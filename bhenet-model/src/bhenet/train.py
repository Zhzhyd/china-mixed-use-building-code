"""Single-device, batch-one training with explicit sample manifests."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch

from .dataloader import read_h5_sample
from .labels import CLASS_CODES
from .preprocessing import PreprocessConfig, prepare_sample
from .runtime import (
    capture_rng_state, checkpoint_metadata, copy_checkpoint, load_checkpoint, restore_rng_state, sample_signature,
    save_checkpoint, seed_everything, tokenizer_signature, training_arguments, validate_input_contract,
)


def _files(paths, name, *, allow_empty=False):
    if isinstance(paths, (str, bytes, Path)):
        raise TypeError(f"{name} must contain individual HDF5 paths")
    result = tuple(Path(path).expanduser().resolve() for path in paths)
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate paths")
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def _forward(model, tokenizer, filename, configuration):
    record = read_h5_sample(filename, raster_names=configuration.raster_names,
                            raster_channels=configuration.raster_channels)
    batch = prepare_sample(record, tokenizer, configuration, require_target=True)
    logits = model(*batch.inputs)
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, len(CLASS_CODES)):
        raise ValueError(f"{filename}: model must return logits with shape (1, 29)")
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError(f"{filename}: non-finite logits")
    target = batch.target.to(device=logits.device, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, target)
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"{filename}: non-finite loss")
    return batch, logits, loss


def _sample_record(batch, logits, loss):
    target = int(batch.target.item())
    prediction = int(logits.detach().argmax(dim=-1).item())
    return {
        "filename": str(batch.filename), "identifier": str(batch.identifier),
        "target": target, "target_label": CLASS_CODES[target],
        "prediction": prediction, "prediction_label": CLASS_CODES[prediction],
        "loss": float(loss.detach().item()), "tokens": int(batch.token_count),
    }


def _metrics(records):
    counts = [0] * len(CLASS_CODES)
    correct = [0] * len(CLASS_CODES)
    for row in records:
        counts[row["target"]] += 1
        correct[row["target"]] += int(row["target"] == row["prediction"])
    recalls = [right / count for right, count in zip(correct, counts) if count]
    return {
        "count": len(records), "loss": sum(row["loss"] for row in records) / len(records),
        "OA29": sum(correct) / len(records), "macro_recall": sum(recalls) / len(CLASS_CODES),
        "represented_macro_recall": sum(recalls) / len(recalls),
        "class_counts": counts, "class_correct": correct,
    }


def evaluate(model, tokenizer, files, *, preprocess_config=None):
    """Evaluate each explicit file once, preserving RNG and module modes.

    Macro recall averages all 29 classes, counting absent reference classes as
    zero, as in the source driver. Represented-class macro recall is also logged.
    """
    paths = _files(files, "validation files")
    configuration = preprocess_config or PreprocessConfig()
    validate_input_contract(model, configuration)
    rng = capture_rng_state()
    modes = [(module, module.training) for module in model.modules()]
    records = []
    try:
        model.eval()
        with torch.inference_mode():
            for path in paths:
                batch, logits, loss = _forward(model, tokenizer, path, configuration)
                records.append(_sample_record(batch, logits, loss))
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng_state(rng)
    return _metrics(records), records


def _write_log(handle, event):
    handle.write(json.dumps(event, allow_nan=False, ensure_ascii=False) + "\n")
    handle.flush()


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _recover_epoch_log(path, history):
    """Restore the committed journal, preserving any interrupted-write evidence."""
    existing = None
    if path.exists():
        try:
            existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (ValueError, UnicodeError):
            pass
    if existing == history:
        return
    if path.exists():
        recovered = path.with_name(f"recovered_epochs_{uuid.uuid4().hex}.jsonl")
        path.replace(recovered)
    with path.open("x", encoding="utf-8") as handle:
        for row in history:
            _write_log(handle, row)


def run_training(
    model, tokenizer, train_files, val_files, output_dir, *, epochs=400,
    learning_rate=2e-5, weight_decay=0.01, scheduler_t_max=400, eta_min=1e-5,
    seed=2025, preprocess_config=None, resume=None, metadata=None,
):
    """Train through ``epochs`` completed epochs, optionally continuing ``last.pt``.

    The cosine scheduler advances after every optimizer update, preserving the
    source protocol's T_max=400 update steps. Validation is optional and never
    inferred. Its all-29-class macro recall selects ``best.pt``; ``last.pt``
    always records the last completed epoch. Both contain all model weights,
    optimizer/scheduler state and RNG streams. Increasing epochs on resume is
    allowed; changing inputs, preprocessing or optimizer settings is rejected.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This training entry point uses one process and one device")
    _positive_integer("epochs", epochs)
    _positive_integer("scheduler_t_max", scheduler_t_max)
    for name, value, positive in (("learning_rate", learning_rate, True),
                                  ("weight_decay", weight_decay, False), ("eta_min", eta_min, False)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (positive and value == 0):
            raise ValueError(f"{name} is outside its allowed range")
    if eta_min > learning_rate:
        raise ValueError("eta_min must not exceed learning_rate")
    configuration = preprocess_config or PreprocessConfig()
    if not isinstance(configuration, PreprocessConfig):
        raise TypeError("preprocess_config must be a PreprocessConfig")
    validate_input_contract(model, configuration)
    if configuration.max_tokens is not None:
        raise ValueError("Training requires max_tokens=None to preserve complete sequences")
    training = _files(train_files, "training files")
    validation = _files(val_files, "validation files", allow_empty=True)
    if set(training) & set(validation):
        raise ValueError("Training and validation files overlap")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    settings = {
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "scheduler_t_max": scheduler_t_max, "eta_min": eta_min, "seed": seed,
        "batch_size": 1, "scheduler_step": "optimizer_step",
        "selection_metric": "validation_macro_recall_all29",
    }
    reserved = {"training_settings", "preprocessing", "train_signature", "validation_signature",
                "tokenizer_signature", "training_state"}
    if not isinstance(metadata, (dict, type(None))) or reserved.intersection(metadata or {}):
        raise ValueError("metadata must be an object without reserved training keys")
    expected = json.loads(json.dumps({
        **(metadata or {}), "training_settings": settings, "preprocessing": asdict(configuration),
        "train_signature": sample_signature(training), "validation_signature": sample_signature(validation),
        "tokenizer_signature": tokenizer_signature(tokenizer),
    }, allow_nan=False))
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and not directory.is_dir():
        raise ValueError("output_dir must be a directory")
    if resume is None and directory.exists() and any(directory.iterdir()):
        raise FileExistsError("A new run requires an empty output directory; use --resume for an existing run")
    if resume is not None and Path(resume).expanduser().resolve() != directory / "last.pt":
        raise ValueError("Resume requires last.pt in the same output directory")
    seed_everything(seed)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=eta_min)
    completed, global_step, best_epoch, best_score = 0, 0, None, None
    metrics_path, samples_path = directory / "epochs.jsonl", directory / "samples.jsonl"
    history = []
    if resume is not None:
        saved = load_checkpoint(resume, model, optimizer=optimizer, scheduler=scheduler,
                                expected_metadata=expected, restore_rng=True)
        completed = saved["epoch"]
        state = saved["metadata"]["training_state"]
        global_step, best_epoch, best_score = state["global_step"], state["best_epoch"], state["best_macro_recall"]
        if isinstance(completed, bool) or not isinstance(completed, int) or not 1 <= completed <= epochs:
            raise ValueError("Resume epoch lies outside the requested completed-epoch range")
        if global_step != completed * len(training) or scheduler.last_epoch != global_step:
            raise ValueError("Resume checkpoint is not a completed training epoch")
        if not samples_path.is_file():
            raise ValueError("Resume requires the original sample log")
        history = state["epoch_records"]
        if [row.get("epoch") for row in history] != list(range(1, completed + 1)):
            raise ValueError("Checkpoint history does not match its completed epochs")
        if history[-1].get("global_step") != global_step:
            raise ValueError("Epoch log and checkpoint global steps differ")
        if validation:
            if best_epoch is None or best_score is None or not 1 <= best_epoch <= completed:
                raise ValueError("Resume checkpoint has no valid validation-selection history")
            selected = max(history, key=lambda row: row["validation"]["macro_recall"])
            if (selected["epoch"], selected["validation"]["macro_recall"]) != (best_epoch, best_score):
                raise ValueError("Epoch log and checkpoint best-selection history differ")
            if best_epoch == completed:
                copy_checkpoint(directory / "last.pt", directory / "best.pt")
            else:
                best = checkpoint_metadata(directory / "best.pt")
                if best["epoch"] != best_epoch or any(best["metadata"].get(key) != value for key, value in expected.items()):
                    raise ValueError("Historical best checkpoint differs from committed selection history")
                if best["metadata"]["training_state"]["best_macro_recall"] != best_score:
                    raise ValueError("Historical best checkpoint has a different selection score")
        _recover_epoch_log(metrics_path, history)
    directory.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    with metrics_path.open("a", encoding="utf-8") as epoch_log, samples_path.open("a", encoding="utf-8") as sample_log:
        for epoch in range(completed, epochs):
            started = time.perf_counter()
            model.train()
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(len(training), generator=generator).tolist()
            train_records = []
            for position, index in enumerate(order, start=1):
                optimizer.zero_grad(set_to_none=True)
                batch, logits, loss = _forward(model, tokenizer, training[index], configuration)
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                       for parameter in parameters):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"{training[index]}: non-finite gradient; optimizer update was not applied")
                lr_used = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                global_step += 1
                row = _sample_record(batch, logits, loss)
                train_records.append(row)
                _write_log(sample_log, {
                    "event": "train_sample", "attempt_id": attempt_id, "epoch": epoch + 1,
                    "position": position, "global_step": global_step, "lr_used": lr_used,
                    "lr_next": optimizer.param_groups[0]["lr"], **row,
                })
            optimizer.zero_grad(set_to_none=True)
            validation_metrics = None
            if validation:
                validation_metrics, rows = evaluate(model, tokenizer, validation, preprocess_config=configuration)
                for position, row in enumerate(rows, start=1):
                    _write_log(sample_log, {"event": "validation_sample", "attempt_id": attempt_id,
                                           "epoch": epoch + 1, "position": position, **row})
            improved = validation_metrics is not None and (best_score is None or validation_metrics["macro_recall"] > best_score)
            if improved:
                best_score, best_epoch = validation_metrics["macro_recall"], epoch + 1
            record = {
                "event": "epoch_complete", "epoch": epoch + 1, "global_step": global_step,
                "train": _metrics(train_records), "validation": validation_metrics,
                "best_epoch": best_epoch, "best_macro_recall": best_score,
                "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started,
            }
            snapshot_metadata = {**expected, "training_state": {
                "global_step": global_step, "best_epoch": best_epoch, "best_macro_recall": best_score,
                "epoch_records": history + [record],
            }}
            rng = capture_rng_state()
            # last.pt is the atomic commit. Its history repairs a lagging journal,
            # and its weights repair an interrupted update of the selected best.
            save_checkpoint(directory / "last.pt", model, optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch + 1, metadata=snapshot_metadata, rng_state=rng)
            if improved:
                copy_checkpoint(directory / "last.pt", directory / "best.pt")
            _write_log(epoch_log, record)
            history.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
    return {"epochs": epochs, "global_step": global_step, "best_epoch": best_epoch,
            "best_macro_recall": best_score, "last_checkpoint": str(directory / "last.pt"),
            "best_checkpoint": str(directory / "best.pt") if validation else None,
            "epoch_records": history}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON configuration with explicit training/validation manifests")
    parser.add_argument("--resume", help="Continue an existing run from its last.pt checkpoint")
    args = parser.parse_args(argv)
    run_training(**training_arguments(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
