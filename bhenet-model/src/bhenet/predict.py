"""Per-building inference from an explicit HDF5 sample manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile

import torch

from .dataloader import H5BuildingDataset
from .labels import CLASS_CODES
from .preprocessing import PreprocessConfig, prepare_sample
from .runtime import (build_runtime, checkpoint_metadata, load_checkpoint, load_model_weights,
                      load_run_config, read_manifest, tokenizer_signature, tokenizer_files_signature,
                      validate_input_contract)


def run_prediction(model, tokenizer, files, output_csv, *, checkpoint,
                   preprocess_config=None, checkpoint_state_key=None) -> dict:
    """Restore a complete classifier and write ordered predictions atomically.

    Reference labels are ignored. Softmax scores are model scores, not calibrated
    probabilities. A user-supplied legacy state_key is an explicit compatibility
    choice; matching keys/shapes and historical label order remain required.
    """
    target = Path(output_csv).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Prediction output already exists: {target}")
    files = [Path(p).expanduser().resolve() for p in files]
    if not files or len(set(files)) != len(files):
        raise ValueError("Prediction requires a nonempty, unique sample list")
    configuration = preprocess_config or PreprocessConfig()
    validate_input_contract(model, configuration)
    if checkpoint_state_key is None:
        payload = load_checkpoint(checkpoint, model, expected_metadata={"tokenizer_signature": tokenizer_signature(tokenizer)})
        saved = payload["metadata"].get("preprocessing")
        if saved is None:
            raise ValueError("Checkpoint lacks the training preprocessing configuration")
        requested = json.loads(json.dumps(asdict(configuration)))
        if {k: v for k, v in saved.items() if k != "max_tokens"} != {k: v for k, v in requested.items() if k != "max_tokens"}:
            raise ValueError("Inference preprocessing differs from the training checkpoint")
    else:
        load_model_weights(checkpoint, model, state_key=checkpoint_state_key)
    dataset = H5BuildingDataset(files, raster_names=configuration.raster_names,
                               raster_channels=configuration.raster_channels)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    fields = ["row", "filename", "building_id", "predicted_code", "BF1p", "BF2p",
              "softmax_score", "tokens_before_truncation", "tokens_used"]
    fields += [f"score_{code}" for code in CLASS_CODES]
    truncated = 0
    model.eval()
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row, record in enumerate(dataset):
                sample = prepare_sample(record, tokenizer, configuration, require_target=False)
                logits = model(*sample.inputs)
                if tuple(logits.shape) != (1, len(CLASS_CODES)) or not torch.isfinite(logits).all():
                    raise ValueError(f"Nonfinite or malformed logits for {sample.filename}")
                scores = logits.float().softmax(dim=-1)[0].cpu().tolist()
                index = int(logits.argmax(dim=-1).item())
                code = CLASS_CODES[index]
                used = sample.inputs[0].shape[1]
                truncated += int(configuration.max_tokens is not None and sample.token_count > configuration.max_tokens)
                result = dict(row=row, filename=sample.filename, building_id=sample.identifier,
                              predicted_code=code, BF1p=code[0], BF2p=code[1:] or "",
                              softmax_score=scores[index], tokens_before_truncation=sample.token_count,
                              tokens_used=used)
                result.update({f"score_{label}": value for label, value in zip(CLASS_CODES, scores)})
                writer.writerow(result)
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusively reserve the final filename before replacement; preserve existing files.
        with target.open("x", encoding="utf-8"):
            pass
        try:
            os.replace(temporary, target)
        except OSError:
            # Remove only this call's empty reservation, never an existing result.
            if target.is_file() and target.stat().st_size == 0:
                target.unlink()
            raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"samples": len(files), "truncated_samples": truncated, "output_csv": str(target)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Run configuration JSON")
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    settings = config["inference"]
    files = read_manifest(config["data"]["predict_manifest"])
    checkpoint = Path(settings["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint}")
    if Path(settings["output_csv"]).exists():
        raise FileExistsError(settings["output_csv"])
    if settings.get("checkpoint_state_key") is None:
        saved_files = checkpoint_metadata(checkpoint)["metadata"].get("tokenizer_files")
        if saved_files != tokenizer_files_signature(config["model"]["text_model_dir"]):
            raise ValueError("Local tokenizer files differ from the training checkpoint")
    preprocessing = PreprocessConfig(**config.get("preprocessing", {}))
    if "max_tokens" in settings:
        preprocessing = replace(preprocessing, max_tokens=settings["max_tokens"])
    model, tokenizer = build_runtime(config)
    result = run_prediction(model, tokenizer, files, settings["output_csv"], checkpoint=checkpoint,
                            preprocess_config=preprocessing,
                            checkpoint_state_key=settings.get("checkpoint_state_key"))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
