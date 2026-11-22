"""Explicit configuration, reproducibility and full-checkpoint I/O."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from dataclasses import asdict

import torch

from .labels import CLASS_CODES
from .config import ModelConfig
from .model import build_bhenet


CHECKPOINT_FORMAT = "bhenet_full_v1"


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    import numpy as np
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    import numpy as np
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the saved number of visible CUDA devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_signature(files) -> str:
    """Hash ordered filenames and file contents; order is part of split identity."""
    records = [(str(Path(p).resolve()), file_sha256(p)) for p in files]
    return hashlib.sha256(json.dumps(records, ensure_ascii=False).encode("utf-8")).hexdigest()


def tokenizer_signature(tokenizer) -> str:
    """Identify vocabulary, special tokens and tokenization settings."""
    attributes = ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id",
                  "padding_side", "truncation_side", "model_max_length",
                  "add_bos_token", "add_eos_token", "legacy")
    details = {name: getattr(tokenizer, name, None) for name in attributes}
    details["class"] = type(tokenizer).__module__ + "." + type(tokenizer).__qualname__
    details["vocabulary"] = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    details["special_tokens"] = {k: str(v) for k, v in getattr(tokenizer, "special_tokens_map", {}).items()}
    return hashlib.sha256(json.dumps(details, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def tokenizer_files_signature(directory) -> dict:
    names = ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.json", "merges.txt")
    assets = {name: file_sha256(Path(directory) / name) for name in names if (Path(directory) / name).is_file()}
    if not assets:
        raise FileNotFoundError("Local tokenizer assets were not found")
    return assets


def read_manifest(filename: str | Path) -> list[Path]:
    """Read a JSON array of HDF5 paths, relative to the manifest location."""
    path = Path(filename).expanduser().resolve()
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries or any(not isinstance(p, str) or not p.strip() for p in entries):
        raise ValueError("A sample manifest must be a nonempty JSON array of file paths")
    files = [(path.parent / Path(p).expanduser()).resolve() for p in entries]
    if len(set(files)) != len(files):
        raise ValueError("A sample manifest contains duplicate resolved file paths")
    for item in files:
        if not item.is_file():
            raise FileNotFoundError(f"Sample file does not exist: {item}")
    return files


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def _model_contract(model) -> dict:
    configuration = getattr(model, "config", None)
    return {
        "config": asdict(configuration) if configuration is not None else None,
        "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
    }


def save_checkpoint(path, model, *, optimizer=None, scheduler=None, epoch, metadata, rng_state=None) -> None:
    """Atomically replace an explicit run checkpoint with complete model state.

    epoch is the number of fully completed epochs. All frozen weights, including
    the initialized frozen text projection, are retained alongside adapters.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a completed-epoch count")
    # Require portable primitive metadata before writing a potentially large file.
    clean_metadata = json.loads(json.dumps(metadata, allow_nan=False))
    payload = {
        "format": CHECKPOINT_FORMAT, "model": _cpu_tree(model.state_dict()),
        "class_codes": list(CLASS_CODES), "model_contract": _model_contract(model),
        "epoch": epoch, "metadata": clean_metadata,
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
        "optimizer": _cpu_tree(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_version": str(torch.__version__),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_load(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def checkpoint_metadata(path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a bhenet_full_v1 checkpoint")
    return {name: payload[name] for name in ("format", "epoch", "metadata", "class_codes", "model_contract")}


def copy_checkpoint(source, target) -> None:
    """Atomically retain a completed checkpoint, using a same-filesystem hard link."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        os.unlink(temporary)
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model, *, optimizer=None, scheduler=None, expected_metadata=None, restore_rng=False) -> dict:
    """Load this package's full checkpoint strictly; never infer key remapping."""
    payload = _safe_load(path)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a bhenet_full_v1 checkpoint; use explicit model-state import for other formats")
    if payload.get("class_codes") != list(CLASS_CODES):
        raise ValueError("Checkpoint class order differs from CLASS_CODES")
    if payload.get("model_contract") != _model_contract(model):
        raise ValueError("Checkpoint architecture or trainable parameter set differs")
    for name, value in (expected_metadata or {}).items():
        if payload.get("metadata", {}).get(name) != value:
            raise ValueError(f"Checkpoint metadata mismatch: {name}")
    if optimizer is not None and payload.get("optimizer") is None:
        raise ValueError("Checkpoint has no optimizer state")
    if scheduler is not None and payload.get("scheduler") is None:
        raise ValueError("Checkpoint has no scheduler state")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


def load_model_weights(path, model, *, state_key: str | None = None) -> None:
    """Explicit import of a trusted tensor state dictionary or a named container.

    Set state_key='net' for compatible historical training containers. Tensor
    keys must already match; wrapper prefixes and redundant aliases are not removed.
    """
    payload = _safe_load(path)
    state = payload if state_key is None else payload[state_key]
    if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("The selected model state must be a nonempty tensor dictionary")
    model.load_state_dict(state, strict=True)


def _section(config, name, allowed):
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {name} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} configuration keys: {sorted(unknown)}")
    return value


def load_run_config(filename) -> dict:
    path = Path(filename).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or set(config) - {"model", "data", "training", "inference", "preprocessing"}:
        raise ValueError("Configuration must contain only model/data/training/inference/preprocessing sections")
    _section(config, "model", {"text_model_dir", "vision_checkpoint", "device", "text_dtype", "train_text_projection", "initial_weights", "initial_state_key"})
    _section(config, "data", {"train_manifest", "validation_manifest", "predict_manifest"})
    _section(config, "training", {"output_dir", "epochs", "learning_rate", "weight_decay", "scheduler_t_max", "eta_min", "seed"})
    _section(config, "inference", {"checkpoint", "checkpoint_state_key", "output_csv", "max_tokens"})
    from .preprocessing import PreprocessConfig
    _section(config, "preprocessing", set(PreprocessConfig.__dataclass_fields__))
    def resolve(value):
        return str((path.parent / Path(value).expanduser()).resolve()) if value is not None else None
    path_keys = {"model": ("text_model_dir", "vision_checkpoint", "initial_weights"),
                 "data": ("train_manifest", "validation_manifest", "predict_manifest"),
                 "training": ("output_dir",), "inference": ("checkpoint", "output_csv")}
    for section, keys in path_keys.items():
        for key in keys:
            if key in config.get(section, {}):
                config[section][key] = resolve(config[section][key])
    return config


def build_runtime(config: dict):
    """Construct the model and local tokenizer with explicit random initialization."""
    settings = config["model"]
    seed_everything(config.get("training", {}).get("seed", 2025))
    torch.set_float32_matmul_precision("high")
    dtypes = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16, None: None}
    if settings.get("text_dtype") not in dtypes:
        raise ValueError("text_dtype must be float16, bfloat16 or float32")
    from .preprocessing import PreprocessConfig
    preparation = PreprocessConfig(**config.get("preprocessing", {}))
    architecture = ModelConfig(grid_size=preparation.grid_size,
                               raster_names=preparation.raster_names,
                               raster_channels=preparation.raster_channels)
    model = build_bhenet(settings["text_model_dir"], settings["vision_checkpoint"],
                         config=architecture,
                         device=settings.get("device", "cpu"), text_dtype=dtypes[settings.get("text_dtype")],
                         train_text_projection=settings.get("train_text_projection", False))
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(settings["text_model_dir"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        # Masked tokenizer padding is removed before the historical ID-2 fill.
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires an existing padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def validate_input_contract(model, configuration) -> None:
    """Check prepared raster names, channels and grid against the model."""
    architecture = getattr(model, "config", None)
    if isinstance(architecture, ModelConfig):
        for name in ("grid_size", "raster_names", "raster_channels"):
            if getattr(architecture, name) != getattr(configuration, name):
                raise ValueError(f"Model and preprocessing disagree on {name}")
        if architecture.aux_channels != 1:
            raise ValueError("HDF5 preprocessing requires a single building-raster channel")


def training_arguments(filename) -> dict:
    from .preprocessing import PreprocessConfig
    config = load_run_config(filename)
    train_files = read_manifest(config["data"]["train_manifest"])
    validation_manifest = config["data"].get("validation_manifest")
    val_files = read_manifest(validation_manifest) if validation_manifest else []
    if set(train_files) & set(val_files):
        raise ValueError("Training and validation manifests overlap")
    settings = dict(config["training"])
    output_dir = settings.pop("output_dir")
    preprocess = PreprocessConfig(**config.get("preprocessing", {}))
    if preprocess.max_tokens is not None:
        raise ValueError("Training preserves long token sequences; preprocessing.max_tokens must be null")
    model, tokenizer = build_runtime(config)
    if config["model"].get("initial_weights"):
        load_model_weights(config["model"]["initial_weights"], model,
                           state_key=config["model"].get("initial_state_key"))
    return dict(model=model, tokenizer=tokenizer, train_files=train_files, val_files=val_files,
                output_dir=output_dir, preprocess_config=preprocess,
                metadata={"model_settings": config["model"],
                          "tokenizer_files": tokenizer_files_signature(config["model"]["text_model_dir"])}, **settings)
