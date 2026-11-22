"""Optional local pretrained backbones for the multimodal architecture.

The factory initializes the Atom/LLaMA and MobileSAM backbone architecture from
local checkpoint files, with new low-rank adapters. Use trusted local files and
restore the matching complete state dictionary for a trained classifier.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import torch
from torch import nn


PathLike = Union[str, Path]


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")


def _require_targets(module: nn.Module, targets: tuple[str, ...], name: str) -> None:
    available = {key.rsplit(".", 1)[-1] for key, _ in module.named_modules()}
    missing = set(targets) - available
    if missing:
        raise ValueError(
            f"{name} is missing required LoRA target modules: {sorted(missing)}. "
            "Check that the checkpoint matches the documented backbone architecture."
        )


def _adapters_only(module: nn.Module, name: str) -> None:
    """Explicitly freeze the base weights, independent of PEFT defaults."""
    found = False
    for parameter_name, parameter in module.named_parameters():
        is_adapter = any(part.startswith("lora_") for part in parameter_name.split("."))
        parameter.requires_grad_(is_adapter)
        found |= is_adapter
    if not found:
        raise RuntimeError(f"PEFT did not create any LoRA parameters for {name}.")


def build_backbones(
    text_model_dir: PathLike,
    vision_checkpoint: PathLike,
    *,
    feature_dim: int = 256,
    device: Union[str, torch.device] = "cpu",
    text_dtype: torch.dtype | None = None,
    lora_rank: int = 8,
    lora_alpha: float = 32,
    lora_dropout: float = 0.05,
    train_text_projection: bool = False,
) -> tuple[nn.Module, nn.Module]:
    """Construct local LLaMA and MobileSAM ``vit_t`` encoders with RS-LoRA.

    Parameters
    ----------
    text_model_dir:
        Local Atom-7B-Chat/LLaMA checkpoint directory accepted by
        ``LlamaForCausalLM.from_pretrained``. Network downloads and custom
        checkpoint Python code are disabled.
    vision_checkpoint:
        Trusted local MobileSAM checkpoint. Only its image encoder is returned.
    feature_dim:
        Shared encoder width. The pretrained MobileSAM ``vit_t`` implementation
        has 256 output channels, so this factory requires 256.
    device:
        Explicit destination; defaults to CPU and does not alter CUDA settings.
    text_dtype:
        Floating-point dtype for the text backbone and new projection. Defaults
        to float16 on CUDA and float32 otherwise. The visual encoder uses
        float32, as in the original model. Adapter dtype follows PEFT.
    lora_rank, lora_alpha, lora_dropout:
        RS-LoRA settings, defaulting to rank 8, alpha 32 and dropout 0.05. Text
        targets are ``q_proj``/``v_proj``; image targets are ``qkv``.
    train_text_projection:
        Whether the replacement, bias-free ``lm_head`` is trainable. The default
        keeps it frozen, matching the original PEFT-wrapped model. Enabling it
        is an explicit change in the trainable parameter set.

    Returns
    -------
    text_encoder, image_encoder:
        The PEFT-wrapped ``LlamaForCausalLM`` (``TOKEN_CLS`` wrapper, retained from
        the original implementation) and MobileSAM image encoder. The text
        output's ``logits`` are per-token features, not vocabulary scores.

    Notes
    -----
    The new projection and adapters are initialized here, not restored from a
    building-function checkpoint. Reproduction requires loading the complete
    compatible trained model state separately. This function is architecture
    construction with explicit configuration and parameter freezing.
    """
    _positive_integer(feature_dim, "feature_dim")
    if feature_dim != 256:
        raise ValueError("MobileSAM vit_t has 256 output channels; feature_dim must be 256.")
    _positive_integer(lora_rank, "lora_rank")
    if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, (int, float)):
        raise ValueError("lora_alpha must be a positive finite number.")
    if not math.isfinite(lora_alpha) or lora_alpha <= 0:
        raise ValueError("lora_alpha must be a positive finite number.")
    if isinstance(lora_dropout, bool) or not isinstance(lora_dropout, (int, float)):
        raise ValueError("lora_dropout must be a finite number in [0, 1).")
    if not math.isfinite(lora_dropout) or not 0 <= lora_dropout < 1:
        raise ValueError("lora_dropout must be a finite number in [0, 1).")
    if not isinstance(train_text_projection, bool):
        raise TypeError("train_text_projection must be a bool.")

    text_path = Path(text_model_dir).expanduser()
    vision_path = Path(vision_checkpoint).expanduser()
    if not text_path.is_dir():
        raise FileNotFoundError(f"Local text model directory does not exist: {text_path}")
    if not vision_path.is_file():
        raise FileNotFoundError(f"Local MobileSAM checkpoint does not exist: {vision_path}")
    if not (text_path / "config.json").is_file():
        raise FileNotFoundError(f"Text model config.json is missing: {text_path}")

    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid device: {device!r}.") from exc
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")
        if target_device.index is not None and target_device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {target_device.index}.")
    if text_dtype is None:
        text_dtype = torch.float16 if target_device.type == "cuda" else torch.float32
    if text_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("text_dtype must be torch.float16, torch.bfloat16, or torch.float32.")

    try:
        from transformers import LlamaForCausalLM
    except ImportError as exc:
        raise ImportError("The optional text backbone requires the 'transformers' package.") from exc
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("The optional pretrained backbones require the 'peft' package.") from exc
    try:
        from mobile_sam import sam_model_registry
    except ImportError as exc:
        raise ImportError("The optional image backbone requires the 'mobile_sam' package.") from exc
    if "vit_t" not in sam_model_registry:
        raise ValueError("The installed mobile_sam does not register the required 'vit_t' encoder.")

    try:
        text_encoder = LlamaForCausalLM.from_pretrained(
            str(text_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=text_dtype,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not load the local LLaMA checkpoint at {text_path}: {exc}") from exc
    hidden_size = getattr(getattr(text_encoder, "config", None), "hidden_size", None)
    _positive_integer(hidden_size, "text checkpoint config.hidden_size")
    _require_targets(text_encoder, ("q_proj", "v_proj"), "The text encoder")
    projection = nn.Linear(hidden_size, feature_dim, bias=False, dtype=text_dtype)
    text_encoder.lm_head = projection
    text_lora = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=True,
        target_modules=["q_proj", "v_proj"],
        task_type="TOKEN_CLS",
    )
    text_encoder = get_peft_model(text_encoder, text_lora)
    _adapters_only(text_encoder, "the text encoder")
    if not any(child is projection for child in text_encoder.modules()):
        raise RuntimeError(
            "PEFT replaced the text projection unexpectedly; check the installed PEFT version."
        )
    projection.requires_grad_(train_text_projection)
    if any(parameter.requires_grad != train_text_projection for parameter in projection.parameters()):
        raise RuntimeError("Could not apply the requested text projection freezing policy.")
    text_encoder.to(device=target_device)

    sam = sam_model_registry["vit_t"](checkpoint=str(vision_path))
    image_encoder = getattr(sam, "image_encoder", None)
    if not isinstance(image_encoder, nn.Module):
        raise ValueError("The MobileSAM checkpoint did not construct an image encoder.")
    _require_targets(image_encoder, ("qkv",), "The image encoder")
    image_lora = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=True,
        target_modules=["qkv"],
    )
    image_encoder = get_peft_model(image_encoder, image_lora)
    _adapters_only(image_encoder, "the image encoder")
    image_encoder.to(device=target_device, dtype=torch.float32)
    return text_encoder, image_encoder
