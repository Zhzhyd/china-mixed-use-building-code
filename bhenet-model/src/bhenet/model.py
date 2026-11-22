"""Composition of text, image and spatial-fusion modules."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .fusion import FusionHead, _raster_inputs


def _placement(module: nn.Module, fallback: Tensor) -> tuple[torch.device, torch.dtype]:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.device, parameter.dtype
    for buffer in module.buffers():
        if buffer.is_floating_point():
            return buffer.device, buffer.dtype
    return fallback.device, fallback.dtype


class BHENet(nn.Module):
    """Building classifier with injected backbone modules.

    Backbones must return token features in ``.logits`` (B,L,D) and image
    features (B,D,G,G), respectively. Inputs are already prepared tensors;
    the tensor contract defines their encoding, dimensions and numeric scale.
    The output is unnormalized class logits. All tokens participate in
    attention, including any padding supplied by the caller, as in the
    research implementation.

    Whole backbone modules may occupy different devices. Internal sharding
    of a backbone is outside this interface. No device is selected globally.
    """

    def __init__(
        self,
        language_model: nn.Module,
        image_model_encoder: nn.Module,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(language_model, nn.Module) or not isinstance(image_model_encoder, nn.Module):
            raise TypeError("Both backbones must be torch.nn.Module instances")
        if config is not None and not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self.config = config or ModelConfig()
        # Retain the established names for backbone and initial fusion modules.
        self.language_model = language_model
        self.image_model_encoder = image_model_encoder
        self.fusion_model = FusionHead(
            feature_dim=self.config.feature_dim,
            head_dim=self.config.head_dim,
            grid_size=self.config.grid_size,
            aux_channels=self.config.aux_channels,
            num_classes=self.config.num_classes,
            raster_names=self.config.raster_names,
            raster_channels=self.config.raster_channels,
        )

    def forward(
        self,
        text: Tensor,
        image: Tensor,
        x_mat: Tensor,
        y_mat: Tensor,
        xy_mat_dist: Tensor,
        pois_cor_x: Tensor,
        pois_cor_y: Tensor,
        pois_cor_dist: Tensor,
        label: Tensor,
        rasters: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Evaluate model modules; ``label`` is the auxiliary raster, not a target.

        text: integer token IDs (B,L); image: prepared floating RGB (B,3,H,W).
        Image positions: three floating tensors (B,1,G,G).
        Text positions: three floating tensors (B,L,1).
        label: floating auxiliary tensor (B,A,G,G).
        rasters: configured additional rasters, each (B,C,G,G); road is required
        by default and is fused after the building raster.
        """
        if not isinstance(text, Tensor) or text.ndim != 2 or min(text.shape) < 1:
            raise ValueError("text must have nonempty shape (batch, tokens)")
        if text.dtype not in (torch.int32, torch.int64):
            raise TypeError("text must contain integer token IDs (int32 or int64)")
        if not isinstance(image, Tensor) or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape (batch, 3, height, width)")
        if image.shape[0] != text.shape[0] or min(image.shape) < 1:
            raise ValueError("image and text require matching, nonempty batches")
        if not image.is_floating_point():
            raise TypeError("image must be a prepared floating-point tensor")
        positions = (x_mat, y_mat, xy_mat_dist, pois_cor_x, pois_cor_y, pois_cor_dist, label)
        for value in positions:
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError("Position and auxiliary inputs must be floating-point tensors")
        batch, length = text.shape
        size = self.config.grid_size
        expected = [(batch, 1, size, size)] * 3 + [(batch, length, 1)] * 3
        expected.append((batch, self.config.aux_channels, size, size))
        for value, shape in zip(positions, expected):
            if tuple(value.shape) != shape:
                raise ValueError(f"Position/auxiliary shape {tuple(value.shape)}; expected {shape}")
        rasters = _raster_inputs(rasters, self.config.raster_names, self.config.raster_channels, batch, size)
        text_device, _ = _placement(self.language_model, text)
        image_device, image_dtype = _placement(self.image_model_encoder, image)
        fusion_device, fusion_dtype = _placement(self.fusion_model, image)
        text_output = self.language_model(text.to(device=text_device, dtype=torch.long))
        if not hasattr(text_output, "logits") or not isinstance(text_output.logits, Tensor):
            raise TypeError("language_model must return a tensor in its .logits attribute")
        image_output = self.image_model_encoder(image.to(device=image_device, dtype=image_dtype))
        if not isinstance(image_output, Tensor):
            raise TypeError("image_model_encoder must return a tensor")
        return self.fusion_model(
            text_output.logits.to(device=fusion_device, dtype=fusion_dtype),
            image_output.to(device=fusion_device, dtype=fusion_dtype),
            *(value.to(device=fusion_device, dtype=fusion_dtype) for value in positions),
            rasters={name: rasters[name].to(device=fusion_device, dtype=fusion_dtype)
                     for name in self.config.raster_names},
        )


def build_bhenet(
    text_model_dir: str | Path,
    vision_checkpoint: str | Path,
    *,
    config: ModelConfig | None = None,
    device: str | torch.device = "cpu",
    text_dtype: torch.dtype | None = None,
    train_text_projection: bool = False,
) -> BHENet:
    """Construct the research backbone family from existing local weights only.

    LoRA adapters, the token projection and the fusion head, including additional
    raster stages, are initialized,
    not a trained building classifier. Load a matching full model state
    dictionary separately with ``load_state_dict(..., strict=True)``.
    Local backbone files and explicit settings define model construction.
    """
    from .backbones import build_backbones

    configuration = config or ModelConfig()
    if not isinstance(configuration, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    if configuration.feature_dim != 256 or configuration.grid_size != 64:
        raise ValueError("The MobileSAM builder requires feature_dim=256 and grid_size=64")
    text_backbone, image_backbone = build_backbones(
        text_model_dir,
        vision_checkpoint,
        feature_dim=configuration.feature_dim,
        device=device,
        text_dtype=text_dtype,
        train_text_projection=train_text_projection,
    )
    model = BHENet(text_backbone, image_backbone, configuration)
    model.fusion_model.to(device=device, dtype=torch.float32)
    return model
