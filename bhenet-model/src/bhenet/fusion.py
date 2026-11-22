"""Position-aware text, image and auxiliary-raster feature fusion."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .config import normalize_raster_spec
from .layers import (
    CrossAttentionBlock,
    RMSNorm,
    _floating_tensor,
    _matches_parameter,
    _positive_int,
    _same_storage_type,
)


def _raster_inputs(rasters, names, channels, batch, grid, reference=None):
    if rasters is None:
        rasters = {}
    if not isinstance(rasters, Mapping):
        raise TypeError("rasters must be a mapping from configured names to tensors")
    if set(rasters) != set(names):
        missing = [name for name in names if name not in rasters]
        extra = [str(name) for name in rasters if name not in names]
        raise ValueError(f"Raster keys differ from configuration; missing={missing}, extra={extra}")
    for name, count in zip(names, channels):
        tensor = rasters[name]
        _floating_tensor(f"rasters[{name!r}]", tensor, 4)
        if tuple(tensor.shape) != (batch, count, grid, grid):
            raise ValueError(f"Raster {name!r} must have shape {(batch, count, grid, grid)}")
        if reference is not None:
            _same_storage_type(f"rasters[{name!r}]", tensor, reference)
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Raster {name!r} must contain only finite values")
    return rasters


class RasterFusionStage(nn.Module):
    """One spatially conditioned raster stage following the building stage.

    Raster tokens are ``RMSNorm(Linear(raster)) + position_features``. The
    projected image-position features preserve the alignment of each raster
    cell with its location before these tokens become attention keys/values.
    """

    def __init__(self, *, feature_dim=256, head_dim=32, grid_size=64, channels=1):
        super().__init__()
        _positive_int("grid_size", grid_size)
        _positive_int("channels", channels)
        _positive_int("feature_dim", feature_dim)
        self.grid_size, self.channels, self.feature_dim = grid_size, channels, feature_dim
        self.projection = nn.Linear(channels, feature_dim, bias=False)
        self.norm = RMSNorm(feature_dim)
        self.attention = CrossAttentionBlock(feature_dim=feature_dim, head_dim=head_dim)

    def forward(self, query: Tensor, raster: Tensor, position_features: Tensor) -> Tensor:
        _floating_tensor("query", query, 3)
        _floating_tensor("position_features", position_features, 3)
        _same_storage_type("position_features", position_features, query)
        shape = (query.shape[0], self.grid_size ** 2, self.feature_dim)
        if tuple(query.shape) != shape or tuple(position_features.shape) != shape:
            raise ValueError(f"Raster-stage query and position_features must have shape {shape}")
        _raster_inputs({"stage": raster}, ("stage",), (self.channels,), query.shape[0], self.grid_size, query)
        tokens = raster.reshape(query.shape[0], self.channels, self.grid_size ** 2).transpose(1, 2)
        tokens = self.norm(self.projection(tokens)) + position_features
        return self.attention(queryd=query, keyd=tokens)


class FusionHead(nn.Module):
    """Fuse encoded features and return unnormalized class logits.

    Inputs are prepared tensors, not raw images, POI records or GIS files.
    Image features and the three image-position channels share a square spatial
    grid. The three text-position channels are aligned to the encoded tokens.
    ``label`` denotes an auxiliary raster, not a target class label.

    The initial attention operations are image-to-image-position,
    text-to-text-position, image-to-text, and fused-image-to-building-raster.
    Configured raster stages follow serially, with the road stage enabled by
    default. Each has its own projection, normalization and cross-attention.
    The number of image patches is fixed by ``grid_size`` at construction time.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 256,
        head_dim: int = 32,
        grid_size: int = 64,
        aux_channels: int = 1,
        num_classes: int = 29,
        raster_names: tuple[str, ...] = ("road",),
        raster_channels: tuple[int, ...] = (1,),
    ) -> None:
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("head_dim", head_dim),
            ("grid_size", grid_size),
            ("aux_channels", aux_channels),
            ("num_classes", num_classes),
        ):
            _positive_int(name, value)
        if feature_dim % head_dim:
            raise ValueError("feature_dim must be divisible by head_dim.")
        self.input_dim = feature_dim
        self.image_patch = grid_size
        self.label_num = aux_channels
        self.num_classes = num_classes
        self.raster_names, self.raster_channels = normalize_raster_spec(raster_names, raster_channels)

        self.p_convert_txt = nn.Linear(3, self.input_dim, bias=False)
        self.p_convert_img = nn.Linear(3, self.input_dim, bias=False)
        self.txt_norm = RMSNorm(self.input_dim)
        self.img_norm = RMSNorm(self.input_dim)
        self.convert_lab = nn.Linear(self.label_num, self.input_dim, bias=False)
        self.lab_norm = RMSNorm(self.input_dim)
        self.txt_posi_fusion = CrossAttentionBlock(feature_dim=feature_dim, head_dim=head_dim)
        self.img_posi_fusion = CrossAttentionBlock(feature_dim=feature_dim, head_dim=head_dim)
        self.attention_fusion_block_1 = CrossAttentionBlock(feature_dim=feature_dim, head_dim=head_dim)
        self.attention_fusion_block_2 = CrossAttentionBlock(feature_dim=feature_dim, head_dim=head_dim)
        self.raster_branches = nn.ModuleDict({
            f"raster_{name}": RasterFusionStage(feature_dim=feature_dim, head_dim=head_dim,
                                               grid_size=grid_size, channels=channels)
            for name, channels in zip(self.raster_names, self.raster_channels)
        })
        self.o1 = nn.Linear(grid_size ** 2, 1, bias=False)
        self.o2 = nn.Linear(feature_dim, num_classes, bias=False)

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
        """Return ``(B, num_classes)`` logits without a softmax.

        Shapes are ``text=(B,T,D)``, ``image=(B,D,G,G)``, each image
        position channel ``(B,1,G,G)``, each text position channel
        ``(B,T,1)``, and ``label=(B,A,G,G)``. All tensors must share a
        floating-point dtype and device. ``D``, ``G`` and ``A`` are the
        configured feature dimension, grid size and auxiliary channel count.
        ``rasters`` must contain precisely the configured names, each with shape
        ``(B, raster_channels[i], G, G)`` and finite floating-point values.
        Non-contiguous tensors are supported without changing their ordering.
        """
        _floating_tensor("text", text, 3)
        _floating_tensor("image", image, 4)
        _same_storage_type("image", image, text)
        _matches_parameter("text", text, self.p_convert_txt.weight)
        batch, tokens, features = text.shape
        grid = self.image_patch
        if features != self.input_dim:
            raise ValueError(f"text must have feature dimension {self.input_dim}.")
        if tuple(image.shape) != (batch, self.input_dim, grid, grid):
            raise ValueError(f"image must have shape {(batch, self.input_dim, grid, grid)}.")

        for name, tensor in (("x_mat", x_mat), ("y_mat", y_mat), ("xy_mat_dist", xy_mat_dist)):
            _floating_tensor(name, tensor, 4)
            _same_storage_type(name, tensor, text)
            if tuple(tensor.shape) != (batch, 1, grid, grid):
                raise ValueError(f"{name} must have shape {(batch, 1, grid, grid)}.")
        for name, tensor in (
            ("pois_cor_x", pois_cor_x), ("pois_cor_y", pois_cor_y), ("pois_cor_dist", pois_cor_dist)
        ):
            _floating_tensor(name, tensor, 3)
            _same_storage_type(name, tensor, text)
            if tuple(tensor.shape) != (batch, tokens, 1):
                raise ValueError(f"{name} must have shape {(batch, tokens, 1)}.")
        _floating_tensor("label", label, 4)
        _same_storage_type("label", label, text)
        if tuple(label.shape) != (batch, self.label_num, grid, grid):
            raise ValueError(f"label must have shape {(batch, self.label_num, grid, grid)}.")
        rasters = _raster_inputs(rasters, self.raster_names, self.raster_channels, batch, grid, text)

        p_img = torch.cat((x_mat, y_mat, xy_mat_dist), dim=1)
        p_txt = torch.cat((pois_cor_x, pois_cor_y, pois_cor_dist), dim=-1)
        p_img = self.img_norm(
            self.p_convert_img(p_img.reshape(batch, 3, grid * grid).transpose(1, 2))
        )
        p_txt = self.txt_norm(self.p_convert_txt(p_txt))
        auxiliary = self.lab_norm(
            self.convert_lab(label.reshape(batch, self.label_num, grid * grid).transpose(1, 2))
        )

        image = self.img_posi_fusion(queryd=image, keyd=p_img)
        text = self.txt_posi_fusion(queryd=text, keyd=p_txt)
        output = self.attention_fusion_block_1(queryd=image, keyd=text)
        output = self.attention_fusion_block_2(queryd=output, keyd=auxiliary)
        for name in self.raster_names:
            output = self.raster_branches[f"raster_{name}"](output, rasters[name], p_img)
        return self.o2(self.o1(output.transpose(1, 2)).reshape(batch, self.input_dim))
