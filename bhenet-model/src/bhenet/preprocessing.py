"""Prepare one stored building using the research training transformations.

Coordinates in a sample are already normalized together by the data producer.
They are not longitude/latitude, and this module does not normalize them again.
The returned tensors include their batch-of-one dimension and remain on the
CPU. Configured named rasters form a final mapping after the original nine
inputs. Every raster must share the RGB crop's extent and orientation; the
data producer must establish this geographic alignment before preprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import normalize_raster_spec
from .labels import CLASS_CODES


_EMPTY_POI_TEXT = "名称：无有效名称，大类：无有效类别"
_CLASS_INDEX = {code: index for index, code in enumerate(CLASS_CODES)}


@dataclass(frozen=True)
class PreprocessConfig:
    """Image, grid and token settings shared by training and inference.

    Training pads short token sequences but never truncates them. An explicit
    ``max_tokens`` enables inference truncation and must be at least
    ``min_tokens``. Use equal minimum and maximum values to reproduce the
    historical prediction script's fixed token length.
    """

    image_size: int = 1024
    grid_size: int = 64
    min_tokens: int = 512
    max_tokens: int | None = None
    padding_token_id: int = 2
    pixel_mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    pixel_std: tuple[float, float, float] = (58.395, 57.12, 57.375)
    raster_names: tuple[str, ...] = ("road",)
    raster_channels: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        raster_names, raster_channels = normalize_raster_spec(self.raster_names, self.raster_channels)
        object.__setattr__(self, "raster_names", raster_names)
        object.__setattr__(self, "raster_channels", raster_channels)
        for name in ("image_size", "grid_size", "min_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_tokens is not None:
            if type(self.max_tokens) is not int or self.max_tokens < self.min_tokens:
                raise ValueError("max_tokens must be an integer >= min_tokens, or None")
        if type(self.padding_token_id) is not int or self.padding_token_id < 0:
            raise ValueError("padding_token_id must be a nonnegative integer")
        for name in ("pixel_mean", "pixel_std"):
            try:
                values = tuple(float(value) for value in getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain three finite numbers") from exc
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite numbers")
            if name == "pixel_std" and any(value <= 0 for value in values):
                raise ValueError("pixel_std values must be positive")
            # JSON round trips produce lists; retain the immutable tuple API.
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class PreparedSample:
    """Model inputs plus the target and source identity for one building.

    ``inputs`` follows ``BHENet.forward``: token IDs, RGB image, x/y image
    grids, image distance, x/y token coordinates, token distance, auxiliary
    building raster, then a mapping of configured raster names to tensors.
    With no named rasters, the mapping is omitted and the original nine-input
    interface is retained. ``token_count`` is after removing tokenizer padding
    and before any final padding or inference truncation.
    """

    inputs: tuple[Tensor | dict[str, Tensor], ...]
    target: Tensor | None
    identifier: Any
    filename: str | None
    token_count: int


def _float_tensor(value: Any, name: str, *, dtype: torch.dtype | None = torch.float32) -> Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if tensor.is_complex():
        raise ValueError(f"{name} must be a real numeric array")
    if dtype is None:
        dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
    # Construct directly at the requested precision: Python coordinate lists
    # must not first round through torch's default float32 dtype.
    tensor = torch.as_tensor(value, dtype=dtype, device="cpu").detach().clone()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _shape(tensor: Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}; got {tuple(tensor.shape)}")


def _raster(
    value: Any, name: str, channels: int, *, dtype: torch.dtype = torch.float32,
) -> Tensor:
    tensor = _float_tensor(value, name, dtype=dtype)
    if tensor.ndim != 3 or tensor.shape[0] != channels or min(tensor.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape ({channels}, H, W)")
    return tensor.unsqueeze(0)


def _prepare_rasters(sample: Mapping[str, Any], config: PreprocessConfig) -> dict[str, Tensor]:
    raw = sample.get("rasters", {})
    if not isinstance(raw, Mapping):
        raise ValueError("Sample rasters must be a mapping from configured names to arrays")
    missing = [name for name in config.raster_names if name not in raw]
    unexpected = [name for name in raw if name not in config.raster_names]
    if missing or unexpected:
        raise ValueError(f"Sample rasters must match configured names; missing={missing}, unexpected={unexpected}")
    prepared = {}
    for name, channels in zip(config.raster_names, config.raster_channels):
        # Validate occupancy before float32 conversion could round a value to
        # exactly 0 or 1. Road is not encoded like the building auxiliary input.
        dtype = torch.float64 if name == "road" else torch.float32
        raster = _raster(raw[name], f"rasters[{name!r}]", channels, dtype=dtype)
        if name == "road" and not ((raster == 0) | (raster == 1)).all():
            raise ValueError("rasters['road'] must contain binary occupancy values 0 or 1")
        prepared[name] = F.interpolate(
            raster.float(), size=(config.grid_size, config.grid_size), mode="nearest",
        )
    return prepared


def _target(sample: Mapping[str, Any]) -> Tensor:
    bf1, bf2 = sample.get("BF1"), sample.get("BF2")
    if not isinstance(bf1, str) or not isinstance(bf2, str):
        raise ValueError("Training requires text BF1 and BF2 labels; use literal 'None' for absent BF2")
    code = bf1 if bf2 == "None" else bf1 + bf2
    code = "E" if code == "EE" else code
    if code not in _CLASS_INDEX:
        raise ValueError(f"Unknown training class {code!r} from BF1={bf1!r}, BF2={bf2!r}")
    return torch.tensor([_CLASS_INDEX[code]], dtype=torch.long)


def _token_tensors(tokenizer: Any, descriptions: list[str]) -> tuple[Tensor, Tensor]:
    encoded = tokenizer(descriptions, return_tensors="pt", padding=True)
    try:
        if isinstance(encoded, Mapping):
            ids, mask = encoded["input_ids"], encoded["attention_mask"]
        else:
            ids, mask = encoded.input_ids, encoded.attention_mask
        ids, mask = torch.as_tensor(ids), torch.as_tensor(mask)
    except (KeyError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Tokenizer must return input_ids and attention_mask tensors") from exc
    if ids.ndim != 2 or ids.shape[0] != len(descriptions) or ids.shape[1] < 1:
        raise ValueError("Tokenizer input_ids must have shape (number of POI descriptions, words)")
    if ids.is_floating_point() or ids.is_complex() or ids.dtype == torch.bool:
        raise ValueError("Tokenizer input_ids must contain integer token IDs")
    ids = ids.detach().to(device="cpu", dtype=torch.long).clone()
    if (ids < 0).any():
        raise ValueError("Tokenizer input_ids must contain nonnegative token IDs")
    if mask.shape != ids.shape or mask.is_complex():
        raise ValueError("Tokenizer attention_mask must match input_ids shape and contain 0 or 1")
    mask = mask.detach().to(device="cpu")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Tokenizer attention_mask must contain only 0 or 1")
    mask = mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("Tokenizer must produce at least one active token per POI description")
    return ids, mask


def prepare_sample(
    sample: Mapping[str, Any],
    tokenizer: Any,
    config: PreprocessConfig | None = None,
    *,
    require_target: bool = True,
) -> PreparedSample:
    """Return the CPU inputs consumed by ``BHENet.forward``.

    Accept an individual raw record, such as ``read_h5_sample`` returns, or
    ``DataLoader(..., batch_size=None)`` yields. Batched raw records are not
    supported. No source arrays or tensors are modified. In inference mode
    (``require_target=False``), BF1/BF2 are ignored and target is ``None``.
    """
    if not isinstance(sample, Mapping):
        raise TypeError("sample must be an individual building mapping")
    config = PreprocessConfig() if config is None else config
    if not isinstance(config, PreprocessConfig):
        raise TypeError("config must be a PreprocessConfig")
    if require_target and config.max_tokens is not None:
        raise ValueError("Training does not truncate tokens; max_tokens must be None")
    target = _target(sample) if require_target else None
    required = (
        "out_image", "out_label_0", "build_center", "pois_infor",
        "pois_cor_x", "pois_cor_y", "x_mat", "y_mat",
    )
    missing = [name for name in required if name not in sample]
    if missing:
        raise ValueError(f"Sample is missing required fields: {', '.join(missing)}")
    rasters = _prepare_rasters(sample, config)

    image = _raster(sample["out_image"], "out_image", 3)
    mean = torch.tensor(config.pixel_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(config.pixel_std, dtype=torch.float32).view(1, 3, 1, 1)
    image = F.interpolate(
        (image - mean) / std, size=(config.image_size, config.image_size),
        mode="bilinear", align_corners=False, antialias=False,
    )
    label = _raster(sample["out_label_0"], "out_label_0", 1)
    label[label > 0] = 2
    label[label == 0] = 1
    label = F.interpolate(label, size=(config.grid_size, config.grid_size), mode="nearest")

    # Historical DataLoader collation promotes the two Python centre values
    # to float64. Distances are computed with those values and then cast back.
    center = _float_tensor(sample["build_center"], "build_center", dtype=torch.float64)
    _shape(center, (2,), "build_center")
    x_mat = _float_tensor(sample["x_mat"], "x_mat", dtype=None)
    y_mat = _float_tensor(sample["y_mat"], "y_mat", dtype=None)
    for name, grid in (("x_mat", x_mat), ("y_mat", y_mat)):
        _shape(grid, (config.grid_size, config.grid_size), name)
    x_mat, y_mat = x_mat[None, None], y_mat[None, None]
    xy_mat_dist = (((x_mat - center[0:1]) ** 2 + (y_mat - center[1:2]) ** 2) ** 0.5).float()
    x_mat, y_mat = x_mat.float(), y_mat.float()

    descriptions = sample["pois_infor"]
    if isinstance(descriptions, (str, bytes)) or not isinstance(descriptions, Sequence):
        raise ValueError("pois_infor must be a sequence of text descriptions")
    if any(not isinstance(value, str) for value in descriptions):
        raise ValueError("pois_infor must contain only text descriptions")
    descriptions = list(descriptions)
    poi_x = _float_tensor(sample["pois_cor_x"], "pois_cor_x")
    poi_y = _float_tensor(sample["pois_cor_y"], "pois_cor_y")
    for name, coordinate in (("pois_cor_x", poi_x), ("pois_cor_y", poi_y)):
        if tuple(coordinate.shape) != (len(descriptions),):
            raise ValueError(f"{name} must match the number of POI descriptions")
    if not descriptions:
        descriptions = [_EMPTY_POI_TEXT]
        poi_x, poi_y = center[0:1].float(), center[1:2].float()

    ids, mask = _token_tensors(tokenizer, descriptions)
    words = ids.shape[1]
    active = mask.reshape(-1)
    input_ids = ids.reshape(-1)[active].unsqueeze(0)
    poi_x = poi_x[:, None].repeat(1, words).reshape(-1)[active][None, :, None]
    poi_y = poi_y[:, None].repeat(1, words).reshape(-1)[active][None, :, None]
    poi_dist = (((poi_x - center[0:1]) ** 2 + (poi_y - center[1:2]) ** 2) ** 0.5).float()
    token_count = input_ids.shape[1]

    if config.max_tokens is not None and token_count > config.max_tokens:
        input_ids = input_ids[:, :config.max_tokens]
        poi_x, poi_y, poi_dist = (value[:, :config.max_tokens] for value in (poi_x, poi_y, poi_dist))
    padding = config.min_tokens - input_ids.shape[1]
    if padding > 0:
        input_ids = F.pad(input_ids, (0, padding), value=config.padding_token_id)
        poi_x, poi_y, poi_dist = (
            F.pad(value, (0, 0, 0, padding), value=0) for value in (poi_x, poi_y, poi_dist)
        )
    inputs = (input_ids, image, x_mat, y_mat, xy_mat_dist, poi_x, poi_y, poi_dist, label)
    if any(not torch.isfinite(value).all() for value in inputs):
        raise ValueError("Preprocessing produced non-finite values")
    if config.raster_names:
        inputs = (*inputs, rasters)
    filename = sample.get("filename")
    return PreparedSample(
        inputs=inputs, target=target, identifier=sample.get("index"),
        filename=None if filename is None else str(filename), token_count=token_count,
    )
