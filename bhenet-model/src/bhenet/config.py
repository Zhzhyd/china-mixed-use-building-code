"""Dimensions and structural configuration of the model."""

from dataclasses import dataclass
from collections.abc import Sequence
import re


def normalize_raster_spec(names, channels) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Validate ordered additional raster branches, accepting tuples or JSON lists."""
    for name, value in (("raster_names", names), ("raster_channels", channels)):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"{name} must be an ordered sequence")
    names, channels = tuple(names), tuple(channels)
    if len(names) != len(channels):
        raise ValueError("raster_names and raster_channels must have the same length")
    for name in names:
        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise ValueError("Raster names must start with a lowercase letter and contain lowercase letters, digits or underscores")
        if name in {"building", "label_0", "image"}:
            raise ValueError(f"Raster name {name!r} is reserved")
    if len(set(names)) != len(names):
        raise ValueError("raster_names must not contain duplicates")
    for value in channels:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("raster_channels must contain positive integers")
    return names, channels


@dataclass(frozen=True)
class ModelConfig:
    """Dimensions and ordered raster stages; smaller dimensions support unit tests.

    The research configuration has 256 features, eight attention heads,
    a 64-by-64 feature map, one building-raster channel and 29 output classes.
    The default adds a road stage after building fusion. Additional raster stages
    follow the configured order. Empty raster tuples retain the original stages.
    """

    feature_dim: int = 256
    head_dim: int = 32
    grid_size: int = 64
    aux_channels: int = 1
    num_classes: int = 29
    raster_names: tuple[str, ...] = ("road",)
    raster_channels: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        for name in ("feature_dim", "head_dim", "grid_size", "aux_channels", "num_classes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.feature_dim % self.head_dim:
            raise ValueError("feature_dim must be divisible by head_dim")
        names, channels = normalize_raster_spec(self.raster_names, self.raster_channels)
        object.__setattr__(self, "raster_names", names)
        object.__setattr__(self, "raster_channels", channels)
