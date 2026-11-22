"""Read one building per HDF5 file with configured auxiliary raster inputs.

Stored dataset    Shape           Meaning
BF1, BF2          scalar text     Background / building-use labels
index             scalar          Building identifier
pois_infor        (P,) text        Ordered POI descriptions
build_center      (2,)            Normalized building-centre coordinates
pois_cor_x/y      (P,)            Normalized coordinates aligned with POI text
x_mat, y_mat      (G, G)          Normalized image-position grids (research G=64)
out_image         (3, H, W)       Stored RGB crop in channel-first order
out_label_0       (1, Ha, Wa)      Stored single-channel building raster
out_road          (1, Hr, Wr)      Binary road occupancy (required by default)
out_{name}        (C, Hn, Wn)      Each additional configured raster branch

Each returned sample retains the stored labels, order, values and numeric
dtypes. Position grids are torch tensors, raster arrays are NumPy arrays, and
coordinates are lists, matching the research HDF5 reader. Image and auxiliary
crop sizes may differ, but all rasters must cover the same crop extent and
have the same orientation as the RGB image. Numeric shape checks cannot verify
that geographic alignment; the data producer must ensure it. Named rasters are
returned in ``sample['rasters']`` in configured order. The filename is included
for sample identification.
The example in README describes the link from these fields to model tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import torch
from torch.utils.data import Dataset

from .config import normalize_raster_spec


class BuildingSample(TypedDict):
    BF1: str
    BF2: str
    index: Any
    pois_infor: list[str]
    build_center: list[float]
    pois_cor_x: list[float]
    pois_cor_y: list[float]
    x_mat: torch.Tensor
    y_mat: torch.Tensor
    out_image: Any
    out_label_0: Any
    rasters: dict[str, Any]
    filename: str


_REQUIRED = (
    "BF1", "BF2", "index", "pois_infor", "build_center", "pois_cor_x",
    "pois_cor_y", "x_mat", "y_mat", "out_image", "out_label_0",
)


def read_h5_sample(
    filename: str | Path,
    *,
    raster_names: Sequence[str] = ("road",),
    raster_channels: Sequence[int] = (1,),
) -> BuildingSample:
    """Read and validate a building record, opening the HDF5 file read-only.

    A fresh file handle is closed before return, permitting worker-local reads
    with PyTorch DataLoader. Numeric values and labels (including literal
    ``'None'`` in BF2) are preserved. Each configured name requires its own
    ``out_{name}`` dataset. Road occupancy must contain finite binary 0/1
    values. Other named rasters retain their finite numeric values and dtypes.
    Empty raster-name/channel sequences select the original building-only
    schema. Missing rasters are reported rather than synthesized.
    """
    raster_names, raster_channels = normalize_raster_spec(raster_names, raster_channels)
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise ImportError("HDF5 loading requires the 'data' extra: pip install '.[data]'") from exc

    path = Path(filename).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Building HDF5 file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        required = _REQUIRED + tuple(f"out_{name}" for name in raster_names)
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path}: missing HDF5 datasets: {', '.join(missing)}")
        if any(not isinstance(handle[name], h5py.Dataset) for name in required):
            raise ValueError(f"{path}: required fields must be HDF5 datasets")

        def text(name: str, ndim: int):
            field = handle[name]
            if field.ndim != ndim or h5py.check_string_dtype(field.dtype) is None:
                raise ValueError(f"{path}: {name} must be a {ndim}-dimensional text dataset")
            return field.asstr(encoding="utf-8")[()]

        bf1, bf2 = str(text("BF1", 0)), str(text("BF2", 0))
        descriptions = text("pois_infor", 1).tolist()
        if handle["index"].ndim != 0:
            raise ValueError(f"{path}: index must be a scalar identifier")
        identifier = handle["index"][()]
        if isinstance(identifier, np.generic):
            identifier = identifier.item()
        if isinstance(identifier, bytes):
            identifier = identifier.decode("utf-8")

        arrays = {name: handle[name][()] for name in _REQUIRED[4:]}
        for name, value in arrays.items():
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
                raise ValueError(f"{path}: {name} must be a real numeric array")
        if arrays["build_center"].shape != (2,):
            raise ValueError(f"{path}: build_center must have shape (2,)")
        for name in ("pois_cor_x", "pois_cor_y"):
            if arrays[name].shape != (len(descriptions),):
                raise ValueError(f"{path}: {name} must match the number of POI descriptions")
        grid = arrays["x_mat"]
        if grid.ndim != 2 or min(grid.shape) < 1 or grid.shape[0] != grid.shape[1]:
            raise ValueError(f"{path}: x_mat must be a nonempty square position grid")
        if arrays["y_mat"].shape != grid.shape:
            raise ValueError(f"{path}: x_mat and y_mat must have the same shape")
        for name, channels in (("out_image", 3), ("out_label_0", 1)):
            value = arrays[name]
            if value.ndim != 3 or value.shape[0] != channels or min(value.shape) < 1:
                raise ValueError(f"{path}: {name} must have nonempty shape ({channels}, H, W)")

        rasters = {}
        for name, channels in zip(raster_names, raster_channels):
            field = f"out_{name}"
            value = handle[field][()]
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
                raise ValueError(f"{path}: {field} must be a real numeric array")
            if value.ndim != 3 or value.shape[0] != channels or min(value.shape) < 1:
                raise ValueError(f"{path}: {field} must have nonempty shape ({channels}, H, W)")
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: {field} must contain only finite values")
            if name == "road" and not ((value == 0) | (value == 1)).all():
                raise ValueError(f"{path}: out_road must contain binary occupancy values 0 or 1")
            rasters[name] = value

    return BuildingSample(
        BF1=bf1, BF2=bf2, index=identifier, pois_infor=descriptions,
        build_center=arrays["build_center"].tolist(),
        pois_cor_x=arrays["pois_cor_x"].tolist(),
        pois_cor_y=arrays["pois_cor_y"].tolist(),
        x_mat=torch.from_numpy(arrays["x_mat"]), y_mat=torch.from_numpy(arrays["y_mat"]),
        out_image=arrays["out_image"], out_label_0=arrays["out_label_0"],
        rasters=rasters, filename=str(path),
    )


class H5BuildingDataset(Dataset):
    """Explicitly ordered collection of per-building HDF5 files.

    ``files`` is an ordered sequence of paths. Files are read lazily in
    ``__getitem__`` and variable-length text/crop arrays retain their shapes.
    Use ``DataLoader(dataset, batch_size=None)`` for individual-record iteration.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        *,
        raster_names: Sequence[str] = ("road",),
        raster_channels: Sequence[int] = (1,),
    ) -> None:
        if isinstance(files, (str, bytes, Path)) or not isinstance(files, Sequence):
            raise TypeError("files must be a sequence of individual HDF5 paths")
        self.files = tuple(Path(filename).expanduser() for filename in files)
        self.raster_names, self.raster_channels = normalize_raster_spec(raster_names, raster_channels)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> BuildingSample:
        return read_h5_sample(
            self.files[index], raster_names=self.raster_names, raster_channels=self.raster_channels,
        )
