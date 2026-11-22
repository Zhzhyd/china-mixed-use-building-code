"""BHENet architecture and per-building HDF5 sample interface."""

from .config import ModelConfig
from .dataloader import H5BuildingDataset, read_h5_sample
from .fusion import FusionHead, RasterFusionStage
from .labels import CLASS_CODES
from .model import BHENet, build_bhenet
from .preprocessing import PreprocessConfig, PreparedSample, prepare_sample

__version__ = "0.1.0"
__all__ = ["BHENet", "CLASS_CODES", "FusionHead", "H5BuildingDataset", "ModelConfig", "build_bhenet", "read_h5_sample",
           "PreprocessConfig", "PreparedSample", "prepare_sample", "RasterFusionStage"]
