"""midas-fit-grain: PyTorch FF-HEDM grain refiner.

Drop-in replacement for the C executables ``FitPosOrStrainsOMP`` /
``FitPosOrStrainsGPU``. Public API is intentionally small; everything else lives
under module-private submodules.
"""

__version__ = "0.5.1"

from .config import FitConfig
from .device import resolve_device, resolve_dtype
from .io_binary import (
    ExtraInfoSpot,
    GrainResult,
    PerSpotFit,
    read_extra_info,
    read_orient_pos_fit,
    read_fit_best,
    read_key,
    write_orient_pos_fit_row,
    write_fit_best_row,
    write_key_row,
    write_process_key_row,
)
from .batch import MatchBatch, ObservedBatch, batch_residuals
from .matching import MatchResult, associate, ring_slot_lookup
from .observations import ObservedSpots
from .refine import GrainFitResult, refine_grain
from .refine_block import BlockFitResult, refine_block
from .residuals import grain_residuals
from .spec_residual import HEDMResidualBundle, hedm_spot_residual

__all__ = [
    "FitConfig",
    "HEDMResidualBundle",
    "hedm_spot_residual",
    "ExtraInfoSpot",
    "GrainResult",
    "PerSpotFit",
    "MatchResult",
    "MatchBatch",
    "ObservedBatch",
    "ObservedSpots",
    "GrainFitResult",
    "BlockFitResult",
    "associate",
    "batch_residuals",
    "grain_residuals",
    "refine_block",
    "read_extra_info",
    "read_orient_pos_fit",
    "read_fit_best",
    "read_key",
    "refine_grain",
    "ring_slot_lookup",
    "write_orient_pos_fit_row",
    "write_fit_best_row",
    "write_key_row",
    "write_process_key_row",
    "resolve_device",
    "resolve_dtype",
    "__version__",
]
