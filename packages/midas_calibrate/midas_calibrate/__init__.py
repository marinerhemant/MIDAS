"""midas-calibrate — native Python/Torch detector calibration.

Public API:

    from midas_calibrate import CalibrationParams, build_ring_table, refine_geometry

    params = CalibrationParams.from_file("calib.txt")
    rt = build_ring_table(params)
    result = autocalibrate(params)            # full pipeline
"""

# Register the HDF5 filter plugins (bitshuffle, LZ4, zstd) that Eiger / Dectris
# and ESRF files are written with.  hdf5plugin ships the plugin binaries but
# only registers them with the HDF5 library when it is imported — declaring it
# as a dependency is not enough.  Without this, reading such a dataset fails
# with "can't open directory (/usr/local/lib/plugin)".
try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

from .params import CalibrationParams
from .rings import RingTable, build_ring_table
from .refine import FittedPoint, RefineResult, refine_geometry
from .orchestrator import CalibrationResult, IterRecord, autocalibrate
from .estep import CakeProfile, integrate_cake, run_estep

__version__ = "0.4.2"

__all__ = [
    "CakeProfile",
    "CalibrationParams",
    "CalibrationResult",
    "FittedPoint",
    "IterRecord",
    "RefineResult",
    "RingTable",
    "autocalibrate",
    "build_ring_table",
    "integrate_cake",
    "refine_geometry",
    "run_estep",
]
