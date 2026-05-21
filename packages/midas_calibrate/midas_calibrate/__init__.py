"""midas-calibrate — native Python/Torch detector calibration.

Public API:

    from midas_calibrate import CalibrationParams, build_ring_table, refine_geometry

    params = CalibrationParams.from_file("calib.txt")
    rt = build_ring_table(params)
    result = autocalibrate(params)            # full pipeline
"""
from .params import CalibrationParams
from .rings import RingTable, build_ring_table
from .refine import FittedPoint, RefineResult, refine_geometry
from .orchestrator import CalibrationResult, IterRecord, autocalibrate
from .estep import CakeProfile, integrate_cake, run_estep

__version__ = "0.2.4"

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
