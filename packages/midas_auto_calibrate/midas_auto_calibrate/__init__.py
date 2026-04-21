"""midas-auto-calibrate: Fully automated detector geometry calibration.

Wraps the MIDAS `MIDASCalibrant` binary (shipped inside the wheel) to run
end-to-end calibration of synchrotron area detectors against a powder calibrant
without manual peak picking or ring assignment.
"""

__version__ = "0.1.0"

from . import data
from ._binaries import MidasBinaryNotFoundError, midas_bin
from ._config import CalibrationConfig, write_params_file
from .benchmark import BenchmarkResult, benchmark
from .calibrate import CalibrationResult, run_calibration
from .geometry import DetectorGeometry
from .progressive import ProgressiveResult, calibrate_progressive

# Alias for ergonomic top-level use: `mac.auto_calibrate(...)` reads better
# than `mac.run_calibration(...)` as the package's headline verb.
auto_calibrate = run_calibration

__all__ = [
    "__version__",
    "BenchmarkResult",
    "CalibrationConfig",
    "CalibrationResult",
    "DetectorGeometry",
    "MidasBinaryNotFoundError",
    "ProgressiveResult",
    "auto_calibrate",
    "benchmark",
    "calibrate_progressive",
    "data",
    "midas_bin",
    "run_calibration",
    "write_params_file",
]
