"""midas-integrate: radial integration for synchrotron area detectors.

High-level API (fleshed out across v0.1.0 weeks 8-12):
    - ``Integrator``      wraps ``MIDASIntegrator`` (CPU + GPU backends).
    - ``correct_image``   produces a geometrically rectified TIFF for
                           Dioptas/pyFAI/GSAS-II interop.
    - ``Mapper``          pre-computes pixel→(R,η) binning.
    - ``benchmark``       pyFAI head-to-head harness.
    - ``stream.Server``   socket-based streaming integration server.

See the release plan under ``/Users/hsharma/.claude/plans/`` for the
delivery order.
"""

__version__ = "0.1.0"

from . import stream
from ._binaries import MidasBinaryNotFoundError, midas_bin
from ._config import IntegrationConfig, write_params_file
from .correct import (
    Panel,
    correct_image,
    correct_images,
    generate_panels,
    load_panel_shifts,
    write_tiff,
)
from .integrate import IntegrationResult, Integrator
from .io import make_zarr_zip
from .mapper import MapArtifacts, MapHeader, Mapper
from .peakfit import fit_peaks_1d, load_peaks_h5, pseudo_voigt

__all__ = [
    "__version__",
    "IntegrationConfig",
    "IntegrationResult",
    "Integrator",
    "MapArtifacts",
    "MapHeader",
    "Mapper",
    "MidasBinaryNotFoundError",
    "Panel",
    "correct_image",
    "correct_images",
    "fit_peaks_1d",
    "generate_panels",
    "load_panel_shifts",
    "load_peaks_h5",
    "make_zarr_zip",
    "midas_bin",
    "pseudo_voigt",
    "stream",
    "write_params_file",
    "write_tiff",
]
