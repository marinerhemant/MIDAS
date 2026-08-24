"""midas-tomo: gridrec filtered back-projection CT reconstruction.

Wraps the MIDAS gridrec engine (a Fourier-domain FBP using prolate spheroidal
wave function interpolation) behind a NumPy API. The heavy lifting is done by
the bundled ``MIDAS_TOMO`` binary, compiled at install time from the vendored
C in ``c_src/``; ``MIDAS_TOMO_GPU`` is built too when CUDA is available.

Quick start::

    from midas_tomo import run_tomo
    recon = run_tomo(data, dark, whites, "/path/to/work", thetas, shifts=1.0)

If the binary could not be built (no compiler, or missing FFTW / HDF5 /
OpenMP) the package still imports; :func:`midas_tomo.backend_c.available`
returns ``False`` and :func:`midas_tomo.backend_c.why_unavailable` explains
what to install.
"""

from __future__ import annotations

# Register the HDF5 filter plugins (bitshuffle, LZ4, zstd) that Eiger / Dectris
# and ESRF files are written with.  hdf5plugin ships the plugin binaries but
# only registers them with the HDF5 library when it is imported — declaring it
# as a dependency is not enough.  Without this, reading such a dataset fails
# with "can't open directory (/usr/local/lib/plugin)".
try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

__version__ = "0.2.0"

from pathlib import Path

from . import backend_c
from .api import read_recon_cube, run_tomo, run_tomo_from_sinos, write_thetas
from .center import find_center, sharpness, shift_values_for_search
from .cleanup import (
    default_cleanup_grid,
    load_cleanup_grid,
    ring_metric,
    run_tomo_cleanup_sweep,
)
from .config import FILTERS, TomoConfig, next_power_of_2, parse_shift_arg

__all__ = [
    "__version__",
    "FILTERS",
    "TomoConfig",
    "backend_c",
    "c_src_dir",
    "default_cleanup_grid",
    "find_center",
    "load_cleanup_grid",
    "next_power_of_2",
    "parse_shift_arg",
    "read_recon_cube",
    "ring_metric",
    "run_tomo",
    "run_tomo_cleanup_sweep",
    "run_tomo_from_sinos",
    "sharpness",
    "shift_values_for_search",
    "write_thetas",
]


def c_src_dir() -> Path:
    """Directory holding the vendored gridrec C sources.

    Mirrors the ``midas_ckernel.c_src_dir()`` convention so a downstream
    scikit-build-core package can compile against these sources instead of
    re-vendoring them::

        import midas_tomo
        include_dir = midas_tomo.c_src_dir()

    The sources ship in the sdist. They may be absent from a wheel built with
    ``sdist.include`` disabled, so callers should check ``.is_dir()``.
    """
    return Path(__file__).resolve().parent.parent / "c_src"
