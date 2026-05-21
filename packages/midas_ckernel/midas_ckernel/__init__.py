"""midas_ckernel — shared C kernels for the MIDAS c-omp indexer and refiner.

This package ships ONE copy of the diffraction forward model and the linear
-algebra / optimizer primitives that both ``midas_index`` (orientation search)
and ``midas_fit_grain`` (position + strain fit) compile against. It exposes no
runtime Python API beyond locating its bundled C sources so downstream
scikit-build-core builds can ``#include`` and compile them.

Bundled sources (``midas_ckernel.c_src_dir()``):

================================  ============================================
forward.{c,h}                     unified diffraction forward simulator
nelder_mead.{c,h}                 NLopt-free bounded Nelder-Mead simplex
MIDAS_Math.{c,h}                  linear-algebra primitives
GetMisorientation.{c,h}           cubic-symmetry misorientation
IndexerConsolidatedIO.h           shared I/O struct/enum definitions
MIDAS_Limits.h                    global array-size limits
================================  ============================================

The forward model is verified BIT-IDENTICAL to the legacy indexer
``CalcDiffrSpots`` and equal-to-ULP (~3e-9) with the legacy refiner
``CalcDiffractionSpots`` — see ``tests/parity_test.c``.
"""
from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

__all__ = ["c_src_dir", "headers", "sources"]

_C_SRC = Path(__file__).resolve().parent.parent / "c_src"

# Public C sources (compiled by downstream builds). forward.c pulls in
# MIDAS_Math.c; nelder_mead.c / GetMisorientation.c are independent.
_SOURCES = ("forward.c", "MIDAS_Math.c", "nelder_mead.c", "GetMisorientation.c")
_HEADERS = (
    "forward.h",
    "nelder_mead.h",
    "MIDAS_Math.h",
    "GetMisorientation.h",
    "IndexerConsolidatedIO.h",
    "MIDAS_Limits.h",
)


def c_src_dir() -> str:
    """Absolute path to the bundled ``c_src`` directory (for ``-I`` include)."""
    return str(_C_SRC)


def headers() -> list[str]:
    """Absolute paths to the bundled public headers."""
    return [str(_C_SRC / h) for h in _HEADERS]


def sources() -> list[str]:
    """Absolute paths to the bundled compilable C sources."""
    return [str(_C_SRC / s) for s in _SOURCES]
