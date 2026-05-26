"""midas-process-grains: pure-Python FF-HEDM grain-determination + strain.

Drop-in replacement for ``FF_HEDM/src/ProcessGrains.c``. Reads the binary
outputs of the upstream pipeline (``IndexBest{,Full}.bin``, ``FitBest.bin``,
``Key.bin``, ``OrientPosFit.bin``, ``ProcessKey.bin``) and emits the canonical
MIDAS grain artefacts (``Grains.csv``, ``SpotMatrix.csv``, ``GrainIDsKey.csv``).

Three operating modes (`mode=` kwarg):

  * ``"legacy"``      — bit-for-bit reproduce the current C ProcessGrains
                        output (used for regression tests during migration).
  * ``"paper_claim"`` — the §3.6 spec from the MIDAS methodology paper that
                        the current C code does not actually enforce
                        (90% shared peaks, 0.01° misorientation, 15 µm pos).
  * ``"spot_aware"``  — DEFAULT. Symmetry-aware row-aligned per-hkl SpotID
                        consistency, Jaccard pre-screen, union-of-cluster
                        emission, lstsq strain. No position gate.
"""

from __future__ import annotations

__version__ = "0.4.4"

from .params import ProcessGrainsParams, read_paramstest_pg

__all__ = [
    "__version__",
    "ProcessGrainsParams",
    "read_paramstest_pg",
]


def __getattr__(name):
    """Lazy import of pipeline-level symbols (avoid module cycles during build-up)."""
    if name == "ProcessGrains":
        from .pipeline import ProcessGrains
        return ProcessGrains
    if name == "ProcessGrainsResult":
        from .result import ProcessGrainsResult
        return ProcessGrainsResult
    raise AttributeError(f"module 'midas_process_grains' has no attribute {name!r}")
