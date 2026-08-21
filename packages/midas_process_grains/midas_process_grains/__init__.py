"""midas-process-grains: pure-Python FF-HEDM grain-determination + strain.

Drop-in replacement for ``FF_HEDM/src/ProcessGrains.c``. Reads the binary
outputs of the upstream pipeline (``IndexBest{,Full}.bin``, ``FitBest.bin``,
``Key.bin``, ``OrientPosFit.bin``, ``ProcessKey.bin``) and emits the canonical
MIDAS grain artefacts (``Grains.csv``, ``SpotMatrix.csv``, ``GrainIDsKey.csv``).

Operating modes (`mode=` kwarg):

  * ``"c_parity"``    — **DEFAULT**, and what ``midas-pipeline`` runs. Replicates
                        C ProcessGrains (datasetA Ni: 6150 grains vs C's 6138,
                        matched pairs agreeing to 0.0000° and 0.000 µm) and is
                        the most accurate against EBSD. Dispatched by the CLI to
                        ``compute.c_parity_run``, not through ``ProcessGrains.run``.
  * ``"legacy"``      — bit-for-bit reproduce the current C ProcessGrains
                        output (used for regression tests during migration).
  * ``"paper_claim"`` — the §3.6 spec from the MIDAS methodology paper that
                        the current C code does not actually enforce
                        (90% shared peaks, 0.01° misorientation, 15 µm pos).
  * ``"adaptive"``    — the spot-aware pipeline with the misorientation
                        threshold derived at run time (see ``modes.py``).
  * ``"spot_aware"``  — **DISABLED; never runs.** Every entry point raises
                        ``SPOT_AWARE_DISABLED``. Against EBSD on shade_LSHR it
                        traded −11.6 pp of precision for +0.1 pp of recall, and
                        on a 20-ID alumina rod it placed 4.1 % of its grains
                        outside the physical sample. *Why* it does this is not
                        diagnosed — it is disabled on its output, not on a root
                        cause, so do not re-enable it on one dataset looking
                        better. Use ``"c_parity"``.

Every mode that reads ``FitBest.bin`` also writes the signed per-spot residual
sidecar ``processgrains_diagnostics.h5:/residuals`` — ``c_parity`` included as
of 0.9.2, which closed the gap where the *default* mode produced none.
``"legacy"`` never reads FitBest so its residuals are empty by design, and the
CLI-only ``mode="physics"`` (``v4_pipeline``) writes none at all. See the
README for the schema and the per-mode table.
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

__version__ = "0.10.0"

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
