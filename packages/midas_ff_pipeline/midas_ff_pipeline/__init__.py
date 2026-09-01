"""midas-ff-pipeline — RETIRED. A thin argv shim over ``midas-pipeline``.

.. deprecated:: 0.4.0
   ``midas-ff-pipeline`` has been consolidated into ``midas-pipeline`` — FF is
   the single-scan degeneracy of PF, and one orchestrator covers both. Use
   ``midas-pipeline run --scan-mode ff …`` (CLI) or
   ``midas_pipeline.Pipeline(config_with_ScanGeometry.ff())`` (API).
   This package is removed at 1.0.0.

**There is no longer a public API here.** ``Pipeline``, ``PipelineConfig``,
``DetectorConfig``, ``LayerSelection``, ``MachineConfig`` and ``LayerResult``
were re-exported until 0.7.0 and are gone; import them from ``midas_pipeline``,
which is where the maintained implementations always were. Keeping the
re-exports would have kept the whole stage tree alive behind them, which is the
two-places-both-half-right trap this consolidation exists to close.

What remains: :func:`midas_ff_pipeline.cli.main` (argv translation → delegate)
and the :mod:`midas_ff_pipeline.testing` re-export shim.
"""
from __future__ import annotations

try:  # pragma: no cover - environment-dependent
    import hdf5plugin as _hdf5plugin  # noqa: F401
except ImportError:  # HDF5 files with no plugin filter still read fine
    pass

import warnings as _warnings
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("midas-ff-pipeline")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.0.0.dev0"

_warnings.warn(
    "midas-ff-pipeline is retired and now only translates argv to "
    "midas-pipeline. Use `midas-pipeline run --scan-mode ff` directly; "
    "this package is removed at 1.0.0.",
    DeprecationWarning, stacklevel=2,
)

__all__ = ["__version__"]
