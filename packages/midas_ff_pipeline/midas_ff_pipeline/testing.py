"""Back-compat shim — synthetic FF dataset helpers moved to midas-pipeline.

.. deprecated:: 0.4.0
   These helpers now live in :mod:`midas_pipeline.testing`. Imports from
   ``midas_ff_pipeline.testing`` continue to work but emit
   :class:`DeprecationWarning`. After the 1.0.0 removal of
   ``midas-ff-pipeline``, use ``from midas_pipeline.testing import ...``
   directly.

This shim re-exports the three public entry points
(``generate_synthetic_dataset``, ``generate_pinwheel_synthetic_dataset``,
``generate_multidet_synthetic_dataset``) plus the ``_find_midas_home`` helper
that downstream callers (parity-gate scripts, notebook builders) sometimes
reach into.
"""

from __future__ import annotations

import warnings as _warnings

# Re-export the canonical implementations.
from midas_pipeline.testing import (  # noqa: F401
    generate_synthetic_dataset,
    generate_pinwheel_synthetic_dataset,
    generate_multidet_synthetic_dataset,
    _find_midas_home,
)

_warnings.warn(
    "midas_ff_pipeline.testing has moved to midas_pipeline.testing. "
    "Update your imports to `from midas_pipeline.testing import ...`. "
    "This shim will be removed when midas-ff-pipeline 1.0.0 deletes the "
    "package.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "generate_synthetic_dataset",
    "generate_pinwheel_synthetic_dataset",
    "generate_multidet_synthetic_dataset",
]
