"""Reconstruction backends for pf-HEDM (FBP, MLEM, OS-EM, voxelmap).

Three backend families:

- ``fbp_recon`` — thin wrapper around ``TOMO.midas_tomo_python.run_tomo_from_sinos``.
  The TOMO module is **not relocated** (people use it standalone);
  this module just imports it via a ``sys.path`` hop.
- ``mlem_recon`` / ``osem_recon`` — full port of ``utils/mlem_recon.py``.
  Tensor inputs flow through a differentiable, device-portable torch
  path; ndarray inputs run the legacy NumPy code verbatim. The
  original ``utils/mlem_recon.py`` stays in place untouched.
- ``voxelmap_recon`` — port of ``pf_MIDAS.py:voxelmap_recon`` that
  swaps ``calcMiso`` for ``midas_stress.orientation.misorientation_om_batch``
  per the repo-wide orientation-primitive rule.
"""

from __future__ import annotations

from .fbp import fbp_recon, fbp_recon_per_grain
# NOT re-exported: the legacy alias ``mlem`` (= mlem_recon). A package attribute named ``mlem`` shadows
# the ``recon.mlem`` submodule, so ``import midas_pipeline.recon.mlem as M`` bound the FUNCTION and
# ``M.mlem_recon`` raised. The alias still lives in the module: ``from midas_pipeline.recon.mlem import mlem``.
from .mlem import (
    back_project,
    forward_project,
    mlem_recon,
    osem,
    osem_recon,
)
from .voxelmap import voxelmap_recon

__all__ = [
    "fbp_recon",
    "fbp_recon_per_grain",
    "mlem_recon",
    "osem_recon",
    "osem",         # legacy alias
    "forward_project",
    "back_project",
    "voxelmap_recon",
]
