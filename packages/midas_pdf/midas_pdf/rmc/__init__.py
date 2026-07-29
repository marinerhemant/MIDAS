"""Reverse Monte Carlo for disordered systems.

The small-box refiner (``midas_pdf.structure.refine_structure``) parametrises
a *crystal* — one unit cell, a handful of scalars. For glasses, liquids,
defective crystals, and other disordered systems we need a *configuration*
of atoms: an explicit supercell whose pair distances reproduce the measured
G(r).  RMC is the standard technique: start with a supercell, propose local
moves, accept/reject on χ² against the measured G(r).

This subpackage:

  * :class:`Supercell` — atoms (species, Cartesian positions, cell) with
    periodic-boundary-condition pair-distance evaluation.
  * :func:`supercell_G_r` — Gaussian-broadened G(r) forward model matching
    the ``pdffit_gr`` convention (rev-5 checkpoint).

Day 2+ (deferred) will add MC moves + Metropolis acceptance +
constraints + ensemble diagnostics.  See ``dev/RMC_PLAN.md``.
"""

from .supercell import Supercell
from .histogram import supercell_G_r, pair_distance_histogram
from .moves import (
    DisplaceMove, SwapMove, chi2_supercell, metropolis_step,
)
from .gc_moves import (
    InsertMove, RemoveMove, grand_canonical_metropolis_step,
)
from .cluster_moves import (
    ClusterDisplaceMove, RigidRotationMove, cluster_metropolis_step,
)
from .driver import rmc_refine, RMCResult
from .analysis import (
    partial_g_r, coordination_number, ergodicity_diagnostics,
    CoordinationBias,
)
from .ensemble import (
    RMCEnsembleResult, rmc_refine_ensemble,
    ensemble_partial_g_r, ensemble_coordination, ensemble_G_r,
)

__all__ = [
    "Supercell",
    "supercell_G_r",
    "pair_distance_histogram",
    "DisplaceMove", "SwapMove", "chi2_supercell", "metropolis_step",
    "InsertMove", "RemoveMove", "grand_canonical_metropolis_step",
    "ClusterDisplaceMove", "RigidRotationMove", "cluster_metropolis_step",
    "rmc_refine", "RMCResult",
    "partial_g_r", "coordination_number", "ergodicity_diagnostics",
    "CoordinationBias",
    "RMCEnsembleResult", "rmc_refine_ensemble",
    "ensemble_partial_g_r", "ensemble_coordination", "ensemble_G_r",
]
