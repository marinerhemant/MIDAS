"""CPFEM coupling and Tier-1 self-consistent elastic predictions.

The ``damask_io`` / ``fepx_io`` / ``prisms_io`` stubs are intentionally
signature-only — they give collaborators a fixed import surface to target.

The ``elastic_sc`` module implements the Tier-1 self-consistent elastic
prediction: per-grain σ, ε, and U under macroscopic load, in Reuss /
Voigt / Hill-Eshelby self-consistent bounds. No fitting parameters —
inputs are grain orientations + single-crystal stiffness + macroscopic
stress. Used in the demk Cu-Al paper to test whether orientation alone
predicts the matrix-twin elastic-energy asymmetry.
"""

from . import damask_io, fepx_io, prisms_io
from .elastic_sc import (
    per_grain_lab_stiffness,
    hill_average_isotropic,
    reuss_per_grain,
    voigt_per_grain,
    kroner_self_consistent,
    per_grain_energy,
)

__all__ = [
    "damask_io",
    "fepx_io",
    "prisms_io",
    "per_grain_lab_stiffness",
    "hill_average_isotropic",
    "reuss_per_grain",
    "voigt_per_grain",
    "kroner_self_consistent",
    "per_grain_energy",
]
