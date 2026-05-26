"""midas-uq: Cross-validation based uncertainty quantification for HEDM grain refinement.

Three diagnostics, three modalities (far-field, point-focused, near-field):

  * `half_half`           - random K-split resampling -> empirical covariance
  * `jackknife`           - leave-one-out -> per-observation influence
  * `laplace_covariance`  - inverse-Hessian Gaussian posterior approximation
  * `rfree_gap`           - train/holdout loss gap, refinement R_free analog

For FF and pf, the observation unit is a measured spot. For NF, the
observation unit is an omega frame in the detector image stack. The
public API is symmetric across modalities; the difference is the
`mode` argument.

Companion paper:
    Sharma, H., Park, J.-S., Kenesei, P., Andrejevic, N. & Cherukara, M.
    (2026). Cross-Validation Based Uncertainty Quantification for HEDM
    Grain Refinement. IUCrJ (in prep).

Builds on the differentiable forward model of paper I:
    Sharma, Andrejevic, Zhang, Cherukara (2026). An End-to-End
    Differentiable Forward Model for HEDM. IUCrJ (in press).
"""

__version__ = "0.1.0"

from ._common import GrainState, misori_deg, misori_deg_sym
from .spots import (
    half_half_spots,
    jackknife_spots,
    HalfHalfResult,
    JackknifeResult,
)
from .images import (
    half_half_frames,
    jackknife_frames,
)
from .laplace import laplace_covariance, LaplaceResult
from .rfree import rfree_gap

# Fixed-assignment UQ — the correct shape for scoring already-refined grains
# (default half_half / jackknife do dynamic re-association at every step,
# which is fine for initial refinement but degenerate for refined inputs).
from .fixed_assignment import (
    freeze_associations,
    per_spot_residuals,
    fit_grain_spots_fixed,
    half_half_fixed,
    jackknife_fixed,
    trust_score,
    ResidualsResult,
    HalfHalfFixedResult,
    JackknifeFixedResult,
    TrustScore,
)

# Refiner-anchored UQ — read the refiner's own per-spot predictions from
# FitBest.bin, no forward-model re-prediction. Fastest, no convention drift,
# recommended for population-scale auditing of an already-refined recon.
from .refiner_anchored import (
    per_grain_residuals,
    bootstrap_uq,
    trust_score_anchored,
    PerGrainResiduals,
    BootstrapUQ,
    TrustScoreAnchored,
)

# Generic dispatch entry points (mode-aware)
from .api import half_half, jackknife

__all__ = [
    "__version__",
    "GrainState",
    "misori_deg",
    "misori_deg_sym",
    # Generic dispatch
    "half_half",
    "jackknife",
    # FF / pf (spot-based)
    "half_half_spots",
    "jackknife_spots",
    # NF (frame-based)
    "half_half_frames",
    "jackknife_frames",
    # Other diagnostics
    "laplace_covariance",
    "rfree_gap",
    # Fixed-assignment UQ (recommended for refined-grain trust scoring)
    "freeze_associations",
    "per_spot_residuals",
    "fit_grain_spots_fixed",
    "half_half_fixed",
    "jackknife_fixed",
    "trust_score",
    # Refiner-anchored UQ
    "per_grain_residuals",
    "bootstrap_uq",
    "trust_score_anchored",
    "PerGrainResiduals",
    "BootstrapUQ",
    "TrustScoreAnchored",
    # Result types
    "HalfHalfResult",
    "JackknifeResult",
    "LaplaceResult",
    "ResidualsResult",
    "HalfHalfFixedResult",
    "JackknifeFixedResult",
    "TrustScore",
]
