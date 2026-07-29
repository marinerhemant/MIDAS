"""Per-grain ODF parameterizations.

Three first-class implementations behind the GrainODF interface:
  - ParticleODF       (4a)  K orientations + simplex weights
  - BinghamMixtureODF (4b)  Mixture of Bingham distributions on quaternions
  - VoxelGridODF      (4c)  Regular angle-axis grid with node weights

All three live in a small angle-axis ball anchored at a grain-averaged
orientation R_avg. The ball half-width (theta_max) bounds the trust region.

Orientation primitives (axis-angle <-> matrix, matrix -> quaternion) are
delegated to ``midas_stress.orientation``: a single tested source of truth
across the MIDAS toolkit.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from midas_stress.orientation import (
    axis_angle_to_orient_mat,
    orient_mat_to_quat,
)


# ---------------------------------------------------------------------------
#  SO(3) helpers (thin wrappers around midas_stress)
# ---------------------------------------------------------------------------


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) -> rotation matrix (..., 3, 3).

    Convention: ``norm(axis_angle)`` is the rotation angle in radians;
    ``axis_angle / norm`` is the unit axis. Delegates to
    ``midas_stress.orientation.axis_angle_to_orient_mat`` (torch backend) for
    the actual Rodrigues formula.
    """
    eps = 1e-9
    norm = axis_angle.norm(dim=-1, keepdim=True).clamp_min(eps)
    axis = axis_angle / norm
    angle_deg = norm.squeeze(-1) * (180.0 / math.pi)
    R = axis_angle_to_orient_mat(axis, angle_deg)
    # At zero rotation, axis_angle / eps is arbitrary; identity is the correct limit.
    near_zero = (norm.squeeze(-1) < 10.0 * eps).unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return torch.where(near_zero, I.expand_as(R), R)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) -> rotation vector (..., 3) (axis * angle).

    Used for diagnostics and visualization (e.g., projecting recovered ODF
    samples into axis-angle space). Computed via the trace + asymmetric-part
    formula; midas_stress does not yet expose a torch-batch matrix→axis-angle
    helper but exposes matrix→quaternion, which we could route through if
    the future need arises.
    """
    cos_t = torch.clamp(
        0.5 * (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0),
        -1.0 + 1e-7, 1.0 - 1e-7,
    )
    theta = torch.acos(cos_t)
    sin_t = torch.sin(theta)
    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    near_zero = theta < 1e-7
    factor = torch.where(
        near_zero,
        torch.ones_like(theta),
        theta / (2.0 * sin_t + 1e-12),
    )
    return axis * factor.unsqueeze(-1)


# ---------------------------------------------------------------------------
#  Base interface
# ---------------------------------------------------------------------------


class GrainODF(nn.Module):
    """Common interface for per-grain ODF parameterizations.

    Subclasses must implement ``sample()``. ``support()`` and ``density()``
    are optional.
    """

    R_avg: torch.Tensor  # (3, 3)
    theta_max: float

    def __init__(self, R_avg: torch.Tensor, theta_max: float = math.radians(5.0)):
        super().__init__()
        self.register_buffer("R_avg", R_avg.detach().clone().to(torch.float64))
        self.theta_max = float(theta_max)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (R: (K, 3, 3), w: (K,)) -- orientations and simplex weights."""
        raise NotImplementedError

    @property
    def K(self) -> int:
        """Number of orientation samples returned by sample()."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  4a. Particle ODF
# ---------------------------------------------------------------------------


class ParticleODF(GrainODF):
    """ODF as K free orientation particles with softmax weights.

    Each particle is parameterized by a 3-vector axis-angle perturbation
    around R_avg. Weight logits are softmaxed to a simplex.

    Initialization: K particles uniformly sampled inside the trust region,
    weights uniform.
    """

    def __init__(
        self,
        R_avg: torch.Tensor,
        K: int = 64,
        theta_max: float = math.radians(5.0),
        seed: Optional[int] = None,
        init_axis_angle: Optional[torch.Tensor] = None,
    ):
        """Initialize K particles in axis-angle space around R_avg.

        Default init: uniform direction × uniform magnitude in the
        theta_max ball.

        With ``init_axis_angle`` (shape (K, 3)): use those as the initial
        particle positions verbatim. Intended for GOE-init: sample K
        positions from the orientation feasibility set computed by the
        GOE construction (e.g., midas_grain_odf.goe.sample_envelope) and
        pass them in. Particles start inside the per-spot-centroid
        feasibility set rather than in random parts of the trust region.
        """
        super().__init__(R_avg, theta_max=theta_max)
        self._K = int(K)

        if init_axis_angle is not None:
            init_axis_angle = torch.as_tensor(init_axis_angle, dtype=torch.float64)
            if init_axis_angle.shape != (self._K, 3):
                raise ValueError(
                    f"init_axis_angle shape {tuple(init_axis_angle.shape)} "
                    f"does not match expected ({self._K}, 3)"
                )
            axis_angles = init_axis_angle.clone()
        else:
            if seed is not None:
                g = torch.Generator().manual_seed(seed)
            else:
                g = None
            # Uniform-ish random angle-axis vectors inside theta_max ball.
            # Direction uniform on S^2; magnitude uniform in [0, theta_max].
            # (Slightly biases small angles -- fine for initialization.)
            directions = torch.randn(self._K, 3, generator=g, dtype=torch.float64)
            directions = directions / directions.norm(dim=-1, keepdim=True)
            magnitudes = torch.rand(self._K, 1, generator=g, dtype=torch.float64) * theta_max
            axis_angles = directions * magnitudes

        self.axis_angle = nn.Parameter(axis_angles)
        self.logits = nn.Parameter(torch.zeros(self._K, dtype=torch.float64))

    @property
    def K(self) -> int:
        return self._K

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Soft trust-region clipping: scale axis-angle vectors whose norm
        # exceeds theta_max back to theta_max (preserves direction).
        norms = self.axis_angle.norm(dim=-1, keepdim=True)
        scale = torch.where(
            norms > self.theta_max,
            self.theta_max / (norms + 1e-12),
            torch.ones_like(norms),
        )
        clipped = self.axis_angle * scale

        delta_R = axis_angle_to_matrix(clipped)            # (K, 3, 3)
        R = self.R_avg.unsqueeze(0) @ delta_R              # (K, 3, 3)
        w = torch.softmax(self.logits, dim=0)              # (K,)
        return R, w


# ---------------------------------------------------------------------------
#  4b. Bingham mixture ODF
# ---------------------------------------------------------------------------


class BinghamMixtureODF(GrainODF):
    """Mixture of Bingham-like concentrated distributions on quaternions.

    For tractability we use a Bingham *surrogate* parameterized by:
      - mode quaternion mu_m  (4D unit vector)
      - concentration kappa_m (positive scalar)
    sampling K points around each mode, with mixture weights pi_m.
    Density is approximated by an isotropic concentration on quaternions:
        p(q | mu, kappa) ∝ exp(kappa * (q · mu)^2)
    which is the standard antipodally-symmetric von-Mises-Fisher-on-S^3 used
    when full Bingham normalization is overkill. For MVP this gives smooth,
    differentiable mixture densities; the full Bingham normalization can
    replace this without API change.

    The mixture is sampled deterministically: K samples per mode at
    fixed offsets (uniform in axis-angle, magnitude scaled by 1/sqrt(kappa))
    so the forward pass is reproducible.
    """

    def __init__(
        self,
        R_avg: torch.Tensor,
        n_modes: int = 3,
        K_per_mode: int = 32,
        theta_max: float = math.radians(5.0),
        seed: Optional[int] = None,
    ):
        super().__init__(R_avg, theta_max=theta_max)
        self.n_modes = int(n_modes)
        self.K_per_mode = int(K_per_mode)

        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        else:
            g = None

        # Initialize modes near identity (in delta-rotation space) with a small spread.
        # Mode parameter: axis-angle delta from R_avg.
        mode_axis_angle = torch.randn(self.n_modes, 3, generator=g, dtype=torch.float64) * 0.2 * theta_max
        self.mode_axis_angle = nn.Parameter(mode_axis_angle)
        self.log_kappa = nn.Parameter(
            torch.full((self.n_modes,), math.log(50.0), dtype=torch.float64)
        )
        self.mixture_logits = nn.Parameter(torch.zeros(self.n_modes, dtype=torch.float64))

        # Fixed sampling offsets per mode (deterministic). We use a halton-like
        # uniform draw; magnitudes will be modulated at sample-time by 1/sqrt(kappa).
        directions = torch.randn(self.K_per_mode, 3, generator=g, dtype=torch.float64)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        magnitudes = torch.rand(self.K_per_mode, 1, generator=g, dtype=torch.float64)
        # Scale to roughly fill the unit ball; per-mode scaling happens at sample time.
        offsets = directions * magnitudes
        self.register_buffer("_sample_offsets", offsets)  # (K_per_mode, 3)

    @property
    def K(self) -> int:
        return self.n_modes * self.K_per_mode

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-mode scale: 1/sqrt(kappa), capped at theta_max.
        kappa = torch.exp(self.log_kappa)                  # (M,)
        sigma = 1.0 / torch.sqrt(kappa.clamp(min=1.0))     # (M,)
        sigma = sigma.clamp(max=self.theta_max)            # (M,)

        # Mode delta (clipped to trust region) -> rotation matrix.
        norms = self.mode_axis_angle.norm(dim=-1, keepdim=True)
        scale = torch.where(
            norms > self.theta_max,
            self.theta_max / (norms + 1e-12),
            torch.ones_like(norms),
        )
        mode_aa_clipped = self.mode_axis_angle * scale     # (M, 3)
        R_mode = axis_angle_to_matrix(mode_aa_clipped)     # (M, 3, 3)

        # Per-sample offsets scaled by sigma_m, applied as additional rotations.
        # offsets: (K_per_mode, 3) -> (1, K_per_mode, 3) * (M, 1, 1) -> (M, K_per_mode, 3)
        offsets_scaled = self._sample_offsets.unsqueeze(0) * sigma.reshape(-1, 1, 1)
        R_offsets = axis_angle_to_matrix(offsets_scaled)   # (M, K_per_mode, 3, 3)

        # Compose: R_sample = R_avg @ R_mode @ R_offset
        R = self.R_avg.unsqueeze(0).unsqueeze(0) @ \
            R_mode.unsqueeze(1) @ R_offsets                # (M, K_per_mode, 3, 3)
        R = R.reshape(self.n_modes * self.K_per_mode, 3, 3)

        # Compute Bingham-surrogate density at each sample (relative to its mode).
        # density(q | mode) ∝ exp(kappa * (q·mu)^2). For samples drawn near the
        # mode, this is a smooth bump; we evaluate it analytically rather than
        # via Monte Carlo.
        q_samples = orient_mat_to_quat(R)                  # (M*K_per_mode, 4)
        # Compute mode quaternions from R_avg @ R_mode.
        R_modes_full = self.R_avg.unsqueeze(0) @ R_mode    # (M, 3, 3)
        q_modes = orient_mat_to_quat(R_modes_full)         # (M, 4)
        # Replicate q_modes per K_per_mode samples
        q_modes_rep = q_modes.unsqueeze(1).expand(-1, self.K_per_mode, -1).reshape(-1, 4)
        dot = (q_samples * q_modes_rep).sum(dim=-1)        # (M*K_per_mode,)

        kappa_rep = kappa.unsqueeze(1).expand(-1, self.K_per_mode).reshape(-1)
        log_density_per_sample = kappa_rep * (dot ** 2)

        # Mixture weights π_m
        pi = torch.softmax(self.mixture_logits, dim=0)     # (M,)
        pi_rep = pi.unsqueeze(1).expand(-1, self.K_per_mode).reshape(-1)

        unnorm = pi_rep * torch.exp(log_density_per_sample - log_density_per_sample.max())
        w = unnorm / (unnorm.sum() + 1e-12)
        return R, w


# ---------------------------------------------------------------------------
#  4b'. Anisotropic single-mode Bingham ODF (axis-angle parameterization)
# ---------------------------------------------------------------------------


class AnisotropicBinghamODF(GrainODF):
    """Single-mode Bingham-like distribution with anisotropic concentration
    in axis-angle space.

    Designed for the "real-data ODF" use case where intra-grain spread
    is highly anisotropic — typically broad along one slip-system rotation
    axis (producing eta-aligned spot streaks on the detector) and tight
    along the other two axes-angle directions.

    Parameters
    ----------
    mode_axis_angle : nn.Parameter shape (3,)
        Axis-angle vector of the Bingham mode center, relative to R_avg.
        Initialized at ~0 (small Gaussian).
    log_sigma_aa : nn.Parameter shape (3,)
        Per-axis log standard deviation of the axis-angle Gaussian (one
        scalar per axis-angle component). Anisotropy is captured by
        having different sigma_aa values across the three axes.
        Initialized to log(theta_max / 4) on each axis (mildly broad).

    The K samples are deterministic offsets drawn from a fixed unit
    Gaussian (registered as a buffer for reproducibility); at sample time
    each sample is multiplied componentwise by exp(log_sigma_aa) and
    added to mode_axis_angle. All sample weights are uniform = 1/K.

    Why no learnable per-particle weights? At a single Bingham mode the
    spread is fully captured by the (mode_axis_angle, log_sigma_aa)
    parameters; per-particle softmax weights would just duplicate that
    information and create the logits-mixing problem documented in
    project_lbfgs_logits_masking.
    """

    def __init__(
        self,
        R_avg: torch.Tensor,
        K: int = 64,
        theta_max: float = math.radians(5.0),
        seed: Optional[int] = None,
        init_sigma_frac: float = 0.25,
    ):
        super().__init__(R_avg, theta_max=theta_max)
        self._K = int(K)

        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        else:
            g = None

        # Mode at zero axis-angle (R_avg) plus a tiny jitter for symmetry breaking.
        mode = torch.randn(3, generator=g, dtype=torch.float64) * 0.01 * theta_max
        self.mode_axis_angle = nn.Parameter(mode)

        # log sigma per axis: start at init_sigma_frac × theta_max (mildly broad)
        sigma_init = init_sigma_frac * theta_max
        self.log_sigma_aa = nn.Parameter(
            torch.full((3,), math.log(sigma_init), dtype=torch.float64)
        )

        # Fixed unit-Gaussian samples
        std_norm = torch.randn(self._K, 3, generator=g, dtype=torch.float64)
        # Subtract mean to ensure samples are centered (avoid bias in mode estimation)
        std_norm = std_norm - std_norm.mean(dim=0, keepdim=True)
        self.register_buffer("_std_normal", std_norm)        # (K, 3)

    @property
    def K(self) -> int:
        return self._K

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-axis sigma in axis-angle space (clamped to theta_max).
        sigma = torch.exp(self.log_sigma_aa).clamp(max=self.theta_max)   # (3,)
        # Sample axis-angles: mode + std_normal * sigma (per-axis).
        offsets = self._std_normal * sigma.unsqueeze(0)                  # (K, 3)
        aa = self.mode_axis_angle.unsqueeze(0) + offsets                 # (K, 3)
        # Trust-region clip (per particle).
        norms = aa.norm(dim=-1, keepdim=True)
        scale = torch.where(
            norms > self.theta_max,
            self.theta_max / (norms + 1e-12),
            torch.ones_like(norms),
        )
        aa_clipped = aa * scale
        delta_R = axis_angle_to_matrix(aa_clipped)                       # (K, 3, 3)
        R = self.R_avg.unsqueeze(0) @ delta_R                            # (K, 3, 3)
        # Uniform weights (the anisotropy lives in the position distribution).
        w = torch.full((self._K,), 1.0 / self._K,
                        dtype=torch.float64, device=self.R_avg.device)
        return R, w


# ---------------------------------------------------------------------------
#  4c. Voxel-grid ODF
# ---------------------------------------------------------------------------


class VoxelGridODF(GrainODF):
    """Regular grid in axis-angle space with softmax weights at each node.

    Grid nodes live at (i, j, k) indices in [0, N)^3, mapped linearly to
    axis-angle vectors in [-theta_max, theta_max]^3. (Cube, not ball — a
    weight outside the inscribed ball still gets density but we keep the
    cube for grid simplicity. Trust-region penalty handled in losses.)
    """

    def __init__(
        self,
        R_avg: torch.Tensor,
        n_per_axis: int = 5,
        theta_max: float = math.radians(5.0),
    ):
        super().__init__(R_avg, theta_max=theta_max)
        self.n_per_axis = int(n_per_axis)

        # Axis-angle coordinates of grid nodes: linspace in [-theta_max, theta_max].
        coords = torch.linspace(-theta_max, theta_max, self.n_per_axis, dtype=torch.float64)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
        grid_aa = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # (N^3, 3)
        self.register_buffer("_grid_aa", grid_aa)

        # One logit per node.
        self.logits = nn.Parameter(torch.zeros(self.n_per_axis ** 3, dtype=torch.float64))

    @property
    def K(self) -> int:
        return self.n_per_axis ** 3

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_R = axis_angle_to_matrix(self._grid_aa)      # (N^3, 3, 3)
        R = self.R_avg.unsqueeze(0) @ delta_R              # (N^3, 3, 3)
        w = torch.softmax(self.logits, dim=0)              # (N^3,)
        return R, w


def particle_spread_stats(odf, *, within_deg: float = 1.0) -> dict:
    """Robust particle-spread statistics for a fitted ODF (E4c).

    Returns a dict with BOTH the weighted-RMS spread and robust
    alternatives. wRMS scales with ``theta_max`` (the prior-dominated
    tails inflate it: measured 1.75× at 2× trust region while the
    weighted MEDIAN moved only 1.24×) — report the weighted median (+
    the weight fraction within ``within_deg``) alongside wRMS, and
    prefer the median when comparing across trust-region settings.

    Keys: ``wrms_deg``, ``weighted_median_deg``, ``weight_within``,
    ``within_deg``, ``n_particles``.
    """
    with torch.no_grad():
        R, w = odf.sample()
        w = w.reshape(-1).to(torch.float64)
        w = w / w.sum().clamp(min=1e-300)
        # Angle of each particle from R_avg (convention-free: straight from
        # the rotation matrices, so it works for every ODF family).
        R_avg = odf.R_avg.to(R.dtype)
        R_rel = R_avg.transpose(-1, -2) @ R                       # (K, 3, 3)
        tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
        cos_t = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
        th_deg = torch.rad2deg(torch.acos(cos_t)).to(torch.float64)

        wrms = torch.sqrt((w * th_deg ** 2).sum())
        order = torch.argsort(th_deg)
        cum = torch.cumsum(w[order], dim=0)
        k = int(torch.searchsorted(cum, torch.tensor(0.5, dtype=cum.dtype)))
        k = min(k, th_deg.numel() - 1)
        wmed = th_deg[order[k]]
        within = w[th_deg <= float(within_deg)].sum()
    return {
        "wrms_deg": float(wrms),
        "weighted_median_deg": float(wmed),
        "weight_within": float(within),
        "within_deg": float(within_deg),
        "n_particles": int(th_deg.numel()),
    }
