"""Batched (multi-grain) refinement plumbing.

Per the design plan: a refinement block contains ``B`` grains, each with a
ragged number of matched spots. We pad to ``S_max`` and carry a boolean
mask. The forward model is already batched, so one forward call produces
``(B, K=2, M)`` predicted spots for the whole block.

For ADAM, L-BFGS, and Nelder–Mead the per-grain loss is fully separable
(no cross-grain coupling), so we sum over the batch and run a single
optimizer over the flat ``12·B``-vector. For Levenberg–Marquardt the
Jacobian is block-diagonal, so we treat each grain independently — but
all forward + matching work is still done in one batched pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import math

import torch

from midas_diffract import HEDMForwardModel, SpotDescriptors  # type: ignore

from .config import LossKind
from .matching import MatchResult
from .observations import ObservedSpots

DEG2RAD = math.pi / 180.0


@dataclass
class ObservedBatch:
    """Padded ``(B, S_max, …)`` observed-spot batch.

    ``valid`` is a boolean mask: True where row b has a real spot at
    index s (i.e. s < n_spots[b]).
    """
    n_grains: int
    s_max: int
    spot_id:    torch.Tensor   # (B, S_max) int64
    ring_nr:    torch.Tensor   # (B, S_max) int64
    y_lab:      torch.Tensor   # (B, S_max) um
    z_lab:      torch.Tensor   # (B, S_max) um
    omega:      torch.Tensor   # (B, S_max) rad
    eta:        torch.Tensor   # (B, S_max) rad
    two_theta:  torch.Tensor   # (B, S_max) rad
    valid:      torch.Tensor   # (B, S_max) bool
    n_spots:    torch.Tensor   # (B,) int — real spot count per grain

    @classmethod
    def pack(cls, grains: Sequence[ObservedSpots],
             *, device: torch.device, dtype: torch.dtype) -> "ObservedBatch":
        """Pad a list of per-grain :class:`ObservedSpots` into a batch."""
        if not grains:
            raise ValueError("ObservedBatch.pack: empty grain list")
        B = len(grains)
        s_per = [int(g.n_spots) for g in grains]
        s_max = max(s_per) if s_per else 0

        def _empty_int(): return torch.zeros((B, s_max), dtype=torch.int64, device=device)
        def _empty_f():   return torch.zeros((B, s_max), dtype=dtype, device=device)

        spot_id = _empty_int()
        ring_nr = _empty_int()
        y_lab = _empty_f()
        z_lab = _empty_f()
        omega = _empty_f()
        eta = _empty_f()
        two_theta = _empty_f()
        valid = torch.zeros((B, s_max), dtype=torch.bool, device=device)

        for b, g in enumerate(grains):
            n = s_per[b]
            spot_id[b, :n] = g.spot_id.to(device)
            ring_nr[b, :n] = g.ring_nr.to(device)
            y_lab[b, :n] = g.y_lab.to(device=device, dtype=dtype)
            z_lab[b, :n] = g.z_lab.to(device=device, dtype=dtype)
            omega[b, :n] = g.omega.to(device=device, dtype=dtype)
            eta[b, :n] = g.eta.to(device=device, dtype=dtype)
            two_theta[b, :n] = g.two_theta.to(device=device, dtype=dtype)
            valid[b, :n] = True

        return cls(
            n_grains=B, s_max=s_max,
            spot_id=spot_id, ring_nr=ring_nr,
            y_lab=y_lab, z_lab=z_lab,
            omega=omega, eta=eta, two_theta=two_theta,
            valid=valid,
            n_spots=torch.tensor(s_per, dtype=torch.int64, device=device),
        )

    def g_unit_lab(self) -> torch.Tensor:
        """Unit observed G-vectors in the lab frame, shape ``(B, S_max, 3)``."""
        theta = self.two_theta * 0.5
        eta = self.eta
        omega = self.omega
        c_th = torch.cos(theta)
        g_om = torch.stack([
            -torch.sin(theta),
            c_th * torch.sin(eta),
            c_th * torch.cos(eta),
        ], dim=-1)
        c_w = torch.cos(omega)
        s_w = torch.sin(omega)
        g_lab = torch.stack([
            c_w * g_om[..., 0] + s_w * g_om[..., 1],
            -s_w * g_om[..., 0] + c_w * g_om[..., 1],
            g_om[..., 2],
        ], dim=-1)
        return g_lab / g_lab.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class MatchBatch:
    """Padded ``(B, S_max)`` association into the predicted ``(K, M)`` grid."""
    k_idx: torch.Tensor   # (B, S_max) int64
    m_idx: torch.Tensor   # (B, S_max) int64
    mask:  torch.Tensor   # (B, S_max) bool

    @classmethod
    def pack(cls, matches: Sequence[MatchResult], *,
             s_max: int, device: torch.device) -> "MatchBatch":
        B = len(matches)
        k_idx = torch.zeros((B, s_max), dtype=torch.int64, device=device)
        m_idx = torch.zeros((B, s_max), dtype=torch.int64, device=device)
        mask = torch.zeros((B, s_max), dtype=torch.bool, device=device)
        for b, m in enumerate(matches):
            n = int(m.k_idx.shape[0])
            k_idx[b, :n] = m.k_idx.to(device)
            m_idx[b, :n] = m.m_idx.to(device)
            mask[b, :n] = m.mask.to(device)
        return cls(k_idx=k_idx, m_idx=m_idx, mask=mask)


# ---------------------------------------------------------------------------
# Batched residual computation
# ---------------------------------------------------------------------------


def _angular_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a - b
    return ((diff + math.pi) % (2.0 * math.pi)) - math.pi


def batch_residuals(
    model: HEDMForwardModel,
    *,
    grain_position: torch.Tensor,    # (B, 3) um
    grain_euler:    torch.Tensor,    # (B, 3) rad
    grain_lattice:  torch.Tensor,    # (B, 6)
    obs:    ObservedBatch,
    match:  MatchBatch,
    kind:   LossKind,
    px: float, y_BC: float, z_BC: float,
) -> torch.Tensor:
    """Compute per-(grain, spot) residuals on a padded batch.

    Returns ``(B, S_max, K_residual)``. Padded slots and unmatched spots
    have residual zero so they don't contribute to ``(r ** 2).sum()``.
    """
    B = obs.n_grains
    spots = model(
        grain_euler.view(B, 1, 3),
        grain_position.view(B, 1, 3),
        lattice_params=grain_lattice.view(B, 6),
    )
    # Forward output for FF/single-distance is shape (B, K=2, M) for
    # omega/eta/two_theta/y_pixel/z_pixel/valid. (No leading distance dim.)
    omega = spots.omega
    eta = spots.eta
    two_theta = spots.two_theta
    y_pixel = spots.y_pixel
    z_pixel = spots.z_pixel
    valid_pred = spots.valid

    # Sanity: (B, K, M)
    assert omega.dim() == 3 and omega.shape[0] == B, omega.shape
    K = omega.shape[1]
    M = omega.shape[2]

    flat_idx = match.k_idx * M + match.m_idx  # (B, S_max)

    def _pick(t: torch.Tensor) -> torch.Tensor:
        # t: (B, K, M) -> flatten last two -> (B, K*M) -> gather over S
        return t.reshape(B, K * M).gather(1, flat_idx)

    pred_omega   = _pick(omega)
    pred_eta     = _pick(eta)
    pred_2theta  = _pick(two_theta)
    pred_y       = _pick(y_pixel)
    pred_z       = _pick(z_pixel)

    if kind == "pixel":
        raise ValueError(
            "The 'pixel' loss is disabled (2D, omits omega -> orientation "
            "drift). Use the full 3D loss 'full3d'. See dev/REFINEMENT_DRIFT_FIX.md."
        )

    elif kind == "full3d":
        # Detector position (y,z) + omega (scaled by spot pixel-radius to an
        # azimuthal arc, comparable to Δy/Δz). See residuals.grain_residuals.
        obs_y_pixel = y_BC - obs.y_lab / px
        obs_z_pixel = z_BC + obs.z_lab / px
        r_px = torch.sqrt((pred_y - y_BC) ** 2 + (pred_z - z_BC) ** 2)
        res = torch.stack([
            pred_y - obs_y_pixel,
            pred_z - obs_z_pixel,
            _angular_diff(pred_omega, obs.omega) * r_px,
        ], dim=-1)

    elif kind == "angular":
        res = torch.stack([
            _angular_diff(pred_2theta, obs.two_theta),
            _angular_diff(pred_eta, obs.eta),
            _angular_diff(pred_omega, obs.omega),
        ], dim=-1)

    elif kind == "internal_angle":
        # Build predicted unit g-vector from (2θ, η, ω); compare to obs.
        theta = pred_2theta * 0.5
        c_th = torch.cos(theta)
        g_om = torch.stack([
            -torch.sin(theta),
            c_th * torch.sin(pred_eta),
            c_th * torch.cos(pred_eta),
        ], dim=-1)
        c_w = torch.cos(pred_omega)
        s_w = torch.sin(pred_omega)
        g_pred = torch.stack([
            c_w * g_om[..., 0] + s_w * g_om[..., 1],
            -s_w * g_om[..., 0] + c_w * g_om[..., 1],
            g_om[..., 2],
        ], dim=-1)
        g_pred = g_pred / g_pred.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        g_obs = obs.g_unit_lab()
        cos_ang = (g_pred * g_obs).sum(dim=-1).abs().clamp(0.0, 1.0 - 1e-12)
        res = torch.acos(cos_ang).unsqueeze(-1)

    else:
        raise ValueError(f"unknown loss kind {kind!r}")

    # Combined mask: valid spot AND matched.
    full_mask = (obs.valid & match.mask).to(res.dtype).unsqueeze(-1)
    return res * full_mask
