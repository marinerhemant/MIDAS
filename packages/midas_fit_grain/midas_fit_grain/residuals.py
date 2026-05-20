"""Per-grain residual computation in pixel / angular / internal-angle space.

All three losses share the same plumbing:
  1. Run :class:`HEDMForwardModel` to get a ``SpotDescriptors`` for the
     current ``(euler, position, lattice)`` state.
  2. Use a precomputed :class:`MatchResult` (from ``matching.associate``) to
     gather the predicted slot for each observed spot.
  3. Compute residuals according to ``kind``.

The predicted-spot gather is differentiable; the choice of which slot to
pick is *not* (it's a discrete index produced by ``associate``). This is
intentional: re-association happens between solver phases, not inside.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch

from midas_diffract import HEDMForwardModel, SpotDescriptors  # type: ignore

from .matching import MatchResult
from .observations import ObservedSpots

LossKind = Literal["pixel", "full3d", "angular", "internal_angle"]


def _gather_pred(spots: SpotDescriptors,
                 match: MatchResult) -> dict[str, torch.Tensor]:
    """Pick predicted (omega, eta, two_theta, y_pixel, z_pixel) per observed spot.

    ``spots`` carries shape ``(K, M)`` (single-grain) tensors. ``match`` has
    ``(S,)`` indices. The returned dict has matching ``(S,)`` tensors for
    every field.
    """
    S = match.k_idx.shape[0]
    if S == 0:
        zero = match.k_idx.new_zeros((0,), dtype=spots.omega.dtype)
        return {k: zero for k in ("omega", "eta", "two_theta", "y_pixel", "z_pixel")}

    flat_idx = match.k_idx * spots.omega.shape[-1] + match.m_idx  # (S,)

    def _pick(t: torch.Tensor) -> torch.Tensor:
        # t can be (K, M) or (D, K, M) for multi-distance — we squeeze D=1 only.
        if t.dim() == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t.reshape(-1).gather(0, flat_idx)

    return {
        "omega": _pick(spots.omega),
        "eta": _pick(spots.eta),
        "two_theta": _pick(spots.two_theta),
        "y_pixel": _pick(spots.y_pixel),
        "z_pixel": _pick(spots.z_pixel),
    }


def _angular_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrap ``a - b`` into [-π, π]."""
    diff = a - b
    return ((diff + math.pi) % (2.0 * math.pi)) - math.pi


def grain_residuals(
    model: HEDMForwardModel,
    *,
    grain_euler: torch.Tensor,         # (3,)  rad — refining
    grain_position: torch.Tensor,      # (3,)  um  — refining
    grain_lattice: torch.Tensor,       # (6,)  refining
    obs: ObservedSpots,
    match: MatchResult,
    kind: LossKind,
    px: float,
    y_BC: float,
    z_BC: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute residuals for one grain. Returns ``(S, K_residual)`` tensor.

    ``K_residual`` is 2 for ``pixel``, 3 for ``angular``, 1 for
    ``internal_angle`` (a scalar per spot). Rows for unmatched spots are
    zeroed via the mask so they don't affect ``(r ** 2).sum()``-style losses.

    The returned tensor is differentiable wrt ``grain_euler``,
    ``grain_position``, ``grain_lattice`` (via the forward model and
    autograd).
    """
    if obs.n_spots == 0 or match.k_idx.shape[0] == 0:
        return torch.zeros((0, _residual_dim(kind)),
                           dtype=grain_euler.dtype, device=grain_euler.device)

    # Forward pass: shape (1, 1, 3) -> spots (B=1, N=1, K=2, M)
    euler = grain_euler.view(1, 1, 3)
    pos = grain_position.view(1, 1, 3)
    latc = grain_lattice.view(1, 6)
    spots = model(euler, pos, lattice_params=latc)

    # Squeeze the (B=1, N=1) leading dims away. After project_to_detector with
    # single distance, omega/eta/two_theta have shape (1, 1, K=2, M) -> (K, M)
    # because the forward concatenates omega solutions along the K axis.
    def _drop_lead(t: torch.Tensor) -> torch.Tensor:
        # forward outputs (..., K, M); for (B=1, N=1) the leading is (1, 1, ...)
        # but project_to_detector folds positions into K = 2*N, so for N=1, K=2
        # and the (B=1) just adds a single batch dim — squeeze to (K, M).
        while t.dim() > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
            if t.dim() == 0:
                break
        return t

    spots_sg = SpotDescriptors(
        omega=_drop_lead(spots.omega),
        eta=_drop_lead(spots.eta),
        two_theta=_drop_lead(spots.two_theta),
        y_pixel=_drop_lead(spots.y_pixel),
        z_pixel=_drop_lead(spots.z_pixel),
        frame_nr=_drop_lead(spots.frame_nr),
        valid=_drop_lead(spots.valid),
    )

    pick = _gather_pred(spots_sg, match)

    if kind == "pixel":
        # DISABLED 2026-05-20. The 'pixel' loss fit only the 2 detector
        # coordinates (y_pixel, z_pixel) and OMITTED omega, leaving the
        # crystal free to rotate about the omega direction. On real PF data
        # this let per-voxel orientations drift up to ~20° from the (correct)
        # indexer seed while the loss kept dropping — see
        # dev/REFINEMENT_DRIFT_FIX.md. A HEDM spot is 3D (y, z, omega); always
        # use a full 3D loss. Use 'angular' (2theta, eta, omega).
        raise ValueError(
            "The 'pixel' loss is disabled: it is a 2D (y,z) loss that omits "
            "omega and lets orientation drift freely (≈20° on real PF data). "
            "Use the full 3D loss 'full3d' (y, z, omega) or 'angular'."
        )

    elif kind == "full3d":
        # Full 3D spot loss: detector position (y_pixel, z_pixel) AND omega.
        # The 2-D 'pixel' loss omitted omega → orientation drifted freely; the
        # 'angular' loss includes omega but (in this forward model) drops
        # sensitivity to grain POSITION. 'full3d' keeps both: y,z constrain
        # position, omega constrains the rotation. Omega (rad) is scaled by the
        # spot's pixel radius from the beam centre so its residual is an
        # azimuthal-arc displacement in pixels, comparable to Δy/Δz.
        obs_y_pixel = y_BC - obs.y_lab / px
        obs_z_pixel = z_BC + obs.z_lab / px
        r_px = torch.sqrt((pick["y_pixel"] - y_BC) ** 2
                          + (pick["z_pixel"] - z_BC) ** 2)
        res = torch.stack([
            pick["y_pixel"] - obs_y_pixel,
            pick["z_pixel"] - obs_z_pixel,
            _angular_diff(pick["omega"], obs.omega) * r_px,
        ], dim=-1)  # (S, 3)

    elif kind == "angular":
        res = torch.stack([
            _angular_diff(pick["two_theta"], obs.two_theta),
            _angular_diff(pick["eta"], obs.eta),
            _angular_diff(pick["omega"], obs.omega),
        ], dim=-1)  # (S, 3)

    elif kind == "internal_angle":
        # Compare unit g-vectors in the lab frame. Friedel pairs (g, -g)
        # give the same diffraction spot, so we minimize over the
        # antiparallel direction by taking |g_pred · g_obs|.
        g_obs = obs.g_unit_lab()  # (S, 3)
        g_pred = _g_unit_from_pred(pick)
        cos_ang = (g_pred * g_obs).sum(dim=-1).abs().clamp(0.0, 1.0)
        # acos at exactly 1.0 has zero gradient on float64; nudge with a
        # tiny epsilon-clamp on the upper side so backward stays defined.
        cos_ang = cos_ang.clamp(max=1.0 - 1e-12)
        res = torch.acos(cos_ang).unsqueeze(-1)  # (S, 1) in [0, π/2]

    else:
        raise ValueError(f"unknown loss kind {kind!r}")

    # Apply mask: unmatched spots contribute zero residual.
    mask = match.mask.to(res.dtype).unsqueeze(-1)
    res = res * mask

    if weights is not None:
        res = res * weights.view(-1, 1).to(res.dtype)

    return res


def _g_unit_from_pred(pick: dict[str, torch.Tensor]) -> torch.Tensor:
    """Predicted unit G-vector in lab frame, built from (2θ, η, ω)."""
    theta = pick["two_theta"] * 0.5
    eta = pick["eta"]
    omega = pick["omega"]
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


def _residual_dim(kind: LossKind) -> int:
    return {"pixel": 2, "full3d": 3, "angular": 3, "internal_angle": 1}[kind]
