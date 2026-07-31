"""Differentiable surrogate for ``CalcFracOverlap`` and the forward-model
glue layer.

Bridges :mod:`midas_diffract`'s :class:`HEDMForwardModel` and the
fit drivers in this package. Provides:

- :func:`build_forward_model` — construct a model from
  :class:`midas_nf_fitorientation.params.FitParams` plus an HKL table.
- :func:`forward_with_overrides` — call the model with optional
  per-call replacement of geometry tensors (``Lsd``, ``y_BC``,
  ``z_BC``, ``tilts``, ``wedge``). The replacement tensors carry their
  own autograd history, so gradients flow back through them rather
  than through the model's internal ``nn.Parameter`` buffers.
- :func:`soft_overlap_loss` — full closure body: run the forward with
  overrides, sample the obs volume, return ``1 − mean overlap`` plus
  optional Tikhonov terms.

The override pattern is the cleanest way to mix L-BFGS leaves
(``TanhBox.u``) with the existing forward graph without touching the
upstream package.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import numpy as np
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry

from .obs_volume import ObsVolume
from .params import FitParams
from .reparam import LsdEncoding, TanhBox


# ---------------------------------------------------------------------------
#  HKL → Cartesian helper (avoids a hard dependency on midas-hkls)
# ---------------------------------------------------------------------------

def cartesian_B_matrix(latc: tuple) -> np.ndarray:
    """B matrix for ``hkls_cart = hkls_int @ B^T`` in Cartesian 1/Å.

    Replicates the convention used in
    :func:`midas_diffract.forward.HEDMForwardModel.correct_hkls_latc`,
    which itself ports ``CorrectHKLsLatC`` from
    ``FF_HEDM/src/FitPosOrStrainsDoubleDataset.c:214-252``.
    """
    a, b, c, alpha_d, beta_d, gamma_d = latc
    d2r = math.pi / 180.0
    alpha = alpha_d * d2r
    beta = beta_d * d2r
    gamma = gamma_d * d2r
    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta), math.cos(beta)
    sg, cg = math.sin(gamma), math.cos(gamma)

    eps = 1e-7
    gamma_pr = math.acos(max(-1 + eps, min(1 - eps,
                              (ca * cb - cg) / (sa * sb + eps))))
    beta_pr = math.acos(max(-1 + eps, min(1 - eps,
                             (cg * ca - cb) / (sg * sa + eps))))
    sin_beta_pr = math.sin(beta_pr)

    vol = a * b * c * sa * sin_beta_pr * sg
    a_pr = b * c * sa / (vol + eps)
    b_pr = c * a * sb / (vol + eps)
    c_pr = a * b * sg / (vol + eps)

    return np.array([
        [a_pr, b_pr * math.cos(gamma_pr), c_pr * math.cos(beta_pr)],
        [0.0,  b_pr * math.sin(gamma_pr), -c_pr * sin_beta_pr * ca],
        [0.0,  0.0,                       c_pr * sin_beta_pr * sa],
    ])


def hkls_cart_thetas(
    hkls_int: np.ndarray,
    latc: tuple,
    wavelength_A: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Compute Cartesian G-vectors and Bragg angles from integer HKLs.

    Returns
    -------
    hkls_cart : ndarray (M, 3) float64, units of 1/Å
    thetas    : ndarray (M,) float64, **radians**
    """
    B = cartesian_B_matrix(latc)
    hkls_cart = hkls_int @ B.T
    g_mag = np.linalg.norm(hkls_cart, axis=-1)
    s = g_mag * wavelength_A / 2.0
    s = np.clip(s, -1.0 + 1e-12, 1.0 - 1e-12)
    thetas = np.arcsin(s)
    return hkls_cart, thetas


# ---------------------------------------------------------------------------
#  Forward-model construction
# ---------------------------------------------------------------------------

def build_forward_model(
    p: FitParams,
    hkls_int: np.ndarray,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    *,
    hkls_cart: Optional[np.ndarray] = None,
    thetas_rad: Optional[np.ndarray] = None,
) -> HEDMForwardModel:
    """Construct an :class:`HEDMForwardModel` from FitParams + HKL list.

    The model is built in NF-HEDM mode (``flip_y=False``,
    ``multi_mode='layered'``), with all geometry tensors at zero
    ``requires_grad``. Fit drivers opt parameters in via the override
    mechanism, not by mutating ``model.<param>.requires_grad`` directly.

    Parameters
    ----------
    hkls_int : ndarray (M, 3)
        Integer Miller indices. Used by the strain-refinement path
        (:meth:`HEDMForwardModel.correct_hkls_latc`).
    hkls_cart : ndarray (M, 3), optional
        Precomputed Cartesian G-vectors in 1/Å (typically read from
        ``hkls.csv`` columns 5–7 via :func:`io.read_hkls`). If not
        supplied, we compute them from ``hkls_int`` and the lattice
        constant via the B-matrix in :func:`hkls_cart_thetas`. The
        precomputed path is preferred — it matches what
        ``MakeDiffrSpots`` actually wrote into ``DiffractionSpots.bin``,
        avoiding any chance of double-applying B.
    thetas_rad : ndarray (M,), optional
        Bragg angles in radians. If ``hkls_cart`` is supplied we
        compute thetas from ``|G|*λ/2`` to keep the two consistent;
        pass ``thetas_rad`` only if you have non-default theta values
        you want to preserve verbatim.
    """
    if hkls_cart is None:
        hkls_cart, thetas = hkls_cart_thetas(
            hkls_int, p.lattice_constant, p.wavelength,
        )
    else:
        if thetas_rad is None:
            g_mag = np.linalg.norm(hkls_cart, axis=-1)
            s = np.clip(g_mag * p.wavelength / 2.0, -1 + 1e-12, 1 - 1e-12)
            thetas = np.arcsin(s)
        else:
            thetas = thetas_rad

    Lsd_list = list(p.Lsd) if p.n_distances > 1 else float(p.Lsd[0])
    yBC_list = list(p.ybc) if p.n_distances > 1 else float(p.ybc[0])
    zBC_list = list(p.zbc) if p.n_distances > 1 else float(p.zbc[0])

    geom = HEDMGeometry(
        Lsd=Lsd_list,
        y_BC=yBC_list,
        z_BC=zBC_list,
        px=p.px,
        omega_start=p.omega_start,
        omega_step=p.omega_step,
        n_frames=p.n_frames_per_distance,
        n_pixels_y=p.n_pixels_y,
        n_pixels_z=p.n_pixels_z,
        min_eta=p.exclude_pole_angle,
        wavelength=p.wavelength,
        tx=p.tx,
        ty=p.ty,
        tz=p.tz,
        flip_y=False,           # NF-HEDM convention
        wedge=p.wedge,
        multi_mode="layered",
        # Paired OmegaRange/BoxSize gate. The C driver applies this inside
        # CalcDiffrSpots_Furnace before CalcFracOverlap ever sees the spot
        # list (NF_HEDM/src/SharedFuncsFit.c:830-837), so a spot outside the
        # box is excluded from the ConfidenceIndex denominator rather than
        # counted as a miss. Both lists empty => filter off (unchanged).
        omega_ranges=list(p.omega_ranges) or None,
        box_sizes=list(p.box_sizes) or None,
    )

    model = HEDMForwardModel(
        hkls=torch.tensor(hkls_cart, dtype=dtype),
        thetas=torch.tensor(thetas, dtype=dtype),
        geometry=geom,
        hkls_int=torch.tensor(hkls_int, dtype=dtype),
        device=torch.device(device),
    )

    return model


# ---------------------------------------------------------------------------
#  Override context
# ---------------------------------------------------------------------------

@dataclass
class GeometryOverrides:
    """Per-call replacement tensors for geometry buffers.

    Any field set to ``None`` falls through to the model's own internal
    parameter. Any field set to a tensor temporarily replaces the
    corresponding attribute via :func:`object.__setattr__` (bypassing
    ``nn.Module.__setattr__`` so we don't have to convert plain tensors
    to ``nn.Parameter``). Gradients flow through the override tensor
    back to whatever leaf produced it — typically a
    :class:`TanhBox.u`.

    Shapes
    ------
    Lsd, y_BC, z_BC : (n_distances,) — float32 or float64
    tilts           : (n_distances, 3) — degrees (tx, ty, tz)
    wedge           : scalar — degrees
    """
    Lsd: Optional[torch.Tensor] = None
    y_BC: Optional[torch.Tensor] = None
    z_BC: Optional[torch.Tensor] = None
    tilts: Optional[torch.Tensor] = None
    wedge: Optional[torch.Tensor] = None


@contextmanager
def overrides(
    model: HEDMForwardModel,
    ov: GeometryOverrides,
) -> Iterator[None]:
    """Temporarily replace model geometry attributes with the supplied
    tensors. Restores the originals on context exit.

    The model uses ``self._Lsd_eff`` (a property that adds ``_Lsd`` and
    ``1000 * _Lsd_delta_mm``); since we override ``_Lsd`` and never
    touch ``_Lsd_delta_mm``, the property still works. The ``_has_tilts``
    fast-path flag becomes stale during the override, so we toggle it
    on whenever we override ``tilts`` (``_apply_nf_tilt`` will then
    actually run instead of skipping).
    """
    saves: Dict[str, object] = {}
    saves["_has_tilts_orig"] = model._has_tilts

    def _set(name: str, value):
        saves[name] = getattr(model, name)
        object.__setattr__(model, name, value)

    if ov.Lsd is not None:
        _set("_Lsd", ov.Lsd)
    if ov.y_BC is not None:
        _set("_y_BC", ov.y_BC)
    if ov.z_BC is not None:
        _set("_z_BC", ov.z_BC)
    if ov.tilts is not None:
        _set("tilts", ov.tilts)
        # Ensure NF tilt application actually runs
        model._has_tilts = True
    if ov.wedge is not None:
        _set("wedge", ov.wedge)
        model._has_wedge = bool(abs(float(ov.wedge.detach())) > 0.0)

    try:
        yield
    finally:
        for name, val in saves.items():
            if name == "_has_tilts_orig":
                continue
            object.__setattr__(model, name, val)
        model._has_tilts = saves["_has_tilts_orig"]


# ---------------------------------------------------------------------------
#  Soft overlap loss
# ---------------------------------------------------------------------------

def soft_overlap(
    model: HEDMForwardModel,
    obs: ObsVolume,
    euler: torch.Tensor,
    position_um: torch.Tensor,
    sigma_px: float,
    geom_ov: Optional[GeometryOverrides] = None,
) -> torch.Tensor:
    """Forward + sample → soft FracOverlap in ``[0, 1]``.

    Parameters
    ----------
    model : HEDMForwardModel
    obs : ObsVolume
    euler : Tensor (3,) or (N, 3)
        Bunge ZXZ angles in radians. A 1D input is treated as a single
        grain.
    position_um : Tensor (3,) or (N, 3)
        Voxel centroid in lab frame, microns. NF triangular voxels are
        approximated by their centroid; ``sigma_px`` widens the
        Gaussian splat to cover the triangle footprint.
    sigma_px : float
        Std-dev of the Gaussian splat in detector pixels. For
        sub-pixel voxels (``gs ≤ px``) use 1.0; for larger voxels use
        roughly ``gs / (px * sqrt(3))``.

    Returns
    -------
    Tensor scalar — overlap fraction in ``[0, 1]``. Higher = better.
    """
    if euler.ndim == 1:
        euler = euler.unsqueeze(0)
    if position_um.ndim == 1:
        position_um = position_um.unsqueeze(0)

    geom_ov = geom_ov or GeometryOverrides()
    with overrides(model, geom_ov):
        spots = model(euler, position_um)

    # The forward squeezes the implicit batch dim, so for a single grain
    # the shapes are:
    #   single-distance (D=1, layered): (K, M)            — no D dim
    #   multi-distance  (D>1, layered): (D, K, M)         — leading D dim
    # ``ObsVolume.soft_fraction`` handles both.
    return obs.soft_fraction(
        frame_nr=spots.frame_nr,
        y_pixel=spots.y_pixel,
        z_pixel=spots.z_pixel,
        valid=spots.valid,
        sigma_px=sigma_px,
    )


def soft_overlap_loss(
    model: HEDMForwardModel,
    obs: ObsVolume,
    euler: torch.Tensor,
    position_um: torch.Tensor,
    sigma_px: float,
    geom_ov: Optional[GeometryOverrides] = None,
    tikhonov_terms: Optional[Iterator[torch.Tensor]] = None,
) -> torch.Tensor:
    """``loss = (1 − soft_overlap) + sum(Tikhonov terms)``.

    Drop-in body for an L-BFGS closure. ``tikhonov_terms`` is any
    iterable of pre-computed scalar tensors (typically
    :meth:`TanhBox.tikhonov` outputs).
    """
    overlap = soft_overlap(model, obs, euler, position_um, sigma_px, geom_ov)
    loss = 1.0 - overlap
    if tikhonov_terms:
        for t in tikhonov_terms:
            loss = loss + t
    return loss


# ---------------------------------------------------------------------------
#  Batched per-grain forward (used by the vectorised NM polish)
# ---------------------------------------------------------------------------

def forward_batched_grains(
    model: HEDMForwardModel,
    eul: torch.Tensor,
    pos: torch.Tensor,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """Run the forward model on ``B`` independent grain fits at once.

    The native :meth:`HEDMForwardModel.forward` packs the two omega
    solutions along the first axis of every output (so for ``B`` input
    grains it returns ``(2B, M)`` rather than ``(B, 2, M)``). For
    fitting we want ``(B, K, M)`` so the per-grain hard-fraction
    reduction at the end summarises to one scalar per grain. This
    helper does the reshape + permute and hands back the four tensors
    the obs sampler needs.

    Returns
    -------
    frame_nr, valid : Tensor (B, K=2, M)
    y_pixel, z_pixel : Tensor (D, B, K=2, M) under multi-distance
        ``"layered"`` mode with ``D > 1``; otherwise ``(B, K=2, M)``
        (the ``D=1`` collapse the model already does).
    """
    if eul.ndim != 2 or eul.shape[-1] != 3:
        raise ValueError(f"eul must be (B, 3); got {eul.shape}")
    if pos.shape != eul.shape:
        raise ValueError(
            f"pos must match eul shape (B, 3); got {pos.shape}"
        )
    B = eul.shape[0]
    K = 2
    spots = model(eul, pos)

    frame_nr = spots.frame_nr.reshape(K, B, -1).permute(1, 0, 2).contiguous()
    valid = spots.valid.reshape(K, B, -1).permute(1, 0, 2).contiguous()

    if spots.y_pixel.ndim == 3:
        D = spots.y_pixel.shape[0]
        y_pixel = (
            spots.y_pixel.reshape(D, K, B, -1)
            .permute(0, 2, 1, 3).contiguous()
        )
        z_pixel = (
            spots.z_pixel.reshape(D, K, B, -1)
            .permute(0, 2, 1, 3).contiguous()
        )
    else:
        y_pixel = (
            spots.y_pixel.reshape(K, B, -1)
            .permute(1, 0, 2).contiguous()
        )
        z_pixel = (
            spots.z_pixel.reshape(K, B, -1)
            .permute(1, 0, 2).contiguous()
        )

    return frame_nr, valid, y_pixel, z_pixel


# ---------------------------------------------------------------------------
#  Helper: choose σ_px from voxel size
# ---------------------------------------------------------------------------

def auto_sigma_px(gs_um: float, px_um: float, override: Optional[float] = None) -> float:
    """Pick a Gaussian splat σ in pixels for a triangular voxel of edge
    ``gs_um``. ``override`` (if not None) wins.

    Heuristic: a triangle of edge ``L`` has inscribed-circle radius
    ``L / (2*sqrt(3))``, which we use as a rough Gaussian "radius".
    Clamped at 1 px (the minimum below which trilinear sampling is
    already smooth enough).
    """
    if override is not None and override > 0:
        return float(override)
    return max(1.0, gs_um / (2.0 * math.sqrt(3.0) * px_um))
