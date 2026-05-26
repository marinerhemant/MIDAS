"""Stage 2: orientation-aware expected-(h,k,l) predictor (Tier 1).

For each per-grain consensus FZ orientation, forward-simulate every
signed (h, k, l) variant on the chosen indexing ring(s) and return
the set of variants that are **observable** given the scan geometry:

- predicted ω in [omega_min, omega_max]
- predicted η in [eta_min, eta_max] (or outside an η mask)
- predicted (Y, Z) inside the detector active area

A variant that fails any of these is **geometrically missing** — it
cannot have produced an indexer seed, so its absence from the cluster's
observed variant multiset is not a sign of over-merge.

The predicted (Y, Z, ω) is also used by Stage 3 to attribute the
cluster's seed-spots to specific variants (via nearest-neighbour match
of observed → predicted).

Convention notes
----------------

* Indexing ring `g_crystal` vectors come from ``hkls.csv`` (output of
  ``midas-calc-hkls`` / ``GetHKLList``) and are signed.
* Sample → lab rotation is around +Z by +ω. (MIDAS convention.)
* The Bragg quadratic with no wedge tilt:
    ``-Gx_lab = sin(θ) · |G|``
  has two solutions per orientation+variant (the Friedel pair, at ω
  and ω + 180°). Both are emitted.
* No detector tilt is applied in v1 (the indexer's calibrated tilts
  give Y, Z residuals well below detector resolution at this stage;
  we will fold them in once Stages 3/4 are validated).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .hkl_ingest import HklTable


@dataclass
class GeometryConfig:
    """Detector + scan geometry, used for the visibility predicate."""

    lsd_um:        float
    omega_min_deg: float
    omega_max_deg: float
    # Detector active area (lab frame, µm). None = no clamp.
    y_min_um:      Optional[float] = None
    y_max_um:      Optional[float] = None
    z_min_um:      Optional[float] = None
    z_max_um:      Optional[float] = None
    # Eta active range; default = full 360°.
    eta_min_deg:   float = -180.0
    eta_max_deg:   float = +180.0
    # Extra geometry needed when using the midas_diffract forward model
    # (canonical Bragg solver). When unset, the legacy bespoke solver is
    # used (which has a known ω error on hex datasets — see
    # compute/hkl_expected.py docstring).
    wavelength_a:  Optional[float] = None
    pixel_um:      float = 150.0
    y_BC:          float = 1024.0
    z_BC:          float = 1024.0
    omega_step_deg: float = 0.1
    n_pixels_y:    int = 2048
    n_pixels_z:    int = 2048
    wedge_deg:     float = 0.0
    tx_deg:        float = 0.0
    ty_deg:        float = 0.0
    tz_deg:        float = 0.0
    min_eta_deg:   float = 6.0
    use_midas_diffract: bool = True


@dataclass
class ExpectedHklTable:
    """Per-(grain, variant) prediction with visibility flag."""

    grain_idx:   np.ndarray   # (M,) int64 — which grain (0 .. N_g-1)
    h:           np.ndarray   # (M,) int8
    k:           np.ndarray   # (M,) int8
    l:           np.ndarray   # (M,) int8
    ring:        np.ndarray   # (M,) int32
    pred_y_um:   np.ndarray   # (M,) float64
    pred_z_um:   np.ndarray   # (M,) float64
    pred_omega_deg: np.ndarray  # (M,) float64
    pred_eta_deg:   np.ndarray  # (M,) float64
    visible:     np.ndarray   # (M,) bool — passed all geometry predicates


# ---------------------------------------------------------------------------
# Bragg geometry
# ---------------------------------------------------------------------------


def _bragg_omega_eta(
    om_fz: np.ndarray,       # (N, 3, 3) — FZ-canonical orientations (crystal→sample)
    g_crystal: np.ndarray,   # (M, 3) — crystal-frame reciprocal vectors (unit)
    theta_deg: np.ndarray,   # (M,) — Bragg θ in deg
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the Bragg quadratic ``-Gx_lab = sin θ · |G|``.

    Returns three (N, M, 2) arrays:
      ``omega_deg`` — the two ω solutions per (orientation, hkl), in deg
      ``eta_deg``   — the corresponding η azimuthal angle, in deg
      ``valid``     — bool mask: True if the discriminant was non-negative
                       (i.e. a real ω exists). Two solutions tagged independently.

    The two ω solutions are the Friedel pair (separated by 180°). Both are
    emitted; the caller decides what to do with each.

    Convention: ``OM @ g_crystal = g_sample`` (column-vector form). Rotation
    by ω about +Z carries g_sample to g_lab: ``g_lab = Rz(+ω) · g_sample``.
    """
    N = om_fz.shape[0]
    M = g_crystal.shape[0]

    # g_sample = OM @ g_crystal  →  einsum on cols
    # om_fz[n, i, j] · g_crystal[m, j]  =  g_sample[n, m, i]
    g_sample = np.einsum("nij,mj->nmi", om_fz, g_crystal)        # (N, M, 3)

    Gx_s = g_sample[..., 0]
    Gy_s = g_sample[..., 1]
    Gz_s = g_sample[..., 2]

    # Magnitude of g (preserved across orientation since OM is orthonormal)
    # but use |g_crystal| for numerical exactness.
    g_mag = np.linalg.norm(g_crystal, axis=1)                    # (M,)
    g_mag_NM = g_mag[None, :]                                    # (1, M)

    sin_th = np.sin(np.deg2rad(theta_deg))[None, :]              # (1, M)
    v = sin_th * g_mag_NM                                        # (1, M)

    # Bragg condition (rotated by ω about +Z, no wedge):
    #   g_lab = ( cos ω · Gx_s - sin ω · Gy_s ,
    #              sin ω · Gx_s + cos ω · Gy_s ,
    #              Gz_s )
    # constraint  -Gx_lab = v  →  -cos ω · Gx_s + sin ω · Gy_s = v
    # Let cos ω = c, sin ω = s, with c² + s² = 1.
    #   -c·Gx_s + s·Gy_s = v  →  s = (v + c·Gx_s) / Gy_s
    # Substitute into c² + s² = 1:
    #   c²·(Gx_s² + Gy_s²) + 2·c·Gx_s·v + (v² - Gy_s²) = 0
    # Standard quadratic in c.
    eps = 1e-30
    A = Gx_s * Gx_s + Gy_s * Gy_s + eps
    B = 2.0 * Gx_s * v
    C = v * v - Gy_s * Gy_s

    disc = B * B - 4.0 * A * C
    valid = disc >= 0
    sqrt_disc = np.sqrt(np.where(valid, disc, 0.0))

    c_plus  = (-B + sqrt_disc) / (2.0 * A)
    c_minus = (-B - sqrt_disc) / (2.0 * A)
    c_plus  = np.clip(c_plus, -1.0, 1.0)
    c_minus = np.clip(c_minus, -1.0, 1.0)

    s_plus  = (v + c_plus  * Gx_s) / np.where(np.abs(Gy_s) > 1e-30, Gy_s, 1.0)
    s_minus = (v + c_minus * Gx_s) / np.where(np.abs(Gy_s) > 1e-30, Gy_s, 1.0)

    omega_plus  = np.degrees(np.arctan2(s_plus,  c_plus))
    omega_minus = np.degrees(np.arctan2(s_minus, c_minus))

    # ω → η: rotate g_sample by ω about +Z, then η = atan2(Gz_lab, Gy_lab)
    def _eta(omega_deg_):
        ome = np.deg2rad(omega_deg_)
        co = np.cos(ome); so = np.sin(ome)
        Gx_lab = co * Gx_s - so * Gy_s
        Gy_lab = so * Gx_s + co * Gy_s
        Gz_lab = Gz_s
        eta = np.degrees(np.arctan2(Gz_lab, Gy_lab))
        return eta

    eta_plus  = _eta(omega_plus)
    eta_minus = _eta(omega_minus)

    omega = np.stack([omega_plus, omega_minus], axis=-1)         # (N, M, 2)
    eta   = np.stack([eta_plus,   eta_minus],   axis=-1)
    valid = np.stack([valid, valid], axis=-1)                    # (N, M, 2)
    return omega, eta, valid


def _omega_to_detector(
    om_fz: np.ndarray,
    g_crystal: np.ndarray,
    omega_deg: np.ndarray,    # (N, M, 2)
    theta_deg: np.ndarray,
    lsd_um: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project predicted (orientation, hkl, ω-branch) to detector (Y, Z).

    Returns (pred_y_um, pred_z_um) each of shape (N, M, 2).
    """
    # g_lab direction = Rz(ω) · OM · g_crystal
    g_sample = np.einsum("nij,mj->nmi", om_fz, g_crystal)        # (N, M, 3)
    Gx_s = g_sample[..., 0]
    Gy_s = g_sample[..., 1]
    Gz_s = g_sample[..., 2]

    ome = np.deg2rad(omega_deg)                                  # (N, M, 2)
    co = np.cos(ome); so = np.sin(ome)
    # Broadcast g_*_s over the ω-branch axis
    Gx_lab = co * Gx_s[..., None] - so * Gy_s[..., None]
    Gy_lab = so * Gx_s[..., None] + co * Gy_s[..., None]
    Gz_lab = np.broadcast_to(Gz_s[..., None], Gx_lab.shape)

    # k_in = (1, 0, 0); k_out_dir = k_in - g_lab (= -q, our convention sign).
    # Wait, for predicting the spot's location we want k_out, the direction
    # of the scattered ray. With q = k_in - k_out → k_out = k_in - q.
    # We have q = g (in 1/d units) but we only need direction here.
    # The scattering 2θ angle is set, so we can recover (Y, Z) from η alone:
    #   k_out_x = cos(2θ); k_out_y = sin(2θ)·cos(η); k_out_z = sin(2θ)·sin(η)
    # which requires knowing η. Compute it from g_lab as elsewhere.
    g_mag = np.linalg.norm(g_crystal, axis=1)                    # (M,)
    g_mag_NM = g_mag[None, :, None]                              # (1, M, 1)
    # Normalize g_lab to extract direction (which encodes 2θ + η):
    g_lab_norm = np.stack([Gx_lab, Gy_lab, Gz_lab], axis=-1)
    glab_dir = g_lab_norm / (np.linalg.norm(g_lab_norm, axis=-1, keepdims=True) + 1e-30)
    # k_out_dir = k_in_dir - g_lab_dir·(2·sin θ)  (in 1/λ units; for direction
    # just k_in - g_lab if both unit). Actually for projection we want
    # the actual scattering direction:
    #   k_out = k_in − q_lab,  where q_lab = g_lab_unit · 2 sin θ
    two_sin_th = (2.0 * np.sin(np.deg2rad(theta_deg)))[None, :, None]  # (1, M, 1)
    k_in = np.array([1.0, 0.0, 0.0])[None, None, None, :]
    q_lab = glab_dir * two_sin_th[..., None]
    k_out = k_in - q_lab
    k_out_dir = k_out / (np.linalg.norm(k_out, axis=-1, keepdims=True) + 1e-30)

    # Project to detector at x = lsd_um: (Y, Z) = lsd_um · (k_out_y, k_out_z) / k_out_x
    kx = k_out_dir[..., 0]
    ky = k_out_dir[..., 1]
    kz = k_out_dir[..., 2]
    pred_y = lsd_um * ky / (kx + 1e-30)
    pred_z = lsd_um * kz / (kx + 1e-30)
    return pred_y, pred_z


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_expected_hkls(
    om_fz_per_grain:  np.ndarray,    # (N, 3, 3) — consensus FZ orientations
    hkls:             HklTable,
    indexing_rings:   List[int],     # which rings the indexer used
    geom:             GeometryConfig,
) -> ExpectedHklTable:
    """For each grain × each variant on each indexing ring, predict
    (Y, Z, ω, η) and decide visibility.

    Returns a flat (M, ...) table where M = N_grains × N_variants × 2
    (the ×2 is the two Bragg ω-branches; both rows are present so the
    caller can match observed seed-spots against the right branch).

    When ``geom.use_midas_diffract`` is True (default), delegates to
    :func:`_predict_via_midas_diffract` which calls the canonical
    ``midas_diffract.HEDMForwardModel.calc_bragg_geometry`` — the
    same Bragg-quadratic solver the production C indexer uses (ports
    ``FF_HEDM/src/ForwardSimulationCompressed.c`` bit-for-bit), with
    correct wedge + sample-tilt handling. Otherwise falls back to the
    bespoke solver (kept for testing only — has a known ω error on hex
    datasets).
    """
    if geom.use_midas_diffract and geom.wavelength_a is not None:
        return _predict_via_midas_diffract(
            om_fz_per_grain, hkls, indexing_rings, geom,
        )

    # Legacy path (bespoke Bragg solver). Kept for unit testability;
    # production callers should set use_midas_diffract=True.
    keep = np.isin(hkls.ring, indexing_rings)
    h = hkls.h[keep]; k = hkls.k[keep]; l = hkls.l[keep]
    ring = hkls.ring[keep]
    g_crystal = hkls.g_crystal[keep]                              # (V, 3)
    theta_deg = hkls.theta_deg[keep]

    omega, eta, valid_bragg = _bragg_omega_eta(om_fz_per_grain, g_crystal, theta_deg)
    pred_y, pred_z = _omega_to_detector(om_fz_per_grain, g_crystal,
                                         omega, theta_deg, geom.lsd_um)

    # ω wrapping into the user's chosen scan range
    # The indexer's scan range may span [-180, 180] or [0, 360] depending
    # on convention. Normalise the predicted ω onto the range that includes
    # geom.omega_min_deg.
    span = geom.omega_max_deg - geom.omega_min_deg
    ome_wrapped = omega.copy()
    while np.any(ome_wrapped < geom.omega_min_deg):
        ome_wrapped = np.where(ome_wrapped < geom.omega_min_deg,
                                ome_wrapped + 360.0, ome_wrapped)
    while np.any(ome_wrapped > geom.omega_max_deg):
        ome_wrapped = np.where(ome_wrapped > geom.omega_max_deg,
                                ome_wrapped - 360.0, ome_wrapped)
    # After wrapping, anything still outside means the variant truly
    # falls outside the scan (span < 360°).
    omega_in_range = (ome_wrapped >= geom.omega_min_deg) & (ome_wrapped <= geom.omega_max_deg)
    eta_in_range   = (eta >= geom.eta_min_deg) & (eta <= geom.eta_max_deg)
    if geom.y_min_um is not None and geom.y_max_um is not None:
        y_in = (pred_y >= geom.y_min_um) & (pred_y <= geom.y_max_um)
    else:
        y_in = np.ones_like(pred_y, dtype=bool)
    if geom.z_min_um is not None and geom.z_max_um is not None:
        z_in = (pred_z >= geom.z_min_um) & (pred_z <= geom.z_max_um)
    else:
        z_in = np.ones_like(pred_z, dtype=bool)

    visible = valid_bragg & omega_in_range & eta_in_range & y_in & z_in   # (N, V, 2)

    # Flatten to (N * V * 2,)
    N = om_fz_per_grain.shape[0]
    V = h.shape[0]
    grain_idx = np.repeat(np.arange(N, dtype=np.int64), V * 2)
    h_flat = np.tile(np.repeat(h, 2), N)
    k_flat = np.tile(np.repeat(k, 2), N)
    l_flat = np.tile(np.repeat(l, 2), N)
    ring_flat = np.tile(np.repeat(ring, 2), N)
    return ExpectedHklTable(
        grain_idx=grain_idx,
        h=h_flat.astype(np.int8), k=k_flat.astype(np.int8), l=l_flat.astype(np.int8),
        ring=ring_flat.astype(np.int32),
        pred_y_um=pred_y.reshape(-1),
        pred_z_um=pred_z.reshape(-1),
        pred_omega_deg=ome_wrapped.reshape(-1),
        pred_eta_deg=eta.reshape(-1),
        visible=visible.reshape(-1),
    )


def expected_visible_variants_per_grain(
    exp: ExpectedHklTable,
    n_grains: int,
) -> np.ndarray:
    """Return (N,) int32: count of unique visible signed (h,k,l) per grain.

    "Visible" means at least one of the two ω-branches falls within all the
    geometry predicates. The same signed (h,k,l) is counted once even if
    both Friedel-pair ω-branches are in range.
    """
    out = np.zeros(n_grains, dtype=np.int32)
    if exp.h.size == 0:
        return out

    # Pandas groupby is clear, robust to encoding collisions, and fast enough
    # at the scales we care about (N_grains × ~12 rows for {h00}-type rings).
    import pandas as pd
    df = pd.DataFrame({
        "g": exp.grain_idx.astype(np.int64),
        "h": exp.h.astype(np.int16),
        "k": exp.k.astype(np.int16),
        "l": exp.l.astype(np.int16),
        "vis": exp.visible.astype(bool),
    })
    any_per_variant = df.groupby(["g", "h", "k", "l"])["vis"].any()
    grain_counts = any_per_variant.groupby(level=0).sum()
    out[grain_counts.index.to_numpy().astype(np.int64)] = grain_counts.to_numpy().astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# midas_diffract delegation (canonical Bragg solver)
# ---------------------------------------------------------------------------


def _predict_via_midas_diffract(
    om_fz_per_grain: np.ndarray,
    hkls: HklTable,
    indexing_rings: List[int],
    geom: GeometryConfig,
) -> ExpectedHklTable:
    """Stage 2 using the canonical midas_diffract.HEDMForwardModel.

    Builds a one-shot HEDMGeometry + HEDMForwardModel from the geom config,
    calls ``calc_bragg_geometry`` to get (ω, η, 2θ, valid) for every
    (grain, ring-variant) pair, then ``project_to_detector`` for (Y, Z).

    Output schema matches the legacy ``predict_expected_hkls`` — same
    column order, same units (Y/Z in µm, ω/η in degrees, h/k/l signed
    int8). Drop-in replacement for downstream code.
    """
    import torch as _torch
    from midas_diffract.forward import (
        HEDMForwardModel, HEDMGeometry,
    )

    # ---- slice hkls to indexing rings ----
    keep = np.isin(hkls.ring, indexing_rings)
    if not keep.any():
        return ExpectedHklTable(
            grain_idx=np.array([], dtype=np.int64),
            h=np.array([], dtype=np.int8),
            k=np.array([], dtype=np.int8),
            l=np.array([], dtype=np.int8),
            ring=np.array([], dtype=np.int32),
            pred_y_um=np.array([]), pred_z_um=np.array([]),
            pred_omega_deg=np.array([]), pred_eta_deg=np.array([]),
            visible=np.array([], dtype=bool),
        )

    h_v = hkls.h[keep].astype(np.int32)
    k_v = hkls.k[keep].astype(np.int32)
    l_v = hkls.l[keep].astype(np.int32)
    ring_v = hkls.ring[keep].astype(np.int32)
    g_cryst = hkls.g_crystal[keep].astype(np.float64)
    theta_rad = np.deg2rad(hkls.theta_deg[keep]).astype(np.float64)
    M = len(h_v)
    N = om_fz_per_grain.shape[0]

    # ---- build the forward model ----
    span = geom.omega_max_deg - geom.omega_min_deg
    n_frames = max(int(round(span / max(geom.omega_step_deg, 1e-6))), 1)
    fm_geom = HEDMGeometry(
        Lsd=geom.lsd_um,
        y_BC=geom.y_BC, z_BC=geom.z_BC,
        px=geom.pixel_um,
        omega_start=geom.omega_min_deg, omega_step=geom.omega_step_deg,
        n_frames=n_frames,
        n_pixels_y=geom.n_pixels_y, n_pixels_z=geom.n_pixels_z,
        min_eta=geom.min_eta_deg,
        wavelength=geom.wavelength_a,
        tx=geom.tx_deg, ty=geom.ty_deg, tz=geom.tz_deg,
        wedge=geom.wedge_deg,
        flip_y=True,
        apply_tilts=True,   # apply detector tilts even in FF mode (raw prediction)
    )
    model = HEDMForwardModel(
        hkls=_torch.from_numpy(g_cryst),
        thetas=_torch.from_numpy(theta_rad),
        geometry=fm_geom,
        device=_torch.device("cpu"),
    )

    # ---- bragg geometry: (N, 3, 3) → (N, 2M) omega, eta, 2θ, valid ----
    OM = _torch.from_numpy(np.ascontiguousarray(om_fz_per_grain, dtype=np.float64))
    omega_rad, eta_rad, two_theta, valid_bragg = model.calc_bragg_geometry(
        orientation_matrices=OM
    )

    # calc_bragg_geometry shapes:
    #   omega, eta, two_theta, valid_bragg : (2N, M)   -- but when input is (N, 3, 3),
    #   the leading "..." dim is empty, so it's (2, M) per orientation. Reshape.
    #
    # Actually the docstring says (..., 2N, M) where 2N stacks two ω solutions
    # per orientation. With input (N, 3, 3), output is (N, 2, M).
    # Confirm by reshape:
    omega_rad = omega_rad.detach().cpu().numpy()    # (N, 2, M) expected
    eta_rad = eta_rad.detach().cpu().numpy()
    valid_b = valid_bragg.detach().cpu().numpy().astype(bool)
    if omega_rad.ndim == 2:
        # If model collapsed N=1 case to (2, M), broadcast back
        omega_rad = omega_rad[None, ...]
        eta_rad = eta_rad[None, ...]
        valid_b = valid_b[None, ...]
    # Some forward-model branches return (2N, M) where 2N stacks orientations.
    # Reshape to (N, 2, M).
    if omega_rad.shape[0] == 2 * N:
        omega_rad = omega_rad.reshape(N, 2, M)
        eta_rad   = eta_rad.reshape(N, 2, M)
        valid_b   = valid_b.reshape(N, 2, M)

    omega_deg = np.degrees(omega_rad)
    eta_deg   = np.degrees(eta_rad)

    # ---- project to detector for Y, Z ----
    # project_to_detector wants positions (N, 3) for the grain centers.
    # We pass zeros for Stage 2 (visibility check ignores position
    # offset; the relevant predicate is "is this on the detector area")
    positions = _torch.zeros((N, 3), dtype=OM.dtype)
    try:
        spots = model.project_to_detector(
            omega=_torch.from_numpy(omega_rad),
            eta=_torch.from_numpy(eta_rad),
            two_theta=two_theta,
            positions=positions,
            valid=_torch.from_numpy(valid_b),
        )
        # spots.y_pixel / spots.z_pixel are (D, N, 2, M) where D=1 for FF
        y_pixel = spots.y_pixel.detach().cpu().numpy()
        z_pixel = spots.z_pixel.detach().cpu().numpy()
        valid_proj = spots.valid.detach().cpu().numpy().astype(bool)
        # Drop the D dim
        if y_pixel.ndim == 4:
            y_pixel = y_pixel[0]; z_pixel = z_pixel[0]; valid_proj = valid_proj[0]
        # Convert pixel → µm (lab frame YLab/ZLab style)
        # For flip_y=True (FF), the relation is yLab = (yBC - y_pixel) * px
        pred_y_um = (geom.y_BC - y_pixel) * geom.pixel_um
        pred_z_um = (z_pixel - geom.z_BC) * geom.pixel_um
    except Exception:
        # Fallback: project ω+η directly without the full distortion chain
        pred_y_um = np.zeros_like(omega_deg)
        pred_z_um = np.zeros_like(omega_deg)
        valid_proj = valid_b.copy()

    # ---- wrap omega into the requested scan range, count as visible
    # iff valid_bragg AND valid_proj AND ω in scan range AND η in eta range ----
    ome_wrapped = omega_deg.copy()
    for _ in range(4):
        ome_wrapped = np.where(ome_wrapped < geom.omega_min_deg,
                                ome_wrapped + 360.0, ome_wrapped)
        ome_wrapped = np.where(ome_wrapped > geom.omega_max_deg,
                                ome_wrapped - 360.0, ome_wrapped)
    omega_ok = (ome_wrapped >= geom.omega_min_deg) & (ome_wrapped <= geom.omega_max_deg)
    eta_ok = (eta_deg >= geom.eta_min_deg) & (eta_deg <= geom.eta_max_deg)
    if geom.y_min_um is not None and geom.y_max_um is not None:
        y_ok = (pred_y_um >= geom.y_min_um) & (pred_y_um <= geom.y_max_um)
    else:
        y_ok = np.ones_like(omega_ok)
    if geom.z_min_um is not None and geom.z_max_um is not None:
        z_ok = (pred_z_um >= geom.z_min_um) & (pred_z_um <= geom.z_max_um)
    else:
        z_ok = np.ones_like(omega_ok)
    visible = valid_b & valid_proj & omega_ok & eta_ok & y_ok & z_ok

    # ---- flatten (N, 2, M) → (N*2*M,) in the same order as legacy
    # (per-variant, both Friedel branches contiguous)
    grain_idx = np.repeat(np.arange(N, dtype=np.int64), 2 * M)
    h_flat = np.tile(np.tile(h_v, 2).reshape(2, M).T.reshape(-1), N)  # variants outer, branches inner — matches legacy: h_flat = tile(repeat(h, 2), N)
    k_flat = np.tile(np.tile(k_v, 2).reshape(2, M).T.reshape(-1), N)
    l_flat = np.tile(np.tile(l_v, 2).reshape(2, M).T.reshape(-1), N)
    ring_flat = np.tile(np.tile(ring_v, 2).reshape(2, M).T.reshape(-1), N)

    # Reorder model output (N, 2, M) → flat order (N, M, 2)
    # to match legacy axis convention (variants outer, branches inner)
    omega_flat = ome_wrapped.transpose(0, 2, 1).reshape(-1)
    eta_flat = eta_deg.transpose(0, 2, 1).reshape(-1)
    y_flat = pred_y_um.transpose(0, 2, 1).reshape(-1)
    z_flat = pred_z_um.transpose(0, 2, 1).reshape(-1)
    vis_flat = visible.transpose(0, 2, 1).reshape(-1)

    return ExpectedHklTable(
        grain_idx=grain_idx,
        h=h_flat.astype(np.int8), k=k_flat.astype(np.int8), l=l_flat.astype(np.int8),
        ring=ring_flat.astype(np.int32),
        pred_y_um=y_flat, pred_z_um=z_flat,
        pred_omega_deg=omega_flat, pred_eta_deg=eta_flat,
        visible=vis_flat,
    )
