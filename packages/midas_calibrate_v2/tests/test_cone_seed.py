"""Cone-aware BC refinement on synthetic tilted-detector rings.

Locks in the production wire-in of :func:`midas_calibrate_v2.seed.cone.cone_aware_bc_refine`
that follows the Hough-circle vote in :func:`hough_seed_bc_lsd`.

The synthetic forward projection mirrors the POC at
``dev/paper/runners/run_cone_aware_seed.py``.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest


def _project_ring(
    bc_y: float,
    bc_z: float,
    Lsd_um: float,
    ty_deg: float,
    tz_deg: float,
    two_theta_deg: float,
    p_x: float,
    n_points: int = 360,
    rng: np.random.Generator | None = None,
    jitter_px: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-project a Debye-Scherrer cone of half-angle 2theta onto a
    detector tilted by (ty, tz).  Anchored at BC so 2theta=0 hits exactly
    (BC_y, BC_z) on the detector.  Mirrors the POC at
    dev/paper/runners/run_cone_aware_seed.py."""
    if rng is None:
        rng = np.random.default_rng(0)
    eta = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    tt = math.radians(two_theta_deg)
    cy_t, sy_t = math.cos(math.radians(ty_deg)), math.sin(math.radians(ty_deg))
    cz_t, sz_t = math.cos(math.radians(tz_deg)), math.sin(math.radians(tz_deg))
    ex_det = np.array([cz_t * cy_t, sz_t * cy_t, -sy_t])
    ey_det = np.array([-sz_t, cz_t, 0.0])
    ez_det = np.array([cz_t * sy_t, sz_t * sy_t, cy_t])
    p0 = np.array([Lsd_um, 0.0, 0.0])
    sx = math.cos(tt)
    sy = math.sin(tt) * np.sin(eta)
    sz = math.sin(tt) * np.cos(eta)
    cone = np.column_stack([np.full_like(sy, sx), sy, sz])
    denom = cone @ ex_det
    keep = np.abs(denom) > 1e-12
    t_num = float(p0 @ ex_det)
    t = np.where(keep, t_num / denom, 0.0)
    pts = t[:, None] * cone
    rel = pts - p0
    Y = (rel @ ey_det) / p_x + bc_y
    Z = (rel @ ez_det) / p_x + bc_z
    if jitter_px > 0:
        Y = Y + rng.normal(0.0, jitter_px, Y.shape)
        Z = Z + rng.normal(0.0, jitter_px, Z.shape)
    return Y[keep], Z[keep]


def _build_image(
    bc_y: float,
    bc_z: float,
    Lsd_um: float,
    p_x: float,
    ty_deg: float,
    tt_deg: np.ndarray,
    npy: int = 2048,
    npz: int = 2048,
) -> np.ndarray:
    """Render synthetic edge points into an above-MAD detector image."""
    img = np.zeros((npz, npy), dtype=np.float64)
    rng = np.random.default_rng(0)
    for tt in tt_deg:
        Y, Z = _project_ring(
            bc_y, bc_z, Lsd_um, ty_deg=ty_deg, tz_deg=0.0,
            two_theta_deg=float(tt), p_x=p_x,
            n_points=720, rng=rng, jitter_px=0.5,
        )
        yi = np.round(Y).astype(np.int64)
        zi = np.round(Z).astype(np.int64)
        in_frame = (yi >= 0) & (yi < npy) & (zi >= 0) & (zi < npz)
        img[zi[in_frame], yi[in_frame]] += 50.0
    return img


def _ceo2_two_theta(Lsd_um: float, p_x: float, lam_A: float = 0.197,
                    a_A: float = 5.4116) -> Tuple[np.ndarray, np.ndarray]:
    hkl_sq = np.array([3, 4, 8, 11, 12, 16, 19, 20, 24, 27, 32], dtype=np.float64)
    hkl_norm = np.sqrt(hkl_sq)
    sintheta = lam_A * hkl_norm / (2.0 * a_A)
    keep = sintheta < 0.95
    tt = 2.0 * np.degrees(np.arcsin(sintheta[keep]))
    radii_px = Lsd_um * np.tan(np.radians(tt)) / p_x
    return tt, radii_px


def test_fit_ellipse_recovers_circle_centre():
    from midas_calibrate_v2.seed.cone import fit_ellipse
    rng = np.random.default_rng(0)
    eta = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    cy, cz, r = 100.0, 75.0, 50.0
    y = cy + r * np.cos(eta) + rng.normal(0, 0.2, eta.size)
    z = cz + r * np.sin(eta) + rng.normal(0, 0.2, eta.size)
    centre = fit_ellipse(y, z)
    assert centre is not None
    assert abs(centre[0] - cy) < 0.5
    assert abs(centre[1] - cz) < 0.5


def test_cone_refine_recovers_bc_with_clean_rings():
    """Validates the math: given clean per-ring edge points (rings
    individually identified) on a 15 degree tilted detector, the
    ellipse-fit + (2theta -> 0) extrapolation recovers the true BC
    to sub-pixel precision regardless of the seed.

    This is the strongest claim that the algorithm validates; the
    production hook in :func:`hough_seed_bc_lsd` chains this with
    the Hough vote but is opt-in because per-ring ring isolation
    from a tilt-biased seed BC requires further engineering before
    it can be enabled by default without regressing on clean cases.
    """
    from midas_calibrate_v2.seed.cone import fit_ellipse
    bc_y, bc_z = 1024.0, 1024.0
    Lsd, p_x = 900_000.0, 150.0
    ty_deg = 15.0
    tt_deg, _ = _ceo2_two_theta(Lsd, p_x)

    rng = np.random.default_rng(0)
    cy_per_ring, cz_per_ring = [], []
    for tt in tt_deg:
        Y, Z = _project_ring(
            bc_y, bc_z, Lsd, ty_deg=ty_deg, tz_deg=0.0,
            two_theta_deg=float(tt), p_x=p_x,
            n_points=720, rng=rng, jitter_px=0.5,
        )
        fit = fit_ellipse(Y, Z)
        assert fit is not None, f"ellipse fit failed at 2theta={tt}"
        cy_per_ring.append(fit[0])
        cz_per_ring.append(fit[1])

    x = np.tan(np.radians(tt_deg)) ** 2
    py = np.polyfit(x, cy_per_ring, 1)
    pz = np.polyfit(x, cz_per_ring, 1)
    err = math.hypot(py[1] - bc_y, pz[1] - bc_z)
    assert err < 0.5, (
        f"clean-ring extrapolation BC error {err:.3f} px exceeds 0.5 px"
    )


def test_cone_refine_does_no_harm_when_seed_is_truth():
    """When the seed BC is at truth and rings are perfect circles
    (zero tilt), the production cone-aware refinement must not move
    the BC away from truth by more than a small numerical amount."""
    from midas_calibrate_v2.seed.cone import cone_aware_bc_refine
    bc_y, bc_z = 1024.0, 1024.0
    Lsd, p_x = 900_000.0, 150.0
    tt_deg, radii_px = _ceo2_two_theta(Lsd, p_x)
    img = _build_image(bc_y, bc_z, Lsd, p_x, ty_deg=0.0, tt_deg=tt_deg)
    bc_y_out, bc_z_out, n_used = cone_aware_bc_refine(
        img,
        bc_y_seed=bc_y,
        bc_z_seed=bc_z,
        sim_radii_px=radii_px,
        two_theta_rad=np.radians(tt_deg),
    )
    err = math.hypot(bc_y_out - bc_y, bc_z_out - bc_z)
    assert err < 5.0, (
        f"refinement at zero tilt drifted from truth by {err:.2f} px"
    )


def test_cone_refine_falls_back_when_no_rings():
    """When the seed BC is far enough that no edges land in the gating
    band, the refinement must return the seed BC and n_used=0."""
    from midas_calibrate_v2.seed.cone import cone_aware_bc_refine
    img = np.zeros((512, 512), dtype=np.float64)
    bc_y, bc_z, _ = cone_aware_bc_refine(
        img,
        bc_y_seed=256.0,
        bc_z_seed=256.0,
        sim_radii_px=np.array([50.0, 100.0, 150.0]),
        two_theta_rad=np.radians(np.array([3.0, 6.0, 9.0])),
    )
    assert bc_y == 256.0 and bc_z == 256.0


def test_cone_refine_with_tilt_prior_recovers_bc_at_15_degree_tilt():
    """With a tilt prior accurate to within a few degrees, the
    refinement gates edges around the predicted ellipse-centre per
    ring (not around the biased seed BC) and recovers the true BC
    to within the LM basin width even when the Hough seed is off.
    """
    from midas_calibrate_v2.seed.cone import (
        cone_aware_bc_refine_with_tilt_prior,
    )
    bc_y, bc_z = 1024.0, 1024.0
    Lsd, p_x = 900_000.0, 150.0
    ty_truth_deg = 15.0
    tt_deg, radii_px = _ceo2_two_theta(Lsd, p_x)
    img = _build_image(bc_y, bc_z, Lsd, p_x, ty_truth_deg, tt_deg)

    # Tilt prior offset by 3 degrees from truth — well within the
    # +-5 degree user-supplied prior accuracy the API contract assumes.
    ty_prior_deg = 12.0

    # Hough seed is biased on tilted detectors; pretend the user got
    # this much from the Hough vote (35-50 px in our synthetic image).
    bc_y_seed = bc_y + 30.0
    bc_z_seed = bc_z + 20.0

    bc_y_out, bc_z_out, n_used = cone_aware_bc_refine_with_tilt_prior(
        img,
        bc_y_seed=bc_y_seed,
        bc_z_seed=bc_z_seed,
        Lsd_um=Lsd,
        p_x_um=p_x,
        sim_radii_px=radii_px,
        two_theta_rad=np.radians(tt_deg),
        tilt_y_prior_rad=math.radians(ty_prior_deg),
        tilt_z_prior_rad=0.0,
    )
    err_seed = math.hypot(bc_y_seed - bc_y, bc_z_seed - bc_z)
    err_out = math.hypot(bc_y_out - bc_y, bc_z_out - bc_z)
    assert n_used >= 3, f"too few rings used: {n_used}"
    assert err_out < err_seed, (
        f"refinement did not improve seed: {err_seed:.2f} -> {err_out:.2f} px"
    )
    # 5 px is well inside the 60 px LM basin and tighter than the
    # `cone_aware_bc_refine` (no-tilt-prior) variant achieves at the
    # same tilt and seed bias.
    assert err_out < 5.0, (
        f"cone-aware (tilt prior) BC error {err_out:.2f} px exceeds 5 px"
    )


def test_cone_refine_with_zero_tilt_prior_matches_seed_at_zero_tilt():
    """Sanity: zero tilt + zero tilt prior = production cone refinement
    must not displace the BC from a near-truth seed."""
    from midas_calibrate_v2.seed.cone import (
        cone_aware_bc_refine_with_tilt_prior,
    )
    bc_y, bc_z = 1024.0, 1024.0
    Lsd, p_x = 900_000.0, 150.0
    tt_deg, radii_px = _ceo2_two_theta(Lsd, p_x)
    img = _build_image(bc_y, bc_z, Lsd, p_x, ty_deg=0.0, tt_deg=tt_deg)
    bc_y_out, bc_z_out, n = cone_aware_bc_refine_with_tilt_prior(
        img,
        bc_y_seed=bc_y, bc_z_seed=bc_z,
        Lsd_um=Lsd, p_x_um=p_x,
        sim_radii_px=radii_px,
        two_theta_rad=np.radians(tt_deg),
        tilt_y_prior_rad=0.0, tilt_z_prior_rad=0.0,
    )
    err = math.hypot(bc_y_out - bc_y, bc_z_out - bc_z)
    assert err < 2.0, f"zero-tilt drift {err:.2f} px exceeds 2 px"


def test_hough_seed_with_cone_aware_chains_does_not_break_pipeline():
    """End-to-end opt-in: hough_seed_bc_lsd called with
    ``cone_aware=True`` and ``two_theta_rad`` returns a finite BC that
    keeps the seed within the LM basin width on a tilted detector.

    This locks in the API surface (kwargs accepted, return signature
    unchanged); robust ring isolation that brings the refined BC to
    sub-pixel precision is future work.
    """
    from midas_calibrate_v2.seed.hough import hough_seed_bc_lsd
    bc_y, bc_z = 1024.0, 1024.0
    Lsd, p_x = 900_000.0, 150.0
    tt_deg, radii_px = _ceo2_two_theta(Lsd, p_x)
    img = _build_image(bc_y, bc_z, Lsd, p_x, ty_deg=10.0, tt_deg=tt_deg)
    bc_y_out, bc_z_out, lsd_out, n_match = hough_seed_bc_lsd(
        img,
        sim_radii_px=radii_px,
        initial_lsd=Lsd,
        bc_guess=(bc_y, bc_z),
        bc_search_radius_px=80.0,
        n_lsd_candidates=4,
        sample_step_pixels=1,
        accumulator_blur_sigma=1.0,
        two_theta_rad=np.radians(tt_deg),
        cone_aware=True,
    )
    err = math.hypot(bc_y_out - bc_y, bc_z_out - bc_z)
    assert math.isfinite(err)
    # 60 px is the documented LM basin width.
    assert err < 60.0, (
        f"chained Hough+cone BC error {err:.2f} px exceeds LM basin"
    )
