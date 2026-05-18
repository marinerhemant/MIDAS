"""Cone-aware BC refinement for tilted detectors.

The Hough-circle seeder (:mod:`.hough`) assumes that Debye-Scherrer
rings are circles on the detector.  This holds at zero detector tilt;
under non-trivial tilt the cone-detector intersection is an ellipse
whose centre is offset from the true beam centre by

.. math::

    \\Delta = L_\\mathrm{sd}\\,\\tan(\\alpha)\\,\\tan^{2}(2\\theta),

where :math:`\\alpha` is the detector tilt and :math:`2\\theta` is the
ring opening angle.  Each ring fits to a different ellipse centre;
extrapolating the per-ring centres to :math:`2\\theta \\to 0` recovers
the true BC exactly.  This module implements that refinement on top
of an existing (tilt-biased) Hough seed.

The synthetic POC validating this approach against a circle-Hough
baseline is at
``packages/midas_calibrate_v2/dev/paper/runners/run_cone_aware_seed.py``;
on a Varex geometry at 15 degrees tilt with 60 px BC perturbation,
the refinement recovers BC to sub-pixel precision (~260x improvement
over the naive Hough-circle limit).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def fit_ellipse(y: np.ndarray, z: np.ndarray) -> Optional[Tuple[float, float]]:
    """Fitzgibbon-Pilu-Fisher direct least-squares ellipse centre fit.

    Solves the constrained generalised eigenvalue problem
    :math:`\\min \\|D a\\|^2` subject to :math:`4ac - b^2 = 1` for the
    conic :math:`a y^2 + b y z + c z^2 + d y + e z + f = 0`.

    Parameters
    ----------
    y, z : 1-D float arrays of pixel coordinates on a single ring.

    Returns
    -------
    ``(centre_y, centre_z)`` if a valid ellipse fit is found, else
    ``None``.
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if y.size < 6:
        return None
    D = np.column_stack([y * y, y * z, z * z, y, z, np.ones_like(y)])
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = 2.0
    C[2, 0] = 2.0
    C[1, 1] = -1.0
    try:
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.solve(S, C))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(eig_vals).any():
        return None
    pos = int(np.argmax(eig_vals.real))
    a = eig_vecs[:, pos].real
    A_, B_, Cc_, D_, E_, _F_ = a
    disc = B_ * B_ - 4.0 * A_ * Cc_
    if abs(disc) < 1e-14:
        return None
    cy = (2.0 * Cc_ * D_ - B_ * E_) / disc
    cz = (2.0 * A_ * E_ - B_ * D_) / disc
    if not (np.isfinite(cy) and np.isfinite(cz)):
        return None
    return float(cy), float(cz)


def _edge_pixels(
    image: np.ndarray,
    *,
    panel_mask: Optional[np.ndarray] = None,
    edge_thresh_mad_factor: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(y_edge, z_edge)`` for above-MAD pixels.

    Mirrors the threshold rule used by :func:`.hough.hough_circle_bc`
    so that the cone-aware refinement consumes the same edge set.
    """
    img = np.asarray(image, dtype=np.float64)
    if panel_mask is not None:
        from .mask import erode_mask
        m = erode_mask(panel_mask.astype(bool), iterations=2)
        img = np.where(m, img, 0.0)
    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    cutoff = med + edge_thresh_mad_factor * (1.4826 * mad + 1.0)
    edges = img > cutoff
    if panel_mask is not None:
        from .mask import erode_mask
        edges &= erode_mask(panel_mask.astype(bool), iterations=2)
    z_edge, y_edge = np.where(edges)
    return y_edge.astype(np.float64), z_edge.astype(np.float64)


def _robust_linear_intercept(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    min_kept: int,
    mad_factor: float = 2.5,
) -> Tuple[float, float, int]:
    """Linear fit ``(y, z) = a*x + b`` with MAD-based outlier rejection.

    Used to extrapolate per-ring ellipse centres to ``x = tan^2(2theta) = 0``
    when individual ellipse fits may be polluted by neighbouring-ring
    contamination.
    """
    if x.size < min_kept:
        return float("nan"), float("nan"), 0
    py = np.polyfit(x, y, 1)
    pz = np.polyfit(x, z, 1)
    res_y = y - np.polyval(py, x)
    res_z = z - np.polyval(pz, x)
    res = np.hypot(res_y, res_z)
    mad = float(np.median(np.abs(res - np.median(res))))
    if mad <= 0.0:
        return float(py[1]), float(pz[1]), int(x.size)
    keep = res < mad_factor * 1.4826 * mad + np.median(res)
    if int(keep.sum()) < min_kept:
        # Outlier rejection killed too many rings; fall back to the
        # naive fit on all rings, which at least uses the data we have.
        return float(py[1]), float(pz[1]), int(x.size)
    py = np.polyfit(x[keep], y[keep], 1)
    pz = np.polyfit(x[keep], z[keep], 1)
    return float(py[1]), float(pz[1]), int(keep.sum())


def cone_aware_bc_refine(
    image: np.ndarray,
    *,
    bc_y_seed: float,
    bc_z_seed: float,
    sim_radii_px: Sequence[float],
    two_theta_rad: Sequence[float],
    panel_mask: Optional[np.ndarray] = None,
    edge_thresh_mad_factor: float = 3.0,
    radius_tol_frac: float = 0.10,
    radius_tol_min_px: float = 15.0,
    min_points_per_ring: int = 60,
    min_rings_for_extrapolation: int = 2,
    n_iterations: int = 2,
) -> Tuple[float, float, int]:
    """Refine a tilt-biased Hough BC seed by per-ring ellipse fits.

    Edge pixels within ``radius_tol_frac`` of each predicted ring radius
    (computed from the seed BC) are gathered into per-ring point sets;
    each set is fit to an ellipse via :func:`fit_ellipse`; the resulting
    per-ring centres are linearly extrapolated in
    :math:`(\\tan^{2}(2\\theta), \\text{centre})` to the
    :math:`2\\theta \\to 0` intercept, which is the true BC.

    Parameters
    ----------
    image : 2-D detector frame (same convention as
        :func:`.hough.hough_circle_bc`).
    bc_y_seed, bc_z_seed : Hough-vote BC, biased on tilted detectors.
    sim_radii_px : predicted ring radii in pixels at the current Lsd.
    two_theta_rad : matching 2theta values (radians); same length as
        ``sim_radii_px``.
    panel_mask : optional binary mask applied during edge detection.
    edge_thresh_mad_factor : MAD threshold for edge pixels (mirrors
        :func:`.hough.hough_circle_bc`).
    radius_tol_frac : per-ring radial gating fraction; pixels within
        ``r * (1 - tol)`` to ``r * (1 + tol)`` of the seed BC are
        candidates for that ring.  The default of 10% is wide enough
        to capture an ellipse offset by the worst-case
        :math:`L_\\mathrm{sd}\\,\\tan(\\alpha)\\,\\tan^{2}(2\\theta)`
        bias even at 15 degrees of tilt.
    radius_tol_min_px : absolute lower bound on the gating half-width;
        used when ``r * radius_tol_frac`` is small (small rings or
        downsampled images).
    n_iterations : number of refine-then-regate passes.  The first
        pass uses the wide gating; subsequent passes re-gate around
        the refined BC, which lets the per-ring ellipses contribute
        more pixels and stabilises the centre estimate.
    min_points_per_ring : minimum number of pixels required for a ring
        to enter the ellipse fit.
    min_rings_for_extrapolation : minimum number of rings with valid
        ellipse fits required for the BC extrapolation.  Below this
        the refinement falls back to the seed BC.

    Returns
    -------
    ``(bc_y, bc_z, n_rings_used)``.  When the refinement is unsafe
    (too few rings, ill-conditioned ellipse fits) the seed values
    are returned with ``n_rings_used == 0``.
    """
    sim_radii = np.asarray(sim_radii_px, dtype=np.float64)
    tt_rad = np.asarray(two_theta_rad, dtype=np.float64)
    if sim_radii.size != tt_rad.size:
        raise ValueError("sim_radii_px and two_theta_rad must have the same length")
    if sim_radii.size < min_rings_for_extrapolation:
        return float(bc_y_seed), float(bc_z_seed), 0

    y_edge, z_edge = _edge_pixels(
        image,
        panel_mask=panel_mask,
        edge_thresh_mad_factor=edge_thresh_mad_factor,
    )
    if y_edge.size == 0:
        return float(bc_y_seed), float(bc_z_seed), 0

    bc_y_cur = float(bc_y_seed)
    bc_z_cur = float(bc_z_seed)
    n_rings = 0
    for it in range(max(1, int(n_iterations))):
        # Tighten the gating on subsequent passes once the BC has been
        # corrected by the first ellipse-extrapolation step; this lets
        # high-2theta rings contribute cleaner per-ring centres.
        if it == 0:
            half_floor = float(radius_tol_min_px)
            tol_frac = float(radius_tol_frac)
        else:
            half_floor = max(float(radius_tol_min_px) * 0.5, 5.0)
            tol_frac = float(radius_tol_frac) * 0.5

        rho = np.hypot(y_edge - bc_y_cur, z_edge - bc_z_cur)
        cy_per_ring: list[float] = []
        cz_per_ring: list[float] = []
        tt_per_ring: list[float] = []
        for r, tt in zip(sim_radii, tt_rad):
            if r <= 0 or not np.isfinite(r):
                continue
            half_width = max(r * tol_frac, half_floor)
            lo = r - half_width
            hi = r + half_width
            sel = (rho >= lo) & (rho <= hi)
            n = int(sel.sum())
            if n < min_points_per_ring:
                continue
            fit = fit_ellipse(y_edge[sel], z_edge[sel])
            if fit is None:
                continue
            cy, cz = fit
            if (abs(cy - bc_y_cur) > 4.0 * r
                    or abs(cz - bc_z_cur) > 4.0 * r):
                continue
            cy_per_ring.append(cy)
            cz_per_ring.append(cz)
            tt_per_ring.append(float(tt))

        n_rings = len(tt_per_ring)
        if n_rings < min_rings_for_extrapolation:
            # Iteration produced no usable rings; return the best-so-far
            # BC and signal failure with n_rings=0 only on the first
            # pass (a successful first pass followed by a noisy second
            # pass keeps the first pass's refinement).
            if it == 0:
                return float(bc_y_seed), float(bc_z_seed), 0
            break

        x_extrap = np.tan(np.asarray(tt_per_ring, dtype=np.float64)) ** 2
        cy_arr = np.asarray(cy_per_ring, dtype=np.float64)
        cz_arr = np.asarray(cz_per_ring, dtype=np.float64)

        # Robustify: drop per-ring centres whose residual to an initial
        # linear fit exceeds 2.5*MAD.  Imperfect ring grouping (pixels
        # bleeding from a neighbouring ring into the gating window)
        # produces individual ellipse-centre outliers that would
        # otherwise dominate the (only 5-10 ring) linear regression.
        bc_y_next, bc_z_next, n_kept = _robust_linear_intercept(
            x_extrap, cy_arr, cz_arr,
            min_kept=min_rings_for_extrapolation,
        )
        if not (np.isfinite(bc_y_next) and np.isfinite(bc_z_next)):
            if it == 0:
                return float(bc_y_seed), float(bc_z_seed), 0
            break
        bc_y_cur, bc_z_cur = bc_y_next, bc_z_next
        n_rings = n_kept

    return bc_y_cur, bc_z_cur, n_rings


def cone_aware_bc_refine_with_tilt_prior(
    image: np.ndarray,
    *,
    bc_y_seed: float,
    bc_z_seed: float,
    Lsd_um: float,
    p_x_um: float,
    sim_radii_px: Sequence[float],
    two_theta_rad: Sequence[float],
    tilt_y_prior_rad: float,
    tilt_z_prior_rad: float,
    panel_mask: Optional[np.ndarray] = None,
    edge_thresh_mad_factor: float = 3.0,
    radius_tol_px: float = 12.0,
    min_points_per_ring: int = 60,
    min_rings_for_extrapolation: int = 2,
) -> Tuple[float, float, int]:
    """Cone-aware BC refinement using a user-supplied tilt prior.

    On a detector tilted by :math:`(\\alpha_y, \\alpha_z)` the
    Debye-Scherrer cone of half-angle :math:`2\\theta` projects to an
    ellipse whose centre is offset from the true beam centre by

    .. math::

        (\\Delta_y, \\Delta_z) \\approx (L_\\mathrm{sd}/p_x)\\,
            \\tan^{2}(2\\theta)\\,
            \\bigl(\\tan(\\alpha_z),\\,\\tan(\\alpha_y)\\bigr).

    Given a tilt prior accurate to within roughly :math:`\\pm 5^\\circ`,
    the production refinement gates edge pixels by distance from the
    \\emph{predicted ellipse centre} (rather than from the tilt-biased
    seed BC); this isolates each ring cleanly even when the seed BC
    sits outside the LM basin.  Per-ring ellipse fits are then
    extrapolated in :math:`(\\tan^{2}(2\\theta), \\text{centre})` to
    the :math:`2\\theta \\to 0` intercept, which is the true BC.

    Parameters
    ----------
    image : 2-D detector frame.
    bc_y_seed, bc_z_seed : Hough-vote BC seed (may be tilt-biased).
    Lsd_um : sample-to-detector distance, micrometres.
    p_x_um : pixel pitch, micrometres.
    sim_radii_px : predicted ring radii in pixels at the seed Lsd.
    two_theta_rad : matching 2theta values (radians).
    tilt_y_prior_rad, tilt_z_prior_rad : user-supplied tilt angles.
        Accuracy of about :math:`\\pm 5^\\circ` is sufficient for the
        gating to capture the per-ring ellipses.  Pass ``0.0`` for
        nominal-perpendicular detectors.
    radius_tol_px : absolute radial gating half-width in pixels around
        the predicted ellipse-centre + radius.  The default of 12 px
        is wide enough to absorb the residual error left by a roughly
        :math:`\\pm 5^\\circ` tilt-prior mismatch and the
        ellipse-vs-circle deformation along the ring.

    Returns
    -------
    ``(bc_y, bc_z, n_rings_used)``.  Falls back to the seed BC with
    ``n_rings_used == 0`` when fewer than ``min_rings_for_extrapolation``
    rings yield valid ellipse fits.
    """
    sim_radii = np.asarray(sim_radii_px, dtype=np.float64)
    tt_rad = np.asarray(two_theta_rad, dtype=np.float64)
    if sim_radii.size != tt_rad.size:
        raise ValueError(
            "sim_radii_px and two_theta_rad must have the same length"
        )
    if sim_radii.size < min_rings_for_extrapolation:
        return float(bc_y_seed), float(bc_z_seed), 0

    y_edge, z_edge = _edge_pixels(
        image,
        panel_mask=panel_mask,
        edge_thresh_mad_factor=edge_thresh_mad_factor,
    )
    if y_edge.size == 0:
        return float(bc_y_seed), float(bc_z_seed), 0

    # Predicted per-ring ellipse-centre offsets from the BC seed under
    # the tilt prior (small-tilt approximation; accurate for the prior
    # ranges where users supply this hook).
    tan_y = float(np.tan(tilt_y_prior_rad))
    tan_z = float(np.tan(tilt_z_prior_rad))
    Lsd_over_px = float(Lsd_um) / float(p_x_um)
    delta_y = Lsd_over_px * tan_z * np.tan(tt_rad) ** 2
    delta_z = Lsd_over_px * tan_y * np.tan(tt_rad) ** 2

    cy_per_ring: list[float] = []
    cz_per_ring: list[float] = []
    tt_per_ring: list[float] = []
    for r, tt, dy, dz in zip(sim_radii, tt_rad, delta_y, delta_z):
        if r <= 0 or not np.isfinite(r):
            continue
        # Gate around the predicted ellipse centre at this ring (not
        # around the biased seed BC).
        cy_pred = float(bc_y_seed) + float(dy)
        cz_pred = float(bc_z_seed) + float(dz)
        rho = np.hypot(y_edge - cy_pred, z_edge - cz_pred)
        sel = (rho >= r - radius_tol_px) & (rho <= r + radius_tol_px)
        n = int(sel.sum())
        if n < min_points_per_ring:
            continue
        fit = fit_ellipse(y_edge[sel], z_edge[sel])
        if fit is None:
            continue
        cy, cz = fit
        # Reject implausibly far ellipse centres (a degenerate fit on
        # noise can return a centre far from the prior).
        if (abs(cy - cy_pred) > 4.0 * radius_tol_px
                or abs(cz - cz_pred) > 4.0 * radius_tol_px):
            continue
        cy_per_ring.append(cy)
        cz_per_ring.append(cz)
        tt_per_ring.append(float(tt))

    n_rings = len(tt_per_ring)
    if n_rings < min_rings_for_extrapolation:
        return float(bc_y_seed), float(bc_z_seed), 0

    x_extrap = np.tan(np.asarray(tt_per_ring, dtype=np.float64)) ** 2
    cy_arr = np.asarray(cy_per_ring, dtype=np.float64)
    cz_arr = np.asarray(cz_per_ring, dtype=np.float64)
    bc_y, bc_z, n_kept = _robust_linear_intercept(
        x_extrap, cy_arr, cz_arr,
        min_kept=min_rings_for_extrapolation,
    )
    if not (np.isfinite(bc_y) and np.isfinite(bc_z)):
        return float(bc_y_seed), float(bc_z_seed), 0
    return float(bc_y), float(bc_z), int(n_kept)


__all__ = [
    "fit_ellipse",
    "cone_aware_bc_refine",
    "cone_aware_bc_refine_with_tilt_prior",
]
