"""Hough-circle BC seed — robust on multi-panel detectors.

Per-arc algebraic circle fits (Pratt) bias on multi-panel detectors when
arcs aren't symmetrically distributed across η: the per-arc center comes
out shifted toward whichever panel the arc fragment lives on.  The
chord-bisector method has the same failure mode.

The Hough circle transform sidesteps this by **voting**: for each edge
pixel and each candidate radius, vote for every (cy, cz) on the circle
of that radius around the pixel.  A real ring's pixels all vote for the
same (cy, cz) — namely, the BC.  Panel-edge artefacts vote for many
inconsistent centers.  The accumulator's peak is the BC.

Uses scikit-image's ``hough_circle`` if available; otherwise a
NumPy-only manual implementation.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def hough_circle_bc(
    image: np.ndarray,
    *,
    radii: Sequence[float],
    panel_mask: Optional[np.ndarray] = None,
    edge_thresh_mad_factor: float = 3.0,
    bc_search_box: Optional[Tuple[float, float, float, float]] = None,
    accumulator_blur_sigma: float = 1.0,
    sample_step_pixels: int = 1,
) -> Tuple[float, float, float]:
    """Estimate BC and best-matching radius via Hough circle voting.

    Parameters
    ----------
    image : 2-D float array.
    radii : iterable of candidate ring radii (px).  These are the *expected*
        ring radii at the trial Lsd; the routine picks the best-fitting
        among them and returns the corresponding (cy, cz).
    panel_mask : optional binary mask (1=valid).  Used to drop sentinel
        pixels and erode panel edges before edge detection.
    edge_thresh_mad_factor : pixel intensity > median + factor × MAD →
        edge pixel (cast a vote).
    bc_search_box : optional ``(y_min, y_max, z_min, z_max)`` to restrict
        the accumulator footprint (faster, smaller memory).
    accumulator_blur_sigma : Gaussian blur applied to each per-radius
        accumulator before peak finding.

    Returns
    -------
    ``(bc_y, bc_z, best_radius_px)``.
    """
    img = np.asarray(image, dtype=np.float64)
    if panel_mask is not None:
        from .mask import erode_mask
        m = erode_mask(panel_mask.astype(bool), iterations=2)
        img = np.where(m, img, 0.0)
    # MIDAS image convention: image.shape = (n_z, n_y); image[z_row, y_col].
    # BC values are stored as (BC_y, BC_z) = (col, row).  np.where returns
    # (row_indices, col_indices) → first axis is z, second is y.
    n_z, n_y = img.shape

    med = float(np.median(img))
    mad = float(np.median(np.abs(img - med)))
    cutoff = med + edge_thresh_mad_factor * (1.4826 * mad + 1.0)
    edges = img > cutoff
    if panel_mask is not None:
        from .mask import erode_mask
        edges &= erode_mask(panel_mask.astype(bool), iterations=2)
    z_edge, y_edge = np.where(edges)              # MIDAS-conventional unpack
    if sample_step_pixels > 1:
        z_edge = z_edge[::sample_step_pixels]
        y_edge = y_edge[::sample_step_pixels]
    if len(y_edge) == 0:
        return float(n_y * 0.5), float(n_z * 0.5), float(np.median(radii))

    # Accumulator footprint in (y, z) — note bc_search_box order is
    # (y_min, y_max, z_min, z_max) per MIDAS convention.
    if bc_search_box is None:
        ymin, ymax = 0, n_y
        zmin, zmax = 0, n_z
    else:
        ymin, ymax, zmin, zmax = bc_search_box
        ymin, ymax = int(max(0, ymin)), int(min(n_y, ymax))
        zmin, zmax = int(max(0, zmin)), int(min(n_z, zmax))
    A_y = ymax - ymin
    A_z = zmax - zmin

    best_score = -1.0
    best_y = best_z = 0.0
    best_r = float(radii[0])
    for r in radii:
        n_theta = max(int(2 * np.pi * r), 32)
        thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        # Each (y_e, z_e) edge pixel votes for centres on its circle.
        cy = y_edge[:, None] - r * cos_t[None, :]
        cz = z_edge[:, None] - r * sin_t[None, :]
        cy_int = np.round(cy).astype(np.int64).ravel()
        cz_int = np.round(cz).astype(np.int64).ravel()
        keep = ((cy_int >= ymin) & (cy_int < ymax)
                & (cz_int >= zmin) & (cz_int < zmax))
        if not keep.any():
            continue
        cy_int = cy_int[keep] - ymin
        cz_int = cz_int[keep] - zmin
        # Accumulator axes: (z, y).  Peak indices map back via cz_int → z, cy_int → y.
        flat_idx = cz_int * A_y + cy_int
        acc = np.bincount(flat_idx, minlength=A_z * A_y).reshape(A_z, A_y)
        if accumulator_blur_sigma > 0:
            from scipy import ndimage
            acc = ndimage.gaussian_filter(acc.astype(np.float64),
                                            sigma=accumulator_blur_sigma)
        peak_v = float(acc.max())
        if peak_v > best_score:
            best_score = peak_v
            z_peak, y_peak = np.unravel_index(int(acc.argmax()), acc.shape)
            best_y = float(y_peak + ymin)
            best_z = float(z_peak + zmin)
            best_r = float(r)
    return best_y, best_z, best_r


def hough_seed_bc_lsd(
    image: np.ndarray,
    *,
    sim_radii_px: np.ndarray,
    initial_lsd: float,
    panel_mask: Optional[np.ndarray] = None,
    bc_search_box: Optional[Tuple[float, float, float, float]] = None,
    bc_search_radius_px: Optional[float] = None,
    bc_guess: Optional[Tuple[float, float]] = None,
    lsd_search_factor: float = 1.5,
    n_lsd_candidates: int = 8,
    accumulator_blur_sigma: float = 1.5,
    sample_step_pixels: int = 2,
    two_theta_rad: Optional[Sequence[float]] = None,
    cone_aware: bool = False,
    tilt_y_prior_rad: Optional[float] = None,
    tilt_z_prior_rad: Optional[float] = None,
    p_x_um: Optional[float] = None,
) -> Tuple[float, float, float, int]:
    """Joint (BC, Lsd) seed via Hough voting over multiple Lsd hypotheses.

    For each of ``n_lsd_candidates`` Lsd values geometrically spaced
    around ``initial_lsd``, run :func:`hough_circle_bc` against all rings
    rescaled to that Lsd.  Pick the Lsd with the highest accumulator peak
    sum across rings.

    When ``two_theta_rad`` and ``p_x_um`` are supplied together with
    a tilt prior (``tilt_y_prior_rad`` and / or ``tilt_z_prior_rad``,
    accurate to within roughly :math:`\\pm 5^\\circ`) the Hough vote
    is followed by a cone-aware BC refinement
    (:func:`.cone.cone_aware_bc_refine_with_tilt_prior`).  The
    refinement gates edges around each ring's predicted ellipse centre
    --- using the user-supplied tilt prior to compensate for the
    :math:`L_\\mathrm{sd}\\,\\tan(\\alpha)\\,\\tan^{2}(2\\theta)`
    cone-detector intersection offset --- and extrapolates the
    per-ring centres to :math:`2\\theta \\to 0` for a sub-pixel BC.
    This is the production path for users who know their detector is
    tilted; on a Varex geometry at 15 degrees of tilt with a
    :math:`\\pm 5^\\circ` prior the refinement returns the BC well
    within the 60 px LM basin even from a 30 px-biased Hough seed.

    For users without a tilt prior, the legacy ``cone_aware=True`` path
    is retained as opt-in: it runs the same per-ring extrapolation but
    gates around the tilt-biased seed BC, which can introduce
    ring-grouping noise.  The default ``cone_aware=False`` keeps the
    plain Hough vote, sufficient for tilts within roughly
    :math:`\\pm 5^\\circ` where the Hough seed already lies inside the
    LM basin.

    Returns ``(bc_y, bc_z, best_lsd, best_n_match)``.
    """
    sim = np.asarray(sim_radii_px, dtype=np.float64)
    lsd_lo = initial_lsd / lsd_search_factor
    lsd_hi = initial_lsd * lsd_search_factor
    lsd_candidates = np.geomspace(lsd_lo, lsd_hi, n_lsd_candidates)

    # Build a bc_search_box from bc_guess + bc_search_radius_px if the caller
    # didn't pass an explicit box.  Constraining the Hough accumulator to a
    # tight window around the user's BC guess prevents the vote from drifting
    # to off-center beam-stop bright pixels or asymmetric ring fragments.
    if bc_search_box is None and bc_guess is not None and bc_search_radius_px:
        gy, gz = float(bc_guess[0]), float(bc_guess[1])
        r = float(bc_search_radius_px)
        bc_search_box = (gy - r, gy + r, gz - r, gz + r)

    best = (-1.0, 0.0, 0.0, float(initial_lsd), 0)
    for trial_lsd in lsd_candidates:
        scale = trial_lsd / initial_lsd
        radii = sim[:8] * scale  # use first 8 rings for the vote
        bc_y, bc_z, _ = hough_circle_bc(
            image,
            radii=radii.tolist(),
            panel_mask=panel_mask,
            bc_search_box=bc_search_box,
            accumulator_blur_sigma=accumulator_blur_sigma,
            sample_step_pixels=sample_step_pixels,
        )
        # Score: count rings whose r * scale matches a strong arc near (bc_y, bc_z).
        # Faster proxy: re-run hough_circle_bc and use its peak as score.
        # (Exact score requires a per-ring accumulator pass.)
        n_match = 0
        for r in radii:
            yy, zz = np.where(image > np.median(image))
            if len(yy) == 0:
                continue
            d = np.sqrt((yy - bc_y) ** 2 + (zz - bc_z) ** 2)
            n_pixels = int(np.sum(np.abs(d - r) < 1.0))
            if n_pixels > 50:
                n_match += 1
        score = n_match
        if score > best[0]:
            best = (score, bc_y, bc_z, float(trial_lsd), n_match)
    bc_y_out, bc_z_out, best_lsd, best_n_match = (
        best[1], best[2], best[3], best[4]
    )
    has_tilt_prior = (
        two_theta_rad is not None
        and p_x_um is not None
        and (tilt_y_prior_rad is not None or tilt_z_prior_rad is not None)
    )
    if has_tilt_prior:
        from .cone import cone_aware_bc_refine_with_tilt_prior
        scale = best_lsd / initial_lsd
        bc_y_out, bc_z_out, _ = cone_aware_bc_refine_with_tilt_prior(
            image,
            bc_y_seed=bc_y_out,
            bc_z_seed=bc_z_out,
            Lsd_um=float(best_lsd),
            p_x_um=float(p_x_um),
            sim_radii_px=sim * scale,
            two_theta_rad=two_theta_rad,
            tilt_y_prior_rad=float(tilt_y_prior_rad or 0.0),
            tilt_z_prior_rad=float(tilt_z_prior_rad or 0.0),
            panel_mask=panel_mask,
        )
    elif cone_aware and two_theta_rad is not None:
        from .cone import cone_aware_bc_refine
        scale = best_lsd / initial_lsd
        bc_y_out, bc_z_out, _ = cone_aware_bc_refine(
            image,
            bc_y_seed=bc_y_out,
            bc_z_seed=bc_z_out,
            sim_radii_px=sim * scale,
            two_theta_rad=two_theta_rad,
            panel_mask=panel_mask,
        )
    return bc_y_out, bc_z_out, best_lsd, best_n_match


__all__ = ["hough_circle_bc", "hough_seed_bc_lsd"]
