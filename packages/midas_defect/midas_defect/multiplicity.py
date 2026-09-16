"""Is one `find_blobs_3d` connected component one real object, or more?

`find_blobs_3d`'s watershed splitter (`ingest._split_blob`) decides multiplicity by scale-free
intensity prominence alone -- no statistical or physical model of "how many real objects is
this." On a genuinely complex sample that over-fragments elongated/streak-like features into
arbitrary pieces (measured: a real connected component watershed cut into 6 sub-peaks along what
turned out to be a real second Bragg-like feature plus ordinary peak non-Gaussianity, not 6
separate objects) -- and forcing a "how many Gaussians" question onto a genuinely elongated
rod/diffuse feature is the wrong model class entirely, not just a mis-tuned one.

This module does two things, deliberately not one thing that tries to do both:

1. A cheap, deterministic shape check (`classify_component_shape`) that routes a component to
   the model that can actually explain it: a compact, Gaussian-shaped candidate for the K=1-vs-K=2
   resolver below, or an elongated one that should go to the package's EXISTING rod/diffuse tools
   (`rod_profile.py`, `bragg_diffuse.py`) instead of being force-fit as Gaussians.
2. A calibrated K=1-vs-K=2 multi-Gaussian resolver (`resolve_multiplicity`), hard-capped at K=2 --
   testing K=3 on a real, validated K=2 case re-explained the DOMINANT peak's own non-Gaussianity
   rather than finding new structure, so this module does not offer it. The significance threshold
   is always a caller-supplied, per-dataset-measured `NullCalibration` (`calibrate_null`), never a
   baked-in default: raw BIC improvement correlates strongly with blob size (bigger/more complex
   blobs "improve" more from a second Gaussian with no second object involved), so the same
   `sigma_rtn` discipline applies here -- ``manuals/solve-cell/phase-3-refine.md``: "YOUR sample's
   measured residual budget... the package default is one DAC sample's, not universal."

What this module does NOT do: automatic end-to-end rod/diffuse profiling. `rod_path_geometry`/
`profile_along`/`classify_voxels` need an indexed orientation, which does not exist at raw-ingest
time -- `spots_to_qsample` is the missing coordinate bridge (`raster._ingest_position`'s own
private pixel_to_qlab -> qlab_to_qsample pattern, made reusable), not a new rod fitter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.optimize import least_squares

__all__ = [
    "ShapeReference", "empirical_shape_reference", "classify_component_shape",
    "GaussianComponent", "MultiplicityFit", "fit_k_gaussians",
    "NullCalibration", "calibrate_null", "resolve_multiplicity",
    "CentroidEnvelope", "bootstrap_centroid_envelope",
    "spots_to_qsample",
]

_SHAPE_COLS = ("skew_length", "skew_width", "kurt_length", "kurt_width")


def _shape_badness(row_or_df):
    """Sum of |skew|+|kurt| across the in-plane axes -- 0 for a symmetric, mesokurtic
    (Gaussian-like) spot; large for an asymmetric or bimodal one. Shared formula between
    `empirical_shape_reference` and `classify_component_shape` so "clean" means the same
    thing in both places.
    """
    return sum(abs(row_or_df[c]) for c in _SHAPE_COLS)


# ---------------------------------------------------------------------------
# 1. shape classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapeReference:
    """What a real, single, clean spot looks like IN THIS DATASET -- measured, not assumed."""
    max_extent_px: float
    max_aspect: float
    max_omega_width_frames: float
    n_reference_blobs: int


def empirical_shape_reference(spots: pd.DataFrame, *, badness_max: float = 1.0,
                              margin: float = 4.0) -> ShapeReference:
    """Measure "what does a clean single spot look like" from THIS dataset's own currently-
    unsplit, low-shape-badness blobs -- the reference both `classify_component_shape`'s
    rod/diffuse cut and `resolve_multiplicity`'s extent bound use.

    `spots` is a `find_blobs_3d`-shaped DataFrame (must carry `blob_id`, `length_px`, `aspect`,
    `omega_width_frames`, and `_SHAPE_COLS`). Raises `ValueError` if no blob clears the
    `badness_max` bar -- silently falling back to an arbitrary number would defeat the entire
    point of measuring this from data.
    """
    counts = spots["blob_id"].value_counts()
    unsplit = spots[spots["blob_id"].isin(counts[counts == 1].index)]
    badness = _shape_badness(unsplit)
    clean = unsplit[badness < badness_max]
    if len(clean) == 0:
        raise ValueError(
            f"no unsplit blob has shape badness < {badness_max} -- cannot measure a reference "
            f"without at least one clean spot. Loosen badness_max or inspect this dataset by hand."
        )
    return ShapeReference(
        max_extent_px=float(margin * clean["length_px"].max()),
        max_aspect=float(margin * clean["aspect"].replace(np.inf, np.nan).max()),
        max_omega_width_frames=float(margin * clean["omega_width_frames"].max()),
        n_reference_blobs=len(clean),
    )


def classify_component_shape(row: pd.Series, ref: ShapeReference,
                             badness_max: float = 1.0
                             ) -> Literal["clean", "compact", "rod_like"]:
    """Route one `find_blobs_3d` row to the model that can explain it.

    "rod_like": elongated well beyond this dataset's own clean single spots, in-plane
    (`aspect`) or in omega (`omega_width_frames`) -- decline to Gaussian-fit; hand off to
    `rod_profile.py`/`bragg_diffuse.py` (needs an orientation this function does not have).
    "clean": not rod-like, low shape badness, single sub_id under its blob_id -- nothing to do.
    "compact": otherwise -- a candidate for `resolve_multiplicity`.
    """
    aspect = row["aspect"]
    if (np.isfinite(aspect) and aspect > ref.max_aspect) or \
       row["omega_width_frames"] > ref.max_omega_width_frames:
        return "rod_like"
    if _shape_badness(row) < badness_max and row.get("sub_id", 1) == 1:
        return "clean"
    return "compact"


# ---------------------------------------------------------------------------
# 2. calibrated K=1-vs-K=2 multi-Gaussian resolution
# ---------------------------------------------------------------------------

@dataclass
class GaussianComponent:
    amplitude: float
    mean: np.ndarray            # (3,) in (frame, row, col)
    cov: np.ndarray             # (3, 3)


@dataclass
class MultiplicityFit:
    k: int
    baseline: float
    components: List[GaussianComponent]
    rss: float                  # in sqrt-intensity space -- see fit_k_gaussians
    bic: float
    n_voxels: int


def _unpack(theta: np.ndarray, k: int):
    baseline = theta[0]
    comps = []
    off = 1
    for _ in range(k):
        amp = theta[off]
        mu = theta[off + 1:off + 4]
        l = theta[off + 4:off + 10]
        L = np.array([[l[0], 0, 0], [l[1], l[2], 0], [l[3], l[4], l[5]]])
        comps.append((amp, mu, L))
        off += 10
    return baseline, comps


def _predict(theta: np.ndarray, k: int, coords: np.ndarray) -> np.ndarray:
    baseline, comps = _unpack(theta, k)
    pred = np.full(coords.shape[0], baseline)
    for amp, mu, L in comps:
        d = coords - mu
        u = d @ L.T
        pred = pred + amp * np.exp(-0.5 * np.sum(u * u, axis=1))
    return pred


def _residuals(theta: np.ndarray, k: int, coords: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """sqrt-space residuals: real detector counts here span ~0 to ~1e6 in one connected
    component, so RAW least-squares residuals are dominated ~1e10:1 by the single brightest
    voxel and a second component collapses to a no-op. Counts are approximately Poisson, for
    which sqrt(count) has approximately CONSTANT variance -- the standard variance-stabilizing
    fix, not an arbitrary reweighting. (Found by direct failure during prototyping: an
    unconstrained fit in raw-count space converged to a negative-amplitude "component"
    contributing ~0 everywhere.)
    """
    pred = np.clip(_predict(theta, k, coords), 0, None)
    return np.sqrt(pred) - np.sqrt(np.clip(intensity, 0, None))


def _weighted_moments_init(coords: np.ndarray, intensity: np.ndarray):
    w = np.clip(intensity, 0, None)
    W = w.sum()
    mu = (coords * w[:, None]).sum(0) / W
    d = coords - mu
    cov = (d[:, :, None] * d[:, None, :] * w[:, None, None]).sum(0) / W + np.eye(3) * 1e-3
    return mu, cov


def _cov_to_Lchol(cov: np.ndarray) -> np.ndarray:
    prec = np.linalg.inv(cov)
    Lp = np.linalg.cholesky(prec)
    return np.array([Lp[0, 0], Lp[1, 0], Lp[1, 1], Lp[2, 0], Lp[2, 1], Lp[2, 2]])


def _residual_seeded_second_component(coords, intensity, theta1):
    """Seed component 2 from where the K=1 fit's OWN residual is largest, >=1.5-sigma
    outside component 1's own core -- greedy/matching-pursuit, not a blind restart.

    Generic k-means restarts on the raw intensity-weighted cloud were found, during
    prototyping, to re-split the DOMINANT peak's own (mildly non-Gaussian) core into two
    nearby components instead of finding a real, much fainter second feature -- the easiest
    way to reduce residual when the initializer has no reason to look for anything specific.
    Looking at the K=1 residual directly targets "what does this model fail to explain."
    """
    _, comps1 = _unpack(theta1, 1)
    amp1, mu1, L1 = comps1[0]
    pred1 = _predict(theta1, 1, coords)
    resid = np.clip(intensity - pred1, 0, None)
    cov1 = np.linalg.inv(L1.T @ L1)
    d = coords - mu1
    maha1 = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov1), d), 0))
    away = maha1 > 1.5
    if not away.any() or resid[away].max() <= 0:
        return None
    seed_idx = np.flatnonzero(away)[np.argmax(resid[away])]
    mu2 = coords[seed_idx]
    near = np.linalg.norm(coords - mu2, axis=1) < 3.0
    w = resid[near]
    if w.sum() <= 0 or near.sum() < 4:
        cov2 = np.eye(3)
    else:
        dd = coords[near] - mu2
        cov2 = (dd[:, :, None] * dd[:, None, :] * w[:, None, None]).sum(0) / w.sum() + np.eye(3) * 1e-2
    amp2 = float(resid[seed_idx])
    return (float(amp1), mu1, cov1), (amp2, mu2, cov2)


def fit_k_gaussians(coords: np.ndarray, intensity: np.ndarray, k: int, *,
                    max_extent_px: float, n_restarts: int = 3, seed: int = 0) -> MultiplicityFit:
    """Fit K anisotropic 3-D Gaussians + a constant baseline to raw voxel data.

    `coords` is `(N, 3)` in `(frame, row, col)`; `intensity` is `(N,)`. `max_extent_px` bounds
    every component's standard deviation along any axis (via the Cholesky-of-precision diagonal)
    -- an UNCONSTRAINED fit was found, during prototyping, to reach a degenerate ~2e6-pixel-wide
    "component" that soaks up residual as a flat ridge rather than representing anything real.
    Get `max_extent_px` from `ShapeReference.max_extent_px` (`empirical_shape_reference`), not a
    guessed number. Amplitude is bounded >= 0 (independent domains' intensities ADD; nothing
    here can subtract -- an unbounded fit was found to use a negative-amplitude component to
    sharpen a single peak's non-Gaussian core instead of finding a real second object).

    Only K=1 and K=2 are supported. `resolve_multiplicity` never asks for more; K=3 was found,
    during prototyping, to re-explain the dominant component's own non-Gaussianity rather than
    find new structure, even on a case where K=2 was independently validated as correct.
    """
    if k not in (1, 2):
        raise ValueError(f"fit_k_gaussians only supports k in (1, 2), got {k}")
    rng = np.random.default_rng(seed)
    n = len(intensity)
    baseline0 = float(np.percentile(intensity, 10))
    lo, hi = coords.min(0) - 3, coords.max(0) + 3
    l_min = 1.0 / max_extent_px
    lb = [-np.inf] + [0.0, lo[0], lo[1], lo[2], l_min, -np.inf, l_min, -np.inf, -np.inf, l_min] * k
    ub = [np.inf] + [np.inf, hi[0], hi[1], hi[2], np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] * k

    n_trials = n_restarts + (1 if k == 2 else 0)
    best = None
    for trial in range(n_trials):
        if k == 1:
            mu0, cov0 = _weighted_moments_init(coords, intensity)
            comps0 = [(float(intensity.max() - baseline0), mu0, cov0)]
        elif trial == n_trials - 1:
            fit1 = fit_k_gaussians(coords, intensity, 1, max_extent_px=max_extent_px,
                                   n_restarts=n_restarts, seed=seed)
            theta1 = np.concatenate([[fit1.baseline],
                                     [fit1.components[0].amplitude], fit1.components[0].mean,
                                     _cov_to_Lchol(fit1.components[0].cov)])
            seeded = _residual_seeded_second_component(coords, intensity, theta1)
            if seeded is None:
                continue
            comps0 = list(seeded)
        else:
            w = np.clip(intensity - np.percentile(intensity, 50), 0, None)
            if w.sum() <= 0:
                w = np.clip(intensity, 1e-6, None)
            reps = rng.poisson(w / w.sum() * 500 + 0.01)
            pts = np.repeat(coords, reps, axis=0)
            if len(pts) < k * 5:
                pts = coords
            _, labels_km = kmeans2(pts.astype(float), k, seed=seed + trial, minit="++")
            comps0 = []
            for c in range(k):
                sel = labels_km == c
                pts_c = pts[sel] if sel.sum() >= 4 else coords
                mu_c = pts_c.mean(0)
                cov_c = np.cov(pts_c.T) + np.eye(3) * 1e-3 if pts_c.shape[0] > 4 else np.eye(3)
                comps0.append((float(intensity.max() - baseline0) / k, mu_c, cov_c))

        theta0 = [baseline0]
        for amp, mu, cov in comps0:
            theta0 += [amp, *mu, *_cov_to_Lchol(cov)]
        theta0 = np.clip(np.array(theta0, float), lb, ub)

        res = least_squares(_residuals, theta0, args=(k, coords, intensity),
                            method="trf", bounds=(lb, ub), max_nfev=5000)
        rss = float(np.sum(res.fun ** 2))
        if best is None or rss < best[1]:
            best = (res.x, rss)

    baseline, comps = _unpack(best[0], k)
    components = [GaussianComponent(amplitude=float(amp), mean=np.asarray(mu),
                                    cov=np.linalg.inv(L.T @ L)) for amp, mu, L in comps]
    n_params = 1 + k * 10
    sigma2 = max(best[1] / n, 1e-6)
    bic = n * np.log(sigma2) + n_params * np.log(n)
    return MultiplicityFit(k=k, baseline=float(baseline), components=components,
                           rss=best[1], bic=bic, n_voxels=n)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

@dataclass
class NullCalibration:
    """delta_BIC(K1->K2) ~ slope*log10(n_voxels) + intercept, fit on blobs the CURRENT
    segmentation already treats as single -- how much does BIC "improve" from an extra
    Gaussian with NO second object involved, purely from ordinary non-Gaussianity and size.
    Measured per dataset, exactly like `sigma_rtn`: never assume this transfers.
    """
    slope: float
    intercept: float
    residual_std: float
    n_calibration_blobs: int

    def z_score(self, delta_bic: float, n_voxels: int) -> float:
        predicted = self.slope * np.log10(n_voxels) + self.intercept
        if self.residual_std <= 0:
            return -np.inf if delta_bic < predicted else np.inf
        return (delta_bic - predicted) / self.residual_std


def calibrate_null(reference_coords_intensities, *, max_extent_px: float) -> NullCalibration:
    """Measure the delta_BIC(K1->K2)-vs-size null trend from a set of reference blobs the
    current segmentation already treats as single (e.g. `find_blobs_3d` blobs whose `blob_id`
    appears exactly once). `reference_coords_intensities` is an iterable of `(coords, intensity)`
    pairs, each already in the `(N,3) frame/row/col` + `(N,)` format `fit_k_gaussians` expects
    (build these from `find_blobs_3d`'s `labels` output, one pair per unsplit blob_id).

    Raises `ValueError` with fewer than 8 reference blobs -- a size trend fit on fewer points
    is not a calibration, it is a guess dressed up as one.
    """
    pairs = list(reference_coords_intensities)
    if len(pairs) < 8:
        raise ValueError(
            f"calibrate_null needs >= 8 reference blobs to fit a size trend, got {len(pairs)}"
        )
    log_n, delta_bic = [], []
    for coords, intensity in pairs:
        n = len(intensity)
        if n < 3 * 21:
            continue  # too small to identifiably fit K=2 at all -- not usable as a null point
        fit1 = fit_k_gaussians(coords, intensity, 1, max_extent_px=max_extent_px)
        fit2 = fit_k_gaussians(coords, intensity, 2, max_extent_px=max_extent_px)
        log_n.append(np.log10(n))
        delta_bic.append(fit2.bic - fit1.bic)
    if len(log_n) < 8:
        raise ValueError(
            f"only {len(log_n)} reference blobs had >= 63 voxels (needed to fit K=2 at all) -- "
            f"need >= 8 to calibrate a null."
        )
    log_n, delta_bic = np.array(log_n), np.array(delta_bic)
    slope, intercept = np.polyfit(log_n, delta_bic, 1)
    residual = delta_bic - (slope * log_n + intercept)
    return NullCalibration(slope=float(slope), intercept=float(intercept),
                           residual_std=float(residual.std()), n_calibration_blobs=len(log_n))


def resolve_multiplicity(coords: np.ndarray, intensity: np.ndarray, *, max_extent_px: float,
                         null: NullCalibration, min_z: float = 2.0) -> MultiplicityFit:
    """Fit K=1; accept K=2 only if its delta_BIC clears `null`'s size-corrected trend by
    `min_z` residual standard deviations. Never tries K=3 (see module docstring).

    `min_z=2.0` is deliberately stricter than the ~1-sigma the one real, independently-
    corroborated case validated during prototyping cleared -- conservative until this is
    calibrated on more than one position. Do not lower it without new calibration evidence.

    **Only call this on components `classify_component_shape` has already called "compact".**
    Found directly during testing: on a genuinely single but heavy-tailed (Laplace/exponential
    -decay, i.e. rod-like) feature, this function explains it as two co-located Gaussians of
    different width -- a legitimate scale-mixture approximation of a non-Gaussian peak, not a
    bug, and no calibration threshold fixes it, because the model class itself (K discrete
    Gaussians) is wrong for that shape. The classifier exists to keep rod-like components away
    from this function entirely, not to be a redundant second check after the fact.
    """
    fit1 = fit_k_gaussians(coords, intensity, 1, max_extent_px=max_extent_px)
    if len(intensity) < 3 * 21:
        return fit1   # not enough voxels to identifiably test K=2 at all
    fit2 = fit_k_gaussians(coords, intensity, 2, max_extent_px=max_extent_px)
    z = null.z_score(fit2.bic - fit1.bic, len(intensity))
    return fit2 if z <= -min_z else fit1


# ---------------------------------------------------------------------------
# 3. per-peak centroid + envelope
# ---------------------------------------------------------------------------

@dataclass
class CentroidEnvelope:
    """A fitted centroid's own positional uncertainty -- the "report the spread, not just the
    point estimate" discipline `05_raster_lattice_single_point.ipynb`'s Steps 5/6 already apply
    to orientation and cell, extended down to the individual peak. Segmentation is not fully
    solved (see `resolve_multiplicity`'s own docstring); this does not fix that, it makes
    whatever centroid IS reported carry an honest uncertainty instead of a bare point value.
    """
    k: int
    centroids: np.ndarray          # (k, 3) mean of the bootstrap centroids per component
    bootstrap_means: np.ndarray    # (n_kept, k, 3)
    spread_std: np.ndarray         # (k, 3) per-axis std across bootstrap draws
    percentile_2_5: np.ndarray     # (k, 3)
    percentile_97_5: np.ndarray    # (k, 3)
    n_boot: int
    n_kept: int                    # draws that converged; a failed refit is skipped, not zero-filled
    n_voxels: int
    low_n_warning: bool            # True when a with-replacement bootstrap's variance estimate
                                    # is unreliable at this voxel count (n_voxels < 30) -- an
                                    # honestly-labeled envelope is still reported, not refused,
                                    # matching bootstrap_orientation_uncertainty's own philosophy


def bootstrap_centroid_envelope(coords: np.ndarray, intensity: np.ndarray, *, k: int,
                                max_extent_px: float, n_boot: int = 200,
                                seed: int = 0) -> CentroidEnvelope:
    """Bootstrap a FIXED-k fit's own centroid uncertainty by resampling voxels with replacement
    (`refine_cell_joint`'s own bootstrap pattern -- resample size N from N voxels with
    replacement, not `bootstrap_orientation_uncertainty`'s without-replacement subsampling).

    Deliberately does NOT re-run `resolve_multiplicity`'s own K-selection per draw -- call
    `resolve_multiplicity` once to decide `k`, then this function to get THAT decision's own
    positional uncertainty. Bootstrapping the model-selection itself (does k flip between 1 and
    2 across draws) is a different, harder question this function does not answer; conflating
    the two would make a segmentation-ambiguity problem look like an ordinary fit-noise problem,
    the same distinction Steps 5/6 draw between bootstrapping a domain's orientation/cell (fixed
    domain, resampled reflections) and deciding whether that domain exists at all in the first
    place (a different, one-time question, answered before any bootstrap starts).
    """
    rng = np.random.default_rng(seed)
    n = len(intensity)

    # Point estimate first, purely to fix a consistent LABELING of the k components -- a
    # fresh fit_k_gaussians call on each resample has no reason to return "component 0" as
    # the same physical peak from draw to draw (label switching, the standard failure mode
    # of bootstrapping a mixture model). Found directly while testing this function: naive
    # per-draw averaging of unaligned components landed BETWEEN two real, well-separated
    # peaks -- neither centroid the bootstrap draws' own point estimate agreed on.
    point_fit = fit_k_gaussians(coords, intensity, k, max_extent_px=max_extent_px)
    point_means = np.array([c.mean for c in point_fit.components])   # (k, 3)

    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            fit = fit_k_gaussians(coords[idx], intensity[idx], k, max_extent_px=max_extent_px)
        except Exception:
            continue   # a degenerate resample failing to fit is skipped, not zero-filled
        draw_means = np.array([c.mean for c in fit.components])
        if k == 1:
            draws.append(draw_means)
            continue
        # align this draw's components to the point estimate's own labeling by nearest
        # centroid (k is always 1 or 2 here, so a brute-force assignment is exact and cheap)
        d00 = np.linalg.norm(draw_means[0] - point_means[0])
        d01 = np.linalg.norm(draw_means[0] - point_means[1])
        aligned = draw_means if d00 <= d01 else draw_means[::-1]
        draws.append(aligned)
    n_kept = len(draws)
    if n_kept == 0:
        raise RuntimeError(f"all {n_boot} bootstrap resamples failed to fit at k={k} -- "
                           f"the point estimate itself is not trustworthy enough to bootstrap")
    bootstrap_means = np.array(draws)   # (n_kept, k, 3)
    return CentroidEnvelope(
        k=k,
        centroids=bootstrap_means.mean(axis=0),
        bootstrap_means=bootstrap_means,
        spread_std=bootstrap_means.std(axis=0, ddof=1) if n_kept > 1 else np.zeros((k, 3)),
        percentile_2_5=np.percentile(bootstrap_means, 2.5, axis=0),
        percentile_97_5=np.percentile(bootstrap_means, 97.5, axis=0),
        n_boot=n_boot,
        n_kept=n_kept,
        n_voxels=n,
        low_n_warning=n < 30,
    )


# ---------------------------------------------------------------------------
# 4. rod/diffuse routing: the missing coordinate bridge
# ---------------------------------------------------------------------------

def spots_to_qsample(spots: pd.DataFrame, geom, omega_sign: int = 1) -> np.ndarray:
    """Detector-space `find_blobs_3d` rows -> sample-frame q (N,3), the format
    `bragg_diffuse.classify_voxels`/`rod_profile.rod_path_geometry` expect.

    Exactly `raster._ingest_position`'s own private two-step pattern (pixel_to_qlab then
    qlab_to_qsample), made public and reusable instead of re-derived by every caller. Only
    meaningful once an orientation/domain exists to classify against -- this function does not
    decide Bragg vs diffuse itself, it only gets the coordinates into the right frame for
    `bragg_diffuse.predicted_reflection_points`/`classify_voxels` or
    `rod_profile.rod_path_geometry`/`profile_along` to consume.
    """
    import torch
    from .geometry import pixel_to_qlab, qlab_to_qsample

    omega_deg = geom.omega_first_deg + geom.omega_step_deg * spots["frame"].values
    qlab = pixel_to_qlab(spots["row"].values.astype(float), spots["col"].values.astype(float),
                         geom, device="cpu")
    omega_rad = torch.deg2rad(torch.as_tensor(omega_sign * omega_deg, dtype=qlab.dtype))
    return qlab_to_qsample(qlab, omega_rad).detach().cpu().numpy()
