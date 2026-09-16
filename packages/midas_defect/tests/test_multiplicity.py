"""Is one find_blobs_3d connected component one real object, or more?

Built on synthetic data with known ground truth. The properties pinned here are the ones
found, during prototyping on real S5 data, to matter:

- the shape router sends elongated components away from the Gaussian-mixture resolver
- a single (if non-Gaussian) peak must not be spuriously split
- a genuine, well-separated double must be found
- a naive size-blind BIC comparison over-splits big/complex single features -- the
  size-corrected null must catch this, not just raw BIC-min
- the calibration must generalize to a held-out blob, not just memorize its own sample
- a fainter peak's bootstrap centroid envelope must be wider than a brighter one's
- a genuine double's two components each get their own, separately-computed envelope
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import pytest

from midas_defect.multiplicity import (
    ShapeReference, empirical_shape_reference, classify_component_shape,
    fit_k_gaussians, calibrate_null, resolve_multiplicity, spots_to_qsample,
    bootstrap_centroid_envelope,
)

RNG = np.random.default_rng(20260914)


def _make_blob(centers, covs, amps, *, box=24, baseline=50.0, keep_thresh=150.0, seed=0):
    """A synthetic 3-D (frame, row, col) intensity cloud: sum of anisotropic Gaussians plus
    Poisson-ish noise, thresholded like a real background-subtracted detector region.
    """
    rng = np.random.default_rng(seed)
    ff, rr, cc = np.mgrid[0:box, 0:box, 0:box].astype(float)
    coords_all = np.stack([ff.ravel(), rr.ravel(), cc.ravel()], axis=1)
    truth = np.full(coords_all.shape[0], baseline)
    for c, cov, a in zip(centers, covs, amps):
        d = coords_all - np.asarray(c)
        inv = np.linalg.inv(np.asarray(cov))
        m2 = np.einsum("ni,ij,nj->n", d, inv, d)
        truth = truth + a * np.exp(-0.5 * m2)
    noisy = rng.normal(truth, np.sqrt(np.clip(truth, 1, None)))
    keep = noisy > keep_thresh
    return coords_all[keep], noisy[keep]


def _elongated_single_feature(box, decay_len, sigma_perp, amp, *, seed):
    """A genuinely SINGLE, continuously elongated feature -- Laplace (exponential) decay
    along the row axis, Gaussian in frame/col -- not a sum of discrete Gaussian bumps (that
    would just BE two objects, and correctly getting split would not test anything). This
    is deliberately non-Gaussian along its long axis, the way a real streak/rod is, while
    remaining one connected, unimodal feature.
    """
    rng = np.random.default_rng(seed)
    ff, rr, cc = np.mgrid[0:box, 0:box, 0:box].astype(float)
    c0 = box / 2
    d_row = np.abs(rr - c0)
    d_frame = ff - c0
    d_col = cc - c0
    truth = 50.0 + amp * np.exp(-d_row / decay_len) * \
        np.exp(-0.5 * (d_frame / sigma_perp) ** 2) * np.exp(-0.5 * (d_col / sigma_perp) ** 2)
    noisy = rng.normal(truth, np.sqrt(np.clip(truth, 1, None))).ravel()
    keep = noisy > 150.0
    coords = np.stack([ff.ravel(), rr.ravel(), cc.ravel()], axis=1)
    return coords[keep], noisy[keep]


def _single_gaussian(box, sigma, amp, *, seed, wobble_frac=0.02):
    """A "clean" single spot with a small, fixed amount of realistic non-Gaussianity (a
    faint, very close secondary lobe) -- pure synthetic Gaussians are exactly the model
    being fit and would never show the size-dependent BIC "improvement" real detector data
    does; this makes the calibration test meaningful rather than trivial.
    """
    c0 = np.array([box / 2, box / 2, box / 2])
    cov = np.diag([sigma * 0.7, sigma, sigma * 1.1]) ** 2
    wobble = c0 + np.array([0.3, 0.4, -0.3])
    return _make_blob([c0, wobble], [cov, cov * 0.5], [amp, amp * wobble_frac],
                      box=box, seed=seed)


# --------------------------------------------------------------- 1. shape classification

def _row(*, aspect, omega_width_frames, skew_length=0.0, skew_width=0.0,
        kurt_length=0.0, kurt_width=0.0, sub_id=1):
    return pd.Series(dict(aspect=aspect, omega_width_frames=omega_width_frames,
                          skew_length=skew_length, skew_width=skew_width,
                          kurt_length=kurt_length, kurt_width=kurt_width, sub_id=sub_id))


def test_classify_routes_elongated_components_away_from_the_gaussian_resolver():
    ref = ShapeReference(max_extent_px=10.0, max_aspect=4.0, max_omega_width_frames=3.0,
                         n_reference_blobs=20)
    assert classify_component_shape(_row(aspect=1.2, omega_width_frames=0.8), ref) == "clean"
    assert classify_component_shape(_row(aspect=8.0, omega_width_frames=0.8), ref) == "rod_like"
    assert classify_component_shape(_row(aspect=1.2, omega_width_frames=5.0), ref) == "rod_like"
    assert classify_component_shape(
        _row(aspect=1.5, omega_width_frames=1.0, skew_length=0.9, kurt_length=-0.9),
        ref) == "compact"


def test_empirical_shape_reference_measures_from_clean_unsplit_blobs_only():
    spots = pd.DataFrame([
        # blob_id=1: clean, unsplit -> counts toward the reference
        dict(blob_id=1, sub_id=1, length_px=2.0, aspect=1.3, omega_width_frames=1.0,
            skew_length=0.05, skew_width=0.02, kurt_length=-0.1, kurt_width=0.05),
        # blob_id=2: split into 2 sub-peaks -> excluded regardless of shape
        dict(blob_id=2, sub_id=1, length_px=9.0, aspect=6.0, omega_width_frames=4.0,
            skew_length=0.0, skew_width=0.0, kurt_length=0.0, kurt_width=0.0),
        dict(blob_id=2, sub_id=2, length_px=9.0, aspect=6.0, omega_width_frames=4.0,
            skew_length=0.0, skew_width=0.0, kurt_length=0.0, kurt_width=0.0),
        # blob_id=3: unsplit but high shape badness -> excluded
        dict(blob_id=3, sub_id=1, length_px=5.0, aspect=2.0, omega_width_frames=1.5,
            skew_length=1.5, skew_width=1.5, kurt_length=1.5, kurt_width=1.5),
    ])
    ref = empirical_shape_reference(spots, badness_max=1.0, margin=4.0)
    assert ref.n_reference_blobs == 1
    assert ref.max_extent_px == pytest.approx(4.0 * 2.0)
    assert ref.max_aspect == pytest.approx(4.0 * 1.3)


def test_empirical_shape_reference_refuses_to_guess_with_no_clean_blob():
    spots = pd.DataFrame([
        dict(blob_id=1, sub_id=1, length_px=5.0, aspect=2.0, omega_width_frames=1.5,
            skew_length=2.0, skew_width=2.0, kurt_length=2.0, kurt_width=2.0),
    ])
    with pytest.raises(ValueError):
        empirical_shape_reference(spots, badness_max=1.0)


# ---------------------------------------------------------- 2. single vs double resolution

def test_fit_k_gaussians_recovers_a_known_single_gaussian():
    coords, intensity = _single_gaussian(box=20, sigma=2.0, amp=6000.0, seed=1, wobble_frac=0.0)
    fit = fit_k_gaussians(coords, intensity, 1, max_extent_px=30.0)
    assert fit.k == 1
    truth = np.array([10.0, 10.0, 10.0])
    assert np.allclose(fit.components[0].mean, truth, atol=0.5)
    assert fit.components[0].amplitude > 1000


def test_resolve_multiplicity_does_not_split_a_single_peak():
    calib_pairs = [_single_gaussian(box=20, sigma=s, amp=a, seed=100 + i)
                  for i, (s, a) in enumerate(
                      [(1.5, 3000), (1.8, 4000), (2.0, 5000), (2.2, 6000),
                       (2.5, 8000), (1.6, 3500), (2.1, 5500), (1.9, 4500)])]
    null = calibrate_null(calib_pairs, max_extent_px=30.0)
    assert np.isfinite(null.slope) and np.isfinite(null.intercept)

    coords, intensity = _single_gaussian(box=20, sigma=2.0, amp=5000.0, seed=999)
    fit = resolve_multiplicity(coords, intensity, max_extent_px=30.0, null=null, min_z=2.0)
    assert fit.k == 1


def test_resolve_multiplicity_finds_a_genuine_well_separated_double():
    box = 24
    c1 = np.array([box / 2 - 4, box / 2, box / 2])
    c2 = np.array([box / 2 + 4, box / 2, box / 2])
    cov = np.diag([1.5, 1.8, 1.8]) ** 2
    coords, intensity = _make_blob([c1, c2], [cov, cov], [6000.0, 5000.0], box=box, seed=7)

    # a permissive null: this test is about detection power, not the size-correction itself
    null_permissive = calibrate_null(
        [_single_gaussian(box=box, sigma=s, amp=a, seed=200 + i)
         for i, (s, a) in enumerate([(1.5, 3000), (1.8, 4000), (2.0, 5000), (2.2, 6000),
                                     (1.6, 3500), (2.1, 5500), (1.9, 4500), (1.7, 4200)])],
        max_extent_px=30.0)
    fit = resolve_multiplicity(coords, intensity, max_extent_px=30.0,
                               null=null_permissive, min_z=1.0)
    assert fit.k == 2
    means = sorted([c.mean for c in fit.components], key=lambda m: m[0])
    assert np.allclose(means[0], c1, atol=1.0)
    assert np.allclose(means[1], c2, atol=1.0)


def test_classifier_routes_the_elongated_single_feature_away_before_it_reaches_the_resolver():
    """The blob_id=1050 lesson, correctly scoped: `resolve_multiplicity` on its own was found,
    during this test's development, to explain a genuinely single but heavy-tailed (Laplace/
    exponential-decay) streak as TWO co-located Gaussians of different width -- a legitimate
    scale-mixture approximation of a non-Gaussian peak, not a bug in the fit. That is exactly
    why the architecture puts a shape classifier IN FRONT of the Gaussian-mixture resolver:
    `resolve_multiplicity` was never meant to be robust to arbitrary non-Gaussian shapes on its
    own, `classify_component_shape` is supposed to route them away first. This test checks that
    actual guarantee, using `find_blobs_3d`'s own moment computation (not a hand-rolled
    approximation) so it exercises the real integration point.
    """
    from midas_defect.ingest import find_blobs_3d

    box = 40
    coords, intensity = _elongated_single_feature(box, decay_len=8.0, sigma_perp=1.3,
                                                  amp=8000.0, seed=42)
    stack = np.full((box, box, box), 30.0, dtype=np.float32)
    for (f, r, c), v in zip(coords.astype(int), intensity):
        stack[f, r, c] = v
    mask = np.zeros((box, box), bool)
    streak_df = find_blobs_3d(stack, mask, threshold=150.0, min_vol=10, split_ratio=1e9,
                              gap_bridge=0)
    assert len(streak_df) == 1
    streak_row = streak_df.iloc[0]

    # a handful of ordinary compact spots, for the reference population
    compact_stack = np.full((box, box, box), 30.0, dtype=np.float32)
    rng = np.random.default_rng(43)
    for i, (fc, rc, cc) in enumerate([(10, 10, 10), (10, 25, 10), (30, 10, 25), (30, 25, 30)]):
        ff, rr, cc_ = np.mgrid[0:box, 0:box, 0:box]
        d2 = (ff - fc) ** 2 / 2.0 ** 2 + (rr - rc) ** 2 / 1.6 ** 2 + (cc_ - cc) ** 2 / 1.8 ** 2
        compact_stack += (4000.0 * np.exp(-0.5 * d2)).astype(np.float32)
    compact_df = find_blobs_3d(compact_stack, mask, threshold=150.0, min_vol=10,
                               split_ratio=1e9, gap_bridge=0)
    assert len(compact_df) == 4

    ref = empirical_shape_reference(compact_df, badness_max=2.0, margin=4.0)
    assert classify_component_shape(streak_row, ref) == "rod_like"
    for _, row in compact_df.iterrows():
        assert classify_component_shape(row, ref) != "rod_like"


def test_calibrate_null_generalizes_to_a_held_out_blob_size():
    calib_pairs = [_single_gaussian(box=20, sigma=s, amp=a, seed=400 + i)
                  for i, (s, a) in enumerate(
                      [(1.4, 2500), (1.7, 3500), (2.0, 5000), (2.3, 6500),
                       (2.6, 8500), (1.5, 3000), (1.9, 4500), (2.1, 5500)])]
    null = calibrate_null(calib_pairs, max_extent_px=30.0)
    assert null.n_calibration_blobs >= 8
    assert null.residual_std >= 0

    # a held-out clean single blob, a size NOT in the calibration set
    coords, intensity = _single_gaussian(box=20, sigma=3.0, amp=10000.0, seed=999999)
    fit = resolve_multiplicity(coords, intensity, max_extent_px=30.0, null=null, min_z=2.0)
    assert fit.k == 1


def test_calibrate_null_refuses_too_few_reference_blobs():
    pairs = [_single_gaussian(box=20, sigma=2.0, amp=5000.0, seed=i) for i in range(3)]
    with pytest.raises(ValueError):
        calibrate_null(pairs, max_extent_px=30.0)


def test_fit_k_gaussians_rejects_unsupported_k():
    coords, intensity = _single_gaussian(box=20, sigma=2.0, amp=5000.0, seed=1)
    with pytest.raises(ValueError):
        fit_k_gaussians(coords, intensity, 3, max_extent_px=30.0)


# ------------------------------------------------------- 2b. per-peak centroid + envelope

def test_fainter_peak_has_a_wider_bootstrap_envelope():
    """The 05 notebook's "report the spread" discipline, extended to one peak: less signal
    must mean a wider, more honest envelope, not the same tight number regardless of SNR.
    """
    box = 20
    bright_coords, bright_intensity = _single_gaussian(box=box, sigma=2.0, amp=8000.0,
                                                       seed=11, wobble_frac=0.0)
    faint_coords, faint_intensity = _single_gaussian(box=box, sigma=2.0, amp=250.0,
                                                     seed=12, wobble_frac=0.0)

    bright_env = bootstrap_centroid_envelope(bright_coords, bright_intensity, k=1,
                                             max_extent_px=30.0, n_boot=100, seed=0)
    faint_env = bootstrap_centroid_envelope(faint_coords, faint_intensity, k=1,
                                            max_extent_px=30.0, n_boot=100, seed=0)

    assert bright_env.n_kept > 0 and faint_env.n_kept > 0
    # compare the row-axis spread specifically (index 1 of frame,row,col) -- robust to any
    # one axis happening to tie
    assert faint_env.spread_std[0, 1] > bright_env.spread_std[0, 1]


def test_double_gaussian_components_each_get_their_own_envelope():
    box = 24
    c1 = np.array([box / 2 - 4, box / 2, box / 2])
    c2 = np.array([box / 2 + 4, box / 2, box / 2])
    cov = np.diag([1.5, 1.8, 1.8]) ** 2
    coords, intensity = _make_blob([c1, c2], [cov, cov], [6000.0, 5000.0], box=box, seed=7)

    env = bootstrap_centroid_envelope(coords, intensity, k=2, max_extent_px=30.0,
                                      n_boot=100, seed=0)
    assert env.k == 2
    assert env.centroids.shape == (2, 3)
    assert env.spread_std.shape == (2, 3)
    means = sorted(env.centroids, key=lambda m: m[0])
    assert np.allclose(means[0], c1, atol=1.5)
    assert np.allclose(means[1], c2, atol=1.5)
    # both components' envelopes should be tight relative to their 8-unit separation --
    # otherwise this "genuine double" fixture is not actually well-resolved
    assert env.spread_std[:, 0].max() < 4.0   # row-axis std, well under half the separation


def test_bootstrap_envelope_warns_at_low_voxel_count():
    coords, intensity = _single_gaussian(box=10, sigma=0.8, amp=2000.0, seed=5, wobble_frac=0.0)
    assert len(intensity) < 30   # this fixture's own precondition
    env = bootstrap_centroid_envelope(coords, intensity, k=1, max_extent_px=30.0,
                                      n_boot=50, seed=0)
    assert env.low_n_warning is True


def test_bootstrap_envelope_reports_n_kept_not_just_n_boot():
    coords, intensity = _single_gaussian(box=20, sigma=2.0, amp=5000.0, seed=1, wobble_frac=0.0)
    env = bootstrap_centroid_envelope(coords, intensity, k=1, max_extent_px=30.0,
                                      n_boot=40, seed=0)
    assert env.n_boot == 40
    assert 0 < env.n_kept <= 40
    assert env.bootstrap_means.shape == (env.n_kept, 1, 3)


# --------------------------------------------------------------- 3. rod/diffuse bridge

def test_spots_to_qsample_matches_the_underlying_geometry_calls():
    from midas_defect.geometry import Geometry, pixel_to_qlab, qlab_to_qsample
    import torch

    geom = Geometry(lsd_um=150_000.0, bcy_px=512.0, bcz_px=512.0, px_um=200.0,
                    wavelength_A=0.25, n_pix_y=1024, n_pix_z=1024,
                    omega_first_deg=-10.0, omega_step_deg=1.0, n_frames=20)
    spots = pd.DataFrame(dict(row=[500.0, 480.0, 520.0], col=[505.0, 490.0, 515.0],
                              frame=[0.0, 5.0, 10.0]))

    got = spots_to_qsample(spots, geom, omega_sign=1)

    omega_deg = geom.omega_first_deg + geom.omega_step_deg * spots["frame"].values
    qlab = pixel_to_qlab(spots["row"].values, spots["col"].values, geom, device="cpu")
    expected = qlab_to_qsample(
        qlab, torch.deg2rad(torch.as_tensor(omega_deg, dtype=qlab.dtype))
    ).detach().cpu().numpy()

    assert got.shape == (3, 3)
    assert np.allclose(got, expected)


def test_spots_to_qsample_honours_omega_sign():
    from midas_defect.geometry import Geometry

    geom = Geometry(lsd_um=150_000.0, bcy_px=512.0, bcz_px=512.0, px_um=200.0,
                    wavelength_A=0.25, n_pix_y=1024, n_pix_z=1024,
                    omega_first_deg=-10.0, omega_step_deg=1.0, n_frames=20)
    spots = pd.DataFrame(dict(row=[480.0], col=[490.0], frame=[5.0]))

    q_pos = spots_to_qsample(spots, geom, omega_sign=1)
    q_neg = spots_to_qsample(spots, geom, omega_sign=-1)
    assert not np.allclose(q_pos, q_neg)
