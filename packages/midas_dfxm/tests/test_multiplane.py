"""multiplane.py must recover a planted answer and must not repeat either bug found building
it on the real Mg-4Al ID03 campaign: a grain mask threshold that assumes background is the
majority (wrong on a crop already mostly grain), and skipping the per-frame pedestal before
accumulating a moment (lets the angular baseline-removal step swallow it instead, silently).
"""
import numpy as np
import pytest

from midas_dfxm.multiplane import (
    accumulate_marginals, angular_moments, derive_grain_mask, predicted_fov_row_gradient,
    strain_and_tilt,
)
from midas_dfxm.rocking import RockingScan, example_rocking_scan

pytestmark = pytest.mark.unit


def _mesh_scan(seed=0, pedestal=0.0):
    scan = example_rocking_scan("mesh", seed=seed)
    if pedestal:
        scan = RockingScan.from_arrays(scan.frames + pedestal, scan.motors, meta=scan.meta,
                                       source=scan.source)
    return scan


def test_accumulate_marginals_recovers_the_known_mesh_tilt():
    scan = _mesh_scan()
    mu_g, chi_g, Mmu, Mchi, Mob = accumulate_marginals([scan], mu_key="th", chi_key="chi")
    assert Mmu.shape == (len(mu_g),) + scan.frames.shape[1:]
    assert Mob.shape == (1,) + scan.frames.shape[1:]
    mu_m = angular_moments(Mmu, mu_g)
    truth = scan.meta["truth"]["tilt_mdeg"][..., 0]  # th tilt, mdeg
    fit_mdeg = 1000 * (mu_m["com"] - np.median(mu_m["com"]))
    # bright pixels only: the corners carry almost no signal (beam taper), noisy there
    bright = mu_m["S0"] > np.percentile(mu_m["S0"], 70)
    r = np.corrcoef(fit_mdeg[bright], truth[bright])[0, 1]
    assert r > 0.9


def test_accumulate_marginals_rejects_non_tilt2d_scan():
    scan = example_rocking_scan("single")
    with pytest.raises(ValueError, match="tilt2d"):
        accumulate_marginals([scan], mu_key="th", chi_key="chi")


def test_accumulate_marginals_rejects_mismatched_frame_shapes():
    a = _mesh_scan(seed=1)
    b = _mesh_scan(seed=2)
    b = RockingScan.from_arrays(b.frames[:, :-2, :-2], b.motors, meta=b.meta, source=b.source)
    with pytest.raises(ValueError, match="frame shape"):
        accumulate_marginals([a, b], mu_key="th", chi_key="chi")


def test_pedestal_subtraction_prevents_baseline_from_absorbing_it():
    """Regression test for the bug found live: skipping the per-frame pedestal let
    angular_moments's baseline-removal absorb ~91% instead of the true ~O(30%)."""
    scan_clean = _mesh_scan(seed=0)
    scan_pedestal = _mesh_scan(seed=0, pedestal=5000.0)   # large uniform per-frame offset

    mu_g, _, Mmu_clean, _, _ = accumulate_marginals([scan_clean], mu_key="th", chi_key="chi",
                                                     pedestal_percentile=5.0)
    frac_clean = np.median(angular_moments(Mmu_clean, mu_g)["frac_base"])

    # accumulate_marginals subtracts the per-frame pedestal even when a large one is added
    mu_g2, _, Mmu_fixed, _, _ = accumulate_marginals([scan_pedestal], mu_key="th", chi_key="chi",
                                                      pedestal_percentile=5.0)
    frac_fixed = np.median(angular_moments(Mmu_fixed, mu_g2)["frac_base"])
    assert frac_fixed == pytest.approx(frac_clean, abs=1e-9), (
        "accumulate_marginals should remove a large per-frame pedestal before it reaches "
        "angular_moments -- a uniform additive pedestal must not change frac_base at all")

    # what skipping that step does: bin raw frames directly (the first version's bug)
    frames = scan_pedestal.frames
    mu_round = np.round(scan_pedestal.motors["th"], 3)
    mu_g3 = np.sort(np.unique(mu_round))
    Mmu_broken = np.zeros((len(mu_g3),) + frames.shape[1:])
    for i, g in enumerate(mu_g3):
        Mmu_broken[i] = frames[mu_round == g].sum(0)
    frac_broken = np.median(angular_moments(Mmu_broken, mu_g3)["frac_base"])
    assert frac_broken > frac_fixed + 0.2, (
        "skipping the per-frame pedestal should inflate frac_base well above the fixed value "
        "(real data: ~0.91 broken vs ~0.33 fixed)")


def _textured_crop(H=200, W=200, background_frac=0.13, seed=0):
    """A bright, streaky 'grain' filling most of the field with a true-background margin only
    at the corners -- the exact scenario (crop already mostly foreground) that broke the naive
    background_percentile=50 mask on the real data."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    # a border of width m leaves foreground fraction ((H-2m)/H)**2; invert for m
    margin = int(round(H * (1 - np.sqrt(1 - background_frac)) / 2))
    true_grain = np.ones((H, W), bool)
    true_grain[:margin] = False; true_grain[-margin:] = False
    true_grain[:, :margin] = False; true_grain[:, -margin:] = False
    texture = 5000 + 3000 * np.sin(xx / 11.0) * np.cos(yy / 7.0 + xx / 23.0)
    image = np.where(true_grain, texture, 50.0) + rng.normal(0, 20, (H, W))
    return image, true_grain


def test_derive_grain_mask_recovers_a_mostly_foreground_crop():
    image, truth = _textured_crop()
    mask = derive_grain_mask(image, background_percentile=5.0, threshold_frac=0.10)
    iou = (mask & truth).sum() / (mask | truth).sum()
    assert iou > 0.9


def test_derive_grain_mask_background_percentile_must_match_the_crop():
    """The exact bug: background_percentile=50 assumes background is the majority. On a crop
    that's mostly grain, it picks a threshold near the grain's own median and fragments it."""
    image, truth = _textured_crop()
    mask = derive_grain_mask(image, background_percentile=50.0, threshold_frac=0.15)
    iou = (mask & truth).sum() / (mask | truth).sum()
    assert iou < 0.7, "background_percentile=50 on a mostly-foreground crop should NOT recover it"


def test_derive_grain_mask_refuses_an_empty_threshold():
    image = np.full((32, 32), 100.0)
    with pytest.raises(ValueError, match="no pixel exceeds"):
        derive_grain_mask(image, background_percentile=5.0, threshold_frac=1e6)


def test_strain_and_tilt_matches_hand_computed_geometry():
    grain = np.ones((3, 3), bool)
    ob_com = np.array([[16.0, 16.02, 16.0], [16.0, 16.0, 16.0], [16.0, 16.0, 16.04]])
    mu_com = np.full((3, 3), 20.5); mu_com[0, 0] = 20.51
    chi_com = np.full((3, 3), -5.0)
    mu_m = {"com": mu_com}; chi_m = {"com": chi_com}; ob_m = {"com": ob_com}
    LAM = 0.7293188
    out = strain_and_tilt(mu_m, chi_m, ob_m, grain, LAM)

    th = np.deg2rad(16.0) / 2.0
    assert out.theta_B_deg == pytest.approx(np.rad2deg(th), abs=1e-9)
    assert out.d_spacing_A == pytest.approx(LAM / (2 * np.sin(th)), rel=1e-12)
    # the +0.02 deg obpitch pixel: strain = -(d(2theta)/2)/tan(theta)
    expected = -(np.deg2rad(0.02) / 2.0) / np.tan(th)
    assert out.strain[0, 1] == pytest.approx(expected, rel=1e-9)
    assert out.strain[1, 1] == pytest.approx(0.0, abs=1e-12)
    assert out.tilt_mu[0, 0] == pytest.approx(0.01, abs=1e-9)
    assert np.all(out.tilt_chi == 0)


def test_predicted_fov_row_gradient_matches_the_verified_mg4al_number():
    """Same numbers as the Mg-4Al g9 ID03 campaign (RETRACTION_strain.md / verified live in
    reduce_id03_scan.ipynb Part C): predicted ~2.21 ue/row."""
    ob_g = np.array([16.1191, 16.1391, 16.1591, 16.1791, 16.1991, 16.2191, 16.2391, 16.2591])
    ffz = np.array([1438.9406, 1440.8322, 1442.7238, 1444.6158, 1446.5079, 1448.4010,
                   1450.2941, 1452.1876])
    pred = predicted_fov_row_gradient(ffz=ffz, ob_g=ob_g, obx=264.0, pixel_um=6.5,
                                      magnification=2.0, theta_B_deg=8.09928)
    assert pred == pytest.approx(2.2145, abs=0.01)


def test_predicted_fov_row_gradient_refuses_a_negative_distance():
    ob_g = np.array([16.0, 16.02])
    ffz = np.array([1.0, 1.0])   # zero ffz step -> zero sample-to-detector distance
    with pytest.raises(ValueError, match="objective->detector distance"):
        predicted_fov_row_gradient(ffz=ffz, ob_g=ob_g, obx=264.0, pixel_um=6.5,
                                   magnification=2.0, theta_B_deg=8.0)
