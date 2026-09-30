"""Tests for midas_peakfit.tracks (synthetic data only)."""
import numpy as np

from midas_peakfit.tracks import fixed_pixel_flags, merge_detections, window_test

N = 120


def test_repeated_detections_merge_into_one_feature():
    rng = np.random.default_rng(0)
    # feature A: 30 detections jittered around (20, 30); feature B: 5 around (80, 90)
    fa = np.arange(30); ra = 20 + rng.normal(0, 0.4, 30); ca = 30 + rng.normal(0, 0.4, 30)
    fb = np.arange(100, 105); rb = 80 + rng.normal(0, 0.4, 5); cb = 90 + rng.normal(0, 0.4, 5)
    d = np.r_[np.full(30, 1.5), np.full(5, 2.0)]
    out = merge_detections(np.r_[fa, fb], np.r_[ra, rb], np.r_[ca, cb], radius=3.0,
                           values={"d": d})
    assert len(out["row"]) == 2
    assert sorted(out["n_det"].tolist()) == [5, 30]
    i = int(np.argmax(out["n_det"]))
    assert out["first"][i] == 0 and out["last"][i] == 29
    assert abs(out["d"][i] - 1.5) < 1e-12


def test_min_det_drops_short_features():
    out = merge_detections([0, 1, 50], [10, 10.5, 60], [10, 10.2, 60], radius=3.0, min_det=2)
    assert len(out["row"]) == 1 and out["n_det"][0] == 2
    assert out["label"].tolist()[2] == -1


def _frames(rng, lam, n):
    return sum(rng.poisson(lam) for _ in range(n)).astype(float)


def test_window_test_separates_vanishing_spot_from_persistent_and_noise():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:N, 0:N]
    ok = np.ones((N, N), bool)
    bkg = np.full((N, N), 0.1)

    def spot(r0, c0, flux):
        return flux * np.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / (2 * 1.7 ** 2)) / (2 * np.pi * 1.7 ** 2)

    vanishing, persistent, nothing = (30, 30), (60, 90), (95, 20)
    lam_a = bkg + spot(*vanishing, 3.0) + spot(*persistent, 3.0)   # 3 counts / frame each
    lam_b = bkg + spot(*persistent, 3.0)
    na = nb = 200
    A, B = _frames(rng, lam_a, na), _frames(rng, lam_b, nb)
    rows = np.array([vanishing[0], persistent[0], nothing[0]], float)
    cols = np.array([vanishing[1], persistent[1], nothing[1]], float)
    t = window_test(A, bkg * na, na, B, bkg * nb, nb, ok, rows, cols)
    assert t.present_a.tolist() == [True, True, False]
    assert t.absent_b.tolist() == [True, False, False]


def test_fixed_pixel_flag():
    f = fixed_pixel_flags(np.array([10.0, 50.0]), np.array([10.0, 50.0]),
                          np.array([11.0, 200.0]), np.array([10.5, 200.0]), radius=3.0)
    assert f.tolist() == [True, False]
