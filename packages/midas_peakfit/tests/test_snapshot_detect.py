"""Tests for midas_peakfit.snapshot_detect (synthetic data only).

Each test encodes a failure mode seen on real still-frame data:
- a per-ring mean background turns single photons in a low-background region
  into highly significant "spots" when the background varies with azimuth;
- an absolute per-pixel count floor that scales with the number of summed
  frames cancels the gain from summing;
- the threshold must come from null images, so the false-alarm rate is what was
  asked for.
"""
import numpy as np
import pytest

from midas_peakfit.snapshot_detect import (
    SnapshotDetectorConfig,
    calibrate_threshold,
    detect_spots,
    local_background,
    ring_mean_background,
    score_map,
    valid_region,
)

N = 160


def _geometry():
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    r = np.hypot(yy - N / 2, xx - N / 2)
    ring = np.floor(r / 2).astype(int)                      # ring bins, 2 px wide
    ok = np.ones((N, N), bool)
    ok[:, 78:82] = False                                     # a gap column
    return yy, xx, r, ring, ok


def _background(yy, xx, level=0.1, gradient=0.0):
    """Smooth background; ``gradient`` adds azimuthal (left-right) variation."""
    return level * (1.0 + gradient * (xx / N))


def _plant(lam, spots, sigma=1.7):
    yy, xx = np.mgrid[0:N, 0:N]
    out = lam.copy()
    for (r0, c0, flux) in spots:
        out += flux * np.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return out


SPOTS = [(30.3, 30.6, 0), (40.2, 120.7, 0), (120.5, 35.1, 0), (125.4, 125.2, 0), (60.6, 100.3, 0)]


def _recovered(det, spots, tol=2.0):
    if len(det) == 0:
        return 0
    return sum(np.min(np.hypot(det[:, 0] - r, det[:, 1] - c)) < tol for r, c, _ in spots)


@pytest.mark.parametrize("stat", ["gauss", "poisson"])
def test_recovers_planted_spots_at_low_background(stat):
    rng = np.random.default_rng(1)
    yy, xx, r, ring, ok = _geometry()
    lam = _background(yy, xx, 0.1)
    spots = [(a, b, 80.0) for a, b, _ in SPOTS]
    img = np.where(ok, rng.poisson(_plant(lam, spots)), -1).astype(float)
    ok2 = img >= 0
    cfg = SnapshotDetectorConfig(statistic=stat, n_null_images=20, local_box=21)
    coarse = ring_mean_background(img, ok2, ring)[1]
    T = calibrate_threshold(img, ok2, coarse, cfg, np.random.default_rng(2), ring_index=ring)
    det = detect_spots(img, ok2, coarse, T, cfg)
    assert _recovered(det, spots) == len(spots)
    assert len(det) <= len(spots) + 2          # no flood of false peaks


@pytest.mark.parametrize("stat", ["gauss", "poisson"])
def test_false_alarm_rate_matches_target_on_pure_poisson(stat):
    yy, xx, r, ring, ok = _geometry()
    lam = _background(yy, xx, 0.3)
    cfg = SnapshotDetectorConfig(statistic=stat, n_null_images=30, fa_per_image=0.1, local_box=21)
    rng = np.random.default_rng(3)
    img0 = np.where(ok, rng.poisson(lam), 0).astype(float)
    coarse0 = ring_mean_background(img0, ok, ring)[1]
    T = calibrate_threshold(img0, ok, coarse0, cfg, np.random.default_rng(4), ring_index=ring)
    n = 0
    for k in range(30):
        img = np.where(ok, rng.poisson(lam), 0).astype(float)
        coarse = ring_mean_background(img, ok, ring)[1]
        n += len(detect_spots(img, ok, coarse, T, cfg))
    assert n / 30 < 0.5        # target 0.1; generous bound for 30 draws


def test_ring_mean_background_trap_is_avoided_by_local_background():
    """Background that varies strongly with azimuth inside a ring: scoring against
    the ring mean flags many single photons; scoring against the local background
    does not."""
    rng = np.random.default_rng(5)
    yy, xx, r, ring, ok = _geometry()
    lam = np.where(xx < N / 2, 0.01, 0.5)                    # one dark half, one bright half
    img = np.where(ok, rng.poisson(lam), 0).astype(float)
    cfg = SnapshotDetectorConfig(statistic="gauss", local_box=15)
    coarse = ring_mean_background(img, ok, ring)[1]
    safe = valid_region(ok, cfg.edge_px, 0)
    _, s_ring = score_map(img, ok, coarse, cfg)
    _, s_local = score_map(img, ok, local_background(img, ok, coarse, cfg.local_box), cfg)
    # Against the ring mean the bright half's ordinary photons look significant, so
    # the ring-mean score has a much heavier upper tail than the local one.
    q_ring = np.percentile(s_ring[safe], 99.9)
    q_local = np.percentile(s_local[safe], 99.9)
    assert q_ring > 1.5 * q_local


def test_summing_lowers_the_detection_floor():
    """With no absolute count floor, summing W frames of a stationary spot must
    detect spots that single frames miss."""
    yy, xx, r, ring, ok = _geometry()
    lam = _background(yy, xx, 0.2)
    flux = 12.0
    spots = [(a, b, flux) for a, b, _ in SPOTS]
    cfg = SnapshotDetectorConfig(statistic="poisson", n_null_images=20, local_box=21)

    def recovered(W, seed):
        rng = np.random.default_rng(seed)
        img = sum(np.where(ok, rng.poisson(_plant(lam, spots)), 0).astype(float) for _ in range(W))
        coarse = ring_mean_background(img, ok, ring)[1]
        T = calibrate_threshold(img, ok, coarse, cfg, np.random.default_rng(seed + 1), ring_index=ring)
        return _recovered(detect_spots(img, ok, coarse, T, cfg), spots)

    r1 = np.mean([recovered(1, s) for s in (10, 20)])
    r25 = np.mean([recovered(25, s) for s in (30, 40)])
    assert r25 >= len(SPOTS) - 1
    assert r25 > r1


def test_margin_excludes_spots_next_to_invalid_pixels():
    rng = np.random.default_rng(7)
    yy, xx, r, ring, ok = _geometry()
    lam = _background(yy, xx, 0.1)
    near_gap = [(50.2, 87.5, 200.0)]                          # 6 px from the gap edge
    img = np.where(ok, rng.poisson(_plant(lam, near_gap)), 0).astype(float)
    coarse = ring_mean_background(img, ok, ring)[1]
    cfg0 = SnapshotDetectorConfig(n_null_images=10, local_box=21, margin_px=0)
    cfg8 = SnapshotDetectorConfig(n_null_images=10, local_box=21, margin_px=8)
    T = calibrate_threshold(img, ok, coarse, cfg0, np.random.default_rng(8), ring_index=ring)
    assert _recovered(detect_spots(img, ok, coarse, T, cfg0), near_gap) == 1
    assert _recovered(detect_spots(img, ok, coarse, T, cfg8), near_gap) == 0


def test_measure_spot_sigma_recovers_the_true_width():
    from midas_peakfit.snapshot_detect import measure_spot_sigma, ring_mean_background
    rng = np.random.default_rng(3)
    n = 300
    yy, xx = np.mgrid[0:n, 0:n]
    ring = (np.hypot(yy - n / 2, xx - n / 2) // 4).astype(int)
    for true_sig in (1.0, 1.7):
        lam = np.full((n, n), 0.3)
        for r in range(30, n - 30, 24):
            for c in range(30, n - 30, 24):
                lam += 400 * np.exp(-((yy - r - 0.3) ** 2 + (xx - c + 0.2) ** 2) / (2 * true_sig ** 2)) / (2 * np.pi * true_sig ** 2)
        img = rng.poisson(lam).astype(float)
        ok = np.ones_like(img, bool)
        coarse = ring_mean_background(img, ok, ring)[1]
        sg, k = measure_spot_sigma(img, ok, coarse, SnapshotDetectorConfig(sigma_px=1.0))
        assert k >= 50
        assert abs(sg - true_sig) < 0.12 * true_sig, (true_sig, sg)
