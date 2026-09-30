"""The support estimator on planted truth: broad peaks that fill most of the scan (the case
window="peak" drops), narrow peaks, cut peaks, noise, 2-D meshes with missing cells, 3-D grids,
theta-2theta axis detection and scan stacking."""
import math
import numpy as np
import pytest

from midas_dfxm.rocking import RockingScan
from midas_dfxm.support import support_curve, grid_axes, reduce_support, stack_scans, _build_grid


def curves_1d(x, c, fwhm, amp, base, n, rng, noise=True):
    s = fwhm / 2.3548
    lam = base + amp * np.exp(-0.5 * ((x[:, None] - c[None]) / s) ** 2)
    return (rng.poisson(lam) if noise else lam).astype(float)


def test_broad_peak_filling_the_scan_is_kept_and_unbiased():
    rng = np.random.default_rng(0)
    x = np.linspace(-0.02, 0.02, 41)                          # 41 points, 1 mdeg step
    N = 2000
    c = rng.uniform(-0.004, 0.004, N)
    D = curves_1d(x, c, 0.015, 1000.0, 120.0, N, rng)        # FWHM 15 mdeg: ~15 points above half
    o = support_curve(D, (41,), [x])
    assert np.mean(o["truncated"]) < 0.01
    assert np.mean(o["snr"] >= 10) > 0.99
    err = o["centre"][:, 0] - c
    assert abs(np.mean(err)) < 2e-5 and np.std(err) < 2e-4    # < 0.2 mdeg scatter, no bias
    # the Gaussian's tail never reaches the floor inside this scan (0.8-43 counts at the ends), so
    # the floor is not observable: the baseline must equal pedestal + the TRUE tail level where the
    # estimator looks (the 3-frame band at the end farther from the peak), within noise
    s_ = 0.015 / 2.3548
    tail = 1000.0 * np.exp(-0.5 * ((x[:, None] - c[None]) / s_) ** 2)
    far_hi = c < 0                                            # peak left of centre: far end is high
    tail_end = np.where(far_hi, tail[-3:].mean(0), tail[:3].mean(0))
    fallback = ~o["baseline_ok"]
    assert abs(np.median((o["baseline"] - 120.0 - tail_end)[fallback])) < 2.0
    true_S = 1000.0 * 0.015 / 2.3548 * math.sqrt(2 * math.pi) / 0.001
    assert abs(np.median(o["intensity"]) / true_S - 1) < 0.03


def test_narrow_peak():
    rng = np.random.default_rng(1)
    x = np.linspace(-0.01, 0.01, 21)
    N = 2000
    c = rng.uniform(-0.002, 0.002, N)
    D = curves_1d(x, c, 0.002, 300.0, 100.0, N, rng)
    o = support_curve(D, (21,), [x])
    err = o["centre"][:, 0] - c
    # pad=3 admits ~6 noise frames around a 2-point-wide peak: scatter 0.16 mdeg at pad 2, 0.21 at
    # pad 3 (sweep 2026-09-27); pad 3 is kept because pad <= 2 lost 6 % of a weak peak's intensity
    assert abs(np.mean(err)) < 5e-5 and np.std(err) < 2.5e-4
    assert np.mean(o["truncated"]) < 0.01
    assert abs(np.median(o["fwhm"][:, 0]) - 0.002) < 0.0006


def test_cut_peak_is_flagged_and_contained_one_is_not():
    rng = np.random.default_rng(2)
    x = np.linspace(-0.02, 0.02, 41)
    N = 500
    cut = curves_1d(x, np.full(N, 0.019), 0.010, 800.0, 100.0, N, rng)
    inside = curves_1d(x, np.full(N, 0.0), 0.010, 800.0, 100.0, N, rng)
    assert support_curve(cut, (41,), [x])["truncated"].mean() > 0.95
    assert support_curve(inside, (41,), [x])["truncated"].mean() < 0.02
    assert (support_curve(cut, (41,), [x])["cut"][:, 0, 1]).mean() > 0.95   # the HIGH side


def test_noise_estimate_on_flat_poisson():
    rng = np.random.default_rng(3)
    D = rng.poisson(400.0, (41, 3000)).astype(float)
    o = support_curve(D, (41,), [np.arange(41) * 0.001])
    assert abs(np.median(o["sigma_frame"]) / 20.0 - 1) < 0.1
    assert np.mean(o["snr"] >= 10) < 0.005                   # pure noise is (almost) never lit


def test_asymmetric_and_double_peaks_raise_the_shape_flags():
    x = np.linspace(-0.02, 0.02, 41)
    g = np.exp(-0.5 * (x / 0.003) ** 2)
    asym = np.where(x < 0, np.exp(-0.5 * (x / 0.001) ** 2), np.exp(-x / 0.008))
    two = np.exp(-0.5 * ((x + 0.012) / 0.002) ** 2) + 0.8 * np.exp(-0.5 * ((x - 0.012) / 0.002) ** 2)
    D = 100 + 1000 * np.stack([g, asym, two], 1)
    o = support_curve(D, (41,), [x])
    assert o["shape_resid"][0] < 0.05
    assert o["shape_resid"][1] > 3 * o["shape_resid"][0]
    assert o["main_share"][0] > 0.99 and o["main_share"][2] < 0.7


def test_2d_mesh_correlated_gaussian_with_missing_cells():
    rng = np.random.default_rng(4)
    a = np.linspace(-0.02, 0.02, 21); b = np.linspace(-0.05, 0.05, 11)
    A, B = np.meshgrid(a, b, indexing="ij")
    N = 400
    ca = rng.uniform(-0.003, 0.003, N); cb = rng.uniform(-0.01, 0.01, N)
    Sg = np.array([[0.004 ** 2, 0.5 * 0.004 * 0.012], [0.5 * 0.004 * 0.012, 0.012 ** 2]])
    P = np.linalg.inv(Sg)
    da = A[..., None] - ca; db = B[..., None] - cb
    q = P[0, 0] * da * da + 2 * P[0, 1] * da * db + P[1, 1] * db * db
    D = rng.poisson(100 + 2000 * np.exp(-0.5 * q)).astype(float)
    D[3, 7] = np.nan; D[15, 2] = np.nan                       # two unmeasured cells
    o = support_curve(D, (21, 11), [a, b])
    assert np.abs(o["centre"][:, 0] - ca).mean() < 3e-4
    assert np.abs(o["centre"][:, 1] - cb).mean() < 1e-3
    assert np.mean(o["truncated"]) < 0.05
    cov = np.nanmedian(o["cov"], 0)
    assert abs(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]) - 0.5) < 0.12


def test_3d_grid():
    rng = np.random.default_rng(5)
    ax = [np.linspace(-0.01, 0.01, 11), np.linspace(-0.03, 0.03, 7), np.linspace(-0.02, 0.02, 9)]
    G = np.meshgrid(*ax, indexing="ij")
    N = 200
    c = np.stack([rng.uniform(-0.002, 0.002, N), rng.uniform(-0.005, 0.005, N),
                  rng.uniform(-0.003, 0.003, N)], 1)
    q = sum(((G[i][..., None] - c[:, i]) / w) ** 2 for i, w in enumerate((0.003, 0.01, 0.006)))
    D = rng.poisson(100 + 3000 * np.exp(-0.5 * q)).astype(float)
    o = support_curve(D, (11, 7, 9), ax)
    assert (np.abs(o["centre"] - c).mean(0) < np.array([3e-4, 1.5e-3, 6e-4])).all()


def test_theta_2theta_axis_detection_and_stack():
    th = 15.0 + np.arange(21) * 0.001
    m1 = {"th": th, "tth": 2 * th, "chi": np.zeros(21)}
    ax = grid_axes(m1)
    assert [a[2] for a in ax] == ["strain"]                   # theta tracks 2theta/2: one axis
    m2 = {"th": np.tile(15.0 + np.arange(5) * 0.001, 4), "chi": np.repeat([-0.02, 0, 0.02, 0.04], 5)}
    shape, flat, coords = _build_grid(grid_axes(m2))
    assert shape == (5, 4)
    rng = np.random.default_rng(6)
    scans = []
    for chi in (-0.02, 0.0, 0.02):
        th = 15.0 + np.arange(15) * 0.001
        fr = rng.poisson(100, (15, 4, 5)).astype(np.float32)
        scans.append(RockingScan.from_arrays(fr, {"th": th, "chi": np.full(15, chi)}))
    st = stack_scans(scans)
    assert st.frames.shape == (45, 4, 5)
    maps = reduce_support(st, split_half=False)
    assert maps.grid_shape == (15, 3) and maps.value.shape == (4, 5, 2)


def test_full_frame_noise_is_not_lit_and_a_blob_is():
    rng = np.random.default_rng(7)
    th = 15.0 + np.arange(31) * 0.001
    fr = rng.poisson(120.0, (31, 200, 200)).astype(np.float32)
    yy, xx = np.mgrid[:200, :200]
    blob = ((yy - 100) ** 2 + (xx - 60) ** 2) < 20 ** 2
    fr[:, blob] += (400 * np.exp(-0.5 * ((th - th[15]) / 0.004) ** 2))[:, None]
    m = reduce_support(RockingScan.from_arrays(fr, {"th": th}), split_half=True)
    assert m.lit[blob].mean() > 0.99
    assert m.lit[~blob].mean() < 2e-4


def test_sensitivity_small_on_a_clean_peak():
    from midas_dfxm.support import support_sensitivity
    rng = np.random.default_rng(8)
    th = 15.0 + np.arange(41) * 0.001
    fr = rng.poisson(120.0, (41, 64, 64)).astype(np.float32)
    yy, xx = np.mgrid[:64, :64]
    c = th[20] + 0.004 * np.sin(yy / 10.0)
    fr += (800 * np.exp(-0.5 * ((th[:, None, None] - c[None]) / 0.004) ** 2)).astype(np.float32)
    rows, maps = support_sensitivity(RockingScan.from_arrays(fr, {"th": th}))
    for r in rows:
        assert r["axes"][0]["block_rms"] < 0.3              # mdeg, against a +-4 mdeg planted field
        assert abs(r["axes"][0]["block_slope"] - 1) < 0.05


def test_neighbourhood_support_recovers_a_broad_low_curve():
    """datasetJ S996 holes: counts spread over a wide angle, per-frame SNR ~3 (point = mean of 10
    repeats). The pixel's own argmax seeds on noise; a 5x5 neighbourhood decides the support, the
    pixel keeps its own counts, and the half-max width is taken on the smoothed curve."""
    rng = np.random.default_rng(11)
    th = 8.29 + np.arange(101) * 0.0005
    H = W = 60
    prof = np.exp(-0.5 * ((th - th[50]) / 0.005) ** 2)       # FWHM 11.8 mdeg, ~24 points
    lam = 20.0 + 4.5 * prof[:, None, None] * np.ones((1, H, W))
    fr = (rng.poisson(10 * lam) / 10.0).astype(np.float32)    # 10 repeats averaged: sigma ~1.45
    scan = RockingScan.from_arrays(fr, {"th": th})
    true_S = 4.5 * prof.sum()
    own = reduce_support(scan, support_smooth=1, lit_neighbours=0, split_half=False)
    nb = reduce_support(scan, support_smooth=5, lit_neighbours=0, split_half=False)
    assert np.nanmedian(own.intensity) < 0.8 * true_S         # the failure it fixes
    assert abs(np.nanmedian(nb.intensity) / true_S - 1) < 0.08
    assert nb.lit.mean() > 0.95 and nb.truncated.mean() < 0.02
    assert abs(np.nanmedian(nb.fwhm_mdeg[..., 0]) - 11.8) < 2.0
    assert abs(np.nanmedian(nb.centre_deg[..., 0]) - th[50]) < 3e-4


def test_centre_stays_in_range_and_point_parity_split_works():
    """Single-frame-per-point scans (datasetJ scans_single): even/odd split leaves every other cell
    empty -- sigma was NaN everywhere; and a near-zero support sum must not give a wild centre."""
    rng = np.random.default_rng(12)
    th = 15.0 + np.arange(41) * 0.001
    fr = rng.poisson(120.0, (41, 40, 40)).astype(np.float32)
    fr[:, :20] += (300 * np.exp(-0.5 * ((th - th[20]) / 0.003) ** 2))[:, None, None].astype(np.float32)
    m = reduce_support(RockingScan.from_arrays(fr, {"th": th}), lit_neighbours=0)
    c = m.centre_deg[..., 0]
    assert np.all(~np.isfinite(c) | ((c >= th[0]) & (c <= th[-1])))
    assert m.split == "point parity" and np.isfinite(m.sigma_global[0]) and m.sigma_global[0] > 0
    assert np.isfinite(m.sigma[:20][m.lit[:20]]).mean() > 0.9
