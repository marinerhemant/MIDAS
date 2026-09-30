"""midas_dfxm.populations: known-truth phantoms for the per-pixel two-population test."""
import numpy as np
import pytest

from midas_dfxm import populations as pp

X = 15.4732 + 0.01 * np.arange(61)          # the S168 theta grid (61 points, 10 mdeg)
FW = 0.12                                    # single-population FWHM (deg), ~S168


def _gauss(c, amp):
    return amp * np.exp(-0.5 * ((X[:, None, None] - c) / (FW / 2.3548)) ** 2)


def _frames(signal, seed, base=110.0, read=4.0):
    rng = np.random.default_rng(seed)
    lam = base + signal
    return (lam + rng.standard_normal(lam.shape) * np.sqrt(read ** 2 + 1.2 * np.clip(signal, 0, None) / 2)).astype(np.float32)


def _run(F, lit, base=110.0):
    b = np.full(lit.shape, base, np.float32)
    sig = np.full(int(lit.sum()), np.sqrt(4.0 ** 2 + 0.0))
    return pp.pixel_populations(F, X, b, lit, sig)


def test_decide_splits_two_peaks_in_angle_order():
    y = 100 * np.exp(-0.5 * ((X - 15.70) / 0.05) ** 2) + 60 * np.exp(-0.5 * ((X - 15.89) / 0.05) ** 2)
    n, S, E, _ = pp.decide(y[None].astype(float), np.array([1.0]), 5.0)
    assert n[0] == 2
    assert S[0, 0] < S[0, 1]                                  # populations come out in angle order
    assert X[S[0, 1]] < 15.89 and X[E[0, 0]] > 15.70


def test_single_peak_not_split():
    y = 100 * np.exp(-0.5 * ((X - 15.75) / 0.05) ** 2)
    zs, _ = pp.calibrate_zstar(X, FW)
    n, *_ = pp.decide(y[None].astype(float), np.array([1.0]), zs)
    assert n[0] == 1


def test_two_population_fraction_recovers_planted_half():
    H = W = 32
    lit = np.ones((H, W), bool)
    two = np.zeros((H, W), bool); two[:, W // 2:] = True        # right half: two populations, 50/50
    amp = 300.0
    sig = _gauss(np.full((H, W), 15.70), np.where(two, 0.5 * amp, amp)) + _gauss(np.full((H, W), 15.89), np.where(two, 0.5 * amp, 0.0))
    P = _run(_frames(sig, 1), lit)
    tp = pp.two_population(P, edge=15.80)
    both = tp["both"].reshape(H, W)
    # interior columns only: the 5 x 5 decision curve mixes the two halves within 2 px of the seam
    assert both[:, W // 2 + 3:].mean() > 0.9
    assert both[:, :W // 2 - 3].mean() < 0.02
    assert abs(tp["T"] - 0.5) < 0.1
    assert abs(tp["high_centre"] - 15.89) < 0.01
    sep = pp.separation_bootstrap(tp, lit, block=8, n_boot=200)
    assert abs(sep["median_mdeg"] - 190.0) < 15.0


def test_single_population_with_steep_gradient_is_not_two():
    # one population per pixel whose centre sweeps 0.30 deg across the image: the 5 x 5 decision curve
    # sees neighbours ~10-50 mdeg apart; this must not be called two populations
    H = W = 32
    lit = np.ones((H, W), bool)
    c = 15.60 + 0.30 * np.linspace(0, 1, W)[None, :].repeat(H, 0)
    F = _frames(_gauss(c, 300.0), 2)
    P = _run(F, lit)
    tp = pp.two_population(P, edge=15.75)
    assert tp["T"] < 0.02


def test_single_population_null_has_no_two_population_pixels():
    H = W = 24
    lit = np.ones((H, W), bool)
    two = np.zeros((H, W), bool); two[:, W // 2:] = True
    sig = _gauss(np.full((H, W), 15.70), np.where(two, 150.0, 300.0)) + _gauss(np.full((H, W), 15.89), np.where(two, 150.0, 0.0))
    F = _frames(sig, 3)
    b = np.full((H, W), 110.0, np.float32)
    null = pp.single_population_null(F, X, b, lit, np.full((H, W), 4.0), FW, kind="gauss", seed=0, edge=15.80)
    assert null["T"] < 0.01


def test_band_edges_ignore_excluded_frame():
    rng = np.random.default_rng(0)
    C = np.concatenate([rng.normal(15.70, 0.02, 4000), rng.normal(15.89, 0.02, 2000)])[:, None]
    I = np.ones_like(C)
    valid = np.ones(len(X), bool); valid[32] = False              # 15.7932, near the valley
    C = np.where(np.abs(C - X[32]) < 0.005, np.nan, C)            # nothing lands in the excluded bin
    cuts, _, _ = pp.band_edges(C, I, X, valid=valid)
    assert len(cuts) == 1 and abs(cuts[0] - X[32]) > 1e-6


def test_unknown_null_kind_raises():
    with pytest.raises(ValueError):
        pp.single_population_null(np.zeros((61, 4, 4), np.float32), X, np.zeros((4, 4)), np.ones((4, 4), bool),
                                  np.ones((4, 4)), FW, kind="nope", edge=15.8)


def test_confirmed_populations_gate_on_peak_snr():
    H = W = 24
    lit = np.ones((H, W), bool)
    rng = np.random.default_rng(5)
    F = (110.0 + rng.standard_normal((len(X), H, W)) * 4.0).astype(np.float32)          # pure noise
    F[:, :, W // 2:] += _gauss(np.full((H, W // 2), 15.75), 300.0)                       # bright single peak, right half
    P = _run(F, lit)
    assert P["peak_snr"].shape == (H * W,)
    conf = pp.confirmed_populations(P).any(1).reshape(H, W)
    assert conf[:, W // 2 + 2:].mean() > 0.95                      # bright half confirmed
    assert conf[:, :W // 2 - 2].mean() < 0.02                      # noise half not confirmed at peak SNR < 5
    assert (P["peak_snr"].reshape(H, W)[:, :W // 2 - 2] < 5).mean() > 0.9
