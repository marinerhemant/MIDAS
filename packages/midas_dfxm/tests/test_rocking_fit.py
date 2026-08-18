"""Single-axis rocking-curve fit + likelihood-ratio model selection.

These tests are the guards that make the K=2-vs-K=1 comparison honest. A two-component
fit can always reduce the residual, so the interesting question is never "does it fit
better" but "does it fit better for a legitimate reason". Each test below pins one way it
could win illegitimately.
"""
import math

import numpy as np
import pytest
import torch

from midas_dfxm import fit_rocking_curve, rocking_lrt, rocking_nll

DTYPE = torch.float64


def gauss(theta, c, fwhm, amp):
    s = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return amp * torch.exp(-0.5 * ((theta - c) / s) ** 2)


def scan(n=200, rng=0.2, center=10.4):
    return torch.linspace(center - rng / 2, center + rng / 2, n, dtype=DTYPE)


def test_single_component_recovers_center_and_width():
    th = scan()
    truth_c, truth_w = 10.41, 0.024
    data = gauss(th, truth_c, truth_w, 1000.0)[None] + 5.0
    fit = fit_rocking_curve(data, th, n_components=1, sigma=1.0, steps=800)
    assert abs(float(fit["center"][0, 0]) - truth_c) < 1e-3
    assert abs(float(fit["width"][0, 0]) - truth_w) < 3e-3


def test_two_component_recovers_split_pair():
    """The core capability: a genuinely bimodal curve gives back both centres."""
    th = scan()
    c1, c2, w = 10.388, 10.418, 0.020
    data = (gauss(th, c1, w, 1000.0) + gauss(th, c2, w, 700.0))[None] + 5.0
    fit = fit_rocking_curve(data, th, n_components=2, sigma=1.0, steps=1500)
    got = sorted(float(x) for x in fit["center"][0])
    assert abs(got[0] - c1) < 3e-3, got
    assert abs(got[1] - c2) < 3e-3, got


def test_lrt_separates_bimodal_from_unimodal():
    """The statistic must be large on a true doublet and small on a true singlet.

    This is the property the moment-based bimodality coefficient lacked at coarse
    sampling: it flagged broad curves, not double ones.
    """
    th = scan()
    w, noise = 0.020, 3.0
    g = torch.Generator().manual_seed(0)
    uni = gauss(th, 10.40, w, 1000.0)[None] + 5.0
    bi = (gauss(th, 10.388, w, 1000.0) + gauss(th, 10.418, w, 700.0))[None] + 5.0
    stats = []
    for truth in (uni, bi):
        d = truth + noise * torch.randn(truth.shape, generator=g, dtype=DTYPE)
        f1 = fit_rocking_curve(d, th, n_components=1, sigma=noise, steps=1200)
        f2 = fit_rocking_curve(d, th, n_components=2, sigma=noise, steps=1500)
        stats.append(float(rocking_lrt(f1, f2)[0]))
    uni_stat, bi_stat = stats
    assert bi_stat > 50.0, stats
    assert bi_stat > 10.0 * max(uni_stat, 1.0), stats


def test_broad_single_peak_does_not_masquerade_as_a_doublet():
    """The exact failure mode that retracted the NaMnO2 analysis.

    A WIDE unimodal curve must not score like a doublet. Moment-based bimodality
    correlated +0.24..0.36 with curve width; the LRT must not.
    """
    th = scan()
    noise = 3.0
    g = torch.Generator().manual_seed(1)
    narrow = gauss(th, 10.40, 0.015, 1000.0)[None] + 5.0
    broad = gauss(th, 10.40, 0.060, 1000.0)[None] + 5.0
    out = []
    for truth in (narrow, broad):
        d = truth + noise * torch.randn(truth.shape, generator=g, dtype=DTYPE)
        f1 = fit_rocking_curve(d, th, n_components=1, sigma=noise, steps=1200)
        f2 = fit_rocking_curve(d, th, n_components=2, sigma=noise, steps=1500)
        out.append(float(rocking_lrt(f1, f2)[0]))
    # both are single peaks, so neither should look like a doublet, and crucially the
    # broad one must not score higher just for being broad
    assert out[1] < 50.0, out
    assert out[1] < 10.0 * max(out[0], 1.0) + 30.0, out


def test_warm_start_makes_the_lrt_nonnegative():
    """K=2 nests K=1, so the likelihood ratio can never be negative.

    A cold-started K=2 fit routinely lands *worse* than K=1 on a single peak (its second
    component initialises out in the tail and Adam cannot fully retract it), which shows
    up as a large negative LRT -- an optimiser failure that would otherwise be read as
    evidence. Warm-starting from the K=1 solution makes the guarantee hold numerically.
    """
    th = scan(n=120)
    g = torch.Generator().manual_seed(4)
    noise = 4.0
    data = torch.stack([
        gauss(th, 10.40, 0.024, 4000.0) + 20.0,            # single
        gauss(th, 10.39, 0.024, 2400.0) + gauss(th, 10.415, 0.024, 1600.0) + 20.0,
    ])
    data = data + noise * torch.randn(data.shape, generator=g, dtype=DTYPE)
    f1 = fit_rocking_curve(data, th, n_components=1, sigma=noise, steps=800)
    cold = fit_rocking_curve(data, th, n_components=2, sigma=noise, steps=800)
    warm = fit_rocking_curve(data, th, n_components=2, sigma=noise, steps=800,
                             init_from=f1)
    lrt_warm = rocking_lrt(f1, warm)
    assert float(lrt_warm.min()) >= -1e-6, float(lrt_warm.min())
    # and it must still find the real doublet, not just sit at the K=1 answer
    assert float(lrt_warm[1]) > 100.0, float(lrt_warm[1])
    # the warm start is never worse than the cold one
    assert float(warm["nll"].sum()) <= float(cold["nll"].sum()) + 1e-6


def test_matches_an_independent_scipy_fit():
    """Our optimum must be scipy's optimum, not merely a stable place we always land.

    This is the test that caught the real defect: the fit converged reliably -- same
    answer from every starting point -- to a likelihood 47 worse than scipy's, because
    a softplus-parameterised background starting near zero has a near-dead Jacobian
    column that Levenberg-Marquardt damping then pins in place. Reproducibility is not
    correctness; only an independent implementation catches that.
    """
    from scipy.optimize import curve_fit

    rng = np.random.default_rng(0)
    th_np = np.linspace(10.3896, 10.3896 + 0.199, 200)
    hw = 0.5 * 0.024
    mu = 4000.0 * np.exp(-np.log(2.0) * ((th_np - 10.4896) / hw) ** 2) + 5.0
    dark = 20.0
    y = rng.poisson(mu + dark) - dark
    sg = np.sqrt(np.clip(y, 0, None) + dark)

    def model(t, c, w, a, b):
        return a * np.exp(-np.log(2.0) * ((t - c) / (0.5 * w)) ** 2) + b

    p, _ = curve_fit(model, th_np, y, p0=[10.4896, 0.024, 4000, 5], sigma=sg,
                     maxfev=200000)
    ref = 0.5 * (((model(th_np, *p) - y) / sg) ** 2).sum()

    fit = fit_rocking_curve(torch.as_tensor(y, dtype=DTYPE)[None],
                            torch.as_tensor(th_np, dtype=DTYPE), n_components=1,
                            sigma=torch.as_tensor(sg, dtype=DTYPE)[None],
                            steps=100, polish=150)
    assert float(fit["nll"][0]) <= ref + 0.05, (float(fit["nll"][0]), ref)
    assert abs(float(fit["width"][0, 0]) - p[1]) < 1e-4


@pytest.mark.parametrize("K,fit_eta", [(1, False), (2, False), (2, True)])
def test_analytic_jacobian_matches_autodiff(K, fit_eta):
    """The hand-derived LM Jacobian must equal the autodiff one.

    Hand-derived derivatives are exactly the kind of thing that is silently wrong -- a
    dropped chain-rule factor still converges, just to the wrong place, and every
    downstream likelihood ratio inherits the error. So the analytic path is validated
    against vmap(jacrev) rather than trusted, for every component count and with the
    pseudo-Voigt mixing both off and on.
    """
    th = scan(n=64)
    g = torch.Generator().manual_seed(7)
    data = torch.stack([
        gauss(th, 10.40, 0.025, 900.0) + 5.0,
        gauss(th, 10.395, 0.02, 600.0) + gauss(th, 10.42, 0.03, 400.0) + 5.0,
    ]) + 3.0 * torch.randn((2, len(th)), generator=g, dtype=DTYPE)
    sig = torch.sqrt(data.clamp_min(1.0))
    kw = dict(n_components=K, sigma=sig, fit_eta=fit_eta, steps=5)
    a = fit_rocking_curve(data, th, polish=3, jacobian="analytic", **kw)
    b = fit_rocking_curve(data, th, polish=3, jacobian="autodiff", **kw)
    # identical Jacobians => identical LM trajectory => identical optimum
    for key in ("center", "width", "amplitude", "background", "nll"):
        assert torch.allclose(a[key], b[key], atol=1e-9, rtol=1e-7), (
            key, a[key], b[key])


def test_free_eta_never_fits_worse_than_the_gaussian():
    """The pseudo-Voigt NESTS the Gaussian at eta=0, so with a warm start it must never
    fit worse -- on every pixel, not merely most of them.

    This is the guarantee an independent MLE satisfied at 0% violations while this fitter
    violated it on 2.8% of pixels, because Adam runs before the LM polish and is not
    monotone, so it could walk away from a good warm start. Fixed by keeping the raw
    starts in the candidate set.
    """
    th = scan(n=120)
    g = torch.Generator().manual_seed(11)
    truth = torch.stack([
        gauss(th, 10.40, 0.024, 4000.0) + 20.0,                      # pure Gaussian
        gauss(th, 10.40, 0.018, 3000.0) + gauss(th, 10.40, 0.060, 900.0) + 20.0,
    ])
    data = truth.repeat(30, 1)
    data = data + torch.sqrt(data.clamp_min(1.0)) * torch.randn(
        data.shape, generator=g, dtype=DTYPE)
    sig = torch.sqrt(data.clamp_min(1.0))
    G = fit_rocking_curve(data, th, n_components=1, sigma=sig, steps=100, polish=150)
    V = fit_rocking_curve(data, th, n_components=1, sigma=sig, steps=100, polish=150,
                          fit_eta=True, init_from=G)
    # tolerance: the guard candidate sits at eta = sigmoid(-20) = 2e-9, so it
    # reproduces the Gaussian to ~1e-9 relative rather than exactly
    worse = (V["nll"] > G["nll"] + 1e-6).sum().item()
    assert worse == 0, f"{worse}/{len(data)} pixels violate nesting"
    # and it must still find the tail on the genuinely tailed curves
    tailed = V["eta"][1::2, 0]
    assert float(tailed.median()) > 0.05, float(tailed.median())


def test_eta_free_null_is_not_pinned_at_the_bound():
    """"The null returns eta = 0.000" is uninformative when eta is bounded below at 0.

    sigmoid() cannot go negative, so the median pins to the boundary whether or not the
    fit is unbiased. The informative check is that a Gaussian truth does not produce a
    LARGE eta -- reported here as the fraction above a threshold, not as a median.
    """
    th = scan(n=100)
    g = torch.Generator().manual_seed(12)
    truth = (gauss(th, 10.40, 0.024, 3000.0) + 20.0).repeat(60, 1)
    data = truth + torch.sqrt(truth.clamp_min(1.0)) * torch.randn(
        truth.shape, generator=g, dtype=DTYPE)
    sig = torch.sqrt(data.clamp_min(1.0))
    G = fit_rocking_curve(data, th, n_components=1, sigma=sig, steps=100, polish=150)
    V = fit_rocking_curve(data, th, n_components=1, sigma=sig, steps=100, polish=150,
                          fit_eta=True, init_from=G)
    eta = V["eta"][:, 0]
    assert float((eta > 0.3).float().mean()) < 0.15, float((eta > 0.3).float().mean())


def test_background_can_go_negative():
    """Background-subtracted data legitimately sits slightly below zero, and a
    positivity-constrained background silently biases the whole fit there."""
    th = scan(n=80)
    data = (gauss(th, 10.40, 0.03, 900.0) - 12.0)[None]
    fit = fit_rocking_curve(data, th, n_components=1, sigma=4.0, steps=100, polish=150)
    assert float(fit["background"][0]) < -5.0, float(fit["background"][0])
    assert abs(float(fit["center"][0, 0]) - 10.40) < 3e-3


def test_init_from_rejects_a_larger_model():
    th = scan(n=60)
    data = (gauss(th, 10.40, 0.03, 500.0) + 5.0)[None]
    f2 = fit_rocking_curve(data, th, n_components=2, sigma=2.0, steps=200)
    with pytest.raises(ValueError):
        fit_rocking_curve(data, th, n_components=1, sigma=2.0, steps=10, init_from=f2)


def test_width_floor_blocks_single_point_spike():
    """Without a width floor a second component collapses onto one noisy point.

    That degeneracy makes K=2 win on *every* pixel, which is a fit artefact, not physics.
    """
    th = scan(n=60, rng=0.2)
    data = gauss(th, 10.40, 0.030, 1000.0)[None] + 5.0
    data[0, 7] += 400.0                                  # one hot point
    fit = fit_rocking_curve(data, th, n_components=2, sigma=3.0, steps=1500)
    step = fit["theta_step"]
    assert float(fit["width"].min()) >= step - 1e-9


def test_secondary_center_cannot_escape_the_bound():
    """Unconstrained mixtures run to the scan edges on real peaks; the bound stops that."""
    th = scan()
    g = torch.Generator().manual_seed(2)
    data = 5.0 + 3.0 * torch.abs(torch.randn((1, len(th)), generator=g, dtype=DTYPE))
    off = 0.02
    fit = fit_rocking_curve(data, th, n_components=2, sigma=3.0,
                            max_offset=off, steps=800)
    sep = float((fit["center"][0, 1] - fit["center"][0, 0]).abs())
    assert sep <= off + 1e-9, sep


def test_eta_zero_is_exactly_gaussian():
    """fit_eta=False must be the pure-Gaussian model, so `pseudo-Voigt` is opt-in."""
    th = scan(n=51)
    data = gauss(th, 10.40, 0.03, 500.0)[None] + 1.0
    fit = fit_rocking_curve(data, th, n_components=1, sigma=1.0, steps=300)
    assert float(fit["eta"].abs().max()) == 0.0


def test_poisson_and_gaussian_nll_both_run_and_prefer_the_truth():
    th = scan(n=80)
    truth = gauss(th, 10.40, 0.03, 800.0)[None] + 20.0
    g = torch.Generator().manual_seed(3)
    data = torch.poisson(truth, generator=g).to(DTYPE)
    for sig in (None, math.sqrt(20.0 + 800.0)):
        fit = fit_rocking_curve(data, th, n_components=1, sigma=sig, steps=800)
        assert abs(float(fit["center"][0, 0]) - 10.40) < 5e-3
        # the fitted model must beat a flat model of the same mean
        flat = data.mean(-1, keepdim=True).expand_as(data)
        s = None if sig is None else torch.tensor(sig, dtype=DTYPE)
        assert float(rocking_nll(fit["model"], data, s)) < float(rocking_nll(flat, data, s))


def test_per_point_sigma_downweights_the_peak_top():
    """Per-point sigma must be accepted and must actually change the weighting.

    Constant sigma over-weights high-count points; photon noise there is larger. If the
    two NLLs were equal the per-point option would be decorative.
    """
    th = scan(n=80)
    data = (gauss(th, 10.40, 0.03, 900.0) + 20.0)[None]
    model = (gauss(th, 10.405, 0.03, 900.0) + 20.0)[None]
    flat = torch.tensor(5.0, dtype=DTYPE)
    perpt = torch.sqrt(data.clamp_min(1.0))
    n_flat = float(rocking_nll(model, data, flat))
    n_pt = float(rocking_nll(model, data, perpt))
    assert n_pt != pytest.approx(n_flat, rel=1e-6)
    # a wrong shape must be rejected, not silently broadcast
    with pytest.raises(ValueError):
        rocking_nll(model, data, torch.ones(3, dtype=DTYPE))


def test_per_pixel_sigma_shape_accepted():
    th = scan(n=60)
    data = torch.stack([gauss(th, 10.40, 0.03, 500.0) + 5.0,
                        gauss(th, 10.41, 0.03, 900.0) + 5.0])
    sig = torch.tensor([2.0, 3.0], dtype=DTYPE)
    fit = fit_rocking_curve(data, th, n_components=1, sigma=sig, steps=600)
    assert abs(float(fit["center"][0, 0]) - 10.40) < 5e-3
    assert abs(float(fit["center"][1, 0]) - 10.41) < 5e-3


def test_batched_pixels_are_independent():
    """Per-pixel fits must not leak into each other -- the map would be smoothed by the
    optimiser rather than by the sample."""
    th = scan(n=100)
    a = gauss(th, 10.38, 0.02, 900.0) + 5.0
    b = gauss(th, 10.42, 0.02, 900.0) + 5.0
    data = torch.stack([a, b])
    fit = fit_rocking_curve(data, th, n_components=1, sigma=2.0, steps=800)
    assert abs(float(fit["center"][0, 0]) - 10.38) < 3e-3
    assert abs(float(fit["center"][1, 0]) - 10.42) < 3e-3


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_device_portable(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    dt = torch.float32
    th = scan(n=60).to(device=device, dtype=dt)
    data = (gauss(th, 10.40, 0.03, 800.0) + 5.0)[None]
    fit = fit_rocking_curve(data, th, n_components=1, sigma=2.0, steps=400)
    assert fit["center"].device.type == device
    assert abs(float(fit["center"][0, 0]) - 10.40) < 5e-3


def test_low_signal_center_stays_bounded():
    """Regression: the main peak centre was unbounded, so on noisy / no-peak pixels the LM step
    drove it to ~1e8. It must now stay within the scanned range (+ a small margin), and real
    peaks must still be recovered."""
    th = scan(n=61, rng=0.8, center=0.0)                 # +/- 0.4
    gen = torch.Generator().manual_seed(3)
    P = 200
    data = 0.02 * torch.rand(P, len(th), generator=gen, dtype=DTYPE)   # low-level noise, no peak
    data[:20] = data[:20] + gauss(th, 0.10, 0.05, 1.0)                 # 20 pixels carry a real peak
    out = fit_rocking_curve(data, th, steps=120, polish=120)
    c = out["center"][:, 0]
    assert torch.isfinite(c).all(), "non-finite centres"
    margin = 0.1 * float(th.max() - th.min())
    assert float(c.abs().max()) <= float(th.max()) + margin + 1e-6, \
        f"centre diverged: max|c| = {float(c.abs().max()):.2e}"
    assert float((c[:20] - 0.10).abs().median()) < 0.02, "real peaks not recovered"
