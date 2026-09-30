"""Censored intensity fit: recovers per-grain scales and the |F| exponent under thresholding."""
from __future__ import annotations

import copy
import math
import numpy as np
import pytest
import torch

from midas_nf_fitorientation.intensity import (
    ContributionTable, IntensityModel, RecordedPixels, fit_intensity, lorentz_polarization,
)

NF, NY, NZ = 40, 96, 96


def _table(rng, n_grains=6, rows_per_grain=120, n_refl=5):
    g = np.repeat(np.arange(n_grains), rows_per_grain)
    R = g.size
    return ContributionTable(
        grain=g, refl=rng.integers(0, n_refl, R), sol=np.zeros(R, np.int8),
        frame=rng.uniform(2, NF - 3, R), y=rng.uniform(4, NY - 5, R), z=rng.uniform(4, NZ - 5, R),
        w_geom=rng.uniform(0.5, 2.0, R), log_F=np.log(np.linspace(1.0, 4.0, n_refl)), n_grains=n_grains)


def _record(table, s_true, p_true, sigma_psf, sigma, blanket, rng):
    m = IntensityModel(table, NF, NY, NZ, sigma_psf=sigma_psf, p=p_true)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true))
        mu = m().numpy()
    x = mu + rng.normal(0, sigma, mu.size)
    rec = x > blanket
    k = m.support_key[rec]
    f, rem = np.divmod(k, NY * NZ); y, z = np.divmod(rem, NZ)
    return RecordedPixels.from_arrays(f, y, z, x[rec] - blanket, NY, NZ)


def test_lorentz_polarization_limits():
    lp = lorentz_polarization(np.radians([10.0]), np.radians([90.0]))
    s = np.sin(np.radians(10.0))
    assert lp[0] == pytest.approx((1 - s ** 2) / s)


def test_support_mass_conserved():
    rng = np.random.default_rng(1)
    t = _table(rng)
    m = IntensityModel(t, NF, NY, NZ, sigma_psf=0.9, p=2.0)
    with torch.no_grad():
        total = float(m().sum()); expect = float(m.row_amplitude().sum())
    assert total == pytest.approx(expect, rel=1e-10)


def test_censored_fit_recovers_scales_and_p():
    rng = np.random.default_rng(0)
    t = _table(rng)
    s_true = np.exp(rng.normal(0, 0.5, t.n_grains)) * 3.0
    rec = _record(t, s_true, 2.0, 1.0, 2.0, 5.0, rng)
    fit = fit_intensity(t, rec, NF, NY, NZ, blanket=5.0, p_init=1.0, sigma_psf_init=0.7)
    assert fit.p == pytest.approx(2.0, abs=0.08)
    assert fit.sigma_psf == pytest.approx(1.0, abs=0.1)
    rel = fit.relative_scales(); truth = s_true / np.exp(np.log(s_true).mean())
    assert np.median(np.abs(np.log(rel / truth))) < 0.05


def test_median_radius_matches_scipy_on_support():
    """With every neighbour inside the support, the model's median equals scipy's 3x3 median."""
    from scipy.ndimage import median_filter
    rng = np.random.default_rng(3)
    t = _table(rng, n_grains=2, rows_per_grain=400)
    m0 = IntensityModel(t, NF, NY, NZ, psf_radius=2, sigma_psf=1.0)
    m1 = IntensityModel(t, NF, NY, NZ, psf_radius=2, sigma_psf=1.0, median_radius=1)
    with torch.no_grad():
        raw = m0().numpy(); med = m1().numpy()
    img = np.zeros((NF, NY, NZ)); f, rem = np.divmod(m0.support_key, NY * NZ); y, z = np.divmod(rem, NZ)
    img[f, y, z] = raw
    ref = np.stack([median_filter(img[k], size=3, mode="constant") for k in range(NF)])[f, y, z]
    assert np.allclose(med, ref)


def test_tie_scales_and_exclusion_and_heldout():
    from midas_nf_fitorientation.intensity import heldout_nll
    rng = np.random.default_rng(5)
    t = _table(rng)
    s_true = np.exp(rng.normal(0, 0.5, t.n_grains)) * 3.0
    rec = _record(t, s_true, 2.0, 1.0, 2.0, 5.0, rng)
    full = fit_intensity(t, rec, NF, NY, NZ, blanket=5.0, p_init=1.0, sigma_psf_init=0.7)
    one = fit_intensity(t, rec, NF, NY, NZ, blanket=5.0, p_init=1.0, sigma_psf_init=0.7, tie_scales=True)
    assert np.ptp(one.scales) == 0 and one.scales.size == t.n_grains
    k, nf, _, _ = heldout_nll(t, rec, full, NF, NY, NZ, blanket=5.0)
    _, no, _, _ = heldout_nll(t, rec, one, NF, NY, NZ, blanket=5.0)
    assert nf.mean() < no.mean()                      # per-grain scales explain the data better
    ex = k[:50]
    k2, n2, _, _ = heldout_nll(t, rec, full, NF, NY, NZ, blanket=5.0, exclude_keys=ex)
    assert k2.size == k.size - 50 and not np.isin(ex, k2).any()


def test_anisotropic_psf_recovered():
    rng = np.random.default_rng(7)
    t = _table(rng, rows_per_grain=150)
    s_true = np.exp(rng.normal(0, 0.3, t.n_grains)) * 6.0
    m = IntensityModel(t, NF, NY, NZ, psf_radius=5, sigma_psf=(0.8, 2.0), p=2.0)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true)); mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); y, z = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, y, z, x[rec] - 5.0, NY, NZ)
    fit = fit_intensity(t, R, NF, NY, NZ, blanket=5.0, psf_radius=5, sigma_psf_init=1.2, p_init=1.5)
    assert fit.sigma_psf_yz[0] == pytest.approx(0.8, abs=0.1)
    assert fit.sigma_psf_yz[1] == pytest.approx(2.0, abs=0.15)


def test_heteroscedastic_noise_gain_recovered():
    rng = np.random.default_rng(11)
    t = _table(rng, rows_per_grain=200)
    s_true = np.exp(rng.normal(0, 0.3, t.n_grains)) * 8.0
    m = IntensityModel(t, NF, NY, NZ, sigma_psf=1.0, p=2.0)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true)); mu = m().numpy()
    x = mu + rng.normal(0, 1, mu.size) * np.sqrt(4.0 + 2.0 * mu); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); y, z = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, y, z, x[rec] - 5.0, NY, NZ)
    het = fit_intensity(t, R, NF, NY, NZ, blanket=5.0, noise_model="het", p_init=1.5)
    const = fit_intensity(t, R, NF, NY, NZ, blanket=5.0, p_init=1.5)
    assert het.noise_gain == pytest.approx(2.0, rel=0.2)
    assert het.sigma_noise == pytest.approx(2.0, rel=0.2)
    assert het.nll < const.nll
    assert abs(het.p - 2.0) < abs(const.p - 2.0) + 0.02


def test_splat_blur_conserves_mass_and_matches_psf_at_integer_positions():
    from midas_nf_fitorientation.intensity import SplatBlurModel
    rng = np.random.default_rng(13)
    t = _table(rng, n_grains=3, rows_per_grain=60)
    t.y = np.rint(t.y); t.z = np.rint(t.z); t.frame = np.floor(t.frame) + 0.5      # integer pixel, frame centre
    a = SplatBlurModel(t, NF, NY, NZ, blur_radius=3, sigma_psf=1.1, frame_model="linear")
    b = IntensityModel(t, NF, NY, NZ, psf_radius=3, sigma_psf=1.1)
    with torch.no_grad():
        mu_a, mu_b = a().numpy(), b().numpy()
        assert mu_a.sum() == pytest.approx(float(a.row_amplitude().sum()), rel=1e-10)
    pos = np.searchsorted(a.support_key, b.support_key)
    assert np.array_equal(a.support_key[pos], b.support_key)
    assert np.allclose(mu_a[pos], mu_b)
    other = np.setdiff1d(np.arange(a.n_support), pos)
    assert np.allclose(mu_a[other], 0.0)


def test_barycentric_subpoints_fill_the_triangle():
    from midas_nf_fitorientation.intensity import _barycentric_subpoints
    for n in (1, 2, 4):
        b = _barycentric_subpoints(n)
        assert b.shape == (n * n, 3) and np.all(b > 0) and np.allclose(b.sum(1), 1)
        assert np.allclose(b.mean(0), 1 / 3)


def test_splat_fit_recovers_p_and_psf():
    from midas_nf_fitorientation.intensity import SplatBlurModel
    rng = np.random.default_rng(17)
    t = _table(rng, rows_per_grain=150)
    s_true = np.exp(rng.normal(0, 0.3, t.n_grains)) * 6.0
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=5, sigma_psf=(0.7, 1.8), p=2.0)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true)); mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); y, z = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, y, z, x[rec] - 5.0, NY, NZ)
    fit = fit_intensity(t, R, NF, NY, NZ, blanket=5.0, psf_radius=5, sigma_psf_init=1.2, p_init=1.5, model_kind="splat")
    assert fit.p == pytest.approx(2.0, abs=0.08)
    assert fit.sigma_psf_yz[0] == pytest.approx(0.7, abs=0.15) and fit.sigma_psf_yz[1] == pytest.approx(1.8, abs=0.15)


def test_gauss_frame_model_hard_limit_and_recovery():
    """sigma_omega -> 0 puts each row in the frame containing it (the C code's floor); sigma_omega is recoverable."""
    from midas_nf_fitorientation.intensity import SplatBlurModel
    rng = np.random.default_rng(19)
    t = _table(rng, n_grains=4, rows_per_grain=150)
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=3, sigma_psf=0.8, sigma_omega=1e-3)
    with torch.no_grad():
        fw = m.frame_weights().numpy()
    assert np.all(fw.argmax(1) == 1)                                        # slot 1 == floor(frame)
    assert np.mean(fw[:, 1] > 0.999) > 0.98                                  # only rows at a bin edge leak
    s_true = np.exp(rng.normal(0, 0.3, t.n_grains)) * 8.0
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=4, sigma_psf=(0.8, 1.5), sigma_omega=0.35, p=2.0)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true)); mu = m().numpy()
        assert mu.sum() == pytest.approx(float(m.row_amplitude().sum()), rel=1e-8)
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); y, z = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, y, z, x[rec] - 5.0, NY, NZ)
    fit = fit_intensity(t, R, NF, NY, NZ, blanket=5.0, psf_radius=4, sigma_psf_init=1.0, p_init=1.5,
                        model_kind="splat", sigma_omega_init=0.15)
    assert fit.sigma_omega == pytest.approx(0.35, abs=0.07)
    assert fit.p == pytest.approx(2.0, abs=0.08)


def test_spot_level_fit_recovers_p_and_scales_with_correct_shape():
    """Correctness: with the true spot shape assumed, the spot-level censored fit recovers p and grain scales.
    (Shape-insensitivity is NOT guaranteed: on faint, mostly-censored spots a wrong assumed PSF biases p;
    that is tested on real data, not assumed.)"""
    from midas_nf_fitorientation.intensity import SplatBlurModel, fit_spot_intensity
    rng = np.random.default_rng(23)
    t = _table(rng, n_grains=6, rows_per_grain=150)
    s_true = np.exp(rng.normal(0, 0.4, t.n_grains)) * 10.0
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=5, sigma_psf=(0.8, 2.0), sigma_omega=0.15, p=2.0)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true)); mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); y, z = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, y, z, x[rec] - 5.0, NY, NZ)
    fit = fit_spot_intensity(t, R, NF, NY, NZ, blanket=5.0, sigma_psf=(0.8, 2.0), sigma_omega=0.15,
                             sigma_noise_init=2.0, p_init=1.0)
    assert fit.p == pytest.approx(2.0, abs=0.05)
    rel = fit.scales / np.exp(np.log(fit.scales).mean()); tru = s_true / np.exp(np.log(s_true).mean())
    assert np.max(np.abs(np.log(rel / tru))) < 0.03


def test_spot_fit_tie_keeps_spots_and_logt_recovers_p():
    from midas_nf_fitorientation.intensity import SplatBlurModel, fit_spot_intensity
    rng = np.random.default_rng(29)
    t = _table(rng, n_grains=6, rows_per_grain=150)
    s_true = np.exp(rng.normal(0, 0.4, t.n_grains)) * 10.0
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=5, sigma_psf=(0.8, 2.0), sigma_omega=0.15, p=2.0)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true)); mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); y, z = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, y, z, x[rec] - 5.0, NY, NZ)
    kw = dict(blanket=5.0, sigma_psf=(0.8, 2.0), sigma_omega=0.15, sigma_noise_init=2.0, p_init=1.0)
    full = fit_spot_intensity(t, R, NF, NY, NZ, **kw)
    one = fit_spot_intensity(t, R, NF, NY, NZ, tie_scales=True, **kw)
    assert one.n_spots == full.n_spots and np.array_equal(one.spot_ids, full.spot_ids)
    assert np.ptp(one.scales) == 0 and one.nll > full.nll
    lt = fit_spot_intensity(t, R, NF, NY, NZ, likelihood="logt", **kw)
    assert lt.p == pytest.approx(2.0, abs=0.1)


def test_splat_chunked_equals_unchunked():
    from midas_nf_fitorientation.intensity import SplatBlurModel
    rng = np.random.default_rng(31)
    t = _table(rng, n_grains=4, rows_per_grain=200)
    a = SplatBlurModel(t, NF, NY, NZ, blur_radius=3, sigma_psf=(0.9, 1.7), sigma_omega=0.2)
    b = SplatBlurModel(t, NF, NY, NZ, blur_radius=3, sigma_psf=(0.9, 1.7), sigma_omega=0.2); b.chunk_rows = 97
    ga = torch.autograd.grad(a().sum(), [a.p, a.log_sigma_omega])
    gb = torch.autograd.grad(b().sum(), [b.p, b.log_sigma_omega])
    with torch.no_grad():
        assert torch.allclose(a(), b())
    assert all(torch.allclose(x, y) for x, y in zip(ga, gb))


def test_column_unit_recovers_p_and_sees_position():
    """Profile (spot x column) likelihood: recovers p like the spot fit, and — unlike spot sums — penalises a
    row set shifted along y by 3 px (the thing a twin gap looks like)."""
    from midas_nf_fitorientation.intensity import SplatBlurModel, fit_spot_intensity
    rng = np.random.default_rng(37)
    t = _table(rng, n_grains=5, rows_per_grain=150)
    # make each spot an extended streak: add 8 companion rows per row along y (a "grain" projected along the streak)
    reps = 8
    ext = ContributionTable(np.repeat(t.grain, reps), np.repeat(t.refl, reps), np.repeat(t.sol, reps),
                            np.repeat(t.frame, reps), np.repeat(t.y, reps) + np.tile(np.arange(reps), t.grain.size),
                            np.repeat(t.z, reps), np.repeat(t.w_geom, reps), t.log_F, t.n_grains)
    s_true = np.exp(rng.normal(0, 0.3, t.n_grains)) * 6.0
    m = SplatBlurModel(ext, NF, NY, NZ, blur_radius=4, sigma_psf=(0.8, 1.5), sigma_omega=0.15, p=2.0)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true)); mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); y, z = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, y, z, x[rec] - 5.0, NY, NZ)
    kw = dict(blanket=5.0, sigma_psf=(0.8, 1.5), sigma_omega=0.15, sigma_noise_init=2.0, likelihood="logt")
    col = fit_spot_intensity(ext, R, NF, NY, NZ, unit="column", p_init=1.0, **kw)
    assert col.p == pytest.approx(2.0, abs=0.1)
    # shift grain 0's rows by +3 px in y: evaluate both likelihoods with the fitted parameters, nothing refitted
    sh = copy.copy(ext); sh.y = ext.y.copy(); sh.y[ext.grain == 0] += 3.0
    def nll(table, unit):
        return fit_spot_intensity(table, R, NF, NY, NZ, unit=unit, scales=col.scales, fit_p=False, p_init=col.p,
                                  kappa_init=max(col.kappa, 1e-3), fit_kappa=False, max_iter=0, **kw).spot_nll.sum()
    d_col = nll(sh, "column") - nll(ext, "column")
    d_spot = nll(sh, "spot") - nll(ext, "spot")
    assert d_col > 0 and d_col > 3 * max(d_spot, 1e-9)


def test_profile_fit_measures_sigma_z():
    from midas_nf_fitorientation.intensity import SplatBlurModel, fit_spot_intensity
    rng = np.random.default_rng(41)
    t = _table(rng, n_grains=5, rows_per_grain=200)
    s_true = np.exp(rng.normal(0, 0.3, t.n_grains)) * 6.0
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=5, sigma_psf=(0.8, 2.2), sigma_omega=0.15, p=2.0)
    with torch.no_grad():
        m.log_s[:] = torch.log(torch.tensor(s_true)); mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); y, z = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, y, z, x[rec] - 5.0, NY, NZ)
    fit = fit_spot_intensity(t, R, NF, NY, NZ, blanket=5.0, blur_radius=5, sigma_psf=(0.8, 1.2), sigma_omega=0.15,
                             sigma_noise_init=2.0, likelihood="logt", unit="column", fit_sigma_z=True, p_init=1.5)
    assert fit.sigma_psf_yz[0] == pytest.approx(0.8, abs=1e-6)          # sigma_y untouched
    assert fit.sigma_psf_yz[1] == pytest.approx(2.2, abs=0.35)
    assert fit.p == pytest.approx(2.0, abs=0.12)


def _mixture_scene(rng, S=12, K=3, n_refl=6, reps=4):
    """S sites x K candidate orientations, each candidate with its own spots (grain label site*K + k)."""
    g, r, fr, y, z = [], [], [], [], []
    for lab in range(S * K):
        for m in range(n_refl):
            y0 = rng.uniform(4, NY - 5 - reps)
            for j in range(reps):                       # a short streak along y
                g.append(lab); r.append(m); fr.append(rng.uniform(2, NF - 3)); y.append(y0 + j); z.append(rng.uniform(4, NZ - 5))
    g = np.array(g); R = g.size
    fr = np.array(fr).reshape(-1, reps)[:, :1].repeat(reps, 1).ravel()            # one frame per spot
    z = np.array(z).reshape(-1, reps)[:, :1].repeat(reps, 1).ravel()
    t = ContributionTable(grain=g, refl=np.array(r), sol=np.zeros(R, np.int8), frame=fr, y=np.array(y), z=z,
                          w_geom=np.ones(R), log_F=np.log(np.linspace(1.5, 3.0, n_refl)), n_grains=S * K)
    comp = g                                            # component = site*K + k = grain label
    return t, comp, np.arange(S * K) // K, np.arange(S * K) % K


def test_mixture_recovers_fractions_and_ignores_decoy():
    from midas_nf_fitorientation.intensity import Mixture, SplatBlurModel, fit_spot_intensity
    rng = np.random.default_rng(53)
    S, K = 12, 3
    t, comp, site, cand = _mixture_scene(rng, S, K)
    f_true = np.zeros((S, K)); f_true[:6, 0] = 1.0; f_true[6:, 0] = 0.3; f_true[6:, 1] = 0.7   # candidate 2 = decoy
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=4, sigma_psf=(0.8, 1.5), sigma_omega=0.15, p=2.0)
    with torch.no_grad():
        m.log_s[:] = math.log(8.0)
        m.w_geom *= torch.tensor(f_true[site[comp], cand[comp]])
        mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); yy, zz = np.divmod(rem, NZ)
    Rp = RecordedPixels.from_arrays(f, yy, zz, x[rec] - 5.0, NY, NZ)
    pairs = np.array([(i, i + 1) for i in range(S - 1) if i != 5])
    kw = dict(blanket=5.0, sigma_psf=(0.8, 1.5), sigma_omega=0.15, sigma_noise_init=2.0, likelihood="logt",
              unit="column", fit_p=False, p_init=2.0)
    for lam in (0.0, 0.01):
        mx = Mixture(row_comp=comp, comp_site=site, comp_cand=cand, n_cand=K,
                     scale_group=np.zeros(S * K, np.int64), pairs=pairs, tv_lambda=lam)
        fit = fit_spot_intensity(t, Rp, NF, NY, NZ, mixture=mx, **kw)
        assert fit.fractions.shape == (S, K)
        assert np.allclose(fit.fractions.sum(1), 1.0)
        assert np.abs(fit.fractions - f_true).mean() < 0.1
        assert fit.fractions[:, 2].mean() < 0.05                                   # the decoy stays empty
        assert fit.scales[0] == pytest.approx(8.0, rel=0.2)
        assert fit.tv is not None


def test_mixture_bounded_logits_recover_and_stay_bounded():
    from midas_nf_fitorientation.intensity import Mixture, SplatBlurModel, fit_spot_intensity
    rng = np.random.default_rng(59)
    S, K = 12, 3
    t, comp, site, cand = _mixture_scene(rng, S, K)
    f_true = np.zeros((S, K)); f_true[:6, 0] = 1.0; f_true[6:, 1] = 1.0
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=4, sigma_psf=(0.8, 1.5), sigma_omega=0.15, p=2.0)
    with torch.no_grad():
        m.log_s[:] = math.log(8.0); m.w_geom *= torch.tensor(f_true[site[comp], cand[comp]]); mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); yy, zz = np.divmod(rem, NZ)
    Rp = RecordedPixels.from_arrays(f, yy, zz, x[rec] - 5.0, NY, NZ)
    init = np.full((S, K), -3.0); init[:, 0] = 0.0               # start every site on candidate 0 (wrong for sites 6-11)
    mx = Mixture(row_comp=comp, comp_site=site, comp_cand=cand, n_cand=K, scale_group=np.zeros(S * K, np.int64),
                 init_logits=init, logit_bound=5.0)
    fit = fit_spot_intensity(t, Rp, NF, NY, NZ, mixture=mx, blanket=5.0, sigma_psf=(0.8, 1.5), sigma_omega=0.15,
                             sigma_noise_init=2.0, likelihood="logt", unit="column", fit_p=False, p_init=2.0,
                             fit_kappa=False, kappa_init=0.3, fit_sigma_noise=False)
    assert np.abs(fit.logits).max() <= 5.0 + 1e-9
    assert np.abs(fit.fractions - f_true).mean() < 0.1


def test_mixture_float32_model_saturated_start_no_overflow():
    """Phase 2 v2 crashed with a float32 L-BFGS overflow. A float32 model with sites started deep in saturation on the
    WRONG candidate (logit gap 25) must neither raise nor return NaN, and the wall must bound the logits. It also
    DOCUMENTS the trap: from saturation the gradient fit cannot switch a site (softmax gradient f(1-f) ~ 0), which is
    why a discrete search, not this optimiser, has to find absent twins."""
    from midas_nf_fitorientation.intensity import Mixture, SplatBlurModel, fit_spot_intensity
    rng = np.random.default_rng(61)
    S, K = 12, 3
    t, comp, site, cand = _mixture_scene(rng, S, K)
    f_true = np.zeros((S, K)); f_true[:6, 0] = 1.0; f_true[6:, 1] = 1.0
    m = SplatBlurModel(t, NF, NY, NZ, blur_radius=4, sigma_psf=(0.8, 1.5), sigma_omega=0.15, p=2.0)
    with torch.no_grad():
        m.log_s[:] = math.log(8.0); m.w_geom *= torch.tensor(f_true[site[comp], cand[comp]]); mu = m().numpy()
    x = mu + rng.normal(0, 2.0, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); yy, zz = np.divmod(rem, NZ)
    Rp = RecordedPixels.from_arrays(f, yy, zz, x[rec] - 5.0, NY, NZ)
    init = np.full((S, K), -25.0); init[:, 0] = 0.0
    mx = Mixture(row_comp=comp, comp_site=site, comp_cand=cand, n_cand=K, scale_group=np.zeros(S * K, np.int64),
                 init_logits=init)
    fit = fit_spot_intensity(t, Rp, NF, NY, NZ, mixture=mx, blanket=5.0, sigma_psf=(0.8, 1.5), sigma_omega=0.15,
                             sigma_noise_init=2.0, likelihood="logt", unit="column", fit_p=False, p_init=2.0,
                             fit_kappa=False, kappa_init=0.3, fit_sigma_noise=False, dtype=torch.float32)
    assert np.all(np.isfinite(fit.fractions))
    assert np.abs(fit.logits).max() <= 15.0 + 1.0                            # held by the soft wall (+ small overshoot)
    assert fit.fractions[6:, 1].max() < 1e-3                                 # the trap: wrongly started sites never switch
