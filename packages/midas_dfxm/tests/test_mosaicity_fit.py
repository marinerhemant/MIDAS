"""Physics-forward orientation+mosaicity fitting: deconvolution, regularization, sub-grains."""
import numpy as np
import pytest
import torch

from midas_dfxm.mosaicity_fit import fit_orientation_mosaicity, moment_orientation

DT = torch.float64


def _synth(P=150, m=41, sig_m=0.12, sig_r=0.15, noise=400, seed=0, ori_amp=0.3):
    rng = np.random.default_rng(seed)
    ax = np.linspace(-1, 1, m)
    CH, PH = np.meshgrid(ax, ax, indexing="ij")
    chi = CH.reshape(-1); phi = PH.reshape(-1)
    c0 = rng.uniform(-ori_amp, ori_amp, P); p0 = rng.uniform(-ori_amp, ori_amp, P)
    sig_tot = np.sqrt(sig_m**2 + sig_r**2)
    d = np.exp(-0.5 * ((chi[None]-c0[:, None])**2 + (phi[None]-p0[:, None])**2) / sig_tot**2)
    d = rng.poisson(np.clip(d*noise, 0, None)) / noise
    return (torch.tensor(d, dtype=DT), torch.tensor(chi, dtype=DT), torch.tensor(phi, dtype=DT),
            np.stack([c0, p0], -1), sig_m, sig_r)


@pytest.mark.unit
def test_recovers_orientation_and_deconvolved_mosaic():
    data, chi, phi, truth, sig_m, sig_r = _synth()
    res = np.eye(2) * sig_r**2
    out = fit_orientation_mosaicity(data, chi, phi, res, n_components=1, steps=500, lr=0.03)
    ori = out["orientation"][:, 0, :].numpy()
    mos = out["mosaic_cov"][:, 0].numpy()
    mos_sig = np.sqrt(0.5 * (mos[:, 0, 0] + mos[:, 1, 1]))
    # orientation recovered
    assert np.abs(ori - truth).mean() < 0.02
    # INTRINSIC mosaic recovered (deconvolved) -- close to true sig_m, NOT the convolved total
    assert abs(mos_sig.mean() - sig_m) < 0.02
    assert mos_sig.mean() < np.sqrt(sig_m**2 + sig_r**2) - 0.02  # genuinely deconvolved


@pytest.mark.unit
def test_beats_moment_on_intrinsic_mosaic():
    data, chi, phi, truth, sig_m, sig_r = _synth()
    res = np.eye(2) * sig_r**2
    out = fit_orientation_mosaicity(data, chi, phi, res, steps=500, lr=0.03)
    mos = out["mosaic_cov"][:, 0].numpy()
    ours = np.sqrt(0.5 * (mos[:, 0, 0] + mos[:, 1, 1])).mean()
    # moment second-moment width (convolved) overestimates the intrinsic mosaic
    mom = out["moment"]
    w = data.clamp_min(0); ws = w.sum(-1)
    var = ((w*(chi[None]-mom[:, 0:1])**2).sum(-1)/ws + (w*(phi[None]-mom[:, 1:2])**2).sum(-1)/ws)/2
    moment_sig = float(var.sqrt().mean())
    assert abs(ours - sig_m) < abs(moment_sig - sig_m)          # ours closer to truth
    assert moment_sig > sig_m + 0.03                            # moment clearly overestimates


@pytest.mark.slow
def test_regularization_smooths_the_field():
    # Spatial smoothness is an available global option; verify it DOES what it claims
    # (produces a smoother orientation field). Honest note: for orientation the per-pixel
    # peak position is already a robust estimator, so regularisation's accuracy benefit
    # is modest/regime-dependent (unlike the ill-posed strain-tensor inverse).
    m, ny, nx = 25, 10, 10
    ax = np.linspace(-1, 1, m); CH, PH = np.meshgrid(ax, ax, indexing="ij")
    chi = torch.tensor(CH.reshape(-1), dtype=DT); phi = torch.tensor(PH.reshape(-1), dtype=DT)
    c0 = np.repeat(np.linspace(-0.3, 0.3, nx)[None, :], ny, 0).reshape(-1)
    rng = np.random.default_rng(0)
    d = np.exp(-0.5*((CH.reshape(-1)[None]-c0[:, None])**2 + (PH.reshape(-1)[None])**2)/0.18**2)
    d = rng.poisson(np.clip(d*8, 0, None))/8.0
    data = torch.tensor(d, dtype=DT); res = np.eye(2)*0.10**2
    per = fit_orientation_mosaicity(data, chi, phi, res, shape=(ny, nx), lambda_smooth=0.0, steps=350)
    reg = fit_orientation_mosaicity(data, chi, phi, res, shape=(ny, nx), lambda_smooth=300.0, steps=350)
    def tv(o): return np.abs(np.diff(o[:, 0, 0].numpy().reshape(ny, nx), axis=0)).mean()
    assert tv(reg["orientation"]) < tv(per["orientation"])     # regularised field is smoother


@pytest.mark.unit
def test_two_component_resolves_subgrains():
    rng = np.random.default_rng(2); m = 41; ax = np.linspace(-1.5, 1.5, m)
    CH, PH = np.meshgrid(ax, ax, indexing="ij"); chi = torch.tensor(CH.reshape(-1), dtype=DT); phi = torch.tensor(PH.reshape(-1), dtype=DT)
    P = 60; c1 = rng.uniform(-0.2, 0.2, P); sep = 0.7
    d = (np.exp(-0.5*((CH.reshape(-1)[None]-c1[:, None])**2+(PH.reshape(-1)[None])**2)/0.12**2)
         + 0.6*np.exp(-0.5*((CH.reshape(-1)[None]-(c1+sep)[:, None])**2+(PH.reshape(-1)[None])**2)/0.12**2))
    d = rng.poisson(np.clip(d*400, 0, None))/400.0
    out = fit_orientation_mosaicity(torch.tensor(d, dtype=DT), chi, phi, np.eye(2)*0.05**2,
                                    n_components=2, steps=600, lr=0.03)
    ori = out["orientation"].numpy()  # (P,2,2)
    recovered_sep = np.abs(ori[:, 0, 0] - ori[:, 1, 0])
    assert np.median(recovered_sep) == pytest.approx(sep, abs=0.2)  # resolves the two components


@pytest.mark.unit
def test_offcenter_peak_not_driven_offgrid():
    """Regression: a peak far from the scan centre must be located there, not driven off the
    grid. The old init (fixed fat width + unbounded centre) put a clean Gaussian at (0.20,0.20)
    at (0.265,0.530) -- phi off the +/-0.4 grid entirely -- and inflated the mosaic width."""
    n = 17
    ax = torch.linspace(-0.4, 0.4, n, dtype=DT)
    CH, PH = torch.meshgrid(ax, ax, indexing="ij")
    chi = CH.reshape(-1); phi = PH.reshape(-1)
    res = np.diag([0.02 ** 2, 0.02 ** 2])
    for (c0, p0, w) in [(0.20, 0.20, 0.08), (0.35, 0.35, 0.06), (-0.30, 0.10, 0.07)]:
        d = torch.exp(-0.5 * (((chi - c0) / w) ** 2 + ((phi - p0) / w) ** 2))[None, :].repeat(2, 1)
        out = fit_orientation_mosaicity(d, chi, phi, res, steps=400)
        o = out["orientation"][0, 0]
        assert abs(float(o[0]) - c0) < 0.02 and abs(float(o[1]) - p0) < 0.02, \
            f"peak ({c0},{p0}) recovered at ({float(o[0]):.3f},{float(o[1]):.3f}) -- driven off-grid"
        assert float(ax.min()) <= float(o[0]) <= float(ax.max())
        assert float(ax.min()) <= float(o[1]) <= float(ax.max())
        # width must not balloon (intrinsic ~ w after deconvolution, not ~0.5 deg)
        fwhm = 2.3548 * float(torch.sqrt(torch.diagonal(out["mosaic_cov"][0, 0]).clamp_min(0)).mean())
        assert fwhm < 0.3, f"mosaic FWHM {fwhm:.2f} deg inflated (center was mislocated)"
