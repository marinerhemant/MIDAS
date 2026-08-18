"""Detector realism: PSF, photon statistics, read noise."""
import math

import pytest
import torch

from midas_dct_tt.detector import (add_photon_noise, add_read_noise, apply_psf,
                                   gaussian_kernel_1d, simulate_detector)

DT = torch.float64


def _spot(nu=64, nv=64, cu=31.0, cv=27.0, w=2.0):
    u = torch.arange(nu, dtype=DT).view(-1, 1)
    v = torch.arange(nv, dtype=DT).view(1, -1)
    return torch.exp(-0.5 * (((u - cu) / w) ** 2 + ((v - cv) / w) ** 2))


def _centroid(img):
    nu, nv = img.shape[-2:]
    u = torch.arange(nu, dtype=img.dtype).view(-1, 1)
    v = torch.arange(nv, dtype=img.dtype).view(1, -1)
    m = img.sum()
    return float((img * u).sum() / m), float((img * v).sum() / m)


def test_kernel_is_normalised():
    for s in (0.5, 1.0, 3.7):
        assert float(gaussian_kernel_1d(s).sum()) == pytest.approx(1.0, abs=1e-12)


def test_psf_conserves_flux_away_from_edges():
    img = _spot()
    assert float(apply_psf(img, 1.5).sum()) == pytest.approx(float(img.sum()), rel=1e-10)


def test_psf_preserves_centroid_in_the_interior():
    """A symmetric PSF must not move an INTERIOR spot."""
    img = _spot()
    c0 = _centroid(img)
    c1 = _centroid(apply_psf(img, 2.0))
    assert c1[0] == pytest.approx(c0[0], abs=1e-9)
    assert c1[1] == pytest.approx(c0[1], abs=1e-9)


def test_psf_moves_the_centroid_near_an_edge():
    """...and near an edge it DOES move it, by enough to matter.

    Zero padding truncates the kernel asymmetrically. At sigma=2 (radius 8) a spot
    4 px from the edge shifts ~0.29 px -- 43% of the 0.66 um centroid separation
    the B2 result rests on. Pinned so the interior test is never mistaken for a
    general guarantee.
    """
    shifts = []
    for cu in (31.0, 8.0, 4.0):
        img = _spot(cu=cu)
        shifts.append(abs(_centroid(apply_psf(img, 2.0))[0] - _centroid(img)[0]))
    assert shifts[0] < 1e-9
    assert shifts[2] > 0.1, f"edge bias should be large, got {shifts[2]:.3e}"
    assert shifts[0] < shifts[1] < shifts[2]


def test_psf_broadens_and_lowers_the_peak():
    img = _spot()
    b = apply_psf(img, 2.0)
    assert float(b.max()) < float(img.max())
    sec = ((b > 0.5 * b.max()).sum(), (img > 0.5 * img.max()).sum())
    assert sec[0] > sec[1]


def test_psf_zero_sigma_is_identity():
    img = _spot()
    assert torch.equal(apply_psf(img, 0.0), img)


def test_psf_rejects_negative_sigma():
    with pytest.raises(ValueError, match="sigma_px must be > 0"):
        gaussian_kernel_1d(-1.0)


def test_psf_is_differentiable():
    img = _spot().requires_grad_(True)
    apply_psf(img, 1.5).sum().backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


def test_psf_handles_batch_dims():
    stack = torch.stack([_spot(), _spot(cu=10.0)])
    out = apply_psf(stack, 1.0)
    assert out.shape == stack.shape
    assert torch.allclose(out[0], apply_psf(stack[0], 1.0))


def test_psf_loses_flux_off_the_edge():
    img = torch.zeros(32, 32, dtype=DT)
    img[0, 0] = 1.0
    assert float(apply_psf(img, 2.0).sum()) < 0.99


def test_photon_noise_has_the_right_relative_scale():
    """Relative noise must be 1/sqrt(photons) -- the number worth quoting."""
    img = torch.ones(200, 200, dtype=DT)
    g = torch.Generator().manual_seed(0)
    for ppu in (1e2, 1e4):
        n = add_photon_noise(img, ppu, generator=g)
        assert float(n.mean()) == pytest.approx(1.0, rel=0.02)
        assert float(n.std()) == pytest.approx(1.0 / math.sqrt(ppu), rel=0.1)


def test_photon_noise_is_reproducible_and_rejects_bad_input():
    img = _spot()
    a = add_photon_noise(img, 1e3, generator=torch.Generator().manual_seed(3))
    b = add_photon_noise(img, 1e3, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="photons_per_unit must be > 0"):
        add_photon_noise(img, 0.0)


def test_read_noise_scale_and_identity():
    img = torch.zeros(300, 300, dtype=DT)
    g = torch.Generator().manual_seed(1)
    assert float(add_read_noise(img, 0.25, generator=g).std()) == pytest.approx(0.25, rel=0.05)
    assert torch.equal(add_read_noise(img, 0.0), img)
    with pytest.raises(ValueError, match="sigma must be >= 0"):
        add_read_noise(img, -1.0)


def _lag1_autocorr(r):
    a = r[:, :-1].reshape(-1)
    b = r[:, 1:].reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def test_simulate_detector_applies_psf_after_poisson():
    """Order is physical, and the signature is the noise CORRELATION structure.

    Poisson-then-PSF (correct) gives spatially correlated noise; PSF-then-Poisson
    gives white noise. A least-squares fit assumes independent residuals, so
    getting this backwards misstates exactly the thing the fit relies on.
    """
    img = torch.ones(128, 128, dtype=DT) * 10.0
    g = torch.Generator().manual_seed(7)
    correct = simulate_detector(img, psf_px=1.5, photons_per_unit=50.0, generator=g)
    g = torch.Generator().manual_seed(7)
    wrong = add_photon_noise(apply_psf(img, 1.5), 50.0, generator=g)
    ref = apply_psf(img, 1.5)
    ac_c = _lag1_autocorr(correct - ref)
    ac_w = _lag1_autocorr(wrong - ref)
    # Bound, not a value: this single-Poisson model is the m-bar -> inf limit.
    # A real indirect detector (m-bar ~ 10-70 p.e./x-ray) sits near 0.5, so the
    # assertion is that the noise is CORRELATED, not that it reaches 0.9.
    assert ac_c > 0.3, f"Poisson-then-PSF noise must be correlated, got {ac_c:.3f}"
    assert abs(ac_w) < 0.1, f"PSF-then-Poisson noise must be white, got {ac_w:.3f}"


def test_blur_reduces_per_pixel_noise_because_it_averages_photons():
    """Real physics, not an artefact: the correct order has lower per-pixel std."""
    img = torch.ones(128, 128, dtype=DT) * 10.0
    g = torch.Generator().manual_seed(11)
    blurred = simulate_detector(img, psf_px=1.5, photons_per_unit=50.0, generator=g)
    g = torch.Generator().manual_seed(11)
    unblurred = add_photon_noise(img, 50.0, generator=g)
    assert float((blurred - apply_psf(img, 1.5)).std()) < float((unblurred - img).std())


def test_simulate_detector_noiseless_passthrough():
    img = _spot()
    assert torch.equal(simulate_detector(img), img)


# --- agreement with midas_dfxm.detector ------------------------------------
def test_psf_agrees_with_midas_dfxm_in_the_interior():
    """Pin the two implementations together where both apply, so they cannot drift.

    They differ only by kernel radius (3 sigma there, 4 sigma here). Away from the
    edges that is a ~1e-3 effect on a normalised spot.
    """
    from midas_dfxm.detector import apply_psf as dfxm_psf
    img = _spot()
    for s in (1.0, 2.0):
        a, b = dfxm_psf(img, s), apply_psf(img, s)
        assert float((a - b).abs().max()) < 3e-3
        assert float(a.sum()) == pytest.approx(float(b.sum()), rel=1e-9)


def test_padding_choice_is_deliberate_and_differs_from_dfxm():
    """midas_dfxm reflect-pads (creates flux at an edge); this zero-pads (loses it).

    Zero padding is the physical one for a detector, and it matters for TT because
    the spot moves during a psi scan.
    """
    from midas_dfxm.detector import apply_psf as dfxm_psf
    edge = _spot(cu=4.0)
    kept_dfxm = float(dfxm_psf(edge, 2.0).sum() / edge.sum())
    kept_tt = float(apply_psf(edge, 2.0).sum() / edge.sum())
    assert kept_dfxm > 1.0, "reflect padding should over-conserve at an edge"
    assert kept_tt < 1.0, "zero padding should lose flux at an edge"


def test_dfxm_psf_cannot_batch_which_is_why_this_module_exists():
    from midas_dfxm.detector import apply_psf as dfxm_psf
    stack = torch.stack([_spot(), _spot(cu=20.0)])
    with pytest.raises(NotImplementedError):
        dfxm_psf(stack, 1.0)
    assert apply_psf(stack, 1.0).shape == stack.shape


# --- two-stage (indirect detection) noise ----------------------------------
def _C_coeffs(sigma):
    from midas_dct_tt.detector import gaussian_kernel_1d
    k = gaussian_kernel_1d(sigma, dtype=DT)
    c0 = float((k * k).sum()); c1 = float((k[:-1] * k[1:]).sum())
    return c0 * c0, c1 * c0          # separable 2-D


def test_two_stage_matches_the_analytic_correlation():
    """rho(1) = C(1)/(1/m_bar + C(0)) -- the closed form for a Poisson cluster
    process. Pinned across three decades of light yield."""
    from midas_dct_tt.detector import add_two_stage_noise
    sig = 1.5
    C0, C1 = _C_coeffs(sig)
    img = torch.ones(220, 220, dtype=DT)
    for mb in (5.0, 30.0, 100.0, 1000.0):
        g = torch.Generator().manual_seed(4)
        s = add_two_stage_noise(img, 500.0, mb, sig, generator=g)
        r = (s - apply_psf(img, sig))[30:-30, 30:-30]
        got = _lag1_autocorr(r)
        assert got == pytest.approx(C1 / (1.0 / mb + C0), abs=0.03), f"m_bar={mb}"


def test_single_stage_is_the_infinite_light_yield_limit():
    """The existing model is optimistic: a real detector at m_bar ~ 30 has about
    half the correlation and twice the variance."""
    from midas_dct_tt.detector import add_two_stage_noise
    sig, img = 1.5, torch.ones(220, 220, dtype=DT)
    C0, _ = _C_coeffs(sig)
    ref = apply_psf(img, sig)
    var = {}
    for mb in (30.0, 1000.0):
        g = torch.Generator().manual_seed(9)
        r = (add_two_stage_noise(img, 500.0, mb, sig, generator=g) - ref)[30:-30, 30:-30]
        var[mb] = float(r.var())
    # In returned units Var is proportional to (1/m_bar + C0), so the ratio is
    # fixed by the closed form -- assert that, not a guessed threshold.
    predicted = (1.0 / 30.0 + C0) / (1.0 / 1000.0 + C0)
    assert var[30.0] / var[1000.0] == pytest.approx(predicted, rel=0.10)
    assert predicted > 1.8, "a real detector should carry ~2x the variance"

    g = torch.Generator().manual_seed(9)
    one = simulate_detector(img, psf_px=sig, photons_per_unit=500.0 * 30.0, generator=g)
    assert _lag1_autocorr(one - ref) > _lag1_autocorr(
        add_two_stage_noise(img, 500.0, 30.0, sig,
                            generator=torch.Generator().manual_seed(9)) - ref)


def test_two_stage_is_unbiased():
    from midas_dct_tt.detector import add_two_stage_noise
    img = torch.ones(200, 200, dtype=DT) * 2.0
    g = torch.Generator().manual_seed(11)
    s = add_two_stage_noise(img, 400.0, 40.0, 1.0, generator=g)
    assert float(s[20:-20, 20:-20].mean()) == pytest.approx(2.0, rel=0.01)


def test_simulate_detector_routes_to_two_stage_and_validates_input():
    img = torch.ones(64, 64, dtype=DT)
    with pytest.raises(ValueError, match="needs photons_per_unit"):
        simulate_detector(img, photoelectrons_per_xray=30.0)
    g = torch.Generator().manual_seed(2)
    out = simulate_detector(img, psf_px=1.0, photons_per_unit=300.0,
                            photoelectrons_per_xray=30.0, generator=g)
    assert out.shape == img.shape and torch.isfinite(out).all()


def test_two_stage_rejects_nonpositive_rates():
    from midas_dct_tt.detector import add_two_stage_noise
    img = torch.ones(16, 16, dtype=DT)
    for a, b in ((0.0, 30.0), (300.0, 0.0)):
        with pytest.raises(ValueError, match="must be > 0"):
            add_two_stage_noise(img, a, b, 1.0)


# --- effective degrees of freedom under PSF correlation --------------------
def test_effective_dof_is_one_without_a_psf():
    from midas_dct_tt import effective_dof
    for mode in ("chi2", "mean"):
        assert effective_dof((64, 64), 0.0, mode=mode) == pytest.approx(1.0, abs=1e-12)


def test_effective_dof_falls_with_psf_width():
    from midas_dct_tt import effective_dof
    prev = 1.1
    for sig in (0.5, 0.8, 1.5, 2.5):
        v = effective_dof((120, 120), sig, photoelectrons_per_xray=30.0, mode="mean")
        assert 0.0 < v < prev
        prev = v


def test_the_two_modes_differ_and_mean_is_harsher():
    """Conflating them is worth ~3x. The k=0 mode is where correlated noise piles
    up, so a global parameter suffers far more than a goodness-of-fit sum."""
    from midas_dct_tt import effective_dof
    kw = dict(photoelectrons_per_xray=30.0)
    c = effective_dof((120, 120), 1.5, mode="chi2", **kw)
    m = effective_dof((120, 120), 1.5, mode="mean", **kw)
    assert c == pytest.approx(0.2231, rel=0.02)
    assert m == pytest.approx(0.0665, rel=0.02)
    assert m < c / 3.0


def test_effective_dof_matches_direct_simulation():
    """400 realisations vs the spectral prediction."""
    import statistics as st

    from midas_dct_tt import effective_dof
    from midas_dct_tt.detector import add_two_stage_noise
    img = torch.ones(160, 160, dtype=DT)
    ref = apply_psf(img, 1.5)
    means, last = [], None
    for s in range(400):
        g = torch.Generator().manual_seed(s)
        d = (add_two_stage_noise(img, 500.0, 30.0, 1.5, generator=g) - ref)[20:-20, 20:-20]
        means.append(float(d.mean())); last = d
    emp = st.pvariance([float(x) for x in last.reshape(-1)]) / (
        st.pvariance(means) * last.numel())
    pred = effective_dof(tuple(last.shape), 1.5, photoelectrons_per_xray=30.0, mode="mean")
    assert emp == pytest.approx(pred, rel=0.20), f"empirical {emp:.4f} vs predicted {pred:.4f}"


def test_effective_dof_rejects_unknown_mode():
    from midas_dct_tt import effective_dof
    with pytest.raises(ValueError, match="mode must be"):
        effective_dof((32, 32), 1.0, mode="rms")
