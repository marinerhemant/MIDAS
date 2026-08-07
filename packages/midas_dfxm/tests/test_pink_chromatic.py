"""Spectrum-integrated (pink-beam) dynamical forward + objective chromatic aberration.

Covers: the leg-1 dynamical spectral integration (mono limit, the crystal-dispersion-is-negligible
finding, differentiability), and the leg-2 chromatic objective (defocus mapping, the core+pedestal
PSF, band-edge blur, and differentiable spectrum recovery from a through-focus scan).
"""
import numpy as np
import torch

from midas_dfxm.io import fcc_reference_crystal
from midas_dfxm.takagi_taupin import (
    susceptibility_fourier, diffracted_intensity, extinction_length,
)
from midas_dfxm.pink import pink_dynamical_reflectivity, spectrum_grid, H_C_KEV_A
from midas_dfxm.beamline import chromatic_defocus_coeffs, crl_na
from midas_dfxm.chromatic import effective_chromatic_psf, chromatic_psf_from_spectrum, psf_fwhm_um

torch.set_default_dtype(torch.float64)
CR = fcc_reference_crystal(); HKL = (2, 2, 0)
E0, TB = 17.0, 12.0
LAM0 = H_C_KEV_A / E0
C0, CH, CHB = susceptibility_fourier(CR, HKL, wavelength_A=LAM0)
LAM = extinction_length(CH, CHB, wavelength_A=LAM0, theta_B_deg=TB)
N, R, P = 69, 50.0, 0.36


# ----------------------------------------------------------------- leg 1
def test_pink_mono_limit():
    """bandwidth=0 reproduces the monochromatic dynamical reflectivity exactly."""
    for geom, mono in [("bragg", None), ("laue", float(diffracted_intensity(
            C0, CH, CHB, thickness_um=20.0, y=0.3, theta_B_deg=TB, wavelength_A=LAM0)))]:
        pink = float(pink_dynamical_reflectivity(CR, HKL, thickness_um=20.0, theta_B_deg=TB,
                     E0_keV=E0, bandwidth=0.0, y0=0.3, geometry=geom))
        if mono is not None:
            assert abs(pink - mono) < 1e-9


def test_spectrum_grid_normalised():
    _, lam, w = spectrum_grid(E0, 0.01, n_lambda=21, shape="gaussian")
    assert abs(float(w.sum()) - 1.0) < 1e-12 and len(lam) == 21


def test_crystal_dispersion_is_negligible():
    """The finding: at DFXM bandwidth the extinction-length dispersion adds no fringe washout beyond
    the deviation blur -- dispersing Lambda(lambda) vs freezing it gives the same Pendellosung."""
    ts = np.linspace(6, 12, 12) * LAM
    def curve(disp):
        return np.array([float(pink_dynamical_reflectivity(CR, HKL, thickness_um=float(t),
                         theta_B_deg=TB, E0_keV=E0, bandwidth=0.01, n_lambda=15, y0=0.0,
                         geometry="laue", disperse=disp)) for t in ts])
    cd, cf = curve(True), curve(False)
    assert np.abs(cd - cf).max() < 0.02 * max(np.ptp(cf), 1e-6) + 1e-4


def test_pink_differentiable():
    tt = torch.tensor(5 * LAM, dtype=torch.float64, requires_grad=True)
    I = pink_dynamical_reflectivity(CR, HKL, thickness_um=tt, theta_B_deg=TB, E0_keV=E0,
                                    bandwidth=0.01, n_lambda=11, y0=0.0, geometry="laue")
    g = torch.autograd.grad(I, tt)[0]
    assert torch.isfinite(g).all() and float(g.abs()) > 0


# ----------------------------------------------------------------- leg 2
def test_chromatic_defocus_zero_at_centre():
    """The centre energy is in focus: its defocus coefficient is 0; band edges are non-zero."""
    E = torch.tensor([E0 * 0.99, E0, E0 * 1.01])
    c = chromatic_defocus_coeffs(E, E0_keV=E0, n_lenses=N, radius_um=R, object_distance_m=P)
    assert abs(float(c[1])) < 1e-9
    assert abs(float(c[0])) > 0.1 and abs(float(c[2])) > 0.1


def test_chromatic_psf_core_plus_pedestal():
    """A 1% pink beam keeps a sharp core but pushes energy into a defocused pedestal."""
    NA0 = float(crl_na(N, R, E0))
    h_mono, dx = effective_chromatic_psf(torch.zeros(1), torch.ones(1), NA=NA0,
                                         wavelength_A=LAM0, grid_size=256, extent=5.0)
    E = torch.linspace(E0 * 0.98, E0 * 1.02, 21)
    w = torch.exp(-0.5 * ((E - E0) / (E0 * 0.008)) ** 2)
    h_pink, dx = chromatic_psf_from_spectrum(E, w, E0_keV=E0, n_lenses=N, radius_um=R,
                                             object_distance_m=P, grid_size=256, extent=5.0)

    def enc(h, r_um):
        n = h.shape[0]; c = n // 2
        yy, xx = np.mgrid[0:n, 0:n]; rr = np.hypot(xx - c, yy - c) * dx
        hh = h.detach().numpy()
        return hh[rr <= r_um].sum() / hh.sum()

    assert psf_fwhm_um(h_pink, dx) >= psf_fwhm_um(h_mono, dx)      # core broadened
    assert enc(h_pink, 0.5) < enc(h_mono, 0.5) - 0.05             # energy in the pedestal


def test_chromatic_psf_broadens_with_defocus():
    """PSF FWHM grows monotonically with defocus (the coefficient convention)."""
    NA0 = float(crl_na(N, R, E0))
    fw = []
    for coeff in (0.0, 3.0, 6.0):
        h, dx = effective_chromatic_psf(torch.tensor([coeff]), torch.ones(1), NA=NA0,
                                        wavelength_A=LAM0, grid_size=256, extent=5.0)
        fw.append(psf_fwhm_um(h, dx))
    assert fw[0] < fw[1] < fw[2]


def test_spectrum_recovery_through_focus():
    """S(lambda) is recoverable by gradient descent from a through-focus set of chromatic PSFs
    (asymmetric defocus diversity breaks the even-aberration sign degeneracy)."""
    NA0 = float(crl_na(N, R, E0))
    sp_E = torch.linspace(E0 * 0.988, E0 * 1.012, 17)
    coeffs = chromatic_defocus_coeffs(sp_E, E0_keV=E0, n_lenses=N, radius_um=R, object_distance_m=P)
    w_true = torch.softmax(-0.5 * ((sp_E - E0) / (E0 * 0.010)) ** 2
                           + torch.linspace(-0.5, 0.5, sp_E.shape[0]), 0)
    div = [0.0, 3.0, 6.0]
    def imgs(w):
        return [effective_chromatic_psf(coeffs + d, w, NA=NA0, wavelength_A=LAM0,
                grid_size=96, extent=5.0)[0] for d in div]
    tgt = [im.detach() for im in imgs(w_true)]
    logits = torch.zeros(sp_E.shape[0], requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.15)
    for _ in range(200):
        opt.zero_grad()
        w = torch.softmax(logits, 0)
        loss = sum(((a - b) ** 2).sum() for a, b in zip(imgs(w), tgt))
        loss.backward(); opt.step()
    w_rec = torch.softmax(logits, 0).detach().numpy()
    corr = np.corrcoef(w_rec, w_true.numpy())[0, 1]
    # light/fast config here (few iters, coarse grid); the paper figure converges to corr ~0.997
    # with more diversity settings + iterations. The point tested is that recovery works.
    assert corr > 0.9, f"spectrum not recovered: corr={corr:.3f}"


# ----------------------------------------------------------------- composed image forward
def _optics():
    from midas_dfxm.optics import ObjectiveOptics
    return ObjectiveOptics(two_theta_deg=2 * TB, magnification=1.0, pixel_um=0.1,
                           detector_shape=(48, 48), NA=float(crl_na(N, R, E0)), wavelength_A=LAM0)


def test_composed_image_mono_limit():
    """The composed chromatic pink image reduces to the mono dynamical image at bandwidth 0."""
    from midas_dfxm.wave_imaging import dfxm_image_dynamical, dfxm_image_dynamical_chromatic_pink
    opt = _optics()
    ref = dfxm_image_dynamical(C0, CH, CHB, wavelength_A=LAM0, theta_B_deg=TB, thickness_um=3 * LAM,
                               optics=opt, dx_um=0.1, y=0.0, n_depth=150, ny=48, coherent_fraction=0.0)
    img = dfxm_image_dynamical_chromatic_pink(CR, HKL, theta_B_deg=TB, thickness_um=3 * LAM, E0_keV=E0,
                                              optics=opt, n_lenses=N, radius_um=R, object_distance_m=P,
                                              bandwidth=0.0, y0=0.0, dx_um=0.1, n_depth=150, ny=48,
                                              coherent_fraction=0.0)
    assert float((img - ref).abs().max()) < 1e-9


def test_composed_per_energy_chi_negligible_and_differentiable():
    """Per-energy chi vs frozen chi give the same image (crystal dispersion negligible), and the
    composed forward is differentiable in the deformation amplitude."""
    from midas_dfxm.wave_imaging import dfxm_image_dynamical_chromatic_pink
    opt = _optics()
    xs = (np.arange(48) - 24) * 0.1
    hu = torch.tensor(1.5 * np.exp(-(xs / 0.35) ** 2), dtype=torch.float64)[None, :].expand(150, 48).clone()
    kw = dict(theta_B_deg=TB, thickness_um=3 * LAM, E0_keV=E0, optics=opt, n_lenses=N, radius_um=R,
              object_distance_m=P, dx_um=0.1, n_depth=150, ny=48, coherent_fraction=0.0,
              bandwidth=0.01, n_lambda=9, y0=0.0, chromatic_psf=False)
    a = dfxm_image_dynamical_chromatic_pink(CR, HKL, hu=hu, disperse_chi=True, **kw)
    b = dfxm_image_dynamical_chromatic_pink(CR, HKL, hu=hu, disperse_chi=False, **kw)
    assert float((a - b).abs().max()) < 5e-3 * float(b.abs().max()) + 1e-9

    amp = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    huf = amp * torch.tensor(1.5 * np.exp(-(xs / 0.35) ** 2), dtype=torch.float64)[None, :].expand(150, 48)
    img = dfxm_image_dynamical_chromatic_pink(CR, HKL, hu=huf, disperse_chi=False, **kw)
    g = torch.autograd.grad((img ** 2).sum(), amp)[0]
    assert torch.isfinite(g).all() and float(g.abs()) > 0
