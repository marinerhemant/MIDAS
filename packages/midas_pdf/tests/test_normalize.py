import torch

from midas_hkls import form_factor
from midas_integrate_v2.pdf import normalize_to_S

from midas_pdf import Composition, faber_ziman_S


def _q():
    return torch.linspace(0.5, 22.0, 200, dtype=torch.float64)


def test_monoatomic_reduces_to_normalize_to_S():
    """Single-element FZ must equal the existing monoatomic normalize_to_S."""
    q = _q()
    s2 = (q / (4.0 * torch.pi)) ** 2
    f = form_factor(s2, "Ni")
    f2 = f * f
    # Some synthetic measured intensity and Compton.
    I = f2 * (1.0 + 0.2 * torch.sin(2.5 * q))
    comp_term = 3.0 * (1.0 - torch.exp(-1.4 * q))  # arbitrary precomputed Compton

    S_ref, sig_ref = normalize_to_S(
        I, q=q, atomic_form_factor_squared=f2,
        compton=comp_term, sigma_intensity=torch.sqrt(I.abs()),
    )
    comp = Composition({"Ni": 1})
    S_fz, sig_fz = faber_ziman_S(
        I, q, comp, wavelength_A=0.1665, scale=1.0,
        compton=comp_term, sigma_intensity=torch.sqrt(I.abs()),
    )
    assert torch.allclose(S_fz, S_ref, atol=1e-10)
    assert torch.allclose(sig_fz, sig_ref, atol=1e-10)


def test_sigma_scales_with_intensity_sigma():
    q = _q()
    comp = Composition({"Si": 1, "O": 2})
    I = torch.ones_like(q) * 100.0
    sig = torch.sqrt(I)
    _, sigma_S_1 = faber_ziman_S(I, q, comp, wavelength_A=0.1665,
                                 sigma_intensity=sig)
    _, sigma_S_2 = faber_ziman_S(I, q, comp, wavelength_A=0.1665,
                                 sigma_intensity=2.0 * sig)
    assert torch.allclose(sigma_S_2, 2.0 * sigma_S_1, atol=1e-12)


def test_no_sigma_gives_zero():
    q = _q()
    comp = Composition({"Si": 1, "O": 2})
    I = torch.ones_like(q) * 50.0
    _, sigma_S = faber_ziman_S(I, q, comp, wavelength_A=0.1665)
    assert torch.allclose(sigma_S, torch.zeros_like(sigma_S))


def test_gradcheck_scale_and_fractions():
    q = torch.linspace(1.0, 12.0, 25, dtype=torch.float64)
    comp = Composition({"Si": 1, "O": 2})
    I = torch.linspace(10.0, 200.0, 25, dtype=torch.float64)

    def fn(scale, frac):
        S, _ = faber_ziman_S(I, q, comp, wavelength_A=0.1665,
                             scale=scale, compton=True, fractions=frac)
        return S

    scale = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    frac = torch.tensor([0.34, 0.66], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (scale, frac), atol=1e-6)
