import torch

from midas_pdf import Composition, delta_pdf, i_of_q_to_Gr, significant_mask


def _synthetic_iq(comp, q, *, r0=1.62, shell=4.0, broadening=0.05):
    """Single-coordination-shell model coherent intensity for a test sample.

    I_coh(Q) = ⟨f²⟩ · S_model(Q),  S_model = 1 + (sin(Q r0)/(Q r0)) e^{-Q²σ²/2}
    (a clean nearest-neighbour shell at r0) plus the Compton background.
    """
    f_avg, f2_avg = comp.form_factor_averages(q)
    debye = torch.sin(q * r0) / (q * r0) * torch.exp(-0.5 * (q * broadening) ** 2)
    S_model = 1.0 + shell * debye
    I_coh = f2_avg * S_model
    cmp = comp.compton(q, wavelength_A=0.1665)
    return I_coh + cmp


def test_pipeline_runs_end_to_end():
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.7, 22.0, 1500, dtype=torch.float64)
    r = torch.linspace(0.0, 10.0, 800, dtype=torch.float64)
    I = _synthetic_iq(comp, q)
    sigma_I = torch.sqrt(I.clamp(min=1.0))

    G, sigma_G, S = i_of_q_to_Gr(
        q, I, comp, r, wavelength_A=0.1665, sigma_intensity=sigma_I,
        compton=True, q_max=20.0,
    )
    assert G.shape == r.shape
    assert sigma_G.shape == r.shape
    assert torch.all(sigma_G >= 0)
    assert torch.all(torch.isfinite(G))
    # S(Q) → ~1 at high Q (Faber-Ziman asymptote), allow for the test model.
    assert abs(float(S[-50:].mean()) - 1.0) < 0.3


def test_pipeline_is_differentiable():
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.7, 20.0, 600, dtype=torch.float64)
    r = torch.linspace(0.5, 8.0, 300, dtype=torch.float64)
    I = _synthetic_iq(comp, q)

    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    G, _, _ = i_of_q_to_Gr(q, I, comp, r, wavelength_A=0.1665,
                           scale=scale, compton=True, q_max=18.0)
    G.pow(2).sum().backward()
    assert scale.grad is not None and torch.isfinite(scale.grad)


def test_delta_pdf_significance():
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.7, 20.0, 1200, dtype=torch.float64)
    r = torch.linspace(0.0, 10.0, 700, dtype=torch.float64)

    I_a = _synthetic_iq(comp, q, r0=1.62)
    I_b = _synthetic_iq(comp, q, r0=1.70)  # shell expands (e.g. heating)
    sig = torch.sqrt(I_a.clamp(min=1.0))

    G_a, sig_a, _ = i_of_q_to_Gr(q, I_a, comp, r, wavelength_A=0.1665,
                                 sigma_intensity=sig, q_max=18.0)
    G_b, sig_b, _ = i_of_q_to_Gr(q, I_b, comp, r, wavelength_A=0.1665,
                                 sigma_intensity=sig, q_max=18.0)

    dG, sig_dG = delta_pdf(G_a, G_b, sigma_a=sig_a, sigma_b=sig_b)
    assert dG.shape == r.shape
    # independent-error addition
    assert torch.allclose(sig_dG, torch.sqrt(sig_a**2 + sig_b**2), atol=1e-12)
    mask = significant_mask(dG, sig_dG, n_sigma=3.0)
    # a real shell shift should produce at least some significant points
    assert bool(mask.any())
