"""The propagated σ_G must be *correct*, not just present.

Validates the analytic 1σ on G(r) from the full polyatomic pipeline against a
Monte-Carlo bootstrap: draw many noisy I(Q) realizations consistent with the
stated σ_I, push each through I(Q)→S(Q)→G(r), and compare the ensemble standard
deviation to the analytic σ_G. (The propagation assumes Q-bin independence, so
the bootstrap uses independent per-bin noise — the regime it is derived for.)
"""
import numpy as np
import torch

from midas_pdf import Composition, i_of_q_to_Gr


def test_sigma_G_matches_bootstrap():
    comp = Composition({"Si": 1, "O": 2})
    q = torch.linspace(0.7, 20.0, 900, dtype=torch.float64)
    r = torch.linspace(0.5, 8.0, 240, dtype=torch.float64)

    f_avg, f2_avg = comp.form_factor_averages(q)
    debye = torch.sin(q * 1.62) / (q * 1.62) * torch.exp(-0.5 * (q * 0.05) ** 2)
    I0 = f2_avg * (1.0 + 4.0 * debye) + comp.compton(q, wavelength_A=0.1665)
    sigma_I = 0.02 * I0.abs() + 0.5         # heteroscedastic per-bin σ

    # analytic propagated σ_G
    _, sigma_G_analytic, _ = i_of_q_to_Gr(
        q, I0, comp, r, wavelength_A=0.1665, sigma_intensity=sigma_I,
        compton=True, q_max=18.0,
    )

    # Monte-Carlo bootstrap with independent per-bin Gaussian noise
    n_trials = 400
    gen = torch.Generator().manual_seed(7)
    G_samples = torch.empty((n_trials, r.numel()), dtype=torch.float64)
    for t in range(n_trials):
        noise = torch.randn(q.shape, generator=gen, dtype=torch.float64) * sigma_I
        G_t, _, _ = i_of_q_to_Gr(
            q, I0 + noise, comp, r, wavelength_A=0.1665,
            compton=True, q_max=18.0,
        )
        G_samples[t] = G_t
    sigma_G_mc = G_samples.std(dim=0, unbiased=True)

    # Compare over the bulk (skip the smallest r where σ is tiny/edge effects).
    m = r > 1.0
    ratio = (sigma_G_analytic[m] / sigma_G_mc[m].clamp(min=1e-12))
    med = float(ratio.median())
    assert 0.9 < med < 1.1, f"median analytic/MC σ ratio = {med:.3f}"
