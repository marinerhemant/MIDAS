"""07 — Delta-PDF for time-resolved studies, with significance testing.

Because uncertainty is propagated, sigma^2(dG) = sigma_1^2 + sigma_2^2, so a
change between two states can be flagged as statistically real (>n sigma).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import Composition, delta_pdf, i_of_q_to_Gr, significant_mask

WL = 0.1665
comp = Composition({"Si": 1, "O": 2})
q = torch.linspace(0.7, 20.0, 1500, dtype=torch.float64)
r = torch.linspace(0.0, 10.0, 800, dtype=torch.float64)


def state(r0):
    f_avg, f2_avg = comp.form_factor_averages(q)
    debye = torch.sin(q * r0) / (q * r0) * torch.exp(-0.5 * (q * 0.05) ** 2)
    I = f2_avg * (1.0 + 4.0 * debye) + comp.compton(q, wavelength_A=WL)
    return i_of_q_to_Gr(q, I, comp, r, wavelength_A=WL,
                        sigma_intensity=torch.sqrt(I.clamp(min=1.0)), q_max=18.0)


G_a, sig_a, _ = state(1.62)         # before
G_b, sig_b, _ = state(1.70)         # after (e.g. shell expands on heating)
dG, sig_dG = delta_pdf(G_a, G_b, sigma_a=sig_a, sigma_b=sig_b)
mask = significant_mask(dG, sig_dG, n_sigma=3.0)

print(f"{int(mask.sum())} of {r.numel()} r-points show a >3σ change")
rsig = r[mask]
if rsig.numel():
    print(f"significant changes between r = {float(rsig.min()):.2f} and "
          f"{float(rsig.max()):.2f} Å (the shifting first shell)")
