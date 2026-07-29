"""06 — The correlation-function family: F(Q), g(r), T(r), RDF.

Different total-scattering communities report different functions. From one
G(r) and the number density rho0 we produce all of them (Keen 2001), with the
correct sigma scaling.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import (Composition, i_of_q_to_Gr, pair_distribution_g,
                       radial_distribution_R, structure_function_F,
                       total_correlation_T)

WL, RHO = 0.1665, 0.0709
comp = Composition({"Si": 1, "O": 2})
q = torch.linspace(0.7, 22.0, 2000, dtype=torch.float64)
r = torch.linspace(0.05, 10.0, 900, dtype=torch.float64)

f_avg, f2_avg = comp.form_factor_averages(q)
debye = torch.sin(q * 1.62) / (q * 1.62) * torch.exp(-0.5 * (q * 0.05) ** 2)
I = f2_avg * (1.0 + 4.0 * debye) + comp.compton(q, wavelength_A=WL)
G, _, S = i_of_q_to_Gr(q, I, comp, r, wavelength_A=WL, q_max=20.0)

F, _ = structure_function_F(q, S)
g, _ = pair_distribution_g(r, G, number_density=RHO)
T, _ = total_correlation_T(r, G, number_density=RHO)
R, _ = radial_distribution_R(r, G, number_density=RHO)

print("F(Q) = Q[S-1]:   max |F| =", round(float(F.abs().max()), 3))
print("g(r):            g at large r ->", round(float(g[-1]), 3), "(should be ~1)")
print("T(r) = G + 4πrρ: T at large r ->", round(float(T[-1]), 3))
print("RDF R(r):        peak value   ->", round(float(R.max()), 3),
      "(integral over a peak = coordination number)")
