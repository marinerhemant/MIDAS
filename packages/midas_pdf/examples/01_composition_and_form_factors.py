"""01 — Composition: number-fraction-weighted form-factor averages.

⟨f⟩(Q) and ⟨f²⟩(Q) are the polyatomic Faber-Ziman building blocks; their
difference is the Laue self-scattering term (zero for one element).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import Composition

comp = Composition({"Si": 1, "O": 2})          # SiO2, number fractions 1/3, 2/3
print("composition:", comp.as_dict())

q = torch.linspace(0.0, 20.0, 5, dtype=torch.float64)
f_avg, f2_avg = comp.form_factor_averages(q)
laue = comp.laue(q)

print(f"{'Q':>6} {'<f>':>10} {'<f^2>':>10} {'Laue':>10}")
for qi, fa, f2, l in zip(q, f_avg, f2_avg, laue):
    print(f"{float(qi):6.1f} {float(fa):10.4f} {float(f2):10.4f} {float(l):10.4f}")

# f(0) ~ Z: <f>(0) should be (14 + 2*8)/3 = 10
print(f"\n<f>(Q=0) = {float(f_avg[0]):.3f}  (≈ Z-weighted 10.0)")

# differentiable in composition
frac = torch.tensor([0.34, 0.66], dtype=torch.float64, requires_grad=True)
comp.form_factor_averages(q, fractions=frac)[0].sum().backward()
print("d<f>/d(fractions):", frac.grad.tolist())
