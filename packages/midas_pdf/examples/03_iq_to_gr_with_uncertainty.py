"""03 — Full I(Q) -> G(r) with a propagated 1-sigma band.

The headline capability: every G(r) point carries a validated uncertainty.
Saves a figure with the +/-1 sigma band.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from midas_pdf import Composition, i_of_q_to_Gr

WL = 0.1665
comp = Composition({"Si": 1, "O": 2})
q = torch.linspace(0.7, 22.0, 2000, dtype=torch.float64)
r = torch.linspace(0.0, 10.0, 900, dtype=torch.float64)

f_avg, f2_avg = comp.form_factor_averages(q)
debye = torch.sin(q * 1.62) / (q * 1.62) * torch.exp(-0.5 * (q * 0.05) ** 2)
I = f2_avg * (1.0 + 4.0 * debye) + comp.compton(q, wavelength_A=WL)
sigma_I = torch.sqrt(I.clamp(min=1.0))

G, sigma_G, S = i_of_q_to_Gr(q, I, comp, r, wavelength_A=WL,
                             sigma_intensity=sigma_I, q_max=20.0)
peak = float(r[torch.argmax(G)])
print(f"first G(r) peak near r = {peak:.2f} Å (input shell at 1.62 Å)")
print(f"median sigma_G = {float(sigma_G.median()):.4f}")

out = Path(__file__).parent / "figures"; out.mkdir(exist_ok=True)
rn, Gn, Gs = r.numpy(), G.numpy(), sigma_G.numpy()
plt.figure(figsize=(7, 4))
plt.fill_between(rn, Gn - Gs, Gn + Gs, alpha=0.3, label="±1σ")
plt.plot(rn, Gn, lw=1.0, label="G(r)")
plt.axhline(0, color="0.8", lw=0.6); plt.xlabel("r (Å)"); plt.ylabel("G(r)")
plt.legend(); plt.tight_layout()
plt.savefig(out / "example_03_gr.png", dpi=130)
print(f"figure -> {out / 'example_03_gr.png'}")
