"""04 — The complete chain: detector pixels -> G(r) (slow ~30-60 s).

Uses midas-integrate-v2 polygon binning under the hood. Renders a synthetic
powder image of a Ni dimer, integrates, normalizes, and Fourier transforms;
the recovered first-shell peak sits at the true bond length.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_integrate_v2.spec import IntegrationSpec
from midas_pdf import Composition
from midas_pdf.frontend import image_to_Gr
from midas_pdf.validate import debye_scattering_intensity, synthetic_powder_image

WL = 0.1665
n = 384                                  # small detector to keep it quick

s = IntegrationSpec()
s.NrPixelsY = s.NrPixelsZ = n
s.pxY = s.pxZ = 150.0
s.Lsd = torch.tensor(70000.0, dtype=torch.float64)
s.BC_y = torch.tensor(n / 2.0, dtype=torch.float64)
s.BC_z = torch.tensor(n / 2.0, dtype=torch.float64)
s.Wavelength = torch.tensor(WL, dtype=torch.float64)
s.RMin, s.RMax, s.RBinSize, s.EtaBinSize = 5.0, n / 2 - 5, 1.0, 5.0

d = 2.50
q_prof = torch.linspace(0.3, 20.0, 2000, dtype=torch.float64)
I_prof = debye_scattering_intensity(
    q_prof, ["Ni", "Ni"],
    torch.tensor([[0, 0, 0], [d, 0, 0]], dtype=torch.float64), thermal_B=0.3)
img = synthetic_powder_image(s, q_prof, I_prof, counts=2e5, seed=5)

r = torch.linspace(0.5, 6.0, 1000, dtype=torch.float64)
print("integrating + transforming (polygon binning, please wait)...")
Q, G, sigma_G, S = image_to_Gr(img, s, Composition({"Ni": 1}), r,
                               compton=False, q_min=0.8, q_max=15.0)
win = (r >= 1.5) & (r <= 4.0)
print(f"recovered bond length: {float(r[win][torch.argmax(G[win])]):.3f} Å  (true {d})")
print(f"sigma_G carried through from pixel counting stats: median {float(sigma_G.median()):.4f}")
