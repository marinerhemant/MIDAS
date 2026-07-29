"""09 — Hubbell incoherent (Compton) scattering + Breit-Dirac recoil.

The tabulated Hubbell incoherent scattering function (the data GudrunX/PDFgetX
use) replaces the coarse IT94 analytic form, which has the wrong shape.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import Composition
from midas_pdf.compton import breit_dirac_factor

WL = 0.1665
comp = Composition({"Si": 1, "O": 2})
q = torch.tensor([2.0, 8.0, 16.0, 28.0], dtype=torch.float64)

hub = comp.compton(q, wavelength_A=WL, method="hubbell", breit_dirac=False)
hub_bd = comp.compton(q, wavelength_A=WL, method="hubbell", breit_dirac=True)
it94 = comp.compton(q, wavelength_A=WL, method="it94")
R = breit_dirac_factor(q, wavelength_A=WL)

print(f"{'Q':>5} {'Hubbell':>9} {'+BreitD':>9} {'IT94(old)':>10} {'BD factor':>10}")
for i in range(len(q)):
    print(f"{float(q[i]):5.1f} {float(hub[i]):9.3f} {float(hub_bd[i]):9.3f} "
          f"{float(it94[i]):10.3f} {float(R[i]):10.3f}")
print("\nHubbell rises monotonically toward <Z>=10; IT94 peaks early and falls (wrong).")
