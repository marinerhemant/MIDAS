"""08 — Model-free validation via the Debye scattering equation.

A finite atom cluster has an exact powder-averaged I(Q) (Debye). Running it
through midas-pdf must give a G(r) whose peaks sit at the true interatomic
distances — a physics-level check needing no external PDF software.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import Composition, i_of_q_to_Gr
from midas_pdf.validate import debye_scattering_intensity, interatomic_distances

WL = 0.1665
# a small linear chain of 3 Ni atoms at 2.5 Å spacing -> distances 2.5 and 5.0
pos = torch.tensor([[0., 0., 0.], [2.5, 0., 0.], [5.0, 0., 0.]], dtype=torch.float64)
D = interatomic_distances(pos)
print("interatomic distances:", sorted({round(float(x), 2) for x in D.flatten() if x > 0}))

q = torch.linspace(0.5, 24.0, 3000, dtype=torch.float64)
r = torch.linspace(0.5, 7.0, 1300, dtype=torch.float64)
I = debye_scattering_intensity(q, ["Ni"] * 3, pos, thermal_B=0.3)
G, _, _ = i_of_q_to_Gr(q, I, Composition({"Ni": 1}), r,
                       wavelength_A=WL, compton=False, q_max=22.0)

# find peaks in the physical region (r > 1.5 Å; below the nearest distance the
# reduced PDF is just the -4πρ₀r baseline plus termination ripple).
from scipy.signal import find_peaks
import numpy as np
phys = r > 1.5
Gp = G.numpy()[phys.numpy()]
rp = r.numpy()[phys.numpy()]
pk, _ = find_peaks(Gp, height=float(Gp.max()) * 0.1)
print("recovered G(r) peaks at r =", [round(float(rp[i]), 2) for i in pk], "Å")
print("(expected near 2.5 and 5.0 Å; the 2.5 Å pair is stronger — two bonds vs one)")
