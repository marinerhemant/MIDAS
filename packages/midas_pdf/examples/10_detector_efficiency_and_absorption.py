"""10 — Q-dependent detector efficiency and sample self-absorption.

Both are backed by the NIST mass-attenuation tables already in midas-hkls.
A finite-thickness sensor detects more at high scattering angle (longer path);
divide I(Q) by the efficiency to remove that high-Q tilt.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import (apply_detector_efficiency, detector_efficiency,
                       flat_plate_transmission, linear_attenuation_um)

WL = 0.1665
q = torch.tensor([2.0, 10.0, 20.0, 28.0], dtype=torch.float64)

mu_cdte = linear_attenuation_um({"Cd": 0.468, "Te": 0.532}, WL, density_g_cm3=5.85)
print(f"CdTe linear attenuation at {WL} Å: {mu_cdte:.3e} 1/µm")

eta = detector_efficiency(q, wavelength_A=WL, material={"Cd": 0.468, "Te": 0.532},
                          thickness_um=1000.0, density_g_cm3=5.85)
print(f"\n{'Q':>5} {'efficiency η':>13}")
for qi, e in zip(q, eta):
    print(f"{float(qi):5.1f} {float(e):13.4f}")

# applying the correction divides out the tilt
I = torch.ones_like(q)
I_corr, _ = apply_detector_efficiency(I, q, wavelength_A=WL, material="Si",
                                      thickness_um=450.0)
print("\nSi sensor correction factor 1/η:", [round(float(x), 3) for x in I_corr])

A = flat_plate_transmission(q, wavelength_A=WL, mu_um=mu_cdte * 0.01, thickness_um=200.0)
print("flat-plate self-absorption A(Q):", [round(float(x), 4) for x in A])
