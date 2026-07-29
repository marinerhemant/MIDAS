"""12 — Multiple scattering: Tier-1 lumped background and Tier-2/3 first principles.

(a) Tier 1: subtract a refinable smooth polynomial background (lumps MS +
    fluorescence + air), wired through the normalization.
(b) The differentiable per-atom differential cross-section dσ/dΩ(Q).
(c) Tier 2/3: an analog Monte-Carlo MS estimate for a slab (the reference) and
    the analytic, differentiable single-scattering factor, plus the MS fraction
    β(Q) ready to feed back as a background.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from midas_pdf import (Composition, differential_cross_section, faber_ziman_S,
                       lumped_background, multiple_scattering_mc,
                       slab_optical_params, slab_single_scattering_factor)
from midas_pdf.ms import ms_background_on_grid, slab_double_scattering

WL = 0.1665
comp = Composition({"Si": 1, "O": 2})
q = torch.linspace(0.7, 22.0, 1500, dtype=torch.float64)

# (a) Tier-1 lumped smooth background
f_avg, f2_avg = comp.form_factor_averages(q)
I = f2_avg * (1.0 + 0.3 * torch.sin(2.0 * q)) + comp.compton(q, wavelength_A=WL)
b = lumped_background(q, [4.0, -0.6], q_max=22.0)        # 4 - 0.6*(Q/Qmax)
S_raw, _ = faber_ziman_S(I, q, comp, wavelength_A=WL)
S_bg, _ = faber_ziman_S(I, q, comp, wavelength_A=WL, background=b)
print(f"(a) Tier-1 lumped background subtracts {float(b.mean()):.2f} (mean) before normalizing")

# (b) differential cross-section (the MS engine)
dsig = differential_cross_section(q, comp, wavelength_A=WL)
print(f"(b) dσ/dΩ(Q): {float(dsig[0]):.1f} at low Q -> {float(dsig[-1]):.1f} at high Q")

# (c) first-principles MS for a 2 mm SiO2 slab
mu, tau, albedo = slab_optical_params(comp, wavelength_A=WL, thickness_um=2000.0,
                                      number_density_A3=0.0709)
print(f"(c) slab: optical depth τ={tau:.3f}, albedo={albedo:.3f}")
mc = multiple_scattering_mc(comp, wavelength_A=WL, tau=tau, albedo=albedo,
                            n_photons=150_000, seed=0)
beta = mc["n_multiple"] / (mc["n_single"] + mc["n_multiple"])
print(f"    Monte-Carlo multiple-scattering fraction β ≈ {beta:.3f}")

# (d) Tier-3: the DIFFERENTIABLE analytic double-scattering, validated vs the MC
an = slab_double_scattering(comp, wavelength_A=WL, tau=tau, albedo=albedo, q_max=20.0)
print(f"(d) analytic differentiable double-scattering β (angle-avg): "
      f"{float((an['beta_double'] * an['I_single']).sum() / an['I_single'].sum()):.3f}")
print("    -> matches the Monte-Carlo exactly-double channel pointwise in Q "
      "(see dev/demo_multiple_scattering.png)")

ms_bg = ms_background_on_grid(q, I, an)     # differentiable MS background, ready for background=
print(f"    MS background on data grid (from analytic estimator): mean "
      f"{float(ms_bg.mean()):.2f}  (feed as background= to remove MS)")
