"""13 — Small-box PDF structure refinement (PDFfit-style), error-aware.

Build a model G(r) from a crystal structure (FCC Ni), fit a noisy observation,
and recover the lattice parameter / displacement / scale — WITH parameter
uncertainties from the autograd Hessian and a calibrated chi^2 (the error-aware
features PDFgui does only crudely). Fully differentiable, so the structure could
be co-refined with geometry / normalization / multiple scattering.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_pdf.structure import build_pair_list, pdffit_gr, refine_structure

# truth
a0, U0, s0 = 3.524, 0.0055, 1.0
ct = Crystal(lattice=Lattice(a0, a0, a0, 90, 90, 90),
             space_group=SpaceGroup.from_number(225),
             atoms=[Atom(element="Ni", fract=(0, 0, 0))], name="Ni").to_torch()
pairs = build_pair_list(ct, r_max=10.0)
r = torch.linspace(1.5, 10.0, 1700, dtype=torch.float64)
lp0 = torch.tensor([a0, a0, a0, 90, 90, 90], dtype=torch.float64)
G_true = pdffit_gr(ct, r, pairs, scale=s0, u_iso=U0, lattice_params=lp0)

# synthetic observation with known sigma
rng = np.random.default_rng(0)
sigma = 0.03 * torch.ones_like(G_true)
G_obs = G_true + torch.tensor(rng.normal(0, 0.03, size=G_true.shape), dtype=torch.float64)

# refine from deliberately wrong starting values
res = refine_structure(ct, r, G_obs, pairs, sigma_obs=sigma,
                       init_a=3.60, init_u_iso=0.010, init_scale=0.8, steps=150)

print("           truth      fitted ± 1σ (from autograd Hessian)")
print(f"  a     {a0:9.4f}   {res.fitted['a']:.5f} ± {res.uncertainty['a']:.2e} Å")
print(f"  U_iso {U0:9.5f}   {res.fitted['u_iso']:.5f} ± {res.uncertainty['u_iso']:.2e} Å²")
print(f"  scale {s0:9.3f}   {res.fitted['scale']:.4f} ± {res.uncertainty['scale']:.2e}")
print(f"\n  chi²/ndof = {res.chi2_reduced:.3f}  (≈1 ⇒ error-aware fit is well-calibrated)")
