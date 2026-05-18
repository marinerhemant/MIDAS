"""Builds notebook 17: σ → Rietveld chain (Item 34 paper material)."""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
    ("md", """\
# 17 — σ Chain: Pixel → MIDAS → TOPAS Rietveld

This notebook accompanies the methods paper (item 34 of the
implementation plan). Goal: show the full uncertainty chain from
Poisson pixel σ all the way to lattice-parameter / phase-fraction σ
that TOPAS reports.

What you'll see:

1. Synthesise a Ni standard frame with planted Poisson noise.
2. Integrate via the polygon kernel → I(2θ) + σ_I.
3. Bootstrap n=30 Monte-Carlo realisations to validate the analytic σ.
4. Write FXYE for TOPAS Rietveld (σ in column 3 = ESD as TOPAS expects).
5. Methodology notes for the actual TOPAS pass.
"""),
    ("code", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    PolygonBinGeometry, integrate_polygon_with_variance, spec_from_v1_params,
)
from midas_integrate_v2.io import build_provenance, write_fxye
"""),
    ("md", "## 1. Synthesize Ni standard"),
    ("code", """\
NY = NZ = 256
rng = np.random.default_rng(2026)
BC_y, BC_z = NY/2, NZ/2
Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
R = np.sqrt((Y - BC_y)**2 + (Z - BC_z)**2)
img_mean = np.full((NZ, NY), 50.0)
for r0 in [25, 30, 45, 55, 65, 80]:
    img_mean += 600.0 * np.exp(-((R - r0)/1.4)**2)
sample = rng.poisson(img_mean).astype(float)
"""),
    ("md", "## 2. Integrate with σ"),
    ("code", """\
p = IntegrationParams(
    NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0, Lsd=1_500_000.0,
    BC_y=BC_y, BC_z=BC_z, RhoD=120.0,
    RMin=10.0, RMax=120.0, RBinSize=0.5,
    EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0, Wavelength=0.18,
)
spec = spec_from_v1_params(p, requires_grad=False)
geom = PolygonBinGeometry.from_spec(spec)
img_t = torch.as_tensor(sample, dtype=torch.float64)
mean2d, sig2d = integrate_polygon_with_variance(img_t, geom)
n_eta = mean2d.shape[0]
I = mean2d.mean(dim=0).numpy()
sigma_analytic = (torch.sqrt((sig2d**2).sum(dim=0)) / n_eta).numpy()
R_axis = spec.RMin + (np.arange(I.shape[0]) + 0.5) * spec.RBinSize
two_theta_deg = np.degrees(np.arctan(R_axis * spec.pxY / float(spec.Lsd)))
print("integrated", I.shape)
"""),
    ("md", "## 3. Bootstrap to validate analytic σ"),
    ("code", """\
n_boot = 30
boot = np.zeros((n_boot, I.shape[0]))
for k in range(n_boot):
    b = rng.poisson(img_mean).astype(float)
    m_b, _ = integrate_polygon_with_variance(torch.as_tensor(b, dtype=torch.float64), geom)
    boot[k] = m_b.mean(dim=0).numpy()
sigma_boot = boot.std(axis=0, ddof=1)
valid = sigma_analytic > 0
median_relerr = float(np.median(np.abs(sigma_analytic[valid] - sigma_boot[valid])
                                  / np.maximum(sigma_boot[valid], 1e-30)))
print(f"median |σ_analytic - σ_bootstrap| / σ_bootstrap = {median_relerr:.3f}")
"""),
    ("md", """\
## 4. Write FXYE for TOPAS

TOPAS reads the third column as ESD (= σ, one standard deviation).
GSAS-II reads the same. Both use χ²-weighting based on this column.
"""),
    ("code", """\
from pathlib import Path
out = Path("/tmp/sigma_rietveld_nb")
out.mkdir(exist_ok=True)
md = build_provenance(spec, integrate_mode="polygon", extra={"demo": "sigma_rietveld_nb"})
write_fxye(out/"ni_for_topas.fxye",
            r_axis=two_theta_deg, intensity=I, sigma=sigma_analytic,
            x_unit="degrees_2theta", title="ni_standard for TOPAS Rietveld",
            metadata=md)
print(f"wrote {out / 'ni_for_topas.fxye'}")
"""),
    ("md", """\
## 5. From here to TOPAS lattice σ

The methods-paper claim: when TOPAS ingests this FXYE, its
covariance matrix at the optimum carries σ on lattice parameter,
phase fraction, isotropic ADP, and so on. The propagated σ_a / a is
the *bottom of the chain* that started with one Poisson photon.

For the actual TOPAS pass the user runs (outside the notebook):

```
TOPAS topas_master.inp ni_for_topas.fxye
```

where ``topas_master.inp`` ``#include``s a phase block produced by
``midas_hkls.io.write_topas_phase`` (Item 13). The output ``.out``
file carries one ESD per refined parameter — these are the ``σ_a``,
``σ_phase_fraction`` etc. values that close the chain.
"""),
]


def main():
    out = Path(__file__).parent / "17_sigma_to_rietveld.ipynb"
    write_notebook(out, CELLS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
