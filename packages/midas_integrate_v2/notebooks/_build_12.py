"""Builds notebook 12: APS PDF + operando demo (Item 7 composite)."""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
    ("md", """\
# 12 — APS PDF + Operando Demo

Maps the runner ``dev/paper/runners/run_aps_meeting_demo.py`` into a
walkthrough notebook for adoption training. Synthetic data, no
external services.

What this covers (one cell each, in order):

1. Synthesize a CeO2-like ring frame + a different empty-cell
   reference frame.
2. Apply spatial dezinger + cosmic-ray rejection.
3. Empty-cell subtraction (refinable scale, auto-fit).
4. Polygon-bin integrate with full σ propagation.
5. PDF transform via `integrate_to_Gr_with_variance`.
6. Write DAT (PDFgetX3), FXYE (TOPAS), ESG (MAUD/MILK) outputs.
7. η-coverage diagnostic on a full-aperture vs DAC mask.
"""),
    ("code", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    EmptySubtraction, PolygonBinGeometry,
    integrate_polygon_with_variance, spec_from_v1_params,
)
from midas_integrate_v2.dac import build_gasket_mask, eta_coverage_per_ring
from midas_integrate_v2.io import build_provenance, write_dat, write_esg, write_fxye
from midas_integrate_v2.pdf import R_px_to_Q, integrate_to_Gr_with_variance
from midas_integrate_v2.streaming import reject_cosmic_rays, reject_spatial_spikes
"""),
    ("md", "## 1. Synthesize sample + empty"),
    ("code", """\
NY = NZ = 384
rng = np.random.default_rng(7)
BC_y, BC_z = NY/2, NZ/2
Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
R = np.sqrt((Y - BC_y)**2 + (Z - BC_z)**2)

def ring_frame(intensity_scale, ring_radii, plant_spikes=20):
    img = np.zeros((NZ, NY))
    for r0 in ring_radii:
        img += intensity_scale * np.exp(-((R - r0)/1.5)**2)
    img = rng.poisson(img + 5.0).astype(float)
    if plant_spikes:
        flat_idx = rng.choice(img.size, size=plant_spikes, replace=False)
        img.flat[flat_idx] = 5_000.0
    return img

sample = ring_frame(900, np.linspace(20, 180, 6), plant_spikes=20)
empty = ring_frame(80, [40, 90], plant_spikes=0)
sample.shape, empty.shape
"""),
    ("md", "## 2. Spatial dezinger + cosmic-ray rejection"),
    ("code", """\
sample_dz, mask_dz = reject_spatial_spikes(sample, n_sigma=5.0, method="laplacian")
print(f"flagged {int(mask_dz.sum())} pixel-spikes")
stack = np.tile(sample_dz[None], (3, 1, 1))
clean, _ = reject_cosmic_rays(stack, n_sigma=5.0)
sample_clean = clean[1]
"""),
    ("md", "## 3. Build spec + empty-cell subtraction"),
    ("code", """\
p = IntegrationParams(
    NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0, Lsd=1_500_000.0,
    BC_y=NY/2, BC_z=NZ/2, RhoD=200.0,
    RMin=10.0, RMax=200.0, RBinSize=0.5,
    EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
    Wavelength=0.18,
)
spec = spec_from_v1_params(p, requires_grad=False)
es = EmptySubtraction(torch.as_tensor(empty, dtype=torch.float64),
                       scale=1.0, clip_negative=True)
sample_t = torch.as_tensor(sample_clean, dtype=torch.float64)
sub = es(sample_t)
"""),
    ("md", "## 4. Polygon integrate with σ"),
    ("code", """\
geom = PolygonBinGeometry.from_spec(spec)
mean2d, sigma2d = integrate_polygon_with_variance(sub, geom)
n_eta = mean2d.shape[0]
I = mean2d.mean(dim=0)
sigma_I = torch.sqrt((sigma2d**2).sum(dim=0)) / n_eta
R_axis = spec.RMin + (np.arange(I.shape[0]) + 0.5) * spec.RBinSize
Q_axis = R_px_to_Q(torch.as_tensor(R_axis), Lsd_um=spec.Lsd, px_um=spec.pxY,
                    lambda_A=spec.Wavelength).numpy()
two_theta_deg = np.degrees(np.arctan(R_axis * spec.pxY / float(spec.Lsd)))
"""),
    ("md", "## 5. PDF G(r) with σ propagation"),
    ("code", """\
r_grid = torch.arange(0.5, 10.01, 0.02, dtype=torch.float64)
r, G, sigma_G = integrate_to_Gr_with_variance(
    sub, spec, r_grid, Q_min=0.5, Q_max=15.0, Q_step=0.01, window="lorch",
)
print(f"G(r) shape {G.shape}, σ_G median {float(sigma_G.median()):.4e}")
"""),
    ("md", "## 6. Write DAT, FXYE, ESG"),
    ("code", """\
from pathlib import Path
out = Path("/tmp/aps_demo_nb")
out.mkdir(exist_ok=True)
md = build_provenance(spec, integrate_mode="polygon", extra={"demo": "aps_meeting_nb"})
write_dat(out/"sample.dat", q_axis_invA=Q_axis, intensity=I.numpy(), sigma=sigma_I.numpy(), metadata=md)
write_fxye(out/"sample.fxye", r_axis=two_theta_deg, intensity=I.numpy(),
            sigma=sigma_I.numpy(), x_unit="degrees_2theta", title="aps_demo_nb")
write_esg(out/"sample.esg", two_theta_deg=two_theta_deg, intensity=I.numpy(),
           sigma=sigma_I.numpy(), wavelength_A=float(spec.Wavelength))
print(list(out.iterdir()))
"""),
    ("md", "## 7. η-coverage diagnostic"),
    ("code", """\
mask = build_gasket_mask(NY, NZ, BC=(float(spec.BC_y), float(spec.BC_z)),
                          eta_open_deg=(-180.0, 180.0), symmetry="single")
cov = eta_coverage_per_ring(spec, mask, torch.tensor(np.linspace(spec.RMin, spec.RMax, 6)))
print("η-coverage per ring:", cov.numpy())
"""),
]


def main():
    out = Path(__file__).parent / "12_aps_pdf_operando_demo.ipynb"
    write_notebook(out, CELLS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
