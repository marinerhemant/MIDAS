"""Build .ipynb files from a maintainable cell-list source.

    cd packages/midas_pink/notebooks
    python _build.py
    python _build.py 01_spectrum_aware_recovery
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).parent
Cell = Tuple[str, str]


def _make_cell(kind: str, source: str, *, idx: int) -> dict:
    src_lines = source.splitlines(keepends=True)
    cell_id = f"cell-{idx:03d}"
    if kind == "md":
        return {"id": cell_id, "cell_type": "markdown",
                "metadata": {}, "source": src_lines}
    if kind == "py":
        return {"id": cell_id, "cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [], "source": src_lines}
    raise ValueError(f"unknown cell kind {kind!r}")


def write_notebook(name: str, cells: List[Cell]) -> Path:
    nb = {
        "cells": [_make_cell(k, s, idx=i) for i, (k, s) in enumerate(cells)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (midas_env)",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out_path = HERE / f"{name}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path


# =====================================================================
# 01 — Spectrum-aware plant-and-recover (mono -> pink)
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — `midas-pink`: spectrum-aware plant-and-recover

`midas-pink` extends the monochromatic `midas-diffract` forward model to
an arbitrary illumination spectrum `S(E)` by integrating per-energy mono
forward evaluations as a discrete weighted sum. The same loss, optimiser,
and parameterisation cover **monochromatic, pink, and white** HEDM.

This notebook does a small synthetic round trip:

1. Build a `ParameterisedSpectrum` (a narrow pink Gaussian).
2. Build a per-energy mono **bank** for an FCC grain.
3. **Plant** a grain, splat its observed ROIs (the "measurement").
4. **Recover** orientation + lattice from a perturbed seed with
   `recover_grain_state`.
5. Show the monochromatic → pink extension is just the spectrum width.

CPU + synthetic; runs in well under a minute.
"""),
    ("py", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
DEG2RAD = math.pi / 180.0
print("torch", torch.__version__, "| device: cpu")
"""),
    ("md", """\
## Step 1 — A parameterised pink spectrum

`ParameterisedSpectrum` holds softmax-normalised energy weights on a
fixed dense energy grid. Here a fixed Gaussian centred at 71.6764 keV
with relative bandwidth 1e-2 (a narrow pink beam). `fixed=True` means the
weights are not refined — we treat `S(E)` as known.
"""),
    ("py", """\
import midas_pink as mp

spec = mp.ParameterisedSpectrum(
    E0_keV=71.6764, half_bw=0.03, n_samples=21,
    init_kind="gaussian", init_rel_bw=1e-2, fixed=True,
    dtype=torch.float64,
)
w = spec.weights()
print("n energies:", spec.energies_keV.shape[0])
print("weights sum to:", float(w.sum()))
print("spectrum centroid:", float((w * spec.energies_keV).sum()), "keV")
print("wavelength range:", float(spec.lambdas_A.min()), "->",
      float(spec.lambdas_A.max()), "A")
"""),
    ("md", """\
## Step 2 — Build the per-energy mono bank

`build_pink_bank` constructs one mono `HEDMForwardModel` per energy
sample from a geometry factory `lam_A -> HEDMGeometry`. The grain's
predicted pattern is then the spectrum-weighted sum over the bank.
"""),
    ("py", """\
import midas_diffract as md
from midas_hkls import SpaceGroup, Lattice

def geom_factory(lam_A):
    return md.HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=-180.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048,
        min_eta=6.0, wavelength=lam_A,
    )

bank = mp.build_pink_bank(
    spec,
    space_group=SpaceGroup.from_number(225),       # FCC
    lattice=Lattice.for_system("cubic", a=4.078),
    geom_factory=geom_factory, two_theta_max_deg=6.0,
    device="cpu", dtype=torch.float64,
)
print("mono models in bank:", len(bank.models))
print("reflections:", bank.hkls_int.shape[0])
"""),
    ("md", """\
## Step 3 — Plant a grain and splat the observed ROIs

`plan_rois_from_state` picks a small detector window (ROI) around each
predicted spot of the planted grain; `splat_rois` renders the
spectrum-integrated intensity into those windows with a Gaussian PSF.
These rendered ROIs are our synthetic **measurement**.
"""),
    ("py", """\
gt_euler = torch.tensor([45.0, 30.0, 60.0]) * DEG2RAD
gt_pos   = torch.zeros(3)
gt_latc  = torch.tensor([4.078, 4.078, 4.078, 90.0, 90.0, 90.0])

plan = mp.plan_rois_from_state(
    bank, gt_euler.unsqueeze(0), gt_pos.unsqueeze(0),
    lattice_params=gt_latc, roi_h=21, roi_w=21,
)
observed = mp.splat_rois(
    bank, plan, gt_euler.unsqueeze(0), gt_pos.unsqueeze(0), gt_latc,
    sigma_psf_px=1.5,
)
print("ROIs kept:", plan.n_kept, "| observed tensor:", tuple(observed.shape))
print("observed peak intensity:", round(float(observed.max()), 4))
"""),
    ("md", """\
## Step 4 — Recover (orientation + lattice) from a perturbed seed

Perturb the truth (~0.3° orientation, ~5e-4 Å lattice) and let
`recover_grain_state` run its L-BFGS phase schedule. We add a centroid
term so the basin of attraction is the whole panel, not just the ROI
window.
"""),
    ("py", """\
init_euler = gt_euler + 0.3 * DEG2RAD * torch.randn(3)
init_latc  = gt_latc + 5e-4 * torch.randn(6)

cfg = mp.RecoveryConfig(
    sigma_psf_px=1.5,
    phase1_steps=20, phase2_steps=20, phase3_steps=20,
    fit_orientation=True, fit_lattice=True, fit_position=False,
    centroid_loss_weight=1.0, image_loss_weight=1.0,
)
result = mp.recover_grain_state(
    bank, plan, observed,
    init_euler=init_euler, init_position=gt_pos, init_lattice=init_latc,
    cfg=cfg,
)
print("final loss:", f"{result['final_loss']:.3e}")
"""),
    ("md", """\
### How good is the recovery?

Convert recovered vs. truth Euler angles to a misorientation and report
the lattice error. We use the forward model's own `euler2mat` so the
convention matches the one used to plant the grain.
"""),
    ("py", """\
from midas_diffract import HEDMForwardModel

def misori_deg(e_a, e_b):
    Ra = HEDMForwardModel.euler2mat(e_a)
    Rb = HEDMForwardModel.euler2mat(e_b)
    dR = Ra @ Rb.T
    cos_t = ((torch.trace(dR) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.acos(cos_t) / DEG2RAD)

seed_err = misori_deg(init_euler, gt_euler)
rec_err  = misori_deg(result["euler"], gt_euler)
lat_err  = float((result["lattice"][:3] - gt_latc[:3]).abs().max())
print(f"orientation error  seed={seed_err:.4f} deg  ->  recovered={rec_err:.4f} deg")
print(f"lattice |Δa| max   = {lat_err:.2e} A")
"""),
    ("md", """\
## Monochromatic → pink: it's just the spectrum width

A *monochromatic* model is the limit of a delta spectrum
(`init_kind="delta"`); a *pink* model widens `init_rel_bw`. The bank,
plan, splat, and recovery API are byte-for-byte identical — only the
`ParameterisedSpectrum` changes. We verify the delta-spectrum bank holds
a single energy and that the splat still produces finite ROIs.
"""),
    ("py", """\
spec_mono = mp.ParameterisedSpectrum(
    E0_keV=71.6764, half_bw=0.02, n_samples=5,
    init_kind="delta", fixed=True, dtype=torch.float64,
)
wm = spec_mono.weights()
print("delta-spectrum weights:", np.round(wm.tolist(), 4),
      "(mass concentrated at the central energy)")

bank_mono = mp.build_pink_bank(
    spec_mono, space_group=SpaceGroup.from_number(225),
    lattice=Lattice.for_system("cubic", a=4.078),
    geom_factory=geom_factory, two_theta_max_deg=6.0,
    device="cpu", dtype=torch.float64,
)
plan_m = mp.plan_rois_from_state(
    bank_mono, gt_euler.unsqueeze(0), gt_pos.unsqueeze(0),
    lattice_params=gt_latc, roi_h=21, roi_w=21,
)
rois_m = mp.splat_rois(bank_mono, plan_m, gt_euler.unsqueeze(0),
                       gt_pos.unsqueeze(0), gt_latc, sigma_psf_px=1.5)
print("mono ROIs:", tuple(rois_m.shape), "| finite:", bool(torch.isfinite(rois_m).all()))
print("\\nSame code path; pink simply sums more per-energy mono "
      "evaluations under a wider S(E).")
"""),
    ("md", """\
## Where to go next

- `recover_joint` — refine `S(E)` *and* the grain state together when the
  spectrum is unknown (needs a learnable, non-fixed spectrum and a
  `centroid_E0_penalty` to break the energy↔lattice degeneracy).
- `recover_two_stage` — centroid alignment then image-profile refinement.
- `splat_rois_3d` — per-frame (ω) ROIs when orientation DOF shift spots
  primarily in ω.

The full paper protocols (bandwidth sweeps, noise studies, calibrant
spectrum fits) live in `dev/paper/scripts/`.
"""),
]


def build_all(only: str | None = None) -> None:
    notebooks = {"01_spectrum_aware_recovery": NB_01}
    for name, cells in notebooks.items():
        if only and only not in name:
            continue
        p = write_notebook(name, cells)
        print("wrote", p)


if __name__ == "__main__":
    build_all(sys.argv[1] if len(sys.argv) > 1 else None)
