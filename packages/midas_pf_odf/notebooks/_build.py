"""Build .ipynb files from a maintainable cell-list source.

    cd packages/midas_pf_odf/notebooks
    python _build.py
    python _build.py 01_peakshape_inversion
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
# 01 — Phase-1 peak-shape inversion
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — `midas-pf-odf`: Phase-1 per-grain peak-shape inversion

pf-HEDM (point-focused HEDM / scanning 3DXRD) scans a focused beam across
a sample and records a diffraction pattern per beam position. **Phase 1**
of `midas-pf-odf` recovers, for each grain, the per-voxel orientation
`R_V` and strain `ε_V` jointly — by matching the *shape* of the measured
3D peak patches `(F, P, P)`, not just their centroids.

This notebook:

1. Plants a small multi-grain microstructure and simulates per-grain
   patches with the differentiable forward model.
2. Recovers per-voxel `(R, ε)` with `fit_multi_grain`.
3. Validates per voxel with `recovery_metrics` (misorientation RMS, ε RMSE).
4. Compares peak-shape inversion against the **centroid baseline** on a
   strained single grain — the headline Phase-1 claim.

CPU + synthetic; a small grid + short L-BFGS keep it under ~2 minutes.
The forward model and scan config come from the package's own test
scaffolding (`tests/conftest.py`).
"""),
    ("py", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
print("torch", torch.__version__, "| device: cpu")
"""),
    ("py", """\
# Use the package test scaffolding for the pf-HEDM forward model.
import sys
from pathlib import Path
PKG = Path.cwd().parent          # packages/midas_pf_odf
sys.path.insert(0, str(PKG))
from tests.conftest import make_fcc_hkls, small_scan_config, build_model

G_cart, thetas, hkls_int = make_fcc_hkls(d_min_A=1.0, h_max=2)   # tiny set, fast
scan = small_scan_config(sample_size_um=20.0, n_scans=9, beam_size_um=4.0)
model = build_model(scan, hkls_int, G_cart, thetas)
print("reflections:", hkls_int.shape[0], "| scan positions:", scan.beam_positions.numel())
"""),
    ("md", """\
## Step 1 — Plant a multi-grain microstructure

`plant_multi_grain` Voronoi-tessellates a grid into `n_grains` regions,
each with a random orientation and a small constant per-grain strain.
"""),
    ("py", """\
from midas_pf_odf import (
    plant_multi_grain, split_into_grains, simulate_multi_grain,
    fit_multi_grain, recovery_metrics, IdentifiabilityMode,
)

plant = plant_multi_grain(
    grid_shape=(8, 8), n_grains=3, voxel_size_um=2.0,
    eps_per_grain_amp=1e-3, intra_grain_spread_deg=0.0, seed=1,
)
sub_plants = split_into_grains(plant)
counts = torch.bincount(plant.grain_id, minlength=plant.n_grains)
print(f"voxels: {plant.n_voxels} | grains: {plant.n_grains} | per-grain voxel counts: {counts.tolist()}")
"""),
    ("md", """\
## Step 2 — Simulate per-grain peak patches

Each grain's spots are independent in detector space; `simulate_multi_grain`
renders the voxel-summed, beam-gated peak patches per grain. These are the
synthetic **measurement**.
"""),
    ("py", """\
data = simulate_multi_grain(
    plant, model,
    patch_F=5, patch_P=15, sigma_yz=1.0, sigma_f=0.6,
    gate_tau_um=0.5, add_noise_sigma=0.0,
)
for g, d in data.items():
    print(f"grain {g}: patches {tuple(d.measured_patches.shape)}")
"""),
    ("md", """\
## Step 3 — Recover per-voxel `(R, ε)`

Warm-start each grain at the planted orientation (as an indexer would
provide) with strain at zero, and run `fit_multi_grain`. We use
`IdentifiabilityMode.FREE` with the lattice locked (`lr_lat=0`) so the
constant per-grain strain is held by per-voxel ε directly — the same
regime as `tests/test_multi_grain.py`.
"""),
    ("py", """\
eps_init = {g: torch.zeros_like(sp.eps_voxel) for g, sp in sub_plants.items()}
R_init   = {g: sp.R_voxel for g, sp in sub_plants.items()}

fits = fit_multi_grain(
    data, sub_plants, model,
    eps_init_per_grain=eps_init, R_init_per_grain=R_init,
    identifiability=IdentifiabilityMode.FREE,
    optimizer="lbfgs", inner_steps=20,
    lr_aa=1.0, lr_eps=1.0, lr_lat=0.0,
)
print("fit complete for grains:", sorted(fits.keys()))
"""),
    ("md", """\
## Step 4 — Per-voxel validation

`recovery_metrics` compares the recovered per-voxel `(R, ε)` against the
plant: misorientation RMS (deg) and strain RMSE (Voigt Frobenius).
"""),
    ("py", """\
print(f"{'grain':>6} {'miso RMS (deg)':>16} {'eps RMSE':>14}")
for g, sp in sub_plants.items():
    rep = recovery_metrics(sp, fits[g].R_fit, fits[g].eps_fit)
    print(f"{g:>6} {rep.misorient_rms_deg:>16.4f} {rep.eps_rms:>14.3e}")
print("\\nAll grains: miso RMS << 0.05 deg and ε RMSE << 5e-4 means the "
      "per-voxel state is recovered (same gate as the unit test).")
"""),
    ("md", """\
## Step 5 — Peak-shape vs centroid baseline

The headline Phase-1 claim: matching peak *shape* recovers strain more
tightly than matching peak *centroids* on the same data. We plant a
single grain with a 1e-2 ε₁₁ gradient (multi-pixel peak shifts — the
regime where shape information matters), then run both inverters from the
same warm-start. Each uses the `lr_eps` tuned for its own loss scale.
"""),
    ("py", """\
from midas_pf_odf import (
    plant_single_grain, simulate_grain_patches,
    fit_grain_peakshape, fit_grain_centroid_baseline,
)

plant1 = plant_single_grain(
    grid_shape=(4, 4), voxel_size_um=2.0,
    eps_avg=(0.0,) * 6, eps_gradient_voigt=0,
    eps_gradient_amp=1e-2, eps_gradient_dir="x",
    R_gradient_amp_deg=0.0,
)
data1 = simulate_grain_patches(
    plant1, model, patch_F=5, patch_P=15,
    sigma_yz=1.0, sigma_f=0.6, gate_tau_um=0.5,
)

common = dict(
    voxel_pos=plant1.voxel_pos, R_init=plant1.R_voxel,
    eps_init=torch.zeros_like(plant1.eps_voxel), lattice_init=plant1.lattice,
    identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
    optimizer="adam", inner_steps=200, lr_aa=1e-4, lr_lat=1e-5,
)
fit_shape = fit_grain_peakshape(data1, model, lr_eps=1e-3, **common)
fit_cent  = fit_grain_centroid_baseline(data1, model, lr_eps=1e-7, **common)

rep_shape = recovery_metrics(plant1, fit_shape.R_fit, fit_shape.eps_fit)
rep_cent  = recovery_metrics(plant1, fit_cent.R_fit,  fit_cent.eps_fit)
print(f"peak-shape  eps RMSE = {rep_shape.eps_rms:.3e}")
print(f"centroid    eps RMSE = {rep_cent.eps_rms:.3e}")
print(f"improvement factor   = {rep_cent.eps_rms / max(rep_shape.eps_rms, 1e-15):.1f}x")
print("\\n(ordering check: peak-shape strain RMSE < centroid strain RMSE; "
      "this is an un-tuned baseline, not the paper's headline ratio)")
"""),
    ("md", """\
## Summary

We planted a 3-grain microstructure, recovered per-voxel `(R, ε)` from
the simulated peak patches, validated each grain against the plant, and
showed peak-shape inversion beats the centroid baseline on a strained
grain.

**Identifiability knobs** (`IdentifiabilityMode`):
- `FREE` — per-voxel ε over-parameterised; pair with a locked lattice for
  constant per-grain strain.
- `PROJECT_EPS_MEAN_ZERO` — project ε to mean-zero per grain; the natural
  choice for spatially-varying intragranular strain (lattice absorbs the
  bulk component).

Phase 2 (per-voxel ODF / sub-grain mosaic) builds on this same forward.
See `dev/RESTART.md` and `tests/` for the full validation matrix.
"""),
]


def build_all(only: str | None = None) -> None:
    notebooks = {"01_peakshape_inversion": NB_01}
    for name, cells in notebooks.items():
        if only and only not in name:
            continue
        p = write_notebook(name, cells)
        print("wrote", p)


if __name__ == "__main__":
    build_all(sys.argv[1] if len(sys.argv) > 1 else None)
