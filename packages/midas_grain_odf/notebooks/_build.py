"""Build .ipynb files from a maintainable cell-list source.

    cd packages/midas_grain_odf/notebooks
    python _build.py
    python _build.py 01_odf_round_trip
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
# 01 — Synthetic per-grain ODF round trip
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — `midas-grain-odf`: synthetic per-grain ODF round trip

`midas-grain-odf` recovers a per-grain **orientation distribution
function** (ODF) — the intensity-weighted spread of sub-orientations
inside a single grain — from FF-HEDM spot *shapes*, not just centroids.

The plant-and-recover loop:

1. Plant a known ODF as a few discrete orientation "particles" with
   weights, all near a grain-average orientation `R_avg`.
2. Forward-simulate each particle's spots and splat them, weighted by the
   ODF, into measured spot **patches**.
3. Initialise a `ParticleODF` and run `fit_grain_odf` to recover the
   particle orientations + weights from the patches.
4. Report how much recovered ODF mass lands near the planted particles.

We then look briefly at the three ODF parameterisations
(`ParticleODF`, `BinghamMixtureODF`, `VoxelGridODF`).

Everything is CPU + synthetic. This notebook uses a small particle count
and a short optimisation so it finishes in roughly a minute.
"""),
    ("py", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import torch

torch.manual_seed(7)
np.random.seed(7)
torch.set_default_dtype(torch.float64)
DEG = math.pi / 180.0
print("torch", torch.__version__, "| device: cpu")
"""),
    ("md", """\
## Step 1 — Forward model and a grain-average orientation

We reuse the package's own test scaffolding (`tests/conftest.py`) for the
FCC forward model and a uniformly-random orientation — the exact ground
truth the unit tests use.
"""),
    ("py", """\
import sys
from pathlib import Path
PKG = Path.cwd().parent          # packages/midas_grain_odf
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "tests"))
from conftest import make_model, random_orientation

model = make_model()
R_avg = random_orientation(seed=11).to(torch.float64)
position = torch.zeros(3, dtype=torch.float64)
print("forward model + grain-average orientation ready")
"""),
    ("md", """\
## Step 2 — Plant a known ODF (3 particles in a tight ball)

The planted ODF is three orientations within a ~0.06° ball of `R_avg`,
with asymmetric simplex weights. `axis_angle_to_matrix` turns small
axis-angle deltas into rotation matrices composed onto `R_avg`.
"""),
    ("py", """\
from midas_grain_odf.odf import axis_angle_to_matrix

aa_planted = torch.tensor([
    [0.00 * DEG, 0.00 * DEG, 0.00 * DEG],
    [0.05 * DEG, 0.00 * DEG, 0.00 * DEG],
    [0.00 * DEG, 0.04 * DEG, 0.03 * DEG],
], dtype=torch.float64)
w_planted = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
w_planted = w_planted / w_planted.sum()

R_planted = R_avg.unsqueeze(0) @ axis_angle_to_matrix(aa_planted)   # (3, 3, 3)
print("planted particles:", R_planted.shape[0], "| weights:", w_planted.tolist())
"""),
    ("md", """\
## Step 3 — Synthesise measured spot patches

For each spot that is valid for both `R_avg` and the planted particles,
splat the ODF-weighted predictions into a `(patch_F, patch_P, patch_P)`
intensity patch anchored at the `R_avg`-predicted centre. This mirrors
`tests/test_synth_particle.py::synthesize_measurements`.
"""),
    ("py", """\
from midas_grain_odf.forward_helpers import forward_orientations
from midas_grain_odf.spot_extract import SpotPatchSpec, splat_spots_to_patches

sigma_yz, sigma_f = 1.0, 0.6
patch_F, patch_P = 7, 21          # smaller patch than the test for speed

spots_p = forward_orientations(model, R_planted, position)
sy_p = spots_p.y_pixel.reshape(R_planted.shape[0], -1)
sz_p = spots_p.z_pixel.reshape(R_planted.shape[0], -1)
sf_p = spots_p.frame_nr.reshape(R_planted.shape[0], -1)
sv_p = spots_p.valid.reshape(R_planted.shape[0], -1)

spots_avg = forward_orientations(model, R_avg.unsqueeze(0), position)
sy_a = spots_avg.y_pixel.reshape(-1)
sz_a = spots_avg.z_pixel.reshape(-1)
sf_a = spots_avg.frame_nr.reshape(-1)
sv_a = spots_avg.valid.reshape(-1)

valid_global = (sv_a > 0.5) & (sv_p.sum(dim=0) > 0)
idx = torch.nonzero(valid_global, as_tuple=False).squeeze(-1)

sy_sel, sz_sel = sy_p[:, idx], sz_p[:, idx]
sf_sel, sv_sel = sf_p[:, idx], sv_p[:, idx]
spec = SpotPatchSpec(
    n_spots=int(idx.numel()), patch_F=patch_F, patch_P=patch_P,
    sigma_yz=sigma_yz, sigma_f=sigma_f,
    anchor_y=sy_a[idx].clone(), anchor_z=sz_a[idx].clone(), anchor_f=sf_a[idx].clone(),
)
patches = splat_spots_to_patches(spec, sy_sel, sz_sel, sf_sel, w_planted, sv_sel)

w_norm = (w_planted.reshape(-1, 1) * sv_sel).sum(dim=0).clamp(min=1e-12)
meas_y = (w_planted.reshape(-1, 1) * sv_sel * sy_sel).sum(dim=0) / w_norm
meas_z = (w_planted.reshape(-1, 1) * sv_sel * sz_sel).sum(dim=0) / w_norm
meas_f = (w_planted.reshape(-1, 1) * sv_sel * sf_sel).sum(dim=0) / w_norm
print(f"measured spots: {int(idx.numel())} | patch peak {float(patches.max()):.4f}")
"""),
    ("md", """\
## Step 4 — Fit the ODF back

Initialise a `ParticleODF` with `K=24` particles inside a ±0.15° ball
(deliberately looser and more numerous than planted) and run
`fit_grain_odf`. The fit alternates updating particle axis-angles
(`lr_axis_angle`) and simplex weight logits (`lr_logits`).
"""),
    ("py", """\
from midas_grain_odf.odf import ParticleODF
from midas_grain_odf.inversion import fit_grain_odf

odf = ParticleODF(R_avg=R_avg.clone(), K=24, theta_max=0.15 * DEG, seed=42).to(torch.float64)

result = fit_grain_odf(
    odf=odf, model=model, position=position,
    measured_y=meas_y, measured_z=meas_z, measured_f=meas_f,
    measured_patches=patches, spot_indexer=idx,
    patch_F=patch_F, patch_P=patch_P, sigma_yz=sigma_yz, sigma_f=sigma_f,
    delta_iters=2, inner_steps=200,
    lr_axis_angle=1e-4, lr_logits=0.1, verbose=False,
)
print(f"delta iters run = {result.delta_iters_run} | converged = {result.converged}")
print(f"loss: {result.losses[0]:.3e} -> {result.losses[-1]:.3e} "
      f"(ratio {result.losses[-1] / result.losses[0]:.2e})")
"""),
    ("md", """\
## Step 5 — Intensity-weighted recovery metric

For each recovered particle, check whether it sits within 0.05° of a
planted particle, then sum the recovered simplex **weight** in that
neighbourhood. This is the intensity-weighted recovery: the fraction of
ODF mass placed correctly.
"""),
    ("py", """\
R_rec, w_rec = result.odf.sample()
trace = torch.einsum("kij,pij->pk", R_rec.detach(), R_planted.detach())
angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))         # (P, K)
near = (angle < 0.05 * DEG).any(dim=0).double()                     # (K,)
mass_near = float((w_rec.detach() * near).sum())
print(f"recovered ODF mass within 0.05 deg of a planted particle: {mass_near:.3f}")
print(f"planted weights: {w_planted.tolist()}")
print("\\n(>0.6 means the intensity-weighted ODF is recovered — same gate "
      "as tests/test_synth_particle.py)")
"""),
    ("md", """\
## ODF parameterisations

The package exposes three differentiable ODF families with the *same*
`fit_grain_odf` interface — pick by how you want to model the
intra-grain spread:

- **`ParticleODF`** (used above) — `K` discrete weighted orientations.
  Most flexible; good for arbitrary / multi-modal sub-grain structure.
- **`BinghamMixtureODF`** — a mixture of Bingham distributions on SO(3).
  Compact, smooth, parametric mosaic.
- **`VoxelGridODF`** — a dense weighted grid of orientations in a ball
  around `R_avg`. Non-parametric, highest resolution, most parameters.

All three subclass `GrainODF` and expose `.sample() -> (R, weights)`, so
they are drop-in interchangeable in the inverter.
"""),
    ("py", """\
from midas_grain_odf import ParticleODF, BinghamMixtureODF, VoxelGridODF, GrainODF

for cls in (ParticleODF, BinghamMixtureODF, VoxelGridODF):
    print(f"{cls.__name__:20s}  subclass of GrainODF: {issubclass(cls, GrainODF)}")

# Each yields a (R, w) pair via .sample(); shapes differ by parameterisation.
bingham = BinghamMixtureODF(R_avg=R_avg.clone(), n_modes=2, seed=0).to(torch.float64)
R_b, w_b = bingham.sample()
print("\\nBinghamMixtureODF.sample ->", tuple(R_b.shape), "orientations,",
      tuple(w_b.shape), "weights (sum=%.3f)" % float(w_b.sum()))
"""),
    ("md", """\
## Summary

We planted a 3-particle ODF, rendered ODF-weighted spot patches,
recovered it with a 24-particle `ParticleODF`, and confirmed the
intensity-weighted mass lands on the planted orientations. Swapping in
`BinghamMixtureODF` or `VoxelGridODF` needs only the constructor line.

For the heavier validation cases (larger spreads, asymmetric Bingham,
voxel-grid round trips) see `tests/test_synth_*.py`.
"""),
]


def build_all(only: str | None = None) -> None:
    notebooks = {"01_odf_round_trip": NB_01}
    for name, cells in notebooks.items():
        if only and only not in name:
            continue
        p = write_notebook(name, cells)
        print("wrote", p)


if __name__ == "__main__":
    build_all(sys.argv[1] if len(sys.argv) > 1 else None)
