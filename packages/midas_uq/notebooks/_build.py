"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where
kind is "md" or "py".  Run this script to (re)generate the notebooks.

    cd packages/midas_uq/notebooks
    python _build.py                  # rebuild all
    python _build.py 01_quickstart    # rebuild one
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
# 01 — Quickstart: the four UQ diagnostics on one synthetic grain
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — `midas-uq` Quickstart: four UQ diagnostics on one synthetic grain

`midas-uq` answers, for a refined HEDM grain and *without any ground
truth*:

1. **`half_half`** — how reproducible is the refined state under random
   K-splits of the measured spots? (empirical posterior spread)
2. **`jackknife`** — which individual spots drive the fit, and which look
   corrupted? (leave-one-out influence)
3. **`laplace_covariance`** — what is the local Gaussian posterior from
   the inverse Hessian? (analytic baseline)
4. **`rfree_gap`** — is the refinement overfitting in the low-spots-per-DOF
   regime? (train vs holdout loss)

We **plant** a known FCC grain, forward-simulate its spots with
`midas-diffract`, treat those as the "measurement", perturb the seed,
and run each diagnostic. Everything is CPU + synthetic and runs in well
under a minute.
"""),
    ("py", """\
# midas-diffract / numpy share an OpenMP runtime with torch; allow the
# duplicate load so the import doesn't abort on macOS.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
DEG2RAD = math.pi / 180.0
DEVICE = "cpu"          # this package is CPU-only here
torch.set_default_dtype(torch.float64)
print("torch", torch.__version__, "| device:", DEVICE)
"""),
    ("md", """\
## Step 1 — Build the forward model and plant a grain

We use the same paper-I FF geometry the package tests use: a 2048² panel
at 1 m, 0.25° ω steps, FCC Au (a = 4.08 Å) at 71.7 keV. The reflection
list comes from `midas-hkls` (no C `GetHKLList` dependency).
"""),
    ("py", """\
from midas_diffract import HEDMForwardModel, HEDMGeometry, hkls_for_forward_model
from midas_hkls import SpaceGroup, Lattice

geom = HEDMGeometry(
    Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
    omega_start=0.0, omega_step=0.25, n_frames=1440,
    n_pixels_y=2048, n_pixels_z=2048,
    min_eta=6.0, wavelength=0.172979,
)
sg = SpaceGroup.from_number(225)              # FCC
lat = Lattice.for_system("cubic", a=4.08)     # Au
hkls_cart, thetas, hkls_int = hkls_for_forward_model(
    sg, lat, wavelength_A=geom.wavelength, two_theta_max_deg=15.0,
)
model = HEDMForwardModel(hkls=hkls_cart, thetas=thetas,
                         geometry=geom, hkls_int=hkls_int)

gt_euler = torch.tensor([45.0, 30.0, 60.0]) * DEG2RAD
gt_latc  = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0])
gt_pos   = torch.zeros(3)
print("planted grain:", gt_euler.tolist(), "rad")
"""),
    ("md", """\
## Step 2 — Synthesise the "measured" spots

Forward-simulate the planted grain and keep only the valid spots in
**angular** space `(2θ, η, ω)` in radians — this is exactly the
observation format `half_half`/`jackknife`/`laplace`/`rfree` expect for
FF and pf modes.
"""),
    ("py", """\
spots = model(gt_euler.unsqueeze(0), gt_pos.unsqueeze(0), lattice_params=gt_latc)
ang, valid = HEDMForwardModel.predict_spot_coords(spots, space="angular")
obs = ang.squeeze()[valid.squeeze() > 0.5].detach().clone()
print(f"valid spots (observations): {obs.shape[0]}")
assert obs.shape[0] >= 12, "need enough spots for K-splitting"
"""),
    ("md", """\
## Step 3 — A perturbed seed

A real workflow seeds `midas-uq` from an indexer/Grains.csv row, not the
truth. We mimic that by perturbing the orientation by ~0.5° and the
lattice by ~0.001 Å.
"""),
    ("py", """\
import midas_uq as muq

init = muq.GrainState(
    euler_rad=gt_euler + 0.5 * DEG2RAD * torch.randn(3),
    latc=gt_latc + 1e-3 * torch.randn(6),
    pos=gt_pos,
)
print("seed euler perturbation (deg):",
      np.round(((init.euler_rad - gt_euler) / DEG2RAD).tolist(), 3))
"""),
    ("md", """\
## Diagnostic 1 — `half_half` (K-split reproducibility)

Each of `n_splits` random partitions refines two independent halves of
the spot set; the disagreement between the two refined states is the
empirical UQ. `mode="ff"` (and `"pf"`) use the spot-based path; `"nf"`
would use frames.
"""),
    ("py", """\
hh = muq.half_half(model, init, obs, mode="ff", n_splits=5, phase_steps=(8, 8, 5))
print("per-split misorientation disagreement (deg):",
      np.round(hh.misori_deg, 4))
print(f"misori   median={hh.misori_median_deg:.4f}  p90={hh.misori_p90_deg:.4f} deg")
print(f"lattice  median={hh.lattice_median_A:.2e}  p90={hh.lattice_p90_A:.2e} A")
print("\\nInterpretation: small, tight spread => the refinement is "
      "reproducible under resampling (well-determined grain).")
"""),
    ("md", """\
## Diagnostic 2 — `jackknife` (per-spot influence)

Leave-one-out: refit dropping each spot in turn, measure how far the
refined orientation moves. High-influence spots are leverage points or
corruption candidates — the drill-down on grains `half_half` flags.
"""),
    ("py", """\
jk = muq.jackknife(model, init, obs, mode="ff")
top = jk.top_k(5, by="misori")
print("5 most influential spots (index, influence in deg):")
for i in top:
    print(f"  spot {int(i):3d}:  {jk.influence_misori_deg[int(i)]:.4e} deg")
print(f"\\nmean influence = {jk.influence_misori_deg.mean():.4e} deg  "
      f"(uniform low influence => no single spot dominates the fit)")
"""),
    ("md", """\
## Diagnostic 3 — `laplace_covariance` (Hessian baseline)

The Laplace approximation builds the local Gaussian posterior on
`(euler | latc)` from the inverse Hessian of the NLL at the converged
state. `sigma_vec` is the per-coordinate measurement noise
(σ_2θ, σ_η, σ_ω) in radians. We pass `refine_first=True` so it converges
the perturbed seed before forming the Hessian.
"""),
    ("py", """\
sigma_vec = torch.tensor([0.5, 0.5, 0.5]) * DEG2RAD * 0.1   # ~0.05 deg noise
lp = muq.laplace_covariance(model, init, obs, sigma_vec,
                            refine_first=True, n_mc_samples=1000)
print(f"misori  p95 = {lp.misori_p95_deg:.4f} deg")
print(f"lattice p95 = {lp.lattice_p95_A:.2e} A")
print(f"Hessian condition number = {lp.condition_number:.3e}")
print("\\nInterpretation: a Laplace p95 that disagrees with the "
      "half_half spread is itself a flag for a non-Gaussian / "
      "multi-basin posterior.")
"""),
    ("md", """\
## Diagnostic 4 — `rfree_gap` (overfitting check)

Split the spots into a train and a holdout set, refine on train only, and
track both losses across the three-phase L-BFGS. A holdout loss that
sits far above the train loss (large `gap_final`) means the refinement is
explaining noise — the low-spots-per-DOF overfitting regime.
"""),
    ("py", """\
rf = muq.rfree_gap(model, init, obs, train_fraction=0.5)
print(f"n_train={rf.n_train}  n_hold={rf.n_hold}")
print(f"final train loss   = {rf.train_losses[-1]:.4e}")
print(f"final holdout loss = {rf.holdout_losses[-1]:.4e}")
print(f"R_free-style gap   = {rf.gap_final:.3f}  "
      "(near 0 => generalises; large => overfit)")
"""),
    ("md", """\
## Mode dispatch: `ff` / `pf` / `nf`

`half_half` and `jackknife` are mode-aware. `"ff"` and `"pf"` share the
spot-based implementation (observations are `(N, 3)` angular coords);
`"nf"` switches to the frame-based path (observations are an `(F, H, W)`
image stack). Unknown modes raise.
"""),
    ("py", """\
# ff and pf accept the identical spot observation tensor:
hh_pf = muq.half_half(model, init, obs, mode="pf", n_splits=3, phase_steps=(6, 6, 4))
print("pf-mode half_half median misori (deg):",
      round(hh_pf.misori_median_deg, 4))

try:
    muq.half_half(model, init, obs, mode="garbage")
except ValueError as e:
    print("unknown mode correctly rejected:", str(e)[:60], "...")
"""),
    ("md", """\
## Summary

| Diagnostic | What it surfaces | Cost |
|---|---|---|
| `rfree_gap` | overfitting at low n_obs/DOF | 1× refine |
| `half_half` | noise + model misspec + local-minimum basin | ~10× refine |
| `jackknife` | per-spot leverage / corruption | N_obs × refine |
| `laplace_covariance` | local Gaussian baseline | 1× Hessian |

Recommended population workflow: `half_half` over all grains, then
`jackknife` on the flagged ones, with `laplace_covariance` as a
complementary Gaussian baseline. See `dev/paper/` in the repo for the
Ti-7Al / Park22 population studies.
"""),
]


def build_all(only: str | None = None) -> None:
    notebooks = {"01_quickstart": NB_01}
    for name, cells in notebooks.items():
        if only and only not in name:
            continue
        p = write_notebook(name, cells)
        print("wrote", p)


if __name__ == "__main__":
    build_all(sys.argv[1] if len(sys.argv) > 1 else None)
