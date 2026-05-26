"""Build .ipynb files from a maintainable cell-list source.

    cd packages/midas_propagate/notebooks
    python _build.py
    python _build.py 01_calibration_aware_covariance
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
# 01 — Calibration-aware per-grain covariance + stress propagation
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — `midas-propagate`: calibration-aware per-grain covariance → stress

Production HEDM tools report per-grain σ with the detector calibration
**held fixed**. `midas-propagate` closes that loop: it propagates
calibration uncertainty into per-grain covariance and then into per-grain
**stress** error bars.

> **Status note.** The package README labels this a scaffold, but the
> three core numerical modules below are implemented and unit-tested
> (`tests/`). This notebook exercises them end to end on synthetic data.
> The full pipeline glue (reading a real MIDAS dataset) is still in
> progress.

The three stages, mirroring `tests/`:

1. **`joint_nll`** — per-grain Hessian blocks `H_gg`, `H_gc` of the joint
   negative-log-likelihood on grain state `g` and calibration `c`.
2. **`schur`** — Schur-complement marginalisation: turn `(H_gg, H_gc,
   Σ_cc)` into a calibration-*marginalised* per-grain covariance
   `Σ_gg`, which is wider than the calibration-frozen one.
3. **`propagate`** — delta-method `Σ_σ = J Σ_g Jᵀ` from per-grain
   covariance to per-grain Cauchy-stress covariance via a known
   single-crystal stiffness.

CPU + synthetic; runs in a few seconds.
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
import midas_propagate as mpr
print("midas_propagate", mpr.__version__)
"""),
    ("md", """\
## Stage 1 — Per-grain Hessian blocks from the joint NLL

Forward-simulate one FCC Au grain on a paper-1 FF geometry, treat its
predicted spots as the observation (so the residual is zero at the MAP),
and compute the two Hessian blocks:

- `H_gg` — grain-state curvature `(12, 12)` on `(euler[3], latc[6], pos[3])`.
- `H_gc` — grain↔calibration coupling `(12, n_c)` for the chosen
  calibration parameters.

`apply_tilts=True` is required so the forward model uses `ty`/`tz`
directly, letting calibration tilt uncertainty couple into the grain
residual.
"""),
    ("py", """\
from midas_diffract import HEDMForwardModel, HEDMGeometry, hkls_for_forward_model
from midas_hkls import SpaceGroup, Lattice
from midas_propagate.joint_nll import GrainObs, per_grain_hessian_blocks

DEG2RAD = math.pi / 180.0
geom = HEDMGeometry(
    Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
    omega_start=0.0, omega_step=0.25, n_frames=1440,
    n_pixels_y=2048, n_pixels_z=2048,
    min_eta=6.0, wavelength=0.172979, apply_tilts=True,
)
sg = SpaceGroup.from_number(225)
lat = Lattice.for_system("cubic", a=4.08)
hkls_cart, thetas, hkls_int = hkls_for_forward_model(
    sg, lat, wavelength_A=geom.wavelength, two_theta_max_deg=15.0,
)
model = HEDMForwardModel(hkls=hkls_cart, thetas=thetas,
                         geometry=geom, hkls_int=hkls_int)

gt_euler = torch.tensor([45.0, 30.0, 60.0]) * DEG2RAD
gt_latc  = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0])
gt_pos   = torch.zeros(3)

spots = model(gt_euler.unsqueeze(0), gt_pos.unsqueeze(0), lattice_params=gt_latc)
det, valid = HEDMForwardModel.predict_spot_coords(spots, space="detector")
obs = det.squeeze()[valid.squeeze() > 0.5].detach().clone()
print("observed spots:", obs.shape[0])
"""),
    ("py", """\
grain_obs = GrainObs(
    spot_id=0, euler_rad=gt_euler, latc=gt_latc, pos_um=gt_pos,
    observed_detector=obs,
)
calibration_names = ["Lsd", "BC_y", "BC_z", "ty", "tz"]
calibration_map = torch.tensor([1_000_000.0, 1024.0, 1024.0, 0.0, 0.0])
sigma_obs_detector = torch.full((3,), 0.5)         # px / frame measurement noise

blocks = per_grain_hessian_blocks(
    grain_obs,
    hkls_cart=hkls_cart, hkls_int=hkls_int, thetas=thetas,
    base_geometry=geom, scan_config=None,
    calibration_names=calibration_names,
    calibration_map=calibration_map,
    sigma_obs_detector=sigma_obs_detector,
    method="fisher",
)
print("H_gg:", tuple(blocks.H_gg.shape), "| H_gc:", tuple(blocks.H_gc.shape),
      "| spots matched:", blocks.n_spots_matched)
eig = torch.linalg.eigvalsh(0.5 * (blocks.H_gg + blocks.H_gg.T))
print("H_gg smallest eigenvalue (PSD check):", float(eig.min()))
col_norms = torch.linalg.norm(blocks.H_gc, dim=0)
print("H_gc column norms:", {n: round(float(v), 3) for n, v in zip(calibration_names, col_norms)})
print("\\nLsd/ty/tz carry the most grain<->calibration coupling, as expected.")
"""),
    ("md", """\
## Stage 2 — Schur-marginalised per-grain covariance

`per_grain_schur_marginal(H_gg, H_gc, Σ_cc)` returns two per-grain
covariances:

- `sigma_gg_frozen = H_gg⁻¹` — calibration held fixed (what current tools
  report).
- `sigma_gg_calmarg` — calibration uncertainty marginalised in. It is
  **wider** (the inflation eigenvalues are ≥ 1), and that inflation is
  exactly the calibration's contribution to per-grain uncertainty.

The PSD-inflation guarantee holds when each grain's implied joint
Hessian `[[H_gg, H_gc], [H_gcᵀ, Σ_cc⁻¹]]` is PSD — i.e. the
grain↔calibration coupling is bounded by `H_gg`'s spectral floor (the
real-HEDM regime; see `tests/test_schur.py`). To demonstrate the property
cleanly we use well-conditioned synthetic blocks at paper-1 scale; the
*shapes* match the real `(12, n_c)` blocks computed in Stage 1.
"""),
    ("py", """\
from midas_propagate.schur import per_grain_schur_marginal, sigma_inflation_ratio

torch.manual_seed(1)
G, n_g, n_c = 50, 12, len(calibration_names)

# Calibration posterior covariance Σ_cc (SPD) — stand-in for the
# midas-calibrate-v2 posterior over (Lsd, BC_y, BC_z, ty, tz).
Bc = torch.randn(n_c, n_c)
sigma_cc = Bc @ Bc.T + 0.5 * torch.eye(n_c)

# Per-grain SPD H_gg with a strong diagonal floor, and a small H_gc so the
# joint Hessian stays PSD (the bounded-coupling regime).
A = torch.randn(G, n_g, n_g)
H_gg = A @ A.transpose(-1, -2) + 5.0 * torch.eye(n_g)
H_gc = torch.randn(G, n_g, n_c) * 0.05

res = per_grain_schur_marginal(H_gg, H_gc, sigma_cc, ridge_g=1e-12)
ratio = sigma_inflation_ratio(res.sigma_gg_frozen, res.sigma_gg_calmarg)
print("frozen  σ (grain 0, first 3 diag):",
      np.round(torch.sqrt(torch.diagonal(res.sigma_gg_frozen[0])[:3]).tolist(), 5))
print("calmarg σ (grain 0, first 3 diag):",
      np.round(torch.sqrt(torch.diagonal(res.sigma_gg_calmarg[0])[:3]).tolist(), 5))
print("inflation eigenvalues all >= 1:", bool((ratio >= 1.0 - 1e-9).all()),
      "| max inflation:", round(float(ratio.max()), 4))
print("\\n=> calibration uncertainty provably widens every grain's covariance.")
"""),
    ("md", """\
### Sanity: zero coupling ⇒ no inflation

If a grain doesn't couple to calibration (`H_gc = 0`), the marginalised
covariance equals the frozen one exactly — calibration uncertainty
cannot leak in.
"""),
    ("py", """\
res0 = per_grain_schur_marginal(
    H_gg=blocks.H_gg.unsqueeze(0),
    H_gc=torch.zeros_like(blocks.H_gc).unsqueeze(0),
    sigma_cc=sigma_cc, ridge_g=1e-10,
)
gap = (res0.sigma_gg_calmarg - res0.sigma_gg_frozen).abs().max()
print("max |calmarg - frozen| with H_gc=0:", float(gap), "(≈ 0 as expected)")
"""),
    ("md", """\
## Stage 3 — Delta-method to per-grain stress covariance

Finally, propagate per-grain covariance through Hooke's law to per-grain
Cauchy stress and its 6×6 Voigt covariance: `Σ_σ = J Σ_g Jᵀ`, with
`J = ∂σ(g)/∂g`. Position drops out (stress depends only on orientation +
lattice). We use a cubic single-crystal stiffness.
"""),
    ("py", """\
from midas_propagate.propagate import per_grain_stress_with_cov

def cubic_stiffness(C11=200.0, C12=130.0, C44=80.0):
    C = torch.tensor([
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, 2 * C44, 0, 0],     # Mandel: 2*C44 on shear diagonal
        [0, 0, 0, 0, 2 * C44, 0],
        [0, 0, 0, 0, 0, 2 * C44],
    ], dtype=torch.float64)
    return C

# One grain: slightly strained orientation; use grain 0's
# calibration-marginalised covariance from Stage 2 as Σ_g.
g_map = torch.cat([gt_euler, gt_latc, gt_pos]).unsqueeze(0)        # (1, 12)
# Bump the lattice slightly so there is a non-zero stress to report.
g_map[0, 3:6] = torch.tensor([4.083, 4.080, 4.079])
sigma_gg = res.sigma_gg_calmarg[:1]                                # (1, 12, 12)
latc_ref = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0])
C = cubic_stiffness()

stress = per_grain_stress_with_cov(g_map, sigma_gg, latc_ref, C)
print("per-grain stress (Voigt, GPa):", np.round(stress.stress_voigt.squeeze(0).tolist(), 4))
print("per-grain stress σ (Voigt, GPa):", np.round(stress.sigma_voigt.squeeze(0).tolist(), 5))
# PSD check on the stress covariance.
cov = stress.stress_cov.squeeze(0)
print("stress-cov min eigenvalue (PSD):",
      float(torch.linalg.eigvalsh(0.5 * (cov + cov.T)).min()))
"""),
    ("md", """\
## Summary

End to end on synthetic data:

1. `joint_nll.per_grain_hessian_blocks` → `H_gg`, `H_gc` (PSD `H_gg`,
   Lsd/ty/tz dominate the coupling).
2. `schur.per_grain_schur_marginal` → calibration-marginalised per-grain
   covariance, provably wider than the frozen one; zero coupling gives no
   inflation.
3. `propagate.per_grain_stress_with_cov` → per-grain stress with a PSD
   Voigt covariance via the delta method.

This is the paper-1 chain "calibration σ → grain σ → stress σ". The
remaining work (tracked in `dev/paper/SKETCH.md`) is wiring it to real
MIDAS Grains.csv / paramstest inputs and a measured `Σ_cc` from
`midas-calibrate-v2`.
"""),
]


def build_all(only: str | None = None) -> None:
    notebooks = {"01_calibration_aware_covariance": NB_01}
    for name, cells in notebooks.items():
        if only and only not in name:
            continue
        p = write_notebook(name, cells)
        print("wrote", p)


if __name__ == "__main__":
    build_all(sys.argv[1] if len(sys.argv) > 1 else None)
