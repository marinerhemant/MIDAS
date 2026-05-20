"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where
kind is "md" or "py" and source is the markdown / Python source.
Run this script once to (re)generate every .ipynb in this directory.

Why a builder?  Editing raw .ipynb JSON is tedious and the JSON
diffs are unreviewable.  This file is the source of truth; the
.ipynb files are derived artefacts.

Usage:
    cd packages/midas_calibrate/notebooks
    python _build.py             # rebuild all notebooks
    python _build.py 00_getting_started   # rebuild one
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).parent

Cell = Tuple[str, str]    # (kind, source)


def _make_cell(kind: str, source: str, *, idx: int) -> dict:
    src_lines = source.splitlines(keepends=True)
    cell_id = f"cell-{idx:03d}"
    if kind == "md":
        return {
            "id": cell_id,
            "cell_type": "markdown",
            "metadata": {},
            "source": src_lines,
        }
    if kind == "py":
        return {
            "id": cell_id,
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src_lines,
        }
    raise ValueError(f"unknown cell kind {kind!r}")


def write_notebook(name: str, cells: List[Cell]) -> Path:
    nb = {
        "cells": [_make_cell(k, s, idx=i) for i, (k, s) in enumerate(cells)],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (midas_env)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = HERE / f"{name}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path


# =====================================================================
# A shared synthetic-data preamble
# =====================================================================
#
# Every notebook here is SELF-CONTAINED and uses only synthetic data —
# no calibrant TIFFs on disk, no network.  The synthetic CeO2 ring
# image is rendered by the same forward model the package's own
# end-to-end test uses (tests/test_e2e_synthetic.py): a tilted Varex-
# like 1024x1024 detector with bright Gaussian rings on a noisy
# background.  This keeps the notebooks fast (~5-45 s) and reproducible
# on any CPU.

SYNTH_PREAMBLE = """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS OpenMP guard
import numpy as np

from midas_integrate.geometry import build_tilt_matrix, pixel_to_REta
from midas_calibrate import CalibrationParams, build_ring_table


def make_truth() -> CalibrationParams:
    \"\"\"Known-truth CeO2 geometry on a small 1024x1024 detector.\"\"\"
    p = CalibrationParams()
    p.NrPixelsY = 1024; p.NrPixelsZ = 1024
    p.pxY = 200.0; p.pxZ = 200.0
    p.Lsd = 1_000_000.0
    p.BC_y = 512.0; p.BC_z = 512.0
    p.tx = 0.0; p.ty = 0.4; p.tz = 0.25
    p.Wavelength = 0.173
    p.SpaceGroup = 225
    p.LatticeConstant = (5.411, 5.411, 5.411, 90.0, 90.0, 90.0)
    p.MaxRingRad = 480.0
    p.MinRingRad = 0.0
    p.RhoD = 512.0
    p.Width = 1500.0
    p.EtaBinSize = 10.0
    p.RBinSize = 1.0
    p.nIterations = 4
    p.RemoveOutliersBetweenIters = False
    p.SNRMin = 1.5
    p.tolLsd = 5000.0; p.tolBC = 8.0; p.tolTilts = 1.0
    p.tolDistortion = 0.0
    p.Refine = {
        'Lsd': True, 'BC': True, 'ty': True, 'tz': True,
        'Wavelength': False, 'Parallax': False,
        **{f'p{i}': False for i in range(15)},
    }
    return p


def simulate_image(params: CalibrationParams, ring_thickness_px: float = 1.5) -> np.ndarray:
    \"\"\"Render a 2D image: bright Gaussian rings on a noisy background.\"\"\"
    rt = build_ring_table(params)
    NY, NZ = params.NrPixelsY, params.NrPixelsZ
    px = 0.5 * (params.pxY + params.pxZ)
    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)
    Y_grid, Z_grid = np.meshgrid(np.arange(NY, dtype=np.float64),
                                 np.arange(NZ, dtype=np.float64))
    R_pix, _ = pixel_to_REta(
        Y_grid, Z_grid, Ycen=params.BC_y, Zcen=params.BC_z, TRs=TRs,
        Lsd=params.Lsd, RhoD=params.RhoD, px=px, parallax=params.Parallax,
    )
    img = np.full(R_pix.shape, 50.0, dtype=np.float64)
    rng = np.random.default_rng(0)
    img += rng.normal(0, 5.0, size=img.shape)
    for r_ideal in rt.r_ideal_px:
        I_amp = 1000.0 / (1.0 + r_ideal / 100.0)
        img += I_amp * np.exp(-0.5 * ((R_pix - r_ideal) / ring_thickness_px) ** 2)
    return img
"""


# =====================================================================
# NB 00 — Getting started
# =====================================================================

NB_00: List[Cell] = [
    ("md", """\
# 00 — Getting Started with `midas_calibrate`

`midas_calibrate` is the production native-Python/Torch reference
engine for MIDAS detector-geometry calibration.  It replaces the C
chain `AutoCalibrateZarr -> CalibrantIntegratorOMP -> CalibrationCore`
with the same paramstest input format and byte-compatible output,
running on CPU.

This notebook is **fully self-contained** — it renders a synthetic
CeO2 calibrant image, runs the full E↔M alternating engine, inspects
convergence, and writes a refined paramstest.  No data files, no
network.  Runtime ~30-45 s on a CPU.

The engine:

* **E-step** — integrates the image into a 2D (R, η) cake at the
  current geometry, fits each ring's radial peak.
* **M-step** — Levenberg-Marquardt minimisation of the per-(ring, η)
  pseudo-strain residual w.r.t. geometry (Lsd, BC, tilts, distortion).
* Repeat for `nIterations`.
"""),
    ("py", SYNTH_PREAMBLE),
    ("md", """\
## Step 1 — Render the synthetic calibrant image

`make_truth()` defines a known geometry; `simulate_image()` projects
the CeO2 ring table onto a tilted 1024x1024 detector.
"""),
    ("py", """\
truth = make_truth()
image = simulate_image(truth)
print(f'image: shape={image.shape}, min={image.min():.0f}, '
      f'max={image.max():.0f}, mean={image.mean():.0f}')

rt = build_ring_table(truth)
print(f'CeO2 rings in field of view: {len(rt.r_ideal_px)}')
print(f'truth: Lsd={truth.Lsd:.0f} um  BC=({truth.BC_y},{truth.BC_z})  '
      f'ty={truth.ty}  tz={truth.tz}')
"""),
    ("md", """\
## Step 2 — Build a perturbed seed

A real calibration starts from an approximate geometry.  We perturb
the truth: +300 um on Lsd, ~1.5 px on BC, ~0.05 deg on the tilts.
This is the seed handed to `autocalibrate`.
"""),
    ("py", """\
seed = make_truth()
seed.Lsd += 300.0
seed.BC_y += 1.5
seed.BC_z -= 1.0
seed.ty -= 0.05
seed.tz += 0.06
print(f'seed:  Lsd={seed.Lsd:.0f}  BC=({seed.BC_y},{seed.BC_z})  '
      f'ty={seed.ty:.3f}  tz={seed.tz:.3f}')
"""),
    ("md", """\
## Step 3 — Run the full E↔M engine

`autocalibrate(params, image)` returns a `CalibrationResult` with the
refined `params` and a per-iteration `history`.
"""),
    ("py", """\
import time
from midas_calibrate import autocalibrate

t0 = time.time()
result = autocalibrate(seed, image, verbose=False)
elapsed = time.time() - t0

final = result.history[-1]
print(f'elapsed: {elapsed:.1f} s   ({len(result.history)} iterations)')
print(f'final mean strain: {final.mean_strain_uE:.1f} ue')
print(f'fits used in last iter: {final.n_fitted}')
"""),
    ("md", """\
## Step 4 — Inspect convergence

The `history` records strain and geometry at every iteration.  A
healthy calibration shows the pseudo-strain dropping monotonically and
the geometry locking onto the truth.
"""),
    ("py", """\
print(f'{"iter":>4s}  {"strain (ue)":>12s}  {"Lsd (um)":>12s}  '
      f'{"BC_y":>9s}  {"BC_z":>9s}  {"ty":>8s}  {"tz":>8s}  {"n_fit":>6s}')
for h in result.history:
    print(f'{h.iteration:>4d}  {h.mean_strain_uE:>12.1f}  {h.Lsd:>12.1f}  '
          f'{h.BC_y:>9.4f}  {h.BC_z:>9.4f}  {h.ty:>8.4f}  {h.tz:>8.4f}  '
          f'{h.n_fitted:>6d}')
"""),
    ("py", """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

iters = [h.iteration for h in result.history]
strain = [h.mean_strain_uE for h in result.history]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(iters, strain, 'o-', color='C0')
ax.set_xlabel('iteration')
ax.set_ylabel('mean pseudo-strain (ue)')
ax.set_yscale('log')
ax.set_title('midas_calibrate convergence (synthetic CeO2)')
ax.grid(alpha=0.3)
fig.tight_layout()
out_png = 'getting_started_convergence.png'
fig.savefig(out_png, dpi=120)
plt.close(fig)
print(f'wrote {out_png}')
"""),
    ("md", """\
## Step 5 — Recovery vs truth

Compare the refined geometry to the known truth.  Lsd and BC should
recover to sub-pixel / sub-100-um accuracy; the tilts to a few
millidegrees.
"""),
    ("py", """\
p = result.params
print(f'{"param":<8s}  {"truth":>12s}  {"refined":>12s}  {"error":>12s}')
print(f'{"Lsd":<8s}  {truth.Lsd:>12.2f}  {p.Lsd:>12.2f}  {p.Lsd-truth.Lsd:>+12.2f} um')
print(f'{"BC_y":<8s}  {truth.BC_y:>12.4f}  {p.BC_y:>12.4f}  {p.BC_y-truth.BC_y:>+12.4f} px')
print(f'{"BC_z":<8s}  {truth.BC_z:>12.4f}  {p.BC_z:>12.4f}  {p.BC_z-truth.BC_z:>+12.4f} px')
print(f'{"ty":<8s}  {truth.ty:>12.4f}  {p.ty:>12.4f}  {p.ty-truth.ty:>+12.4f} deg')
print(f'{"tz":<8s}  {truth.tz:>12.4f}  {p.tz:>12.4f}  {p.tz-truth.tz:>+12.4f} deg')
"""),
    ("md", """\
## Step 6 — Write the refined paramstest

`result.params.write(path)` writes a paramstest in the same format the
C MIDAS pipeline reads — directly consumable by downstream integration
and FF/NF-HEDM analysis.
"""),
    ("py", """\
import tempfile, os
out_dir = tempfile.mkdtemp(prefix='midas_calib_nb_')
out_path = os.path.join(out_dir, 'calib_refined.txt')
result.params.write(out_path)
print(f'wrote refined paramstest: {out_path}')
print('--- first lines ---')
with open(out_path) as f:
    for line in list(f)[:12]:
        print(line.rstrip())
"""),
    ("md", """\
## What you learned

1. `autocalibrate(params, image)` runs the full alternating E↔M engine.
2. `result.history` carries per-iteration strain + geometry for
   convergence diagnostics.
3. `result.params.write(path)` emits a C-compatible refined paramstest.

See **01_v1_vs_v2_comparison** for a head-to-head against the
differentiable `midas_calibrate_v2` engine on the same synthetic image.
"""),
]


# =====================================================================
# NB 01 — v1 vs v2 comparison
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — `midas_calibrate` (v1) vs `midas_calibrate_v2`

`midas_calibrate` is the **production reference** E↔M engine.
`midas_calibrate_v2` is a fully-differentiable re-implementation that
adds Bayesian (Laplace) uncertainty, multi-panel/multi-distance
support, four-stage refinement, and per-ring basis extension — at the
cost of a different internal parameterisation.

This notebook runs the **same synthetic CeO2 image** through both
engines and compares strain, geometry recovery, and runtime.  It is
self-contained (synthetic data only) and runs in ~10 s on a CPU.
"""),
    ("py", SYNTH_PREAMBLE),
    ("md", """\
## Build the shared problem

One truth geometry, one rendered image, one perturbed seed — fed
identically to both engines.
"""),
    ("py", """\
truth = make_truth()
image = simulate_image(truth)

def make_seed():
    s = make_truth()
    s.Lsd += 300.0; s.BC_y += 1.5; s.BC_z -= 1.0
    s.ty -= 0.05; s.tz += 0.06
    return s

print(f'truth: Lsd={truth.Lsd:.0f}  BC=({truth.BC_y},{truth.BC_z})  '
      f'ty={truth.ty}  tz={truth.tz}')
print(f'image shape: {image.shape}')
"""),
    ("md", """\
## Engine A — v1 `autocalibrate`

The production alternating E↔M engine.
"""),
    ("py", """\
import time
from midas_calibrate import autocalibrate

t0 = time.time()
res_v1 = autocalibrate(make_seed(), image, verbose=False)
t_v1 = time.time() - t0
f1 = res_v1.history[-1]
p1 = res_v1.params
print(f'v1: {t_v1:.1f} s  strain={f1.mean_strain_uE:.1f} ue  '
      f'Lsd={p1.Lsd:.1f}  BC=({p1.BC_y:.3f},{p1.BC_z:.3f})')
"""),
    ("md", """\
## Engine B — v2 `autocalibrate_pv`

The differentiable single-image pipeline (cake + pseudo-Voigt
peak-fit E-step, autograd Levenberg-Marquardt M-step).  We build a v2
spec from the **same** seed and freeze the distortion harmonics (the
synthetic image has no distortion), matching the v1 refine set
(Lsd, BC, ty, tz).
"""),
    ("py", """\
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params
from midas_calibrate_v2.forward.distortion import P_COEF_NAMES
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv

seed_v2 = make_seed()
spec = spec_from_v1_params(seed_v2)
for nm in P_COEF_NAMES:           # freeze distortion to match the v1 refine set
    if nm in spec.parameters:
        spec.parameters[nm].refined = False
        spec.parameters[nm].init = 0.0

t0 = time.time()
res_v2 = autocalibrate_pv(
    seed_v2, image, spec=spec,
    n_iter=3, reuse_fits=True, verbose=False, distribution_report=False,
)
t_v2 = time.time() - t0
f2 = res_v2.history[-1]
print(f'v2: {t_v2:.1f} s  strain={f2.mean_strain_uE:.1f} ue  '
      f'Lsd={f2.Lsd:.1f}  BC=({f2.BC_y:.3f},{f2.BC_z:.3f})')
"""),
    ("md", """\
## Head-to-head

Both engines minimise a per-(ring, η) pseudo-strain residual but have
independent E-step extraction code and M-step parameterisations.  We
compare against the **same** truth.
"""),
    ("py", """\
rows = [
    ('engine',        'v1', 'v2'),
    ('runtime (s)',   f'{t_v1:.1f}', f'{t_v2:.1f}'),
    ('final strain (ue)', f'{f1.mean_strain_uE:.1f}', f'{f2.mean_strain_uE:.1f}'),
    ('Lsd err (um)',  f'{p1.Lsd-truth.Lsd:+.1f}', f'{f2.Lsd-truth.Lsd:+.1f}'),
    ('BC_y err (px)', f'{p1.BC_y-truth.BC_y:+.4f}', f'{f2.BC_y-truth.BC_y:+.4f}'),
    ('BC_z err (px)', f'{p1.BC_z-truth.BC_z:+.4f}', f'{f2.BC_z-truth.BC_z:+.4f}'),
]
w = 22
for label, a, b in rows:
    print(f'{label:<{w}s}  {a:>12s}  {b:>12s}')
"""),
    ("md", """\
## Notes on interpreting the comparison

* **Lsd and beam-centre** recover to sub-pixel / ~sub-100-um accuracy
  in both engines — these are the well-conditioned parameters.
* **Runtime**: v2's `reuse_fits=True` does the cake + pV fit once and
  re-uses fitted positions across LM iterations, so it is typically
  faster per-image than v1's re-extract-every-iter loop on this small
  synthetic.
* **Strain floor** differs between engines because the E-step
  centroiding and the M-step parameterisation (v2 uses a bounded
  Logit reparam + autograd Jacobian) are independent implementations.
  On a clean synthetic both land in the same regime; on real
  distorted detectors v2's residual-correction map and four-stage
  spline (see the v2 notebooks) push the floor lower.
* **Tilt convention**: the synthetic rings are radially symmetric, so
  the tilt is only weakly constrained by ring radius alone; the two
  engines can settle on different (ty, tz) that fit the same rings
  comparably.  For tilt-sensitive validation use an azimuthally
  structured pattern or the v2 `cone_aware_seed` notebook (12).

## When to use which

| Need | Engine |
|---|---|
| Drop-in C-compatible production calibration | **v1** `midas_calibrate` |
| Per-parameter Bayesian σ (Laplace / NUTS) | **v2** |
| Multi-panel (Pilatus/Eiger) or multi-distance | **v2** |
| Four-stage per-ring + TPS-spline residual map | **v2** |
| Joint powder + FF-HEDM calibration | `midas_joint_ff_calibrate` |
"""),
]


# =====================================================================
# Notebook registry
# =====================================================================

NOTEBOOKS = {
    "00_getting_started":    NB_00,
    "01_v1_vs_v2_comparison": NB_01,
}


def main(argv):
    if len(argv) > 1:
        for t in argv[1:]:
            if t not in NOTEBOOKS:
                print(f"unknown notebook: {t}")
                print(f"available: {list(NOTEBOOKS)}")
                return 1
        for t in argv[1:]:
            p = write_notebook(t, NOTEBOOKS[t])
            print(f"wrote {p}")
    else:
        for name, cells in NOTEBOOKS.items():
            p = write_notebook(name, cells)
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
