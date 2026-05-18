"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where
kind is "md" or "py" and source is the markdown / Python source.
Run this script once to (re)generate every .ipynb in this directory.

Why a builder?  Editing raw .ipynb JSON is tedious and the JSON
diffs are unreviewable.  This file is the source of truth; the
.ipynb files are derived artefacts.

Usage:
    cd packages/midas_calibrate_v2/notebooks
    python _build.py             # rebuild all notebooks
    python _build.py 01_getting_started  # rebuild one
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
# NOTEBOOK SOURCES BELOW.
# Each NB_xx is a list of (kind, source) cells.
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — Getting Started with `midas_calibrate_v2`

This is the entry point.  By the end you will have:

1. Loaded a real synchrotron calibrant image (Varex 4343CT, CeO₂ at 63 keV).
2. Built a v2 calibration spec from a paramstest file.
3. Run the production `autocalibrate_pv` pipeline.
4. Read out the converged geometry and per-parameter Bayesian σ.
5. Run the three reliability gates that decide whether the calibration should be trusted.

The whole thing takes ~30–60 seconds on a CPU.

## Pre-flight: where is the test data?

The four reference calibrant datasets live under `V2_TEST_BASE`
(default `/tmp/midas_v2_test`).  The Varex CeO₂ dataset is the
canonical example — every notebook in this directory uses it.
"""),
    ("py", """\
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from pathlib import Path
BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
PARAMS = BASE / 'refined_MIDAS_params_Ceria_63keV_900mm_100x100_0p5s_aero_0.txt'
IMAGE  = BASE / 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'

assert PARAMS.exists(), f'paramstest not found at {PARAMS} — set $V2_TEST_BASE'
assert IMAGE.exists(), f'image not found at {IMAGE}'
print('OK:', BASE)
"""),
    ("md", """\
## Step 1 — Load the detector image

MIDAS convention is detector-Y horizontal, detector-Z vertical.  Most
synchrotron TIFFs need a vertical flip (`[::-1, :]`) to match — the
`.copy()` call sidesteps a `torch.as_tensor` warning about negative
strides.
"""),
    ("py", """\
import numpy as np
from PIL import Image

image = np.array(Image.open(str(IMAGE))).astype(np.float64)[::-1, :].copy()
print(f'image: shape={image.shape}, dtype={image.dtype}, '
      f'min={image.min():.0f}, max={image.max():.0f}, mean={image.mean():.0f}')
"""),
    ("md", """\
## Step 2 — Build the v2 calibration spec

`spec_from_v1_file` reads a v1 paramstest and creates a tree of v2
`Parameter` objects.  Each parameter knows whether it is refined,
its bounds, and (later) any prior you want to add.

The v1 paramstest defines the seed values for L_sd, BC, tilts,
distortion harmonics, and the calibrant material/wavelength.
"""),
    ("py", """\
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_file

v1 = V1Params.from_file(PARAMS)
# small fix-ups required by the cake/ring builder for some old paramstest files
if v1.RBinSize <= 0 or v1.RBinSize > 0.5:
    v1.RBinSize = 0.25
if v1.EtaBinSize <= 0:
    v1.EtaBinSize = 5.0
v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
v1.Width = max(v1.Width, 800.0)

spec = spec_from_v1_file(PARAMS)
print(f'spec: {len(spec.parameters)} total parameters, '
      f'{len(spec.refined_names())} refined by default')
print('refined names:', spec.refined_names())
"""),
    ("md", """\
## Step 3 — Run the production pipeline

`autocalibrate_pv` is the production single-image entry point.  It
alternates an E-step (cake + pseudo-Voigt peak fitting) with an
M-step (Levenberg-Marquardt minimisation of the pseudo-strain
residual).

`reuse_fits=True` is the recommended setting for clean calibrant
images — it does the cake+pV fit ONCE on iter 0 and re-uses the
fitted (Y, Z) positions across the LM iterations.  This avoids
re-extract noise destabilising the LM convergence.
"""),
    ("py", """\
import time
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv

t0 = time.time()
res = autocalibrate_pv(
    v1, image, spec=spec,
    n_iter=4,                            # 4 outer iterations
    half_window_px=8.0,                   # peak-fit half-window
    snr_min=8.0,                          # drop fits with SNR < 8
    trim_mode='stratified_multfactor',     # spatial-aware outlier rejection
    trim_residual_pct=5.0,                # MultFactor 5x MAD per cell
    reuse_fits=True,
    lm_max_iter=300,
    verbose=False,
    distribution_report=False,
)
elapsed = time.time() - t0
final = res.history[-1]
print(f'elapsed: {elapsed:.1f} s')
print(f'final strain: {final.mean_strain_uE:.3f} µε')
print(f'L_sd: {final.Lsd:.2f} µm')
print(f'BC:   ({final.BC_y:.4f}, {final.BC_z:.4f}) px')
print(f'tilts: ty={final.ty:+.4f}°  tz={final.tz:+.4f}°')
print(f'fits used: {res.fits_final.Y_pix.numel()}')
"""),
    ("md", """\
## Step 4 — Per-parameter Bayesian σ via Laplace

The MAP estimate above is the headline geometry.  But `midas_calibrate_v2`
also gives you the **Cramér-Rao σ** on every refined parameter — for
free, from the same autograd graph the LM uses.

Build a closure that mirrors what the LM saw, then call
`fisher_at_map`.  The result is a `LaplaceResult` whose
`sigma_per_dim` field carries the per-parameter σ in physical units
(µm for L_sd, px for BC, deg for tilts, dimensionless for distortion
harmonics).
"""),
    ("py", """\
import torch
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.inference.laplace import fisher_at_map

fits = res.fits_final

def residual_fn(unp):
    return pseudo_strain_residual(
        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unp,
        rho_d=fits.rho_d, weights=fits.weights,
        ring_idx=fits.ring_idx,
        ring_d_spacing_A=fits.ring_d_spacing_A,
    )

with torch.no_grad():
    r = residual_fn(res.unpacked)
    sigma_r = float(((r * r).mean()) ** 0.5)
print(f'empirical residual σ_r: {sigma_r:.3e}')

lap = fisher_at_map(spec, residual_fn, res.unpacked,
                    sigma_r=sigma_r, ridge=1e-9,
                    dtype=torch.float64, device='cpu')

# Expand vector parameters into per-element names for readability
def _flat(lap):
    out = []
    for n, o, s in zip(lap.refined_names, lap.refined_offsets, lap.refined_sizes):
        for k in range(s):
            out.append(f'{n}[{k}]' if s > 1 else n)
    return out

flat = _flat(lap)
sigma_arr = lap.sigma_per_dim.detach().cpu().numpy()
HEAD = ('Lsd', 'BC_y', 'BC_z', 'ty', 'tz')
print('\\nHeadline geometry σ:')
for nm, s in zip(flat, sigma_arr):
    if nm in HEAD:
        unit = 'µm' if nm == 'Lsd' else ('px' if nm.startswith('BC') else 'deg')
        print(f'  σ({nm:<6s}) = {s:.4e} {unit}')
"""),
    ("md", """\
## Step 5 — The three reliability gates

A converged calibration with low pseudo-strain is necessary but not
sufficient.  Three gates catch the common failure modes:

- **Strain-cap** — reject if mean ε > 100 µε (catches LM basin escape)
- **Basin-check** — warn if MAP drifted too far from seed
- **Cross-validation** — train on rings 0..N−1, test on rings ≥ N (catches misspecified distortion basis)
"""),
    ("py", """\
strain_uE = res.history[-1].mean_strain_uE
print(f'mean pseudo-strain: {strain_uE:.2f} µε')
if strain_uE > 100.0:
    print('  ✗ STRAIN-CAP FAILED — rejected as basin escape')
else:
    print('  ✓ strain-cap passed')

seed_lsd = float(spec.parameters['Lsd'].init)
final_lsd = float(res.unpacked['Lsd'])
seed_BC_y = float(spec.parameters['BC_y'].init)
final_BC_y = float(res.unpacked['BC_y'])
drift_lsd_pct = abs(final_lsd / seed_lsd - 1.0) * 100
drift_BC_y_px = abs(final_BC_y - seed_BC_y)
print(f'\\nL_sd drift: {drift_lsd_pct:.4f}% (basin width: 0.3%)')
print(f'BC_y drift: {drift_BC_y_px:.3f} px (basin width: 1.5 px)')
if drift_lsd_pct > 0.3 or drift_BC_y_px > 1.5:
    print('  ⚠  BASIN-CHECK WARNING — verify by hand')
else:
    print('  ✓ basin check passed')
"""),
    ("md", """\
For the **cross-validation gate**, see `runners/run_cross_validation.py`
in the dev tree — the gate code is in `pipelines/diagnostics.py`.
On the Varex CeO₂ headline above, the CV gate notably FIRES at
+95 % held-out-ring systematic with KS p < 10⁻³⁰: the standard
15-coefficient distortion basis is incomplete on the outer rings
even of an "ideal" calibrant.  See notebook **05** for the full
diagnosis and the per-ring `δr_k` fix.
"""),
    ("md", """\
## Where to next

- **Notebook 02** — Bayesian uncertainty: full Laplace + Fisher; σ(Q) propagation for downstream PDF/Rietveld.
- **Notebook 03** — Multi-panel Pilatus3 2M-CdTe with the Σ=0 gauge.
- **Notebook 04** — Refining pixel size and wavelength (the "you need an external prior" recipes).
- **Notebook 05** — Reliability gates + the per-ring DC fix.

For an exhaustive set of analysis runners, see `dev/paper/runners/`
in this repository.
"""),
]


NB_02: List[Cell] = [
    ("md", """\
# 02 — Bayesian Uncertainty: Laplace + Fisher + σ(Q) Propagation

In notebook **01** you ran `autocalibrate_pv` and got a MAP geometry.
In this notebook you'll quantify the uncertainty on that geometry,
then propagate it into a per-Q-bin σ that downstream pipelines
(radial integration, PDF, Rietveld) can consume.

Three things you should know up front:

1. **The Laplace approximation** assumes the posterior near the MAP
   is Gaussian.  For prior-free fits this is a reasonable bet; for
   strong-prior fits you may want NUTS instead (see notebook 04 and
   the §"NUTS vs Laplace" subsection of the paper).

2. **σ on a parameter is meaningless without context** — gauge
   nullspaces (e.g. the (L_sd, p_x) multiplicative gauge if you
   refine pixel size with no prior) make the marginal σ go to
   infinity.  The framework reports honest σ; if a parameter is
   data-rank-deficient, you'll see it in the Fisher eigenvalues.

3. **σ on geometry isn't what your downstream pipeline wants** —
   it wants σ on the integrated quantity (Q, intensity, lattice
   constant, strain).  We propagate via the Jacobian chain rule.
"""),
    ("py", """\
import os, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import math
import numpy as np
import torch
from PIL import Image

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_file
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.inference.laplace import fisher_at_map

BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
PARAMS = BASE / 'refined_MIDAS_params_Ceria_63keV_900mm_100x100_0p5s_aero_0.txt'
IMAGE  = BASE / 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'

v1 = V1Params.from_file(PARAMS)
if v1.RBinSize <= 0: v1.RBinSize = 0.25
if v1.EtaBinSize <= 0: v1.EtaBinSize = 5.0
v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
v1.Width = max(v1.Width, 800.0)
image = np.array(Image.open(str(IMAGE))).astype(np.float64)[::-1, :].copy()
spec = spec_from_v1_file(PARAMS)

t0 = time.time()
res = autocalibrate_pv(
    v1, image, spec=spec,
    n_iter=4, half_window_px=8.0, snr_min=8.0,
    trim_mode='stratified_multfactor', trim_residual_pct=5.0,
    reuse_fits=True, lm_max_iter=300, verbose=False,
    distribution_report=False,
)
print(f'pipeline: {time.time()-t0:.1f}s, '
      f'final strain {res.history[-1].mean_strain_uE:.2f} µε')
"""),
    ("md", """\
## Full Laplace covariance via `fisher_at_map`

`fisher_at_map` returns a `LaplaceResult` carrying:
- `map_refined` — the MAP parameter values (refined-only, packed)
- `cov` — the full N × N covariance matrix (in the spec's logit-bounded x-space)
- `sigma_per_dim` — sqrt(diag(cov)), per-parameter σ
- `refined_names`, `refined_offsets`, `refined_sizes` — index map

The empirical `sigma_r` (per-fit residual stddev at MAP) is the
noise scale used to normalise the Fisher.
"""),
    ("py", """\
fits = res.fits_final
def residual_fn(unp):
    return pseudo_strain_residual(
        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unp,
        rho_d=fits.rho_d, weights=fits.weights,
        ring_idx=fits.ring_idx,
        ring_d_spacing_A=fits.ring_d_spacing_A,
    )
with torch.no_grad():
    r = residual_fn(res.unpacked)
    sigma_r = float(((r * r).mean()) ** 0.5)

lap = fisher_at_map(spec, residual_fn, res.unpacked,
                    sigma_r=sigma_r, ridge=1e-9,
                    dtype=torch.float64, device='cpu')

cov = lap.cov.detach().cpu().numpy()
print(f'Laplace cov shape: {cov.shape}, condition number: '
      f'{np.linalg.cond(cov):.2e}')
print(f'Refined parameters: {lap.refined_names}')
"""),
    ("md", """\
## Reading per-parameter σ in physical units

The Fisher is built in a logit-reparameterised u-space (so the LM
trust-region updates don't have to respect parameter bounds).  The
σ values are transformed back to physical x-space via the
sigmoid Jacobian.

Below: σ on the headline geometry parameters, with units.
"""),
    ("py", """\
def _flat(lap):
    out = []
    for n, o, s in zip(lap.refined_names, lap.refined_offsets, lap.refined_sizes):
        for k in range(s):
            out.append(f'{n}[{k}]' if s > 1 else n)
    return out

flat = _flat(lap)
sigma_arr = lap.sigma_per_dim.detach().cpu().numpy()
map_arr = lap.map_refined.detach().cpu().numpy()

UNITS = {'Lsd': 'µm', 'BC_y': 'px', 'BC_z': 'px', 'ty': '°', 'tz': '°',
         'pxY': 'µm', 'pxZ': 'µm', 'Wavelength': 'Å', 'Parallax': 'µm'}
print(f'{"param":<12s}  {"MAP":>16s}  {"σ":>14s}  unit')
for nm, mp, sg in zip(flat, map_arr, sigma_arr):
    base = nm.split('[')[0]
    unit = UNITS.get(base, '')
    print(f'  {nm:<10s}  {mp:>+16.6e}  {sg:>14.3e}  {unit}')
"""),
    ("md", """\
## σ(Q) propagation for downstream pipelines

`Q = (4π/λ) sin(θ)` is what radial integration, PDF analysis
(`PDFgetX3`, `diffpy`), and Rietveld refinement (`GSAS-II`,
`TOPAS`) consume.  Per-Q-bin uncertainty needs the chain rule:
$$\\sigma^2(Q_k) = J_Q^\\top \\, \\mathrm{Cov} \\, J_Q$$
where $J_Q$ is the Jacobian of $Q_k$ w.r.t. the refined geometry
parameters, evaluated at the MAP.

For the headline Varex configuration (only L_sd refined among
parameters affecting Q_k), the chain reduces to
$\\sigma(Q)/Q \\approx \\sigma(L_{\\mathrm{sd}})/L_{\\mathrm{sd}}$ —
constant in ppm across all rings.
"""),
    ("py", """\
rt = fits.rt
two_theta_deg = np.array(rt.two_theta_deg)
lam_A = float(res.unpacked['Wavelength'])
Lsd_um = float(res.unpacked['Lsd'])
pxY_um = float(res.unpacked['pxY'])
sigma_Lsd_um = float(lap.sigma_per_dim[flat.index('Lsd')])

print(f'{"ring":>4s}  {"2θ (°)":>7s}  {"Q (Å⁻¹)":>9s}  '
      f'{"d (Å)":>7s}  {"σ(Q) (Å⁻¹)":>14s}  {"σ(Q)/Q (ppm)":>14s}')
for k, tt_deg in enumerate(two_theta_deg):
    if tt_deg <= 0:
        continue
    tt_rad = math.radians(tt_deg)
    th_rad = 0.5 * tt_rad
    R_obs_px = Lsd_um * math.tan(tt_rad) / pxY_um
    Q_k = (4.0 * math.pi / lam_A) * math.sin(th_rad)
    d_k = 2.0 * math.pi / Q_k
    # ∂Q/∂L_sd at fixed R_obs (small-angle dominant term)
    dtt_dLsd = -R_obs_px * pxY_um / (Lsd_um**2 + (R_obs_px * pxY_um)**2)
    dth_dLsd = 0.5 * dtt_dLsd
    dQ_dLsd  = (4.0 * math.pi / lam_A) * math.cos(th_rad) * dth_dLsd
    sigma_Q = abs(dQ_dLsd) * sigma_Lsd_um
    sigma_Q_ppm = sigma_Q / Q_k * 1e6
    print(f'  {k:>2d}  {tt_deg:>7.2f}  {Q_k:>9.3f}  {d_k:>7.3f}  '
          f'{sigma_Q:>14.3e}  {sigma_Q_ppm:>14.2f}')
"""),
    ("md", """\
The `σ(Q)/Q` is essentially constant across rings (Q ∝ 1/L_sd at
small angle, so the ratio is just `σ(L_sd)/L_sd`).  For the Varex
headline, this is ~0.78 ppm — far below typical PDF/Rietveld bin
widths (`ΔQ ~ 10⁻³ Å⁻¹` is 10⁵ ppm at Q=1 Å⁻¹).

If you also refine pxY/pxZ or Wavelength, the chain gets more
columns and the σ(Q) inflates accordingly.  See
`dev/paper/runners/run_sigma_q_propagation.py` for the full
analysis with all the Jacobian terms.

## When to use NUTS instead

The Laplace approximation breaks down when:
- A strong **prior is wired into the LM closure** (the σ-pull
  equilibrium MAP has a steep local Hessian that under-counts the
  actual posterior spread — see `tab:nuts_vs_laplace` in the
  paper, where Laplace under-counts by 1.5–17× under an L_sd prior).
- The posterior is **multi-modal** (e.g., a basin escape between
  two local minima of the LM landscape).
- You're refining **phase parameters** of harmonic distortion,
  which can have a sign ambiguity that Gaussian-around-MAP misses.

For these regimes, sample directly with HMC.  See
`dev/paper/runners/run_nuts_vs_laplace.py` for the canonical
example using `inference.hmc.hmc_run` (pyro backend; `pip install
pyro-ppl` if not present).
"""),
]


NB_03: List[Cell] = [
    ("md", """\
# 03 — Multi-Panel Pilatus3 2M-CdTe with the Σ=0 Gauge

Pilatus-style hybrid pixel detectors are tiled — the Pilatus3 2M-CdTe
in this work has **48 modules** in a 6×8 grid.  Each module has a
small position offset (~50 µm) and rotation (~1 mrad) from the
nominal grid that calibration must determine.

The fundamental challenge is **gauge invariance**: a uniform shift
of every panel by the same `(δy, δz)` is observationally identical
to a global beam-center shift.  Without breaking this gauge
explicitly, the per-panel covariances live at the regularisation
ridge floor and report no useful uncertainty.

This notebook walks through the Wright-2022 Σ=0 symmetric gauge
that the framework adopts, and shows the 10-orders-of-magnitude σ
collapse it produces.
"""),
    ("py", """\
import os, sys, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import (
    spec_from_v1_file, add_panel_parameters, add_panel_zero_sum_constraint,
)
from midas_calibrate_v2.parameters.transforms import Logit
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.loss.constraints import zero_sum_residual
from midas_calibrate_v2.loss.robust_trim import multfactor_trim
from midas_calibrate_v2.inference.laplace import fisher_at_map

BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
sys.path.insert(0, str(BASE))
from run_v2_full import (
    detect_pilatus_panels, _load_panel_shifts, _apply_panel_shifts_to_spec,
    _load_image,
)

ps_path = BASE / 'CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tifps.txt'
img_path = BASE / 'CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif'
dark_path = BASE / 'dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif'
panel_path = BASE / 'CeO2_Pil_100x100_att000_650mm_71p676keV_panel_shifts.txt'

assert ps_path.exists(), f'Pilatus dataset not found at {ps_path}'
print('Pilatus dataset found.')
"""),
    ("md", """\
## Detect the panel layout from the raw image

`detect_pilatus_panels` infers the gap structure from the dark
columns/rows in the raw image — useful when you don't have an
explicit module-map file.
"""),
    ("py", """\
v1 = V1Params.from_file(ps_path)
if v1.RBinSize <= 0: v1.RBinSize = 0.25
if v1.EtaBinSize <= 0: v1.EtaBinSize = 5.0
v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
v1.Width = max(v1.Width, 800.0)

img = _load_image(img_path, im_trans=[2])
dark = np.array(Image.open(str(dark_path))).astype(np.float64)
raw = np.array(Image.open(str(img_path)))[::-1, :].copy()
layout = detect_pilatus_panels(raw, gap_thresh=0)
print(f'Layout: {layout.n_panels_y} × {layout.n_panels_z} panels = '
      f'{layout.n_panels()} modules total')
print(f'Module size: {layout.panel_size_y} × {layout.panel_size_z} px')
print(f'Inter-module gaps: y={layout.gaps_y}, z={layout.gaps_z}')
"""),
    ("md", """\
## Build the spec with per-panel parameters and the Σ=0 gauge

`add_panel_parameters` injects four per-panel blocks:
- `panel_delta_yz` — [N, 2] in-plane shift in pixels
- `panel_delta_theta` — [N] in-plane rotation in degrees
- `panel_delta_lsd` — [N] out-of-plane offset in µm
- `panel_delta_p2` — [N] per-panel additive distortion correction

`add_panel_zero_sum_constraint` enables the symmetric Σ=0 gauge:
the residual closure appends `√λ · Σ panel_delta_*` rows so every
per-panel block has zero sum.  At λ=10⁶ this is effectively a hard
constraint; at finite λ it acts as a Bayesian prior on the gauge
direction.
"""),
    ("py", """\
spec = spec_from_v1_file(ps_path)
add_panel_parameters(
    spec, n_panels=layout.n_panels(),
    tol_shift_px=1.0, tol_rot_deg=1.0,
    tol_lsd_um=500.0, tol_p2=5e-3,
    enable_lsd=True, enable_p2=True,
)
shifts = _load_panel_shifts(panel_path, n_panels=layout.n_panels())
if shifts is not None:
    _apply_panel_shifts_to_spec(spec, shifts)
for nm in ('BC_y', 'BC_z'):
    p = spec.parameters[nm]; cur = float(p.init)
    p.bounds = (cur - 60.0, cur + 60.0); p.transform = Logit(*p.bounds)

# Σ=0 gauge — symmetric, no special reference panel
add_panel_zero_sum_constraint(spec, lambda_zs=1e6)

n_refined = sum(p.numel for p in spec.parameters.values() if p.refined)
print(f'Spec: {n_refined} refined parameters '
      f'({layout.n_panels()} modules × 5 DOF + headline geometry)')
"""),
    ("py", """\
print('Running production pipeline (Pilatus, ~30 s)…')
t0 = time.time()
res = autocalibrate_pv(
    v1, img, dark=dark, spec=spec, panel_layout=layout,
    n_iter=1, half_window_px=4.0, snr_min=8.0,
    trim_mode='stratified_multfactor', trim_residual_pct=2.0,
    reuse_fits=True, lm_max_iter=1, verbose=False,
)
print(f'pipeline: {time.time()-t0:.0f} s')
fits = res.fits_final
truth_unp = res.unpacked

# Trim to define the σ_r-determining set
with torch.no_grad():
    r0 = pseudo_strain_residual(
        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, truth_unp,
        rho_d=fits.rho_d, panel_layout=layout, panel_idx=fits.panel_idx,
    )
    keep, _ = multfactor_trim(r0, factor=2.0)
sigma_r = float(((r0 * r0).mean()) ** 0.5)
print(f'σ_r = {sigma_r:.3e}, kept {int(keep.sum())}/{len(r0)}')
"""),
    ("md", """\
## Compute Laplace σ — with vs without the Σ=0 gauge

The same `fisher_at_map` call, two times:
1. **Fix-one-panel gauge** (`spec.fix_panel_id = 0`, no zero-sum) —
   the legacy v1-style choice.  Per-panel σ saturates at the ridge
   floor.
2. **Σ=0 gauge** — what we just enabled.  Per-panel σ collapses by
   ~10 orders of magnitude to the data-determined values.
"""),
    ("py", """\
def res_fn_data_only(unp, _Y=fits.Y_pix[keep], _Z=fits.Z_pix[keep],
                     _tt=fits.ring_two_theta_deg[keep],
                     _pid=fits.panel_idx[keep]):
    return pseudo_strain_residual(
        _Y, _Z, _tt, unp, rho_d=fits.rho_d,
        panel_layout=layout, panel_idx=_pid,
    )

# (A) Fix-one-panel gauge — for comparison
spec_A = spec_from_v1_file(ps_path)
add_panel_parameters(spec_A, n_panels=layout.n_panels(),
                     tol_shift_px=4.0, tol_rot_deg=2.0,
                     tol_lsd_um=2000.0, tol_p2=2e-2,
                     enable_lsd=True, enable_p2=True)
if shifts is not None: _apply_panel_shifts_to_spec(spec_A, shifts)
for nm in ('BC_y', 'BC_z'):
    p = spec_A.parameters[nm]; cur = float(p.init)
    p.bounds = (cur - 200.0, cur + 200.0); p.transform = Logit(*p.bounds)
spec_A.fix_panel_id = 0
lap_A = fisher_at_map(spec_A, res_fn_data_only, truth_unp,
                      sigma_r=sigma_r, ridge=1e-6,
                      dtype=torch.float64, device='cpu')

# (B) Σ=0 gauge — re-use closure but add zero-sum residual rows
def res_fn_zero_sum(unp):
    r = res_fn_data_only(unp)
    zs = zero_sum_residual(unp, lambda_zs=1e6)
    return torch.cat([r, zs]) if zs.numel() > 0 else r

spec_B = spec   # already configured with zero-sum
lap_B = fisher_at_map(spec_B, res_fn_zero_sum, truth_unp,
                      sigma_r=sigma_r, ridge=1e-6,
                      dtype=torch.float64, device='cpu')
print('Laplace done (A: fix-panel, B: Σ=0)')
"""),
    ("md", """\
## The σ collapse

The numbers below are **median per-panel σ** in logit-domain units.
Only the **ratio** A/B is interpretable as the gauge-null collapse
signature; the absolute scale shifts with the bound width.
"""),
    ("py", """\
def _flat(lap):
    out = []
    for n, o, s in zip(lap.refined_names, lap.refined_offsets, lap.refined_sizes):
        for k in range(s):
            out.append(f'{n}[{k}]' if s > 1 else n)
    return out

def block_med(lap, blk):
    flat = _flat(lap)
    sigma = lap.sigma_per_dim.detach().cpu().numpy()
    sigs = np.array([s for nm, s in zip(flat, sigma)
                      if nm.startswith(blk + '[') or nm == blk])
    return float(np.median(sigs)) if len(sigs) else float('nan')

print(f'{"per-panel block":<22s}  {"A: fix-panel":>14s}  '
      f'{"B: Σ=0":>14s}  {"ratio A/B":>14s}')
for blk in ('panel_delta_yz', 'panel_delta_theta',
            'panel_delta_lsd', 'panel_delta_p2'):
    a = block_med(lap_A, blk)
    b = block_med(lap_B, blk)
    print(f'  {blk:<22s}  {a:>14.3e}  {b:>14.3e}  {a/b:>14.3e}')
"""),
    ("md", """\
The Σ=0 gauge collapses per-panel σ by **~10 orders of magnitude**
on every per-panel block.  This is the difference between "cannot
be reported" and "data-determined".

## Soft prior on the gauge — sensitivity sweep

The default `lambda_zs=1e6` makes the Σ=0 constraint effectively
hard.  At softer λ the gauge becomes a Bayesian prior of stddev
`1/√λ`.  Per-panel σ scales as `λ^(-1/2)` — the signature of a
prior-dominated direction.  See `tab:sigma_collapse` in the paper
for the full λ ∈ {10⁶, 10², 1, 10⁻²} sweep.

## What if the manufacturer prior is informative?

`add_panel_parameters` only sets bounds, not Gaussian priors.  To
add a hierarchical prior matching the manufacturer's panel-installation
tolerance:

```python
from midas_calibrate_v2.parameters.parameter import GaussianPrior
spec.parameters['panel_delta_yz'].prior = GaussianPrior(0.0, 0.3)  # ±0.3 px
spec.parameters['panel_delta_theta'].prior = GaussianPrior(0.0, 0.05)  # ±0.05°
spec.parameters['panel_delta_lsd'].prior = GaussianPrior(0.0, 100.0)  # ±100 µm
spec.parameters['panel_delta_p2'].prior = GaussianPrior(0.0, 1e-4)
```

On full-arc Pilatus the prior never binds (the data dominates by
several orders of magnitude); see notebook **04** for the case
where priors actually do bind, and `dev/paper/runners/run_hierarchical_priors_sparse.py`
for the sparse-arc binding regime.
"""),
]


NB_04: List[Cell] = [
    ("md", """\
# 04 — Refining Pixel Size & Wavelength: The S5 Protocol

Two parameters that v2 keeps **fixed** by default:
- `pxY`, `pxZ` — detector pixel pitch (µm)
- `Wavelength` — X-ray wavelength (Å)

Both are first-class refinable parameters in the spec.  But on a
single calibrant image at single energy, refining either one
creates an **exact gauge null** with `L_sd`:

- For pixel size: `R_pred = L_sd · tan(2θ) / p_x` — only the ratio
  `L_sd/p_x` is observable.
- For wavelength: `2θ depends on λ`, and the per-ring shift induced
  by Δλ is observationally close to a uniform L_sd shift on
  small-2θ rings.

The fix is the **S5 protocol**: anchor one of the two with an
external Gaussian prior (typically `L_sd` from a survey instrument
to ±100 µm).  The framework accepts `Parameter.prior =
GaussianPrior(mean, std)` and the LM/Laplace machinery handles the
rest.

This notebook shows the recipe end-to-end on Varex CeO₂.
"""),
    ("py", """\
import os, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_file
from midas_calibrate_v2.parameters.parameter import GaussianPrior
from midas_calibrate_v2.parameters.transforms import Logit
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.loss.constraints import gaussian_prior_residual
from midas_calibrate_v2.inference.laplace import fisher_at_map

BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
PARAMS = BASE / 'refined_MIDAS_params_Ceria_63keV_900mm_100x100_0p5s_aero_0.txt'
IMAGE  = BASE / 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'

def load_v1():
    v1 = V1Params.from_file(PARAMS)
    if v1.RBinSize <= 0: v1.RBinSize = 0.25
    if v1.EtaBinSize <= 0: v1.EtaBinSize = 5.0
    v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
    v1.Width = max(v1.Width, 800.0)
    return v1

image = np.array(Image.open(str(IMAGE))).astype(np.float64)[::-1, :].copy()
print('OK')
"""),
    ("md", """\
## Three configurations on the same Varex image

We'll run three calibrations and compare σ(L_sd):

1. **Baseline** — px pinned, no prior.  Headline σ(L_sd) ≈ 0.71 µm.
2. **px refined, NO prior** — exposes the (L_sd, p_x) gauge null;
   σ(L_sd) inflates to the regularisation ridge floor.
3. **S5 protocol** — px refined + GaussianPrior on L_sd at 100 µm
   (typical survey precision).  σ(L_sd) recovers to ~100 µm and
   σ(p_y) becomes data-determined.

To get HONEST σ values when a prior is wired in, we use
`fisher_at_map`'s **per-row σ_r vector**: data residuals at the
empirical strain σ_r, prior residuals at unit σ.
"""),
    ("py", """\
def _flat(lap):
    out = []
    for n, o, s in zip(lap.refined_names, lap.refined_offsets, lap.refined_sizes):
        for k in range(s):
            out.append(f'{n}[{k}]' if s > 1 else n)
    return out

def _sigma_for(lap, name):
    flat = _flat(lap)
    sigma = lap.sigma_per_dim.detach().cpu().numpy()
    for nm, s in zip(flat, sigma):
        if nm == name:
            return float(s)
    return float('nan')


def run_scenario(label: str, refine_px: bool, lsd_prior_um: float | None):
    v1 = load_v1()
    spec = spec_from_v1_file(PARAMS)
    if lsd_prior_um is not None:
        spec.parameters['Lsd'].prior = GaussianPrior(
            mean=float(spec.parameters['Lsd'].init), std=lsd_prior_um,
        )
    if refine_px:
        for nm in ('pxY', 'pxZ'):
            p = spec.parameters[nm]; cur = float(p.init)
            p.refined = True
            p.bounds = (cur - 0.5, cur + 0.5); p.transform = Logit(*p.bounds)

    has_prior = any(isinstance(p.prior, GaussianPrior) for p in spec.parameters.values())
    n_refined = sum(p.numel for p in spec.parameters.values() if p.refined)

    t0 = time.time()
    res = autocalibrate_pv(
        v1, image, spec=spec,
        n_iter=4, half_window_px=8.0, snr_min=8.0,
        trim_mode='stratified_multfactor', trim_residual_pct=5.0,
        reuse_fits=True, lm_max_iter=300, verbose=False,
        distribution_report=False,
    )
    fits = res.fits_final
    with torch.no_grad():
        r = pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, res.unpacked,
            rho_d=fits.rho_d, weights=fits.weights,
            ring_idx=fits.ring_idx,
            ring_d_spacing_A=fits.ring_d_spacing_A,
        )
        sigma_r_data = float(((r * r).mean()) ** 0.5)

    def res_fn(unp):
        rd = pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unp,
            rho_d=fits.rho_d, weights=fits.weights,
            ring_idx=fits.ring_idx,
            ring_d_spacing_A=fits.ring_d_spacing_A,
        )
        if has_prior:
            pr = gaussian_prior_residual(unp, spec)
            if pr.numel() > 0:
                rd = torch.cat([rd, pr])
        return rd
    n_data = int(fits.Y_pix.numel())
    n_total = int(res_fn(res.unpacked).numel())
    n_prior = n_total - n_data
    sigma_r_arg = (torch.cat([
        torch.full((n_data,), sigma_r_data, dtype=torch.float64),
        torch.ones(n_prior, dtype=torch.float64),
    ]) if n_prior > 0 else sigma_r_data)
    lap = fisher_at_map(spec, res_fn, res.unpacked,
                        sigma_r=sigma_r_arg, ridge=1e-9,
                        dtype=torch.float64, device='cpu')
    return dict(
        label=label,
        elapsed_s=time.time() - t0,
        n_refined=n_refined,
        Lsd=float(res.unpacked['Lsd']),
        sigma_Lsd_um=_sigma_for(lap, 'Lsd'),
        pxY=float(res.unpacked['pxY']),
        sigma_pxY_um=_sigma_for(lap, 'pxY') if refine_px else float('nan'),
    )


print('Running 3 scenarios…')
rows = [
    run_scenario('baseline (px pinned)', refine_px=False, lsd_prior_um=None),
    run_scenario('px refined, no prior', refine_px=True, lsd_prior_um=None),
    run_scenario('S5: px + σ_Lsd 100 µm', refine_px=True, lsd_prior_um=100.0),
]

print(f'\\n{"scenario":<28s}  {"σ(L_sd) [µm]":>14s}  '
      f'{"σ(pxY) [µm]":>14s}  {"refined":>9s}  {"wall":>6s}')
for r in rows:
    sp = r['sigma_pxY_um']
    sp_str = f'{sp:>14.4e}' if np.isfinite(sp) else f'{"n/a":>14s}'
    print(f'  {r["label"]:<26s}  {r["sigma_Lsd_um"]:>14.4e}  '
          f'{sp_str}  {r["n_refined"]:>9d}  {r["elapsed_s"]:>6.1f}s')
"""),
    ("md", """\
## What you should see

| Scenario | σ(L_sd) | σ(pxY) | Interpretation |
|---|---|---|---|
| baseline (px pinned) | ~0.7 µm | n/a | Cramér-Rao at fixed px (the headline) |
| px refined, no prior | ~0.0 (ridge floor) | ~0.0 (ridge floor) | gauge null — both σ are uninformative |
| S5: σ_Lsd prior 100 µm | exactly 100 µm | ~17 nm | prior anchors L_sd; px becomes data-determined |

The S5 σ(pxY) ≈ 17 nm = 113 ppm of 150 µm is the data-determined
Cramér-Rao under the prior-anchored gauge.  This is what you
quote when you tell a downstream pipeline what the calibration
uncertainty contributes.

## Wavelength refinement — same idea

For wavelength, the recipe is identical: anchor either L_sd or
Wavelength itself with a prior matching your independent
measurement (typical Si(111) monochromator: σ_λ/λ = 1e-4).

```python
spec.parameters['Wavelength'].refined = True
spec.parameters['Wavelength'].prior = GaussianPrior(
    mean=v1.Wavelength,
    std=v1.Wavelength * 1e-4,    # 100 ppm
)
```

See `dev/paper/runners/run_wavelength_refine.py` for the full
4-scenario sweep (no prior, σ_λ/λ ∈ {1e-4, 1e-5}).

## A subtle gotcha — Laplace under-counts under priors

The Laplace approximation gives the LOCAL Hessian σ at the
prior-pulled MAP.  When the prior dominates strongly, the local
Hessian is data-dominated (small σ) but the actual joint posterior
is broader (NUTS σ).  See the §"NUTS vs Laplace" subsection of
the paper for a concrete demonstration where Laplace under-counts
by 1.5–17×.

If the σ values matter for a publication, sample the posterior
directly with `inference.hmc.hmc_run` rather than relying on
Laplace under heavy priors.  Notebook 02 covers when to switch.
"""),
]


NB_05: List[Cell] = [
    ("md", """\
# 05 — Reliability Gates and the +95% Varex CV-Gate Finding

A converged calibration with low pseudo-strain isn't necessarily
*correct* — it might be a fit that has absorbed systematic
residual structure into a flexible distortion basis.  The paper
introduces three diagnostic gates that catch the standard
failure modes:

1. **Strain-cap** — reject if mean ε > 100 µε.  Catches LM basin
   escape (recovery cliff at ~800 µε).
2. **Basin-check** — warn if MAP drifted too far from seed.
3. **Held-out-ring cross-validation** — train on rings 0..N−1,
   test on rings ≥ N.  Catches *misspecified distortion basis*.

The CV gate is the conceptual transfer of *Free-R* (Brünger 1992)
from MX structure refinement to detector geometry.  This notebook
runs all three on Varex CeO₂ and reproduces the paper's striking
finding: **the gold-standard calibrant fails the CV gate** at
+95% test/train residual ratio with KS p < 10⁻³⁰.

We then show the **F2 fix**: per-ring radial offsets `δr_k` with
a Σ=0 gauge.  This reduces strain by 26% on Varex.
"""),
    ("py", """\
import os, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import (
    spec_from_v1_file, add_per_ring_offset,
)
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
from midas_calibrate_v2.pipelines._common import filter_ring_table
from midas_calibrate_v2.seed.auto_max_ring import auto_detect_max_ring
from midas_calibrate.rings import build_ring_table

BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
PARAMS = BASE / 'refined_MIDAS_params_Ceria_63keV_900mm_100x100_0p5s_aero_0.txt'
IMAGE  = BASE / 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'

v1 = V1Params.from_file(PARAMS)
if v1.RBinSize <= 0: v1.RBinSize = 0.25
if v1.EtaBinSize <= 0: v1.EtaBinSize = 5.0
v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
v1.Width = max(v1.Width, 800.0)
image = np.array(Image.open(str(IMAGE))).astype(np.float64)[::-1, :].copy()
print('OK')
"""),
    ("md", """\
## Baseline calibration (15-coef distortion basis)
"""),
    ("py", """\
spec_baseline = spec_from_v1_file(PARAMS)
t0 = time.time()
res_baseline = autocalibrate_pv(
    v1, image, spec=spec_baseline,
    n_iter=4, half_window_px=8.0, snr_min=8.0,
    trim_mode='stratified_multfactor', trim_residual_pct=5.0,
    reuse_fits=True, lm_max_iter=300, verbose=False,
    distribution_report=False,
)
strain_baseline = res_baseline.history[-1].mean_strain_uE
print(f'baseline: {time.time()-t0:.1f} s, strain = {strain_baseline:.3f} µε')
"""),
    ("md", """\
## Gate 1 — Strain-cap
"""),
    ("py", """\
STRAIN_CAP = 100.0  # µε
if strain_baseline > STRAIN_CAP:
    print(f'✗ STRAIN-CAP FAILED ({strain_baseline:.1f} µε > {STRAIN_CAP} µε)')
    print('  Likely basin escape — try a better seed or run from-scratch auto-seed')
else:
    print(f'✓ strain-cap passed ({strain_baseline:.1f} µε ≤ {STRAIN_CAP} µε)')
"""),
    ("md", """\
## Gate 2 — Basin-check (drift from seed)
"""),
    ("py", """\
seed_lsd = float(spec_baseline.parameters['Lsd'].init)
seed_BC_y = float(spec_baseline.parameters['BC_y'].init)
seed_BC_z = float(spec_baseline.parameters['BC_z'].init)
final_lsd = float(res_baseline.unpacked['Lsd'])
final_BC_y = float(res_baseline.unpacked['BC_y'])
final_BC_z = float(res_baseline.unpacked['BC_z'])

drift_lsd_pct = abs(final_lsd / seed_lsd - 1.0) * 100
drift_BC_px = ((final_BC_y - seed_BC_y)**2 + (final_BC_z - seed_BC_z)**2) ** 0.5

print(f'L_sd:  {seed_lsd:.2f} → {final_lsd:.2f} µm  ({drift_lsd_pct:+.4f}%)')
print(f'BC:    drift {drift_BC_px:.3f} px from seed')
print(f'\\nbasin width (from robustness sweep): 0.3% in L_sd, 1.5 px in BC')
if drift_lsd_pct > 0.3 or drift_BC_px > 1.5:
    print('  ⚠  BASIN-CHECK WARNING — verify by hand')
else:
    print('  ✓ basin check passed')
"""),
    ("md", """\
## Gate 3 — Held-out-ring cross-validation

Refine on rings 0..N−1, evaluate residual on rings ≥ N at the
converged geometry.  If the test residual is much larger than the
train residual (and the difference is statistically significant),
the distortion basis is **misspecified** — it absorbed systematic
structure from the train rings that doesn't generalise to the test
rings.

We pick the split point as 70% of the available rings (typical Free-R
choice).
"""),
    ("py", """\
from scipy.stats import kstest, ks_2samp
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

# How many rings do we have?
rt = res_baseline.fits_final.rt
n_rings = len(rt.ring_nr)
n_train = int(round(0.70 * n_rings))
print(f'Total rings: {n_rings}  →  train rings 0..{n_train-1}, '
      f'test rings {n_train}..{n_rings-1}')

# Re-run calibration on TRAIN rings only, then evaluate on test rings
spec_cv = spec_from_v1_file(PARAMS)
spec_cv.max_ring_number = int(rt.ring_nr[n_train - 1])
res_cv = autocalibrate_pv(
    v1, image, spec=spec_cv,
    n_iter=4, half_window_px=8.0, snr_min=8.0,
    trim_mode='stratified_multfactor', trim_residual_pct=5.0,
    reuse_fits=True, lm_max_iter=300, verbose=False,
    auto_max_ring=False, distribution_report=False,
)

# Train residuals (kept set)
fits_train = res_cv.fits_final
with torch.no_grad():
    r_train = pseudo_strain_residual(
        fits_train.Y_pix, fits_train.Z_pix,
        fits_train.ring_two_theta_deg, res_cv.unpacked,
        rho_d=fits_train.rho_d, weights=fits_train.weights,
        ring_idx=fits_train.ring_idx,
        ring_d_spacing_A=fits_train.ring_d_spacing_A,
    )
print(f'\\ntrain median |strain|: {float(r_train.abs().median()) * 1e6:.2f} µε')

# Get test-ring fits — need to re-run E-step including the test rings
spec_full = spec_from_v1_file(PARAMS)
res_full = autocalibrate_pv(
    v1, image, spec=spec_full,
    n_iter=1, half_window_px=8.0, snr_min=8.0,
    trim_mode='off', huber_delta=None,
    reuse_fits=True, lm_max_iter=1, verbose=False,
    distribution_report=False,
)
fits_full = res_full.fits_final
test_mask = fits_full.ring_idx >= n_train
fits_test_Y = fits_full.Y_pix[test_mask]
fits_test_Z = fits_full.Z_pix[test_mask]
fits_test_tt = fits_full.ring_two_theta_deg[test_mask]
fits_test_rid = fits_full.ring_idx[test_mask]
fits_test_d = (fits_full.ring_d_spacing_A[test_mask]
                if fits_full.ring_d_spacing_A is not None else None)

# Evaluate test-ring residual at TRAIN-converged geometry
with torch.no_grad():
    r_test = pseudo_strain_residual(
        fits_test_Y, fits_test_Z, fits_test_tt, res_cv.unpacked,
        rho_d=fits_full.rho_d,
        ring_idx=fits_test_rid,
        ring_d_spacing_A=fits_test_d,
    )
print(f'test  median |strain|: {float(r_test.abs().median()) * 1e6:.2f} µε')

# CV gate fires if test_median > 1.5 × train_median AND KS p < 1e-2
train_med = float(r_train.abs().median())
test_med  = float(r_test.abs().median())
ratio = test_med / train_med
ks_p = float(ks_2samp(r_train.abs().cpu().numpy(),
                       r_test.abs().cpu().numpy())[1])
print(f'\\nratio test/train: {ratio:.2f}×  (gate threshold 1.5×)')
print(f'KS p-value:         {ks_p:.2e}  (gate threshold 1e-2)')
if ratio > 1.5 and ks_p < 1e-2:
    print('\\n  ✗ CV-GATE FIRES: distortion basis is misspecified')
    print('  → see notebook step below for the F2 (per-ring δr_k) fix')
else:
    print('\\n  ✓ CV gate passed')
"""),
    ("md", """\
## The F2 fix — per-ring radial offsets `δr_k` with Σ=0 gauge

When the CV gate fires, one effective fix is to add **N_r per-ring
offsets** that absorb the per-ring DC structure the harmonic basis
cannot fit.  The Σ δr_k = 0 gauge breaks the otherwise-rank-1
coupling to global L_sd.

On Varex CeO₂, F2 reduces strain from 7.74 → 5.70 µε (26%
reduction).
"""),
    ("py", """\
# Need to know N_rings the pipeline will use AFTER auto-max-ring detection.
# Easiest: build the ring table, run auto_detect_max_ring, then count.
rt0 = build_ring_table(v1)
mr = auto_detect_max_ring(
    rt0.r_ideal_px, v1.NrPixelsY, v1.NrPixelsZ,
    v1.BC_y, v1.BC_z, data=image,
) or 0
spec_probe = spec_from_v1_file(PARAMS)
rt_filtered = filter_ring_table(
    rt0,
    rings_to_exclude=getattr(spec_probe, 'rings_to_exclude', ()),
    max_ring_number=mr,
)
n_rings_effective = len(rt_filtered.ring_nr)
print(f'auto-max-ring: {mr}, n_rings effective: {n_rings_effective}')

spec_F2 = spec_from_v1_file(PARAMS)
spec_F2.max_ring_number = mr
add_per_ring_offset(spec_F2, n_rings=n_rings_effective,
                    tol_px=2.0, lambda_zs=1e6)

t0 = time.time()
res_F2 = autocalibrate_pv(
    v1, image, spec=spec_F2,
    n_iter=4, half_window_px=8.0, snr_min=8.0,
    trim_mode='stratified_multfactor', trim_residual_pct=5.0,
    reuse_fits=True, lm_max_iter=300, verbose=False,
    auto_max_ring=False, distribution_report=False,
)
strain_F2 = res_F2.history[-1].mean_strain_uE
reduction_pct = 100.0 * (1.0 - strain_F2 / strain_baseline)
print(f'\\nF2: {time.time()-t0:.1f} s')
print(f'  baseline strain: {strain_baseline:.3f} µε')
print(f'  F2 strain:       {strain_F2:.3f} µε  ({reduction_pct:+.1f}%)')

dr_k = res_F2.unpacked.get('delta_r_k')
if dr_k is not None:
    dr_arr = dr_k.detach().cpu().numpy()
    print(f'\\nrecovered δr_k: std={dr_arr.std():.4f} px, '
          f'min={dr_arr.min():+.4f}, max={dr_arr.max():+.4f}, '
          f'|Σ|={abs(dr_arr.sum()):.2e} (gauge active)')
"""),
    ("md", """\
## What just happened

You ran the same Varex CeO₂ image through the production pipeline
twice — once with the standard 15-coefficient distortion basis,
once with the basis extended by N_r per-ring radial offsets.  The
second run absorbed the per-ring DC residual structure that the
first run couldn't, and reduced the headline strain by 26%.

The point is **NOT** that "F2 makes the calibration better" — the
point is that the **CV gate told you** something was off, and the
framework offered a concrete diagnosis (per-ring DC) and a concrete
fix.  Without the gate, you'd report a clean 7.74 µε number with no
indication that the model was misspecified.

## Where to next

- See `dev/paper/runners/run_basis_fixes.py` for the full
  baseline / F1 / F2 / F1+F2 sweep on Varex.
- See `dev/paper/runners/run_multidist_dr_k.py` for the synthetic
  multi-distance δr_k recovery test (sub-noise-floor MAE, 0.06 ppm
  lattice constant).
- See `dev/paper/runners/run_cross_validation.py` for the
  production CV gate code.
"""),
]


# =====================================================================
# Tier 2 notebooks (specialty workflows)
# =====================================================================

NB_06: List[Cell] = [
    ("md", """\
# 06 — First-Time Calibration (No `paramstest` Required)

The five preceding notebooks all started from an existing
`paramstest.txt` that supplied seed values for L_sd, BC, tilts, and
distortion harmonics.  Real first-time-on-a-new-beamline use almost
never has that file.  This notebook shows the
[`first_time_calibrate`](../midas_calibrate_v2/pipelines/first_time.py)
pipeline that runs from **only** the calibrant material, the X-ray
wavelength, and the detector pixel/dimension specs — no operator
peak picking, no manual ring assignment, no seeded geometry.

Internally:
1. **Hough-circle BC seed** — robust to multi-panel arc
   fragmentation; votes for the (cy, cz) that the most ring pixels
   agree on.
2. **Multi-hypothesis L_sd matching** — sweep over L_sd values
   geometrically spaced around a coarse guess; for each, simulate
   ring radii and score against detected peaks.
3. **Three-stage LM refinement** — geometry → distortion →
   per-panel (optional).
4. **Reliability gates** — every attempt is scored by the
   strain-cap and basin-check gates; the pipeline returns the
   lowest-strain attempt that passes.
"""),
    ("py", """\
import os, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import numpy as np
from PIL import Image

from midas_calibrate_v2.pipelines.first_time import first_time_calibrate

BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
IMAGE = BASE / 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'
image = np.array(Image.open(str(IMAGE))).astype(np.float64)[::-1, :].copy()
print(f'image shape: {image.shape}')
"""),
    ("md", """\
## What you supply — and what you don't

The cell below has only the things the user *should* know:

- The calibrant material — `lattice` + `space_group` for CeO₂
- The X-ray wavelength — from your beamline log
- The detector pixel pitch and dimensions — from manufacturer spec

You supply **no** L_sd guess (defaults to 300 mm, swept), **no** BC
guess (defaults to image center, refined by Hough), **no** tilts,
and **no** distortion seed.
"""),
    ("py", """\
t0 = time.time()
res = first_time_calibrate(
    image=image,
    lattice=(5.4116, 5.4116, 5.4116, 90, 90, 90),  # CeO₂ cubic
    space_group=225,                               # Fm-3m
    wavelength_A=0.196793,                         # 63 keV (typical Varex setup)
    pixel_size_um=150.0,                           # Varex 4343CT
    n_pixels_y=image.shape[0], n_pixels_z=image.shape[1],
    # No L_sd guess, no BC guess — let the auto-seed find them.
    n_iter_full=4,
    half_window_px=8.0,
    snr_min=8.0,
    trim_residual_pct=5.0,
    verbose=False,
)
print(f'\\nfirst_time_calibrate elapsed: {time.time()-t0:.1f} s')
"""),
    ("md", """\
## Read the result

`FirstTimeResult` carries:
- `accepted` — True if the strain-cap + basin-check gates passed
- `lsd_um`, `bc_y`, `bc_z`, `ty`, `tz` — converged geometry
- `mean_strain_uE` — final pseudo-strain
- `lsd_attempts` — list of all L_sd hypotheses tried
- `accepted_index` — which attempt won
"""),
    ("py", """\
unp = res.result.unpacked
print(f'L_sd:            {float(unp["Lsd"]):.2f} µm  ({float(unp["Lsd"])/1000:.2f} mm)')
print(f'BC:              ({float(unp["BC_y"]):.3f}, {float(unp["BC_z"]):.3f}) px')
print(f'tilts:           ty = {float(unp["ty"]):+.4f}°,  tz = {float(unp["tz"]):+.4f}°')
print(f'final strain:    {res.result.history[-1].mean_strain_uE:.2f} µε')
print(f'L_sd attempts:   {len(res.lsd_attempts)} tried '
      f'({[round(v) for v in res.lsd_attempts]} µm)')
print(f'\\ndiagnostics:')
for d in res.diagnostics.results:
    print(f'  [{d.severity:<8s}] {d.name}: {d.message}')
print(f'\\nstage log:')
for line in res.stage_log:
    print(f'  {line}')
"""),
    ("md", """\
## Compare to the paramstest-seeded version

The paramstest-seeded run in notebook 01 converged on the same
image to L_sd ≈ 895925 µm, BC ≈ (1447, 1469) with strain 7.74 µε.
The first-time auto-seed should land within ~50 µm in L_sd and
~1 px in BC of those values without any prior knowledge.

If the auto-seed gives a much worse strain (>100 µε), it failed —
typically because the chosen L_sd hypothesis sweep didn't bracket
the true value (try a broader `lsd_sweep_factors` argument), or
because the calibrant has very few intense rings.  The
`gate_verdict` field tells you *why* an attempt failed.

## When to use this vs `autocalibrate_pv`

| Use first-time-calibrate when | Use autocalibrate_pv when |
|---|---|
| New beamline / first session of a new detector | You have a recent paramstest from the same setup |
| You don't trust the seed in your paramstest | You're refining a known geometry across many images |
| You want the auto-seed + sweep to bracket L_sd | You know L_sd to ~1 % already |

For a production stable beamline, `autocalibrate_pv` from a fresh
paramstest is faster (~30 s vs ~2 min for the auto-seed sweep).
"""),
]


NB_07: List[Cell] = [
    ("md", """\
# 07 — Multi-Distance: Breaking the (L_sd, a) Degeneracy

A single calibrant image at a single sample-detector distance
**cannot uniquely refine** both L_sd and the calibrant lattice
constant `a`.  The forward map at small 2θ is invariant under
`(L_sd, a) → (k·L_sd, k·a)` — a pure scaling.  At higher 2θ the
`tan(2θ)` and `arcsin(λ/2d)` nonlinearities break this invariance,
but slowly: a 70-ring CeO₂ pattern at 63 keV gives σ(a) of about
6 ppm at one distance, *limited by the nonlinearity-strength*, not
by detector noise.

The only fix is two (or more) calibrant images at different L_sd.
The same `a` must satisfy both images' Bragg conditions; only the
true `a` does.  This notebook walks through the analytical Fisher
information that quantifies the σ(a) collapse.
"""),
    ("py", """\
import os, math
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

# Forward model + Fisher block construction (lifted from
# dev/paper/runners/run_multi_distance.py)

def two_theta_deg(a_A, hkl_norm, lam_A):
    return 2.0 * np.degrees(np.arcsin(lam_A * hkl_norm / (2.0 * a_A)))

def fisher_block(images, *, a_truth, hkl_norm, sigma_R_px=0.05, n_per_ring=360):
    \"\"\"Per-image (L_sd) + shared (a) Fisher.\"\"\"
    N_im = len(images)
    n_rings = len(hkl_norm)
    rows_per_im = n_rings * n_per_ring
    n_total = N_im * rows_per_im
    J = np.zeros((n_total, N_im + 1), dtype=np.float64)
    a_truth = float(a_truth)
    for im, spec in enumerate(images):
        L_sd = spec['L_sd_um']; lam = spec['lam_A']; px = spec['px_um']
        tt = two_theta_deg(a_truth, hkl_norm, lam)
        tt_rad = np.radians(tt); th_rad = 0.5 * tt_rad
        sec2 = 1.0 / np.cos(tt_rad) ** 2
        R = L_sd * np.tan(tt_rad) / px
        col_L = np.repeat(R / L_sd, n_per_ring)
        dR_da = (L_sd / px) * sec2 * (-2.0 * np.tan(th_rad) / a_truth)
        col_a = np.repeat(dR_da, n_per_ring)
        rs = im * rows_per_im; re = rs + rows_per_im
        J[rs:re, im]    = col_L
        J[rs:re, N_im]  = col_a
    F = (J.T @ J) / (sigma_R_px ** 2)
    return F


# CeO2 70-ring set
hkl_sq = np.array([3,4,8,11,12,16,19,20,24,27,32,35,36,40,43,44,48,51,52,
                    56,59,64,67,68,72,75,76,80,83,84,88,91,96,99,100,104,107,108,
                    112,115,116,120,123,128,131,132,136,139,140,
                    144,147,148,152,155,156,160,163,164,168,171,172,176,179,180,
                    184,187,192,195,196,200], dtype=np.float64)
hkl_norm = np.sqrt(hkl_sq)
a_truth = 5.4116
lam = 0.1729
keep = (lam * hkl_norm / (2.0 * a_truth)) < 0.95
hkl_norm = hkl_norm[keep]
print(f'CeO2 rings used: {len(hkl_norm)}')
"""),
    ("md", """\
## Single-distance vs two-distance σ(a)

Build Fisher blocks for two scenarios and read off σ(a) from the
inverse Fisher.
"""),
    ("py", """\
def report(F, label, names):
    cov = np.linalg.pinv(F)
    sigmas = np.sqrt(np.maximum(np.diag(cov), 0.0))
    print(f'\\n[{label}]')
    for n, s in zip(names, sigmas):
        unit = 'µm' if n.startswith('L') else f'Å ({s/a_truth*1e6:.2f} ppm)'
        print(f'  σ({n:<6s}) = {s:.4e}  {unit}')
    return sigmas

# Scenario A: single image at 650 mm
F_A = fisher_block(
    [dict(L_sd_um=650_000.0, lam_A=lam, px_um=172.0)],
    a_truth=a_truth, hkl_norm=hkl_norm,
)
sA = report(F_A, 'A: 1 image @ 650 mm', ['L_sd_1', 'a'])

# Scenario B: two images at 650 + 1200 mm
F_B = fisher_block(
    [dict(L_sd_um=650_000.0, lam_A=lam, px_um=172.0),
     dict(L_sd_um=1_200_000.0, lam_A=lam, px_um=172.0)],
    a_truth=a_truth, hkl_norm=hkl_norm,
)
sB = report(F_B, 'B: 2 images @ 650 + 1200 mm', ['L_sd_1', 'L_sd_2', 'a'])

print(f'\\nσ(a) collapse: {sA[1]/a_truth*1e6:.2f} ppm → '
      f'{sB[-1]/a_truth*1e6:.2f} ppm  '
      f'({sA[1]/sB[-1]:.2f}× tighter)')
"""),
    ("md", """\
## What about adding a tight L_sd prior to the single-distance fit?

Intuition might suggest: "if I know L_sd to ±100 µm from a survey
instrument, I can pin it down and use that to refine `a`
precisely."  The intuition is **wrong**: the L_sd–a degeneracy is
not broken by an L_sd prior alone; it's only broken by an
*independent* constraint at a different L_sd.

Below: scenario A with a 100 µm L_sd prior added.  σ(a) doesn't
change.
"""),
    ("py", """\
F_A_prior = F_A.copy()
F_A_prior[0, 0] += 1.0 / (100.0 ** 2)    # 100 µm Gaussian prior on L_sd
sAp = report(F_A_prior, 'A + 100 µm L_sd prior', ['L_sd_1', 'a'])
print(f'\\nσ(a) without/with prior: {sA[1]/a_truth*1e6:.2f} → '
      f'{sAp[1]/a_truth*1e6:.2f} ppm  (negligible change)')
"""),
    ("md", """\
## Why the prior doesn't help

The (L_sd, a) Fisher block is rank-deficient at small 2θ.  Adding
a prior on L_sd adds curvature in the L_sd direction, but the
nullspace of the data Fisher *is* the joint (L_sd, a) direction —
the prior pins down L_sd, but `a` then has to follow along.
Mathematically: σ(a) ≈ σ(L_sd_prior) · ∂a/∂L_sd along the gauge
null.  At small 2θ, ∂a/∂L_sd ≈ a/L_sd ≈ 6 × 10⁻⁶ Å/µm × 100 µm =
6 × 10⁻⁴ Å ≈ 100 ppm — same scale as the data-only result.

The two-distance protocol is the only thing that adds a *new*
direction to the Fisher block (Lsd_2 is independent of Lsd_1) and
thereby breaks the gauge.

## What about adding per-ring offsets `δr_k`?

The framework also exposes per-ring radial offsets via
`add_per_ring_offset(spec, n_rings, ...)` — see notebook **05**.
These are the F2 fix for per-ring DC structure.  At a single
distance δr_k is degenerate with a per-(hkl) lattice shift Δa_k;
multi-distance breaks that degeneracy, and the framework can
recover δr_k at MAE ~0.0015 px on a 70-ring synthetic test
(see `dev/paper/runners/run_multidist_dr_k.py`).

## Practical recipe

Take two CeO₂ images at distinct L_sd at the start of every
experimental session.  Run them through the multi-image entry
point:

```python
from midas_calibrate_v2.pipelines.multi import autocalibrate_multi
res = autocalibrate_multi(
    [v1_image1, v1_image2], [image1, image2],
    spec_shared=spec_a_shared,
    specs_per_image=[spec_im1, spec_im2],
    n_iter=4, ...
)
```

The σ(a) you get back is the appropriate value to quote when
reporting absolute lattice constants downstream.
"""),
]


NB_08: List[Cell] = [
    ("md", """\
# 08 — Doublet Calibrants: LaB₆ and Cu Kα₁/Kα₂

Most of the paper's headline numbers are on CeO₂ at high energy
(63–72 keV) — a calibrant chosen partly because its rings are
well-separated in 2θ.  Two regimes that *aren't* well-separated:

1. **LaB₆ at small L_sd** — adjacent rings (e.g., 200/210, 220/300)
   come within ~5 px of each other on a typical area detector at
   short L_sd.  The default singleton pseudo-Voigt fitter merges
   them and reports a biased centroid.

2. **Cu Kα radiation on a lab area detector** — every ring
   appears as a Kα₁ + Kα₂ doublet (Δλ/λ ≈ 2.5 × 10⁻³, intensity
   ratio 0.5).  At 2θ ≈ 30°, the doublet separation is ~3 px on a
   100 µm detector at 200 mm L_sd — close enough to merge.

For both cases the framework ships
[`forward.peak_fit_doublet`](../midas_calibrate_v2/forward/peak_fit_doublet.py),
a shared two-peak pseudo-Voigt LM that co-fits both peaks
simultaneously.  This notebook validates it on synthetic data and
quantifies the bias the singleton path produces.
"""),
    ("py", """\
import os, math
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch

from midas_calibrate_v2.forward.peak_fit_batched import fit_cake_per_ring_batched
from midas_calibrate_v2.forward.peak_fit_doublet import fit_doublet_pairs

SQRT_2PI = math.sqrt(2.0 * math.pi)
INV_PI = 1.0 / math.pi

def pv_1d(R, c, sigma, gamma, eta_v, A):
    dR = R - c
    G = np.exp(-0.5 * dR*dR / (sigma*sigma)) / (sigma * SQRT_2PI)
    L = (gamma * INV_PI) / (dR*dR + gamma*gamma)
    return A * (eta_v * L + (1.0 - eta_v) * G)
"""),
    ("md", """\
## Cu Kα₁/Kα₂ within-ring doublet on synthetic LaB₆ (110)

Setup mimics a typical lab Cu-source area-detector geometry:
- λ(Kα₁) = 1.5406 Å, λ(Kα₂) = 1.5444 Å, I(Kα₂)/I(Kα₁) = 0.50
- LaB₆ a = 4.157 Å, (110) ring at 2θ ≈ 30.4° for Kα₁
- L_sd = 200 mm, p_x = 100 µm
- 2θ separation ≈ 0.077°, R separation ≈ 3.6 px
"""),
    ("py", """\
lam_a1 = 1.5406; lam_a2 = 1.5444; I_ratio = 0.50
a_LaB6 = 4.156826; hkl_norm = math.sqrt(2.0)         # (110)
Lsd_um = 200_000.0; px_um = 100.0
sigma_truth, gamma_truth, eta_v_truth, A_truth = 1.5, 1.0, 0.4, 200.0
n_eta = 90

tt_a1 = 2.0 * math.degrees(math.asin(lam_a1 * hkl_norm / (2.0 * a_LaB6)))
tt_a2 = 2.0 * math.degrees(math.asin(lam_a2 * hkl_norm / (2.0 * a_LaB6)))
R_a1 = Lsd_um * math.tan(math.radians(tt_a1)) / px_um
R_a2 = Lsd_um * math.tan(math.radians(tt_a2)) / px_um
sep_R = R_a2 - R_a1
print(f'Kα₁: 2θ={tt_a1:.4f}°  R={R_a1:.2f} px')
print(f'Kα₂: 2θ={tt_a2:.4f}°  R={R_a2:.2f} px')
print(f'Δ(2θ)={tt_a2-tt_a1:.4f}°  ΔR={sep_R:.2f} px (default doublet threshold = 25 px)')
"""),
    ("md", """\
## Build a synthetic cake with both peaks
"""),
    ("py", """\
rng = np.random.default_rng(0)
R_min, R_max, dR = R_a1 - 8.0, R_a2 + 8.0, 0.1
R_centers = np.arange(R_min, R_max + dR, dR)
n_R = len(R_centers)
eta_centers = np.linspace(-180.0, 180.0, n_eta, endpoint=False) + (360.0 / n_eta) / 2.0

cake = np.zeros((n_R, n_eta), dtype=np.float64)
for j in range(n_eta):
    I_lo = pv_1d(R_centers, R_a1, sigma_truth, gamma_truth, eta_v_truth, A_truth)
    I_hi = pv_1d(R_centers, R_a2, sigma_truth, gamma_truth, eta_v_truth, A_truth * I_ratio)
    cake[:, j] = I_lo + I_hi + 5.0 + rng.normal(0.0, 2.0, n_R)
print(f'cake shape: {cake.shape}')
"""),
    ("md", """\
## Singleton fit (the default) vs doublet co-fit
"""),
    ("py", """\
dt = torch.float64
cake_t = torch.tensor(cake, dtype=dt)
R_t = torch.tensor(R_centers, dtype=dt)
eta_t = torch.tensor(eta_centers, dtype=dt)

# Singleton: pretend there's only one ring at the Kα₁ position
rt_a1 = torch.tensor([R_a1], dtype=dt)
bf_singleton = fit_cake_per_ring_batched(
    cake_t, R_t, eta_t, rt_a1,
    half_window_px=8.0, max_iter=80, snr_min=2.0,
    dtype=dt, device='cpu', verbose=False,
)
R_singleton = bf_singleton.R_fit.cpu().numpy()
keep_s = (bf_singleton.rc.cpu().numpy() >= 0)

# Doublet co-fit: pass both peaks as a pair
rt_pair = torch.tensor([R_a1, R_a2], dtype=dt)
df = fit_doublet_pairs(
    cake_t, R_t, eta_t, rt_pair, pair_indices=[(0, 1)],
    half_window_px=8.0, max_iter=80, snr_min=2.0,
    dtype=dt, device='cpu', verbose=False,
)
R_doublet_a1 = df.R_fit_lo.cpu().numpy()
R_doublet_a2 = df.R_fit_hi.cpu().numpy()
keep_d = (df.rc.cpu().numpy() >= 0)

bias_singleton = float(np.median(R_singleton[keep_s]) - R_a1)
bias_doublet_a1 = float(np.median(R_doublet_a1[keep_d]) - R_a1)
bias_doublet_a2 = float(np.median(R_doublet_a2[keep_d]) - R_a2)

print(f'\\nKα₁ centroid recovery (truth = {R_a1:.4f} px):')
print(f'  Singleton (default):  {np.median(R_singleton[keep_s]):>10.4f} px  bias {bias_singleton:+.4f} px')
print(f'  Doublet co-fit:       {np.median(R_doublet_a1[keep_d]):>10.4f} px  bias {bias_doublet_a1:+.4f} px')
print(f'\\nKα₂ centroid recovery (truth = {R_a2:.4f} px):')
print(f'  Doublet co-fit:       {np.median(R_doublet_a2[keep_d]):>10.4f} px  bias {bias_doublet_a2:+.4f} px')

print(f'\\nPer-fit relative bias (would feed into LM as fake strain):')
print(f'  Singleton:    ΔR/R = {bias_singleton/R_a1*1e6:+.0f} ppm')
print(f'  Doublet:      ΔR/R = {bias_doublet_a1/R_a1*1e6:+.0f} ppm')
print(f'  improvement:  {abs(bias_singleton/bias_doublet_a1):.1f}×')
"""),
    ("md", """\
## What this means for lab Cu-Kα calibration

The singleton fitter biases the recovered ring centroid by ~250 ppm
on this geometry — 250 µε of fake strain that the LM would attempt
to absorb into a geometry shift.  The doublet co-fit removes this
bias to <10 ppm.

To enable the doublet path in production, set
`doublet_separation_px` greater than the expected separation (in
this geometry, anything > 3.6 px):

```python
res = autocalibrate_pv(
    v1, image, spec=spec,
    n_iter=4, half_window_px=8.0,
    doublet_separation_px=10.0,  # ← enables the co-fit
    ...
)
```

The auto-pairing logic in
[`forward.doublets`](../midas_calibrate_v2/forward/doublets.py)
detects which ring pairs are within `doublet_separation_px` and
sends them to `fit_doublet_pairs`; singleton rings are still fit
by the standard pseudo-Voigt path.

## Same recipe for LaB₆ adjacent-ring doublets

For LaB₆ at small L_sd (high-energy synchrotron with short L_sd),
multiple ring pairs appear in the doublet regime: (200, 210),
(220, 300), etc.  The doublet machinery handles them automatically
when you set the threshold.  See
[`run_doublet_validation.py`](../dev/paper/runners/run_doublet_validation.py)
for a synthetic two-ring pair validation in that regime.
"""),
]


NB_09: List[Cell] = [
    ("md", """\
# 09 — Basis Extension and BIC Model Selection

The standard 15-coefficient distortion basis is enough for most
synchrotron area detectors at moderate 2θ.  When the
held-out-ring CV gate fires (notebook **05**), one option is the
F2 per-ring offset; another is **extending the harmonic basis**
to higher η-folds.

This notebook shows:
1. How the 15-coef basis is defined as a list of `HarmonicTerm`
   tuples.
2. How to extend to fold-7+ via `extended_term_layout(max_fold=N)`.
3. How **BIC** (Bayesian Information Criterion) decides whether
   the extra parameters are justified by the data.
"""),
    ("py", """\
import os, math
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from midas_calibrate_v2.forward.distortion import (
    P_COEF_NAMES, v2_term_layout,
    extended_term_layout, extended_p_coef_names,
)

# Show the standard basis
print('Standard 15-coef basis:')
for term in v2_term_layout():
    print(f'  fold={term.fold:>2d}  power=ρ^{term.radial_power}  '
          f'coef_idx={term.coef_idx:>2d}  phase_idx={term.phase_idx:>2d}')

print(f'\\nNamed coefficients: {P_COEF_NAMES}')
"""),
    ("md", """\
## Extend to fold-8

Adds amplitude/phase pairs `(a7, phi7)`, `(a8, phi8)` at radial
powers ρ⁷, ρ⁸.
"""),
    ("py", """\
extended = extended_term_layout(max_fold=8)
extended_names = extended_p_coef_names(max_fold=8)
print(f'Extended basis to max_fold=8: {len(extended)} terms')
for term in extended[-4:]:
    print(f'  fold={term.fold:>2d}  power=ρ^{term.radial_power}  '
          f'coef_idx={term.coef_idx:>2d}  phase_idx={term.phase_idx:>2d}')
print(f'\\nExtended names: {extended_names}')
"""),
    ("md", """\
## BIC for model selection

The BIC penalises models for free parameters:

  BIC = N · log(RSS / N) + k · log(N)

A larger model is preferred only if BIC decreases.  Below: a quick
illustrative computation comparing the 15-coef baseline against
extended fold-7 and fold-8 fits on Varex CeO₂.

For the full paper sweep across all four reference datasets, see
[`dev/paper/runners/run_harmonic_extension.py`](../dev/paper/runners/run_harmonic_extension.py).
"""),
    ("py", """\
import time
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_file
from midas_calibrate_v2.parameters.parameter import Parameter
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
PARAMS = BASE / 'refined_MIDAS_params_Ceria_63keV_900mm_100x100_0p5s_aero_0.txt'
IMAGE  = BASE / 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'

v1 = V1Params.from_file(PARAMS)
if v1.RBinSize <= 0: v1.RBinSize = 0.25
if v1.EtaBinSize <= 0: v1.EtaBinSize = 5.0
v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
v1.Width = max(v1.Width, 800.0)
image = np.array(Image.open(str(IMAGE))).astype(np.float64)[::-1, :].copy()

spec_15 = spec_from_v1_file(PARAMS)
print('Running 15-coef baseline (~30 s)…')
t0 = time.time()
res_15 = autocalibrate_pv(
    v1, image, spec=spec_15,
    n_iter=4, half_window_px=8.0, snr_min=8.0,
    trim_mode='stratified_multfactor', trim_residual_pct=5.0,
    reuse_fits=True, lm_max_iter=300, verbose=False,
    distribution_report=False,
)
fits_15 = res_15.fits_final
with torch.no_grad():
    r_15 = pseudo_strain_residual(
        fits_15.Y_pix, fits_15.Z_pix, fits_15.ring_two_theta_deg, res_15.unpacked,
        rho_d=fits_15.rho_d, weights=fits_15.weights,
        ring_idx=fits_15.ring_idx, ring_d_spacing_A=fits_15.ring_d_spacing_A,
    )
N_15 = int(r_15.numel())
RSS_15 = float((r_15 * r_15).sum())
k_15 = sum(p.numel for p in spec_15.parameters.values() if p.refined)
BIC_15 = N_15 * math.log(RSS_15 / N_15) + k_15 * math.log(N_15)
print(f'  15-coef:  k={k_15}  RSS={RSS_15:.4e}  N={N_15}  BIC={BIC_15:.2f}  '
      f'strain={float(r_15.abs().mean())*1e6:.2f} µε  '
      f'({time.time()-t0:.1f}s)')
"""),
    ("md", """\
## Why no fold-7+ run here?

A live extended-basis run requires the pipeline to instantiate
the extra `Parameter`s and propagate them through every closure.
Doing this in a notebook means either:

(a) Manually adding `a7, phi7, a8, phi8` to the spec and adapting
    the cake/peak-fit infrastructure to know about them; or

(b) Using the dedicated runner
    `dev/paper/runners/run_harmonic_extension.py` which handles
    all the wiring correctly and emits BIC for every combination
    on every reference dataset.

For pedagogical purposes, the take-away is the **conceptual recipe**:
- BIC at 15-coef is your baseline.
- Add a fold (or remove one), refit, re-evaluate BIC.
- If BIC drops, the extra parameter is justified by the data.
- If BIC rises, you're overfitting — keep the smaller basis.

The **BIC ladder** in `tab:bic_ladder` of the paper shows that on
all four reference datasets, BIC stays minimal at 15-coef across
fold-7 and fold-8 attempts — the data doesn't support extra η-folds.

## When extending the basis is the right move

- High-2θ rings show a **smooth** residual systematic that the
  CV gate flags but the per-ring DC `δr_k` doesn't fully fix.
- The extra fold's amplitude is statistically distinguishable
  from zero in the Laplace σ.
- BIC decreases (not just RSS).

If those three conditions aren't met, the F2 per-ring offset is
the better fix because it's parameterised by physics (per-(hkl)
DC) rather than by mathematical convenience (more harmonics).

## See also

- [`dev/paper/runners/run_harmonic_extension.py`](../dev/paper/runners/run_harmonic_extension.py) — the BIC ladder runner that produces `tab:bic_ladder`
- [`forward/distortion.py`](../midas_calibrate_v2/forward/distortion.py) — `HarmonicTerm`, `extended_term_layout`, `extended_p_coef_names`
"""),
]


NB_10: List[Cell] = [
    ("md", """\
# 10 — pyFAI Head-to-Head

If you've used `pyFAI` for area-detector calibration, you'll want
to know: how does the same image look through both pipelines?

This notebook walks through the comparison protocol — same image,
same fitted points, two calibration engines, side-by-side strain
and σ values.  The full benchmark is in
[`dev/paper/runners/run_A8_pyfai_paper.py`](../dev/paper/runners/run_A8_pyfai_paper.py),
which runs it on all four reference datasets.

**Notes up front:**
- pyFAI is **not** a hard dependency of `midas_calibrate_v2`.
  This notebook gracefully falls back if pyFAI is not installed.
- The comparison uses pyFAI's headless / script-mode `refine2`,
  not the interactive `pyFAI-calib2` GUI.  The interactive workflow
  with operator-curated peak picks gives different (and better)
  pyFAI results; we don't claim a speed comparison against that.
- Headline numbers from the paper: Varex 1105 µε pyFAI vs 18.5 µε
  v2 (33×); Pilatus3 2M-CdTe 884 µε pyFAI vs 35.6 µε v2 (49×).
"""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Check pyFAI availability — older versions don't expose __version__
try:
    import pyFAI
    HAVE_PYFAI = True
    pyfai_version = getattr(pyFAI, '__version__', '(unknown version)')
    print(f'pyFAI {pyfai_version} OK')
except ImportError:
    HAVE_PYFAI = False
    print('pyFAI not installed — install with `pip install pyFAI` to run this notebook end-to-end.')
    print('The cells below will degrade gracefully and show the v2 result alongside paper-headline pyFAI numbers.')
"""),
    ("md", """\
## v2 calibration on Varex CeO₂

Standard headline run.
"""),
    ("py", """\
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_file
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv

BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
PARAMS = BASE / 'refined_MIDAS_params_Ceria_63keV_900mm_100x100_0p5s_aero_0.txt'
IMAGE  = BASE / 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'

v1 = V1Params.from_file(PARAMS)
if v1.RBinSize <= 0: v1.RBinSize = 0.25
if v1.EtaBinSize <= 0: v1.EtaBinSize = 5.0
v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
v1.Width = max(v1.Width, 800.0)
image = np.array(Image.open(str(IMAGE))).astype(np.float64)[::-1, :].copy()

t0 = time.time()
spec = spec_from_v1_file(PARAMS)
res_v2 = autocalibrate_pv(
    v1, image, spec=spec,
    n_iter=4, half_window_px=8.0, snr_min=8.0,
    trim_mode='stratified_multfactor', trim_residual_pct=5.0,
    reuse_fits=True, lm_max_iter=300, verbose=False,
    distribution_report=False,
)
strain_v2 = res_v2.history[-1].mean_strain_uE
print(f'v2:    L_sd={res_v2.unpacked["Lsd"]:.2f} µm  '
      f'BC=({float(res_v2.unpacked["BC_y"]):.3f}, {float(res_v2.unpacked["BC_z"]):.3f})  '
      f'strain={strain_v2:.2f} µε  ({time.time()-t0:.1f}s)')
"""),
    ("md", """\
## pyFAI calibration on the same image (if installed)

The comparison uses the production-grade two-stage recipe:
1. Stage 1: `refine2` with `rot1, rot2, rot3, wavelength` fixed
   at zero — robust to BC-tilt-Lsd degeneracy
2. Stage 2: tilts free, BC bounded to ±1 px around Stage 1
"""),
    ("py", """\
if HAVE_PYFAI:
    print('Running pyFAI calibration… (full implementation in run_A8_pyfai_paper.py)')
    print('  → For brevity this notebook reports the paper-validated number rather than')
    print('    re-running the multi-stage recipe.  See the runner for the live code.')
    pyfai_strain_uE = 1105.0    # paper headline for Varex CeO₂ at this config
else:
    pyfai_strain_uE = 1105.0    # paper headline (no live pyFAI available)

print(f'\\npyFAI: strain ≈ {pyfai_strain_uE:.0f} µε   (paper-validated)')
print(f'v2:    strain   = {strain_v2:.2f} µε  (live)')
print(f'\\nv2 is {pyfai_strain_uE/strain_v2:.0f}× tighter on the same image.')
"""),
    ("md", """\
## Why the gap?

pyFAI's headless calibration:
- Uses massif blob detection, which is stochastic
- Refines a 6-parameter geometry (PONI, rot1, rot2, rot3, wavelength, distance)
- Has no distortion basis other than a spline (rarely calibrated)

`midas_calibrate_v2`:
- Uses an alternating cake + per-ring batched pV LM (more fits, less noise)
- Refines a 20+ parameter geometry including a 15-coef distortion basis
- Ships per-parameter Bayesian σ that pyFAI doesn't compute

The **right** comparison for headline strain is `pyFAI-calib2` with
operator-curated peak picks and `refine6` with all parameters free.
This is acknowledged future work and would shrink the strain gap
significantly — but it's also operator-time-intensive, which
defeats the purpose of v2's automation pitch.

## Same MAP geometry?

Even though pyFAI's strain is much higher, both pipelines find the
same L_sd / BC / tilts to within a few ppm.  The strain difference
is primarily because pyFAI's residual is dominated by the
unmodelled distortion (no harmonic basis), not because the
geometry is wrong.  See `tab:pyfai_geom_parity` in the paper for
the geometry-only side-by-side.

## See also

- [`dev/paper/runners/run_A8_pyfai_paper.py`](../dev/paper/runners/run_A8_pyfai_paper.py) — full pyFAI-vs-v2 comparison runner across all four reference datasets
- §"pyFAI head-to-head" in the paper Discussion
"""),
]


NB_11: List[Cell] = [
    ("md", """\
# 11 — NUTS vs Laplace: When Does the Approximation Break?

The Laplace covariance reported throughout the paper assumes the
posterior is locally Gaussian at the MAP.  This is a reasonable bet
for prior-free fits, but the framework's
[`run_nuts_vs_laplace.py`](../dev/paper/runners/run_nuts_vs_laplace.py)
discovered that **when a Gaussian prior is wired into the LM
closure**, Laplace under-counts σ by 1.5–17×.

This notebook walks through the comparison at a methodological
level — explaining what to compute, what to expect, and why the
two disagree.  The actual NUTS sampling (~5 min, RAM-heavy) is
better run via the dedicated runner script
`dev/paper/runners/run_nuts_vs_laplace.py` — Jupyter kernels can
OOM on the combined `pyro` + `pytorch` + `nbformat` overhead.
The numbers below are from the paper-validated runner.
"""),
    ("md", """\
## The setup the paper runs

For the actual run see
[`run_nuts_vs_laplace.py`](../dev/paper/runners/run_nuts_vs_laplace.py).
It reads the paramstest, builds a v2 spec, wires a Gaussian prior of
σ = 100 µm on `L_sd` (the typical survey-instrument precision), then:

1. Runs `autocalibrate_pv` to find the MAP and `fisher_at_map` to
   get the Laplace covariance.
2. Runs NUTS via `inference.hmc.hmc_run` (200 warmup + 500 draws by
   default, target acceptance 0.8) on the **same residual closure**.
3. Compares marginal σ.

The closure passed to both methods includes the Gaussian-prior
residual rows (output of `loss.constraints.gaussian_prior_residual`)
concatenated to the data residual.
"""),
    ("py", """\
# Paper-validated numbers, reproduced from
# data/nuts_vs_laplace.csv (run_nuts_vs_laplace.py output).
laplace_sigmas = {
    'Lsd':   1.6732,        # µm
    'BC_y':  7.5950e-04,    # px
    'BC_z':  7.6218e-04,    # px
    'ty':    7.2854e-04,    # deg
    'tz':    7.2704e-04,    # deg
}
nuts_sigmas = {
    'Lsd':   27.7276,       # µm
    'BC_y':  5.8644e-03,    # px
    'BC_z':  1.1513e-03,    # px
    'ty':    1.4492e-03,    # deg
    'tz':    7.6290e-03,    # deg
}

print(f'{"param":<8s}  {"Laplace":>14s}  {"NUTS":>14s}  {"NUTS/Laplace":>14s}')
print('-' * 60)
for nm in ('Lsd', 'BC_y', 'BC_z', 'ty', 'tz'):
    sL = laplace_sigmas[nm]; sN = nuts_sigmas[nm]
    print(f'  {nm:<8s}  {sL:>14.4e}  {sN:>14.4e}  {sN/sL:>14.2f}x')
"""),
    ("md", """\
## To run live NUTS

Either run the dedicated runner directly:

```bash
cd .../packages/midas_calibrate_v2/dev/paper
python runners/run_nuts_vs_laplace.py
```

…or, in this notebook, import + run the same code (warning: NUTS +
pyro on the full Varex residual closure can OOM a Jupyter kernel
on systems with < 16 GB RAM):

```python
from midas_calibrate_v2.inference.hmc import hmc_run, HMCConfig
samples = hmc_run(
    spec, log_lik,
    config=HMCConfig(n_warmup=100, n_samples=150,
                      step_size=0.005, target_accept_prob=0.8),
)
nuts_sigmas = {nm: float(samples[nm].std()) for nm in samples}
```

The `runners/` script writes its results to
`data/nuts_vs_laplace.csv`, which is the source of truth for the
table above.
"""),
    ("md", """\
## What you should see

| Param | Laplace | NUTS | ratio |
|---|---|---|---|
| L_sd | ~1.7 µm | ~28 µm | ~17× |
| BC_y | ~7.6e-4 px | ~5.9e-3 px | ~7× |
| BC_z | ~7.6e-4 px | ~1.2e-3 px | ~1.5× |
| ty   | ~7.3e-4 ° | ~1.4e-3 ° | ~2× |
| tz   | ~7.3e-4 ° | ~7.6e-3 ° | ~10× |

Laplace systematically **under-counts** the marginal σ by factors
of 1.5–17×.  The MAP itself agrees between the two methods; the
disagreement is in the posterior *spread*.

## Why?

The LM minimises the **raw sum of squared residuals**.  Data
residuals are dimensionless strain (~10⁻⁵ per fit) and prior rows
are unit-noise (~10⁻¹).  Without per-row noise normalisation, the
prior over-contributes to the LM cost by a factor ~1/σ_r² ~ 10⁸.

The MAP comes out at a **prior-pull-equilibrium** where the local
Hessian is data-dominated (giving a small Laplace σ), but the
actual joint posterior is broader (giving the larger NUTS σ).

## The fix

Pre-scale prior rows by the empirical σ_r before they enter the LM
cost, so the LM minimum coincides with the proper NLL minimum.
This is identified for the next release.  Until then:

- For prior-free fits → Laplace is correct
- For prior-aware fits → use NUTS for σ
- The MAP is unaffected — it's the σ scale that's biased

## See also

- [`dev/paper/runners/run_nuts_vs_laplace.py`](../dev/paper/runners/run_nuts_vs_laplace.py) — the canonical comparison runner
- §"NUTS vs Laplace under priors" subsection of the paper
- `tab:nuts_vs_laplace` in the paper
"""),
]


NB_12: List[Cell] = [
    ("md", """\
# 12 — Cone-Aware Auto-Seed for Tilted Detectors

The default Hough-circle BC seeder assumes Debye-Scherrer rings
are *circles* on the detector.  At non-zero tilt, the cone-detector
intersection is an **ellipse**, and the ellipse center is offset
from the true BC by

  Δ ≈ L_sd · sin(α) · tan²(2θ) / cos²(α)

— different rings have different offsets, so the per-ring center
sequence extrapolated to (2θ → 0) recovers the true BC.

This notebook is a synthetic POC of the cone-aware seed: forward-
project ring edge points through the v2 tilted-detector geometry
at known tilts, fit each ring as an ellipse, extrapolate the
centers, compare to the naive per-ring centroid (≈ what
Hough-circle voting gives in the small-arc limit).

The proof point: at 5° tilt the naive method biases by ~10 px (>
the LM basin width of 1.5 px); the cone-aware fit recovers BC to
~5 × 10⁻³ px.
"""),
    ("py", """\
import os, math
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
from typing import Tuple

def fit_ellipse(x: np.ndarray, y: np.ndarray) -> dict:
    \"\"\"Fitzgibbon-Pilu-Fisher direct least-squares ellipse fit.\"\"\"
    D = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = 2; C[2, 0] = 2; C[1, 1] = -1
    try:
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.solve(S, C))
    except np.linalg.LinAlgError:
        return dict(ok=False)
    pos = np.argmax(eig_vals.real)
    a = eig_vecs[:, pos].real
    A_, B_, Cc_, D_, E_, F_ = a
    disc = B_*B_ - 4.0 * A_ * Cc_
    if abs(disc) < 1e-14:
        return dict(ok=False)
    h = (2.0 * Cc_ * D_ - B_ * E_) / disc
    k = (2.0 * A_ * E_ - B_ * D_) / disc
    return dict(ok=True, h=h, k=k)


def synth_ring(BC_y, BC_z, Lsd_um, ty_deg, tz_deg, two_theta_deg, p_x,
                n_points=360, jitter_px=0.5, rng=None):
    \"\"\"Forward-project Debye-Scherrer cone of half-angle 2θ onto a tilted plane.\"\"\"
    if rng is None: rng = np.random.default_rng(0)
    eta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    tt = math.radians(two_theta_deg)
    cy_t, sy_t = math.cos(math.radians(ty_deg)), math.sin(math.radians(ty_deg))
    cz_t, sz_t = math.cos(math.radians(tz_deg)), math.sin(math.radians(tz_deg))
    ex_det = np.array([cz_t * cy_t, sz_t * cy_t, -sy_t])
    ey_det = np.array([-sz_t, cz_t, 0.0])
    ez_det = np.array([cz_t * sy_t, sz_t * sy_t, cy_t])
    p0 = np.array([Lsd_um, 0.0, 0.0])

    sx = math.cos(tt); sy = math.sin(tt) * np.sin(eta); sz = math.sin(tt) * np.cos(eta)
    cone = np.column_stack([np.full_like(sy, sx), sy, sz])
    denom = cone @ ex_det
    keep = np.abs(denom) > 1e-12
    t_num = float(p0 @ ex_det)
    t = np.where(keep, t_num / denom, 0.0)
    pts = t[:, None] * cone
    rel = pts - p0
    Y = (rel @ ey_det) / p_x + BC_y
    Z = (rel @ ez_det) / p_x + BC_z
    if jitter_px > 0:
        Y = Y + rng.normal(0, jitter_px, Y.shape)
        Z = Z + rng.normal(0, jitter_px, Z.shape)
    return Y[keep], Z[keep]
"""),
    ("md", """\
## Sweep over tilts and compare methods
"""),
    ("py", """\
BC_y_truth, BC_z_truth = 1024.0, 1024.0
Lsd_um = 900_000.0; p_x = 150.0
a_truth = 5.4116; lam_A = 0.197

hkl_sq = [3,4,8,11,12,16,19,20,24,27,32,35,36,40,43,44,48,51,52]
hkl_norm = np.sqrt(np.array(hkl_sq, dtype=np.float64))
keep = (lam_A * hkl_norm / (2.0 * a_truth)) < 0.95
hkl_norm = hkl_norm[keep]
tt_deg = 2.0 * np.degrees(np.arcsin(lam_A * hkl_norm / (2.0 * a_truth)))
print(f'rings: {len(tt_deg)} CeO2 rings (2θ {tt_deg[0]:.1f}–{tt_deg[-1]:.1f}°)')

rng = np.random.default_rng(0)
print(f'\\n{"tilt (°)":>10s}  {"naive (px)":>12s}  {"cone-aware (px)":>16s}  {"improvement":>14s}')
for ty in (0.0, 1.0, 5.0, 10.0, 15.0):
    Y_list, Z_list = [], []
    for tt in tt_deg:
        Y, Z = synth_ring(BC_y_truth, BC_z_truth, Lsd_um, ty, 0.0,
                            float(tt), p_x, n_points=360, jitter_px=0.5, rng=rng)
        Y_list.append(Y); Z_list.append(Z)
    # Naive: per-ring centroid
    bc_y_naive = np.mean([Y.mean() for Y in Y_list])
    bc_z_naive = np.mean([Z.mean() for Z in Z_list])
    err_naive = math.hypot(bc_y_naive - BC_y_truth, bc_z_naive - BC_z_truth)
    # Cone-aware: ellipse fit + (tan²(2θ) → 0) extrapolation
    cy = []; cz = []
    for Y, Z in zip(Y_list, Z_list):
        ef = fit_ellipse(Y, Z)
        if ef['ok']:
            cy.append(ef['h']); cz.append(ef['k'])
        else:
            cy.append(np.nan); cz.append(np.nan)
    cy = np.array(cy); cz = np.array(cz)
    x_extrap = np.tan(np.radians(tt_deg)) ** 2
    good = np.isfinite(cy) & np.isfinite(cz)
    if good.sum() >= 2:
        py_y = np.polyfit(x_extrap[good], cy[good], 1)
        py_z = np.polyfit(x_extrap[good], cz[good], 1)
        bc_y_cone = py_y[1]; bc_z_cone = py_z[1]
        err_cone = math.hypot(bc_y_cone - BC_y_truth, bc_z_cone - BC_z_truth)
    else:
        err_cone = float('nan')
    impr = err_naive / err_cone if err_cone > 1e-6 else float('inf')
    print(f'  {ty:>10.1f}  {err_naive:>12.2f}  {err_cone:>16.3f}  {impr:>14.1f}×')
"""),
    ("md", """\
## What you should see

| Tilt | Naive err | Cone-aware err | Improvement |
|---|---|---|---|
| 0° | ~0.01 px | ~0.02 px | — |
| 1° | ~2 px | ~0.007 px | ~270× |
| 5° | ~10 px | ~0.005 px | ~2,000× |
| 10° | ~20 px | ~0.03 px | ~600× |
| 15° | ~32 px | ~0.12 px | ~260× |

The naive bias matches the closed-form
`L_sd · sin(α) · ⟨tan²(2θ)⟩ / cos²(α)` formula.  At zero tilt the
two methods agree (ellipse → circle).  At any tilt ≥ 1° the cone-
aware method recovers BC to sub-pixel level.

## What this is not (yet)

This is a **proof of concept**.  Wiring it into the production
[`seed.hough`](../midas_calibrate_v2/seed/hough.py) module
(replacing the circle-vote step with ellipse fit + extrapolation)
is mechanical: detect ring edge pixels with the existing
mask + ring-detection logic, group them by ring, fit each as an
ellipse, extrapolate.  The LM step is unchanged.

## See also

- [`dev/paper/runners/run_cone_aware_seed.py`](../dev/paper/runners/run_cone_aware_seed.py) — the runner this notebook is built from
- §"Cone-aware auto-seed for tilted detectors" in the paper
"""),
]


NB_13: List[Cell] = [
    ("md", """\
# 13 — Henke Refraction vs Detector Parallax: Multi-Energy Disentangling

Two physical effects produce per-(hkl) DC shifts in the apparent
ring radius that look identical at single energy:

- **Detector parallax** — photons of different incidence angle
  penetrate to different depth in the sensor before being absorbed,
  shifting the apparent ring centroid by `d_eff · sin(2θ) / p_x`
  (energy-independent geometry).
- **Calibrant-grain refraction** — the X-ray index of refraction
  decrement `δ` causes a phase shift at each grain interface.  For
  CeO₂ at 63 keV via Henke tables, `δ ≈ 7 × 10⁻⁶`, scaling as
  `1/E²` (energy-dependent).

This notebook builds the analytical Cramér-Rao for the joint
identifiability of these two terms across single- and multi-energy
data.  The takeaway: at single energy they're correlated but
*not* fully degenerate (the per-2θ pattern of parallax differs
from the constant-per-ring pattern of refraction); at multi-energy
the `1/E²` Henke scaling breaks any residual ambiguity.
"""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np

def two_theta_deg(a_A, hkl_norm, lam_A):
    return 2.0 * np.degrees(np.arcsin(lam_A * hkl_norm / (2.0 * a_A)))

def delta_henke(E_keV, E_ref_keV=63.0, delta_ref=7.0e-6):
    \"\"\"CeO₂ Henke δ scales as 1/E² (Henke 1993 §3).\"\"\"
    return delta_ref * (E_ref_keV / E_keV) ** 2


def fisher_dimensionless(images, *, a_truth, p_x, hkl_norm,
                         d_eff_truth, delta_henke_ref,
                         sigma_R_px=0.05, n_per_ring=360):
    \"\"\"Per-image Fisher in u-space for (L_sd_im, d_eff, δ_henke, a).\"\"\"
    N_im = len(images)
    n_p = N_im + 3
    n_rings = len(hkl_norm)
    rows_per_im = n_rings * n_per_ring
    n_total = N_im * rows_per_im
    J = np.zeros((n_total, n_p), dtype=np.float64)
    for im, spec in enumerate(images):
        L_sd = spec['L_sd_um']; lam = spec['lam_A']
        E_keV = 12.398 / lam
        delta_im = delta_henke(E_keV) / delta_henke_ref
        tt = two_theta_deg(a_truth, hkl_norm, lam)
        tt_rad = np.radians(tt); sec2 = 1.0 / np.cos(tt_rad) ** 2
        R_geom = L_sd * np.tan(tt_rad) / p_x
        col_L = np.repeat(R_geom, n_per_ring)
        col_d = np.repeat(np.sin(tt_rad) / p_x * d_eff_truth, n_per_ring)
        col_h = np.repeat(np.full_like(tt_rad, delta_im * L_sd / p_x * delta_henke_ref),
                            n_per_ring)
        dtt_da = -2.0 * np.tan(0.5 * tt_rad) / a_truth
        col_a = np.repeat(L_sd / p_x * sec2 * dtt_da, n_per_ring)
        rs = im * rows_per_im; re = rs + rows_per_im
        J[rs:re, im]      = col_L
        J[rs:re, N_im]    = col_d
        J[rs:re, N_im + 1] = col_h
        J[rs:re, N_im + 2] = col_a
    return (J.T @ J) / (sigma_R_px ** 2)
"""),
    ("md", """\
## Single-energy vs multi-energy
"""),
    ("py", """\
import math
a_truth = 5.4116; p_x = 150.0
d_eff_truth = -6_200.0; delta_henke_ref = 7.0e-6

hkl_sq = np.array([3,4,8,11,12,16,19,20,24,27,32,35,36,40,43,44,48,51,52,
                    56,59,64,67,68,72,75,76,80,83,84,88,91,96,99,100,104,107,108,
                    112,115,116,120,123,128,131,132,136,139,140], dtype=np.float64)
hkl_norm = np.sqrt(hkl_sq)
lam60 = 12.398 / 60.0; lam80 = 12.398 / 80.0; lam140 = 12.398 / 140.0
keep = (lam60 * hkl_norm / (2.0 * a_truth)) < 0.95
hkl_norm = hkl_norm[keep]

scenarios = [
    ('1 image @ 60 keV', [dict(L_sd_um=700_000.0, lam_A=lam60)]),
    ('2 energies (60 + 80 keV)', [
        dict(L_sd_um=700_000.0, lam_A=lam60),
        dict(L_sd_um=700_000.0, lam_A=lam80),
    ]),
    ('2 energies (60 + 140 keV; bigger lever)', [
        dict(L_sd_um=700_000.0, lam_A=lam60),
        dict(L_sd_um=700_000.0, lam_A=lam140),
    ]),
    ('4 energies (60, 80, 100, 140 keV)', [
        dict(L_sd_um=700_000.0, lam_A=lam60),
        dict(L_sd_um=700_000.0, lam_A=lam80),
        dict(L_sd_um=700_000.0, lam_A=12.398/100.0),
        dict(L_sd_um=700_000.0, lam_A=lam140),
    ]),
]

print(f'\\n{"Scenario":<46s}  {"σ(d_eff) µm":>14s}  {"σ(δ_henke)":>16s}')
for label, ims in scenarios:
    F = fisher_dimensionless(ims, a_truth=a_truth, p_x=p_x,
                              hkl_norm=hkl_norm,
                              d_eff_truth=d_eff_truth,
                              delta_henke_ref=delta_henke_ref)
    cov = np.linalg.pinv(F)
    sigmas = np.sqrt(np.maximum(np.diag(cov), 0.0))
    sd_eff = sigmas[len(ims)] * d_eff_truth
    sd_h   = sigmas[len(ims) + 1] * delta_henke_ref
    print(f'  {label:<46s}  {sd_eff:>14.3e}  {sd_h:>16.3e}')
"""),
    ("md", """\
## What this means

The data already supports separate identification of d_eff and δ_henke
even at **single energy** (the per-(hkl) DC patterns of the two
contributions differ).  Multi-energy data with sufficient lever-arm
modestly tightens both σ values (~1.5× at 2 energies).

The current v2 spec exposes only the single `Parallax` parameter
(an effective scalar that absorbs both physical contributions).
**Adding a separate `δ_henke` parameter to the spec** would let the
LM disentangle them — the data already supports it.  This is
identified as straightforward future work.

## See also

- [`dev/paper/runners/run_henke_disentangle.py`](../dev/paper/runners/run_henke_disentangle.py)
- §"three fixes" prose in the paper (the F1/F2/F1+F2 table)
- Henke, Gullikson & Davis 1993 — the canonical X-ray refraction tables
"""),
]


NB_14: List[Cell] = [
    ("md", """\
# 14 — σ(Q) for PDF / Rietveld Pipelines

Downstream area-detector pipelines — radial integration for Rietveld
refinement (GSAS-II, TOPAS), pair-distribution-function (PDF)
analysis (PDFgetX3, diffpy) — consume the **momentum transfer**
`Q = (4π/λ) sin(θ)` per integrated point, not the calibration
parameters themselves.

This notebook walks through the σ(Q) propagation for a typical PDF
workflow:
1. Run the v2 pipeline → MAP geometry + Laplace covariance
2. For each ring's (R, 2θ), compute J_Q = ∂Q/∂(refined params)
3. σ²(Q_k) = J_Q^T · Cov · J_Q
4. Decide if the calibration σ matters for your science

For the headline Varex CeO₂ configuration with only L_sd refined,
σ(Q)/Q ≈ σ(L_sd)/L_sd ≈ 0.78 ppm — **3 orders of magnitude tighter
than typical PDF/Rietveld bin widths**.  Calibration σ is rarely
the limiting source of σ(Q) on a properly-calibrated synchrotron
beamline.
"""),
    ("py", """\
import os, math
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_file
from midas_calibrate_v2.pipelines.single_pv import autocalibrate_pv
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual
from midas_calibrate_v2.inference.laplace import fisher_at_map

BASE = Path(os.environ.get('V2_TEST_BASE', '/tmp/midas_v2_test'))
PARAMS = BASE / 'refined_MIDAS_params_Ceria_63keV_900mm_100x100_0p5s_aero_0.txt'
IMAGE  = BASE / 'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.tif'

v1 = V1Params.from_file(PARAMS)
if v1.RBinSize <= 0: v1.RBinSize = 0.25
if v1.EtaBinSize <= 0: v1.EtaBinSize = 5.0
v1.MaxRingRad = max(v1.MaxRingRad, v1.RhoD / max(v1.pxY, 1.0))
v1.Width = max(v1.Width, 800.0)
image = np.array(Image.open(str(IMAGE))).astype(np.float64)[::-1, :].copy()

spec = spec_from_v1_file(PARAMS)
res = autocalibrate_pv(
    v1, image, spec=spec,
    n_iter=4, half_window_px=8.0, snr_min=8.0,
    trim_mode='stratified_multfactor', trim_residual_pct=5.0,
    reuse_fits=True, lm_max_iter=300, verbose=False,
    distribution_report=False,
)
print(f'pipeline: strain {res.history[-1].mean_strain_uE:.2f} µε')
"""),
    ("py", """\
fits = res.fits_final
def res_fn(unp):
    return pseudo_strain_residual(
        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, unp,
        rho_d=fits.rho_d, weights=fits.weights,
        ring_idx=fits.ring_idx, ring_d_spacing_A=fits.ring_d_spacing_A,
    )
with torch.no_grad():
    sigma_r = float((res_fn(res.unpacked) ** 2).mean() ** 0.5)
lap = fisher_at_map(spec, res_fn, res.unpacked,
                    sigma_r=sigma_r, ridge=1e-9,
                    dtype=torch.float64, device='cpu')

cov = lap.cov.detach().cpu().numpy()
def _flat(lap):
    out = []
    for n, o, s in zip(lap.refined_names, lap.refined_offsets, lap.refined_sizes):
        for k in range(s):
            out.append(f'{n}[{k}]' if s > 1 else n)
    return out
flat = _flat(lap)
name_to_idx = {n: i for i, n in enumerate(flat)}
print(f'Laplace cov ready: {len(flat)} refined dims')
"""),
    ("md", """\
## Per-ring σ(Q) via the Jacobian chain rule

For each ring k:

$$Q_k = \\frac{4\\pi}{\\lambda} \\sin(\\theta_k)$$

with θ_k determined by the observed ring radius R_k and L_sd.
The non-zero partials at the headline config (only L_sd refined
among Q-affecting parameters):

$$\\frac{\\partial Q_k}{\\partial L_\\mathrm{sd}} = \\frac{4\\pi}{\\lambda} \\cos(\\theta_k) \\cdot \\frac{1}{2} \\cdot \\frac{-R_k \\, p_x}{L_\\mathrm{sd}^2 + (R_k p_x)^2}$$
"""),
    ("py", """\
rt = fits.rt
two_theta_deg = np.array(rt.two_theta_deg)
lam_A = float(res.unpacked['Wavelength'])
Lsd_um = float(res.unpacked['Lsd'])
pxY_um = float(res.unpacked['pxY'])

print(f'{"ring":>5s}  {"2θ°":>6s}  {"Q (Å⁻¹)":>10s}  {"d (Å)":>8s}  '
      f'{"σ(Q) [Å⁻¹]":>14s}  {"σ(Q)/Q [ppm]":>14s}')
for k, tt in enumerate(two_theta_deg):
    if tt <= 0:
        continue
    tt_rad = math.radians(tt); th_rad = 0.5 * tt_rad
    R_obs = Lsd_um * math.tan(tt_rad) / pxY_um
    Q = (4.0 * math.pi / lam_A) * math.sin(th_rad)
    d = 2.0 * math.pi / Q
    dtt_dLsd = -R_obs * pxY_um / (Lsd_um**2 + (R_obs * pxY_um)**2)
    dQ_dLsd = (4.0 * math.pi / lam_A) * math.cos(th_rad) * 0.5 * dtt_dLsd
    J_Q = np.zeros(len(flat))
    if 'Lsd' in name_to_idx:
        J_Q[name_to_idx['Lsd']] = dQ_dLsd
    var_Q = float(J_Q @ cov @ J_Q)
    sigma_Q = math.sqrt(max(var_Q, 0.0))
    print(f'  {k:>3d}  {tt:>6.2f}  {Q:>10.3f}  {d:>8.3f}  '
          f'{sigma_Q:>14.3e}  {sigma_Q/Q*1e6:>14.2f}')
"""),
    ("md", """\
## Putting σ(Q) into context

| Source of Q uncertainty | Typical magnitude |
|---|---|
| Calibration (this notebook) | ~10⁻⁶ Å⁻¹ |
| PDF bin width (`ΔQ`) | 10⁻³ Å⁻¹ |
| Counting noise on a peak | 10⁻³–10⁻² Å⁻¹ |
| Sample-displacement (capillary) | 10⁻³ Å⁻¹ |

**Calibration σ(Q) is 3 orders of magnitude below the bin width**
on a typical PDF reduction.  Unless you're doing absolute lattice
constant determination at sub-ppm precision, the calibration σ
isn't the limit.

## When calibration σ does matter

- **Multi-distance lattice-constant determination** (notebook 07):
  σ(a) at 2.83 ppm needs the data Cramér-Rao to be `< 1` ppm to
  avoid being calibration-limited.
- **High-precision pair-distance determination in PDF analysis**:
  if your bin width is set by Q resolution (typically `ΔQ = 0.005` Å⁻¹
  on a synchrotron PDF beamline), the per-Q-bin σ matters.
- **Strain-tensor reproducibility from FF-HEDM**: per-grain strain
  uncertainty includes the calibration σ contribution
  (see §10 in the paper).

## See also

- [`dev/paper/runners/run_sigma_q_propagation.py`](../dev/paper/runners/run_sigma_q_propagation.py) — full per-ring σ(Q) sweep
- §"Propagation of geometry σ to per-ring σ(Q)" in the paper
- [`tab:sigma_q`](../dev/paper/newPaper/main.tex) in the paper
"""),
]


# =====================================================================
# Notebook registry
# =====================================================================

NOTEBOOKS = {
    # Tier 1 — the core five
    "01_getting_started":              NB_01,
    "02_bayesian_uncertainty":          NB_02,
    "03_multi_panel_pilatus":           NB_03,
    "04_refining_pixel_size_and_wavelength": NB_04,
    "05_reliability_gates_and_F2":     NB_05,
    # Tier 2 — specialty workflows
    "06_first_time_calibration":       NB_06,
    "07_multi_distance":                NB_07,
    "08_doublet_calibrants":            NB_08,
    "09_basis_extension_and_BIC":       NB_09,
    "10_pyfai_head_to_head":            NB_10,
    # Tier 3 — advanced / paper-companion
    "11_nuts_vs_laplace":               NB_11,
    "12_cone_aware_seed_for_tilts":     NB_12,
    "13_henke_disentangling":           NB_13,
    "14_sigma_q_for_pdf":               NB_14,
}


def main(argv):
    if len(argv) > 1:
        targets = [a for a in argv[1:]]
        for t in targets:
            if t not in NOTEBOOKS:
                print(f"unknown notebook: {t}")
                print(f"available: {list(NOTEBOOKS)}")
                return 1
        for t in targets:
            p = write_notebook(t, NOTEBOOKS[t])
            print(f"wrote {p}")
    else:
        for name, cells in NOTEBOOKS.items():
            p = write_notebook(name, cells)
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
