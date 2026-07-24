"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where
kind is "md" or "py" and source is the markdown / Python source.
Run this script once to (re)generate every .ipynb in this directory.

Why a builder?  Editing raw .ipynb JSON is tedious and the JSON
diffs are unreviewable.  This file is the source of truth; the
.ipynb files are derived artefacts.

Usage:
    cd packages/midas_joint_ff_calibrate/notebooks
    python _build.py             # rebuild all notebooks
    python _build.py 00_getting_started   # rebuild one
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
# Shared synthetic-Pilatus preamble
# =====================================================================
#
# Every notebook here is SELF-CONTAINED and uses only synthetic data —
# no Zarr files, no real datasets, no subprocesses, no network.  We
# reuse the package's own validated synthetic generators from
# ``runners.run_synthetic_pilatus_joint`` (the source of paper figure 1)
# but shrink the configuration (fewer panels / grains / rings) so the
# notebooks run in a few seconds on a CPU.  The forward generators,
# residual closures, and Fisher diagnostic are EXACTLY the production
# code paths — only the problem size is reduced.

PREAMBLE = """\
import os, math, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS OpenMP guard
import numpy as np
import torch

import midas_peakfit as mp
from midas_calibrate_v2.forward.panels import PanelLayout
from midas_hkls import Lattice, SpaceGroup
from midas_diffract.hkls import hkls_for_forward_model

# The package's validated synthetic generators (paper fig. 1 source).
import midas_joint_ff_calibrate.runners.run_synthetic_pilatus_joint as R
from midas_joint_ff_calibrate.loss import JointWeights, joint_residual
from midas_joint_ff_calibrate.pipelines.identifiability import fisher_block_rank
from midas_joint_ff_calibrate.pipelines.alternating import AlternatingDriver
from midas_joint_ff_calibrate.pipelines.full_joint import FullJointDriver

# ---- shrink the problem for notebook speed (production paths unchanged)
R.N_GRAINS = 24
R.N_PANELS_Y = 4
R.N_PANELS_Z = 4
R.PANEL_SIZE_Y = 150
R.PANEL_SIZE_Z = 150
R.LSD_UM = 7.0e5            # closer detector -> more panels see Au rings
R.N_RINGS = 6
R.TWO_THETA_MAX_DEG = 14.0
R.N_POWDER_PER_RING = 180

# Loss weights (1/sigma per modality so neither dominates) + gauge lambda.
W_POWDER, W_HEDM, LAMBDA_GAUGE = 1.0e4, 10.0, 1.0e6


def build_problem(seed: int = 2026):
    \"\"\"Build (layout, truth, grains, ring 2theta/d, powder+HEDM obs).\"\"\"
    layout = PanelLayout.regular(R.N_PANELS_Y, R.N_PANELS_Z,
                                 R.PANEL_SIZE_Y, R.PANEL_SIZE_Z,
                                 gap_y=R.GAP_Y, gap_z=R.GAP_Z)
    truth = R.sample_truth(layout, seed=seed)
    grain_eulers, grain_pos, grain_lat = R.sample_truth_grains(seed=seed + 1)

    sg = SpaceGroup.from_number(225)                 # Fm-3m (Au)
    lat = Lattice.for_system('cubic', a=R.AU_LATTICE_A)
    _, thetas, _ = hkls_for_forward_model(
        sg, lat, wavelength_A=R.WAVELENGTH_A,
        two_theta_max_deg=R.TWO_THETA_MAX_DEG, expand_equivalents=False)
    ring_tt, _ = torch.unique(2 * thetas * 180.0 / math.pi,
                              return_inverse=True, sorted=True)
    ring_tt = ring_tt.double()[:R.N_RINGS]
    ring_d = R.WAVELENGTH_A / (2.0 * torch.sin(ring_tt * math.pi / 360.0))

    powder_obs = R.generate_powder_observations(layout, truth, ring_tt, seed=seed + 2)
    hedm_obs = R.generate_hedm_observations(
        layout, truth, grain_eulers, grain_pos, grain_lat, seed=seed + 3)
    return dict(layout=layout, truth=truth,
                grain_eulers=grain_eulers, grain_pos=grain_pos, grain_lat=grain_lat,
                ring_tt=ring_tt, ring_d=ring_d,
                powder_obs=powder_obs, hedm_obs=hedm_obs)


def build_spec(prob, *, refine_grains=False, refine_panels=True):
    \"\"\"Canonical joint spec: geometry + per-panel deltas + grain blocks.\"\"\"
    truth = prob['truth']; layout = prob['layout']
    spec = mp.ParameterSpec()
    spec.add(mp.Parameter('Lsd', init=truth.Lsd + 50.0,
                          bounds=(truth.Lsd - 5e3, truth.Lsd + 5e3)))
    spec.add(mp.Parameter('BC_y', init=truth.BC_y + 0.3,
                          bounds=(truth.BC_y - 5.0, truth.BC_y + 5.0)))
    spec.add(mp.Parameter('BC_z', init=truth.BC_z - 0.2,
                          bounds=(truth.BC_z - 5.0, truth.BC_z + 5.0)))
    spec.add(mp.Parameter('ty', init=0.0, refined=False))
    spec.add(mp.Parameter('tz', init=0.0, refined=False))
    spec.add(mp.Parameter('Wavelength', init=R.WAVELENGTH_A, refined=False))
    spec.add(mp.Parameter('pxY', init=R.PX_UM, refined=False))
    spec.add(mp.Parameter('pxZ', init=R.PX_UM, refined=False))
    spec.add(mp.Parameter('RhoD', init=200000.0, refined=False))
    spec.add(mp.Parameter('panel_delta_yz',
                          init=torch.zeros(layout.n_panels(), 2, dtype=torch.float64),
                          bounds=(-3.0, 3.0),
                          prior=mp.GaussianPrior(mean=0.0, std=0.5),
                          refined=refine_panels))
    spec.add(mp.Parameter('panel_delta_theta',
                          init=torch.zeros(layout.n_panels(), dtype=torch.float64),
                          refined=False))
    spec.add(mp.Parameter('grain_euler', init=prob['grain_eulers'],
                          bounds=(-2 * math.pi, 2 * math.pi), refined=refine_grains))
    spec.add(mp.Parameter('grain_pos', init=prob['grain_pos'],
                          bounds=(-1000.0, 1000.0), refined=refine_grains))
    spec.add(mp.Parameter('grain_lattice', init=prob['grain_lat'], refined=False))
    return spec


def make_closures(prob, spec):
    \"\"\"Return (joint, powder_only, hedm_only) gauge-free residual closures.\"\"\"
    pf = R.make_powder_residual(prob['powder_obs'], prob['layout'],
                                prob['ring_tt'], ring_d_spacing_A=prob['ring_d'])
    hf = R.make_hedm_residual(prob['hedm_obs'], prob['layout'])

    def joint_fn(u):
        return joint_residual(
            u, powder_residual_fn=pf, hedm_residual_fn=hf, spec=spec,
            weights=JointWeights(w_powder=W_POWDER, w_hedm=W_HEDM,
                                 lambda_gauge=LAMBDA_GAUGE),
            gauge_blocks=[])

    def powder_only(u):   return W_POWDER * pf(u)
    def hedm_only(u):     return W_HEDM * hf(u)
    return joint_fn, powder_only, hedm_only


def seed_unpacked(spec):
    \"\"\"Dict of every parameter at its init value, as float64 tensors.\"\"\"
    u = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    for n in u:
        if not isinstance(u[n], torch.Tensor):
            u[n] = torch.tensor(u[n], dtype=torch.float64)
    return u


def covered_panels(prob):
    p = set(prob['powder_obs']['panel_idx'].tolist())
    h = set(prob['hedm_obs']['panel_idx'].tolist())
    return p, h, (p | h)
"""


# =====================================================================
# NB 00 — Getting started: build_joint_spec + canonical naming
# =====================================================================

NB_00: List[Cell] = [
    ("md", """\
# 00 — Getting Started: the joint spec and its canonical names

`midas_joint_ff_calibrate` performs **joint powder + FF-HEDM
differentiable calibration**.  Its job is to wire three existing
differentiable packages together over a single
`midas_peakfit.ParameterSpec`:

* `midas_calibrate_v2.loss.pseudo_strain` — the powder-calibrant residual
* `midas_fit_grain` / `midas_diffract` — the FF-HEDM spot residual
* `midas_peakfit` — spec / pack / LM / Laplace / Σ=0 gauge

This notebook builds the unified spec with `build_joint_spec`, prints
the **canonical parameter naming**, and shows the refined / frozen
split.  Self-contained synthetic data; runs in seconds.
"""),
    ("py", PREAMBLE),
    ("md", """\
## The canonical parameter names

The joint spec has two families of parameters under fixed names:

**Geometry + detector (shared across all modalities)**

| name | meaning | unit |
|---|---|---|
| `Lsd` | sample-detector distance | µm |
| `BC_y`, `BC_z` | beam centre | px |
| `ty`, `tz` | detector tilts | deg |
| `Wavelength` | X-ray wavelength | Å |
| `pxY`, `pxZ` | pixel pitch | µm |
| `RhoD` | distortion reference radius | µm |
| `panel_delta_yz` | per-panel (δy, δz) shift | px, shape (N_panel, 2) |
| `panel_delta_theta` | per-panel in-plane rotation | rad, shape (N_panel,) |

**HEDM grain nuisance blocks (appended by `build_joint_spec`)**

| name | meaning | unit |
|---|---|---|
| `grain_euler` | per-grain orientation (ZXZ) | rad, shape (N_g, 3) |
| `grain_pos` | per-grain centroid | µm, shape (N_g, 3) |
| `grain_lattice` | per-grain lattice (Voigt a,b,c,α,β,γ) | Å/deg, shape (N_g, 6) |
"""),
    ("md", """\
## Build a powder spec, then extend it with `build_joint_spec`

`build_joint_spec` takes a **powder** `CalibrationSpec` (e.g. from
`midas_calibrate_v2`) and appends the three grain blocks under the
canonical names above.  Default refinement follows the
alternating-driver convention: orientations + positions **on**,
strains **off** (refined in a separate pass).
"""),
    ("py", """\
from midas_joint_ff_calibrate import build_joint_spec

prob = build_problem()
n_g = R.N_GRAINS

# A minimal powder spec (geometry + per-panel deltas).  In production this
# comes from midas_calibrate_v2.compat.from_v1.spec_from_v1_file(...).
powder_spec = build_spec(prob, refine_grains=False, refine_panels=True)
# build_spec already added grain blocks for the notebook helper; for a clean
# demonstration of build_joint_spec we drop them and re-add via the API.
for nm in ('grain_euler', 'grain_pos', 'grain_lattice'):
    if nm in powder_spec.parameters:
        del powder_spec.parameters[nm]

joint_spec = build_joint_spec(
    powder_spec=powder_spec,
    grain_eulers_init=prob['grain_eulers'],
    grain_positions_init=prob['grain_pos'],
    grain_lattices_init=prob['grain_lat'],
    refine_grain_orientation=True,
    refine_grain_position=True,
    refine_grain_strain=False,
)
print(f'joint spec: {len(joint_spec.parameters)} parameter blocks, '
      f'{n_g} grains')
"""),
    ("md", """\
## Inspect the refined / frozen split

`spec.refined_names()` lists the blocks the LM will move.  Note grain
orientation + position are refined; lattice (strain) is frozen.
"""),
    ("py", """\
print('REFINED blocks:')
for n in joint_spec.refined_names():
    p = joint_spec.parameters[n]
    shape = tuple(p.init.shape) if hasattr(p.init, 'shape') and p.init.dim() else '()'
    print(f'  {n:<20s} shape={str(shape):<10s}')

print('\\nFROZEN blocks:')
for n, p in joint_spec.parameters.items():
    if n not in joint_spec.refined_names():
        shape = tuple(p.init.shape) if hasattr(p.init, 'shape') and p.init.dim() else '()'
        print(f'  {n:<20s} shape={str(shape):<10s}')
"""),
    ("md", """\
## Pack / unpack round-trip

The spec packs to a flat refined vector for the LM, and unpacks back
to a name->tensor dict for the residual closures.
"""),
    ("py", """\
x, info = mp.pack_spec(joint_spec)
unpacked = mp.unpack_spec(x, info, joint_spec)
print(f'packed refined vector length: {x.numel()}')
print(f'unpacked keys: {sorted(unpacked.keys())}')
print(f"grain_euler shape: {tuple(unpacked['grain_euler'].shape)}")
print(f"panel_delta_yz shape: {tuple(unpacked['panel_delta_yz'].shape)}")
"""),
    ("md", """\
## Next

* **01** — the recommended `AlternatingDriver` on synthetic Pilatus.
* **02** — `FullJointDriver` + Laplace covariance (per-panel σ).
* **03** — the Fisher block-rank diagnostic (paper headline).
* **04** — the (Lsd, λ) gauge story.
"""),
]


# =====================================================================
# NB 01 — Alternating driver
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — Alternating driver on synthetic Pilatus 6×8 (here 4×4)

`AlternatingDriver` is the **recommended default** per the paper's
implementation plan: an outer loop that alternates two LM passes —

* **Pass A**: geometry + per-panel deltas + grain orientation/position
* **Pass B**: grain strain (lattice) only

This decoupling keeps the well-conditioned geometry/orientation block
away from the weakly-conditioned strain block, improving convergence.

Self-contained synthetic data; runs in ~10 s.
"""),
    ("py", PREAMBLE),
    ("md", """\
## Build the synthetic problem

A multi-panel detector with prescribed per-panel `(δy, δz)` shifts, a
powder CeO2/Au-ring pattern, and FF-HEDM spots from random Au grains —
all at the **same** truth geometry.
"""),
    ("py", """\
prob = build_problem()
layout = prob['layout']; truth = prob['truth']
p_pan, h_pan, cov = covered_panels(prob)
print(f'detector: {layout.n_panels()} panels '
      f'({R.N_PANELS_Y}x{R.N_PANELS_Z})')
print(f'truth Lsd = {truth.Lsd:.1f} um  BC = ({truth.BC_y:.1f}, {truth.BC_z:.1f}) px')
print(f'truth panel_delta sigma = {truth.panel_delta_yz.std().item():.4f} px')
print(f"powder peaks: {prob['powder_obs']['Y'].numel()}  "
      f"HEDM spots: {prob['hedm_obs']['Y'].numel()}")
print(f'panels covered by some modality: {len(cov)}/{layout.n_panels()}')
"""),
    ("md", """\
## Build the spec and the joint residual closure

We refine geometry + per-panel deltas (the headline block).  Grain
orientation/position can be opted in; here we keep them frozen near
truth to isolate the panel-delta recovery.
"""),
    ("py", """\
spec = build_spec(prob, refine_grains=False, refine_panels=True)
joint_fn, powder_only, hedm_only = make_closures(prob, spec)
print('refined:', spec.refined_names())
"""),
    ("md", """\
## Run the AlternatingDriver

`pass_a_thaw` defaults to geometry + grain orientation/position;
`pass_b_thaw` is `grain_lattice`.  Here grain blocks are frozen so
pass A refines geometry + panel deltas and pass B is a no-op — the
driver still demonstrates the outer-loop machinery and converges.
"""),
    ("py", """\
drv = AlternatingDriver(
    spec=spec,
    residual_fn=joint_fn,
    pass_a_thaw=['Lsd', 'BC_y', 'BC_z', 'panel_delta_yz'],
    pass_b_thaw=['grain_lattice'],
    lm_config_a=mp.GenericLMConfig(max_iter=60, ftol_rel=1e-9),
    lm_config_b=mp.GenericLMConfig(max_iter=20, ftol_rel=1e-9),
    n_outer_max=3,
    fallback_span=2.0,
)
t0 = time.time()
res = drv.run(verbose=True)
print(f'\\nconverged={res.converged}  n_outer={res.n_outer}  '
      f'time={time.time()-t0:.1f}s')
print(f'cost history: {[f"{c:.3e}" for c in res.cost_history]}')
"""),
    ("md", """\
## Recovery vs truth (covered panels)

The Σ=0 gauge / Gaussian prior fixes the global panel-shift mean
(which is otherwise absorbed into BC), so we subtract the covered-set
mean error before comparing — a gauge ambiguity, not a misfit.
"""),
    ("py", """\
unp = res.unpacked
pdyz = unp['panel_delta_yz']
err_raw = pdyz - truth.panel_delta_yz
covered_mask = torch.zeros(layout.n_panels(), dtype=torch.bool)
for k in cov:
    covered_mask[k] = True
err_cov = err_raw[covered_mask]
gauge_bias = err_cov.mean(dim=0, keepdim=True)
err_corr = err_cov - gauge_bias
print(f'gauge bias (absorbed into BC): '
      f'({gauge_bias[0,0]:+.4f}, {gauge_bias[0,1]:+.4f}) px')
print(f'panel_delta gauge-corrected RMS: {err_corr.pow(2).mean().sqrt():.4f} px')
print(f'panel_delta max error: {err_corr.abs().max():.4f} px '
      f'(truth sigma {truth.panel_delta_yz.std():.4f} px)')
print(f"Lsd:  truth={truth.Lsd:.1f}  MAP={float(unp['Lsd']):.1f}  "
      f"err={float(unp['Lsd'])-truth.Lsd:+.1f} um")
print(f"BC_y: truth={truth.BC_y:.2f}  MAP={float(unp['BC_y']):.4f}")
print(f"BC_z: truth={truth.BC_z:.2f}  MAP={float(unp['BC_z']):.4f}")
"""),
    ("md", """\
## Takeaway

The alternating driver recovers per-panel shifts to a small fraction
of the truth scatter while jointly refining `Lsd`/`BC`.  For the full
joint refinement with per-parameter Bayesian σ, see **02**.
"""),
]


# =====================================================================
# NB 02 — Full-joint driver + Laplace covariance
# =====================================================================

NB_02: List[Cell] = [
    ("md", """\
# 02 — Full-joint driver + Laplace covariance (per-panel σ)

`FullJointDriver` refines **every** refined parameter at once in a
single LM, then computes the **Laplace (Cramér-Rao) covariance at the
MAP** — for free, from the same autograd Jacobian the LM used.  This
gives a per-parameter σ, including a σ on every panel's (δy, δz).

Self-contained synthetic data; runs in ~10 s.
"""),
    ("py", PREAMBLE),
    ("py", """\
prob = build_problem()
layout = prob['layout']; truth = prob['truth']
spec = build_spec(prob, refine_grains=False, refine_panels=True)
joint_fn, powder_only, hedm_only = make_closures(prob, spec)
p_pan, h_pan, cov = covered_panels(prob)
print(f'{layout.n_panels()} panels, {len(cov)} covered; refined: {spec.refined_names()}')
"""),
    ("md", """\
## Run FullJointDriver with `compute_laplace=True`

`sigma_r` is the residual noise scale (the post-weighting residual is
~unit-variance, so `sigma_r=1.0`).  The driver returns a
`LaplaceResult` with `sigma_per_dim` aligned to `refined_names`.
"""),
    ("py", """\
drv = FullJointDriver(
    spec=spec,
    residual_fn=joint_fn,
    lm_config=mp.GenericLMConfig(max_iter=80, ftol_rel=1e-10, xtol_rel=1e-10),
    fallback_span=2.0,
    sigma_r=1.0,
    compute_laplace=True,
)
t0 = time.time()
res = drv.run()
print(f'rc={res.rc}  cost={res.cost:.4e}  time={time.time()-t0:.1f}s')
print(f'laplace computed: {res.laplace is not None}')
lap = res.laplace
print(f'sigma_per_dim length: {lap.sigma_per_dim.numel()}  '
      f'(refined names: {lap.refined_names})')
"""),
    ("md", """\
## Per-parameter σ on geometry

Read off the σ on the scalar geometry parameters.
"""),
    ("py", """\
names = lap.refined_names
sig = lap.sigma_per_dim
# The flat sigma vector expands vector params; map scalar names to their
# first (only) entry by walking offsets.
import numpy as np
offset = 0
for n in names:
    p = spec.parameters[n]
    size = int(np.prod(p.init.shape)) if hasattr(p.init, 'shape') and p.init.dim() else 1
    if size == 1:
        print(f'  sigma({n:<14s}) = {float(sig[offset]):.4e}')
    offset += size
"""),
    ("md", """\
## Per-panel σ map on `panel_delta_yz`

The headline Bayesian output: a σ for each panel's (δy, δz).  Panels
seen by data have small σ; uncovered panels saturate at the prior σ
(0.5 px).  We reshape the panel block of `sigma_per_dim` to a grid.
"""),
    ("py", """\
# Find the slice of the flat sigma vector belonging to panel_delta_yz.
offset = 0
panel_sigma = None
for n in names:
    p = spec.parameters[n]
    size = int(np.prod(p.init.shape)) if hasattr(p.init, 'shape') and p.init.dim() else 1
    if n == 'panel_delta_yz':
        panel_sigma = sig[offset:offset + size].reshape(layout.n_panels(), 2)
        break
    offset += size

covered_mask = torch.zeros(layout.n_panels(), dtype=torch.bool)
for k in cov:
    covered_mask[k] = True

sig_norm = panel_sigma.norm(dim=1)
print(f'{"panel":>6s}  {"covered":>8s}  {"sigma_dy":>10s}  {"sigma_dz":>10s}')
for k in range(layout.n_panels()):
    print(f'{k:>6d}  {str(bool(covered_mask[k])):>8s}  '
          f'{float(panel_sigma[k,0]):>10.4f}  {float(panel_sigma[k,1]):>10.4f}')
print(f'\\nmedian sigma (covered)   = {float(sig_norm[covered_mask].median()):.4f} px')
print(f'median sigma (uncovered) = {float(sig_norm[~covered_mask].median()):.4f} px '
      f'(should approx the 0.5 px prior)')
"""),
    ("py", """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

grid = sig_norm.numpy().reshape(R.N_PANELS_Y, R.N_PANELS_Z).astype(float)
cov2d = covered_mask.numpy().reshape(R.N_PANELS_Y, R.N_PANELS_Z)
grid_masked = grid.copy()
grid_masked[~cov2d] = np.nan
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(grid_masked, cmap='viridis', origin='lower')
ax.set_title('Per-panel sigma(|delta_yz|) at MAP (covered panels)')
ax.set_xlabel('panel col'); ax.set_ylabel('panel row')
fig.colorbar(im, ax=ax, label='sigma (px)')
fig.tight_layout()
fig.savefig('full_joint_panel_sigma.png', dpi=120)
plt.close(fig)
print('wrote full_joint_panel_sigma.png')
"""),
    ("md", """\
## Takeaway

`FullJointDriver(... compute_laplace=True)` gives a calibrated σ on
every panel shift in one shot.  The next notebook (**03**) explains
*why* the joint fit is needed at all: powder alone leaves the
per-panel block rank-deficient.
"""),
]


# =====================================================================
# NB 03 — Fisher block-rank diagnostic (paper headline)
# =====================================================================

NB_03: List[Cell] = [
    ("md", """\
# 03 — Fisher block-rank diagnostic: powder rank-deficient → joint full-rank

This is the **headline result of the paper**.  For the per-panel
`(δy, δz)` block:

* **Powder-only** data Fisher is **rank-deficient** — a single
  calibrant image cannot independently determine every panel's shift
  (paper-3 §9 result; some panels under-/un-sampled by rings).
* **HEDM evidence** adds independent constraints (spots from many
  grains hit panels from many directions).
* **Joint** powder + HEDM is **full-rank** on the covered panels.

We compute `fisher_block_rank` on the gauge-free **data** Fisher under
each modality.  Self-contained synthetic; runs in seconds (no LM).
"""),
    ("py", PREAMBLE),
    ("md", """\
## Build a prior-free spec for the data-only rank diagnostic

The rank-deficiency is a property of the **data alone**.  A Gaussian
prior on `panel_delta_yz` (used in the production fit, notebook 02)
adds curvature in *every* panel direction and would mask the
deficiency — so for this diagnostic we build the panel block with **no
prior** (and plain bounds), exposing the raw data information content.
"""),
    ("py", """\
prob = build_problem()
layout = prob['layout']

# Prior-free spec: identical to build_spec but panel_delta_yz has no prior.
spec = build_spec(prob, refine_grains=False, refine_panels=True)
del spec.parameters['panel_delta_yz']
spec.add(mp.Parameter('panel_delta_yz',
                      init=torch.zeros(layout.n_panels(), 2, dtype=torch.float64),
                      bounds=(-3.0, 3.0)))             # NO prior
joint_fn, powder_only, hedm_only = make_closures(prob, spec)

p_pan, h_pan, cov = covered_panels(prob)
n_block = 2 * layout.n_panels()           # (dy, dz) per panel
n_covered = len(cov)
print(f'{layout.n_panels()} panels; powder sees {len(p_pan)}, '
      f'HEDM sees {len(h_pan)}, union {n_covered}')
print(f'panel_delta_yz block size = {n_block} '
      f'(full rank on covered = 2*{n_covered} = {2*n_covered})')
"""),
    ("md", """\
## Data Fisher rank under each modality

A full-rank covered block has rank `2 * n_covered`.  Powder-only falls
short; HEDM and joint reach it.
"""),
    ("py", """\
seed_unp = seed_unpacked(spec)
reports = {}
print(f'{"modality":<12s}  {"rank":>6s}  {"of":>5s}  {"cond":>11s}  '
      f'{"sigma_med":>11s}')
for label, fn in [('powder-only', powder_only),
                  ('hedm-only', hedm_only),
                  ('joint', joint_fn)]:
    rep = fisher_block_rank(spec, fn, seed_unp,
                            block_names=['panel_delta_yz'],
                            sigma_r=1.0, fallback_span=2.0)
    reports[label] = rep
    sig = rep.sigma_per_dim
    print(f'{label:<12s}  {rep.rank:>6d}  {sig.numel():>5d}  '
          f'{rep.condition_number:>11.2e}  {float(sig.median()):>11.3e}')

print(f'\\n2*covered = {2*n_covered}')
print(f"powder-only rank-deficient vs covered: "
      f"{reports['powder-only'].rank < 2*n_covered}")
print(f"joint reaches full rank on covered:    "
      f"{reports['joint'].rank == 2*n_covered}")
"""),
    ("md", """\
## Why the difference?

* The **powder** ring at radius `R` only constrains the panel shift
  along the radial direction at the η where the ring crosses that
  panel.  Panels with no ring crossing (or only a tangential one) are
  rank-deficient.
* **HEDM spots** from many grains strike each panel from many
  directions, constraining both `δy` and `δz` per panel.
* The **joint** loss stacks both residuals, so the combined Jacobian
  spans the full per-panel space wherever *either* modality sees a
  panel.

## The nullspace directions

`fisher_block_rank` returns the rank-deficient directions for the
powder-only block — the panel-shift combinations the powder data
cannot see.
"""),
    ("py", """\
null = reports['powder-only'].nullspace_directions
print(f'powder-only nullspace: {null.shape[0]} unconstrained direction(s) '
      f'in the {null.shape[1]}-dim panel block')
if null.shape[0] > 0:
    # Which panels dominate the first nullspace direction?
    d0 = null[0].reshape(layout.n_panels(), 2).norm(dim=1)
    top = torch.argsort(d0, descending=True)[:5]
    print('panels with largest weight in nullspace dir 0:',
          [int(t) for t in top])
"""),
    ("md", """\
## Takeaway

The figure the paper leads with: **powder-only is rank-deficient on
the per-panel block; HEDM evidence makes the joint problem full-rank.**
This is *why* joint calibration is necessary for densely-tiled
multi-panel detectors (Pilatus / Eiger).
"""),
]


# =====================================================================
# NB 04 — (Lsd, wavelength) gauge breaking
# =====================================================================

NB_04: List[Cell] = [
    ("md", """\
# 04 — Breaking the (L_sd, λ) gauge

FF-HEDM spot pixel positions behave, in the small-angle limit, like

$$R_\\mathrm{pix} \\propto \\frac{L_\\mathrm{sd}\\,\\lambda}{d_\\mathrm{grain}}$$

where `d_grain` depends on the (refined) grain lattice.  So the
transform `(L_sd, λ) → (k·L_sd, λ/k)` leaves spot pixels approximately
unchanged — **(L_sd, λ) is a near-null mode of the HEDM data Fisher.**

A **powder calibrant** with an *independently known* `d_calibrant`
breaks this: `R_powder ∝ L_sd·λ / d_calibrant_known`, where
`d_calibrant_known` is fixed (Au reference value), not scaled by `k`.

We compute the 2×2 Fisher block on `(Lsd, Wavelength)` at truth under
powder-only / HEDM-only / joint and read off the condition number.
Self-contained synthetic; no LM needed.
"""),
    ("py", PREAMBLE),
    ("py", """\
prob = build_problem()
truth = prob['truth']; layout = prob['layout']

# A demo spec with Lsd + Wavelength refined, everything else frozen at truth.
spec = mp.ParameterSpec()
spec.add(mp.Parameter('Lsd', init=truth.Lsd, bounds=(truth.Lsd - 5e3, truth.Lsd + 5e3)))
spec.add(mp.Parameter('BC_y', init=truth.BC_y, refined=False))
spec.add(mp.Parameter('BC_z', init=truth.BC_z, refined=False))
spec.add(mp.Parameter('ty', init=0.0, refined=False))
spec.add(mp.Parameter('tz', init=0.0, refined=False))
spec.add(mp.Parameter('Wavelength', init=R.WAVELENGTH_A,
                      bounds=(R.WAVELENGTH_A * 0.998, R.WAVELENGTH_A * 1.002)))
spec.add(mp.Parameter('pxY', init=R.PX_UM, refined=False))
spec.add(mp.Parameter('pxZ', init=R.PX_UM, refined=False))
spec.add(mp.Parameter('RhoD', init=200000.0, refined=False))
spec.add(mp.Parameter('panel_delta_yz',
                      init=torch.zeros(layout.n_panels(), 2, dtype=torch.float64),
                      refined=False))
spec.add(mp.Parameter('panel_delta_theta',
                      init=torch.zeros(layout.n_panels(), dtype=torch.float64),
                      refined=False))
spec.add(mp.Parameter('grain_euler', init=prob['grain_eulers'],
                      bounds=(-2 * math.pi, 2 * math.pi), refined=False))
spec.add(mp.Parameter('grain_pos', init=prob['grain_pos'],
                      bounds=(-1000.0, 1000.0), refined=False))
spec.add(mp.Parameter('grain_lattice', init=prob['grain_lat'], refined=False))

joint_fn, powder_only, hedm_only = make_closures(prob, spec)
truth_unp = seed_unpacked(spec)
print('refined:', spec.refined_names())
"""),
    ("md", """\
## The 2×2 Fisher block on (L_sd, λ)

A large eigenvalue ratio (high condition number) means a near-gauge:
the data barely constrains one direction in (L_sd, λ) space.  HEDM-only
should be near-degenerate; powder and joint should break it.
"""),
    ("py", """\
print(f'{"modality":<12s}  {"lambda_min":>12s}  {"lambda_max":>12s}  '
      f'{"cond":>12s}')
for label, fn in [('powder-only', powder_only),
                  ('hedm-only', hedm_only),
                  ('joint', joint_fn)]:
    rep = fisher_block_rank(spec, fn, truth_unp,
                            block_names=['Lsd', 'Wavelength'],
                            sigma_r=1.0, fallback_span=2.0)
    F = rep.fisher.detach()
    eig = torch.linalg.eigvalsh(F).clamp(min=0.0)
    lo, hi = float(eig.min()), float(eig.max())
    cond = hi / max(lo, 1e-300)
    print(f'{label:<12s}  {lo:>12.3e}  {hi:>12.3e}  {cond:>12.3e}')
"""),
    ("md", """\
## Reading the result

* **HEDM-only**: the condition number is large — the `(L_sd, λ)`
  direction is a near-null mode, exactly the gauge described above.
  You cannot separately determine `L_sd` and `λ` from HEDM spots alone
  when the grain lattice is also free.
* **Powder-only**: the known calibrant d-spacing anchors the product
  `L_sd·λ` to a fixed ring radius, breaking the gauge (lower
  condition number).
* **Joint**: tightest of all — both modalities contribute.

## Practical recipe

If your experiment refines wavelength alongside geometry, you **must**
include a powder calibrant (with a trusted reference lattice) in the
joint fit — HEDM spots alone leave `(L_sd, λ)` unidentifiable.  This is
the calibration analogue of the multi-distance `(L_sd, a)` story in
`midas_calibrate_v2` notebook 22.
"""),
]


# =====================================================================
# NB 05 — grain-based tx (the powder-blind geometry)
# =====================================================================
#
# Unlike 00-04 (which share the synthetic-Pilatus PREAMBLE above), this
# notebook needs the single-panel FF HEDM forward model and the raw-pixel
# DetCor path, so it carries its own preamble.  Part 1-3 are synthetic and
# self-contained (they execute in the CI sweep); part 4 is the production
# entry point against a real reconstruction and is skipped when no layer
# directory is configured.
#
# The synthetic generators below are lifted from
# ``tests/test_grain_refine.py`` so the notebook and the regression tests
# demonstrate the same thing on the same construction.

PREAMBLE_TX = """\
import os, math
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS OpenMP guard
import numpy as np
import torch

import midas_peakfit as mp
from midas_peakfit import Parameter
from midas_calibrate.geometry_torch import build_tilt_matrix_torch, pixel_to_REta_torch
from midas_diffract import HEDMForwardModel
from midas_diffract.forward import HEDMGeometry
from midas_diffract.hkls import hkls_for_forward_model
from midas_hkls import Lattice, SpaceGroup
from midas_fit_grain.matching import MatchResult
from midas_fit_grain.observations import ObservedSpots
from midas_joint_ff_calibrate.grain_refine import make_residual
from midas_joint_ff_calibrate.spec import build_joint_spec
from midas_stress.orientation import euler_to_orient_mat, orient_mat_to_euler

torch.manual_seed(0)
DT = torch.float64

# One synthetic FF geometry: Ni FCC, Varex-like 2048 x 2048 @ 150 um, 1 m.
LSD, BCY, BCZ, PX = 1.0e6, 1024.0, 1024.0, 150.0
RHOD, NPIX = 1024.0 * PX, 2048


def build_model():
    sg = SpaceGroup.from_number(225)                       # Ni, Fm-3m
    lat = Lattice(3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=0.2066, two_theta_max_deg=18.0,
        expand_equivalents=True)
    geom = HEDMGeometry(
        Lsd=LSD, y_BC=BCY, z_BC=BCZ, px=PX, omega_start=-180.0, omega_step=0.25,
        n_frames=1440, n_pixels_y=NPIX, n_pixels_z=NPIX, min_eta=6.0,
        wavelength=0.2066, tx=0.0, ty=0.0, tz=0.0, wedge=0.0,
        flip_y=True, apply_tilts=False, multi_mode='layered')
    return HEDMForwardModel(hkls_cart, thetas, geom, hkls_int=hkls_int.float())


def bake_tx_into_raw(yp_ideal, zp_ideal, tx_true_deg):
    \"\"\"RAW pixels such that DetCor(raw, tx_true) == the ideal tilt-free pixels.

    With ty=tz=0 and no distortion the pure-tx tilt is an exact 2-D rotation
    of the centred (Yc, Zc) detector coordinates, so we simply apply its
    inverse.\"\"\"
    T = build_tilt_matrix_torch(torch.tensor(tx_true_deg, dtype=DT),
                                torch.tensor(0.0, dtype=DT),
                                torch.tensor(0.0, dtype=DT))
    Rot = T[1:, 1:]                                # 2x2 rotation on (Yc, Zc)
    Yc = (-yp_ideal + BCY) * PX
    Zc = (zp_ideal - BCZ) * PX
    raw = torch.stack([Yc, Zc], dim=-1) @ Rot      # orthonormal => == Rot^T . v
    return BCY - raw[..., 0] / PX, BCZ + raw[..., 1] / PX


def build_synth(tx_true_deg, n_grains=8, seed=0, max_spots=40):
    \"\"\"Forward-model grains in ideal space, bake tx_true into the RAW pixels,
    and assemble (observations, matches, raw_yz) with an identity match.\"\"\"
    model = build_model()
    rng = np.random.default_rng(seed)
    eulers = rng.uniform(-math.pi, math.pi, size=(n_grains, 3))
    eulers[:, 1] = rng.uniform(0, math.pi, size=n_grains)      # Phi in [0, pi]
    positions = np.zeros((n_grains, 3))
    lattices = np.tile(np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), (n_grains, 1))

    observations, matches, raw_yz = [], [], []
    eul_t = torch.from_numpy(eulers).to(DT)
    pos_t = torch.from_numpy(positions).to(DT)
    lat_t = torch.from_numpy(lattices).to(DT)

    def sq(t):
        while t.dim() > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t

    for g in range(n_grains):
        s = model(eul_t[g].view(1, 1, 3), pos_t[g].view(1, 1, 3),
                  lattice_params=lat_t[g].view(1, 6))
        valid = sq(s.valid).bool()
        yp, zp = sq(s.y_pixel).double(), sq(s.z_pixel).double()
        om, eta, tth = sq(s.omega).double(), sq(s.eta).double(), sq(s.two_theta).double()
        M = valid.shape[1]
        ks, msq = torch.where(valid)
        if ks.numel() == 0:
            continue
        if ks.numel() > max_spots:
            sel = torch.randperm(ks.numel())[:max_spots]
            ks, msq = ks[sel], msq[sel]
        flat = ks * M + msq
        yp_i, zp_i = yp.reshape(-1)[flat], zp.reshape(-1)[flat]
        om_i = om.reshape(-1)[flat]
        eta_i, tth_i = eta.reshape(-1)[flat], tth.reshape(-1)[flat]
        yp_raw, zp_raw = bake_tx_into_raw(yp_i, zp_i, tx_true_deg)

        # Observed lab (Y, Z): the ideal prediction R=Lsd*tan(2theta),
        # (Y,Z)=(-R sin eta, R cos eta), rotated about the beam by -tx_true.
        # make_residual rotates the OBSERVED by the trial tx and matches it to
        # the ideal prediction, so the (Y,Z) minimum sits exactly at tx_true.
        R_id = LSD * torch.tan(tth_i)
        pred_vec = torch.stack([-R_id * torch.sin(eta_i),
                                R_id * torch.cos(eta_i)], dim=-1)
        T_tx = build_tilt_matrix_torch(torch.tensor(tx_true_deg, dtype=DT),
                                       torch.tensor(0.0, dtype=DT),
                                       torch.tensor(0.0, dtype=DT))
        obs_vec = pred_vec @ T_tx[1:, 1:]
        S = ks.numel()
        observations.append(ObservedSpots(
            spot_id=torch.arange(S), ring_nr=torch.zeros(S, dtype=torch.int64),
            y_lab=obs_vec[..., 0], z_lab=obs_vec[..., 1],
            omega=om_i, eta=eta_i, two_theta=tth_i,
            grain_radius=torch.full((S,), 50.0, dtype=DT),
            fit_rmse=torch.zeros(S, dtype=DT), y_orig=torch.zeros(S, dtype=DT),
            z_orig=torch.zeros(S, dtype=DT), omega_ini=om_i.clone(),
            mask_touched=torch.zeros(S, dtype=torch.bool)))
        matches.append(MatchResult(
            k_idx=ks.long(), m_idx=msq.long(), mask=torch.ones(S, dtype=torch.bool),
            delta_omega=torch.zeros(S, dtype=DT), delta_eta=torch.zeros(S, dtype=DT)))
        raw_yz.append((yp_raw, zp_raw))
    return model, observations, matches, raw_yz, eulers, positions, lattices


def fixed_geo():
    return dict(
        Lsd=torch.tensor(LSD, dtype=DT), BC_y=torch.tensor(BCY, dtype=DT),
        BC_z=torch.tensor(BCZ, dtype=DT), ty=torch.tensor(0.0, dtype=DT),
        tz=torch.tensor(0.0, dtype=DT), px=torch.tensor(PX, dtype=DT),
        RhoD=torch.tensor(RHOD, dtype=DT), p_coeffs=torch.zeros(15, dtype=DT))


def make_spec(eulers, positions, lattices, tx_init=0.0, free_pose=False):
    \"\"\"tx + the three grain blocks. free_pose=True thaws grain orientation --
    used ONLY as the negative control in part 2.\"\"\"
    spec = mp.ParameterSpec()
    spec.add(Parameter('tx', init=torch.tensor(tx_init, dtype=DT), refined=True,
                       bounds=(-5.0, 5.0)))
    spec = build_joint_spec(
        powder_spec=spec,
        grain_eulers_init=torch.from_numpy(eulers).to(DT),
        grain_positions_init=torch.from_numpy(positions).to(DT),
        grain_lattices_init=torch.from_numpy(lattices).to(DT),
        refine_grain_orientation=free_pose, refine_grain_position=False,
        refine_grain_strain=False)
    if free_pose:
        spec.parameters['grain_euler'].bounds = (-2 * math.pi, 2 * math.pi)
    return spec


def cost_at(spec, resid, **overrides):
    u = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    for k, v in overrides.items():
        u[k] = torch.tensor(np.asarray(v), dtype=DT)
    return float((resid(u) ** 2).sum())


def rotate_grains_about_beam(eul, delta_deg):
    \"\"\"Rotate every grain orientation about the LAB beam axis (X) by delta.

    This is the operation that -- if the tx <-> orientation degeneracy were
    exact -- would let the grains absorb a tx error. Part 2 measures whether
    it actually does.\"\"\"
    d = math.radians(delta_deg)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(d), -math.sin(d)],
                   [0.0, math.sin(d), math.cos(d)]])
    out = np.zeros_like(eul)
    for i, e in enumerate(eul):
        OM = np.asarray(euler_to_orient_mat(e)).reshape(3, 3)
        out[i] = orient_mat_to_euler((Rx @ OM).reshape(9))
    return out


print('preamble ready')
"""


NB_05: List[Cell] = [
    ("md", """\
# 05 — Recovering `tx` from grains (the geometry powder cannot see)

A powder / calibrant fit (`midas-calibrate-v2`) pins `Lsd`, the beam centre,
`ty`, `tz` and the distortion — but it is **structurally blind to `tx`**, the
in-plane detector rotation about the beam. Debye–Scherrer rings are
rotationally symmetric about the beam axis, so rotating the detector in its
own plane maps every ring onto itself. No powder pattern, however good, carries
information about `tx`.

`tx` can only be pinned by **single-crystal spots**, whose azimuthal positions
break that symmetry. That is what this package's `grain-tx` step does, and it
is why an FF reconstruction is properly a **two-pass** procedure:

1. calibrate on the calibrant → reconstruct with `tx = 0`;
2. refine `tx` from the recovered grains → **re-reconstruct**.

Leaving `tx` at 0 when it is not 0 shifts every spot by `R·sin(tx)` — for a
real Ni dataset, `tx = 0.049°` at `R ≈ 150 mm` is a ~130 µm azimuthal shift,
which corrupts indexing *and* the refined positions and strains.

### What this notebook covers

| Part | Data | Runs in CI |
| --- | --- | --- |
| 1. `tx` moves η, not R | synthetic | yes |
| 2. Why the grain pose must be frozen | synthetic | yes |
| 3. Recovering a known `tx` from 0 | synthetic | yes |
| 4. The production path on a real reconstruction | bring-your-own | no |

Parts 1–3 are self-contained — no dataset, no subprocess, no network. They use
the same generators as `tests/test_grain_refine.py`.
"""),
    ("py", PREAMBLE_TX),
    ("md", """\
## 1. `tx` rotates η and leaves R alone

Take spots whose RAW pixels contain a known `tx`, then run the DetCor
transform (`pixel_to_REta_torch`) twice — once with the correct `tx`, once with
`tx = 0` — and compare the recovered radius `R` and azimuth `η`.

This measurement is the whole reason `kind="pixel"` is **disabled** for the
`grain-tx` step: a radial/pixel-distance residual barely moves with `tx`, so it
cannot constrain it. The loss has to be η-sensitive.
"""),
    ("py", """\
TX_TRUE = 0.40                      # degrees
model, obs, matches, raw_yz, eulers, positions, lattices = build_synth(TX_TRUE)
fg = fixed_geo()
print(f'{len(obs)} grains, {sum(int(m.mask.sum()) for m in matches)} spots')

Yr, Zr = raw_yz[0]
def detcor(tx_deg):
    return pixel_to_REta_torch(
        Yr, Zr, Lsd=fg['Lsd'], BC_y=fg['BC_y'], BC_z=fg['BC_z'],
        tx=torch.tensor(tx_deg, dtype=DT), ty=fg['ty'], tz=fg['tz'],
        p_coeffs=fg['p_coeffs'], parallax=torch.zeros((), dtype=DT),
        px=fg['px'], rho_d=fg['RhoD'])

print(f'\\n{"tx used":>9s}  {"tx error":>10s}  {"max |dR|":>12s}  {"max |d.eta|":>12s}')
print(f'{"(deg)":>9s}  {"(deg)":>10s}  {"(um)":>12s}  {"(deg)":>12s}')
R_ref, eta_ref = detcor(TX_TRUE)
for tx_try in (0.0, 0.1, 0.2, 0.4):
    R_t, eta_t = detcor(tx_try)
    dR = float((R_ref - R_t).abs().max()) * float(fg['px'])
    deta = float((eta_ref - eta_t).abs().max())
    print(f'{tx_try:>9.2f}  {TX_TRUE - tx_try:>10.2f}  {dR:>12.6f}  {deta:>12.4f}')
print('\\n=> d.eta tracks the tx error 1:1; dR is zero to numerical precision.')
print('   A radial (pixel) loss is blind to tx; an angular loss is not.')
"""),
    ("md", """\
## 2. Why the grain pose must be frozen

`tx` rotates the **observed** pattern about the beam. A grain's orientation
rotates the **predicted** pattern about the beam. That looks like a textbook
degeneracy — counter-rotate every grain and you should be able to absorb any
`tx` — and it is the stated reason production `refine_geometry_from_grains`
**holds grain orientation, position and strain fixed** for the `tx` step.

It is worth checking rather than repeating, because it turns out to be only
half true. Two measurements below:

1. a **direct probe** — offset `tx`, counter-rotate every grain about the beam
   by the same angle, and see whether the cost comes back;
2. the **operational test** — thaw the grain orientations and refine `tx`
   alongside them.

First, the frozen-pose cost scan, as a baseline.
"""),
    ("py", """\
spec_fixed = make_spec(eulers, positions, lattices, tx_init=0.0)
resid = make_residual(model, obs, matches, raw_yz, fixed_geo=fg, kind='angular')

scan = np.linspace(-0.6, 0.6, 49)
costs = np.array([cost_at(spec_fixed, resid, tx=float(t)) for t in scan])
print(f'cost minimum at tx = {scan[costs.argmin()]:+.3f} deg  (true {TX_TRUE:+.3f})')
print(f'cost at tx=0      : {cost_at(spec_fixed, resid, tx=0.0):.4e}')
print(f'cost at tx=tx_true: {cost_at(spec_fixed, resid, tx=TX_TRUE):.4e}')
"""),
    ("py", """\
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.6, 4.0))
ax.semilogy(scan, costs, 'b.-', lw=1.2, ms=4)
ax.axvline(TX_TRUE, color='g', ls='--', lw=1.5, label=f'true tx = {TX_TRUE}')
ax.axvline(0.0, color='r', ls=':', lw=1.5, label='powder pass (tx = 0)')
ax.set_xlabel('trial tx (deg)'); ax.set_ylabel('sum of squared residuals')
ax.set_title('Frozen-pose cost has a clean minimum at the true tx')
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig('grain_tx_cost_scan.png', dpi=120)
plt.close(fig)
print('wrote grain_tx_cost_scan.png')
"""),
    ("md", """\
### Probe 1 — can counter-rotating the grains actually absorb a `tx` error?

Offset `tx` by `delta`, rotate every grain about the lab beam axis by the same
`delta`, and compare against the *unrotated* cost at that same offset `tx`. If
the degeneracy were exact the counter-rotated cost would return to the floor.
The unrotated column is the control that makes the comparison meaningful.
"""),
    ("py", """\
print(f'{"delta":>7s}  {"cost @ tx+delta":>17s}  {"...+ grains rotated":>21s}  {"verdict":>12s}')
print(f'{"(deg)":>7s}  {"(no rotation)":>17s}  {"(counter-rotated)":>21s}  {"":>12s}')
for delta in (0.05, 0.10, 0.20):
    c_plain = cost_at(spec_fixed, resid, tx=TX_TRUE + delta)
    c_rot = cost_at(spec_fixed, resid, tx=TX_TRUE + delta,
                    grain_euler=rotate_grains_about_beam(eulers, delta))
    verdict = 'absorbed' if c_rot < 0.1 * c_plain else 'NOT absorbed'
    print(f'{delta:>7.2f}  {c_plain:>17.4e}  {c_rot:>21.4e}  {verdict:>12s}')
print(f'\\nfloor (tx exact, pose exact): {cost_at(spec_fixed, resid, tx=TX_TRUE):.4e}')
print('\\n=> Counter-rotating the grains does NOT recover the cost - it makes it')
print('   WORSE. The degeneracy is not exact, because tx leaves omega untouched')
print('   while rotating a grain about the beam shifts the omega at which each')
print('   spot diffracts. The omega term in the 3-D loss is what breaks it.')
"""),
    ("md", """\
### Probe 2 — the operational test

The degeneracy is not exact, so in principle `tx` and the grain orientations are
jointly identifiable. In practice: thaw the orientations (24 extra parameters
for 8 grains) and refine `tx` alongside them from the same start.
"""),
    ("py", """\
def run_lm(spec, resid, max_iter=80):
    u0 = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    c0 = float((resid(u0) ** 2).sum())
    u, c, rc = mp.lm_minimise(
        spec, resid,
        config=mp.GenericLMConfig(max_iter=max_iter, ftol_rel=1e-12, xtol_rel=1e-12),
        fallback_span=2.0)
    return float(u['tx']), c0, float(c), rc

tx_fixed, c0_f, c1_f, rc_f = run_lm(
    make_spec(eulers, positions, lattices, tx_init=0.0), resid)
tx_free, c0_v, c1_v, rc_v = run_lm(
    make_spec(eulers, positions, lattices, tx_init=0.0, free_pose=True), resid)

print(f'{"grain pose":<14s}  {"tx recovered":>13s}  {"error":>10s}  {"cost drop":>22s}')
print(f'{"FROZEN":<14s}  {tx_fixed:>13.6f}  {abs(tx_fixed - TX_TRUE):>10.2e}  '
      f'{c0_f:>9.2e} -> {c1_f:>9.2e}')
print(f'{"FREE":<14s}  {tx_free:>13.6f}  {abs(tx_free - TX_TRUE):>10.2e}  '
      f'{c0_v:>9.2e} -> {c1_v:>9.2e}')
print(f'\\ntrue tx = {TX_TRUE}')
"""),
    ("md", """\
### What the two probes together say

The exact-degeneracy story is **wrong**, and the measurement says so: counter-
rotating the grains does not absorb a `tx` offset, it makes the fit worse. `tx`
moves `η` only — `2θ` and `ω` are invariant under it — whereas rotating a grain
about the beam also changes the `ω` at which each of its spots diffracts. That
`ω` mismatch is exactly the *multi-grain ω-coupling* that makes `tx`
identifiable in the first place.

But probe 2 shows the practical answer is unchanged: with the pose thawed the
refinement **fails anyway** — it stalls near the start, barely moving the cost
and leaving `tx` at ~0 instead of 0.4°. Near-degenerate directions plus 24
weakly-determined nuisance parameters wreck the conditioning, even though the
problem is formally identifiable.

So: freeze the pose. Not because `tx` is unidentifiable with it free, but
because the frozen-pose problem is a one-parameter fit with a clean minimum
(the scan above) and the thawed one is not. Pose and strain get refined
downstream in `process_grains`, where they belong.
"""),
    ("md", """\
## 3. Recovering a known `tx` from zero

The real use case: the powder pass handed us `tx = 0`, and we want the truth
back. Both η-sensitive losses work — `angular` (the 3-D `2θ, η, ω` residual,
the default) and `internal_angle` (the angle between predicted and observed
g-vectors). `pixel` is rejected by the API for the reason measured in part 1.
"""),
    ("py", """\
for kind in ('angular', 'internal_angle'):
    r = make_residual(model, obs, matches, raw_yz, fixed_geo=fg, kind=kind)
    tx_rec, c0, c1, rc = run_lm(make_spec(eulers, positions, lattices, tx_init=0.0), r)
    print(f'{kind:<16s}  tx = {tx_rec:+.6f}  (true {TX_TRUE:+.3f}, '
          f'err {abs(tx_rec - TX_TRUE):.2e})  cost {c0:.2e} -> {c1:.2e}  rc={rc}')
"""),
    ("md", """\
## 4. The production path — a real reconstruction

Everything above is the *why*. On real data you call one function (or one CLI
command) against a finished FF reconstruction. It reads `Grains.csv`,
`SpotMatrix.csv` and `hkls.csv` from the layer directory, picks the
best-**fitting** grains (smallest mean g-vector angle at the seed pose — not the
highest confidence, which admits badly-fit grains whose residuals swamp `tx`'s
sub-degree signal), fits one shared `tx` across all of them, and writes a
corrected parameter file.

> **The trap.** Pass the **master** FF parameter file — the one
> `build_paramstest` / `FitSetupParams` wrote, carrying `OmegaStep`,
> `NrFilesPerSweep`, `NrPixelsY/Z`. Do **not** pass the stripped per-layer
> `paramstest.txt` that the refiner consumes: it drops the ω-scan and detector
> keys, `OmegaStep` silently defaults to 0, every predicted spot is invalid, and
> you get `matched spots = 0` and `tx = 0`. There is a guard that raises on
> this now, but the symptom is otherwise baffling.

Set the two paths below to run it. The cell no-ops if they are not set, which
is how it stays in the automated notebook sweep.
"""),
    ("py", """\
from pathlib import Path

# ---- point these at a finished FF reconstruction -------------------------
LAYER_DIR    = None      # e.g. Path('/data/ni_recon/LayerNr_1')
MASTER_PARAM = None      # e.g. Path('/data/nb_ps_ni_v2.txt')  <-- MASTER, not the layer's
# -------------------------------------------------------------------------

if LAYER_DIR is None or MASTER_PARAM is None:
    print('No dataset configured - skipping the real-data step.')
    print('Set LAYER_DIR and MASTER_PARAM above to run it.')
else:
    from midas_joint_ff_calibrate.grain_refine import refine_geometry_from_grains
    out_ps = Path(LAYER_DIR) / 'paramstest_graintx.txt'
    res = refine_geometry_from_grains(
        paramstest=MASTER_PARAM, layer_dir=LAYER_DIR,
        refine_params=('tx',),          # add 'Wedge' to co-refine the rotation axis
        kind='angular', max_grains=50, max_iter=50,
        out_paramstest=out_ps, device='cpu')
    print(f'grains={res.n_grains}  matched spots={res.n_spots_matched}  rc={res.rc}')
    print(f'cost: {res.cost_init:.4e} -> {res.cost_final:.4e}')
    for k, v in res.refined.items():
        print(f'  {k} = {v:+.6f} deg')
    print(f'wrote {res.paramstest_out}')
"""),
    ("md", """\
### The same thing from a terminal

```bash
midas-joint-ff-calibrate grain-tx \\
    --paramstest /data/nb_ps_ni_v2.txt \\
    --layer-dir  /data/ni_recon/LayerNr_1 \\
    --refine tx --kind angular --max-grains 50 \\
    --out /data/ni_recon/LayerNr_1/paramstest_graintx.txt
```

Or as an optional pipeline stage, which runs it automatically after
`process_grains` and writes the corrected parameter file for you:

```bash
midas-pipeline run --scan-mode ff ... --grain-geometry-run --grain-geometry-refine tx
```

## Completing the second pass

`grain-tx` writes a corrected copy of the parameter file you gave it. To
actually benefit, fold the refined `tx` into your **master** parameter file and
re-run the pipeline from the start — `tx` enters at the *transforms* step, so
indexing and refinement both have to be redone:

```python
import re
tx_ref = res.refined['tx']
txt = MASTER_PARAM.read_text()
txt = (re.sub(r'(?m)^tx\\b.*$', f'tx {tx_ref:.6f}', txt)
       if re.search(r'(?m)^tx\\b', txt) else txt + f'\\ntx {tx_ref:.6f}\\n')
MASTER_PARAM.with_name(MASTER_PARAM.stem + '_tx.txt').write_text(txt)
```

Then compare pass 1 against pass 2 — grain count, mean confidence, and the
strain spread should all improve. A worked end-to-end example (raw CBF →
calibrate → reconstruct → `grain-tx` → re-reconstruct) is
`midas_pipeline` notebook
`06_ff_cbf_real_data_calibrate_and_reconstruct.ipynb`, section 5.

## Limitations

* **Only `tx` and `Wedge` can be refined here.** `refine_geometry_from_grains`
  builds its spec with those two; `Lsd`, `BC_y`, `BC_z`, `ty`, `tz` and the
  distortion coefficients are passed as fixed geometry, so asking for them
  raises `KeyError`. The archived C `FitGrain` fit ten geometry parameters —
  that breadth has not been ported, because `tx` is the one powder genuinely
  cannot see.
* **`tx` needs several grains.** One grain cannot separate `tx` from its own
  orientation; the ω-coupling across differently-oriented grains is what breaks
  the degeneracy. The default `max_grains=50` is a reasonable floor.
* **`tx` is η-only.** It leaves `2θ` and `ω` untouched, so it does not fix a
  wrong `Lsd` or a hydrostatic strain offset. For those, see the `d0` recovery
  in `midas_pipeline` notebook 06 section 6.
"""),
]


# =====================================================================
# Notebook registry
# =====================================================================

NOTEBOOKS = {
    "00_getting_started":          NB_00,
    "01_alternating_driver":       NB_01,
    "02_full_joint_laplace":       NB_02,
    "03_fisher_block_rank":        NB_03,
    "04_lsd_wavelength_gauge":     NB_04,
    "05_grain_tx_from_grains":     NB_05,
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
