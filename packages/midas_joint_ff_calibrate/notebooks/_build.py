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
# Notebook registry
# =====================================================================

NOTEBOOKS = {
    "00_getting_started":          NB_00,
    "01_alternating_driver":       NB_01,
    "02_full_joint_laplace":       NB_02,
    "03_fisher_block_rank":        NB_03,
    "04_lsd_wavelength_gauge":     NB_04,
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
