"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where
kind is "md" or "py" and source is the markdown / Python source.
Run this script once to (re)generate every .ipynb in this directory.

The .ipynb files are derived artefacts; this file is the source of truth.

Usage:
    cd packages/midas_fit_grain/notebooks
    python _build.py                       # rebuild all notebooks
    python _build.py 01_single_grain_refinement
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
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = HERE / f"{name}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path


# =====================================================================
# NB 01 — Single-grain refinement quickstart
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — Single-grain refinement quickstart (`midas_fit_grain`)

`midas_fit_grain` is the PyTorch drop-in for the C executable
`FitPosOrStrainsOMP`. For each indexed grain it refines **12 parameters** —
position (3 µm), Bunge Euler angles (3 rad), and lattice constants (6) — by
minimising the mismatch between observed and forward-modelled diffraction
spots.

This notebook is a self-contained tour of the three knobs you choose when you
call `refine_grain`:

| knob | options | what it controls |
|------|---------|------------------|
| **solver** | `lbfgs` (default), `adam`, `lm`, `nelder_mead` | the optimiser |
| **loss** | `pixel` (default), `angular`, `internal_angle` | the residual definition |
| **mode** | `iterative` (default), `all_at_once` | re-match between phases or not |

In production these inputs come from the indexer (`ExtraInfo.bin` +
`BestPos_*.csv`). Here we build a **synthetic single-grain fixture** using the
package's own test helper (`tests/_synthetic.py`) — a small far-field
`HEDMForwardModel`, a known ground-truth grain, and the noise-free spot list it
produces. Everything runs on CPU in float64 in a few seconds.
"""),
    ("py", """\
import os
# midas_diffract / torch pull in OpenMP; allow the duplicate runtime on macOS.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import math
import sys
from pathlib import Path

import torch

# The synthetic single-grain fixture lives in the package's test suite. Add the
# tests dir to sys.path so we can reuse the exact generator the unit tests use
# (this is ground-truth API usage, not an invented helper). The package itself
# may be pip-installed into site-packages, so resolve the source tree from this
# notebook's location (notebooks/ sits next to tests/ in the repo).
import midas_fit_grain

def _find_tests_dir():
    candidates = [Path.cwd() / 'tests', Path.cwd().parent / 'tests']
    # also try the source tree adjacent to the installed package, if editable
    try:
        candidates.append(Path(midas_fit_grain.__file__).resolve().parents[1] / 'tests')
    except Exception:
        pass
    for c in candidates:
        if (c / '_synthetic.py').exists():
            return c
    raise FileNotFoundError(f'cannot locate tests/_synthetic.py; tried {candidates}')

TESTS_DIR = _find_tests_dir()
sys.path.insert(0, str(TESTS_DIR))

from _synthetic import make_synthetic, fixture_to_observed, gt_match  # noqa: E402

DEVICE = torch.device('cpu')   # CPU only — no CUDA/MPS required
DTYPE  = torch.float64
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
print('midas_fit_grain', midas_fit_grain.__version__)
"""),
    ("md", """\
## 1 — Build the synthetic grain + its observed spots

`make_synthetic` returns a `SyntheticFixture` carrying:

- `model` — a far-field `HEDMForwardModel` (cubic *a*=4.04 Å, λ=0.1729 Å,
  Lsd=1 m, 2048² detector, 200 µm pixels).
- `gt_position` / `gt_euler` / `gt_lattice` — the ground-truth grain state.
- the noise-free observed spot list that grain produces.

`fixture_to_observed` packs the spots into an `ObservedSpots` (the same struct
the refiner consumes from `ExtraInfo.bin` in production).
"""),
    ("py", """\
fix = make_synthetic(device=DEVICE, dtype=DTYPE)
obs = fixture_to_observed(fix, device=DEVICE, dtype=DTYPE)

print('ground-truth grain:')
print('  position (um):', fix.gt_position.tolist())
print('  euler (deg)  :', [round(a * RAD2DEG, 3) for a in fix.gt_euler.tolist()])
print('  lattice      :', fix.gt_lattice.tolist())
print(f'observed spots : {obs.n_spots}')
print(f'rings (slots)  : {fix.ring_numbers}')
"""),
    ("md", """\
## 2 — Configure the fit (`FitConfig`)

`FitConfig` holds the geometry + the three knobs. The fields below mirror what
the unit tests build. We start with the production defaults:
`solver=lbfgs`, `loss=pixel`, `mode=iterative`.
"""),
    ("py", """\
from midas_fit_grain import FitConfig

def build_cfg(*, solver='lbfgs', loss='pixel', mode='iterative'):
    return FitConfig(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()),
        SpaceGroup=225,
        RingNumbers=fix.ring_numbers,
        RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=2.0,
        EtaBinSize=2.0, OmeBinSize=2.0, MinEta=6.0,
        solver=solver, mode=mode, loss=loss,
        max_iter=200, ftol=1e-8, xtol=1e-9,
        phase_steps=(8, 8, 8, 8),
    )

cfg = build_cfg()
print('solver =', cfg.solver, '| loss =', cfg.loss, '| mode =', cfg.mode)
"""),
    ("md", """\
## 3 — Perturb the seed, then refine

Real seeds from the indexer are tight; here we deliberately offset the grain by
~1 µm and a small Euler rotation so the optimiser has work to do. `refine_grain`
is keyword-only after `cfg`. We pass `precomputed_match=gt_match(...)` to mirror
the per-spot reflection pairing the indexer hands over in `BestPos_*.csv`.
"""),
    ("py", """\
from midas_fit_grain import refine_grain
from midas_diffract import HEDMForwardModel

def misori_deg(ea, eb):
    Ra = HEDMForwardModel.euler2mat(ea)
    Rb = HEDMForwardModel.euler2mat(eb)
    trace = (Ra.T @ Rb).diagonal().sum()
    return float(torch.acos(((trace - 1) / 2).clamp(-1.0, 1.0))) * RAD2DEG

# Seed = ground truth nudged off (stay inside the smooth basin).
init_pos = fix.gt_position.clone() + torch.tensor([1.0, -0.5, 0.3], dtype=DTYPE)
init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
init_lat = fix.gt_lattice.clone()
match_seed = gt_match(fix, device=DEVICE, dtype=DTYPE)

res = refine_grain(
    cfg, model=fix.model, obs=obs,
    init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
    pred_ring_slot=fix.pred_ring_slot,
    precomputed_match=match_seed,
)

print(f'converged       : {res.converged}')
print(f'spots matched   : {res.n_matched} / {obs.n_spots}')
print(f'loss            : {res.history[0]:.4e}  ->  {res.history[-1]:.4e}')
print(f'|Δposition| (um): {(res.position - fix.gt_position).norm().item():.4f}')
print(f'misori (deg)    : {misori_deg(res.euler, fix.gt_euler):.5f}')
"""),
    ("md", """\
## 4 — Loss functions: `pixel` vs `angular` vs `internal_angle`

| loss | residual | refines |
|------|----------|---------|
| `pixel` | (y, z) detector pixel positions | position + orientation + strain |
| `angular` | (2θ, η, ω) in radians | orientation (pose) |
| `internal_angle` | angle between predicted/observed ĝ vectors | orientation only |

`pixel` is the only loss that constrains **position** (the others are pose-only
because angular/g-vector geometry is translation-blind). The loop below
confirms each loss drives orientation toward truth.
"""),
    ("py", """\
for loss in ('pixel', 'angular', 'internal_angle'):
    cfg_l = build_cfg(loss=loss)
    # internal_angle / angular are pose-only: start position at truth.
    ip = fix.gt_position.clone() if loss != 'pixel' else init_pos
    ie = fix.gt_euler.clone() + (0.5 if loss == 'internal_angle' else 0.05) * DEG2RAD
    r = refine_grain(
        cfg_l, model=fix.model, obs=obs,
        init_position=ip, init_euler=ie, init_lattice=fix.gt_lattice.clone(),
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=(match_seed if loss == 'pixel' else None),
    )
    pe = (r.position - fix.gt_position).norm().item()
    print(f'{loss:>14s}: loss {r.history[0]:.3e}->{r.history[-1]:.3e}  '
          f'misori={misori_deg(r.euler, fix.gt_euler):.5f} deg  |Δpos|={pe:.4f} um')
"""),
    ("md", """\
## 5 — Solvers: `lbfgs` vs `adam`

`lbfgs` (quasi-Newton, default) converges fastest on this smooth synthetic.
`adam` is per-parameter scale-invariant — handy for mixed-unit problems — but
needs a learning rate roughly the size of the perturbation, so we use an
angular (single-scale) loss for it. `lm` (Levenberg–Marquardt) and
`nelder_mead` are also available via the same `solver=` argument.
"""),
    ("py", """\
for solver in ('lbfgs', 'adam'):
    cfg_s = build_cfg(solver=solver, loss='angular')
    if solver == 'adam':
        cfg_s.phase_steps = (40, 40, 40, 40)
    r = refine_grain(
        cfg_s, model=fix.model, obs=obs,
        init_position=fix.gt_position.clone(),
        init_euler=fix.gt_euler.clone() + 0.1 * DEG2RAD,
        init_lattice=fix.gt_lattice.clone(),
        pred_ring_slot=fix.pred_ring_slot,
    )
    print(f'{solver:>10s}: loss {r.history[0]:.3e} -> {r.history[-1]:.3e}  '
          f'(n_iters={len(r.history)})  misori={misori_deg(r.euler, fix.gt_euler):.5f} deg')
"""),
    ("md", """\
## 6 — Fit modes: `iterative` vs `all_at_once`

- `iterative` — position → re-match → orientation → re-match → strain →
  re-match → joint polish. This matches `FitPosOrStrainsOMP`'s default and is
  more robust when the seed association is imperfect.
- `all_at_once` — all 12 parameters jointly; spot association is computed once
  at entry and never updated mid-fit. Faster, fewer kernel launches.

Both recover the grain here because the GT match is exact.
"""),
    ("py", """\
for mode in ('iterative', 'all_at_once'):
    cfg_m = build_cfg(mode=mode, loss='pixel')
    r = refine_grain(
        cfg_m, model=fix.model, obs=obs,
        init_position=init_pos.clone(),
        init_euler=init_eul.clone(),
        init_lattice=init_lat.clone(),
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=match_seed,
    )
    print(f'{mode:>12s}: loss {r.history[0]:.3e} -> {r.history[-1]:.3e}  '
          f'|Δpos|={(r.position - fix.gt_position).norm().item():.4f} um  '
          f'misori={misori_deg(r.euler, fix.gt_euler):.5f} deg')
"""),
    ("md", """\
## 7 — The `GrainFitResult`

`refine_grain` returns a `GrainFitResult` with the refined state plus
diagnostics: `position`, `euler`, `lattice`, `final_loss`, `n_matched`,
`history` (per-iteration loss), `converged`, `match` (the spot→reflection
pairing), and `per_spot_residuals`.

In a real run, `refine_block` batches many grains and writes the byte-identical
`OrientPosFit.bin` / `FitBest.bin` / `Key.bin` that `ff_MIDAS.py`'s merge stage
consumes (see `io_binary.write_orient_pos_fit_row` & friends).
"""),
    ("py", """\
print('GrainFitResult fields:')
print('  position (um) :', [round(x, 4) for x in res.position.tolist()])
print('  euler (deg)   :', [round(x * RAD2DEG, 4) for x in res.euler.tolist()])
print('  lattice       :', [round(x, 5) for x in res.lattice.tolist()])
print('  final_loss    :', f'{res.final_loss:.4e}')
print('  n_matched     :', res.n_matched)
print('  converged     :', res.converged)
print('  history len   :', len(res.history))
print('  per-spot resid:', tuple(res.per_spot_residuals.shape))
"""),
    ("md", """\
## Recap

- Built a synthetic single grain from the package test fixture and refined it
  on CPU/float64.
- Walked every **solver** (`lbfgs`/`adam`), **loss** (`pixel`/`angular`/
  `internal_angle`), and **mode** (`iterative`/`all_at_once`).
- `pixel` loss is the only one that constrains position; angular/internal_angle
  are pose-only.

**Next:** the CLI `midas-fit-grain paramstest.txt <blockNr> <numBlocks>
<numLines> <numProcs> [--solver ...] [--loss ...] [--mode ...]` runs the same
refiner over real indexer output inside the `ff_MIDAS` pipeline.
"""),
]


NOTEBOOKS = {
    "01_single_grain_refinement": NB_01,
}


def main(argv: List[str]) -> None:
    targets = argv or list(NOTEBOOKS)
    for name in targets:
        if name not in NOTEBOOKS:
            raise SystemExit(f"unknown notebook {name!r}; choices: {list(NOTEBOOKS)}")
        path = write_notebook(name, NOTEBOOKS[name])
        print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
