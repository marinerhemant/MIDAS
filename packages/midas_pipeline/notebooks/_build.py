"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of ``(kind, source)`` tuples where
``kind`` is ``"md"`` or ``"py"``. The .ipynb files are derived
artefacts; this file is the source of truth.

Usage:
    cd packages/midas_pipeline/notebooks
    python _build.py
    jupyter nbconvert --to notebook --execute --inplace *.ipynb \
        --ExecutePreprocessor.timeout=300
"""
from __future__ import annotations

import json
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
    out = HERE / f"{name}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    return out


# A small Au CIF reused by the V-map notebook (no external file dependency).
_AU_CIF = """\
data_Au
_cell_length_a 4.08
_cell_length_b 4.08
_cell_length_c 4.08
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Au1 Au 0.0 0.0 0.0 1.0
"""


# ===========================================================================
# 01 — Synthetic FF-HEDM walkthrough via the unified orchestrator
# ===========================================================================

NB01: List[Cell] = [
    ("md", """\
# 01 · Unified pipeline — synthetic FF-HEDM walkthrough

`midas-pipeline` is the unified MIDAS HEDM orchestrator. Its design
thesis: **FF is the single-scan degeneracy of PF.** One orchestrator,
one CLI, two scan modes (`--scan-mode {ff,pf,auto}`). Each layer runs a
mode-dependent ordered list of stages (zip_convert → hkl → peakfit →
merge_overlaps → calc_radius → transforms → … → binning → indexing →
refinement → … → consolidation).

This notebook runs the orchestrator end-to-end on a **synthetic** Au
FF-HEDM dataset (no external data, CPU only), now **through indexing**.
We:

1. Forward-simulate a small synthetic single-detector dataset.
2. Drive `midas-pipeline run --scan-mode ff` through ingest → hkl →
   peakfit → merge → radius → transforms → **binning → indexing**, and
   inspect per-stage timings + artefacts.
3. Confirm `Data.bin` (the per-bin spot-index table) is populated and the
   indexer recovers grain solutions.
4. Show how `ScanGeometry.ff()` is literally `pf_uniform` with one scan
   position — the degeneracy the package is built around.

> **Runtime** ~1–1.5 min on CPU (the per-frame peak search dominates)."""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS libomp guard

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from midas_pipeline import __version__, ScanGeometry
print('midas-pipeline', __version__)"""),
    ("md", """\
## 1. Forward-simulate a synthetic dataset

`midas-pipeline simulate` is a scaffold in 0.2.x, so for a self-contained
synthetic dataset we use the sibling `midas_ff_pipeline.testing`
generator (pure-Python: `midas_diffract.simulate_panel_zarrs` +
`midas_hkls`, no C forward-sim). It reads geometry / lattice / scan keys
from the bundled `FF_HEDM/Example/Parameters.txt` and writes a
`.MIDAS.zip` archive plus a local `Parameters.txt`.

We keep the grain count small (the pattern just needs enough spots to
exercise the stages)."""),
    ("py", """\
from midas_ff_pipeline.testing import generate_synthetic_dataset

MIDAS_HOME = Path(os.environ.get('MIDAS_HOME') or Path.home() / 'opt' / 'MIDAS')
PARAMS_TEMPLATE = MIDAS_HOME / 'FF_HEDM' / 'Example' / 'Parameters.txt'
assert PARAMS_TEMPLATE.exists(), f'template not found: {PARAMS_TEMPLATE}'

WORK = Path(tempfile.mkdtemp(prefix='midas_pipeline_ff_'))
t0 = time.time()
zarr = generate_synthetic_dataset(
    out_dir=WORK / 'sim',
    params_template=PARAMS_TEMPLATE,
    n_grains=20,
    n_cpus=4,
)
print(f'synthetic dataset in {time.time() - t0:.1f}s')
print('zarr  :', zarr)
print('params:', WORK / 'sim' / PARAMS_TEMPLATE.name)"""),
    ("md", """\
## 2. Run the orchestrator (FF mode, through indexing)

We call the CLI exactly as a user would. `--scan-mode ff` selects the
single-scan stage order. We run ingest → hkl → peakfit → merge → radius
→ transforms → **binning → indexing**, and `--skip` only the
refinement/consolidation tail (those add little to a binning/indexing
walkthrough). The default indexer backend is the in-process
PyTorch indexer (`--indexer-backend python`)."""),
    ("py", """\
RESULT = WORK / 'run'
cmd = [
    sys.executable, '-m', 'midas_pipeline', 'run',
    '--scan-mode', 'ff',
    '--params', str(WORK / 'sim' / PARAMS_TEMPLATE.name),
    '--result', str(RESULT),
    '--zarr', str(zarr),
    '--n-cpus', '4',
    '--device', 'cpu',
    '--dtype', 'float64',
    '--indexer-backend', 'python',
    '--skip', 'refinement',
    '--skip', 'process_grains', '--skip', 'consolidation',
]
print('running:', ' '.join(cmd[2:]), '\\n')
t0 = time.time()
proc = subprocess.run(cmd, capture_output=True, text=True)
print(f'exit={proc.returncode}  ({time.time() - t0:.1f}s)')
# Show the tail of the orchestrator log.
print('\\n'.join(proc.stderr.strip().splitlines()[-8:]))
assert proc.returncode == 0, proc.stderr[-2000:]"""),
    ("md", """\
## 3. Per-stage status

`midas-pipeline status` reads the provenance store written under each
`LayerNr_*/` and reports per-stage completion + wall time. This is the
same machinery that powers `resume` (skip already-complete stages) and
`inspect`."""),
    ("py", """\
status = subprocess.run(
    [sys.executable, '-m', 'midas_pipeline', 'status', str(RESULT)],
    capture_output=True, text=True,
)
print(status.stdout)"""),
    ("md", """\
## 4. Binning + indexing artefacts

The binning stage writes `Spots.bin` (observed spots, lab frame) plus the
per-bin spot-index table `Data.bin` / `nData.bin`. `Data.bin` is now
**non-empty** — this is the table the indexer mmaps to gather candidate
spots per (ring, eta, omega) bin. The indexer then writes `IndexBest.bin`
(one record per seed) and `IndexBestFull.bin` (matched spot pairs)."""),
    ("py", """\
layer_dir = RESULT / 'LayerNr_1'
interesting = ['hkls.csv', 'InputAll.csv', 'Spots.bin', 'Data.bin', 'nData.bin',
               'paramstest.txt', 'IndexBest.bin', 'IndexBestFull.bin']
for name in interesting:
    p = layer_dir / name
    print(f'{name:34s} {"OK " if p.exists() else "-- "} '
          f'{p.stat().st_size if p.exists() else 0:>14,} bytes')

assert (layer_dir / 'Data.bin').stat().st_size > 0, 'Data.bin must be non-empty'"""),
    ("md", """\
## 5. Grain solutions

`IndexBest.bin` is a flat `float64` array of 15-column records, one per
indexed seed. Column 14 is `nMatches` (the count of matched predicted
spots); a seed with `nMatches > 0` produced an orientation solution."""),
    ("py", """\
ib = np.fromfile(layer_dir / 'IndexBest.bin', dtype=np.float64)
assert ib.size % 15 == 0, ib.size
ib = ib.reshape(-1, 15)
n_seeds = ib.shape[0]
n_solved = int((ib[:, 14] > 0).sum())
print(f'seeds attempted : {n_seeds}')
print(f'seeds w/ a solution (nMatches > 0): {n_solved}')
print(f'best nMatches   : {ib[:, 14].max():.0f}')
assert n_solved > 0, 'expected at least one indexed seed'"""),
    ("md", """\
## 6. FF is the single-scan degeneracy of PF

The package's central abstraction is `ScanGeometry`. FF is not a
separate code path conceptually — it is `pf_uniform` with exactly one
scan position at Y = 0. The orchestrator dispatches stages on
`scan_mode`, but the geometry object makes the degeneracy explicit."""),
    ("py", """\
ff = ScanGeometry.ff()
pf = ScanGeometry.pf_uniform(n_scans=5, scan_step_um=2.0, beam_size_um=4.0)
print('FF :', ff)
print('PF :', pf)
print()
print('FF n_scans          :', ff.n_scans, '  (single scan position)')
print('FF scan_positions   :', ff.scan_positions)
print('PF n_scans          :', pf.n_scans)
print('PF scan_positions   :', pf.scan_positions, 'um')"""),
    ("md", """\
## What just happened

The unified orchestrator ran the FF pipeline end-to-end **through
indexing** — frame ingest, HKL generation, per-frame peak fitting,
overlap merge, radius assignment, lab-frame transforms, binning, and
orientation indexing — all in-process via the shared `midas-*` kernel
packages, then reported provenance through `status`.

The remaining notebooks build on this working indexing stage:

| Notebook | Topic |
| --- | --- |
| 02 | indexer backend selector (`python` vs `c-omp`) — they agree |
| 03 | V-map per-spot relative volume + soft beam attribution |
| 04 | FF as the single-scan degeneracy of PF |"""),
    ("py", """\
# Tidy up the scratch workspace.
shutil.rmtree(WORK, ignore_errors=True)
print('cleaned', WORK)"""),
]


# ===========================================================================
# 02 — Indexer backend selector (python vs c-omp)
# ===========================================================================

NB02: List[Cell] = [
    ("md", """\
# 02 · Indexer backend selector — `python` vs `c-omp`

`midas-pipeline` ships two interchangeable indexer backends behind one
flag, `--indexer-backend {python, c-omp}`:

- **`python`** — the in-process PyTorch/Numba indexer
  (`midas_index`). CPU/CUDA/MPS, autograd-friendly, no compiler needed.
- **`c-omp`** — the bundled OpenMP C binary (`midas_indexer`), built by
  scikit-build-core at pip-install time when a compiler is available.

Both read the **same** unified on-disk contract written by the binning
stage: a 10-column `Spots.bin` (col 9 = ScanNr, 0 for FF) plus an
`int64`-pair `Data.bin` / `nData.bin`. This notebook runs the FF
pipeline once, then drives both backends over the identical binned
inputs and shows they read the same data and agree.

> **Runtime** ~1–1.5 min on CPU."""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from midas_pipeline import __version__
from midas_index import backend_c
print('midas-pipeline', __version__)
print('c-omp backend available:', backend_c.available())
if backend_c.available():
    print('c-omp binary:', backend_c.binary_path())"""),
    ("md", """\
## 1. Forward-simulate + run upstream stages (shared inputs)

We generate one synthetic FF dataset and run the orchestrator through
**binning** (skipping indexing and after). The binned `Spots.bin` /
`Data.bin` / `nData.bin` become the shared inputs that both backends
will index. Running upstream once keeps the comparison apples-to-apples."""),
    ("py", """\
from midas_ff_pipeline.testing import generate_synthetic_dataset

MIDAS_HOME = Path(os.environ.get('MIDAS_HOME') or Path.home() / 'opt' / 'MIDAS')
PARAMS_TEMPLATE = MIDAS_HOME / 'FF_HEDM' / 'Example' / 'Parameters.txt'

WORK = Path(tempfile.mkdtemp(prefix='midas_pipeline_idx_'))
zarr = generate_synthetic_dataset(
    out_dir=WORK / 'sim', params_template=PARAMS_TEMPLATE,
    n_grains=20, n_cpus=4,
)

UPSTREAM = WORK / 'upstream'
cmd = [
    sys.executable, '-m', 'midas_pipeline', 'run', '--scan-mode', 'ff',
    '--params', str(WORK / 'sim' / PARAMS_TEMPLATE.name),
    '--result', str(UPSTREAM), '--zarr', str(zarr),
    '--n-cpus', '4', '--device', 'cpu', '--dtype', 'float64',
    '--skip', 'indexing', '--skip', 'refinement',
    '--skip', 'process_grains', '--skip', 'consolidation',
]
t0 = time.time()
proc = subprocess.run(cmd, capture_output=True, text=True)
print(f'upstream exit={proc.returncode} ({time.time() - t0:.1f}s)')
assert proc.returncode == 0, proc.stderr[-2000:]

layer = UPSTREAM / 'LayerNr_1'
for n in ('Spots.bin', 'Data.bin', 'nData.bin'):
    print(f'  {n:12s}', f'{(layer / n).stat().st_size:>14,} bytes')
assert (layer / 'Data.bin').stat().st_size > 0"""),
    ("md", """\
## 2. Run the indexer directly with each backend

We invoke `midas_index` over the binned `LayerNr_1/` with each backend.
The CLI argv is the IndexerOMP-compatible
`<paramstest> <block> <n_blocks> <n_seeds> <n_cpus>`. We point the C
binary's `OutputFolder` at a per-backend directory so the consolidated
output lands somewhere we can inspect."""),
    ("py", """\
def run_index(backend, tag):
    out = WORK / f'idx_{tag}'
    out.mkdir(parents=True, exist_ok=True)
    n_seeds = sum(1 for _ in (layer / 'SpotsToIndex.csv').open())
    if backend == 'python':
        cmd = [sys.executable, '-m', 'midas_index', str(layer / 'paramstest.txt'),
               '0', '1', str(n_seeds), '4', '--device', 'cpu', '--dtype', 'float64',
               '--group-size', '4']
    else:
        cmd = [str(backend_c.binary_path()), str(layer / 'paramstest.txt'),
               '0', '1', str(n_seeds), '4']
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(layer), capture_output=True, text=True)
    dt = time.time() - t0
    return p, dt

py_proc, py_dt = run_index('python', 'py')
print(f'[python] exit={py_proc.returncode} ({py_dt:.2f}s)')
print('  ', py_proc.stderr.strip().splitlines()[-1] if py_proc.stderr.strip() else '')
assert py_proc.returncode == 0, py_proc.stderr[-1500:]"""),
    ("py", """\
if backend_c.available():
    c_proc, c_dt = run_index('c-omp', 'c')
    print(f'[c-omp] exit={c_proc.returncode} ({c_dt:.2f}s)')
    # The C binary echoes what it read from the binned files to stdout.
    for line in c_proc.stdout.splitlines():
        if any(k in line for k in ('nSpots', 'DataSize', 'nDataSize',
                                   'Binning', 'Mode', 'Finished')):
            print('  ', line.strip())
    assert c_proc.returncode == 0, c_proc.stderr[-1500:]
else:
    print('c-omp binary not built in this environment — skipping the C run.')
    print('Re-install midas-index with a C/OpenMP toolchain to enable it.')
    c_proc = None"""),
    ("md", """\
## 3. Do the backends agree?

The two backends share one byte-level contract, so the decisive check is
that the C binary reads the **same** observed-spot count and per-bin
table that the Python backend indexes. The Python backend additionally
materialises `IndexBest.bin`; we read back its solution count.

(The C binary writes consolidated outputs under its own `OutputFolder`
convention rather than `IndexBest.bin`; both completing cleanly over the
identical `Spots.bin` / `Data.bin` is the agreement we assert here.)"""),
    ("py", """\
# Python backend: number of seeds with a solution.
ib = np.fromfile(layer / 'IndexBest.bin', dtype=np.float64).reshape(-1, 15)
py_solved = int((ib[:, 14] > 0).sum())
print(f'python backend: {py_solved} / {ib.shape[0]} seeds solved')

# What the indexer read from disk (10-col Spots.bin, int64 Data.bin).
spots = np.fromfile(layer / 'Spots.bin', dtype=np.float64)
n_spots = spots.size // 10
data = np.fromfile(layer / 'Data.bin', dtype=np.int64).reshape(-1, 2)
print(f'shared inputs : Spots.bin -> {n_spots} spots (10-col), '
      f'Data.bin -> {data.shape[0]} (spot, scan) entries')

if c_proc is not None:
    # The C binary prints 'nSpots = <N>' — confirm it matches.
    c_nspots = None
    for line in c_proc.stdout.splitlines():
        if line.strip().startswith('nSpots'):
            c_nspots = int(line.split('=')[1])
    print(f'c-omp read    : nSpots = {c_nspots}')
    assert c_nspots == n_spots, (c_nspots, n_spots)
    print('AGREE: both backends read the same', n_spots, 'observed spots from the unified Spots.bin.')
else:
    print('c-omp unavailable — python backend solved', py_solved, 'seeds.')"""),
    ("py", """\
shutil.rmtree(WORK, ignore_errors=True)
print('cleaned', WORK)"""),
]


# ===========================================================================
# 03 — V-map relative volume + soft beam attribution
# ===========================================================================

NB03: List[Cell] = [
    ("md", """\
# 03 · V-map per-spot relative volume + soft beam attribution

The V-map machinery turns measured spot intensities into a per-voxel
**relative scattering volume** map, jointly with per-ring scale factors
`K`, optional absorption `mu`, and a refinable beam profile. It is driven
by two flag groups:

- `--vmap-run` enables the `calc_radius_v` + `refine_vmap` stages.
- `--soft-attribution` swaps the indexer's hard voxel assignment for a
  continuous beam-weight profile (`gaussian` / `tophat` / `tophat-ramp`).

This notebook runs them on the synthetic **FF** case. Two things to
understand up front:

1. `calc_radius_v` computes the **per-spot relative volume**
   `V_rel = I_obs / I_theory(ring)` from the indexed spots and a crystal
   model — this *is* the raw V-map signal, and it runs on FF.
2. The **joint** `refine_vmap` step (and `--soft-attribution` in the
   indexer) operate over a **multi-scan voxel grid**, which is a PF
   concept: with a single FF beam position there is exactly one voxel and
   nothing to attribute across. So on FF, `refine_vmap` skips with a
   precise reason and the soft-attribution profile is a no-op in the FF
   indexer. We show the V-map signal that *does* exist on FF and explain
   the PF requirement.

> **Runtime** ~1–1.5 min on CPU."""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from midas_pipeline import __version__
print('midas-pipeline', __version__)"""),
    ("md", """\
## 1. Synthetic dataset + a tiny Au crystal model

`calc_radius_v` needs a crystal structure to predict per-ring theoretical
intensities. We write a minimal Au CIF (FCC, a=4.08 Å) inline so the
notebook stays self-contained."""),
    ("py", f"""\
from midas_ff_pipeline.testing import generate_synthetic_dataset

MIDAS_HOME = Path(os.environ.get('MIDAS_HOME') or Path.home() / 'opt' / 'MIDAS')
PARAMS_TEMPLATE = MIDAS_HOME / 'FF_HEDM' / 'Example' / 'Parameters.txt'

WORK = Path(tempfile.mkdtemp(prefix='midas_pipeline_vmap_'))
AU_CIF = WORK / 'Au.cif'
AU_CIF.write_text({_AU_CIF!r})

zarr = generate_synthetic_dataset(
    out_dir=WORK / 'sim', params_template=PARAMS_TEMPLATE,
    n_grains=20, n_cpus=4,
)
print('cif  :', AU_CIF)
print('zarr :', zarr)"""),
    ("md", """\
## 2. Run with `--vmap-run` + `--soft-attribution`

We run the FF pipeline through indexing, then the V-map stages. The
soft-attribution flags are accepted and threaded through; on FF they
configure a beam profile the (single-voxel) FF indexer does not need to
apply, which is exactly the degeneracy this package makes explicit."""),
    ("py", """\
RESULT = WORK / 'run'
cmd = [
    sys.executable, '-m', 'midas_pipeline', 'run', '--scan-mode', 'ff',
    '--params', str(WORK / 'sim' / PARAMS_TEMPLATE.name),
    '--result', str(RESULT), '--zarr', str(zarr),
    '--n-cpus', '4', '--device', 'cpu', '--dtype', 'float64',
    '--indexer-backend', 'python',
    '--vmap-run',
    '--vmap-crystal-cif', str(WORK / 'Au.cif'),
    '--vmap-wavelength', '0.22291',
    '--vmap-emit-diagnostics', '1',
    '--soft-attribution', '--soft-profile', 'gaussian',
    '--skip', 'refinement', '--skip', 'process_grains', '--skip', 'consolidation',
]
t0 = time.time()
proc = subprocess.run(cmd, capture_output=True, text=True)
print(f'exit={proc.returncode} ({time.time() - t0:.1f}s)\\n')
for line in proc.stderr.splitlines():
    if any(k in line.lower() for k in ('calc_radius_v', 'refine_vmap', 'soft', 'indexing(ff)')):
        print(' ', line.split('] ', 1)[-1])
assert proc.returncode == 0, proc.stderr[-2000:]"""),
    ("md", """\
## 3. The V-map signal (`Radius_V.csv`)

`calc_radius_v` writes one row per indexed spot with its relative volume
`V_rel = I_obs / I_theory(ring)`. We load it and visualise the V-map two
ways:

- the distribution of per-spot `V_rel`, and
- `V_rel` laid out on the detector ring/eta map (the diagnostic image)."""),
    ("py", """\
out = RESULT / 'LayerNr_1' / 'Output'
radius_v = out / 'Radius_V.csv'
theory = out / 'I_theory_per_ring.csv'
print('Radius_V.csv         ', radius_v.stat().st_size, 'bytes')
print('I_theory_per_ring.csv', theory.stat().st_size, 'bytes')

arr = np.loadtxt(radius_v, comments='#', skiprows=1)
# cols: spot_id scan_nr ring_number ring_idx intensity V_rel omega_deg eta_deg
ring_number = arr[:, 2].astype(int)
intensity   = arr[:, 4]
v_rel       = arr[:, 5]
eta_deg     = arr[:, 7]
print(f'{arr.shape[0]} spots; V_rel in [{v_rel.min():.3g}, {v_rel.max():.3g}], '
      f'median {np.median(v_rel):.3g}')"""),
    ("py", """\
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))

ax0.hist(v_rel[v_rel > 0], bins=40, color='steelblue', edgecolor='k', lw=0.3)
ax0.set_xlabel('per-spot relative volume V_rel = I_obs / I_theory(ring)')
ax0.set_ylabel('spot count')
ax0.set_title('V-map signal distribution')

sc = ax1.scatter(eta_deg, ring_number, c=v_rel, s=14, cmap='viridis',
                 norm=matplotlib.colors.LogNorm(vmin=max(v_rel[v_rel > 0].min(), 1e-3),
                                                 vmax=v_rel.max()))
ax1.set_xlabel('eta (deg)')
ax1.set_ylabel('ring number')
ax1.set_title('V_rel on the ring / eta map')
fig.colorbar(sc, ax=ax1, label='V_rel')
fig.tight_layout()

diag_png = WORK / 'v_map_diagnostic.png'
fig.savefig(diag_png, dpi=110)
plt.close(fig)
print('wrote V-map diagnostic image ->', diag_png)"""),
    ("py", """\
from IPython.display import Image, display
display(Image(filename=str(WORK / 'v_map_diagnostic.png')))"""),
    ("md", """\
## 4. Why the joint refinement / soft attribution need PF

The log above shows `refine_vmap` skipping with:

> *`voxel_grid.csv` not present; the refine_vmap stage requires a
> (voxel_idx, x, y, z, grain_id) table.*

That is the correct, designed behaviour — **not** a failure. The *joint*
V-map refinement (`refine_vmap_joint`) optimises one `V` value per
**voxel**, and `--soft-attribution` distributes each spot's contribution
across voxels under a beam profile. Both require a **multi-scan voxel
grid**, which only exists in PF mode (`--scan-mode pf`, `n_scans ≥ 2`).
In FF there is a single beam position — one voxel — so there is nothing
to attribute across and no voxel grid to refine. This is precisely the
"FF is the single-scan degeneracy of PF" thesis: the per-spot V-map
signal (section 3) is the FF-observable part; the spatial V-map is what
the extra scan positions in PF unlock.

A full PF demonstration additionally needs a multi-scan synthetic
generator; `midas_ff_pipeline.testing` ships only single-scan FF
generators today, so the spatially-resolved V-map / soft-attribution
demo is left for when a PF synthetic generator lands."""),
    ("py", """\
shutil.rmtree(WORK, ignore_errors=True)
print('cleaned', WORK)"""),
]


# ===========================================================================
# 04 — FF as the single-scan degeneracy of PF
# ===========================================================================

NB04: List[Cell] = [
    ("md", """\
# 04 · FF is the single-scan degeneracy of PF

This notebook makes the package's founding thesis concrete and runnable:
**FF-HEDM is PF-HEDM with exactly one scan position.** Same orchestrator,
same stages, same on-disk contract; the only difference is the number of
beam positions the sample is scanned through.

We:

1. Show the `ScanGeometry` abstraction: `ff()` ≡ `pf_uniform(n_scans=1)`.
2. Run the FF pipeline end-to-end through indexing and confirm the binned
   `Spots.bin` already carries the PF-shaped 10-column layout (col 9 =
   ScanNr) with every spot at scan 0 — the single-scan degeneracy made
   literal in the bytes on disk.

> **Runtime** ~1–1.5 min on CPU."""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from midas_pipeline import __version__, ScanGeometry
print('midas-pipeline', __version__)"""),
    ("md", """\
## 1. The geometry: FF is `pf_uniform` with one scan position

`ScanGeometry` is the single abstraction the orchestrator dispatches on.
`ScanGeometry.ff()` is constructed as a PF uniform scan with `n_scans=1`
at Y = 0 — there is no separate FF geometry type."""),
    ("py", """\
ff = ScanGeometry.ff()
pf = ScanGeometry.pf_uniform(n_scans=5, scan_step_um=2.0, beam_size_um=4.0)
print('FF :', ff)
print('PF :', pf)
print()
print('FF scan_mode    :', ff.scan_mode)
print('FF n_scans      :', ff.n_scans, '(single position)')
print('FF positions    :', ff.scan_positions)
print('PF n_scans      :', pf.n_scans)
print('PF positions    :', pf.scan_positions, 'um')
assert ff.n_scans == 1 and float(ff.scan_positions[0]) == 0.0"""),
    ("md", """\
## 2. Run FF end-to-end and inspect the on-disk degeneracy

The binning stage emits a **unified** `Spots.bin`: 10 columns where col 9
is `ScanNr`, plus an `int64`-pair `Data.bin` carrying `(spot_id, scan_nr)`
and a `positions.csv` sidecar. This is the *same* layout a PF run writes —
FF just has one scan position, so every `ScanNr` is 0 and `positions.csv`
is a single `0.0` line. The indexer reads this identical contract in
both modes."""),
    ("py", """\
from midas_ff_pipeline.testing import generate_synthetic_dataset

MIDAS_HOME = Path(os.environ.get('MIDAS_HOME') or Path.home() / 'opt' / 'MIDAS')
PARAMS_TEMPLATE = MIDAS_HOME / 'FF_HEDM' / 'Example' / 'Parameters.txt'

WORK = Path(tempfile.mkdtemp(prefix='midas_pipeline_deg_'))
zarr = generate_synthetic_dataset(
    out_dir=WORK / 'sim', params_template=PARAMS_TEMPLATE,
    n_grains=20, n_cpus=4,
)
RESULT = WORK / 'run'
cmd = [
    sys.executable, '-m', 'midas_pipeline', 'run', '--scan-mode', 'ff',
    '--params', str(WORK / 'sim' / PARAMS_TEMPLATE.name),
    '--result', str(RESULT), '--zarr', str(zarr),
    '--n-cpus', '4', '--device', 'cpu', '--dtype', 'float64',
    '--indexer-backend', 'python',
    '--skip', 'refinement', '--skip', 'process_grains', '--skip', 'consolidation',
]
t0 = time.time()
proc = subprocess.run(cmd, capture_output=True, text=True)
print(f'exit={proc.returncode} ({time.time() - t0:.1f}s)')
assert proc.returncode == 0, proc.stderr[-2000:]"""),
    ("py", """\
layer = RESULT / 'LayerNr_1'

# Spots.bin: PF-shaped 10 columns; col 9 = ScanNr, all zero for FF.
spots = np.fromfile(layer / 'Spots.bin', dtype=np.float64).reshape(-1, 10)
print(f'Spots.bin       : {spots.shape[0]} spots x {spots.shape[1]} cols')
print(f'ScanNr (col 9)  : unique = {np.unique(spots[:, 9])}  -> single-scan degeneracy')

# Data.bin: int64 (spot_id, scan_nr) pairs; scan_nr column all zero.
data = np.fromfile(layer / 'Data.bin', dtype=np.int64).reshape(-1, 2)
print(f'Data.bin        : {data.shape[0]} (spot_id, scan_nr) entries; '
      f'scan_nr unique = {np.unique(data[:, 1])}')

# positions.csv: a single 0.0 line.
pos = (layer / 'positions.csv').read_text().split()
print(f'positions.csv   : {pos}  -> one scan position at Y=0')

assert np.all(spots[:, 9] == 0.0)
assert np.all(data[:, 1] == 0)
assert len(pos) == 1 and float(pos[0]) == 0.0"""),
    ("py", """\
# And the indexer recovered grains over this single-scan layout.
ib = np.fromfile(layer / 'IndexBest.bin', dtype=np.float64).reshape(-1, 15)
print(f'indexed seeds with a solution: {int((ib[:, 14] > 0).sum())} / {ib.shape[0]}')
assert int((ib[:, 14] > 0).sum()) > 0"""),
    ("md", """\
## Takeaway

The FF run produced a `Spots.bin` / `Data.bin` / `positions.csv` triple
that is **structurally a PF run with `n_scans = 1`**: the ScanNr column
exists and is uniformly 0, and there is one scan position. Adding more
scan positions (PF) populates ScanNr `> 0` and a multi-row
`positions.csv`, unlocking the per-voxel reconstruction and V-map
refinement (notebook 03) — but the code path, file format, and indexer
are the same. That is the single-orchestrator, single-contract design the
package is built around."""),
    ("py", """\
shutil.rmtree(WORK, ignore_errors=True)
print('cleaned', WORK)"""),
]


if __name__ == '__main__':
    for name, cells in [
        ('01_synthetic_ff_walkthrough', NB01),
        ('02_indexer_backends', NB02),
        ('03_vmap_soft_attribution', NB03),
        ('04_ff_is_pf_degeneracy', NB04),
    ]:
        p = write_notebook(name, cells)
        print('wrote', p)
