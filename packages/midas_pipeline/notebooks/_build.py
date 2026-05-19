"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of ``(kind, source)`` tuples where
``kind`` is ``"md"`` or ``"py"``. The .ipynb files are derived
artefacts; this file is the source of truth.

Usage:
    cd packages/midas_pipeline/notebooks
    python _build.py
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
FF-HEDM dataset (no external data, CPU only). We:

1. Forward-simulate a small synthetic single-detector dataset.
2. Drive `midas-pipeline run --scan-mode ff` through its upstream
   stages and inspect per-stage timings + artefacts.
3. Show how `ScanGeometry.ff()` is literally `pf_uniform` with one scan
   position — the degeneracy the package is built around.

> **Runtime** ~1.5–2 min on CPU (the per-frame peak search dominates)."""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # macOS libomp guard

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from midas_pipeline import __version__, ScanGeometry
print('midas-pipeline', __version__)"""),
    ("md", """\
## 1. Forward-simulate a synthetic dataset

`midas-pipeline simulate` is a scaffold in 0.2.0, so for a self-contained
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
    n_grains=50,
    n_cpus=4,
)
print(f'synthetic dataset in {time.time() - t0:.1f}s')
print('zarr  :', zarr)
print('params:', WORK / 'sim' / PARAMS_TEMPLATE.name)"""),
    ("md", """\
## 2. Run the orchestrator (FF mode, upstream stages)

We call the CLI exactly as a user would. `--scan-mode ff` selects the
single-scan stage order. The upstream stages — frame ingest, HKL list,
per-frame peak fitting, overlap merge, radius, lab-frame transforms, and
binning — are all exercised here.

We `--skip` the indexing-and-after stages in this quickstart (see the
note at the bottom): on the current build the FF binning stage writes an
empty `Data.bin`, which the indexer cannot mmap. The upstream half of
the pipeline runs clean and is what this walkthrough showcases."""),
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
    '--skip', 'indexing', '--skip', 'refinement',
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
    ("py", """\
# What artefacts did the upstream stages leave behind?
layer_dir = RESULT / 'LayerNr_1'
interesting = ['hkls.csv', 'InputAll.csv', 'Spots.bin',
               'Result_StartNr_1_EndNr_1440.csv',
               'Radius_StartNr_1_EndNr_1440.csv', 'paramstest.txt']
for name in interesting:
    p = layer_dir / name
    print(f'{name:34s} {"OK " if p.exists() else "-- "} '
          f'{p.stat().st_size if p.exists() else 0:>12,} bytes')"""),
    ("md", """\
## 4. FF is the single-scan degeneracy of PF

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
## What just happened, and what's next

The unified orchestrator ran the FF upstream stages — frame ingest, HKL
generation, per-frame peak fitting, overlap merge, radius assignment,
lab-frame transforms, and binning — all in-process via the shared
`midas-*` kernel packages, then reported provenance through `status`.

**Indexing and downstream (deferred in this notebook).** On the current
0.2.0 build the FF binning stage writes a zero-byte `Data.bin` (the
per-bin spot-index table), so both indexer backends fail with
`cannot mmap an empty file`:

```
ValueError: cannot mmap an empty file        # python backend
mmap ./Data.bin failed: Invalid argument     # c-omp backend
```

Until the binning stage populates `Data.bin`, the full
indexing → refinement → consolidation tail (and therefore a grain
table) is not reproducible from this notebook. The remaining planned
notebooks depend on a working indexing stage:

| Notebook | Topic | Status |
| --- | --- | --- |
| 02 | indexer backend selector (`python` vs `c-omp`) | blocked on indexing |
| 03 | V-map joint refinement + soft beam attribution | blocked on indexing |
| 04 | FF as PF degeneracy, full run | blocked on indexing |

The conceptual FF↔PF degeneracy is shown in section 4 above; the full
end-to-end PF path additionally needs a multi-scan synthetic generator
(`midas-pipeline simulate` is a scaffold in 0.2.0)."""),
    ("py", """\
# Tidy up the scratch workspace.
shutil.rmtree(WORK, ignore_errors=True)
print('cleaned', WORK)"""),
]


if __name__ == '__main__':
    p = write_notebook('01_synthetic_ff_walkthrough', NB01)
    print('wrote', p)
