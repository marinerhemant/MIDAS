"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where kind is
"md" or "py". The .ipynb files are derived artefacts; this file is the source
of truth.

Usage:
    cd packages/midas_process_grains/notebooks
    python _build.py                    # rebuild all notebooks
    python _build.py 01_ff_grain_consolidation
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
        return {"id": cell_id, "cell_type": "markdown", "metadata": {},
                "source": src_lines}
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
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = HERE / f"{name}.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path


# =====================================================================
# NB 01 — FF grain consolidation (c_parity mode)
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — FF grain consolidation with `midas_process_grains` (`c_parity` mode)

`midas_process_grains` is the pure-Python (PyTorch) replacement for
`FF_HEDM/src/ProcessGrains.c`. It is the **final** stage of FF-HEDM analysis:
it takes the per-seed refinement output and consolidates thousands of
overlapping single-grain solutions into a deduplicated grain list with strain
tensors, writing the canonical MIDAS artefacts:

| file | columns | content |
|------|---------|---------|
| `Grains.csv` | 47 | one row per kept grain — orientation, position, lattice, strain |
| `SpotMatrix.csv` | 12 | one row per (grain, matched spot) |
| `GrainIDsKey.csv` | — | one line per kept grain (the cluster→grain map) |

The shippable mode is **`c_parity`**, which mirrors the C source line-for-line
(bit-identical on real data except the Kenesei strain solver). This notebook
runs it end-to-end on a **tiny synthetic run directory** we build from scratch,
using the exact binary schemas the test suite (`tests/conftest.py`) uses — so
no real pipeline output or GPU is needed. CPU, a couple of seconds.

## The pipeline in one sentence

> Stage 1 groups seeds that **share spots** (via `ProcessKey`) **and** are
> within 0.4° misorientation into clusters → Pass A dedups clusters whose
> representatives are within 0.1° and 5 µm → a confidence cut (≥0.05) → write
> the three CSVs.
"""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch + OpenMP on macOS

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import midas_process_grains
print('midas_process_grains', midas_process_grains.__version__)
"""),
    ("md", """\
## 1 — Build a tiny synthetic run directory

The C pipeline (and this package) reads a fixed run-directory layout. We write
the binary inputs by hand. The schemas below match `tests/conftest.py`:

```
<run_dir>/
  paramstest.txt
  hkls.csv
  InputAllExtraInfoFittingAll.csv        # needed only for SpotMatrix.csv
  Results/OrientPosFit.bin               # (n_seeds, 27) float64 — refined seeds
  Results/Key.bin                        # (n_seeds, 2)  int32   — alive flag, NrIDsPerID
  Results/ProcessKey.bin                 # (n_seeds, 5000) int32 — spot-overlap candidates
  Output/IndexBest.bin                   # (n_seeds, 15) float64
  Output/IndexBestFull.bin               # (n_seeds, 5000, 2) float64
  Output/FitBest.bin                     # (n_seeds, 5000, 22) float64 — matched spots
```

**Our scenario:** 4 refined seeds.
- Seeds 1 & 2 are the *same grain* — identical orientation (0.1° apart),
  positions 5 µm and 8 µm, and they reference each other in `ProcessKey`
  (shared spots). → these **merge** into one grain.
- Seeds 3 & 4 are distinct grains (20° and 40° rotated, far away). → kept
  separately.

Expected output: **3 grains**.
"""),
    ("py", """\
def rot_z(deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

RUN_DIR = Path(tempfile.mkdtemp(prefix='pg_demo_'))
(RUN_DIR / 'Output').mkdir()
(RUN_DIR / 'Results').mkdir()

n_seeds = 4
MAX_N_HKLS = 5000
oms = [np.eye(3), rot_z(0.1), rot_z(20.0), rot_z(40.0)]   # seed1≈seed2
positions = np.array([[5, 0, 0], [8, 0, 0], [200, 0, 0], [400, 0, 0]], float)

# --- OrientPosFit.bin: (n, 27) float64 -------------------------------------
# col 0 = seed SpotID; 1..9 = orientation matrix; 11..13 = position (µm);
# 15..20 = lattice (a,b,c,α,β,γ); 22..24 = (DiffPos, DiffOme, IA);
# 25 = GrainRadius; 26 = Confidence. Cols 10/14/21 carry the SpId sentinel.
opf = np.zeros((n_seeds, 27), dtype=np.float64)
for i in range(n_seeds):
    sp = i + 1
    opf[i, 0] = opf[i, 10] = opf[i, 14] = opf[i, 21] = float(sp)
    opf[i, 1:10] = oms[i].reshape(9)
    opf[i, 11:14] = positions[i]
    opf[i, 15:21] = [3.6, 3.6, 3.6, 90.0, 90.0, 90.0]   # FCC Cu
    opf[i, 22:25] = [0.5, 0.05, 0.01 * (i + 1)]          # min IA picks the rep
    opf[i, 25] = 5.0 + i
    opf[i, 26] = 0.95                                    # > 0.05 confidence floor
opf.tofile(RUN_DIR / 'Results' / 'OrientPosFit.bin')

# --- Key.bin: (n, 2) int32 — [alive, NrIDsPerID] ---------------------------
nr_ids = np.array([5, 5, 5, 5], dtype=np.int32)
key = np.zeros((n_seeds, 2), dtype=np.int32)
key[:, 0] = 1            # all alive
key[:, 1] = nr_ids
key.tofile(RUN_DIR / 'Results' / 'Key.bin')

# --- ProcessKey.bin: (n, 5000) int32 — candidate SEED IDs (1-indexed) ------
# Stage 1 only considers merging seeds that reference each other here.
pk = np.zeros((n_seeds, MAX_N_HKLS), dtype=np.int32)
pk[0, :1] = [2]          # seed 1 ↔ seed 2  (these two will merge)
pk[1, :1] = [1]
pk[2, :1] = [2]          # seeds 3, 4: candidates whose misori is too large to merge
pk[3, :1] = [3]
pk.tofile(RUN_DIR / 'Results' / 'ProcessKey.bin')

# --- IndexBest / IndexBestFull / FitBest -----------------------------------
ib = np.zeros((n_seeds, 15), dtype=np.float64)
ibf = np.zeros((n_seeds, MAX_N_HKLS, 2), dtype=np.float64)
fb = np.zeros((n_seeds, MAX_N_HKLS, 22), dtype=np.float64)
for i in range(n_seeds):
    ib[i, 0] = 0.01
    ib[i, 1:10] = oms[i].reshape(9)
    ib[i, 13] = 50.0
    ib[i, 14] = nr_ids[i]
    sids = np.arange(5 * i + 1, 5 * i + 6)   # 5 matched spots/grain, small contiguous IDs
    ibf[i, :5, 0] = sids
    fb[i, :5, 0] = sids
ib.tofile(RUN_DIR / 'Output' / 'IndexBest.bin')
ibf.tofile(RUN_DIR / 'Output' / 'IndexBestFull.bin')
fb.tofile(RUN_DIR / 'Output' / 'FitBest.bin')
print('run dir:', RUN_DIR)
"""),
    ("md", """\
## 2 — Write `paramstest.txt`, `hkls.csv`, and the spot table

`paramstest.txt` carries the geometry and the consolidation thresholds.
`hkls.csv` is the reflection table (a full {111} orbit is enough that every
cubic symmetry op maps inside it). `InputAllExtraInfoFittingAll.csv` carries
the per-spot detector positions — needed only so `SpotMatrix.csv` can be
written; row `N-1` describes `SpotID = N`.
"""),
    ("py", """\
(RUN_DIR / 'paramstest.txt').write_text(
    'LatticeParameter 3.6 3.6 3.6 90.0 90.0 90.0;\\n'
    'Wavelength 0.172979;\\n'
    'Distance 800000.0;\\n'
    'px 200.0;\\n'
    'SpaceGroup 225;\\n'
    'RingNumbers 1;\\n'
    'RingRadii 60000.0;\\n'
    'MinNrSpots 1;\\n'
    'MisoriTol 0.25;\\n'
    'OutputFolder ' + str(RUN_DIR / 'Output') + '\\n'
    'ResultFolder ' + str(RUN_DIR / 'Results') + '\\n'
)

# hkls.csv: full {111} orbit.
hkl_text = 'h k l D-spacing RingNr g1 g2 g3 Theta(deg) 2Theta(deg) Radius\\n'
for h in (-1, 1):
    for k in (-1, 1):
        for l in (-1, 1):
            hkl_text += f'{h} {k} {l} 2.0784 1 0 0 0 2.39 4.78 60000.0\\n'
(RUN_DIR / 'hkls.csv').write_text(hkl_text)

# InputAllExtraInfoFittingAll.csv: 1 header + 20 whitespace rows (SpotID 1..20).
# Column meaning (subset the C loader keeps): 0=YLab 1=ZLab 2=Omega 4=SpotID
# 5=RingNr 6=Eta 7=2θ 11=YOrig(DetH) 12=ZOrig(DetV) 13=OmeRaw.
lines = [' '.join(f'c{j}' for j in range(18)) + '\\n']
for sid in range(1, 21):
    cols = [0.0] * 18
    cols[0]  = 100.0 + sid          # YLab
    cols[1]  = 200.0 + sid          # ZLab
    cols[2]  = 10.0 + 0.1 * sid     # Omega
    cols[4]  = float(sid)           # SpotID
    cols[5]  = 1.0                  # RingNr
    cols[6]  = 5.0 + 0.01 * sid     # Eta
    cols[7]  = 4.78                 # 2θ
    cols[11] = 1024.0 + sid         # DetectorHor
    cols[12] = 1024.0 - sid         # DetectorVert
    cols[13] = 10.0 + 0.1 * sid     # OmeRaw
    lines.append(' '.join(f'{c:.5f}' for c in cols) + '\\n')
(RUN_DIR / 'InputAllExtraInfoFittingAll.csv').write_text(''.join(lines))
print('wrote paramstest.txt, hkls.csv, InputAllExtraInfoFittingAll.csv')
"""),
    ("md", """\
## 3 — Run the `c_parity` consolidation

`run_c_parity_pipeline_from_disk` is the documented library entry point
(`device='cpu'` keeps it GPU-free). The thresholds default to the C values:
Stage-1 misori 0.4°, Pass-A misori 0.1° + 5 µm, confidence ≥ 0.05.
"""),
    ("py", """\
from midas_process_grains.compute.c_parity_run import run_c_parity_pipeline_from_disk

result = run_c_parity_pipeline_from_disk(
    run_dir=RUN_DIR,
    out_dir=RUN_DIR,
    device='cpu',           # CPU only — no CUDA required
    min_nr_spots=1,         # keep every cluster
    write_spot_matrix=True,
)
print('\\nwrote:', sorted(p.name for p in RUN_DIR.glob('*.csv')))
"""),
    ("md", """\
## 4 — Read `Grains.csv` and explain the columns

`Grains.csv` has a `%`-commented header block (NumGrains, lattice, phase info)
then a tab-separated table with these 47 columns:

| group | columns | meaning |
|-------|---------|---------|
| identity | `GrainID` | the rep seed's SpotID |
| orientation | `O11..O33` (9) | row-major orientation matrix |
| position | `X, Y, Z` | grain centroid (µm, lab frame) |
| lattice | `a, b, c, α, β, γ` | refined unit cell |
| fit quality | `DiffPos, DiffOme, DiffAngle, GrainRadius, Confidence` | per-grain residuals |
| strain | `eFab11..eFab33` (9) | Fable-Beaudoin strain (closed form from lattice) |
| strain | `eKen11..eKen33` (9), `RMSErrorStrain` | Kenesei per-spot least-squares strain |
| phase + Euler | `PhaseNr, Eul0, Eul1, Eul2` | phase index + Bunge Euler (**radians**) |

We expect **3 grains** — seeds 1 & 2 collapsed to one (the rep is the one with
the smaller internal angle).
"""),
    ("py", """\
grains_path = RUN_DIR / 'Grains.csv'
print('── %-header ──')
header = [ln for ln in grains_path.read_text().splitlines() if ln.startswith('%')]
print('\\n'.join(header[:8]))

grains = pd.read_csv(grains_path, sep='\\t', comment='%', header=None)
# The data header line (starting with '%GrainID') is commented; assign names.
cols = (['GrainID'] + [f'O{i}{j}' for i in (1, 2, 3) for j in (1, 2, 3)] +
        ['X', 'Y', 'Z', 'a', 'b', 'c', 'alpha', 'beta', 'gamma',
         'DiffPos', 'DiffOme', 'DiffAngle', 'GrainRadius', 'Confidence'] +
        [f'eFab{i}{j}' for i in (1, 2, 3) for j in (1, 2, 3)] +
        [f'eKen{i}{j}' for i in (1, 2, 3) for j in (1, 2, 3)] +
        ['RMSErrorStrain', 'PhaseNr', 'Eul0', 'Eul1', 'Eul2'])
grains.columns = cols[:grains.shape[1]]
print(f'\\n{len(grains)} grains, {grains.shape[1]} columns')
grains[['GrainID', 'X', 'Y', 'Z', 'Confidence', 'Eul0', 'Eul1', 'Eul2']]
"""),
    ("md", """\
## 5 — Read `SpotMatrix.csv` and `GrainIDsKey.csv`

`SpotMatrix.csv` has one row per (kept grain, matched spot) — 12 tab-separated
columns: `GrainID, SpotID, Omega, DetectorHor, DetectorVert, OmeRaw, Eta,
RingNr, YLab, ZLab, Theta, StrainError`. The detector/η/ω values are looked up
from `InputAllExtraInfoFittingAll.csv` by `SpotID`.

`GrainIDsKey.csv` is the compact cluster→grain map (one line per kept grain).
"""),
    ("py", """\
# Each data row ends in a trailing tab (C printf "...%lf\\t\\n"), so pandas sees
# one extra all-NaN column — read headerless, drop it, then name the 12 columns.
sm = pd.read_csv(RUN_DIR / 'SpotMatrix.csv', sep='\\t', comment='%', header=None)
sm = sm.dropna(axis=1, how='all')
sm.columns = ['GrainID', 'SpotID', 'Omega', 'DetectorHor', 'DetectorVert',
              'OmeRaw', 'Eta', 'RingNr', 'YLab', 'ZLab', 'Theta', 'StrainError']
print(f'SpotMatrix: {len(sm)} rows (grain, spot) pairs')
print('spots per grain:')
print(sm.groupby('GrainID')['SpotID'].count())
print('\\n── GrainIDsKey.csv ──')
print((RUN_DIR / 'GrainIDsKey.csv').read_text())
sm.head()
"""),
    ("md", """\
## Recap

- Built a 4-seed synthetic FF-HEDM run directory with the exact binary schemas
  the C `ProcessGrains` consumes.
- Ran `run_c_parity_pipeline_from_disk` (CPU) — Stage 1 merged the two
  co-oriented spot-sharing seeds into one grain, leaving **3 grains**.
- Read and annotated all three outputs: the 47-column `Grains.csv`, the
  12-column `SpotMatrix.csv`, and `GrainIDsKey.csv`.

**On real data:** swap the hand-built binaries for an actual MIDAS run
directory and call the same function (or the CLI
`midas-process-grains paramstest.txt <nCPU> --mode c_parity --device cpu`).
The package is bit-identical to C `ProcessGrains` on every column except the
Kenesei strain tensor (≤ 35 µε, solver-convergence difference).
"""),
]


NOTEBOOKS = {
    "01_ff_grain_consolidation": NB_01,
}


def main(argv: List[str]) -> None:
    targets = argv or list(NOTEBOOKS)
    for name in targets:
        if name not in NOTEBOOKS:
            raise SystemExit(f"unknown notebook {name!r}; choices: {list(NOTEBOOKS)}")
        print(f"wrote {write_notebook(name, NOTEBOOKS[name])}")


if __name__ == "__main__":
    main(sys.argv[1:])
