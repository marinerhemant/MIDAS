"""Build .ipynb files from a maintainable cell-list source.

The .ipynb files are derived artefacts; this file is the source of
truth. Run once to (re)generate every notebook in this directory.

Usage:
    cd packages/midas_index/notebooks
    python _build.py                       # rebuild all
    python _build.py 01_ff_indexing        # rebuild one
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
        return {"id": cell_id, "cell_type": "markdown", "metadata": {}, "source": src_lines}
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
# NB 01 — single-grain FF indexing on a small synthetic dataset
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — FF-HEDM indexing on a small synthetic dataset

`midas-index` is a pure-Python / PyTorch FF-HEDM indexer — a
drop-in for the C `IndexerOMP`, with one device switch for
CPU / CUDA / MPS. This notebook runs the **Python backend on CPU**
over a small synthetic FF dataset that ships with the package's
tests (5 Cu grains, 4 rings).

By the end you will have:

1. Loaded a `paramstest.txt` with `Indexer.from_param_file`.
2. Loaded the binary observations (`Spots.bin`, `Data.bin`,
   `nData.bin`, `hkls.csv`, `SpotsToIndex.csv`).
3. Run the indexer on a few seed spots.
4. Read out per-seed orientation, position, and match statistics.

It runs in well under a minute on a laptop CPU.
"""),
    ("py", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+numba libomp coexist
import shutil, tempfile, time
from pathlib import Path

import numpy as np
import torch

from midas_index import Indexer
from midas_index import backend_c

print("torch:", torch.__version__, "| device: cpu")
# The bundled C binary is an optional accelerator. We probe it here but
# run the portable Python backend below regardless.
print("C backend (midas_indexer) available:", backend_c.available())
"""),
    ("md", """\
## 1. Stage the synthetic dataset

The package ships a tiny FF reference dataset under
`tests/data/ref_dataset/` (5 grains, 4 rings). We copy it into a
scratch workspace and point `OutputFolder` at a writable subdir.
The indexer reads `hkls.csv` by relative path, so we `chdir` into the
workspace.
"""),
    ("py", """\
import midas_index
PKG = Path(midas_index.__file__).resolve().parent
DATA = PKG.parent / "tests" / "data" / "ref_dataset"
assert DATA.exists(), f"reference dataset not found at {DATA}"

WORK = Path(tempfile.mkdtemp(prefix="midas_index_nb_"))
for f in ("paramstest.txt", "Spots.bin", "Data.bin", "nData.bin",
          "hkls.csv", "SpotsToIndex.csv"):
    shutil.copy2(DATA / f, WORK / f)
(WORK / "out").mkdir()

# Re-point OutputFolder at our scratch dir.
import re
ptxt = (WORK / "paramstest.txt").read_text()
ptxt = re.sub(r"OutputFolder .*", f"OutputFolder {WORK}/out", ptxt)
(WORK / "paramstest.txt").write_text(ptxt)

os.chdir(WORK)
print("workspace:", WORK)
print()
print((WORK / "paramstest.txt").read_text())
"""),
    ("md", """\
## 2. Build the indexer and load observations

`Indexer.from_param_file` parses `paramstest.txt` into typed
`IndexerParams`. `load_observations(cwd=".")` reads the binary spot
table, the (Data/nData) spatial bins, the hkls, and the list of seed
spot IDs.
"""),
    ("py", """\
indexer = Indexer.from_param_file("paramstest.txt", device="cpu", dtype="float64")
indexer.load_observations(cwd=".")

obs = indexer._observations
print("observed spots :", obs["spots"].shape)
print("hkls (real)    :", obs["hkls_real"].shape)
print("seed spot ids  :", obs["spot_ids"][:10], "...")
print("ring numbers   :", indexer.params.RingNumbers)
print("space group    :", indexer.params.SpaceGroup,
      "| lattice:", indexer.params.LatticeConstant)
"""),
    ("md", """\
## 3. Run the indexer

`run()` evaluates each seed spot: it enumerates candidate
orientations, forward-simulates their theoretical spots, matches
against the observed bins, and keeps the best (orientation, position)
tuple. We index the first few seeds to keep it fast; drop
`n_spots_to_index` to index them all.
"""),
    ("py", """\
t0 = time.time()
result = indexer.run(
    block_nr=0, n_blocks=1,
    n_spots_to_index=5,    # first 5 seeds; None = all
    num_procs=1,
    backend="python",      # portable path. "c-omp" uses the bundled binary.
)
print(f"indexed {len(result.seeds)} seeds in {time.time() - t0:.2f} s")
"""),
    ("md", """\
## 4. Inspect the recovered grains

Each `SeedResult` carries the best orientation matrix, the sample-frame
position, and the matched-spot bookkeeping (`n_matches` of `n_t_spots`
theoretical spots, mean internal angle `avg_ia`).
"""),
    ("py", """\
print(f"{'spot':>5} {'n_match':>7} {'n_theor':>7} {'frac':>6} {'avg_ia(deg)':>11}")
for s in result.seeds:
    print(f"{s.spot_id:5d} {s.n_matches:7d} {s.n_t_spots:7d} "
          f"{s.frac_matches:6.2f} {float(s.avg_ia):11.4f}")

if result.seeds:
    s0 = result.seeds[0]
    print()
    print("seed", s0.spot_id, "best orientation matrix (row-major 3x3):")
    print(np.asarray(s0.best_or_mat).reshape(3, 3))
    print("sample-frame position (µm):", np.asarray(s0.best_pos))
"""),
    ("md", """\
## Notes

- **Device portability**: pass `device="cuda"` or `device="mps"` to
  `from_param_file`; everything else is identical. (This machine is
  CPU-only.)
- **C backend**: `run(..., backend="c-omp")` shells out to the bundled
  `midas_indexer` binary (when `backend_c.available()` is `True`),
  writing consolidated `IndexBest_all.bin` to `OutputFolder`. The
  Python backend is bit-identical to C `IndexerOMP` on the parity gate.
- **CLI**: the same run is `midas-index paramstest.txt 0 1 1000 8`.

See [02 — soft beam attribution](02_soft_attribution.ipynb) and
[03 — scanning / PF indexing](03_scanning_pf.ipynb).
"""),
]


# =====================================================================
# NB 02 — soft beam attribution
# =====================================================================

NB_02: List[Cell] = [
    ("md", """\
# 02 — Soft beam attribution

In scanning / PF-HEDM the indexer must decide which observed spots a
candidate voxel "owns", based on how far the voxel's projected
position sits from the illuminating beam. The legacy behavior is a
**hard window**: a spot counts iff it falls within `ScanPosTol` µm.

`midas-index` generalizes this into **soft attribution**: each
candidate match gets a *weight* in `[0, 1]` from a beam-profile
function. The C backend writes these weights to the
`IndexBest_weights_all.bin` sidecar (1.0 everywhere in the legacy
`none` mode, so downstream code can always rely on the file). This
notebook demonstrates the weighting kernels and how they flow into
`compare_spots` scoring — all on CPU, in a couple of seconds.
"""),
    ("py", """\
import os
# torch + numba (pulled in by the matcher) can both ship a libomp on
# macOS; allow the duplicate so the kernel doesn't abort. Matches the
# package's own test harness.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import torch

from midas_index.compute.soft_attribution import (
    hard_window_fn, soft_top_hat_fn, soft_gaussian_fn,
)

print("torch:", torch.__version__)
"""),
    ("md", """\
## 1. The three attribution kernels

Each kernel maps a beam-distance `d` (µm) to a weight in `[0, 1]`:

- `hard_window_fn(tol)` — the legacy binary filter (`1` if `d < tol`).
- `soft_top_hat_fn(width, fall_off)` — flat top, linear edge ramp.
- `soft_gaussian_fn(fwhm)` — Gaussian; weight = 0.5 at `d = FWHM/2`.

Configured in `paramstest.txt` via `SoftAttrMode` /
`SoftAttrFwhm` / `SoftAttrTruncate` / `SoftAttrFalloff`.
"""),
    ("py", """\
d = torch.linspace(0, 12, 13, dtype=torch.float64)

hard = hard_window_fn(5.0)(d)
tophat = soft_top_hat_fn(10.0, fall_off_um=4.0)(d)     # half-width 5, ramp to 9
gauss = soft_gaussian_fn(10.0)(d)                       # 0.5 at d = 5

print(f"{'d(µm)':>6} {'hard(<5)':>9} {'top_hat':>9} {'gaussian':>9}")
for i in range(len(d)):
    print(f"{float(d[i]):6.1f} {float(hard[i]):9.3f} "
          f"{float(tophat[i]):9.3f} {float(gauss[i]):9.3f}")

# Sanity checks matching the package's tests.
assert math.isclose(float(soft_gaussian_fn(10.0)(torch.tensor([5.0]))[0]), 0.5, abs_tol=1e-9)
assert math.isclose(float(soft_top_hat_fn(10.0, fall_off_um=4.0)(torch.tensor([6.0]))[0]), 0.75, rel_tol=1e-9)
print("\\nGaussian half-max at FWHM/2 ✓   top-hat ramp value ✓")
"""),
    ("md", """\
## 2. Differentiable in the distance

The soft kernels are smooth, so the per-match weight carries an
autograd gradient w.r.t. the voxel↔beam distance — useful for joint
calibration / grain-mapping. (The hard window is, by design, not
differentiable.)
"""),
    ("py", """\
dd = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
w = soft_gaussian_fn(10.0)(dd).sum()
w.backward()
print("d/d(distance) of total gaussian weight:", dd.grad.tolist())
assert torch.autograd.gradcheck(
    lambda x: soft_gaussian_fn(10.0)(x).sum(),
    (torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True),),
    eps=1e-6, atol=1e-7,
)
print("gradcheck passed ✓")
"""),
    ("md", """\
## 3. Weights feeding the matcher

`compute.matching.compare_spots` accepts a `soft_beam_weight_fn`. When
provided, it returns `weighted_n_matches` / `weighted_frac_matches`
alongside the integer counts (the values that populate
`IndexBest_weights_all.bin`). We build a minimal single-voxel,
single-theoretical-spot scan setup and compare hard vs soft scoring.
"""),
    ("py", """\
from midas_index.compute.matching import (
    compare_spots, build_eta_margins, build_ome_margins,
)

def matching_kwargs():
    return dict(
        eta_margins=build_eta_margins(
            ring_radii={1: 30000.0}, margin_eta=20.0, stepsize_orient_deg=0.5,
            device=torch.device("cpu"), dtype=torch.float64),
        ome_margins=build_ome_margins(
            margin_ome=10.0, stepsize_orient_deg=0.5,
            device=torch.device("cpu"), dtype=torch.float64),
        eta_bin_size=0.1, ome_bin_size=5.0,
        n_eta_bins=3600, n_ome_bins=72,
        rings_to_reject=torch.tensor([], dtype=torch.int64),
        margin_rad=50.0, margin_radial=50.0,
    )

def build_bin(eta, ome, ring_nr, eta_bin, ome_bin, n_eta, n_ome, n_rows):
    pos = ((ring_nr - 1) * n_eta * n_ome
           + int((180 + eta) / eta_bin) * n_ome
           + int((180 + ome) / ome_bin))
    ndata = torch.zeros(2 * (pos + 10), dtype=torch.int32)
    ndata[2 * pos] = n_rows
    return torch.arange(n_rows, dtype=torch.int32), ndata

# One observed spot at ω=90°; voxel at (10, 0) µm → beam projection = 10 µm.
omega = 90.0
obs = torch.tensor(
    [(10.0, 5.0, omega, 30000.0, 101, 1, 12.0, 1.5, 0.0, 0)],
    dtype=torch.float64)
theor = torch.tensor(
    [[0, 0, 0, 0, 0, 0, omega, 0, 0, 1, 10.0, 5.0, 12.0, 0.0]],
    dtype=torch.float64).unsqueeze(0)
valid = torch.ones((1, 1), dtype=torch.bool)
kw = matching_kwargs()
bin_data, bin_ndata = build_bin(12.0, omega, 1, kw["eta_bin_size"],
                                kw["ome_bin_size"], kw["n_eta_bins"],
                                kw["n_ome_bins"], n_rows=1)

common = dict(
    obs=obs, theor=theor, valid=valid,
    bin_data=bin_data, bin_ndata=bin_ndata,
    ref_rad=torch.tensor([30000.0], dtype=torch.float64),
    scan_positions=torch.tensor([4.0], dtype=torch.float64),  # beam at 4 µm
    voxel_xy=torch.tensor([[10.0, 0.0]], dtype=torch.float64),
    scan_pos_tol_um=20.0, **kw,
)

res_hard = compare_spots(**common)
res_soft = compare_spots(**common, soft_beam_weight_fn=soft_top_hat_fn(10.0, fall_off_um=4.0))

print("beam distance = |10 - 4| = 6 µm")
print("hard counts   : n_matches =", int(res_hard.n_matches.item()),
      "| weighted_n_matches =", res_hard.weighted_n_matches)
print("soft (top-hat): n_matches =", int(res_soft.n_matches.item()),
      "| weighted_n_matches =", float(res_soft.weighted_n_matches[0]))
print()
print("→ both count the spot, but soft attribution down-weights it to")
print("  (5 + 4 - 6) / 4 = 0.75 because it sits on the beam-edge ramp.")
"""),
    ("md", """\
## Recap

- Soft attribution turns the hard `ScanPosTol` window into a weighted
  one; the weights land in `IndexBest_weights_all.bin`.
- Three kernels: `hard_window_fn`, `soft_top_hat_fn`, `soft_gaussian_fn`.
- They are autograd-differentiable in beam distance.
- Enable in production via `SoftAttrMode gaussian` (etc.) in
  `paramstest.txt`; mode `none` preserves legacy behavior bit-for-bit.
"""),
]


# =====================================================================
# NB 03 — scanning / PF indexing
# =====================================================================

NB_03: List[Cell] = [
    ("md", """\
# 03 — Scanning / PF indexing (voxel grid)

The same unified indexer drives **PF-HEDM / scanning** workflows: it
sweeps a grid of sample voxels, and for each voxel runs the seed
pipeline with the scan-position beam filter active. Output is the
consolidated `IndexBest_all.bin` (one block of records per voxel),
readable by the `pf_MIDAS` downstream.

A full scanning fixture is heavy (hundreds of voxels, ~GB bins), so
this notebook builds a **tiny synthetic in-memory case** — a 3×3
voxel grid with minimal observations — to exercise the
`run_scanning` orchestration + the consolidated reader end-to-end in
a couple of seconds on CPU.
"""),
    ("py", """\
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch+numba libomp coexist
import tempfile
from pathlib import Path

import numpy as np
import torch

from midas_index import Indexer
from midas_index.params import IndexerParams
from midas_index.io.consolidated import read_index_best_all

WORK = Path(tempfile.mkdtemp(prefix="midas_index_scan_nb_"))
print("workspace:", WORK)
"""),
    ("md", """\
## 1. Minimal scanning params + observations

We construct `IndexerParams` directly (no file) with the scan filter
enabled (`scan_pos_tol_um`, single-sided Friedel — the C default) and
hand the indexer a minimal observation set so it doesn't read any
files. The voxel grid is the Cartesian product of the 1-D scan
positions.
"""),
    ("py", """\
p = IndexerParams(
    px=200.0, Distance=1e6, Wavelength=0.18,
    SpaceGroup=225,
    EtaBinSize=0.1, OmeBinSize=0.1,
    StepsizeOrient=0.5,
    MarginEta=2.0, MarginOme=0.5, MarginRad=10.0, MarginRadial=10.0,
    RingNumbers=[1],
    RingRadii={1: 30000.0},
    scan_pos_tol_um=2.0,
    friedel_symmetric_scan_filter=True,
    multi_solution_output=True,
)
ind = Indexer(p, device="cpu")

# Minimal 10-column PF observations (one dummy spot, scan_nr=0) so
# load_observations() doesn't need on-disk binaries.
ind._observations = {
    "spots":     np.zeros((1, 10), dtype=np.float64),
    "bin_data":  np.zeros(0, dtype=np.int32),
    "bin_ndata": np.zeros(0, dtype=np.int32),
    "hkls_real": np.zeros((1, 6), dtype=np.float64),
    "hkls_int":  np.zeros((1, 4), dtype=np.int64),
    "spot_ids":  np.zeros(0, dtype=np.int64),
}
print("scan filter tol (µm):", p.scan_pos_tol_um,
      "| friedel-symmetric:", p.friedel_symmetric_scan_filter)
"""),
    ("md", """\
## 2. Run the scanning indexer

`run_scanning` iterates the `n_scans × n_scans` voxel grid built from
`scan_positions` and writes the consolidated `IndexBest_all.bin`. It
returns the number of voxels processed. (Large grids can be sharded
with `voxel_block_nr` / `voxel_n_blocks`.)
"""),
    ("py", """\
scan_positions = np.array([0.0, 5.0, 10.0])     # 3 positions → 3×3 = 9 voxels
out_path = WORK / "IndexBest_all.bin"

n_vox = ind.run_scanning(
    scan_positions=scan_positions,
    out_path=out_path,
    num_procs=1,
    backend="python",
)
print("voxels processed:", n_vox)
print("wrote:", out_path.name, f"({out_path.stat().st_size} bytes)")
"""),
    ("md", """\
## 3. Read back the consolidated output

`read_index_best_all` parses the consolidated binary: a header, a
per-voxel solution-count array, and the records. With no real
observations every voxel has zero solutions — but the byte layout,
voxel count, and reader round-trip are exactly what a production
scanning run produces.
"""),
    ("py", """\
res = read_index_best_all(out_path)
print("n_voxels in file :", res.n_voxels)
print("solutions/voxel  :", res.n_sol_arr.tolist())
print()
print("Voxel grid (Cartesian product of scan positions, µm):")
n = len(scan_positions)
for i in range(n):
    for j in range(n):
        v = i * n + j
        print(f"  voxel {v}: (x={scan_positions[j]:5.1f}, y={scan_positions[i]:5.1f})"
              f"  n_solutions={int(res.n_sol_arr[v])}")
"""),
    ("md", """\
## Recap

- One indexer, two modes: FF is the `n_scans=1` specialization of the
  scanning algorithm; PF activates the per-voxel scan-position filter.
- `Indexer.run_scanning(scan_positions, out_path=…)` writes the
  consolidated `IndexBest_all.bin` over the voxel grid.
- `read_index_best_all` round-trips it.
- For real PF data, point `load_observations(cwd=…)` at a directory
  with 10-column `Spots.bin` + scanning `Data.bin`/`nData.bin`, and
  pass the real `positions.csv` values as `scan_positions`. The
  `c-omp` backend handles production-scale grids.
"""),
]


NOTEBOOKS = {
    "01_ff_indexing": NB_01,
    "02_soft_attribution": NB_02,
    "03_scanning_pf": NB_03,
}


def main(argv):
    if len(argv) > 1:
        for t in argv[1:]:
            if t not in NOTEBOOKS:
                print(f"unknown notebook: {t}\navailable: {list(NOTEBOOKS)}")
                return 1
        for t in argv[1:]:
            print("wrote", write_notebook(t, NOTEBOOKS[t]))
    else:
        for name, cells in NOTEBOOKS.items():
            print("wrote", write_notebook(name, cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
