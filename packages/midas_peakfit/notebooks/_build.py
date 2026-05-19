"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where kind is
"md" or "py". The .ipynb files are derived artefacts; this file is the source
of truth.

Usage:
    cd packages/midas_peakfit/notebooks
    python _build.py                    # rebuild all notebooks
    python _build.py 01_batched_lm_peakfit
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
# NB 01 — Standalone batched LM peak fitting
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — Batched LM peak fitting with `midas_peakfit`

`midas_peakfit` is the differentiable-PyTorch replacement for the C tool
`PeaksFittingOMPZarrRefactor`. It is the **first** stage of FF-HEDM analysis:
it finds and fits 2D pseudo-Voigt peaks on each detector frame of a
`*.MIDAS.zip` Zarr archive and writes the binary peak files the rest of the
pipeline indexes.

Key differences from the C tool:

- **Optimiser:** NLopt Nelder-Mead → batched **Levenberg-Marquardt** (autograd).
- **Backend:** OpenMP → PyTorch — runs on CPU or CUDA, fp64 or fp32.
- **Output:** byte-compatible `AllPeaks_PS.bin` / `AllPeaks_PX.bin`.

This notebook builds a tiny synthetic `*.MIDAS.zip` from scratch (the exact
schema the test suite uses), runs the full pipeline on **CPU/fp64**, and reads
back the two binary outputs. A few seconds, no GPU, no real data.
"""),
    ("py", """\
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # torch + OpenMP on macOS

import tempfile
from pathlib import Path

import numpy as np
import zarr

import midas_peakfit
print('midas_peakfit', midas_peakfit.__version__)
"""),
    ("md", """\
## 1 — Build a synthetic `*.MIDAS.zip`

The pipeline reads a Zarr archive with the image stack under `exchange/data`
and the geometry/analysis parameters under
`analysis/process/analysis_parameters` + `measurement/process/scan_parameters`.

We plant **pseudo-Gaussian peaks** at known positions across 3 frames:

| frame | planted peaks (Y, Z) |
|-------|----------------------|
| 0 | (60, 70) and (180, 200) |
| 1 | (128, 128) |
| 2 | none (flat background) |

so we can verify the fit recovers them. (Schema mirrors
`tests/conftest.py::synthetic_zarr`.)
"""),
    ("py", """\
tmpdir = Path(tempfile.mkdtemp(prefix='pf_demo_'))
zip_path = tmpdir / 'synthetic.MIDAS.zip'

nFrames, NrPixelsZ, NrPixelsY = 3, 256, 256
Y = np.arange(NrPixelsY, dtype=np.float64)
Z = np.arange(NrPixelsZ, dtype=np.float64)
Yg, Zg = np.meshgrid(Y, Z, indexing='xy')      # (Z, Y)

def gauss2d(y0, z0, amp, sig):
    return amp * np.exp(-((Yg - y0) ** 2 + (Zg - z0) ** 2) / (2 * sig * sig))

data = np.zeros((nFrames, NrPixelsZ, NrPixelsY), dtype=np.uint16)
data[0] = (gauss2d(60, 70, 1500, 4) + gauss2d(180, 200, 2200, 4) + 5).astype(np.uint16)
data[1] = (gauss2d(128, 128, 3000, 4) + 5).astype(np.uint16)
data[2] = np.full_like(data[2], 5)             # background only

with zarr.ZipStore(str(zip_path), mode='w') as store:
    root = zarr.open_group(store=store, mode='w')
    root.create_dataset('exchange/data', data=data, chunks=(1, NrPixelsZ, NrPixelsY))
    ap = root.require_group('analysis/process/analysis_parameters')
    sp = root.require_group('measurement/process/scan_parameters')

    # Geometry + ring/peak controls.
    ap.create_dataset('YCen', data=np.array([128.0]))       # beam center Y (px)
    ap.create_dataset('ZCen', data=np.array([128.0]))       # beam center Z (px)
    ap.create_dataset('PixelSize', data=np.array([200.0]))  # um/px
    ap.create_dataset('Lsd', data=np.array([1e6]))          # sample-detector (um)
    ap.create_dataset('Wavelength', data=np.array([0.18]))  # Angstrom
    ap.create_dataset('RhoD', data=np.array([NrPixelsY * 200.0]))
    ap.create_dataset('Width', data=np.array([10000.0]))    # ring half-width (um)
    ap.create_dataset('DoFullImage', data=np.array([1], dtype=np.int32))
    ap.create_dataset('RingThresh', data=np.array([[1, 50.0]]))   # ring 1, thresh 50
    ap.create_dataset('MinNrPx', data=np.array([3], dtype=np.int32))
    ap.create_dataset('MaxNrPx', data=np.array([10000], dtype=np.int32))
    ap.create_dataset('MaxNPeaks', data=np.array([20], dtype=np.int32))
    ap.create_dataset('UpperBoundThreshold', data=np.array([14000.0]))
    ap.create_dataset('ResultFolder', data=np.bytes_(str(tmpdir).encode()))

    sp.create_dataset('start', data=np.array([0.0]))        # omega start (deg)
    sp.create_dataset('step', data=np.array([1.0]))         # omega step (deg)
    sp.create_dataset('doPeakFit', data=np.array([1], dtype=np.int32))

print('wrote', zip_path, f'({zip_path.stat().st_size/1024:.0f} KB)')
"""),
    ("md", """\
## 2 — Run the pipeline (CPU, fp64)

`midas_peakfit.orchestrator.run` is the programmatic entry point. The first
four positional args mirror the C tool exactly:
`data_file, block_nr, n_blocks, num_procs`. Block 0 of 1 = process all frames.

`device` and `dtype` are the two backend knobs:

- `device='cpu'` (here) or `'cuda'`.
- `dtype='float64'` (default, what the C tool uses) or `'float32'` for speed.
"""),
    ("py", """\
from midas_peakfit.orchestrator import run

summary = run(
    str(zip_path),
    block_nr=0, n_blocks=1, num_procs=1,    # process all frames in one block
    result_folder_cli=str(tmpdir),
    device='cpu',                            # CPU only — no CUDA required
    dtype='float64',                         # fp64 (use 'float32' for speed)
)
print('\\nsummary:', {k: summary[k] for k in ('n_frames_done', 'total_time')})
print('PS file:', summary['ps_path'])
print('PX file:', summary['px_path'])
"""),
    ("md", """\
## 3 — Read `AllPeaks_PS.bin` (the peak summary)

`AllPeaks_PS.bin` holds, per frame, a `(nPeaks, 29)` table of fitted peak
parameters. Column 0 is SpotID; columns 3 and 4 are the fitted peak centre
`(YCen, ZCen)` in pixels; column 18 is the LM return code (0 = converged).
The decoder `read_ps` returns a `PSData` with `n_frames`, `n_peaks` (per
frame), and `rows_per_frame` (a list of `(nPeaks, 29)` arrays).
"""),
    ("py", """\
from midas_peakfit.compat.reference_decoder import read_ps
from midas_peakfit.postfit import N_PEAK_COLS

ps = read_ps(summary['ps_path'])
print(f'PS columns per peak : {N_PEAK_COLS}')
print(f'frames              : {ps.n_frames}')
print(f'peaks per frame     : {ps.n_peaks.tolist()}   (expect [2, 1, 0])')

print('\\nFrame 0 fitted peak centres (YCen, ZCen) vs planted:')
planted = {0: [(60.0, 70.0), (180.0, 200.0)], 1: [(128.0, 128.0)]}
for f in (0, 1):
    rows = ps.rows_per_frame[f]
    yz = rows[:, [3, 4]]
    print(f'  frame {f}:')
    for (ty, tz) in planted[f]:
        d = np.linalg.norm(yz - np.array([ty, tz]), axis=1)
        i = int(d.argmin())
        print(f'    planted ({ty:6.1f}, {tz:6.1f}) -> fitted '
              f'({yz[i, 0]:7.3f}, {yz[i, 1]:7.3f})  |Δ|={d[i]:.3f} px  '
              f'returnCode={int(rows[i, 18])}')
"""),
    ("md", """\
## 4 — Read `AllPeaks_PX.bin` (the per-peak pixel sets)

`AllPeaks_PX.bin` stores, for every peak, the list of detector pixels that
belonged to that peak's connected region. `read_px` returns a `PXData` with
`n_frames`, `n_peaks`, `nr_pixels` (the detector edge length), and
`pixels_per_frame` — a per-frame list of `(pixel_y, pixel_z)` arrays, one per
peak.
"""),
    ("py", """\
from midas_peakfit.compat.reference_decoder import read_px

px = read_px(summary['px_path'])
print(f'frames          : {px.n_frames}')
print(f'peaks per frame : {px.n_peaks.tolist()}')
print(f'detector size   : {px.nr_pixels} px')

# Pixel count of each peak in frame 0.
for pk, (py, pz) in enumerate(px.pixels_per_frame[0]):
    print(f'  frame 0 peak {pk}: {py.size} pixels, '
          f'Y∈[{py.min()},{py.max()}]  Z∈[{pz.min()},{pz.max()}]')
"""),
    ("md", """\
## 5 — Switching device / dtype

The same call runs on `float32` (faster, slightly looser) or on CUDA if a GPU
is present. Here we re-run on CPU/float32 and confirm the peaks still land in
the same place — the LM converges to the same basin regardless of precision.
"""),
    ("py", """\
summary32 = run(
    str(zip_path), block_nr=0, n_blocks=1, num_procs=1,
    result_folder_cli=str(tmpdir / 'f32'),
    device='cpu', dtype='float32',
)
ps32 = read_ps(summary32['ps_path'])
yz64 = ps.rows_per_frame[0][:, [3, 4]]
yz32 = ps32.rows_per_frame[0][:, [3, 4]]
# Match each fp32 peak to the nearest fp64 peak.
maxdiff = max(np.linalg.norm(yz64 - p, axis=1).min() for p in yz32)
print(f'fp32 peaks per frame: {ps32.n_peaks.tolist()}')
print(f'max fp32-vs-fp64 centre disagreement: {maxdiff:.4f} px')
print('\\nTo run on a GPU instead:  run(..., device=\"cuda\", dtype=\"float32\")')
"""),
    ("md", """\
## Recap

- Built a synthetic 3-frame `*.MIDAS.zip` with peaks planted at known
  positions.
- Ran the batched-LM peak fitter on **CPU/fp64** via
  `midas_peakfit.orchestrator.run` — it recovered every planted peak to
  sub-pixel accuracy with `returnCode=0`.
- Read both binary outputs: `AllPeaks_PS.bin` (29-column peak summary) and
  `AllPeaks_PX.bin` (per-peak pixel sets).
- Showed the `device` / `dtype` knobs; fp32 lands in the same basin as fp64.

**On real data:** point `run(...)` (or the `peakfit_torch DataFile.MIDAS.zip
blockNr nBlocks numProcs` CLI) at a real `*.MIDAS.zip` produced by
`midas_zipper`. The output feeds directly into the FF-HEDM indexing stage.
"""),
]


NOTEBOOKS = {
    "01_batched_lm_peakfit": NB_01,
}


def main(argv: List[str]) -> None:
    targets = argv or list(NOTEBOOKS)
    for name in targets:
        if name not in NOTEBOOKS:
            raise SystemExit(f"unknown notebook {name!r}; choices: {list(NOTEBOOKS)}")
        print(f"wrote {write_notebook(name, NOTEBOOKS[name])}")


if __name__ == "__main__":
    main(sys.argv[1:])
