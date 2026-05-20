"""Build .ipynb files from a maintainable cell-list source.

The .ipynb files are derived artefacts; this file is the source of
truth. Run once to (re)generate every notebook in this directory.

Usage:
    cd packages/midas_transforms/notebooks
    python _build.py                  # rebuild all
    python _build.py 01_per_stage     # rebuild one
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


# Shared synthetic-data helper, embedded in both notebooks so each is
# self-contained. Builds a tiny FF dataset (3 rings × 4 η × N ω frames)
# whose peaks land exactly on the configured ring radii.
SYNTH_HELPER = '''\
import math
import numpy as np
from midas_transforms.merge.core import (
    COL_SPOTID, COL_II, COL_OMEGA, COL_YCEN, COL_ZCEN, COL_IMAX,
    COL_RADIUS, COL_ETA, COL_SIGMAR, COL_SIGMAETA, COL_NRPX,
    COL_NRPXTOT, COL_RAWSUM, N_PEAK_COLS,
)

# Geometry shared across the notebook.
NR_PIXELS = 2048
PX_UM = 200.0          # pixel size
LSD_UM = 1_000_000.0   # sample-detector distance
YCEN = ZCEN = 1024.0   # beam center (px)
RINGS = [1, 2, 3]
RING_RADII_UM = [94949.4, 109761.7, 155930.4]   # ring radii in µm
N_FRAMES = 4
ETAS = [-120.0, -40.0, 40.0, 120.0]

def synth_frames():
    """One (n_peaks, 29) AllPeaks_PS-layout array per ω frame.

    Each peak is placed on its ring at the chosen η, so its observed
    radius matches the hkls ring radius and survives calc_radius.
    """
    frames = []
    sid = 1
    for fi in range(N_FRAMES):
        rows = []
        for rr in RING_RADII_UM:
            r_px = rr / PX_UM
            for eta in ETAS:
                y = YCEN - r_px * math.sin(math.radians(eta))
                z = ZCEN + r_px * math.cos(math.radians(eta))
                row = np.zeros(N_PEAK_COLS)
                row[COL_SPOTID] = sid; sid += 1
                row[COL_II] = 100.0
                row[COL_OMEGA] = fi * 0.25
                row[COL_YCEN] = y; row[COL_ZCEN] = z
                row[COL_IMAX] = 200.0
                row[COL_RADIUS] = r_px        # radial distance (px) → ring radius
                row[COL_ETA] = eta
                row[COL_SIGMAR] = 1.0; row[COL_SIGMAETA] = 1.0
                row[COL_NRPX] = 9; row[COL_NRPXTOT] = 9
                row[COL_RAWSUM] = 100.0
                rows.append(row)
        frames.append(np.array(rows))
    return frames

def write_hkls(path):
    with open(path, "w") as f:
        f.write("h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\\n")
        for rn, rr in zip(RINGS, RING_RADII_UM):
            f.write(f"1 1 1 2.0 {rn} 0 0 0 1.0 2.0 {rr}\\n")
'''


# =====================================================================
# NB 01 — per-stage walkthrough (merge → radius → fit_setup → bin)
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — Per-stage walkthrough: merge → radius → fit-setup → bin

`midas-transforms` is the pure-Python / PyTorch port of the four
C binaries that sit between peak-fitting and indexing in the FF-HEDM
workflow:

| Stage | C binary | Python entry |
|---|---|---|
| 1. merge overlapping peaks | `MergeOverlappingPeaksAllZarr` | `merge_overlapping_peaks` |
| 2. compute ring radii      | `CalcRadiusAllZarr`           | `calc_radius` |
| 3. transform + filter      | `FitSetupZarr`                | `fit_setup` |
| 4. bin for the indexer     | `SaveBinData`                 | `bin_data` |

This notebook runs all four **stage-by-stage on CPU**, feeding each
stage's in-memory output to the next (every stage takes a NumPy
array and a `write=False` flag). The input is a small **synthetic**
FF dataset (3 rings × 4 η × 4 ω frames) — no zarr file needed at
this layer.
"""),
    ("py", SYNTH_HELPER + """
import tempfile
from pathlib import Path
import torch
from midas_transforms.params import ZarrParams

print("torch:", torch.__version__, "| device: cpu")
WORK = Path(tempfile.mkdtemp(prefix="midas_transforms_nb_"))
write_hkls(WORK / "hkls.csv")
print("workspace:", WORK)
"""),
    ("md", """\
## 0. Parameters

The C stages read geometry from the Zarr's `analysis_parameters`
group; here we build the equivalent `ZarrParams` object directly. The
ring radii and `Width` (radial match window, µm) gate which peaks
survive `calc_radius`.
"""),
    ("py", """\
zp = ZarrParams()
zp.Lsd = LSD_UM
zp.Wavelength = 0.18
zp.PixelSize = PX_UM
zp.YCen = YCEN
zp.ZCen = ZCEN
zp.NrPixels = NR_PIXELS
zp.SpaceGroup = 225
zp.LatticeConstant = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
zp.RingThresh = [(rn, 100.0) for rn in RINGS]
zp.OmegaStart = 0.0
zp.OmegaStep = 0.25
zp.Width = 2000.0          # radial match window (µm)
zp.Hbeam = 2000.0
zp.Rsample = 1000.0
zp.MarginRadius = zp.MarginRadial = 2000.0
zp.MarginEta = 500.0
zp.MarginOme = 2.0
zp.EtaBinSize = zp.OmeBinSize = 5.0
zp.StepSizeOrient = 0.2
zp.StepSizePos = 100.0
zp.EndNr = N_FRAMES
print("rings:", RINGS, "| radii (µm):", RING_RADII_UM)
"""),
    ("md", """\
## Stage 1 — merge overlapping peaks

`merge_overlapping_peaks` consolidates peaks that recur across
adjacent ω frames into single spots. We pass the synthetic frames
in-memory via `frames=` (bypassing the on-disk `AllPeaks_PS.bin`)
and `write=False`. Output is the `(N, 17)` merged spot table.
"""),
    ("py", """\
from midas_transforms import merge_overlapping_peaks

frames = synth_frames()
n_in = sum(f.shape[0] for f in frames)
merged = merge_overlapping_peaks(
    frames=frames, result_folder=WORK,
    nr_pixels=NR_PIXELS, device="cpu", dtype="float64", write=False,
)
merge_arr = merged.peaks.cpu().numpy()
print(f"in: {n_in} raw peaks across {len(frames)} frames "
      f"→ out: {merge_arr.shape[0]} merged spots, {merge_arr.shape[1]} cols")
"""),
    ("md", """\
## Stage 2 — compute ring radii

`calc_radius` matches each merged spot to its Bragg ring (radial
window `|R_obs - R_ring| < Width`), computes 2θ, and emits the
`(N, 24)` radius table. The ring radii come from `hkls.csv`.
"""),
    ("py", """\
from midas_transforms import calc_radius

radius = calc_radius(
    result_folder=WORK, zarr_params=zp, result_array=merge_arr,
    start_nr=1, end_nr=N_FRAMES, device="cpu", dtype="float64", write=False,
)
radius_arr = radius.spots.cpu().numpy()
print("radius table:", radius_arr.shape, "(N, 24)")
print("ring numbers found:", sorted(set(int(r) for r in radius_arr[:, 13])))
"""),
    ("md", """\
## Stage 3 — transform + filter (fit-setup)

`fit_setup` applies the detector transform (tilts / distortion / ω
correction), filters to in-spec spots, and produces the 8-column
`InputAll` + the extra-info matrix that `bin_data` consumes. With
`DoFit==1` it would also refine geometry; here we keep it as the
pass-through transform.
"""),
    ("py", """\
from midas_transforms import fit_setup

fs = fit_setup(
    result_folder=WORK, zarr_params=zp, radius_array=radius_arr,
    start_nr=1, end_nr=N_FRAMES, device="cpu", dtype="float64", write=False,
)
inputall = fs.spots_inputall.cpu().numpy()
extra = fs.extra.cpu().numpy()
print("InputAll:", inputall.shape, "(N, 8)")
print("ExtraInfo:", extra.shape)
print("InputAll cols = [YLab, ZLab, Omega, GrainRadius, SpotID, RingNr, Eta, Ttheta]")
print("first spot:", np.round(inputall[0], 3))
"""),
    ("md", """\
## Stage 4 — bin for the indexer

`bin_data` is the drop-in for `SaveBinData`. It writes the four
binaries the indexer reads: `Spots.bin`, `ExtraInfo.bin`, and the
`Data.bin` / `nData.bin` spatial-bin pair. We pass
the fit-setup arrays in-memory and `write=True` to land the files.
"""),
    ("py", """\
from midas_transforms import bin_data

bins = bin_data(
    result_folder=WORK, paramstest=fs.paramstest,
    spots_inputall=inputall, extra_inputall=extra,
    device="cpu", dtype="float64", write=True,
)
print("Spots.bin   :", tuple(bins.spots.shape))
print("ExtraInfo   :", tuple(bins.extra_info.shape))
print("Data / nData:", tuple(bins.data.shape), "/", tuple(bins.ndata.shape))
print("eta/ome/ring bins:", bins.n_eta_bins, bins.n_ome_bins, bins.n_ring_bins)
print()
print("files written to workspace:")
for p in sorted(WORK.glob("*.bin")):
    print(f"  {p.name:16s} {p.stat().st_size} bytes")
"""),
    ("md", """\
## Recap

We ran the full FF intermediate chain stage-by-stage, threading each
stage's in-memory array into the next:

```
frames → merge_overlapping_peaks → calc_radius → fit_setup → bin_data
```

The `*.bin` files now in the workspace are exactly what
[`midas-index`](../../midas_index/) consumes. Notebook
[02](02_pipeline_from_zarr.ipynb) runs the same four stages as one
chained `Pipeline.from_zarr(...)` call, reading a synthetic Zarr
archive end-to-end.
"""),
]


# =====================================================================
# NB 02 — Pipeline.from_zarr on a synthetic Zarr
# =====================================================================

NB_02: List[Cell] = [
    ("md", """\
# 02 — `Pipeline.from_zarr` end-to-end on a synthetic Zarr

Notebook 01 ran the four transforms stage-by-stage. The `Pipeline`
class chains all four with the intermediates kept on-device (no CSV /
binary disk round-trips between stages) and writes only the final
outputs on `dump()`.

This notebook builds a **synthetic `*.MIDAS.zip` Zarr archive** plus
the `AllPeaks_PS.bin` peak blob that the production peak-fitter would
emit, then runs `Pipeline.from_zarr(...).run()` and `.dump()`. CPU
only, a few seconds.
"""),
    ("py", SYNTH_HELPER + """
import tempfile
from pathlib import Path
import zarr
import torch

WORK = Path(tempfile.mkdtemp(prefix="midas_transforms_pipe_"))
(WORK / "Temp").mkdir()
print("torch:", torch.__version__, "| device: cpu")
print("workspace:", WORK)
"""),
    ("md", """\
## 1. Write a synthetic `*.MIDAS.zip`

A MIDAS analysis Zarr stores parameters under
`analysis/process/analysis_parameters`. We write the keys
`read_zarr_params` requires (geometry, ring thresholds, lattice,
margins). This is the same group `midas_zipper` produces from raw
detector data.
"""),
    ("py", """\
zip_fn = WORK / "synthFF_000001.analysis.MIDAS.zip"
store = zarr.ZipStore(str(zip_fn), mode="w")
root = zarr.group(store=store)
ap = root.create_group("analysis").create_group("process").create_group("analysis_parameters")

def put(key, value, dt):
    ap.create_dataset(key, data=np.array(value, dtype=dt))

for key, val in [
    ("Lsd", [LSD_UM]), ("Wavelength", [0.18]), ("PixelSize", [PX_UM]),
    ("YCen", [YCEN]), ("ZCen", [ZCEN]),
    ("tx", [0.0]), ("ty", [0.0]), ("tz", [0.0]),
    ("OmegaStart", [0.0]), ("OmegaStep", [0.25]),
    ("Width", [2000.0]), ("Hbeam", [2000.0]), ("Rsample", [1000.0]),
    ("MarginRadius", [2000.0]), ("MarginRadial", [2000.0]),
    ("MarginEta", [500.0]), ("MarginOme", [2.0]),
    ("EtaBinSize", [5.0]), ("OmeBinSize", [5.0]),
    ("StepSizeOrient", [0.2]), ("StepSizePos", [100.0]),
]:
    put(key, val, np.double)
put("LatticeParameter", [3.6, 3.6, 3.6, 90, 90, 90], np.double)
put("RingThresh", [[rn, 100.0] for rn in RINGS], np.double)
put("NrPixels", [NR_PIXELS], np.int32)
put("SpaceGroup", [225], np.int32)
store.close()

write_hkls(WORK / "hkls.csv")   # ring radii table the radius stage reads
print("wrote", zip_fn.name, f"({zip_fn.stat().st_size} bytes)")
"""),
    ("md", """\
## 2. Write the synthetic `AllPeaks_PS.bin`

`Pipeline` reads the consolidated peak blob the peak-fitter writes
(`Temp/AllPeaks_PS.bin`). Its layout is: `int32 n_frames`, then
per-frame `int32` peak counts, then per-frame `int64` byte offsets,
then the float64 `(n_peaks, 29)` blocks. We serialize our synthetic
frames into exactly that format.
"""),
    ("py", """\
frames = synth_frames()
counts = np.array([f.shape[0] for f in frames], dtype=np.int32)
n_frames = len(frames)
header = 4 + n_frames * 4 + n_frames * 8
offsets, off = [], header
for f in frames:
    offsets.append(off)
    off += f.shape[0] * N_PEAK_COLS * 8

ps_bin = WORK / "Temp" / "AllPeaks_PS.bin"
with open(ps_bin, "wb") as fh:
    fh.write(np.array([n_frames], dtype=np.int32).tobytes())
    fh.write(counts.tobytes())
    fh.write(np.array(offsets, dtype=np.int64).tobytes())
    for f in frames:
        fh.write(f.astype(np.float64).tobytes())
print("wrote", ps_bin.relative_to(WORK), f"({ps_bin.stat().st_size} bytes),"
      f" {int(counts.sum())} peaks across {n_frames} frames")
"""),
    ("md", """\
## 3. Build the pipeline and run all four stages

`Pipeline.from_zarr` parses the Zarr params and locates the
`AllPeaks_PS.bin`. `run()` executes merge → radius → fit-setup → bin
with the intermediates staying as on-device tensors. (We set `EndNr`
to our frame count, which a real scan's Zarr would carry.)
"""),
    ("py", """\
from midas_transforms import Pipeline

pipe = Pipeline.from_zarr(zip_fn, result_folder=WORK, device="cpu", dtype="float64")
pipe.zarr_params.EndNr = N_FRAMES
result = pipe.run()

print("merge      :", tuple(result.merge.peaks.shape), "(N, 17)")
print("radius     :", tuple(result.radius.spots.shape), "(N, 24)")
print("fit_setup  :", tuple(result.fit_setup.spots_inputall.shape), "(N, 8)")
print("bins/Spots :", tuple(result.bins.spots.shape))
print("bins/Data  :", tuple(result.bins.data.shape),
      "| nData:", tuple(result.bins.ndata.shape))
"""),
    ("md", """\
## 4. Dump the outputs

`dump(out_dir)` writes every intermediate + final file the per-stage
CLIs would have produced — including `Spots.bin` / `Data.bin` /
`nData.bin` and `SpotsToIndex.csv`, the exact handoff to
`midas-index`.
"""),
    ("py", """\
out_dir = WORK / "out"
pipe.dump(out_dir)
print("dumped files:")
for p in sorted(out_dir.iterdir()):
    print(f"  {p.name}")
"""),
    ("md", """\
## Recap

- `Pipeline.from_zarr(zarr).run()` chains all four transforms with
  on-device intermediates; `dump()` writes the indexer-ready files.
- The synthetic `*.MIDAS.zip` here mirrors what `midas_zipper`
  produces from raw frames; the `AllPeaks_PS.bin` mirrors what
  `midas-peakfit` produces.
- On a GPU box, pass `device="cuda"` to `from_zarr` — the same code
  path, no `.cu` sources. (This machine is CPU-only.)
- CLI equivalent: `midas-transforms pipeline scan.zip --out-dir …`.
"""),
]


NOTEBOOKS = {
    "01_per_stage": NB_01,
    "02_pipeline_from_zarr": NB_02,
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
