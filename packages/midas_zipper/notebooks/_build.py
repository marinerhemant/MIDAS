"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where
kind is "md" or "py". The .ipynb files are derived artefacts; this
file is the source of truth.

Usage:
    cd packages/midas_zipper/notebooks
    python _build.py                 # rebuild all
    python _build.py 01_raw_to_zarr  # rebuild one
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
# NOTEBOOK SOURCES
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — `midas_zipper`: raw frames → `*.MIDAS.zip`

`midas_zipper` turns raw detector data (HDF5 / GE / TIFF / CBF) into
the `*.MIDAS.zip` Zarr archive the rest of MIDAS consumes. It is
pip-portable: pure `numpy` / `h5py` / `zarr` / `numcodecs` / `numba`
/ `pillow` — no MIDAS source tree, no C binaries.

This notebook is fully **self-contained**: we synthesize a small
multi-frame TIFF stack + a minimal parameter file, call
`generate_ff_zip()`, inspect the Zarr keys inside the resulting
`.MIDAS.zip`, and finish by patching a metadata key with
`midas-update-zarr`. Runs in a few seconds on CPU.
"""),
    ("py", """\
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import zarr

import midas_zipper
from midas_zipper import generate_ff_zip

print("midas_zipper version:", midas_zipper.__version__)

WORK = Path(tempfile.mkdtemp(prefix="midas_zipper_nb_"))
RAW = WORK / "raw"
OUT = WORK / "out"
RAW.mkdir(); OUT.mkdir()
print("workspace:", WORK)
"""),
    ("md", """\
## 1. Synthesize a raw TIFF frame stack

A real FF scan is one frame per ω step. We make a tiny **8-frame**
stack of 64×64 `uint16` images, each with a couple of bright Gaussian
"diffraction spots" on a low background. The filename follows the
MIDAS convention `<FileStem>_<zero-padded-number><Ext>` so the zipper
can locate the sequence.
"""),
    ("py", """\
NR_PIXELS = 64
N_FRAMES = 8
rng = np.random.default_rng(0)

def make_frame(i):
    img = rng.integers(80, 120, size=(NR_PIXELS, NR_PIXELS)).astype(np.uint16)
    yy, xx = np.mgrid[0:NR_PIXELS, 0:NR_PIXELS]
    # Two spots that drift with frame index (stand-ins for Bragg peaks).
    for (cy, cx, amp) in [(20 + i, 40, 1500), (45, 15 + i, 1200)]:
        img += (amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)).astype(np.uint16)
    return img

FILE_STEM = "synthFF"
EXT = ".tif"
START_NR = 1
for i in range(N_FRAMES):
    fn = RAW / f"{FILE_STEM}_{START_NR + i:06d}{EXT}"
    tifffile.imwrite(fn, make_frame(i))

frames = sorted(RAW.glob(f"{FILE_STEM}_*{EXT}"))
print(f"wrote {len(frames)} frames, e.g. {frames[0].name}")
print("frame shape/dtype:", tifffile.imread(frames[0]).shape, tifffile.imread(frames[0]).dtype)
"""),
    ("md", """\
## 2. Minimal parameter file

`generate_ff_zip` reads a MIDAS-style param file. For the TIFF path
it needs the file-layout keys (`RawFolder`, `FileStem`, `Ext`,
`Padding`, `StartFileNrFirstLayer`) plus the geometry / scan metadata
that gets copied into the Zarr's `analysis_parameters` group. We keep
it minimal but representative.
"""),
    ("py", """\
PARAM = f'''\\
RawFolder {RAW}
FileStem {FILE_STEM}
Ext {EXT}
Padding 6
StartFileNrFirstLayer {START_NR}
NrFilesPerSweep 1
LayerNr 1
OmegaStart 0
OmegaStep 0.25
Wavelength 0.172979
Lsd 1000000.0
px 200.0
BC 32 32
NrPixels {NR_PIXELS}
SpaceGroup 225
LatticeConstant 4.08 4.08 4.08 90 90 90
RingThresh 1 100
RingThresh 2 100
'''
param_fn = WORK / "Parameters.txt"
param_fn.write_text(PARAM)
print(param_fn.read_text())
"""),
    ("md", """\
## 3. Generate the `*.MIDAS.zip`

`generate_ff_zip` is a thin programmatic wrapper around the
`midas-ff-zip` CLI — it populates `sys.argv` and runs the same
`main()`. It reads each TIFF, stacks the frames into the Zarr
`exchange/data` array, and writes the parameters into the metadata
groups. Returns 0 on success.
"""),
    ("py", """\
rc = generate_ff_zip(
    result_folder=str(OUT),
    param_file=str(param_fn),
    layer_nr=1,
    num_files_per_scan=N_FRAMES,   # one TIFF == one ω frame; stack all 8
)
print("generate_ff_zip return code:", rc)

zips = list(OUT.glob("*.MIDAS.zip"))
assert zips, "no .MIDAS.zip produced"
zip_fn = zips[0]
print("produced:", zip_fn.name, f"({zip_fn.stat().st_size} bytes)")
"""),
    ("md", """\
## 4. Inspect the Zarr archive

The `.MIDAS.zip` is a zipped Zarr store. We open it read-only and
walk its hierarchy: the frame data lives under `exchange/data`, and
the parameters we passed live under
`analysis/process/analysis_parameters`.
"""),
    ("py", """\
zf = zarr.open(str(zip_fn), mode="r")

print("Top-level groups/arrays:")
def walk(g, prefix=""):
    for name, arr in g.arrays():
        print(f"  [array] {prefix}{name:20s} shape={arr.shape} dtype={arr.dtype}")
    for name, sub in g.groups():
        print(f"  [group] {prefix}{name}/")
        walk(sub, prefix=f"{prefix}{name}/")
walk(zf)

data = zf["exchange/data"]
print()
print("exchange/data shape:", data.shape, " dtype:", data.dtype)
print("frame 0 max / mean :", int(data[0].max()), float(data[0].mean()))
"""),
    ("py", """\
# A few of the parameters we wrote, read back out of the metadata group.
ap = "analysis/process/analysis_parameters"
for key in ("Wavelength", "Lsd", "YCen", "ZCen", "RingThresh"):
    full = f"{ap}/{key}"
    try:
        print(f"  {key:12s} = {zf[full][...]}")
    except KeyError:
        print(f"  {key:12s} = <not present>")
"""),
    ("md", """\
## 5. Patch a metadata key with `midas-update-zarr`

After a zip exists you often need to tweak one scalar (a refined
beam-center, a tilt) without regenerating the whole archive.
`midas-update-zarr` does an in-place key update. We invoke its
`main()` directly with a populated `argv` (same as the CLI). It
`chdir`s into `-folder` and rewrites the key inside the existing
`.zip` with the system `zip` tool, so we pass the archive's basename
plus its folder. Here we bump `YCen` to a refined value.
"""),
    ("py", """\
import sys
from midas_zipper import update_zarr

key = "analysis/process/analysis_parameters/YCen"
print("YCen before:", zf[key][...])

saved = sys.argv
try:
    sys.argv = [
        "midas-update-zarr",
        "-fn", zip_fn.name,        # basename; update_zarr chdir's to -folder
        "-folder", str(OUT),
        "-keyToUpdate", key,
        "-updatedValue", "33.5",
    ]
    update_zarr.main()
finally:
    sys.argv = saved

# Re-open to see the patched value.
zf2 = zarr.open(str(zip_fn), mode="r")
print("YCen after :", zf2[key][...])
"""),
    ("md", """\
## Recap

| Step | Call |
|---|---|
| raw frames → archive | `generate_ff_zip(result_folder=, param_file=, layer_nr=)` |
| inspect | `zarr.open(zip, "r")` → walk `exchange/data` + `analysis/process/analysis_parameters` |
| patch one key in place | `midas-update-zarr -fn … -keyToUpdate … -updatedValue …` |

The same generation runs from the CLI: `midas-ff-zip -resultFolder
… -paramFN … -LayerNr …`. HDF5, GE, and CBF inputs are auto-detected
from the data file extension. See the package
[README](../README.md).
"""),
]


NOTEBOOKS = {
    "01_raw_to_zarr": NB_01,
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
