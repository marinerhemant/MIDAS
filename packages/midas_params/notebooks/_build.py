"""Build .ipynb files from a maintainable cell-list source.

Each notebook is defined as a list of (kind, source) tuples where
kind is "md" or "py" and source is the markdown / Python source.
Run this script once to (re)generate every .ipynb in this directory.

The .ipynb files are derived artefacts; this file is the source of
truth (raw .ipynb JSON is unreviewable in diffs).

Usage:
    cd packages/midas_params/notebooks
    python _build.py             # rebuild all notebooks
    python _build.py 01_quickstart   # rebuild one
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
# NOTEBOOK SOURCES
# =====================================================================

NB_01: List[Cell] = [
    ("md", """\
# 01 — `midas_params`: validate, inspect, rings, discover, wizard

`midas_params` is the parameter-file registry, validator, and wizard
for the MIDAS FF/NF/PF/RI pipelines. This notebook is a fast
end-to-end tour of every public capability on a **self-generated**
sample param file — no network, no external data, runs in a couple
of seconds on CPU.

By the end you will have:

1. Written a small FF `paramstest.txt`.
2. **Validated** it and rendered a human-readable report.
3. Caught a deliberately-broken file with the cross-field rules.
4. **Inspected** the registry (how many keys per pipeline path).
5. Enumerated the visible **Bragg rings** for the crystal + detector.
6. **Discovered** parameters from a raw-frame filename and from a
   calibration file.
7. Run the non-interactive **wizard** to assemble + write a param file.
"""),
    ("py", """\
import tempfile, os
from pathlib import Path

import midas_params
from midas_params import Path as MPath   # the pipeline-path enum (FF/NF/PF/RI)

print("midas_params version:", midas_params.__version__)
# Scratch workspace (self-contained, cleaned up by the OS).
WORK = Path(tempfile.mkdtemp(prefix="midas_params_nb_"))
print("workspace:", WORK)
"""),
    ("md", """\
## 1. A small, valid FF parameter file

We write a minimal-but-coherent FF `paramstest.txt`. Geometry is a
1 m sample-detector distance, 2048², 200 µm pixels, Au (a = 4.08 Å,
FCC, space group 225), four rings, a 360° ω scan in 0.25° steps,
plus the indexing tolerances FF requires.

We also drop a few empty placeholder frames in the workspace so the
later discovery step has a real `RawFolder` to scan.
"""),
    ("py", """\
# Empty placeholder frames (discover_from_file scans the directory).
for n in (1, 2, 3):
    (WORK / f"Au_sample_{n:06d}.ge3").write_bytes(b"")

# RawFolder points at the workspace so the validator's directory check
# passes; OmegaRange matches the scanned [0, 360] range.
SAMPLE = f'''\\
Wavelength 0.172979
Lsd 1000000.0
px 200.0
BC 1024 1024
NrPixels 2048
SpaceGroup 225
LatticeConstant 4.08 4.08 4.08 90 90 90
RhoD 200000
RingThresh 1 100
RingThresh 2 100
RingThresh 3 100
RingThresh 4 100
OmegaStart 0
OmegaStep 0.25
OmegaRange 0 360
StartNr 1
EndNr 1440
RawFolder {WORK}
FileStem Au_sample
Ext .ge3
OverAllRingToIndex 1
Completeness 0.6
StepSizeOrient 0.2
StepSizePos 100
tx 0
ty 0
tz 0
'''
param_fn = WORK / "paramstest.txt"
param_fn.write_text(SAMPLE)
print(param_fn.read_text())
"""),
    ("md", """\
## 2. Validate it

`validate(path_to_file, pipeline_path)` returns a `ValidationReport`.
`format_report` renders it for humans.
"""),
    ("py", """\
from midas_params.validator import validate, format_report

report = validate(str(param_fn), MPath.FF)
print("ok:", report.ok)
print("n errors:", len(report.errors), " n warnings:", len(report.warnings))
print()
print(format_report(report, use_color=False))
"""),
    ("md", """\
## 3. Catch a broken file

The validator runs per-key validators and cross-field consistency
rules. Here we feed it an inconsistent ω-direction (`OmegaStep` is
positive while the scan goes from 180 → -180) and a `StartNr > EndNr`.
"""),
    ("py", """\
BROKEN = '''\\
Wavelength 0.172979
Lsd 1000000.0
px 200.0
SpaceGroup 225
LatticeConstant 4.08 4.08 4.08 90 90 90
RingThresh 1 100
OmegaStart 180
OmegaEnd -180
OmegaStep 0.25
StartNr 100
EndNr 50
'''
broken_fn = WORK / "broken.txt"
broken_fn.write_text(BROKEN)

rep = validate(str(broken_fn), MPath.FF)
print("ok:", rep.ok)
for issue in rep.errors:
    print(f"  [error] rule={issue.rule!r}: {issue.message}")
"""),
    ("md", """\
## 4. Inspect the registry

The registry (`PARAMS`) is the single source of truth — every key is
a JSON-serializable `ParamSpec`. `for_path` / `required_for` scope it
to a pipeline.
"""),
    ("py", """\
from midas_params import PARAMS, for_path, required_for

print("total registered keys:", len(PARAMS))
for p in (MPath.FF, MPath.NF, MPath.PF, MPath.RI):
    n_all = len(for_path(p))
    n_req = len(required_for(p))
    print(f"  {p.name:3s}:  {n_all:3d} keys applicable, {n_req:2d} required")

# Peek at one spec.
spec = {s.name: s for s in PARAMS}["Wavelength"]
print()
print("Wavelength spec:")
print("  type        :", spec.type.name)
print("  units       :", spec.units)
print("  typical     :", spec.typical)
print("  description :", spec.description[:80] if spec.description else None)
"""),
    ("md", """\
## 5. Enumerate visible Bragg rings

`enumerate_rings` projects the crystal's allowed reflections onto the
detector for the given geometry, sorted by 2θ. `recommend_rings`
picks a sensible subset; `format_ring_table` renders it.
"""),
    ("py", """\
from midas_params.rings import enumerate_rings, recommend_rings, format_ring_table

rings = enumerate_rings(
    wavelength=0.172979,
    lsd_um=1_000_000.0,
    lattice=[4.08, 4.08, 4.08, 90, 90, 90],
    space_group=225,
    rho_d_um=200_000.0,
    nr_pixels_y=2048,
    px_um=200.0,
    max_rings=8,
)
print(format_ring_table(rings, use_color=False))
print()
print("recommended ring numbers:", recommend_rings(rings))
"""),
    ("md", """\
## 6. Discover parameters automatically

`discover_from_file` parses a raw-frame file's **name** into
`RawFolder` / `FileStem` / `Padding` / `Ext` / `StartNr` (and scans
the directory for the frame-number range). `discover_from_calibration_file`
reads an existing MIDAS text param file. `merge` combines sources
with priority (earlier wins).
"""),
    ("py", """\
from midas_params import (
    discover_from_file, discover_from_calibration_file, merge,
)

# discover_from_file inspects an on-disk frame (the placeholders we
# created in step 1) and range-scans the directory.
d_name = discover_from_file(str(WORK / "Au_sample_000001.ge3"))
print("from filename:", d_name.extracted)
print()

# From the calibration param file we wrote in step 1.
d_cal = discover_from_calibration_file(str(param_fn))
print("keys discovered from calibration file:", sorted(d_cal.extracted))
print()

# Merge: filename info wins for file-layout keys, calibration for geometry.
seeded = merge(d_name, d_cal)
print("merged FileStem :", seeded.extracted.get("FileStem"))
print("merged Lsd      :", seeded.extracted.get("Lsd"))
print("merged Wavelength:", seeded.extracted.get("Wavelength"))
"""),
    ("md", """\
## 7. Build a param file with the (non-interactive) wizard

`run_wizard(..., non_interactive=True)` seeds from an existing param
file + the dataset frame we just created, fills any remaining
required keys from registry defaults, writes the file, and validates
it. Returns an exit code (0 = the written file validates clean).
"""),
    ("py", """\
from midas_params.wizard import run_wizard

out_fn = WORK / "wizard_out.txt"
code = run_wizard(
    path=MPath.FF,
    output=str(out_fn),
    from_existing=str(param_fn),
    dataset_file=str(WORK / "Au_sample_000001.ge3"),
    non_interactive=True,
)
print("wizard exit code:", code, "(0 = wrote a file that validates clean)")
print()
print(out_fn.read_text())
"""),
    ("md", """\
## Recap

| Capability | Entry point |
|---|---|
| Validate a param file | `midas_params.validator.validate` + `format_report` |
| Inspect the registry | `midas_params.PARAMS`, `for_path`, `required_for` |
| Enumerate Bragg rings | `midas_params.rings.enumerate_rings` / `recommend_rings` |
| Discover from filename / calibration | `discover_from_file`, `discover_from_calibration_file`, `merge` |
| Build a file end-to-end | `midas_params.wizard.run_wizard(..., non_interactive=True)` |

The same operations are available from the CLI: `midas-params
validate|inspect|rings|wizard|diagnose`. See the package
[README](../README.md).
"""),
]


NOTEBOOKS = {
    "01_quickstart": NB_01,
}


def main(argv):
    if len(argv) > 1:
        for t in argv[1:]:
            if t not in NOTEBOOKS:
                print(f"unknown notebook: {t}")
                print(f"available: {list(NOTEBOOKS)}")
                return 1
        for t in argv[1:]:
            print("wrote", write_notebook(t, NOTEBOOKS[t]))
    else:
        for name, cells in NOTEBOOKS.items():
            print("wrote", write_notebook(name, cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
