#!/usr/bin/env python3
"""Regenerate Paper A's Table 4 — MIDAS vs pyFAI on four detectors.

Writes a CSV with the paper's headline metrics for each detector:
    label, midas_ustrain, pyfai_ustrain, accuracy_ratio,
    midas_seconds, pyfai_seconds, speedup

Usage:
    python scripts/regen_table4.py --out table4.csv

The bundled Pilatus CeO₂ frame ships inside the wheel; the other three
detector datasets (Pilatus 6M, Varex 4343CT, Ceria) are ~60 MB combined
and live on Zenodo. Pass ``--data-dir /path/to/zenodo/extract`` to point
at an unpacked archive, or omit to run only the bundled Pilatus row.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import midas_auto_calibrate as mac


@dataclass
class DetectorCase:
    """One row in Table 4: dataset + calibrant config."""

    label: str
    image_name: str          # path relative to data_dir
    dark_name: str
    mask_name: Optional[str]
    pixel_size: float        # µm
    wavelength: float        # Å
    lsd: float               # µm starting guess
    ybc: float
    zbc: float
    nr_pixels_y: int
    nr_pixels_z: int
    im_trans_opt: list[int]
    n_iterations: int = 5
    extra_params: dict = None  # raw MIDAS Parameters.txt knobs; see
                               # FF_HEDM/Example/Calibration/parameters.txt


# Pilatus-specific knobs copied from the bundled paper parameters.txt. The
# 6×8 panel mosaic + ring exclusions + objective weights are essential for
# the paper's ~4 µε pseudo-strain — a monolithic-detector model blows up
# to kilo-µε range.
_PILATUS_KNOBS: dict = {
    # RhoD = distance from off-centre BC to farthest detector corner.
    # Bundled parameters.txt uses 219964.4 µm; my auto-default
    # (half the shorter side * px) underestimates for off-centre BCs.
    "RhoD": 219964.42411013643,
    "Width": 800,
    "OmegaStart": -180, "OmegaStep": 0.25,
    "tolTilts": 3, "tolBC": 20, "tolLsd": 15000,
    "tolP": 2e-3, "tolP4": 1e-4,
    "OutlierIterations": 3, "MultFactor": 2,
    "NormalizeRingWeights": 1, "WeightByRadius": 1,
    "WeightByFitSNR": 1, "L2Objective": 1,
    # Panel geometry
    "NPanelsY": 6, "NPanelsZ": 8,
    "PanelSizeY": 243, "PanelSizeZ": 195,
    "PanelGapsY": [1, 7, 1, 7, 1],
    "PanelGapsZ": [17, 17, 17, 17, 17, 17, 17],
    "FixPanelID": 12, "tolShifts": 1, "tolRotation": 3,
    "PerPanelLsd": 1, "PerPanelDistortion": 1,
    "PanelShiftsFile": "panelshiftsCalibrant.txt",
    "DoubletSeparation": 25,
    # Initial distortion guess (from paper)
    "p0": 0.000230535992, "p1": 0.000172564332,
    "p2": -0.000542224078, "p3": -13.773706892191,
    "p4": 0.001909017437,
}
# Drop high-index noisy rings for the Pilatus detector. MIDAS accepts one
# `RingsToExclude <n>` line per ring — emit as a list-of-lists so
# write_params_file produces the repeated-key form.
_PILATUS_KNOBS["RingsToExclude"] = [[n] for n in range(19, 34)]


# Paper A's four detector configurations. Values are pulled from the paper's
# experimental section — adjust if the camera-ready revises them.
CASES: list[DetectorCase] = [
    DetectorCase(
        # For the bundled Pilatus row we stage files under these names in
        # _stage_bundled_pilatus() below; external-dataset rows use the
        # Zenodo-archive names in their `*_name` fields directly.
        #
        # Starting-point tilts are seeded from the paper's converged values
        # — starting at (0, 0, 0) leaves the optimizer too far from the
        # basin and pseudo-strain bottoms out near 80 µε instead of ~4.
        label="Pilatus (172µm, 71.7 keV)",
        image_name="CeO2_00001.tif",
        dark_name="dark.tif",
        mask_name="mask_upd.tif",
        pixel_size=172.0,
        wavelength=0.172973,
        lsd=657_436.895687981, ybc=685.485459654, zbc=921.034377044,
        nr_pixels_y=1475, nr_pixels_z=1679,
        im_trans_opt=[2],
        extra_params={
            **_PILATUS_KNOBS,
            "ty": 0.200888234849, "tz": 0.446902376310, "tx": 0.0,
        },
    ),
    # Additional cases landed with the Zenodo dataset — see data.zenodo_url().
    # Included here so --data-dir can resolve them when populated.
    DetectorCase(
        label="Pilatus 6M (100x100, 71.7 keV)",
        image_name="CeO2_Pil_6M.tif",
        dark_name="dark_Pil_6M.tif",
        mask_name=None,
        pixel_size=172.0, wavelength=0.172973,
        lsd=650_000, ybc=1237, zbc=1259,
        nr_pixels_y=2463, nr_pixels_z=2527,
        im_trans_opt=[],
    ),
    DetectorCase(
        label="Varex 4343CT (150µm, 63 keV)",
        image_name="CeO2_Varex.tif",
        dark_name="dark_Varex.tif",
        mask_name=None,
        pixel_size=150.0, wavelength=0.19684,
        lsd=900_000, ybc=1438, zbc=1438,
        nr_pixels_y=2880, nr_pixels_z=2880,
        im_trans_opt=[],
    ),
    DetectorCase(
        label="Ceria aero (63 keV, 100x100)",
        image_name="Ceria.tif",
        dark_name="dark_Ceria.tif",
        mask_name=None,
        pixel_size=150.0, wavelength=0.19684,
        lsd=900_000, ybc=1438, zbc=1438,
        nr_pixels_y=2880, nr_pixels_z=2880,
        im_trans_opt=[],
    ),
]


def _stage(case: DetectorCase, src_dir: Path, workdir: Path) -> Path:
    """Copy one case's files into the workdir, return the image path."""
    img = workdir / case.image_name
    shutil.copy(src_dir / case.image_name, img)
    shutil.copy(src_dir / case.dark_name, workdir / "dark.tif")
    if case.mask_name:
        shutil.copy(src_dir / case.mask_name, workdir / "mask_upd.tif")
    return img


def _stage_bundled_pilatus(workdir: Path) -> Path:
    """Stage the wheel-bundled Pilatus data (first row of the table)."""
    # MIDASCalibrant expects numbered filenames (<stem>_<NN...>.<ext>); rename
    # from the bundled name to match.
    img = workdir / "CeO2_00001.tif"
    shutil.copy(mac.data.CEO2_PILATUS, img)
    shutil.copy(mac.data.CEO2_PILATUS_DARK, workdir / "dark.tif")
    shutil.copy(mac.data.CEO2_PILATUS_MASK, workdir / "mask_upd.tif")
    return img


def run_case(case: DetectorCase, image: Path, workdir: Path,
             *, n_cpus: int, include_pyfai: bool) -> dict:
    """Run one detector case, return a row dict for the Table 4 CSV."""
    result = mac.benchmark(
        image,
        material="CeO2",
        wavelength=case.wavelength,
        pixel_size=case.pixel_size,
        nr_pixels_y=case.nr_pixels_y, nr_pixels_z=case.nr_pixels_z,
        lsd=case.lsd, ybc=case.ybc, zbc=case.zbc,
        n_iterations=case.n_iterations,
        n_cpus=n_cpus,
        include_pyfai=include_pyfai,
        work_dir=workdir,
        im_trans_opt=tuple(case.im_trans_opt),
        mask_file=(case.mask_name or "mask_upd.tif"),
        extra_params=case.extra_params or {},
    )
    row = {"label": case.label, **result.as_dict()}
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("table4.csv"),
                        help="Output CSV path.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory holding the Zenodo extract with the "
                             "non-bundled detector images. Omit to run only "
                             "the wheel-bundled Pilatus row.")
    parser.add_argument("--n-cpus", type=int, default=4)
    parser.add_argument("--no-pyfai", action="store_true",
                        help="Skip the pyFAI half of each row (useful when "
                             "pyFAI isn't installed).")
    parser.add_argument("--only-bundled", action="store_true",
                        help="Run only the bundled Pilatus row, ignoring "
                             "--data-dir.")
    args = parser.parse_args(argv)

    rows: list[dict] = []
    include_pyfai = not args.no_pyfai

    for i, case in enumerate(CASES):
        with tempfile.TemporaryDirectory(prefix=f"mac_table4_{i}_") as tmp:
            workdir = Path(tmp)
            try:
                if i == 0 and case.image_name == "CeO2_Pilatus.tif":
                    # First row uses the wheel-bundled data unconditionally.
                    img = _stage_bundled_pilatus(workdir)
                elif args.only_bundled or args.data_dir is None:
                    print(f"  skipping {case.label!r} "
                          f"(no --data-dir or --only-bundled).",
                          file=sys.stderr)
                    continue
                else:
                    src = args.data_dir / case.image_name
                    if not src.exists():
                        print(f"  skipping {case.label!r}: "
                              f"{src} not found.", file=sys.stderr)
                        continue
                    img = _stage(case, args.data_dir, workdir)
            except FileNotFoundError as e:
                print(f"  skipping {case.label!r}: {e}", file=sys.stderr)
                continue

            print(f"  running {case.label!r}…", file=sys.stderr)
            try:
                rows.append(run_case(case, img, workdir,
                                     n_cpus=args.n_cpus,
                                     include_pyfai=include_pyfai))
            except Exception as exc:    # noqa: BLE001 — report and continue
                print(f"  FAILED {case.label!r}: {exc}", file=sys.stderr)

    if not rows:
        print("No rows produced — table not written.", file=sys.stderr)
        return 1

    # Union of all keys across rows (some rows may lack pyFAI columns).
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
