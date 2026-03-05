#!/usr/bin/env python3
"""
Extract 2θ vs intensity lineouts from a series of TIFF/HDF5 images.

For each file, runs IntegratorZarrOMP in direct mode (no Zarr creation)
and converts the resulting lineout.bin into a two-column .xy text file.

Usage:
    python extract_lineouts.py -paramFN params.txt -dataFN map1_{:05d}.tif \
        -startNr 1 -endNr 100 [-nCPUs 4] [-outDir lineouts] [-darkFN dark.tif]

The -dataFN argument is a Python format string with a single integer
placeholder for the frame number, e.g.:
    map1_{:05d}.tif      →  map1_00001.tif … map1_00100.tif
    scan_{}.tif          →  scan_1.tif … scan_100.tif

Output files are named  <stem>_lineout.xy  (e.g. map1_00001_lineout.xy).
"""

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent
MIDAS_BIN = MIDAS_HOME / "FF_HEDM" / "bin"


def read_geometry(param_file: Path) -> dict:
    """Read geometry parameters needed for R → 2θ conversion."""
    geom = {'px': 172.0, 'Lsd': 0.0, 'Wavelength': 0.0,
             'RMin': 10.0, 'RMax': 1200.0, 'RBinSize': 0.25}
    with open(param_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            key = parts[0]
            if key in geom and len(parts) >= 2:
                geom[key] = float(parts[1])
    return geom


def run_integrator(param_file: Path, data_file: Path, work_dir: Path,
                   n_cpus: int = 1, dark_file: Path = None):
    """Run IntegratorZarrOMP in direct mode (-paramFN)."""
    integrator = MIDAS_BIN / "IntegratorZarrOMP"
    if not integrator.exists():
        raise FileNotFoundError(f"IntegratorZarrOMP not found at {integrator}")

    cmd = [
        str(integrator),
        "-paramFN", str(param_file.resolve()),
        "-dataFN", str(data_file.resolve()),
        "-nCPUs", str(n_cpus),
    ]
    if dark_file and dark_file.exists():
        cmd += ["-darkFN", str(dark_file.resolve())]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(work_dir), errors='replace')
    if result.returncode != 0:
        print(f"  WARNING: IntegratorZarrOMP failed (rc={result.returncode})")
        if result.stderr:
            print(f"  STDERR: {result.stderr[-500:]}")
        if result.stdout:
            # errors are often on stdout in MIDAS binaries
            for line in result.stdout.split('\n')[-10:]:
                if line.strip():
                    print(f"  STDOUT: {line.strip()}")
        return False
    return True


def lineout_to_xy(lineout_bin: Path, out_xy: Path, geom: dict):
    """Convert lineout.bin to a 2-column .xy text file (2θ, intensity)."""
    data = lineout_bin.read_bytes()
    n_rbins = len(data) // 8
    if n_rbins == 0:
        print(f"  WARNING: lineout.bin is empty")
        return False

    intensities = struct.unpack(f'{n_rbins}d', data[:n_rbins * 8])

    px = geom['px']
    Lsd = geom['Lsd']
    RMin = geom['RMin']
    RBinSize = geom['RBinSize']

    with open(out_xy, 'w') as f:
        f.write("# 2theta_deg  intensity\n")
        for i in range(n_rbins):
            R_px = RMin + (i + 0.5) * RBinSize
            R_um = R_px * px
            tth_deg = math.degrees(math.atan(R_um / Lsd)) if Lsd > 0 else 0.0
            f.write(f"{tth_deg:.6f}  {intensities[i]:.6f}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2θ vs intensity lineouts from image series.")
    parser.add_argument('-paramFN', required=True,
                        help='MIDAS parameter file')
    parser.add_argument('-dataFN', required=True,
                        help='Data filename pattern with {} placeholder '
                             '(e.g. map1_{:05d}.tif)')
    parser.add_argument('-startNr', type=int, required=True,
                        help='First frame number')
    parser.add_argument('-endNr', type=int, required=True,
                        help='Last frame number (inclusive)')
    parser.add_argument('-nCPUs', type=int, default=1,
                        help='Number of CPUs for IntegratorZarrOMP (default: 1)')
    parser.add_argument('-outDir', default='.',
                        help='Output directory for .xy files (default: cwd)')
    parser.add_argument('-darkFN', default=None,
                        help='Optional dark image file')
    args = parser.parse_args()

    param_file = Path(args.paramFN).resolve()
    if not param_file.exists():
        print(f"ERROR: Parameter file not found: {param_file}")
        sys.exit(1)

    dark_file = Path(args.darkFN).resolve() if args.darkFN else None

    out_dir = Path(args.outDir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = read_geometry(param_file)
    if geom['Lsd'] <= 0:
        print(f"ERROR: Lsd not set in {param_file}")
        sys.exit(1)

    n_total = args.endNr - args.startNr + 1
    n_done = 0
    n_fail = 0

    print(f"  Extract lineouts: {n_total} file(s)")
    print(f"  Pattern: {args.dataFN}")
    print(f"  Range: {args.startNr} → {args.endNr}")
    print(f"  Param: {param_file}")
    print(f"  Output: {out_dir.resolve()}")
    print()

    for nr in range(args.startNr, args.endNr + 1):
        # Build filename from pattern
        try:
            fn = args.dataFN.format(nr)
        except (IndexError, KeyError):
            fn = args.dataFN.replace('{}', str(nr))

        data_file = Path(fn)
        if not data_file.exists():
            # Try relative to param file directory
            data_file = param_file.parent / fn
        if not data_file.exists():
            print(f"  [{nr}] SKIP: {fn} not found")
            n_fail += 1
            continue

        stem = data_file.stem
        out_xy = out_dir / f"{stem}_lineout.xy"

        # Use a temp work directory for Map.bin etc.
        work_dir = Path(tempfile.mkdtemp(prefix=f"lineout_{stem}_"))
        try:
            # Copy Map.bin and nMap.bin if they exist in the param file's dir
            # (avoids re-running DetectorMapper each time)
            for mapf in ("Map.bin", "nMap.bin"):
                src = param_file.parent / mapf
                if src.exists():
                    os.symlink(src, work_dir / mapf)

            print(f"  [{nr}/{args.endNr}] {data_file.name} → {out_xy.name} ...",
                  end="", flush=True)

            ok = run_integrator(param_file, data_file, work_dir,
                                args.nCPUs, dark_file)
            if not ok:
                print(" FAILED")
                n_fail += 1
                continue

            lineout_bin = work_dir / "lineout.bin"
            if not lineout_bin.exists():
                print(" no lineout.bin")
                n_fail += 1
                continue

            if lineout_to_xy(lineout_bin, out_xy, geom):
                n_done += 1
                print(" OK")
            else:
                n_fail += 1
                print(" FAILED (empty)")

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    print()
    print(f"  Done: {n_done}/{n_total} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()
