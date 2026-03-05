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
import concurrent.futures
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


def run_detector_mapper(param_file: Path, work_dir: Path, n_cpus: int = 0):
    """Run DetectorMapper to produce Map.bin + nMap.bin.

    Skips if both files already exist in work_dir.
    """
    map_bin = work_dir / "Map.bin"
    nmap_bin = work_dir / "nMap.bin"
    if map_bin.exists() and nmap_bin.exists():
        print(f"  DetectorMapper: skipped (Map.bin + nMap.bin exist)")
        return True

    mapper = MIDAS_BIN / "DetectorMapper"
    if not mapper.exists():
        raise FileNotFoundError(f"DetectorMapper not found at {mapper}")

    print("  Running DetectorMapper...", end="", flush=True)
    cmd = [str(mapper), str(param_file.resolve())]
    if n_cpus > 0:
        cmd += ["-nCPUs", str(n_cpus)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(work_dir), errors='replace')
    if result.returncode != 0:
        print(f" FAILED (rc={result.returncode})")
        if result.stderr:
            print(f"  STDERR: {result.stderr[-500:]}")
        return False
    print(f" OK ({map_bin.stat().st_size:,} bytes)")
    return True


def _process_one_frame(nr, data_file, param_file, out_dir, out_xy, geom,
                       dark_file, end_nr):
    """Worker function: integrate one frame and write its .xy file.

    Returns (nr, True/False, message).
    """
    stem = data_file.stem
    work_dir = Path(tempfile.mkdtemp(prefix=f"lineout_{stem}_"))
    try:
        # Symlink Map.bin and nMap.bin from the output directory
        for mapf in ("Map.bin", "nMap.bin"):
            src = out_dir / mapf
            if src.exists():
                os.symlink(src, work_dir / mapf)

        # Run IntegratorZarrOMP (single-threaded)
        integrator = MIDAS_BIN / "IntegratorZarrOMP"
        cmd = [
            str(integrator),
            "-paramFN", str(param_file),
            "-dataFN", str(data_file),
            "-nCPUs", "1",
        ]
        if dark_file and dark_file.exists():
            cmd += ["-darkFN", str(dark_file)]

        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(work_dir), errors='replace')
        if result.returncode != 0:
            return (nr, False, "integrator failed")

        lineout_bin = work_dir / "lineout.bin"
        if not lineout_bin.exists():
            return (nr, False, "no lineout.bin")

        # Convert to .xy
        data = lineout_bin.read_bytes()
        n_rbins = len(data) // 8
        if n_rbins == 0:
            return (nr, False, "empty lineout.bin")

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

        return (nr, True, "OK")

    except Exception as e:
        return (nr, False, str(e))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


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
                        help='Number of concurrent integrator instances '
                             '(default: 1)')
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

    out_dir = Path(args.outDir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = read_geometry(param_file)
    if geom['Lsd'] <= 0:
        print(f"ERROR: Lsd not set in {param_file}")
        sys.exit(1)

    n_total = args.endNr - args.startNr + 1
    n_workers = max(1, args.nCPUs)

    print(f"  Extract lineouts: {n_total} file(s), {n_workers} workers")
    print(f"  Pattern: {args.dataFN}")
    print(f"  Range: {args.startNr} → {args.endNr}")
    print(f"  Param: {param_file}")
    print(f"  Output: {out_dir}")
    print()

    # --- Run DetectorMapper once to produce Map.bin + nMap.bin ---
    if not run_detector_mapper(param_file, out_dir, n_workers):
        print("ERROR: DetectorMapper failed, cannot proceed.")
        sys.exit(1)
    print()

    # --- Build list of (nr, data_file, out_xy) jobs ---
    jobs = []
    for nr in range(args.startNr, args.endNr + 1):
        try:
            fn = args.dataFN.format(nr)
        except (IndexError, KeyError):
            fn = args.dataFN.replace('{}', str(nr))

        data_file = Path(fn)
        if not data_file.exists():
            data_file = param_file.parent / fn
        if not data_file.exists():
            print(f"  [{nr}] SKIP: {fn} not found")
            continue

        stem = data_file.stem
        out_xy = out_dir / f"{stem}_lineout.xy"
        jobs.append((nr, data_file.resolve(), out_xy))

    if not jobs:
        print("ERROR: No data files found.")
        sys.exit(1)

    # --- Process frames in parallel ---
    n_done = 0
    n_fail = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_one_frame, nr, df, param_file, out_dir,
                        oxy, geom, dark_file, args.endNr): (nr, df, oxy)
            for nr, df, oxy in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            nr, df, oxy = futures[future]
            frame_nr, ok, msg = future.result()
            if ok:
                n_done += 1
                print(f"  [{frame_nr}/{args.endNr}] {df.name} → {oxy.name} OK")
            else:
                n_fail += 1
                print(f"  [{frame_nr}/{args.endNr}] {df.name} FAILED: {msg}")

    print()
    print(f"  Done: {n_done}/{len(jobs)} succeeded, {n_fail} failed"
          f" ({n_workers} workers)")


if __name__ == "__main__":
    main()
