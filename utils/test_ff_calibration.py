#!/usr/bin/env python
"""
Automated Benchmark Test for FF-HEDM Calibrant Fitting.

Runs CalibrantPanelShiftsOMP on the example CeO2 calibrant data and validates
that the resulting mean strain is below a threshold. This ensures the detector
geometry calibration pipeline produces accurate results.

Usage:
    python test_ff_calibration.py
    python test_ff_calibration.py -nCPUs 4
    python test_ff_calibration.py -paramFN /path/to/custom/parameters.txt
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Benchmark for FF-HEDM Calibrant Fitting")
    parser.add_argument("-nCPUs", type=int, default=1,
                        help="Number of CPUs (passed to CalibrantPanelShiftsOMP)")

    default_param = (Path(__file__).resolve().parent.parent
                     / "FF_HEDM" / "Example" / "Calibration" / "parameters.txt")
    parser.add_argument("-paramFN", type=str, default=str(default_param),
                        help="Path to the calibrant parameter file")

    parser.add_argument("-strainThreshold", type=float, default=50.0,
                        help="Maximum acceptable mean strain in microstrain (default: 50)")

    return parser.parse_args()


def find_binary():
    """Locate CalibrantPanelShiftsOMP binary."""
    midas_home = Path(os.environ.get(
        'MIDAS_HOME', str(Path(__file__).resolve().parent.parent)))
    bin_path = midas_home / "FF_HEDM" / "bin" / "CalibrantPanelShiftsOMP"
    if not bin_path.exists():
        print(f"Error: {bin_path} not found. Please compile first.")
        sys.exit(1)
    return bin_path


def find_hkl_generator():
    """Locate GetHKLList binary."""
    midas_home = Path(os.environ.get(
        'MIDAS_HOME', str(Path(__file__).resolve().parent.parent)))
    bin_path = midas_home / "FF_HEDM" / "bin" / "GetHKLList"
    if not bin_path.exists():
        print(f"Error: {bin_path} not found. Please compile first.")
        sys.exit(1)
    return bin_path


def prepare_working_dir(param_path):
    """Create a temporary working directory with all necessary files."""
    work_dir = Path(tempfile.mkdtemp(prefix="midas_calib_test_"))
    example_dir = param_path.parent

    # Copy data files
    for f in example_dir.iterdir():
        if f.is_file() and f.name != '.DS_Store':
            shutil.copy2(str(f), str(work_dir / f.name))

    # Rewrite parameter file with absolute paths
    new_param = work_dir / "parameters.txt"
    with open(param_path, 'r') as fin, open(new_param, 'w') as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                fout.write(line)
                continue
            parts = stripped.split()
            key = parts[0]
            if key == 'Folder':
                fout.write(f"Folder {work_dir}\n")
            elif key == 'Dark':
                fout.write(f"Dark {work_dir / parts[1]}\n")
            elif key == 'MaskFile':
                fout.write(f"MaskFile {work_dir / parts[1]}\n")
            elif key == 'PanelShiftsFile':
                fout.write(f"PanelShiftsFile {work_dir / parts[1]}\n")
            else:
                fout.write(line)

    return work_dir, new_param


def generate_hkls(param_file, work_dir):
    """Run GetHKLList to generate hkls.csv."""
    hkl_bin = find_hkl_generator()
    cmd = [str(hkl_bin), str(param_file)]
    print(f"  Generating HKL list: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(work_dir),
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: GetHKLList failed.\nstdout: {result.stdout}\nstderr: {result.stderr}")
        sys.exit(1)
    hkls = work_dir / "hkls.csv"
    if not hkls.exists():
        print("Error: hkls.csv was not generated.")
        sys.exit(1)
    n_lines = sum(1 for _ in open(hkls)) - 1
    print(f"  Generated {n_lines} HKL entries.")


def run_calibrant(param_file, nCPUs, work_dir):
    """Run CalibrantPanelShiftsOMP and capture output."""
    calib_bin = find_binary()
    cmd = [str(calib_bin), str(param_file), str(nCPUs)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(work_dir),
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: CalibrantPanelShiftsOMP failed (rc={result.returncode}).")
        print(f"stdout:\n{result.stdout[-2000:]}")
        print(f"stderr:\n{result.stderr[-1000:]}")
        sys.exit(1)
    return result.stdout


def parse_results(output):
    """Parse the calibrant output for key metrics."""
    results = {}

    # Look for the Mean Values section
    in_mean = False
    for line in output.split('\n'):
        if 'Mean Values' in line:
            in_mean = True
            continue
        if 'Copy to par' in line:
            in_mean = False
            continue
        if not in_mean:
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                val = float(parts[1])
                results[key] = val
            except ValueError:
                pass

    # Also look for best-iteration messages
    for line in output.split('\n'):
        m = re.search(r'Best result.*iteration (\d+)/(\d+)', line)
        if m:
            results['bestIter'] = int(m.group(1))
            results['nIters'] = int(m.group(2))

    return results


def main():
    args = parse_args()
    param_path = Path(args.paramFN).resolve()
    if not param_path.exists():
        print(f"Error: Parameter file not found at {param_path}")
        sys.exit(1)

    print("=" * 70)
    print("  FF-HEDM Calibration Benchmark")
    print("=" * 70)
    print(f"  Parameter file: {param_path}")
    print(f"  CPUs: {args.nCPUs}")
    print(f"  Strain threshold: {args.strainThreshold} µε")
    print()

    # 1. Prepare workspace
    print("[1/3] Preparing workspace...")
    work_dir, test_param = prepare_working_dir(param_path)
    print(f"  Working directory: {work_dir}")

    # 2. Generate HKLs
    print("\n[2/3] Generating HKL list...")
    generate_hkls(test_param, work_dir)

    # 3. Run calibration
    print("\n[3/3] Running calibrant fitting...")
    output = run_calibrant(test_param, args.nCPUs, work_dir)

    # Parse and validate results
    results = parse_results(output)

    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)

    if 'MeanStrain' in results:
        mean_us = results['MeanStrain']
        print(f"  MeanStrain:   {mean_us:.2f} µε")
    else:
        print("  Error: MeanStrain not found in output.")
        print("  Tail of output:")
        print(output[-2000:])
        sys.exit(1)

    if 'StdStrain' in results:
        print(f"  StdStrain:    {results['StdStrain']:.2f} µε")
    if 'MedianStr' in results:
        print(f"  MedianStrain: {results['MedianStr']:.2f} µε")
    if 'Q25' in results:
        print(f"  Q25:          {results['Q25']:.2f} µε")
    if 'Q75' in results:
        print(f"  Q75:          {results['Q75']:.2f} µε")
    if 'MinStrain' in results:
        print(f"  MinStrain:    {results['MinStrain']:.2f} µε")
    if 'MaxStrain' in results:
        print(f"  MaxStrain:    {results['MaxStrain']:.2f} µε")
    if 'NPoints' in results:
        print(f"  NPoints:      {int(results['NPoints'])}")
    if 'bestIter' in results:
        print(f"  Best Iter:    {results['bestIter']}/{results['nIters']}")

    # Key parameters
    param_keys = ['Lsd', 'BC', 'ty', 'tz', 'p0', 'p1', 'p2', 'p3', 'p4']
    printed_params = False
    for k in param_keys:
        if k in results:
            if not printed_params:
                print("\n  Refined Parameters:")
                printed_params = True
            print(f"    {k:12s} {results[k]}")

    # Validate
    print("\n" + "=" * 70)
    if mean_us <= args.strainThreshold:
        print(f"  ✅ PASS: MeanStrain {mean_us:.2f} µε ≤ {args.strainThreshold} µε")
    else:
        print(f"  ❌ FAIL: MeanStrain {mean_us:.2f} µε > {args.strainThreshold} µε")
        sys.exit(1)
    print("=" * 70)

    # Cleanup
    try:
        shutil.rmtree(work_dir)
        print(f"\n  Cleaned up: {work_dir}")
    except Exception:
        print(f"\n  Note: Could not clean up {work_dir}")


if __name__ == "__main__":
    main()
