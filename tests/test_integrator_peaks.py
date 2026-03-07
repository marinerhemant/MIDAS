#!/usr/bin/env python3
"""
MIDAS Integrator Peak Fitting Benchmark Test

Uses the CeO2 calibration benchmark data to test IntegratorZarrOMP with
peak fitting. First runs CalibrantPanelShiftsOMP to get optimized geometry,
then integrates and fits peaks, comparing fitted R-positions against
theoretical ring radii to validate integration + fitting accuracy.

Usage:
    python test_integrator_peaks.py
    python test_integrator_peaks.py -nCPUs 8
    python test_integrator_peaks.py -nCPUs 8 --keep-work-dir
"""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from test_common import add_common_args, run_preflight, print_environment

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent
MIDAS_BIN = MIDAS_HOME / "FF_HEDM" / "bin"
CALIB_DIR = MIDAS_HOME / "FF_HEDM" / "Example" / "Calibration"

# Peak fit binary format: 7 doubles per peak
PF_PARAMS_PER_PEAK = 7
# [0]=Imax, [1]=BG, [2]=Mix, [3]=Center, [4]=Sigma, [5]=SNR, [6]=Area

def run_cmd(cmd, cwd=None, check=True, stream=False):
    """Run a command and return stdout."""
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    if stream:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1, cwd=cwd)
        output_lines = []
        progress_keys = ['Iteration', 'MeanStrain', 'StdStrain', 'microstrain',
                         'Number of eta bins', 'Out of', 'Number of function calls',
                         'Lsd ', 'BC ', 'Restoring best', 'Best result']
        for line in proc.stdout:
            output_lines.append(line)
            stripped = line.strip()
            if any(k in stripped for k in progress_keys):
                print(f"    ▸ {stripped}")
        proc.wait()
        output = ''.join(output_lines)
        if check and proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            print(f"  STDERR: {stderr[-1000:]}")
            raise RuntimeError(f"Command failed (rc={proc.returncode})")
        return output
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                                errors='replace')
        if check and result.returncode != 0:
            print(f"  STDERR: {result.stderr}")
            raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(str(c) for c in cmd)}")
        return result.stdout


# =========================================================================
# Step 0: Run calibration to get optimized geometry
# =========================================================================

def prepare_calibration_dir(param_path: Path) -> Path:
    """Create a working directory with calibration files."""
    work_dir = Path(tempfile.mkdtemp(prefix="midas_calib_"))
    example_dir = param_path.parent

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


def run_calibration(param_file: Path, nCPUs: int, work_dir: Path) -> dict:
    """Run CalibrantPanelShiftsOMP and parse optimized parameters."""
    # Generate HKLs
    hkl_bin = MIDAS_BIN / "GetHKLList"
    run_cmd([str(hkl_bin), str(param_file)], cwd=str(work_dir))

    # Run calibrant fitting
    calib_bin = MIDAS_BIN / "CalibrantPanelShiftsOMP"
    output = run_cmd([str(calib_bin), str(param_file), str(nCPUs)],
                     cwd=str(work_dir), stream=True)

    # Parse optimized parameters from "Mean Values" section
    params = {}
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
            if key == 'BC' and len(parts) >= 3:
                params['BCy'] = float(parts[1])
                params['BCz'] = float(parts[2])
            else:
                try:
                    params[key] = float(parts[1])
                except ValueError:
                    pass

    required = ['Lsd', 'ty', 'tz', 'p0', 'p1', 'p2', 'p3', 'p4', 'BCy', 'BCz']
    missing = [k for k in required if k not in params]
    if missing:
        print(f"  WARNING: Could not parse parameters: {missing}")
        print(f"  Output tail:\n{output[-2000:]}")

    return params


def build_integrator_param_file(calib_param_file: Path, optimized: dict,
                                out_path: Path) -> Path:
    """Build a parameter file for IntegratorZarrOMP with optimized geometry.

    Starts from the calibration parameters.txt, updates geometry from the
    optimized results, and ensures RBinSize=0.25 and all integrator params
    are present.
    """
    # Read original params, overriding geometry with optimized values
    overrides = {}
    if 'Lsd' in optimized:
        overrides['Lsd'] = f"{optimized['Lsd']:.12f}"
    if 'BCy' in optimized and 'BCz' in optimized:
        overrides['BC'] = f"{optimized['BCy']:.12f} {optimized['BCz']:.12f}"
    for key in ['ty', 'tz', 'p0', 'p1', 'p2', 'p3', 'p4']:
        if key in optimized:
            overrides[key] = f"{optimized[key]:.12f}"

    # Force RBinSize to 0.25
    overrides['RBinSize'] = '0.25'

    # Keys we've already written
    written_keys = set()

    with open(calib_param_file, 'r') as fin, open(out_path, 'w') as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                fout.write(line)
                continue
            parts = stripped.split()
            key = parts[0]

            # Skip calibrant-only parameters not needed for integration
            skip_keys = {'nIterations', 'OutlierIterations', 'MultFactor',
                         'NormalizeRingWeights', 'WeightByRadius', 'WeightByFitSNR',
                         'L2Objective', 'NPanelsY', 'NPanelsZ', 'PanelSizeY',
                         'PanelSizeZ', 'PanelGapsY', 'PanelGapsZ', 'FixPanelID',
                         'tolShifts', 'tolRotation', 'PerPanelLsd', 'PerPanelDistortion',
                         'PanelShiftsFile', 'DoubletSeparation', 'tolTilts', 'tolBC',
                         'tolLsd', 'tolP', 'tolP4', 'RingsToExclude', 'Width'}
            if key in skip_keys:
                continue

            # Absolutify file paths relative to the source parameter directory
            if key in ('MaskFile', 'MaskFN', 'Dark') and len(parts) > 1:
                abs_path = (calib_param_file.parent / parts[1]).resolve()
                fout.write(f"{key} {abs_path}\n")
                written_keys.add(key)
                continue

            if key in overrides:
                fout.write(f"{key} {overrides[key]}\n")
                written_keys.add(key)
            else:
                fout.write(line)
                written_keys.add(key)

        # Write any overrides that weren't in the original file
        for key, val in overrides.items():
            if key not in written_keys:
                fout.write(f"{key} {val}\n")

    return out_path


# =========================================================================
# Step 1+: Integration and peak fitting
# =========================================================================

def get_ring_radii(param_file: Path) -> list:
    """Run GetHKLList on the parameter file and parse ring radii.

    Returns: list of (RingNr, Radius) tuples with unique ring radii
    """
    hkl_bin = MIDAS_BIN / "GetHKLList"
    stdout = run_cmd([str(hkl_bin), str(param_file), "--stdout"])

    # Output format: "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius"
    # But stdout also contains diagnostic lines like "Will go from ..."
    rings = {}  # RingNr -> Radius
    for line in stdout.strip().split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('h '):
            continue
        parts = stripped.split()
        if len(parts) >= 11:
            try:
                ring_nr = int(float(parts[4]))
                radius = float(parts[10])
                if ring_nr not in rings:
                    rings[ring_nr] = radius
            except (ValueError, IndexError):
                continue  # skip non-data lines

    sorted_rings = sorted(rings.items())
    print(f"  Found {len(sorted_rings)} unique CeO2 rings")
    for nr, r in sorted_rings[:10]:
        print(f"    Ring {nr}: R = {r:.2f} px")
    if len(sorted_rings) > 10:
        print(f"    ... and {len(sorted_rings) - 10} more")
    return sorted_rings


def create_peak_params_file(ring_radii: list, out_path: Path, px: float,
                            max_rings: int = 15) -> int:
    """Create a peak_params.txt file with PeakLocation entries.

    ring_radii are in microns (from GetHKLList); convert to pixels for the integrator.
    """
    n_peaks = min(len(ring_radii), max_rings)
    with open(out_path, 'w') as f:
        f.write("# Auto-generated peak parameters for CeO2 calibration benchmark\n")
        f.write("DoPeakFit 1\n")
        f.write("FitROIPadding 30\n")
        for i in range(n_peaks):
            ring_nr, radius_um = ring_radii[i]
            radius_px = radius_um / px
            f.write(f"PeakLocation {radius_px:.6f}\n")
    print(f"  Wrote {n_peaks} peak locations (px) to {out_path}")
    for nr, r_um in ring_radii[:min(5, n_peaks)]:
        print(f"    Ring {nr}: {r_um:.1f} µm → {r_um/px:.2f} px")
    return n_peaks


def create_autodetect_params_file(out_path: Path, max_peaks: int = 20) -> int:
    """Create a peak_params.txt file for auto-detect mode (no PeakLocation lines)."""
    with open(out_path, 'w') as f:
        f.write("# Auto-detect mode: no PeakLocation specified\n")
        f.write("DoPeakFit 1\n")
        f.write(f"AutoDetectPeaks {max_peaks}\n")
        f.write("SNIPIterations 50\n")
        f.write("FitROIPadding 30\n")
    print(f"  Wrote auto-detect params (maxPeaks={max_peaks}) to {out_path}")
    return max_peaks


def read_fit_per_eta_csv(csv_path: Path) -> list:
    """Parse _fit_per_eta.csv into list of dicts."""
    rows = []
    if not csv_path.exists():
        return rows
    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'frame': int(row['Frame']),
                'eta_bin': int(row['EtaBin']),
                'eta_deg': float(row['EtaCen']),
                'peak_nr': int(row['PeakIdx']),
                'center_px': float(row['R_px']),
                'tth_deg': float(row['TwoTheta_deg']),
                'area': float(row['Area']),
                'fwhm_deg': float(row['FWHM_deg']),
            })
    return rows


def validate_autodetect_per_eta(per_eta_peaks: list, theoretical_radii: list,
                                px: float, n_eta_bins: int = 360):
    """Validate auto-detected peaks against theoretical CeO2 ring radii.

    For each theoretical ring, checks how many eta bins have a matching peak.
    A match occurs when |R_detected - R_theoretical| < tolerance.
    """
    print("\n" + "=" * 70)
    print("  AUTO-DETECT: Per-Eta-Bin Ring Detection Validation")
    print("=" * 70)

    # Theoretical radii in pixels
    ring_radii_px = [(nr, r_um / px) for nr, r_um in theoretical_radii]

    # Group peaks by eta_bin
    from collections import defaultdict
    by_eta = defaultdict(list)
    for p in per_eta_peaks:
        by_eta[p['eta_bin']].append(p)

    n_active_eta = len(by_eta)
    tolerance_px = 2.0  # match within 2 pixels

    print(f"  Active eta bins: {n_active_eta}")
    print(f"  Total detected peaks: {len(per_eta_peaks)}")
    print(f"  Match tolerance: {tolerance_px:.1f} px")
    print()

    print(f"{'Ring':>4} {'R_theory':>10} {'Detected':>10} {'Rate':>8} {'Mean_err':>10} {'Err_ppm':>10}")
    print("-" * 60)

    rings_detected = 0
    all_strains = []

    for ring_nr, R_theory in ring_radii_px:
        n_matched = 0
        center_errors = []

        for eta_bin, peaks in by_eta.items():
            # Find closest detected peak to this ring
            best_err = float('inf')
            for p in peaks:
                err = abs(p['center_px'] - R_theory)
                if err < best_err:
                    best_err = err
            if best_err < tolerance_px:
                n_matched += 1
                center_errors.append(best_err)

        rate = n_matched / max(n_active_eta, 1)
        mean_err = sum(center_errors) / max(len(center_errors), 1) if center_errors else 0
        err_ppm = (mean_err / max(R_theory, 1e-6)) * 1e6

        if rate >= 0.5:
            rings_detected += 1
            all_strains.append(err_ppm)

        status = '✓' if rate >= 0.5 else '✗'
        print(f"  {ring_nr:>3}{status} {R_theory:>10.2f} {n_matched:>8}/{n_active_eta} "
              f"{rate:>7.1%} {mean_err:>10.4f} {err_ppm:>10.1f}")

    print("-" * 60)
    mean_strain = sum(all_strains) / max(len(all_strains), 1) if all_strains else 0

    print(f"  Rings detected (≥50% eta bins): {rings_detected}/{len(ring_radii_px)}")
    if all_strains:
        print(f"  Mean center error: {mean_strain:.1f} ppm")
    print()

    # PASS criteria
    min_rings = 10
    max_err_ppm = 500
    pass_rings = rings_detected >= min_rings
    pass_err = mean_strain < max_err_ppm if all_strains else False

    if pass_rings and pass_err:
        print(f"  ✅ PASS: {rings_detected} rings detected (≥{min_rings}), "
              f"mean error {mean_strain:.0f} ppm (<{max_err_ppm})")
    else:
        reasons = []
        if not pass_rings:
            reasons.append(f"only {rings_detected}/{min_rings} rings detected")
        if not pass_err:
            reasons.append(f"mean error {mean_strain:.0f} ppm (>{max_err_ppm})")
        print(f"  ⚠️  WARNING: {', '.join(reasons)}")


def create_zarr_zip(work_dir: Path, param_file: Path) -> Path:
    """Create a Zarr zip from the calibration TIFF."""
    gen_script = MIDAS_HOME / "utils" / "ffGenerateZipRefactor.py"
    if not gen_script.exists():
        gen_script = MIDAS_HOME / "utils" / "converters" / "midas2zip.py"

    data_file = CALIB_DIR / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif"
    dark_file = CALIB_DIR / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif"

    if not data_file.exists():
        raise FileNotFoundError(f"Calibration data file not found: {data_file}")

    cmd = [
        sys.executable, str(gen_script),
        '-paramFN', str(param_file),
        '-dataFN', str(data_file),
        '-darkFN', str(dark_file),
        '-resultFolder', str(work_dir),
    ]

    try:
        run_cmd(cmd, cwd=str(work_dir))
    except Exception as e:
        print(f"  Warning: ffGenerateZipRefactor failed: {e}")
        print(f"  Trying alternative approach...")
        cmd2 = [
            sys.executable, str(MIDAS_HOME / "FF_HEDM" / "workflows" / "integrator.py"),
            '-paramFN', str(param_file),
            '-dataFN', str(data_file),
            '-darkFN', str(dark_file),
            '-resultFolder', str(work_dir),
            '-convertFiles', '1',
            '-mapDetector', '1',
            '-endFileNr', '-1',
        ]
        run_cmd(cmd2, cwd=str(work_dir))

    zips = list(work_dir.glob("*.MIDAS.zip"))
    if not zips:
        zips = list(work_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No Zarr zip generated in {work_dir}")

    print(f"  Zarr zip: {zips[0]}")
    return zips[0]


def run_integrator_with_peaks(zip_file: Path, peak_params: Path,
                              work_dir: Path, n_cpus: int = 4) -> Path:
    """Run DetectorMapper + IntegratorZarrOMP with peak fitting enabled."""
    # Step A: Run DetectorMapper (Zarr mode) to generate Map.bin/nMap.bin
    mapper = MIDAS_BIN / "DetectorMapper"
    if not mapper.exists():
        raise FileNotFoundError(f"DetectorMapper not found at {mapper}")
    print("  Running DetectorMapper (Zarr mode)...")
    run_cmd([str(mapper), "-zarrFN", str(zip_file),
             "-resultFolder", str(work_dir), "-nCPUs", str(n_cpus)],
            cwd=str(work_dir))

    # Step B: Run IntegratorZarrOMP with peak fitting
    integrator = MIDAS_BIN / "IntegratorZarrOMP"
    if not integrator.exists():
        raise FileNotFoundError(f"IntegratorZarrOMP not found at {integrator}")

    print("  Running IntegratorZarrOMP...")
    cmd = [str(integrator), str(zip_file), str(n_cpus), str(peak_params)]
    stdout = run_cmd(cmd, cwd=str(work_dir))
    # Print key diagnostic lines from integrator output
    diag_keys = ['peak', 'Peak', 'fit.bin', 'DoPeakFit', 'nPeaks',
                 'ROI', 'lineout', 'Warning', 'Error', 'nRBins', 'nEtaBins',
                 'PeakFit', 'REJECTED', 'validCount', 'FAILED']
    for line in stdout.split('\n'):
        stripped = line.strip()
        if stripped and any(k in stripped for k in diag_keys):
            print(f"    ▸ {stripped}")

    # Output files are named per data file (parallel-safe)
    # The integrator uses the zip filename minus .zip as the stem
    data_stem = zip_file.stem  # e.g. "foo.MIDAS" from "foo.MIDAS.zip"
    fit_bin = work_dir / f"{data_stem}_fit.bin"
    lineout_bin = work_dir / f"{data_stem}_lineout.bin"

    if lineout_bin.exists():
        print(f"  {lineout_bin.name}: {lineout_bin.stat().st_size} bytes")
    else:
        print(f"  WARNING: {lineout_bin.name} not generated")

    if fit_bin.exists():
        print(f"  {fit_bin.name}: {fit_bin.stat().st_size} bytes")
    else:
        print(f"  WARNING: {fit_bin.name} not generated")

    return fit_bin


def read_fit_bin(fit_bin: Path, n_peaks: int) -> list:
    """Read fit.bin and return per-peak parameters."""
    if not fit_bin.exists():
        return []

    data = fit_bin.read_bytes()
    expected_size = n_peaks * PF_PARAMS_PER_PEAK * 8

    if len(data) < expected_size:
        print(f"  WARNING: fit.bin size ({len(data)} bytes) < expected ({expected_size} bytes)")
        return []

    values = struct.unpack(f'{n_peaks * PF_PARAMS_PER_PEAK}d', data[:expected_size])

    peaks = []
    for i in range(n_peaks):
        base = i * PF_PARAMS_PER_PEAK
        peaks.append({
            'Area': values[base + 0],
            'Center': values[base + 1],
            'Sig': values[base + 2],
            'Gam': values[base + 3],
            'FWHM': values[base + 4],
            'Eta': values[base + 5],
            'ChiSq': values[base + 6],
        })

    return peaks


def compute_strain_benchmark(theoretical_radii: list, fitted_peaks: list,
                             px: float):
    """Compare fitted peak centers against theoretical ring radii."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Calibration vs Integration Peak Fitting")
    print("=" * 70)
    print(f"{'Ring':>4} {'R_theory':>10} {'R_fitted':>10} {'ΔR':>10} {'ΔR/R (ppm)':>12} {'Area':>10} {'ChiSq':>10}")
    print("-" * 70)

    strains = []
    for i, (ring_nr, R_theory) in enumerate(theoretical_radii):
        if i >= len(fitted_peaks):
            break

        peak = fitted_peaks[i]
        R_fitted = peak['Center']

        if R_fitted == 0 and peak['Area'] == 0:
            print(f"  {ring_nr:>4} {R_theory:>10.2f} {'FAILED':>10} {'--':>10} {'--':>12} {'--':>10} {'--':>10}")
            continue

        delta_R = R_fitted - R_theory
        strain_ppm = (delta_R / R_theory) * 1e6 if R_theory != 0 else 0
        strains.append(strain_ppm)

        print(f"  {ring_nr:>4} {R_theory:>10.2f} {R_fitted:>10.2f} {delta_R:>10.4f} "
              f"{strain_ppm:>12.1f} {peak['Area']:>10.1f} {peak['ChiSq']:>10.1f}")

    if strains:
        import statistics
        mean_strain = statistics.mean(strains)
        std_strain = statistics.stdev(strains) if len(strains) > 1 else 0
        max_strain = max(abs(s) for s in strains)

        print("-" * 70)
        print(f"  Mean strain:  {mean_strain:>10.1f} ppm")
        print(f"  Std strain:   {std_strain:>10.1f} ppm")
        print(f"  Max |strain|: {max_strain:>10.1f} ppm")
        print()

        threshold_ppm = 500
        if max_strain < threshold_ppm:
            print(f"  ✅ PASS: Max strain residual ({max_strain:.0f} ppm) < threshold ({threshold_ppm} ppm)")
        else:
            print(f"  ⚠️  WARNING: Max strain residual ({max_strain:.0f} ppm) > threshold ({threshold_ppm} ppm)")
            print(f"     This may indicate a geometry mismatch between calibration and integration.")
    else:
        print("  No valid strain measurements. All peak fits may have failed.")


def main():
    parser = argparse.ArgumentParser(
        description='Test IntegratorZarrOMP peak fitting with CeO2 calibration data')
    parser.add_argument('-nCPUs', type=int, default=4,
                        help='Number of CPUs (default: 4)')
    parser.add_argument('--keep-work-dir', action='store_true',
                        help='Keep the working directory after test')
    parser.add_argument('--work-dir', type=str, default=None,
                        help='Use a specific working directory')
    parser.add_argument('--max-rings', type=int, default=15,
                        help='Maximum number of rings to fit (default: 15)')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip calibration step, use parameters.txt as-is')
    parser.add_argument('--mode', choices=['specified', 'autodetect'],
                        default='specified',
                        help='Fitting mode: specified (PeakLocation) or autodetect')
    parser.add_argument('--integrator', choices=['cpu', 'gpu'],
                        default='cpu',
                        help='Integrator binary: cpu (IntegratorZarrOMP) or gpu (IntegratorFitPeaksGPUStream)')
    add_common_args(parser)
    args = parser.parse_args()

    # GPU integrator is a streaming server — different workflow
    if args.integrator == 'gpu':
        print("=" * 70)
        print("  GPU Integrator Test")
        print("=" * 70)
        gpu_bin = MIDAS_BIN / "IntegratorFitPeaksGPUStream"
        if not gpu_bin.exists():
            print(f"  ERROR: GPU integrator not found at {gpu_bin}")
            print(f"  Build with: cd ~/opt/MIDAS/build && cmake .. -DUSE_CUDA=ON && cmake --build . --target IntegratorFitPeaksGPUStream -j 8")
            sys.exit(1)
        print(f"  GPU binary found: {gpu_bin}")
        print()
        print("  IntegratorFitPeaksGPUStream is a streaming server.")
        print("  To test GPU peak fitting:")
        print()
        print("  1. Build with CUDA:")
        print("     cd ~/opt/MIDAS/build")
        print("     cmake .. -DUSE_CUDA=ON")
        print("     cmake --build . --target IntegratorFitPeaksGPUStream -j 8")
        print()
        print("  2. Create a combined parameter file with peak params:")
        print("     (DoPeakFit 1, PeakLocation entries, or AutoDetectPeaks N)")
        print()
        print("  3. Run the GPU integrator:")
        print(f"     {gpu_bin} <combined_params.txt>")
        print()
        print("  4. Send data via the streaming pipeline or use direct mode.")
        print()
        print("  HDF5 output: caked_peaks.h5 (written after all frames processed)")
        sys.exit(0)

    param_path = CALIB_DIR / "parameters.txt"
    if not param_path.exists():
        print(f"ERROR: Calibration parameter file not found: {param_path}")
        sys.exit(1)

    print("=" * 70)
    print("  MIDAS Integrator Peak Fitting Benchmark")
    print("=" * 70)
    print(f"  Calibration dir: {CALIB_DIR}")
    print(f"  CPUs: {args.nCPUs}")
    print(f"  Mode: {args.mode}")
    if args.skip_calibration:
        print(f"  ⚡ Calibration skipped (using parameters.txt as-is)")
    print()

    print_environment()

    if not getattr(args, 'skip_preflight', False):
        run_preflight(
            required_binaries=["IntegratorZarrOMP", "DetectorMapper",
                               "CalibrantPanelShiftsOMP", "GetHKLList"],
            required_packages=["numpy"],
            required_data_files=[str(param_path)],
        )

    # Create working directory
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix='midas_peak_test_'))

    print(f"  Working directory: {work_dir}")
    print()

    try:
        # ------------------------------------------------------------------
        # Step 0: Run calibration to get optimized geometry
        # ------------------------------------------------------------------
        if args.skip_calibration:
            print("[0/5] Skipping calibration (--skip-calibration)...")
            # Copy parameter file and use it directly
            integrator_param = work_dir / "integrator_params.txt"
            import shutil as _shutil
            _shutil.copy2(str(param_path), str(integrator_param))
            # Rewrite paths to absolute
            lines = []
            with open(integrator_param, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0] == 'Folder' and len(parts) > 1:
                        lines.append(f"Folder {work_dir}\n")
                    elif parts and parts[0] in ('MaskFile', 'MaskFN') and len(parts) > 1:
                        mask_abs = work_dir / parts[1]
                        lines.append(f"{parts[0]} {mask_abs}\n")
                    elif parts and parts[0] == 'Dark' and len(parts) > 1:
                        dark_abs = work_dir / parts[1]
                        lines.append(f"Dark {dark_abs}\n")
                    else:
                        lines.append(line)
            with open(integrator_param, 'w') as f:
                f.writelines(lines)
            # Copy data files to work dir
            for fp in CALIB_DIR.iterdir():
                if fp.is_file() and fp.name != '.DS_Store':
                    dest = work_dir / fp.name
                    if not dest.exists():
                        _shutil.copy2(str(fp), str(dest))
            # Generate HKLs
            hkl_bin = MIDAS_BIN / "GetHKLList"
            run_cmd([str(hkl_bin), str(integrator_param)], cwd=str(work_dir))
        else:
            print("[0/5] Running calibration (CalibrantPanelShiftsOMP)...")
            calib_dir, calib_param = prepare_calibration_dir(param_path)
            optimized = run_calibration(calib_param, args.nCPUs, calib_dir)

            print("\n  Optimized geometry:")
            for k in ['Lsd', 'BCy', 'BCz', 'ty', 'tz', 'p0', 'p1', 'p2', 'p3', 'p4']:
                if k in optimized:
                    print(f"    {k:6s} {optimized[k]:.12f}")

            # Build integrator parameter file with optimized geometry
            integrator_param = work_dir / "integrator_params.txt"
            build_integrator_param_file(param_path, optimized, integrator_param)
            print(f"\n  Created integrator params: {integrator_param}")

            # Clean up calibration temp dir
            shutil.rmtree(calib_dir, ignore_errors=True)
        print()

        # ------------------------------------------------------------------
        # Step 1: Get theoretical ring radii (using optimized params)
        # ------------------------------------------------------------------
        print("[1/5] Computing theoretical CeO2 ring radii via GetHKLList...")
        ring_radii = get_ring_radii(integrator_param)
        if not ring_radii:
            print("ERROR: No ring radii computed")
            sys.exit(1)
        print()

        # ------------------------------------------------------------------
        # Step 2: Create peak params file
        # ------------------------------------------------------------------
        print("[2/5] Creating peak_params.txt...")
        # Parse px from parameter file for unit conversion
        px = 172.0
        with open(integrator_param) as f:
            for line in f:
                if line.strip().startswith('px '):
                    px = float(line.strip().split()[1])
                    break
        peak_params = work_dir / "peak_params.txt"
        if args.mode == 'autodetect':
            n_peaks = create_autodetect_params_file(peak_params, max_peaks=20)
        else:
            n_peaks = create_peak_params_file(ring_radii, peak_params, px,
                                              max_rings=args.max_rings)
        print()

        # ------------------------------------------------------------------
        # Step 3: Create Zarr zip from calibration data
        # ------------------------------------------------------------------
        print("[3/5] Generating Zarr zip from calibration TIFF...")
        zip_file = create_zarr_zip(work_dir, integrator_param)
        print()

        # ------------------------------------------------------------------
        # Step 4: Run IntegratorZarrOMP with peak fitting
        # ------------------------------------------------------------------
        print("[4/5] Running IntegratorZarrOMP with peak fitting...")
        fit_bin = run_integrator_with_peaks(zip_file, peak_params, work_dir,
                                           n_cpus=args.nCPUs)
        print()

        # ------------------------------------------------------------------
        # Step 5: Read fit results and compute benchmark
        # ------------------------------------------------------------------
        if args.mode == 'autodetect':
            print("[5/5] Validating auto-detected peaks per eta bin...")
            # Find the per-eta CSV
            data_stem = zip_file.stem
            csv_path = work_dir / f"{data_stem}_fit_per_eta.csv"
            per_eta_peaks = read_fit_per_eta_csv(csv_path)
            if per_eta_peaks:
                print(f"  Read {len(per_eta_peaks)} per-eta peak entries from {csv_path.name}")
            else:
                print(f"  WARNING: No per-eta data found in {csv_path}")
            validate_autodetect_per_eta(per_eta_peaks, ring_radii, px)
        else:
            print("[5/5] Computing strain benchmark...")
            fitted_peaks = read_fit_bin(fit_bin, n_peaks)
            if fitted_peaks:
                for i, peak in enumerate(fitted_peaks[:5]):
                    print(f"  Peak {i}: Center={peak['Center']:.4f}, Area={peak['Area']:.2f}")
            else:
                print("  No fit results found")
            # Ring radii from GetHKLList are in microns; convert to pixels
            ring_radii_px = [(nr, r / px) for nr, r in ring_radii[:n_peaks]]
            compute_strain_benchmark(ring_radii_px, fitted_peaks, px)

    finally:
        if not args.keep_work_dir and not args.work_dir:
            print(f"\nCleaning up work directory: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\nWork directory preserved: {work_dir}")


if __name__ == '__main__':
    main()
