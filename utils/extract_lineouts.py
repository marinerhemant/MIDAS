#!/usr/bin/env python3
"""
Extract 2θ vs intensity lineouts from a series of TIFF/HDF5 images.

For each file, runs IntegratorZarrOMP in direct mode (no Zarr creation)
and converts the resulting lineout.bin into a four-column .xy text file
with SNIP background subtraction. Optionally detects and fits peaks
using the GSAS-II pseudo-Voigt profile.

Usage:
    python extract_lineouts.py -paramFN params.txt -dataFN map1_00001.tif \
        -startNr 1 -endNr 100 [-nCPUs 4] [-outDir lineouts] [-darkFN dark.tif] \
        [--snip-iter 50] [--fit-peaks] [--n-peaks 6] [--fit-window 0.5]

The -dataFN argument can be either:
  - A literal filename with a number: map1_00001.tif
    (the last numeric group is auto-replaced with frame numbers)
  - A Python format pattern:  map1_{:05d}.tif

Output files are named  <stem>_lineout.xy  (e.g. map1_00001_lineout.xy).
When --fit-peaks is specified, <stem>_peaks.csv is also written.
"""

import argparse
import concurrent.futures
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent
MIDAS_BIN = MIDAS_HOME / "FF_HEDM" / "bin"


# ─── SNIP Background ────────────────────────────────────────────────────

def snip_background(intensities, n_iter=50):
    """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) background.

    Morháč et al., NIM A 401 (1997) 113–132.

    Parameters
    ----------
    intensities : 1D array of raw intensities
    n_iter      : number of clipping iterations (40–50 typical)

    Returns
    -------
    background : 1D array, same shape as intensities
    """
    y = np.asarray(intensities, dtype=np.float64).copy()
    n = len(y)

    # LLS (log-log-sqrt) transform — compresses dynamic range
    # Standard Morháč formulation: sqrt(y + 1) ensures positive argument
    y = np.log(np.log(np.sqrt(y + 1) + 1) + 1)

    # Iterative clipping with decreasing window (vectorized)
    for m in range(n_iter, 0, -1):
        avg = (y[:-2*m] + y[2*m:]) / 2.0  # average of neighbors at distance m
        y[m:n-m] = np.minimum(y[m:n-m], avg)

    # Inverse LLS transform (exact inverse of forward)
    bg = (np.exp(np.exp(y) - 1) - 1) ** 2 - 1

    return np.maximum(bg, 0)  # clip tiny negative from numerical noise


# ─── GSAS-II Pseudo-Voigt Profile ───────────────────────────────────────

def _tch_eta_fwhm(sig_centideg2, gam_centideg):
    """Thompson-Cox-Hastings: derive total FWHM (deg) and mixing eta
    from GSAS-II parameters.

    sig_centideg2 : Gaussian variance in centideg²
    gam_centideg  : Lorentzian FWHM in centideg
    Returns (FWHM_deg, eta)
    """
    # Convert to FWHM in degrees
    fg = np.sqrt(max(8.0 * np.log(2.0) * max(sig_centideg2, 1e-12), 0)) / 100.0
    fl = max(gam_centideg, 1e-6) / 100.0

    fg2, fg3, fg4, fg5 = fg**2, fg**3, fg**4, fg**5
    fl2, fl3, fl4, fl5 = fl**2, fl**3, fl**4, fl**5
    FWHM = (fg5 + 2.69269*fg4*fl + 2.42843*fg3*fl2 + 4.47163*fg2*fl3
            + 0.07842*fg*fl4 + fl5) ** 0.2
    if FWHM < 1e-15:
        return 1e-15, 0.5
    ratio = fl / FWHM
    eta = np.clip(1.36603*ratio - 0.47719*ratio**2 + 0.11116*ratio**3, 0, 1)
    return FWHM, eta


def pseudo_voigt_no_bg(x, area, center, sig, gam):
    """Area-normalized pseudo-Voigt with GSAS-II parameters (no background).

    Parameters
    ----------
    area   : integrated intensity (area under peak)
    center : peak position (degrees 2θ)
    sig    : Gaussian variance (centideg²)
    gam    : Lorentzian FWHM (centideg)
    """
    FWHM, eta = _tch_eta_fwhm(sig, gam)
    if FWHM < 1e-15:
        FWHM = 1e-15

    # Gaussian component (area-normalized)
    sigma_g = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    G = (1.0 / (sigma_g * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - center) / sigma_g)**2)

    # Lorentzian component (area-normalized)
    half_fwhm = FWHM / 2.0
    L = (half_fwhm / np.pi) / ((x - center)**2 + half_fwhm**2)

    return area * (eta * L + (1.0 - eta) * G)


# ─── Peak Detection ─────────────────────────────────────────────────────

def detect_peaks(tth, corrected, n_peaks=6):
    """Find the n strongest peaks in background-corrected lineout.

    Uses scipy.signal.find_peaks with prominence-based ranking.
    Uses small separation (0.03°) so doublet components are both detected.

    Returns
    -------
    peak_indices : array of indices into tth/corrected, sorted by 2θ
    """
    from scipy.signal import find_peaks

    # Small separation to detect both components of doublets
    dt = np.median(np.diff(tth)) if len(tth) > 1 else 0.004
    min_sep_deg = 0.03
    min_distance = max(5, int(min_sep_deg / dt))

    # Minimum prominence: 1% of max corrected intensity to reject noise
    min_prom = max(np.max(corrected) * 0.01, 1.0)

    peaks, properties = find_peaks(corrected, prominence=min_prom, width=1,
                                   distance=min_distance)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Sort by prominence (descending) and take extra candidates
    # (request 3× n_peaks to have room for multiplet partners)
    prominences = properties['prominences']
    order = np.argsort(prominences)[::-1]
    top = peaks[order[:n_peaks * 3]]

    # Return sorted by 2θ position
    return top[np.argsort(tth[top])]


# ─── Peak Fitting ────────────────────────────────────────────────────────

def _make_multiplet_model(n_peaks):
    """Create an N-peak pseudo-Voigt model function for curve_fit.

    Parameters are packed as: area1, center1, sig1, gam1, area2, ...
    """
    def model(x, *params):
        result = np.zeros_like(x)
        for i in range(n_peaks):
            base = i * 4
            result += pseudo_voigt_no_bg(x, params[base], params[base+1],
                                         params[base+2], params[base+3])
        return result
    return model


def _make_peak_result(center, area, sig, gam, wavelength_A):
    """Build a result dict for one fitted peak."""
    FWHM, eta = _tch_eta_fwhm(sig, gam)
    d_A = 0.0
    if wavelength_A > 0 and center > 0:
        theta_rad = math.radians(center / 2.0)
        if theta_rad > 0:
            d_A = wavelength_A / (2.0 * math.sin(theta_rad))
    return {
        'center_2theta': center,
        'area': area,
        'sig': sig,
        'gam': gam,
        'FWHM_deg': FWHM,
        'eta': eta,
        'd_spacing_A': d_A,
    }


def fit_peaks_gsas(tth, corrected, peak_indices, fit_window_deg=0.15,
                   wavelength_A=0.0, multiplet_threshold=0.15):
    """Fit detected peaks with GSAS-II pseudo-Voigt (no background).

    Peaks closer than multiplet_threshold are grouped and fitted together
    as a multi-peak model (sum of N pseudo-Voigts, N ≤ 3).

    Parameters
    ----------
    tth                 : 2θ array (degrees)
    corrected           : background-subtracted intensities
    peak_indices        : indices of detected peaks
    fit_window_deg      : half-width of fitting ROI (degrees)
    wavelength_A        : wavelength in Å (for d-spacing; 0 = skip)
    multiplet_threshold : max separation (deg) to group as multiplet

    Returns
    -------
    results : list of dicts with fit results per peak
    """
    from scipy.optimize import curve_fit

    results = []
    dt = np.median(np.diff(tth)) if len(tth) > 1 else 0.01

    # Group nearby peaks into multiplets (singlet / doublet / triplet)
    groups = []
    used = set()
    sorted_indices = list(peak_indices)
    for i, idx_i in enumerate(sorted_indices):
        if idx_i in used:
            continue
        group = [idx_i]
        used.add(idx_i)
        # Collect neighbors within threshold (up to 2 more → triplet max)
        for j in range(i + 1, len(sorted_indices)):
            if len(group) >= 3:
                break
            idx_j = sorted_indices[j]
            if idx_j in used:
                continue
            # Check distance to any member of the group
            if any(abs(tth[idx_j] - tth[g]) < multiplet_threshold
                   for g in group):
                group.append(idx_j)
                used.add(idx_j)
        groups.append(group)

    for group in groups:
        n_peaks = len(group)

        if n_peaks == 1:
            # ── Singlet ─────────────────────────────────────────────
            pk_idx = group[0]
            center_guess = tth[pk_idx]
            mask = np.abs(tth - center_guess) <= fit_window_deg
            x_roi = tth[mask]
            y_roi = corrected[mask]
            if len(x_roi) < 5:
                continue

            # Use height at peak index (not ROI max, which might be
            # from a neighbor's tail leaking into the window)
            peak_height = corrected[pk_idx]
            area_guess = peak_height * dt * max(3, len(x_roi) // 4)

            p0 = [area_guess, center_guess, 10.0, 5.0]
            # Center constrained to ±0.05° to prevent snapping to neighbors
            center_tol = 0.05
            bounds_lo = [0, center_guess - center_tol, 0.01, 0.01]
            bounds_hi = [area_guess * 100, center_guess + center_tol,
                         500.0, 500.0]

            try:
                popt, _ = curve_fit(pseudo_voigt_no_bg, x_roi, y_roi,
                                    p0=p0, bounds=(bounds_lo, bounds_hi),
                                    maxfev=5000)
                area, center, sig, gam = popt
                FWHM, _ = _tch_eta_fwhm(sig, gam)
                if FWHM > 0.15:
                    continue  # too wide for a singlet
                results.append(_make_peak_result(center, area, sig, gam,
                                                 wavelength_A))
            except (RuntimeError, ValueError):
                continue

        else:
            # ── Multiplet (doublet or triplet) ──────────────────────
            centers = [tth[g] for g in group]
            mid = np.mean(centers)
            span = max(centers) - min(centers)
            roi_half = max(fit_window_deg, span / 2 + 0.15)
            mask = np.abs(tth - mid) <= roi_half
            x_roi = tth[mask]
            y_roi = corrected[mask]
            if len(x_roi) < 5 * n_peaks:
                continue

            # Build p0, bounds for N peaks: [area, center, sig, gam] × N
            p0 = []
            bounds_lo = []
            bounds_hi = []
            for g_idx in group:
                h = corrected[g_idx] if g_idx < len(corrected) else 100
                a_guess = max(h * dt * 5, 1.0)
                c_guess = tth[g_idx]
                p0.extend([a_guess, c_guess, 10.0, 5.0])
                bounds_lo.extend([0, c_guess - 0.1, 0.01, 0.01])
                bounds_hi.extend([a_guess * 100, c_guess + 0.1, 500.0, 500.0])

            model = _make_multiplet_model(n_peaks)

            try:
                popt, _ = curve_fit(model, x_roi, y_roi,
                                    p0=p0, bounds=(bounds_lo, bounds_hi),
                                    maxfev=10000)
                for k in range(n_peaks):
                    base = k * 4
                    a, c, s, g = popt[base:base+4]
                    results.append(_make_peak_result(c, a, s, g,
                                                     wavelength_A))
            except (RuntimeError, ValueError):
                continue

    # Deduplicate: merge peaks within 0.02° (keep highest area)
    if len(results) > 1:
        results.sort(key=lambda r: r['center_2theta'])
        deduped = [results[0]]
        for r in results[1:]:
            if abs(r['center_2theta'] - deduped[-1]['center_2theta']) < 0.02:
                if r['area'] > deduped[-1]['area']:
                    deduped[-1] = r
            else:
                deduped.append(r)
        results = deduped

    return results


# ─── Geometry Reader ─────────────────────────────────────────────────────

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


# ─── DetectorMapper ──────────────────────────────────────────────────────

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


# ─── Per-Frame Worker ────────────────────────────────────────────────────

def _process_one_frame(nr, data_file, param_file, out_dir, out_xy, geom,
                       dark_file, end_nr, snip_iter, do_fit_peaks, n_peaks,
                       fit_window_deg):
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

        # Read lineout data
        lineout_xy_src = work_dir / f"{stem}_lineout.xy"
        lineout_bin = work_dir / f"{stem}_lineout.bin"

        tth_arr = []
        int_arr = []

        if lineout_xy_src.exists():
            # Parse existing .xy (2-column: 2θ, intensity)
            with open(lineout_xy_src) as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        tth_arr.append(float(parts[0]))
                        int_arr.append(float(parts[1]))
        elif lineout_bin.exists():
            # Fallback: convert binary using parameter-file geometry
            data = lineout_bin.read_bytes()
            n_rbins = len(data) // 8
            if n_rbins == 0:
                return (nr, False, "empty lineout.bin")
            intensities = struct.unpack(f'{n_rbins}d', data[:n_rbins * 8])
            px = geom['px']
            Lsd = geom['Lsd']
            RMin = geom['RMin']
            RBinSize = geom['RBinSize']
            for i in range(n_rbins):
                R_px = RMin + (i + 0.5) * RBinSize
                R_um = R_px * px
                tth_deg = math.degrees(math.atan(R_um / Lsd)) if Lsd > 0 else 0.0
                val = intensities[i]
                if math.isnan(val):
                    continue
                tth_arr.append(tth_deg)
                int_arr.append(val)
        else:
            return (nr, False, f"no {stem}_lineout.xy or _lineout.bin")

        if len(tth_arr) < 10:
            return (nr, False, "too few points in lineout")

        tth_np = np.array(tth_arr)
        int_np = np.array(int_arr)

        # SNIP background
        bg = snip_background(int_np, n_iter=snip_iter)
        corrected = int_np - bg

        # Write 4-column .xy
        with open(out_xy, 'w') as f:
            f.write("# 2theta_deg  intensity  snip_background  corrected\n")
            for i in range(len(tth_np)):
                f.write(f"{tth_np[i]:.6f}  {int_np[i]:.6f}  "
                        f"{bg[i]:.6f}  {corrected[i]:.6f}\n")

        # Peak detection + fitting (optional)
        if do_fit_peaks:
            peak_indices = detect_peaks(tth_np, corrected, n_peaks=n_peaks)
            if len(peak_indices) > 0:
                wavelength = geom.get('Wavelength', 0.0)
                fit_results = fit_peaks_gsas(tth_np, corrected, peak_indices,
                                            fit_window_deg=fit_window_deg,
                                            wavelength_A=wavelength)
                if fit_results:
                    peaks_csv = out_xy.parent / f"{stem}_peaks.csv"
                    with open(peaks_csv, 'w') as f:
                        f.write("# peak_nr,center_2theta,area,"
                                "sig_centideg2,gam_centideg,"
                                "FWHM_deg,eta,d_spacing_A\n")
                        for k, r in enumerate(fit_results, 1):
                            f.write(f"{k},{r['center_2theta']:.6f},"
                                    f"{r['area']:.2f},"
                                    f"{r['sig']:.4f},{r['gam']:.4f},"
                                    f"{r['FWHM_deg']:.6f},{r['eta']:.4f},"
                                    f"{r['d_spacing_A']:.6f}\n")

        return (nr, True, "OK")

    except Exception as e:
        import traceback
        return (nr, False, f"{e}\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract 2θ vs intensity lineouts from image series "
                    "with SNIP background subtraction and optional peak fitting.")
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
    parser.add_argument('-inputDir', default='.',
                        help='Input directory containing data files (default: cwd)')
    parser.add_argument('-outDir', default='.',
                        help='Output directory for .xy files (default: cwd)')
    parser.add_argument('-darkFN', default=None,
                        help='Optional dark image file')
    # SNIP background
    parser.add_argument('--snip-iter', type=int, default=50,
                        help='SNIP background iterations (default: 50)')
    # Peak fitting
    parser.add_argument('--fit-peaks', action='store_true',
                        help='Enable peak detection and GSAS-II pseudo-Voigt fitting')
    parser.add_argument('--n-peaks', type=int, default=6,
                        help='Number of strongest peaks to detect (default: 6)')
    parser.add_argument('--fit-window', type=float, default=0.15,
                        help='Peak fit ROI half-width in degrees 2θ (default: 0.15)')
    args = parser.parse_args()

    param_file = Path(args.paramFN).resolve()
    if not param_file.exists():
        print(f"ERROR: Parameter file not found: {param_file}")
        sys.exit(1)

    dark_file = Path(args.darkFN).resolve() if args.darkFN else None

    input_dir = Path(args.inputDir).resolve()
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
    print(f"  Input:  {input_dir}")
    print(f"  Param:  {param_file}")
    print(f"  Output: {out_dir}")
    print(f"  SNIP iterations: {args.snip_iter}")
    if args.fit_peaks:
        print(f"  Peak fitting: enabled ({args.n_peaks} peaks, "
              f"±{args.fit_window}° window)")
    print()

    # --- Copy auxiliary files (mask, dark) to output dir if needed ---
    with open(param_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or len(parts) < 2:
                continue
            if parts[0] in ('MaskFile', 'MaskFN', 'Dark'):
                src = Path(parts[1])
                if not src.is_absolute():
                    src = param_file.parent / src
                src = src.resolve()
                dst = out_dir / src.name
                if src.exists() and not dst.exists():
                    shutil.copy2(str(src), str(dst))
                    print(f"  Copied {parts[0]}: {src.name} → {out_dir.name}/")

    # --- Run DetectorMapper once to produce Map.bin + nMap.bin ---
    if not run_detector_mapper(param_file, out_dir, n_workers):
        print("ERROR: DetectorMapper failed, cannot proceed.")
        sys.exit(1)
    print()

    # --- Build filename pattern ---
    data_pattern = args.dataFN
    if '{' not in data_pattern:
        stem_part = Path(data_pattern).stem
        m = list(re.finditer(r'\d+', stem_part))
        if m:
            last_match = m[-1]
            width = len(last_match.group())
            ext = Path(data_pattern).suffix
            prefix = stem_part[:last_match.start()]
            suffix = stem_part[last_match.end():]
            dir_part = str(Path(data_pattern).parent)
            if dir_part == '.':
                data_pattern = f"{prefix}{{:0{width}d}}{suffix}{ext}"
            else:
                data_pattern = f"{dir_part}/{prefix}{{:0{width}d}}{suffix}{ext}"
            print(f"  Auto-detected pattern: {data_pattern}")
        else:
            print(f"ERROR: No numeric group found in {data_pattern}")
            sys.exit(1)

    # --- Build list of (nr, data_file, out_xy) jobs ---
    jobs = []
    for nr in range(args.startNr, args.endNr + 1):
        try:
            fn = data_pattern.format(nr)
        except (IndexError, KeyError):
            fn = data_pattern.replace('{}', str(nr))

        data_file = input_dir / fn
        if not data_file.exists():
            data_file = Path(fn)
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
                        oxy, geom, dark_file, args.endNr,
                        args.snip_iter, args.fit_peaks, args.n_peaks,
                        args.fit_window): (nr, df, oxy)
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
