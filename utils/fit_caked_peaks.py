#!/usr/bin/env python3
"""
Batch peak fitting for caked integrator output.

Reads ``*.caked.hdf.zarr.zip`` files produced by integrator.py /
integrator_batch_process.py, applies SNIP background subtraction,
detects and fits peaks with GSAS-II pseudo-Voigt profiles for every
(OmegaSumFrame, eta-bin) slice, and writes results to HDF5.

The (frame × eta) grid is embarrassingly parallel and processed with
``concurrent.futures.ProcessPoolExecutor``.

Usage:
    python fit_caked_peaks.py -zarrFN data.caked.hdf.zarr.zip \\
        [-nCPUs 8] [--snip-iter 50] [--n-peaks 6] [--fit-window 0.15] \\
        [--frames 0,-1] [--outDir .] [-paramFN params.txt]
"""

import argparse
import concurrent.futures
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import zarr

# Import shared peak-fitting machinery from extract_lineouts
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_lineouts import (
    snip_background,
    detect_peaks,
    fit_peaks_gsas,
    pseudo_voigt_no_bg,
    _tch_eta_fwhm,
)


# ── Geometry reader ─────────────────────────────────────────────────────

def read_wavelength(param_file):
    """Read wavelength from a MIDAS parameter file (for d-spacing calc)."""
    if param_file is None:
        return 0.0
    with open(param_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'Wavelength' and len(parts) >= 2:
                return float(parts[1])
    return 0.0


# ── Per-slice worker ────────────────────────────────────────────────────

def _fit_one_slice(args_tuple):
    """Worker: fit peaks in one (frame, eta) slice.

    Parameters
    ----------
    args_tuple : tuple
        (frame_idx, frame_key, eta_idx, tth_axis, profile,
         snip_iter, n_peaks, fit_window_deg, wavelength_A)

    Returns
    -------
    dict with keys: frame_idx, frame_key, eta_idx, eta_deg, peaks
         where peaks is a list of dicts with fit results.
    """
    (frame_idx, frame_key, eta_idx, eta_deg, tth_axis, profile,
     snip_iter, n_peaks, fit_window_deg, wavelength_A) = args_tuple

    result = {
        'frame_idx': frame_idx,
        'frame_key': frame_key,
        'eta_idx': eta_idx,
        'eta_deg': eta_deg,
        'peaks': [],
    }

    # Skip empty or all-zero profiles
    if len(profile) < 10 or np.max(profile) <= 0:
        return result

    try:
        # SNIP background
        bg = snip_background(profile, n_iter=snip_iter)
        corrected = profile - bg

        # Detect peaks
        peak_indices, peak_widths = detect_peaks(tth_axis, corrected,
                                                   n_peaks=n_peaks)
        if len(peak_indices) == 0:
            return result

        # Fit peaks with GSAS-II pseudo-Voigt
        fit_results = fit_peaks_gsas(
            tth_axis, corrected, peak_indices,
            fit_window_deg=fit_window_deg,
            wavelength_A=wavelength_A,
            peak_widths=peak_widths,
        )

        for k, r in enumerate(fit_results):
            result['peaks'].append({
                'peak_nr': k,
                'center_2theta': r['center_2theta'],
                'area': r['area'],
                'sig': r['sig'],
                'gam': r['gam'],
                'FWHM_deg': r['FWHM_deg'],
                'eta_mix': r['eta'],
                'd_spacing_A': r['d_spacing_A'],
                'chi_sq': r['chi_sq'],
            })

    except Exception:
        pass  # silently skip failed slices

    return result


# ── HDF5 output ─────────────────────────────────────────────────────────

def write_results_hdf5(all_results, tth_axis, eta_axis, frame_keys,
                       zarr_path, out_path):
    """Write peak fit results to HDF5.

    Structure:
        /metadata
            zarr_source     : string
            tth_axis        : 1D float array
            eta_axis        : 1D float array
            frame_keys      : 1D string array
        /peaks
            frame_idx       : 1D int
            frame_key       : 1D string
            eta_idx         : 1D int
            eta_deg         : 1D float
            peak_nr         : 1D int
            center_2theta   : 1D float
            area            : 1D float
            sig             : 1D float
            gam             : 1D float
            FWHM_deg        : 1D float
            eta_mix         : 1D float
            d_spacing_A     : 1D float
            chi_sq          : 1D float
    """
    # Flatten all peaks into column arrays
    rows = []
    for r in all_results:
        for p in r['peaks']:
            rows.append({
                'frame_idx': r['frame_idx'],
                'frame_key': r['frame_key'],
                'eta_idx': r['eta_idx'],
                'eta_deg': r['eta_deg'],
                **p,
            })

    n = len(rows)
    if n == 0:
        print("  WARNING: No peaks were fitted — HDF5 will be empty.")

    with h5py.File(str(out_path), 'w') as hf:
        # Metadata
        meta = hf.create_group('metadata')
        meta.attrs['zarr_source'] = str(zarr_path)
        meta.create_dataset('tth_axis', data=np.asarray(tth_axis, dtype=np.float64))
        meta.create_dataset('eta_axis', data=np.asarray(eta_axis, dtype=np.float64))
        dt_str = h5py.string_dtype()
        fk_ds = meta.create_dataset('frame_keys', shape=(len(frame_keys),), dtype=dt_str)
        for i, fk in enumerate(frame_keys):
            fk_ds[i] = fk

        if n == 0:
            return

        # Peak data
        pk = hf.create_group('peaks')

        int_cols = ['frame_idx', 'eta_idx', 'peak_nr']
        float_cols = ['eta_deg', 'center_2theta', 'area', 'sig', 'gam',
                      'FWHM_deg', 'eta_mix', 'd_spacing_A', 'chi_sq']

        for col in int_cols:
            pk.create_dataset(col, data=np.array([r[col] for r in rows], dtype=np.int32))

        for col in float_cols:
            pk.create_dataset(col, data=np.array([r[col] for r in rows], dtype=np.float64))

        # frame_key as variable-length string
        fk_arr = pk.create_dataset('frame_key', shape=(n,), dtype=dt_str)
        for i, r in enumerate(rows):
            fk_arr[i] = r['frame_key']

    print(f"  Wrote {n} peaks to {out_path}")


# ── Main ────────────────────────────────────────────────────────────────

def process_zarr_file(zarr_path, n_cpus=1, snip_iter=50, n_peaks=6,
                      fit_window_deg=0.15, wavelength_A=0.0,
                      frame_selection=None, out_dir=None):
    """Process one zarr file: fit peaks for all (frame, eta) slices.

    Parameters
    ----------
    zarr_path : Path
        Path to *.caked.hdf.zarr.zip
    n_cpus : int
        Number of parallel workers
    snip_iter : int
        SNIP background iterations
    n_peaks : int
        Max peaks to detect per slice
    fit_window_deg : float
        Peak fit ROI half-width in degrees
    wavelength_A : float
        Wavelength in Ångströms (for d-spacing; 0 = skip)
    frame_selection : list or None
        List of frame indices to process (None = all)
    out_dir : Path or None
        Output directory (default: same as zarr file)

    Returns
    -------
    Path to the output HDF5 file
    """
    zarr_path = Path(zarr_path).resolve()
    if out_dir is None:
        out_dir = zarr_path.parent
    else:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  fit_caked_peaks: {zarr_path.name}")
    print(f"{'='*70}")

    # Open zarr
    z = zarr.open(str(zarr_path), mode='r')

    # REtaMap → axes
    retamap = z['REtaMap'][:]
    tth_2d = retamap[1]  # [nRBins, nEtaBins]
    eta_2d = retamap[2]
    tth_axis = tth_2d[:, 0]
    eta_axis = eta_2d[0, :]
    nRBins, nEtaBins = tth_2d.shape
    print(f"  nRBins={nRBins}, nEtaBins={nEtaBins}")
    print(f"  2θ: {tth_axis[0]:.4f} – {tth_axis[-1]:.4f}°")
    print(f"  η:  {eta_axis[0]:.1f} – {eta_axis[-1]:.1f}°")

    # Enumerate frames
    if 'OmegaSumFrame' in z:
        osf = z['OmegaSumFrame']
        frame_keys = sorted(osf.keys(),
                            key=lambda k: int(k.split('_')[-1]))
    elif 'SumFrames' in z:
        frame_keys = ['SumFrames']
    else:
        print("  ERROR: No intensity data found in zarr file")
        return None

    # Apply frame selection
    if frame_selection is not None:
        selected = []
        for idx in frame_selection:
            if idx == -1:
                idx = len(frame_keys) - 1
            if 0 <= idx < len(frame_keys):
                selected.append(idx)
        frame_indices = sorted(set(selected))
    else:
        frame_indices = list(range(len(frame_keys)))

    n_frames = len(frame_indices)
    n_total_slices = n_frames * nEtaBins
    print(f"  Frames: {n_frames}, total slices: {n_total_slices}")
    print(f"  Workers: {n_cpus}")
    print(f"  SNIP iterations: {snip_iter}")
    print(f"  Max peaks/slice: {n_peaks}")
    print(f"  Fit window: ±{fit_window_deg}°")
    if wavelength_A > 0:
        print(f"  Wavelength: {wavelength_A} Å")
    print()

    # Build job list — pre-read all intensity data
    # (zarr random access is fast; reading upfront avoids pickling the zarr object)
    jobs = []
    for fi in frame_indices:
        fk = frame_keys[fi]
        if fk == 'SumFrames':
            intensity = z['SumFrames'][:]
        else:
            intensity = z['OmegaSumFrame'][fk][:]

        for j in range(nEtaBins):
            profile = np.asarray(intensity[:, j], dtype=np.float64)
            jobs.append((
                fi, fk, j, eta_axis[j],
                tth_axis, profile,
                snip_iter, n_peaks, fit_window_deg, wavelength_A,
            ))

    # Process in parallel
    t0 = time.time()
    all_results = []
    n_peaks_total = 0

    if n_cpus <= 1:
        # Serial mode (easier debugging)
        for i, job in enumerate(jobs):
            result = _fit_one_slice(job)
            all_results.append(result)
            n_peaks_total += len(result['peaks'])
            if (i + 1) % 100 == 0 or i == len(jobs) - 1:
                print(f"\r  Progress: {i+1}/{len(jobs)} slices, "
                      f"{n_peaks_total} peaks", end="", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as pool:
            futures = {pool.submit(_fit_one_slice, job): i
                       for i, job in enumerate(jobs)}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                all_results.append(result)
                n_peaks_total += len(result['peaks'])
                done = len(all_results)
                if done % 100 == 0 or done == len(jobs):
                    print(f"\r  Progress: {done}/{len(jobs)} slices, "
                          f"{n_peaks_total} peaks", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s — {n_peaks_total} peaks fitted")

    # Sort results by (frame_idx, eta_idx)
    all_results.sort(key=lambda r: (r['frame_idx'], r['eta_idx']))

    # Write HDF5
    # Output filename: replace .caked.hdf.zarr.zip with _caked_peaks.h5
    stem = zarr_path.name
    for suffix in ('.caked.hdf.zarr.zip', '.zarr.zip', '.zip'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    out_path = out_dir / f"{stem}_caked_peaks.h5"

    write_results_hdf5(all_results, tth_axis, eta_axis, frame_keys,
                       zarr_path, out_path)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch peak fitting for caked integrator output. "
                    "Reads *.caked.hdf.zarr.zip and writes per-peak "
                    "results to HDF5.")
    parser.add_argument('-zarrFN', required=True,
                        help='Path to *.caked.hdf.zarr.zip file')
    parser.add_argument('-nCPUs', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('-paramFN', default=None,
                        help='MIDAS parameter file (for wavelength/d-spacing)')
    parser.add_argument('-outDir', default=None,
                        help='Output directory (default: same as zarr file)')
    parser.add_argument('--snip-iter', type=int, default=50,
                        help='SNIP background iterations (default: 50)')
    parser.add_argument('--n-peaks', type=int, default=6,
                        help='Max peaks to detect per slice (default: 6)')
    parser.add_argument('--fit-window', type=float, default=0.15,
                        help='Peak fit ROI half-width in degrees 2θ (default: 0.15)')
    parser.add_argument('--frames', type=str, default=None,
                        help='Comma-separated frame indices to process '
                             '(default: all, -1 = last)')
    args = parser.parse_args()

    zarr_path = Path(args.zarrFN).resolve()
    if not zarr_path.exists():
        print(f"ERROR: Zarr file not found: {zarr_path}")
        sys.exit(1)

    wavelength = read_wavelength(args.paramFN) if args.paramFN else 0.0

    frame_selection = None
    if args.frames is not None:
        frame_selection = [int(x.strip()) for x in args.frames.split(',')]

    out_path = process_zarr_file(
        zarr_path,
        n_cpus=max(1, args.nCPUs),
        snip_iter=args.snip_iter,
        n_peaks=args.n_peaks,
        fit_window_deg=args.fit_window,
        wavelength_A=wavelength,
        frame_selection=frame_selection,
        out_dir=args.outDir,
    )

    if out_path:
        print(f"\n  Output: {out_path}")
    else:
        print("\n  ERROR: Processing failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
