#!/usr/bin/env python3
"""
Peak Fitting from Integrator Output

Reads the *.caked.hdf.zarr.zip output from integrator.py,
fits Gaussian peaks along the 2theta axis for each eta slice,
and produces a 2D scatter plot of fitted 2theta vs eta.

Usage:
    python plot_integrator_peaks.py <zarr.zip> [options]

Options:
    --min-height    Minimum peak height (default: auto = 5% of max intensity)
    --prominence    Minimum peak prominence (default: auto = 3% of max)
    --fit-window    Half-width in bins for Gaussian fit window (default: 5)
    --save          Save plot to file instead of showing
    --frame         Which OmegaSumFrame to use (-1 = last, default: -1)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def pseudo_voigt(x, amp, mu, sigma, eta, bg):
    """Pseudo-Voigt peak + constant background.

    pV(x) = eta * Lorentzian + (1 - eta) * Gaussian + bg
    where eta is the mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian).
    """
    dx = (x - mu) / sigma
    gauss = np.exp(-0.5 * dx ** 2)
    lorentz = 1.0 / (1.0 + dx ** 2)
    return amp * (eta * lorentz + (1.0 - eta) * gauss) + bg


def fit_peak(tth_arr, intensity, peak_idx, half_window=5):
    """Fit a pseudo-Voigt to a peak in the 1D intensity profile.

    Parameters
    ----------
    tth_arr : ndarray
        2theta values.
    intensity : ndarray
        Intensity values.
    peak_idx : int
        Index of the detected peak.
    half_window : int
        Half-width of the fit window in bins.

    Returns
    -------
    dict or None
        {'tth_fit': float, 'amp': float, 'sigma': float, 'eta': float, 'bg': float}
        or None if fit fails.
    """
    lo = max(0, peak_idx - half_window)
    hi = min(len(tth_arr), peak_idx + half_window + 1)
    x = tth_arr[lo:hi]
    y = intensity[lo:hi]

    if len(x) < 5:
        return None

    # Initial guesses
    amp0 = y.max() - y.min()
    mu0 = tth_arr[peak_idx]
    sigma0 = (x[-1] - x[0]) / 4.0
    eta0 = 0.5  # start halfway between Gaussian and Lorentzian
    bg0 = y.min()

    try:
        popt, _ = curve_fit(
            pseudo_voigt, x, y,
            p0=[amp0, mu0, sigma0, eta0, bg0],
            bounds=([0, x[0], 1e-6, 0.0, -np.inf],
                    [np.inf, x[-1], x[-1] - x[0], 1.0, np.inf]),
            maxfev=3000
        )
        return {'tth_fit': popt[1], 'amp': popt[0], 'sigma': popt[2],
                'eta_mix': popt[3], 'bg': popt[4]}
    except (RuntimeError, ValueError):
        return None


def assign_rings(peaks, tol_deg=0.05):
    """Cluster fitted 2theta values into rings.

    Parameters
    ----------
    peaks : list of dict
        Each with 'tth_fit' key.
    tol_deg : float
        Max 2theta difference to be considered same ring.

    Returns
    -------
    list of dict
        Same peaks with 'ring' key added.
    """
    if not peaks:
        return peaks

    # Sort by 2theta
    sorted_tth = sorted(set(round(p['tth_fit'], 4) for p in peaks))

    # Cluster into rings
    ring_centers = [sorted_tth[0]]
    for tth in sorted_tth[1:]:
        if tth - ring_centers[-1] > tol_deg:
            ring_centers.append(tth)
        else:
            # Update center to running mean
            ring_centers[-1] = (ring_centers[-1] + tth) / 2.0

    # Assign ring number to each peak
    for p in peaks:
        dists = [abs(p['tth_fit'] - c) for c in ring_centers]
        p['ring'] = np.argmin(dists)

    return peaks


def main():
    parser = argparse.ArgumentParser(
        description='Fit peaks in integrator zarr.zip output and plot 2theta vs eta')
    parser.add_argument('zarr_file', help='Path to *.caked.hdf.zarr.zip file (ge1 will be expanded to ge1-ge4)')
    parser.add_argument('--min-height', type=float, default=None,
                        help='Minimum peak height (default: 5%% of max intensity)')
    parser.add_argument('--prominence', type=float, default=None,
                        help='Minimum peak prominence (default: 3%% of max intensity)')
    parser.add_argument('--fit-window', type=int, default=5,
                        help='Half-width in bins for Gaussian fit window (default: 5)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--frame', type=int, default=-1,
                        help='Which OmegaSumFrame to use (-1 = last, default: -1)')
    parser.add_argument('--corr-csv', type=str, default=None,
                        help='Path to _corr.csv file for ideal 2theta lines (auto-derived if not given)')
    args = parser.parse_args()

    # Build list of ge1-ge4 filenames from the input file
    import re
    base_file = args.zarr_file
    ge_match = re.search(r'\.ge(\d)\.', base_file)
    if ge_match:
        zarr_files = []
        for i in range(1, 5):
            f = re.sub(r'\.ge\d\.', f'.ge{i}.', base_file)
            if os.path.exists(f):
                zarr_files.append((f'ge{i}', f))
            else:
                print(f"  Warning: {f} not found, skipping")
    else:
        zarr_files = [('ge1', base_file)]

    print(f"Processing {len(zarr_files)} detector(s): {', '.join(det for det, _ in zarr_files)}")

    # Collect peaks from all detectors
    all_peaks = []

    for det_id, zarr_path in zarr_files:
        print(f"\n{'='*70}")
        print(f"  Detector: {det_id}  —  {os.path.basename(zarr_path)}")
        print(f"{'='*70}")

        z = zarr.open(str(zarr_path), mode='r')

        # REtaMap
        retamap = z['REtaMap'][:]
        tth_2d = retamap[1]
        eta_2d = retamap[2]
        tth_axis = tth_2d[:, 0]
        eta_axis = eta_2d[0, :]
        nRBins, nEtaBins = tth_2d.shape
        print(f"  nRBins={nRBins}, nEtaBins={nEtaBins}")
        print(f"  2theta range: {tth_axis[0]:.4f} – {tth_axis[-1]:.4f} deg")
        print(f"  Eta range: {eta_axis[0]:.4f} – {eta_axis[-1]:.4f} deg")

        # Intensity
        if 'OmegaSumFrame' in z:
            osf = z['OmegaSumFrame']
            frame_keys = sorted(osf.keys(),
                                key=lambda k: int(k.split('_')[-1]))
            if args.frame == -1:
                frame_key = frame_keys[-1]
            elif 0 <= args.frame < len(frame_keys):
                frame_key = frame_keys[args.frame]
            else:
                print(f"  Error: frame {args.frame} out of range, skipping {det_id}")
                continue
            print(f"  Using OmegaSumFrame/{frame_key}")
            intensity = osf[frame_key][:]
        elif 'SumFrames' in z:
            print("  Using SumFrames")
            intensity = z['SumFrames'][:]
        else:
            print(f"  Error: No intensity data in {det_id}, skipping")
            continue

        print(f"  Intensity shape: {intensity.shape}, max={intensity.max():.1f}")

        # Auto-set thresholds per detector
        imax = intensity.max()
        min_height = args.min_height if args.min_height is not None else 0.05 * imax
        prominence = args.prominence if args.prominence is not None else 0.03 * imax
        print(f"  Peak detection: min_height={min_height:.1f}, prominence={prominence:.1f}")

        # Fit peaks for each eta bin
        det_fitted = 0
        det_failed = 0

        print(f"\n  {'EtaBin':>6s}  {'Eta(°)':>8s}  {'2θ_fit':>10s}  {'Amp':>10s}  "
              f"{'Sigma':>10s}  {'η_mix':>6s}  {'BG':>10s}")
        print(f"  {'------':>6s}  {'--------':>8s}  {'----------':>10s}  {'----------':>10s}  "
              f"{'----------':>10s}  {'------':>6s}  {'----------':>10s}")

        for j in range(nEtaBins):
            profile = intensity[:, j]
            eta_val = eta_axis[j]

            if np.max(profile) <= 0:
                continue

            peak_indices, _ = find_peaks(
                profile, height=min_height, prominence=prominence)

            for pi in peak_indices:
                result = fit_peak(tth_axis, profile, pi, half_window=args.fit_window)
                if result is not None:
                    result['eta'] = eta_val
                    result['eta_idx'] = j
                    result['det'] = det_id
                    all_peaks.append(result)
                    det_fitted += 1
                    print(f"  {j:6d}  {eta_val:8.2f}  {result['tth_fit']:10.6f}  "
                          f"{result['amp']:10.1f}  {result['sigma']:10.6f}  "
                          f"{result['eta_mix']:6.3f}  {result['bg']:10.1f}")
                else:
                    det_failed += 1

        print(f"\n  {det_id}: {det_fitted} peaks fitted, {det_failed} failures")

    # ---------- Combined results ----------
    print(f"\n{'='*70}")
    print(f"  COMBINED RESULTS: {len(all_peaks)} peaks across {len(zarr_files)} detector(s)")
    print(f"{'='*70}")

    if not all_peaks:
        print("No peaks found! Try lowering --min-height or --prominence.")
        sys.exit(1)

    # Assign ring numbers
    all_peaks = assign_rings(all_peaks)
    n_rings = max(p['ring'] for p in all_peaks) + 1
    print(f"  Assigned to {n_rings} rings")

    # Load ideal 2theta from _corr.csv if available
    corr_csv = args.corr_csv
    if corr_csv is None:
        base = args.zarr_file
        suffix = '.analysis.MIDAS.zip.caked.hdf.zarr.zip'
        if base.endswith(suffix):
            corr_csv = base[:-len(suffix)] + '.corr.csv'

    ideal_2thetas = None
    if corr_csv and os.path.isfile(corr_csv):
        print(f"  Loading ideal 2theta from {corr_csv}")
        corr_data = np.genfromtxt(corr_csv, skip_header=1)
        ideal_2thetas = np.sort(np.unique(corr_data[:, 6]))
    elif corr_csv:
        print(f"  Warning: corr.csv not found at {corr_csv}")

    # Ring summary
    if ideal_2thetas is not None:
        print(f"\n  {'Ring':>4s}  {'Ideal 2θ':>10s}  {'Mean 2θ':>10s}  {'Std 2θ':>10s}  {'NPoints':>8s}")
        print(f"  {'----':>4s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}")
    else:
        print(f"\n  {'Ring':>4s}  {'Mean 2θ':>10s}  {'Std 2θ':>10s}  {'NPoints':>8s}")
        print(f"  {'----':>4s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}")
    for r in range(n_rings):
        ring_tths = [p['tth_fit'] for p in all_peaks if p['ring'] == r]
        if ring_tths:
            mean_tth = np.mean(ring_tths)
            if ideal_2thetas is not None:
                idx = np.argmin(np.abs(ideal_2thetas - mean_tth))
                ideal_val = ideal_2thetas[idx]
                print(f"  {r:4d}  {ideal_val:10.4f}  {mean_tth:10.4f}  {np.std(ring_tths):10.6f}  {len(ring_tths):8d}")
            else:
                print(f"  {r:4d}  {mean_tth:10.4f}  {np.std(ring_tths):10.6f}  {len(ring_tths):8d}")

    # Plot — all detectors combined, different markers per detector
    det_markers = {'ge1': 'o', 'ge2': 's', 'ge3': '^', 'ge4': 'D'}
    etas = np.array([p['eta'] for p in all_peaks])
    tths = np.array([p['tth_fit'] for p in all_peaks])
    rings = np.array([p['ring'] for p in all_peaks])
    dets = np.array([p['det'] for p in all_peaks])

    fig, ax = plt.subplots(figsize=(14, 8))

    for det_id, _ in zarr_files:
        mask = dets == det_id
        if not np.any(mask):
            continue
        marker = det_markers.get(det_id, 'o')
        scatter = ax.scatter(etas[mask], tths[mask], c=rings[mask], cmap='tab10',
                             vmin=0, vmax=max(n_rings - 1, 1),
                             s=15, alpha=0.7, marker=marker, label=det_id)

    cbar = plt.colorbar(scatter, ax=ax, label='Ring Number')
    cbar.set_ticks(range(n_rings))

    ax.set_xlabel('Eta (deg)', fontsize=12)
    ax.set_ylabel('Fitted 2θ (deg)', fontsize=12)
    ax.set_title('Peak Positions — All Detectors', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper left')

    if ideal_2thetas is not None:
        for tt in ideal_2thetas:
            ax.axhline(tt, color='red', linestyle='--', alpha=0.6, linewidth=2)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches='tight')
        print(f"\n  Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()