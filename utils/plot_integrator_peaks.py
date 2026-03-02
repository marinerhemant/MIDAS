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
    parser.add_argument('zarr_file', help='Path to *.caked.hdf.zarr.zip file')
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

    # Open zarr
    print(f"Opening {args.zarr_file} ...")
    z = zarr.open(str(args.zarr_file), mode='r')

    # Extract REtaMap: shape (4, nRBins, nEtaBins) = [Radius, 2Theta, Eta, BinArea]
    retamap = z['REtaMap'][:]
    tth_2d_raw = retamap[1]   # (nRBins, nEtaBins) - 2theta in degrees
    tth_2d = np.copy(tth_2d_raw)
    eta_2d_raw = retamap[2]   # (nRBins, nEtaBins) - eta in degrees
    eta_2d = np.copy(eta_2d_raw)

    # Filter out points with zero bin area
    valid = retamap[3] > 0
    tth_2d[~valid] = np.nan
    eta_2d[~valid] = np.nan

    # for each eta bin, print 2theta range and min-max values for 2theta
    for i in range(tth_2d.shape[1]):
        tth_range = np.nanmax(tth_2d[:, i]) - np.nanmin(tth_2d[:, i])
        tth_min = np.nanmin(tth_2d[:, i])
        tth_max = np.nanmax(tth_2d[:, i])
        print(f"eta bin {i}: {eta_2d_raw[0, i]:.4f} deg")
        print(f"  2theta range for eta bin {i}: {tth_range:.4f} deg")
        print(f"  2theta min-max for eta bin {i}: {tth_min:.4f} - {tth_max:.4f} deg")

    tth_2d = np.copy(tth_2d_raw)
    eta_2d = np.copy(eta_2d_raw)
    # 2theta and eta axes (constant along the other dimension)
    tth_axis = tth_2d[:, 0]    # 1D: nRBins values
    eta_axis = eta_2d[0, :]    # 1D: nEtaBins values
    nRBins, nEtaBins = tth_2d.shape
    print(f"  nRBins={nRBins}, nEtaBins={nEtaBins}")
    print(f"  2theta range: {tth_axis[0]:.4f} - {tth_axis[-1]:.4f} deg")
    print(f"  Eta range: {eta_axis[0]:.4f} - {eta_axis[-1]:.4f} deg")

    # Find OmegaSumFrame datasets
    if 'OmegaSumFrame' in z:
        osf = z['OmegaSumFrame']
        frame_keys = sorted(osf.keys(),
                            key=lambda k: int(k.split('_')[-1]))
        if args.frame == -1:
            frame_key = frame_keys[-1]
        elif 0 <= args.frame < len(frame_keys):
            frame_key = frame_keys[args.frame]
        else:
            print(f"Error: frame index {args.frame} out of range [0, {len(frame_keys)-1}]")
            sys.exit(1)

        print(f"  Using OmegaSumFrame/{frame_key}")
        intensity = osf[frame_key][:]  # (nRBins, nEtaBins)
    elif 'SumFrames' in z:
        print("  No OmegaSumFrame found, using SumFrames")
        intensity = z['SumFrames'][:]
    else:
        print("Error: No intensity data found in zarr file")
        sys.exit(1)

    print(f"  Intensity shape: {intensity.shape}, max={intensity.max():.1f}")

    # Auto-set thresholds
    imax = intensity.max()
    min_height = args.min_height if args.min_height is not None else 0.05 * imax
    prominence = args.prominence if args.prominence is not None else 0.03 * imax
    print(f"  Peak detection: min_height={min_height:.1f}, prominence={prominence:.1f}")

    # Fit peaks for each eta column
    all_peaks = []
    n_fitted = 0
    n_failed = 0

    for j in range(nEtaBins):
        profile = intensity[:, j]
        eta_val = eta_axis[j]

        # Skip if all zeros
        if np.max(profile) <= 0:
            continue

        # Find peaks
        peak_indices, properties = find_peaks(
            profile, height=min_height, prominence=prominence)

        for pi in peak_indices:
            result = fit_peak(tth_axis, profile, pi, half_window=args.fit_window)
            if result is not None:
                result['eta'] = eta_val
                result['eta_idx'] = j
                all_peaks.append(result)
                n_fitted += 1
            else:
                n_failed += 1

    print(f"\n  Found {n_fitted} peaks ({n_failed} fit failures) across {nEtaBins} eta bins")

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
        # Auto-derive: strip .analysis.MIDAS.zip.caked.hdf.zarr.zip -> .corr.csv
        base = args.zarr_file
        suffix = '.analysis.MIDAS.zip.caked.hdf.zarr.zip'
        if base.endswith(suffix):
            corr_csv = base[:-len(suffix)] + '.corr.csv'

    ideal_2thetas = None
    if corr_csv and os.path.isfile(corr_csv):
        print(f"  Loading ideal 2theta from {corr_csv}")
        corr_data = np.genfromtxt(corr_csv, skip_header=1)
        ideal_2thetas = np.sort(np.unique(corr_data[:, 6]))  # Ideal2Theta column
    elif corr_csv:
        print(f"  Warning: corr.csv not found at {corr_csv}")

    # Print ring summary, 10 decimal places for 2theta
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
                # Find closest ideal 2theta
                idx = np.argmin(np.abs(ideal_2thetas - mean_tth))
                ideal_val = ideal_2thetas[idx]
                print(f"  {r:4d}  {ideal_val:10.10f}  {mean_tth:10.10f}  {np.std(ring_tths):10.10f}  {len(ring_tths):8d}")
            else:
                print(f"  {r:4d}  {mean_tth:10.10f}  {np.std(ring_tths):10.10f}  {len(ring_tths):8d}")

    # Extract arrays for plotting
    etas = np.array([p['eta'] for p in all_peaks])
    tths = np.array([p['tth_fit'] for p in all_peaks])
    rings = np.array([p['ring'] for p in all_peaks])
    amps = np.array([p['amp'] for p in all_peaks])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    scatter = ax.scatter(etas, tths, c=rings, cmap='tab10', s=15, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax, label='Ring Number')
    cbar.set_ticks(range(n_rings))

    ax.set_xlabel('Eta (deg)', fontsize=12)
    ax.set_ylabel('Fitted 2θ (deg)', fontsize=12)
    ax.set_title(f'Peak Positions from {os.path.basename(args.zarr_file)}', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Overlay ideal 2theta lines
    if ideal_2thetas is not None:
        for tt in ideal_2thetas:
            ax.axhline(tt, color='red', linestyle='--', alpha=0.6, linewidth=2,
                       label=f'Ideal 2θ = {tt:.4f}°')
        # De-duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8,
                  loc='upper right', ncol=2)
        print(f"  Added {len(ideal_2thetas)} ideal 2theta lines")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches='tight')
        print(f"\n  Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
