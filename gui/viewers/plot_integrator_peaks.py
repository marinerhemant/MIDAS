#!/usr/bin/env python3
"""
Peak Fitting from Integrator Output (GSAS-II compatible)

Reads the *.caked.hdf.zarr.zip output from integrator.py,
fits area-normalized pseudo-Voigt peaks (TCH mixing, GSAS-II parameter
conventions) along the 2theta axis for each eta slice, and produces
a 2D scatter plot of fitted 2theta vs eta.

GSAS-II parameter conventions
-----------------------------
  sig   Gaussian variance in centideg² (FWHM_G = √(8·ln2·sig)/100 deg)
  gam   Lorentzian FWHM in centideg    (FWHM_L = gam/100 deg)
  int   Integrated intensity (area under peak, above background)
  bg0   Chebyshev T₀ coefficient (constant)
  bg1   Chebyshev T₁ coefficient (linear)

Usage:
    python plot_integrator_peaks.py <zarr.zip> [options]

Options:
    --min-height    Minimum peak height (default: auto = 5% of max intensity)
    --prominence    Minimum peak prominence (default: auto = 3% of max)
    --fit-window    Half-width in bins for peak fit window (default: 50)
    --save          Save plot to file instead of showing
    --frame         Which OmegaSumFrame to use (-1 = last, default: -1)
    --eta-bin       Eta bin index for diagnostic plot (requires --tth-range)
    --tth-range     2theta range [min max] for diagnostic plot (requires --eta-bin)
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ── GSAS-II style profile functions ─────────────────────────────────────

def _tch_eta_fwhm(sig_centideg2, gam_centideg):
    """Thompson-Cox-Hastings: derive total FWHM (deg) and mixing eta
    from GSAS-II parameters.

    sig_centideg2 : Gaussian variance in centideg²
    gam_centideg  : Lorentzian FWHM in centideg
    Returns (FWHM_deg, eta)
    """
    # Convert to FWHM in degrees
    fg = np.sqrt(max(8.0 * np.log(2.0) * max(sig_centideg2, 1e-12), 0)) / 100.0  # Gaussian FWHM (deg)
    fl = max(gam_centideg, 1e-6) / 100.0  # Lorentzian FWHM (deg)

    fg2, fg3, fg4, fg5 = fg**2, fg**3, fg**4, fg**5
    fl2, fl3, fl4, fl5 = fl**2, fl**3, fl**4, fl**5
    FWHM = (fg5 + 2.69269*fg4*fl + 2.42843*fg3*fl2 + 4.47163*fg2*fl3
            + 0.07842*fg*fl4 + fl5) ** 0.2
    if FWHM < 1e-15:
        return 1e-15, 0.5
    ratio = fl / FWHM
    eta = np.clip(1.36603*ratio - 0.47719*ratio**2 + 0.11116*ratio**3, 0, 1)
    return FWHM, eta


def pseudo_voigt_gsas(x, area, center, sig, gam, bg0, bg1, x_lo, x_hi):
    """Area-normalized pseudo-Voigt with GSAS-II parameters and Chebyshev BG.

    Fitted parameters:
      area   : integrated intensity (area under peak above background)
      center : peak position (degrees 2theta)
      sig    : Gaussian variance (centideg²)
      gam    : Lorentzian FWHM (centideg)
      bg0    : Chebyshev T₀ coefficient (constant)
      bg1    : Chebyshev T₁ coefficient (linear slope)

    Fixed parameters:
      x_lo, x_hi : window bounds for Chebyshev normalization
    """
    FWHM, eta = _tch_eta_fwhm(sig, gam)
    if FWHM < 1e-15:
        FWHM = 1e-15

    # Gaussian component (area-normalized)
    sigma_g = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # σ from FWHM
    G = (1.0 / (sigma_g * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - center) / sigma_g)**2)

    # Lorentzian component (area-normalized)
    half_fwhm = FWHM / 2.0
    L = (half_fwhm / np.pi) / ((x - center)**2 + half_fwhm**2)

    # Chebyshev background: T0=1, T1=x_norm
    x_norm = 2.0 * (x - x_lo) / max(x_hi - x_lo, 1e-12) - 1.0
    bg = bg0 + bg1 * x_norm

    return area * (eta * L + (1.0 - eta) * G) + bg


# ── Peak fitting ────────────────────────────────────────────────────────

def fit_peak(tth_arr, intensity, peak_idx, half_window=50):
    """Fit a GSAS-II style area-normalized pseudo-Voigt (TCH) to a peak.

    Parameters
    ----------
    tth_arr : ndarray
        2theta values (degrees).
    intensity : ndarray
        Intensity values.
    peak_idx : int
        Index of the detected peak.
    half_window : int
        Half-width of the fit window in bins (default: 50).

    Returns
    -------
    dict or None
        {'tth_fit', 'area', 'sig', 'gam', 'FWHM', 'eta', 'bg0', 'bg1',
         'Imax_calc', 'x_fit', 'y_fit', 'y_model'}
        or None if fit fails.
    """
    lo = max(0, peak_idx - half_window)
    hi = min(len(tth_arr), peak_idx + half_window + 1)
    x = tth_arr[lo:hi]
    y = intensity[lo:hi]

    if len(x) < 7:
        return None

    x_lo, x_hi = x[0], x[-1]
    span = x_hi - x_lo

    # Initial guesses — estimate FWHM, then convert to GSAS-II units
    bg_est = np.median(np.concatenate([y[:5], y[-5:]]))
    height_est = max(y.max() - bg_est, 1.0)
    center0 = tth_arr[peak_idx]

    # Estimate FWHM (degrees) from half-max crossings
    fwhm_deg = span / 6.0
    half_max = height_est / 2.0 + bg_est
    above = np.where(y > half_max)[0]
    if len(above) >= 2:
        fwhm_deg = x[above[-1]] - x[above[0]]
        if fwhm_deg < span / 50.0:
            fwhm_deg = span / 6.0

    # Convert to GSAS-II units
    fwhm_centideg = fwhm_deg * 100.0
    sig0 = fwhm_centideg**2 / (8.0 * np.log(2.0))   # centideg²
    gam0 = fwhm_centideg                              # centideg

    # Area estimate: height × FWHM × factor (~height × FWHM for pseudo-Voigt)
    area0 = height_est * fwhm_deg * 1.0645  # ~√(π/(4·ln2)) for Gaussian

    bg0_init = bg_est
    bg1_init = 0.0

    # Wrapper to fix x_lo, x_hi
    def model(xx, area, center, sig, gam, bg0, bg1):
        return pseudo_voigt_gsas(xx, area, center, sig, gam, bg0, bg1, x_lo, x_hi)

    try:
        popt, pcov = curve_fit(
            model, x, y,
            p0=[area0, center0, sig0, gam0, bg0_init, bg1_init],
            bounds=([0,      x_lo,    1e-3,    1e-3,  -np.inf, -np.inf],
                    [np.inf, x_hi,    1e8,     1e6,    np.inf,  np.inf]),
            maxfev=10000
        )
        area, center, sig, gam, bg0, bg1 = popt
        FWHM, eta = _tch_eta_fwhm(sig, gam)
        # Compute peak height from area and profile shape
        sigma_g = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        G_peak = 1.0 / (sigma_g * np.sqrt(2.0 * np.pi)) if sigma_g > 0 else 0
        L_peak = (2.0 / (np.pi * FWHM)) if FWHM > 0 else 0
        Imax_calc = area * (eta * L_peak + (1.0 - eta) * G_peak)

        y_model = model(x, *popt)

        return {'tth_fit': center, 'area': area, 'sig': sig, 'gam': gam,
                'FWHM': FWHM, 'eta': eta, 'bg0': bg0, 'bg1': bg1,
                'Imax_calc': Imax_calc,
                'x_fit': x, 'y_fit': y, 'y_model': y_model}
    except (RuntimeError, ValueError):
        return None


# ── Ring assignment ─────────────────────────────────────────────────────

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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Fit peaks in integrator zarr.zip output (GSAS-II conventions)')
    parser.add_argument('zarr_file', help='Path to *.caked.hdf.zarr.zip file (ge1 will be expanded to ge1-ge4)')
    parser.add_argument('--min-height', type=float, default=None,
                        help='Minimum peak height (default: 5%% of max intensity)')
    parser.add_argument('--prominence', type=float, default=None,
                        help='Minimum peak prominence (default: 3%% of max intensity)')
    parser.add_argument('--fit-window', type=int, default=50,
                        help='Half-width in bins for peak fit window (default: 50)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of showing')
    parser.add_argument('--frame', type=int, default=-1,
                        help='Which OmegaSumFrame to use (-1 = last, default: -1)')
    parser.add_argument('--corr-csv', type=str, default=None,
                        help='Path to _corr.csv file for ideal 2theta lines (auto-derived if not given)')

    # Diagnostic plot options
    parser.add_argument('--eta-bin', type=int, default=None,
                        help='Eta bin index for diagnostic profile plot (requires --tth-range)')
    parser.add_argument('--tth-range', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='2theta range [min max] for diagnostic plot (requires --eta-bin)')

    args = parser.parse_args()

    # Validate diagnostic options
    diag_mode = False
    if args.eta_bin is not None and args.tth_range is not None:
        diag_mode = True
    elif args.eta_bin is not None or args.tth_range is not None:
        print("Warning: both --eta-bin and --tth-range must be specified for diagnostic plot; ignoring.")

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
    diag_data = None  # will hold diagnostic data for one eta bin

    for det_id, zarr_path in zarr_files:
        print(f"\n{'='*80}")
        print(f"  Detector: {det_id}  —  {os.path.basename(zarr_path)}")
        print(f"{'='*80}")

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

        print(f"\n  {'EtaBin':>6s}  {'Eta(°)':>8s}  {'2θ_fit':>10s}  "
              f"{'Area':>10s}  {'Imax':>10s}  "
              f"{'Sig(cd²)':>10s}  {'Gam(cd)':>10s}  "
              f"{'FWHM(°)':>8s}  {'η':>5s}  {'BG0':>8s}  {'BG1':>8s}")
        print(f"  {'------':>6s}  {'--------':>8s}  {'----------':>10s}  "
              f"{'----------':>10s}  {'----------':>10s}  "
              f"{'----------':>10s}  {'----------':>10s}  "
              f"{'--------':>8s}  {'-----':>5s}  {'--------':>8s}  {'--------':>8s}")

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
                          f"{result['area']:10.2f}  {result['Imax_calc']:10.1f}  "
                          f"{result['sig']:10.2f}  {result['gam']:10.2f}  "
                          f"{result['FWHM']:8.5f}  "
                          f"{result['eta']:5.3f}  {result['bg0']:8.1f}  {result['bg1']:8.2f}")
                else:
                    det_failed += 1

            # Capture diagnostic data for selected eta bin
            if diag_mode and j == args.eta_bin and det_id == zarr_files[0][0]:
                diag_data = {
                    'tth_axis': tth_axis,
                    'profile': profile,
                    'eta_val': eta_val,
                    'eta_idx': j,
                    'det': det_id,
                    'peaks': [p for p in all_peaks if p.get('eta_idx') == j
                              and p.get('det') == det_id],
                }

        print(f"\n  {det_id}: {det_fitted} peaks fitted, {det_failed} failures")

    # ---------- Combined results ----------
    print(f"\n{'='*80}")
    print(f"  COMBINED RESULTS: {len(all_peaks)} peaks across {len(zarr_files)} detector(s)")
    print(f"{'='*80}")

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

    # ── Diagnostic: per-peak printout and profile plot for selected eta bin ──
    if diag_mode and diag_data is not None:
        tth_lo, tth_hi = args.tth_range
        eta_idx = args.eta_bin
        eta_val = diag_data['eta_val']
        profile = diag_data['profile']
        tth_axis = diag_data['tth_axis']

        # Filter peaks in the requested 2theta range
        diag_peaks = [p for p in all_peaks
                      if p.get('eta_idx') == eta_idx
                      and p.get('det') == diag_data['det']
                      and tth_lo <= p['tth_fit'] <= tth_hi]

        print(f"\n{'='*80}")
        print(f"  DIAGNOSTIC: Eta bin {eta_idx} (η = {eta_val:.2f}°), "
              f"2θ = [{tth_lo:.4f}, {tth_hi:.4f}]")
        print(f"{'='*80}")

        if diag_peaks:
            print(f"\n  {'#':>3s}  {'2θ_fit':>10s}  {'Area':>12s}  {'Imax':>10s}  "
                  f"{'sig(cd²)':>10s}  {'gam(cd)':>10s}  "
                  f"{'FWHM(°)':>10s}  {'η_mix':>6s}  {'BG0':>10s}  {'BG1':>10s}")
            print(f"  {'---':>3s}  {'----------':>10s}  {'------------':>12s}  {'----------':>10s}  "
                  f"{'----------':>10s}  {'----------':>10s}  "
                  f"{'----------':>10s}  {'------':>6s}  {'----------':>10s}  {'----------':>10s}")
            for i, p in enumerate(diag_peaks):
                print(f"  {i:3d}  {p['tth_fit']:10.6f}  {p['area']:12.4f}  {p['Imax_calc']:10.2f}  "
                      f"{p['sig']:10.4f}  {p['gam']:10.4f}  "
                      f"{p['FWHM']:10.6f}  {p['eta']:6.4f}  "
                      f"{p['bg0']:10.2f}  {p['bg1']:10.4f}")

            # Build composite model for the 2theta range
            mask = (tth_axis >= tth_lo) & (tth_axis <= tth_hi)
            x_diag = tth_axis[mask]
            y_diag = profile[mask]

            if len(x_diag) > 0:
                x_lo_w, x_hi_w = x_diag[0], x_diag[-1]

                # Dense grid for smooth fitted curve (10× data density)
                x_smooth = np.linspace(x_lo_w, x_hi_w, max(len(x_diag) * 10, 500))

                # Sum peak contributions on both grids
                y_total_smooth = np.zeros_like(x_smooth, dtype=float)
                y_bg_smooth = np.zeros_like(x_smooth, dtype=float)
                y_model_data = np.zeros_like(x_diag, dtype=float)  # for residual

                for p in diag_peaks:
                    # Smooth curve
                    y_pk_s = pseudo_voigt_gsas(
                        x_smooth, p['area'], p['tth_fit'],
                        p['sig'], p['gam'], p['bg0'], p['bg1'],
                        x_lo_w, x_hi_w)
                    x_norm_s = 2.0 * (x_smooth - x_lo_w) / max(x_hi_w - x_lo_w, 1e-12) - 1.0
                    bg_s = p['bg0'] + p['bg1'] * x_norm_s
                    y_total_smooth += (y_pk_s - bg_s)
                    y_bg_smooth += bg_s

                    # Data-grid model (for residual)
                    y_pk_d = pseudo_voigt_gsas(
                        x_diag, p['area'], p['tth_fit'],
                        p['sig'], p['gam'], p['bg0'], p['bg1'],
                        x_lo_w, x_hi_w)
                    y_model_data += y_pk_d

                if len(diag_peaks) > 1:
                    y_bg_smooth /= len(diag_peaks)
                    # Adjust model_data for averaged background
                    x_norm_d = 2.0 * (x_diag - x_lo_w) / max(x_hi_w - x_lo_w, 1e-12) - 1.0
                    bg_avg = sum(p['bg0'] + p['bg1'] * x_norm_d for p in diag_peaks) / len(diag_peaks)
                    y_model_data = y_model_data - sum(
                        p['bg0'] + p['bg1'] * x_norm_d for p in diag_peaks) + bg_avg * len(diag_peaks)

                y_model_smooth = y_total_smooth + y_bg_smooth

                # ── Diagnostic plot ──
                fig_diag, ax_diag = plt.subplots(figsize=(12, 6))

                # Observed data: blue crosses
                ax_diag.plot(x_diag, y_diag, '+', color='#2196F3', markersize=6,
                             label='Observed', zorder=2)

                # Fitted envelope: smooth green line
                ax_diag.plot(x_smooth, y_model_smooth, '-', color='#4CAF50', linewidth=1.8,
                             label='Fit (total)', zorder=3)

                # Background: smooth red line
                ax_diag.plot(x_smooth, y_bg_smooth, '-', color='#F44336', linewidth=1.2,
                             label='Background', zorder=2)

                # Peak markers: blue dotted vertical lines at each center
                for p in diag_peaks:
                    ax_diag.axvline(p['tth_fit'], color='#2196F3', linestyle=':',
                                    alpha=0.7, linewidth=1)

                # 2theta range boundaries
                ax_diag.axvline(tth_lo, color='#4CAF50', linestyle='--',
                                alpha=0.5, linewidth=1, label='Range bounds')
                ax_diag.axvline(tth_hi, color='#F44336', linestyle='--',
                                alpha=0.5, linewidth=1)

                # Residual (offset below)
                residual = y_diag - y_model_data
                res_offset = y_diag.min() - 0.15 * (y_diag.max() - y_diag.min())
                ax_diag.plot(x_diag, residual + res_offset, '-', color='#9E9E9E',
                             linewidth=0.8, label='Residual (offset)')
                ax_diag.axhline(res_offset, color='#9E9E9E', linestyle='-',
                                alpha=0.3, linewidth=0.5)

                ax_diag.set_xlabel('2θ (deg)', fontsize=12)
                ax_diag.set_ylabel('Intensity', fontsize=12)
                ax_diag.set_title(
                    f'Diagnostic: Eta bin {eta_idx} (η={eta_val:.1f}°) — '
                    f'{len(diag_peaks)} peak(s)', fontsize=13)
                ax_diag.legend(fontsize=9, loc='upper right')
                ax_diag.grid(True, alpha=0.2)

                if args.save:
                    diag_save = args.save.rsplit('.', 1)
                    diag_fn = f"{diag_save[0]}_diag.{diag_save[1]}" if len(diag_save) > 1 else f"{args.save}_diag"
                    fig_diag.savefig(diag_fn, dpi=200, bbox_inches='tight')
                    print(f"\n  Saved diagnostic plot to {diag_fn}")

        else:
            print(f"  No peaks found in eta bin {eta_idx} within 2θ [{tth_lo}, {tth_hi}]")
    elif diag_mode:
        print(f"\n  Warning: eta bin {args.eta_bin} not found in data")

    # ── Main scatter plot — all detectors combined ──────────────────────
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