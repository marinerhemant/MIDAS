#!/usr/bin/env python3
"""
Compute per-layer SigmaR and SigmaEta statistics from FF-HEDM peak files.

Scans LayerNr_*/Temp/*_PS.csv files, computes statistics for each layer,
saves distribution histograms, and plots statistics vs layer number.

Usage:
    python peak_sigma_statistics.py /path/to/results
    python peak_sigma_statistics.py /path/to/results --out sigma_report
"""

import argparse
import glob
import json
import os
import warnings
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — plots will be skipped.")


def load_ps_files(temp_dir):
    """Load all _PS.csv files in a Temp directory, return concatenated arrays."""
    ps_files = sorted(glob.glob(os.path.join(temp_dir, '*_PS.csv')))
    if not ps_files:
        return None

    all_rows = []
    for pf in ps_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                data = np.genfromtxt(pf, skip_header=1)
            if data.size == 0:
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            all_rows.append(data)
        except Exception:
            continue

    if not all_rows:
        return None
    return np.vstack(all_rows)


def compute_stats(values):
    """Compute comprehensive statistics for an array of values."""
    if len(values) == 0:
        return {}
    return {
        'count': int(len(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q75': float(np.percentile(values, 75)),
        'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
        'skewness': float(skewness(values)),
        'kurtosis': float(kurtosis(values)),
    }


def skewness(x):
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-15:
        return 0.0
    return np.mean(((x - m) / s) ** 3)


def kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-15:
        return 0.0
    return np.mean(((x - m) / s) ** 4) - 3.0


def main():
    parser = argparse.ArgumentParser(
        description='Compute SigmaR/SigmaEta statistics from FF-HEDM peak files.')
    parser.add_argument('results_dir', help='Directory containing LayerNr_* folders')
    parser.add_argument('--out', default='sigma_statistics',
                        help='Output directory for plots and JSON (default: sigma_statistics)')
    parser.add_argument('--nbins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_dir = os.path.join(results_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Discover LayerNr_* directories
    layer_dirs = sorted(glob.glob(os.path.join(results_dir, 'LayerNr_*')))
    if not layer_dirs:
        print(f"No LayerNr_* directories found in {results_dir}")
        sys.exit(1)

    # Column indices (0-based) from the header
    COL_SIGMA_R = 8      # SigmaR
    COL_SIGMA_ETA = 9    # SigmaEta

    all_results = {}
    layer_nrs = []
    sigmaR_means = []
    sigmaR_medians = []
    sigmaR_stds = []
    sigmaEta_means = []
    sigmaEta_medians = []
    sigmaEta_stds = []
    peak_counts = []

    print(f"Scanning {len(layer_dirs)} LayerNr directories...")
    print(f"{'Layer':>6}  {'Peaks':>7}  {'SigR_mean':>10}  {'SigR_med':>10}  "
          f"{'SigE_mean':>10}  {'SigE_med':>10}")
    print("-" * 72)

    for ld in layer_dirs:
        # Extract layer number
        base = os.path.basename(ld)
        try:
            layer_nr = int(base.split('_')[-1])
        except ValueError:
            continue

        temp_dir = os.path.join(ld, 'Temp')
        if not os.path.isdir(temp_dir):
            continue

        data = load_ps_files(temp_dir)
        if data is None or data.shape[0] == 0:
            continue
        if data.shape[1] <= max(COL_SIGMA_R, COL_SIGMA_ETA):
            continue

        sigmaR = data[:, COL_SIGMA_R]
        sigmaEta = data[:, COL_SIGMA_ETA]

        # Filter out invalid values
        valid = (sigmaR > 0) & (sigmaEta > 0) & np.isfinite(sigmaR) & np.isfinite(sigmaEta)
        sigmaR = sigmaR[valid]
        sigmaEta = sigmaEta[valid]

        if len(sigmaR) == 0:
            continue

        sr_stats = compute_stats(sigmaR)
        se_stats = compute_stats(sigmaEta)

        all_results[layer_nr] = {
            'SigmaR': sr_stats,
            'SigmaEta': se_stats,
        }

        layer_nrs.append(layer_nr)
        sigmaR_means.append(sr_stats['mean'])
        sigmaR_medians.append(sr_stats['median'])
        sigmaR_stds.append(sr_stats['std'])
        sigmaEta_means.append(se_stats['mean'])
        sigmaEta_medians.append(se_stats['median'])
        sigmaEta_stds.append(se_stats['std'])
        peak_counts.append(sr_stats['count'])

        print(f"{layer_nr:>6}  {sr_stats['count']:>7}  {sr_stats['mean']:>10.4f}  "
              f"{sr_stats['median']:>10.4f}  {se_stats['mean']:>10.4f}  {se_stats['median']:>10.4f}")

        # Save per-layer histogram
        if HAS_MPL:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(sigmaR, bins=args.nbins, color='steelblue', edgecolor='white', alpha=0.8)
            axes[0].axvline(sr_stats['mean'], color='red', ls='--', label=f"mean={sr_stats['mean']:.3f}")
            axes[0].axvline(sr_stats['median'], color='orange', ls='--', label=f"med={sr_stats['median']:.3f}")
            axes[0].set_xlabel('SigmaR (px)')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'Layer {layer_nr} — SigmaR ({sr_stats["count"]} peaks)')
            axes[0].legend(fontsize=8)

            axes[1].hist(sigmaEta, bins=args.nbins, color='coral', edgecolor='white', alpha=0.8)
            axes[1].axvline(se_stats['mean'], color='red', ls='--', label=f"mean={se_stats['mean']:.3f}")
            axes[1].axvline(se_stats['median'], color='orange', ls='--', label=f"med={se_stats['median']:.3f}")
            axes[1].set_xlabel('SigmaEta (deg)')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'Layer {layer_nr} — SigmaEta ({se_stats["count"]} peaks)')
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'hist_layer_{layer_nr:04d}.png'), dpi=150)
            plt.close(fig)

    if not layer_nrs:
        print("No valid peak data found in any layer.")
        sys.exit(1)

    # Save JSON summary
    json_path = os.path.join(out_dir, 'sigma_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    print(f"\nStatistics saved to {json_path}")

    # Save CSV summary
    csv_path = os.path.join(out_dir, 'sigma_statistics.csv')
    with open(csv_path, 'w') as f:
        f.write("LayerNr,NrPeaks,"
                "SigmaR_mean,SigmaR_median,SigmaR_std,SigmaR_min,SigmaR_max,SigmaR_q25,SigmaR_q75,"
                "SigmaEta_mean,SigmaEta_median,SigmaEta_std,SigmaEta_min,SigmaEta_max,SigmaEta_q25,SigmaEta_q75\n")
        for i, lnr in enumerate(layer_nrs):
            sr = all_results[lnr]['SigmaR']
            se = all_results[lnr]['SigmaEta']
            f.write(f"{lnr},{peak_counts[i]},"
                    f"{sr['mean']:.6f},{sr['median']:.6f},{sr['std']:.6f},"
                    f"{sr['min']:.6f},{sr['max']:.6f},{sr['q25']:.6f},{sr['q75']:.6f},"
                    f"{se['mean']:.6f},{se['median']:.6f},{se['std']:.6f},"
                    f"{se['min']:.6f},{se['max']:.6f},{se['q25']:.6f},{se['q75']:.6f}\n")
    print(f"CSV saved to {csv_path}")

    # Plot statistics vs layer number
    if HAS_MPL:
        layer_arr = np.array(layer_nrs)

        fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
        fig.suptitle('Peak Shape Statistics vs Layer Number', fontsize=14, fontweight='bold')

        # Row 1: Mean with std band
        axes[0, 0].fill_between(layer_arr,
                                np.array(sigmaR_means) - np.array(sigmaR_stds),
                                np.array(sigmaR_means) + np.array(sigmaR_stds),
                                alpha=0.2, color='steelblue')
        axes[0, 0].plot(layer_arr, sigmaR_means, 'o-', ms=3, color='steelblue', label='Mean')
        axes[0, 0].plot(layer_arr, sigmaR_medians, 's-', ms=3, color='coral', label='Median')
        axes[0, 0].set_ylabel('SigmaR (px)')
        axes[0, 0].set_title('SigmaR — Mean & Median')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].fill_between(layer_arr,
                                np.array(sigmaEta_means) - np.array(sigmaEta_stds),
                                np.array(sigmaEta_means) + np.array(sigmaEta_stds),
                                alpha=0.2, color='coral')
        axes[0, 1].plot(layer_arr, sigmaEta_means, 'o-', ms=3, color='coral', label='Mean')
        axes[0, 1].plot(layer_arr, sigmaEta_medians, 's-', ms=3, color='steelblue', label='Median')
        axes[0, 1].set_ylabel('SigmaEta (deg)')
        axes[0, 1].set_title('SigmaEta — Mean & Median')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Row 2: Quartiles (IQR)
        sr_q25 = [all_results[lnr]['SigmaR']['q25'] for lnr in layer_nrs]
        sr_q75 = [all_results[lnr]['SigmaR']['q75'] for lnr in layer_nrs]
        se_q25 = [all_results[lnr]['SigmaEta']['q25'] for lnr in layer_nrs]
        se_q75 = [all_results[lnr]['SigmaEta']['q75'] for lnr in layer_nrs]

        axes[1, 0].fill_between(layer_arr, sr_q25, sr_q75, alpha=0.3, color='steelblue', label='IQR')
        axes[1, 0].plot(layer_arr, sigmaR_medians, '-', color='steelblue', lw=1.5, label='Median')
        axes[1, 0].set_ylabel('SigmaR (px)')
        axes[1, 0].set_title('SigmaR — IQR range')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].fill_between(layer_arr, se_q25, se_q75, alpha=0.3, color='coral', label='IQR')
        axes[1, 1].plot(layer_arr, sigmaEta_medians, '-', color='coral', lw=1.5, label='Median')
        axes[1, 1].set_ylabel('SigmaEta (deg)')
        axes[1, 1].set_title('SigmaEta — IQR range')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        # Row 3: Peak count + std
        axes[2, 0].bar(layer_arr, peak_counts, color='steelblue', alpha=0.7, width=0.8)
        axes[2, 0].set_xlabel('Layer Number')
        axes[2, 0].set_ylabel('Number of Peaks')
        axes[2, 0].set_title('Peak Count per Layer')
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(layer_arr, sigmaR_stds, 'o-', ms=3, color='steelblue', label='SigmaR std')
        axes[2, 1].plot(layer_arr, sigmaEta_stds, 's-', ms=3, color='coral', label='SigmaEta std')
        axes[2, 1].set_xlabel('Layer Number')
        axes[2, 1].set_ylabel('Std Dev')
        axes[2, 1].set_title('Standard Deviation per Layer')
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        summary_path = os.path.join(out_dir, 'sigma_vs_layer.png')
        plt.savefig(summary_path, dpi=200)
        plt.close(fig)
        print(f"Summary plot saved to {summary_path}")

        # Per-layer histograms are already saved above
        print(f"Per-layer histograms saved to {out_dir}/hist_layer_*.png")

    print(f"\nProcessed {len(layer_nrs)} layers, {sum(peak_counts)} total peaks.")


if __name__ == '__main__':
    main()
