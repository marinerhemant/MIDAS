#!/usr/bin/env python3
"""Generate Fourier harmonic analysis figure for Paper 3 (Calibration).

Decomposes calibrant ΔR residuals into azimuthal harmonics k=1..8 per ring,
showing which harmonics the analytical distortion model captures and which
remain as unmodeled residuals.

Usage:
    python generate_paper3_fourier_analysis.py [corr.csv files...]

If no files given, searches for *stage2_distortion.corr.csv in cwd.
Output: fig_fourier_harmonics.pdf in the same directory as the first input file.
"""

import sys
import os
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def load_corr_csv(fpath):
    """Load a CalibrantIntegratorOMP corr.csv file."""
    with open(fpath) as f:
        lines = f.readlines()
    data_lines = [l.strip() for l in lines[4:] if l.strip() and not l.startswith('%')]
    if len(data_lines) < 10:
        return None
    cols = np.array([[float(x) for x in l.split()] for l in data_lines])
    mask = cols[:, 7] == 0  # non-outlier
    return {
        'eta': cols[mask, 0],
        'strain': cols[mask, 1],
        'delta_r': cols[mask, 16] if cols.shape[1] > 16 else cols[mask, 2] - cols[mask, 12],
        'ring_nr': cols[mask, 10],
        'ideal_r': cols[mask, 12],
    }


def fourier_decompose(eta_deg, values, n_harmonics=8):
    """Decompose values into azimuthal Fourier harmonics.

    Returns array of amplitudes for k=1..n_harmonics.
    """
    eta_rad = np.radians(eta_deg)
    A = np.ones((len(eta_rad), 2 * n_harmonics + 1))
    for k in range(1, n_harmonics + 1):
        A[:, 2*k - 1] = np.cos(k * eta_rad)
        A[:, 2*k] = np.sin(k * eta_rad)
    coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    amps = np.zeros(n_harmonics)
    for k in range(1, n_harmonics + 1):
        amps[k-1] = np.sqrt(coeffs[2*k-1]**2 + coeffs[2*k]**2)
    return amps


def generate_fourier_figure(corr_files, output_dir=None):
    """Generate the Fourier harmonic analysis figure."""
    n_harmonics = 8

    # Model coverage: which harmonics have analytical terms
    model_harmonics = {
        1: r'$p_7 \rho^4 \cos(\eta\prime + p_8)$',
        2: r'$p_0 \rho^2 \cos(2\eta\prime + p_6)$',
        3: r'$p_9 \rho^3 \cos(3\eta\prime + p_{10})$',
        4: r'$p_1 \rho^4 \cos(4\eta\prime + p_3)$',
        5: r'$p_{11} \rho^5 \cos(5\eta\prime + p_{12})$',
        6: r'$p_{13} \rho^6 \cos(6\eta\prime + p_{14})$',
    }

    for fpath in corr_files:
        data = load_corr_csv(fpath)
        if data is None:
            print(f"Skipping {fpath}: insufficient data")
            continue

        rings = sorted(np.unique(data['ring_nr']))
        n_rings = len(rings)

        # Filter to rings with sufficient data (>= 100 points for reliable Fourier)
        rings = [rn for rn in rings if np.sum(data['ring_nr'] == rn) >= 100]
        n_rings = len(rings)
        if n_rings < 3:
            print(f"Skipping {fpath}: only {n_rings} rings with >= 100 points")
            continue

        # Build amplitude matrix: [n_rings x n_harmonics]
        amp_matrix = np.zeros((n_rings, n_harmonics))
        for i, rn in enumerate(rings):
            m = data['ring_nr'] == rn
            # Convert deltaR to strain-equivalent for comparability across rings
            amps_um = fourier_decompose(data['eta'][m], data['delta_r'][m], n_harmonics)
            mean_ideal_r = np.mean(data['ideal_r'][m])
            amp_matrix[i, :] = (amps_um / mean_ideal_r) * 1e6  # microstrain

        # Cap extreme values for display
        amp_display = np.clip(amp_matrix, 0, np.percentile(amp_matrix, 98))

        # --- Figure: Heatmap ---
        fig, ax = plt.subplots(1, 1, figsize=(10, max(4, n_rings * 0.35 + 1.5)))

        vmax_display = max(np.percentile(amp_display[amp_display > 0], 95), 1.0) if np.any(amp_display > 0) else 10.0
        im = ax.imshow(amp_display, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest', origin='lower',
                       vmin=0, vmax=vmax_display)

        # Axes
        ax.set_xticks(range(n_harmonics))
        ax.set_xticklabels([f'k={k+1}' for k in range(n_harmonics)], fontsize=11)
        ax.set_yticks(range(n_rings))
        ax.set_yticklabels([f'{int(r)}' for r in rings], fontsize=9)
        ax.set_xlabel('Azimuthal harmonic order', fontsize=12)
        ax.set_ylabel('Ring number', fontsize=12)

        # Mark in-model vs missing harmonics
        for k in range(n_harmonics):
            harmonic = k + 1
            if harmonic in model_harmonics:
                ax.axvline(k, color='green', linewidth=0.5, alpha=0.3)
                ax.text(k, -0.7, '✓', ha='center', va='top', fontsize=10,
                        color='green', fontweight='bold')
            else:
                ax.text(k, -0.7, '✗', ha='center', va='top', fontsize=10,
                        color='red', fontweight='bold')

        # Annotate cells with values
        for i in range(n_rings):
            for j in range(n_harmonics):
                val = amp_matrix[i, j]
                if val > 0.5:
                    color = 'white' if val > np.percentile(amp_matrix[amp_matrix > 0], 70) else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label='Fourier amplitude (µε)', shrink=0.8)

        # Title
        basename = os.path.basename(fpath).replace('_stage2_distortion.corr.csv', '')
        ax.set_title(f'Azimuthal Fourier decomposition of calibrant residuals\n{basename}',
                     fontsize=13)

        # Legend
        in_model = mpatches.Patch(color='green', alpha=0.3, label='In analytical model (p0–p14)')
        not_in = mpatches.Patch(color='red', alpha=0.3, label='Not in model')
        ax.legend(handles=[in_model, not_in], loc='upper right', fontsize=9)

        plt.tight_layout()
        if output_dir is None:
            output_dir = os.path.dirname(fpath) or '.'
        out_path = os.path.join(output_dir, f'fig_fourier_harmonics_{basename}.pdf')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

        # Also generate the combined strain-equivalent summary
        print(f"\nStrain-equivalent Fourier amplitudes (µε) — averaged across rings:")
        print(f"{'k':>3s}  {'amp(µε)':>8s}  {'in model?':>20s}")
        for k in range(n_harmonics):
            mean_amp = np.mean(amp_matrix[:, k])
            status = model_harmonics.get(k+1, 'NOT IN MODEL')
            print(f'{k+1:3d}  {mean_amp:8.2f}  {status}')

        # Print LaTeX table
        print(f"\n% LaTeX table for {basename}")
        print(r"\begin{tabular}{" + "c" * (n_harmonics + 1) + "}")
        print(r"\toprule")
        header = "Ring & " + " & ".join([f"$k={k+1}$" for k in range(n_harmonics)]) + r" \\"
        print(header)
        print(r"\midrule")
        for i, rn in enumerate(rings):
            row = f"{int(rn)}"
            for j in range(n_harmonics):
                val = amp_matrix[i, j]
                row += f" & {val:.1f}"
            row += r" \\"
            print(row)
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob('*stage2_distortion.corr.csv'))
        if not files:
            files = sorted(glob.glob('*.corr.csv'))
            files = [f for f in files if 'integrator_' not in f and 'stage' not in f]

    if not files:
        print("No corr.csv files found. Run calibration first.", file=sys.stderr)
        sys.exit(1)

    # Output to paper directory
    paper_dir = '/Users/hsharma/Documents/3Papers/paper3_calibration'
    if os.path.isdir(paper_dir):
        generate_fourier_figure(files, output_dir=paper_dir)
    else:
        generate_fourier_figure(files)
