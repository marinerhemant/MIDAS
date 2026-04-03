#!/usr/bin/env python3
"""
Paper 3: Automated Calibration — Figure Generation
===================================================
Generates publication-quality figures for the automated calibration paper.

Figures:
  1. Convergence: MeanStrain vs Iteration for all 4 detectors
  2. Per-ring strain breakdown bar chart
  3. pyFAI vs MIDAS: 1D pseudo-strain comparison
  4. pyFAI vs MIDAS: geometry parameter comparison table
  5. 2D strain heatmap (MIDAS with tilts)

Usage:
    python generate_paper3_figures.py [--outdir ./figures]
"""

import argparse
import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDAS_HOME = os.path.dirname(SCRIPT_DIR)

# ── Publication style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# ── Dataset info ───────────────────────────────────────────────────────
DATASETS = {
    'pilatus': {
        'label': 'Pilatus 6M (panels)',
        'short': 'Pilatus',
        'color': '#1565C0',
        'px_um': 172,
        'convergence_csv': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'CeO2_Pil_100x100_att000_650mm_71p676keV_001956..tif.convergence_history.csv.with_panels'),
        'corr_csv': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'integrator_CeO2_Pil_100x100_att000_650mm_71p676keV_001956.corr.csv'),
        'trace_stem': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'CeO2_Pil_100x100_att000_650mm_71p676keV_001956..tif.m_step_trace_iter{}.csv.with_panels'),
    },
    'pilatus_no_panels': {
        'label': 'Pilatus 6M (no panels)',
        'short': 'Pilatus-NP',
        'color': '#64B5F6',
        'px_um': 172,
        'convergence_csv': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'CeO2_Pil_100x100_att000_650mm_71p676keV_001956..tif.convergence_history.csv.no_panels'),
        'corr_csv': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'integrator_CeO2_Pil_100x100_att000_650mm_71p676keV_001956.corr.csv.no_panels'),
        'trace_stem': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'CeO2_Pil_100x100_att000_650mm_71p676keV_001956..tif.m_step_trace_iter{}.csv.no_panels'),
    },
    'varex': {
        'label': 'Varex 4343CT',
        'short': 'Varex',
        'color': '#FF9800',
        'px_um': 150,
        'convergence_csv': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137..tif.convergence_history.csv'),
        'corr_csv': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'integrator_Ceria_63keV_900mm_100x100_0p5s_aero_0_001137.corr.csv'),
        'trace_stem': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137..tif.m_step_trace_iter{}.csv'),
    },
    'ge_offset': {
        'label': 'GE (offset)',
        'short': 'GE',
        'color': '#4CAF50',
        'px_um': 200,
        'convergence_csv': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'CeO2_1s_65pt351keV_1860mm_000007..edf.ge1.convergence_history.csv'),
        'corr_csv': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'integrator_CeO2_1s_65pt351keV_1860mm_000007.edf.corr.csv'),
        'trace_stem': os.path.join(
            MIDAS_HOME, 'FF_HEDM', 'Example', 'Calibration',
            'CeO2_1s_65pt351keV_1860mm_000007..edf.ge1.m_step_trace_iter{}.csv'),
    },
}

# pyFAI benchmark results (from benchmark_pyfai_vs_midas.py --all --strain)
PYFAI_RESULTS = {
    'pilatus': {
        'lsd_um': 657582.7, 'bc_y': 685.00, 'bc_z': 921.00,
        'rot1': 0.0147, 'rot2': 0.0148,
        'strain_1d_median': 100.3,  # µε
    },
    'varex': {
        'lsd_um': 900602.6, 'bc_y': 1446.88, 'bc_z': 1469.02,
        'rot1': 0.0093, 'rot2': -0.0093,
        'strain_1d_median': 164.1,
    },
}

MIDAS_RESULTS = {
    'pilatus': {
        'lsd_um': 657558.8, 'bc_y': 685.40, 'bc_z': 921.04,
        'ty': 0.221, 'tz': 0.527,
        'strain_1d_median': 21.6,
        'mean_strain': 21.6,
    },
    'varex': {
        'lsd_um': 895903.5, 'bc_y': 1446.97, 'bc_z': 1468.91,
        'ty': -0.302, 'tz': 0.393,
        'strain_1d_median': 5.4,
        'mean_strain': 5.4,
    },
    'ge_offset': {
        'lsd_um': 1862584.2, 'bc_y': 2275.26, 'bc_z': 2211.85,
        'ty': 0.017, 'tz': 0.282,
        'mean_strain': 5.6,
        'strain_1d_median': 5.6,
    },
}


def load_convergence(csv_path):
    """Load convergence history CSV (auto-detects CI or ACZ format)."""
    import csv
    data = {'iter': [], 'mean_strain': [], 'std_strain': [],
            'lsd': [], 'bc_y': [], 'bc_z': [], 'ty': [], 'tz': []}
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found")
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['iter'].append(int(row['Iter']))
            # Auto-detect column names (CI format vs ACZ format)
            if 'MeanStrain_ppm' in row:
                data['mean_strain'].append(float(row['MeanStrain_ppm']))
                data['std_strain'].append(float(row['StdStrain_ppm']))
            else:
                data['mean_strain'].append(float(row['MeanStrain_ue']))
                data['std_strain'].append(float(row['StdStrain_ue']))
            data['lsd'].append(float(row['Lsd']))
            if 'ybc' in row:
                data['bc_y'].append(float(row['ybc']))
                data['bc_z'].append(float(row['zbc']))
            else:
                data['bc_y'].append(float(row['BC_Y']))
                data['bc_z'].append(float(row['BC_Z']))
            data['ty'].append(float(row['ty']))
            data['tz'].append(float(row['tz']))
    return {k: np.array(v) for k, v in data.items()}


# ── Figure 1: Convergence ─────────────────────────────────────────────
def fig_convergence(outdir):
    """Convergence of MeanStrain and geometry parameters vs iteration."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    for name, ds in DATASETS.items():
        conv = load_convergence(ds['convergence_csv'])
        if conv is None:
            continue

        iters = conv['iter']
        label = ds['short']
        c = ds['color']

        # Top-left: MeanStrain
        axes[0, 0].semilogy(iters, conv['mean_strain'], '-o', ms=3,
                            color=c, label=label, linewidth=1.5)

        # Top-right: Lsd
        lsd_rel = (conv['lsd'] - conv['lsd'][-1]) / conv['lsd'][-1] * 1e6  # ppm
        axes[0, 1].plot(iters, lsd_rel, '-o', ms=3, color=c,
                        label=label, linewidth=1.5)

        # Bottom-left: BC shift from final
        bc_shift = np.sqrt((conv['bc_y'] - conv['bc_y'][-1])**2 +
                           (conv['bc_z'] - conv['bc_z'][-1])**2)
        axes[1, 0].plot(iters, bc_shift, '-o', ms=3, color=c,
                        label=label, linewidth=1.5)

        # Bottom-right: tilt change
        tilt_mag = np.sqrt(conv['ty']**2 + conv['tz']**2)
        axes[1, 1].plot(iters, tilt_mag, '-o', ms=3, color=c,
                        label=label, linewidth=1.5)

    axes[0, 0].set_ylabel('Mean Pseudo-Strain (µε)')
    axes[0, 0].set_title('(a) Calibration Convergence')
    axes[0, 0].legend(loc='upper right')

    axes[0, 1].set_ylabel('ΔLsd / Lsd (ppm)')
    axes[0, 1].set_yscale('symlog', linthresh=1)
    axes[0, 1].set_title('(b) Sample-Detector Distance')

    axes[1, 0].set_ylabel('ΔBC (pixels)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_title('(c) Beam Center Shift')

    axes[1, 1].set_ylabel('|Tilt| (°)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_title('(d) Detector Tilt Magnitude')

    plt.tight_layout()
    path = os.path.join(outdir, 'fig_convergence.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()


# ── Per-Eval Trajectory ──────────────────────────────────────────────
def load_trace_files(trace_stem, max_iter=20):
    """Load all m_step_trace_iterN.csv files and concatenate."""
    import csv
    data = {k: [] for k in ['eval', 'objective', 'lsd', 'ybc', 'zbc', 'ty', 'tz']}
    iter_boundaries = []  # cumulative eval index where each iteration starts
    cumulative = 0
    for i in range(1, max_iter + 1):
        path = trace_stem.format(i)
        if not os.path.exists(path):
            break
        with open(path) as f:
            reader = csv.DictReader(f)
            n = 0
            for row in reader:
                data['eval'].append(cumulative + int(row['Eval']))
                data['objective'].append(float(row['Objective']))
                data['lsd'].append(float(row['Lsd']))
                data['ybc'].append(float(row['ybc']))
                data['zbc'].append(float(row['zbc']))
                data['ty'].append(float(row['ty']))
                data['tz'].append(float(row['tz']))
                n += 1
        if n > 0:
            iter_boundaries.append(cumulative)
            cumulative += n
    if len(data['eval']) == 0:
        return None
    result = {k: np.array(v) for k, v in data.items()}
    result['iter_boundaries'] = iter_boundaries
    result['n_iters'] = len(iter_boundaries)
    return result


def fig_pereval_trajectory(outdir):
    """Per-evaluation parameter trajectory across all M-step iterations.

    2×2 layout: (a) Objective, (b) Lsd, (c) BC shift, (d) Tilt magnitude.
    """
    plot_datasets = ['pilatus', 'pilatus_no_panels', 'varex', 'ge_offset']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for name in plot_datasets:
        ds = DATASETS.get(name)
        if ds is None or 'trace_stem' not in ds:
            continue
        trace = load_trace_files(ds['trace_stem'])
        if trace is None:
            print(f"  WARNING: No trace files for {name}")
            continue

        c = ds['color']
        label = ds['short']
        evals = trace['eval']

        # (a) Objective function
        axes[0, 0].plot(evals, trace['objective'], '-', color=c,
                        label=label, linewidth=0.7, alpha=0.85)

        # (b) Lsd (relative change from final value in ppm)
        lsd_final = trace['lsd'][-1]
        lsd_rel = (trace['lsd'] - lsd_final) / lsd_final * 1e6
        axes[0, 1].plot(evals, lsd_rel, '-', color=c,
                        label=label, linewidth=0.7, alpha=0.85)

        # (c) BC shift from final (pixels)
        bc_final_y, bc_final_z = trace['ybc'][-1], trace['zbc'][-1]
        bc_shift = np.sqrt((trace['ybc'] - bc_final_y)**2 +
                           (trace['zbc'] - bc_final_z)**2)
        axes[1, 0].plot(evals, bc_shift, '-', color=c,
                        label=label, linewidth=0.7, alpha=0.85)

        # (d) Tilt magnitude
        tilt_mag = np.sqrt(trace['ty']**2 + trace['tz']**2)
        axes[1, 1].plot(evals, tilt_mag, '-', color=c,
                        label=label, linewidth=0.7, alpha=0.85)

        # Mark iteration boundaries
        for bi, b in enumerate(trace['iter_boundaries']):
            if b == 0:
                continue
            lw = 1.5 if bi == 5 else 0.5
            ls = '--' if bi == 5 else ':'
            for ax in axes.flat:
                ax.axvline(b, color=c, linestyle=ls, linewidth=lw, alpha=0.3)

    axes[0, 0].set_ylabel('Objective Function')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('(a) Objective Function')
    axes[0, 0].legend(loc='upper right', fontsize=7)

    axes[0, 1].set_ylabel('ΔLsd / Lsd (ppm)')
    axes[0, 1].set_yscale('symlog', linthresh=1)
    axes[0, 1].set_title('(b) Sample–Detector Distance')

    axes[1, 0].set_ylabel('ΔBC (pixels)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('Cumulative Evaluation')
    axes[1, 0].set_title('(c) Beam Center Shift')

    axes[1, 1].set_ylabel('|Tilt| (°)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Cumulative Evaluation')
    axes[1, 1].set_title('(d) Detector Tilt Magnitude')

    plt.tight_layout()
    path = os.path.join(outdir, 'fig_pereval_trajectory.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()


# ── Figure 2: pyFAI vs MIDAS pseudo-strain ─────────────────────────
def fig_pyfai_comparison(outdir):
    """Bar chart comparing 1D pseudo-strain: pyFAI vs MIDAS."""
    datasets = [d for d in ['pilatus', 'varex', 'ge_offset'] if d in PYFAI_RESULTS]
    labels = [DATASETS[d]['short'] for d in datasets]
    x = np.arange(len(datasets))
    width = 0.35

    pyfai_vals = [PYFAI_RESULTS[d]['strain_1d_median'] for d in datasets]
    midas_vals = [MIDAS_RESULTS[d]['strain_1d_median'] for d in datasets]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars_p = ax.bar(x - width/2, pyfai_vals, width, label='pyFAI (zero tilt)',
                    color='#E57373', edgecolor='#C62828', linewidth=0.8, alpha=0.85)
    bars_m = ax.bar(x + width/2, midas_vals, width, label='MIDAS (full geometry)',
                    color='#64B5F6', edgecolor='#1565C0', linewidth=0.8, alpha=0.85)

    # Add value labels
    for bar in bars_p:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#C62828')
    for bar in bars_m:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='#1565C0')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Median |Pseudo-Strain| (µε)', fontsize=11)
    ax.set_title('1D Pseudo-Strain: pyFAI vs MIDAS Automated Calibration', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, max(pyfai_vals) * 1.25)

    # Add ratio annotations
    for i, (p, m) in enumerate(zip(pyfai_vals, midas_vals)):
        ratio = p / m
        ax.annotate(f'{ratio:.1f}×', xy=(x[i], max(p, m) + 20),
                    ha='center', fontsize=8, color='gray', style='italic')

    plt.tight_layout()
    path = os.path.join(outdir, 'fig_pyfai_comparison.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()


# ── Figure 3: Geometry parameter comparison ────────────────────────
def fig_geometry_table(outdir):
    """Visual comparison of geometry parameters across detectors."""
    datasets = [d for d in ['pilatus', 'varex', 'ge_offset'] if d in PYFAI_RESULTS]
    labels = [DATASETS[d]['short'] for d in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Lsd difference
    lsd_diff = [(PYFAI_RESULTS[d]['lsd_um'] - MIDAS_RESULTS[d]['lsd_um'])
                for d in datasets]
    colors = [DATASETS[d]['color'] for d in datasets]
    bars = axes[0].barh(labels, lsd_diff, color=colors, edgecolor='black',
                        linewidth=0.5, alpha=0.8)
    axes[0].set_xlabel('ΔLsd (µm) [pyFAI − MIDAS]')
    axes[0].set_title('(a) Sample-Detector Distance')
    axes[0].axvline(0, color='black', linewidth=0.5, linestyle='-')
    for bar, val in zip(bars, lsd_diff):
        axes[0].annotate(f'{val:.0f}', xy=(val, bar.get_y() + bar.get_height()/2),
                         xytext=(5 if val > 0 else -5, 0),
                         textcoords="offset points",
                         ha='left' if val > 0 else 'right', va='center', fontsize=9)

    # BC difference
    bc_diff = [np.sqrt((PYFAI_RESULTS[d]['bc_y'] - MIDAS_RESULTS[d]['bc_y'])**2 +
                       (PYFAI_RESULTS[d]['bc_z'] - MIDAS_RESULTS[d]['bc_z'])**2)
               for d in datasets]
    bars = axes[1].barh(labels, bc_diff, color=colors, edgecolor='black',
                        linewidth=0.5, alpha=0.8)
    axes[1].set_xlabel('ΔBC (pixels)')
    axes[1].set_title('(b) Beam Center')
    for bar, val in zip(bars, bc_diff):
        axes[1].annotate(f'{val:.2f}', xy=(val, bar.get_y() + bar.get_height()/2),
                         xytext=(5, 0), textcoords="offset points",
                         ha='left', va='center', fontsize=9)

    # Pseudo-strain ratio
    ratios = [PYFAI_RESULTS[d]['strain_1d_median'] / MIDAS_RESULTS[d]['strain_1d_median']
              for d in datasets]
    bars = axes[2].barh(labels, ratios, color=colors, edgecolor='black',
                        linewidth=0.5, alpha=0.8)
    axes[2].set_xlabel('Strain Ratio (pyFAI / MIDAS)')
    axes[2].set_title('(c) Calibration Quality')
    axes[2].axvline(1, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    for bar, val in zip(bars, ratios):
        axes[2].annotate(f'{val:.1f}×', xy=(val, bar.get_y() + bar.get_height()/2),
                         xytext=(5, 0), textcoords="offset points",
                         ha='left', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(outdir, 'fig_geometry_comparison.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()


# ── Figure 4: Per-ring strain ──────────────────────────────────────
def fig_perring_strain(outdir):
    """Per-ring strain for Varex showing pyFAI vs MIDAS 1D."""
    # Varex data from benchmark output
    ring_idx = list(range(15))
    pyfai_strain = [177.4, 175.3, 169.2, 165.7, 166.1, 163.9, 163.2,
                    165.8, 163.2, 164.1, 159.5, 154.2, 167.8, 153.2, 55.0]
    midas_strain = [-57.0, -55.8, -62.2, -66.9, -66.3, -70.2, -72.1,
                    -70.6, -73.8, -73.9, -78.8, -83.3, -73.3, -84.7, -152.0]
    tth_deg = [3.59, 4.15, 5.87, 6.88, 7.19, 8.30, 9.05,
               9.28, 10.17, 10.79, 11.75, 12.29, 12.46, 13.14, 13.63]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(ring_idx))
    width = 0.35

    ax.bar(x - width/2, pyfai_strain, width, label='pyFAI',
           color='#E57373', edgecolor='#C62828', alpha=0.85, linewidth=0.5)
    ax.bar(x + width/2, midas_strain, width, label='MIDAS',
           color='#64B5F6', edgecolor='#1565C0', alpha=0.85, linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.1f}°' for t in tth_deg], rotation=45, ha='right',
                       fontsize=8)
    ax.set_xlabel('2θ (°)', fontsize=11)
    ax.set_ylabel('Pseudo-Strain (µε)', fontsize=11)
    ax.set_title('Per-Ring Pseudo-Strain: Varex CeO₂ (150µm, 63 keV)', fontsize=12)
    ax.legend(fontsize=10)

    # Annotate consistent bias
    ax.annotate('pyFAI: systematic\npositive bias', xy=(7, 170),
                fontsize=8, color='#C62828', style='italic', ha='center')
    ax.annotate('MIDAS: near-zero\n(symmetric)', xy=(7, -85),
                fontsize=8, color='#1565C0', style='italic', ha='center')

    plt.tight_layout()
    path = os.path.join(outdir, 'fig_perring_strain.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()


# ── Figure 5: Multi-detector summary ──────────────────────────────
def fig_multi_detector(outdir):
    """Summary table figure showing all detector results."""
    datasets = ['pilatus', 'varex', 'ge_offset']

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')

    headers = ['Detector', 'Pixel (µm)', 'Lsd (mm)',
               'MIDAS ε (µε)', 'pyFAI ε (µε)', 'Ratio',
               'ΔLsd (µm)', 'ΔBC (px)', 'Runtime (s)']

    rows = []
    for d in datasets:
        m = MIDAS_RESULTS[d]
        p = PYFAI_RESULTS.get(d)
        ds = DATASETS[d]
        if p is None:
            continue
        ratio = p['strain_1d_median'] / m['strain_1d_median']
        lsd_diff = abs(p['lsd_um'] - m['lsd_um'])
        bc_diff = np.sqrt((p['bc_y'] - m['bc_y'])**2 +
                          (p['bc_z'] - m['bc_z'])**2)
        rows.append([
            ds['label'], f"{ds['px_um']}", f"{m['lsd_um']/1000:.1f}",
            f"{m['strain_1d_median']:.0f}", f"{p['strain_1d_median']:.0f}",
            f"{ratio:.1f}×",
            f"{lsd_diff:.0f}", f"{bc_diff:.1f}", f"~{m.get('mean_strain', 0)*1e6:.0f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#E3F2FD']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color the ratio column
    for i in range(len(rows)):
        table[i+1, 5].set_facecolor('#E8F5E9')

    ax.set_title('Multi-Detector Calibration: MIDAS vs pyFAI Summary', fontsize=12,
                 pad=20)
    plt.tight_layout()
    path = os.path.join(outdir, 'fig_multi_detector.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()


# ── Figure 6 & 7: FitA heatmaps (detector tile boundaries) ────────
def load_corr_csv(csv_path):
    """Load a corr.csv file into a pandas DataFrame."""
    import pandas as pd
    if not csv_path or not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found")
        return None
    # Auto-detect delimiter (CI uses comma, old format uses space)
    with open(csv_path) as f:
        header = f.readline()
    delim = ',' if ',' in header else ' '
    df = pd.read_csv(csv_path, delimiter=delim)
    # Clean column names (header has %Eta)
    df.columns = [c.lstrip('%') for c in df.columns]
    return df


def _plot_fita_panel(ax, fig, df, ds, label_prefix, x_col, y_col,
                     xlabel, ylabel, xlim=None):
    """Plot a single FitA scatter panel.  All points colored by FitA;
    outliers use 'x' marker, inliers use filled circles."""
    import pandas as pd
    if df is None:
        ax.text(0.5, 0.5, f"{ds['label']}\n(data not found)",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title(f"{label_prefix} {ds['label']}")
        return

    # All points colored by FitA
    if 'FitA' not in df.columns:
        ax.text(0.5, 0.5, f"{ds['label']}\n(no FitA column)",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title(f"{label_prefix} {ds['label']}")
        return
    valid = df[df['FitA'] > 0].copy()  # skip invalid
    if len(valid) == 0:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title(f"{label_prefix} {ds['label']}")
        return

    ideal_a = valid['IdealA'].iloc[0]

    # Tight color range: ±0.05% around ideal
    a_range = ideal_a * 0.0005
    vmin = ideal_a - a_range
    vmax = ideal_a + a_range

    # Separate inliers / outliers for marker differentiation
    outlier_col = 'Outlier' if 'Outlier' in valid.columns else None
    if outlier_col:
        inliers = valid[valid[outlier_col] == 0]
        outliers = valid[valid[outlier_col] == 1]
    else:
        inliers = valid
        outliers = pd.DataFrame()

    # Plot outliers first (behind), colored by FitA but with 'x' marker
    if len(outliers) > 0:
        ax.scatter(outliers[x_col], outliers[y_col],
                   c=outliers['FitA'], cmap='jet', s=4, alpha=0.4,
                   vmin=vmin, vmax=vmax, rasterized=True,
                   marker='x', linewidths=0.3)
    # Inliers on top with filled circles
    sc = ax.scatter(inliers[x_col], inliers[y_col],
                    c=inliers['FitA'], cmap='jet', s=4, alpha=0.8,
                    vmin=vmin, vmax=vmax, rasterized=True)

    n_total = len(valid)
    n_inlier = len(inliers)
    ax.set_title(f"{label_prefix} {ds['label']}  "
                 f"({n_inlier}/{n_total} pts, a\u2080={ideal_a:.4f} \u00c5)",
                 fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)

    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('FitA (\u00c5)', fontsize=9)


def fig_fita_reta(outdir):
    """FitA vs R (horizontal) and η (vertical) for all 5 detectors."""
    order = ['varex', 'pilatus', 'pilatus_no_panels', 'ge_offset']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, name in enumerate(order):
        ds = DATASETS[name]
        df = load_corr_csv(ds.get('corr_csv', ''))
        _plot_fita_panel(axes[idx], fig, df, ds,
                         f"({chr(97+idx)})",
                         x_col='RadFit', y_col='Eta',
                         xlabel='R (µm)', ylabel='η (°)')



    fig.suptitle('Fitted Lattice Parameter — R vs η',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, 'fig_fita_r_eta.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()


def fig_fita_yz(outdir):
    """FitA in detector Y-Z space for all 5 detectors."""
    order = ['varex', 'pilatus', 'pilatus_no_panels', 'ge_offset']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, name in enumerate(order):
        ds = DATASETS[name]
        df = load_corr_csv(ds.get('corr_csv', ''))
        if df is not None and 'YRawCorr' in df.columns and 'ZRawCorr' in df.columns:
            _plot_fita_panel(axes[idx], fig, df, ds,
                             f"({chr(97+idx)})",
                             x_col='YRawCorr', y_col='ZRawCorr',
                             xlabel='Y (px)', ylabel='Z (px)')
            axes[idx].set_aspect('equal', adjustable='datalim')
        else:
            axes[idx].text(0.5, 0.5,
                           f"{DATASETS[name]['label']}\n(data not found)",
                           transform=axes[idx].transAxes,
                           ha='center', va='center',
                           fontsize=12, color='gray')
            axes[idx].set_title(f"({chr(97+idx)}) {DATASETS[name]['label']}")



    fig.suptitle('Fitted Lattice Parameter — Detector Y-Z Space',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, 'fig_fita_yz.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()



def _plot_strain_panel(ax, fig, df, ds, label_prefix):
    """Plot a single pseudo-strain scatter panel in Y-Z detector space.
    Strain = (FitA - IdealA) / IdealA × 1e6  (µε).
    Uses a symmetric diverging colormap centred on zero."""
    import pandas as pd
    if df is None:
        ax.text(0.5, 0.5, f"{ds['label']}\n(data not found)",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title(f"{label_prefix} {ds['label']}")
        return

    need_cols = {'YRawCorr', 'ZRawCorr', 'FitA', 'IdealA'}
    if not need_cols.issubset(df.columns):
        # Fall back to Strain column if present
        if 'Strain' in df.columns and 'YRawCorr' in df.columns:
            valid = df[df['FitA'] > 0].copy() if 'FitA' in df.columns else df.copy()
            strain = valid['Strain'].values * 1e6  # ppm → µε
        else:
            ax.text(0.5, 0.5, f"{ds['label']}\n(missing columns)",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            ax.set_title(f"{label_prefix} {ds['label']}")
            return
    else:
        valid = df[df['FitA'] > 0].copy()
        if len(valid) == 0:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(f"{label_prefix} {ds['label']}")
            return
        strain = (valid['FitA'].values - valid['IdealA'].values) / valid['IdealA'].values * 1e6

    # Separate inliers / outliers
    outlier_col = 'Outlier' if 'Outlier' in valid.columns else None
    if outlier_col:
        mask_in = valid[outlier_col].values == 0
    else:
        mask_in = np.ones(len(valid), dtype=bool)

    # Symmetric color range: ±3σ clipped to ±max(150, 3σ)
    abs_lim = max(150, np.nanstd(strain[mask_in]) * 3)
    vmin, vmax = -abs_lim, abs_lim

    # Outliers behind (×-marker)
    if (~mask_in).any():
        ax.scatter(valid['YRawCorr'].values[~mask_in],
                   valid['ZRawCorr'].values[~mask_in],
                   c=strain[~mask_in], cmap='RdBu_r', s=4, alpha=0.3,
                   vmin=vmin, vmax=vmax, rasterized=True,
                   marker='x', linewidths=0.3)
    # Inliers on top
    sc = ax.scatter(valid['YRawCorr'].values[mask_in],
                    valid['ZRawCorr'].values[mask_in],
                    c=strain[mask_in], cmap='RdBu_r', s=4, alpha=0.8,
                    vmin=vmin, vmax=vmax, rasterized=True)

    n_inlier = int(mask_in.sum())
    n_total = len(valid)
    med = np.nanmedian(strain[mask_in])
    ax.set_title(f"{label_prefix} {ds['label']}  "
                 f"({n_inlier}/{n_total} pts, med={med:.1f} µε)",
                 fontsize=10)
    ax.set_xlabel('Y (px)')
    ax.set_ylabel('Z (px)')
    ax.set_aspect('equal', adjustable='datalim')

    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label('Pseudo-Strain (µε)', fontsize=9)


def fig_strain_yz(outdir):
    """Pseudo-strain in detector Y-Z space for all 4 detectors."""
    order = ['varex', 'pilatus', 'pilatus_no_panels', 'ge_offset']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, name in enumerate(order):
        ds = DATASETS[name]
        df = load_corr_csv(ds.get('corr_csv', ''))
        _plot_strain_panel(axes[idx], fig, df, ds, f"({chr(97+idx)})")

    fig.suptitle('Pseudo-Strain — Detector Y-Z Space',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, 'fig_strain_yz.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"  Saved {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate Paper 3 figures')
    parser.add_argument('--outdir', default=os.path.expanduser('~/Documents/3Papers/paper3_calibration'),
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("  Paper 3: Generating Figures")
    print("=" * 60)

    print("\n[1/8] Convergence plots...")
    fig_convergence(args.outdir)

    print("\n[2/8] Per-eval trajectory...")
    fig_pereval_trajectory(args.outdir)

    print("\n[3/8] pyFAI vs MIDAS comparison...")
    fig_pyfai_comparison(args.outdir)

    print("\n[4/8] Geometry comparison...")
    fig_geometry_table(args.outdir)

    print("\n[5/8] FitA R-η (tile boundaries)...")
    fig_fita_reta(args.outdir)

    print("\n[6/8] FitA Y-Z (detector space)...")
    fig_fita_yz(args.outdir)

    print("\n[7/8] Strain Y-Z (detector space)...")
    fig_strain_yz(args.outdir)

    print("\n[8/8] Multi-detector summary...")
    fig_multi_detector(args.outdir)

    print(f"\n  All figures saved to {args.outdir}")


if __name__ == '__main__':
    main()


