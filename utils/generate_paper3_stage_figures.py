#!/usr/bin/env python3
"""Generate stage-by-stage diagnostic figures for Paper 3 (Calibration).

Produces:
1. fig_distortion_maps.pdf — 3-panel (Y,Z) total distortion map per stage
2. fig_strain_vs_eta_stages.pdf — Strain vs Eta scatter per stage, color by RingNr
3. fig_stage_comparison.pdf — Stage 2 vs Stage 4 strain scatter (Y,Z)

Usage:
    python generate_paper3_stage_figures.py [dataset_stem]

If no stem given, uses Ceria_63keV_900mm_100x100_0p5s_aero_0_001137
"""

import sys
import os
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


PAPER_DIR = '/Users/hsharma/Documents/3Papers/paper3_calibration'
CAL_DIR = '/Users/hsharma/opt/MIDAS/FF_HEDM/Example/Calibration'


def load_corr_csv(fpath, include_outliers=False):
    """Load a corr.csv file, return dict of arrays."""
    with open(fpath) as f:
        lines = f.readlines()
    data_lines = [l.strip() for l in lines[4:] if l.strip() and not l.startswith('%')]
    if len(data_lines) < 10:
        return None
    cols = np.array([[float(x) for x in l.split()] for l in data_lines])
    if include_outliers:
        mask = np.ones(len(cols), dtype=bool)
    else:
        mask = cols[:, 7] == 0
    result = {
        'eta': cols[mask, 0],
        'strain_ue': cols[mask, 1] * 1e6,
        'y': cols[mask, 8],
        'z': cols[mask, 9],
        'ring_nr': cols[mask, 10],
        'outlier': cols[mask, 7],
    }
    if cols.shape[1] > 16:
        result['delta_r'] = cols[mask, 16]
    return result


def fig_strain_vs_eta(stem, output_dir=PAPER_DIR):
    """Generate Strain vs Eta scatter plot for each stage."""
    stages = [
        ('stage1_geometry', 'Stage 1: Geometry only'),
        ('stage2_distortion', 'Stage 2: Full distortion'),
        ('stage4_evaluate', 'Stage 4: Distortion + spline'),
    ]

    fig, axes = plt.subplots(len(stages), 1, figsize=(12, 4 * len(stages)),
                              sharex=True)
    if len(stages) == 1:
        axes = [axes]

    cmap = plt.cm.tab10
    found_any = False

    for ax, (stage_name, title) in zip(axes, stages):
        fpath = os.path.join(CAL_DIR, f"{stem}_{stage_name}.corr.csv")
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, f'File not found:\n{os.path.basename(fpath)}',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(title)
            continue

        data = load_corr_csv(fpath, include_outliers=True)
        if data is None:
            continue
        found_any = True

        rings = sorted(np.unique(data['ring_nr']))
        for rn in rings:
            m = data['ring_nr'] == rn
            is_outlier = data['outlier'][m] != 0
            color_idx = int(rn - 1) % 10
            # Plot non-outliers
            m_good = m & (data['outlier'] == 0)
            if m_good.any():
                ax.scatter(data['eta'][m_good], data['strain_ue'][m_good], s=2, alpha=0.5,
                           c=[cmap(color_idx)], label=f'{int(rn)}' if rn <= 13 else None)
            # Plot outliers as hollow markers
            m_bad = m & (data['outlier'] != 0)
            if m_bad.any():
                ax.scatter(data['eta'][m_bad], data['strain_ue'][m_bad], s=8, alpha=0.3,
                           facecolors='none', edgecolors=[cmap(color_idx)], linewidths=0.5)

        non_outlier = data['outlier'] == 0
        mean_strain = np.mean(np.abs(data['strain_ue'][non_outlier])) if non_outlier.any() else 0
        ax.set_title(f'{title}  (Mean |strain| = {mean_strain:.1f} µε)', fontsize=12)
        ax.set_ylabel('Strain (µε)', fontsize=11)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_ylim(-80, 80)

    axes[-1].set_xlabel('η (degrees)', fontsize=11)
    axes[0].legend(title='Ring', ncol=7, fontsize=7, loc='upper right',
                   markerscale=3)

    det_short = DETECTOR_LABELS.get(stem, stem).split('(')[0].strip().replace(' ', '_')
    plt.suptitle(f'Strain vs azimuthal angle — {DETECTOR_LABELS.get(stem, stem)}',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, f'fig_strain_vs_eta_{det_short}.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if found_any:
        print(f"Saved: {out}")
    return found_any


def fig_stage_comparison(stem, output_dir=PAPER_DIR):
    """Generate Stage 2 vs Stage 4 strain scatter in (Y,Z) space."""
    stages = [
        ('stage2_distortion', 'Stage 2: Analytical only'),
        ('stage4_evaluate', 'Stage 4: Analytical + spline'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vmax = 30  # µε

    for ax, (stage_name, title) in zip(axes, stages):
        fpath = os.path.join(CAL_DIR, f"{stem}_{stage_name}.corr.csv")
        if not os.path.exists(fpath):
            ax.text(0.5, 0.5, f'Not found', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_title(title)
            continue

        data = load_corr_csv(fpath, include_outliers=True)
        if data is None:
            continue

        # Plot all points, outliers as X markers
        non_outlier = data['outlier'] == 0
        sc = ax.scatter(data['y'][non_outlier], data['z'][non_outlier],
                        c=data['strain_ue'][non_outlier], s=3,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.7)
        outlier_m = data['outlier'] != 0
        if outlier_m.any():
            ax.scatter(data['y'][outlier_m], data['z'][outlier_m],
                       c=data['strain_ue'][outlier_m], s=12, marker='x',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.4, linewidths=0.5)
        ax.set_xlabel('Y (pixels)', fontsize=11)
        ax.set_ylabel('Z (pixels)', fontsize=11)
        mean_s = np.mean(np.abs(data['strain_ue'][non_outlier])) if non_outlier.any() else 0
        ax.set_title(f'{title}\nMean |strain| = {mean_s:.1f} µε', fontsize=12)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='Strain (µε)', shrink=0.8)

    det_short = DETECTOR_LABELS.get(stem, stem).split('(')[0].strip().replace(' ', '_')
    plt.suptitle(f'Spline improvement — {DETECTOR_LABELS.get(stem, stem)}', fontsize=14)
    plt.tight_layout()
    out = os.path.join(output_dir, f'fig_stage_comparison_{det_short}.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


DETECTOR_LABELS = {
    'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137': 'Varex 4343CT (63 keV, 900 mm)',
    'CeO2_Pil_100x100_att000_650mm_71p676keV_001956': 'Pilatus 6M (72 keV, 650 mm)',
    'CeO2_1s_65pt351keV_1860mm_000007.edf': 'GE Offset (65 keV, 1860 mm)',
}


def fig_distortion_maps(stems, output_dir=PAPER_DIR):
    """Generate 2-panel (Stage 2, Stage 4) distortion maps for multiple detectors."""
    stages = [
        ('stage2_distortion_map', 'Analytical model'),
        ('stage4_evaluate_map', 'Analytical + spline'),
    ]

    n_det = len(stems)
    fig, axes = plt.subplots(n_det, 2, figsize=(14, 6 * n_det))
    if n_det == 1:
        axes = axes.reshape(1, -1)

    for row, stem in enumerate(stems):
        det_label = DETECTOR_LABELS.get(stem, stem)
        for col, (stage_name, stage_title) in enumerate(stages):
            ax = axes[row, col]
            fpath = os.path.join(CAL_DIR, f"{stem}_{stage_name}.png")
            if os.path.exists(fpath):
                img = plt.imread(fpath)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'Not found', transform=ax.transAxes,
                        ha='center', va='center')
            ax.set_title(f'{det_label}\n{stage_title}', fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig_distortion_maps.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    # Datasets used in the paper (exclude bad-distortion Varex)
    PAPER_STEMS = [
        'Ceria_63keV_900mm_100x100_0p5s_aero_0_001137',
        'CeO2_Pil_100x100_att000_650mm_71p676keV_001956',
    ]

    if len(sys.argv) > 1:
        stems = sys.argv[1:]
    else:
        stems = PAPER_STEMS

    os.makedirs(PAPER_DIR, exist_ok=True)

    # Per-detector figures (strain vs eta, stage comparison)
    for stem in stems:
        det_label = DETECTOR_LABELS.get(stem, stem)
        print(f"Generating stage figures for: {det_label}")
        fig_strain_vs_eta(stem)
        fig_stage_comparison(stem)
        print()

    # Multi-detector distortion maps (2 panels per detector)
    fig_distortion_maps(stems)

    # Copy spline-only maps
    import shutil
    for stem in stems:
        spline_src = os.path.join(CAL_DIR, f"{stem}_spline_only_map.png")
        if os.path.exists(spline_src):
            det_short = DETECTOR_LABELS.get(stem, stem).split('(')[0].strip().replace(' ', '_')
            spline_dst = os.path.join(PAPER_DIR, f'fig_spline_map_{det_short}.png')
            shutil.copy2(spline_src, spline_dst)
            print(f"Copied spline map: {spline_dst}")
