#!/usr/bin/env python
"""
Generate a 3-panel figure comparing iterative vs all-at-once refinement.

Reads Grains_iterative.csv, Grains_allatonce.csv, and GrainsSim.csv from
the FF_HEDM/Example directory.  Uses utils/match_grains.py for proper
Hungarian matching with symmetry-aware misorientation.

Usage:
    python tests/plot_fit_comparison.py [--dir FF_HEDM/Example]
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from match_grains import (
    load_grains, aggregate_grains, match_grains,
    _AGG_POS, _AGG_ORIENT, _AGG_QUAT,
)
from calcMiso import MakeSymmetries, rad2deg
from match_grains import _misorientation_angle


def compute_metrics(gt_path, fit_path, sg_num=225):
    """Match grains and compute per-grain errors using Hungarian matching."""
    gt_agg = aggregate_grains([str(gt_path)])
    fit_agg = aggregate_grains([str(fit_path)])

    result = match_grains(
        gt_agg, fit_agg, sg_nr=sg_num,
        mode='combined', weights=(2.0, 50.0),
        remove_duplicates=True,
    )

    # Extract per-match metrics
    gt_raw = load_grains(str(gt_path))
    fit_raw = load_grains(str(fit_path))

    pos_errs = []
    orient_errs = []
    strain_errs = []

    for i1, i2, cost, angle_deg, dist_um in result['matches']:
        pos_errs.append(dist_um)
        orient_errs.append(angle_deg)
        # Lattice params: columns 13-18 in raw Grains.csv (a, b, c, alpha, beta, gamma)
        # For cubic symmetry, a/b/c are interchangeable — try all 6 permutations
        gt_lat = gt_raw[i1, 13:16]    # a, b, c
        fit_lat = fit_raw[i2, 13:16]
        from itertools import permutations
        best_err = min(
            np.mean(np.abs(fit_lat[list(p)] - gt_lat) / gt_lat) * 100
            for p in permutations([0, 1, 2])
        )
        strain_errs.append(best_err)

    n_unmatched_gt = len(result['unmatched_state1'])
    n_unmatched_fit = len(result['unmatched_state2'])

    return (np.array(pos_errs), np.array(orient_errs), np.array(strain_errs),
            n_unmatched_gt, n_unmatched_fit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "FF_HEDM" / "Example"))
    parser.add_argument("--output", type=str, default="refinement_comparison.pdf")
    parser.add_argument("--sg", type=int, default=225)
    args = parser.parse_args()

    work_dir = Path(args.dir)
    gt_path = work_dir / "GrainsSim.csv"
    it_path = work_dir / "Grains_iterative.csv"
    aa_path = work_dir / "Grains_allatonce.csv"

    for p in [gt_path, it_path, aa_path]:
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    print("Computing iterative metrics...")
    pos_it, ori_it, str_it, un_gt_it, un_fit_it = compute_metrics(gt_path, it_path, args.sg)
    print("Computing all-at-once metrics...")
    pos_aa, ori_aa, str_aa, un_gt_aa, un_fit_aa = compute_metrics(gt_path, aa_path, args.sg)

    n_it = len(pos_it)
    n_aa = len(pos_aa)
    print(f"\nMatched: iterative={n_it} (unmatched GT={un_gt_it}, fit={un_fit_it})")
    print(f"         all-at-once={n_aa} (unmatched GT={un_gt_aa}, fit={un_fit_aa})")
    print(f"Iterative  - pos: {np.mean(pos_it):.2f}+/-{np.std(pos_it):.2f} um, "
          f"orient: {np.mean(ori_it):.4f}+/-{np.std(ori_it):.4f} deg, "
          f"strain: {np.mean(str_it):.4f}+/-{np.std(str_it):.4f}%")
    print(f"All-at-once- pos: {np.mean(pos_aa):.2f}+/-{np.std(pos_aa):.2f} um, "
          f"orient: {np.mean(ori_aa):.4f}+/-{np.std(ori_aa):.4f} deg, "
          f"strain: {np.mean(str_aa):.4f}+/-{np.std(str_aa):.4f}%")

    # Filter misindexed grains
    good_it = (ori_it < 5.0) & (pos_it < 100)
    good_aa = (ori_aa < 5.0) & (pos_aa < 100)
    # Use intersection: only grains that are good in both
    n_min = min(len(good_it), len(good_aa))
    good = good_it[:n_min] & good_aa[:n_min]
    pos_it, ori_it, str_it = pos_it[:n_min][good], ori_it[:n_min][good], str_it[:n_min][good]
    pos_aa, ori_aa, str_aa = pos_aa[:n_min][good], ori_aa[:n_min][good], str_aa[:n_min][good]
    n_filt = n_min - np.sum(good)
    n_good = np.sum(good)
    if n_filt > 0:
        print(f"\nFiltered {n_filt} misindexed grains, {n_good} remaining")
        print(f"After filter:")
        print(f"  Iterative  - pos: {np.mean(pos_it):.2f} um, orient: {np.mean(ori_it):.4f} deg, strain: {np.mean(str_it):.4f}%")
        print(f"  All-at-once- pos: {np.mean(pos_aa):.2f} um, orient: {np.mean(ori_aa):.4f} deg, strain: {np.mean(str_aa):.4f}%")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    hist_kw = dict(alpha=0.6, edgecolor='black', linewidth=0.4)
    iter_color = '#2166ac'
    aao_color = '#b2182b'

    # Panel 1: Position error
    ax = axes[0]
    pmax = max(np.percentile(pos_it, 98), np.percentile(pos_aa, 98)) * 1.2
    bins_pos = np.linspace(0, pmax, 25)
    ax.hist(pos_it, bins=bins_pos, label=f'Iterative ($\\mu$={np.mean(pos_it):.1f} $\\mu$m)', color=iter_color, **hist_kw)
    ax.hist(pos_aa, bins=bins_pos, label=f'All-at-once ($\\mu$={np.mean(pos_aa):.1f} $\\mu$m)', color=aao_color, **hist_kw)
    ax.set_xlabel('Position Error ($\\mu$m)', fontsize=12)
    ax.set_ylabel('Number of Grains', fontsize=12)
    ax.set_title('(a) Position', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)

    # Panel 2: Orientation error
    ax = axes[1]
    omax = max(np.percentile(ori_it, 98), np.percentile(ori_aa, 98)) * 1.2
    bins_ori = np.linspace(0, omax, 25)
    ax.hist(ori_it, bins=bins_ori, label=f'Iterative ($\\mu$={np.mean(ori_it):.3f}$^\\circ$)', color=iter_color, **hist_kw)
    ax.hist(ori_aa, bins=bins_ori, label=f'All-at-once ($\\mu$={np.mean(ori_aa):.3f}$^\\circ$)', color=aao_color, **hist_kw)
    ax.set_xlabel('Misorientation ($^\\circ$)', fontsize=12)
    ax.set_ylabel('Number of Grains', fontsize=12)
    ax.set_title('(b) Orientation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)

    # Panel 3: Strain error
    ax = axes[2]
    smax = max(np.percentile(str_it, 98), np.percentile(str_aa, 98)) * 1.2
    bins_str = np.linspace(0, smax, 25)
    ax.hist(str_it, bins=bins_str, label=f'Iterative ($\\mu$={np.mean(str_it):.3f}%)', color=iter_color, **hist_kw)
    ax.hist(str_aa, bins=bins_str, label=f'All-at-once ($\\mu$={np.mean(str_aa):.3f}%)', color=aao_color, **hist_kw)
    ax.set_xlabel('Mean Lattice Parameter Error (%)', fontsize=12)
    ax.set_ylabel('Number of Grains', fontsize=12)
    ax.set_title('(c) Strain', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)

    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    note = f'{n_good} grains'
    if n_filt > 0:
        note += f' ({n_filt} misindexed excluded)'
    fig.suptitle(f'Iterative vs All-at-Once Refinement ({note})',
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = work_dir / args.output
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")


if __name__ == "__main__":
    main()
