#!/usr/bin/env python3
"""
Spatial parity maps: compare CPU vs GPU .mic files.
Produces two PNG maps:
  1. Confidence difference (|cpu_frac - gpu_frac|)
  2. Misorientation angle (degrees) between CPU and GPU Euler angles

Usage: python3 parity_maps.py cpu_benchmark.mic gpu_benchmark.mic [SGNum]
  SGNum defaults to 225 (cubic, FCC aluminum)
"""

import sys
import struct
import numpy as np
import os

# Add MIDAS utils to path for CalcMiso
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDAS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(MIDAS_DIR, 'utils'))
from CalcMiso import GetMisOrientationAngle, Euler2OrientMat

rad2deg = 57.2957795130823


def read_mic(fn):
    """Read binary .mic file: 11 doubles per row."""
    with open(fn, 'rb') as f:
        data = f.read()
    n = len(data) // (11 * 8)
    rows = []
    for i in range(n):
        row = struct.unpack_from('11d', data, i * 11 * 8)
        rows.append(row)
    return rows


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 parity_maps.py cpu.mic gpu.mic [SGNum]")
        sys.exit(1)

    cpu_file = sys.argv[1]
    gpu_file = sys.argv[2]
    sg_num = int(sys.argv[3]) if len(sys.argv) > 3 else 225

    cpu = read_mic(cpu_file)
    gpu = read_mic(gpu_file)
    n = min(len(cpu), len(gpu))
    print(f"Read {n} voxels from each .mic file (SGNum={sg_num})")

    # Extract fields:
    # row[3]=x, row[4]=y, row[7:10]=euler (rad), row[10]=frac
    xs = np.array([r[3] for r in cpu[:n]])
    ys = np.array([r[4] for r in cpu[:n]])

    cpu_frac = np.array([r[10] for r in cpu[:n]])
    gpu_frac = np.array([r[10] for r in gpu[:n]])
    frac_diff = np.abs(cpu_frac - gpu_frac)

    # Misorientation calculation
    miso_deg = np.zeros(n)
    for i in range(n):
        cf = cpu_frac[i]
        gf = gpu_frac[i]
        if cf == 0 and gf == 0:
            miso_deg[i] = 0  # both zero, no orientation
            continue
        cpu_euler = [cpu[i][7], cpu[i][8], cpu[i][9]]  # radians
        gpu_euler = [gpu[i][7], gpu[i][8], gpu[i][9]]  # radians
        try:
            angle_rad, _ = GetMisOrientationAngle(cpu_euler, gpu_euler, sg_num)
            miso_deg[i] = angle_rad * rad2deg
        except Exception:
            miso_deg[i] = 0

    # Statistics
    active = (cpu_frac > 0) | (gpu_frac > 0)
    n_active = np.sum(active)
    match_mask = frac_diff < 0.02
    n_match = np.sum(match_mask & active)
    n_mismatch = np.sum(~match_mask & active)
    print(f"Active voxels: {n_active}")
    print(f"Match (<2% frac diff): {n_match} ({100*n_match/n_active:.1f}%)")
    print(f"Mismatch: {n_mismatch} ({100*n_mismatch/n_active:.1f}%)")
    print(f"Confidence diff: mean={np.mean(frac_diff[active]):.4f}, "
          f"max={np.max(frac_diff[active]):.4f}, "
          f"median={np.median(frac_diff[active]):.4f}")
    print(f"Misorientation: mean={np.mean(miso_deg[active]):.4f}°, "
          f"max={np.max(miso_deg[active]):.4f}°, "
          f"median={np.median(miso_deg[active]):.4f}°")

    # --- Plotting ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    # Determine unique grid positions for scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Map 1: Confidence difference ---
    ax = axes[0]
    sc = ax.scatter(xs, ys, c=frac_diff, cmap='RdYlGn_r', s=8,
                    vmin=0, vmax=max(0.1, np.percentile(frac_diff[active], 95)))
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(f'|CPU frac − GPU frac|\n'
                 f'mean={np.mean(frac_diff[active]):.4f}, '
                 f'match={100*n_match/n_active:.1f}%')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='|Δfrac|')

    # --- Map 2: Misorientation ---
    ax = axes[1]
    vmax_miso = max(1.0, np.percentile(miso_deg[active], 95))
    sc2 = ax.scatter(xs, ys, c=miso_deg, cmap='hot', s=8,
                     vmin=0, vmax=vmax_miso)
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(f'Misorientation CPU vs GPU (°)\n'
                 f'mean={np.mean(miso_deg[active]):.2f}°, '
                 f'max={np.max(miso_deg[active]):.2f}°')
    ax.set_aspect('equal')
    plt.colorbar(sc2, ax=ax, label='Misorientation (°)')

    plt.suptitle(f'CPU vs GPU Phase 2 Parity — {n_active} active voxels, SGNum={sg_num}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_fn = 'parity_maps.png'
    plt.savefig(out_fn, dpi=150)
    print(f"\nSaved spatial parity maps to {out_fn}")


if __name__ == '__main__':
    main()
