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
from math import sin, cos, acos, sqrt, fabs

rad2deg = 57.2957795130823
deg2rad = 0.0174532925199433
EPS = 1e-12

# ── Misorientation functions (from utils/CalcMiso.py) ──

CubSym = [[1,0,0,0],[0.70711,0.70711,0,0],[0,1,0,0],[0.70711,-0.70711,0,0],
           [0.70711,0,0.70711,0],[0,0,1,0],[0.70711,0,-0.70711,0],
           [0.70711,0,0,0.70711],[0,0,0,1],[0.70711,0,0,-0.70711],
           [0.5,0.5,0.5,0.5],[0.5,-0.5,-0.5,-0.5],
           [0.5,-0.5,0.5,0.5],[0.5,0.5,-0.5,-0.5],
           [0.5,0.5,-0.5,0.5],[0.5,-0.5,0.5,-0.5],
           [0.5,-0.5,-0.5,0.5],[0.5,0.5,0.5,-0.5],
           [0,0.70711,0.70711,0],[0,-0.70711,0.70711,0],
           [0,0.70711,0,0.70711],[0,0.70711,0,-0.70711],
           [0,0,0.70711,0.70711],[0,0,0.70711,-0.70711]]

def _normalize(q):
    n = sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
    if n < 1e-15: return [1,0,0,0]
    return [q[0]/n, q[1]/n, q[2]/n, q[3]/n]

def _qprod(q, r):
    Q = [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
         r[1]*q[0]+r[0]*q[1]+r[3]*q[2]-r[2]*q[3],
         r[2]*q[0]+r[0]*q[2]+r[1]*q[3]-r[3]*q[1],
         r[3]*q[0]+r[0]*q[3]+r[2]*q[1]-r[1]*q[2]]
    if Q[0] < 0: Q = [-Q[0],-Q[1],-Q[2],-Q[3]]
    return _normalize(Q)

def _om2quat(m):
    trace = m[0]+m[4]+m[8]
    if trace > 0:
        s = 0.5/sqrt(trace+1.0)
        q = [0.25/s, (m[7]-m[5])*s, (m[2]-m[6])*s, (m[3]-m[1])*s]
    elif m[0]>m[4] and m[0]>m[8]:
        s = 2.0*sqrt(1.0+m[0]-m[4]-m[8])
        q = [(m[7]-m[5])/s, 0.25*s, (m[1]+m[3])/s, (m[2]+m[6])/s]
    elif m[4]>m[8]:
        s = 2.0*sqrt(1.0+m[4]-m[0]-m[8])
        q = [(m[2]-m[6])/s, (m[1]+m[3])/s, 0.25*s, (m[5]+m[7])/s]
    else:
        s = 2.0*sqrt(1.0+m[8]-m[0]-m[4])
        q = [(m[3]-m[1])/s, (m[2]+m[6])/s, (m[5]+m[7])/s, 0.25*s]
    if q[0] < 0: q = [-q[0],-q[1],-q[2],-q[3]]
    return _normalize(q)

def _euler2om(e):
    cps,cph,cth = cos(e[0]),cos(e[1]),cos(e[2])
    sps,sph,sth = sin(e[0]),sin(e[1]),sin(e[2])
    return [cth*cps-sth*cph*sps, -cth*cph*sps-sth*cps, sph*sps,
            cth*sps+sth*cph*cps, cth*cph*cps-sth*sps, -sph*cps,
            sth*sph, cth*sph, cph]

def _bringdown(qin, sym):
    best, maxc = None, -1e10
    for s in sym:
        qt = _qprod(qin, s)
        if qt[0] > maxc:
            maxc = qt[0]; best = qt
    return _normalize(best)

def calc_miso_deg(euler1_rad, euler2_rad, sg=225):
    """Misorientation angle in degrees between two Euler-angle sets (radians)."""
    q1 = _om2quat(_euler2om(euler1_rad))
    q2 = _om2quat(_euler2om(euler2_rad))
    # For cubic (SG 195-230)
    sym = CubSym
    q1f = _bringdown(q1, sym)
    q2f = _bringdown(q2, sym)
    q1f[0] = -q1f[0]
    qp = _qprod(q1f, q2f)
    mv = _bringdown(qp, sym)
    if mv[0] > 1: mv[0] = 1
    return 2*acos(mv[0])*rad2deg


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

    # Fields: row[3]=x, row[4]=y, row[7:10]=euler (rad), row[10]=frac
    xs = np.array([r[3] for r in cpu[:n]])
    ys = np.array([r[4] for r in cpu[:n]])
    cpu_frac = np.array([r[10] for r in cpu[:n]])
    gpu_frac = np.array([r[10] for r in gpu[:n]])
    frac_diff = np.abs(cpu_frac - gpu_frac)

    # Misorientation
    miso_deg = np.zeros(n)
    for i in range(n):
        if cpu_frac[i] == 0 and gpu_frac[i] == 0:
            continue
        e1 = [cpu[i][7], cpu[i][8], cpu[i][9]]
        e2 = [gpu[i][7], gpu[i][8], gpu[i][9]]
        try:
            miso_deg[i] = calc_miso_deg(e1, e2, sg_num)
        except Exception:
            miso_deg[i] = 0

    # Stats
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

    # Plotting
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Map 1: Confidence difference
    ax = axes[0]
    vmax1 = max(0.1, np.percentile(frac_diff[active], 95))
    sc = ax.scatter(xs, ys, c=frac_diff, cmap='RdYlGn_r', s=8, vmin=0, vmax=vmax1)
    ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)')
    ax.set_title(f'|CPU frac − GPU frac|\n'
                 f'mean={np.mean(frac_diff[active]):.4f}, match={100*n_match/n_active:.1f}%')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='|Δfrac|')

    # Map 2: Misorientation
    ax = axes[1]
    vmax2 = max(1.0, np.percentile(miso_deg[active], 95))
    sc2 = ax.scatter(xs, ys, c=miso_deg, cmap='hot', s=8, vmin=0, vmax=vmax2)
    ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)')
    ax.set_title(f'Misorientation CPU vs GPU (°)\n'
                 f'mean={np.mean(miso_deg[active]):.2f}°, max={np.max(miso_deg[active]):.2f}°')
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
