#!/usr/bin/env python3
"""
Spatial parity maps: compare CPU vs GPU .mic files.

Usage:
  2-panel: python3 parity_maps.py cpu.mic gpu.mic [SGNum]
  4-panel: python3 parity_maps.py cpu.mic gpu_float.mic gpu_double.mic [SGNum]
"""

import sys
import struct
import numpy as np
from math import sin, cos, acos, sqrt, fabs

rad2deg = 57.2957795130823
EPS = 1e-12

# ── Misorientation (from CalcMiso.py, cubic only) ──

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

def calc_miso_deg(euler1_rad, euler2_rad):
    q1 = _om2quat(_euler2om(euler1_rad))
    q2 = _om2quat(_euler2om(euler2_rad))
    q1f = _bringdown(q1, CubSym)
    q2f = _bringdown(q2, CubSym)
    q1f[0] = -q1f[0]
    qp = _qprod(q1f, q2f)
    mv = _bringdown(qp, CubSym)
    if mv[0] > 1: mv[0] = 1
    return 2*acos(mv[0])*rad2deg


def read_mic(fn):
    with open(fn, 'rb') as f: data = f.read()
    n = len(data) // (11 * 8)
    return [struct.unpack_from('11d', data, i * 11 * 8) for i in range(n)]


def compute_parity(cpu, gpu, n):
    """Returns (frac_diff, miso_deg, active_mask, stats_dict)."""
    cpu_frac = np.array([r[10] for r in cpu[:n]])
    gpu_frac = np.array([r[10] for r in gpu[:n]])
    frac_diff = np.abs(cpu_frac - gpu_frac)

    miso = np.zeros(n)
    for i in range(n):
        if cpu_frac[i] == 0 and gpu_frac[i] == 0: continue
        try:
            miso[i] = calc_miso_deg(
                [cpu[i][7], cpu[i][8], cpu[i][9]],
                [gpu[i][7], gpu[i][8], gpu[i][9]])
        except: pass

    active = (cpu_frac > 0) | (gpu_frac > 0)
    na = np.sum(active)
    match = np.sum((frac_diff < 0.02) & active)
    stats = dict(n_active=na, n_match=int(match), n_mismatch=int(na-match),
                 frac_mean=np.mean(frac_diff[active]),
                 frac_max=np.max(frac_diff[active]),
                 miso_mean=np.mean(miso[active]),
                 miso_max=np.max(miso[active]),
                 match_pct=100*match/na if na>0 else 0)
    return frac_diff, miso, active, stats


def plot_pair(axes_row, xs, ys, frac_diff, miso, active, stats, label):
    """Plot confidence diff + misorientation on a row of 2 axes."""
    ax1, ax2 = axes_row

    vmax1 = max(0.1, np.percentile(frac_diff[active], 95))
    sc1 = ax1.scatter(xs, ys, c=frac_diff, cmap='RdYlGn_r', s=6, vmin=0, vmax=vmax1)
    ax1.set_xlabel('X (µm)'); ax1.set_ylabel('Y (µm)')
    ax1.set_title(f'{label}: |Δfrac|\n'
                  f'mean={stats["frac_mean"]:.4f}, match={stats["match_pct"]:.1f}%')
    ax1.set_aspect('equal')
    import matplotlib.pyplot as plt
    plt.colorbar(sc1, ax=ax1, label='|Δfrac|')

    vmax2 = max(1.0, np.percentile(miso[active], 95))
    sc2 = ax2.scatter(xs, ys, c=miso, cmap='hot', s=6, vmin=0, vmax=vmax2)
    ax2.set_xlabel('X (µm)'); ax2.set_ylabel('Y (µm)')
    ax2.set_title(f'{label}: Misorientation (°)\n'
                  f'mean={stats["miso_mean"]:.2f}°, max={stats["miso_max"]:.2f}°')
    ax2.set_aspect('equal')
    plt.colorbar(sc2, ax=ax2, label='Miso (°)')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if len(args) < 2:
        print("Usage: parity_maps.py cpu.mic gpu.mic [SGNum]")
        print("       parity_maps.py cpu.mic gpu_float.mic gpu_double.mic [SGNum]")
        sys.exit(1)

    # Detect 2-file or 3-file mode
    try:
        int(args[-1])
        sg = int(args[-1])
        mic_files = args[:-1]
    except ValueError:
        sg = 225
        mic_files = args

    cpu = read_mic(mic_files[0])
    n = len(cpu)
    xs = np.array([r[3] for r in cpu[:n]])
    ys = np.array([r[4] for r in cpu[:n]])

    two_mode = (len(mic_files) == 3)

    if two_mode:
        gpu_float = read_mic(mic_files[1])
        gpu_double = read_mic(mic_files[2])
        n = min(n, len(gpu_float), len(gpu_double))
        xs, ys = xs[:n], ys[:n]

        fd_f, mi_f, act_f, st_f = compute_parity(cpu, gpu_float, n)
        fd_d, mi_d, act_d, st_d = compute_parity(cpu, gpu_double, n)

        print(f"=== GPU Float vs CPU ===")
        print(f"  Active: {st_f['n_active']}, Match: {st_f['n_match']} ({st_f['match_pct']:.1f}%)")
        print(f"  Δfrac: mean={st_f['frac_mean']:.4f}, max={st_f['frac_max']:.4f}")
        print(f"  Miso:  mean={st_f['miso_mean']:.2f}°, max={st_f['miso_max']:.2f}°")
        print(f"\n=== GPU Double vs CPU ===")
        print(f"  Active: {st_d['n_active']}, Match: {st_d['n_match']} ({st_d['match_pct']:.1f}%)")
        print(f"  Δfrac: mean={st_d['frac_mean']:.4f}, max={st_d['frac_max']:.4f}")
        print(f"  Miso:  mean={st_d['miso_mean']:.2f}°, max={st_d['miso_max']:.2f}°")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available"); return

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        plot_pair(axes[0], xs, ys, fd_f, mi_f, act_f, st_f, 'GPU Float')
        plot_pair(axes[1], xs, ys, fd_d, mi_d, act_d, st_d, 'GPU Double')
        plt.suptitle(f'CPU vs GPU Parity — Float vs Double — {st_f["n_active"]} active voxels',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig('parity_maps.png', dpi=150)
        print(f"\nSaved 4-panel figure to parity_maps.png")

    else:
        gpu = read_mic(mic_files[1])
        n = min(n, len(gpu))
        xs, ys = xs[:n], ys[:n]

        fd, mi, act, st = compute_parity(cpu, gpu, n)
        print(f"Active: {st['n_active']}, Match: {st['n_match']} ({st['match_pct']:.1f}%)")
        print(f"Δfrac: mean={st['frac_mean']:.4f}, max={st['frac_max']:.4f}")
        print(f"Miso:  mean={st['miso_mean']:.2f}°, max={st['miso_max']:.2f}°")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available"); return

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plot_pair(axes, xs, ys, fd, mi, act, st, 'GPU')
        plt.suptitle(f'CPU vs GPU Parity — {st["n_active"]} active voxels',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('parity_maps.png', dpi=150)
        print(f"\nSaved 2-panel figure to parity_maps.png")


if __name__ == '__main__':
    main()
