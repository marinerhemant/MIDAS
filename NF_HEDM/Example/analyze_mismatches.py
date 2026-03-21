#!/usr/bin/env python3
"""Analyze mismatching voxels between CPU and GPU mic files."""
import struct, sys, math
import numpy as np

show_all = '--all' in sys.argv

def read_mic(fn):
    with open(fn, 'rb') as f:
        data = f.read()
    n = len(data) // (11 * 8)
    return [struct.unpack_from('11d', data, i * 11 * 8) for i in range(n)]

def euler2mat(e):
    """Euler angles (radians) -> orientation matrix (ZXZ convention)."""
    c1, s1 = math.cos(e[0]), math.sin(e[0])
    c2, s2 = math.cos(e[1]), math.sin(e[1])
    c3, s3 = math.cos(e[2]), math.sin(e[2])
    return np.array([
        [ c1*c3 - s1*c2*s3, -c1*s3 - s1*c2*c3,  s1*s2],
        [ s1*c3 + c1*c2*s3, -s1*s3 + c1*c2*c3, -c1*s2],
        [ s2*s3,              s2*c3,              c2   ]])

# 24 cubic symmetry operators
_cubic_syms = None
def cubic_syms():
    global _cubic_syms
    if _cubic_syms is not None:
        return _cubic_syms
    # Generate from 90-degree rotations about <100> and <111>
    I = np.eye(3)
    rots = [I]
    # 90/180/270 about x,y,z
    for ax in range(3):
        for angle in [math.pi/2, math.pi, 3*math.pi/2]:
            c, s = round(math.cos(angle)), round(math.sin(angle))
            R = np.eye(3)
            axes = [i for i in range(3) if i != ax]
            R[axes[0], axes[0]] = c; R[axes[0], axes[1]] = -s
            R[axes[1], axes[0]] = s; R[axes[1], axes[1]] = c
            rots.append(R)
    # Generate full group by composition
    group = set()
    for r in rots:
        group.add(tuple(r.flatten()))
    changed = True
    while changed:
        changed = False
        new = set()
        for a in list(group):
            for b in list(group):
                ab = np.dot(np.array(a).reshape(3,3), np.array(b).reshape(3,3))
                ab = np.round(ab).astype(int)
                key = tuple(ab.flatten())
                if key not in group:
                    new.add(key)
                    changed = True
        group |= new
    _cubic_syms = [np.array(g).reshape(3,3).astype(float) for g in group]
    return _cubic_syms

def misorientation_cubic(e1, e2):
    """Misorientation angle (degrees) between two orientations, cubic symmetry."""
    R1 = euler2mat(e1)
    R2 = euler2mat(e2)
    dR = R1 @ R2.T
    min_angle = 180.0
    for S in cubic_syms():
        M = S @ dR
        tr = np.clip((np.trace(M) - 1) / 2, -1, 1)
        angle = math.degrees(math.acos(tr))
        if angle < min_angle:
            min_angle = angle
    return min_angle

cpu = read_mic('cpu_benchmark.mic')
gpu = read_mic('gpu_benchmark.mic')
n = min(len(cpu), len(gpu))

print(f"{'VoxIdx':>6} {'CPU_frac':>9} {'GPU_frac':>9} {'Δfrac':>8} {'Miso°':>6} {'CPU_win':>7} {'GPU_win':>7} {'CPU_ori':>7} {'GPU_ori':>7} {'xs':>7} {'ys':>7} {'Category'}")
print("-" * 120)

categories = {'gpu_better': 0, 'cpu_better': 0, 'both_low': 0,
              'screen_diff': 0, 'orientation_diff': 0, 'match': 0, 'both_zero': 0}

rows = []
for i in range(n):
    cf, gf = cpu[i][10], gpu[i][10]
    cwin = int(cpu[i][1])
    gwin = int(gpu[i][1])
    cori = int(cpu[i][0])
    gori = int(gpu[i][0])
    xs = cpu[i][3]
    ys = cpu[i][4]
    delta = gf - cf

    # Compute misorientation
    ce = (cpu[i][7], cpu[i][8], cpu[i][9])
    ge = (gpu[i][7], gpu[i][8], gpu[i][9])
    if cf > 0 and gf > 0:
        miso = misorientation_cubic(ce, ge)
    else:
        miso = 0.0

    if cf == 0 and gf == 0:
        cat = "BOTH_ZERO"
        categories['both_zero'] += 1
    elif abs(cf - gf) < 0.02:
        cat = "MATCH"
        categories['match'] += 1
    elif cwin != gwin:
        cat = "SCREEN_DIFF"
        categories['screen_diff'] += 1
    elif cori != gori:
        cat = "ORI_DIFF"
        categories['orientation_diff'] += 1
    elif gf > cf:
        cat = "GPU_BETTER"
        categories['gpu_better'] += 1
    else:
        cat = "CPU_BETTER"
        categories['cpu_better'] += 1

    if cf < 0.04 and gf < 0.04 and cat not in ('MATCH', 'BOTH_ZERO'):
        cat += " (both_low)"
        categories['both_low'] += 1

    rows.append((i, cf, gf, delta, miso, cwin, gwin, cori, gori, xs, ys, cat))

# Filter and sort
if show_all:
    display = rows
else:
    display = [r for r in rows if r[11] not in ('MATCH', 'BOTH_ZERO')]
    display.sort(key=lambda x: -abs(x[3]))

for m in display:
    i, cf, gf, delta, miso, cwin, gwin, cori, gori, xs, ys, cat = m
    print(f"{i:6d} {cf:9.6f} {gf:9.6f} {delta:+8.5f} {miso:6.2f} {cwin:7d} {gwin:7d} {cori:7d} {gori:7d} {xs:7.1f} {ys:7.1f} {cat}")

total = categories['match'] + categories['screen_diff'] + categories['orientation_diff'] + categories['gpu_better'] + categories['cpu_better']
nmis = categories['screen_diff'] + categories['orientation_diff'] + categories['gpu_better'] + categories['cpu_better']
print(f"\n=== Summary ({n} voxels, {total} active, {nmis} mismatches) ===")
print(f"  Both zero:    {categories['both_zero']}")
print(f"  Match (<2%):  {categories['match']}")
print(f"  Screen diff:  {categories['screen_diff']}")
print(f"  Ori diff:     {categories['orientation_diff']}")
print(f"  GPU better:   {categories['gpu_better']}")
print(f"  CPU better:   {categories['cpu_better']}")

deltas = [abs(r[3]) for r in rows if r[11] not in ('MATCH', 'BOTH_ZERO')]
misos = [r[4] for r in rows if r[11] not in ('MATCH', 'BOTH_ZERO')]
if deltas:
    print(f"\n  |Δfrac| stats: min={min(deltas):.4f} median={sorted(deltas)[len(deltas)//2]:.4f} max={max(deltas):.4f}")
if misos:
    print(f"  Miso°  stats: min={min(misos):.2f} median={sorted(misos)[len(misos)//2]:.2f} max={max(misos):.2f}")
    low_miso = sum(1 for m in misos if m < 2.0)
    print(f"  Miso < 2°:    {low_miso}/{len(misos)} ({100*low_miso/len(misos):.0f}%)")
