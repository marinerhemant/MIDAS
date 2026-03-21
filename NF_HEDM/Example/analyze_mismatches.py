#!/usr/bin/env python3
"""Analyze mismatching voxels between CPU and GPU mic files."""
import struct, sys, math

def read_mic(fn):
    with open(fn, 'rb') as f:
        data = f.read()
    n = len(data) // (11 * 8)
    return [struct.unpack_from('11d', data, i * 11 * 8) for i in range(n)]

cpu = read_mic('cpu_benchmark.mic')
gpu = read_mic('gpu_benchmark.mic')
n = min(len(cpu), len(gpu))

# Fields: [0]=rowNr, [1]=nWinners, [2]=time, [3]=xs, [4]=ys, [5]=gridSize,
#          [6]=UD, [7]=euler0, [8]=euler1, [9]=euler2, [10]=frac

print(f"{'VoxIdx':>6} {'CPU_frac':>9} {'GPU_frac':>9} {'Δfrac':>8} {'CPU_win':>7} {'GPU_win':>7} {'CPU_ori':>7} {'GPU_ori':>7} {'xs':>7} {'ys':>7} {'Category'}")
print("-" * 110)

categories = {'gpu_better': 0, 'cpu_better': 0, 'both_low': 0,
              'screen_diff': 0, 'orientation_diff': 0}

mismatches = []
for i in range(n):
    cf, gf = cpu[i][10], gpu[i][10]
    if cf == 0 and gf == 0:
        continue
    if abs(cf - gf) < 0.02:
        continue

    # Mismatch found
    cwin = int(cpu[i][1])
    gwin = int(gpu[i][1])
    cori = int(cpu[i][0])  # bestRowNr (orientation index)
    gori = int(gpu[i][0])
    xs = cpu[i][3]
    ys = cpu[i][4]
    delta = gf - cf

    # Categorize
    if cwin != gwin:
        cat = "SCREEN_DIFF"
        categories['screen_diff'] += 1
    elif cori != gori:
        cat = "ORI_DIFF"
        categories['orientation_diff'] += 1
    elif gf > cf:
        cat = "GPU_BETTER"
        categories['gpu_better'] += 1
    elif cf > gf:
        cat = "CPU_BETTER"
        categories['cpu_better'] += 1
    else:
        cat = "TIED"

    if cf < 0.04 and gf < 0.04:
        cat += " (both_low)"
        categories['both_low'] += 1

    mismatches.append((i, cf, gf, delta, cwin, gwin, cori, gori, xs, ys, cat))

# Sort by |delta| descending
mismatches.sort(key=lambda x: -abs(x[3]))

for m in mismatches:
    i, cf, gf, delta, cwin, gwin, cori, gori, xs, ys, cat = m
    print(f"{i:6d} {cf:9.6f} {gf:9.6f} {delta:+8.5f} {cwin:7d} {gwin:7d} {cori:7d} {gori:7d} {xs:7.1f} {ys:7.1f} {cat}")

print(f"\n=== Summary ({len(mismatches)} mismatches) ===")
print(f"  Screen diff (different winner count): {categories['screen_diff']}")
print(f"  Orientation diff (same count, diff ori): {categories['orientation_diff']}")
print(f"  GPU better (same ori, GPU frac > CPU):  {categories['gpu_better']}")
print(f"  CPU better (same ori, CPU frac > GPU):  {categories['cpu_better']}")
print(f"  Both low (<0.04, near MinFrac threshold): {categories['both_low']}")

# Distribution of |delta|
deltas = [abs(m[3]) for m in mismatches]
if deltas:
    print(f"\n  |Δfrac| stats: min={min(deltas):.4f} median={sorted(deltas)[len(deltas)//2]:.4f} max={max(deltas):.4f}")
