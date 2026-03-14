#!/usr/bin/env python3
"""
plot_binarea.py — Diagnostic tool for DetectorMapper bin-area computation.

Reads Map.bin + nMap.bin produced by DetectorMapper and plots the total
bin area (sum of fractional pixel areas) as a function of eta for
user-selectable R bins.  A dip at eta = -90, 0, 90, 180° indicates a
bug in the polygon-clipping / intercept logic.

Usage:
    python plot_binarea.py <params.txt> [--rbin R1 R2 ...] [--dir <result_dir>]

If --rbin is not given, a few radii are chosen automatically (10%, 30%,
50%, 70%, 90% of the R range).
"""

import argparse
import math
import os
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── MapHeader constants (must match MapHeader.h) ────────────────────
MAP_HEADER_MAGIC = 0x3050414D  # "MAP0" little-endian
MAP_HEADER_SIZE = 64


def parse_params(fn):
    """Parse a MIDAS text parameter file, return dict of relevant values."""
    p = dict(
        RMin=10.0, RMax=1500.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=1.0,
    )
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            key = toks[0]
            if key in ('RMin', 'RMax', 'RBinSize', 'EtaMin', 'EtaMax',
                       'EtaBinSize'):
                p[key] = float(toks[1])
    return p


def read_map_files(map_fn, nmap_fn):
    """Read Map.bin and nMap.bin, return (pxList_fracs, nPxArr).

    pxList_fracs: 1-D float64 array of fractional areas (flat over all bins).
    nPxArr:       (nTotalBins, 2) int32 array — col 0 = count, col 1 = offset.
    """
    # -- nMap.bin --
    with open(nmap_fn, 'rb') as f:
        header = f.read(MAP_HEADER_SIZE)
        magic = struct.unpack_from('<I', header, 0)[0]
        if magic != MAP_HEADER_MAGIC:
            # Legacy file without header — seek back
            f.seek(0)
        nmap_data = f.read()
    nPxArr = np.frombuffer(nmap_data, dtype=np.int32).reshape(-1, 2)

    # -- Map.bin --
    with open(map_fn, 'rb') as f:
        header = f.read(MAP_HEADER_SIZE)
        magic = struct.unpack_from('<I', header, 0)[0]
        if magic != MAP_HEADER_MAGIC:
            f.seek(0)
        map_data = f.read()

    # Each entry in Map.bin is struct data { int y; int z; double frac; } = 16 bytes
    entry_dtype = np.dtype([('y', '<i4'), ('z', '<i4'), ('frac', '<f8')])
    pxEntries = np.frombuffer(map_data, dtype=entry_dtype)

    return pxEntries, nPxArr


def compute_bin_areas(pxEntries, nPxArr, nRBins, nEtaBins):
    """Compute total area and pixel count per (R, eta) bin."""
    binArea = np.zeros((nRBins, nEtaBins), dtype=np.float64)
    binNPx = np.zeros((nRBins, nEtaBins), dtype=np.int32)
    for rIdx in range(nRBins):
        for eIdx in range(nEtaBins):
            flat = rIdx * nEtaBins + eIdx
            count = nPxArr[flat, 0]
            offset = nPxArr[flat, 1]
            binNPx[rIdx, eIdx] = count
            if count > 0:
                binArea[rIdx, eIdx] = pxEntries['frac'][offset:offset + count].sum()
    return binArea, binNPx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('params', help='MIDAS parameter file (e.g. ps_midas.txt)')
    ap.add_argument('--dir', default='.', help='Directory containing Map.bin/nMap.bin')
    ap.add_argument('--rbin', type=float, nargs='+', default=None,
                    help='R-bin centers (pixels) to plot.  Default: auto-pick 5.')
    ap.add_argument('--rindex', type=int, nargs='+', default=None,
                    help='R-bin indices to plot (0-based). Overrides --rbin.')
    ap.add_argument('-o', '--output', default=None,
                    help='Save figure to file instead of showing')
    args = ap.parse_args()

    # Parse parameters
    p = parse_params(args.params)
    RMin, RMax, RBinSize = p['RMin'], p['RMax'], p['RBinSize']
    EtaMin, EtaMax, EtaBinSize = p['EtaMin'], p['EtaMax'], p['EtaBinSize']

    nRBins_param = int(math.ceil((RMax - RMin) / RBinSize))
    nEtaBins_param = int(math.ceil((EtaMax - EtaMin) / EtaBinSize))

    # Read map files
    map_fn = os.path.join(args.dir, 'Map.bin')
    nmap_fn = os.path.join(args.dir, 'nMap.bin')
    if not os.path.isfile(map_fn) or not os.path.isfile(nmap_fn):
        print(f"ERROR: Cannot find {map_fn} and/or {nmap_fn}", file=sys.stderr)
        print("Run DetectorMapper first, then point --dir to the output folder.",
              file=sys.stderr)
        sys.exit(1)

    pxEntries, nPxArr = read_map_files(map_fn, nmap_fn)
    totalBins = nPxArr.shape[0]

    # Auto-detect grid dimensions from file: totalBins = nRBins * nEtaBins
    # Trust nRBins from params (R range is always specified), derive nEtaBins
    nRBins = nRBins_param
    if totalBins % nRBins == 0:
        nEtaBins = totalBins // nRBins
        if nEtaBins != nEtaBins_param:
            EtaBinSize = (EtaMax - EtaMin) / nEtaBins
            print(f"Auto-detected EtaBinSize = {EtaBinSize:.2f} "
                  f"({nEtaBins} eta bins from file, param said {nEtaBins_param})")
    else:
        nEtaBins = nEtaBins_param
        if totalBins != nRBins * nEtaBins:
            print(f"ERROR: nMap has {totalBins} entries, cannot factor with "
                  f"nRBins={nRBins}. Check parameters.", file=sys.stderr)
            sys.exit(1)

    print(f"Bins: {nRBins} R × {nEtaBins} Eta  ({totalBins} total)")
    print(f"R range: [{RMin}, {RMax}] step {RBinSize}")
    print(f"Eta range: [{EtaMin}, {EtaMax}] step {EtaBinSize:.4f}")
    print(f"Map.bin: {len(pxEntries)} pixel entries")

    # Compute bin areas and pixel counts
    binArea, binNPx = compute_bin_areas(pxEntries, nPxArr, nRBins, nEtaBins)

    # Eta bin centers
    etaCenters = np.array([EtaMin + EtaBinSize * (i + 0.5)
                           for i in range(nEtaBins)])

    # Select R bins to plot
    if args.rindex is not None:
        rIndices = args.rindex
    elif args.rbin is not None:
        # Find closest R-bin index for each requested R value
        rCenters = np.array([RMin + RBinSize * (i + 0.5) for i in range(nRBins)])
        rIndices = [int(np.argmin(np.abs(rCenters - r))) for r in args.rbin]
    else:
        # Auto-pick 5 radii at 10%, 30%, 50%, 70%, 90% of the range
        fracs = [0.10, 0.30, 0.50, 0.70, 0.90]
        rIndices = [int(f * (nRBins - 1)) for f in fracs]

    rCenters = np.array([RMin + RBinSize * (i + 0.5) for i in range(nRBins)])

    # ── Numerical dump near critical angles ─────────────────────────
    critAngles = [-180, -90, 0, 90, 180]
    for rIdx in rIndices[:2]:  # just first 2 R bins
        rVal = rCenters[rIdx]
        print(f"\n── R = {rVal:.1f} px (index {rIdx}) ──")
        for crit in critAngles:
            # Find eta bin closest to this critical angle
            eidx = int(np.argmin(np.abs(etaCenters - crit)))
            lo = max(0, eidx - 2)
            hi = min(nEtaBins, eidx + 3)
            print(f"  near eta={crit}°:")
            for e in range(lo, hi):
                npx = binNPx[rIdx, e]
                area = binArea[rIdx, e]
                avg = area / npx if npx > 0 else 0
                marker = " <<<" if abs(etaCenters[e] - crit) < EtaBinSize else ""
                print(f"    eta={etaCenters[e]:7.1f}°  nPx={npx:4d}  "
                      f"area={area:.4f}  avg={avg:.6f}{marker}")

    # ── Plot: 3 rows per R (area, nPx, area/px) ────────────────────
    fig, axes = plt.subplots(len(rIndices), 1, figsize=(14, 3.5 * len(rIndices)),
                             sharex=True, squeeze=False)

    for ax_row, rIdx in zip(axes, rIndices):
        ax = ax_row[0]
        rVal = rCenters[rIdx]
        area = binArea[rIdx, :]
        npx = binNPx[rIdx, :].astype(float)
        avg_area = np.where(npx > 0, area / npx, 0)

        # Plot area on left axis
        ln1 = ax.plot(etaCenters, area, '-', linewidth=0.8, color='black',
                      label='binArea')
        ax.set_ylabel('Bin Area (px²)', color='black')

        # Plot nPx on right axis
        ax2 = ax.twinx()
        ln2 = ax2.plot(etaCenters, npx, '-', linewidth=0.5, color='red',
                       alpha=0.7, label='nPixels')
        ax2.set_ylabel('nPixels', color='red')

        # Compute deviation stats for nonzero bins
        valid = area[area > 0]
        if len(valid) > 0:
            med = np.median(valid)
            dev_pct = (valid.max() - valid.min()) / med * 100 if med > 0 else 0
        else:
            med = 0; dev_pct = 0

        ax.set_title(f'R = {rVal:.1f} px  (bin index {rIdx})  —  '
                     f'dev = {dev_pct:.4f}%')
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        ax2.ticklabel_format(useOffset=False, style='plain', axis='y')
        ax.grid(True, alpha=0.3)

        # Mark the critical angles
        for crit_eta in critAngles:
            if EtaMin <= crit_eta <= EtaMax:
                ax.axvline(crit_eta, color='red', linewidth=0.7, linestyle='--',
                           alpha=0.6)

        # Combined legend
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', fontsize=7)

    axes[-1][0].set_xlabel('Eta (degrees)')
    fig.suptitle('DetectorMapper binArea vs Eta — diagnostic', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()

