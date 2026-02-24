#!/usr/bin/env python3
"""
Extract grains from a .mic file closest to given (X, Y) positions.

For each query position, finds the nearest point, then extracts all points
whose misorientation from that reference is below a threshold (in degrees).

Usage:
    # Specify positions on command line:
    python extract_grains_mic.py microstructure.mic -pos "100 500" "-80 510"

    # Set misorientation tolerance (degrees) and space group:
    python extract_grains_mic.py microstructure.mic -pos "100 500" -misoTol 5 -sgnum 225

    # Specify positions from a file (one "X Y" pair per line):
    python extract_grains_mic.py microstructure.mic -posfile positions.txt

    # Save extracted points to a file:
    python extract_grains_mic.py microstructure.mic -pos "100 500" -out extracted.csv
"""

import argparse
import os
import sys
import numpy as np

# Add the utils directory to the path so we can import calcMiso
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calcMiso import GetMisOrientationAngle


def load_mic(filepath):
    """Load a .mic file, returning header lines and data array."""
    header_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('%'):
                header_lines.append(line.strip())
            else:
                break

    col_names = header_lines[-1].lstrip('%').split('\t')
    col_names = [c.strip() for c in col_names if c.strip()]

    data = np.genfromtxt(filepath, comments='%')
    return header_lines, col_names, data


def find_grain_by_miso(data, x, y, miso_tol_deg, sgnum, radius=None):
    """Find all points belonging to the same grain as the nearest point to (x,y).

    Grain membership is determined by misorientation angle < miso_tol_deg.

    Args:
        data: Full microstructure array
        x, y: Query position
        miso_tol_deg: Misorientation tolerance in degrees
        sgnum: Space group number for symmetry operations
        radius: Optional spatial radius constraint (microns)

    Returns:
        (nearest_idx, grain_mask, distance_to_nearest)
    """
    xs = data[:, 3]  # X column
    ys = data[:, 4]  # Y column

    dist = np.sqrt((xs - x)**2 + (ys - y)**2)
    nearest_idx = np.argmin(dist)

    # Reference Euler angles (already in radians in .mic files)
    ref_euler = data[nearest_idx, 7:10]

    # Apply spatial radius filter first to reduce computation
    if radius is not None:
        candidates = np.where(dist <= radius)[0]
    else:
        candidates = np.arange(len(data))

    # Compute misorientation for each candidate
    grain_mask = np.zeros(len(data), dtype=bool)
    miso_tol_rad = miso_tol_deg * np.pi / 180.0

    for idx in candidates:
        euler2 = data[idx, 7:10]
        miso_angle, _ = GetMisOrientationAngle(ref_euler, euler2, sgnum)
        if miso_angle <= miso_tol_rad:
            grain_mask[idx] = True

    return nearest_idx, grain_mask, dist[nearest_idx]


def main():
    parser = argparse.ArgumentParser(
        description='Extract grains from .mic file closest to given positions '
                    'using misorientation-based matching.')
    parser.add_argument('micfile', help='Path to .mic microstructure file')
    parser.add_argument('-pos', nargs='+', default=[],
                        help='Positions as "X Y" strings, e.g. -pos "100 500" "-80 510"')
    parser.add_argument('-posfile', type=str, default='',
                        help='File with one "X Y" pair per line')
    parser.add_argument('-misoTol', type=float, default=5.0,
                        help='Misorientation tolerance in degrees (default: 5.0)')
    parser.add_argument('-sgnum', type=int, default=225,
                        help='Space group number for symmetry (default: 225, cubic FCC)')
    parser.add_argument('-radius', type=float, default=None,
                        help='Max spatial search radius (microns). Speeds up large files.')
    parser.add_argument('-minconf', type=float, default=0.0,
                        help='Minimum confidence threshold (default: 0.0)')
    parser.add_argument('-out', type=str, default='',
                        help='Output CSV file for extracted points')
    args = parser.parse_args()

    # Load mic file
    print(f"Loading {args.micfile}...")
    header_lines, col_names, data = load_mic(args.micfile)
    print(f"  {len(data)} points, columns: {col_names}")
    print(f"  Misorientation tolerance: {args.misoTol}°, Space group: {args.sgnum}")

    # Apply confidence filter
    if args.minconf > 0:
        conf_col = 10  # Confidence column
        mask = data[:, conf_col] >= args.minconf
        data = data[mask]
        print(f"  After confidence filter (>= {args.minconf}): {len(data)} points")

    # Collect positions
    positions = []
    for p in args.pos:
        parts = p.split()
        if len(parts) == 2:
            positions.append((float(parts[0]), float(parts[1])))
    if args.posfile:
        with open(args.posfile, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    positions.append((float(parts[0]), float(parts[1])))

    # Interactive mode if no positions given
    if not positions:
        print("\nEnter positions as 'X Y' (one per line, empty line to finish):")
        while True:
            try:
                line = input("  > ").strip()
            except EOFError:
                break
            if not line:
                break
            parts = line.split()
            if len(parts) >= 2:
                positions.append((float(parts[0]), float(parts[1])))

    if not positions:
        print("No positions provided.")
        sys.exit(1)

    # Process each position
    all_extracted = []
    print(f"\n{'Pos':>4}  {'Query X':>10}  {'Query Y':>10}  {'Nearest X':>10}  "
          f"{'Nearest Y':>10}  {'Dist':>8}  {'GrainPts':>9}  "
          f"{'Eul1':>8}  {'Eul2':>8}  {'Eul3':>8}  {'Conf':>6}")
    print("-" * 110)

    for i, (qx, qy) in enumerate(positions):
        nearest_idx, grain_mask, dist = find_grain_by_miso(
            data, qx, qy, args.misoTol, args.sgnum, args.radius)
        grain_pts = data[grain_mask]
        nearest = data[nearest_idx]

        print(f"{i+1:>4}  {qx:>10.2f}  {qy:>10.2f}  {nearest[3]:>10.2f}  "
              f"{nearest[4]:>10.2f}  {dist:>8.2f}  {len(grain_pts):>9}  "
              f"{nearest[7]:>8.4f}  {nearest[8]:>8.4f}  {nearest[9]:>8.4f}  "
              f"{nearest[10]:>6.3f}")

        all_extracted.append(grain_pts)

    # Save output
    if args.out and all_extracted:
        combined = np.vstack(all_extracted)
        with open(args.out, 'w') as f:
            # Write the original .mic header lines
            for h in header_lines:
                f.write(h + '\n')
            # Write data rows
            for row in combined:
                f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')
        print(f"\nSaved {len(combined)} points ({len(header_lines)} header lines) to {args.out}")


if __name__ == '__main__':
    main()

