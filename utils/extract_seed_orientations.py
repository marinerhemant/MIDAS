#!/usr/bin/env python3
"""
Extract seed orientations for a given space group from the binary
master quaternion file + lookup tables.

Can be used as a CLI or imported as a library by nf_MIDAS.py.

Usage:
    python extract_seed_orientations.py <sg_number> [--seed-dir DIR]
"""

import numpy as np
import os
import sys

_TrigType2SGs = {149, 151, 153, 157, 159, 162, 163}


def sg_to_lookup_type(sg):
    """Map space group number to MIDAS symmetry type name."""
    if sg <= 2:   return 'triclinic'
    if sg <= 15:  return 'monoclinic'
    if sg <= 74:  return 'orthorhombic'
    if sg <= 88:  return 'tetragonal_low'
    if sg <= 142: return 'tetragonal_high'
    if sg <= 148: return 'trigonal_low'
    if sg <= 167: return 'trigonal_type2' if sg in _TrigType2SGs else 'trigonal_type1'
    if sg <= 176: return 'hexagonal_low'
    if sg <= 194: return 'hexagonal_high'
    if sg <= 206: return 'cubic_low'
    if sg <= 230: return 'cubic_high'
    raise ValueError(f"Invalid space group number: {sg}")


def ensure_seed_orientations(sg, seed_dir):
    """Return path to CSV seed file for this SG, extracting if needed.

    If the CSV already exists, returns its path immediately.
    Otherwise reads orientations_master.bin + lookup_<type>.bin,
    writes the CSV, and returns its path.

    Parameters
    ----------
    sg : int
        Space group number (1-230).
    seed_dir : str
        Directory containing orientations_master.bin and lookup_*.bin files.

    Returns
    -------
    str
        Path to the CSV seed file (w,x,y,z per line).
    """
    lookup_type = sg_to_lookup_type(sg)
    csv_name = f"seed_{lookup_type}.csv"
    csv_path = os.path.join(seed_dir, csv_name)

    if os.path.exists(csv_path):
        return csv_path

    master_path = os.path.join(seed_dir, 'orientations_master.bin')
    lookup_path = os.path.join(seed_dir, f'lookup_{lookup_type}.bin')

    if not os.path.exists(master_path):
        raise FileNotFoundError(
            f"Master quaternion file not found: {master_path}\n"
            f"Run GenerateSeedLookupTables first (via build.sh).")
    if not os.path.exists(lookup_path):
        raise FileNotFoundError(
            f"Lookup table not found: {lookup_path}\n"
            f"Run GenerateSeedLookupTables first (via build.sh).")

    master = np.fromfile(master_path, dtype=np.float64).reshape(-1, 4)
    indices = np.fromfile(lookup_path, dtype=np.int32)
    seeds = master[indices]

    np.savetxt(csv_path, seeds, delimiter=',', fmt='%.7f')
    return csv_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract seed orientations for a space group.')
    parser.add_argument('sg', type=int, help='Space group number (1-230)')
    parser.add_argument('--seed-dir', default=None,
                        help='Directory with master.bin and lookup files '
                             '(default: NF_HEDM/seedOrientations/)')
    args = parser.parse_args()

    if args.seed_dir is None:
        args.seed_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'NF_HEDM', 'seedOrientations')

    path = ensure_seed_orientations(args.sg, args.seed_dir)
    n = sum(1 for _ in open(path))
    print(f"SG {args.sg} → {sg_to_lookup_type(args.sg)}: {path} ({n:,} orientations)")
