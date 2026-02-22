#!/usr/bin/env python
"""
match_grains.py — Match and stitch FF-HEDM grain reconstructions.

Modes:
  match   Match grains between two sample states (e.g., load steps).
  stitch  Stack layers from the same state into a single grain list.

Library usage:
  from utils.match_grains import load_grains, aggregate_grains, match_grains, stitch_layers

Author: Hemant Sharma / MIDAS
"""

import os
import sys
import argparse
import logging
import glob as globmod
from math import acos, radians

import numpy as np

# Add the directory containing this script so CalcMiso can be imported
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
# Also add parent MIDAS/utils in case of different CWD
_UTILS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), 'utils')
if _UTILS_DIR not in sys.path and os.path.isdir(_UTILS_DIR):
    sys.path.insert(0, _UTILS_DIR)

from calcMiso import (
    MakeSymmetries,
    OrientMat2Quat,
    GetMisOrientationAngleOM,
    BringDownToFundamentalRegionSym,
    QuaternionProduct,
    normalize,
    rad2deg,
    deg2rad,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grains.csv column layout (47 columns after 9-line header)
# ---------------------------------------------------------------------------
GRAINS_HEADER_LINES = 9
GRAINS_COLS = [
    'GrainID',
    'O11', 'O12', 'O13', 'O21', 'O22', 'O23', 'O31', 'O32', 'O33',
    'X', 'Y', 'Z',
    'a', 'b', 'c', 'alpha', 'beta', 'gamma',
    'eFab11', 'eFab12', 'eFab13', 'eFab21', 'eFab22', 'eFab23',
    'eFab31', 'eFab32', 'eFab33',
    'eKen11', 'eKen12', 'eKen13', 'eKen21', 'eKen22', 'eKen23',
    'eKen31', 'eKen32', 'eKen33',
    'RMSErrorStrain', 'Confidence', 'Reserved1', 'Reserved2',
    'PhaseNr', 'Radius', 'Eul0', 'Eul1', 'Eul2', 'Reserved3', 'Reserved4',
]


# ===================================================================
#  I/O helpers
# ===================================================================

def load_grains(path: str) -> np.ndarray:
    """Load grains from a consolidated HDF5 or Grains.csv.

    Tries ``<stem>_consolidated.h5`` first; falls back to ``Grains.csv``.

    Returns:
        2-D float array with one row per grain (same layout as Grains.csv).
    """
    # Try consolidated HDF5 first
    parent = os.path.dirname(os.path.abspath(path))
    h5_candidates = globmod.glob(os.path.join(parent, '*_consolidated.h5'))
    if h5_candidates:
        try:
            import h5py
            h5_path = h5_candidates[0]
            logger.info(f"Reading grains from {h5_path}")
            with h5py.File(h5_path, 'r') as h5:
                return h5['grains/summary'][:]
        except Exception as e:
            logger.warning(f"Could not read HDF5 {h5_candidates[0]}: {e}. "
                           f"Falling back to {path}")

    # Fall back to Grains.csv
    if not os.path.exists(path):
        raise FileNotFoundError(f"Grains file not found: {path}")
    logger.info(f"Reading grains from {path}")
    data = np.genfromtxt(path, skip_header=GRAINS_HEADER_LINES)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _orient_mat_to_quat_batch(orient_mats: np.ndarray) -> np.ndarray:
    """Convert Nx9 orientation matrices to Nx4 quaternions (scalar-first)."""
    n = orient_mats.shape[0]
    quats = np.zeros((n, 4))
    for i in range(n):
        quats[i] = OrientMat2Quat(orient_mats[i].tolist())
    return quats


def _misorientation_angle(quat1, quat2, nr_sym, sym):
    """Compute misorientation angle in degrees between two quaternions."""
    q1FR = BringDownToFundamentalRegionSym(list(quat1), nr_sym, sym)
    q2FR = BringDownToFundamentalRegionSym(list(quat2), nr_sym, sym)
    q1FR[0] = -q1FR[0]
    QP = QuaternionProduct(q1FR, q2FR)
    MisV = BringDownToFundamentalRegionSym(QP, nr_sym, sym)
    if MisV[0] > 1:
        MisV[0] = 1
    angle_rad = 2.0 * acos(MisV[0])
    return angle_rad * rad2deg


# ===================================================================
#  Aggregation
# ===================================================================

def aggregate_grains(
    grains_files: list,
    beam_thickness: float = 0.0,
    offset: tuple = (0.0, 0.0, 0.0),
    offset_table: str = None,
    affine_transform: np.ndarray = None,
) -> np.ndarray:
    """Load and aggregate grains from multiple layer Grains.csv files.

    Each layer's Z positions are shifted by ``layer_index * beam_thickness``.
    An optional per-grain offset table or affine transform is applied.

    Args:
        grains_files: List of paths to Grains.csv (or dirs containing them).
        beam_thickness: Vertical step between layers (µm).
        offset: Global (dx, dy, dz) translation applied to all grains.
        offset_table: Path to CSV with columns ``layer,grainID,dx,dy,dz``.
        affine_transform: 3×4 numpy array [R|t] applied as x' = R@x + t.

    Returns:
        2-D array with columns: [UniqueID, LayerNr, OrigGrainID, <orient 9>,
        X, Y, Z, GrainRadius, Quat0..3] = 19 columns total.
    """
    # Parse per-grain offsets if provided
    per_grain_offsets = {}
    if offset_table and os.path.exists(offset_table):
        ot = np.genfromtxt(offset_table, delimiter=',', skip_header=1)
        if ot.ndim == 1:
            ot = ot.reshape(1, -1)
        for row in ot:
            layer_nr = int(row[0])
            grain_id = int(row[1])
            per_grain_offsets[(layer_nr, grain_id)] = row[2:5]
        logger.info(f"Loaded {len(per_grain_offsets)} per-grain offsets")

    rows = []
    unique_id = 1
    for layer_nr, path in enumerate(grains_files):
        # Resolve directory to Grains.csv
        if os.path.isdir(path):
            path = os.path.join(path, 'Grains.csv')
        data = load_grains(path)

        for row in data:
            grain_id = int(row[0])
            orient = row[1:10]  # 3x3 orientation matrix (row-major)
            pos = row[10:13].copy()  # X, Y, Z
            grain_size = row[42] if row.shape[0] > 42 else 0.0

            # Stack layers vertically
            pos[2] += layer_nr * beam_thickness

            # Apply global offset
            pos[0] += offset[0]
            pos[1] += offset[1]
            pos[2] += offset[2]

            # Apply per-grain offset
            key = (layer_nr, grain_id)
            if key in per_grain_offsets:
                pos += per_grain_offsets[key]

            # Apply affine transform: x' = A @ x + t
            if affine_transform is not None:
                A = affine_transform[:, :3]
                t = affine_transform[:, 3]
                pos = A @ pos + t

            quat = OrientMat2Quat(orient.tolist())

            rows.append([
                unique_id, layer_nr, grain_id,
                *orient, *pos, grain_size, *quat
            ])
            unique_id += 1

    result = np.array(rows)
    logger.info(f"Aggregated {len(rows)} grains from {len(grains_files)} files")
    return result

# Column indices in the aggregated array
_AGG_UID = 0
_AGG_LAYER = 1
_AGG_ORIG_ID = 2
_AGG_ORIENT = slice(3, 12)  # 9 values
_AGG_POS = slice(12, 15)    # X, Y, Z
_AGG_SIZE = 15
_AGG_QUAT = slice(16, 20)   # 4 values


# ===================================================================
#  Matching
# ===================================================================

def compute_cost_matrix(
    state1: np.ndarray,
    state2: np.ndarray,
    sg_nr: int,
    mode: str = 'combined',
    weights: tuple = (1.0, 1.0),
    size_filter: float = 0,
    ref_misorientation: float = 0.0,
) -> np.ndarray:
    """Compute N1×N2 cost matrix for grain matching.

    Args:
        state1, state2: Aggregated grain arrays (from ``aggregate_grains``).
        sg_nr: Space group number (e.g., 225 for FCC).
        mode: ``'orientation'``, ``'position'``, or ``'combined'``.
        weights: (angle_scale_deg, distance_scale_um) for combined mode.
            Cost = misorientation/angle_scale + distance/distance_scale.
        size_filter: If >0, grain size must match within this percentage.
        ref_misorientation: Reference misorientation angle (degrees).
            The cost uses ``|misorientation - ref|`` instead of ``misorientation``.

    Returns:
        N1×N2 cost matrix (float64).
    """
    n1 = state1.shape[0]
    n2 = state2.shape[0]
    cost = np.full((n1, n2), np.inf)

    nr_sym, sym = MakeSymmetries(sg_nr)

    quats1 = state1[:, _AGG_QUAT]
    quats2 = state2[:, _AGG_QUAT]
    pos1 = state1[:, _AGG_POS]
    pos2 = state2[:, _AGG_POS]
    size1 = state1[:, _AGG_SIZE]
    size2 = state2[:, _AGG_SIZE]

    for i in range(n1):
        for j in range(n2):
            # Size filter
            if size_filter > 0:
                if size1[i] > 0 and abs(size1[i] - size2[j]) > size1[i] * 0.01 * size_filter:
                    continue  # leave as inf

            if mode in ('orientation', 'combined'):
                angle = _misorientation_angle(quats1[i], quats2[j], nr_sym, sym)
                angle = abs(angle - ref_misorientation)
            else:
                angle = 0.0

            if mode in ('position', 'combined'):
                diff = pos1[i] - pos2[j]
                dist = np.sqrt(np.sum(diff**2))
            else:
                dist = 0.0

            if mode == 'orientation':
                cost[i, j] = angle
            elif mode == 'position':
                cost[i, j] = dist
            else:  # combined
                cost[i, j] = angle / weights[0] + dist / weights[1]

    return cost


def match_grains(
    state1: np.ndarray,
    state2: np.ndarray,
    sg_nr: int,
    mode: str = 'combined',
    weights: tuple = (1.0, 1.0),
    remove_duplicates: bool = True,
    size_filter: float = 0,
    ref_misorientation: float = 0.0,
) -> dict:
    """Match grains between two states.

    Args:
        state1, state2: Aggregated grain arrays.
        sg_nr, mode, weights, size_filter, ref_misorientation: See ``compute_cost_matrix``.
        remove_duplicates: If True, use Hungarian (optimal 1-to-1). If False, greedy.

    Returns:
        Dictionary with keys:
        - ``matches``: List of (i1, i2, cost, angle_deg, dist_um) tuples.
        - ``unmatched_state1``: Indices of unmatched grains in state1.
        - ``unmatched_state2``: Indices of unmatched grains in state2.
    """
    logger.info(f"Computing cost matrix ({state1.shape[0]} × {state2.shape[0]})...")
    cost = compute_cost_matrix(state1, state2, sg_nr, mode, weights,
                               size_filter, ref_misorientation)

    nr_sym, sym = MakeSymmetries(sg_nr)
    n1, n2 = cost.shape

    if remove_duplicates:
        # Hungarian algorithm (optimal 1-to-1 assignment)
        from scipy.optimize import linear_sum_assignment
        # Replace inf with a large finite value for the solver
        large_val = 1e9
        cost_finite = np.where(np.isinf(cost), large_val, cost)
        row_ind, col_ind = linear_sum_assignment(cost_finite)

        matches = []
        matched1 = set()
        matched2 = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= large_val:
                continue  # not a real match
            angle = _misorientation_angle(
                state1[r, _AGG_QUAT], state2[c, _AGG_QUAT], nr_sym, sym)
            diff = state1[r, _AGG_POS] - state2[c, _AGG_POS]
            dist = np.sqrt(np.sum(diff**2))
            matches.append((r, c, cost[r, c], angle, dist))
            matched1.add(r)
            matched2.add(c)
    else:
        # Greedy: for each grain in state2, find the best match in state1
        matches = []
        matched1 = set()
        matched2 = set()
        for j in range(n2):
            col = cost[:, j]
            best_i = np.argmin(col)
            if np.isinf(col[best_i]):
                continue
            angle = _misorientation_angle(
                state1[best_i, _AGG_QUAT], state2[j, _AGG_QUAT], nr_sym, sym)
            diff = state1[best_i, _AGG_POS] - state2[j, _AGG_POS]
            dist = np.sqrt(np.sum(diff**2))
            matches.append((best_i, j, col[best_i], angle, dist))
            matched1.add(best_i)
            matched2.add(j)

    unmatched1 = sorted(set(range(n1)) - matched1)
    unmatched2 = sorted(set(range(n2)) - matched2)

    logger.info(f"Matched {len(matches)} grain pairs, "
                f"{len(unmatched1)} unmatched in state1, "
                f"{len(unmatched2)} unmatched in state2")

    return {
        'matches': matches,
        'unmatched_state1': unmatched1,
        'unmatched_state2': unmatched2,
    }


# ===================================================================
#  Stitching
# ===================================================================

def stitch_layers(
    grains_files: list,
    beam_thickness: float,
    sg_nr: int,
    misorientation_tol: float = 0.5,
    position_tol: float = 50.0,
) -> np.ndarray:
    """Stitch/merge grains from adjacent layers of the same state.

    Grains from overlapping layer boundaries that have similar orientation
    and position are merged into a single grain.

    Args:
        grains_files: List of Grains.csv paths (one per layer).
        beam_thickness: Vertical step between layers (µm).
        sg_nr: Space group number.
        misorientation_tol: Maximum misorientation (degrees) for merging.
        position_tol: Maximum centroid distance (µm) for merging.

    Returns:
        Aggregated array with duplicates removed.
    """
    agg = aggregate_grains(grains_files, beam_thickness=beam_thickness)
    if agg.shape[0] == 0:
        return agg

    nr_sym, sym = MakeSymmetries(sg_nr)
    n = agg.shape[0]

    # Mark grains to keep (True = keep, False = merged away)
    keep = np.ones(n, dtype=bool)

    # Compare each grain against all later grains
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            # Only consider grains from different layers
            if agg[i, _AGG_LAYER] == agg[j, _AGG_LAYER]:
                continue

            # Position check first (cheap)
            diff = agg[i, _AGG_POS] - agg[j, _AGG_POS]
            dist = np.sqrt(np.sum(diff**2))
            if dist > position_tol:
                continue

            # Orientation check
            angle = _misorientation_angle(
                agg[i, _AGG_QUAT], agg[j, _AGG_QUAT], nr_sym, sym)
            if angle < misorientation_tol:
                # Merge: keep the grain with higher confidence
                # (For simplicity, keep the earlier one and average position)
                agg[i, _AGG_POS] = 0.5 * (agg[i, _AGG_POS] + agg[j, _AGG_POS])
                keep[j] = False
                logger.debug(f"Merged grain {int(agg[j, _AGG_UID])} into "
                             f"{int(agg[i, _AGG_UID])} "
                             f"(angle={angle:.3f}°, dist={dist:.1f}µm)")

    merged = agg[keep]
    logger.info(f"Stitching: {n} → {merged.shape[0]} grains "
                f"({n - merged.shape[0]} duplicates removed)")
    return merged


# ===================================================================
#  Affine from point pairs
# ===================================================================

def fit_affine_from_points(reference_pts: np.ndarray,
                           deformed_pts: np.ndarray) -> np.ndarray:
    """Fit a 3×4 affine transform [R|t] from matched point pairs.

    Solves ``deformed = R @ reference + t`` in least-squares sense.

    Args:
        reference_pts: Nx3 array of reference positions.
        deformed_pts: Nx3 array of deformed positions.

    Returns:
        3×4 affine matrix [R|t].
    """
    n = reference_pts.shape[0]
    # Build system: [x y z 1] @ [R|t]^T = [x' y' z']
    A = np.hstack([reference_pts, np.ones((n, 1))])
    # Solve for each output dim
    result = np.zeros((3, 4))
    for dim in range(3):
        result[dim], _, _, _ = np.linalg.lstsq(A, deformed_pts[:, dim], rcond=None)
    logger.info(f"Fitted affine transform from {n} point pairs")
    return result


# ===================================================================
#  Output
# ===================================================================

def write_output(output_path: str, state1: np.ndarray, state2: np.ndarray,
                 result: dict):
    """Write matching results in MatchGrains.c-compatible TSV format.

    Also writes unmatched grains from both sets.
    """
    header = (
        "UniqueIDState1\tLayerState1\tOrigIDState1\t"
        "UniqueIDState2\tLayerState2\tOrigIDState2\t"
        "Quat0State1\tQuat1State1\tQuat2State1\tQuat3State1\t"
        "Quat0State2\tQuat1State2\tQuat2State2\tQuat3State2\t"
        "XState1\tYState1\tZState1\tXState2\tYState2\tZState2\t"
        "GrainSize1\tGrainSize2\t"
        "CostValue\tMisorientAngle\t"
        "dX\tdY\tdZ\tEuclideanDist\n"
    )

    with open(output_path, 'w') as f:
        f.write(header)

        # Matched pairs
        for i1, i2, cost_val, angle, dist in result['matches']:
            g1 = state1[i1]
            g2 = state2[i2]
            diff = g1[_AGG_POS] - g2[_AGG_POS]
            f.write(f"{int(g1[_AGG_UID])}\t{int(g1[_AGG_LAYER])}\t{int(g1[_AGG_ORIG_ID])}\t")
            f.write(f"{int(g2[_AGG_UID])}\t{int(g2[_AGG_LAYER])}\t{int(g2[_AGG_ORIG_ID])}\t")
            for qi in range(4):
                f.write(f"{g1[16 + qi]:.6f}\t")
            for qi in range(4):
                f.write(f"{g2[16 + qi]:.6f}\t")
            for pi in range(3):
                f.write(f"{g1[12 + pi]:.4f}\t")
            for pi in range(3):
                f.write(f"{g2[12 + pi]:.4f}\t")
            f.write(f"{g1[_AGG_SIZE]:.4f}\t{g2[_AGG_SIZE]:.4f}\t")
            f.write(f"{cost_val:.6f}\t{angle:.6f}\t")
            f.write(f"{diff[0]:.4f}\t{diff[1]:.4f}\t{diff[2]:.4f}\t{dist:.4f}\n")

        # Unmatched state1
        for idx in result['unmatched_state1']:
            g1 = state1[idx]
            f.write(f"{int(g1[_AGG_UID])}\t{int(g1[_AGG_LAYER])}\t{int(g1[_AGG_ORIG_ID])}\t")
            f.write("0\t0\t0\t")
            for qi in range(4):
                f.write(f"{g1[16 + qi]:.6f}\t")
            f.write("0\t0\t0\t0\t")
            for pi in range(3):
                f.write(f"{g1[12 + pi]:.4f}\t")
            f.write("0\t0\t0\t")
            f.write(f"{g1[_AGG_SIZE]:.4f}\t0\t0\t0\t0\t0\t0\t0\n")

        # Unmatched state2
        for idx in result['unmatched_state2']:
            g2 = state2[idx]
            f.write("0\t0\t0\t")
            f.write(f"{int(g2[_AGG_UID])}\t{int(g2[_AGG_LAYER])}\t{int(g2[_AGG_ORIG_ID])}\t")
            f.write("0\t0\t0\t0\t")
            for qi in range(4):
                f.write(f"{g2[16 + qi]:.6f}\t")
            f.write("0\t0\t0\t")
            for pi in range(3):
                f.write(f"{g2[12 + pi]:.4f}\t")
            f.write(f"0\t{g2[_AGG_SIZE]:.4f}\t0\t0\t0\t0\t0\t0\n")

    logger.info(f"Results written to {output_path}")


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Match or stitch FF-HEDM grain reconstructions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Match two load states (single layer each):
  python match_grains.py match \\
    --state1 unloaded/Grains.csv \\
    --state2 loaded/Grains.csv \\
    --space-group 225 --mode combined --weights 2.0 50.0

  # Match multi-layer states:
  python match_grains.py match \\
    --state1 LayerNr_1/Grains.csv LayerNr_2/Grains.csv \\
    --state2 LayerNr_1/Grains.csv LayerNr_2/Grains.csv \\
    --space-group 225 --beam-thickness 200 --mode combined

  # Stitch layers from the same state:
  python match_grains.py stitch \\
    --layers LayerNr_1/Grains.csv LayerNr_2/Grains.csv \\
    --beam-thickness 200 --space-group 225

  # Match with affine deformation correction:
  python match_grains.py match \\
    --state1 unloaded/Grains.csv \\
    --state2 loaded/Grains.csv \\
    --space-group 225 --mode combined \\
    --affine-from-points ref_points.csv def_points.csv
""")

    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- match subcommand ---
    match_parser = subparsers.add_parser('match', help='Match grains between states')
    match_parser.add_argument('--state1', nargs='+', required=True,
                              help='Grains.csv files (or layer dirs) for state 1')
    match_parser.add_argument('--state2', nargs='+', required=True,
                              help='Grains.csv files (or layer dirs) for state 2')
    match_parser.add_argument('--space-group', type=int, required=True,
                              help='Space group number (e.g., 225 for FCC)')
    match_parser.add_argument('--beam-thickness', type=float, default=0,
                              help='Beam thickness for multi-layer stacking (µm)')
    match_parser.add_argument('--mode', choices=['orientation', 'position', 'combined'],
                              default='combined', help='Matching mode (default: combined)')
    match_parser.add_argument('--weights', nargs=2, type=float, default=[1.0, 1.0],
                              metavar=('ANGLE_DEG', 'DIST_UM'),
                              help='Scaling weights for combined mode (degrees, microns)')
    match_parser.add_argument('--offset', nargs=3, type=float, default=[0, 0, 0],
                              metavar=('DX', 'DY', 'DZ'),
                              help='Global offset for state2 (µm)')
    match_parser.add_argument('--offset-table', type=str, default=None,
                              help='CSV with per-grain offsets: layer,grainID,dx,dy,dz')
    match_parser.add_argument('--affine', nargs=12, type=float, default=None,
                              metavar='V', help='3×4 affine matrix (12 values, row-major)')
    match_parser.add_argument('--affine-from-points', nargs=2, type=str, default=None,
                              metavar=('REF_CSV', 'DEF_CSV'),
                              help='Fit affine from matched point pairs')
    match_parser.add_argument('--remove-duplicates', action='store_true', default=True,
                              help='Use Hungarian matching (default: True)')
    match_parser.add_argument('--no-remove-duplicates', action='store_true',
                              help='Use greedy matching (allows many-to-one)')
    match_parser.add_argument('--size-filter', type=float, default=0,
                              help='Grain size must match within this %% (0=disabled)')
    match_parser.add_argument('--ref-misorientation', type=float, default=0,
                              help='Reference misorientation angle (degrees)')
    match_parser.add_argument('--output', type=str, default='MatchedGrains.csv',
                              help='Output file path')
    match_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Enable verbose logging')

    # --- stitch subcommand ---
    stitch_parser = subparsers.add_parser('stitch', help='Stitch layers into one grain list')
    stitch_parser.add_argument('--layers', nargs='+', required=True,
                               help='Grains.csv files (or layer dirs) to stitch')
    stitch_parser.add_argument('--beam-thickness', type=float, required=True,
                               help='Beam thickness between layers (µm)')
    stitch_parser.add_argument('--space-group', type=int, required=True,
                               help='Space group number')
    stitch_parser.add_argument('--misorientation-tol', type=float, default=0.5,
                               help='Max misorientation for merging (degrees)')
    stitch_parser.add_argument('--position-tol', type=float, default=50.0,
                               help='Max centroid distance for merging (µm)')
    stitch_parser.add_argument('--output', type=str, default='StitchedGrains.csv',
                               help='Output file path')
    stitch_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )

    if args.command == 'match':
        # Build affine transform if requested
        affine = None
        if args.affine is not None:
            affine = np.array(args.affine).reshape(3, 4)
        elif args.affine_from_points is not None:
            ref_pts = np.genfromtxt(args.affine_from_points[0], delimiter=',',
                                    skip_header=1)[:, :3]
            def_pts = np.genfromtxt(args.affine_from_points[1], delimiter=',',
                                    skip_header=1)[:, :3]
            affine = fit_affine_from_points(ref_pts, def_pts)

        # Aggregate both states
        state1 = aggregate_grains(
            args.state1, beam_thickness=args.beam_thickness)
        state2 = aggregate_grains(
            args.state2, beam_thickness=args.beam_thickness,
            offset=tuple(args.offset),
            offset_table=args.offset_table,
            affine_transform=affine)

        # Match
        remove_dup = not args.no_remove_duplicates
        result = match_grains(
            state1, state2,
            sg_nr=args.space_group,
            mode=args.mode,
            weights=tuple(args.weights),
            remove_duplicates=remove_dup,
            size_filter=args.size_filter,
            ref_misorientation=args.ref_misorientation)

        write_output(args.output, state1, state2, result)

    elif args.command == 'stitch':
        merged = stitch_layers(
            args.layers,
            beam_thickness=args.beam_thickness,
            sg_nr=args.space_group,
            misorientation_tol=args.misorientation_tol,
            position_tol=args.position_tol)

        # Write stitched output
        header = "UniqueID\tLayerNr\tOrigGrainID\t" + \
                 "\t".join([f"O{i}" for i in range(9)]) + \
                 "\tX\tY\tZ\tGrainSize\tQ0\tQ1\tQ2\tQ3\n"
        with open(args.output, 'w') as f:
            f.write(header)
            for row in merged:
                vals = [f"{v:.6f}" if isinstance(v, float) else str(int(v))
                        for v in row]
                # First 3 are ints
                vals[0] = str(int(row[0]))
                vals[1] = str(int(row[1]))
                vals[2] = str(int(row[2]))
                f.write('\t'.join(vals) + '\n')
        logger.info(f"Stitched {merged.shape[0]} grains → {args.output}")

    print("Done.")


if __name__ == '__main__':
    main()
