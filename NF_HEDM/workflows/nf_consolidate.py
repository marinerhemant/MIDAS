"""
NF-HEDM Consolidated HDF5 — reader + writer module.

Reads the scattered output files from ParseMic (text .mic, binary .map,
.kam, .grod, .grainId, AllMatches.mic, grid.txt) and writes a single
consolidated HDF5.

Usage as library:
    from nf_consolidate import generate_consolidated_hdf5
    generate_consolidated_hdf5(mic_text_path, param_file_text, args)

Usage as CLI:
    python nf_consolidate.py --mic /path/to/MicFileText.mic
"""

import argparse
import glob
import logging
import os
import sys

import h5py
import numpy as np

# Setup path for MIDAS utils — try multiple approaches for robustness
def _find_utils_dir():
    """Find the MIDAS utils/ directory, handling symlinks and install layouts."""
    candidates = []
    # 1. Via realpath (resolves symlinks)
    _here = os.path.dirname(os.path.realpath(__file__))
    candidates.append(os.path.join(_here, '..', '..', 'utils'))
    # 2. Via abspath (no symlink resolution)
    _here2 = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(_here2, '..', '..', 'utils'))
    # 3. Via MIDAS_HOME env var
    midas_home = os.environ.get('MIDAS_HOME', '')
    if midas_home:
        candidates.append(os.path.join(midas_home, 'utils'))
    # 4. Via MIDAS_INSTALL_DIR env var
    midas_install = os.environ.get('MIDAS_INSTALL_DIR', '')
    if midas_install:
        candidates.append(os.path.join(midas_install, 'utils'))
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isdir(c) and os.path.exists(os.path.join(c, 'pipeline_state.py')):
            return c
    # Last resort: return first candidate and let the import error be informative
    return os.path.normpath(candidates[0])

_utils_dir = _find_utils_dir()
sys.path.insert(0, _utils_dir)

from pipeline_state import PipelineH5, COMPRESSION
from version import version_string

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Readers for ParseMic output files
# ──────────────────────────────────────────────────────────────────────

def read_mic_text(mic_path: str) -> np.ndarray:
    """Read a text .mic file (4 header lines, whitespace-delimited columns).

    Column layout (from ParseMic.c WriteMicText):
        0: TriEdgeSize | 1: UpDown | 2: OrientationRowNr
        3: X            | 4: Y
        5: TriEdgeSize2 | 6: OrientationID
        7: Euler0 (rad) | 8: Euler1 (rad) | 9: Euler2 (rad)
        10: Confidence
        11: PhaseNr      | 12: RunTime
        (13: KAM  14: GROD  — if extended by later code)

    Returns:
        (N, C) float64 array.
    """
    return np.genfromtxt(mic_path, skip_header=4)


def read_binary_map(map_path: str):
    """Read a binary .map file written by ParseMic GenerateMap().

    Format:
        4 doubles: xSizeMap, ySizeMap, minXRange, minYRange
        Then 7 * xSizeMap * ySizeMap doubles:
            plane 0: Confidence
            plane 1: Euler0 (rad)
            plane 2: Euler1 (rad)
            plane 3: Euler2 (rad)
            plane 4: OrientationRowNr
            plane 5: PhaseNr
            plane 6: Distance from HEDM voxel

    Returns:
        dict with keys: 'xSize', 'ySize', 'minX', 'minY',
        'orientation' (H, W, 7), 'extent' [minX, maxX, minY, maxY]
    """
    with open(map_path, "rb") as f:
        header = np.fromfile(f, dtype=np.float64, count=4)
        data = np.fromfile(f, dtype=np.float64)

    xs = int(header[0])
    ys = int(header[1])
    min_x = header[2]
    min_y = header[3]
    n = xs * ys

    if data.size < n * 7:
        raise ValueError(
            f"Map file too small: expected {n * 7} values, got {data.size}"
        )

    planes = data[: n * 7].reshape((7, ys, xs))
    return {
        "xSize": xs,
        "ySize": ys,
        "minX": min_x,
        "minY": min_y,
        "orientation": np.transpose(planes, (1, 2, 0)),  # (H, W, 7)
        "extent": [min_x, min_x + xs, min_y, min_y + ys],
    }


def read_single_plane_map(path: str):
    """Read a single-plane binary map (.kam, .grod, .grainId).

    Format: 4 doubles header + xSize*ySize doubles.

    Returns:
        (H, W) float64 array, or None if file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.float64, count=4)
        data = np.fromfile(f, dtype=np.float64)
    xs = int(header[0])
    ys = int(header[1])
    n = xs * ys
    if data.size < n:
        logger.warning(f"Single-plane map {path} too small: {data.size} < {n}")
        return None
    return data[:n].reshape((ys, xs))


def read_all_matches(mic_path: str) -> np.ndarray:
    """Read AllMatches file (text file, 4 header lines, many columns).

    Returns:
        (N, C) float64 array, or None if file doesn't exist.
    """
    # Try direct append first: stem.mic.0 -> stem.mic.0.AllMatches
    all_matches_path = mic_path + ".AllMatches"
    if not os.path.exists(all_matches_path):
        # Try replace-based path: stem.mic -> stem_AllMatches.mic
        all_matches_path = mic_path.replace(".mic", "_AllMatches.mic")
    if not os.path.exists(all_matches_path):
        # Try alternate naming
        base_dir = os.path.dirname(mic_path)
        alt = os.path.join(base_dir, "AllMatches.mic")
        if os.path.exists(alt):
            all_matches_path = alt
        else:
            logger.warning(f"AllMatches file not found for {mic_path}")
            return None
    return np.genfromtxt(all_matches_path, skip_header=4)


def read_grid(grid_path: str) -> np.ndarray:
    """Read grid.txt file.

    Returns:
        (N, 4+) float64 array, or None if not found.
    """
    if not os.path.exists(grid_path):
        return None
    return np.genfromtxt(grid_path, skip_header=1)


# ──────────────────────────────────────────────────────────────────────
#  Parameter extraction
# ──────────────────────────────────────────────────────────────────────

def extract_nf_params(param_text: str) -> dict:
    """Extract key NF parameters from parameter file text.

    Returns dict with keys like SpaceGroupNr, LatticeConstant, etc.
    """
    params = {}
    for line in param_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].rstrip(";")
        val = " ".join(parts[1:]).rstrip(";")
        params[key] = val

    result = {}

    # Space group
    for k in ("SpaceGroup", "SpaceGroupNr", "SGNr"):
        if k in params:
            result["SpaceGroupNr"] = int(params[k])
            break

    # Lattice constant
    if "LatticeConstant" in params:
        result["LatticeConstant"] = np.array(
            [float(x) for x in params["LatticeConstant"].split()[:6]]
        )

    # Grid size
    for k in ("GridSize", "GridSizeGrid"):
        if k in params:
            result["GridSize"] = float(params[k])
            break

    # Global position
    if "GlobalPosition" in params:
        result["GlobalPosition"] = float(params["GlobalPosition"])

    # GB angle
    if "GBAngle" in params:
        result["GBAngle"] = float(params["GBAngle"])

    # NumPhases, PhaseNr
    for k in ("NumPhases", "PhaseNr", "nSaves"):
        if k in params:
            result[k] = int(params[k])

    return result


# ──────────────────────────────────────────────────────────────────────
#  Grain aggregation
# ──────────────────────────────────────────────────────────────────────

def aggregate_grains(mic_data: np.ndarray) -> dict:
    """Compute per-grain aggregated data from mic voxel data.

    Args:
        mic_data: (N, C) array from read_mic_text(). Expects columns:
            3=X, 4=Y, 6=OrientationID, 7/8/9=Euler, 10=Confidence

    Returns:
        dict with grain_id, mean_euler, mean_position, mean_confidence,
        num_voxels arrays.
    """
    if mic_data is None or mic_data.size == 0:
        return {}

    # Filter out zero-confidence voxels
    valid = mic_data[:, 10] > 0
    data = mic_data[valid]

    if data.size == 0:
        return {}

    grain_ids = np.unique(data[:, 6])
    grain_ids = grain_ids[grain_ids >= 0]  # exclude -1 or invalid

    n_grains = len(grain_ids)
    result = {
        "grain_id": grain_ids.astype(np.int32),
        "mean_euler_angles": np.zeros((n_grains, 3)),
        "mean_position": np.zeros((n_grains, 2)),
        "mean_confidence": np.zeros(n_grains),
        "num_voxels": np.zeros(n_grains, dtype=np.int32),
    }

    for i, gid in enumerate(grain_ids):
        mask = data[:, 6] == gid
        g = data[mask]
        result["mean_euler_angles"][i] = g[:, 7:10].mean(axis=0)
        result["mean_position"][i] = g[:, 3:5].mean(axis=0)
        result["mean_confidence"][i] = g[:, 10].mean()
        result["num_voxels"][i] = mask.sum()

    return result


# ──────────────────────────────────────────────────────────────────────
#  Main consolidation function
# ──────────────────────────────────────────────────────────────────────

def generate_consolidated_hdf5(
    mic_text_path: str,
    param_text: str = "",
    args_namespace=None,
    output_path: str = None,
    resolution_label: str = None,
):
    """Generate a consolidated HDF5 from NF-HEDM outputs.

    Args:
        mic_text_path:   Path to the text .mic file (e.g. MicFileText.mic).
        param_text:      Full text of the parameter file (for provenance).
        args_namespace:  argparse.Namespace from the workflow (for restart).
        output_path:     Output H5 path. Default: {mic_text_path}_consolidated.h5
        resolution_label: If set, store under /multi_resolution/{label}/ instead
                         of the root /voxels/.

    Returns:
        Path to the created H5 file.
    """
    if not os.path.exists(mic_text_path):
        raise FileNotFoundError(f"Mic file not found: {mic_text_path}")

    if output_path is None:
        base = os.path.splitext(mic_text_path)[0]
        output_path = base + "_consolidated.h5"

    mic_dir = os.path.dirname(os.path.abspath(mic_text_path))
    mic_base = os.path.splitext(mic_text_path)[0]

    # Determine target group in H5
    if resolution_label:
        voxel_prefix = f"multi_resolution/{resolution_label}/voxels"
    else:
        voxel_prefix = "voxels"

    logger.info(f"Generating consolidated H5: {output_path}")

    with PipelineH5(output_path, "nf_midas", args_namespace, param_text) as ph5:

        # ── Parameters ──
        nf_params = extract_nf_params(param_text)
        for key, val in nf_params.items():
            ph5.write_dataset(f"parameters/{key}", val)

        # ── Read mic text ──
        mic_data = read_mic_text(mic_text_path)
        if mic_data is not None and mic_data.size > 0:
            n_cols = mic_data.shape[1]

            ph5.write_dataset(f"{voxel_prefix}/position", mic_data[:, 3:5])
            ph5.write_dataset(f"{voxel_prefix}/euler_angles", mic_data[:, 7:10])
            ph5.write_dataset(f"{voxel_prefix}/confidence", mic_data[:, 10])

            if n_cols > 2:
                ph5.write_dataset(
                    f"{voxel_prefix}/orientation_row_nr", mic_data[:, 2]
                )
            if n_cols > 6:
                ph5.write_dataset(
                    f"{voxel_prefix}/orientation_id", mic_data[:, 6]
                )
            if n_cols > 0:
                ph5.write_dataset(
                    f"{voxel_prefix}/tri_edge_size", mic_data[:, 0]
                )
            if n_cols > 1:
                ph5.write_dataset(f"{voxel_prefix}/up_down", mic_data[:, 1])
            if n_cols > 11:
                ph5.write_dataset(
                    f"{voxel_prefix}/phase_nr", mic_data[:, 11]
                )
            if n_cols > 12:
                ph5.write_dataset(f"{voxel_prefix}/run_time", mic_data[:, 12])

            logger.info(f"Wrote {mic_data.shape[0]} voxels to /{voxel_prefix}/")

        # ── Grain aggregation (only for root voxels) ──
        if not resolution_label and mic_data is not None:
            grains = aggregate_grains(mic_data)
            if grains:
                for key, val in grains.items():
                    ph5.write_dataset(f"grains/{key}", val)
                logger.info(
                    f"Wrote {len(grains['grain_id'])} grains to /grains/"
                )
                # Reserve strain group
                ph5.h5.require_group("grains/strain")
                ph5.h5["grains/strain"].attrs["status"] = "reserved"

        # ── Binary maps ──
        # Try mic_text_path + ".map" first (for files like "stem.mic.0" -> "stem.mic.0.map")
        # Fall back to splitext-based path (for "stem.mic" -> "stem.map")
        map_path = mic_text_path + ".map"
        if not os.path.exists(map_path):
            map_path = mic_base + ".map"
        if os.path.exists(map_path):
            map_data = read_binary_map(map_path)
            prefix = "maps" if not resolution_label else f"multi_resolution/{resolution_label}/maps"
            ph5.write_dataset(f"{prefix}/orientation", map_data["orientation"])
            ph5.write_dataset(
                f"{prefix}/extent",
                np.array(map_data["extent"]),
            )

            # KAM, GrainID, GROD
            for ext, name in [(".kam", "kam"), (".grod", "grod"), (".grainId", "grain_id")]:
                plane = read_single_plane_map(map_path + ext)
                if plane is not None:
                    ph5.write_dataset(f"{prefix}/{name}", plane)

            logger.info(f"Wrote maps ({map_data['xSize']}x{map_data['ySize']})")

        # ── AllMatches ──
        if not resolution_label:
            all_matches = read_all_matches(mic_text_path)
            if all_matches is not None and all_matches.size > 0:
                ph5.write_dataset("all_matches/data", all_matches)
                logger.info(
                    f"Wrote AllMatches: {all_matches.shape}"
                )

        # ── Grid ──
        grid_path = os.path.join(mic_dir, "grid.txt")
        if not resolution_label and os.path.exists(grid_path):
            grid = read_grid(grid_path)
            if grid is not None:
                ph5.write_dataset("grid/points", grid)
                ph5.write_dataset("grid/num_points", grid.shape[0])

        ph5.mark("consolidation")

    logger.info(f"Consolidated H5 saved: {output_path}")
    return output_path


def add_resolution_to_h5(
    h5_path: str,
    mic_text_path: str,
    resolution_label: str,
    grid_size: float = 0.0,
    pass_type: str = "unseeded",
):
    """Add a resolution loop's data to an existing consolidated H5.

    Used by nf_MIDAS_Multiple_Resolutions to store per-loop results.

    Args:
        h5_path:          Path to existing consolidated H5.
        mic_text_path:    Path to text .mic file for this resolution.
        resolution_label: Label like "loop_0", "loop_1_seeded", etc.
        grid_size:        Grid size for this resolution.
        pass_type:        "unseeded", "seeded", or "merged".
    """
    if not os.path.exists(mic_text_path):
        logger.warning(f"Mic file not found for resolution {resolution_label}: {mic_text_path}")
        return

    mic_data = read_mic_text(mic_text_path)
    mic_base = os.path.splitext(mic_text_path)[0]
    prefix = f"multi_resolution/{resolution_label}"

    with h5py.File(h5_path, "a") as h5:
        grp = h5.require_group(prefix)
        grp.attrs["grid_size"] = grid_size
        grp.attrs["pass_type"] = pass_type

        vp = f"{prefix}/voxels"

        # Write voxel data
        if mic_data is not None and mic_data.size > 0:
            for name in ["position", "euler_angles", "confidence",
                         "orientation_row_nr", "orientation_id",
                         "tri_edge_size", "up_down", "phase_nr", "run_time"]:
                ds_path = f"{vp}/{name}"
                if ds_path in h5:
                    del h5[ds_path]

            h5.create_dataset(f"{vp}/position", data=mic_data[:, 3:5], **COMPRESSION)
            h5.create_dataset(f"{vp}/euler_angles", data=mic_data[:, 7:10], **COMPRESSION)
            h5.create_dataset(f"{vp}/confidence", data=mic_data[:, 10], **COMPRESSION)
            if mic_data.shape[1] > 2:
                h5.create_dataset(f"{vp}/orientation_row_nr", data=mic_data[:, 2], **COMPRESSION)
            if mic_data.shape[1] > 6:
                h5.create_dataset(f"{vp}/orientation_id", data=mic_data[:, 6], **COMPRESSION)
            if mic_data.shape[1] > 0:
                h5.create_dataset(f"{vp}/tri_edge_size", data=mic_data[:, 0], **COMPRESSION)
            if mic_data.shape[1] > 1:
                h5.create_dataset(f"{vp}/up_down", data=mic_data[:, 1], **COMPRESSION)
            if mic_data.shape[1] > 11:
                h5.create_dataset(f"{vp}/phase_nr", data=mic_data[:, 11], **COMPRESSION)
            if mic_data.shape[1] > 12:
                h5.create_dataset(f"{vp}/run_time", data=mic_data[:, 12], **COMPRESSION)

        # Binary map — mic files like "stem.mic.1" have maps at "stem.mic.1.map"
        map_path = mic_text_path + ".map"
        if os.path.exists(map_path):
            map_data = read_binary_map(map_path)
            mp = f"{prefix}/maps"
            for k in ["orientation", "extent", "kam", "grod", "grain_id"]:
                if f"{mp}/{k}" in h5:
                    del h5[f"{mp}/{k}"]
            h5.create_dataset(f"{mp}/orientation", data=map_data["orientation"], **COMPRESSION)
            h5.create_dataset(f"{mp}/extent", data=np.array(map_data["extent"]))
            for ext, name in [(".kam", "kam"), (".grod", "grod"), (".grainId", "grain_id")]:
                plane = read_single_plane_map(map_path + ext)
                if plane is not None:
                    h5.create_dataset(f"{mp}/{name}", data=plane, **COMPRESSION)

    logger.info(f"Added resolution {resolution_label} to {h5_path}")


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate consolidated HDF5 from NF-HEDM outputs."
    )
    parser.add_argument(
        "--mic", required=True, help="Path to text .mic file (for single mode: the mic file; "
        "for --multi-res: the base mic file, e.g. holder3_txt.mic.0)"
    )
    parser.add_argument(
        "--params", default="", help="Path to parameter file (for provenance)"
    )
    parser.add_argument(
        "--output", default=None, help="Output H5 path (default: {mic_base}_consolidated.h5)"
    )
    parser.add_argument(
        "--multi-res", action="store_true",
        help="Auto-discover and consolidate all multi-resolution mic files "
        "(e.g. .mic.0, .mic.1, _all_solutions.1, _merged.1, etc.)"
    )
    args = parser.parse_args()

    param_text = ""
    if args.params and os.path.exists(args.params):
        with open(args.params) as f:
            param_text = f.read()

    if not args.multi_res:
        # Single-file consolidation
        h5_path = generate_consolidated_hdf5(
            mic_text_path=args.mic,
            param_text=param_text,
            args_namespace=vars(args),
            output_path=args.output,
        )
        print(f"Consolidated H5: {h5_path}")
    else:
        # Multi-resolution: auto-discover all mic files
        import re
        mic_path = os.path.abspath(args.mic)
        mic_dir = os.path.dirname(mic_path)

        # Derive base name: strip .mic.N suffix -> base
        # e.g. /path/holder3_txt.mic.0 -> holder3_txt.mic
        base_match = re.match(r'^(.+\.mic)\.\d+$', mic_path)
        if base_match:
            mic_base = base_match.group(1)  # e.g. /path/holder3_txt.mic
        else:
            mic_base = os.path.splitext(mic_path)[0] + ".mic"

        mic_stem = os.path.basename(mic_base)  # e.g. holder3_txt.mic
        # Strip ".mic" for pattern matching
        stem_no_ext = mic_stem.rsplit('.mic', 1)[0]  # e.g. holder3_txt

        output_path = args.output or os.path.join(
            mic_dir, f"{stem_no_ext}_consolidated.h5"
        )

        # 1. Create consolidated H5 from loop 0 (the root)
        loop0_mic = mic_path  # The --mic argument is the .mic.0 file
        if not os.path.exists(loop0_mic):
            print(f"Error: Loop 0 mic file not found: {loop0_mic}")
            sys.exit(1)

        print(f"Creating consolidated H5 from loop 0: {loop0_mic}")
        h5_path = generate_consolidated_hdf5(
            mic_text_path=loop0_mic,
            param_text=param_text,
            args_namespace=vars(args),
            output_path=output_path,
        )

        # Also add as loop_0_unseeded resolution
        add_resolution_to_h5(
            h5_path=h5_path,
            mic_text_path=loop0_mic,
            resolution_label="loop_0_unseeded",
            grid_size=0.0,
            pass_type="unseeded",
        )

        # 2. Auto-discover higher resolution loops
        # Pattern: stem.mic.N (seeded), stem.mic_all_solutions.N (unseeded),
        #          stem.mic_merged.N (merged)
        loop_idx = 1
        while True:
            seeded_mic = os.path.join(mic_dir, f"{mic_stem}.{loop_idx}")
            unseeded_mic = os.path.join(mic_dir, f"{stem_no_ext}_all_solutions.{loop_idx}")
            merged_mic = os.path.join(mic_dir, f"{stem_no_ext}_merged.{loop_idx}")

            has_any = any(os.path.exists(f) for f in [seeded_mic, unseeded_mic, merged_mic])
            if not has_any:
                break

            for mic_file, label, ptype in [
                (seeded_mic, f"loop_{loop_idx}_seeded", "seeded"),
                (unseeded_mic, f"loop_{loop_idx}_unseeded", "unseeded"),
                (merged_mic, f"loop_{loop_idx}_merged", "merged"),
            ]:
                if os.path.exists(mic_file):
                    print(f"  Adding {label}: {os.path.basename(mic_file)}")
                    try:
                        add_resolution_to_h5(
                            h5_path=h5_path,
                            mic_text_path=mic_file,
                            resolution_label=label,
                            grid_size=0.0,
                            pass_type=ptype,
                        )
                    except Exception as e:
                        print(f"  ERROR adding {label}: {e}")

            loop_idx += 1

        print(f"\nConsolidated H5 with {loop_idx - 1} resolution loops: {h5_path}")


# MIDAS version banner
try:
    print(version_string())
except Exception:
    pass

if __name__ == "__main__":
    main()
