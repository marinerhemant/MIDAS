"""NF-HEDM consolidated-HDF5 reader + writer.

Pip-installable port of ``NF_HEDM/workflows/nf_consolidate.py``: reads
the scattered output files from :mod:`parse_mic` (text ``.mic``,
binary ``.map`` / ``.kam`` / ``.grod`` / ``.grainId``, AllMatches
text, ``grid.txt``) and writes a single consolidated HDF5 with
provenance + pipeline-state metadata via :class:`state.PipelineH5`.

Usage as library::

    from midas_nf_pipeline.consolidate import generate_consolidated_hdf5
    generate_consolidated_hdf5(mic_text_path, param_text, args)

Usage as CLI (via the umbrella ``midas-nf-pipeline consolidate`` subcommand).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .state import PipelineH5, COMPRESSION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  ParseMic output readers
# ---------------------------------------------------------------------------

def read_tri_edge_size(mic_path: str | Path) -> float:
    """Pull ``TriEdgeSize`` from the first header line of a text mic file."""
    try:
        with open(mic_path, "r") as f:
            line = f.readline().strip()
            if line.startswith("%TriEdgeSize"):
                return float(line.split()[1])
    except Exception:
        pass
    return 0.0


def read_mic_text(mic_path: str | Path) -> np.ndarray:
    """Load a text ``.mic`` file (4 ``%`` header lines, then voxel rows)."""
    return np.genfromtxt(str(mic_path), skip_header=4)


def read_binary_map(map_path: str | Path):
    """Load a ``.map`` binary written by ParseMic GenerateMap.

    Layout:
      4 doubles header ``[xSize, ySize, minX, minY]`` + 7 planes of
      ``xSize * ySize`` doubles (Confidence, Eul1, Eul2, Eul3,
      OrientationRowNr, PhaseNr, Distance).
    """
    with open(map_path, "rb") as f:
        header = np.fromfile(f, dtype=np.float64, count=4)
        data = np.fromfile(f, dtype=np.float64)
    xs = int(header[0]); ys = int(header[1])
    min_x = float(header[2]); min_y = float(header[3])
    n = xs * ys
    if data.size < n * 7:
        raise ValueError(
            f"Map {map_path}: expected {n * 7} values, got {data.size}"
        )
    planes = data[: n * 7].reshape((7, ys, xs))
    return {
        "xSize": xs, "ySize": ys, "minX": min_x, "minY": min_y,
        "orientation": np.transpose(planes, (1, 2, 0)),
        "extent": [min_x, min_x + xs, min_y, min_y + ys],
    }


def read_single_plane_map(path: str | Path):
    """Single-plane binary map (``.kam``, ``.grod``, ``.grainId``).
    Returns ``(H, W) float64`` or ``None`` if the file is missing.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.float64, count=4)
        data = np.fromfile(f, dtype=np.float64)
    xs = int(header[0]); ys = int(header[1])
    n = xs * ys
    if data.size < n:
        logger.warning(f"Single-plane map {path}: too small ({data.size} < {n})")
        return None
    return data[:n].reshape((ys, xs))


def read_all_matches(mic_path: str | Path) -> Optional[np.ndarray]:
    p = str(mic_path)
    candidates = [p + ".AllMatches", p.replace(".mic", "_AllMatches.mic"),
                  os.path.join(os.path.dirname(p), "AllMatches.mic")]
    for c in candidates:
        if os.path.exists(c):
            return np.genfromtxt(c, skip_header=4)
    return None


def read_grid(grid_path: str | Path) -> Optional[np.ndarray]:
    if not os.path.exists(grid_path):
        return None
    return np.genfromtxt(str(grid_path), skip_header=1)


# ---------------------------------------------------------------------------
#  Parameter extraction
# ---------------------------------------------------------------------------

def extract_nf_params(param_text: str) -> dict:
    # Read only the leading value token(s) of each key, the same rule as
    # params.parse_parameters, so a trailing comment ("GridSize 5 # demo")
    # is ignored here too instead of reaching float() and crashing consolidate.
    raw: dict = {}
    for line in param_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [t.rstrip(";") for t in line.split()]
        if len(parts) < 2 or not parts[1]:
            continue
        raw[parts[0]] = parts[1:]

    out: dict = {}
    for k in ("SpaceGroup", "SpaceGroupNr", "SGNr"):
        if k in raw:
            out["SpaceGroupNr"] = int(raw[k][0])
            break
    if "LatticeConstant" in raw:
        out["LatticeConstant"] = np.array(
            [float(x) for x in raw["LatticeConstant"][:6]]
        )
    for k in ("GridSize", "GridSizeGrid"):
        if k in raw:
            out["GridSize"] = float(raw[k][0])
            break
    if "GlobalPosition" in raw:
        out["GlobalPosition"] = float(raw["GlobalPosition"][0])
    if "GBAngle" in raw:
        out["GBAngle"] = float(raw["GBAngle"][0])
    for k in ("NumPhases", "PhaseNr", "nSaves"):
        if k in raw:
            out[k] = int(raw[k][0])
    return out


# ---------------------------------------------------------------------------
#  Per-grain aggregation
# ---------------------------------------------------------------------------

def aggregate_grains(mic_data: np.ndarray) -> dict:
    """Compute per-grain mean orientation, position, confidence + voxel count."""
    if mic_data is None or mic_data.size == 0:
        return {}
    valid = mic_data[:, 10] > 0
    data = mic_data[valid]
    if data.size == 0:
        return {}
    gids = np.unique(data[:, 6])
    gids = gids[gids >= 0]
    n = len(gids)
    out = {
        "grain_id": gids.astype(np.int32),
        "mean_euler_angles": np.zeros((n, 3)),
        "mean_position": np.zeros((n, 2)),
        "mean_confidence": np.zeros(n),
        "num_voxels": np.zeros(n, dtype=np.int32),
    }
    for i, gid in enumerate(gids):
        mask = data[:, 6] == gid
        g = data[mask]
        out["mean_euler_angles"][i] = g[:, 7:10].mean(axis=0)
        out["mean_position"][i] = g[:, 3:5].mean(axis=0)
        out["mean_confidence"][i] = g[:, 10].mean()
        out["num_voxels"][i] = int(mask.sum())
    return out


# ---------------------------------------------------------------------------
#  Top-level consolidator
# ---------------------------------------------------------------------------

def generate_consolidated_hdf5(
    mic_text_path: str | Path,
    param_text: str = "",
    args_namespace=None,
    output_path: str | Path | None = None,
    resolution_label: Optional[str] = None,
) -> str:
    """Generate (or update) a consolidated HDF5 from one ParseMic output set.

    Parameters
    ----------
    mic_text_path : str
        Path to the text ``.mic`` file (e.g. ``MicFileText.mic``).
    param_text : str
        Full parameter-file text (stored in ``/provenance/parameter_file``).
    args_namespace : argparse.Namespace or dict, optional
        CLI args, JSON-serialised into the H5 for restart support.
    output_path : str, optional
        Default: ``<mic_text_path stem>_consolidated.h5``.
    resolution_label : str, optional
        If set, voxel/map data is stored under
        ``/multi_resolution/<label>/`` instead of the root.
    """
    mic_text_path = str(mic_text_path)
    if not os.path.exists(mic_text_path):
        raise FileNotFoundError(f"Mic file not found: {mic_text_path}")

    if output_path is None:
        base = os.path.splitext(mic_text_path)[0]
        output_path = base + "_consolidated.h5"
    output_path = str(output_path)

    mic_dir = os.path.dirname(os.path.abspath(mic_text_path))
    mic_base = os.path.splitext(mic_text_path)[0]
    voxel_prefix = (
        f"multi_resolution/{resolution_label}/voxels"
        if resolution_label else "voxels"
    )
    map_prefix = (
        f"multi_resolution/{resolution_label}/maps"
        if resolution_label else "maps"
    )

    logger.info(f"Generating consolidated H5: {output_path}")
    with PipelineH5(output_path, "nf_midas", args_namespace, param_text) as ph5:
        for k, v in extract_nf_params(param_text).items():
            ph5.write_dataset(f"parameters/{k}", v)

        mic_data = read_mic_text(mic_text_path)
        if mic_data is not None and mic_data.size > 0:
            ph5.write_dataset(f"{voxel_prefix}/position", mic_data[:, 3:5])
            ph5.write_dataset(f"{voxel_prefix}/euler_angles", mic_data[:, 7:10])
            ph5.write_dataset(f"{voxel_prefix}/confidence", mic_data[:, 10])
            n_cols = mic_data.shape[1]
            if n_cols > 2:
                ph5.write_dataset(f"{voxel_prefix}/orientation_row_nr", mic_data[:, 2])
            if n_cols > 6:
                ph5.write_dataset(f"{voxel_prefix}/orientation_id", mic_data[:, 6])
            if n_cols > 0:
                ph5.write_dataset(f"{voxel_prefix}/tri_edge_size", mic_data[:, 0])
            if n_cols > 1:
                ph5.write_dataset(f"{voxel_prefix}/up_down", mic_data[:, 1])
            if n_cols > 11:
                ph5.write_dataset(f"{voxel_prefix}/phase_nr", mic_data[:, 11])
            if n_cols > 12:
                ph5.write_dataset(f"{voxel_prefix}/run_time", mic_data[:, 12])
            logger.info(f"Wrote {mic_data.shape[0]} voxels to /{voxel_prefix}/")

        if not resolution_label and mic_data is not None:
            grains = aggregate_grains(mic_data)
            if grains:
                for k, v in grains.items():
                    ph5.write_dataset(f"grains/{k}", v)
                ph5.h5.require_group("grains/strain").attrs["status"] = "reserved"
                logger.info(f"Wrote {len(grains['grain_id'])} grains to /grains/")

        # ── binary maps (.map / .kam / .grod / .grainId) ──
        map_path = mic_text_path + ".map"
        if not os.path.exists(map_path):
            map_path = mic_base + ".map"
        if os.path.exists(map_path):
            md = read_binary_map(map_path)
            ph5.write_dataset(f"{map_prefix}/orientation", md["orientation"])
            ph5.write_dataset(f"{map_prefix}/extent", np.array(md["extent"]))
            for ext, name in [(".kam", "kam"), (".grod", "grod"),
                              (".grainId", "grain_id")]:
                plane = read_single_plane_map(map_path + ext)
                if plane is not None:
                    ph5.write_dataset(f"{map_prefix}/{name}", plane)
            logger.info(f"Wrote maps ({md['xSize']}x{md['ySize']})")

        if not resolution_label:
            am = read_all_matches(mic_text_path)
            if am is not None and am.size > 0:
                ph5.write_dataset("all_matches/data", am)
                logger.info(f"Wrote AllMatches: {am.shape}")

        grid_path = os.path.join(mic_dir, "grid.txt")
        if not resolution_label and os.path.exists(grid_path):
            grid = read_grid(grid_path)
            if grid is not None:
                ph5.write_dataset("grid/points", grid)
                ph5.write_dataset("grid/num_points", int(grid.shape[0]))

        ph5.mark("consolidation")

    logger.info(f"Consolidated H5 saved: {output_path}")
    return output_path


def add_resolution_to_h5(
    h5_path: str | Path,
    mic_text_path: str | Path,
    resolution_label: str,
    grid_size: float = 0.0,
    pass_type: str = "unseeded",
) -> None:
    """Append a resolution-loop's data to an existing consolidated H5."""
    mic_text_path = str(mic_text_path)
    if not os.path.exists(mic_text_path):
        logger.warning(f"Mic file missing for {resolution_label}: {mic_text_path}")
        return
    if grid_size == 0.0:
        grid_size = read_tri_edge_size(mic_text_path)
    mic_data = read_mic_text(mic_text_path)
    prefix = f"multi_resolution/{resolution_label}"
    vp = f"{prefix}/voxels"

    with h5py.File(str(h5_path), "a") as h5:
        grp = h5.require_group(prefix)
        grp.attrs["grid_size"] = grid_size
        grp.attrs["pass_type"] = pass_type

        if mic_data is not None and mic_data.size > 0:
            for nm in ("position", "euler_angles", "confidence",
                       "orientation_row_nr", "orientation_id",
                       "tri_edge_size", "up_down", "phase_nr", "run_time"):
                ds = f"{vp}/{nm}"
                if ds in h5:
                    del h5[ds]
            h5.create_dataset(f"{vp}/position", data=mic_data[:, 3:5], **COMPRESSION)
            h5.create_dataset(f"{vp}/euler_angles", data=mic_data[:, 7:10], **COMPRESSION)
            h5.create_dataset(f"{vp}/confidence", data=mic_data[:, 10], **COMPRESSION)
            n_cols = mic_data.shape[1]
            if n_cols > 2:
                h5.create_dataset(f"{vp}/orientation_row_nr", data=mic_data[:, 2], **COMPRESSION)
            if n_cols > 6:
                h5.create_dataset(f"{vp}/orientation_id", data=mic_data[:, 6], **COMPRESSION)
            if n_cols > 0:
                h5.create_dataset(f"{vp}/tri_edge_size", data=mic_data[:, 0], **COMPRESSION)
            if n_cols > 1:
                h5.create_dataset(f"{vp}/up_down", data=mic_data[:, 1], **COMPRESSION)
            if n_cols > 11:
                h5.create_dataset(f"{vp}/phase_nr", data=mic_data[:, 11], **COMPRESSION)
            if n_cols > 12:
                h5.create_dataset(f"{vp}/run_time", data=mic_data[:, 12], **COMPRESSION)

        map_path = mic_text_path + ".map"
        if os.path.exists(map_path):
            md = read_binary_map(map_path)
            mp = f"{prefix}/maps"
            for k in ("orientation", "extent", "kam", "grod", "grain_id"):
                if f"{mp}/{k}" in h5:
                    del h5[f"{mp}/{k}"]
            h5.create_dataset(f"{mp}/orientation", data=md["orientation"], **COMPRESSION)
            h5.create_dataset(f"{mp}/extent", data=np.array(md["extent"]))
            for ext, name in [(".kam", "kam"), (".grod", "grod"),
                              (".grainId", "grain_id")]:
                plane = read_single_plane_map(map_path + ext)
                if plane is not None:
                    h5.create_dataset(f"{mp}/{name}", data=plane, **COMPRESSION)

    logger.info(f"Added resolution {resolution_label} to {h5_path}")
