"""find_grains — unified replacement for findSingleSolutionPF +
findMultipleSolutionsPF.

Two C binaries are retired here:

  - ``FF_HEDM/src/findSingleSolutionPFRefactored.c`` (3014 lines): the
    per-voxel best-orientation pick + cross-voxel dedup + sinogram
    assembly + patch extraction. Replaced by
    :func:`find_grains_single`.
  - ``FF_HEDM/src/findMultipleSolutionsPF.c` (221 lines): per-voxel
    clustering output as ``SpotsToIndex.csv`` (no sinogen, no global
    dedup). Replaced by :func:`find_grains_multiple`.

Both share the per-voxel clustering primitive (:func:`._cluster.per_voxel_cluster`)
and the consolidated-file readers (:mod:`._consolidation_io`). They
differ only in what they output downstream.

Public API
----------

  - :func:`find_grains_single` — replaces findSingleSolutionPF.
  - :func:`find_grains_multiple` — replaces findMultipleSolutionsPF.

Inputs (under ``work_dir/Output/``):
  - ``IndexBest_all.bin``, ``IndexKey_all.bin``, ``IndexBest_IDs_all.bin``
  - ``Spots.bin`` (single mode)
  - ``positions.csv`` (single mode, optional but required for the
    scan-position consistency filter in indexing-mode sinogen)

Outputs (under ``work_dir/Output/``):
  - ``UniqueOrientations.csv``  (single)
  - ``UniqueIndexSingleKey.bin``  (both, single writes one row per voxel)
  - ``SpotsToIndex.csv``  (multiple only)
  - sinos / omegas / nrHKLs / spotMapping / spotMeta (single only)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from ._cluster import (
    PerVoxelClusterResult,
    GlobalClusterResult,
    global_cluster,
    global_cluster_fast,
    per_voxel_cluster,
    per_voxel_cluster_torch,
)
from ._consolidation_io import (
    CONSOLIDATED_KEY_COLS,
    CONSOLIDATED_VALS_COLS,
    ConsolidatedReader,
    open_all_three,
    open_vals,
    open_keys,
    open_ids,
    write_vals_bin,
    write_keys_bin,
    write_ids_bin,
)
from ._geom import ScanGrid, build_scan_grid, read_positions_csv, voxel_to_xy_um
from ._patches import (
    PATCH_HALF_SIZE,
    PATCH_SIZE,
    extract_patch_from_frame,
    extract_patches_from_spot_map,
)
from ._sinogen import (
    SPOTS_ARRAY_COLS,
    SinogenOutputs,
    apply_variant_torch,
    generate_sinograms_tolerance,
)
from ._sinogen_indexing import generate_sinograms_indexing
from ._spot_association import SpotData, SpotList, process_spots
from .._logging import LOG
from ._voxel_keys import (
    read_unique_index_single_key,
    write_spots_to_index_csv,
    write_unique_index_single_key,
    write_unique_orientations_csv,
)

__all__ = [
    # cluster
    "PerVoxelClusterResult", "GlobalClusterResult",
    "global_cluster", "per_voxel_cluster", "per_voxel_cluster_torch",
    # consolidation io
    "ConsolidatedReader", "CONSOLIDATED_KEY_COLS", "CONSOLIDATED_VALS_COLS",
    "open_all_three", "open_vals", "open_keys", "open_ids",
    "write_vals_bin", "write_keys_bin", "write_ids_bin",
    # geom
    "ScanGrid", "build_scan_grid", "read_positions_csv", "voxel_to_xy_um",
    # patches
    "PATCH_HALF_SIZE", "PATCH_SIZE",
    "extract_patch_from_frame", "extract_patches_from_spot_map",
    # sinogen
    "SPOTS_ARRAY_COLS", "SinogenOutputs",
    "apply_variant_torch", "generate_sinograms_tolerance",
    "generate_sinograms_indexing",
    # spot assoc
    "SpotData", "SpotList", "process_spots",
    # voxel keys
    "read_unique_index_single_key", "write_spots_to_index_csv",
    "write_unique_index_single_key", "write_unique_orientations_csv",
    # top-level API
    "FindGrainsArtifacts", "find_grains_single", "find_grains_multiple",
]


@dataclass
class FindGrainsArtifacts:
    """Bookkeeping for what :func:`find_grains_single` / ``_multiple`` wrote.

    A thin layer below :class:`midas_pipeline.results.FindGrainsResult`.
    """

    unique_orientations_csv: str = ""
    unique_index_single_key_bin: str = ""
    spots_to_index_csv: str = ""
    n_unique_grains: int = 0
    sinogen: Optional[SinogenOutputs] = None
    patches_path: Optional[str] = None
    spot_pos_path: Optional[str] = None


def _read_spots_bin(spots_path: Path) -> np.ndarray:
    """Load ``Spots.bin`` (n_spots × 10 float64) from disk."""
    raw = np.frombuffer(spots_path.read_bytes(), dtype=np.float64)
    if raw.size == 0:
        return np.empty((0, SPOTS_ARRAY_COLS), dtype=np.float64)
    if raw.size % SPOTS_ARRAY_COLS != 0:
        raise ValueError(
            f"Spots.bin size {raw.size} not divisible by {SPOTS_ARRAY_COLS}"
        )
    return raw.reshape(-1, SPOTS_ARRAY_COLS)


def _pervoxel_worker(args):
    """Per-voxel pick-best + within-voxel cluster for voxels ``[v0, v1)``.

    Top-level (picklable) for multiprocessing. Each worker opens its OWN
    consolidated readers (mmap — shared page cache, no copy). Returns a list of
    ``(v, OM(9), conf, SpotID, nMatches, nIDs, best_row)`` for valid voxels, in
    ascending ``v``. Byte-identical to the original serial loop body.
    """
    out_dir, v0, v1, space_group, max_ang_deg = args
    vals_r, keys_r, _ = open_all_three(out_dir)
    rows = []
    for v in range(v0, v1):
        vals_v = vals_r.get_vals(v)
        keys_v = keys_r.get_keys(v)
        if vals_v is None or keys_v is None:
            continue
        if int(vals_r.n_sol_arr[v]) <= 0:
            continue
        denom = vals_v[:, 14]
        with np.errstate(divide="ignore", invalid="ignore"):
            confs = np.where(denom > 0, vals_v[:, 15] / denom, 0.0)
        result = per_voxel_cluster(
            vals_v[:, 2:11], confs, vals_v[:, 1], keys_v,
            space_group=space_group, max_ang_deg=max_ang_deg, min_conf=0.0,
        )
        if result.best_row < 0:
            continue
        br = result.best_row
        rows.append((int(v), vals_v[br, 2:11].copy(), float(confs[br]),
                     int(keys_v[br, 0]), int(keys_v[br, 1]), int(keys_v[br, 2]), int(br)))
    return rows


def _per_voxel_pass(out_dir, n_voxels, space_group, max_ang_deg, n_jobs=1):
    """Per-voxel pass over all voxels, optionally parallel across CPU workers.

    The loop is embarrassingly parallel; for DEFORMED maps with many candidate
    solutions per voxel it dominates find_grains runtime. ``n_jobs > 1`` splits
    voxels into contiguous chunks across a process pool. Result is independent of
    ``n_jobs`` (byte-parity): per-voxel arrays are indexed by ``v`` and the
    single-key rows are re-sorted ascending.
    """
    INVALID_U64 = np.uint64(2 ** 64 - 1)
    per_vox_OMs = np.zeros((n_voxels, 9), dtype=np.float64)
    per_vox_confs = np.zeros(n_voxels, dtype=np.float64)
    per_vox_keys = np.zeros((n_voxels, 4), dtype=np.uint64)
    per_vox_keys[:, 0] = INVALID_U64
    single_key_rows: list[tuple[int, np.ndarray]] = []

    if n_jobs and n_jobs > 1 and n_voxels > 0:
        import multiprocessing as mp
        step = (n_voxels + n_jobs - 1) // n_jobs
        chunks = [(str(out_dir), i, min(i + step, n_voxels), space_group, max_ang_deg)
                  for i in range(0, n_voxels, step)]
        ctx = mp.get_context("fork")
        with ctx.Pool(min(n_jobs, len(chunks))) as pool:
            results = pool.map(_pervoxel_worker, chunks)
    else:
        results = [_pervoxel_worker((str(out_dir), 0, n_voxels, space_group, max_ang_deg))]

    for rows in results:
        for (v, om, conf, k0, k1, k2, br) in rows:
            per_vox_OMs[v] = om
            per_vox_confs[v] = conf
            per_vox_keys[v, 0] = np.uint64(k0)
            per_vox_keys[v, 1] = np.uint64(k1)
            per_vox_keys[v, 2] = np.uint64(k2)
            per_vox_keys[v, 3] = np.uint64(br)
            single_key_rows.append((v, np.array([v, k0, k1, k2, br], dtype=np.uint64)))
    single_key_rows.sort(key=lambda x: x[0])
    return per_vox_OMs, per_vox_confs, per_vox_keys, single_key_rows


def find_grains_single(
    work_dir: str | Path,
    space_group: int,
    *,
    sino_mode: Literal["tolerance", "indexing"] = "tolerance",
    extract_patches: bool = False,
    confidence_min: float = 0.5,
    scan_tolerance_um: float = 1.5,
    cluster_misorientation_deg: float = 1.0,
    tol_ome_deg: float = 1.0,
    tol_eta_deg: float = 1.0,
    n_scans: Optional[int] = None,
    output_subdir: str = "Output",
    normalize_sino: bool = False,
    abs_transform: bool = False,
    # P7: soft sino assembly (tolerance mode only — indexing mode unchanged)
    emit_softsum: bool = False,
    soft_weight_fn=None,
    frame_loader=None,
) -> FindGrainsArtifacts:
    """Replace ``findSingleSolutionPFRefactored.c``.

    Workflow:
      1. Open the consolidated indexer files
         (``IndexBest_all.bin``, ``IndexKey_all.bin``, ``IndexBest_IDs_all.bin``).
      2. Run per-voxel best-pick + within-voxel clustering for every voxel.
      3. Run cross-voxel global dedup → unique grains list.
      4. Write ``UniqueOrientations.csv`` + ``UniqueIndexSingleKey.bin``.
      5. Build sinograms (tolerance or indexing mode).
      6. Optionally extract 21×21 patches.

    Parameters
    ----------
    work_dir : path-like
        The MIDAS layer directory. Reads from ``<work_dir>/Output/``,
        writes outputs into the same folder.
    space_group : int
    sino_mode : "tolerance" | "indexing"
        Which sinogram-assembly path to use.
    extract_patches : bool
        Run :func:`._patches.extract_patches_from_spot_map` (requires
        ``frame_loader``).
    confidence_min : float
        ``MIDAS_PF_SINO_CONF_MIN``; env-overridable for ``indexing``
        mode. Ignored for ``tolerance`` mode.
    scan_tolerance_um : float
        ``MIDAS_PF_SINO_SCAN_TOL``; env-overridable for ``indexing``
        mode.
    cluster_misorientation_deg : float
        ``maxAngle`` parameter for both per-voxel and cross-voxel
        clustering.
    tol_ome_deg, tol_eta_deg : float
        Degree tolerances for within-grain spot dedup and tolerance-mode
        sino matching.
    n_scans : int, optional
        If ``None``, inferred from ``n_voxels == n_scans**2``.
    output_subdir : str
        Sub-directory of ``work_dir`` where consolidated files live and
        outputs are written. Defaults to ``"Output"`` (matches C).
    normalize_sino, abs_transform : bool
        Variant transforms for the ``sinos_<nG>_<maxH>_<nS>.bin`` file.
    frame_loader : callable, optional
        Required only when ``extract_patches=True``.
    """
    work = Path(work_dir).resolve()
    out_dir = work / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    vals_r, keys_r, ids_r = open_all_three(out_dir)

    n_voxels = int(vals_r.n_voxels)
    if n_scans is None:
        # Square grid in PF.
        s = int(round(np.sqrt(n_voxels)))
        if s * s != n_voxels:
            raise ValueError(
                f"find_grains_single: n_voxels={n_voxels} is not a square grid; "
                "pass n_scans explicitly."
            )
        n_scans = s

    # --- Per-voxel pass: best-row + within-voxel cluster bookkeeping.
    # Embarrassingly parallel across voxels; dominates runtime on DEFORMED maps
    # with many candidate solutions per voxel. MIDAS_FINDGRAINS_NJOBS>1 fans the
    # voxels across CPU workers (byte-identical to the serial path).
    import os as _osj
    _njobs = int(_osj.environ.get("MIDAS_FINDGRAINS_NJOBS", "1") or "1")
    per_vox_OMs, per_vox_confs, per_vox_keys_for_global, single_key_rows = _per_voxel_pass(
        out_dir, n_voxels, space_group, cluster_misorientation_deg, n_jobs=_njobs,
    )

    single_key_path = out_dir / "UniqueIndexSingleKey.bin"
    write_unique_index_single_key(single_key_path, n_voxels, single_key_rows)

    # --- Cross-voxel global dedup.
    # Cross-voxel dedup. The reference is O(N^2) and degrades catastrophically on
    # DEFORMED maps (intragranular spread defeats the marking shortcut). Default to
    # "auto" = EXACT GPU all-pairs path when a device is available (~100x constant
    # speedup, byte-parity), else the exact reference. Override via
    # MIDAS_FINDGRAINS_CLUSTER = auto | gpu | reference | binned (+ MIDAS_FINDGRAINS_DEVICE).
    # ("binned" is the O(N*k) asymptotic fix but EXPERIMENTAL — see _cluster.py.)
    import os as _os
    _cmethod = _os.environ.get("MIDAS_FINDGRAINS_CLUSTER", "auto").strip().lower()
    _cdev = None
    if _cmethod in ("gpu", "binned"):
        _d = _os.environ.get("MIDAS_FINDGRAINS_DEVICE", "").strip()
        if _d:
            import torch as _torch
            _cdev = _torch.device(_d)
    glob = global_cluster_fast(
        per_vox_OMs=per_vox_OMs,
        per_vox_confs=per_vox_confs,
        per_vox_keys=per_vox_keys_for_global,
        space_group=space_group,
        max_ang_deg=cluster_misorientation_deg,
        invalid_marker=-1,  # uint64 sentinel
        method=_cmethod,
        device=_cdev,
    )

    unique_orientations_csv = out_dir / "UniqueOrientations.csv"
    write_unique_orientations_csv(
        unique_orientations_csv,
        glob.unique_key_arr,
        glob.unique_OM_arr,
    )

    # --- Emit voxel_grid.csv for downstream V-map refinement (P9 TODO(a)).
    # Layout: voxel_idx x_um y_um z_um grain_id
    # Voxel lab positions come from the (xThis, yThis) convention in
    # IndexerScanningOMP.c:1731-1732:
    #     for v = i*n_scans + j: (x, y) = (positions[i], positions[j])
    # Falls back to placeholder zeros if positions.csv is absent.
    voxel_grid_csv = out_dir / "voxel_grid.csv"
    try:
        positions_path = work / "positions.csv"
        if positions_path.exists():
            sg = read_positions_csv(positions_path)
            # spatial_positions are sorted-by-y; the IndexerScanningOMP
            # convention uses sorted positions for the (xThis, yThis) grid.
            pos = sg.spatial_positions
            n_scans_int = int(np.sqrt(n_voxels))
            if n_scans_int * n_scans_int != n_voxels:
                LOG.warning(
                    "find_grains_single: n_voxels=%d not a square — using "
                    "scan_nr-indexed (x, y) = (positions[v], 0).",
                    n_voxels,
                )
                xs = np.array([pos[v] if v < pos.size else 0.0
                               for v in range(n_voxels)])
                ys = np.zeros(n_voxels)
            else:
                i_idx = np.arange(n_voxels) // n_scans_int
                j_idx = np.arange(n_voxels) % n_scans_int
                xs = pos[i_idx]
                ys = pos[j_idx]
        else:
            xs = np.zeros(n_voxels)
            ys = np.zeros(n_voxels)
            LOG.info(
                "find_grains_single: positions.csv absent — voxel positions "
                "in %s are zero placeholders.", voxel_grid_csv.name,
            )
        gid = glob.voxel_to_unique  # int64 (n_voxels,), -1 for invalid
        rows = np.column_stack([
            np.arange(n_voxels, dtype=np.int64),
            xs.astype(np.float64), ys.astype(np.float64),
            np.zeros(n_voxels, dtype=np.float64),    # z = 0 for PF
            gid.astype(np.int64),
        ])
        np.savetxt(
            voxel_grid_csv, rows,
            header="voxel_idx x_um y_um z_um grain_id",
            fmt=["%d", "%.4f", "%.4f", "%.4f", "%d"],
            comments="",
        )
    except Exception as e:  # pragma: no cover
        LOG.warning("find_grains_single: voxel_grid.csv emit failed (%s).", e)
        voxel_grid_csv = None

    artifacts = FindGrainsArtifacts(
        unique_orientations_csv=str(unique_orientations_csv),
        unique_index_single_key_bin=str(single_key_path),
        n_unique_grains=glob.n_uniques,
    )

    # --- Sinogen.
    # Spots.bin may live in either <work>/Output/Spots.bin (Python
    # pipeline default) or <work>/Spots.bin (legacy C-output layout).
    # Try both.
    spots_path = out_dir / "Spots.bin"
    if not spots_path.exists():
        alt = work / "Spots.bin"
        if alt.exists():
            spots_path = alt
    positions_path = work / "positions.csv"
    if not spots_path.exists():
        # No Spots.bin available anywhere — return what we have so far.
        return artifacts
    all_spots = _read_spots_bin(spots_path)

    scan_grid = None
    if positions_path.exists():
        scan_grid = read_positions_csv(positions_path)

    if glob.n_uniques == 0:
        # Nothing to sinogen.
        return artifacts

    if sino_mode == "tolerance":
        spot_list = process_spots(
            unique_key_arr=glob.unique_key_arr,
            all_spots=all_spots,
            keys_reader=keys_r,
            ids_reader=ids_r,
            tol_ome=tol_ome_deg,
            tol_eta=tol_eta_deg,
        )
        sg = generate_sinograms_tolerance(
            spot_list=spot_list,
            n_unique=glob.n_uniques,
            all_spots=all_spots,
            n_scans=n_scans,
            tol_ome=tol_ome_deg,
            tol_eta=tol_eta_deg,
            output_dir=out_dir,
            scan_to_spatial=(scan_grid.scan_to_spatial if scan_grid is not None else None),
            normalize_sino=normalize_sino,
            abs_transform=abs_transform,
            emit_softsum=emit_softsum,
            soft_weight_fn=soft_weight_fn,
        )
    elif sino_mode == "indexing":
        sg = generate_sinograms_indexing(
            unique_key_arr=glob.unique_key_arr,
            unique_OM_arr=glob.unique_OM_arr,
            all_spots=all_spots,
            n_scans=n_scans,
            space_group=space_group,
            max_ang_deg=cluster_misorientation_deg,
            tol_ome=tol_ome_deg,
            tol_eta=tol_eta_deg,
            output_dir=out_dir,
            vals_reader=vals_r,
            keys_reader=keys_r,
            ids_reader=ids_r,
            scan_grid=scan_grid,
            confidence_min=confidence_min,
            scan_tolerance_um=scan_tolerance_um,
            normalize_sino=normalize_sino,
            abs_transform=abs_transform,
        )
    else:
        raise ValueError(f"sino_mode must be 'tolerance' or 'indexing', got {sino_mode!r}")

    artifacts.sinogen = sg

    if extract_patches:
        if frame_loader is None:
            raise ValueError("extract_patches=True requires frame_loader")
        # Read back the sinos_meta + spotMapping we just wrote and feed them
        # to the patch extractor.
        spot_id_arr = np.frombuffer(
            Path(sg.spot_map_path).read_bytes(), dtype=np.int32
        ).reshape(sg.n_grains, sg.max_n_hkls, sg.n_scans)
        spot_meta = np.frombuffer(
            Path(sg.spot_meta_path).read_bytes(), dtype=np.float64
        ).reshape(sg.n_grains, sg.max_n_hkls, sg.n_scans, 4)
        patches_path, spot_pos_path = extract_patches_from_spot_map(
            spot_id_arr=spot_id_arr,
            spot_meta=spot_meta,
            n_grains=sg.n_grains,
            max_n_hkls=sg.max_n_hkls,
            n_scans=sg.n_scans,
            frame_loader=frame_loader,
            output_dir=out_dir,
        )
        artifacts.patches_path = patches_path
        artifacts.spot_pos_path = spot_pos_path

    return artifacts


def find_grains_multiple(
    work_dir: str | Path,
    space_group: int,
    *,
    cluster_misorientation_deg: float = 1.0,
    confidence_min: float = 0.0,
    n_scans: Optional[int] = None,
    output_subdir: str = "Output",
) -> FindGrainsArtifacts:
    """Replace ``findMultipleSolutionsPF.c``.

    For each voxel, run :func:`._cluster.per_voxel_cluster` to get the
    representatives of each within-voxel cluster, then aggregate the
    representative keys into ``SpotsToIndex.csv``.

    Output:
      - ``<work_dir>/SpotsToIndex.csv`` — one row per (voxel, cluster).
      - ``<work_dir>/<output_subdir>/UniqueIndexSingleKey.bin`` — one row
        per voxel = best-of-best (matches C semantics: even though
        findMultipleSolutionsPF only writes the per-voxel ``UniqueIndexKey_*.txt``
        files, the operator typically calls findSingleSolution first to
        get the binary key file. We emit it here too so the contract is
        complete.)
    """
    work = Path(work_dir).resolve()
    out_dir = work / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    vals_r, keys_r, _ids_r = open_all_three(out_dir)
    n_voxels = int(vals_r.n_voxels)
    if n_scans is None:
        s = int(round(np.sqrt(n_voxels)))
        if s * s != n_voxels:
            raise ValueError(
                f"find_grains_multiple: n_voxels={n_voxels} not square; "
                "pass n_scans explicitly."
            )
        n_scans = s

    # Per-voxel cluster reps + best-of-best row for UniqueIndexSingleKey.
    rows_per_voxel: dict[int, np.ndarray] = {}
    single_key_rows: list[tuple[int, np.ndarray]] = []

    for v in range(n_voxels):
        vals_v = vals_r.get_vals(v)
        keys_v = keys_r.get_keys(v)
        if vals_v is None or keys_v is None:
            continue
        n_sol = int(vals_r.n_sol_arr[v])
        if n_sol <= 0:
            continue
        denom = vals_v[:, 14]
        with np.errstate(divide="ignore", invalid="ignore"):
            confs = np.where(denom > 0, vals_v[:, 15] / denom, 0.0)
        ias = vals_v[:, 1]
        OMs = vals_v[:, 2:11]
        # Multiple-mode uses min_conf (default 0.0).
        result = per_voxel_cluster(
            OMs, confs, ias, keys_v,
            space_group=space_group,
            max_ang_deg=cluster_misorientation_deg,
            min_conf=confidence_min,
        )
        if result.best_row < 0:
            continue

        # SpotsToIndex.csv rows: for each within-voxel cluster, emit a row
        # with the cluster representative's (SpotID, nMatches). Cols 2-3
        # are zero placeholders per the C semantics
        # (findMultipleSolutionsPF.c:99-100), col 4 = solution index.
        # We need to find the row index of each unique_keys[i] back in
        # OMs by matching SpotID; per_voxel_cluster doesn't expose that
        # directly so we re-derive: for each cluster representative,
        # search for matching SpotID + nMatches in keys_v.
        # In practice findMultipleSolutionsPF writes uniqueArr[i*5+4] =
        # (size_t)bRN (the row index within the voxel). We synthesize it.
        cluster_rows: list[np.ndarray] = []
        for ui in range(int(result.unique_keys.shape[0])):
            uk = result.unique_keys[ui]
            sid = int(uk[0])
            nm = int(uk[1])
            # Find the bRN: first j where keys_v[j, 0] == sid and keys_v[j, 1] == nm.
            mask = (keys_v[:, 0] == np.uint64(sid)) & (keys_v[:, 1] == np.uint64(nm))
            cand = np.flatnonzero(mask)
            bRN = int(cand[0]) if cand.size > 0 else 0
            cluster_rows.append(np.array([
                sid, nm, 0, 0, bRN,
            ], dtype=np.uint64))
        if cluster_rows:
            rows_per_voxel[v] = np.stack(cluster_rows, axis=0)

        # Best-of-best for UniqueIndexSingleKey.bin.
        br = result.best_row
        single_key_rows.append((v, np.array([
            v,
            int(keys_v[br, 0]),
            int(keys_v[br, 1]),
            int(keys_v[br, 2]),
            br,
        ], dtype=np.uint64)))

    single_key_path = out_dir / "UniqueIndexSingleKey.bin"
    write_unique_index_single_key(single_key_path, n_voxels, single_key_rows)

    spots_to_index_csv = work / "SpotsToIndex.csv"
    write_spots_to_index_csv(spots_to_index_csv, rows_per_voxel)

    n_total = sum(int(rows.shape[0]) for rows in rows_per_voxel.values())

    return FindGrainsArtifacts(
        unique_orientations_csv="",
        unique_index_single_key_bin=str(single_key_path),
        spots_to_index_csv=str(spots_to_index_csv),
        n_unique_grains=n_total,
    )
