"""On-disk driver — file→refine_block→file.

Reads:
  ``<cwd>/SpotsToIndex.csv``   one int per line; row index == seed index
  ``<cwd>/ExtraInfo.bin``      (nSpots, 16) float64
  ``<cwd>/hkls.csv``           reflection list (h k l ds RingNr ... 2θ ...)
  ``<OutputFolder>/IndexBest.bin``       (nSeeds, 15) float64
  ``<OutputFolder>/IndexBestFull.bin``   (nSeeds, 5000, 2) float64

Writes:
  ``<ResultFolder>/Key.bin``
  ``<ResultFolder>/ProcessKey.bin``
  ``<ResultFolder>/OrientPosFit.bin``
  ``<OutputFolder>/FitBest.bin``
  Optional ``<OutputFolder>/FitBest.csv`` (with ``--csv``)

This module is intentionally narrow: ``refine_block_from_disk`` is one
function that the CLI calls, sized so it can also be invoked
programmatically from ``ff_MIDAS.py``.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from midas_diffract import HEDMForwardModel, HEDMGeometry  # type: ignore
from midas_diffract.hkls import _cartesian_B_matrix  # type: ignore

from . import c_port
from .config import FitConfig
from .io_binary import (
    EXTRA_INFO_NCOLS, FIT_BEST_NCOLS, MAX_NHKLS_DEFAULT, ORIENT_POS_FIT_NCOLS,
    GrainResult, read_extra_info,
    write_fit_best_row, write_key_row, write_orient_pos_fit_row,
    write_process_key_row,
)
from .matching import MatchResult
from .observations import ObservedSpots
from .refine_block import BlockFitResult, refine_block

LOG = logging.getLogger(__name__)
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

INDEX_BEST_RECORD_DOUBLES = 15           # IndexerOMP.c:1620
INDEX_BEST_FULL_PER_HKL = 2              # (spot_id, ?) per matched spot


def _read_spots_to_index(path: str | Path) -> np.ndarray:
    """SpotsToIndex.csv: one int per line; returns (n,) int array."""
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


def _read_index_best(path: str | Path,
                     n_seeds: int) -> np.ndarray:
    """``IndexBest.bin`` reader. Returns ``(n_seeds, 15)`` float64."""
    arr = np.fromfile(path, dtype=np.float64)
    full = n_seeds * INDEX_BEST_RECORD_DOUBLES
    if arr.size > full:
        arr = arr[:full]
    if arr.size < full:
        padded = np.zeros(full, dtype=np.float64)
        padded[:arr.size] = arr
        arr = padded
    return arr.reshape(n_seeds, INDEX_BEST_RECORD_DOUBLES)


def _read_index_best_full(path: str | Path, n_seeds: int,
                          max_nhkls: int = MAX_NHKLS_DEFAULT) -> np.ndarray:
    """``IndexBestFull.bin`` reader. Returns ``(n_seeds, max_nhkls, 2)``."""
    arr = np.fromfile(path, dtype=np.float64)
    full = n_seeds * max_nhkls * INDEX_BEST_FULL_PER_HKL
    if arr.size > full:
        arr = arr[:full]
    if arr.size < full:
        padded = np.zeros(full, dtype=np.float64)
        padded[:arr.size] = arr
        arr = padded
    return arr.reshape(n_seeds, max_nhkls, INDEX_BEST_FULL_PER_HKL)


def _read_consolidated_as_ff(out_dir: Path, n_total: int):
    """Adapt the c-omp consolidated ``IndexBest_all.bin`` family to the legacy
    FF ``(index_best, index_best_full)`` shapes the seed loop expects.

    The unified C indexer (and PF) always emit the consolidated family:
      IndexBest_all.bin    — per-voxel candidate records (16 cols/sol)
      IndexKey_all.bin     — per-solution [SpotID, nMatches, nIDs, _]
      IndexBest_IDs_all.bin — concatenated matched observed-spot IDs
    In FF mode each "voxel" is one seed, so we take the top candidate
    (highest completeness) per voxel and map its columns:
      consolidated[2:11]  → FF orient (rec[1:10])
      consolidated[11:14] → FF position (rec[10:13])
      n_observed (rec[14]) = number of matched IDs for that candidate
    The matched IDs become column 0 of ``index_best_full`` so the existing
    per-seed loop (which reads ``index_best_full[row, :n_observed, 0]``)
    works unchanged.
    """
    from .scan_driver import (
        _read_index_best_all, _open_keys_and_ids,
        _matched_ids_for_top, _top_candidate_index,
    )
    n_sol_arr, blocks = _read_index_best_all(out_dir / "IndexBest_all.bin")
    keys_reader, ids_reader = _open_keys_and_ids(out_dir)

    index_best = np.zeros((n_total, INDEX_BEST_RECORD_DOUBLES), dtype=np.float64)
    n_vox = min(len(blocks), n_total)
    matched_ids_per_row: list[np.ndarray] = [
        np.empty(0, dtype=np.int64) for _ in range(n_total)
    ]
    max_n_obs = 1
    n_no_ids = 0
    for v in range(n_vox):
        block = blocks[v]
        top = _top_candidate_index(block)
        if top is None:
            continue
        ids = _matched_ids_for_top(keys_reader, ids_reader, v, top)
        if ids is None or ids.size == 0:
            # No matched-ID record → cannot refine against specific spots;
            # leave the row empty so the seed loop skips it (n_observed==0).
            n_no_ids += 1
            continue
        rec = block[top]
        index_best[v, 1:10] = rec[2:11]      # orientation matrix
        index_best[v, 10:13] = rec[11:14]    # grain position (µm)
        index_best[v, 14] = ids.size         # n_observed
        matched_ids_per_row[v] = ids.astype(np.float64)
        if ids.size > max_n_obs:
            max_n_obs = int(ids.size)

    if keys_reader is None or ids_reader is None:
        LOG.warning(
            "consolidated IndexBest_all.bin found but IndexKey_all.bin / "
            "IndexBest_IDs_all.bin missing — no matched spot IDs available, "
            "all seeds will be skipped. (looked in %s)", out_dir)
    elif n_no_ids:
        LOG.info("consolidated adapter: %d/%d voxels had no matched-ID record",
                 n_no_ids, n_vox)

    index_best_full = np.zeros(
        (n_total, max_n_obs, INDEX_BEST_FULL_PER_HKL), dtype=np.float64)
    for v in range(n_total):
        ids = matched_ids_per_row[v]
        if ids.size:
            index_best_full[v, :ids.size, 0] = ids
    return index_best, index_best_full


def _read_hkls_csv(path: str | Path,
                   ring_numbers: list[int],
                   max_two_theta_deg: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse hkls.csv → ``(hkls_int, two_theta_deg, ring_nr)``.

    * ``hkls_int``: ``(M, 3)`` int64 — Miller indices
    * ``two_theta_deg``: ``(M,)`` float64 — 2θ in degrees
    * ``ring_nr``: ``(M,)`` int64 — reflection's MIDAS ring number (col 4)

    Mirrors ``FitPosOrStrainsOMP.c:2266-2300`` ring filtering.
    """
    h_list: list[tuple[int, int, int, float, int]] = []
    rs = set(ring_numbers)
    with open(path, "r") as fp:
        fp.readline()  # header
        for line in fp:
            parts = line.split()
            if len(parts) < 9:
                continue
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
            rnr = int(parts[4])
            tht = float(parts[8])
            if 2.0 * tht > max_two_theta_deg:
                continue
            if rnr in rs:
                h_list.append((h, k, l, tht, rnr))
    if not h_list:
        raise ValueError(f"{path}: no reflections matched RingNumbers={ring_numbers}")
    hkls_int = np.array([(a, b, c) for (a, b, c, _, _) in h_list], dtype=np.int64)
    thetas_deg = np.array([t for (*_, t, _) in h_list], dtype=np.float64)
    ring_nr = np.array([r for (*_, r) in h_list], dtype=np.int64)
    return hkls_int, thetas_deg, ring_nr


def _build_model(cfg: FitConfig, *,
                 device: torch.device, dtype: torch.dtype,
                 hkls_int: np.ndarray, thetas_deg: np.ndarray,
                 ring_nr: np.ndarray
                 ) -> tuple[HEDMForwardModel, torch.Tensor]:
    """Build the forward model + per-reflection ring-slot tensor.

    Works for any lattice — uses the same Cartesian B-matrix convention
    as :func:`midas_diffract.hkls.hkls_for_forward_model` so the strain
    recompute path inside the model is bit-aligned with our hkls_cart.
    """
    B = _cartesian_B_matrix(tuple(cfg.LatticeConstant))   # (3, 3)
    hkls_cart_np = (B @ hkls_int.astype(np.float64).T).T   # (M, 3) in 1/Å
    thetas_rad = thetas_deg * DEG2RAD

    # Approximate beam center / detector size — these come from the
    # paramstest.txt's calibration block (BC, NrPixels). For now we use
    # placeholders the real driver will plumb through in Phase 5.
    geom = HEDMGeometry(
        Lsd=cfg.Lsd,
        y_BC=getattr(cfg, "y_BC", 1024.0),
        z_BC=getattr(cfg, "z_BC", 1024.0),
        px=cfg.px,
        omega_start=getattr(cfg, "omega_start", -180.0),
        omega_step=getattr(cfg, "omega_step", 0.25),
        n_frames=getattr(cfg, "n_frames", 1440),
        n_pixels_y=getattr(cfg, "n_pixels_y", 2048),
        n_pixels_z=getattr(cfg, "n_pixels_z", 2048),
        min_eta=cfg.MinEta,
        wavelength=cfg.Wavelength,
        flip_y=True,
    )
    model = HEDMForwardModel(
        torch.from_numpy(hkls_cart_np),
        torch.from_numpy(thetas_rad),
        geom,
        hkls_int=torch.from_numpy(hkls_int.astype(np.float64)),
        device=device,
        compile=getattr(cfg, "compile", False),
    )

    # ring slot per reflection: index into cfg.RingNumbers, sourced
    # directly from the per-reflection ring number column in hkls.csv
    # (works for any lattice symmetry).
    rn_to_slot = {int(rn): i for i, rn in enumerate(cfg.RingNumbers)}
    pred_ring_slot = torch.tensor(
        [rn_to_slot.get(int(v), -1) for v in ring_nr],
        dtype=torch.int64, device=device,
    )
    return model, pred_ring_slot


def _orientmat_to_euler_zxz(R: np.ndarray) -> np.ndarray:
    """3×3 ZXZ orientation matrix → Bunge Euler angles in radians.

    Inverts :func:`HEDMForwardModel.euler2mat` (ZXZ convention).
    Gimbal-lock branch (Φ ≈ 0 or π) returns ``φ2 = 0`` and folds the
    rotation into ``φ1`` — same convention as the C ``OrientMat2Euler``.
    """
    R22 = float(np.clip(R[2, 2], -1.0, 1.0))
    Phi = math.acos(R22)
    if abs(math.sin(Phi)) < 1e-9:
        # Gimbal lock — φ1 = atan2(R[1,0], R[0,0]); φ2 = 0.
        phi2 = 0.0
        phi1 = math.atan2(R[1, 0], R[0, 0])
    else:
        phi1 = math.atan2(R[0, 2], -R[1, 2])
        phi2 = math.atan2(R[2, 0],  R[2, 1])
    return np.array([phi1, Phi, phi2], dtype=np.float64)


def _orientmat_to_euler_zxz_batch(Rs: np.ndarray) -> np.ndarray:
    """Vectorised version of :func:`_orientmat_to_euler_zxz`.

    Input ``Rs`` is ``(B, 3, 3)``; output is ``(B, 3)`` — Bunge Euler angles
    in radians, ZXZ convention. Gimbal-lock branch picks the same fallback
    as the scalar function.
    """
    R22 = np.clip(Rs[:, 2, 2], -1.0, 1.0)
    Phi = np.arccos(R22)
    sin_Phi = np.sin(Phi)
    locked = np.abs(sin_Phi) < 1e-9

    # Non-gimbal-lock branch.
    phi1_n = np.arctan2(Rs[:, 0, 2], -Rs[:, 1, 2])
    phi2_n = np.arctan2(Rs[:, 2, 0],  Rs[:, 2, 1])
    # Gimbal-lock branch.
    phi1_l = np.arctan2(Rs[:, 1, 0],  Rs[:, 0, 0])
    phi2_l = np.zeros_like(phi1_l)

    phi1 = np.where(locked, phi1_l, phi1_n)
    phi2 = np.where(locked, phi2_l, phi2_n)
    return np.stack([phi1, Phi, phi2], axis=1)


def _write_empty_key_rows(path: str | Path, rows: list[int]) -> None:
    """Write ``(0, 0)`` int32 pairs for many empty Key.bin slots in one
    buffered ``write()`` call.

    Replaces ~4000 individual ``write_key_row`` open/seek/close cycles
    with one ``ftruncate`` + sparse ``pwrite``. The Key.bin file is
    laid out as ``row_nr × 8 bytes`` (two int32s per row); empty rows
    are all zeros, so we just ensure the file is at least
    ``max(rows)+1`` × 8 bytes long. Any later non-empty row write
    will pwrite over its slot — non-empty wins.
    """
    if not rows:
        return
    max_row = max(rows)
    size = (max_row + 1) * 2 * 4   # 2 × int32
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        cur_size = os.fstat(fd).st_size
        if cur_size < size:
            os.ftruncate(fd, size)
        # File is zero-filled in any region ftruncate just extended; that
        # IS the (SpotID=0, n_spots_comp=0) payload for an empty row.
    finally:
        os.close(fd)


def _preallocate(path: str | Path, size_bytes: int) -> None:
    """Idempotent pre-allocation. Ensures the file is at least ``size_bytes``
    long, extending sparsely with zeros if needed. Safe to call concurrently
    from multiple block workers.

    Without this, sparse-pwrite outputs (OrientPosFit / FitBest / ProcessKey)
    end at the *last written row*, so any seeds skipped at the tail produce a
    file shorter than ``n_seeds`` rows and break downstream readers that
    derive row count from file size (e.g. midas_process_grains.read_all).
    """
    fd = os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        if os.fstat(fd).st_size < size_bytes:
            os.ftruncate(fd, size_bytes)
    finally:
        os.close(fd)


def refine_block_from_disk(
    *,
    cfg: FitConfig,
    param_file: str | Path,
    block_nr: int,
    num_blocks: int,
    num_lines: Optional[int] = None,
    device: torch.device,
    dtype: torch.dtype,
    also_write_csv: bool = False,
) -> int:
    """Load ``block_nr`` from disk, refine, write outputs. Returns # grains."""
    cwd = Path(param_file).resolve().parent
    out_dir = Path(cfg.OutputFolder)
    res_dir = Path(cfg.ResultFolder)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. SpotsToIndex.csv → all seed IDs.
    spots_to_index = _read_spots_to_index(cwd / cfg.IDsFileName)
    n_total = len(spots_to_index)

    # Pre-allocate sparse-pwrite output files so trailing skipped seeds
    # don't truncate the file shorter than n_total rows. Idempotent across
    # block workers. ProcessKey strides MAX_NHKLS int32s per grain (same
    # stride midas_process_grains.io.binary.PROCESS_KEY_INTS expects).
    _preallocate(res_dir / cfg.OrientPosFitFileName,
                 n_total * ORIENT_POS_FIT_NCOLS * 8)
    _preallocate(out_dir / cfg.FitBestFileName,
                 n_total * MAX_NHKLS_DEFAULT * FIT_BEST_NCOLS * 8)
    _preallocate(res_dir / cfg.ProcessKeyFileName,
                 n_total * MAX_NHKLS_DEFAULT * 4)
    if num_lines is not None and num_lines != n_total:
        LOG.warning("num_lines=%d differs from SpotsToIndex (%d); using disk count",
                    num_lines, n_total)

    # 2. Block partition (mirrors C: contiguous slabs).
    chunk = math.ceil(n_total / num_blocks) if num_blocks > 0 else n_total
    row_lo = block_nr * chunk
    row_hi = min(row_lo + chunk, n_total)
    LOG.info("block %d/%d → seed rows [%d, %d) of %d total",
             block_nr, num_blocks, row_lo, row_hi, n_total)
    if row_lo >= row_hi:
        return 0

    # 3. Load shared on-disk artifacts.
    extra = read_extra_info(cwd / "ExtraInfo.bin", mmap=True)
    # Backend-agnostic seed read. The python indexer writes the legacy
    # IndexBest.bin / IndexBestFull.bin; the c-omp (unified C) backend — and
    # every PF run — write the consolidated IndexBest_all.bin family instead.
    # Prefer the legacy files when present; otherwise adapt the consolidated
    # output to the same (index_best, index_best_full) shapes.
    legacy_best = out_dir / "IndexBest.bin"
    consolidated_best = out_dir / "IndexBest_all.bin"
    if legacy_best.exists() and legacy_best.stat().st_size > 0:
        index_best = _read_index_best(legacy_best, n_total)
        index_best_full = _read_index_best_full(
            out_dir / "IndexBestFull.bin", n_total,
        )
    elif consolidated_best.exists() and consolidated_best.stat().st_size > 0:
        LOG.info("indexing output: consolidated IndexBest_all.bin "
                 "(c-omp/unified backend) → adapting to FF seed format")
        index_best, index_best_full = _read_consolidated_as_ff(out_dir, n_total)
    else:
        raise FileNotFoundError(
            f"no indexing output found in {out_dir}: expected IndexBest.bin "
            f"(python backend) or IndexBest_all.bin (c-omp backend)"
        )
    if cfg.RhoD > 0.0 and cfg.Lsd > 0.0:
        max_two_theta_deg = 2.0 * math.degrees(math.atan(cfg.RhoD / cfg.Lsd))
    else:
        # paramstest.txt from torch fit_setup may omit RhoD; fall back to a
        # large sentinel so we don't accidentally filter every reflection.
        max_two_theta_deg = 180.0
    hkls_int, thetas_deg, ring_nr = _read_hkls_csv(
        cwd / "hkls.csv", cfg.RingNumbers,
        max_two_theta_deg=max_two_theta_deg,
    )

    model, pred_ring_slot = _build_model(
        cfg, device=device, dtype=dtype,
        hkls_int=hkls_int, thetas_deg=thetas_deg, ring_nr=ring_nr,
    )

    # 4. Build per-grain inputs.
    #
    # Optimised seed-loading prefix (vs. the original per-row loop with
    # ``ObservedSpots.from_extra_info`` per call):
    #
    #   * ``id_to_row`` is built ONCE here, not 4397× inside from_extra_info
    #     (was ~130 s on park22-scale ExtraInfo with 160k spots).
    #   * Empty-slot ``Key.bin`` writes are batched into one buffered
    #     ``pwrite`` after the loop (was 3956 individual file open/close
    #     cycles → ~4 s).
    #   * The 13 small CPU→GPU tensor copies per grain are deferred to a
    #     single big ``ObservedBatch`` packing inside ``refine_block``;
    #     here we only build CPU-side ``ObservedSpots`` (numpy slices),
    #     which avoids 4397×13 individual ``torch.as_tensor`` calls.
    #   * Orientation matrix → Bunge ZXZ Euler conversion is vectorised
    #     across all valid grains at the end of the loop.
    #
    # Net effect on park22 (218 s seed loading) is targeted to ~10–20 s.
    grains_obs: list[ObservedSpots] = []
    grain_rows: list[int] = []        # rowNr (= offset into Key/OrientPosFit)
    init_pos_list: list[np.ndarray] = []
    init_om_list: list[np.ndarray] = []   # 3×3 orient matrices, vectorised → euler later
    init_lat_list: list[np.ndarray] = []
    seed_ids: list[int] = []
    matches: list[MatchResult] = []
    n_skipped = 0
    skipped_rows: list[int] = []      # for batched empty Key.bin write

    # Build the SpotID → row lookup ONCE. extra is mmap'd; this materialises
    # only the ID column (n_spots × 8 bytes ≈ 1.3 MB at park22-scale).
    extra_all_ids = np.ascontiguousarray(extra[:, 4]).astype(np.int64, copy=False)
    id_to_row = {int(sid): row for row, sid in enumerate(extra_all_ids)}

    cpu_device = torch.device("cpu")
    lat_const_array = np.array(cfg.LatticeConstant, dtype=np.float64)

    for row_nr in range(row_lo, row_hi):
        rec = index_best[row_nr]
        n_observed = int(rec[14])
        if n_observed == 0:
            skipped_rows.append(row_nr)
            n_skipped += 1
            continue

        Orient0 = rec[1:10].reshape(3, 3)
        Pos0 = rec[10:13]
        seed_spot_id = int(spots_to_index[row_nr])

        spot_pairs = index_best_full[row_nr, :n_observed]    # (n_obs, 2)
        spot_ids_np = np.ascontiguousarray(spot_pairs[:, 0]).astype(np.int64,
                                                                    copy=False)

        # Inline lookup using the cached id_to_row dict; bail to the empty-
        # slot path if any SpotID is missing (mirrors C's "skip and zero").
        rows_arr = np.empty(spot_ids_np.shape[0], dtype=np.int64)
        try:
            for i, sid in enumerate(spot_ids_np):
                rows_arr[i] = id_to_row[int(sid)]
        except KeyError as e:
            LOG.warning("row %d: SpotID %s not in ExtraInfo.bin — skipping",
                        row_nr, e)
            skipped_rows.append(row_nr)
            n_skipped += 1
            continue

        block = extra[rows_arr]   # (S, 16) — one slab read per valid grain

        # Build a CPU-tensor ObservedSpots. The big CPU→GPU transfer
        # happens in refine_block via ``ObservedBatch.pack``, which packs
        # all grains into one tensor and copies it across once.
        obs = ObservedSpots(
            spot_id=torch.as_tensor(block[:, 4].astype(np.int64), dtype=torch.int64),
            ring_nr=torch.as_tensor(block[:, 5].astype(np.int64), dtype=torch.int64),
            y_lab=torch.as_tensor(block[:, 0], dtype=dtype),
            z_lab=torch.as_tensor(block[:, 1], dtype=dtype),
            omega=torch.as_tensor(block[:, 2] * DEG2RAD, dtype=dtype),
            eta=torch.as_tensor(block[:, 6] * DEG2RAD, dtype=dtype),
            two_theta=torch.as_tensor(block[:, 7] * DEG2RAD, dtype=dtype),
            grain_radius=torch.as_tensor(block[:, 3], dtype=dtype),
            fit_rmse=torch.as_tensor(block[:, 15], dtype=dtype),
            y_orig=torch.as_tensor(block[:, 9], dtype=dtype),
            z_orig=torch.as_tensor(block[:, 10], dtype=dtype),
            omega_ini=torch.as_tensor(block[:, 8] * DEG2RAD, dtype=dtype),
            mask_touched=torch.as_tensor(block[:, 14], dtype=dtype),
        )

        grains_obs.append(obs)
        grain_rows.append(row_nr)
        seed_ids.append(seed_spot_id)
        init_pos_list.append(Pos0.copy())
        init_om_list.append(Orient0.copy())
        init_lat_list.append(lat_const_array)

        # Pre-built match: one observed spot per HKL slot — but at this
        # stage we don't know which (k, m) slot the indexer picked; we
        # only have the spot ID. So we let refine_block do its initial
        # association via _rematch_batch, which uses ring + closest-omega.
        matches = []   # signal: re-match at init state

    # Batched empty-slot Key.bin writes: all (row_nr, 0, 0) entries at once.
    if skipped_rows:
        _write_empty_key_rows(res_dir / cfg.KeyFileName, skipped_rows)

    if not grains_obs:
        LOG.info("block %d: no valid grains, exiting", block_nr)
        return 0

    # Vectorised Euler conversion across all valid grains at once.
    init_eul_arr = _orientmat_to_euler_zxz_batch(np.stack(init_om_list))   # (B, 3)

    init_positions = torch.from_numpy(np.stack(init_pos_list)).to(device=device, dtype=dtype)
    init_eulers = torch.from_numpy(init_eul_arr).to(device=device, dtype=dtype)
    init_lattices = torch.from_numpy(np.stack(init_lat_list)).to(device=device, dtype=dtype)

    LOG.info("refining %d grains (skipped %d empty slots)",
             len(grains_obs), n_skipped)

    block = refine_block(
        cfg, model=model,
        grains_obs=grains_obs,
        init_positions=init_positions,
        init_eulers=init_eulers,
        init_lattices=init_lattices,
        pred_ring_slot=pred_ring_slot,
    )

    # 5. Write per-grain outputs.
    fit_best_path = out_dir / cfg.FitBestFileName
    orient_path = res_dir / cfg.OrientPosFitFileName
    key_path = res_dir / cfg.KeyFileName
    process_key_path = res_dir / cfg.ProcessKeyFileName

    # Build the (M, 3) integer Miller indices + (M,) ring-number vector
    # used by the C port. `hkls_int` and `ring_nr` were loaded earlier
    # from hkls.csv; reuse them directly.
    omega_ranges_np = np.asarray(cfg.OmegaRanges, dtype=np.float64).reshape(-1, 2)
    box_sizes_np = np.asarray(cfg.BoxSizes, dtype=np.float64).reshape(-1, 4)

    from midas_diffract import HEDMForwardModel as _HFM
    RAD2DEG = 180.0 / math.pi

    n_grains_written = 0
    for g, row_nr, sid, obs in zip(block.grains, grain_rows, seed_ids, grains_obs):
        # Build the C-style spotsYZO (S, 10) view from obs.
        S = int(obs.n_spots)
        spots_yzo = np.zeros((S, 10), dtype=np.float64)
        spots_yzo[:, 0] = obs.y_lab.cpu().numpy()
        spots_yzo[:, 1] = obs.z_lab.cpu().numpy()
        spots_yzo[:, 2] = obs.omega.cpu().numpy() * RAD2DEG
        spots_yzo[:, 3] = obs.spot_id.cpu().numpy()
        spots_yzo[:, 4] = obs.omega_ini.cpu().numpy() * RAD2DEG
        spots_yzo[:, 5] = obs.y_orig.cpu().numpy()
        spots_yzo[:, 6] = obs.z_orig.cpu().numpy()
        spots_yzo[:, 7] = obs.ring_nr.cpu().numpy()
        spots_yzo[:, 8] = obs.mask_touched.cpu().numpy()
        spots_yzo[:, 9] = obs.fit_rmse.cpu().numpy()

        # Refined orientation as a 3×3 numpy matrix.
        OM = _HFM.euler2mat(g.euler.cpu().double()).numpy()
        pos_um = g.position.cpu().numpy().astype(np.float64)
        lat_c = g.lattice.cpu().numpy().astype(np.float64)

        # Run the C-faithful matching + SpotsComp builder.
        spots_comp, spots_yzog, err_ini, n_matched = c_port.calc_angle_errors(
            pos=pos_um,
            orient_mat=OM,
            lat_c=lat_c,
            spots_yzo=spots_yzo,
            hkls_int=hkls_int.astype(np.int64),
            ring_nr_per_hkl=ring_nr.astype(np.int64),
            lsd=cfg.Lsd,
            wavelength=cfg.Wavelength,
            omega_ranges=omega_ranges_np,
            box_sizes=box_sizes_np,
            min_eta=cfg.MinEta,
            wedge_deg=cfg.Wedge,
            chi_deg=cfg.chi,
            weight_mask=getattr(cfg, "weight_by_position_uncertainty", 0) and 1.0 or 1.0,
            weight_fit_rmse=0.0,
        )

        # Write FitBest.bin row.
        if n_matched > 0:
            write_fit_best_row(fit_best_path, row_nr, spots_comp)

        # meanRadius: mean per-spot estimated grain radius across matched
        # spots (uses ExtraInfo col 3 = "GrainRadius" set at peak finding).
        if n_matched > 0:
            grain_radii = obs.grain_radius.cpu().numpy()
            # Need the obs index of each matched spot — recover by SpotID.
            obs_id_to_idx = {
                int(s): i for i, s in enumerate(obs.spot_id.cpu().numpy())
            }
            mean_rad = 0.0
            count = 0
            for i in range(n_matched):
                sid_match = int(spots_comp[i, 0])
                if sid_match in obs_id_to_idx:
                    mean_rad += grain_radii[obs_id_to_idx[sid_match]]
                    count += 1
            mean_radius = mean_rad / count if count > 0 else 0.0
        else:
            mean_radius = 0.0

        # OrientPosFit.bin row.
        completeness = float(n_matched) / max(int(obs.n_spots), 1)
        grain_result = GrainResult(
            SpotID=sid,
            OrientMat=OM.reshape(-1),
            Position=pos_um,
            LatticeFit=lat_c,
            ErrorPos=err_ini[0],
            ErrorOrient=err_ini[1],
            ErrorStrain=err_ini[2],
            meanRadius=mean_radius,
            completeness=completeness,
        )
        write_orient_pos_fit_row(orient_path, row_nr, grain_result)

        # Key.bin: (SpotID, n_matched).
        write_key_row(key_path, row_nr, sid, n_matched)

        # ProcessKey.bin: matched spot IDs per grain (used by ProcessGrains).
        if n_matched > 0:
            valid_ids = spots_comp[:, 0].astype(np.int32)
        else:
            valid_ids = np.array([], dtype=np.int32)
        write_process_key_row(process_key_path, row_nr, valid_ids)
        n_grains_written += 1

    return n_grains_written
