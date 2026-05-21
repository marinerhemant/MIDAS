"""Binary readers for the per-seed records of the FF-HEDM pipeline.

All paths are resolved relative to the run directory (the directory containing
``paramstest.txt``, mirroring the C convention used by ``ProcessGrains.c``).

File schemas
------------

``Output/IndexBest.bin``
    ``[N_seeds, 15]`` float64 (native-endian). Per-seed indexer summary:
    avg_ia, 9-element orient mat, 3-element best position, n_t_spots, n_matches.
    Mirrors ``IndexerOMP.c::WriteBestMatchBin``.

``Output/IndexBestFull.bin``
    ``[N_seeds, MAX_N_HKLS=5000, 2]`` float64 (native-endian). For each seed and
    each theoretical-hkl row, the (matched_obs_SpotID, delta_omega) of the
    matched observed spot (zero-padded). Theoretical-hkl row index is the same
    across all seeds — it indexes into the fixed hkls.csv list filtered by
    RingNumbers.

``Output/FitBest.bin``
    ``[N_seeds, MAX_N_HKLS=5000, 22]`` float64. Per-seed refined per-spot
    record. Column 0 of each 22-double row is the SpotID; columns 1..21 are
    observed/theoretical y/z/omega/eta/etc. and per-spot residuals. Mirrors
    ``FitPosOrStrainsOMP.c:689-702``. *Very* large file — 314 GB on the
    peakfit hard dataset; always mmap'd.

``Results/Key.bin``
    ``[N_seeds, 2]`` int32. (keep_flag, NrIDsPerID).

``Results/OrientPosFit.bin``
    ``[N_seeds, 27]`` float64. Per-seed refined orient/pos/lattice/errors.
    Column layout (per ``FitPosOrStrainsOMP.c:3013-3025``):

    ===== ===========================================
    Index  Meaning
    ===== ===========================================
    0      SpId sentinel (skipped by ProcessGrains.c)
    1-9    OrientsFit (3x3 orientation matrix, row-major)
    10     SpId sentinel
    11-13  PositionsFit (x, y, z) in lab frame, µm
    14     SpId sentinel
    15-20  LatticeParameterFit (a, b, c, α, β, γ)
    21     SpId sentinel
    22-24  ErrorsFin (pos_err, omega_err, internal_angle)
    25     meanRadius (µm)
    26     completeness
    ===== ===========================================

``Results/ProcessKey.bin``
    ``[N_seeds, NR_MAX_IDS_PER_GRAIN=5000]`` int32. Refined matched SpotID list
    per seed (zero-padded). Non-zero entries == matched SpotIDs after
    refinement.

``Results/IDsToKeep.bin`` (**legacy / optional**)
    Some pipelines emit this; we don't depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


MAX_N_HKLS = 5000
NR_MAX_IDS_PER_GRAIN = 5000

INDEX_BEST_DOUBLES = 15
INDEX_BEST_FULL_DOUBLES = 2 * MAX_N_HKLS
FIT_BEST_DOUBLES = 22 * MAX_N_HKLS
ORIENT_POS_FIT_DOUBLES = 27
PROCESS_KEY_INTS = NR_MAX_IDS_PER_GRAIN
KEY_INTS = 2

# Column indices into the OrientPosFit row (after the 4 sentinels).
ORIENT_POS_FIT_LAYOUT = {
    "orient_mat":   slice(1, 10),    # 9 elements row-major
    "position":     slice(11, 14),
    "lattice":      slice(15, 21),   # a, b, c, α, β, γ
    "pos_err":      22,
    "ome_err":      23,
    "internal_ang": 24,              # IAColNr=20 in C OPs; index 24 in this raw record
    "mean_radius":  25,
    "completeness": 26,
}

_NATIVE_ORDER = {"=", "|", "<"} if np.little_endian else {"=", "|", ">"}


def _assert_native(arr: np.ndarray, fname: str) -> None:
    """Fail loud if a binary file appears to be in non-native byte order."""
    bo = arr.dtype.byteorder
    if bo not in _NATIVE_ORDER:
        raise ValueError(
            f"{fname}: dtype byteorder {bo!r} is not native; "
            "midas-process-grains assumes native-endian binaries."
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BinaryInputs:
    """All per-seed binary records mmap'd from disk.

    Each tensor lives on the host (numpy memmap). Promotion to the user device
    happens in the pipeline orchestrator, not here, so we avoid implicitly
    copying multi-GB files into GPU memory.
    """

    n_seeds: int
    index_best: Optional[np.memmap]      # (N, 15) float64, may be None if absent
    index_best_full: Optional[np.memmap] # (N, 5000, 2) float64
    fit_best: Optional[np.memmap]        # (N, 5000, 22) float64
    orient_pos_fit: np.memmap            # (N, 27) float64 — required
    key: np.memmap                       # (N, 2) int32 — required
    process_key: np.memmap               # (N, 5000) int32 — required


# ---------------------------------------------------------------------------
# Per-file readers
# ---------------------------------------------------------------------------


def read_index_best(run_dir: Union[str, Path]) -> np.memmap:
    """Read ``Output/IndexBest.bin`` into a (N, 15) float64 memmap."""
    p = Path(run_dir) / "Output" / "IndexBest.bin"
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.float64, mode="r")
    _assert_native(arr, str(p))
    if arr.size % INDEX_BEST_DOUBLES != 0:
        raise ValueError(
            f"{p} size {arr.size} doubles is not a multiple of "
            f"{INDEX_BEST_DOUBLES}"
        )
    return arr.reshape(-1, INDEX_BEST_DOUBLES)


def read_index_best_full(run_dir: Union[str, Path]) -> np.ndarray:
    """Read the per-seed (matched SpotID, delta-omega) table, (N, 5000, 2).

    Column 0 of the last axis is the matched SpotID (hkl-slot indexed,
    zero-padded). Column 1 is the delta-omega of the match.

    The python indexer writes ``Output/IndexBestFull.bin`` directly. The
    c-omp (unified C) backend emits the consolidated family instead, which
    has no hkl-slot-indexed equivalent — but the *refiner* writes
    ``Output/FitBest.bin`` whose column 0 is the same hkl-slot-indexed
    matched SpotID. When IndexBestFull.bin is absent we synthesize this
    table from FitBest col0 (col1/delta-omega is set to 0; it only feeds
    the residual tiebreak in spot-conflict resolution, not grain count).
    """
    p = Path(run_dir) / "Output" / "IndexBestFull.bin"
    if p.exists():
        arr = np.memmap(p, dtype=np.float64, mode="r")
        _assert_native(arr, str(p))
        if arr.size % INDEX_BEST_FULL_DOUBLES != 0:
            raise ValueError(
                f"{p} size {arr.size} doubles is not a multiple of "
                f"{INDEX_BEST_FULL_DOUBLES}"
            )
        return arr.reshape(-1, MAX_N_HKLS, 2)

    fb_path = Path(run_dir) / "Output" / "FitBest.bin"
    if not fb_path.exists():
        raise FileNotFoundError(
            f"{p} (python backend) and {fb_path} (c-omp backend fallback) "
            "both absent — cannot build the matched-spot table"
        )
    fb = read_fit_best(run_dir)                 # (N, 5000, 22) memmap
    n_seeds = fb.shape[0]
    ibf = np.zeros((n_seeds, MAX_N_HKLS, 2), dtype=np.float64)
    # Contiguous seed-chunked copy (avoids a strided whole-file col-0 gather
    # over NFS); col 0 ← FitBest col 0 (matched SpotID per hkl slot).
    chunk = 512
    for i0 in range(0, n_seeds, chunk):
        i1 = min(i0 + chunk, n_seeds)
        block = np.array(fb[i0:i1])             # (c, 5000, 22) in RAM
        ibf[i0:i1, :, 0] = block[:, :, 0]
    return ibf


def read_fit_best(run_dir: Union[str, Path]) -> np.memmap:
    """Read ``Output/FitBest.bin`` into a (N, 5000, 22) float64 memmap.

    Column 0 of the innermost axis is the SpotID for that matched theoretical
    hkl row. Other columns hold the y/z/omega observed + theoretical values
    and per-spot residuals. See module docstring for the full layout.

    The file may be slightly short of a clean multiple of ``110000`` doubles
    because the C pwrite path on some runs leaves the last seed slot
    uninitialised. We accept up to one trailing slot of slack and truncate
    the readable view; longer-than-expected files are still rejected as
    corrupt.
    """
    p = Path(run_dir) / "Output" / "FitBest.bin"
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.float64, mode="r")
    _assert_native(arr, str(p))
    n_seeds = arr.size // FIT_BEST_DOUBLES
    truncated_size = n_seeds * FIT_BEST_DOUBLES
    if arr.size > truncated_size + FIT_BEST_DOUBLES:
        raise ValueError(
            f"{p} size {arr.size} doubles is far from a multiple of "
            f"{FIT_BEST_DOUBLES}; file is corrupt."
        )
    return np.asarray(arr[:truncated_size]).reshape(n_seeds, MAX_N_HKLS, 22)


def read_orient_pos_fit(run_dir: Union[str, Path]) -> np.memmap:
    """Read ``Results/OrientPosFit.bin`` into a (N, 27) float64 memmap."""
    p = Path(run_dir) / "Results" / "OrientPosFit.bin"
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.float64, mode="r")
    _assert_native(arr, str(p))
    if arr.size % ORIENT_POS_FIT_DOUBLES != 0:
        raise ValueError(
            f"{p} size {arr.size} doubles is not a multiple of "
            f"{ORIENT_POS_FIT_DOUBLES}"
        )
    return arr.reshape(-1, ORIENT_POS_FIT_DOUBLES)


def read_key(run_dir: Union[str, Path]) -> np.memmap:
    """Read ``Results/Key.bin`` into a (N, 2) int32 memmap.

    Columns: (keep_flag, NrIDsPerID). A seed is alive iff ``keep_flag != 0``.
    """
    p = Path(run_dir) / "Results" / "Key.bin"
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.int32, mode="r")
    _assert_native(arr, str(p))
    if arr.size % KEY_INTS != 0:
        raise ValueError(
            f"{p} size {arr.size} ints is not a multiple of {KEY_INTS}"
        )
    return arr.reshape(-1, KEY_INTS)


def read_process_key(run_dir: Union[str, Path]) -> np.memmap:
    """Read ``Results/ProcessKey.bin`` into a (N, 5000) int32 memmap."""
    p = Path(run_dir) / "Results" / "ProcessKey.bin"
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.int32, mode="r")
    _assert_native(arr, str(p))
    n_seeds = arr.size // PROCESS_KEY_INTS
    truncated_size = n_seeds * PROCESS_KEY_INTS
    # The C pwrite path can leave the last partial slot uninitialized; tolerate
    # the file being one slot short of an exact multiple, but never long.
    if arr.size > truncated_size + PROCESS_KEY_INTS:
        raise ValueError(
            f"{p} size {arr.size} ints is far from a multiple of "
            f"{PROCESS_KEY_INTS}; file is corrupt."
        )
    return np.asarray(arr[:truncated_size]).reshape(n_seeds, PROCESS_KEY_INTS)


# ---------------------------------------------------------------------------
# Bundle reader
# ---------------------------------------------------------------------------


def read_all(
    run_dir: Union[str, Path],
    *,
    require_fit_best: bool = True,
    require_index_best_full: bool = True,
) -> BinaryInputs:
    """Read every binary input the PG pipeline needs.

    OrientPosFit / Key / ProcessKey are mandatory. FitBest and IndexBestFull
    are optional only if the caller knows the algorithm path won't touch
    them (e.g. running Phase 1 in isolation).
    """
    rd = Path(run_dir)
    opf = read_orient_pos_fit(rd)
    key = read_key(rd)
    pk = read_process_key(rd)

    # Sanity: row counts must match across files.
    n_seeds = opf.shape[0]
    if key.shape[0] != n_seeds:
        raise ValueError(
            f"Row-count mismatch: OrientPosFit has {n_seeds} seeds, "
            f"Key has {key.shape[0]}."
        )
    if pk.shape[0] not in (n_seeds, n_seeds - 1):
        # ProcessKey.bin can be one short due to pwrite alignment quirks
        # (observed on the peakfit hard dataset).
        raise ValueError(
            f"Row-count mismatch: OrientPosFit has {n_seeds} seeds, "
            f"ProcessKey has {pk.shape[0]}."
        )

    ib: Optional[np.memmap] = None
    try:
        ib = read_index_best(rd)
    except FileNotFoundError:
        pass
    if ib is not None and ib.shape[0] != n_seeds:
        raise ValueError(
            f"IndexBest seed count {ib.shape[0]} != OrientPosFit {n_seeds}"
        )

    ibf: Optional[np.memmap] = None
    if require_index_best_full:
        ibf = read_index_best_full(rd)
        if ibf.shape[0] != n_seeds:
            raise ValueError(
                f"IndexBestFull seed count {ibf.shape[0]} != "
                f"OrientPosFit {n_seeds}"
            )

    fb: Optional[np.memmap] = None
    if require_fit_best:
        fb = read_fit_best(rd)
        if fb.shape[0] not in (n_seeds, n_seeds - 1):
            raise ValueError(
                f"FitBest seed count {fb.shape[0]} != "
                f"OrientPosFit {n_seeds}"
            )

    return BinaryInputs(
        n_seeds=n_seeds,
        index_best=ib,
        index_best_full=ibf,
        fit_best=fb,
        orient_pos_fit=opf,
        key=key,
        process_key=pk,
    )
