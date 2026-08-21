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

    **Short final slot.** The C writer pwrites only ``nSpotsComp`` rows per
    seed at the full 5000-row stride (``FitUnified.c:2297``), so the file
    normally ends mid-slot and its length is NOT a multiple of the stride.
    Interior unwritten seeds are sparse zero holes, which is harmless. The
    final partial slot is served zero-padded by :func:`read_fit_best` — see
    :class:`TailPaddedBinary`. Floor-dividing by the stride instead silently
    deletes the last seed, which is what this reader used to do.

    Note the per-spot record is 22 doubles in the binary but 23 columns in
    the human-readable ``Results/FitBest_*.csv`` dumps: the binary drops the
    trailing ``RingNr``. Binary col N == CSV col N for N in 0..21.

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
    refinement. Same short-final-slot behaviour as ``FitBest.bin``
    (``FitUnified.c:2136`` pwrites ``nSpotsComp`` ints at a 5000-int stride);
    :func:`read_process_key` zero-pads the tail.

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
ORIENT_POS_FIT_DOUBLES = 27       # legacy width
ORIENT_POS_FIT_DOUBLES_V2 = 33    # + pre/post error triples (2026-08-21)
PROCESS_KEY_INTS = NR_MAX_IDS_PER_GRAIN
KEY_INTS = 2

# Column indices into the OrientPosFit row (after the 4 sentinels).
#
# Cols 22/23/24 are a HISTORICAL MIXTURE and are kept that way for
# bug-compatibility: 22 is the post-fit position error
# (FitErrors12D(FinalResult)/nSpotsComp) while 23/24 are the pre-fit ome and
# internal-angle means from CalcAngleErrors at the seed. Nothing in the file
# said so; it is ESTABLISHED (see the package CHANGELOG). Cols 27-32, present
# only in 33-wide files, are the clean same-estimator pre and post triples —
# prefer them for any before/after comparison.
ORIENT_POS_FIT_LAYOUT = {
    "orient_mat":   slice(1, 10),    # 9 elements row-major
    "position":     slice(11, 14),
    "lattice":      slice(15, 21),   # a, b, c, α, β, γ
    "pos_err":      22,              # post-fit (see note above)
    "ome_err":      23,              # PRE-fit
    "internal_ang": 24,              # PRE-fit. IAColNr=20 in C OPs; 24 here
    "mean_radius":  25,
    "completeness": 26,
    # 33-wide files only — KeyError on a 27-wide file, which is the intent:
    # asking for these on an old file should fail loudly, not return the
    # wrong column.
    "pos_err_pre":       27,
    "ome_err_pre":       28,
    "internal_ang_pre":  29,
    "pos_err_post":      30,
    "ome_err_post":      31,
    "internal_ang_post": 32,
}
ORIENT_POS_FIT_PREPOST_COLS = (27, 33)   # slice of the new triples

_NATIVE_ORDER = {"=", "|", "<"} if np.little_endian else {"=", "|", ">"}


def _assert_native(arr: np.ndarray, fname: str) -> None:
    """Fail loud if a binary file appears to be in non-native byte order."""
    bo = arr.dtype.byteorder
    if bo not in _NATIVE_ORDER:
        raise ValueError(
            f"{fname}: dtype byteorder {bo!r} is not native; "
            "midas-process-grains assumes native-endian binaries."
        )


def _resolve(run_dir: Union[str, Path], subfolder: str, filename: str) -> Path:
    """Resolve *filename* under ``run_dir/subfolder`` (c-omp convention) or,
    if absent, directly under ``run_dir`` (the bare-``OutputFolder``/
    ``ResultFolder`` convention the python indexer/refiner write natively —
    see ``midas_index``/``midas_fit_grain``'s ``ResultFolder`` docs). Callers
    still get a ``FileNotFoundError`` naming the subfolder path when neither
    exists, since that's the conventional/expected location."""
    sub = Path(run_dir) / subfolder / filename
    if sub.exists():
        return sub
    bare = Path(run_dir) / filename
    return bare if bare.exists() else sub


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
    p = _resolve(run_dir, "Output", "IndexBest.bin")
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
    p = _resolve(run_dir, "Output", "IndexBestFull.bin")
    if p.exists():
        arr = np.memmap(p, dtype=np.float64, mode="r")
        _assert_native(arr, str(p))
        if arr.size % INDEX_BEST_FULL_DOUBLES != 0:
            raise ValueError(
                f"{p} size {arr.size} doubles is not a multiple of "
                f"{INDEX_BEST_FULL_DOUBLES}"
            )
        return arr.reshape(-1, MAX_N_HKLS, 2)

    fb_path = _resolve(run_dir, "Output", "FitBest.bin")
    if not fb_path.exists():
        # c-omp FF refiner (FitUnified) emits Results/ProcessKey.bin — the
        # matched SpotID per hkl slot per seed — instead of the Output/
        # consolidated FitBest.bin. It carries the col-0 information ibf needs
        # (the matched-SpotID set per seed; col 1 / delta-omega is unused for
        # grain count), so synthesize the table from it.
        pk_path = _resolve(run_dir, "Results", "ProcessKey.bin")
        if pk_path.exists():
            # materialize(): read_process_key returns a per-seed view when the
            # C writer left a short final slot, and that view has no .astype.
            pk = materialize(read_process_key(run_dir))   # (N, 5000) int32
            ibf = np.zeros((pk.shape[0], MAX_N_HKLS, 2), dtype=np.float64)
            ibf[:, :, 0] = pk.astype(np.float64)
            return ibf
        raise FileNotFoundError(
            f"{p} (python backend), {fb_path} (c-omp FitBest) and {pk_path} "
            "(c-omp ProcessKey) all absent — cannot build the matched-spot table"
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


class TailPaddedBinary:
    """Read-only per-seed view over a pwrite-sparse binary whose FINAL slot
    is short, presenting it as though the tail were zero-padded.

    Why this exists
    ---------------
    ``FitUnified.c`` writes ``FitBest.bin`` and ``ProcessKey.bin`` with
    ``pwrite`` at a fixed per-seed stride but a payload of only
    ``nSpotsComp`` records (``FitUnified.c:2297`` and ``:2136``). Unwritten
    interior seeds become sparse zero holes, which is harmless — but the
    **last** seed leaves the file short of a clean multiple of the stride, so
    a reader that floor-divides by the stride drops that seed entirely.
    Silently, and always the last one.

    Measured on a 56,125-seed Ni FF layer (2026-08-21,
    ``ff_refiner_prepost``): ``FitBest.bin`` was 56,124 full slots **+ 87
    rows** and ``ProcessKey.bin`` 56,124 slots **+ 87 ints** — the same seed
    56,124, ``nSpotsComp = 87``. That seed is alive (SpotID 245283,
    ``keep_flag`` set, completeness 0.777, meanRadius 27.3 µm) and both
    ``OrientPosFit.bin`` and ``Key.bin`` see it at 56,125 rows. It was being
    discarded from clustering and from the residual sidecar with no warning,
    and it was also the cause of the "ProcessKey.bin can be one row short of
    OPF" truncation in ``compute/c_parity_run``.

    ``midas_fit_grain.io_binary.read_fit_best`` already zero-pads its tail
    and documents the mechanism; this is the same semantics for the memmap
    path, where materialising the whole file (49 GB on that layer) is not an
    option. Full slots stay zero-copy views into the memmap; only the final
    short slot is materialised (≤ 880 KB for FitBest, ≤ 20 KB for
    ProcessKey).

    Deliberately NOT an ``ndarray`` subclass: ``np.asarray`` on a subclass
    would hand back a base-class view of the *complete* slots only, which is
    exactly the silent truncation this class exists to remove. Bulk
    conversion raises instead — see :meth:`__array__`.
    """

    __slots__ = ("_full", "_tail", "_n_full", "_slot_shape", "_path")

    def __init__(self, full: np.ndarray, tail: Optional[np.ndarray],
                 slot_shape: Tuple[int, ...], path: Path):
        self._full = full                 # (n_full, *slot_shape) memmap view
        self._tail = tail                 # (*slot_shape,) padded copy or None
        self._n_full = int(full.shape[0])
        self._slot_shape = tuple(slot_shape)
        self._path = path

    # -- ndarray-ish surface the readers' callers actually use --------------
    @property
    def shape(self) -> Tuple[int, ...]:
        n = self._n_full + (1 if self._tail is not None else 0)
        return (n,) + self._slot_shape

    @property
    def dtype(self):
        return self._full.dtype

    @property
    def ndim(self) -> int:
        return 1 + len(self._slot_shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return (f"TailPaddedBinary({self._path.name}, shape={self.shape}, "
                f"dtype={self.dtype}, full_slots={self._n_full}, "
                f"padded_tail={self._tail is not None})")

    def __getitem__(self, idx):
        n = self.shape[0]
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            if i < 0:
                i += n
            if not (0 <= i < n):
                raise IndexError(
                    f"seed index {idx} out of range for {n} seeds in "
                    f"{self._path.name}"
                )
            if i < self._n_full:
                return self._full[i]
            return self._tail
        if isinstance(idx, slice):
            want = range(*idx.indices(n))
            if not want:
                return np.empty((0,) + self._slot_shape, dtype=self.dtype)
            # Fast path: entirely inside the memmapped full slots.
            if want.stop <= self._n_full and want.step == 1 and want.start >= 0:
                return self._full[want.start:want.stop]
            parts = []
            full_idx = [i for i in want if i < self._n_full]
            if full_idx:
                parts.append(self._full[full_idx])
            if any(i >= self._n_full for i in want):
                parts.append(self._tail[None, ...])
            return np.concatenate(parts, axis=0)
        raise TypeError(
            f"{type(self).__name__} supports integer and slice indexing on "
            f"the seed axis only (got {type(idx).__name__}). Index one seed "
            f"at a time — the file is too large to address as a whole."
        )

    def __array__(self, dtype=None, copy=None):
        raise TypeError(
            f"refusing to materialise {self._path.name} "
            f"({self.shape[0]} seeds x {np.prod(self._slot_shape):,} "
            f"elements) as a single array.\n"
            f"This file has a short final slot, so it is served as a "
            f"per-seed view; np.asarray() would have to copy the whole "
            f"file into RAM. Index per seed (arr[i]) or slice a chunk "
            f"(arr[i0:i1]), which is what every reader in this package "
            f"does. If you genuinely need it all, call .to_numpy() and "
            f"accept the allocation."
        )

    def to_numpy(self) -> np.ndarray:
        """Materialise every seed, tail included. Allocates the whole file."""
        if self._tail is None:
            return np.asarray(self._full)
        return np.concatenate([np.asarray(self._full),
                               self._tail[None, ...]], axis=0)


def materialize(arr) -> np.ndarray:
    """Full in-RAM copy of a per-seed view, short final slot included.

    Accepts either return type of :func:`read_fit_best` /
    :func:`read_process_key`, so callers do not have to branch on whether the
    file happened to end on a slot boundary.

    Use this ONLY where the whole file is genuinely needed as one array.
    ``ProcessKey.bin`` qualifies (the c_parity clustering indexes it randomly
    across seeds; ~1.1 GB int32 on a 56 k-seed layer). ``FitBest.bin`` does
    **not** — it was 49 GB on that same layer, and every reader in this
    package walks it per seed for exactly that reason.
    """
    if isinstance(arr, TailPaddedBinary):
        return arr.to_numpy()
    return np.array(arr, copy=True)


def _seeds_with_tail(
    arr: np.ndarray,
    slot_elems: int,
    record_elems: int,
    slot_shape: Tuple[int, ...],
    path: Path,
) -> Union[np.ndarray, TailPaddedBinary]:
    """Reshape a flat pwrite-sparse mapping into per-seed slots.

    Returns the plain reshaped view when the file is an exact multiple of the
    stride (the common case — byte-identical to the previous behaviour), and
    a :class:`TailPaddedBinary` when a short final slot is present.

    ``record_elems`` is the element count of one record inside a slot (22 for
    a FitBest row, 1 for a ProcessKey int). A trailing remainder that is not
    a whole number of records means the file is torn, not merely short.
    """
    n_full, remainder = divmod(int(arr.size), slot_elems)
    if remainder == 0:
        return np.asarray(arr).reshape((n_full,) + slot_shape)
    if remainder % record_elems != 0:
        raise ValueError(
            f"{path}: trailing {remainder} elements is not a whole number of "
            f"{record_elems}-element records ({remainder % record_elems} "
            f"left over) — the file is torn, not merely short of a full "
            f"final slot. Refusing to guess where the last seed ends."
        )
    tail = np.zeros(slot_shape, dtype=arr.dtype)
    tail.reshape(-1)[:remainder] = arr[n_full * slot_elems:]
    full = np.asarray(arr[:n_full * slot_elems]).reshape((n_full,) + slot_shape)
    return TailPaddedBinary(full, tail, slot_shape, path)


def read_fit_best(run_dir: Union[str, Path]):
    """Read ``Output/FitBest.bin`` as an (N, 5000, 22) float64 per-seed view.

    Column 0 of the innermost axis is the SpotID for that matched theoretical
    hkl row. Other columns hold the y/z/omega observed + theoretical values
    and per-spot residuals. See module docstring for the full layout.

    The C writer pwrites only ``nSpotsComp`` rows per seed at a 5000-row
    stride, so the final seed leaves the file short of a clean multiple. This
    used to be handled by truncating the view, which silently dropped that
    seed; it is now zero-padded and returned, so ``N`` matches the seed count
    in ``OrientPosFit.bin`` / ``Key.bin``. See :class:`TailPaddedBinary`.

    Returns a plain ndarray view for exact-multiple files, or a
    :class:`TailPaddedBinary` otherwise. Both support ``arr.shape``,
    ``arr[i]`` and ``arr[i0:i1]``; the latter refuses whole-file
    ``np.asarray`` rather than copying 49 GB.
    """
    p = _resolve(run_dir, "Output", "FitBest.bin")
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.float64, mode="r")
    _assert_native(arr, str(p))
    return _seeds_with_tail(
        arr, FIT_BEST_DOUBLES, 22, (MAX_N_HKLS, 22), p,
    )


def read_fit_best_final(run_dir: Union[str, Path]):
    """Read ``Output/FitBestFinal.bin`` — the POST-fit per-spot records.

    Identical layout, stride and short-final-slot behaviour to
    ``FitBest.bin``, but matched at the refined parameters instead of the
    indexer seed. Use this to ask "how well does the REFINED grain explain
    its spots"; ``FitBest.bin`` answers the same question about the seed.

    The two are **not row-aligned** — the post-fit matcher can select a
    different spot set, and ``nSpotsCompFinal`` can differ from
    ``nSpotsComp`` — so join them on SpotID (col 0), never by row index.

    Written only by refiners from 2026-08-21 onward; raises
    ``FileNotFoundError`` on older runs, where the pre-fit ``FitBest.bin``
    is all that exists.
    """
    p = _resolve(run_dir, "Output", "FitBestFinal.bin")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. It is written alongside FitBest.bin by refiners "
            f"from 2026-08-21; an older run has only the pre-fit FitBest.bin."
        )
    arr = np.memmap(p, dtype=np.float64, mode="r")
    _assert_native(arr, str(p))
    return _seeds_with_tail(arr, FIT_BEST_DOUBLES, 22, (MAX_N_HKLS, 22), p)


def _opf_row_is_consistent(arr: np.ndarray, ncols: int) -> bool:
    """Do the SpId sentinels line up when this file is read ``ncols`` wide?

    ``OrientPosFit.bin`` repeats the seed's SpotID in cols 0, 10, 14 and 21
    (``FitUnified.c`` builds OrientsFit[0] / PositionsFit[0] / StrainsFit[0] /
    ErrorsFin[0] all ``= SpId``). Read at the wrong width the rows misalign
    and those four stop agreeing, which makes this a content check that can
    actually fail — necessary because a size divisible by both 27 and 33
    (any multiple of 297 doubles) is arithmetically ambiguous.
    """
    if arr.size % ncols:
        return False
    m = arr.reshape(-1, ncols)
    live = m[m[:, 0] != 0]          # unwritten seeds are all-zero: no signal
    if live.shape[0] == 0:
        return True                 # nothing to contradict
    probe = live[:64]
    return bool(np.all(probe[:, [10, 14, 21]] == probe[:, [0]]))


def read_orient_pos_fit(run_dir: Union[str, Path]) -> np.memmap:
    """Read ``Results/OrientPosFit.bin`` into an (N, 27) or (N, 33) memmap.

    27 doubles per seed through 2026-08-21; **33** once the refiner carries the
    pre/post error triples (cols 27-29 = pre pos/ome/angle, 30-32 = post; see
    ``ORIENT_POS_FIT_LAYOUT``). The width is sniffed rather than assumed so
    that files written by either refiner generation still read.

    Sniffing order: the authoritative seed count from ``Key.bin`` when it is
    present (2 int32 per seed, always written in full, so ``size/8`` is exact);
    otherwise the sentinel-consistency check above. A file that satisfies
    neither width is rejected rather than guessed at.
    """
    p = _resolve(run_dir, "Results", "OrientPosFit.bin")
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.float64, mode="r")
    _assert_native(arr, str(p))

    widths = (ORIENT_POS_FIT_DOUBLES_V2, ORIENT_POS_FIT_DOUBLES)

    # Key.bin is the authoritative seed count when we have it.
    kp = _resolve(run_dir, "Results", "Key.bin")
    if kp.exists():
        n_seeds = kp.stat().st_size // (2 * 4)
        if n_seeds > 0:
            exact = [w for w in widths if arr.size == n_seeds * w]
            if len(exact) == 1:
                return arr.reshape(n_seeds, exact[0])
            # both (impossible for a fixed n_seeds) or neither → fall through

    # Arithmetic first. The sentinel check is only a TIE-BREAKER: plenty of
    # legitimate files (synthetic fixtures, and any writer that does not
    # repeat SpId) leave the sentinel columns unpopulated, and gating on them
    # would reject a perfectly unambiguous file.
    fits = [w for w in widths if arr.size % w == 0]
    if not fits:
        raise ValueError(
            f"{p}: size {arr.size} doubles is a multiple of neither "
            f"{ORIENT_POS_FIT_DOUBLES} nor {ORIENT_POS_FIT_DOUBLES_V2} "
            f"columns. Refusing to guess the row width — a wrong guess "
            f"silently shifts every column."
        )
    if len(fits) == 1:
        return arr.reshape(-1, fits[0])

    # Genuinely ambiguous: any multiple of 297 doubles divides by both.
    consistent = [w for w in fits if _opf_row_is_consistent(arr, w)]
    if len(consistent) == 1:
        return arr.reshape(-1, consistent[0])
    raise ValueError(
        f"{p}: size {arr.size} doubles divides by BOTH "
        f"{ORIENT_POS_FIT_DOUBLES} and {ORIENT_POS_FIT_DOUBLES_V2} columns "
        f"({arr.size // ORIENT_POS_FIT_DOUBLES} vs "
        f"{arr.size // ORIENT_POS_FIT_DOUBLES_V2} seeds), and the SpId "
        f"sentinels in cols 0/10/14/21 do not disambiguate "
        f"({len(consistent)} widths consistent). Refusing to guess — a wrong "
        f"guess silently shifts every column. Supply Results/Key.bin, whose "
        f"seed count is authoritative."
    )


def read_key(run_dir: Union[str, Path]) -> np.memmap:
    """Read ``Results/Key.bin`` into a (N, 2) int32 memmap.

    Columns: (keep_flag, NrIDsPerID). A seed is alive iff ``keep_flag != 0``.
    """
    p = _resolve(run_dir, "Results", "Key.bin")
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.int32, mode="r")
    _assert_native(arr, str(p))
    if arr.size % KEY_INTS != 0:
        raise ValueError(
            f"{p} size {arr.size} ints is not a multiple of {KEY_INTS}"
        )
    return arr.reshape(-1, KEY_INTS)


def read_process_key(run_dir: Union[str, Path]):
    """Read ``Results/ProcessKey.bin`` as an (N, 5000) int32 per-seed view.

    Same short-final-slot situation as :func:`read_fit_best` — the C writer
    pwrites ``nSpotsComp`` ints at a 5000-int stride (``FitUnified.c:2136``),
    so the last seed leaves a partial slot. Previously truncated away, now
    zero-padded and returned. See :class:`TailPaddedBinary`.
    """
    p = _resolve(run_dir, "Results", "ProcessKey.bin")
    if not p.exists():
        raise FileNotFoundError(p)
    arr = np.memmap(p, dtype=np.int32, mode="r")
    _assert_native(arr, str(p))
    return _seeds_with_tail(
        arr, PROCESS_KEY_INTS, 1, (PROCESS_KEY_INTS,), p,
    )


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

    ibf: Optional[np.ndarray] = None
    if require_index_best_full:
        ibf = read_index_best_full(rd)
        # The c-omp refiner skips writing ProcessKey/FitBest for a trailing seed
        # with zero matched spots, so the matched-spot table can be one seed
        # short of OrientPosFit (run-dependent). Pad the missing trailing seed
        # with an empty (all-zero) row rather than failing.
        if ibf.shape[0] == n_seeds - 1:
            ibf = np.concatenate(
                [np.asarray(ibf), np.zeros((1,) + ibf.shape[1:], dtype=ibf.dtype)],
                axis=0)
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
