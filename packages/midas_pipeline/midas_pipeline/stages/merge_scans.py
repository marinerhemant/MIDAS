"""Stage: merge_scans — pure-Python replacement for the C
``mergeScansScanning`` binary (236 LoC).

Weight-averages spots from spatially adjacent scans inside
``(tolPx, tolOme)``. Each merged-output position is the mean of its
input scan positions.

Key C-fidelity points (cover-to-cover read of
``FF_HEDM/src/mergeScansScanning.c``):

1. Spatial sort: per-scan CSVs are read in file order but **merged** in
   spatially-sorted order (sort key = the per-scan Y position).
   ``sort_perm[spatialIdx] = fileIdx`` (lines 18-26 + 58-63 of the C).
2. Each merge group has ``nMerges`` adjacent files. Output count is
   ``floor(nScans / nMerges)``.
3. Per group: read the first spatially-ordered file into ``allSpots``,
   then for each subsequent file find matches against the **previous**
   scan's spots (tracked via ``lastScansSpots`` chain), weight-averaging
   matched spots by GrainRadius and appending unmatched ones.
4. Match predicate (line 180-184 C):
     ``|ring diff| < 0.01 AND |Y diff| < tolPx AND |Z diff| < tolPx AND
       |omega diff| < tolOme``.
   The "0.01" ring tolerance is intentional — ring numbers are integers
   stored as float64.
5. Weight-averaging is over **all 16 columns** including SpotID (col 4)
   — but col 4 is overwritten at the end with the contiguous 1..N
   renumber (C line 210-211).
6. The merged CSV has 16 cols (NOT 18 — the C ``sscanf`` reads 18 but
   the ``fprintf`` writes 16). The merged file is what downstream
   ``SaveBinDataScanning`` reads next.

The inner spot-comparison loop is numba-jitted (``@njit(cache=True)``)
to recover the C's per-iteration speed. There is no torch path here —
the algorithm is pointer-chasing through irregular per-group arrays,
not vectorisable across groups. (Differentiability constraint: N/A —
this is data prep with hard branches, not a compute kernel.)

A pure-Python fallback path (no numba) is exposed for environments
where numba is unavailable. Tests exercise both.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .._logging import LOG
from ..results import MergeScansResult
from ._base import StageContext


# --- Optional numba import ------------------------------------------------

try:
    import numba
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba ships in midas_env
    numba = None
    njit = None
    _NUMBA_AVAILABLE = False


# Internal flag: tests may flip ``MERGE_SCANS_USE_NUMBA`` False to force
# the pure-Python path.
MERGE_SCANS_USE_NUMBA = _NUMBA_AVAILABLE


# ---------------------------------------------------------------------------
# Inner per-group merge loop — numba + pure-Python variants
# ---------------------------------------------------------------------------


def _merge_inner_py(
    all_spots: np.ndarray,
    n_all: int,
    last_scans_spots: np.ndarray,
    n_spots_last_scan: int,
    this_spots: np.ndarray,
    n_this: int,
    this_scans_spots: np.ndarray,
    tol_px: float,
    tol_ome: float,
) -> int:
    """Pure-Python reference of the C inner merge loop (lines 176-205).

    Matches each spot in ``this_spots`` against the previous scan's
    spots (indexed via ``last_scans_spots``) and either weight-averages
    into ``all_spots[j]`` or appends as a new spot at ``all_spots[n_all]``.

    Mutates ``all_spots`` and ``this_scans_spots`` in place. Returns
    the new ``n_all``.
    """
    for i in range(n_this):
        found = 0
        for l in range(n_spots_last_scan):
            j = last_scans_spots[l]
            if abs(this_spots[i, 5] - all_spots[j, 5]) >= 0.01:
                continue
            if abs(this_spots[i, 0] - all_spots[j, 0]) >= tol_px:
                continue
            if abs(this_spots[i, 1] - all_spots[j, 1]) >= tol_px:
                continue
            if abs(this_spots[i, 2] - all_spots[j, 2]) >= tol_ome:
                continue
            # Match — weight-average.
            found = 1
            orig_weight = all_spots[j, 3]
            new_weight = this_spots[i, 3]
            this_scans_spots[i] = j
            denom = orig_weight + new_weight
            for k in range(16):
                all_spots[j, k] = (
                    all_spots[j, k] * orig_weight
                    + this_spots[i, k] * new_weight
                ) / denom
            break
        if found == 0:
            this_scans_spots[i] = n_all
            for k in range(16):
                all_spots[n_all, k] = this_spots[i, k]
            n_all += 1
    return n_all


if _NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=False, nogil=True)
    def _merge_inner_nb(  # pragma: no cover - executed via numba dispatch
        all_spots: np.ndarray,
        n_all: int,
        last_scans_spots: np.ndarray,
        n_spots_last_scan: int,
        this_spots: np.ndarray,
        n_this: int,
        this_scans_spots: np.ndarray,
        tol_px: float,
        tol_ome: float,
    ) -> int:
        for i in range(n_this):
            found = 0
            for l in range(n_spots_last_scan):
                j = last_scans_spots[l]
                if abs(this_spots[i, 5] - all_spots[j, 5]) >= 0.01:
                    continue
                if abs(this_spots[i, 0] - all_spots[j, 0]) >= tol_px:
                    continue
                if abs(this_spots[i, 1] - all_spots[j, 1]) >= tol_px:
                    continue
                if abs(this_spots[i, 2] - all_spots[j, 2]) >= tol_ome:
                    continue
                found = 1
                orig_weight = all_spots[j, 3]
                new_weight = this_spots[i, 3]
                this_scans_spots[i] = j
                denom = orig_weight + new_weight
                for k in range(16):
                    all_spots[j, k] = (
                        all_spots[j, k] * orig_weight
                        + this_spots[i, k] * new_weight
                    ) / denom
                break
            if found == 0:
                this_scans_spots[i] = n_all
                for k in range(16):
                    all_spots[n_all, k] = this_spots[i, k]
                n_all += 1
        return n_all
else:  # pragma: no cover
    _merge_inner_nb = None  # type: ignore


def _merge_inner(*args, **kwargs) -> int:
    """Dispatch to numba or pure-Python based on ``MERGE_SCANS_USE_NUMBA``."""
    if MERGE_SCANS_USE_NUMBA and _merge_inner_nb is not None:
        return _merge_inner_nb(*args, **kwargs)
    return _merge_inner_py(*args, **kwargs)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

# NB: names match the 16 kept DATA columns (cols [0..13, 16, 17] of the
# 18-col InputAllExtra): 11/12 are raw detector pixels, 13 is the
# det-corrected omega, and the trailing pair is maskTouched/FitRMSE. The
# old header mislabeled 13-15 as "IntegratedIntensity RawSumIntensity
# FitRMSE" (the two intensities are exactly what the merge DROPS).
_MERGED_HEADER = (
    "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta "
    "OmegaIni YOrigDetCor ZOrigDetCor YRawPx ZRawPx "
    "OmegaDetCor maskTouched FitRMSE"
)


def _read_per_scan_csv_16(path: Path) -> np.ndarray:
    """Read an 18-col ``InputAllExtraInfoFittingAll{n}.csv`` and return
    the 16 columns the C merger keeps (cols [0..13, 16, 17]).

    Also tolerates 16-col files (already-merged input). 20/21-col files
    carry the appended OrigSpotID/ReturnCode (N2+E3) [+ DetID]; the base
    columns sit at the same indices, so the same keep-list applies.
    """
    arr = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 16), dtype=np.float64)
    arr = np.atleast_2d(arr)
    nc = arr.shape[1]
    if nc == 16:
        return arr
    if nc in (18, 19, 20, 21):
        # 18 base / +DetID / +OrigSpotID,ReturnCode / +both — base cols
        # are positionally identical; drop the dummies (14, 15).
        keep = np.concatenate([np.arange(14), np.arange(16, 18)])
        return arr[:, keep]
    raise ValueError(f"{path}: expected 16, 18..21 cols, got {nc}")


def _read_per_scan_csv_with_header(path: Path) -> tuple[np.ndarray, str]:
    with open(path) as f:
        header = f.readline().rstrip("\n")
    return _read_per_scan_csv_16(path), header


def _write_merged_csv(
    path: Path,
    rows16: np.ndarray,
    header: Optional[str] = None,
) -> None:
    """Write a merged 16-col CSV.

    Matches ``mergeScansScanning.c:213-220``:
      - Header line, then one row per spot.
      - Each value formatted with ``%lf`` (== Python ``%f`` = 6-digit
        fractional precision).
      - Trailing space after each value, newline after each row.
    """
    if rows16.shape[1] != 16:
        raise ValueError(f"merged CSV requires 16 cols, got {rows16.shape[1]}")
    if header is None:
        header = _MERGED_HEADER
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows16:
            for v in r:
                f.write(f"{v:f} ")
            f.write("\n")


# ---------------------------------------------------------------------------
# MergeScansSummary — lightweight in-memory result
# ---------------------------------------------------------------------------


@dataclass
class MergeScansSummary:
    """Per-call result of ``merge_scans()``. Distinct from the stage's
    ``MergeScansResult`` (which is a ``StageResult`` for orchestrator
    use); this lighter struct is what callers of the function get back.
    """

    out_csvs: List[Path]
    positions: np.ndarray
    positions_csv: Path
    n_spots_in: int
    n_spots_out: int


# ---------------------------------------------------------------------------
# Public function-level API (used by tests + the stage shell)
# ---------------------------------------------------------------------------


def merge_scans(
    per_scan_csvs: Sequence[Path],
    scan_positions: np.ndarray,
    *,
    tol_px: float,
    tol_ome: float,
    n_merges: int,
    out_dir: Optional[Path] = None,
    out_csv_template: str = "InputAllExtraInfoFittingAll{n}.csv",
    positions_csv_path: Optional[Path] = None,
    preserve_header: bool = False,
) -> MergeScansSummary:
    """Pure-Python rewrite of ``mergeScansScanning.c``.

    Parameters
    ----------
    per_scan_csvs
        Per-scan ``InputAllExtraInfoFittingAll{n}.csv`` paths (file
        order; spatial reordering done internally).
    scan_positions
        1-D ``(len(per_scan_csvs),)`` array of Y positions in µm.
    tol_px
        Pixel-space tolerance for the (Y, Z) match.
    tol_ome
        Omega-space tolerance for the omega match (degrees).
    n_merges
        Number of adjacent files per merge group. ``n_merges=1`` is
        passthrough (copies inputs). Larger values merge ``n_merges``
        spatially-adjacent files into one. Output count is
        ``floor(n_scans / n_merges)``. ``n_merges == n_scans`` is the
        full merge-all path used by merged-FF seeding.
    out_dir
        Output directory for merged CSVs and ``positions.csv``. Defaults
        to the parent of the first input CSV.
    out_csv_template
        Filename template for merged outputs. ``{n}`` is the merged
        scan index.
    positions_csv_path
        Optional override for the merged-positions CSV path. Defaults
        to ``out_dir / "positions.csv"``.
    preserve_header
        If True, emit the first file's header verbatim. If False
        (default), emit our canonical 16-col header.

    Returns
    -------
    MergeScansSummary
    """
    if n_merges <= 0:
        raise ValueError(f"n_merges must be > 0, got {n_merges}")
    csv_paths = [Path(p) for p in per_scan_csvs]
    n_scans = len(csv_paths)
    scan_positions = np.asarray(scan_positions, dtype=np.float64).ravel()
    if scan_positions.shape[0] != n_scans:
        raise ValueError(
            f"scan_positions has {scan_positions.shape[0]} entries but "
            f"{n_scans} CSVs were supplied"
        )
    if out_dir is None:
        out_dir = csv_paths[0].parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_fin = n_scans // n_merges
    if n_fin == 0:
        raise ValueError(
            f"floor(n_scans={n_scans} / n_merges={n_merges}) == 0 — no "
            "merged outputs would be produced"
        )

    # Spatial sort permutation (ascending Y), C lines 58-63.
    sort_perm = np.argsort(scan_positions, kind="stable")

    positions_new = np.zeros(n_fin, dtype=np.float64)
    n_spots_in_total = 0
    n_spots_out_total = 0
    out_paths: list[Path] = []

    for fin_scan_nr in range(n_fin):
        spatial_start = fin_scan_nr * n_merges
        first_file_idx = int(sort_perm[spatial_start])

        # Pre-scan to size buffers exactly (matches C lines 85-104).
        group_csvs: list[Path] = []
        group_positions_sum = 0.0
        per_file_lengths: list[int] = []
        for ms in range(n_merges):
            file_idx = int(sort_perm[spatial_start + ms])
            csv_path = csv_paths[file_idx]
            group_csvs.append(csv_path)
            group_positions_sum += float(scan_positions[file_idx])
            rows = _read_per_scan_csv_16(csv_path)
            if rows.shape[0] > 0:
                # ``GrainRadius < 0.01`` rows are dropped (C line 137-138).
                mask = rows[:, 3] >= 0.01
                per_file_lengths.append(int(mask.sum()))
            else:
                per_file_lengths.append(0)

        total_lines = int(sum(per_file_lengths))
        max_lines = int(max(per_file_lengths)) if per_file_lengths else 0

        # Pre-allocated like the C ``calloc`` block.
        all_spots = np.zeros((max(total_lines, 1), 16), dtype=np.float64)
        last_scans_spots = np.zeros(max(max_lines, 1), dtype=np.int64)
        this_scans_spots = np.zeros(max(max_lines, 1), dtype=np.int64)

        # Load the first file.
        rows0, header0 = _read_per_scan_csv_with_header(group_csvs[0])
        if rows0.shape[0] > 0:
            mask = rows0[:, 3] >= 0.01
            kept = rows0[mask]
        else:
            kept = np.zeros((0, 16), dtype=np.float64)
        n_all = kept.shape[0]
        n_spots_in_total += n_all
        all_spots[:n_all, :] = kept
        last_scans_spots[:n_all] = np.arange(n_all, dtype=np.int64)
        n_spots_last_scan = n_all

        # Merge each subsequent file.
        for scan_nr in range(1, n_merges):
            this_csv = group_csvs[scan_nr]
            rows_this = _read_per_scan_csv_16(this_csv)
            if rows_this.shape[0] > 0:
                mask_this = rows_this[:, 3] >= 0.01
                this_spots = np.ascontiguousarray(rows_this[mask_this])
            else:
                this_spots = np.zeros((0, 16), dtype=np.float64)
            n_this = this_spots.shape[0]
            n_spots_in_total += n_this

            # Resize buffers if needed.
            if last_scans_spots.shape[0] < n_this:
                new_len = max(n_this, last_scans_spots.shape[0] * 2)
                last_scans_spots = np.resize(last_scans_spots, new_len)
                this_scans_spots = np.resize(this_scans_spots, new_len)

            this_scans_spots[:n_this] = 0

            n_all = _merge_inner(
                all_spots, n_all,
                last_scans_spots, n_spots_last_scan,
                this_spots, n_this,
                this_scans_spots,
                float(tol_px), float(tol_ome),
            )

            # Carry forward (C line 206-207).
            last_scans_spots[:n_this] = this_scans_spots[:n_this]
            n_spots_last_scan = n_this

        # Renumber SpotID 1..n_all (C line 210-211).
        all_spots[:n_all, 4] = np.arange(1, n_all + 1, dtype=np.float64)

        # Write the merged CSV.
        out_path = out_dir / out_csv_template.format(n=fin_scan_nr)
        _write_merged_csv(
            out_path, all_spots[:n_all, :],
            header=(header0 if preserve_header else None),
        )
        out_paths.append(out_path)

        positions_new[fin_scan_nr] = group_positions_sum / n_merges
        n_spots_out_total += n_all

    if positions_csv_path is None:
        positions_csv_path = out_dir / "positions.csv"
    with open(positions_csv_path, "w") as f:
        for y in positions_new:
            f.write(f"{y:f}\n")

    return MergeScansSummary(
        out_csvs=out_paths,
        positions=positions_new,
        positions_csv=positions_csv_path,
        n_spots_in=n_spots_in_total,
        n_spots_out=n_spots_out_total,
    )


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(ctx: StageContext) -> MergeScansResult:
    """Run the merge_scans stage.

    FF mode: no-op (``skipped=True`` ``MergeScansResult``).

    PF mode: stage the per-scan CSVs from ``ctx.layer_dir`` into
    ``original_*.csv`` (idempotent), merge ``n_merges`` spatially-adjacent
    files via the function-level ``merge_scans`` API, write the merged
    CSVs back to ``ctx.layer_dir``, and emit ``positions.csv`` with the
    averaged per-group Ys.
    """
    started = time.time()
    cfg = ctx.config

    if ctx.is_ff:
        finished = time.time()
        return MergeScansResult(
            stage_name="merge_scans",
            started_at=started,
            finished_at=finished,
            duration_s=finished - started,
            inputs={"layer_dir": str(ctx.layer_dir)},
            outputs={},
            metrics={"scan_mode": "ff"},
            skipped=True,
        )

    n_scans = cfg.scan.n_scans
    scan_positions = cfg.scan.scan_positions

    # n_merges resolution priority: env var (tests) → seeding config
    # (``merged_n_merges`` if present) → default 1 (passthrough).
    import os
    env_merges = os.environ.get("MIDAS_PF_N_MERGES")
    if env_merges is not None:
        n_merges = int(env_merges)
    else:
        n_merges = int(getattr(cfg.seeding, "merged_n_merges", 1))
    n_merges = max(1, n_merges)

    # Tolerances from seeding config; sentinel -1 → conservative defaults.
    tol_px = cfg.seeding.merged_tol_px
    if tol_px <= 0:
        tol_px = 2.0
    tol_ome = cfg.seeding.merged_tol_ome
    if tol_ome <= 0:
        tol_ome = 2.0

    # Discover per-scan CSVs in layer_dir.
    per_scan: list[Path] = []
    for s in range(n_scans):
        p = ctx.layer_dir / f"InputAllExtraInfoFittingAll{s}.csv"
        if not p.exists():
            LOG.warning("merge_scans: missing %s — skipping stage", p)
            finished = time.time()
            return MergeScansResult(
                stage_name="merge_scans",
                started_at=started,
                finished_at=finished,
                duration_s=finished - started,
                inputs={"layer_dir": str(ctx.layer_dir)},
                outputs={},
                metrics={"scan_mode": "pf", "reason": f"missing {p.name}"},
                skipped=True,
            )
        per_scan.append(p)

    if n_merges == 1:
        # Passthrough: copy InputAllExtraInfoFittingAll*.csv files to
        # themselves with no merging. We still write ``positions.csv``
        # so downstream stages can rely on its presence.
        with open(ctx.layer_dir / "positions.csv", "w") as f:
            for y in scan_positions:
                f.write(f"{float(y):f}\n")
        finished = time.time()
        return MergeScansResult(
            stage_name="merge_scans",
            started_at=started,
            finished_at=finished,
            duration_s=finished - started,
            inputs={"layer_dir": str(ctx.layer_dir),
                    "n_scans": str(n_scans)},
            outputs={"positions_csv": str(ctx.layer_dir / "positions.csv")},
            metrics={
                "scan_mode": "pf",
                "n_merges": 1,
                "n_scans": n_scans,
                "passthrough": True,
                "numba_used": MERGE_SCANS_USE_NUMBA,
            },
            n_spots_in=0,
            n_spots_out=0,
        )

    # Stage originals (idempotent).
    staged: list[Path] = []
    for p in per_scan:
        orig = p.parent / f"original_{p.name}"
        if not orig.exists():
            shutil.copy(p, orig)
        staged.append(orig)

    summary = merge_scans(
        per_scan_csvs=staged,
        scan_positions=scan_positions,
        tol_px=tol_px,
        tol_ome=tol_ome,
        n_merges=n_merges,
        out_dir=ctx.layer_dir,
    )

    finished = time.time()
    return MergeScansResult(
        stage_name="merge_scans",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        inputs={
            "layer_dir": str(ctx.layer_dir),
            "n_scans": str(n_scans),
            "n_merges": str(n_merges),
            "tol_px": str(tol_px),
            "tol_ome": str(tol_ome),
        },
        outputs={
            "merged_csvs": ",".join(str(p) for p in summary.out_csvs),
            "positions_csv": str(summary.positions_csv),
        },
        metrics={
            "scan_mode": "pf",
            "n_spots_in": summary.n_spots_in,
            "n_spots_out": summary.n_spots_out,
            "n_merged_scans": len(summary.out_csvs),
            "numba_used": MERGE_SCANS_USE_NUMBA,
        },
        merged_csv=str(summary.out_csvs[0]) if summary.out_csvs else "",
        n_spots_in=summary.n_spots_in,
        n_spots_out=summary.n_spots_out,
    )
