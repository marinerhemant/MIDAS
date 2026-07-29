"""Stage D of merged-FF seeding: Grains.csv → UniqueOrientations.csv.

The per-voxel scanning indexer reads its seed orientations from
``UniqueOrientations.csv`` when running with
``-ffSeedOrientations 1`` / ``seeding-mode = ff``. The merged-FF
pipeline produces ``Grains.csv`` from the merged spot file via the FF
indexer; this module converts that table into the seed format the
scanning indexer consumes.

Format contract (matches the parser used by
``pf_MIDAS.py:_orient_score_per_grain``,
``midas_pipeline.find_grains._cluster``, and the legacy seeded path):

  ``UniqueOrientations.csv`` — 14 cols, space-separated, no header.
    cols 0-4   key fields: grainID, RowNr, nSpots, StartRowNr, ListStartPos
                (only grainID is meaningful for seeding; the rest
                are bookkeeping. Set them to 0 unless the caller has
                better values.)
    cols 5-13  9-element orientation matrix (row-major 3×3) — this is
                what the indexer's seeded path reads at columns [5:14].

  ``Grains.csv`` (from FF ``ProcessGrains``) — header line starting
  with ``%GrainID`` followed by columns including an OM block. We
  parse the header to locate the OM columns dynamically (legacy
  ``ProcessGrains`` writes the OM as ``O11 O12 O13 O21 O22 O23 O31
  O32 O33``).

Optional dedup: symmetry-equivalent OMs in ``Grains.csv`` are kept
as separate rows by default (matches the existing seeded path which
tolerates them). Pass ``dedup_misorientation_deg`` to collapse pairs
closer than the given misorientation (radians via
``midas_stress.misorientation_om_batch``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# Column names emitted by ``ProcessGrains`` for the 9 OM entries.
_OM_COL_NAMES = ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33"]


def _parse_grains_csv(path: Path) -> Tuple[np.ndarray, List[int]]:
    """Read ``Grains.csv`` → (OMs (n, 9) float64, grain_ids).

    Locates the header line (starts with ``%GrainID``) and extracts
    the 9 OM columns by name (``O11 .. O33``). Falls back to a fixed
    column range if the header is missing.
    """
    lines = path.read_text().splitlines()
    header_idx: Optional[int] = None
    # The ID column has been spelled both ``GrainID`` and ``ID`` by different
    # ProcessGrains versions (current output starts ``%ID<TAB>O11...``), so key
    # off the OM columns — those are the ones actually consumed here and they
    # are stable. Anchoring on '%GrainID' alone silently rejected valid
    # Grains.csv files and broke FF→PF seeding entirely.
    for i, line in enumerate(lines):
        if line.startswith("%") and "O11" in line.split():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"{path}: no '%' header line containing the OM columns "
            f"{_OM_COL_NAMES} — not a ProcessGrains output file?"
        )
    cols = lines[header_idx][1:].split()         # strip leading '%'
    try:
        om_indices = [cols.index(n) for n in _OM_COL_NAMES]
    except ValueError as e:
        raise ValueError(
            f"{path}: header is missing one of {_OM_COL_NAMES}; got {cols!r}"
        ) from e
    gid_idx = 0
    for _gid_name in ("GrainID", "ID"):
        if _gid_name in cols:
            gid_idx = cols.index(_gid_name)
            break
    rows: List[List[float]] = []
    ids: List[int] = []
    for raw in lines[header_idx + 1:]:
        if not raw.strip() or raw.startswith("%"):
            continue
        toks = raw.split()
        if len(toks) <= max(*om_indices, gid_idx):
            continue
        try:
            rows.append([float(toks[i]) for i in om_indices])
            ids.append(int(float(toks[gid_idx])))
        except (ValueError, IndexError):
            continue
    oms = np.asarray(rows, dtype=np.float64)
    return oms, ids


def _write_unique_orientations(path: Path, oms: np.ndarray,
                               grain_ids: List[int]) -> None:
    """Emit the 14-col ``UniqueOrientations.csv`` consumed by the
    seeded scanning indexer + the find_grains clusterer.
    """
    n = oms.shape[0]
    if n == 0:
        path.write_text("")
        return
    # 5 key cols + 9 OM cols = 14 cols.
    arr = np.zeros((n, 14), dtype=np.float64)
    arr[:, 0] = grain_ids if len(grain_ids) == n else np.arange(1, n + 1)
    # Cols 1-4 (RowNr, nSpots, StartRowNr, ListStartPos) are zero; the
    # seeded indexer ignores them and just keys on cols 5-13.
    arr[:, 5:14] = oms
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, fmt="%.9f")


def _dedup_by_misorientation(oms: np.ndarray, ids: List[int],
                             space_group: int,
                             angle_threshold_deg: float
                             ) -> Tuple[np.ndarray, List[int]]:
    """Drop symmetry-equivalent OMs within ``angle_threshold_deg``.

    O(n^2) pairwise check on the (potentially small) seed set; for the
    typical 5-50 unique grains this is negligible. Returns the
    surviving (OMs, ids).
    """
    if oms.shape[0] <= 1 or angle_threshold_deg <= 0:
        return oms, list(ids)
    from midas_stress.orientation import misorientation_om_batch
    threshold_rad = np.deg2rad(angle_threshold_deg)
    keep = np.ones(oms.shape[0], dtype=bool)
    for i in range(oms.shape[0]):
        if not keep[i]:
            continue
        # Compare oms[i] against all later rows still kept.
        later = np.arange(i + 1, oms.shape[0])
        later = later[keep[later]]
        if later.size == 0:
            continue
        angs = misorientation_om_batch(
            np.tile(oms[i], (later.size, 1)),
            oms[later],
            space_group,
        )
        too_close = np.asarray(angs) < threshold_rad
        for j_pos, drop in zip(later, too_close):
            if drop:
                keep[j_pos] = False
    return oms[keep], [g for g, k in zip(ids, keep) if k]


def grains_csv_to_unique_orientations(
    grains_csv: str | Path,
    unique_orientations_csv: str | Path,
    *,
    space_group: int = 225,
    dedup_misorientation_deg: float = 0.0,     # 0 ⇒ no dedup
) -> int:
    """Convert ``Grains.csv`` to ``UniqueOrientations.csv``.

    Returns the number of seed grains written.

    Parameters
    ----------
    grains_csv : path
        Output of FF ``ProcessGrains`` (header line starts with
        ``%GrainID``; OM cols labeled ``O11 .. O33``).
    unique_orientations_csv : path
        Destination 14-col table that the per-voxel scanning indexer
        consumes when ``seeding-mode = ff`` (or merged-ff).
    space_group : int
        Symmetry group for the optional dedup step (used by
        ``midas_stress.misorientation_om_batch``). Defaults to 225
        (FCC) per project default.
    dedup_misorientation_deg : float
        If > 0, drop symmetry-equivalent OMs within this angle.
        Default 0 leaves the table verbatim (the seeded indexer
        tolerates duplicates).
    """
    grains_csv = Path(grains_csv)
    unique_orientations_csv = Path(unique_orientations_csv)
    oms, ids = _parse_grains_csv(grains_csv)
    if dedup_misorientation_deg > 0:
        oms, ids = _dedup_by_misorientation(
            oms, ids, space_group, dedup_misorientation_deg,
        )
    _write_unique_orientations(unique_orientations_csv, oms, ids)
    return oms.shape[0]
