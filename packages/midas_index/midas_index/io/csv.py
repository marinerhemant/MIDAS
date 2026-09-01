"""CSV / text-format readers.

  hkls.csv           header + 11 columns: h k l D-spacing RingNr g1 g2 g3 Theta(deg) 2Theta(deg) Radius
  SpotsToIndex.csv   one int per line (spot_id); some workflows write 2 ints
                     per line (newID origID) and only the first is consumed
  Grains.csv         MIDAS column-major grain-list format (mode A)

C references:
  - hkls.csv ingestion: FF_HEDM/src/IndexerOMP.c:2192-2227 (only rows whose
    RingNr is in Params.RingNumbers are kept)
  - SpotsToIndex.csv read: IndexerOMP.c:2302-2317
  - Grains.csv read: IndexerOMP.c:2358-2383 (header line + 8 skip lines + rows)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


# ----------------------------------------------------------------------
# hkls.csv
# ----------------------------------------------------------------------

# Columns of the on-disk hkls.csv; we keep the C-side 7-column layout for
# downstream consumers (matches `hkls[n_hkls][0..6]` in IndexerOMP.c:2212-2218):
#   [g1, g2, g3, RingNr, D-spacing, Theta, Radius]
HKLS_OUT_COLS = ("g1", "g2", "g3", "ring_nr", "d_spacing", "theta", "radius")
HKLS_INT_COLS = ("h", "k", "l", "ring_nr")


def read_hkls_csv(
    path: str | Path,
    ring_numbers: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse hkls.csv.

    If `ring_numbers` is given, keep only rows whose RingNr is in that set
    (matching the C filter in IndexerOMP.c:2206).

    Returns
    -------
    hkls_real : ndarray (n_hkls, 7), float64
        Layout `[g1, g2, g3, ring_nr, d_spacing, theta_rad, radius]`. **Theta
        is converted from degrees (as stored in hkls.csv) to radians**, so
        downstream consumers (`HEDMForwardModel`, `compute_ttheta`) get the
        canonical units used throughout midas-diffract / midas-index.
    hkls_int  : ndarray (n_hkls, 4), int32
        Layout `[h, k, l, ring_nr]`. Mirrors C `HKLints[][4]`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"hkls.csv not found at {path}")

    keep = None if ring_numbers is None else set(int(r) for r in ring_numbers)
    real_rows: list[tuple] = []
    int_rows: list[tuple] = []

    with open(path, "r") as fp:
        # First non-blank line is the header (e.g. "h k l D-spacing RingNr ...")
        header_consumed = False
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            if not header_consumed:
                header_consumed = True
                continue
            tokens = line.split()
            if len(tokens) < 11:
                # Skip malformed lines without raising — matches lenient C behavior.
                continue
            h, k, l = int(tokens[0]), int(tokens[1]), int(tokens[2])
            d_spacing = float(tokens[3])
            ring_nr = int(tokens[4])
            if keep is not None and ring_nr not in keep:
                continue
            g1, g2, g3 = float(tokens[5]), float(tokens[6]), float(tokens[7])
            # hkls.csv stores Theta in DEGREES; convert to radians here so all
            # downstream consumers (HEDMForwardModel, ring_ttheta math) see
            # the canonical unit. Mirrors the conversion in midas-diffract.
            theta_rad = float(tokens[8]) * math.pi / 180.0
            # tokens[9] is 2Theta — derivable from theta; not re-stored
            radius = float(tokens[10])
            real_rows.append((g1, g2, g3, float(ring_nr), d_spacing, theta_rad, radius))
            int_rows.append((h, k, l, ring_nr))

    if not real_rows:
        return (
            np.empty((0, 7), dtype=np.float64),
            np.empty((0, 4), dtype=np.int32),
        )
    hkls_real = np.asarray(real_rows, dtype=np.float64)
    hkls_int = np.asarray(int_rows, dtype=np.int32)
    return hkls_real, hkls_int


# ----------------------------------------------------------------------
# SpotsToIndex.csv
# ----------------------------------------------------------------------


def read_spots_to_index_csv(path: str | Path) -> np.ndarray:
    """Load SpotsToIndex.csv.

    Each line carries one int (mode B) or two ints (mode A: `newID origID`).
    Only the first int per line is returned, matching IndexerOMP.c:2313:
        sscanf(aline, "%d", &SpotIDs[i]);
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SpotsToIndex.csv not found at {path}")
    out: list[int] = []
    with open(path, "r") as fp:
        for raw in fp:
            tokens = raw.split()
            if not tokens:
                continue
            try:
                out.append(int(tokens[0]))
            except ValueError:
                continue
    return np.asarray(out, dtype=np.int64)


# ----------------------------------------------------------------------
# Grains.csv (mode A)
# ----------------------------------------------------------------------


#: Positional fallback for a Grains.csv whose preamble names no columns.
#: These are the indices C's ``sscanf`` in IndexerUnified.c:3644-3661 uses and
#: they are correct for every width the format has had (23, 24, 47, 53): the
#: appended blocks all landed to the RIGHT of GrainRadius.
_GRAINS_FALLBACK_IDX = {"id": 0, "om": 1, "x": 10, "y": 11, "z": 12,
                        "radius": 22}


def _grains_column_index(path: Path, lines: list[str]) -> tuple[dict[str, int], int]:
    """Resolve the column indices this reader needs, and where data starts.

    Two positional assumptions used to be baked in here and both were correct
    only by luck of the current file:

    * **A fixed eight-line skip.** The '%' preamble is nine lines TODAY, but it
      is written per-phase (``%NumPhases``, ``%PhaseInfo``, then a SpaceGroup
      and a Lattice Parameter line) and a multi-phase writer would add lines.
      We now consume '%' lines until the data starts, so the count is free to
      grow.
    * **``tokens[22]`` for GrainRadius.** True at every width so far, but the
      file has been widened four times. When the preamble carries a real
      column header we take the index from it.

    Anchoring on ``O11`` rather than on the ID token is deliberate: the column
    is spelled ``GrainID`` by ``c_parity_emit`` and ``ID`` by ``io/csv``, and
    both are on disk.

    Returns ``(index_map, first_data_line)``.
    """
    header: list[str] | None = None
    start = 1
    for i in range(1, len(lines)):
        if not lines[i].lstrip().startswith("%"):
            start = i
            break
        toks = lines[i].lstrip().lstrip("%").split()
        if "O11" in toks:
            header = toks
    else:
        start = len(lines)

    if header is None:
        return dict(_GRAINS_FALLBACK_IDX), start

    idx = dict(_GRAINS_FALLBACK_IDX)
    for key, name in (("x", "X"), ("y", "Y"), ("z", "Z"),
                      ("radius", "GrainRadius")):
        if name in header:
            idx[key] = header.index(name)
    for name in ("GrainID", "ID"):          # both spellings are on disk
        if name in header:
            idx["id"] = header.index(name)
            break
    idx["om"] = header.index("O11")

    # The C path (IndexerUnified.c) can only read this file positionally, so a
    # file whose OM or position block has MOVED would be read differently by
    # the two halves of the same mode-A run. Refuse loudly instead.
    if any(idx[k] != _GRAINS_FALLBACK_IDX[k] for k in ("id", "om", "x", "y", "z")):
        raise ValueError(
            f"{path}: Grains.csv column order has changed -- ID/O11/X/Y/Z are "
            f"at {[idx[k] for k in ('id', 'om', 'x', 'y', 'z')]}, not "
            f"{[_GRAINS_FALLBACK_IDX[k] for k in ('id', 'om', 'x', 'y', 'z')]}. "
            f"The C indexer reads this same file with a fixed sscanf and "
            f"cannot follow the header, so mode A would silently disagree with "
            f"itself. Header: {header!r}")
    for k in range(9):
        want = ("O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33")[k]
        if header[idx["om"] + k] != want:
            raise ValueError(
                f"{path}: orientation block is not contiguous row-major -- "
                f"expected {want} at column {idx['om'] + k}, found "
                f"{header[idx['om'] + k]!r}.")
    return idx, start


def read_grains_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Load Grains.csv (mode A — `isGrainsInput=1`).

    Mirrors IndexerUnified.c:3637-3663. Format:
      line 1:           "%NumGrains <n>"
      then:             a '%' preamble ending in a column-header line
      data lines:       <id> <O11..O33> <X> <Y> <Z> <...> <GrainRadius> <...>

    The preamble length and the GrainRadius column index are read from the
    file rather than hard-coded; see ``_grains_column_index``. Everything else
    stays positional on purpose — the C half of mode A reads the same file with
    a fixed ``sscanf`` and this reader must not diverge from it.

    Returns a dict of arrays keyed by:
      ids        (n,)      int32
      orient_mat (n, 3, 3) float64 (row-major order: O11..O33)
      positions  (n, 3)    float64
      radii      (n,)      float64
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grains.csv not found at {path}")

    lines = path.read_text().splitlines()
    head = lines[0].split() if lines else []
    if len(head) < 2 or not head[0].lstrip("%").startswith("NumGrains"):
        raise ValueError(
            f"Grains.csv: expected '%NumGrains N' on line 1, got {head!r}"
        )
    n_grains = int(head[1])
    idx, start = _grains_column_index(path, lines)
    # Only as many tokens as we actually index. This was a flat 23, which
    # silently skipped every row of a 21-column legacy file (GrainRadius at 19)
    # and returned zero grains.
    min_tokens = max(idx["radius"], idx["om"] + 8, idx["z"]) + 1

    ids = np.empty(n_grains, dtype=np.int32)
    orient_mat = np.empty((n_grains, 3, 3), dtype=np.float64)
    positions = np.empty((n_grains, 3), dtype=np.float64)
    radii = np.zeros(n_grains, dtype=np.float64)

    i = 0
    for raw in lines[start:]:
        if i >= n_grains:
            break
        if raw.lstrip().startswith("%"):    # a comment among the data rows
            continue
        tokens = raw.split()
        if len(tokens) < min_tokens:
            continue
        ids[i] = int(tokens[idx["id"]])
        for k in range(9):
            r, c = divmod(k, 3)
            orient_mat[i, r, c] = float(tokens[idx["om"] + k])
        positions[i, 0] = float(tokens[idx["x"]])
        positions[i, 1] = float(tokens[idx["y"]])
        positions[i, 2] = float(tokens[idx["z"]])
        try:
            radii[i] = float(tokens[idx["radius"]])
        except (IndexError, ValueError):
            pass
        i += 1

    if i != n_grains:
        # Truncate to actual number found (matches lenient C behavior).
        ids = ids[:i]
        orient_mat = orient_mat[:i]
        positions = positions[:i]
        radii = radii[:i]

    return {
        "ids": ids,
        "orient_mat": orient_mat,
        "positions": positions,
        "radii": radii,
    }


def write_spots_to_index_csv(
    path: str | Path,
    rows: Iterable[tuple[int, int]] | Iterable[int],
) -> None:
    """Write SpotsToIndex.csv.

    Mode A produces (newID origID) pairs (IndexerOMP.c:2429); mode B uses
    a single int per line. Accept either shape and write whatever is given.
    """
    path = Path(path)
    with open(path, "w") as fp:
        for row in rows:
            if isinstance(row, (tuple, list, np.ndarray)):
                fp.write(" ".join(str(int(v)) for v in row) + "\n")
            else:
                fp.write(f"{int(row)}\n")
