"""Adapt the c-omp refiner's ``FitBest_<vox>_<sp>.csv`` output into the
``Result_OrientPos_voxel_<vox>.csv`` form that ``midas_pf_odf`` and
``midas_pipeline.stages.consolidation_pf`` consume.

Why an adapter and not a rename
-------------------------------
``FitUnified.c`` writes a **multi-block** CSV per (voxel, seed)::

    line 1 : header
    line 2 : the 39-col refined result  (OM 1-9, pos 11-13, lattice 15-20,
             strain 27-35, Euler 36-38)   <- the only line pf-odf reads
    line 3 : header again
    line 4+: one row per matched spot

``midas_pf_odf.io._read_voxel_result`` does ``np.loadtxt(skiprows=1)`` and
requires a single 1-D row, so it chokes on the trailing header + per-spot rows.
:func:`fitbest_to_result_orientpos` extracts just the refined-result line
(and, when a voxel has several ``FitBest_<vox>_<sp>`` solutions, keeps the
highest-completeness one), writing a clean 2-line
``Result_OrientPos_voxel_<vox>.csv``.

The extracted columns (OM in 1-9, lattice in 15-20) are exactly what pf-odf's
``_read_voxel_result`` reads (``row[1:10]`` / ``row[15:21]``), so this is the
permanent bridge from the fast C refiner to the peak-shape strain code.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# 39 legacy names + the 6 pre/post error columns appended by the refiner from
# 2026-08-21 (FitUnified.c writes them AFTER Eul3 in the CSV, even though they
# are contiguous at 27-32 in the binary OrientPosFit.bin — the CSV has the
# strain tensor and Euler block in between). Cols 22-24 keep their historical
# mixture (22 post-fit, 23/24 pre-fit); prefer the *Pre/*Post pairs.
#
# The header must match the token count of whatever the refiner wrote, or
# every column is labelled wrong. `_header_for` picks the right one.
#
# ASYMMETRIC GUARD — read this before relying on it. `_header_for` protects
# new-binary + NEW-adapter (this file). It CANNOT protect new-binary +
# OLD-adapter, which is the dangerous direction: an old adapter forwards all
# 45 tokens and writes its fixed 39-name header, giving 45 values under 39
# headings, silently. The binary and this package must be updated together or
# not at all; nothing in either can detect the mismatch from the other side.
#
# INDEX WARNING — the same six quantities live at DIFFERENT indices in the two
# carriers. In binary OrientPosFit.bin they are contiguous at 27-32; in this
# CSV they are at 39-44, because the strain tensor (27-35) and Euler block
# (36-38) sit in between. Do not assume a shared index.
_HEADER_39 = (
    "SpotID O11 O12 O13 O21 O22 O23 O31 O32 O33 "
    "SpotID x y z "
    "SpotID a b c alpha beta gamma "
    "SpotID PosErr OmeErr InternalAngle Radius Completeness "
    "E11 E12 E13 E21 E22 E23 E31 E32 E33 "
    "Eul1 Eul2 Eul3\n"
)
# PF SEMANTICS DIFFER FROM FF, and the column names cannot say so on their own.
# `isFF = (gNumScans <= 1)` gates fit stages 2 and 4 (FitUnified.c:1991/2015),
# which are the POSITION fits — so **PF does not refine position at all**. In a
# PF run `PosErrPre`/`PosErrPost` therefore move only through the orientation
# and lattice fit, and the difference is NOT the same quantity as FF's, where
# the position stages are the main thing shrinking it. Do not compare a PF
# PosErr improvement against an FF one, and do not read a small PF change as a
# failed position fit — there was no position fit.
_HEADER_45 = (
    _HEADER_39.rstrip("\n")
    + " PosErrPre OmeErrPre InternalAnglePre"
      " PosErrPost OmeErrPost InternalAnglePost\n"
)
_HEADER = _HEADER_39          # kept for backwards compatibility of the name


def _header_for(n_tokens: int) -> str:
    """Header whose name count matches the data row actually present.

    A 39-name header over a 45-value row (or vice versa) mislabels every
    column downstream and nothing errors, so refuse rather than guess.
    """
    if n_tokens == 39:
        return _HEADER_39
    if n_tokens == 45:
        return _HEADER_45
    raise ValueError(
        f"FitBest result row has {n_tokens} columns; this adapter knows the "
        f"39-column (pre-2026-08-21) and 45-column (with pre/post error "
        f"triples) layouts. Refusing to write a header that does not match "
        f"the data."
    )
_FN = re.compile(r"FitBest_(\d+)_(\d+)\.csv$")
_COMPLETENESS_COL = 26          # 0-indexed within the refined-result row


def _first_data_row(fitbest_path: str) -> list[str] | None:
    """Second physical line (the refined-result row) as whitespace tokens,
    or ``None`` if the file has no data line."""
    with open(fitbest_path) as fh:
        fh.readline()               # header
        data = fh.readline()
    if not data:
        return None
    toks = data.split()
    return toks if toks else None


def fitbest_to_result_orientpos(
    fitbest_dir: str | Path, results_dir: str | Path,
) -> int:
    """Convert every ``FitBest_*.csv`` in *fitbest_dir* into
    ``Result_OrientPos_voxel_<vox>.csv`` in *results_dir*.

    When a voxel has several ``FitBest_<vox>_<sp>`` solutions the
    highest-completeness one (col 26) is kept.

    Returns the number of per-voxel result files written.
    """
    fitbest_dir = Path(fitbest_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    best: dict[int, tuple[float, list[str]]] = {}   # voxNr -> (completeness, tokens)
    for ent in os.scandir(fitbest_dir):
        m = _FN.search(ent.name)
        if not m:
            continue
        vox = int(m.group(1))
        toks = _first_data_row(ent.path)
        if toks is None or len(toks) <= _COMPLETENESS_COL:
            continue
        try:
            comp = float(toks[_COMPLETENESS_COL])
        except ValueError:
            continue
        prev = best.get(vox)
        if prev is None or comp > prev[0]:
            best[vox] = (comp, toks)

    for vox, (_comp, toks) in best.items():
        out = results_dir / f"Result_OrientPos_voxel_{vox}.csv"
        with open(out, "w") as fh:
            fh.write(_header_for(len(toks)))
            fh.write(" ".join(toks) + "\n")
    return len(best)
