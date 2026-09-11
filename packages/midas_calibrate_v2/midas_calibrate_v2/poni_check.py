"""Which reading of a PONI do the DATA agree with? Test a calibration against Friedel pairs.

A ``.poni`` names a point in pyFAI's array order. Whether your reader returns the same row and
column order -- and which package-specific name maps to which axis -- is recorded nowhere in the
file, and a wrong reading is silent. On La3Ni2O7 the delivered PONI was row-flipped against
``tifffile`` by 57 px (``orientation: 3`` is pyFAI's default and declares no flip), and feeding
the right numbers through the wrong axis name put a centre 123 px off.

This module enumerates the readings and lets the sample's own Friedel pairs decide, axis by axis.

It deliberately does NOT use ring sharpness. Tried on those data (2026-09-10) an azimuthal-median
"ring sharpness" score picked the reading 170 px from the true centre on one sample and 131 px on
the other -- single-crystal spots and module gaps give an azimuthal-median profile structure about
a wrong centre as easily as rings do about the right one. A statistic that can fail is necessary;
this one failed.

Usage::

    from midas_integrate_v2.compat.pyfai import read_poni
    p = read_poni(path); px = p["Detector_config"]["pixel1"]
    chk = check_poni_against_friedel(spot_rows_cols,
                                     poni1_px=p["Poni1"]/px, poni2_px=p["Poni2"]/px,
                                     n_rows=1679, n_cols=1475, lsd_um=p["Distance"]*1e6,
                                     pixel_um=px*1e6, rot1_rad=p["Rot1"], rot2_rad=p["Rot2"])
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .friedel import beam_centre_from_pairs

__all__ = ["PoniCheck", "poni_readings", "check_poni_against_friedel"]


@dataclass(frozen=True)
class PoniCheck:
    """A PONI point, tested against a beam centre measured from Friedel pairs."""

    friedel_row: float
    friedel_col: float
    n_pairs: int
    p_value: float
    axes: str                  #: "as given" | "swapped" | "undecidable"
    row_flip: str              #: "as given" | "flipped" | "undecidable"
    col_flip: str              #: "as given" | "flipped" | "undecidable"
    row: float                 #: the PONI point under the decided reading (as given where undecidable)
    col: float
    offset_px: float           #: its distance from the Friedel centre
    expected_offset_px: float  #: point of normal incidence vs beam centre, from the tilts
    within_expected: bool
    readings: Tuple[Tuple[str, float, float, float], ...]   #: (label, row, col, distance), nearest first
    notes: Tuple[str, ...]


def poni_readings(poni1_px: float, poni2_px: float, n_rows: int, n_cols: int):
    """All eight (axis swap x row flip x column flip) readings as ``(label, row, col)`` in YOUR array."""
    out = []
    for swap in (False, True):
        r0, c0 = (poni2_px, poni1_px) if swap else (poni1_px, poni2_px)
        for rf in (False, True):
            for cf in (False, True):
                label = ("axes swapped" if swap else "as given") + (", row-flipped" if rf else "") \
                        + (", col-flipped" if cf else "")
                out.append((label, float(n_rows - 1 - r0 if rf else r0),
                            float(n_cols - 1 - c0 if cf else c0)))
    return out


def _decide(a: float, b: float, target: float, tol: float):
    da, db = abs(a - target), abs(b - target)
    if abs(da - db) <= tol:
        return "undecidable"
    return "as given" if da < db else "flipped"


def check_poni_against_friedel(points: np.ndarray, *, poni1_px: float, poni2_px: float,
                               n_rows: int, n_cols: int, lsd_um: Optional[float] = None,
                               pixel_um: Optional[float] = None, rot1_rad: float = 0.0,
                               rot2_rad: float = 0.0, search_px: float = 150.0,
                               floor_px: float = 5.0, n_null: int = 200,
                               rng_seed: int = 0) -> PoniCheck:
    """Measure the beam centre from Friedel pairs and decide which PONI reading it matches.

    ``points`` are spot ``(row, col)`` in your reader's array order. The centre is measured
    UNSEEDED and again seeded from the nearest reading with a WIDE window, and the run with more
    pairs is kept: a narrow seeded search can return a spurious optimum that still reports
    ``p = 0`` (seen at 6-7 pairs against 44-69 on real data).

    Each axis is decided separately and may come back ``"undecidable"`` -- a column flip cannot
    be judged when the beam centre sits within the tolerance of the detector's central column,
    because both readings are then equally far from it. That is the honest answer, not a failure.

    The tolerance is the larger of ``floor_px`` and 1.5x the expected offset between the point of
    normal incidence and the beam centre, ``lsd_um * tan(hypot(rot1, rot2)) / pixel_um`` (about
    15 px at 0.4 deg and Lsd ~ 350 mm).
    """
    pts = np.asarray(points, float)
    notes = []
    b0 = beam_centre_from_pairs(pts, seed=None, n_null=n_null, rng_seed=rng_seed)
    expected = 0.0
    if lsd_um and pixel_um:
        expected = float(lsd_um * math.tan(math.hypot(rot1_rad, rot2_rad)) / pixel_um)
    tol = max(float(floor_px), 1.5 * expected)
    read = poni_readings(poni1_px, poni2_px, n_rows, n_cols)
    nearest0 = min(read, key=lambda t: math.hypot(t[1] - b0.row, t[2] - b0.col))
    b1 = beam_centre_from_pairs(pts, seed=(nearest0[1], nearest0[2]), search_px=search_px,
                                n_null=n_null, rng_seed=rng_seed)
    b = b1 if b1.n_pairs > b0.n_pairs else b0
    if math.hypot(b1.row - b0.row, b1.col - b0.col) > 2.0:
        notes.append(f"unseeded ({b0.row:.2f}, {b0.col:.2f}; {b0.n_pairs} pairs) and wide-seeded "
                     f"({b1.row:.2f}, {b1.col:.2f}; {b1.n_pairs} pairs) disagree; kept the one with more pairs")
    fr, fc = float(b.row), float(b.col)

    best = {}
    for swap in (False, True):
        r0, c0 = (poni2_px, poni1_px) if swap else (poni1_px, poni2_px)
        best[swap] = math.hypot(min(abs(r0 - fr), abs(n_rows - 1 - r0 - fr)),
                                min(abs(c0 - fc), abs(n_cols - 1 - c0 - fc)))
    if abs(best[False] - best[True]) <= tol or max(best.values()) < 2.0 * min(best.values()):
        axes = "undecidable"; swap = best[True] < best[False]
        notes.append("axis swap undecidable from the centre alone")
    else:
        swap = best[True] < best[False]; axes = "swapped" if swap else "as given"
    r0, c0 = (poni2_px, poni1_px) if swap else (poni1_px, poni2_px)
    row_flip = _decide(r0, n_rows - 1 - r0, fr, tol)
    col_flip = _decide(c0, n_cols - 1 - c0, fc, tol)
    for axis, dec, centre, n in (("row", row_flip, fr, n_rows), ("column", col_flip, fc, n_cols)):
        if dec == "undecidable":
            notes.append(f"{axis} flip undecidable: the beam centre ({centre:.1f}) is within "
                         f"{tol:.1f} px of the detector's central {axis} ({(n - 1) / 2:.1f}) "
                         f"on the scale that matters")
    row = n_rows - 1 - r0 if row_flip == "flipped" else r0
    col = n_cols - 1 - c0 if col_flip == "flipped" else c0
    offset = float(math.hypot(row - fr, col - fc))
    within = offset <= tol
    if not within:
        notes.append(f"even the decided reading is {offset:.1f} px from the Friedel centre, beyond the "
                     f"{tol:.1f} px the tilts allow -- the PONI may belong to another geometry")
    ranked = tuple(sorted(((l, r, c, float(math.hypot(r - fr, c - fc))) for l, r, c in read),
                          key=lambda t: t[3]))
    return PoniCheck(friedel_row=fr, friedel_col=fc, n_pairs=int(b.n_pairs), p_value=float(b.p_value),
                     axes=axes, row_flip=row_flip, col_flip=col_flip, row=float(row), col=float(col),
                     offset_px=offset, expected_offset_px=expected, within_expected=bool(within),
                     readings=ranked, notes=tuple(notes))
