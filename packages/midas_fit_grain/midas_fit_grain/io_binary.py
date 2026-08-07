"""Binary I/O for FitPosOrStrainsOMP-compatible artifacts.

Layouts (little-endian native; the C code uses raw fwrite/pwrite of the host
byte order, so we do too — this matches the on-disk format produced by
MIDAS today):

* ``ExtraInfo.bin``         — ``nSpots × 16 doubles``  (read-only, mmap'd here)
* ``OrientPosFit.bin``      — ``nGrains × 27 doubles`` (one row per
                              ``SpotsToIndex.csv`` entry, addressed by
                              ``rowNr * 27 * 8`` byte offset)
* ``FitBest.bin``           — sparse: each grain occupies
                              ``MaxNHKLS * 22 doubles`` slots starting at
                              ``rowNr * MaxNHKLS * 22 * 8``; only the first
                              ``nSpotsComp`` rows are written, the rest stay
                              zero-filled.
* ``Key.bin``               — ``nGrains × 2 ints32`` ``(SpotID, nSpotsComp)``
* ``ProcessKey.bin``        — ``nGrains × MaxNHKLS ints32``; first
                              ``nSpotsComp`` ints are spot IDs.

All writers are pwrite-style: open ``O_CREAT | O_WRONLY``, seek to
``rowNr × stride``, write the payload, close. Safe for concurrent
multi-process writers as long as each worker owns a disjoint set of rows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# Constants matching FitPosOrStrainsOMP.c
# ----------------------------------------------------------------------------
EXTRA_INFO_NCOLS = 16            # ExtraInfo.bin row width
ORIENT_POS_FIT_NCOLS = 27        # OrientPosFit.bin row width
FIT_BEST_NCOLS = 22              # FitBest.bin row width per matched spot
MAX_NHKLS_DEFAULT = 5000         # FitPosOrStrainsOMP MaxNHKLS — strides FitBest

# ExtraInfo.bin column meanings (binary order; CSV's IntegratedIntensity /
# RawSumIntensity are dummies in the binary — see SaveBinData.c:124-133).
EXTRA_INFO_COLS = {
    0: "YLab",          # detector-corrected + wedge-corrected lab Y (um)
    1: "ZLab",
    2: "Omega",         # wedge-corrected omega (deg)
    3: "GrainRadius",   # ring-derived
    4: "SpotID",
    5: "RingNumber",
    6: "Eta",           # deg
    7: "Ttheta",        # 2theta deg
    8: "OmegaIni",      # raw omega, no wedge correction
    9: "YOrig",         # raw lab Y, no wedge correction
    10: "ZOrig",
    11: "YOrigDetCor",  # detector-corrected only (no wedge)
    12: "ZOrigDetCor",
    13: "OmegaOrigDetCor",
    14: "maskTouched",
    15: "FitRMSE",
}

# ----------------------------------------------------------------------------
# Dataclasses (light wrappers over numpy views)
# ----------------------------------------------------------------------------


@dataclass
class ExtraInfoSpot:
    """One row of ExtraInfo.bin (16 doubles)."""
    YLab: float
    ZLab: float
    Omega: float
    GrainRadius: float
    SpotID: int
    RingNumber: int
    Eta: float
    Ttheta: float
    OmegaIni: float
    YOrig: float
    ZOrig: float
    YOrigDetCor: float
    ZOrigDetCor: float
    OmegaOrigDetCor: float
    maskTouched: float
    FitRMSE: float

    @classmethod
    def from_row(cls, row: np.ndarray) -> "ExtraInfoSpot":
        if row.shape != (EXTRA_INFO_NCOLS,):
            raise ValueError(f"expected length-16, got {row.shape}")
        return cls(
            YLab=float(row[0]), ZLab=float(row[1]), Omega=float(row[2]),
            GrainRadius=float(row[3]), SpotID=int(row[4]),
            RingNumber=int(row[5]), Eta=float(row[6]), Ttheta=float(row[7]),
            OmegaIni=float(row[8]), YOrig=float(row[9]), ZOrig=float(row[10]),
            YOrigDetCor=float(row[11]), ZOrigDetCor=float(row[12]),
            OmegaOrigDetCor=float(row[13]),
            maskTouched=float(row[14]), FitRMSE=float(row[15]),
        )


@dataclass
class GrainResult:
    """One refined grain — packed into 27 doubles for OrientPosFit.bin.

    ``ErrorPos``/``ErrorOme``/``ErrorAngle`` map to OrientPosFit.bin cols
    22-24, which ``midas_process_grains`` reads back as Grains.csv's
    ``DiffPos``/``DiffOme``/``DiffAngle`` (see ``FitPosOrStrainsOMP.c``
    ``CalcAngleErrors``: ``Error[0]``=PosErr, ``Error[1]``=OmeErr,
    ``Error[2]``=IntAngle, and its ``OutMatr[22..24]`` assignment). Do not
    reorder without updating that column contract.
    """
    SpotID: int
    OrientMat: np.ndarray         # (9,) row-major 3x3
    Position: np.ndarray          # (3,)  um
    LatticeFit: np.ndarray        # (6,)  refined a,b,c,alpha,beta,gamma
    ErrorPos: float     # mean matched-spot position error, um
    ErrorOme: float     # mean matched-spot omega error, deg
    ErrorAngle: float   # mean matched-spot internal angle error, deg
    meanRadius: float
    completeness: float

    def to_row(self) -> np.ndarray:
        out = np.zeros(ORIENT_POS_FIT_NCOLS, dtype=np.float64)
        out[0] = self.SpotID
        out[1:10] = self.OrientMat
        out[10] = self.SpotID
        out[11:14] = self.Position
        out[14] = self.SpotID
        out[15:21] = self.LatticeFit
        out[21] = self.SpotID
        out[22] = self.ErrorPos
        out[23] = self.ErrorOme
        out[24] = self.ErrorAngle
        out[25] = self.meanRadius
        out[26] = self.completeness
        return out


@dataclass
class PerSpotFit:
    """One matched-spot row for FitBest.bin (22 doubles).

    Layout follows ``SpotsComp`` in FitPosOrStrainsOMP.c:3032-3037.
    Only the columns explicitly populated by the C code are typed; the
    remainder are passed through as a generic ``double`` vector.
    """
    raw: np.ndarray                # (22,) the entire row, ready to write

    def __post_init__(self):
        if self.raw.shape != (FIT_BEST_NCOLS,):
            raise ValueError(f"expected length-22, got {self.raw.shape}")


# ----------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------


def read_extra_info(path: str | Path,
                    *, mmap: bool = True) -> np.ndarray:
    """Return ``ExtraInfo.bin`` as ``(nSpots, 16)`` float64 array.

    Defaults to memory-mapped read so even multi-GB files cost nothing.
    """
    p = Path(path)
    if mmap:
        arr = np.memmap(p, dtype=np.float64, mode="r")
    else:
        arr = np.fromfile(p, dtype=np.float64)
    if arr.size % EXTRA_INFO_NCOLS != 0:
        raise ValueError(
            f"{p}: size {arr.size * 8} bytes is not a multiple of "
            f"{EXTRA_INFO_NCOLS} doubles per row"
        )
    return arr.reshape(-1, EXTRA_INFO_NCOLS)


def read_orient_pos_fit(path: str | Path,
                        n_grains: Optional[int] = None) -> np.ndarray:
    """Return ``OrientPosFit.bin`` as ``(n_grains, 27)`` float64 array.

    If ``n_grains`` is None, infer from file size. Tolerates pwrite-past-EOF
    truncation: missing tail rows are zero-padded.
    """
    p = Path(path)
    expected_row = ORIENT_POS_FIT_NCOLS * 8
    size = p.stat().st_size
    if n_grains is None:
        if size % expected_row != 0:
            raise ValueError(
                f"{p}: {size} bytes not a multiple of {expected_row}; "
                f"pass n_grains= explicitly"
            )
        n_grains = size // expected_row
    target = n_grains * ORIENT_POS_FIT_NCOLS
    arr = np.fromfile(p, dtype=np.float64)
    if arr.size > target:
        arr = arr[:target]
    if arr.size < target:
        padded = np.zeros(target, dtype=np.float64)
        padded[: arr.size] = arr
        arr = padded
    return arr.reshape(n_grains, ORIENT_POS_FIT_NCOLS)


def read_fit_best(path: str | Path,
                  n_grains: int,
                  max_nhkls: int = MAX_NHKLS_DEFAULT) -> np.ndarray:
    """Return ``FitBest.bin`` as ``(n_grains, max_nhkls, 22)`` float64 array.

    The C code uses pwrite at fixed offsets with a stride of
    ``max_nhkls * 22 doubles`` per grain. pwrite past EOF leaves the file's
    on-disk length at ``offset + payload``; reading from sparse holes yields
    zeros, but a partially-populated tail will simply be missing bytes.
    This reader pads the missing tail with zeros so callers always get a
    fixed ``(n_grains, max_nhkls, 22)`` view.
    """
    p = Path(path)
    full = n_grains * max_nhkls * FIT_BEST_NCOLS
    arr = np.fromfile(p, dtype=np.float64)
    if arr.size > full:
        arr = arr[:full]
    if arr.size < full:
        out = np.zeros(full, dtype=np.float64)
        out[: arr.size] = arr
        arr = out
    return arr.reshape(n_grains, max_nhkls, FIT_BEST_NCOLS)


def read_key(path: str | Path,
             n_grains: Optional[int] = None) -> np.ndarray:
    """Return ``Key.bin`` as ``(n_grains, 2)`` int32 array.

    Like the FitBest reader, this pads with zeros if the file is shorter
    than ``n_grains * 2`` ints (pwrite-past-EOF can leave the file
    truncated when high-numbered rows are unwritten).
    """
    p = Path(path)
    arr = np.fromfile(p, dtype=np.int32)
    if n_grains is None:
        if arr.size % 2 != 0:
            raise ValueError(f"{p}: size {arr.size} ints not a multiple of 2")
        n_grains = arr.size // 2
    target = 2 * n_grains
    if arr.size > target:
        arr = arr[:target]
    if arr.size < target:
        padded = np.zeros(target, dtype=np.int32)
        padded[: arr.size] = arr
        arr = padded
    return arr.reshape(n_grains, 2)


# ----------------------------------------------------------------------------
# Writers (pwrite-style; concurrent-safe per row)
# ----------------------------------------------------------------------------


def _pwrite(path: str | Path, offset: int, payload: bytes) -> None:
    """Open O_CREAT|O_WRONLY, seek+write, close. Mirrors C ``pwrite``.

    Intentionally not buffered: each writer call is one row, and we want
    multiple workers (or repeated calls in the same worker) to stay coherent.
    """
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.lseek(fd, offset, os.SEEK_SET)
        # os.write may short-write on huge buffers; loop just in case.
        view = memoryview(payload)
        n = 0
        while n < len(view):
            n += os.write(fd, view[n:])
    finally:
        os.close(fd)


def write_orient_pos_fit_row(path: str | Path,
                             row_nr: int,
                             grain: GrainResult) -> None:
    """Write a 27-double grain row at offset ``row_nr * 216`` bytes."""
    payload = np.ascontiguousarray(grain.to_row(), dtype=np.float64).tobytes()
    _pwrite(path, row_nr * ORIENT_POS_FIT_NCOLS * 8, payload)


def write_fit_best_row(path: str | Path,
                       row_nr: int,
                       per_spot: np.ndarray,
                       *, max_nhkls: int = MAX_NHKLS_DEFAULT) -> None:
    """Write the matched-spot block for a grain.

    ``per_spot`` is shape ``(nSpotsComp, 22)``. The C code writes only those
    rows; trailing slots in the ``max_nhkls`` stride stay zero-filled by
    the OS.
    """
    if per_spot.ndim != 2 or per_spot.shape[1] != FIT_BEST_NCOLS:
        raise ValueError(f"per_spot must be (nSpots, 22), got {per_spot.shape}")
    if per_spot.shape[0] > max_nhkls:
        raise ValueError(
            f"nSpotsComp={per_spot.shape[0]} exceeds MaxNHKLS={max_nhkls}"
        )
    offset = row_nr * max_nhkls * FIT_BEST_NCOLS * 8
    payload = np.ascontiguousarray(per_spot, dtype=np.float64).tobytes()
    _pwrite(path, offset, payload)


def write_key_row(path: str | Path,
                  row_nr: int,
                  spot_id: int,
                  n_spots_comp: int) -> None:
    """Write the ``(SpotID, nSpotsComp)`` int32 pair at ``row_nr * 8`` bytes."""
    payload = np.array([spot_id, n_spots_comp], dtype=np.int32).tobytes()
    _pwrite(path, row_nr * 2 * 4, payload)


def write_process_key_row(path: str | Path,
                          row_nr: int,
                          spot_ids: np.ndarray,
                          *, max_nhkls: int = MAX_NHKLS_DEFAULT) -> None:
    """Write spot-ID list for a grain into ProcessKey.bin.

    Stride is ``max_nhkls`` int32s per grain.
    """
    arr = np.ascontiguousarray(spot_ids, dtype=np.int32)
    if arr.size > max_nhkls:
        raise ValueError(f"len(spot_ids)={arr.size} > MaxNHKLS={max_nhkls}")
    # SpotIDs are 1-based rows of ExtraInfo.bin. A 0 here is padding that
    # escaped the matcher, and the consumer cannot defend itself: C
    # ProcessGrains computes rowSpotID = SpotID - 1 and dereferences
    # InputMatrix[-1] (ProcessGrains.c:966), which segfaults with no
    # diagnostic. Refuse to write one rather than hand a consumer an index
    # it will turn negative.
    if arr.size and int(arr.min()) < 1:
        bad = np.flatnonzero(arr < 1)
        raise ValueError(
            f"ProcessKey row {row_nr}: {bad.size} non-positive SpotID(s) at "
            f"slots {bad[:8].tolist()} (values {arr[bad[:8]].tolist()}). "
            "SpotIDs are 1-based; 0 means an unfilled observation slot was "
            "counted as a match."
        )
    offset = row_nr * max_nhkls * 4
    _pwrite(path, offset, arr.tobytes())
