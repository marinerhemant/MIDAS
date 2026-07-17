"""CSV readers / writers for the FF-HEDM intermediate stages.

The file formats here are the contract with the C binaries — every column,
every header, every separator must match.

Header / column references:
- ``Result_*.csv``: ``MergeOverlappingPeaksAllZarr.c:357`` after qsort.
- ``Radius_*.csv``: ``CalcRadiusAllZarr.c:412``.
- ``InputAll.csv``: ``FitSetupParamsAllZarr.c`` (8 cols).
- ``InputAllExtraInfoFittingAll.csv``: same source, 18 cols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np


# --- Result_*.csv (merge output) ------------------------------------------

# NB: ``ReturnCode`` (N2) is APPENDED to the legacy 17-col C layout —
# append-at-end only, all in-tree consumers are positional. -1 marks
# "unknown" when a legacy 17-col file is read back.
RESULT_HEADER_LEGACY = (
    "SpotID IntegratedIntensity Omega YCen ZCen IMax MinOme MaxOme "
    "SigmaR SigmaEta NrPx NrPxTot Radius Eta RawSumIntensity maskTouched FitRMSE"
)
RESULT_HEADER = RESULT_HEADER_LEGACY + " ReturnCode"
RESULT_NCOLS_LEGACY = 17
RESULT_NCOLS = 18


def read_result_csv(path: Union[str, Path]) -> np.ndarray:
    """Read Result_*.csv into a (N, 18) float64 array.

    Legacy 17-col files (written before ReturnCode propagation) are padded
    with ReturnCode = -1 ("unknown" — NOT 0, which means "fit OK")."""
    arr = np.loadtxt(path, skiprows=1, dtype=np.float64)
    arr = np.atleast_2d(arr) if arr.size else arr.reshape(0, RESULT_NCOLS)
    if arr.size and arr.shape[1] == RESULT_NCOLS_LEGACY:
        pad = np.full((arr.shape[0], 1), -1.0, dtype=np.float64)
        arr = np.concatenate([arr, pad], axis=1)
    if arr.size and arr.shape[1] != RESULT_NCOLS:
        raise ValueError(
            f"Result CSV has {arr.shape[1]} cols, expected 17 (legacy) or 18"
        )
    return arr.reshape(-1, RESULT_NCOLS)


def write_result_csv(path: Union[str, Path], data: np.ndarray) -> None:
    """Write a (N, 18) array (or legacy (N, 17)) to Result_*.csv; the
    header always matches the column count."""
    if data.shape[1] == RESULT_NCOLS:
        header = RESULT_HEADER
    elif data.shape[1] == RESULT_NCOLS_LEGACY:
        header = RESULT_HEADER_LEGACY
    else:
        raise ValueError(f"expected 17 or 18 columns, got {data.shape[1]}")
    fmt = " ".join(["%.6f"] * data.shape[1])
    with open(path, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, data, fmt=fmt)


# --- Radius_*.csv (calc_radius output) ------------------------------------

# NB: cols 24/25 (``OrigSpotID ReturnCode``) are APPENDED to the legacy
# 24-col C layout (E3/N2). calc_radius renumbers SpotIDs 1..N (and
# duplicates spots matching two rings), so col 0 is NOT the merge-space
# ID — ``OrigSpotID`` is (== Result_StartNr SpotID). -1 = unknown
# (legacy input).
RADIUS_HEADER_LEGACY = (
    "SpotID IntegratedIntensity Omega YCen ZCen IMax MinOme MaxOme "
    "Radius Theta Eta DeltaOmega NImgs RingNr GrainVolume GrainRadius "
    "PowderIntensity SigmaR SigmaEta NrPx NrPxTot RawSumIntensity "
    "maskTouched FitRMSE"
)
RADIUS_HEADER = RADIUS_HEADER_LEGACY + " OrigSpotID ReturnCode"
RADIUS_NCOLS_LEGACY = 24
RADIUS_NCOLS = 26


def read_radius_csv(path: Union[str, Path]) -> np.ndarray:
    """Read Radius_*.csv into a (N, 26) float64 array. Legacy 24-col files
    are padded with OrigSpotID = ReturnCode = -1 ("unknown")."""
    arr = np.loadtxt(path, skiprows=1, dtype=np.float64)
    arr = np.atleast_2d(arr) if arr.size else arr.reshape(0, RADIUS_NCOLS)
    if arr.size and arr.shape[1] == RADIUS_NCOLS_LEGACY:
        pad = np.full((arr.shape[0], 2), -1.0, dtype=np.float64)
        arr = np.concatenate([arr, pad], axis=1)
    if arr.size and arr.shape[1] != RADIUS_NCOLS:
        raise ValueError(
            f"Radius CSV has {arr.shape[1]} cols, expected 24 (legacy) or 26"
        )
    return arr.reshape(-1, RADIUS_NCOLS)


def write_radius_csv(path: Union[str, Path], data: np.ndarray) -> None:
    if data.shape[1] == RADIUS_NCOLS:
        header = RADIUS_HEADER
    elif data.shape[1] == RADIUS_NCOLS_LEGACY:
        header = RADIUS_HEADER_LEGACY
    else:
        raise ValueError(f"expected 24 or 26 columns, got {data.shape[1]}")
    fmt = " ".join(["%.6f"] * data.shape[1])
    with open(path, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, data, fmt=fmt)


# --- InputAll.csv ---------------------------------------------------------

INPUTALL_HEADER = "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta"
INPUTALL_NCOLS = 8
INPUTALL_NCOLS_WITH_DETID = 9


def read_inputall_csv(path: Union[str, Path]) -> np.ndarray:
    """Read InputAll.csv as a (N, 8) array.

    Tolerates the multi-detector pipeline's 9-col format: if the file has 9
    columns the trailing column (DetID) is dropped here. Use
    ``read_inputall_csv_with_detid`` to retrieve DetID alongside the spots.
    """
    arr = np.loadtxt(path, skiprows=1, dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1] if arr.ndim > 0 else 1) if arr.size else arr.reshape(0, INPUTALL_NCOLS)
    if arr.size and arr.shape[1] == INPUTALL_NCOLS_WITH_DETID:
        return arr[:, :INPUTALL_NCOLS]
    if arr.size and arr.shape[1] != INPUTALL_NCOLS:
        raise ValueError(f"InputAll.csv has {arr.shape[1]} cols, expected 8 or 9")
    return arr.reshape(-1, INPUTALL_NCOLS)


def read_inputall_csv_with_detid(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Read 8-col spot rows + an int32 DetID array (length N).

    For an 8-col input the DetID array is filled with 1 (single-detector
    fallback).
    """
    arr = np.loadtxt(path, skiprows=1, dtype=np.float64)
    arr = arr.reshape(-1, arr.shape[-1] if arr.ndim > 0 else 1) if arr.size else arr.reshape(0, INPUTALL_NCOLS)
    if arr.size and arr.shape[1] == INPUTALL_NCOLS_WITH_DETID:
        return arr[:, :INPUTALL_NCOLS], arr[:, INPUTALL_NCOLS].astype(np.int32)
    if arr.size and arr.shape[1] != INPUTALL_NCOLS:
        raise ValueError(f"InputAll.csv has {arr.shape[1]} cols, expected 8 or 9")
    spots = arr.reshape(-1, INPUTALL_NCOLS)
    return spots, np.ones(spots.shape[0], dtype=np.int32)


def write_inputall_csv(path: Union[str, Path], data: np.ndarray) -> None:
    if data.shape[1] != INPUTALL_NCOLS:
        raise ValueError(f"expected {INPUTALL_NCOLS} columns, got {data.shape[1]}")
    fmt = " ".join(["%.6f"] * INPUTALL_NCOLS)
    with open(path, "w") as f:
        f.write(INPUTALL_HEADER + "\n")
        np.savetxt(f, data, fmt=fmt)


# --- InputAllExtraInfoFittingAll.csv --------------------------------------

# NB: names match the DATA actually written by fit_setup (col 13 is the
# det-corrected omega, NOT an intensity; the old header mislabeled cols
# 13-17 as "IntegratedIntensity RawSumIntensity FitRMSE maskTouched
# FitErrCode", one slot to the left of reality, which mis-led any
# name-based reader. Cols 11/12 hold RAW DETECTOR PIXELS (== peaksearch
# YCen/ZCen), not lab-frame um — the old "YOrigNoWedge/ZOrigNoWedge"
# names suggested otherwise. All pipeline consumers are positional.
# The multi-detector pipeline appends a 19th "DetID" column
# (cross_det_merge); readers here tolerate and strip it.)
INPUTALL_EXTRA_HEADER = (
    "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta "
    "OmegaIni YOrigDetCor ZOrigDetCor YRawPx ZRawPx "
    "OmegaDetCor IntegratedIntensity RawSumIntensity maskTouched FitRMSE"
)
# Appended columns (N2 + E3, ONE append-only schema change):
#   col 18 OrigSpotID — merge-space SpotID (== Result_StartNr col 0).
#     fit_setup re-sorts AND renumbers (col 4), and calc_radius renumbers
#     before that — a SpotID join between Result_* and InputAll* spaces
#     silently pairs random spots without this bridge.
#   col 19 ReturnCode — peakfit per-peak returnCode (sticky-first-nonzero
#     over merged constituents; 0 = all constituents fit OK, -1 = unknown/
#     legacy input).
# The multi-detector pipeline may append one more trailing DetID column
# (cross_det_merge). Binary strides (Spots.bin / ExtraInfo.bin) are
# UNTOUCHED — the appended cols never reach the binaries.
INPUTALL_EXTRA_APPENDED_HEADER = INPUTALL_EXTRA_HEADER + " OrigSpotID ReturnCode"
INPUTALL_EXTRA_NCOLS = 18
INPUTALL_EXTRA_NCOLS_WITH_DETID = 19
INPUTALL_EXTRA_NCOLS_APPENDED = 20
INPUTALL_EXTRA_NCOLS_APPENDED_WITH_DETID = 21


def _load_inputall_extra_any(path: Union[str, Path]) -> np.ndarray:
    arr = np.loadtxt(path, skiprows=1, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, INPUTALL_EXTRA_NCOLS_APPENDED)
    arr = np.atleast_2d(arr)
    if arr.shape[1] not in (INPUTALL_EXTRA_NCOLS,
                            INPUTALL_EXTRA_NCOLS_WITH_DETID,
                            INPUTALL_EXTRA_NCOLS_APPENDED,
                            INPUTALL_EXTRA_NCOLS_APPENDED_WITH_DETID):
        raise ValueError(
            f"InputAllExtra has {arr.shape[1]} cols, expected 18/19 "
            "(legacy [+DetID]) or 20/21 (appended [+DetID])"
        )
    return arr


def read_inputall_extra_csv(path: Union[str, Path]) -> np.ndarray:
    """Read InputAllExtraInfoFittingAll.csv as the (N, 18) BASE columns.

    Appended columns (OrigSpotID/ReturnCode, cols 18/19) and a trailing
    multi-detector DetID are stripped, so every positional consumer —
    including the ExtraInfo.bin builder (fixed 16-double stride) — sees
    the legacy layout. Use :func:`read_inputall_extra_csv_appended` to
    also retrieve the appended columns."""
    arr = _load_inputall_extra_any(path)
    return arr[:, :INPUTALL_EXTRA_NCOLS].reshape(-1, INPUTALL_EXTRA_NCOLS)


def read_inputall_extra_csv_appended(
    path: Union[str, Path],
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Read InputAllExtra returning ``(base18, orig_spot_id, return_code)``.

    For legacy 18/19-col files the appended arrays are -1 ("unknown")."""
    arr = _load_inputall_extra_any(path)
    base = arr[:, :INPUTALL_EXTRA_NCOLS].reshape(-1, INPUTALL_EXTRA_NCOLS)
    if arr.shape[1] >= INPUTALL_EXTRA_NCOLS_APPENDED:
        return base, arr[:, 18].copy(), arr[:, 19].copy()
    unknown = np.full(base.shape[0], -1.0, dtype=np.float64)
    return base, unknown.copy(), unknown.copy()


def write_inputall_extra_csv(path: Union[str, Path], data: np.ndarray) -> None:
    """Write InputAllExtra; accepts the (N, 20) appended layout or the
    legacy (N, 18); the header always matches the column count."""
    if data.shape[1] == INPUTALL_EXTRA_NCOLS_APPENDED:
        header = INPUTALL_EXTRA_APPENDED_HEADER
    elif data.shape[1] == INPUTALL_EXTRA_NCOLS:
        header = INPUTALL_EXTRA_HEADER
    else:
        raise ValueError(f"expected 18 or 20 columns, got {data.shape[1]}")
    fmt = " ".join(["%.6f"] * data.shape[1])
    with open(path, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, data, fmt=fmt)


# --- SpotsToIndex.csv -----------------------------------------------------


def read_spots_to_index(path: Union[str, Path]) -> np.ndarray:
    """Single-column CSV of spot IDs."""
    return np.loadtxt(path, dtype=np.int64).reshape(-1)


def write_spots_to_index(path: Union[str, Path], spot_ids: Iterable[int]) -> None:
    with open(path, "w") as f:
        for sid in spot_ids:
            f.write(f"{int(sid)}\n")


# --- hkls.csv -------------------------------------------------------------


def read_hkls_csv(path: Union[str, Path]) -> np.ndarray:
    """Load hkls.csv as a (N, ncols) float64 array.

    The MIDAS hkls.csv schema is:
        col 0..2  : h, k, l
        col 3..5  : ds, theta, multiplicity (or similar — packages vary)
        col 4     : ring radius (px) in the modern schema
        ...
    We return the raw float matrix; consumers index by column.
    """
    return np.loadtxt(path, skiprows=1, dtype=np.float64)
