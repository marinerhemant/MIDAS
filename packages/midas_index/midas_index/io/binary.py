"""Binary file readers (mmap'd, native endian).

Files (paths relative to ``cwd = dirname(OutputFolder)`` per C convention,
see IndexerOMP.c:2231-2236):

  Spots.bin     [n_spots, 9] float64                observed spots in lab frame
  Data.bin      flat int32                          spot rows per (ring,eta,omega) bin
  nData.bin     flat int32 (interleaved 2 per bin)  per-bin (count, data_offset)

Mirrors `ReadSpots` (IndexerOMP.c:2118), `ReadBins` (line 2085).
BigDetector is deprecated; no `read_big_det` is provided.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NATIVE_ORDER = {"=", "|", "<"} if np.little_endian else {"=", "|", ">"}


def _assert_native(arr: np.ndarray, fname: str) -> None:
    """Fail loud if a file appears to be in non-native byte order."""
    bo = arr.dtype.byteorder
    if bo not in _NATIVE_ORDER:
        raise ValueError(
            f"{fname}: dtype byteorder {bo!r} is not native; "
            "the indexer assumes native-endian binaries."
        )


def read_spots(cwd: str | Path) -> tuple[int, np.ndarray]:
    """Read Spots.bin into a [n_spots, 9-or-10] float64 array via mmap.

    Auto-detects FF (9 cols) vs PF / scanning (10 cols — FF nine plus
    ``ScanNr`` at col 9, matching the layout emitted by
    ``SaveBinDataScanning.c:394-409`` and consumed by the scan-aware
    branch in ``compute.matching.compare_spots``).

    Returns (n_spots, ObsSpotsLab).
    """
    cwd = Path(cwd)
    path = cwd / "Spots.bin"
    if not path.exists():
        raise FileNotFoundError(f"Spots.bin not found at {path}")
    arr = np.memmap(path, dtype=np.float64, mode="r")
    _assert_native(arr, "Spots.bin")
    n_cols = _resolve_spots_ncols(cwd, arr)
    n_spots = arr.size // n_cols
    return n_spots, arr.reshape(n_spots, n_cols)


def _resolve_spots_ncols(cwd: Path, arr: np.ndarray) -> int:
    """Decide whether Spots.bin is 9-col (legacy FF) or 10-col (unified).

    The byte count alone is ambiguous whenever ``n_spots`` makes the total
    double-count divisible by *both* 9 and 10 (e.g. 855 spots → 8550 doubles
    = 855·10 = 950·9). The unified binner (``midas_transforms.bin_data``)
    always writes the 10-col layout (col 9 = ScanNr, 0 for FF) together with
    int64 Data/nData and a ``positions.csv`` sidecar; the legacy C ``SaveBinData``
    wrote 9 cols with int32 bins and no positions.csv. Resolve the ambiguity
    in favour of the unified 10-col layout when its evidence is present —
    otherwise the FF read path mis-routes to the int32 bin reader and
    misinterprets every int64 Data.bin entry (→ zero matches).

    Resolution order:
      1. Unambiguous by divisibility → use it.
      2. Ambiguous → 10-col iff (positions.csv exists) OR (the candidate
         ScanNr column is integer-valued and non-negative, as the unified
         binner guarantees). Else fall back to 9-col.
    """
    div9 = arr.size % 9 == 0
    div10 = arr.size % 10 == 0
    if div10 and not div9:
        return 10
    if div9 and not div10:
        return 9
    if not div9 and not div10:
        raise ValueError(
            f"Spots.bin size {arr.size * 8} bytes is not a multiple of "
            "9 (FF) or 10 (PF/scanning) doubles."
        )
    # Ambiguous: divisible by both 9 and 10.
    if (Path(cwd) / "positions.csv").exists():
        return 10
    # Sniff the candidate ScanNr column (col 9 of the 10-col view).
    scan_col = np.asarray(arr.reshape(-1, 10)[:, 9])
    if np.all(scan_col >= 0) and np.all(scan_col == np.floor(scan_col)):
        return 10
    return 9


RING_SLOTS_FILENAME = "RingSlots.csv"


def read_ring_slots(cwd: str | Path) -> list[int] | None:
    """Ring numbers in slot order from ``cwd/RingSlots.csv``; None if absent.

    Format (written by midas_transforms ``bin_data`` / ``bin_data_scanning``
    next to ``nData.bin``): header ``RingNr Slot``, then one ``<ring> <slot>``
    pair per line; slots are exactly ``0..N-1``. Malformed -> ``ValueError``.
    """
    path = Path(cwd) / RING_SLOTS_FILENAME
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith("RingNr"):
        raise ValueError(f"{path}: first line must be the header 'RingNr Slot'")
    by_slot: dict[int, int] = {}
    rings: set[int] = set()
    for i, ln in enumerate(lines[1:], start=2):
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 2:
            raise ValueError(f"{path} line {i}: expected 'RingNr Slot'")
        r, sl = int(parts[0]), int(parts[1])
        if not (1 <= r < 500) or r in rings or sl in by_slot:
            raise ValueError(f"{path} line {i}: bad or duplicate ring {r} / slot {sl}")
        rings.add(r)
        by_slot[sl] = r
    if not by_slot or sorted(by_slot) != list(range(len(by_slot))):
        raise ValueError(f"{path}: slots must be exactly 0..N-1")
    return [by_slot[k] for k in range(len(by_slot))]


def _ring_slots_to_legacy(
    ndata: np.ndarray,
    cwd: str | Path,
    *,
    itemsize: int,
    n_eta_bins: int | None,
    n_ome_bins: int | None,
    highest_ring: int | None,
) -> np.ndarray:
    """Validate nData.bin's size against its layout; return it LEGACY-shaped.

    With ``RingSlots.csv`` present the file's ring axis is compact (one slab
    per slot); it must be exactly ``n_slots * n_eta * n_ome * 2`` entries and
    is expanded in RAM to the legacy ``slot = RingNr - 1`` layout the Python
    matchers address (empty slabs for rings with no slot; offsets are the
    exclusive cumsum of counts, exactly what a legacy writer produces). So
    only the on-disk table shrinks on this path, not the in-RAM one.

    Without the sidecar the file must hold whole ring slabs and at least
    ``highest_ring`` of them. The size checks run only when the bin geometry
    is known (``n_eta_bins`` and ``n_ome_bins`` given).
    """
    slots = read_ring_slots(cwd)
    geom = n_eta_bins is not None and n_ome_bins is not None
    slab = (int(n_eta_bins) * int(n_ome_bins) * 2) if geom else None
    nbytes = ndata.size * itemsize
    if slots is None:
        if geom and highest_ring is not None:
            need = int(highest_ring) * slab
            if slab == 0 or ndata.size % slab != 0 or ndata.size < need:
                raise ValueError(
                    f"nData.bin is {nbytes} bytes, not the legacy layout for "
                    f"HighestRingNo={highest_ring} x {n_eta_bins} eta x "
                    f"{n_ome_bins} ome bins (>= {need * itemsize} bytes, whole "
                    "ring slabs). No RingSlots.csv next to it: if the table "
                    "was written with compact ring slots its sidecar is "
                    "missing. Re-run binning."
                )
        return ndata
    n_slots = len(slots)
    if geom:
        if ndata.size != n_slots * slab:
            raise ValueError(
                f"nData.bin is {nbytes} bytes but RingSlots.csv lists {n_slots} "
                f"ring slots x {n_eta_bins} eta x {n_ome_bins} ome bins = "
                f"{n_slots * slab * itemsize} bytes. The table and its sidecar "
                "do not belong together. Re-run binning."
            )
    elif ndata.size % (2 * n_slots) != 0:
        raise ValueError(
            f"nData.bin ({nbytes} bytes) is not {n_slots} whole ring slots"
        )
    per_slot = ndata.size // (2 * n_slots)                 # bins per slab
    n_legacy = max(max(slots), int(highest_ring or 0))
    counts = np.zeros((n_legacy, per_slot), dtype=np.int64)
    cmp_counts = np.asarray(ndata).reshape(n_slots, per_slot, 2)[:, :, 0]
    for sl, r in enumerate(slots):
        counts[r - 1] = cmp_counts[sl]
    counts = counts.ravel()
    offsets = np.zeros_like(counts)
    offsets[1:] = np.cumsum(counts[:-1])
    out = np.empty(2 * counts.size, dtype=ndata.dtype)
    out[0::2] = counts
    out[1::2] = offsets
    return out


def read_bins(
    cwd: str | Path,
    *,
    n_eta_bins: int | None = None,
    n_ome_bins: int | None = None,
    highest_ring: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read Data.bin + nData.bin as int32 mmaps.

    A compact (``RingSlots.csv``) table is size-checked and expanded to the
    legacy layout in RAM; see :func:`_ring_slots_to_legacy`.

    Returns (data, ndata) where:
      data[k]       = spot row stored in flat layout
      ndata[2*pos]   = nspots in bin `pos`
      ndata[2*pos+1] = data offset for bin `pos`

    The bin index is `pos = ring * (n_eta_bins * n_ome_bins) + iEta * n_ome_bins + iOme`,
    computed in `compute.binning`.
    """
    cwd = Path(cwd)
    data_path = cwd / "Data.bin"
    ndata_path = cwd / "nData.bin"
    if not data_path.exists():
        raise FileNotFoundError(f"Data.bin not found at {data_path}")
    if not ndata_path.exists():
        raise FileNotFoundError(f"nData.bin not found at {ndata_path}")
    data = np.memmap(data_path, dtype=np.int32, mode="r")
    ndata = np.memmap(ndata_path, dtype=np.int32, mode="r")
    _assert_native(data, "Data.bin")
    _assert_native(ndata, "nData.bin")
    if ndata.size % 2 != 0:
        raise ValueError(
            f"nData.bin size {ndata.size * 4} bytes is not a multiple of 2 int32s"
        )
    ndata = _ring_slots_to_legacy(
        ndata, cwd, itemsize=4, n_eta_bins=n_eta_bins,
        n_ome_bins=n_ome_bins, highest_ring=highest_ring)
    return data, ndata


def read_bins_scanning(
    cwd: str | Path,
    *,
    n_eta_bins: int | None = None,
    n_ome_bins: int | None = None,
    highest_ring: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read scanning-mode Data.bin + nData.bin (int64) and project to FF layout.

    A compact (``RingSlots.csv``) table is size-checked and expanded to the
    legacy layout in RAM; see :func:`_ring_slots_to_legacy`.

    ``SaveBinDataScanning.c:672-700`` writes both files as ``size_t``
    (int64 on x86_64):

      - ``nData.bin``: int64 pairs ``(count, offset_in_spot_units)`` per
        bin, 16 bytes per bin.
      - ``Data.bin``:  int64 pairs ``(spot_id, scan_nr)`` per spot,
        16 bytes per spot.

    The Python scan filter looks up the per-spot ``scan_nr`` via
    ``obs[spot_id, 9]`` (the PF Spots.bin's ScanNr column), so we only
    need ``spot_id`` from Data.bin for candidate gather. The result is
    returned in the SAME int32 (data, ndata) layout as :func:`read_bins`
    so the rest of the indexer pipeline is byte-identical:

      - data_out = spot_ids only (every 2nd int64 from Data.bin) as int32.
      - ndata_out = interleaved (count, offset) as int32, where offset is
        already in "spot units" matching the FF Data.bin layout.

    Returns
    -------
    (data, ndata) : tuple of np.ndarray (int32 in-RAM copies, not mmaps)

    Notes
    -----
    Unlike :func:`read_bins`, this returns in-RAM int32 copies (not
    mmaps) — the on-disk file is int64 and the dense candidate gather
    indexes the array directly, so we must materialise to int32 once.
    For the 5-grain × 15-scan fixture the copy is ~5 MB Data + ~520 MB
    nData, well within chiltepin / build host RAM.
    """
    cwd = Path(cwd)
    data_path = cwd / "Data.bin"
    ndata_path = cwd / "nData.bin"
    if not data_path.exists():
        raise FileNotFoundError(f"Data.bin not found at {data_path}")
    if not ndata_path.exists():
        raise FileNotFoundError(f"nData.bin not found at {ndata_path}")
    data64 = np.memmap(data_path, dtype=np.int64, mode="r")
    ndata64 = np.memmap(ndata_path, dtype=np.int64, mode="r")
    _assert_native(data64, "Data.bin (scanning)")
    _assert_native(ndata64, "nData.bin (scanning)")
    if data64.size % 2 != 0:
        raise ValueError(
            f"Data.bin (scanning) size {data64.size * 8} bytes is not a "
            "multiple of 2 int64 (expected (spot_id, scan_nr) pairs)."
        )
    if ndata64.size % 2 != 0:
        raise ValueError(
            f"nData.bin (scanning) size {ndata64.size * 8} bytes is not a "
            "multiple of 2 int64 (expected (count, offset) pairs)."
        )
    spot_ids = data64.reshape(-1, 2)[:, 0].astype(np.int32, copy=True)
    ndata64 = _ring_slots_to_legacy(
        ndata64, cwd, itemsize=8, n_eta_bins=n_eta_bins,
        n_ome_bins=n_ome_bins, highest_ring=highest_ring)
    ndata_i32 = ndata64.astype(np.int32, copy=True)
    return spot_ids, ndata_i32
