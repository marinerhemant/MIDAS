"""voxel_binner: drop-in replacement for the C ``SaveBinDataScanning`` binary.

The C source is ``FF_HEDM/src/SaveBinDataScanning.c`` (744 LoC). This Python
port:

1. Reads ``N`` per-scan ``InputAllExtraInfoFittingAll{n}.csv`` files from a
   directory (one per scan position).
2. Filters spots with ``GrainRadius <= 0.0001`` (matches the C semantic at
   ``SaveBinDataScanning.c:283``).
3. Tags each retained spot with its scan number.
4. Sorts globally by ``(RingNumber, Omega, Eta)`` (matches the C
   ``cmpfunc`` at lines 58-78).
5. Renumbers SpotID 1..N in sorted order, recording the original
   ``(NewID, OrigID, ScanNr)`` triple in ``IDsMergedScanning.csv``.
6. Computes the per-spot ``RadiusDistIdeal = sqrt(YLab² + ZLab²) -
   RingRadii[ring]`` (matches ``CalcDistanceIdealRing``).
7. Writes:

   - ``Spots.bin`` — ``(N, 10)`` float64 (the 9 FF columns + scanNr as col
     9).
   - ``ExtraInfo.bin`` — ``(N, 16)`` float64 (same as FF mode).
   - ``IDsMergedScanning.csv`` — CSV with header ``NewID,OrigID,ScanNr``.
   - ``voxel_scan_pos.bin`` — float64 ``(n_scans,)`` — the 1-D scan-Y
     array (the new Python-only sidecar; the C indexer reads
     ``positions.csv``, which we ALSO emit when ``write_positions_csv``
     is True so the legacy binary continues to work).
   - When ``NoSaveAll == 0``: ``Data.bin`` + ``nData.bin`` as ``size_t``
     pairs per ``(ring, iEta, iOme)`` bin (PF layout, matches the C).

When called with ``scan_positions is None`` the function delegates to the
plain ``bin_data`` entry point — i.e. **the FF regression gate is the
identity of FF behaviour**.

The differentiable + device-portable contract: the radius / binning maths
all routes through the existing ``bin_data.core`` torch primitives, so
this module inherits its differentiability and device portability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from ..io import binary as bio
from ..params import ParamsTest, read_paramstest
from . import core as bin_core
from .core import (
    BinDataResult,
    _bin_assignment,
    _build_ring_radii,
    _compute_radius_dist_ideal,
    _radius_dist_ideal_numpy,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VoxelBinDataResult(BinDataResult):
    """Result of ``bin_data_scanning``.

    Inherits all FF-mode fields from ``BinDataResult`` plus three PF
    extras: the ``scan_nr`` per spot, the 1-D ``scan_positions`` array
    written to ``voxel_scan_pos.bin``, and the ``id_map`` table that
    pairs renumbered SpotIDs with their original (per-scan) IDs.
    """

    scan_nr: Optional[torch.Tensor] = None          # (N,) int64 — col 9 of Spots.bin
    scan_positions: Optional[np.ndarray] = None     # (n_scans,) float64
    id_map: Optional[np.ndarray] = None             # (N, 3) int — (NewID, OrigID, ScanNr)
    n_scans: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_per_scan_csv(
    csv_path: Path,
) -> np.ndarray:
    """Read one ``InputAllExtraInfoFittingAll{n}.csv`` file as a (N, 16) array.

    The file has 18 cols on disk (the C ``sscanf`` format); we drop cols
    14 and 15 (the C ``dummy0``, ``dummy1`` sinks) and keep cols
    [0..13, 16, 17] in that order — matching the in-memory ``InpData``
    layout used by ``SaveBinDataScanning.c:233-244``.
    """
    arr = np.loadtxt(csv_path, skiprows=1, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 16), dtype=np.float64)
    arr = np.atleast_2d(arr)
    if arr.shape[1] == 16:
        # Already in compressed form (e.g. mergeScansScanning output).
        return arr
    if arr.shape[1] in (18, 19, 20, 21):
        # 18 base / +DetID / +OrigSpotID,ReturnCode (N2+E3) / +both — the
        # base cols are positionally identical in every variant; drop the
        # dummies (14, 15) and any appended cols. The 16-double binary
        # stride is never widened.
        keep = np.concatenate([np.arange(14), np.arange(16, 18)])
        return arr[:, keep]
    raise ValueError(
        f"{csv_path}: expected 16, 18..21 cols, got {arr.shape[1]}"
    )


def _sort_index_ring_omega_eta(rows: np.ndarray) -> np.ndarray:
    """Return the permutation that sorts rows by (RingNumber, Omega, Eta).

    Mirrors the C qsort comparator at ``SaveBinDataScanning.c:58-78``:
    primary key = col 5 (RingNumber), secondary = col 2 (Omega),
    tertiary = col 6 (Eta).

    ``numpy.lexsort`` keys are processed from LAST to FIRST, so we pass
    ``(eta, omega, ring_number)``.
    """
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return np.lexsort((rows[:, 6], rows[:, 2], rows[:, 5]))


def _filter_valid(rows: np.ndarray) -> np.ndarray:
    """Drop rows with ``|GrainRadius| <= 0.0001``.

    Mirrors ``SaveBinDataScanning.c:283`` — ``if (fabs(... Values[3]) >
    0.0001) nSpots++``.
    """
    if rows.shape[0] == 0:
        return rows
    mask = np.abs(rows[:, 3]) > 0.0001
    return rows[mask]


# ---------------------------------------------------------------------------
# Vectorised binning (PF variant — Data.bin stores (rowno, scanno) pairs)
# ---------------------------------------------------------------------------


def _bin_to_data_ndata_scanning(
    arrays: list,
    *,
    n_ring_bins: int,
    n_eta_bins: int,
    n_ome_bins: int,
):
    """PF variant of ``_bin_to_data_ndata``: outputs (rowno, scanno) pairs.

    ``arrays`` is a mutable list ``[out_spot_idx, out_ring, out_ieta,
    out_iome, out_scan_nr]`` that is CLEARED on entry: at multi-1e9 pairs
    each of these int64 tensors is tens of GB, and taking sole ownership
    (caller keeps no references) lets the prompt ``del``s below actually
    free them. Peak memory is ~5 pair-length tensors (the argsort) instead
    of ~10 for the naive path. Output is unchanged.

    Layout (per ``SaveBinDataScanning.c:560-705``):
      - Output is grouped by ring, then eta, then ome (matches FF binner).
      - Per spot in a bin, write the pair ``(rowno, scanno)`` as two
        ``size_t`` values, so Data.bin total size is ``2 * sum(counts) *
        sizeof(size_t)``.
      - ``nData.bin`` stores ``(count, offset)`` per bin, where offset is
        the running total of spots (NOT pairs) preceding this bin in the
        ring/eta/ome traversal order.
    """
    out_spot_idx, out_ring, out_ieta, out_iome, out_scan_nr = arrays
    arrays.clear()
    device = out_spot_idx.device

    # Modulo wrap (negative-aware), same as FF binner.
    ieta_mod = (out_ieta % n_eta_bins + n_eta_bins) % n_eta_bins
    del out_ieta
    iome_mod = (out_iome % n_ome_bins + n_ome_bins) % n_ome_bins
    del out_iome

    # iRing in C is ``ringnr - 1``; ring-bin axis is [0, HighestRingNo).
    iring = out_ring - 1
    del out_ring

    # Drop entries whose ring index is out of range. (Defensive; ring_nr is
    # filtered upstream — fast-path the all-true case to avoid 5 full copies.)
    mask = (iring >= 0) & (iring < n_ring_bins)
    if not bool(mask.all()):
        iring = iring[mask]
        ieta_mod = ieta_mod[mask]
        iome_mod = iome_mod[mask]
        out_spot_idx = out_spot_idx[mask]
        out_scan_nr = out_scan_nr[mask]
    del mask

    # Composite bin id (ring outer, eta middle, ome inner) — same order
    # as the FF binner writes.
    bin_id = (iring.long() * n_eta_bins + ieta_mod.long()) * n_ome_bins + iome_mod.long()
    del iring, ieta_mod, iome_mod

    if out_spot_idx.numel() == 0:
        total_bins = n_ring_bins * n_eta_bins * n_ome_bins
        counts = torch.zeros(total_bins, dtype=torch.int64, device=device)
        offsets = torch.zeros(total_bins, dtype=torch.int64, device=device)
        ndata = torch.stack([counts, offsets], dim=1)
        data = torch.zeros((0, 2), dtype=torch.int64, device=device)
        return data, ndata

    # Stable secondary sort by spot_idx for deterministic order within a bin.
    # The composite key is built IN-PLACE on bin_id, and (bin_id, spot_idx)
    # are recovered exactly from the sorted key by divmod — this avoids two
    # extra pair-length tensors vs. the keep-everything formulation.
    max_spot = int(out_spot_idx.max().item()) + 2
    bin_id.mul_(max_spot).add_(out_spot_idx)          # bin_id := composite
    del out_spot_idx
    order = torch.argsort(bin_id, stable=True)
    sorted_comp = bin_id[order]
    del bin_id
    sorted_bin_id = sorted_comp // max_spot
    sorted_spot_idx = sorted_comp - sorted_bin_id * max_spot
    del sorted_comp
    sorted_scan_nr = out_scan_nr[order]
    del out_scan_nr, order

    # Counts per bin.
    total_bins = n_ring_bins * n_eta_bins * n_ome_bins
    counts = torch.zeros(total_bins, dtype=torch.int64, device=device)
    counts.scatter_add_(0, sorted_bin_id, torch.ones_like(sorted_bin_id))

    # Offsets are cumulative spot count (NOT pair count) — the C code
    # increments ``globalCounter += localNDataVal`` (not 2x), matching
    # the way the indexer dereferences ``data[2 * offset + 0/1]``.
    offsets = torch.zeros(total_bins, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts[:-1], dim=0)

    ndata = torch.stack([counts, offsets], dim=1).to(torch.int64)
    data_pairs = torch.stack([sorted_spot_idx.to(torch.int64),
                              sorted_scan_nr.to(torch.int64)], dim=1)
    return data_pairs, ndata


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def bin_data_scanning(
    result_folder: Union[str, Path] = ".",
    *,
    n_scans: int,
    scan_positions: Sequence[float],
    out_dir: Optional[Union[str, Path]] = None,
    paramstest: Optional[ParamsTest] = None,
    paramstest_path: Optional[Union[str, Path]] = None,
    csv_template: str = "InputAllExtraInfoFittingAll{n}.csv",
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    write: bool = True,
    write_positions_csv: bool = True,
    strict_files: bool = False,
) -> VoxelBinDataResult:
    """Drop-in replacement for the C ``SaveBinDataScanning`` binary.

    PF / scanning entry point. Reads ``n_scans`` per-scan
    ``InputAllExtraInfoFittingAll{n}.csv`` files (one per scan position),
    filters by ``GrainRadius``, sorts by ``(RingNumber, Omega, Eta)``,
    renumbers SpotID, and writes the indexer-shaped ``Spots.bin`` (10
    cols), ``ExtraInfo.bin``, ``IDsMergedScanning.csv``, and the new
    Python-side ``voxel_scan_pos.bin`` sidecar.

    Parameters
    ----------
    result_folder
        Directory to read per-scan inputs from / write outputs to.
    n_scans
        Number of per-scan CSVs (``InputAllExtraInfoFittingAll0.csv`` ..
        ``InputAllExtraInfoFittingAll{n-1}.csv``).
    scan_positions
        1-D Y array (in µm) — one position per scan. Length == n_scans.
    out_dir
        Optional output dir override (defaults to ``result_folder``).
    paramstest, paramstest_path
        Optional in-memory params or override path.
    csv_template
        Template for per-scan CSV filenames. ``{n}`` is the scan index.
    device, dtype
        Torch device / dtype.
    write
        If False, skip disk writes and return the in-memory result.
    write_positions_csv
        If True (default), also emit ``positions.csv`` (one Y per line)
        for the legacy C indexer.
    strict_files
        If True, error when any per-scan file is missing. If False, log
        + skip (default, matches the C binary behaviour).

    Returns
    -------
    VoxelBinDataResult
    """
    rf = Path(result_folder)
    out_dir = Path(out_dir) if out_dir is not None else rf

    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)

    if n_scans <= 0:
        raise ValueError(f"n_scans must be > 0, got {n_scans}")
    scan_positions = np.asarray(scan_positions, dtype=np.float64).ravel()
    if scan_positions.shape[0] != n_scans:
        raise ValueError(
            f"scan_positions has {scan_positions.shape[0]} entries but "
            f"n_scans={n_scans}"
        )

    if paramstest is None:
        ppath = paramstest_path if paramstest_path is not None else rf / "paramstest.txt"
        paramstest = read_paramstest(ppath)

    # Validate required parameters (C lines 119-170).
    for name in ("MarginOme", "MarginEta", "EtaBinSize", "OmeBinSize", "StepSizeOrient"):
        if getattr(paramstest, name) <= 0:
            raise ValueError(f"Required parameter '{name}' missing or <= 0")
    if len(paramstest.RingNumbers) == 0:
        raise ValueError("No 'RingNumbers' entries found in paramstest")
    if len(paramstest.RingRadii) != len(paramstest.RingNumbers):
        raise ValueError(
            f"NrOfRings({len(paramstest.RingRadii)}) != "
            f"NoRingNumbers({len(paramstest.RingNumbers)})"
        )

    # Read + concatenate all per-scan CSVs, tagging each row with its scan.
    parts: List[np.ndarray] = []
    scan_tags: List[np.ndarray] = []
    n_found = 0
    for scan_nr in range(n_scans):
        csv_path = rf / csv_template.format(n=scan_nr)
        if not csv_path.exists():
            if strict_files:
                raise FileNotFoundError(f"Per-scan CSV missing: {csv_path}")
            continue
        rows = _read_per_scan_csv(csv_path)
        if rows.shape[0] == 0:
            continue
        rows = _filter_valid(rows)
        if rows.shape[0] == 0:
            continue
        parts.append(rows)
        scan_tags.append(np.full(rows.shape[0], scan_nr, dtype=np.int64))
        n_found += 1

    if n_found == 0:
        raise FileNotFoundError(
            f"No per-scan CSVs found in {rf} matching {csv_template}"
        )

    rows = np.concatenate(parts, axis=0) if parts else np.zeros((0, 16), dtype=np.float64)
    scan_nrs = np.concatenate(scan_tags, axis=0) if scan_tags else np.zeros(0, dtype=np.int64)

    if rows.shape[0] == 0:
        raise ValueError(
            f"No valid spots found across {n_scans} scans (after "
            "GrainRadius filter). Aborting."
        )

    # Sort globally by (RingNumber, Omega, Eta) — matches C qsort.
    order = _sort_index_ring_omega_eta(rows)
    rows = rows[order]
    scan_nrs = scan_nrs[order]
    orig_spot_ids = rows[:, 4].astype(np.int64).copy()

    # Renumber SpotID 1..N (matches C lines 374-378).
    n_spots = rows.shape[0]
    new_ids = np.arange(1, n_spots + 1, dtype=np.float64)
    rows[:, 4] = new_ids

    # Compute RadiusDistIdeal (col 8 of the FF spots layout) and assemble
    # the 10-col Spots.bin: first 8 cols of rows + radius_dist + scanNr.
    yl = rows[:, 0]
    zl = rows[:, 1]
    ring_np = rows[:, 5].astype(np.int64)
    ring_radii_np = np.zeros(500, dtype=np.float64)
    for r, rad in zip(paramstest.RingNumbers, paramstest.RingRadii):
        if 0 <= r < 500:
            ring_radii_np[r] = rad
    # Byte-exact CPU path matches the C `sqrt(y*y + z*z) - RingRadii[ring]`.
    rad_dist_np = _radius_dist_ideal_numpy(yl, zl, ring_np, ring_radii_np)

    spots_np = np.empty((n_spots, 10), dtype=np.float64)
    spots_np[:, :8] = rows[:, :8]
    spots_np[:, 8] = rad_dist_np
    spots_np[:, 9] = scan_nrs.astype(np.float64)

    # ExtraInfo.bin: 16 cols. The in-memory ``rows`` already has 16
    # (cols [0..13, 16, 17] from the original 18-col CSV).
    extra_np = rows[:, :16].astype(np.float64).copy()

    # Tensors for the binning step.
    spots_t = torch.from_numpy(spots_np[:, :8].copy()).to(device=dev, dtype=dt)
    spots_full_t = torch.from_numpy(spots_np).to(device=dev, dtype=dt)
    extra_t = torch.from_numpy(extra_np).to(device=dev, dtype=dt)
    scan_nrs_t = torch.from_numpy(scan_nrs).to(device=dev)

    # ID map: (NewID, OrigID, ScanNr).
    id_map = np.column_stack([
        new_ids.astype(np.int64),
        orig_spot_ids,
        scan_nrs.astype(np.int64),
    ])

    if write:
        bio.write_spots_bin(out_dir / "Spots.bin", spots_np)
        bio.write_extrainfo_bin(out_dir / "ExtraInfo.bin", extra_np)
        bio.write_voxel_scan_pos_bin(out_dir / "voxel_scan_pos.bin", scan_positions)
        # IDsMergedScanning.csv (C lines 474-485).
        with open(out_dir / "IDsMergedScanning.csv", "w") as f:
            f.write("NewID,OrigID,ScanNr\n")
            for row in id_map:
                f.write(f"{int(row[0])},{int(row[1])},{int(row[2])}\n")
        if write_positions_csv:
            with open(out_dir / "positions.csv", "w") as f:
                for y in scan_positions:
                    f.write(f"{y:.6f}\n")

    if paramstest.NoSaveAll == 1:
        return VoxelBinDataResult(
            spots=spots_full_t, extra_info=extra_t,
            scan_nr=scan_nrs_t, scan_positions=scan_positions,
            id_map=id_map, n_scans=n_scans,
            paramstest=paramstest,
        )

    # Now bin spots into (ring, iEta, iOme) bins for the Data.bin index.
    ring_radii_t = _build_ring_radii(paramstest).to(device=dev, dtype=dt)
    n_ring_bins = paramstest.highest_ring_no
    n_eta_bins = math.ceil(360.0 / paramstest.EtaBinSize)
    n_ome_bins = math.ceil(360.0 / paramstest.OmeBinSize)

    # Note: C stores ``rowno = i`` (0-based) and ``scanno`` in Data.bin.
    # Our ``out_spot_idx`` is 0-based row index after the global sort, so
    # it matches ``rowno`` semantics 1:1.
    #
    # PER-RING processing: Data.bin's layout is RING-MAJOR (ring, eta, ome),
    # so packing one ring at a time and concatenating the per-ring data
    # slices is BIT-IDENTICAL to the all-at-once path while dividing the
    # peak pair-array memory by the number of active rings. Each ring is
    # selected by zeroing every other ring's radius (reusing
    # ``_bin_assignment``'s existing radius>0 keep filter). ``ndata``
    # counts accumulate into the global bin table; offsets are the running
    # cumsum over the ring-major traversal, computed once at the end.
    # The 5 pair-length arrays are handed over in a container so the callee
    # takes SOLE ownership (it clears the list) — required for its internal
    # frees to work; at multi-1e9 pairs each array is tens of GB.
    total_bins = n_ring_bins * n_eta_bins * n_ome_bins
    counts_total = torch.zeros(total_bins, dtype=torch.int64, device=dev)
    _data_parts = []
    _active_rings = [r for r in range(ring_radii_t.shape[0])
                     if float(ring_radii_t[r].item()) > 0]
    for _r in _active_rings:
        _rr_one = torch.zeros_like(ring_radii_t)
        _rr_one[_r] = ring_radii_t[_r]
        _pair_arrays = list(_bin_assignment(
            spots_t,
            _rr_one,
            margin_ome=paramstest.MarginOme,
            margin_eta=paramstest.MarginEta,
            eta_bin_size=paramstest.EtaBinSize,
            ome_bin_size=paramstest.OmeBinSize,
            step_size_orient=paramstest.StepSizeOrient,
        ))
        # Look up scan_nr per output triple from spot index.
        _pair_arrays.append(scan_nrs_t[_pair_arrays[0]])
        _data_r, _ndata_r = _bin_to_data_ndata_scanning(
            _pair_arrays,
            n_ring_bins=n_ring_bins, n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        )
        counts_total += _ndata_r[:, 0]
        del _ndata_r
        if _data_r.shape[0]:
            _data_parts.append(_data_r)
        del _data_r

    data_pairs = (torch.cat(_data_parts) if _data_parts
                  else torch.zeros((0, 2), dtype=torch.int64, device=dev))
    _data_parts.clear()
    offsets_total = torch.zeros(total_bins, dtype=torch.int64, device=dev)
    offsets_total[1:] = torch.cumsum(counts_total[:-1], dim=0)
    ndata = torch.stack([counts_total, offsets_total], dim=1)

    if write:
        bio.write_data_ndata_bin_scanning(
            out_dir / "Data.bin", out_dir / "nData.bin",
            data_pairs.detach().cpu().numpy().astype(np.uint64),
            ndata.detach().cpu().numpy().astype(np.uint64),
        )

    return VoxelBinDataResult(
        spots=spots_full_t, extra_info=extra_t,
        data=data_pairs, ndata=ndata,
        n_ring_bins=n_ring_bins, n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        scan_nr=scan_nrs_t, scan_positions=scan_positions,
        id_map=id_map, n_scans=n_scans,
        paramstest=paramstest,
    )


# ---------------------------------------------------------------------------
# Unified dispatch entry point
# ---------------------------------------------------------------------------


def bin_data_unified(
    result_folder: Union[str, Path] = ".",
    *,
    scan_positions: Optional[Sequence[float]] = None,
    n_scans: Optional[int] = None,
    beam_size_um: float = 0.0,
    out_dir: Optional[Union[str, Path]] = None,
    csv_template: str = "InputAllExtraInfoFittingAll{n}.csv",
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    write: bool = True,
    **kwargs,
) -> BinDataResult:
    """Unified entry point: dispatches to ``bin_data`` (FF) or
    ``bin_data_scanning`` (PF) based on ``scan_positions``.

    Parameters
    ----------
    scan_positions
        If None (default), runs FF-mode ``bin_data``. The output is
        bit-identical to today's FF binner.
        If provided, runs PF-mode ``bin_data_scanning``.
    n_scans
        Required when ``scan_positions`` is provided. Must equal
        ``len(scan_positions)``.
    beam_size_um
        PF-only; recorded for downstream stages. Not used in the binning
        math directly (the scan tolerance is consumed by the indexer).
    """
    if scan_positions is None:
        # FF mode — strict identity to today's ``bin_data``.
        from .core import bin_data
        return bin_data(
            result_folder, out_dir=out_dir,
            device=device, dtype=dtype, write=write, **kwargs,
        )

    # PF mode.
    if n_scans is None:
        n_scans = len(scan_positions)
    return bin_data_scanning(
        result_folder, n_scans=n_scans, scan_positions=scan_positions,
        out_dir=out_dir, csv_template=csv_template,
        device=device, dtype=dtype, write=write, **kwargs,
    )
