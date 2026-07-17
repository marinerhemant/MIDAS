"""bin_data: drop-in replacement for ``SaveBinData``.

The C source is `FF_HEDM/src/SaveBinData.c` (341 LoC). The torch port:

1. Reads ``InputAll.csv`` (8 cols) and ``InputAllExtraInfoFittingAll.csv``
   (18 cols) into tensors on ``device``.
2. Computes the per-spot ``RadiusDistIdeal = radius_obs - ring_radii[ring_nr]``.
3. Writes ``Spots.bin`` (Nx9 float64) and ``ExtraInfo.bin`` (Nx16 float64).
4. If ``NoSaveAll == 0``: builds the per-(ring, eta-bin, ome-bin) lookup
   table and writes ``Data.bin`` (int32 ragged) and ``nData.bin`` (count/offset
   pairs, int32).

The (eta, ome) bin assignment per spot uses the C margin formulae
(``SaveBinData.c:265-271``):

    omemargin = MarginOme + 0.5 * StepSizeOrient / |sin(eta_deg)|
    etamargin = rad2deg * atan(MarginEta / RingRadii[ring]) + 0.5 * StepSizeOrient

Then for each spot, all bins in ``[iEtaMin..iEtaMax] × [iOmeMin..iOmeMax]``
mod ``n_eta``, ``n_ome`` receive the spot's index.

The vectorised torch path emits one ``(spot_idx, ring, iEta, iOme)`` tuple
per (spot, eta-bin, ome-bin) triple, sorts by ``(ring, iEta, iOme, spot_idx)``,
then ``unique_consecutive`` to recover counts and offsets per bin. The
``spot_idx`` secondary key makes the output bit-stable across runs and
between CPU and GPU (the C version's order is implicitly the spot
iteration order, which is row order in InputAll.csv == ascending
``spot_idx`` for non-empty bins).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from ..io import binary as bio
from ..io import csv as csv_io
from ..params import ParamsTest, read_paramstest


@dataclass
class BinDataResult:
    """In-memory result of the bin_data stage. Tensors live on ``device``.

    Used by ``Pipeline`` to pass to a downstream consumer (e.g. ``midas-index``)
    without going through disk.

    After the Phase 5 format unification (2026-05), ``spots`` always has 10
    columns (col 9 = ScanNr, 0 for FF runs); ``data``/``ndata`` are still
    int32 in-memory but get expanded to int64-pair (data, scan_nr) /
    (count, offset) layout on disk via ``write_data_ndata_bin_scanning`` so
    the unified ``midas_indexer`` C binary and the Python indexer share one
    byte layout regardless of FF vs PF mode.
    """

    spots: torch.Tensor                        # (N, 10) float64 — col 9 = ScanNr
    extra_info: torch.Tensor                   # (N, 16) float64
    data: Optional[torch.Tensor] = None        # (T,) int32 spot rows (in-mem)
    ndata: Optional[torch.Tensor] = None       # (M, 2) int32 (count, offset) (in-mem)
    n_ring_bins: int = 0
    n_eta_bins: int = 0
    n_ome_bins: int = 0
    paramstest: Optional[ParamsTest] = field(default=None)


# ---------------------------------------------------------------------------
# Pure-tensor kernels
# ---------------------------------------------------------------------------

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


# ---- libm.fma binding (for byte-exact y² + z² matching the C build) -----

def _resolve_libm_fma():
    """Return a vectorised libm ``fma`` if available, else ``None``.

    Used to reproduce clang's default ``-ffp-contract=on`` FMA fusion of
    ``y*y + z*z`` so that ``Spots.bin`` col 8 (RadiusDistIdeal) is byte-exact
    against the C ``SaveBinData`` output. Falls back to plain ``y*y + z*z``
    when libm or its ``fma`` symbol is unavailable (Windows, oddball libcs).
    """
    import ctypes
    import ctypes.util
    name = ctypes.util.find_library("m") or "libm.dylib"
    try:
        libm = ctypes.CDLL(name)
        libm.fma.restype = ctypes.c_double
        libm.fma.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    except (OSError, AttributeError):
        return None
    return np.vectorize(libm.fma, otypes=[np.float64])


_FMA = _resolve_libm_fma()


def _radius_dist_ideal_numpy(
    yl: np.ndarray, zl: np.ndarray, ring_nr: np.ndarray, ring_radii: np.ndarray,
) -> np.ndarray:
    """Compute ``sqrt(y*y + z*z) - RingRadii[ring]`` using FMA when available.

    The byte-exact CPU path. The ``y*y + z*z`` fused expression is what the
    C ``SaveBinData`` code compiles to under clang's default FMA contraction.
    """
    z2 = zl * zl
    if _FMA is not None:
        s = _FMA(yl, yl, z2)
    else:
        s = yl * yl + z2
    return np.sqrt(s) - ring_radii[ring_nr]


def _build_ring_radii(p: ParamsTest, max_n_rings: int = 500) -> torch.Tensor:
    """Return a 1-D tensor ``[max_n_rings]`` of radii indexed by ring number.

    Mirrors ``SaveBinData.c:170-174`` — ``RingRadii[RingNumbers[i]] = RingRadiiUser[i]``
    with everything else zeroed.
    """
    out = np.zeros(max_n_rings, dtype=np.float64)
    for r, rad in zip(p.RingNumbers, p.RingRadii):
        if 0 <= r < max_n_rings:
            out[r] = rad
    return torch.from_numpy(out)


def _compute_radius_dist_ideal(
    spots_first8: torch.Tensor, ring_radii: torch.Tensor
) -> torch.Tensor:
    """Per spot: distance from observed radius (col 8 of InputAll, after Ttheta) to ideal ring radius.

    InputAll.csv columns are
       0=YLab, 1=ZLab, 2=Omega, 3=GrainRadius, 4=SpotID,
       5=RingNumber, 6=Eta, 7=Ttheta
    The radius-distance is ``sqrt(YLab^2 + ZLab^2) - RingRadii[RingNumber]``.

    The C version (``CalcDistanceIdealRing``) recomputes this per spot before
    writing; we replicate the same formula.
    """
    yl = spots_first8[:, 0]
    zl = spots_first8[:, 1]
    ring_nr = spots_first8[:, 5].long()
    radius = torch.sqrt(yl * yl + zl * zl)
    ideal = ring_radii.to(radius.device)[ring_nr]
    return radius - ideal


def _bin_assignment(
    spots_first8: torch.Tensor,
    ring_radii: torch.Tensor,
    margin_ome: float,
    margin_eta: float,
    eta_bin_size: float,
    ome_bin_size: float,
    step_size_orient: float,
):
    """Vectorised per-spot bin assignment.

    Returns: (data_idx, ring_idx, eta_idx, ome_idx) — flattened triples.

    Notes (from SaveBinData.c:260-289):
      - Margins are spot-specific; the eta margin depends on ring radius and
        the ome margin depends on |sin(eta)|.
      - Bins wrap [0, 360) modulo n_eta and n_ome.
      - The C code does ``omemin = 180 + omega - omemargin`` and floors by
        bin size, generating an integer range [iOmeMin, iOmeMax]. We replicate.
    """
    device = spots_first8.device
    dtype = spots_first8.dtype

    omega = spots_first8[:, 2]
    ring_nr = spots_first8[:, 5].long()
    eta = spots_first8[:, 6]

    # Filter spots to those whose ring has a configured radius (>0).
    rrng = ring_radii.to(device=device, dtype=dtype)
    rad_for_ring = rrng[ring_nr]
    keep = rad_for_ring > 0
    if not keep.any():
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty

    spot_idx_all = torch.arange(spots_first8.shape[0], device=device)[keep]
    omega = omega[keep]
    eta = eta[keep]
    ring_nr = ring_nr[keep]
    rad_for_ring = rad_for_ring[keep]

    # Per-spot ome margin: avoid division by zero when sin(eta)=0 (eta=0 or 180)
    # and cap at 180° (half-period). Past 180°, every omega bin is already
    # covered after modulo wrap, so a larger margin only inflates the flat
    # (spot, eta, ome) triple list. The C path (SaveBinData.c:270-275) lets
    # the value blow up and relies on undefined int-cast saturation; the
    # vectorised path needs an explicit cap to avoid 1e10+ element tensors.
    sin_eta_abs = torch.abs(torch.sin(eta * _DEG2RAD))
    sin_eta_abs = torch.clamp(sin_eta_abs, min=1e-12)
    ome_margin_per = margin_ome + 0.5 * step_size_orient / sin_eta_abs
    ome_margin_per = torch.clamp(ome_margin_per, max=180.0)
    eta_margin_per = _RAD2DEG * torch.atan(margin_eta / rad_for_ring) + 0.5 * step_size_orient

    omemin = 180.0 + omega - ome_margin_per
    omemax = 180.0 + omega + ome_margin_per
    etamin = 180.0 + eta - eta_margin_per
    etamax = 180.0 + eta + eta_margin_per

    iome_min = torch.floor(omemin / ome_bin_size).long()
    iome_max = torch.floor(omemax / ome_bin_size).long()
    ieta_min = torch.floor(etamin / eta_bin_size).long()
    ieta_max = torch.floor(etamax / eta_bin_size).long()

    # Per-spot range count
    n_eta_per = (ieta_max - ieta_min + 1).clamp(min=0)
    n_ome_per = (iome_max - iome_min + 1).clamp(min=0)
    n_pairs_per = n_eta_per * n_ome_per

    total = int(n_pairs_per.sum().item())
    if total == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty, empty

    # Build flat arrays of (spot_idx, ring, iEta, iOme) in SPOT CHUNKS.
    # The all-at-once expansion holds ~10 pair-length int64 tensors
    # simultaneously; at multi-1e9 pairs (dense pf-HEDM layers: 100k+ spots
    # x ~60 bins each x 259 scans) that exceeds even 250 GB RAM — observed
    # as CUDA OOM (60 GiB single tensor) and kernel OOM-kills. Chunking over
    # spots bounds the transient to ~O(pair budget) while producing
    # BIT-IDENTICAL output: the expansion is per-spot (pos is the LOCAL
    # offset within each spot's segment), and chunks preserve spot order,
    # so the concatenated flat arrays match the unchunked path exactly.
    # The layout stays (iEta outer, iOme inner) matching C
    # ``for iEta0 ... for iOme0`` (SaveBinData.c): pos = i_eta * n_ome + i_ome.
    pair_budget = int(os.environ.get("MIDAS_BIN_PAIR_CHUNK", str(2 ** 28)))
    if device.type == "mps":
        # Multi-chunk int64 ops segfault on MPS (dev-Mac only; small data).
        # A single chunk reproduces the legacy all-at-once path, which is
        # fine there. Linux CPU/CUDA (production) chunk normally.
        pair_budget = 2 ** 62
    # Chunk-boundary bookkeeping runs on CPU regardless of the compute
    # device: the tensors are spot-length (small), and int64 searchsorted
    # segfaults on MPS.
    cum = torch.cumsum(n_pairs_per, dim=0).to("cpu")
    n_kept = int(n_pairs_per.shape[0])
    outs_spot, outs_ring, outs_ieta, outs_iome = [], [], [], []
    start = 0
    while start < n_kept:
        base = int(cum[start - 1].item()) if start > 0 else 0
        end = int(torch.searchsorted(cum, base + pair_budget, right=True).item())
        end = min(max(end, start + 1), n_kept)          # >=1 spot per chunk
        npp = n_pairs_per[start:end]
        c_total = int(npp.sum().item())
        if c_total == 0:
            start = end
            continue
        seg_starts = torch.cumsum(npp, dim=0) - npp
        pos = torch.arange(c_total, device=device) - torch.repeat_interleave(seg_starts, npp)
        c_n_ome = torch.repeat_interleave(n_ome_per[start:end], npp).clamp(min=1)
        eta_off = pos // c_n_ome
        ome_off = pos - eta_off * c_n_ome
        del pos, c_n_ome
        outs_spot.append(torch.repeat_interleave(spot_idx_all[start:end], npp))
        outs_ring.append(torch.repeat_interleave(ring_nr[start:end], npp))
        outs_ieta.append(torch.repeat_interleave(ieta_min[start:end], npp) + eta_off)
        outs_iome.append(torch.repeat_interleave(iome_min[start:end], npp) + ome_off)
        del eta_off, ome_off
        start = end

    # Concatenate one output at a time, freeing chunk pieces as we go, to
    # keep the peak at ~(4 outputs + 1 cat copy) rather than 2x everything.
    out_spot_idx = torch.cat(outs_spot); outs_spot.clear()
    out_ring = torch.cat(outs_ring); outs_ring.clear()
    out_ieta = torch.cat(outs_ieta); outs_ieta.clear()
    out_iome = torch.cat(outs_iome); outs_iome.clear()

    return out_spot_idx, out_ring, out_ieta, out_iome


def _bin_to_data_ndata(
    out_spot_idx: torch.Tensor,
    out_ring: torch.Tensor,
    out_ieta: torch.Tensor,
    out_iome: torch.Tensor,
    n_ring_bins: int,
    n_eta_bins: int,
    n_ome_bins: int,
):
    """Convert per-(spot, eta, ome) triples to ``(Data, nData)`` arrays.

    Layout matches ``SaveBinData.c:308-322`` — ring-major, eta-major, ome-major.
    Bins wrap modulo n_eta / n_ome.
    """
    device = out_spot_idx.device

    # NB memory hygiene: at multi-1e9 pairs each pair-length int64 tensor is
    # tens of GB; intermediates are freed (del) as soon as they are consumed
    # so the peak stays at ~6 pair-length tensors (the argsort) instead of ~10.

    # Modulo wrap (negative-aware).
    ieta_mod = (out_ieta % n_eta_bins + n_eta_bins) % n_eta_bins
    del out_ieta
    iome_mod = (out_iome % n_ome_bins + n_ome_bins) % n_ome_bins
    del out_iome

    # iRing in C is `ringnr - 1`; ring-bin axis is [0, HighestRingNo).
    iring = out_ring - 1
    del out_ring

    # Drop entries whose ring index is out of range. (Defensive; ring_nr is
    # filtered upstream — fast-path the all-true case to avoid 4 full copies.)
    mask = (iring >= 0) & (iring < n_ring_bins)
    if not bool(mask.all()):
        iring = iring[mask]
        ieta_mod = ieta_mod[mask]
        iome_mod = iome_mod[mask]
        out_spot_idx = out_spot_idx[mask]
    del mask

    # Composite bin id for sorting (ring outer, eta middle, ome inner).
    bin_id = (iring.long() * n_eta_bins + ieta_mod.long()) * n_ome_bins + iome_mod.long()
    del iring, ieta_mod, iome_mod
    # Stable sort by (bin_id, spot_idx) for deterministic insertion order.
    composite = bin_id * (out_spot_idx.max().item() + 2 if out_spot_idx.numel() else 1) + out_spot_idx
    order = torch.argsort(composite, stable=True)
    del composite
    sorted_bin_id = bin_id[order]
    del bin_id
    sorted_spot_idx = out_spot_idx[order]
    del out_spot_idx, order

    # Counts per bin: scatter add ones.
    total_bins = n_ring_bins * n_eta_bins * n_ome_bins
    counts = torch.zeros(total_bins, dtype=torch.int64, device=device)
    counts.scatter_add_(0, sorted_bin_id, torch.ones_like(sorted_bin_id))

    # Offsets are cumulative by visit order. Since bins are in
    # ring-major / eta-major / ome-major order in the output, the offset
    # for each bin is the running total of all preceding bins.
    offsets = torch.zeros(total_bins, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts[:-1], dim=0)

    # Pack ndata as (count, offset) per bin.
    ndata = torch.stack([counts, offsets], dim=1).to(torch.int32)
    data = sorted_spot_idx.to(torch.int32)
    return data, ndata


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def bin_data(
    result_folder: Union[str, Path] = ".",
    *,
    inputall_csv: Optional[Union[str, Path]] = None,
    inputall_extra_csv: Optional[Union[str, Path]] = None,
    paramstest_path: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    paramstest: Optional[ParamsTest] = None,
    spots_inputall: Optional[np.ndarray] = None,
    extra_inputall: Optional[np.ndarray] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    write: bool = True,
) -> BinDataResult:
    """Run the binning stage. Drop-in replacement for the C ``SaveBinData`` binary.

    Defaults match the C binary's argv-less convention: read ``InputAll.csv``,
    ``InputAllExtraInfoFittingAll.csv``, and ``paramstest.txt`` from the
    current directory; write outputs to the same directory.

    Parameters
    ----------
    result_folder
        Directory to read inputs from / write outputs to (when no override).
    inputall_csv, inputall_extra_csv, paramstest_path
        Optional input file overrides.
    out_dir
        Optional output directory override.
    paramstest, spots_inputall, extra_inputall
        Optional in-memory inputs (used by ``Pipeline``).
    device, dtype
        Torch device / dtype.
    write
        If ``False``, skip disk writes and return only the in-memory result.

    Returns
    -------
    BinDataResult
    """
    rf = Path(result_folder)
    out_dir = Path(out_dir) if out_dir is not None else rf

    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)

    # Inputs.
    if paramstest is None:
        ppath = paramstest_path if paramstest_path is not None else rf / "paramstest.txt"
        paramstest = read_paramstest(ppath)

    if spots_inputall is None:
        ipath = inputall_csv if inputall_csv is not None else rf / "InputAll.csv"
        spots_inputall, det_ids_in = csv_io.read_inputall_csv_with_detid(ipath)
    else:
        det_ids_in = np.ones(spots_inputall.shape[0], dtype=np.int32)
    if extra_inputall is None:
        ipath = inputall_extra_csv if inputall_extra_csv is not None else rf / "InputAllExtraInfoFittingAll.csv"
        extra_inputall = csv_io.read_inputall_extra_csv(ipath)

    if spots_inputall.shape[0] != extra_inputall.shape[0]:
        raise ValueError(
            f"InputAll ({spots_inputall.shape[0]} rows) and "
            f"InputAllExtraInfoFittingAll ({extra_inputall.shape[0]} rows) "
            "must agree on row count."
        )

    n_spots = spots_inputall.shape[0]
    if n_spots == 0:
        raise ValueError("No spots in InputAll.csv. Aborting.")

    # Move to device.
    spots_t = torch.from_numpy(spots_inputall.astype(np.float64)).to(device=dev, dtype=dt)
    extra_t = torch.from_numpy(extra_inputall.astype(np.float64)).to(device=dev, dtype=dt)

    # Compute Spots.bin layout (cols 0-7 + RadiusDistIdeal). For multi-det
    # runs each spot uses its own panel's ring radius (from RingRadii_DetN
    # blocks). Single-det runs fall through with one global table.
    ring_radii = _build_ring_radii(paramstest).to(device=dev, dtype=dt)
    has_per_det = bool(paramstest.RingRadiiPerDet)
    if has_per_det:
        # Per-spot ideal radius: lookup by (det_id, ring_nr). Start at NaN
        # so we can detect spots whose panel/ring isn't in the per-det map.
        ring_np = spots_inputall[:, 5].astype(np.int64)
        ideal_per_spot = np.full(spots_inputall.shape[0], np.nan, dtype=np.float64)
        for det_id, table in paramstest.RingRadiiPerDet.items():
            mask = det_ids_in == det_id
            if not mask.any():
                continue
            for rn, rad in table.items():
                ideal_per_spot[mask & (ring_np == rn)] = rad
        # Fall back to the global ring_radii for any unmapped spot.
        unfilled = np.isnan(ideal_per_spot)
        if unfilled.any():
            for r, rad in zip(paramstest.RingNumbers, paramstest.RingRadii):
                if 0 <= r < 500:
                    ideal_per_spot[unfilled & (ring_np == r)] = rad
            ideal_per_spot[np.isnan(ideal_per_spot)] = 0.0
        yl_np = spots_inputall[:, 0].astype(np.float64)
        zl_np = spots_inputall[:, 1].astype(np.float64)
        rad_dist_np = np.sqrt(yl_np * yl_np + zl_np * zl_np) - ideal_per_spot
        rad_dist = torch.from_numpy(rad_dist_np).to(device=dev, dtype=dt)
    elif dev.type == "cpu" and dt == torch.float64:
        # Byte-exact C-parity path (single-detector legacy).
        yl_np = spots_inputall[:, 0].astype(np.float64)
        zl_np = spots_inputall[:, 1].astype(np.float64)
        ring_np = spots_inputall[:, 5].astype(np.int64)
        ring_radii_np = np.zeros(500, dtype=np.float64)
        for r, rad in zip(paramstest.RingNumbers, paramstest.RingRadii):
            if 0 <= r < 500:
                ring_radii_np[r] = rad
        rad_dist_np = _radius_dist_ideal_numpy(yl_np, zl_np, ring_np, ring_radii_np)
        rad_dist = torch.from_numpy(rad_dist_np)
    else:
        rad_dist = _compute_radius_dist_ideal(spots_t, ring_radii)
    spots_out = torch.cat([spots_t, rad_dist.unsqueeze(1)], dim=1)  # (N, 9)
    # Phase 5: append ScanNr=0 column → (N, 10). The unified midas_indexer
    # always expects 10-col Spots.bin even in FF mode; the extra column is
    # zero for non-scanning runs.
    scan_nr_col = torch.zeros(spots_out.shape[0], 1,
                              dtype=spots_out.dtype, device=spots_out.device)
    spots_out = torch.cat([spots_out, scan_nr_col], dim=1)  # (N, 10)
    # ExtraInfo.bin is 16 cols; drop CSV cols 14 and 15 (the C version's dummy0/dummy1).
    # See SaveBinData.c — sscanf maps CSV[16, 17] to AllSpots[14, 15].
    # The 20-col appended layout (OrigSpotID/ReturnCode, N2+E3) reduces to
    # the same 16 binary cols — the binary stride is NEVER widened (C
    # consumers mmap fixed 16-double rows).
    if extra_t.shape[1] in (18, 20):
        extra_out = torch.cat([extra_t[:, :14], extra_t[:, 16:18]], dim=1)
    elif extra_t.shape[1] == 16:
        extra_out = extra_t
    else:
        raise ValueError(
            "InputAllExtraInfoFittingAll must have 16, 18, or 20 cols, "
            f"got {extra_t.shape[1]}"
        )

    if write:
        bio.write_spots_bin(out_dir / "Spots.bin", spots_out.detach().cpu().numpy().astype(np.float64))
        bio.write_extrainfo_bin(out_dir / "ExtraInfo.bin", extra_out.detach().cpu().numpy().astype(np.float64))
        # Multi-detector side-car: int32 DetID per spot row, in the same order
        # as Spots.bin. Always emitted (single-det runs see all 1s) so the
        # downstream torch tools can rely on its presence.
        np.asarray(det_ids_in, dtype=np.int32).tofile(out_dir / "Spots_det.bin")

    if paramstest.NoSaveAll == 1:
        return BinDataResult(
            spots=spots_out, extra_info=extra_out,
            paramstest=paramstest,
        )

    # Determine bin counts.
    n_ring_bins = paramstest.highest_ring_no
    n_eta_bins = math.ceil(360.0 / paramstest.EtaBinSize)
    n_ome_bins = math.ceil(360.0 / paramstest.OmeBinSize)

    out_spot_idx, out_ring, out_ieta, out_iome = _bin_assignment(
        spots_t,
        ring_radii,
        margin_ome=paramstest.MarginOme,
        margin_eta=paramstest.MarginEta,
        eta_bin_size=paramstest.EtaBinSize,
        ome_bin_size=paramstest.OmeBinSize,
        step_size_orient=paramstest.StepSizeOrient,
    )
    data, ndata = _bin_to_data_ndata(
        out_spot_idx, out_ring, out_ieta, out_iome,
        n_ring_bins=n_ring_bins, n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
    )

    if write:
        # Phase 5: always write the int64-pair format that the unified
        # midas_indexer C binary and read_bins_scanning() consume.
        # FF inputs get scan_nr=0 on every Data.bin entry.
        data_np = data.detach().cpu().numpy().astype(np.int64)
        data_pairs = np.zeros((data_np.size, 2), dtype=np.uint64)
        data_pairs[:, 0] = data_np.astype(np.uint64)
        ndata_np = ndata.detach().cpu().numpy().reshape(-1, 2).astype(np.uint64)
        bio.write_data_ndata_bin_scanning(
            out_dir / "Data.bin", out_dir / "nData.bin",
            data_pairs, ndata_np,
        )
        # Always emit positions.csv (single line "0.0" for FF). The unified
        # midas_indexer auto-detects mode from this file's row count.
        positions_path = out_dir / "positions.csv"
        if not positions_path.exists():
            positions_path.write_text("0.000000\n")

    return BinDataResult(
        spots=spots_out, extra_info=extra_out,
        data=data, ndata=ndata,
        n_ring_bins=n_ring_bins, n_eta_bins=n_eta_bins, n_ome_bins=n_ome_bins,
        paramstest=paramstest,
    )
