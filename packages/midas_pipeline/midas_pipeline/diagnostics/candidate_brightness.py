"""Per-candidate matched-spot brightness for PF point-by-point indexing.

The indexer matches spots on geometry alone (``CompareSpots``), so a voxel's
winner is the candidate with the highest completeness no matter how bright its
spots are. On a dense layer that lets a weakly diffracting orientation tie with
the grain that is actually there. On ESRF ma5608 the orientations that fit
voxels at completeness >= 0.8 without winning any had matched spots ~2.6x
brighter than a random orientation's chance matches and ~5.5x dimmer than the
winner's (claim 47602aed3339, 4/4 lenses) -- real but dim, and invisible to
completeness.

This pass puts a number on it. For every candidate in ``IndexBest_all.bin`` it
averages ``ln(I / ring median)`` over the candidate's matched spots
(``IndexBest_IDs_all.bin``), where ``I`` is the spot's IntegratedIntensity and
the ring median is over every observed spot of that ring. Intensity is not in
``ExtraInfo.bin`` (binning drops those columns), so it is read back from the
per-scan ``InputAllExtraInfoFittingAll{n}.csv`` through
``IDsMergedScanning.csv``.

Output ``<layer>/Output/CandidateBrightness.npz``:

- ``brightness``   float32 (n_candidates,)  mean ln(I/ring median), NaN below
  ``min_hits`` matched spots; candidates in IndexBest_all.bin order.
- ``n_hits``       int32   (n_candidates,)
- ``cand_start``   int64   (n_voxels,)  first candidate of each voxel.
- per voxel: ``winner`` (index within the voxel, argmax completeness, ties to
  the lower internal angle as the C does), ``winner_brightness``,
  ``contender`` (best other candidate within ``margin`` of the winner's
  completeness and at or above ``conf_min``; -1 if none),
  ``contender_brightness``, ``contender_completeness_gap``.

A contender that is much dimmer than the winner is the "dim grain behind the
mapped one" signature; a contender that is brighter is where the winner may be
wrong. Neither changes the map unless the opt-in tie-break is used.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

LOG = logging.getLogger("midas_pipeline.candidate_brightness")

BRIGHTNESS_NAME = "CandidateBrightness.npz"


def _read_scan_intensities(csv_path: Path):
    """(OrigSpotID, RingNumber, IntegratedIntensity) from one per-scan CSV,
    located by header name (the column count varies between 16 and 21)."""
    with open(csv_path) as f:
        header = f.readline().split()
    need = ("SpotID", "RingNumber", "IntegratedIntensity")
    if not all(n in header for n in need):
        raise ValueError(f"{csv_path}: header lacks {need}; got {header}")
    cols = [header.index(n) for n in need]
    arr = np.loadtxt(csv_path, skiprows=1, usecols=cols, ndmin=2)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), arr[:, 2]


def spot_log_intensity(layer_dir: Union[str, Path], n_scans: int,
                       csv_template: str = "InputAllExtraInfoFittingAll{n}.csv",
                       ) -> np.ndarray:
    """``ln(I / ring median)`` indexed by merged spot ID (NewID); index 0 and
    spots without a positive intensity are NaN."""
    layer_dir = Path(layer_dir)
    idmap = np.loadtxt(layer_dir / "IDsMergedScanning.csv", delimiter=",",
                       skiprows=1, dtype=np.int64, ndmin=2)
    new_id, orig_id, scan = idmap[:, 0], idmap[:, 1], idmap[:, 2]
    inten = np.full(int(new_id.max()) + 1, np.nan)
    ring = np.full(inten.shape, -1, dtype=np.int64)
    order = np.argsort(scan, kind="stable")
    bounds = np.searchsorted(scan[order], np.arange(n_scans + 1))
    for s in range(n_scans):
        rows = order[bounds[s]:bounds[s + 1]]
        if rows.size == 0:
            continue
        sid, rg, I = _read_scan_intensities(layer_dir / csv_template.format(n=s))
        srt = np.argsort(sid)
        pos = np.clip(np.searchsorted(sid[srt], orig_id[rows]), 0, len(sid) - 1)
        hit = sid[srt[pos]] == orig_id[rows]
        inten[new_id[rows[hit]]] = I[srt[pos[hit]]]
        ring[new_id[rows[hit]]] = rg[srt[pos[hit]]]
    out = np.full(inten.shape, np.nan)
    ok = np.isfinite(inten) & (inten > 0)
    for r in np.unique(ring[ok]):
        m = ok & (ring == r)
        out[m] = np.log(inten[m] / np.median(inten[m]))
    LOG.info("candidate_brightness: intensities for %d / %d merged spots",
             int(np.isfinite(out).sum()), len(new_id))
    return out


def candidate_brightness(layer_dir: Union[str, Path], n_scans: int, *,
                         output_subdir: str = "Output", min_hits: int = 5,
                         margin: float = 0.05, conf_min: float = 0.8,
                         chunk_ids: int = 20_000_000,
                         write: bool = True) -> dict:
    """Compute and (by default) write ``Output/CandidateBrightness.npz``."""
    from ..find_grains._consolidation_io import open_all_three

    layer_dir = Path(layer_dir)
    out_dir = layer_dir / output_subdir
    vals_r, keys_r, ids_r = open_all_three(out_dir)
    n_vox = vals_r.n_voxels
    n_sol = vals_r.n_sol_arr.astype(np.int64)
    n_cand = int(n_sol.sum())
    lnI = spot_log_intensity(layer_dir, n_scans)

    # All three payloads are contiguous in voxel order (checked here, relied
    # on below), so candidates and their IDs can be read as flat arrays.
    def _flat(r, width, dtype, count):
        nz = r.n_sol_arr > 0
        ends = r.off_arr + r.n_sol_arr.astype(np.int64) * width
        if nz.any() and not np.all(r.off_arr[1:][nz[1:]] == ends[:-1][nz[1:]]):
            raise ValueError(f"{r.path}: voxel blocks are not contiguous")
        start = int(r.off_arr[np.argmax(nz)]) if nz.any() else r.header_size
        return np.frombuffer(r.raw, dtype=dtype, count=count, offset=start)

    vals = _flat(vals_r, 128, np.float64, n_cand * 16).reshape(n_cand, 16)
    keys = _flat(keys_r, 32, np.uint64, n_cand * 4).reshape(n_cand, 4)
    n_ids_tot = int(ids_r.n_sol_arr.astype(np.int64).sum())
    ids = _flat(ids_r, 4, np.int32, n_ids_tot)
    n_ids = keys[:, 2].astype(np.int64)
    if int(n_ids.sum()) != n_ids_tot:
        raise ValueError("IndexKey nIDs do not add up to IndexBest_IDs_all.bin")

    # Chunked sum of ln(I) and count of finite values per candidate.
    sums = np.zeros(n_cand)
    cnts = np.zeros(n_cand, dtype=np.int64)
    id_start = np.concatenate([[0], np.cumsum(n_ids)])
    c0 = 0
    while c0 < n_cand:
        c1 = int(np.searchsorted(id_start, id_start[c0] + chunk_ids, side="right")) - 1
        c1 = min(max(c1, c0 + 1), n_cand)
        a, b = id_start[c0], id_start[c1]
        v = lnI[np.clip(ids[a:b], 0, len(lnI) - 1)]
        fin = np.isfinite(v)
        owner = np.repeat(np.arange(c0, c1), n_ids[c0:c1])
        sums += np.bincount(owner, weights=np.where(fin, v, 0.0), minlength=n_cand)
        cnts += np.bincount(owner, weights=fin, minlength=n_cand).astype(np.int64)
        c0 = c1
    bright = np.where(cnts >= min_hits, sums / np.maximum(cnts, 1), np.nan).astype(np.float32)

    # Per-voxel winner / contender.
    comp = vals[:, 15] / np.maximum(vals[:, 14], 1.0)
    ia = vals[:, 1]
    cand_start = np.concatenate([[0], np.cumsum(n_sol)])[:-1]
    winner = np.full(n_vox, -1, dtype=np.int32)
    contender = np.full(n_vox, -1, dtype=np.int32)
    wb = np.full(n_vox, np.nan, dtype=np.float32)
    cb = np.full(n_vox, np.nan, dtype=np.float32)
    gap = np.full(n_vox, np.nan, dtype=np.float32)
    for vx in np.flatnonzero(n_sol > 0):
        s, n = cand_start[vx], n_sol[vx]
        c = comp[s:s + n]
        best = np.flatnonzero(c == c.max())
        w = int(best[np.argmin(ia[s:s + n][best])]) if best.size > 1 else int(best[0])
        winner[vx] = w
        wb[vx] = bright[s + w]
        others = np.flatnonzero((c >= c[w] - margin) & (c >= conf_min))
        others = others[others != w]
        if others.size:
            k = int(others[np.argmax(c[others])])
            contender[vx] = k
            cb[vx] = bright[s + k]
            gap[vx] = c[w] - c[k]

    res = dict(brightness=bright, n_hits=cnts.astype(np.int32),
               cand_start=cand_start.astype(np.int64), winner=winner,
               winner_brightness=wb, contender=contender,
               contender_brightness=cb, contender_completeness_gap=gap,
               min_hits=np.int32(min_hits), margin=np.float32(margin),
               conf_min=np.float32(conf_min))
    n_amb = int((contender >= 0).sum())
    dim = np.isfinite(cb) & np.isfinite(wb)
    LOG.info("candidate_brightness: %d voxels, %d with a contender within %.3g of the "
             "winner at completeness >= %.2f; median winner-contender brightness "
             "gap %.3f ln", n_vox, n_amb, margin, conf_min,
             float(np.median(wb[dim] - cb[dim])) if dim.any() else float("nan"))
    if write:
        np.savez(out_dir / BRIGHTNESS_NAME, **res)
    return res


def brightness_tiebreak(comp: np.ndarray, ia: np.ndarray,
                        bright: Optional[np.ndarray], margin: float) -> int:
    """Winner index for one voxel's candidates.

    ``margin <= 0`` or no brightness: the indexer's rule (max completeness,
    ties to the lower internal angle). Otherwise, among candidates within
    ``margin`` of the best completeness, the brightest wins (NaN brightness
    never beats a finite one); ties fall back to the indexer's rule.
    """
    best = np.flatnonzero(comp == comp.max())
    default = int(best[np.argmin(ia[best])]) if best.size > 1 else int(best[0])
    if margin <= 0 or bright is None:
        return default
    near = np.flatnonzero(comp >= comp.max() - margin)
    b = np.where(np.isfinite(bright[near]), bright[near], -np.inf)
    if not np.isfinite(b.max()):
        return default
    top = near[b == b.max()]
    if default in top:
        return default
    return int(top[np.argmax(comp[top])])
