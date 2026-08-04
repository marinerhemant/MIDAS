"""Is the noise entering at DETECTION, or manufactured downstream?

Two measurements are in tension at RingThresh 10:
  * midas-ring-thresh: 92% of per-frame BLOBS have SNR > 5  (clean)
  * the spot audit:    63% of merged SPOTS have SNR <= 5    (dirty)

They differ in two ways at once -- the objects scored (per-frame blobs at
detection vs merged/fitted spots after merge_overlaps) and the estimator
(in-band annulus vs 81x81 box). So vary one at a time: a 2x2 of
{stage} x {estimator}, all on the SAME frames.

  if a row differs but columns agree -> the STAGE matters: blobs are clean and
      merging/fitting manufactures the noise, i.e. RingThresh is fine.
  if a column differs but rows agree -> the ESTIMATOR matters: one of the two
      SNR definitions is wrong and the disagreement is an artefact.

Coordinates: blob pixels are in the CORRECTED frame (correct_frame applies
transpose_square), while YRawPx/ZRawPx index the raw frame. The mapping is
calibrated on known-indexed spots rather than assumed -- guessing it wrong is
how this investigation went astray twice already.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/au3_cubes_ff_000008")
RESULT = R / "results/LayerNr_1"
ZIP = RESULT / "Au3_cubes_ff_000008.MIDAS.zip"
AUD = R / "spot_noise_audit/spot_audit_snr.csv"
NFRAMES = 30
THR = 10.0
BG_INNER, BG_OUTER = 6, 20      # estimator E1 (calculator)
BOX_HALF = 40                   # estimator E2 (audit)


def e1(img, rows, cols, valid):
    """In-band annulus, MAD -- the calculator's estimator."""
    from midas_peakfit.ring_thresh import blob_snr
    return blob_snr(img, np.asarray(rows), np.asarray(cols), valid=valid)


def e2(img, row, col):
    """81x81 box, MAD -- the spot audit's estimator.

    MUST be given the RAW dark-subtracted frame, never the corrected one: the
    corrected frame is zero outside the ring bands, so an 81x81 box is mostly
    zeros, its MAD collapses to ~0 and the SNR explodes to ~1e10. That is what
    the first run of this script produced.
    """
    r0, r1 = max(0, row - BOX_HALF), min(img.shape[0], row + BOX_HALF)
    c0, c1 = max(0, col - BOX_HALF), min(img.shape[1], col + BOX_HALF)
    box = img[r0:r1, c0:c1]
    if box.size < 64:
        return np.nan
    bg = float(np.median(box))
    sd = float(np.median(np.abs(box - bg))) * 1.4826 + 1e-9
    sr0, sr1 = max(0, row - 2), min(img.shape[0], row + 3)
    sc0, sc1 = max(0, col - 2), min(img.shape[1], col + 3)
    return (float(img[sr0:sr1, sc0:sc1].max()) - bg) / sd


def summarize(name, vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        print(f"  {name:34s}      n=0")
        return
    print(f"  {name:34s} n={v.size:5d}  median {np.median(v):9.1f}  "
          f"frac>5 {(v > 5).mean():6.1%}")


def main():
    from midas_peakfit.background import build_background_bins
    from midas_peakfit.connected import filter_regions_by_size, find_regions
    from midas_peakfit.geometry import compute_good_coords, compute_rt_eta, load_ring_radii
    from midas_peakfit.orchestrator import _build_panels
    from midas_peakfit.preprocess import (
        apply_threshold, correct_frame, prepare_dark, prepare_flood,
    )
    from midas_peakfit.zarr_io import (
        frame_omega, load_corrections, parse_zarr_params, read_frame,
    )

    p = parse_zarr_params(str(ZIP))
    p.ResultFolder = str(RESULT)
    panels = _build_panels(p)
    load_corrections(str(ZIP), p)
    rads = np.asarray(load_ring_radii(p, p.ResultFolder), float)[: p.nRingsThresh]
    Rt, Eta = compute_rt_eta(p, panels)
    bins = build_background_bins(Rt, Eta, rads, float(p.Width), n_sectors=36)
    band = bins.in_band

    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    flood = prepare_flood(p.flood, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    gc_band = np.where(band, 1e-12, 0.0)
    orig = np.array(p.Thresholds, float).copy()
    p.Thresholds = np.full_like(orig, THR)
    gc = compute_good_coords(p, panels, rads)

    m = pd.read_csv(AUD)
    omes = np.array([frame_omega(p, i + p.skipFrame) for i in range(p.nFrames)])
    m["frame"] = [int(np.argmin(np.abs(omes - o))) for o in m["Omega"]]

    idxs = np.linspace(0, p.nFrames - 1, NFRAMES).astype(int)

    # ── calibrate the blob-frame <-> raw-frame coordinate mapping ──────────
    idx_spots = m[m["indexed"]]
    cal = idx_spots[idx_spots["frame"].isin(idxs)]
    if len(cal) < 5:
        cal = idx_spots.head(20)
        idxs = np.unique(np.concatenate([idxs, cal["frame"].to_numpy()]))
    print(f"calibrating coordinate convention on {len(cal)} indexed spots")
    scores = {"(Z,Y)": [], "(Y,Z)": []}
    for fi in np.unique(cal["frame"]):
        raw = read_frame(str(ZIP), int(fi) + p.skipFrame)
        rawsub = raw.astype(np.float64) - dark
        for _, r in cal[cal["frame"] == fi].iterrows():
            z, y = int(round(r["ZRawPx"])), int(round(r["YRawPx"]))
            scores["(Z,Y)"].append(e2(rawsub, z, y))
            scores["(Y,Z)"].append(e2(rawsub, y, z))
    best, best_med = None, -np.inf
    for k, v in scores.items():
        med = float(np.nanmedian(v))
        print(f"  RAW-frame indexing row,col = {k}: median SNR {med:9.1f}")
        if med > best_med:
            best, best_med = k, med
    print(f"  -> using {best}")
    if best_med < 20:
        raise SystemExit("ABORT: neither convention recovers known-real spots.")
    flip = (best == "(Y,Z)")

    # ── the 2x2 ────────────────────────────────────────────────────────────
    blob_e1, blob_e2, spot_e1, spot_e2 = [], [], [], []
    for fi in idxs:
        raw = read_frame(str(ZIP), int(fi) + p.skipFrame)
        ung = correct_frame(
            raw, NrPixels=p.NrPixels, NrPixelsY=p.NrPixelsY, NrPixelsZ=p.NrPixelsZ,
            transform_options=p.TransOpt, dark=dark, flood=flood,
            good_coords=gc_band, bc=p.bc,
            bad_px_intensity=p.BadPxIntensity, make_map=p.makeMap)
        gated = apply_threshold(ung, gc)
        # corrected[i,j] == rawsq[j,i] (transpose_square), so a blob at
        # corrected (r,c) sits at raw (c,r).
        rawsub = raw.astype(np.float64) - dark

        # STAGE A -- per-frame blobs at detection
        for reg in filter_regions_by_size(find_regions(gated, gc), p.minNrPx, p.maxNrPx):
            blob_e1.append(e1(ung, reg.pixel_rows, reg.pixel_cols, band))
            cr = int(round(reg.pixel_rows.mean())); cc = int(round(reg.pixel_cols.mean()))
            blob_e2.append(e2(rawsub, cc, cr))          # transpose back to raw

        # STAGE B -- merged/fitted spots recorded on this frame
        for _, r in m[m["frame"] == int(fi)].iterrows():
            z, y = int(round(r["ZRawPx"])), int(round(r["YRawPx"]))
            # Two frames, two conventions -- do NOT reuse one for the other.
            # Calibration above establishes spots sit at raw (Z, Y); since
            # corrected(r,c) == raw(c,r), the SAME spot is at corrected (Y, Z).
            if not (0 <= z < rawsub.shape[0] and 0 <= y < rawsub.shape[1]):
                continue
            spot_e2.append(e2(rawsub, z, y))            # raw frame,      (Z,Y)
            if 0 <= y < ung.shape[0] and 0 <= z < ung.shape[1]:
                spot_e1.append(e1(ung, np.array([y]), np.array([z]), band))

    p.Thresholds = orig
    print(f"\n=== 2x2: stage x estimator, RingThresh {THR:.0f}, "
          f"{len(idxs)} frames ===")
    print("\nE1 = in-band annulus (calculator)")
    summarize("  stage A: detected blobs", blob_e1)
    summarize("  stage B: merged/fitted spots", spot_e1)
    print("\nE2 = 81x81 box (spot audit)")
    summarize("  stage A: detected blobs", blob_e2)
    summarize("  stage B: merged/fitted spots", spot_e2)


if __name__ == "__main__":
    main()
