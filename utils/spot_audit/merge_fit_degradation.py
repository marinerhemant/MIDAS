"""Why does a clean blob become a dirty recorded spot?

Established (Lab Notebook 6b): the SNR>5 fraction halves from detection
(53.4%) to the recorded spot list (28.8%) under one estimator. Three candidate
mechanisms, tested here:

  A. FIT DIVERGENCE -- the fit fails and puts the centroid where there is no
     intensity. Lead: ReturnCode != -1 on 58.4% of unindexed spots vs 9.2% of
     indexed.
  B. FIT DISPLACEMENT -- even when it "converges", the recorded position drifts
     off the blob that produced it. Tested by measuring, per frame, the
     distance from each recorded spot to the nearest DETECTED blob, and how SNR
     falls with that distance.
  C. MERGE CHAINING -- merge_overlaps joins weak single-frame detections that
     should not join. Tested via NImgs and the merge multiplicity.

A and C need only the audit CSV. B needs the frames.
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


def band(name, v, extra=""):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        print(f"  {name:38s} n=0")
        return
    print(f"  {name:38s} n={v.size:5d}  median SNR {np.median(v):8.1f}  "
          f"frac>5 {(v > 5).mean():6.1%} {extra}")


def main():
    m = pd.read_csv(AUD)
    snr = m["own_snr"].to_numpy()
    ok = np.isfinite(snr)
    ring = m["RingNumber"].to_numpy().astype(int)
    real = ok & (ring > 0)

    print("=" * 72)
    print("A. FIT DIVERGENCE — does ReturnCode explain the low SNR?")
    print("=" * 72)
    rc = m["ReturnCode"].to_numpy()
    for label, sel in (("ReturnCode == -1 (converged)", real & (rc == -1)),
                       ("ReturnCode != -1 (did NOT)", real & (rc != -1))):
        band(label, snr[sel])
    conv, div = real & (rc == -1), real & (rc != -1)
    print(f"\n  population split: {int(conv.sum())} converged / "
          f"{int(div.sum())} diverged  ({div.sum()/max(real.sum(),1):.1%} diverged)")
    # If divergence were the whole story, converged spots would be clean.
    print(f"  -> if divergence were the whole story, the converged row would be "
          f"~clean.\n     It is {100*(snr[conv] > 5).mean():.1f}% clean.")

    print()
    print("=" * 72)
    print("C. MERGE CHAINING — does merge multiplicity track dirtiness?")
    print("=" * 72)
    nim = m["NImgs"].to_numpy()
    for lo, hi, lbl in ((1, 1, "NImgs == 1 (single frame)"),
                        (2, 2, "NImgs == 2"),
                        (3, 5, "NImgs 3-5"),
                        (6, 10**9, "NImgs >= 6")):
        band(lbl, snr[real & (nim >= lo) & (nim <= hi)])

    print("\n  FitRMSE by cleanliness (a bad fit should show a bad RMSE):")
    for lbl, sel in (("SNR > 5 ", real & (snr > 5)), ("SNR <= 5", real & (snr <= 5))):
        v = m.loc[sel, "FitRMSE"].to_numpy()
        print(f"    {lbl} n={v.size:5d}  median FitRMSE {np.median(v):10.3g}")

    print()
    print("=" * 72)
    print("B. FIT DISPLACEMENT — is the recorded position ON a detected blob?")
    print("=" * 72)
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
    Rt, _ = compute_rt_eta(p, panels)
    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    flood = prepare_flood(p.flood, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    in_band = np.zeros(Rt.shape, bool)
    for rad in rads:
        in_band |= (Rt > rad - p.Width) & (Rt < rad + p.Width)
    gc_band = np.where(in_band, 1e-12, 0.0)
    orig = np.array(p.Thresholds, float).copy()
    p.Thresholds = np.full_like(orig, THR)
    gc = compute_good_coords(p, panels, rads)

    omes = np.array([frame_omega(p, i + p.skipFrame) for i in range(p.nFrames)])
    m["frame"] = [int(np.argmin(np.abs(omes - o))) for o in m["Omega"]]
    idxs = np.linspace(0, p.nFrames - 1, NFRAMES).astype(int)

    dists, dsnr, drc = [], [], []
    for fi in idxs:
        raw = read_frame(str(ZIP), int(fi) + p.skipFrame)
        ung = correct_frame(
            raw, NrPixels=p.NrPixels, NrPixelsY=p.NrPixelsY, NrPixelsZ=p.NrPixelsZ,
            transform_options=p.TransOpt, dark=dark, flood=flood,
            good_coords=gc_band, bc=p.bc,
            bad_px_intensity=p.BadPxIntensity, make_map=p.makeMap)
        regs = filter_regions_by_size(find_regions(apply_threshold(ung, gc), gc),
                                      p.minNrPx, p.maxNrPx)
        if not regs:
            continue
        # blob centroid in CORRECTED frame (row=Y, col=Z) -> raw (row=Z, col=Y)
        braw = np.array([[reg.pixel_cols.mean(), reg.pixel_rows.mean()]
                         for reg in regs])
        sub = m[m["frame"] == int(fi)]
        for _, r in sub.iterrows():
            if not np.isfinite(r["own_snr"]) or r["RingNumber"] <= 0:
                continue
            d = np.hypot(braw[:, 0] - r["ZRawPx"], braw[:, 1] - r["YRawPx"]).min()
            dists.append(d); dsnr.append(r["own_snr"]); drc.append(r["ReturnCode"])

    dists = np.asarray(dists); dsnr = np.asarray(dsnr); drc = np.asarray(drc)
    print(f"  {len(dists)} recorded spots on {len(idxs)} sampled frames\n")
    print(f"  {'distance to nearest blob':32s} {'n':>6s} {'median SNR':>12s} "
          f"{'frac>5':>8s}")
    for lo, hi, lbl in ((0, 2, "on the blob   (<= 2 px)"),
                        (2, 5, "near          (2-5 px)"),
                        (5, 20, "off           (5-20 px)"),
                        (20, 1e9, "far           (> 20 px)")):
        s = (dists >= lo) & (dists < hi)
        if s.sum() == 0:
            print(f"  {lbl:32s} {0:6d}")
            continue
        print(f"  {lbl:32s} {int(s.sum()):6d} {np.median(dsnr[s]):12.1f} "
              f"{(dsnr[s] > 5).mean():8.1%}")
    far = dists > 5
    print(f"\n  spots recorded MORE THAN 5 px from any detected blob: "
          f"{int(far.sum())}/{len(dists)} ({far.mean():.1%})")
    if far.sum():
        print(f"    of those, ReturnCode != -1: {(drc[far] != -1).mean():.1%}")
        print(f"    of those, SNR > 5         : {(dsnr[far] > 5).mean():.1%}")
    near = dists <= 2
    if near.sum():
        print(f"  spots ON a blob (<=2 px): {int(near.sum())}, "
              f"SNR>5 {(dsnr[near] > 5).mean():.1%}")
    p.Thresholds = orig


if __name__ == "__main__":
    main()
