"""Own-frame SNR for EVERY spot, so the credible-spot count is exact.

The +-90 deg control run already established omega-localization per stratum
(indexed 1822x, unindexed ~2.2x, padding 1.0x). The ~2.2x for unindexed is
what pure selection bias produces on its own: the pixel was chosen BECAUSE
it was a local maximum on that frame, so it beats the same pixel elsewhere
without any diffraction. So localization is not the discriminator -- absolute
SNR is.

This measures own-frame SNR for all 2076 spots (one read per frame) rather
than extrapolating a percentage from 70 samples, which carried +-5% binomial
error and ~+-80 spots of slop in the headline count.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/gdata/dm/1ID/2026/pokharel_jul26/analysis/au3_cubes_ff_000008")
AUD = R / "spot_noise_audit/spot_audit.csv"
ZIP = R / "results/LayerNr_1/Au3_cubes_ff_000008.MIDAS.zip"
OUT = R / "spot_noise_audit"
SIG_HALF, BG_HALF = 2, 40


def snr_at(img, row, col):
    if not (0 <= row < img.shape[0] and 0 <= col < img.shape[1]):
        return np.nan
    r0, r1 = max(0, row - SIG_HALF), min(img.shape[0], row + SIG_HALF + 1)
    c0, c1 = max(0, col - SIG_HALF), min(img.shape[1], col + SIG_HALF + 1)
    br0, br1 = max(0, row - BG_HALF), min(img.shape[0], row + BG_HALF)
    bc0, bc1 = max(0, col - BG_HALF), min(img.shape[1], col + BG_HALF)
    ring = img[br0:br1, bc0:bc1]
    bg = float(np.median(ring))
    sd = float(np.median(np.abs(ring - bg))) * 1.4826 + 1e-9
    return (float(img[r0:r1, c0:c1].max()) - bg) / sd


def main():
    from midas_peakfit.preprocess import prepare_dark
    from midas_peakfit.zarr_io import (
        frame_omega, load_corrections, parse_zarr_params, read_frame,
    )
    p = parse_zarr_params(str(ZIP))
    load_corrections(str(ZIP), p)
    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    omes = np.array([frame_omega(p, i + p.skipFrame) for i in range(p.nFrames)])

    m = pd.read_csv(AUD)
    snr = np.full(len(m), np.nan)
    by_frame = defaultdict(list)
    for i, r in m.iterrows():
        row, col = int(round(r["ZRawPx"])), int(round(r["YRawPx"]))
        if 0 <= row < 2048 and 0 <= col < 2048:
            by_frame[int(np.argmin(np.abs(omes - float(r["Omega"]))))].append(
                (i, row, col))
    print(f"{len(m)} spots over {len(by_frame)} unique frames")
    for k, (fi, group) in enumerate(sorted(by_frame.items())):
        img = read_frame(str(ZIP), fi + p.skipFrame).astype(np.float64) - dark
        for i, row, col in group:
            snr[i] = snr_at(img, row, col)
        if k % 100 == 0:
            print(f"  ...{k}/{len(by_frame)}", flush=True)

    m["own_snr"] = snr
    m.to_csv(OUT / "spot_audit_snr.csv", index=False)

    idx = m["indexed"].to_numpy().astype(bool)
    ring = m["RingNumber"].to_numpy().astype(int)
    nim = m["NImgs"].to_numpy()
    ok = np.isfinite(snr)

    print("\n=== own-frame SNR at the recorded pixel, ALL spots ===")
    print(f"  {'stratum':32s} {'n':>6s} {'median':>9s} {'SNR>5':>8s} "
          f"{'SNR>10':>8s} {'count>5':>8s}")
    strata = [
        ("indexed", idx & ok),
        ("unindexed, ring>0, NImgs>=2", ~idx & ok & (ring > 0) & (nim >= 2)),
        ("unindexed, ring>0, NImgs==1", ~idx & ok & (ring > 0) & (nim <= 1)),
        ("ring-0 padding", ok & (ring == 0)),
    ]
    tot_real_unidx = 0
    for name, sel in strata:
        v = snr[sel]
        if len(v) == 0:
            continue
        n5 = int((v > 5).sum())
        if name.startswith("unindexed"):
            tot_real_unidx += n5
        print(f"  {name:32s} {len(v):6d} {np.median(v):9.1f} "
              f"{(v>5).mean():8.1%} {(v>10).mean():8.1%} {n5:8d}")

    n_idx = int((idx & ok).sum())
    denom = n_idx + tot_real_unidx
    print(f"\n=== corrected completeness ===")
    print(f"  credible spots (SNR>5): {denom}  "
          f"= {n_idx} indexed + {tot_real_unidx} unindexed")
    print(f"  completeness over credible spots: {n_idx/max(denom,1):.1%}")
    print(f"  (raw, over all 2076 rows:          {n_idx/len(m):.1%})")

    # implied grain size of the CREDIBLE unindexed spots only
    sel = ~idx & ok & (ring > 0) & (snr > 5)
    gr = m.loc[sel, "GrainRadius"].to_numpy()
    gi = m.loc[idx & ok, "GrainRadius"].to_numpy()
    if len(gr):
        print(f"\n=== implied grain size, CREDIBLE spots only ===")
        print(f"  indexed            n={len(gi):4d}  median {np.median(gi):7.1f} um")
        print(f"  unindexed (SNR>5)  n={len(gr):4d}  median {np.median(gr):7.1f} um"
              f"   p25 {np.percentile(gr,25):.1f}  p75 {np.percentile(gr,75):.1f}")
        rr = np.median(gi) / max(np.median(gr), 1e-9)
        print(f"  -> {rr:.1f}x smaller radius, {rr**3:.0f}x smaller volume")


if __name__ == "__main__":
    main()
