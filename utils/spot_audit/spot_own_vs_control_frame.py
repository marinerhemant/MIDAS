"""Decisive test: is the intensity at a spot's pixel LOCALIZED IN OMEGA?

The previous figure was drawn with vmax = 99.5th percentile of each 25x25
patch, so a bright neighbour saturated the scale and hid weak spots at the
marker. That was a plotting artifact, not a result. Redone here properly.

The real question is not "is there intensity at the recorded pixel" (a peak
finder put it there -- circular), but whether that intensity is SPECIFIC to
the spot's own omega frame:

  * a real Bragg reflection is time-localized -- bright on its own frame,
    gone 90 deg of rotation later at the same detector pixel;
  * a hot pixel / dead column / scattering artifact is ALWAYS there, so it
    looks identical on a control frame;
  * pure noise is at background on both.

So for each spot: SNR at (ZRawPx, YRawPx) on its OWN frame, and at the SAME
pixel on control frames at omega +-90 deg. The ratio separates all three.
This also kills the hot-pixel explanation for the Friedel excess: mirrored
hot pixels would pair on every frame, but would show no own-vs-control
contrast.

Frames are read once each (spots grouped by frame) because each read pulls
from a 3.7 GB zarr.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("/gdata/dm/1ID/2026/pokharel_jul26/analysis/au3_cubes_ff_000008")
AUD = R / "spot_noise_audit/spot_audit.csv"
ZIP = R / "results/LayerNr_1/Au3_cubes_ff_000008.MIDAS.zip"
OUT = R / "spot_noise_audit"
RNG = np.random.default_rng(3)
NSAMP = 70
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
    nF = len(omes)

    def frame_of(ome):
        return int(np.argmin(np.abs(omes - ome)))

    m = pd.read_csv(AUD)
    idx = m["indexed"].to_numpy().astype(bool)
    nim = m["NImgs"].to_numpy()
    ring = m["RingNumber"].to_numpy().astype(int)

    strata = {
        "indexed": idx,
        "unindexed NImgs>=2": ~idx & (nim >= 2),
        "unindexed NImgs==1": ~idx & (nim <= 1) & (ring > 0),
        "ring-0 padding": ring == 0,
    }

    # build the full work list first, so each frame is read exactly once
    jobs = []   # (stratum, spot_row_index, frame, row, col, kind)
    for name, sel in strata.items():
        w = np.where(sel)[0]
        if len(w) == 0:
            continue
        take = RNG.choice(w, size=min(NSAMP, len(w)), replace=False)
        for i in take:
            r = m.iloc[i]
            row, col = int(round(r["ZRawPx"])), int(round(r["YRawPx"]))
            if not (0 <= row < 2048 and 0 <= col < 2048):
                continue
            f0 = frame_of(float(r["Omega"]))
            jobs.append((name, i, f0, row, col, "own"))
            for d in (+90.0, -90.0):
                jobs.append((name, i, (f0 + int(d / 0.25)) % nF, row, col, "ctl"))

    by_frame = defaultdict(list)
    for j in jobs:
        by_frame[j[2]].append(j)
    print(f"{len(jobs)} measurements over {len(by_frame)} unique frames")

    res = defaultdict(lambda: defaultdict(list))
    for k, (fi, group) in enumerate(sorted(by_frame.items())):
        img = read_frame(str(ZIP), fi + p.skipFrame).astype(np.float64) - dark
        for name, i, _, row, col, kind in group:
            res[name][kind].append(snr_at(img, row, col))
        if k % 50 == 0:
            print(f"  ...{k}/{len(by_frame)} frames")

    print("\nSNR at the recorded pixel: OWN frame vs SAME pixel +-90 deg away")
    print(f"  {'stratum':22s} {'n':>5s} {'own med':>9s} {'ctl med':>9s} "
          f"{'own/ctl':>8s} {'own>5':>7s} {'ctl>5':>7s}")
    summary = {}
    for name in strata:
        own = np.array([v for v in res[name]["own"] if np.isfinite(v)])
        ctl = np.array([v for v in res[name]["ctl"] if np.isfinite(v)])
        if len(own) == 0:
            continue
        mo, mc = np.median(own), np.median(ctl)
        summary[name] = (own, ctl)
        print(f"  {name:22s} {len(own):5d} {mo:9.1f} {mc:9.1f} "
              f"{mo/max(mc,1e-9):8.2f} {(own>5).mean():7.1%} {(ctl>5).mean():7.1%}")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    names = list(summary)
    for k, name in enumerate(names):
        own, ctl = summary[name]
        for arr, off, c, lab in ((own, -0.16, "#2b6cb0", "own frame"),
                                 (ctl, +0.16, "#b0b0b0", "+-90 deg control")):
            x = k + off + RNG.normal(0, 0.035, len(arr))
            ax.scatter(x, np.clip(arr, 0.05, None), s=9, c=c, alpha=0.55,
                       label=lab if k == 0 else None)
            ax.plot([k + off - 0.11, k + off + 0.11],
                    [np.median(arr)] * 2, c="k", lw=2)
    ax.set_yscale("log")
    ax.axhline(5, ls="--", c="r", lw=1, label="SNR = 5")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("SNR at the recorded detector pixel")
    ax.set_title("Is the intensity localized in omega?\n"
                 "real Bragg spot: high on its own frame, background 90 deg away",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = OUT / "own_vs_control_frame.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
