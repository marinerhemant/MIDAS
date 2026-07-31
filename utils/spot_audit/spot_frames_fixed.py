"""Corrected raw-frame figure (replaces spot_strata_frames.png).

The first version scaled each panel to vmax = 99.5th percentile OF THAT PATCH.
Any bright neighbour inside the 25x25 window then drove vmax up and made a
genuinely weak spot at the marker invisible -- so several panels looked like
"nothing there" when the spot was simply faint. That was a plotting artifact
and the figure should not have been read as evidence.

Fixed here: every panel is scaled to its OWN local background,
vmin = bg, vmax = bg + 8*sigma_MAD, computed from a 81x81 box. A spot at
SNR 8 saturates, a noise excursion at SNR 2 stays dark, and the scale is
independent of neighbours. Each panel is annotated with the measured
own-frame SNR at the marker so the picture and the number agree.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("/gdata/dm/1ID/2026/pokharel_jul26/analysis/au3_cubes_ff_000008")
SNRCSV = R / "spot_noise_audit/spot_audit_snr.csv"
ZIP = R / "results/LayerNr_1/Au3_cubes_ff_000008.MIDAS.zip"
OUT = R / "spot_noise_audit/spot_strata_frames_fixed.png"
RNG = np.random.default_rng(11)
HALF, BG = 12, 40


def main():
    from midas_peakfit.preprocess import prepare_dark
    from midas_peakfit.zarr_io import (
        frame_omega, load_corrections, parse_zarr_params, read_frame,
    )
    p = parse_zarr_params(str(ZIP))
    load_corrections(str(ZIP), p)
    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    omes = np.array([frame_omega(p, i + p.skipFrame) for i in range(p.nFrames)])

    m = pd.read_csv(SNRCSV)
    idx = m["indexed"].to_numpy().astype(bool)
    nim = m["NImgs"].to_numpy()
    ring = m["RingNumber"].to_numpy().astype(int)
    snr = m["own_snr"].to_numpy()

    def sample(sel, n=8):
        w = np.where(sel & np.isfinite(snr))[0]
        if len(w) == 0:
            return []
        return list(RNG.choice(w, size=min(n, len(w)), replace=False))

    rows = [
        ("INDEXED", sample(idx), "#2b6cb0"),
        ("UNindexed\nSNR > 5", sample(~idx & (ring > 0) & (snr > 5)), "#2f855a"),
        ("UNindexed\nSNR < 5 (typical)", sample(~idx & (ring > 0) & (snr <= 5)),
         "#e8453c"),
        ("ring-0 padding", sample(ring == 0), "#777777"),
    ]
    ncol = 8
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(1.95 * ncol, 2.25 * len(rows)))
    for ax_row, (label, sel, colr) in zip(axes, rows):
        for k, ax in enumerate(ax_row):
            ax.set_xticks([]); ax.set_yticks([])
            if k >= len(sel):
                ax.axis("off"); continue
            r = m.iloc[sel[k]]
            fi = int(np.argmin(np.abs(omes - float(r["Omega"]))))
            img = read_frame(str(ZIP), fi + p.skipFrame).astype(np.float64) - dark
            row = int(round(float(r["ZRawPx"]))); col = int(round(float(r["YRawPx"])))
            br0, br1 = max(0, row - BG), min(img.shape[0], row + BG)
            bc0, bc1 = max(0, col - BG), min(img.shape[1], col + BG)
            box = img[br0:br1, bc0:bc1]
            bg = float(np.median(box))
            sd = float(np.median(np.abs(box - bg))) * 1.4826 + 1e-9
            r0, r1 = max(0, row - HALF), min(img.shape[0], row + HALF + 1)
            c0, c1 = max(0, col - HALF), min(img.shape[1], col + HALF + 1)
            ax.imshow(img[r0:r1, c0:c1], cmap="magma", vmin=bg, vmax=bg + 8 * sd)
            ax.plot(col - c0, row - r0, "o", ms=13, mfc="none", mec=colr, mew=1.6)
            ax.set_title(f"SNR {float(r['own_snr']):.0f}  N{int(r['NImgs'])}",
                         fontsize=7.5, color=colr)
            for s in ax.spines.values():
                s.set_edgecolor(colr); s.set_linewidth(1.6)
        ax_row[0].set_ylabel(label, fontsize=8.5, color=colr)
        ax_row[0].axis("on"); ax_row[0].set_xticks([]); ax_row[0].set_yticks([])

    fig.suptitle(
        "Raw dark-subtracted frames at each spot's recorded pixel — every panel "
        "scaled to its OWN local background (vmin=bg, vmax=bg+8σ)\n"
        "so a faint spot cannot be hidden by a bright neighbour, unlike the "
        "first version of this figure",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
