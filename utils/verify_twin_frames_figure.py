"""Figure: the Sigma3 twin's OWN spots, on the raw frames.

Companion to twin_vs_frames.py, which established:
  * pixel convention img[z,y] with (+z, -y) — median SNR 1469 vs ~1.0 for all
    seven alternatives, so the mapping is determined, not fitted;
  * twin-only spots 30/30 at SNR>5, median 1718, min 351.

This renders the evidence: raw (dark-subtracted) patches centred on the
PREDICTED pixel for spots that belong to the twin alone, alongside the
parent's own spots for comparison. If the twin were an artifact of reflections
borrowed from the parent, its exclusive spots would be empty boxes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/au3_cubes_ff_000008")
ZIP = R / "results/LayerNr_1/Au3_cubes_ff_000008.MIDAS.zip"
CDIR = R / "c_ff_fmt"
OUT = R / "twin_vs_frames.png"
PARENT, TWIN = 1000, 1177
BCY, BCZ, PX = 1018.718310, 1076.544304, 200.0
HALF = 14


def spot_sets():
    out = {}
    for ln in (CDIR / "SpotMatrix.csv").read_text().splitlines():
        if ln.startswith("%") or not ln.strip():
            continue
        f = ln.split("\t")
        out.setdefault(int(float(f[0])), set()).add(int(float(f[1])))
    return out


def to_px(y_um, z_um):
    """Calibrated convention: image[row=z, col=y], +z and -y."""
    return int(round(BCZ + z_um / PX)), int(round(BCY - y_um / PX))


def main():
    from midas_peakfit.preprocess import prepare_dark
    from midas_peakfit.zarr_io import (
        frame_omega, load_corrections, parse_zarr_params, read_frame,
    )

    p = parse_zarr_params(str(ZIP))
    load_corrections(str(ZIP), p)
    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    omes = np.array([frame_omega(p, i + p.skipFrame) for i in range(p.nFrames)])

    ei = np.fromfile(CDIR / "ExtraInfo.bin", dtype=np.float64).reshape(-1, 16)
    by_id = {int(r[4]): r for r in ei}

    sm = spot_sets()
    only_twin = sorted(sm[TWIN] - sm[PARENT])
    only_par = sorted(sm[PARENT] - sm[TWIN])

    def patch(sid):
        r = by_id[sid]
        ome = float(r[8])
        fi = int(np.argmin(np.abs(omes - ome)))
        img = read_frame(str(ZIP), fi + p.skipFrame).astype(np.float64) - dark
        row, col = to_px(float(r[9]), float(r[10]))
        r0, r1 = max(0, row - HALF), min(img.shape[0], row + HALF + 1)
        c0, c1 = max(0, col - HALF), min(img.shape[1], col + HALF + 1)
        return img[r0:r1, c0:c1], ome, fi, row - r0, col - c0

    ncol = 8
    sel_t = only_twin[:: max(1, len(only_twin) // ncol)][:ncol]
    sel_p = only_par[:: max(1, len(only_par) // ncol)][:ncol]

    fig, axes = plt.subplots(2, ncol, figsize=(2.05 * ncol, 5.0))
    for ax_row, sids, name, colr in (
        (axes[0], sel_t, f"TWIN {TWIN} only", "#e8453c"),
        (axes[1], sel_p, f"PARENT {PARENT} only", "#2b6cb0"),
    ):
        for ax, sid in zip(ax_row, sids):
            pa, ome, fi, ry, rx = patch(sid)
            v = np.percentile(pa, 99.5)
            ax.imshow(pa, cmap="magma", vmin=0, vmax=max(v, 1))
            ax.plot(rx, ry, marker="o", ms=15, mfc="none", mec=colr, mew=1.8)
            ax.set_title(f"#{sid}  ω={ome:+.1f}°", fontsize=8, color=colr)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(colr); s.set_linewidth(1.6)

    fig.suptitle(
        "Σ3 twin verified against the raw frames — dark-subtracted patches at the "
        "PREDICTED pixel\n"
        f"twin {TWIN} has {len(only_twin)} spots of its own (not shared with "
        f"parent {PARENT}); sampled 30/30 at SNR>5, median SNR 1718",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
