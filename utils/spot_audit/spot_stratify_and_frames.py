"""Where does the Friedel signal live, and what do these spots look like raw?

The null said chance pairing is ~25% and the unindexed spots pair at 77.8%,
so a real physical pairing exists somewhere in that population. But it was
measured over ALL 1891 unindexed spots, most of which are weak single-frame
events. Two very different readings:

  (a) the pairing comes from a credible minority (multi-frame, brighter)
      and the single-frame majority pairs at chance -> the junk really is
      junk, and the credible spots are unindexed REAL diffraction;
  (b) the pairing is uniform across the whole population -> even the weak
      single-frame events are real diffraction (e.g. a fine-grained or
      powder-like component), and "noise" is the wrong word for them.

So: stratify the pairing rate by NImgs and by brightness, each against its
own shuffled null. Then LOOK AT THE RAW FRAMES -- dark-subtracted patches at
the recorded detector pixel for each stratum, which is the only thing that
settles what these events physically are.

Pixel convention: YRawPx is the COLUMN, ZRawPx is the ROW. Verified against
the independently calibrated mapping (col = BCY - y/px, row = BCZ + z/px)
from the twin verification -- both agree to <0.5 px on row 1.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/au3_cubes_ff_000008")
AUD = R / "spot_noise_audit/spot_audit.csv"
ZIP = R / "results/LayerNr_1/Au3_cubes_ff_000008.MIDAS.zip"
OUT = R / "spot_noise_audit"
RNG = np.random.default_rng(7)
ETA_TOL = OME_TOL = 1.0
HALF = 12


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def pair_hits(eta, ome, ring):
    e_t, o_t = wrap(180.0 - eta), wrap(ome + 180.0)
    hit = np.zeros(len(eta), bool)
    for r in np.unique(ring):
        sel = np.where(ring == r)[0]
        if len(sel) < 2:
            continue
        de = np.abs(wrap(e_t[sel][:, None] - eta[sel][None, :]))
        do = np.abs(wrap(o_t[sel][:, None] - ome[sel][None, :]))
        ok = (de < ETA_TOL) & (do < OME_TOL)
        np.fill_diagonal(ok, False)
        hit[sel] = ok.any(axis=1)
    return hit


def null_rate(eta, ome, ring, mask, n=12):
    out = []
    for _ in range(n):
        e2 = eta.copy()
        for r in np.unique(ring):
            sel = np.where(ring == r)[0]
            e2[sel] = RNG.permutation(e2[sel])
        out.append(pair_hits(e2, ome, ring)[mask].mean())
    return float(np.mean(out)), float(np.std(out))


def stratify(m):
    eta = m["Eta"].to_numpy()
    ome = m["Omega"].to_numpy()
    ring = m["RingNumber"].to_numpy().astype(int)
    idx = m["indexed"].to_numpy().astype(bool)
    nim = m["NImgs"].to_numpy()
    imax = m["IMax"].to_numpy()
    obs = pair_hits(eta, ome, ring)

    print("Friedel pairing by stratum (observed vs shuffled-eta null)\n")
    print(f"  {'stratum':38s} {'n':>6s} {'obs':>7s} {'null':>13s} {'excess':>8s}")
    strata = [
        ("indexed (positive control)", idx),
        ("unindexed, NImgs>=2", ~idx & (nim >= 2)),
        ("unindexed, NImgs==1", ~idx & (nim <= 1)),
        ("unindexed, NImgs==1, IMax<20", ~idx & (nim <= 1) & (imax < 20)),
        ("unindexed, NImgs>=2, IMax>=100", ~idx & (nim >= 2) & (imax >= 100)),
        ("ring-0 padding rows", ring == 0),
    ]
    for name, sel in strata:
        if sel.sum() < 5:
            print(f"  {name:38s} {int(sel.sum()):6d}   (too few)")
            continue
        o = obs[sel].mean()
        mu, sd = null_rate(eta, ome, ring, sel)
        print(f"  {name:38s} {int(sel.sum()):6d} {o:7.1%} "
              f"{mu:6.1%} +-{sd:4.1%} {o-mu:+8.1%}")
    return obs


def frames_figure(m):
    from midas_peakfit.preprocess import prepare_dark
    from midas_peakfit.zarr_io import (
        frame_omega, load_corrections, parse_zarr_params, read_frame,
    )
    p = parse_zarr_params(str(ZIP))
    load_corrections(str(ZIP), p)
    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    omes = np.array([frame_omega(p, i + p.skipFrame) for i in range(p.nFrames)])

    idx = m["indexed"].to_numpy().astype(bool)
    nim = m["NImgs"].to_numpy()
    ring = m["RingNumber"].to_numpy().astype(int)

    def pick(sel, n, by="IMax"):
        d = m[sel].sort_values(by, ascending=False)
        if len(d) == 0:
            return d
        step = max(1, len(d) // n)
        return d.iloc[::step].head(n)

    rows = [
        ("INDEXED (real)", pick(idx, 8), "#2b6cb0"),
        ("UNindexed NImgs>=2\n(brightest)", pick(~idx & (nim >= 2), 8), "#2f855a"),
        ("UNindexed NImgs==1\n(typical)", pick(~idx & (nim <= 1) & (ring > 0), 8),
         "#e8453c"),
    ]
    ncol = 8
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.0 * ncol, 2.35 * len(rows)))
    for ax_row, (label, d, colr) in zip(axes, rows):
        for k, ax in enumerate(ax_row):
            ax.set_xticks([]); ax.set_yticks([])
            if k >= len(d):
                ax.axis("off"); continue
            r = d.iloc[k]
            fi = int(np.argmin(np.abs(omes - float(r["Omega"]))))
            img = read_frame(str(ZIP), fi + p.skipFrame).astype(np.float64) - dark
            row = int(round(float(r["ZRawPx"]))); col = int(round(float(r["YRawPx"])))
            r0, r1 = max(0, row - HALF), min(img.shape[0], row + HALF + 1)
            c0, c1 = max(0, col - HALF), min(img.shape[1], col + HALF + 1)
            pa = img[r0:r1, c0:c1]
            v = np.percentile(pa, 99.5) if pa.size else 1
            ax.imshow(pa, cmap="magma", vmin=0, vmax=max(v, 1))
            ax.plot(col - c0, row - r0, "o", ms=13, mfc="none", mec=colr, mew=1.6)
            ax.set_title(f"IMax {float(r['IMax']):.0f}  N{int(r['NImgs'])}",
                         fontsize=7.5, color=colr)
            for s in ax.spines.values():
                s.set_edgecolor(colr); s.set_linewidth(1.6)
        ax_row[0].set_ylabel(label, fontsize=9, color=colr)
        ax_row[0].axis("on"); ax_row[0].set_xticks([]); ax_row[0].set_yticks([])

    fig.suptitle(
        "Raw dark-subtracted frames at each spot's recorded detector pixel "
        "(same colour scale rule, 25x25 px)\n"
        "top = indexed; middle = unindexed but multi-frame; "
        "bottom = unindexed single-frame (the bulk of the 1891)",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = OUT / "spot_strata_frames.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


def grain_sizes(m):
    """What grain size would each spot imply, IF it is real diffraction?

    calc_radius turns a spot's integrated intensity into GrainVolume by
    ratio against the ring's PowderIntensity, then GrainRadius from that.
    CAVEAT: for a noise event this number is meaningless -- it is just a
    monotone function of intensity. It answers "if real, how big?", not
    "is it real?". Read it alongside the raw frames, not instead of them.
    """
    idx = m["indexed"].to_numpy().astype(bool)
    nim = m["NImgs"].to_numpy()
    ring = m["RingNumber"].to_numpy().astype(int)
    print("\n\nImplied grain size per spot (GrainRadius, um)\n")
    print(f"  {'stratum':34s} {'n':>6s} {'p25':>8s} {'med':>8s} {'p75':>8s} "
          f"{'vol vs indexed':>16s}")
    ref = np.median(m.loc[idx, "GrainRadius"])
    for name, sel in (
        ("indexed", idx),
        ("unindexed, NImgs>=2", ~idx & (nim >= 2)),
        ("unindexed, NImgs==1 (ring>0)", ~idx & (nim <= 1) & (ring > 0)),
    ):
        d = m.loc[sel, "GrainRadius"].to_numpy()
        if len(d) < 3:
            continue
        med = np.median(d)
        vr = (med / ref) ** 3 if ref > 0 else np.nan
        print(f"  {name:34s} {len(d):6d} {np.percentile(d,25):8.2f} "
              f"{med:8.2f} {np.percentile(d,75):8.2f} {vr:15.2e}x")
    print(f"\n  indexed median radius {ref:.1f} um; a spot at the unindexed "
          f"median implies a grain")
    med_u = np.median(m.loc[~idx & (ring > 0), "GrainRadius"])
    print(f"  ~{ref/max(med_u,1e-9):.0f}x smaller in radius, "
          f"~{(ref/max(med_u,1e-9))**3:.0f}x smaller in VOLUME.")
    tot_i = m.loc[idx, "GrainVolume"].sum()
    tot_u = m.loc[~idx & (ring > 0), "GrainVolume"].sum()
    print(f"\n  summed GrainVolume (arb): indexed {tot_i:.3e}, "
          f"unindexed {tot_u:.3e}  -> unindexed carry "
          f"{tot_u/max(tot_i+tot_u,1e-9):.1%} of the implied diffracting volume")


def main():
    m = pd.read_csv(AUD)
    stratify(m)
    grain_sizes(m)
    frames_figure(m)


if __name__ == "__main__":
    main()
