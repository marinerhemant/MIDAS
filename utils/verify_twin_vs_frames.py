"""Reconcile the Sigma3 twin claim against the RAW FRAMES.

The claim so far is orientation-only: C grains {1000,1014,918,899} and
{1177,1176} sit ~60 deg apart about <111>. That is a statement about a model.
Before it counts it needs independent evidence in the measured data.

The circular version of this check is "do the matched spots have intensity" —
of course they do, a peak finder put them there. The honest version asks
whether the TWIN has spots that are ITS OWN:

    grain 1000 (parent): 112 spots
    grain 1177 (twin):   116 spots
    shared:               43     -> 73 belong to the twin alone

If those land on real intensity the twin is not an artifact of reflections
borrowed from the parent — the failure mode that killed the Zn/Cu epitaxy claim
in the Fuller Laue campaign, where a phase scored on peaks owned by the
substrate.

The detector convention (which of y/z is the image row, and the signs) is NOT
assumed: it is calibrated on spots already known to be real, and the run is
abandoned if no convention recovers them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

R = Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/au3_cubes_ff_000008")
ZIP = R / "results/LayerNr_1/Au3_cubes_ff_000008.MIDAS.zip"
CDIR = R / "c_ff_fmt"
PARENT, TWIN = 1000, 1177
HALF = 6                 # signal box half-width, px
BOX = 60                 # local-background box half-width, px
BCY, BCZ, PX = 1018.718310, 1076.544304, 200.0     # from the CeO2 calibration


def spot_sets():
    out = {}
    for ln in (CDIR / "SpotMatrix.csv").read_text().splitlines():
        if ln.startswith("%") or not ln.strip():
            continue
        f = ln.split("\t")
        out.setdefault(int(float(f[0])), set()).add(int(float(f[1])))
    return out


CONVENTIONS = {                       # name -> (row_from, col_from, srow, scol)
    "img[z,y] ++": ("z", "y", +1, +1),
    "img[z,y] +-": ("z", "y", +1, -1),
    "img[z,y] -+": ("z", "y", -1, +1),
    "img[z,y] --": ("z", "y", -1, -1),
    "img[y,z] ++": ("y", "z", +1, +1),
    "img[y,z] +-": ("y", "z", +1, -1),
    "img[y,z] -+": ("y", "z", -1, +1),
    "img[y,z] --": ("y", "z", -1, -1),
}


def to_px(conv, y_um, z_um):
    rf, cf, sr, sc = CONVENTIONS[conv]
    val = {"y": (y_um, BCY), "z": (z_um, BCZ)}
    r_um, r_bc = val[rf]
    c_um, c_bc = val[cf]
    return int(round(r_bc + sr * r_um / PX)), int(round(c_bc + sc * c_um / PX))


def snr_at(img, row, col):
    if not (0 <= row < img.shape[0] and 0 <= col < img.shape[1]):
        return None
    r0, r1 = max(0, row - HALF), min(img.shape[0], row + HALF + 1)
    c0, c1 = max(0, col - HALF), min(img.shape[1], col + HALF + 1)
    br0, br1 = max(0, row - BOX), min(img.shape[0], row + BOX)
    bc0, bc1 = max(0, col - BOX), min(img.shape[1], col + BOX)
    ring = img[br0:br1, bc0:bc1]
    bg = float(np.median(ring))
    sig = float(np.median(np.abs(ring - bg))) * 1.4826 + 1e-9
    return (float(img[r0:r1, c0:c1].max()) - bg) / sig


def main():
    from midas_peakfit.preprocess import prepare_dark
    from midas_peakfit.zarr_io import (
        frame_omega, load_corrections, parse_zarr_params, read_frame,
    )

    p = parse_zarr_params(str(ZIP))
    load_corrections(str(ZIP), p)
    dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    print(f"zarr nFrames={p.nFrames} skipFrame={p.skipFrame} "
          f"TransOpt={p.TransOpt}  dark mean={float(np.mean(dark)):.1f}")

    ei = np.fromfile(CDIR / "ExtraInfo.bin", dtype=np.float64).reshape(-1, 16)
    by_id = {int(r[4]): r for r in ei}       # 8=omega_ini 9=y_orig 10=z_orig
    omes = np.array([frame_omega(p, i + p.skipFrame) for i in range(p.nFrames)])

    def frame_img(ome):
        fi = int(np.argmin(np.abs(omes - ome)))
        raw = read_frame(str(ZIP), fi + p.skipFrame).astype(np.float64)
        return fi, raw - dark

    sm = spot_sets()
    shared = sorted(sm[PARENT] & sm[TWIN])
    only_twin = sorted(sm[TWIN] - sm[PARENT])
    only_par = sorted(sm[PARENT] - sm[TWIN])
    print(f"parent {PARENT}: {len(sm[PARENT])}  twin {TWIN}: {len(sm[TWIN])}  "
          f"shared: {len(shared)}  twin-only: {len(only_twin)}  "
          f"parent-only: {len(only_par)}")

    # --- calibrate the convention on KNOWN-REAL (matched) spots -------------
    cal = (shared + only_par)[:12]
    cache = {}
    for sid in cal:
        r = by_id[sid]
        cache[sid] = frame_img(float(r[8]))[1]
    print("\nconvention calibration on 12 known-real spots (median SNR):")
    best, best_med = None, -1e9
    for conv in CONVENTIONS:
        vals = []
        for sid in cal:
            r = by_id[sid]
            row, col = to_px(conv, float(r[9]), float(r[10]))
            s = snr_at(cache[sid], row, col)
            if s is not None:
                vals.append(s)
        med = float(np.median(vals)) if vals else -1e9
        print(f"  {conv:14s} n={len(vals):3d}  median SNR {med:9.1f}")
        if med > best_med:
            best, best_med = conv, med
    print(f"  -> using {best!r} (median SNR {best_med:.1f})")
    if best_med < 5:
        print("  ABORT: no convention recovers known-real spots; the "
              "pixel mapping is wrong, so any twin verdict would be garbage.")
        return

    # --- now the actual question -------------------------------------------
    for label, sids in (("TWIN-ONLY   ", only_twin),
                        ("PARENT-ONLY ", only_par),
                        ("SHARED      ", shared)):
        step = max(1, len(sids) // 30)
        sample = sids[::step][:30]
        vals = []
        for sid in sample:
            r = by_id[sid]
            _, img = frame_img(float(r[8]))
            row, col = to_px(best, float(r[9]), float(r[10]))
            s = snr_at(img, row, col)
            if s is not None:
                vals.append(s)
        v = np.array(vals)
        print(f"\n{label} n={len(v):3d}  SNR>5: {int((v > 5).sum())}/{len(v)}  "
              f"median {np.median(v):8.1f}  min {v.min():7.1f}  "
              f"max {v.max():8.1f}")


if __name__ == "__main__":
    main()
