#!/usr/bin/env python3
"""Look at the raw frames of one Varex sweep BEFORE any threshold or model (ff-hedm RUNBOOK 5a, rule 13).

Usage:  python look_frames.py <data.vrx.h5> <dark_before.vrx.h5> <out_prefix> [bc_y bc_z]

What it writes
  <out_prefix>_frames.png   one frame, the median of ~9 spread frames (static structure), the dark, histogram
  <out_prefix>_radial.png   radial profile of the median frame about the BC guess (rings show as bumps)
  <out_prefix>_stats.json   numbers: dark mean/max, pedestal after dark subtraction, fraction of pixels above each
                            candidate preproc threshold (the zipper zeroes pixels below dark_mean + thresh),
                            local-background (51 px median) and the spot excess above it.
The dark is taken as the zipper does: mean of the dark frames after skipping the first.
Frame 0 is skipped (SkipFrame 1). Block-max downsampling keeps single-pixel spots visible.
"""
import json
import sys

import h5py
import numpy as np
from scipy.ndimage import median_filter

THR = [10, 20, 30, 50, 75, 100, 150, 200, 300]


def dark_mean(path):
    with h5py.File(path, "r") as f:
        for k in ("bright", "dark", "data"):
            d = f[f"exchange/{k}"]
            if d.shape[0] > 1 and float(np.asarray(d[min(2, d.shape[0] - 1)]).mean()) > 0:
                return np.asarray(d[1:], dtype=np.float64).mean(0), k, d.shape[0]
        d = f["exchange/bright"]
        return np.asarray(d[:], dtype=np.float64).mean(0), "bright(all-zero?)", d.shape[0]


def bmax(a, b=4):
    n = a.shape[0] // b * b
    return a[:n, :n].reshape(n // b, b, n // b, b).max((1, 3))


def main():
    data_fn, dark_fn, pre = sys.argv[1:4]
    bcy, bcz = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (1450.9, 1467.4)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dm, dloc, ndark = dark_mean(dark_fn)
    out = {"data": data_fn, "dark_file": dark_fn, "darkLoc": dloc, "n_dark_frames": ndark,
           "dark_mean": float(dm.mean()), "dark_max_pixel_mean": float(dm.max()),
           "dark_p99.9": float(np.percentile(dm, 99.9))}
    with h5py.File(data_fn, "r") as f:
        d = f["exchange/data"]
        n = d.shape[0]
        idx = sorted({int(x) for x in np.linspace(1, n - 1, 9)})
        fr = {i: np.asarray(d[i], dtype=np.float64) - dm for i in idx}
    mid = idx[len(idx) // 2]
    med = np.median(np.stack([fr[i] for i in idx]), 0)
    out["frames_used"] = idx
    for tag, im in (("frame_mid", fr[mid]), ("median9", med)):
        out[f"{tag}_median"] = float(np.median(im))
        out[f"{tag}_mean"] = float(im.mean())
        out[f"{tag}_p99"] = float(np.percentile(im, 99))
        out[f"{tag}_p99.9"] = float(np.percentile(im, 99.9))
        out[f"{tag}_max"] = float(im.max())
    # threshold occupancy, the preproc-thresh rule: fraction of pixels left non-zero by the zipper
    out["frac_above_thresh"] = {str(t): float(np.mean([(fr[i] >= t).mean() for i in idx[:3]])) for t in THR}
    # local background: 51 px median of a downsampled frame, spot excess above it
    sm = fr[mid][:2880, :2880]
    bg = np.kron(median_filter(sm[::4, ::4], size=13), np.ones((4, 4)))
    ex = sm - bg
    mad = 1.4826 * np.median(np.abs(ex - np.median(ex)))
    out["local_bg_mean"] = float(bg.mean())
    out["local_bg_p1_p99"] = [float(np.percentile(bg, 1)), float(np.percentile(bg, 99))]
    out["excess_sigma_MAD"] = float(mad)
    out["frac_excess_gt_5sigma"] = float((ex > 5 * mad).mean())
    out["frac_excess_gt_50"] = float((ex > 50).mean())

    fig, ax = plt.subplots(2, 2, figsize=(13, 12))
    for a, im, ttl in ((ax[0, 0], fr[mid], f"frame {mid} - dark (log, 4x block-max)"),
                       (ax[0, 1], med, "median of 9 frames (static background)")):
        a.imshow(np.log10(1 + np.clip(bmax(im), 0, None)), cmap="viridis", origin="upper")
        a.plot(bcz / 4, bcy / 4, "r+", ms=14)
        a.set_title(ttl)
    ax[1, 0].imshow(np.log10(1 + np.clip(bmax(dm), 0, None)), cmap="magma")
    ax[1, 0].set_title(f"dark mean (from exchange/{dloc}, {ndark} frames)")
    v = fr[mid].ravel()
    ax[1, 1].hist(np.clip(v, -50, 1500), bins=200, log=True)
    ax[1, 1].set_title("pixel values after dark subtraction (clipped -50..1500)")
    fig.tight_layout()
    fig.savefig(pre + "_frames.png", dpi=70)
    plt.close(fig)

    yy, zz = np.indices(med.shape)
    r = np.hypot(yy - bcy, zz - bcz)
    rb = r.astype(int).ravel()
    prof = np.bincount(rb, weights=med.ravel()) / np.maximum(np.bincount(rb), 1)
    fig, a = plt.subplots(figsize=(10, 4))
    a.plot(prof[:2000])
    a.set_xlabel("radius from BC guess (px)")
    a.set_ylabel("mean of median9 (counts)")
    a.set_title("radial profile, rings are bumps (BC is a guess)")
    fig.tight_layout()
    fig.savefig(pre + "_radial.png", dpi=70)
    plt.close(fig)
    json.dump(out, open(pre + "_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))}, indent=1))
    print("frac_above_thresh", out["frac_above_thresh"])


if __name__ == "__main__":
    main()
