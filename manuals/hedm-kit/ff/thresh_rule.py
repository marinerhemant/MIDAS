#!/usr/bin/env python3
"""Evaluate the preproc-threshold rule on background-subtracted frames (frames only, no peak search).

Usage:  python thresh_rule.py <data.vrx.h5> <bgdark.h5> <out_prefix> [frames=367,727,1087] [occupancy=0.005]

The zipper keeps a pixel iff frame >= dark_mean + thresh. For each grid value we report the fraction of pixels kept
(mean over the test frames, which must NOT be among the frames the median was built from) and pick the smallest grid
value with fraction <= occupancy; NONE if there is none (then STOP and ask). Also writes a figure: the residual
frame, the median image, and a full-resolution residual crop on the strongest ring, so you can see whether spots survive.
"""
import json
import sys

import h5py
import numpy as np

GRID = [10, 20, 30, 50, 75, 100, 150, 200, 300]
INFO = [400, 500, 750, 1000, 2000]          # reported for information only, never selected


def bmax(a, b=4):
    n = a.shape[0] // b * b
    return a[:n, :n].reshape(n // b, b, n // b, b).max((1, 3))


def main():
    data_fn, bg_fn, pre = sys.argv[1:4]
    frames = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [367, 727, 1087]
    occ = float(sys.argv[5]) if len(sys.argv) > 5 else 0.005
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with h5py.File(bg_fn, "r") as g:
        used = set(int(i) for i in g["exchange/dark"].attrs["frame_indices"])
        P = np.asarray(g["exchange/dark"][0], dtype=np.float64)
    assert not (set(frames) & used), "test frames must be off the median lattice"
    with h5py.File(data_fn, "r") as f:
        R = [np.asarray(f["exchange/data"][i], dtype=np.float64) - P for i in frames]
    out = {"data": data_fn, "bgdark": bg_fn, "test_frames": frames, "occupancy_target": occ}
    out["frac_kept"] = {str(t): float(np.mean([(r >= t).mean() for r in R])) for t in GRID + INFO}
    sel = [t for t in GRID if out["frac_kept"][str(t)] <= occ]
    out["selected_threshold"] = sel[0] if sel else "NONE"
    r0 = R[len(R) // 2]
    out["resid_median"] = float(np.median(r0))
    out["resid_mean"] = float(r0.mean())
    out["resid_sigma_MAD"] = float(1.4826 * np.median(np.abs(r0 - np.median(r0))))
    out["resid_p99"], out["resid_p99.9"], out["resid_max"] = (float(np.percentile(r0, 99)), float(np.percentile(r0, 99.9)), float(r0.max()))
    out["P_mean"] = float(P.mean())
    json.dump(out, open(pre + "_thresh.json", "w"), indent=1)
    print(f"selected threshold: {out['selected_threshold']}")
    print("frac kept:", {k: round(v, 5) for k, v in out["frac_kept"].items()})
    print(f"resid median {out['resid_median']:.1f} mean {out['resid_mean']:.1f} sigmaMAD {out['resid_sigma_MAD']:.1f} p99 {out['resid_p99']:.0f} p99.9 {out['resid_p99.9']:.0f} max {out['resid_max']:.0f}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 12))
    ax[0, 0].imshow(np.log10(1 + np.clip(bmax(r0), 0, None)), cmap="viridis")
    ax[0, 0].set_title(f"frame {frames[len(R)//2]} minus pseudo-dark (log10, 4x block-max)")
    ax[0, 1].imshow(np.log10(1 + bmax(P)), cmap="viridis")
    ax[0, 1].set_title("pseudo-dark P (temporal median)")
    r = np.hypot(*(np.indices(P.shape) - np.array([1450.9, 1467.4])[:, None, None]))
    mean_r = np.bincount(r.astype(int).ravel(), weights=P.ravel()) / np.maximum(np.bincount(r.astype(int).ravel()), 1)
    p0 = int(np.argmax(mean_r[200:1300])) + 200
    cy, cz = 1451, int(1467 + p0)
    sl = (slice(cy - 80, cy + 80), slice(cz - 80, cz + 80))
    v = np.percentile(r0[sl], 99.8)
    ax[1, 0].imshow(r0[sl], vmin=0, vmax=max(v, 1), cmap="gray")
    ax[1, 0].set_title(f"residual, full-res crop on ring r~{p0}px (clip 0..p99.8)")
    ax[1, 1].hist(np.clip(r0.ravel(), -300, 1500), bins=200, log=True)
    ax[1, 1].set_title("residual pixel values (clipped -300..1500)")
    fig.tight_layout()
    fig.savefig(pre + "_resid.png", dpi=70)


if __name__ == "__main__":
    main()
