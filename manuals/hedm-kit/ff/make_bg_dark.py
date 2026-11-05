#!/usr/bin/env python3
"""Pre-step B: build a temporal-median pseudo-dark for a dense-ring sweep.

Usage:  python make_bg_dark.py <data.vrx.h5> <out_bgdark.h5> [K=120] [stride=12]

Why: on composite samples the rings are continuous bands plus a background plateau of ~200-300 counts, so no
zipper threshold (it keeps pixels >= dark_mean + thresh) isolates spots. The per-pixel median over the sweep
removes everything that is static in omega and keeps spots that last a few frames.
Output: <out> with exchange/dark = 2 identical float32 frames (the zipper averages dark_frames[SkipFrame:], and
falls back to frame 0), so set in Parameters.txt:  Dark <out>   darkLoc /exchange/dark
Frames used: 1 + stride*j, j = 0..K-1 (frame 0 is the throwaway). Test frames for the threshold rule are
chosen OFF this lattice (index = 7 mod 12).
"""
import datetime
import os
import sys

import h5py
import numpy as np


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    with h5py.File(src, "r") as f:
        d = f["exchange/data"]
        n = d.shape[0]
        idx = [1 + stride * j for j in range(K)]
        assert idx[-1] < n, f"K*stride too large for {n} frames"
        stack = np.empty((K,) + d.shape[1:], dtype=d.dtype)
        for j, i in enumerate(idx):
            stack[j] = d[i]
    P = np.median(stack, axis=0).astype(np.float32)
    del stack
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with h5py.File(out, "w") as g:
        ds = g.create_dataset("exchange/dark", data=np.stack([P, P]), compression="gzip", compression_opts=1)
        ds.attrs["source"] = os.path.abspath(src)
        ds.attrs["frame_indices"] = np.asarray(idx)
        ds.attrs["method"] = "per-pixel median over frame_indices of exchange/data (raw)"
        ds.attrs["made"] = datetime.datetime.now().isoformat(timespec="seconds")
        ds.attrs["script"] = "hedm-kit/ff/make_bg_dark.py"
    print(f"{out}: median of {K} frames of {os.path.basename(src)}; P mean {P.mean():.1f} median {np.median(P):.1f} max {P.max():.0f}")


if __name__ == "__main__":
    main()
