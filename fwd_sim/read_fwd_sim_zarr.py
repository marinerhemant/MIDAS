#!/usr/bin/env python3
"""
Read ForwardSimulationCompressed output (Zarr v2 in zip, blosc/zstd compressed).

Output format:
  - Zip archive containing Zarr v2 store
  - Array at exchange/data/ with shape (nFrames, NrPixels, NrPixels), dtype int32
  - Chunks: (1, NrPixels, NrPixels) — one frame per chunk
  - Compression: blosc with zstd, clevel=3, bitshuffle

Usage:
  python read_fwd_sim_zarr.py <output.zip> [--frame N] [--sum] [--max] [--save out.npy]
"""

import json
import sys
import zipfile

import numpy as np
from numcodecs import Blosc


def read_zarr_zip(zip_path):
    """Read the full 3D array from a ForwardSimulation Zarr zip file.

    Returns:
        data: np.ndarray of shape (nFrames, NrPixels, NrPixels), dtype int32
        metadata: dict with shape, chunks, dtype info from .zarray
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        zarray = json.loads(zf.read("exchange/data/.zarray"))
        shape = tuple(zarray["shape"])
        chunks = tuple(zarray["chunks"])
        dtype = np.dtype(zarray["dtype"])

        comp_cfg = zarray.get("compressor", None)
        if comp_cfg and comp_cfg.get("id") == "blosc":
            decompressor = Blosc(
                cname=comp_cfg.get("cname", "zstd"),
                clevel=comp_cfg.get("clevel", 3),
                shuffle=comp_cfg.get("shuffle", 2),
            )
        else:
            decompressor = None

        nframes = shape[0]
        data = np.zeros(shape, dtype=dtype)

        for i in range(nframes):
            chunk_key = f"exchange/data/{i}.0.0"
            try:
                raw = zf.read(chunk_key)
            except KeyError:
                continue
            if decompressor is not None:
                buf = decompressor.decode(raw)
                frame = np.frombuffer(buf, dtype=dtype).reshape(chunks[1], chunks[2])
            else:
                frame = np.frombuffer(raw, dtype=dtype).reshape(chunks[1], chunks[2])
            data[i] = frame

    return data, zarray


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Read ForwardSimulationCompressed Zarr output"
    )
    parser.add_argument("zip_path", help="Path to the output .zip file")
    parser.add_argument(
        "--frame", type=int, default=None, help="Display a single frame index"
    )
    parser.add_argument(
        "--sum", action="store_true", help="Display sum projection over all frames"
    )
    parser.add_argument(
        "--max", action="store_true", help="Display max projection over all frames"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save full array to .npy file"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plotting (just print info)"
    )
    args = parser.parse_args()

    print(f"Reading: {args.zip_path}")
    data, meta = read_zarr_zip(args.zip_path)
    print(f"Shape: {data.shape}, dtype: {data.dtype}")
    print(f"Value range: [{data.min()}, {data.max()}]")
    print(f"Non-zero pixels: {np.count_nonzero(data)} / {data.size}")
    nframes = data.shape[0]
    nonzero_frames = np.count_nonzero(np.any(data != 0, axis=(1, 2)))
    print(f"Frames with signal: {nonzero_frames} / {nframes}")

    if args.save:
        np.save(args.save, data)
        print(f"Saved to: {args.save}")

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    if args.frame is not None:
        img = data[args.frame].astype(float)
        title = f"Frame {args.frame}"
    elif args.sum:
        img = data.sum(axis=0).astype(float)
        title = "Sum projection"
    elif args.max:
        img = data.max(axis=0).astype(float)
        title = "Max projection"
    else:
        img = data.max(axis=0).astype(float)
        title = "Max projection (default)"

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(img, origin="lower", cmap="gray_r")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    out_png = args.zip_path.replace(".zip", f"_{title.replace(' ', '_').lower()}.png")
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
