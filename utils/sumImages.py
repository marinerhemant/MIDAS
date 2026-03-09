#!/usr/bin/env python
"""Sum a numbered sequence of image files (TIF, etc.) into a single output image."""

import argparse
import os
import sys
import numpy as np
from PIL import Image


def build_filename(folder, stem, nr, padding, ext):
    """Construct a filename like  folder/stem00042.tif"""
    return os.path.join(folder, f"{stem}{nr:0{padding}d}.{ext}")


def sum_images(folder, stem, start, end, padding, ext):
    """Read images [start..end] and return (summed_array, count)."""
    accumulated = None
    count = 0
    for nr in range(start, end + 1):
        fname = build_filename(folder, stem, nr, padding, ext)
        if not os.path.isfile(fname):
            print(f"WARNING: {fname} not found – skipping")
            continue
        img = np.array(Image.open(fname), dtype=np.float64)
        if accumulated is None:
            accumulated = img.copy()
        else:
            if img.shape != accumulated.shape:
                print(f"ERROR: shape mismatch for {fname}: "
                      f"expected {accumulated.shape}, got {img.shape}")
                sys.exit(1)
            accumulated += img
        count += 1
    return accumulated, count


def main():
    parser = argparse.ArgumentParser(
        description="Sum a numbered sequence of images into one output file."
    )
    parser.add_argument("Folder",   help="Directory containing the images")
    parser.add_argument("FileStem", help="Common filename prefix (before the number)")
    parser.add_argument("startNr",  type=int, help="First file number (inclusive)")
    parser.add_argument("endNr",    type=int, help="Last  file number (inclusive)")
    parser.add_argument("Padding",  type=int, help="Zero-padding width for the number")
    parser.add_argument("Ext",      help="File extension without dot (e.g. tif)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output filename (default: <Folder>/<FileStem>_sum.tif)")
    args = parser.parse_args()

    total, count = sum_images(
        args.Folder, args.FileStem, args.startNr, args.endNr, args.Padding, args.Ext
    )

    if total is None:
        print("ERROR: no images were loaded.")
        sys.exit(1)

    out_path = args.output or os.path.join(
        args.Folder, f"{args.FileStem}_sum.tif"
    )

    # Choose a dtype that preserves precision
    if total.max() <= np.iinfo(np.uint16).max and total.min() >= 0 and np.all(total == total.astype(np.uint16)):
        out_img = Image.fromarray(total.astype(np.uint16))
    elif total.max() <= np.iinfo(np.uint32).max and total.min() >= 0 and np.all(total == total.astype(np.uint32)):
        out_img = Image.fromarray(total.astype(np.uint32))
    else:
        out_img = Image.fromarray(total.astype(np.float32))

    out_img.save(out_path)
    print(f"Summed {count} images → {out_path}  "
          f"(shape {total.shape}, dtype {out_img.mode}, "
          f"min={total.min():.1f}, max={total.max():.1f})")


if __name__ == "__main__":
    main()
