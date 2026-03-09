#!/usr/bin/env python
"""Sum a numbered sequence of image files (TIF, etc.) into a single output image.

Supports parallel loading via multiprocessing (-j flag).
"""

import argparse
import os
import sys
import multiprocessing
import numpy as np
from PIL import Image


def build_filename(folder, stem, nr, padding, ext):
    """Construct a filename like  folder/stem00042.tif"""
    return os.path.join(folder, f"{stem}{nr:0{padding}d}.{ext}")


def _load_one(fname):
    """Load a single image file and return (array, fname) or (None, fname)."""
    if not os.path.isfile(fname):
        return None, fname
    return np.array(Image.open(fname), dtype=np.float64), fname


def sum_images(folder, stem, start, end, padding, ext, jobs=1):
    """Read images [start..end] and return (summed_array, count).

    Parameters
    ----------
    jobs : int
        Number of parallel workers for file I/O.  1 = serial.
    """
    fnames = [build_filename(folder, stem, nr, padding, ext)
              for nr in range(start, end + 1)]

    if jobs == 1:
        results = [_load_one(f) for f in fnames]
    else:
        with multiprocessing.Pool(processes=jobs) as pool:
            results = pool.map(_load_one, fnames)

    accumulated = None
    count = 0
    for img, fname in results:
        if img is None:
            print(f"WARNING: {fname} not found – skipping")
            continue
        if accumulated is None:
            accumulated = img
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
    parser.add_argument("-j", "--jobs", type=int,
                        default=multiprocessing.cpu_count(),
                        help="Number of parallel workers (default: all CPUs)")
    args = parser.parse_args()

    total, count = sum_images(
        args.Folder, args.FileStem, args.startNr, args.endNr,
        args.Padding, args.Ext, jobs=args.jobs
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
    print(f"Summed {count} images ({args.jobs} workers) → {out_path}  "
          f"(shape {total.shape}, dtype {out_img.mode}, "
          f"min={total.min():.1f}, max={total.max():.1f})")


if __name__ == "__main__":
    main()

