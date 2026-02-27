#!/usr/bin/env python
"""Generate a detector mask TIFF from an image and intensity values.

Reads an input image (e.g., a dark frame TIFF), identifies all pixels whose
intensity matches any of the given values (e.g., -1, -2 for gap/bad pixels),
and writes a mask TIFF with 1 at those positions and 0 elsewhere.

Usage:
    python generate_mask.py <image> <intensity1> [intensity2 ...] [-o output.tif]

Example:
    python generate_mask.py dark.tif -1 -2 -o mask.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def generate_mask(image_path, mask_intensities, output_path=None):
    """Generate a mask TIFF from an image and intensity values.

    Args:
        image_path: Path to input image (TIFF, etc.)
        mask_intensities: List of intensity values to mask (e.g., [-1, -2])
        output_path: Output mask TIFF path (default: <input>_mask.tif)

    Returns:
        Path to the generated mask TIFF
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img = np.array(Image.open(image_path))
    print(f"Input image: {image_path} (shape={img.shape}, dtype={img.dtype})")

    # Build mask: 1 where pixel matches ANY of the given intensities, 0 elsewhere.
    mask = np.zeros(img.shape, dtype=np.uint8)
    for val in mask_intensities:
        matches = (img == val)
        n = np.sum(matches)
        mask[matches] = 1
        print(f"  intensity={val}: {n} pixels masked")

    total_masked = np.sum(mask)
    total_pixels = mask.size
    print(f"Total masked: {total_masked}/{total_pixels} ({100*total_masked/total_pixels:.2f}%)")

    # Determine output path
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_mask.tif"
    else:
        output_path = Path(output_path)

    # Save as uint8 TIFF
    Image.fromarray(mask).save(str(output_path))
    print(f"Mask written to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a detector mask TIFF from an image and intensity values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python generate_mask.py dark.tif -1 -2 -o mask.tif",
    )
    parser.add_argument("image", help="Input image file (TIFF, etc.)")
    parser.add_argument(
        "intensities",
        nargs="+",
        type=float,
        help="Intensity values to mask (pixels matching these become 1)",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output mask TIFF path (default: <input>_mask.tif)"
    )

    args = parser.parse_args()
    generate_mask(args.image, args.intensities, args.output)


if __name__ == "__main__":
    main()
