#!/usr/bin/env python
"""Generate a detector mask TIFF from an image and intensity values.

Reads an input image (e.g., a dark frame TIFF), identifies all pixels whose
intensity matches any of the given values (e.g., -1, -2 for gap/bad pixels),
and writes a mask TIFF with 1 at those positions and 0 elsewhere.

Alternatively, if a .mask file is provided (which is a TIFF in disguise),
it is read directly: pixels with value 1 are masked, 0 are unmasked.

Usage:
    python generate_mask.py dark.tif -1 -2 -o mask.tif
    python generate_mask.py detector.mask -o mask.tif
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


def convert_mask_file(mask_path, output_path=None):
    """Convert a .mask file (TIFF in disguise) to a proper mask TIFF.

    In the .mask format, pixels with value 1 are masked and 0 are unmasked.
    The output is a uint8 TIFF with the same convention.

    Args:
        mask_path: Path to .mask file
        output_path: Output mask TIFF path (default: <input>_mask.tif)

    Returns:
        Path to the generated mask TIFF
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    img = np.array(Image.open(mask_path))
    print(f"Input .mask file: {mask_path} (shape={img.shape}, dtype={img.dtype})")

    # Ensure binary: anything != 0 is masked
    mask = (img != 0).astype(np.uint8)
    total_masked = np.sum(mask)
    total_pixels = mask.size
    print(f"Masked pixels: {total_masked}/{total_pixels} ({100*total_masked/total_pixels:.2f}%)")

    if output_path is None:
        output_path = mask_path.parent / f"{mask_path.stem}_mask.tif"
    else:
        output_path = Path(output_path)

    Image.fromarray(mask).save(str(output_path))
    print(f"Mask written to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a detector mask TIFF from an image and intensity values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python generate_mask.py dark.tif -1 -2 -o mask.tif\n"
               "  python generate_mask.py detector.mask -o mask.tif",
    )
    parser.add_argument("image", help="Input image file (TIFF, .mask, etc.)")
    parser.add_argument(
        "intensities",
        nargs="*",
        type=float,
        help="Intensity values to mask (not needed for .mask files)",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output mask TIFF path (default: <input>_mask.tif)"
    )

    args = parser.parse_args()

    if Path(args.image).suffix.lower() == '.mask':
        convert_mask_file(args.image, args.output)
    else:
        if not args.intensities:
            parser.error("intensity values are required for non-.mask files")
        generate_mask(args.image, args.intensities, args.output)


# MIDAS version banner
try:
    from version import version_string as _vs
    print(_vs())
except Exception:
    pass

if __name__ == "__main__":
    main()
