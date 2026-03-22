#!/usr/bin/env python3
"""
Paper 3 Validation: Synthetic Calibration Test Matrix
=====================================================
Tests the auto_guess_tilted() function on synthetic images with known
ground truth geometry. This validates the direct geometry fit approach
for tilted-detector auto-calibration.

Test Matrix:
  - untilted / tilted (5°, 8°)
  - no gaps / with panel gaps (3×4 Pilatus-like, 17px gaps)

Each test generates a synthetic CeO₂ diffraction image, extracts arc
pixels, and runs auto_guess_tilted() with an intentionally offset
initial guess (BC+15px, Lsd+3%) to test recovery accuracy.

Usage:
    python validate_calibration.py

Expected results:
    - BC recovery: <0.5 px from 15px offset
    - Tilt recovery: <0.1° from 0° initial guess
    - Gap-insensitive: gaps don't degrade accuracy

Coordinate convention:
    - Synthetic image after .T is img[z_row, y_col] (TIFF layout)
    - MIDAS uses (y_px, z_px) where y_px is the Y pixel index
    - skimage regionprops returns (row, col) = (z_idx, y_idx)
    - We swap to (y_px, z_px) before calling auto_guess_tilted
"""

import sys
import os

# Ensure MIDAS utils are importable
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

import numpy as np
from synthetic_tilt_experiment import (
    generate_synthetic_image,
    generate_panel_gap_mask,
    compute_ring_radii,
    NPY, NPZ, PX_UM, LSD_UM, BC_Y, BC_Z, WAVELENGTH_A, A0,
)
from AutoCalibrateZarr import auto_guess_tilted, _pixel_to_R

from skimage import measure


def extract_arc_coords(img, threshold=100):
    """Extract arc pixel coordinates from a synthetic image.

    The image is in TIFF layout (z_row, y_col). Returns coordinates
    as (y_px, z_px) in MIDAS convention.

    Parameters
    ----------
    img : ndarray (nz, ny)
        Synthetic image in TIFF layout.
    threshold : float
        Intensity above background to consider as arc pixel.

    Returns
    -------
    coords : ndarray (N, 2)
        Arc pixel coordinates as (y_px, z_px).
    """
    bg = np.median(img)
    binary = (img > bg + threshold).astype(np.uint8)
    labels, n = measure.label(binary, return_num=True)
    props = measure.regionprops(labels)
    coords = []
    for rp in props:
        if rp.area >= 300:
            # rp.coords = (z_row, y_col); swap to (y_col, z_row) = (y_px, z_px)
            swapped = rp.coords[:, ::-1]
            coords.append(swapped)
    if coords:
        return np.vstack(coords)
    return np.zeros((0, 2))


def test_auto_guess_tilted(ty_true, tz_true, add_gaps=False, tag='test'):
    """Run a single test of auto_guess_tilted with known ground truth.

    Parameters
    ----------
    ty_true, tz_true : float
        True tilt angles in degrees.
    add_gaps : bool
        Whether to apply Pilatus-like panel gaps.
    tag : str
        Test case identifier for output.

    Returns
    -------
    result : dict or None
        Error metrics, or None if test failed.
    """
    print(f"\n{'='*70}")
    print(f"Test: ty={ty_true}°, tz={tz_true}°, gaps={add_gaps}  [{tag}]")
    print(f"{'='*70}")

    # Generate image (returns img[z, y] after transpose)
    img, ring_radii = generate_synthetic_image(ty_true, tz_true)
    R_rings = np.array([r for r, _ in ring_radii])

    # Apply gaps if needed
    if add_gaps:
        mask = generate_panel_gap_mask(ny=NPY, nz=NPZ,
                                        n_panels_y=3, n_panels_z=4,
                                        gap_y=17, gap_z=17)
        img[mask == 1] = 0
        n_masked = np.sum(mask == 1)
        print(f"  Applied gap mask: {n_masked} pixels masked "
              f"({100*n_masked/(NPY*NPZ):.1f}%)")

    # Extract arc coordinates as (y_px, z_px)
    arc_coords = extract_arc_coords(img)
    print(f"  Extracted {len(arc_coords)} arc pixels")

    if len(arc_coords) == 0:
        print("  ERROR: No arc pixels found!")
        return None

    # Initial BC guess with deliberate error (15px Y offset, -10px Z offset)
    bc_init = (BC_Y + 15, BC_Z - 10)
    lsd_init = LSD_UM * 1.03  # 3% Lsd error

    print(f"  Initial guess: BC=({bc_init[0]:.1f}, {bc_init[1]:.1f}), "
          f"Lsd={lsd_init:.0f}")
    print(f"  True values:   BC=({BC_Y:.1f}, {BC_Z:.1f}), "
          f"Lsd={LSD_UM:.0f}, ty={ty_true}°, tz={tz_true}°")

    result = auto_guess_tilted(
        arc_coords, R_rings,
        bc_init=bc_init,
        lsd_init=lsd_init,
        px=PX_UM,
        max_tilt=20.0
    )

    # Compute errors
    err_bc_y = result['bc_y'] - BC_Y
    err_bc_z = result['bc_z'] - BC_Z
    err_lsd = (result['lsd'] - LSD_UM) / LSD_UM * 100
    err_ty = result['ty'] - ty_true
    err_tz = result['tz'] - tz_true

    print(f"\n  Results:")
    print(f"  {'Param':>8s}  {'True':>10s}  {'Fitted':>10s}  {'Error':>10s}")
    print(f"  {'-'*45}")
    print(f"  {'BC_Y':>8s}  {BC_Y:>10.2f}  {result['bc_y']:>10.2f}  "
          f"{err_bc_y:>+10.2f} px")
    print(f"  {'BC_Z':>8s}  {BC_Z:>10.2f}  {result['bc_z']:>10.2f}  "
          f"{err_bc_z:>+10.2f} px")
    print(f"  {'Lsd':>8s}  {LSD_UM:>10.0f}  {result['lsd']:>10.0f}  "
          f"{err_lsd:>+10.3f} %")
    print(f"  {'ty':>8s}  {ty_true:>10.3f}  {result['ty']:>10.3f}  "
          f"{err_ty:>+10.3f} °")
    print(f"  {'tz':>8s}  {tz_true:>10.3f}  {result['tz']:>10.3f}  "
          f"{err_tz:>+10.3f} °")
    print(f"  Residual: {result['residual']:.6f} px²")

    return {
        'tag': tag,
        'ty_true': ty_true, 'tz_true': tz_true,
        'add_gaps': add_gaps,
        'err_bc_y': err_bc_y, 'err_bc_z': err_bc_z,
        'err_lsd_pct': err_lsd,
        'err_ty': err_ty, 'err_tz': err_tz,
        'residual': result['residual'],
    }


def main():
    """Run the full 2×2 validation matrix."""
    print("Paper 3 Calibration Validation: auto_guess_tilted()")
    print("=" * 70)

    test_cases = [
        (0.0, 0.0, False, 'untilted_no_gaps'),
        (0.0, 0.0, True,  'untilted_with_gaps'),
        (5.0, 8.0, False, 'tilted_no_gaps'),
        (5.0, 8.0, True,  'tilted_with_gaps'),
    ]

    results = []
    for ty, tz, gaps, tag in test_cases:
        r = test_auto_guess_tilted(ty, tz, add_gaps=gaps, tag=tag)
        if r:
            results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Tag':>25s}  {'ΔBC_Y':>8s}  {'ΔBC_Z':>8s}  {'ΔLsd%':>8s}  "
          f"{'Δty°':>8s}  {'Δtz°':>8s}  {'Resid':>8s}")
    print(f"  {'-'*76}")
    for r in results:
        print(f"  {r['tag']:>25s}  {r['err_bc_y']:>+8.2f}  {r['err_bc_z']:>+8.2f}  "
              f"{r['err_lsd_pct']:>+8.3f}  {r['err_ty']:>+8.3f}  "
              f"{r['err_tz']:>+8.3f}  {r['residual']:>8.4f}")

    # Check pass/fail criteria
    all_pass = True
    for r in results:
        if abs(r['err_bc_y']) > 1.0 or abs(r['err_bc_z']) > 1.0:
            print(f"  FAIL: {r['tag']} — BC error > 1.0 px")
            all_pass = False
        if abs(r['err_ty']) > 0.5 or abs(r['err_tz']) > 0.5:
            print(f"  FAIL: {r['tag']} — tilt error > 0.5°")
            all_pass = False

    if all_pass:
        print("\n  ✅ ALL TESTS PASSED")
    else:
        print("\n  ❌ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
