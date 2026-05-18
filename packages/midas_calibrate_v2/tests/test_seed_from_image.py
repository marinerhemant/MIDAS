"""Regression tests for ``seed_from_image``.

These pin three things that were broken at some point and silently shipped
wrong results:

1. **Speed**: ``_detect_arcs`` used a per-label ``labels == k`` scan that
   degenerated to hours on real images with hundreds of thousands of
   connected components.  Test that a 2880² synthetic image completes
   in well under a minute.

2. **Y/Z convention**: ``_detect_arcs`` returns coords as ``(row, col)`` =
   ``(z, y)`` in MIDAS image convention; ``circle_fit`` expects ``(y, z)``.
   The swap was missing for years, hidden whenever BC_y ≈ BC_z.  Test
   with deliberately asymmetric BC to lock the convention in place.

3. **Lsd matcher**: ``max_det_start=3`` would lock onto the first 3
   detected radii — frequently beamstop / noise on real images.  Test
   that small spurious rings before the beamstop don't poison the Lsd
   estimate.

These run in <10 s without any external data: each test draws synthetic
CeO2-like rings on a 2880×2880 grid.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest


# ============================================================ helpers

def _synth_image(NrPixelsY: int, NrPixelsZ: int,
                  BC_y: float, BC_z: float,
                  ring_radii_px,
                  *,
                  ring_intensity: float = 1000.0,
                  ring_width_px: float = 1.5,
                  spurious_radii_px=(),
                  noise: float = 50.0,
                  rng_seed: int = 0) -> np.ndarray:
    """Build a synthetic Debye-Scherrer image with the given rings.

    Returns a ``(NrPixelsZ, NrPixelsY)`` float32 array in MIDAS image
    convention (row=z, col=y).  Rings are Gaussian-profile radial.
    """
    rng = np.random.default_rng(rng_seed)
    z, y = np.meshgrid(np.arange(NrPixelsZ), np.arange(NrPixelsY), indexing="ij")
    R = np.sqrt((y - BC_y) ** 2 + (z - BC_z) ** 2)
    img = np.full((NrPixelsZ, NrPixelsY), 100.0, dtype=np.float32)
    for r0 in list(ring_radii_px) + list(spurious_radii_px):
        intensity = (ring_intensity
                     if r0 in ring_radii_px else 0.5 * ring_intensity)
        img += intensity * np.exp(-0.5 * ((R - r0) / ring_width_px) ** 2)
    img += rng.normal(0.0, noise, size=img.shape).astype(np.float32)
    return img


def _ceo2_sim_radii(Lsd_um: float, px_um: float, wavelength_A: float,
                     a_A: float = 5.4116) -> np.ndarray:
    """First few CeO2 ring radii in pixels at a given Lsd.  Uses
    :func:`midas_hkls.generate_hkls` so the extinction rules are the
    canonical Hall-symbol ones, not a hand-rolled fcc-parity switch."""
    from midas_hkls import SpaceGroup, Lattice, generate_hkls
    refs = generate_hkls(
        SpaceGroup.from_number(225),
        Lattice(a=a_A, b=a_A, c=a_A, alpha=90.0, beta=90.0, gamma=90.0),
        wavelength_A=wavelength_A, two_theta_max_deg=28.0,
    )
    out = [Lsd_um * math.tan(math.radians(r.two_theta_deg)) / px_um
           for r in refs]
    return np.array(sorted(set(round(r, 3) for r in out)))


# ============================================================ tests

def test_seed_recovers_known_geometry_in_under_30s():
    """Round-trip: build synthetic image at (BC_y, BC_z, Lsd) → seed should
    recover them, and finish in well under a minute on a 2880² image."""
    import midas_calibrate_v2.seed     # forces module preload (diplib order)
    from midas_calibrate_v2.seed import seed_from_image

    NY = NZ = 2880
    BC_y_true, BC_z_true = 1430.0, 1472.0
    Lsd_true = 840_000.0
    PX = 150.0
    LAM = 0.184139

    # First 6 CeO2 rings at the true Lsd.
    radii_true = _ceo2_sim_radii(Lsd_true, PX, LAM)[:6]
    img = _synth_image(NY, NZ, BC_y_true, BC_z_true, radii_true,
                         spurious_radii_px=[40.0, 70.0])    # beamstop-edge noise

    sim_radii = _ceo2_sim_radii(1_000_000.0, PX, LAM)        # nominal 1 m

    t0 = time.time()
    seed = seed_from_image(
        image=img, sim_radii_px=sim_radii,
        initial_lsd=1_000_000.0,
        npy=NY, npz=NZ,
        skip_median=True,                # synthetic image is already clean
    )
    elapsed = time.time() - t0

    # 1. Speed — the buggy version took 35+ minutes on 2880².
    assert elapsed < 30.0, (
        f"seed_from_image took {elapsed:.1f}s — regression of the "
        f"per-label-scan bug (should be <10s)"
    )

    # 2. BC convention: bc_y and bc_z must not be swapped.  Difference
    # between true values is 42 px, so a swap would be VERY visible.
    assert abs(seed.bc_y - BC_y_true) < 5.0, (
        f"bc_y={seed.bc_y:.2f} differs from truth {BC_y_true} by >5 px — "
        f"likely (row,col)→(y,z) swap regression"
    )
    assert abs(seed.bc_z - BC_z_true) < 5.0, (
        f"bc_z={seed.bc_z:.2f} differs from truth {BC_z_true} by >5 px — "
        f"likely (row,col)→(y,z) swap regression"
    )

    # 3. Lsd matcher: the spurious sub-beamstop rings must not lock the
    # matcher onto a wrong correspondence.
    assert abs(seed.Lsd - Lsd_true) / Lsd_true < 0.02, (
        f"Lsd={seed.Lsd:.0f} µm differs from truth {Lsd_true:.0f} by "
        f">2% — likely max_det_start regression (spurious beamstop rings "
        f"poisoning the multi-hypothesis matcher)"
    )

    # Should match at least a handful of real rings.
    assert seed.n_rings >= 3


def test_seed_bc_y_z_not_swapped_with_asymmetric_bc():
    """Targeted swap-detector: build with BC_y much smaller than BC_z, so
    a swap would be off by ~hundreds of pixels."""
    import midas_calibrate_v2.seed
    from midas_calibrate_v2.seed import seed_from_image

    NY = NZ = 2048
    BC_y_true, BC_z_true = 800.0, 1300.0     # 500-px asymmetry
    Lsd_true = 700_000.0
    PX = 200.0
    LAM = 0.189714

    radii_true = _ceo2_sim_radii(Lsd_true, PX, LAM)[:5]
    img = _synth_image(NY, NZ, BC_y_true, BC_z_true, radii_true)

    seed = seed_from_image(
        image=img, sim_radii_px=_ceo2_sim_radii(1_000_000.0, PX, LAM),
        initial_lsd=1_000_000.0, npy=NY, npz=NZ,
        skip_median=True,
    )
    # Tolerance 10 px (synthetic rings are clean; sub-px is realistic but
    # we keep slack for noise).
    assert abs(seed.bc_y - BC_y_true) < 10.0, (
        f"bc_y={seed.bc_y:.2f} (truth {BC_y_true}) — likely swapped with bc_z"
    )
    assert abs(seed.bc_z - BC_z_true) < 10.0, (
        f"bc_z={seed.bc_z:.2f} (truth {BC_z_true}) — likely swapped with bc_y"
    )


def test_seed_min_radius_filter_rejects_beamstop_noise():
    """The min_ring_radius_px filter must drop rings below the beamstop
    so the Lsd matcher doesn't pick a wrong correspondence."""
    import midas_calibrate_v2.seed
    from midas_calibrate_v2.seed import seed_from_image

    NY = NZ = 2048
    BC_y_true, BC_z_true = 1024.0, 1024.0
    Lsd_true = 940_000.0
    PX = 200.0
    LAM = 0.184139

    radii_true = _ceo2_sim_radii(Lsd_true, PX, LAM)[:6]
    # Strong spurious rings at tiny R (would totally mislead matcher with
    # max_det_start=3 and no min_ring_radius filter).
    img = _synth_image(NY, NZ, BC_y_true, BC_z_true, radii_true,
                        spurious_radii_px=[20.0, 40.0, 60.0, 80.0],
                        ring_intensity=2000.0)

    seed = seed_from_image(
        image=img, sim_radii_px=_ceo2_sim_radii(1_000_000.0, PX, LAM),
        initial_lsd=1_000_000.0, npy=NY, npz=NZ,
        skip_median=True,
        min_ring_radius_px=100.0,
    )
    # With the filter, recovered Lsd should still be close to truth.
    assert abs(seed.Lsd - Lsd_true) / Lsd_true < 0.02, (
        f"Lsd={seed.Lsd:.0f} µm differs from truth {Lsd_true:.0f} by >2% "
        f"— min_ring_radius_px filter is not working"
    )


def test_detect_arcs_scales_subminute_on_full_detector():
    """Pure-noise threshold can blow up to 100k+ connected components.  The
    bincount/find_objects rewrite of the kept-region loop must keep this
    under control."""
    import midas_calibrate_v2.seed
    from midas_calibrate_v2.seed.from_image import _detect_arcs

    NY = NZ = 2880
    # Noisy image: forces many small connected components after thresholding.
    rng = np.random.default_rng(seed=123)
    img = rng.normal(0.0, 100.0, size=(NZ, NY)).astype(np.float32)
    img = np.clip(img, 0, None)         # one-sided to drive many CCs

    t0 = time.time()
    _, _, kept = _detect_arcs(img, skip_median=True, min_area=50)
    elapsed = time.time() - t0
    assert elapsed < 60.0, (
        f"_detect_arcs took {elapsed:.1f}s on noisy 2880² — kept-region "
        f"loop regression"
    )
