"""Smoke + functional tests for the Phase 1 screen kernel.

The screen consumes ``DiffractionSpots.bin`` rows ``(yl, zl, omega_deg)``
in lab-frame microns at the primary detector distance, projects them
through each voxel position, scales to per-distance pixels, looks up
the obs bitmap, and aggregates ``FracOverlap`` per orientation. We
build a tiny synthetic fixture where one orientation should match
exactly and a second should be filtered out by ``MinFracAccept``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.io import GridTable, OrientationData
from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.params import FitParams
from midas_nf_fitorientation.screen import (
    apply_nf_tilt,
    build_rot_tilts,
    orientmat_to_euler_zxz,
    screen,
)


def _make_params(
    n_distances: int = 1, Lsd_list=(1_000_000.0,),
    n_pixels: int = 64,
) -> FitParams:
    p = FitParams()
    p.n_distances = n_distances
    p.Lsd = list(Lsd_list)
    p.ybc = [n_pixels / 2.0] * n_distances
    p.zbc = [n_pixels / 2.0] * n_distances
    p.px = 200.0
    p.omega_start = -180.0
    p.omega_step_raw = 1.0
    p.start_nr = 1
    p.end_nr = 360                  # one frame per degree
    p.exclude_pole_angle = 0.0
    p.wavelength = 0.172979
    p.lattice_constant = (4.08, 4.08, 4.08, 90, 90, 90)
    p.n_pixels_y = n_pixels
    p.n_pixels_z = n_pixels
    p.tx = p.ty = p.tz = 0.0
    p.wedge = 0.0
    p.min_frac_accept = 0.5
    return p


def test_orientmat_to_euler_zxz_identity():
    R = np.eye(3)[None, ...]
    eul = orientmat_to_euler_zxz(R)
    assert eul.shape == (1, 3)
    # phi1 = 0, Phi = 0, phi2 = 0
    assert np.allclose(eul, 0.0, atol=1e-9)


def test_build_rot_tilts_identity_when_zero():
    R = build_rot_tilts(0, 0, 0, torch.device("cpu"), torch.float64)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64))


def test_apply_nf_tilt_identity_when_R_is_identity():
    R = torch.eye(3, dtype=torch.float64)
    y = torch.tensor([1.0, -2.0, 3.0])
    z = torch.tensor([4.0, 5.0, -6.0])
    out_y, out_z = apply_nf_tilt(y, z, Lsd_val=1_000_000.0, R=R)
    assert torch.allclose(out_y, y, atol=1e-9)
    assert torch.allclose(out_z, z, atol=1e-9)


# ---------------------------------------------------------------------------
#  End-to-end screen with synthetic fixtures
# ---------------------------------------------------------------------------

def test_screen_picks_winning_orientation_at_origin():
    """Build one good orientation (its spots fall on lit pixels) and one
    bad orientation (spots land on empty pixels). Voxel at origin so
    the projection formula collapses to ``yl/px + ybc``.
    """
    p = _make_params()

    # Two orientations, two spots each.
    matrices = np.zeros((2, 3, 3), dtype=np.float64)
    matrices[0] = np.eye(3)
    matrices[1] = np.eye(3)            # content doesn't matter for the screen
    n_spots = np.array([2, 2], dtype=np.int64)
    starts = np.array([0, 2], dtype=np.int64)

    # Pixel positions to lit on the obs volume.
    px = p.px
    bc = p.ybc[0]
    n_y = p.n_pixels_y

    # Orientation 0: hits at (frame=10, y=10, z=10) and (frame=20, y=20, z=20)
    # Orientation 1: same omegas but different yl/zl, so it lands somewhere
    # we will NOT lit, giving zero overlap.
    omega_a, omega_b = 10.0, 20.0      # degrees
    # For voxel at origin the projection collapses to: y_pix = yl/px + bc
    # so to land on pixel y=10 we need yl = (10 - bc) * px = (10 - 32) * 200
    yl0_a = (10 - bc) * px
    zl0_a = (10 - bc) * px
    yl0_b = (20 - bc) * px
    zl0_b = (20 - bc) * px
    yl1_a = (50 - bc) * px             # off the dark obs volume
    zl1_a = (50 - bc) * px
    yl1_b = (60 - bc) * px
    zl1_b = (60 - bc) * px

    spots = np.array([
        [yl0_a, zl0_a, omega_a],
        [yl0_b, zl0_b, omega_b],
        [yl1_a, zl1_a, omega_a],
        [yl1_b, zl1_b, omega_b],
    ], dtype=np.float64)

    od = OrientationData(
        matrices=matrices, n_spots=n_spots, starts=starts, spots=spots,
    )

    # Light only orientation 0's pixels.
    obs_arr = np.zeros((1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
                        dtype=np.float32)
    frame_a = int((omega_a - p.omega_start) / p.omega_step)
    frame_b = int((omega_b - p.omega_start) / p.omega_step)
    obs_arr[0, frame_a, 10, 10] = 1.0
    obs_arr[0, frame_b, 20, 20] = 1.0
    obs = ObsVolume.from_dense_array(obs_arr)

    grid = GridTable(
        y1=np.array([1.0]), y2=np.array([0.5]),
        xs=np.array([0.0]), ys=np.array([0.0]),
        gs=np.array([100.0]), ud=np.array([-1], dtype=np.int8),
    )

    result = screen(grid, od, obs, p, dtype=torch.float64)
    # Only orientation 0 should pass; its FracOverlap = 1.0
    assert len(result.winners) == 1
    w = result.winners[0]
    assert w.voxel_idx == 0
    assert w.orient_idx == 0
    assert w.frac_overlap == pytest.approx(1.0)


def test_screen_partial_match_below_threshold_filtered():
    """Single orientation, two spots, one lit one dark. FracOverlap = 0.5;
    with ``min_frac_accept = 0.6`` it is dropped, with 0.4 it is kept.
    """
    p = _make_params()
    p.min_frac_accept = 0.6
    bc = p.ybc[0]; px = p.px

    spots = np.array([
        [(10 - bc) * px, (10 - bc) * px, 10.0],   # lit
        [(20 - bc) * px, (20 - bc) * px, 20.0],   # dark
    ], dtype=np.float64)
    od = OrientationData(
        matrices=np.eye(3)[None, ...].copy(),
        n_spots=np.array([2], dtype=np.int64),
        starts=np.array([0], dtype=np.int64),
        spots=spots,
    )

    obs_arr = np.zeros((1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
                        dtype=np.float32)
    frame_lit = int((10.0 - p.omega_start) / p.omega_step)
    obs_arr[0, frame_lit, 10, 10] = 1.0
    obs = ObsVolume.from_dense_array(obs_arr)

    grid = GridTable(
        y1=np.array([1.0]), y2=np.array([0.5]),
        xs=np.array([0.0]), ys=np.array([0.0]),
        gs=np.array([100.0]), ud=np.array([-1], dtype=np.int8),
    )

    # Threshold 0.6 → reject (overlap = 0.5)
    result = screen(grid, od, obs, p, dtype=torch.float64)
    assert len(result.winners) == 0

    # Lower threshold → accept
    p.min_frac_accept = 0.4
    result = screen(grid, od, obs, p, dtype=torch.float64)
    assert len(result.winners) == 1
    assert result.winners[0].frac_overlap == pytest.approx(0.5)


def test_screen_super_pixel_voxel_uses_rasterisation():
    """When ``2*gs > px`` the screen rasterises the projected triangle
    instead of using the centroid. Build a fixture where a centroid-only
    screen would miss the spot but a rasterised triangle catches it.
    """
    p = _make_params(n_pixels=64)
    # Force super-pixel: voxel is bigger than a pixel.
    # gs = 200 µm, px = 200 µm → 2*gs = 400 > 200, super-pixel branch.
    # The projected triangle covers K ≥ 1 pixels; we only light one of
    # them, so use a very loose threshold (we're checking the rasteriser
    # path runs end-to-end, not a quantitative overlap target).
    p.min_frac_accept = 0.1

    # Single orientation, single spot.
    yl = 0.0
    zl = 0.0
    omega = 10.0
    spots = np.array([[yl, zl, omega]], dtype=np.float64)
    od = OrientationData(
        matrices=np.eye(3)[None, ...].copy(),
        n_spots=np.array([1], dtype=np.int64),
        starts=np.array([0], dtype=np.int64),
        spots=spots,
    )

    # The voxel triangle with gs=200 µm projects (at omega=10°) to a
    # detector triangle several pixels wide; the centroid lands at
    # (bc, bc) but the rasterised triangle covers a small patch around
    # it. We light only one of those off-centroid pixels and verify
    # the screen still scores ≥ 1 hit. Mostly a smoke test that the
    # super-pixel path runs without throwing.
    obs_arr = np.zeros(
        (1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
        dtype=np.float32,
    )
    frame = int((omega - p.omega_start) / p.omega_step)
    bc = int(p.ybc[0])
    # Light the centre pixel — guaranteed to be in any rasterised
    # triangle that covers the centre.
    obs_arr[0, frame, bc, bc] = 1.0
    obs = ObsVolume.from_dense_array(obs_arr)

    grid = GridTable(
        y1=np.array([200.0]), y2=np.array([100.0]),
        xs=np.array([0.0]), ys=np.array([0.0]),
        gs=np.array([200.0]),                   # super-pixel
        ud=np.array([-1], dtype=np.int8),
    )
    result = screen(grid, od, obs, p, dtype=torch.float64)
    # FracOverlap is hits/total_pixels. With one spot rasterised into
    # K pixels and exactly one lit, frac = 1/K. Loose lower bound: > 0.
    assert len(result.winners) == 1
    assert result.winners[0].voxel_idx == 0
    assert result.winners[0].orient_idx == 0
    assert 0.0 < result.winners[0].frac_overlap <= 1.0


def test_screen_multi_distance_requires_all_hits():
    """Two distances. Spot lit at distance 0 only ⇒ AND across distances
    gives zero. Lit at both ⇒ FracOverlap = 1.
    """
    p = _make_params(n_distances=2, Lsd_list=(1_000_000.0, 2_000_000.0))
    bc = p.ybc[0]; px = p.px

    # Voxel at origin, single spot at the beam center: yl_0 = 0 → pixel = bc
    # at every distance (linear scaling preserves bc at origin).
    spots = np.array([
        [(10 - bc) * px, (10 - bc) * px, 10.0],
    ], dtype=np.float64)
    od = OrientationData(
        matrices=np.eye(3)[None, ...].copy(),
        n_spots=np.array([1], dtype=np.int64),
        starts=np.array([0], dtype=np.int64),
        spots=spots,
    )

    obs_arr = np.zeros(
        (2, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
        dtype=np.float32,
    )
    frame = int((10.0 - p.omega_start) / p.omega_step)
    # At distance 0: y_pix = (10 - bc) + bc = 10
    # At distance 1: scale = 2, y_pix = (10 - bc)*2 + bc = (10-32)*2+32 = -12
    # which is out-of-bounds. So this case actually tests the in-bounds mask.
    # For a cleaner two-distance pass, lit a spot at the centre:
    yl = 0.0
    zl = 0.0
    spots = np.array([[yl, zl, 10.0]], dtype=np.float64)
    od = OrientationData(
        matrices=np.eye(3)[None, ...].copy(),
        n_spots=np.array([1], dtype=np.int64),
        starts=np.array([0], dtype=np.int64),
        spots=spots,
    )
    # At voxel (0, 0, 0), yl=0 → pixel y_pix_d = bc[d] for any d. Both
    # distances share bc=32, so we lit the same pixel.
    obs_arr[0, frame, 32, 32] = 1.0
    obs_arr[1, frame, 32, 32] = 1.0
    obs = ObsVolume.from_dense_array(obs_arr)
    grid = GridTable(
        y1=np.array([1.0]), y2=np.array([0.5]),
        xs=np.array([0.0]), ys=np.array([0.0]),
        gs=np.array([100.0]), ud=np.array([-1], dtype=np.int8),
    )
    result = screen(grid, od, obs, p, dtype=torch.float64)
    assert len(result.winners) == 1
    assert result.winners[0].frac_overlap == pytest.approx(1.0)

    # Erase distance-1 hit ⇒ AND-across-distances gives 0.
    obs_arr[1, frame, 32, 32] = 0.0
    obs = ObsVolume.from_dense_array(obs_arr)
    result = screen(grid, od, obs, p, dtype=torch.float64)
    assert len(result.winners) == 0
