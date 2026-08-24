"""The tomography grid is the third frame, and the one with no convention.

A diffraction experiment at 1-ID has three coordinate systems that must agree:
MIDAS lab (x beam, y outboard, z up), APS lab (x outboard, y up, z beam), and —
once a tomogram is used as a sample shape — the reconstruction grid.

The first two are pinned by ``R_MIDAS_TO_APS`` and unit-tested elsewhere. The
third is the problem: a reconstruction cube's shape lives in its filename, its
pixel size lives in an acquisition config, and the rotation-axis position is an
*output* of the reconstruction. Nothing in the file says which way is up or
which way the sample turns.

These tests pin the parts that can be pinned and, more importantly, pin the
refusals — because the failure mode here is a mirrored sample, which
reconstructs perfectly and is invisible downstream. Same class as the omega
sign.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_stress.frames import (
    R_APS_TO_MIDAS,
    R_MIDAS_TO_APS,
    TOMO_IN_PLANE,
    tomo_grid_to_midas,
    tomo_slice_for_z,
)


# --------------------------------------------------------------- the axis map

def test_the_midas_aps_permutation_is_what_the_beamline_says():
    """x_MIDAS = z_APS, y_MIDAS = x_APS, z_MIDAS = y_APS."""
    beam_midas = np.array([1.0, 0.0, 0.0])
    outboard_midas = np.array([0.0, 1.0, 0.0])
    up_midas = np.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(R_MIDAS_TO_APS @ beam_midas, [0, 0, 1])      # -> z_APS
    np.testing.assert_allclose(R_MIDAS_TO_APS @ outboard_midas, [1, 0, 0])  # -> x_APS
    np.testing.assert_allclose(R_MIDAS_TO_APS @ up_midas, [0, 1, 0])        # -> y_APS


def test_the_permutation_is_a_rotation_not_a_reflection():
    """det = +1. A cyclic permutation preserves handedness; a swap would not,
    and a handedness flip here would mirror every reconstruction."""
    assert np.isclose(np.linalg.det(R_MIDAS_TO_APS), 1.0)
    np.testing.assert_allclose(R_MIDAS_TO_APS @ R_APS_TO_MIDAS, np.eye(3), atol=1e-15)


# ------------------------------------------------------- the vertical is the tie

def test_slice_is_the_vertical_and_carries_the_stage_position():
    """slice -> MIDAS z is what registers a tomogram to an FF/NF layer."""
    x, y, z = tomo_grid_to_midas(
        slice_idx=10, iy=0, ix=0,
        pixel_size_um=1.17, slice_pitch_um=1.17,
        rot_axis_ix=0, rot_axis_iy=0, slice0_z_um=250.0,
    )
    assert z == pytest.approx(10 * 1.17 + 250.0)
    assert (x, y) == (0.0, 0.0)


def test_slice_pitch_is_separate_from_pixel_size():
    """Vertical detector binning breaks the isotropic assumption."""
    _, _, z = tomo_grid_to_midas(
        slice_idx=4, iy=0, ix=0,
        pixel_size_um=1.17, slice_pitch_um=2.34,      # 2x vertical binning
        rot_axis_ix=0, rot_axis_iy=0,
    )
    assert z == pytest.approx(4 * 2.34)


def test_in_plane_is_measured_from_the_rotation_axis_not_the_grid_corner():
    """The axis is the origin; it is an output of the shift sweep."""
    x, y, _ = tomo_grid_to_midas(
        slice_idx=0, iy=100, ix=140,
        pixel_size_um=2.0, slice_pitch_um=2.0,
        rot_axis_ix=120, rot_axis_iy=100,
    )
    assert x == pytest.approx((140 - 120) * 2.0)     # +20 px along +x
    assert y == pytest.approx(0.0)                    # on the axis in y


# ------------------------------------------------------------ the refusals

def test_in_plane_handedness_is_not_guessed():
    with pytest.raises(ValueError, match="no safe default"):
        tomo_grid_to_midas(0, 0, 0, pixel_size_um=1.0, slice_pitch_um=1.0,
                           rot_axis_ix=0, rot_axis_iy=0, in_plane="nonsense")


def test_all_eight_in_plane_choices_are_orthonormal():
    """Each is a signed axis assignment, so each 2x2 must be orthogonal —
    otherwise a choice would shear the reconstruction, not just mirror it."""
    for name, ((ax_x, ax_y), (ay_x, ay_y)) in TOMO_IN_PLANE.items():
        M = np.array([[ax_x, ay_x], [ax_y, ay_y]])
        np.testing.assert_allclose(M @ M.T, np.eye(2), atol=1e-15,
                                   err_msg=f"{name} is not orthonormal")
        assert abs(round(float(np.linalg.det(M)))) == 1, name


def test_mirroring_actually_changes_the_answer():
    """Guard against a no-op: if 'xy' and '-xy' agreed, the knob would be
    decorative and the mirror it exists to prevent would go undetected."""
    kw = dict(pixel_size_um=1.0, slice_pitch_um=1.0, rot_axis_ix=0, rot_axis_iy=0)
    a = tomo_grid_to_midas(0, 3, 7, in_plane="xy", **kw)
    b = tomo_grid_to_midas(0, 3, 7, in_plane="-xy", **kw)
    c = tomo_grid_to_midas(0, 3, 7, in_plane="yx", **kw)
    assert a != b and a != c


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_pixel_size_must_be_supplied_and_positive(bad):
    with pytest.raises(ValueError, match="pixel_size_um"):
        tomo_grid_to_midas(0, 0, 0, pixel_size_um=bad, slice_pitch_um=1.0,
                           rot_axis_ix=0, rot_axis_iy=0)


# --------------------------------------------------------- z -> slice lookup

def test_z_to_slice_round_trips():
    kw = dict(slice_pitch_um=1.17, slice0_z_um=250.0)
    for s in (0, 5, 199):
        _, _, z = tomo_grid_to_midas(s, 0, 0, pixel_size_um=1.17,
                                     rot_axis_ix=0, rot_axis_iy=0, **kw)
        assert tomo_slice_for_z(z, **kw) == s


def test_z_outside_the_reconstruction_raises_rather_than_clamping():
    """Clamping would extrapolate the sample mask past the tomographic field
    of view — fabricating path length, which is worse than failing."""
    with pytest.raises(ValueError, match="does not cover this layer"):
        tomo_slice_for_z(1000.0, slice_pitch_um=1.0, slice0_z_um=0.0, n_slices=100)
    with pytest.raises(ValueError, match="does not cover this layer"):
        tomo_slice_for_z(-5.0, slice_pitch_um=1.0, slice0_z_um=0.0, n_slices=100)
    # in range is fine
    assert tomo_slice_for_z(50.0, slice_pitch_um=1.0, slice0_z_um=0.0,
                            n_slices=100) == 50


def test_ff_layer_to_tomo_slice_is_a_read_not_a_fit():
    """The realistic use: an FF layer at a known stage Z picks its tomo slice.

    Worked with the bt_1id_jun25b-style numbers — 1.17 um slices, tomogram
    starting at stage z = 100 um, FF layer measured at z = 350 um.
    """
    s = tomo_slice_for_z(350.0, slice_pitch_um=1.17, slice0_z_um=100.0,
                         n_slices=2048)
    assert s == round((350.0 - 100.0) / 1.17)
    _, _, z_back = tomo_grid_to_midas(s, 0, 0, pixel_size_um=1.17,
                                      slice_pitch_um=1.17, rot_axis_ix=0,
                                      rot_axis_iy=0, slice0_z_um=100.0)
    assert abs(z_back - 350.0) <= 1.17 / 2 + 1e-9


# ---------------------------------------------------------- the inverse map

def test_midas_to_tomo_grid_inverts_tomo_grid_to_midas_exactly():
    """Forward then back must be the identity for every handedness. A
    hand-written inverse that is only approximately orthogonal would pass a
    loose tolerance and then leak half a voxel into every path length."""
    from midas_stress.frames import (
        TOMO_IN_PLANE, midas_to_tomo_grid, tomo_grid_to_midas,
    )

    rng = np.random.default_rng(0)
    s0, iy0, ix0 = (rng.uniform(0, 30, 200), rng.uniform(0, 40, 200),
                    rng.uniform(0, 50, 200))
    kw = dict(pixel_size_um=1.37, slice_pitch_um=2.9,
              rot_axis_ix=24.5, rot_axis_iy=19.5, slice0_z_um=-412.5)
    for ip in sorted(TOMO_IN_PLANE):
        x, y, z = tomo_grid_to_midas(s0, iy0, ix0, in_plane=ip, **kw)
        s1, iy1, ix1 = midas_to_tomo_grid(x, y, z, in_plane=ip, **kw)
        np.testing.assert_allclose(s1, s0, atol=1e-12, err_msg=ip)
        np.testing.assert_allclose(iy1, iy0, atol=1e-12, err_msg=ip)
        np.testing.assert_allclose(ix1, ix0, atol=1e-12, err_msg=ip)


def test_the_inverse_does_not_round_to_a_voxel():
    """Rounding here would silently extend a sample mask by half a voxel in
    every direction; the caller has to own that choice."""
    from midas_stress.frames import midas_to_tomo_grid

    s, iy, ix = midas_to_tomo_grid(
        0.5, 0.0, 0.0, pixel_size_um=1.0, slice_pitch_um=1.0,
        rot_axis_ix=0.0, rot_axis_iy=0.0,
    )
    assert float(ix) == 0.5


def test_the_inverse_refuses_an_unknown_handedness():
    from midas_stress.frames import midas_to_tomo_grid

    with pytest.raises(ValueError, match="in_plane must be one of"):
        midas_to_tomo_grid(0.0, 0.0, 0.0, pixel_size_um=1.0,
                           slice_pitch_um=1.0, rot_axis_ix=0.0,
                           rot_axis_iy=0.0, in_plane="diagonal")
