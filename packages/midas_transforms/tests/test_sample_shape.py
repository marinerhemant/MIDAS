"""The illuminated volume — the quantity ``V_gauge`` was always standing in for.

Grain volume scales linearly with ``V_gauge`` and grain radius with its cube
root, and ``V_gauge = Hbeam * pi * Rsample^2`` is built from two deliberately
generous SEARCH BOUNDS. So the absolute scale of every reported grain size is a
canned constant. :class:`SampleShape` supplies the measured replacement.

These tests check the arithmetic against closed forms, and — more importantly —
the refusals, because the failure modes here are silent: a mirrored shape
reconstructs perfectly, and a wrong pixel size gives a sharp, plausible
reconstruction of an object of the wrong size.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_transforms.geometry.sample_shape import SampleShape


# ------------------------------------------------------------ analytic shapes

def test_cylinder_volume_matches_pi_r2_h():
    d, h, px = 200.0, 300.0, 2.0
    s = SampleShape.cylinder(diameter_um=d, height_um=h, pixel_size_um=px)
    want = math.pi * (d / 2) ** 2 * h
    got = s.total_volume_um3()
    assert abs(got - want) / want < 0.01, f"{got:.0f} vs {want:.0f}"


def test_supersampling_reduces_the_boundary_error():
    """The boundary is a surface term and it lands straight in every volume."""
    d, h, px = 60.0, 10.0, 2.0            # only 30 px across: boundary matters
    want = math.pi * (d / 2) ** 2 * h
    e1 = abs(SampleShape.cylinder(diameter_um=d, height_um=h, pixel_size_um=px,
                                  supersample=1).total_volume_um3() - want) / want
    e8 = abs(SampleShape.cylinder(diameter_um=d, height_um=h, pixel_size_um=px,
                                  supersample=8).total_volume_um3() - want) / want
    assert e8 < e1, f"supersampling made it worse: {e1:.4f} -> {e8:.4f}"
    assert e8 < 0.005


def test_box_volume_is_exact():
    s = SampleShape.box(size_x_um=100.0, size_y_um=60.0, height_um=40.0,
                        pixel_size_um=2.0)
    assert s.total_volume_um3() == pytest.approx(100.0 * 60.0 * 40.0, rel=1e-9)


# -------------------------------------------------------- illuminated volume

def test_beam_taller_than_the_sample_lights_all_of_it():
    s = SampleShape.cylinder(diameter_um=100.0, height_um=50.0, pixel_size_um=2.0)
    assert s.illuminated_volume_um3(beam_height_um=1e6) == \
        pytest.approx(s.total_volume_um3())
    assert s.illuminated_volume_um3() == pytest.approx(s.total_volume_um3())


def test_a_slab_beam_lights_a_slab():
    """The FF case: a beam thinner than the sample cuts a disc out of the rod."""
    d, h, px, bh = 100.0, 400.0, 2.0, 40.0
    s = SampleShape.cylinder(diameter_um=d, height_um=h, pixel_size_um=px)
    got = s.illuminated_volume_um3(beam_height_um=bh, beam_centre_z_um=0.0)
    want = math.pi * (d / 2) ** 2 * bh
    assert abs(got - want) / want < 0.05, f"{got:.0f} vs {want:.0f}"


def test_a_cylinder_is_flat_in_omega_and_that_is_why_V1_has_no_power():
    """A cylinder's illuminated volume does not vary with rotation.

    That is a *property of the sample*, and it means the V1 sinogram
    registration check cannot detect anything on it. Compute the modulation
    before claiming a registration was verified.
    """
    s = SampleShape.cylinder(diameter_um=80.0, height_um=20.0, pixel_size_um=2.0)
    v = s.illuminated_volume_sinogram(np.arange(0, 180, 15.0),
                                      beam_width_um=40.0)
    assert v.std() / v.mean() < 0.02, "a cylinder should be flat in omega"


def test_a_box_is_NOT_flat_in_omega_so_V1_can_work():
    """The contrast that makes the previous test meaningful rather than a bug."""
    s = SampleShape.box(size_x_um=200.0, size_y_um=50.0, height_um=20.0,
                        pixel_size_um=2.0)
    v = s.illuminated_volume_sinogram(np.arange(0, 180, 15.0),
                                      beam_width_um=60.0)
    assert v.std() / v.mean() > 0.05, "an elongated box must modulate with omega"


def test_gauge_volume_ratio_is_the_factor_grain_volumes_are_wrong_by():
    """Worked on the FF reference run's own numbers.

    ``paramstest.txt`` there has no ``Vsample`` line, so
    V_gauge = Hbeam * pi * Rsample^2 = 2000 * pi * 2000^2 = 2.513e10 um^3,
    built entirely from two search bounds.
    """
    v_gauge = 2000.0 * math.pi * 2000.0 ** 2
    s = SampleShape.cylinder(diameter_um=1000.0, height_um=2000.0,
                             pixel_size_um=20.0)
    r = s.gauge_volume_ratio(v_gauge, beam_height_um=200.0)
    # a 1 mm rod lit over 200 um is pi*500^2*200 = 1.571e8 um^3
    assert r == pytest.approx(math.pi * 500.0 ** 2 * 200.0 / v_gauge, rel=0.05)
    assert r < 1.0, "V_gauge from search bounds must overstate the lit volume"
    # radius error is the cube root
    assert (1.0 / r) ** (1 / 3) > 3.0


# ------------------------------------------------------------- the refusals

def test_pixel_size_and_pitch_must_be_positive():
    occ = np.ones((2, 4, 4))
    for kw in ({"pixel_size_um": 0.0}, {"slice_pitch_um": -1.0}):
        base = dict(pixel_size_um=1.0, slice_pitch_um=1.0,
                    rot_axis_ix=0, rot_axis_iy=0)
        base.update(kw)
        with pytest.raises(ValueError, match="must be > 0"):
            SampleShape(occupancy=occ, **base)


def test_in_plane_handedness_is_not_guessed():
    with pytest.raises(ValueError, match="no safe default"):
        SampleShape(occupancy=np.ones((1, 2, 2)), pixel_size_um=1.0,
                    slice_pitch_um=1.0, rot_axis_ix=0, rot_axis_iy=0,
                    in_plane="sideways")


def test_occupancy_must_be_a_fraction_not_a_density():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SampleShape(occupancy=np.full((1, 2, 2), 4.7), pixel_size_um=1.0,
                    slice_pitch_um=1.0, rot_axis_ix=0, rot_axis_iy=0)


def test_occupancy_must_be_3d():
    with pytest.raises(ValueError, match="must be 3-D"):
        SampleShape(occupancy=np.ones((4, 4)), pixel_size_um=1.0,
                    slice_pitch_um=1.0, rot_axis_ix=0, rot_axis_iy=0)


def test_anisotropic_voxels_are_refused_by_to_sample_grid():
    """SampleGrid assumes a cubic voxel; a silent mismatch would give the ray
    tracer the wrong length along z."""
    pytest.importorskip("torch")
    s = SampleShape.cylinder(diameter_um=20.0, height_um=10.0,
                             pixel_size_um=1.0, slice_pitch_um=2.0)
    with pytest.raises(ValueError, match="cubic voxels"):
        s.to_sample_grid()


# ------------------------------------------------------------ frame plumbing

def test_the_vertical_carries_the_stage_position():
    """slice -> MIDAS z, offset by slice0_z_um: the FF/NF registration."""
    s = SampleShape.cylinder(diameter_um=20.0, height_um=30.0,
                             pixel_size_um=2.0, centre_z_um=500.0)
    z = s.voxel_positions_um()[..., 2]
    assert z.mean() == pytest.approx(500.0, abs=1.0)


def test_to_sample_grid_populates_the_topology_absorption_needs():
    """``from_arrays`` leaves grid_shape None and absorption_factor raises on
    exactly that — which is why the existing V-map path cannot reach it."""
    pytest.importorskip("torch")
    s = SampleShape.cylinder(diameter_um=20.0, height_um=10.0, pixel_size_um=2.0)
    g = s.to_sample_grid()
    assert g.grid_shape is not None and g.grid_origin_um is not None
    assert g.n_voxels == int(np.prod(s.shape))
    assert bool(g.sample_mask.any())
