"""Reading a reconstruction into a :class:`SampleShape`.

The happy paths here are short. Most of these tests are about the refusals,
because every one of them guards a failure that is *silent*: a mirrored mask
reconstructs perfectly, a wrong pixel size gives a sharp reconstruction of the
wrong-sized object, and picking shift 0 out of a 501-entry sweep gives a
mis-registered volume with no error anywhere.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_transforms.geometry import tomo
from midas_transforms.geometry.sample_shape import SampleShape


# ------------------------------------------------------------------ fixtures

def _disc_cube(n_shifts=1, n_slices=3, x=64, radius=10.0, value=1.0):
    """A cube with a filled disc on the rotation axis in every slice."""
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    c = (x - 1) / 2.0
    disc = (np.hypot(ix - c, iy - c) <= radius).astype(np.float32) * value
    cube = np.zeros((n_shifts, n_slices, x, x), dtype=np.float32)
    cube[...] = disc
    return cube, c


def _write_bin(tmp_path, cube, stem="recon"):
    ns, nsl, y, x = cube.shape
    p = (tmp_path / f"{stem}_NrShifts_{ns:03d}_NrSlices_{nsl:05d}"
                    f"_XDim_{x:06d}_YDim_{y:06d}_float32.bin")
    cube.astype(np.float32).tofile(p)
    return p


# --------------------------------------------------------------- the filename

def test_filename_carries_the_only_copy_of_the_shape(tmp_path):
    cube, _ = _disc_cube(n_shifts=2, n_slices=5, x=32)
    p = _write_bin(tmp_path, cube)
    meta = tomo.parse_recon_filename(p)
    assert meta == {"n_shifts": 2, "n_slices": 5, "xdim": 32, "ydim": 32}


def test_a_renamed_recon_is_refused_not_guessed(tmp_path):
    cube, _ = _disc_cube()
    p = _write_bin(tmp_path, cube)
    renamed = tmp_path / "recon.bin"
    p.rename(renamed)
    with pytest.raises(ValueError, match="not a MIDAS-tomo reconstruction"):
        tomo.parse_recon_filename(renamed)


def test_a_truncated_file_is_caught_by_the_size_cross_check(tmp_path):
    """The name and the bytes are independent; disagreement means one is wrong."""
    cube, c = _disc_cube(n_slices=4, x=32)
    p = _write_bin(tmp_path, cube)
    with p.open("r+b") as f:
        f.truncate(p.stat().st_size - 4096)
    with pytest.raises(ValueError, match="filename says"):
        tomo.from_midas_tomo_bin(
            p, pixel_size_um=1.0, rot_axis_ix=c, rot_axis_iy=c,
            in_plane="xy", threshold=0.5,
        )


def test_the_cleanup_sweep_axis_is_parsed(tmp_path):
    meta = tomo.parse_recon_filename(
        "r_NrCleanup_007_NrShifts_003_NrSlices_00010"
        "_XDim_000512_YDim_000512_float32.bin"
    )
    assert meta["n_cleanup"] == 7 and meta["n_shifts"] == 3


# ------------------------------------------------------------- the shift sweep

def test_a_multi_shift_cube_refuses_to_pick_one_for_you(tmp_path):
    """501 candidates, 500 of them mis-registered. Index 0 is not a default."""
    cube, c = _disc_cube(n_shifts=4, x=32)
    p = _write_bin(tmp_path, cube)
    with pytest.raises(ValueError, match="SWEEP over candidate shifts"):
        tomo.from_midas_tomo_bin(
            p, pixel_size_um=1.0, rot_axis_ix=c, rot_axis_iy=c,
            in_plane="xy", threshold=0.5,
        )


def test_a_single_shift_cube_needs_no_index(tmp_path):
    """The refusal exists because of ambiguity; with one entry there is none."""
    cube, c = _disc_cube(n_shifts=1, x=32)
    p = _write_bin(tmp_path, cube)
    s = tomo.from_midas_tomo_bin(
        p, pixel_size_um=1.0, rot_axis_ix=c, rot_axis_iy=c,
        in_plane="xy", threshold=0.5,
    )
    assert s.provenance["shift_index"] == 0


def test_the_chosen_shift_is_the_one_read(tmp_path):
    """A weak assertion would pass on any index; give the shifts different sizes."""
    x = 48
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    c = (x - 1) / 2.0
    r = np.hypot(ix - c, iy - c)
    cube = np.stack([(r <= rad).astype(np.float32) for rad in (4.0, 8.0, 12.0)])
    cube = cube[:, None, :, :]                      # (shift, slice, y, x)
    p = _write_bin(tmp_path, cube)
    got = [
        tomo.from_midas_tomo_bin(
            p, pixel_size_um=1.0, rot_axis_ix=c, rot_axis_iy=c,
            in_plane="xy", threshold=0.5, shift_index=i,
        ).occupancy.sum()
        for i in range(3)
    ]
    assert got[0] < got[1] < got[2]


def test_shift_index_out_of_range_raises(tmp_path):
    cube, c = _disc_cube(n_shifts=2, x=32)
    p = _write_bin(tmp_path, cube)
    with pytest.raises(IndexError):
        tomo.from_midas_tomo_bin(
            p, pixel_size_um=1.0, rot_axis_ix=c, rot_axis_iy=c,
            in_plane="xy", threshold=0.5, shift_index=5,
        )


# ------------------------------------------------------------- the pad region

def test_a_little_corner_ringing_is_CLIPPED_not_rejected():
    """``recon_xdim = next_pow2(det_xdim)``: a 1365-wide detector gives a 2048
    grid, so a third of every slice is padding no ray ever sampled. It is not
    data, so it is zeroed -- but a few rung-up corner voxels must not throw
    away the whole reconstruction. Measured on bt_1id_jun25b NMC811 s5, the old
    reject-on-any rule refused all 12 thresholds, including ones where 0.5 %
    of the mask was outside."""
    x = 64
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    vol = (np.hypot(ix - 31.5, iy - 31.5) <= 8.0).astype(np.float32)[None]
    vol[0, 2, 2] = 1.0                      # one corner voxel of ringing
    s = tomo.from_array(
        vol, pixel_size_um=1.0, rot_axis_ix=31.5, rot_axis_iy=31.5,
        in_plane="xy", threshold=0.5, det_xdim=40,
    )
    assert s.provenance["pad_occupancy_clipped"] == 1
    assert s.occupancy[0, 2, 2] == 0.0, "the corner voxel must be zeroed"
    assert s.provenance["pad_occupancy_clipped_fraction"] < 0.01


def test_a_LARGE_overflow_still_raises_and_names_truncation():
    """The case that is not ringing: the sample is wider than the field of
    view, the reconstruction cups, and no threshold gives a usable mask."""
    x = 64
    vol = np.zeros((1, x, x), dtype=np.float32)
    vol[0, :8, :8] = 1.0                    # a whole corner block
    with pytest.raises(ValueError, match="wider than the field of view"):
        tomo.from_array(
            vol, pixel_size_um=1.0, rot_axis_ix=31.5, rot_axis_iy=31.5,
            in_plane="xy", threshold=0.5, det_xdim=40,
        )


def test_the_pad_tolerance_is_adjustable_and_zero_restores_strictness():
    x = 64
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    vol = (np.hypot(ix - 31.5, iy - 31.5) <= 8.0).astype(np.float32)[None]
    vol[0, 2, 2] = 1.0
    with pytest.raises(ValueError, match="above the 0.00 % limit"):
        tomo.from_array(
            vol, pixel_size_um=1.0, rot_axis_ix=31.5, rot_axis_iy=31.5,
            in_plane="xy", threshold=0.5, det_xdim=40, max_pad_fraction=0.0,
        )


def test_a_mask_inside_the_reconstructible_disc_passes():
    x = 64
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    vol = (np.hypot(ix - 31.5, iy - 31.5) <= 8.0).astype(np.float32)[None]
    s = tomo.from_array(
        vol, pixel_size_um=1.0, rot_axis_ix=31.5, rot_axis_iy=31.5,
        in_plane="xy", threshold=0.5, det_xdim=40,
    )
    assert s.provenance["pad_occupancy_clipped"] == 0
    assert s.provenance["reconstructible_radius_px"] == 20.0


def test_without_det_xdim_the_check_is_weaker_and_says_so():
    """It still catches a corner, but the annulus between det_xdim/2 and X/2
    is unguarded — the provenance has to record which check ran."""
    x = 64
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    vol = (np.hypot(ix - 31.5, iy - 31.5) <= 25.0).astype(np.float32)[None]
    s = tomo.from_array(
        vol, pixel_size_um=1.0, rot_axis_ix=31.5, rot_axis_iy=31.5,
        in_plane="xy", threshold=0.5,
    )
    assert "weaker check" in s.provenance["pad_check_basis"]
    # ... and the same volume IS refused once the detector width is known
    with pytest.raises(ValueError, match="wider than the field of view"):
        tomo.from_array(
            vol, pixel_size_um=1.0, rot_axis_ix=31.5, rot_axis_iy=31.5,
            in_plane="xy", threshold=0.5, det_xdim=40,
        )


# ---------------------------------------------------------------- the volume

def test_the_read_volume_matches_pi_r2_h(tmp_path):
    """End to end: bytes on disk -> a volume in um^3 that closes on the disc."""
    r_px, px, pitch, nsl = 12.0, 2.5, 4.0, 6
    cube, c = _disc_cube(n_slices=nsl, x=64, radius=r_px)
    p = _write_bin(tmp_path, cube)
    s = tomo.from_midas_tomo_bin(
        p, pixel_size_um=px, slice_pitch_um=pitch, rot_axis_ix=c, rot_axis_iy=c,
        in_plane="xy", threshold=0.5, det_xdim=48,
    )
    want = math.pi * (r_px * px) ** 2 * (nsl * pitch)
    assert abs(s.total_volume_um3() - want) / want < 0.05


def test_slice_range_reads_a_subset_and_moves_slice0_z(tmp_path):
    """An FF layer needs a few slices out of hundreds, and the z origin has to
    follow — otherwise the subset silently registers to the wrong height."""
    cube, c = _disc_cube(n_slices=10, x=32)
    p = _write_bin(tmp_path, cube)
    s = tomo.from_midas_tomo_bin(
        p, pixel_size_um=1.0, slice_pitch_um=3.0, rot_axis_ix=c, rot_axis_iy=c,
        in_plane="xy", threshold=0.5, slice_range=(4, 7), slice0_z_um=100.0,
    )
    assert s.shape[0] == 3
    assert s.slice0_z_um == pytest.approx(100.0 + 4 * 3.0)
    assert s.voxel_positions_um()[0, 0, 0, 2] == pytest.approx(112.0)


def test_a_bad_slice_range_raises(tmp_path):
    cube, c = _disc_cube(n_slices=5, x=32)
    p = _write_bin(tmp_path, cube)
    with pytest.raises(ValueError, match="half-open range"):
        tomo.from_midas_tomo_bin(
            p, pixel_size_um=1.0, rot_axis_ix=c, rot_axis_iy=c,
            in_plane="xy", threshold=0.5, slice_range=(3, 99),
        )


# ------------------------------------------------------------- the refusals

def test_in_plane_handedness_has_no_default():
    with pytest.raises(TypeError):
        tomo.from_array(                              # type: ignore[call-arg]
            np.ones((1, 4, 4)), pixel_size_um=1.0,
            rot_axis_ix=1.5, rot_axis_iy=1.5, threshold=0.5,
        )


def test_a_mirrored_in_plane_choice_actually_mirrors_the_sample():
    """The reason handedness cannot be defaulted: it is silent in the volume
    and visible only in position. Total volume is identical; y flips sign."""
    vol = np.zeros((1, 8, 8), dtype=np.float32)
    vol[0, 1, 5] = 1.0
    kw = dict(pixel_size_um=1.0, rot_axis_ix=3.5, rot_axis_iy=3.5,
              threshold=0.5)
    a = tomo.from_array(vol, in_plane="xy", **kw)
    b = tomo.from_array(vol, in_plane="x-y", **kw)
    assert a.total_volume_um3() == b.total_volume_um3()
    ya = a.voxel_positions_um()[0, 1, 5, 1]
    yb = b.voxel_positions_um()[0, 1, 5, 1]
    assert ya == pytest.approx(-yb) and ya != 0.0


def test_non_finite_reconstruction_values_are_refused():
    vol = np.ones((1, 4, 4), dtype=np.float32)
    vol[0, 2, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tomo.from_array(vol, pixel_size_um=1.0, rot_axis_ix=1.5,
                        rot_axis_iy=1.5, in_plane="xy", threshold=0.5)


def test_provenance_says_the_registration_is_unverified():
    """A SampleShape carries no other record of it, and a tomogram read from
    disk has been registered by nobody."""
    vol = np.zeros((1, 8, 8), dtype=np.float32)
    vol[0, 3:5, 3:5] = 1.0
    s = tomo.from_array(vol, pixel_size_um=1.0, rot_axis_ix=3.5,
                        rot_axis_iy=3.5, in_plane="xy", threshold=0.5)
    assert "NOT verified" in s.provenance["registration"]


def test_a_filled_square_is_refused_because_its_corners_cannot_be_data():
    """A reconstruction whose corners are fully occupied is not a
    reconstruction -- the corners are outside the field of view of every
    projection, so 21 % of that mask is provably artefact."""
    with pytest.raises(ValueError, match="wider than the field of view"):
        tomo.from_array(np.ones((1, 8, 8)), pixel_size_um=1.0, rot_axis_ix=3.5,
                        rot_axis_iy=3.5, in_plane="xy", threshold=0.5)


# -------------------------------------------------------- threshold sweeping

def test_a_high_contrast_object_is_stationary_in_the_threshold():
    x = 48
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    vol = (np.hypot(ix - 23.5, iy - 23.5) <= 10.0).astype(np.float64)[None]
    r = tomo.threshold_sensitivity(vol, [0.2, 0.4, 0.6, 0.8],
                                   voxel_volume_um3=1.0)
    assert r["stationary"] and r["fractional_spread"] == 0.0


def test_a_smooth_blob_is_NOT_stationary_and_the_band_is_reported():
    """The case that must not be reported as a measurement: no contrast, so the
    'volume' is whatever threshold was picked."""
    x = 48
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    r2 = ((ix - 23.5) ** 2 + (iy - 23.5) ** 2) / (2 * 8.0 ** 2)
    vol = np.exp(-r2)[None]
    r = tomo.threshold_sensitivity(vol, [0.2, 0.4, 0.6, 0.8],
                                   voxel_volume_um3=1.0)
    assert not r["stationary"]
    assert r["radius_spread"] > 1.3      # tens of percent in reported radius


def test_threshold_sensitivity_needs_more_than_one_threshold():
    with pytest.raises(ValueError, match="at least two"):
        tomo.threshold_sensitivity(np.ones((1, 4, 4)), [0.5],
                                   voxel_volume_um3=1.0)


# --------------------------------------------------------- legacy square mask

def test_the_legacy_square_uint8_mask_round_trips(tmp_path):
    n = 40
    iy, ix = np.mgrid[0:n, 0:n]
    img = (np.hypot(ix - 19.5, iy - 19.5) <= 8.0).astype(np.uint8)
    p = tmp_path / "TomoImage.bin"
    img.tofile(p)
    s = tomo.from_square_uint8(p, pixel_size_um=2.0, in_plane="xy",
                               slice_pitch_um=50.0)
    assert s.shape == (1, n, n)
    assert s.total_volume_um3() == pytest.approx(int(img.sum()) * 4.0 * 50.0)
    assert s.provenance["side_px"] == n


def test_a_non_square_legacy_file_is_refused(tmp_path):
    p = tmp_path / "bad.bin"
    np.zeros(1000, dtype=np.uint8).tofile(p)     # 1000 is not a perfect square
    with pytest.raises(ValueError, match="not a perfect square"):
        tomo.load_square_tomo(p)


# ------------------------------------------------------------- NXtomoproc

def test_nxtomoproc_round_trips_through_the_midas_tomo_writer(tmp_path):
    h5py = pytest.importorskip("h5py")
    cube, c = _disc_cube(n_shifts=3, n_slices=4, x=32, radius=9.0)
    p = tmp_path / "recon.h5"
    with h5py.File(p, "w") as hf:
        rec = hf.create_group("entry/reconstruction")
        d = rec.create_dataset("data", data=cube)
        d.attrs["axes"] = "shift:slice:y:x"
        rec.create_dataset("axis_shift", data=np.array([-1.0, 0.0, 1.0]))
    s = tomo.from_nxtomoproc(
        p, pixel_size_um=1.5, rot_axis_ix=c, rot_axis_iy=c, in_plane="xy",
        threshold=0.5, shift_index=1, det_xdim=24,
    )
    assert s.provenance["axis_shift_px"] == 0.0
    assert s.shape == (4, 32, 32)


def test_a_transposed_nxtomoproc_cube_is_refused(tmp_path):
    """A (slice, shift, y, x) cube reconstructs a sample rotated into the beam
    and is otherwise perfectly well-formed."""
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "recon.h5"
    with h5py.File(p, "w") as hf:
        d = hf.create_dataset("entry/reconstruction/data",
                              data=np.ones((2, 2, 8, 8), dtype=np.float32))
        d.attrs["axes"] = "slice:shift:y:x"
    with pytest.raises(ValueError, match="declares axes"):
        tomo.from_nxtomoproc(p, pixel_size_um=1.0, rot_axis_ix=3.5,
                             rot_axis_iy=3.5, in_plane="xy", threshold=0.5,
                             shift_index=0)


# ------------------------------------------- what the whole thing is here for

def test_a_read_shape_replaces_V_gauge_and_the_factor_is_large():
    """The point of Phase 3, on the FF reference run's own gauge volume.

    ``paramstest.txt`` there has no ``Vsample``, so
    V_gauge = Hbeam * pi * Rsample^2 = 2000 * pi * 2000^2, built from two
    search bounds. A 400 um rod lit over 200 um is four orders smaller.
    """
    v_gauge = 2000.0 * math.pi * 2000.0 ** 2
    x = 128
    iy, ix = np.mgrid[0:x, 0:x].astype(np.float64)
    disc = (np.hypot(ix - 63.5, iy - 63.5) <= 40.0).astype(np.float32)
    vol = np.repeat(disc[None], 40, axis=0)          # 40 slices
    s = tomo.from_array(vol, pixel_size_um=5.0, slice_pitch_um=10.0,
                        rot_axis_ix=63.5, rot_axis_iy=63.5, in_plane="xy",
                        threshold=0.5, det_xdim=100)
    ratio = s.gauge_volume_ratio(v_gauge, beam_height_um=200.0)
    assert ratio < 0.01
    assert (1.0 / ratio) ** (1 / 3) > 4.0    # radii overstated 4x or more
    assert isinstance(s, SampleShape)


def test_percentile_thresholds_pin_radius_spread_to_a_CONSTANT():
    """A check that cannot fail, pinned so the number is never quoted again.

    With thresholds at percentiles of the data, the volume runs from ~50 % of
    the voxels to ~0.5 % by construction -- a ratio of 100 -- so
    ``radius_spread`` reports 100**(1/3) = 4.642 for any input whatsoever.
    Measured on six visibly different bt_1id_jun25b reconstructions: 4.642 every
    time. ``fractional_spread`` does still vary and is the one to use.
    """
    rng = np.random.default_rng(0)
    got = []
    for shape, scale in (((4, 40, 40), 1.0), ((6, 30, 30), 1e-4),
                         ((3, 50, 50), 900.0)):
        vol = rng.gamma(2.0, scale, shape)
        ths = np.linspace(*np.percentile(vol, [50, 99.5]), 12)
        r = tomo.threshold_sensitivity(vol, ths, voxel_volume_um3=1.0)
        got.append(r["radius_spread"])
    for g in got:
        assert g == pytest.approx(100.0 ** (1 / 3), rel=0.02), got
