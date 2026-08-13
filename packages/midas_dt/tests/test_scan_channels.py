"""Raw-file layout, snake detection, channels and geometry parsing."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.channels import Channel, channels_from_legacy_params
from midas_dt.geometry import (
    DTGeometry,
    geometry_from_legacy_params,
    parse_legacy_params,
)
from midas_dt.scan import PILATUS_1475x1679, DTScan, RawFormat, detect_snake


# ------------------------------------------------------------- raw format
def test_u3o8_arithmetic_is_exact():
    """The real file sizes must divide with no remainder.

    14,274,698,292 = 8192 + 1441 x (1475 x 1679 x 4) and the dark
    99,069,192 = 8192 + 10 x (...). Exactness is what confirms the layout is
    one header per FILE rather than per frame.
    """
    fmt = PILATUS_1475x1679
    assert fmt.frame_bytes == 1475 * 1679 * 4 == 9_906_100
    assert 8192 + 1441 * fmt.frame_bytes == 14_274_698_292
    assert 8192 + 10 * fmt.frame_bytes == 99_069_192


def _write_raw(path, fmt: RawFormat, n_frames: int, fill=None):
    with open(path, "wb") as f:
        f.write(b"\0" * fmt.header_bytes)
        for i in range(n_frames):
            frame = (np.full(fmt.frame_shape, i, dtype=fmt.dtype)
                     if fill is None else fill[i].astype(fmt.dtype))
            frame.tofile(f)
    return path


@pytest.fixture
def small_fmt():
    return RawFormat(n_pixels_y=8, n_pixels_z=6, header_bytes=64)


def test_n_frames_from_size(tmp_path, small_fmt):
    p = _write_raw(tmp_path / "a.raw", small_fmt, 5)
    assert small_fmt.n_frames(p) == 5


def test_truncated_file_names_the_problem(tmp_path, small_fmt):
    p = _write_raw(tmp_path / "a.raw", small_fmt, 5)
    with open(p, "r+b") as f:          # lop off part of the last frame
        f.truncate(p.stat().st_size - 17)
    with pytest.raises(ValueError, match="not a whole number of"):
        small_fmt.n_frames(p)


def test_header_only_file_rejected(tmp_path, small_fmt):
    p = tmp_path / "h.raw"
    p.write_bytes(b"\0" * small_fmt.header_bytes)
    with pytest.raises(ValueError, match="truncated|header"):
        small_fmt.n_frames(p)


def test_memmap_does_not_read(tmp_path, small_fmt):
    p = _write_raw(tmp_path / "a.raw", small_fmt, 4)
    mm = small_fmt.memmap(p)
    assert isinstance(mm, np.memmap)
    assert mm.shape == (4, 6, 8)
    assert mm[2, 0, 0] == 2          # frame index is the fill value


# ------------------------------------------------------------------- scan
def test_scan_drops_the_throwaway_first_frame(tmp_path, small_fmt):
    """1-ID writes a junk first frame in every acquisition."""
    files = [_write_raw(tmp_path / f"s_{i:06d}.raw", small_fmt, 5) for i in range(3)]
    scan = DTScan(files=files, fmt=small_fmt, omega_deg=np.arange(4.0))
    assert scan.n_frames == 4
    assert scan.first_frame == 1
    # frame 0 of the scan is frame 1 of the file
    assert scan.frame(0, 0)[0, 0] == 1


def test_scan_can_keep_the_first_frame(tmp_path, small_fmt):
    files = [_write_raw(tmp_path / f"s_{i:06d}.raw", small_fmt, 5) for i in range(3)]
    scan = DTScan(files=files, fmt=small_fmt, omega_deg=np.arange(5.0),
                  drop_first_frame=False)
    assert scan.n_frames == 5
    assert scan.frame(0, 0)[0, 0] == 0


def test_scan_applies_the_detector_flip(tmp_path):
    fmt = RawFormat(n_pixels_y=4, n_pixels_z=3, header_bytes=16, flip_vertical=True)
    fill = np.stack([np.arange(12).reshape(3, 4)] * 2)
    p = _write_raw(tmp_path / "s_000000.raw", fmt, 2, fill=fill)
    scan = DTScan(files=[p], fmt=fmt, omega_deg=np.zeros(1))
    got = scan.frame(0, 0)
    np.testing.assert_array_equal(got, np.arange(12).reshape(3, 4)[::-1])


def test_missing_file_is_reported_with_a_count(tmp_path, small_fmt):
    good = _write_raw(tmp_path / "s_000000.raw", small_fmt, 3)
    with pytest.raises(FileNotFoundError, match="missing"):
        DTScan(files=[good, tmp_path / "nope.raw"], fmt=small_fmt)


def test_translation_index_is_bounds_checked(tmp_path, small_fmt):
    p = _write_raw(tmp_path / "s_000000.raw", small_fmt, 3)
    scan = DTScan(files=[p], fmt=small_fmt, omega_deg=np.zeros(2))
    with pytest.raises(IndexError, match="out of range"):
        scan.translation(1)


def test_from_stem_negates_omega(tmp_path, small_fmt):
    for n in range(161, 164):
        _write_raw(tmp_path / f"u3o8_{n:06d}.raw", small_fmt, 5)
    scan = DTScan.from_stem(tmp_path, "u3o8", 161, 163, fmt=small_fmt,
                            start_omega=180.25, omega_step=-0.25)
    assert scan.n_translations == 3
    assert scan.n_frames == 4
    # nominal for the first KEPT frame is 180.25 + (-0.25 * 1) = 180.0
    assert scan.omega_deg[0] == pytest.approx(-180.0)


# ------------------------------------------------------------------ snake
def _snake_profiles(n_trans=6, n_frames=40, snake=False, seed=0):
    """Neighbouring translations share a smooth omega-dependent signal.

    The profile must be ASYMMETRIC under reversal or the test proves nothing.
    An earlier version used ``sin(linspace(0, 3*pi))``, for which
    ``sin(3*pi - x) == sin(x)`` -- identical to its own reverse, so no snake
    is detectable in it by construction and the test failed for the right
    reason. Two off-centre bumps of different heights cannot be mistaken for
    their mirror image.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_frames)
    base = (1.0 * np.exp(-((x - 0.25) / 0.08) ** 2)
            + 0.4 * np.exp(-((x - 0.60) / 0.05) ** 2)
            + 0.3 * x)
    out = np.stack([base + 0.01 * rng.normal(size=n_frames) for _ in range(n_trans)])
    if snake:
        out[1::2] = out[1::2, ::-1]
    return out


def test_the_snake_fixture_is_actually_asymmetric():
    """Guard the guard: a reversal-symmetric profile makes the snake tests vacuous."""
    base = _snake_profiles(n_trans=1, snake=False)[0]
    assert not np.allclose(base, base[::-1], atol=0.05), (
        "the test profile is symmetric under reversal, so it cannot "
        "distinguish a snake scan from a normal one"
    )


def test_detects_a_snake_scan():
    is_snake, gain = detect_snake(_snake_profiles(snake=True))
    assert is_snake
    assert gain > 1.05


def test_does_not_flag_a_unidirectional_scan():
    is_snake, gain = detect_snake(_snake_profiles(snake=False))
    assert not is_snake


def test_detect_snake_needs_enough_translations():
    with pytest.raises(ValueError, match="at least 3"):
        detect_snake(np.zeros((2, 10)))


def test_detect_snake_rejects_wrong_rank():
    with pytest.raises(ValueError, match="must be 2-D"):
        detect_snake(np.zeros((3, 4, 5)))


# --------------------------------------------------------------- channels
def test_channel_bin_counts():
    c = Channel(105, 125, eta_min=-180, eta_max=180, r_bin=0.25, eta_bin=3.0)
    assert c.n_r == 80
    assert c.n_eta == 120
    assert c.n_sinograms == 9600


def test_channel_centred_uses_half_width():
    """Legacy `RadiusToFit 118 10` means 118 +/- 10, not a 10-wide window."""
    c = Channel.centred(118, 10)
    assert (c.r_min, c.r_max) == (108, 128)


@pytest.mark.parametrize("kw,msg", [
    (dict(r_min=125, r_max=105), "r_max must exceed"),
    (dict(r_min=-1, r_max=10), "non-negative"),
    (dict(r_min=1, r_max=10, eta_min=10, eta_max=-10), "eta_max must exceed"),
    (dict(r_min=1, r_max=10, eta_min=-200, eta_max=10), r"\[-180, 180\]"),
    (dict(r_min=1, r_max=10, r_bin=0), "bin sizes must be positive"),
    (dict(r_min=1, r_max=10, n_peaks=0), "n_peaks must be"),
])
def test_channel_validation(kw, msg):
    with pytest.raises(ValueError, match=msg):
        Channel(**kw)


def test_peak_centres_must_lie_inside_the_window():
    with pytest.raises(ValueError, match="outside the radius window"):
        Channel(105, 125, n_peaks=1, peak_centres=(200.0,))


def test_peak_centres_count_must_match_n_peaks():
    with pytest.raises(ValueError, match="n_peaks is"):
        Channel(105, 125, n_peaks=3, peak_centres=(110.0, 120.0))


def test_channels_from_legacy_params_cross_product():
    """The legacy scripts pair every radius with every eta."""
    params = {"rads": [118, 237], "etas": [0], "Rwidth": 10, "etaWidth": 180,
              "RBinSize": 0.25, "EtaBinSize": 3}
    chans = channels_from_legacy_params(params)
    assert len(chans) == 2
    assert (chans[0].r_min, chans[0].r_max) == (108, 128)


def test_legacy_multipeak_assigns_centres_to_their_own_window():
    """Rcenters are listed globally; each must land in the window it falls in."""
    params = {"rads": [118, 250], "etas": [0], "Rwidth": 10, "multipeak": 1,
              "Rcenters": [118, 243, 248, 254]}
    chans = channels_from_legacy_params(params)
    assert chans[0].peak_centres == (118.0,)
    assert set(chans[1].peak_centres) == {243.0, 248.0, 254.0}
    assert chans[1].n_peaks == 3


def test_legacy_params_without_rads_is_an_error():
    with pytest.raises(ValueError, match="no 'rads'"):
        channels_from_legacy_params({"etas": [0]})


# --------------------------------------------------------------- geometry
LEGACY_PS_DT = """
# 2022 U3O8 geometry
NrPixelsY 1475
NrPixelsZ 1679
ImTransOpt 2
px 172
BytesPerPx 4
HeadSize 8192
Wavelength 0.136994    # 55.618 keV (Ho-edge)
Lsd 1071098.336
BC 790.3118888 864.5394861
ty -0.07913612768
tz 0.05595587143
p0 0.000603981776
p1 -0.000076495455
p2 0.0000487658065
p3 -13.20896847
RhoD 150000
startNr 161
endNr 215
nFrames 1441
startOme 180.25
omeStep -0.25
BadRotation 1
rads 195
Rwidth 10
etas 0
etaWidth 180
RBinSize 0.25
EtaBinSize 3
"""


def test_parse_legacy_params(tmp_path):
    p = tmp_path / "ps_dt.txt"
    p.write_text(LEGACY_PS_DT)
    got = parse_legacy_params(p)
    assert got["Lsd"] == pytest.approx(1071098.336)
    assert got["BC"] == [pytest.approx(790.3118888), pytest.approx(864.5394861)]
    assert got["NrPixelsY"] == 1475
    assert got["BadRotation"] == 1
    assert got["startOme"] == pytest.approx(180.25)
    assert got["omeStep"] == pytest.approx(-0.25)


def test_geometry_from_legacy_params_and_energy(tmp_path):
    p = tmp_path / "ps_dt.txt"
    p.write_text(LEGACY_PS_DT)
    geo = geometry_from_legacy_params(p)
    assert geo.lsd_um == pytest.approx(1071098.336)
    assert geo.bc_y_px == pytest.approx(790.3118888)
    assert geo.n_pixels_z == 1679
    assert geo.distortion["p3"] == pytest.approx(-13.20896847)
    # The file's own comment says 55.618 keV; the wavelength says otherwise.
    assert geo.energy_kev == pytest.approx(90.5, abs=0.1)


def test_geometry_missing_key_names_it(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("NrPixelsY 1475\nNrPixelsZ 1679\n")
    with pytest.raises(KeyError, match="Lsd"):
        geometry_from_legacy_params(p)


def test_from_calibration_rejects_a_foreign_object():
    from midas_dt.geometry import from_calibration

    class NotAResult:
        pass

    with pytest.raises(TypeError, match="AutoCalibrationResult"):
        from_calibration(NotAResult())


# ------------------------------------------------- legacy p0..p3 -> v2 terms
def test_legacy_distortion_maps_by_the_canonical_permutation(tmp_path):
    """p0..p3 are NOT positional in v2.

    midas_distortion.V1_TO_V2_DISTORTION says p0->a2, p1->a4, p2->iso_R2,
    p3->phi4. The natural wrong guess is p0->iso_R2, which would silently
    distort every radius. Pinned here so the mapping cannot drift.
    """
    pytest.importorskip("midas_distortion")
    p = tmp_path / "ps_dt.txt"
    p.write_text(LEGACY_PS_DT)
    geo = geometry_from_legacy_params(p)
    v2 = geo.v2_distortion()
    assert v2["a2"] == pytest.approx(0.000603981776)      # p0
    assert v2["a4"] == pytest.approx(-0.000076495455)     # p1
    assert v2["iso_R2"] == pytest.approx(0.0000487658065) # p2
    assert v2["phi4"] == pytest.approx(-13.20896847)      # p3
    assert "iso_R4" not in v2 and "iso_R6" not in v2      # p5, p4 not supplied


def test_no_distortion_gives_no_terms():
    geo = DTGeometry(lsd_um=1e6, bc_y_px=1.0, bc_z_px=1.0, px_um=172,
                     n_pixels_y=10, n_pixels_z=10, wavelength_a=0.137)
    assert geo.v2_distortion() == {}


def test_energy_from_wavelength():
    geo = DTGeometry(lsd_um=1e6, bc_y_px=1.0, bc_z_px=1.0, px_um=172,
                     n_pixels_y=10, n_pixels_z=10, wavelength_a=0.136994)
    assert geo.energy_kev == pytest.approx(90.5, abs=0.1)
