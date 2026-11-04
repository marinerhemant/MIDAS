"""CBF reader: header numbers, regular-sweep check, ordering by frame number, refusal to wrap integers. fabio cannot WRITE CBF loops, so frames are
served by a stand-in for ``fabio.open`` that returns the object layout fabio produces for a real file (checked on the Zenodo 7265997 frames:
``header['loop_'] = [[names, [row dicts]]]``)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

fabio = pytest.importorskip("fabio")

from midas_sx.io_frames import list_cbf_frames, read_cbf_scan          # noqa: E402


def _fake(tmp_path, monkeypatch, n=4, step=1.0, start=0.0, bad_at=None, big=False):
    frames = {}
    for i in range(n):
        p = tmp_path / f"scan_2_{i + 1:03d}.cbf"
        p.write_bytes(b"")
        data = (np.arange(64 * 48).reshape(48, 64) % 200 + i).astype(np.int32)
        if big and i == 1:
            data[0, 0] = 70000
        ang = start + i * step + (0.3 if bad_at == i else 0.0)
        hdr = {"_diffrn_radiation_wavelength.wavelength": "0.708457", "_diffrn_measurement.sample_detector_distance": "85.9285",
               "_diffrn_data_frame.details": "HEADER_BYTES=  512;\r\nPIXEL_SIZE=0.1024;\r\nBIN=2x2;", "_array_intensities.gain": "2.54",
               "_array_intensities.overload": "65535", "_diffrn_detector.type": "ADSC QUANTUM210", "_diffrn_scan_frame.integration_time": "1.0",
               "loop_": [[[b"_diffrn_scan_axis.axis_id", b"_diffrn_scan_axis.angle_start"],
                          [{b"_diffrn_scan_axis.axis_id": b"GONIOMETER_PHI", b"_diffrn_scan_axis.angle_start": f"{ang:.2f}".encode()},
                           {b"_diffrn_scan_axis.axis_id": b"DETECTOR_Z", b"_diffrn_scan_axis.angle_start": b"0.00"}]]]}
        frames[str(p)] = SimpleNamespace(data=data, header=hdr)
    monkeypatch.setattr(fabio, "open", lambda path: frames[str(path)])
    return frames


def test_header_numbers_and_regular_sweep(tmp_path, monkeypatch):
    _fake(tmp_path, monkeypatch)
    stack, h = read_cbf_scan(str(tmp_path))
    assert stack.shape == (4, 48, 64) and stack.dtype == np.uint16
    assert h.n_frames == 4 and h.wavelength_A == pytest.approx(0.708457) and h.distance_mm == pytest.approx(85.9285)
    assert h.pixel_mm == pytest.approx(0.1024) and h.gain == pytest.approx(2.54) and h.overload == 65535
    assert h.angle_start == 0.0 and h.angle_step == pytest.approx(1.0) and h.angle_steps_regular


def test_irregular_sweep_is_flagged(tmp_path, monkeypatch):
    _fake(tmp_path, monkeypatch, bad_at=2)
    _, h = read_cbf_scan(str(tmp_path))
    assert not h.angle_steps_regular and h.notes


def test_frames_are_sorted_by_number_not_lexically(tmp_path):
    for n in (2, 10, 1):
        (tmp_path / f"s_{n}.cbf").write_bytes(b"")
    assert [p.split("/")[-1] for p in list_cbf_frames(str(tmp_path))] == ["s_1.cbf", "s_2.cbf", "s_10.cbf"]
    with pytest.raises(FileNotFoundError):
        list_cbf_frames(str(tmp_path / "nothing"))


def test_values_above_the_dtype_range_raise(tmp_path, monkeypatch):
    _fake(tmp_path, monkeypatch, big=True)
    with pytest.raises(ValueError, match="do not fit"):
        read_cbf_scan(str(tmp_path))
    stack, h = read_cbf_scan(str(tmp_path), dtype=np.int32)
    assert stack.max() == 70000 and h.n_undefined == 0


def test_negative_voxels_are_invalid_flags_set_to_zero_and_counted(tmp_path, monkeypatch):
    frames = _fake(tmp_path, monkeypatch)
    f2 = tmp_path / "scan_2_002.cbf"
    d = fabio.open(str(f2)).data
    d[3, 4] = -1
    d[5, 6] = -1
    stack, h = read_cbf_scan(str(tmp_path))
    assert h.n_undefined == 2 and stack[1, 3, 4] == 0 and stack[1, 5, 6] == 0 and stack.min() == 0
    assert any("below zero" in n for n in h.notes)


def test_missing_scan_axis_raises(tmp_path, monkeypatch):
    _fake(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="angle_start"):
        read_cbf_scan(str(tmp_path), scan_axis="GONIOMETER_OMEGA")


# ------------------------------------------------------------------ Bruker SFRM
def _fake_sfrm(tmp_path, monkeypatch, starts, incr="1.000000", axis="2", big=None, mask_name=None):
    from midas_sx.io_frames import read_sfrm_scan  # noqa: F401
    frames = {}
    for i, a in enumerate(starts):
        p = tmp_path / f"mo_run_01_{i + 1:04d}.sfrm"
        p.write_bytes(b"")
        data = (np.arange(48 * 32).reshape(48, 32) % 90 + 10 + i).astype(np.int32)
        if big is not None and i == 1:
            data[0, 0] = big
        hdr = {"WAVELEN": "0.710730         0.709300         0.713590         0.632290", "DISTANC": "2.796100                           3.800100",
               "DETTYPE": "CMOS-PHOTONII        36.954918   1.004000    0 0.425000    0.035000    1", "ZOOM": "0.500000               0.500000               1.000000",
               "CENTER": "384.300000       512.800000       384.300000       512.800000", "ANGLES": f"9.684000         {a:.6f}        299.977800       319.681000",
               "START": f"{a:.6f}", "INCREME": incr, "AXIS": axis, "ELAPSDR": "4.000000", "MODEL": "Booster with KappaStage", "LOWTEMP": "1 TEMP: 130.00", "TYPE": "FAST SCAN FRAME"}
        frames[str(p)] = SimpleNamespace(data=data, header=hdr)
    if mask_name:
        p = tmp_path / mask_name
        p.write_bytes(b"")
        m = np.ones((48, 32), np.uint8); m[3, 4] = 0
        frames[str(p)] = SimpleNamespace(data=m, header={"TYPE": "ACTIVE MASK"})
    monkeypatch.setattr(fabio, "open", lambda path: frames[str(path)])
    return frames


def test_sfrm_header_numbers_and_wrap_through_360(tmp_path, monkeypatch):
    from midas_sx.io_frames import read_sfrm_scan
    _fake_sfrm(tmp_path, monkeypatch, [358.0, 359.0, 0.0, 1.0])                      # the run crosses 360
    stack, h = read_sfrm_scan(str(tmp_path), prefix="mo_run_01_")
    assert stack.shape == (4, 48, 32) and stack.dtype == np.uint32
    assert h.wavelength_A == pytest.approx(0.71073) and h.distance_mm == pytest.approx(38.001) and h.scan_axis == "OMEGA"
    assert h.pixel_mm == pytest.approx(10.0 / 36.954918 * 0.5) and h.angle_start == 358.0 and h.angle_step == 1.0 and h.angle_steps_regular
    assert h.extra["detector_2theta_deg"] == pytest.approx(9.684) and h.extra["chi_deg"] == pytest.approx(319.681) and h.extra["centre_xy_px"] == (384.3, 512.8)
    assert list(h.extra["angle_starts_unwrapped"]) == pytest.approx([358.0, 359.0, 360.0, 361.0])


def test_sfrm_irregular_run_is_flagged_and_phi_axis_named(tmp_path, monkeypatch):
    from midas_sx.io_frames import read_sfrm_scan
    _fake_sfrm(tmp_path, monkeypatch, [10.0, 11.0, 12.5, 13.5], axis="3")
    _, h = read_sfrm_scan(str(tmp_path))
    assert not h.angle_steps_regular and h.scan_axis == "PHI" and h.notes


def test_sfrm_large_counts_fit_the_default_dtype_and_overflow_raises(tmp_path, monkeypatch):
    from midas_sx.io_frames import read_sfrm_scan
    _fake_sfrm(tmp_path, monkeypatch, [0.0, 1.0, 2.0], big=518255)                  # a real Bragg pixel of the Zenodo set (> 65535)
    stack, _ = read_sfrm_scan(str(tmp_path))
    assert stack.max() == 518255
    with pytest.raises(ValueError, match="do not fit"):
        read_sfrm_scan(str(tmp_path), dtype=np.uint16)


def test_sfrm_prefix_keeps_one_run_and_the_mask_frame_is_dead_pixels(tmp_path, monkeypatch):
    from midas_sx.io_frames import list_sfrm_frames, read_sfrm_mask
    _fake_sfrm(tmp_path, monkeypatch, [0.0, 1.0], mask_name="mo_run_xa_01_0001.sfrm")
    (tmp_path / "other_02_0001.sfrm").write_bytes(b"")
    assert len(list_sfrm_frames(str(tmp_path), prefix="mo_run_01_")) == 2
    assert all("xa" not in p for p in list_sfrm_frames(str(tmp_path)))               # the _xa_01_0001 frame is part of no run
    m = read_sfrm_mask(str(tmp_path / "mo_run_xa_01_0001.sfrm"))
    assert m.dtype == bool and m[3, 4] and m.sum() == 1
    with pytest.raises(ValueError, match="not an active-mask"):
        read_sfrm_mask(str(tmp_path / "mo_run_01_0001.sfrm"))


# ------------------------------------------------------------------ Dectris Pilatus CBF
PILATUS_TEXT = """# Detector: PILATUS 2M, S/N 24-0107 Diamond
# 2024-03-11T19:23:14.765
# Pixel_size 172e-6 m x 172e-6 m
# Silicon sensor, thickness 0.000320 m
# Exposure_time 0.1964000 s
# Tau = 124.0e-09 s
# Count_cutoff 791927 counts
# Wavelength 0.68890 A
# Detector_distance 0.16000 m
# Beam_xy (740.35, 862.41) pixels
# Start_angle {start:.4f} deg.
# Angle_increment 0.2000 deg.
# Detector_2theta 20.0000 deg.
# Polarization 0.990
# Kappa 0.0000 deg.
# Phi 120.0000 deg.
# Omega {start:.4f} deg.
# Chi -54.7000 deg.
# Oscillation_axis OMEGA
"""


def _fake_pilatus(tmp_path, monkeypatch, n=4, step_shift=0.0, big=None, name="pgw_02_{:05d}.cbf"):
    frames = {}
    for i in range(n):
        p = tmp_path / name.format(i + 1)
        p.write_bytes(b"")
        data = np.zeros((40, 30), np.int64)
        data[5 + i, 7] = 50 + i
        data[:, 10] = -1                                      # a module gap
        data[20, 20] = -2                                     # an excluded pixel
        if big is not None and i == 1:
            data[3, 3] = big
        start = -155.0 + 0.2 * i + (step_shift if i == 2 else 0.0)
        frames[str(p)] = SimpleNamespace(data=data, header={"_array_data.header_convention": "PILATUS_1.2", "_array_data.header_contents": PILATUS_TEXT.format(start=start)})
    monkeypatch.setattr(fabio, "open", lambda path: frames[str(path)])


def test_pilatus_header_numbers_units_and_dead_mask(tmp_path, monkeypatch):
    from midas_sx.io_frames import read_pilatus_cbf_scan
    _fake_pilatus(tmp_path, monkeypatch)
    stack, h = read_pilatus_cbf_scan(str(tmp_path))
    assert stack.shape == (4, 40, 30) and stack.dtype == np.uint32 and stack.min() == 0
    assert h.wavelength_A == pytest.approx(0.6889) and h.distance_mm == pytest.approx(160.0) and h.pixel_mm == pytest.approx(0.172)
    assert h.scan_axis == "OMEGA" and h.angle_start == -155.0 and h.angle_step == pytest.approx(0.2) and h.angle_steps_regular and h.overload == 791927
    ex = h.extra
    assert ex["beam_xy_px"] == (740.35, 862.41) and ex["detector_2theta_deg"] == 20.0 and ex["polarization"] == pytest.approx(0.99) and ex["chi_deg"] == -54.7
    assert ex["dead_mask"].dtype == bool and ex["dead_mask"].sum() == 40 + 1 and ex["dead_mask"][20, 20] and ex["dead_mask"][:, 10].all()
    assert h.n_undefined == 4 * 41 and stack[1, 6, 7] == 51


def test_pilatus_irregular_sweep_and_overflow(tmp_path, monkeypatch):
    from midas_sx.io_frames import read_pilatus_cbf_scan
    _fake_pilatus(tmp_path, monkeypatch, step_shift=0.05)
    _, h = read_pilatus_cbf_scan(str(tmp_path))
    assert not h.angle_steps_regular and h.notes
    for f in list(tmp_path.glob("*.cbf")):
        f.unlink()
    _fake_pilatus(tmp_path, monkeypatch, big=2 ** 33)
    with pytest.raises(ValueError, match="do not fit"):
        read_pilatus_cbf_scan(str(tmp_path))


def test_pilatus_rejects_a_header_that_is_not_pilatus(tmp_path, monkeypatch):
    from midas_sx.io_frames import read_pilatus_cbf_scan
    _fake_pilatus(tmp_path, monkeypatch, n=2)
    for p in tmp_path.glob("*.cbf"):
        fabio.open(str(p)).header["_array_data.header_contents"] = "# nothing\n"
    with pytest.raises(ValueError, match="not a Pilatus"):
        read_pilatus_cbf_scan(str(tmp_path))
    _fake_pilatus(tmp_path, monkeypatch, n=2, name="other_{:03d}.cbf")
    assert read_pilatus_cbf_scan(str(tmp_path), prefix="other_")[1].n_frames == 2


def test_pilatus_keep_invalid_keeps_the_flags_for_build_mask(tmp_path, monkeypatch):
    from midas_defect.ingest import build_mask
    from midas_sx.io_frames import read_pilatus_cbf_scan
    _fake_pilatus(tmp_path, monkeypatch)
    stack, h = read_pilatus_cbf_scan(str(tmp_path), keep_invalid=True)
    assert stack.dtype == np.int32 and stack.min() == -2 and (stack[0, :, 10] == -1).all()
    m = build_mask(stack, low_count_threshold=0.0, grow=0).mask                    # a zero-background detector: no 'low count' pixels, only the flagged ones
    assert m.sum() == 41 and m[20, 20] and m[:, 10].all()
    assert build_mask(stack, low_count_threshold=20.0, grow=0).mask.mean() > 0.9    # the default threshold would mask the whole detector: it is meant for a pedestal
