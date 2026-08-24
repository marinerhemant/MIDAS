"""Reading a scan's own record instead of hand-counting its frames.

The load-bearing test here is
:func:`test_the_derived_layout_matches_the_hand_counted_prepare_script` — the
parser has to reproduce four indices that a human counted by eye and that were
then used to make every NMC811 reconstruction in the tree. Those numbers are an
external answer; the parser cannot be tuned to them without the arithmetic
actually closing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_tomo.scanrecord import parse_metastr, read_scan_record

# --------------------------------------------------------------- fixtures
# Transcribed verbatim from the real files (read 2026-08-23):
#   bt_1id_jun25b: metadata/bt_1id_jun25b/nmc811s5tomo1/nmc811s5tomo1_TomoFastScan.dat
#   bt_1id_jul26:  new_data/bt_1id_jul26/tomo_Ce_ht525_s2/tomo_Ce_ht525_s2_TomoFastScan.dat

BT_1ID_JUN25B = """Beginning of tomography scan at: Sun Jun 22 01:44:28 2025
Path: /home/beams/S1IDUSER/mnt/s1c/bt_1id_jun25b/tomo
Image prefix: nmc811s5tomo1/nmc811s5tomo1
Exposure time (s): 0.2
Energy (keV): 51.93
tomo_metastr: D~100.000000mm, FLIR-GH1, 5X, 0.708 um/px, aero axis, left handed --- scantype=3, scan start/end/step angles=-180.000000/180.000000/0.100000 deg, exp. time=0.200000 sec, sumimg#=1, shiftmotor=samXE, sampleshift=-5.000 mm at rot. angle aero=-180.000 deg, WF#=10, DF#=10, Proj#=3601, settling=0.000000 sec, padding time=0.100000 sec, Gap time for fastsweep=0.070000 sec --- prefix=nmc811s5tomo1/nmc811s5tomo1, tomoparfile=bt_1id_jun25b_tomopar.par, TomoScanFile=nmc811s5tomo1/nmc811s5tomo1_TomoFastScan.dat
White field image sequence starts at: 7317
Number of white field images: 10
Start collecting front white...Done.
First image sequence number: 7317
Last image sequence number: 10947
Number of images taken in this scan: 3631
Total time elapsed:  17.331  min
"""

BT_1ID_JUL26 = """Beginning of tomography scan at: Thu Jul 30 15:53:17 2026
Path: /home/beams/S1IDUSER/mnt/s1c/bt_1id_jul26/tomo
Image prefix: tomo_Ce_ht525_s2/tomo_Ce_ht525_s2
Exposure time (s): 0.5
Energy (keV): 95
tomo_metastr: D~100.000000mm, FLIR-GH1, 5X, 0.69 um/px, aero axis, left handed, imgZE=100.000000 --- scantype=3, scan start/end/step angles=-180.000000/180.000000/0.200000 deg, exp. time=0.500000 sec, sumimg#=1, shiftmotor=samZE, sampleshift=-7.000 mm at rot. angle aero=-90.000 deg, WF#=10, DF#=10, Proj#=1801, settling=0.000000 sec, padding time=0.100000 sec, Gap time for fastsweep=0.070000 sec --- prefix=tomo_Ce_ht525_s2/tomo_Ce_ht525_s2, tomoparfile=bt_1id_jul26_tomopar.par, TomoScanFile=tomo_Ce_ht525_s2/tomo_Ce_ht525_s2_TomoFastScan.dat
White field image sequence starts at: 74
Number of white field images: 10
Start collecting front white...Done.
First image sequence number: 74
Last image sequence number: 1904
Number of images taken in this scan: 1831
"""


def _write(tmp_path, text, name="scan_TomoFastScan.dat"):
    p = tmp_path / name
    p.write_text(text)
    return p


# ------------------------------------------------------- THE gate

def test_the_derived_layout_matches_the_hand_counted_prepare_script(tmp_path):
    """``prepare_data_nmc811_s5_tomo1.py`` hard-codes four indices a human
    counted. Every NMC811 reconstruction in the tree was made from them, so
    they are an independent answer, not one this parser can be fitted to.

        startNrWhite  = 7317
        startNrData   = 7327
        startNrWhite2 = 10928
        startNrDark   = 10938
    """
    s = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    got = {b.role: b.start for b in s.blocks}
    assert got == {
        "front_white": 7317,
        "projections": 7327,
        "back_white": 10928,
        "dark": 10938,
    }
    assert s.block("projections").count == 3601
    assert all(b.count == 10 for b in s.blocks if b.role != "projections")
    assert s.last_image == 10947          # closes on the recorded last image


def test_the_pixel_size_is_the_one_tomocupy_args_gets_wrong(tmp_path):
    """0.708 um from the scan's own record; tomocupy_args.yml says 1.17, which
    is the PointGrey value. A 1.65x error here is 4.5x in every volume."""
    s = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    assert s.pixel_size_um == 0.708
    assert s.detector == "FLIR-GH1" and s.magnification == "5X"
    assert read_scan_record(_write(tmp_path, BT_1ID_JUL26, "b.dat")).pixel_size_um == 0.69


# ---------------------------------------------------------- the metastr

def test_an_inserted_field_does_not_shift_the_positional_reads(tmp_path):
    """bt_1id_jul26's optics section carries an extra ``imgZE=100.000000`` that
    bt_1id_jun25b's does not. A positional comma reader would take that as the
    handedness and shift everything after it."""
    a = read_scan_record(_write(tmp_path, BT_1ID_JUN25B, "a.dat"))
    b = read_scan_record(_write(tmp_path, BT_1ID_JUL26, "b.dat"))
    for s in (a, b):
        assert s.handedness == "left"
        assert s.rotation_axis == "aero"
        assert s.detector == "FLIR-GH1"
        assert s.propagation_mm == 100.0


def test_angles_and_counts_come_off_the_metastr(tmp_path):
    s = read_scan_record(_write(tmp_path, BT_1ID_JUL26))
    assert (s.omega_start_deg, s.omega_end_deg, s.omega_step_deg) == (-180.0, 180.0, 0.2)
    assert (s.n_projections, s.n_white, s.n_dark) == (1801, 10, 10)
    assert s.energy_kev == 95.0 and s.exposure_s == 0.5
    assert s.shift_motor == "samZE"


def test_parse_metastr_on_the_bare_string():
    d = parse_metastr(
        "tomo_metastr: D~50.000000mm, PointGrey, 7.5X, 0.7813 um/px, "
        "RAMS3 axis, right handed --- WF#=5, DF#=3, Proj#=900, "
        "scan start/end/step angles=0.000000/180.000000/0.200000 deg"
    )
    assert d["pixel_size_um"] == 0.7813
    assert d["magnification"] == "7.5X"
    assert d["handedness"] == "right"
    assert d["rotation_axis"] == "rams3"
    assert d["detector"] == "PointGrey"
    assert (d["n_white"], d["n_dark"], d["n_projections"]) == (5, 3, 900)


# ------------------------------------------------------------ the omega sign

def test_the_aero_sign_is_applied_and_matches_the_beamline_driver(tmp_path):
    """``midas_tomo_python_nmc811_s5_tomo1.py`` uses
    ``np.arange(180, -180.1, -0.1)`` against a metastr that reads
    ``-180/180/0.100``. That is exactly the negation, so the rule is confirmed
    by the beamline's own script and not only by the standing convention."""
    s = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    th = s.thetas()
    assert th.size == 3601
    assert th[0] == pytest.approx(180.0)
    assert th[-1] == pytest.approx(-180.0)
    np.testing.assert_allclose(th, np.arange(180, -180.05, -0.1), atol=1e-9)


def test_the_recorded_angles_are_still_reachable(tmp_path):
    s = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    raw = s.thetas(apply_aero_sign=False)
    assert raw[0] == pytest.approx(-180.0) and raw[-1] == pytest.approx(180.0)
    assert s.is_aero


def test_a_non_aero_stage_is_left_alone(tmp_path):
    txt = BT_1ID_JUN25B.replace("aero axis", "RAMS3 axis")
    s = read_scan_record(_write(tmp_path, txt))
    assert not s.is_aero
    np.testing.assert_allclose(s.thetas(), s.thetas(apply_aero_sign=False))


# --------------------------------------------------------------- refusals

def test_no_metastr_is_refused_rather_than_defaulted(tmp_path):
    txt = "\n".join(l for l in BT_1ID_JUN25B.splitlines() if "tomo_metastr" not in l)
    with pytest.raises(ValueError, match="no tomo_metastr"):
        read_scan_record(_write(tmp_path, txt))


def test_frame_counts_that_do_not_close_are_refused(tmp_path):
    """The failure this guards: an off-by-one block boundary averages
    projections into the flat field, which is silent and ruins every slice."""
    txt = BT_1ID_JUN25B.replace("Proj#=3601", "Proj#=3590")
    with pytest.raises(ValueError, match="frame counts do not close"):
        read_scan_record(_write(tmp_path, txt))


def test_a_layout_that_overshoots_the_last_image_is_refused(tmp_path):
    txt = BT_1ID_JUN25B.replace("Last image sequence number: 10947",
                            "Last image sequence number: 10999")
    with pytest.raises(ValueError, match="derived layout ends at"):
        read_scan_record(_write(tmp_path, txt))


def test_a_scan_with_no_back_white_is_handled_not_assumed(tmp_path):
    """bt_1id_jun25b has flats at both ends; not every scan does. The count
    arithmetic decides, so a front-white-only scan parses correctly instead of
    stealing ten projections."""
    txt = (BT_1ID_JUN25B
           .replace("Number of images taken in this scan: 3631",
                    "Number of images taken in this scan: 3621")
           .replace("Last image sequence number: 10947",
                    "Last image sequence number: 10937"))
    s = read_scan_record(_write(tmp_path, txt))
    assert not s.has_back_white
    assert {b.role for b in s.blocks} == {"front_white", "projections", "dark"}
    assert s.block("dark").start == 10928
    assert s.block("projections").count == 3601


def test_missing_totals_are_refused_because_nothing_can_be_cross_checked(tmp_path):
    txt = "\n".join(
        l for l in BT_1ID_JUN25B.splitlines()
        if "Last image sequence" not in l and "Number of images taken" not in l
    )
    with pytest.raises(ValueError, match="cannot be cross-checked"):
        read_scan_record(_write(tmp_path, txt))


# -------------------------------------------------------------- the extras

def test_frame_paths_use_a_local_root_not_the_acquisition_path(tmp_path):
    """The recorded ``Path:`` is the acquisition machine's view and is usually
    not mounted where the analysis runs."""
    s = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    paths = s.frame_paths("dark", root="/scratch/here")
    assert len(paths) == 10
    assert paths[0].as_posix() == \
        "/scratch/here/nmc811s5tomo1/nmc811s5tomo1_010938.tif"
    assert paths[-1].name == "nmc811s5tomo1_010947.tif"


def test_asking_for_a_block_that_does_not_exist_names_the_ones_that_do(tmp_path):
    txt = (BT_1ID_JUN25B
           .replace("Number of images taken in this scan: 3631",
                    "Number of images taken in this scan: 3621")
           .replace("Last image sequence number: 10947",
                    "Last image sequence number: 10937"))
    s = read_scan_record(_write(tmp_path, txt))
    with pytest.raises(KeyError, match="front_white"):
        s.frame_paths("back_white")


def test_fov_needs_the_column_count_because_the_record_has_no_crop(tmp_path):
    """The bt_1id_jun25b .raw is a 128x128 crop out of a much larger frame, so a
    self-computed FOV would be wrong by whatever the crop was."""
    s = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    assert s.fov_um(128) == pytest.approx(90.624)
    with pytest.raises(ValueError):
        s.fov_um(0)


def test_provenance_names_where_the_pixel_size_did_NOT_come_from(tmp_path):
    s = read_scan_record(_write(tmp_path, BT_1ID_JUN25B))
    p = s.provenance()
    assert "NOT tomocupy_args.yml" in p["pixel_size_source"]
    assert p["omega_sign_applied"] == "negated (aero)"
    assert "not a TOMO_IN_PLANE assignment" in p["handedness_note"]
    assert p["blocks"]["projections"] == [7327, 3601]
