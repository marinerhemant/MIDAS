"""6-ID-C readers must refuse every ambiguity that has already produced a wrong map.

The two layouts are written here exactly as they appear on disk (file naming, motor-table
columns, repeats), with the traps planted: a motor log from another beamtime, file names
that do not sort in acquisition order, a hot first repeat, a lost file, a zero-byte frame,
and point-major frames read as repeat-major.
"""
import csv
import math
import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from midas_dfxm import (check_frame_order, example_rocking_scan, find_motor_tables,  # noqa: E402
                        load_6idc_scan, read_motor_table, reduce_rocking)

pytestmark = pytest.mark.unit


def _curves(M, H=32, W=32, seed=0, amp=3000.0, ped=400.0):
    rng = np.random.default_rng(seed)
    x = 15.838 + 0.005 * np.arange(M)
    yy, xx = np.mgrid[0:H, 0:W]
    centre = x[M // 2] + 0.020 * (xx / (W - 1) - 0.5)
    sig = 0.020 / 2.3548
    lam = amp * np.exp(-0.5 * ((x[:, None, None] - centre[None]) / sig) ** 2) + ped
    return rng.poisson(lam).astype(np.uint16), x


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _named_scan(root, frames, x, name=lambda i, th: f"NaMnO2_P_10_20-2_{2 * th - 1.426:.4f}_{th:.4f}_-0.812_ 1.tif",
                rows_per_frame=1, seed=0):
    d = os.path.join(root, "Plumb_Dec_data_S006", "S006")
    os.makedirs(d, exist_ok=True)
    order = np.random.default_rng(seed).permutation(len(x))       # creation order is irrelevant
    for i in order:
        tifffile.imwrite(os.path.join(d, name(i, x[i])), frames[i])
    mdir = os.path.join(root, "Plumb_Dec_motors", "motors")
    os.makedirs(mdir, exist_ok=True)
    rows = []
    for i, th in enumerate(np.repeat(x, rows_per_frame)):
        rows.append([i, 30.25, f"{th:.4f}", 0.0, 0.0, 0.0, 1.5])
    table = os.path.join(mdir, "S006_motorInfo.csv")
    _write_csv(table, ["image number", "two-theta", "theta", "chi", "phi", "nu", "sample x"], rows)
    return d, table


def test_named_layout_confirms_the_order_from_a_filename_field(tmp_path):
    frames, x = _curves(41)
    d, table = _named_scan(str(tmp_path), frames, x)
    scan = load_6idc_scan(d, table)
    assert scan.scan_type == "tilt" and scan.axes == ("theta",)
    assert any("file order confirmed" in n for n in scan.notes)
    np.testing.assert_array_equal(scan.frames, frames.astype(np.float32))
    assert check_frame_order(scan).consistent


def test_named_layout_refuses_a_log_from_another_beamtime(tmp_path):
    frames, x = _curves(41)
    d, _ = _named_scan(str(tmp_path), frames, x)
    july = os.path.join(str(tmp_path), "motors")
    os.makedirs(july)
    _write_csv(os.path.join(july, "S006_motorInfo.csv"),
               ["image number", "two-theta", "theta"],
               [[i, 22.874, 8.597 + 0.010 * (i // 50)] for i in range(41 * 50)])
    with pytest.raises(ValueError, match="2050 rows for 41 frames"):
        load_6idc_scan(d, os.path.join(july, "S006_motorInfo.csv"))


def test_two_candidate_tables_are_refused_not_picked(tmp_path):
    frames, x = _curves(21)
    d, _ = _named_scan(str(tmp_path), frames, x)
    other = os.path.join(str(tmp_path), "motors")
    os.makedirs(other)
    _write_csv(os.path.join(other, "S006_motorInfo.csv"), ["theta"], [[1.0]] * 21)
    assert len(find_motor_tables(d)) == 2
    with pytest.raises(ValueError, match="found 2 motor tables"):
        load_6idc_scan(d)


def test_names_that_do_not_sort_in_row_order_are_reordered_by_the_angle(tmp_path):
    frames, x = _curves(21)
    ids = np.random.default_rng(7).permutation(21) * 13 + 100        # natural sort scrambles
    d, table = _named_scan(str(tmp_path), frames, x,
                           name=lambda i, th: f"frame_{ids[i]}_{th:.4f}.tif")
    scan = load_6idc_scan(d, table)
    assert any("REORDERED" in n for n in scan.notes)
    np.testing.assert_array_equal(scan.frames, frames.astype(np.float32))


def test_a_zero_byte_frame_is_refused_by_name(tmp_path):
    frames, x = _curves(11)
    d, table = _named_scan(str(tmp_path), frames, x)
    victim = sorted(os.listdir(d))[3]
    open(os.path.join(d, victim), "w").close()
    with pytest.raises(ValueError, match="zero-byte"):
        load_6idc_scan(d, table)


def _indexed_scan(root, npts=31, R=4, hot=0.043, seed=0, H=24, W=24, amp=2000.0, ped=300.0,
                  two_theta=False):
    """scans_raw/DFXM_S190/data_NNNNN.tif, point-major, plus its motor_files table.

    ``two_theta=True`` logs tth = 2 th (a theta-2theta scan), the Dec-2025 ``th2th_DFXM`` macro.
    """
    rng = np.random.default_rng(seed)
    x = 15.4732 + 0.010 * np.arange(npts)
    yy, xx = np.mgrid[0:H, 0:W]
    centre = x[npts // 2] + 0.040 * (xx / (W - 1) - 0.5) + 0.02 * (yy > H // 2)
    lam = amp * np.exp(-0.5 * ((x[:, None, None] - centre[None]) / 0.012) ** 2) + ped
    reps = rng.poisson(lam[:, None], size=(npts, R, H, W)).astype(np.float64)
    reps[:, 0] *= 1.0 + hot
    d = os.path.join(root, "data", "scans_raw", "DFXM_S190")
    os.makedirs(d)
    for pt in range(npts):
        for r in range(R):
            tifffile.imwrite(os.path.join(d, f"data_{pt * R + r:05d}.tif"),
                             np.clip(np.rint(reps[pt, r]), 0, 65535).astype(np.uint16))
    mdir = os.path.join(root, "data", "motor_files")
    os.makedirs(mdir)
    table = os.path.join(mdir, "Ba122_Cu_3p4_motor_information_S190_0.5s.csv")
    _write_csv(table, ["Num", "tth", "th", "chi", "phi", "mono", "symz", "symy", "symx"],
               [[i, f"{2 * th:.6f}" if two_theta else 25.5254, f"{th:.6f}", 0.0, 0.0, 5.674611,
                 -9.5, 1.26, -1.6954] for i, th in enumerate(x)])
    return d, table, np.rint(reps)


def test_indexed_layout_reads_point_major_and_drops_a_hot_first_repeat(tmp_path):
    d, table, reps = _indexed_scan(str(tmp_path))
    assert find_motor_tables(d) == [os.path.abspath(table)]
    scan = load_6idc_scan(d)                               # table found automatically
    assert scan.scan_type == "tilt" and scan.axes == ("th",)
    assert any("DROPPED" in n for n in scan.notes)
    assert scan.n_repeats == 3 and scan.halves is not None
    want = reps[:, 1:].mean(1)
    np.testing.assert_allclose(scan.frames, want, rtol=1e-3)
    assert abs(scan.meta["energy_keV_si111"] - 19.995) < 0.002
    assert check_frame_order(scan).consistent


def test_repeat_major_misreading_of_point_major_data_fails_the_order_check(tmp_path):
    d, table, _ = _indexed_scan(str(tmp_path), R=2, hot=0.0)
    wrong = load_6idc_scan(d, table, order="repeat")
    assert not check_frame_order(wrong).consistent
    right = load_6idc_scan(d, table, order="point")
    assert check_frame_order(right).consistent


def test_a_lost_file_is_refused(tmp_path):
    d, table, _ = _indexed_scan(str(tmp_path), npts=11, R=2, hot=0.0)
    os.remove(os.path.join(d, "data_00021.tif"))           # the last frame: count no longer divides
    with pytest.raises(ValueError, match="not a whole number of repeats"):
        load_6idc_scan(d, table)
    os.remove(os.path.join(d, "data_00005.tif"))           # a middle frame: indices have a gap
    with pytest.raises(ValueError, match="not contiguous"):
        load_6idc_scan(d, table)


def test_reading_and_reducing_end_to_end(tmp_path):
    d, table, _ = _indexed_scan(str(tmp_path), R=2, hot=0.0)
    scan = load_6idc_scan(d, table)
    maps = reduce_rocking(scan)
    assert maps.split == "repeat parity"
    assert maps.lit.mean() > 0.9 and np.isfinite(maps.sigma_global)
    grad = np.nanmedian(np.diff(maps.value, axis=1))       # planted 40 mdeg across 23 px
    assert abs(grad - 40.0 / 23.0) < 0.3, grad


def test_theta_two_theta_indexed_scan_reads_and_reduces_in_microstrain(tmp_path):
    """The Dec-2025 th2th_DFXM layout: tth moves with th; the map is a d-spacing change."""
    d, table, _ = _indexed_scan(str(tmp_path), R=2, hot=0.0, two_theta=True)
    scan = load_6idc_scan(d, table)
    assert scan.scan_type == "strain" and scan.axes == ("tth",)
    np.testing.assert_allclose(scan.coordinate, 15.4732 + 0.010 * np.arange(31), atol=1e-6)
    assert check_frame_order(scan).consistent
    cot = 1.0 / np.tan(np.radians(scan.two_theta_deg / 2.0))
    for window in ("peak", "fixed"):
        maps = reduce_rocking(scan, window=window)
        assert maps.unit == "microstrain" and maps.lit.mean() > 0.9
        grad = np.nanmedian(np.diff(maps.value, axis=1))   # planted +40 mdeg of theta across 23 px
        want = -cot * (0.040 / 23.0) * np.pi / 180.0 * 1e6
        assert abs(grad - want) < 0.2 * abs(want), (window, grad, want)
    ex = example_rocking_scan("single", two_theta=True)
    assert ex.scan_type == "strain" and reduce_rocking(ex).unit == "microstrain"


def test_two_repeats_are_never_auto_dropped(tmp_path):
    """With 2 repeats, dropping the first leaves one: no halves, so no repeat check at all.

    A misread scan then passed on the adjacent-frame test alone (verify 6b35197c71b9).
    """
    d, table, _ = _indexed_scan(str(tmp_path), R=2, hot=0.043)
    scan = load_6idc_scan(d, table)
    assert scan.n_repeats == 2 and scan.halves is not None
    assert not scan.meta["first_repeat_dropped"]
    assert any("first repeat" in n and "kept" in n for n in scan.notes)
    explicit = load_6idc_scan(d, table, drop_first_repeat=True)
    assert explicit.n_repeats == 1 and explicit.halves is None


def test_weak_signal_repeat_major_misread_is_caught(tmp_path):
    """A repeat-major misread of point-major data with 3 repeats and a weak signal.

    With the repeat test's pixels chosen on the averaged stack this passed as CONSISTENT in
    96 of 100 synthetic scans (verify 6b35197c71b9, statistics lens).
    """
    passed = 0
    for seed in range(5):
        root = os.path.join(str(tmp_path), f"s{seed}")
        d, table, _ = _indexed_scan(root, R=3, hot=0.0, amp=200.0, ped=300.0, seed=seed)
        right = check_frame_order(load_6idc_scan(d, table))
        assert right.consistent, right.message
        wrong = check_frame_order(load_6idc_scan(d, table, order="repeat"))
        passed += int(wrong.consistent)
    assert passed == 0, passed


@pytest.mark.xfail(strict=True, reason="known limit, documented in check_frame_order: a brighter "
                   "first repeat, with a point count sharing a factor with the repeat count, lets "
                   "a repeat-major misread pass")
def test_hot_first_repeat_aliasing_does_not_pass_a_misread(tmp_path):
    """A brighter first repeat, with a point count that shares a factor with the repeat count.

    Read repeat-major, both halves of a misread point then hold first-repeat frames, and the
    repeat test passes the misread: 40 of 40 at 51 points, 3 repeats and peak 50 over pedestal
    300 (51 = 3 x 17). Removing each frame's level from the statistic did not fix it; it let
    strong-signal misreads through instead. This is a strict xfail that records the limit: if a
    later change makes it pass, the suite reports XPASS, and the mark should come off. The
    correct reading must still be confirmed, or a scan with no contrast would pass for nothing.
    """
    right_ok = misread_passed = 0
    for seed in range(4):
        root = os.path.join(str(tmp_path), f"s{seed}")
        d, table, _ = _indexed_scan(root, npts=51, R=3, hot=0.043, amp=50.0, ped=300.0,
                                    H=64, W=64, seed=seed)
        right_ok += int(check_frame_order(load_6idc_scan(d, table)).consistent)
        misread_passed += int(check_frame_order(load_6idc_scan(d, table, order="repeat")).consistent)
    assert right_ok >= 3, right_ok
    assert misread_passed == 0, misread_passed


def test_read_motor_table_drops_empty_columns_and_keeps_numbers(tmp_path):
    p = os.path.join(str(tmp_path), "t.csv")
    _write_csv(p, ["theta", "note", "chi"], [["8.5", "x", ""], ["8.6", "y", "0.1"]])
    t = read_motor_table(p)
    assert set(t) == {"theta", "chi"}
    assert math.isnan(t["chi"][0]) and t["theta"][1] == 8.6
