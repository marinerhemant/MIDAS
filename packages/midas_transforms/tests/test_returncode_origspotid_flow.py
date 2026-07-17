"""N2 + E3 regression tests: ``ReturnCode`` and ``OrigSpotID`` flow.

Three SpotID spaces exist: peaksearch/merge (Result_StartNr), calc_radius
(renumbered 1..N, spots near two rings DUPLICATED), and fit_setup
(re-sorted + renumbered). Joining on SpotID across spaces silently pairs
random spots — this invalidated two emerson analyses before it was
caught. The fix threads two APPENDED columns end-to-end:

  merge Result col 17 = ReturnCode (peakfit col 18; sticky-first-nonzero)
  radius cols 24/25   = OrigSpotID (merge-space), ReturnCode
  InputAllExtra 18/19 = OrigSpotID, ReturnCode

Binary strides (Spots.bin / ExtraInfo.bin) are never widened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_transforms.io import csv as csv_io
from midas_transforms.merge.core import (
    COL_ETA, COL_II, COL_IMAX, COL_OMEGA, COL_RADIUS, COL_RETCODE,
    COL_SPOTID, COL_YCEN, COL_ZCEN, N_PEAK_COLS,
    _finalise_row, _merge_frames, _seed_current_from_frame,
)


def _peak(spot_id, omega, y, z, ii=100.0, radius=500.0, eta=45.0, retcode=0.0):
    row = np.zeros(N_PEAK_COLS, dtype=np.float64)
    row[COL_SPOTID] = spot_id
    row[COL_II] = ii
    row[COL_OMEGA] = omega
    row[COL_YCEN] = y
    row[COL_ZCEN] = z
    row[COL_IMAX] = ii / 2
    row[COL_RADIUS] = radius
    row[COL_ETA] = eta
    row[COL_RETCODE] = retcode
    return row


# ---------------------------------------------------------------------------
# merge: ReturnCode carried + sticky-first-nonzero
# ---------------------------------------------------------------------------


def test_merge_carries_returncode_sticky_nonzero():
    # Frame 1: one clean peak; frame 2: same spot, FAILED fit (retcode 3);
    # frame 3: same spot, clean again. The merged spot must record 3.
    f1 = np.stack([_peak(1, 10.0, 100.0, 200.0, retcode=0.0)])
    f2 = np.stack([_peak(1, 10.25, 100.2, 200.1, retcode=3.0)])
    f3 = np.stack([_peak(1, 10.5, 100.1, 200.0, retcode=0.0)])
    out, merge_map = _merge_frames([f1, f2, f3], overlap_length=5.0)
    assert out.shape == (1, 18)
    assert out[0, 17] == 3.0, "sticky-first-nonzero ReturnCode lost in merge"
    # A clean merged spot stays 0.
    g = np.stack([_peak(7, 10.0, -300.0, 50.0, eta=-120.0)])
    out2, _ = _merge_frames([g], overlap_length=5.0)
    assert out2[0, 17] == 0.0


def test_result_csv_round_trip_and_legacy_pad(tmp_path: Path):
    out, _ = _merge_frames(
        [np.stack([_peak(1, 10.0, 100.0, 200.0, retcode=2.0)])], overlap_length=5.0)
    f = tmp_path / "Result_StartNr_1_EndNr_10.csv"
    csv_io.write_result_csv(f, out)
    head = f.read_text().splitlines()[0]
    assert head.split()[-1] == "ReturnCode"
    back = csv_io.read_result_csv(f)
    assert back.shape[1] == 18 and back[0, 17] == 2.0
    # Legacy 17-col file → padded with -1 (unknown), never 0 (fit OK).
    csv_io.write_result_csv(tmp_path / "legacy.csv", out[:, :17])
    legacy = csv_io.read_result_csv(tmp_path / "legacy.csv")
    assert legacy.shape[1] == 18 and legacy[0, 17] == -1.0


# ---------------------------------------------------------------------------
# calc_radius: OrigSpotID survives renumber + two-ring duplication
# ---------------------------------------------------------------------------


def _radius_from_result(result18: np.ndarray, ring_radii_um, ring_numbers):
    import torch
    from midas_transforms.radius.core import _filter_and_compute_radius
    out, _, _ = _filter_and_compute_radius(
        torch.from_numpy(result18),
        torch.tensor(ring_numbers, dtype=torch.int64),
        torch.tensor(ring_radii_um, dtype=torch.float64),
        width_px=2000.0, px_um=200.0, Lsd_um=1_000_000.0,
        OmegaStep=0.25, Hbeam=2000.0, Rsample=1000.0, Vsample=0.0,
        DiscModel=0, DiscArea=0.0, n_frames=1440,
    )
    return out.numpy()


def test_radius_carries_origspotid_through_duplication():
    # Two spots: merge-space IDs 11 and 42. Spot 42's radius (600 px ·
    # 200 µm = 120000 µm) sits within Width of BOTH rings → duplicated,
    # renumbered — col 0 becomes 1..3 while col 24 keeps 11/42/42.
    res = np.zeros((2, 18), dtype=np.float64)
    res[0, 0], res[1, 0] = 11.0, 42.0
    res[0, 1] = res[1, 1] = 100.0            # IntInt
    res[0, 12], res[1, 12] = 500.0, 600.0    # Radius px (=100000 / 120000 um)
    res[0, 13], res[1, 13] = 45.0, -120.0    # Eta
    res[0, 17], res[1, 17] = 0.0, 5.0        # ReturnCode
    # Rings at 100000 (spot 11) and 119000+121000 um — spot 42 (120000 um)
    # is within Width=2000 of BOTH of the last two.
    out = _radius_from_result(res, [100_000.0, 119_000.0, 121_000.0], [1, 2, 3])
    assert out.shape[1] == 26
    np.testing.assert_array_equal(out[:, 0], np.arange(1, out.shape[0] + 1))
    dup = out[out[:, 24] == 42.0]
    assert dup.shape[0] == 2, "two-ring spot must appear once per ring"
    np.testing.assert_array_equal(dup[:, 25], [5.0, 5.0])
    assert (out[out[:, 24] == 11.0][:, 25] == 0.0).all()


# ---------------------------------------------------------------------------
# fit_setup: appended cols land in InputAllExtra; binary strides unchanged
# ---------------------------------------------------------------------------


def test_fit_setup_spotsinfo_carries_appended_cols():
    from midas_transforms.fit_setup.core import (
        _radius_csv_to_spotsinfo, _sort_per_ring_renumber,
    )
    rad = np.zeros((3, 26), dtype=np.float64)
    rad[:, 0] = [1, 2, 3]
    rad[:, 2] = [30.0, 10.0, 20.0]     # Omega (drives the re-sort)
    rad[:, 13] = [2, 2, 2]             # RingNr
    rad[:, 24] = [11.0, 42.0, 99.0]    # OrigSpotID
    rad[:, 25] = [0.0, 5.0, 0.0]       # ReturnCode
    si = _radius_csv_to_spotsinfo(rad)
    assert si.shape == (3, 12)
    sorted_si, _, _ = _sort_per_ring_renumber(si, [2])
    # Re-sorted by omega (10, 20, 30) + renumbered 1..3; OrigSpotID must
    # follow its row.
    np.testing.assert_array_equal(sorted_si[:, 0], [1, 2, 3])
    np.testing.assert_array_equal(sorted_si[:, 10], [42.0, 99.0, 11.0])
    np.testing.assert_array_equal(sorted_si[:, 11], [5.0, 0.0, 0.0])
    # Legacy 24-col radius input → -1 (unknown).
    si_legacy = _radius_csv_to_spotsinfo(rad[:, :24])
    assert (si_legacy[:, 10] == -1.0).all() and (si_legacy[:, 11] == -1.0).all()


def test_inputall_extra_readers_and_binary_stride(tmp_path: Path):
    from midas_transforms.bin_data import bin_data
    from midas_transforms.params import ParamsTest, write_paramstest
    import math

    # 20-col InputAllExtra + 8-col InputAll for one ring-2 spot pair.
    p = ParamsTest()
    p.Wavelength = 0.18; p.Lsd = 1_000_000.0; p.px = 200.0
    p.MarginOme = 1.0; p.MarginEta = 500.0
    p.EtaBinSize = 5.0; p.OmeBinSize = 5.0; p.StepSizeOrient = 0.2
    p.RingNumbers = [2]; p.RingRadii = [700.0]; p.RingToIndex = 2
    p.BeamSize = 2000.0
    p.LatticeConstant = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    write_paramstest(p, tmp_path / "paramstest.txt")

    rows8 = []
    extra20 = []
    for i, eta in enumerate((45.0, -120.0)):
        rr = 700.0
        yl = -rr * math.sin(math.radians(eta)) * p.px
        zl = rr * math.cos(math.radians(eta)) * p.px
        rows8.append([yl, zl, 10.0 + i, 5.0, i + 1, 2, eta,
                      math.degrees(math.atan2(rr * p.px, p.Lsd))])
        e = np.zeros(20)
        e[:8] = rows8[-1]
        e[18] = 40.0 + i          # OrigSpotID
        e[19] = float(i)          # ReturnCode
        extra20.append(e)
    csv_io.write_inputall_csv(tmp_path / "InputAll.csv",
                              np.array(rows8, dtype=np.float64))
    csv_io.write_inputall_extra_csv(tmp_path / "InputAllExtraInfoFittingAll.csv",
                                    np.array(extra20, dtype=np.float64))

    # Reader contract: base reader strips; appended reader returns them.
    base = csv_io.read_inputall_extra_csv(tmp_path / "InputAllExtraInfoFittingAll.csv")
    assert base.shape[1] == 18
    b, orig, ret = csv_io.read_inputall_extra_csv_appended(
        tmp_path / "InputAllExtraInfoFittingAll.csv")
    np.testing.assert_array_equal(orig, [40.0, 41.0])
    np.testing.assert_array_equal(ret, [0.0, 1.0])

    # ExtraInfo.bin stride must stay 16 float64 per row.
    bin_data(result_folder=tmp_path)
    n_rows = 2
    raw = np.fromfile(tmp_path / "ExtraInfo.bin", dtype=np.float64)
    assert raw.size == n_rows * 16, (
        f"ExtraInfo.bin stride changed: {raw.size} doubles for {n_rows} rows"
    )


def test_finalise_row_shape():
    cur, _ = _seed_current_from_frame(
        np.stack([_peak(9, 1.0, 10.0, 20.0, retcode=7.0)]), frame_nr=0)
    row = _finalise_row(cur[0], spot_id=1)
    assert row.shape == (18,) and row[17] == 7.0
