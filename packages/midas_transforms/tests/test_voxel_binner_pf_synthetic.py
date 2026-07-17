"""voxel_binner PF-mode synthetic tests.

Hand-crafted 3-scan × 4-spot fixture. Verifies:
- Per-scan tagging (col 9 of Spots.bin == scan_nr).
- Global sort key (ring, omega, eta) matches the C qsort comparator.
- SpotID renumbering preserves a stable 1..N mapping.
- ``voxel_scan_pos.bin`` sidecar is written with the right dtype/length.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from midas_transforms import bin_data_scanning, bin_data_unified
from midas_transforms.bin_data.voxel_binner import VoxelBinDataResult
from midas_transforms.io import binary as bio
from midas_transforms.io import csv as csv_io
from midas_transforms.params import ParamsTest, write_paramstest


def _make_paramstest() -> ParamsTest:
    p = ParamsTest()
    p.Wavelength = 0.18
    p.Lsd = 1_000_000.0
    p.px = 200.0
    p.MarginOme = 1.0
    p.MarginEta = 500.0
    p.EtaBinSize = 5.0
    p.OmeBinSize = 5.0
    p.StepSizeOrient = 0.2
    p.NoSaveAll = 0
    p.RingNumbers = [1, 2, 3]
    p.RingRadii = [500.0, 700.0, 900.0]
    p.LatticeConstant = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    p.SpaceGroup = 225
    return p


def _make_one_csv(rows18: np.ndarray) -> np.ndarray:
    """Build an 18-col CSV row block. Cols 0..13 = base spot, 14/15 =
    dummies, 16/17 = trailing fitting cols."""
    n = rows18.shape[0]
    out = np.zeros((n, 18), dtype=np.float64)
    out[:, :14] = rows18[:, :14]
    out[:, 14] = rows18[:, 14]   # dummy 0
    out[:, 15] = rows18[:, 15]   # dummy 1
    out[:, 16] = rows18[:, 16]
    out[:, 17] = rows18[:, 17]
    return out


@pytest.fixture
def tmp_pf_scan_dir(tmp_path: Path) -> Path:
    """Three per-scan CSVs, each with 4 spots."""
    p = _make_paramstest()
    write_paramstest(p, tmp_path / "paramstest.txt")

    # Per-scan synthetic data. For each scan, write 4 spots covering rings 1,2,3.
    # GrainRadius > 0.0001 so they all pass the filter.
    # Layout cols 0..13: YLab, ZLab, Omega, GrainRadius, SpotID,
    #                    RingNumber, Eta, Ttheta, OmegaIni,
    #                    YOrigDetCor, ZOrigDetCor, YRawPx, ZRawPx,
    #                    IntegratedIntensity
    # Cols 14..15: dummies
    # Cols 16..17: RawSumIntensity, FitRMSE (or similar)
    rng = np.random.default_rng(seed=42)
    for scan_nr in range(3):
        rows = []
        # Per-scan ring/eta grid (deliberately varies in omega across scans).
        for spot_idx, (ring_nr, ring_rad) in enumerate([(1, 500.0), (2, 700.0),
                                                          (3, 900.0), (1, 500.0)]):
            # Eta varies per spot; omega varies per scan.
            eta = (-90.0 + 45.0 * spot_idx + 30.0 * scan_nr) % 360 - 180
            omega = -50.0 + 20.0 * scan_nr + 7.0 * spot_idx
            ttheta = math.degrees(math.atan2(ring_rad * p.px, p.Lsd))
            yl = -ring_rad * math.sin(math.radians(eta)) * p.px
            zl = ring_rad * math.cos(math.radians(eta)) * p.px
            spot_id = scan_nr * 100 + spot_idx + 1   # per-scan unique
            grain_radius = 5.0 + 0.1 * spot_idx
            r = np.zeros(18, dtype=np.float64)
            r[0] = yl
            r[1] = zl
            r[2] = omega
            r[3] = grain_radius
            r[4] = spot_id
            r[5] = ring_nr
            r[6] = eta
            r[7] = ttheta
            r[8] = omega
            r[9] = yl
            r[10] = zl
            r[11] = yl
            r[12] = zl
            r[13] = 1000.0 + 5.0 * spot_idx
            r[14] = 0.0
            r[15] = 0.0
            r[16] = 1000.0 + 5.0 * spot_idx     # RawSumIntensity (close)
            r[17] = 0.01                         # FitRMSE
            rows.append(r)
        rows18 = np.array(rows, dtype=np.float64)
        csv_io.write_inputall_extra_csv(
            tmp_path / f"InputAllExtraInfoFittingAll{scan_nr}.csv",
            rows18,
        )
    return tmp_path


def test_pf_mode_writes_all_files(tmp_pf_scan_dir: Path):
    res = bin_data_scanning(
        result_folder=tmp_pf_scan_dir,
        n_scans=3,
        scan_positions=np.array([-4.0, 0.0, 4.0]),
    )
    for name in (
        "Spots.bin", "ExtraInfo.bin",
        "Data.bin", "nData.bin",
        "IDsMergedScanning.csv",
        "voxel_scan_pos.bin", "positions.csv",
    ):
        assert (tmp_pf_scan_dir / name).exists(), f"missing {name}"
    assert res.n_scans == 3
    assert res.id_map.shape[0] == 12   # 3 scans × 4 spots
    assert res.scan_positions is not None
    np.testing.assert_array_equal(res.scan_positions, np.array([-4.0, 0.0, 4.0]))


def test_pf_spots_bin_has_10_cols(tmp_pf_scan_dir: Path):
    bin_data_scanning(
        result_folder=tmp_pf_scan_dir,
        n_scans=3,
        scan_positions=np.array([-4.0, 0.0, 4.0]),
    )
    spots = bio.read_spots_bin(tmp_pf_scan_dir / "Spots.bin", ncols=10)
    assert spots.shape[1] == 10
    assert spots.shape[0] == 12
    # Col 9 must be a valid scanNr (0, 1, or 2) for every row.
    scan_nrs = spots[:, 9].astype(np.int64)
    assert set(scan_nrs.tolist()) == {0, 1, 2}


def test_pf_spotid_renumbered_1_to_n(tmp_pf_scan_dir: Path):
    bin_data_scanning(
        result_folder=tmp_pf_scan_dir,
        n_scans=3,
        scan_positions=np.array([-4.0, 0.0, 4.0]),
    )
    spots = bio.read_spots_bin(tmp_pf_scan_dir / "Spots.bin", ncols=10)
    # SpotID is col 4. Must be 1..12 in sorted order.
    np.testing.assert_array_equal(
        spots[:, 4].astype(np.int64),
        np.arange(1, 13, dtype=np.int64),
    )


def test_pf_global_sort_ring_omega_eta(tmp_pf_scan_dir: Path):
    bin_data_scanning(
        result_folder=tmp_pf_scan_dir,
        n_scans=3,
        scan_positions=np.array([-4.0, 0.0, 4.0]),
    )
    spots = bio.read_spots_bin(tmp_pf_scan_dir / "Spots.bin", ncols=10)
    # The C qsort key is (col5 ring, col2 omega, col6 eta). Going down the
    # rows, the (ring, omega, eta) tuple must be monotonically non-decreasing
    # under the same key.
    keys = list(zip(spots[:, 5], spots[:, 2], spots[:, 6]))
    for k1, k2 in zip(keys[:-1], keys[1:]):
        assert k1 <= k2, f"sort violated: {k1} > {k2}"


def test_pf_id_map_csv_matches_result(tmp_pf_scan_dir: Path):
    res = bin_data_scanning(
        result_folder=tmp_pf_scan_dir,
        n_scans=3,
        scan_positions=np.array([-4.0, 0.0, 4.0]),
    )
    # Read the CSV back and compare to the in-memory ID map.
    import csv
    csv_path = tmp_pf_scan_dir / "IDsMergedScanning.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [(int(r["NewID"]), int(r["OrigID"]), int(r["ScanNr"])) for r in reader]
    expected = [tuple(int(x) for x in row) for row in res.id_map.tolist()]
    assert rows == expected


def test_pf_dispatch_via_unified(tmp_pf_scan_dir: Path):
    """``bin_data_unified(scan_positions=...)`` routes to PF mode."""
    res = bin_data_unified(
        result_folder=tmp_pf_scan_dir,
        scan_positions=np.array([-4.0, 0.0, 4.0]),
        n_scans=3,
    )
    assert (tmp_pf_scan_dir / "voxel_scan_pos.bin").exists()
    # PF result type is VoxelBinDataResult, FF is BinDataResult.
    assert isinstance(res, VoxelBinDataResult)
    assert res.scan_positions is not None
