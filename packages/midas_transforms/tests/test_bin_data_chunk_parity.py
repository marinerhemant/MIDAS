"""FF/PF bit-parity gate for the chunked pair expansion (N3).

``_bin_assignment`` expands (spot x eta-bin x ome-bin) pairs in spot chunks
bounded by ``MIDAS_BIN_PAIR_CHUNK`` (default 2**28 pairs) to avoid OOM on
dense pf-HEDM layers. The chunked path must be BIT-IDENTICAL to the
single-chunk (legacy all-at-once) path on both binner flavours:

- FF ``bin_data``            (the midas-transforms Pipeline / FF stage path)
- PF ``bin_data_scanning``   (per-ring pack + (rowno, scanno) pair writer)

A tiny chunk budget forces one-spot-per-chunk, maximising chunk-boundary
coverage on a small fixture. FF binning has regressed silently before
(pipeline 0.2.0 zero-byte Data.bin), hence byte-level hashes on every
output file, not just array equality.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from midas_transforms import bin_data, bin_data_scanning
from midas_transforms.io import csv as csv_io
from midas_transforms.params import ParamsTest, write_paramstest

FF_OUTPUTS = ("Spots.bin", "ExtraInfo.bin", "Data.bin", "nData.bin")
PF_OUTPUTS = ("Spots.bin", "ExtraInfo.bin", "Data.bin", "nData.bin",
              "IDsMergedScanning.csv", "voxel_scan_pos.bin", "positions.csv")


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _hash_outputs(d: Path, names) -> dict:
    out = {}
    for fn in names:
        assert (d / fn).exists(), f"missing output {fn} in {d}"
        assert (d / fn).stat().st_size > 0, f"zero-byte output {fn} in {d}"
        out[fn] = _hash_file(d / fn)
    return out


# ---------------------------------------------------------------------------
# FF flavour
# ---------------------------------------------------------------------------


def test_ff_bin_data_chunked_bit_identical(
    tmp_inputall_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """FF ``bin_data``: forced multi-chunk == single-chunk, byte-for-byte."""
    baseline = tmp_path / "single_chunk"
    chunked = tmp_path / "multi_chunk"
    for d in (baseline, chunked):
        d.mkdir()
        for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv",
                   "paramstest.txt"):
            shutil.copy(tmp_inputall_dir / fn, d / fn)

    # Effectively unchunked (far above the fixture's total pair count).
    monkeypatch.setenv("MIDAS_BIN_PAIR_CHUNK", str(2 ** 62))
    bin_data(result_folder=baseline)

    # 1-pair budget -> one spot per chunk (the >=1-spot floor kicks in),
    # maximising chunk boundaries.
    monkeypatch.setenv("MIDAS_BIN_PAIR_CHUNK", "1")
    bin_data(result_folder=chunked)

    assert _hash_outputs(baseline, FF_OUTPUTS) == _hash_outputs(chunked, FF_OUTPUTS)


def test_ff_bin_data_default_equals_unchunked(
    tmp_inputall_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The default budget (env unset) must also match the unchunked path."""
    baseline = tmp_path / "unchunked"
    default = tmp_path / "default"
    for d in (baseline, default):
        d.mkdir()
        for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv",
                   "paramstest.txt"):
            shutil.copy(tmp_inputall_dir / fn, d / fn)

    monkeypatch.setenv("MIDAS_BIN_PAIR_CHUNK", str(2 ** 62))
    bin_data(result_folder=baseline)

    monkeypatch.delenv("MIDAS_BIN_PAIR_CHUNK", raising=False)
    bin_data(result_folder=default)

    assert _hash_outputs(baseline, FF_OUTPUTS) == _hash_outputs(default, FF_OUTPUTS)


# ---------------------------------------------------------------------------
# PF flavour (per-ring pack + scanning writer)
# ---------------------------------------------------------------------------


def _write_pf_fixture(d: Path, n_scans: int = 3) -> np.ndarray:
    """3-scan x 4-spot PF fixture (rings 1,2,3; mirrors the voxel_binner
    synthetic tests)."""
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
    write_paramstest(p, d / "paramstest.txt")

    for scan_nr in range(n_scans):
        rows = []
        for spot_idx, (ring_nr, ring_rad) in enumerate(
            [(1, 500.0), (2, 700.0), (3, 900.0), (1, 500.0)]
        ):
            eta = (-90.0 + 45.0 * spot_idx + 30.0 * scan_nr) % 360 - 180
            omega = -50.0 + 20.0 * scan_nr + 7.0 * spot_idx
            ttheta = math.degrees(math.atan2(ring_rad * p.px, p.Lsd))
            yl = -ring_rad * math.sin(math.radians(eta)) * p.px
            zl = ring_rad * math.cos(math.radians(eta)) * p.px
            r = np.zeros(18, dtype=np.float64)
            r[0], r[1], r[2] = yl, zl, omega
            r[3] = 5.0 + 0.1 * spot_idx          # GrainRadius
            r[4] = scan_nr * 100 + spot_idx + 1  # SpotID
            r[5], r[6], r[7] = ring_nr, eta, ttheta
            r[8] = omega
            r[9], r[10], r[11], r[12] = yl, zl, yl, zl
            r[13] = 1000.0 + 5.0 * spot_idx
            r[16] = 1000.0 + 5.0 * spot_idx
            r[17] = 0.01
            rows.append(r)
        csv_io.write_inputall_extra_csv(
            d / f"InputAllExtraInfoFittingAll{scan_nr}.csv",
            np.array(rows, dtype=np.float64),
        )
    return np.array([-4.0, 0.0, 4.0][:n_scans], dtype=np.float64)


def test_pf_bin_data_scanning_chunked_bit_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """PF ``bin_data_scanning``: forced multi-chunk == single-chunk,
    byte-for-byte, through the per-ring pack + concat."""
    baseline = tmp_path / "single_chunk"
    chunked = tmp_path / "multi_chunk"
    hashes = {}
    for d, budget in ((baseline, str(2 ** 62)), (chunked, "1")):
        d.mkdir()
        scan_positions = _write_pf_fixture(d)
        monkeypatch.setenv("MIDAS_BIN_PAIR_CHUNK", budget)
        bin_data_scanning(
            result_folder=d,
            n_scans=scan_positions.shape[0],
            scan_positions=scan_positions,
        )
        hashes[d.name] = _hash_outputs(d, PF_OUTPUTS)

    assert hashes["single_chunk"] == hashes["multi_chunk"]


def test_pf_ndata_counts_consistent_under_chunking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """nData bin counts must sum to the number of Data.bin pairs, and the
    offsets must be the exclusive cumsum of the counts (ring-major
    traversal), regardless of chunking."""
    monkeypatch.setenv("MIDAS_BIN_PAIR_CHUNK", "1")
    scan_positions = _write_pf_fixture(tmp_path)
    bin_data_scanning(
        result_folder=tmp_path,
        n_scans=scan_positions.shape[0],
        scan_positions=scan_positions,
    )
    ndata = np.fromfile(tmp_path / "nData.bin", dtype=np.uint64).reshape(-1, 2)
    data = np.fromfile(tmp_path / "Data.bin", dtype=np.uint64).reshape(-1, 2)
    counts, offsets = ndata[:, 0].astype(np.int64), ndata[:, 1].astype(np.int64)
    assert counts.sum() == data.shape[0]
    np.testing.assert_array_equal(
        offsets, np.concatenate([[0], np.cumsum(counts)[:-1]])
    )
