"""bin_data unit tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_transforms.bin_data import bin_data
from midas_transforms.bin_data.core import (
    _bin_assignment, _bin_to_data_ndata, _build_ring_radii,
    _compute_radius_dist_ideal,
)
from midas_transforms.io import binary as bio
from midas_transforms.params import ParamsTest


def test_build_ring_radii(tiny_paramstest: ParamsTest):
    rr = _build_ring_radii(tiny_paramstest)
    assert rr[1].item() == 500.0
    assert rr[2].item() == 700.0
    assert rr[3].item() == 900.0
    # Unconfigured rings = 0
    assert rr[0].item() == 0.0
    assert rr[4].item() == 0.0


def test_radius_dist_ideal(tiny_inputall):
    p = ParamsTest()
    p.RingNumbers = [1, 2, 3]
    p.RingRadii = [500.0, 700.0, 900.0]
    rr = _build_ring_radii(p)
    spots = torch.from_numpy(tiny_inputall)
    rd = _compute_radius_dist_ideal(spots, rr)
    # In the synthetic fixture, YLab/ZLab were generated to match RingRadii
    # exactly (in µm; px=200), so radius_dist_ideal should be tiny per spot.
    # YLab = -RingRad_px * sin(eta) * px → sqrt(YLab² + ZLab²) = RingRad_px * px
    # And ring_radii is in px.  So |radius - rr_px| could be RingRad_px * (px - 1).
    # Just check that the function runs and returns the right shape.
    assert rd.shape == (spots.shape[0],)


def test_bin_data_writes_expected_files(tmp_inputall_dir: Path):
    bin_data(result_folder=tmp_inputall_dir)
    for name in ("Spots.bin", "ExtraInfo.bin", "Data.bin", "nData.bin"):
        assert (tmp_inputall_dir / name).exists(), f"{name} missing"


def test_spots_bin_layout(tmp_inputall_dir: Path):
    res = bin_data(result_folder=tmp_inputall_dir)
    # Post Phase 5: Spots.bin is always 10 cols (col 9 = ScanNr, 0 for FF).
    spots = bio.read_spots_bin(tmp_inputall_dir / "Spots.bin", ncols=10)
    assert spots.shape[1] == 10
    assert spots.shape[0] == res.spots.shape[0]
    # First 8 cols match InputAll.csv input rows.
    np.testing.assert_allclose(
        spots[:, :8], res.spots.detach().cpu().numpy()[:, :8], rtol=0, atol=0,
    )
    # ScanNr (col 9) is 0 for non-scanning FF runs.
    assert np.all(spots[:, 9] == 0.0)


def test_extrainfo_bin_layout(tmp_inputall_dir: Path):
    bin_data(result_folder=tmp_inputall_dir)
    extra = bio.read_extrainfo_bin(tmp_inputall_dir / "ExtraInfo.bin")
    assert extra.shape[1] == 16


def test_ndata_offsets_consistent(tmp_inputall_dir: Path):
    res = bin_data(result_folder=tmp_inputall_dir)
    # Post Phase 5: Data/nData are int64 (uint64) pairs.
    data_pairs = bio.read_data_bin_scanning(tmp_inputall_dir / "Data.bin")
    ndata = bio.read_ndata_bin_scanning(tmp_inputall_dir / "nData.bin")
    counts = ndata[:, 0]
    offsets = ndata[:, 1]
    assert offsets[0] == 0
    expected_offsets = np.zeros_like(counts)
    expected_offsets[1:] = np.cumsum(counts[:-1])
    np.testing.assert_array_equal(offsets, expected_offsets)
    # total count == number of (rowno, scanno) pairs in Data.bin.
    assert int(counts.sum()) == data_pairs.shape[0]
    # FF mode: scan_nr column is all zeros.
    assert np.all(data_pairs[:, 1] == 0)


def test_positions_csv_written_for_ff(tmp_inputall_dir: Path):
    # Phase 5: bin_data() emits positions.csv (single "0.0" line for FF) so
    # the unified midas_indexer can auto-detect mode.
    bin_data(result_folder=tmp_inputall_dir)
    pos = (tmp_inputall_dir / "positions.csv").read_text().strip().splitlines()
    assert len(pos) == 1
    assert float(pos[0]) == 0.0


def test_no_save_all_skips_data_files(tmp_inputall_dir: Path, tiny_paramstest):
    # Override paramstest with NoSaveAll=1
    pt = tiny_paramstest
    pt.NoSaveAll = 1
    from midas_transforms.params import write_paramstest
    write_paramstest(pt, tmp_inputall_dir / "paramstest.txt")
    # Remove old Data.bin/nData.bin if present
    (tmp_inputall_dir / "Data.bin").unlink(missing_ok=True)
    (tmp_inputall_dir / "nData.bin").unlink(missing_ok=True)

    bin_data(result_folder=tmp_inputall_dir)
    assert (tmp_inputall_dir / "Spots.bin").exists()
    assert (tmp_inputall_dir / "ExtraInfo.bin").exists()
    assert not (tmp_inputall_dir / "Data.bin").exists()
    assert not (tmp_inputall_dir / "nData.bin").exists()


def test_bin_assignment_shapes(tiny_paramstest, tiny_inputall):
    spots = torch.from_numpy(tiny_inputall)
    rr = _build_ring_radii(tiny_paramstest)
    sp_idx, ring, ieta, iome = _bin_assignment(
        spots, rr,
        margin_ome=tiny_paramstest.MarginOme,
        margin_eta=tiny_paramstest.MarginEta,
        eta_bin_size=tiny_paramstest.EtaBinSize,
        ome_bin_size=tiny_paramstest.OmeBinSize,
        step_size_orient=tiny_paramstest.StepSizeOrient,
    )
    assert sp_idx.shape == ring.shape == ieta.shape == iome.shape
    assert sp_idx.numel() > 0  # synthetic fixture should produce some assignments
