"""Tests for IndexBest.bin / IndexBestFull.bin output writers."""

import os

import numpy as np
import pytest
import torch

from midas_index.compute.constants import MAX_N_HKLS
from midas_index.io import open_output_files, close_output_files, write_seed_record
from midas_index.io.output import (
    INDEX_BEST_FULL_RECORD_BYTES,
    INDEX_BEST_RECORD_BYTES,
    INDEX_BEST_RECORD_DOUBLES,
    write_full_record,
)
from midas_index.result import SeedResult


def _make_seed(spot_id=1, dtype=torch.float64):
    R = torch.eye(3, dtype=dtype) * 1.0
    R[0, 1] = 0.5  # break symmetry to detect column-major ordering bugs
    pos = torch.tensor([10.0, 20.0, 30.0], dtype=dtype)
    matched = torch.tensor([7, 11, 13], dtype=torch.int64)
    return SeedResult(
        spot_id=spot_id,
        best_or_mat=R,
        best_pos=pos,
        n_matches=3,
        n_t_spots=42,
        n_t_frac_calc=40,
        frac_matches=0.075,
        avg_ia=0.123,
        matched_ids=matched,
    )


def test_open_truncates_to_full_size_on_block_zero(tmp_path):
    fd_best, fd_full = open_output_files(tmp_path, n_total_seeds=10, block_nr=0)
    try:
        st_best = os.fstat(fd_best)
        st_full = os.fstat(fd_full)
    finally:
        close_output_files(fd_best, fd_full)

    assert st_best.st_size == 10 * INDEX_BEST_RECORD_BYTES
    assert st_full.st_size == 10 * INDEX_BEST_FULL_RECORD_BYTES


def test_open_does_not_truncate_for_later_blocks(tmp_path):
    # First, block 0 creates and zero-fills.
    fd_best, fd_full = open_output_files(tmp_path, n_total_seeds=10, block_nr=0)
    close_output_files(fd_best, fd_full)
    initial = os.path.getsize(tmp_path / "IndexBest.bin")

    # Block 1 must NOT truncate — files keep their size.
    fd_best, fd_full = open_output_files(tmp_path, n_total_seeds=10, block_nr=1)
    try:
        assert os.fstat(fd_best).st_size == initial
    finally:
        close_output_files(fd_best, fd_full)


def test_seed_record_roundtrip(tmp_path):
    fd_best, fd_full = open_output_files(tmp_path, n_total_seeds=5, block_nr=0)
    try:
        seed = _make_seed()
        write_seed_record(fd_best, seed, offset_loc=2)
    finally:
        close_output_files(fd_best, fd_full)

    raw = np.fromfile(tmp_path / "IndexBest.bin", dtype=np.float64)
    assert raw.size == 5 * INDEX_BEST_RECORD_DOUBLES

    rec = raw[2 * INDEX_BEST_RECORD_DOUBLES : 3 * INDEX_BEST_RECORD_DOUBLES]
    # [0]: avg_ia
    assert rec[0] == pytest.approx(0.123)
    # [1..9]: orientation matrix flat row-major
    expected_R = np.eye(3) * 1.0
    expected_R[0, 1] = 0.5
    np.testing.assert_array_equal(rec[1:10], expected_R.reshape(-1))
    # [10..12]: position
    np.testing.assert_array_equal(rec[10:13], [10.0, 20.0, 30.0])
    # [13]: n_t_spots (matches C IndexerOMP.c:1620-1628 layout)
    assert rec[13] == 42.0
    # [14]: n_matches as double
    assert rec[14] == 3.0

    # Untouched slots (0, 1, 3, 4) must read as zeros
    others = np.concatenate([
        raw[0 * INDEX_BEST_RECORD_DOUBLES : 1 * INDEX_BEST_RECORD_DOUBLES],
        raw[1 * INDEX_BEST_RECORD_DOUBLES : 2 * INDEX_BEST_RECORD_DOUBLES],
        raw[3 * INDEX_BEST_RECORD_DOUBLES : 4 * INDEX_BEST_RECORD_DOUBLES],
        raw[4 * INDEX_BEST_RECORD_DOUBLES : 5 * INDEX_BEST_RECORD_DOUBLES],
    ])
    np.testing.assert_array_equal(others, np.zeros_like(others))


def test_full_record_roundtrip(tmp_path):
    fd_best, fd_full = open_output_files(tmp_path, n_total_seeds=2, block_nr=0)
    try:
        # 4 matched theoretical spots: (matched_obs_id, delta_omega)
        matched_pairs = np.array(
            [[101, 0.001], [102, -0.002], [103, 0.003], [104, -0.004]],
            dtype=np.float64,
        )
        write_full_record(fd_full, matched_pairs, offset_loc=1)
    finally:
        close_output_files(fd_best, fd_full)

    raw = np.fromfile(tmp_path / "IndexBestFull.bin", dtype=np.float64)
    assert raw.size == 2 * MAX_N_HKLS * 2

    slot1 = raw[MAX_N_HKLS * 2 :]  # offset_loc == 1
    np.testing.assert_array_equal(slot1[:8], matched_pairs.reshape(-1))
    # Remainder padded with zeros
    np.testing.assert_array_equal(slot1[8:], np.zeros(slot1.size - 8))


# --- P8 TODO(b): IndexBestWeighted.bin (soft attribution) ---


def test_open_output_files_opens_weighted_when_requested(tmp_path):
    from midas_index.io.output import INDEX_BEST_WEIGHTED_BYTES
    fd_best, fd_full, fd_w = open_output_files(
        tmp_path, n_total_seeds=4, block_nr=0, open_weighted=True,
    )
    try:
        st = os.fstat(fd_w)
        assert st.st_size == 4 * INDEX_BEST_WEIGHTED_BYTES
    finally:
        close_output_files(fd_best, fd_full, fd_w)
    assert (tmp_path / "IndexBestWeighted.bin").exists()


def test_write_block_emits_weighted_only_when_soft_seeds_present(tmp_path):
    from midas_index.io.output import write_block
    from midas_index.result import IndexerResult, SeedResult

    # Two seeds: one without weighted data, one with.
    s1 = _make_seed(spot_id=1)
    s2 = _make_seed(spot_id=2)
    s2.weighted_n_matches = 2.7
    s2.weighted_frac_matches = 0.42
    block = IndexerResult(block_nr=0, n_blocks=1, seeds=[s1, s2])

    write_block(block, tmp_path, n_total_seeds=2, block_nr=0)
    # IndexBestWeighted.bin should now exist; offset 0 (s1) = zeros,
    # offset 1 (s2) = (2.7, 0.42).
    raw = np.fromfile(tmp_path / "IndexBestWeighted.bin", dtype=np.float64)
    assert raw.shape == (4,)   # 2 seeds × 2 doubles
    assert raw[0] == 0.0 and raw[1] == 0.0
    assert raw[2] == pytest.approx(2.7)
    assert raw[3] == pytest.approx(0.42)


def test_write_block_skips_weighted_when_no_soft_seeds(tmp_path):
    from midas_index.io.output import write_block
    from midas_index.result import IndexerResult

    block = IndexerResult(
        block_nr=0, n_blocks=1,
        seeds=[_make_seed(spot_id=1), _make_seed(spot_id=2)],
    )
    write_block(block, tmp_path, n_total_seeds=2, block_nr=0)
    # No weighted_n_matches on any seed → no IndexBestWeighted.bin.
    assert not (tmp_path / "IndexBestWeighted.bin").exists()


def test_write_seed_record_dtype_promotes_to_float64(tmp_path):
    """Float32 seeds (GPU path) still write 15 doubles per slot."""
    fd_best, fd_full = open_output_files(tmp_path, n_total_seeds=1, block_nr=0)
    try:
        seed = _make_seed(dtype=torch.float32)
        write_seed_record(fd_best, seed, offset_loc=0)
    finally:
        close_output_files(fd_best, fd_full)

    raw = np.fromfile(tmp_path / "IndexBest.bin", dtype=np.float64)
    assert raw.size == INDEX_BEST_RECORD_DOUBLES
    # Values should round-trip to ~float32 precision after the dtype promotion.
    assert raw[0] == pytest.approx(0.123, rel=1e-6)
    assert raw[10] == pytest.approx(10.0)
