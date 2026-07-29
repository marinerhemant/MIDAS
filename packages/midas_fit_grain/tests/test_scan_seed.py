"""Tests for scan_seed.write_pf_seed_file — the 5-col SpotsToIndex.csv the
c-omp PF refiner needs, synthesised from IndexBest_all.bin."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from midas_fit_grain.scan_seed import write_pf_seed_file


def _write_index_best_all(path: Path, per_voxel: List[np.ndarray]) -> None:
    """Write a synthetic IndexBest_all.bin (see scan_driver._read_index_best_all):
    int32 n_voxels, int32 n_sol[], int64 off[], float64 vals[total,16]."""
    n_voxels = len(per_voxel)
    n_sol = np.array([int(r.shape[0]) for r in per_voxel], dtype=np.int32)
    off = np.zeros(n_voxels, dtype=np.int64)          # recomputable; unused by reader
    with open(path, "wb") as f:
        f.write(np.int32(n_voxels).tobytes())
        f.write(n_sol.tobytes())
        f.write(off.tobytes())
        for rec in per_voxel:
            if rec.shape[0]:
                f.write(np.ascontiguousarray(rec, dtype=np.float64).tobytes())


def _sol(n_expected: float, n_matched: float) -> np.ndarray:
    """One 16-col solution record with the completeness-relevant cols set
    (col14 = NrExpected, col15 = NrMatched)."""
    r = np.zeros(16, dtype=np.float64)
    r[14] = n_expected
    r[15] = n_matched
    return r


def test_seed_picks_best_completeness_and_skips_empty(tmp_path: Path):
    # voxel 0: two solutions, best is #1 (comp 0.6 > 0.3), nSpots = 6
    v0 = np.stack([_sol(10, 3), _sol(10, 6)])
    # voxel 1: empty -> skipped
    v1 = np.zeros((0, 16))
    # voxel 2: one solution, comp 0.5, nSpots = 5
    v2 = _sol(10, 5).reshape(1, 16)
    idx = tmp_path / "IndexBest_all.bin"
    _write_index_best_all(idx, [v0, v1, v2])

    out = tmp_path / "SpotsToIndex.csv"
    n = write_pf_seed_file(idx, out)

    assert n == 2                                     # voxel 1 skipped
    rows = [ln.split() for ln in out.read_text().split("\n") if ln.strip()]
    # voxNr SpId nSpotsBest unused bestSolIdx
    assert rows[0] == ["0", "0", "6", "0", "1"]
    assert rows[1] == ["2", "2", "5", "0", "0"]


def test_seed_all_empty_writes_no_rows(tmp_path: Path):
    idx = tmp_path / "IndexBest_all.bin"
    _write_index_best_all(idx, [np.zeros((0, 16))] * 3)
    out = tmp_path / "SpotsToIndex.csv"
    assert write_pf_seed_file(idx, out) == 0
    assert out.read_text().strip() == ""


def test_seed_completeness_uses_matched_over_expected(tmp_path: Path):
    # higher raw nMatched but lower completeness must NOT win
    v0 = np.stack([_sol(4, 3),     # comp 0.75, nMatched 3
                   _sol(100, 5)])  # comp 0.05, nMatched 5
    idx = tmp_path / "IndexBest_all.bin"
    _write_index_best_all(idx, [v0])
    out = tmp_path / "SpotsToIndex.csv"
    write_pf_seed_file(idx, out)
    row = out.read_text().split()
    assert row[2] == "3" and row[4] == "0"            # picked the comp-0.75 sol
