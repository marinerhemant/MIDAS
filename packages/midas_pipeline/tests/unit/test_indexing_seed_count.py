"""The seed counter must read whichever backend actually wrote the seeds.

Counting only the legacy ``IndexBest.bin`` made the pipeline log
"0 / 56196 seeds with non-zero data" on every c-omp run while that run went on
to produce 11475 grains. A diagnostic pinned to zero cannot report a real zero,
which is the one thing it is for.
"""

import numpy as np
import pytest

from midas_pipeline.stages.indexing import _count_indexed_seeds


def _write_legacy(path, n_seeds, n_solved):
    a = np.zeros((n_seeds, 15), dtype=np.float64)
    a[:n_solved, 14] = 37.0                     # n_observed
    a.tofile(path)


def _write_consolidated(path, n_sol_per_vox, n_observed_per_vox):
    """IndexBest_all.bin: int32 nVox, int32 nSol[], int64 off[], (nSol,16) f64."""
    n_vox = len(n_sol_per_vox)
    n_sol = np.asarray(n_sol_per_vox, dtype=np.int32)
    off = np.zeros(n_vox, dtype=np.int64)
    rows = []
    for n, nobs in zip(n_sol_per_vox, n_observed_per_vox):
        for k in range(n):
            r = np.zeros(16, dtype=np.float64)
            r[14] = 100.0                       # n_expected
            r[15] = float(nobs)                 # n_observed
            rows.append(r)
    blob = (np.int32(n_vox).tobytes() + n_sol.tobytes() + off.tobytes()
            + (np.vstack(rows) if rows else np.zeros((0, 16))).astype(
                np.float64).tobytes())
    path.write_bytes(blob)


def test_counts_legacy_backend(tmp_path):
    p = tmp_path / "IndexBest.bin"
    _write_legacy(p, 100, 73)
    n, src = _count_indexed_seeds(tmp_path, p)
    assert n == 73
    assert src == p


def test_counts_comp_backend(tmp_path):
    """The regression: c-omp writes IndexBest_all.bin and nothing else."""
    p = tmp_path / "IndexBest_all.bin"
    _write_consolidated(p, [1, 2, 1, 0, 3], [12, 8, 0, 0, 40])
    legacy = tmp_path / "IndexBest.bin"          # deliberately absent
    n, src = _count_indexed_seeds(tmp_path, legacy)
    assert n == 3, "voxels 0,1,4 have a solution with n_observed>0"
    assert src == p


def test_reports_a_genuine_zero(tmp_path):
    """A real zero must be distinguishable from 'I looked in the wrong file'."""
    p = tmp_path / "IndexBest_all.bin"
    _write_consolidated(p, [1, 1], [0, 0])
    n, src = _count_indexed_seeds(tmp_path, tmp_path / "IndexBest.bin")
    assert n == 0
    assert src == p                              # counted, and found zero


def test_no_seed_file_at_all_is_distinguishable(tmp_path):
    n, src = _count_indexed_seeds(tmp_path, tmp_path / "IndexBest.bin")
    assert n == 0
    assert src is None                           # nothing to count, not a zero


def test_legacy_takes_precedence_when_both_exist(tmp_path):
    legacy = tmp_path / "IndexBest.bin"
    _write_legacy(legacy, 50, 11)
    _write_consolidated(tmp_path / "IndexBest_all.bin", [1] * 50, [9] * 50)
    n, src = _count_indexed_seeds(tmp_path, legacy)
    assert (n, src) == (11, legacy)


def test_ignores_a_truncated_legacy_file(tmp_path):
    """A short read must not be silently counted as a valid seed table."""
    p = tmp_path / "IndexBest.bin"
    p.write_bytes(np.zeros(15 * 3 + 7, dtype=np.float64).tobytes())
    _write_consolidated(tmp_path / "IndexBest_all.bin", [1, 1], [5, 5])
    n, src = _count_indexed_seeds(tmp_path, p)
    assert n == 2 and src.name == "IndexBest_all.bin"
