"""Progress reporting for the per-voxel pass, and the parity it must not break.

find_grains had no instrumentation at all: on s5/L3 it sat for 94 minutes
looking exactly like a hang, and the only way to tell was to attach to the
process. Reporting it needs results as they COMPLETE rather than in submission
order, which is only safe because per-voxel results are scattered by voxel
index and the single-key rows are re-sorted -- so chunk order cannot reach the
output. That safety is the thing worth pinning here.
"""

from __future__ import annotations

import numpy as np

from midas_pipeline.find_grains import _per_voxel_pass, _pervoxel_worker
from midas_pipeline.find_grains import (
    write_ids_bin,
    write_keys_bin,
    write_vals_bin,
)


def _build_out_dir(tmp_path, n_vox: int, *, seed: int = 7):
    """Consolidated triplet for ``n_vox`` voxels, one candidate each.

    Orientations vary per voxel so the chunks are not trivially identical.
    """
    out_dir = tmp_path / "Output"
    out_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    vals, keys, ids = [], [], []
    for v in range(n_vox):
        row = np.zeros(16, dtype=np.float64)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        row[2:11] = np.array([
            1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
        ], dtype=np.float64)
        row[1] = v + 1
        row[14] = 4.0
        row[15] = float(2 + (v % 3))          # varying confidence
        vals.append(row.reshape(1, 16))
        keys.append(np.array([[v + 1, 4, 4, 0]], dtype=np.uint64))
        ids.append(np.array([1, 2, 3, 4], dtype=np.int32))
    write_vals_bin(out_dir / "IndexBest_all.bin", vals)
    write_keys_bin(out_dir / "IndexKey_all.bin", keys)
    write_ids_bin(out_dir / "IndexBest_IDs_all.bin", ids)
    return out_dir


def test_worker_returns_its_span_so_progress_counts_voxels():
    # `rows` only covers VALID voxels, so its length cannot be used to measure
    # how much of the grid a chunk swept. The span has to come back too.
    import inspect
    src = inspect.getsource(_pervoxel_worker)
    assert "return v0, v1, rows" in src


def test_progress_reaches_every_voxel(tmp_path):
    out_dir = _build_out_dir(tmp_path, 36)
    seen: list[tuple[int, int, str]] = []
    _per_voxel_pass(out_dir, 36, 225, 1.0, n_jobs=4,
                    progress_cb=lambda d, t, u: seen.append((d, t, u)))
    assert seen, "no progress was reported"
    assert seen[-1][0] == 36            # every voxel accounted for
    assert all(t == 36 for _d, t, _u in seen)
    assert all(u == "voxels" for _d, _t, u in seen)
    # Monotone: a bar that goes backwards is worse than none.
    assert [d for d, _t, _u in seen] == sorted(d for d, _t, _u in seen)


def test_serial_path_also_reports(tmp_path):
    out_dir = _build_out_dir(tmp_path, 9)
    seen: list[int] = []
    _per_voxel_pass(out_dir, 9, 225, 1.0, n_jobs=1,
                    progress_cb=lambda d, t, u: seen.append(d))
    assert seen and seen[-1] == 9


def test_results_identical_serial_vs_parallel(tmp_path):
    """The parity the imap_unordered change must not break."""
    out_dir = _build_out_dir(tmp_path, 36)
    a = _per_voxel_pass(out_dir, 36, 225, 1.0, n_jobs=1)
    b = _per_voxel_pass(out_dir, 36, 225, 1.0, n_jobs=5)

    np.testing.assert_array_equal(a[0], b[0])      # per-voxel OMs
    np.testing.assert_array_equal(a[1], b[1])      # confidences
    np.testing.assert_array_equal(a[2], b[2])      # keys
    assert [v for v, _r in a[3]] == [v for v, _r in b[3]]
    for (_va, ra), (_vb, rb) in zip(a[3], b[3]):
        np.testing.assert_array_equal(ra, rb)


def test_progress_does_not_change_results(tmp_path):
    out_dir = _build_out_dir(tmp_path, 16)
    quiet = _per_voxel_pass(out_dir, 16, 225, 1.0, n_jobs=4)
    noisy = _per_voxel_pass(out_dir, 16, 225, 1.0, n_jobs=4,
                            progress_cb=lambda d, t, u: None)
    np.testing.assert_array_equal(quiet[0], noisy[0])
    np.testing.assert_array_equal(quiet[2], noisy[2])


def test_a_raising_callback_does_not_fail_the_stage(tmp_path):
    out_dir = _build_out_dir(tmp_path, 16)

    def cb(_d, _t, _u):
        raise ValueError("reporter exploded")

    res = _per_voxel_pass(out_dir, 16, 225, 1.0, n_jobs=4, progress_cb=cb)
    assert res[0].shape == (16, 9)
