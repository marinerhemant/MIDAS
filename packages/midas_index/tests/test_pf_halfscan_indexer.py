"""PF half-scan support in the C indexer: VoxelGridFile and PFCoverageAwareCompleteness.

Both are off by default and the default path is pinned bit-identical by
test_unified_c_parity. Here:

* a VoxelGridFile that lists the historical positions x positions grid must
  reproduce the legacy golden BYTE FOR BYTE (the file path is exact);
* coverage-aware completeness on a FULL scan may change only voxels outside the
  scanned disc (|r| > R + ScanPosTol), where some predicted spots have no scan
  position to land on -- every voxel inside must be byte-identical.

The half-scan recovery itself (cut 9 of the 15 scans, recover the far side of
the axis) is measured by the dev script
``$ANALYSIS/datasetK_halfscan_dev/eval_halfscan.py``;
it needs a rewritten 1 GB nData.bin, too heavy for the unit suite.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from midas_index import backend_c
from midas_index.io.consolidated import read_index_best_all, split_records_by_voxel

FIX = Path(__file__).parent / "data" / "scanning_5grain_golden"
GOLDEN = FIX / "golden" / "IndexBest_all.bin"


def _ready() -> bool:
    return all((FIX / f).is_file() for f in ("Spots.bin", "Data.bin", "nData.bin")) \
        and GOLDEN.is_file()


pytestmark = [
    pytest.mark.skipif(not backend_c.available(), reason="midas_indexer C binary not built"),
    pytest.mark.skipif(not _ready(), reason="PF fixture binaries absent (regen via build.py)"),
]


def _run(tmp: Path, extra: list[str]):
    (tmp / "Output").mkdir(parents=True)
    for f in ("Spots.bin", "Data.bin", "nData.bin"):
        os.symlink(os.path.realpath(FIX / f), tmp / f)
    if (FIX / "RingSlots.csv").exists():      # compact-ring-slot sidecar
        shutil.copy2(FIX / "RingSlots.csv", tmp / "RingSlots.csv")
    for f in ("positions.csv", "hkls.csv", "SpotsToIndex.csv"):
        shutil.copy2(FIX / f, tmp / f)
    lines = [f"OutputFolder {tmp / 'Output'}" if l.startswith("OutputFolder ") else l
             for l in (FIX / "paramstest.txt").read_text().splitlines()] + extra
    (tmp / "paramstest.txt").write_text("\n".join(lines) + "\n")
    proc = backend_c.run_indexer(tmp / "paramstest.txt", n_work=15, num_procs=1,
                                 extra_env={"OMP_NUM_THREADS": "1"}, cwd=tmp)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    return (tmp / "Output" / "IndexBest_all.bin"), proc.stdout.decode("utf-8", errors="replace")


def _positions():
    return np.sort(np.loadtxt(FIX / "positions.csv"))


def test_voxel_grid_file_of_the_legacy_grid_is_byte_identical(tmp_path):
    pos = _positions()
    grid = tmp_path / "grid.txt"
    grid.write_text("# x y\n" + "".join(f"{x:.17g} {y:.17g}\n" for x in pos for y in pos))
    out, stdout = _run(tmp_path / "run", [f"VoxelGridFile {grid}"])
    if "VoxelGridFile:" not in stdout:
        pytest.skip("this midas_indexer build predates VoxelGridFile")
    assert out.read_bytes() == GOLDEN.read_bytes()


def test_coverage_changes_only_voxels_outside_the_scanned_disc(tmp_path):
    out, stdout = _run(tmp_path / "run", ["PFCoverageAwareCompleteness 1"])
    if "PFCoverageAwareCompleteness: on" not in stdout:
        pytest.skip("this midas_indexer build predates PFCoverageAwareCompleteness")
    pos = _positions()
    n = pos.size
    R = float(np.max(np.abs(pos)))
    tol = 5.0 / 2.0                     # fixture BeamSize 5.0, no ScanPosTol
    got = split_records_by_voxel(read_index_best_all(out))
    ref = split_records_by_voxel(read_index_best_all(GOLDEN))
    n_changed = 0
    for v in range(n * n):
        i, j = divmod(v, n)
        r = float(np.hypot(pos[i], pos[j]))
        a, b = ref[v], got[v]
        same = (a is None and b is None) or (
            a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b))
        if r <= R + tol:
            assert same, f"voxel {v} at r={r:.1f} is inside the scanned disc but changed"
        n_changed += not same
    assert n_changed > 0, "coverage never engaged: the corners should change"
