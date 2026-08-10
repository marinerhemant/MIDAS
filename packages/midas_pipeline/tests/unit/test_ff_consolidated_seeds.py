"""FF indexing must handle the CONSOLIDATED seed family end to end.

github.com/marinerhemant/MIDAS issues/68: a c-omp FF run logged
"indexing(FF): 0 / 6399 seeds with non-zero data", refined nothing, and died in
midas_process_grains on `cannot mmap an empty file`. The reporter concluded
midas_indexer takes the PF branch in FF mode and should be changed to write
IndexBest.bin.

It should not. Measured on the datasetA Ni layer:

    py_run  (python backend, FF)   Output/ holds ONLY IndexBest_all.bin,
                                   IndexKey_all.bin, IndexBest_IDs_all.bin,
                                   IndexBest_weights_all.bin
    c_run   (classical IndexerOMP) Output/ holds ONLY IndexBest.bin,
                                   IndexBestFull.bin

The consolidated family is the FF contract for BOTH modern backends; the legacy
pair now comes only from the classical binary. Both refiners already consume
the consolidated form (midas_fit_grain.driver adapts it; FitUnified.c probes it
first). The writer was right — the reader was counting only the legacy name,
and the stage then advertised legacy paths that had never been written.

This is the FF-mode c-omp test issues/68 asked for as its point 3.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.stages.indexing import _count_indexed_seeds


def _write_consolidated(out_dir: Path, n_vox: int, n_observed) -> Path:
    """IndexBest_all.bin: int32 nVox | int32 nSol[nVox] | int64 off[nVox]
    | float64 (total_sol, 16), col 15 = n_observed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_sol = np.ones(n_vox, dtype=np.int32)
    off = np.arange(n_vox, dtype=np.int64)
    recs = np.zeros((n_vox, 16), dtype=np.float64)
    recs[:, 14] = 100.0                       # n_expected
    recs[:, 15] = np.asarray(n_observed, dtype=np.float64)
    p = out_dir / "IndexBest_all.bin"
    with p.open("wb") as f:
        f.write(struct.pack("<i", n_vox))
        f.write(n_sol.tobytes())
        f.write(off.tobytes())
        f.write(recs.tobytes())
    return p


def _write_legacy(out_dir: Path, n_seeds: int, n_observed) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = np.zeros((n_seeds, 15), dtype=np.float64)
    rec[:, 14] = np.asarray(n_observed, dtype=np.float64)
    p = out_dir / "IndexBest.bin"
    rec.tofile(p)
    return p


def test_consolidated_seeds_are_counted(tmp_path):
    """The exact false negative from the report: c-omp wrote seeds, the
    pipeline said zero."""
    out = tmp_path / "Output"
    _write_consolidated(out, 6399, [100] * 6370 + [0] * 29)
    n, src = _count_indexed_seeds(out, out / "IndexBest.bin")
    assert n == 6370, "must not report 0 when the consolidated family is there"
    assert src is not None and src.name == "IndexBest_all.bin"


def test_legacy_seeds_are_still_counted(tmp_path):
    out = tmp_path / "Output"
    _write_legacy(out, 10, [5] * 7 + [0] * 3)
    n, src = _count_indexed_seeds(out, out / "IndexBest.bin")
    assert n == 7
    assert src is not None and src.name == "IndexBest.bin"


def test_legacy_wins_when_both_are_present(tmp_path):
    """A run seeded from the classical binary and re-indexed must not
    double-count; the legacy pair is checked first and is authoritative."""
    out = tmp_path / "Output"
    _write_legacy(out, 10, [5] * 10)
    _write_consolidated(out, 4, [1, 1, 1, 1])
    n, src = _count_indexed_seeds(out, out / "IndexBest.bin")
    assert (n, src.name) == (10, "IndexBest.bin")


def test_no_seed_file_is_distinguishable_from_zero_seeds(tmp_path):
    """The distinction the whole report turns on."""
    out = tmp_path / "Output"
    out.mkdir()
    n_missing, src_missing = _count_indexed_seeds(out, out / "IndexBest.bin")
    assert (n_missing, src_missing) == (0, None)

    _write_consolidated(out, 5, [0] * 5)
    n_zero, src_zero = _count_indexed_seeds(out, out / "IndexBest.bin")
    assert n_zero == 0
    assert src_zero is not None, (
        "an honest zero must be distinguishable from a missing file — one is "
        "a result, the other is a broken contract"
    )


@pytest.mark.parametrize("n_vox", [1, 3, 257])
def test_ragged_solution_counts(tmp_path, n_vox):
    """nSol is per-voxel and need not be 1; the offset walk must survive it."""
    out = tmp_path / f"Output{n_vox}"
    out.mkdir(parents=True)
    n_sol = np.full(n_vox, 2, dtype=np.int32)
    off = (np.arange(n_vox, dtype=np.int64) * 2)
    recs = np.zeros((n_vox * 2, 16), dtype=np.float64)
    recs[:, 15] = 7.0                          # every solution has observations
    p = out / "IndexBest_all.bin"
    with p.open("wb") as f:
        f.write(struct.pack("<i", n_vox))
        f.write(n_sol.tobytes())
        f.write(off.tobytes())
        f.write(recs.tobytes())
    n, src = _count_indexed_seeds(out, out / "IndexBest.bin")
    assert n == n_vox, "one voxel with any solved candidate counts once"
