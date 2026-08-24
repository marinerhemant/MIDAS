"""The exact progress lines the c-omp binaries emit must keep parsing.

``progress.py``'s regex and the ``printf`` calls in ``IndexerUnified.c`` /
``FitUnified.c`` are a contract across a language boundary with no compiler to
check it: if either side is edited alone, progress silently stops working and
nothing fails. These pin the real strings.

The PF indexer reports ``seeds`` rather than ``voxels`` because voxels are far
too coarse -- one voxel is ~2 thread-hours on a dense s1 layer, so a 361-voxel
grid on 64 threads gives a bar that cannot move for the first two hours. Its
line carries a trailing ``(voxels done N/M)`` that the regex must ignore.
"""

from __future__ import annotations

import pytest

from midas_pipeline.progress import format_sub, parse_progress_line


def test_pf_indexer_seed_line_with_trailing_voxel_count():
    p = parse_progress_line(
        "  progress: 1234567/9876543 seeds, 5123.4 seeds/s, "
        "elapsed 350.0s (voxels done 12/361)"
    )
    assert p is not None
    assert (p["done"], p["total"], p["unit"]) == (1234567, 9876543, "seeds")
    assert p["rate"] == 5123.4


def test_ff_indexer_voxel_line():
    p = parse_progress_line(
        "  progress: 180/36100 voxels, 3.4 vox/s, elapsed 53.0s")
    assert p is not None
    assert (p["done"], p["total"], p["unit"]) == (180, 36100, "voxels")
    assert p["rate"] == 3.4


def test_refiner_seed_line():
    p = parse_progress_line(
        "  progress: 40/361 seeds, 0.8 seeds/s, elapsed 50.0s")
    assert p is not None
    assert (p["done"], p["total"], p["unit"]) == (40, 361, "seeds")


def test_non_progress_output_is_ignored():
    for line in ("Writing consolidated output files...\n",
                 "Reading parameters from file: paramstest.txt.\n",
                 "Finished, time elapsed: 407.236 seconds.\n",
                 ""):
        assert parse_progress_line(line) is None


def test_zero_total_is_rejected_not_divided_by():
    assert parse_progress_line("  progress: 0/0 voxels, 0.0 vox/s") is None


@pytest.mark.parametrize("done,total,expect_pct", [(0, 361, "0%"),
                                                   (180, 361, "50%"),
                                                   (361, 361, "100%")])
def test_rendering_percentages(done, total, expect_pct):
    out = format_sub({"stage": "indexing", "done": done, "total": total,
                      "unit": "voxels", "rate": None, "at": 0.0})
    assert expect_pct in out
