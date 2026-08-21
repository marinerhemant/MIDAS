"""The PF carrier: FitBest_*.csv -> Result_OrientPos_voxel_*.csv at 39 or 45 cols.

PF writes none of the OrientPosFit/Key/ProcessKey/FitBest **binaries** — only
``SpotDiagnostics.bin`` — but it *does* write the per-voxel ``FitBest_*.csv``,
because that writer (``FitUnified.c:2270``) sits outside the FF-only guard. So
widening the CSV result row is what carries the pre/post error triples into PF,
and this adapter is what propagates them: it forwards **all** tokens, so the
only thing that must keep up is the header.

A 39-name header over a 45-value row mislabels every column downstream and
nothing raises, which is why ``_header_for`` refuses rather than guesses.

Note the guard is ASYMMETRIC and cannot be made otherwise from this side: it
protects new-binary + new-adapter, but an *old* adapter given a 45-column row
writes its fixed 39-name header silently. Binary and package move together.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_fit_grain.fitbest_adapter import (
    _HEADER_39,
    _HEADER_45,
    _header_for,
    fitbest_to_result_orientpos,
)

_SPOT_HEADER = "SpotID\tYObsCorrPos\tZObsCorrPos\n"


def _write_fitbest(path, n_result_cols, completeness=0.8, tag=1.0):
    """One FitBest_<vox>_<sp>.csv: header, result row, header, spot rows."""
    row = np.arange(n_result_cols, dtype=float) * tag
    row[26] = completeness                      # _COMPLETENESS_COL
    hdr = _HEADER_39 if n_result_cols == 39 else _HEADER_45
    with open(path, "w") as f:
        f.write(hdr.replace(" ", "\t"))
        f.write("\t".join(f"{v:.6f}" for v in row) + "\n")
        f.write(_SPOT_HEADER)
        for i in range(3):
            f.write(f"{i+1}\t{i*10.0}\t{i*20.0}\n")
    return row


def _read_result(path):
    names = open(path).readline().split()
    vals = open(path).read().splitlines()[1].split()
    return names, vals


@pytest.mark.parametrize("ncols", [39, 45])
def test_round_trip_at_both_widths(tmp_path, ncols):
    fb = tmp_path / "FitBest_comp"; fb.mkdir()
    res = tmp_path / "Results"
    row = _write_fitbest(fb / "FitBest_000007_000000123.csv", ncols)

    n = fitbest_to_result_orientpos(fb, res)
    assert n == 1
    names, vals = _read_result(res / "Result_OrientPos_voxel_7.csv")
    assert len(names) == ncols, "header must match the data width"
    assert len(vals) == ncols
    np.testing.assert_allclose([float(v) for v in vals], row)
    # the columns pf-odf reads must not have moved
    np.testing.assert_allclose([float(v) for v in vals[1:10]], row[1:10])
    np.testing.assert_allclose([float(v) for v in vals[15:21]], row[15:21])
    assert float(vals[26]) == 0.8


def test_45_col_header_names_the_new_columns(tmp_path):
    fb = tmp_path / "FitBest_comp"; fb.mkdir()
    res = tmp_path / "Results"
    _write_fitbest(fb / "FitBest_000001_000000001.csv", 45)
    fitbest_to_result_orientpos(fb, res)
    names, _ = _read_result(res / "Result_OrientPos_voxel_1.csv")
    assert names[39:] == ["PosErrPre", "OmeErrPre", "InternalAnglePre",
                          "PosErrPost", "OmeErrPost", "InternalAnglePost"]
    # ...and they are at 39-44 here, NOT the 27-32 they occupy in the binary
    assert names[27] == "E11" and names[36] == "Eul1"


def test_refuses_a_width_it_cannot_label(tmp_path):
    with pytest.raises(ValueError, match="Refusing to write a header"):
        _header_for(41)
    assert _header_for(39) is _HEADER_39
    assert _header_for(45) is _HEADER_45


def test_highest_completeness_solution_still_wins(tmp_path):
    """Multi-solution voxels: the pick must not change with the width."""
    fb = tmp_path / "FitBest_comp"; fb.mkdir()
    res = tmp_path / "Results"
    _write_fitbest(fb / "FitBest_000003_000000001.csv", 45,
                   completeness=0.30, tag=1.0)
    _write_fitbest(fb / "FitBest_000003_000000009.csv", 45,
                   completeness=0.95, tag=2.0)
    fitbest_to_result_orientpos(fb, res)
    _, vals = _read_result(res / "Result_OrientPos_voxel_3.csv")
    assert float(vals[26]) == 0.95
    assert float(vals[1]) == 2.0, "the tag=2.0 (higher-completeness) row won"
