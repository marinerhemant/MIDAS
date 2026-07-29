"""Tests for fitbest_adapter.fitbest_to_result_orientpos — c-omp FitBest_*.csv
-> pf-odf-readable Result_OrientPos_voxel_*.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from midas_fit_grain.fitbest_adapter import fitbest_to_result_orientpos

_HEADER = "\t".join([
    "SpotID", "O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
    "SpotID", "x", "y", "z", "SpotID", "a", "b", "c", "alpha", "beta", "gamma",
    "SpotID", "PosErr", "OmeErr", "InternalAngle", "Radius", "Completeness",
    "E11", "E12", "E13", "E21", "E22", "E23", "E31", "E32", "E33",
    "Eul1", "Eul2", "Eul3"]) + "\n"


def _write_fitbest(path: Path, om, lattice, completeness, n_spot_rows=3):
    """Multi-block FitBest CSV: header, 39-col result row, header again, per-spot
    rows — matching FitUnified.c's writer."""
    row = np.zeros(39)
    row[0] = row[10] = row[14] = row[21] = 42          # SpotID labels
    row[1:10] = np.asarray(om).ravel()
    row[15:21] = lattice
    row[25] = 1.0                                       # radius
    row[26] = completeness
    with open(path, "w") as f:
        f.write(_HEADER)
        f.write("\t".join(f"{v:.6f}" for v in row) + "\t\n")
        f.write(_HEADER)                                # repeated header block
        for i in range(n_spot_rows):
            f.write("\t".join(str(float(i)) for _ in range(23)) + "\t\n")


def test_adapter_extracts_result_row_and_is_pf_odf_readable(tmp_path: Path):
    fb = tmp_path / "fitbest"; fb.mkdir()
    res = tmp_path / "Results"
    om = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    lat = [3.61, 3.61, 3.61, 90.0, 90.0, 90.0]
    _write_fitbest(fb / "FitBest_000007_000000000.csv", om, lat, 0.55)

    n = fitbest_to_result_orientpos(fb, res)
    assert n == 1
    out = res / "Result_OrientPos_voxel_7.csv"
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) == 2                              # header + single data row

    # Exactly the columns pf-odf._read_voxel_result reads: OM row[1:10], lat row[15:21]
    row = np.array(lines[1].split(), dtype=float)
    np.testing.assert_allclose(row[1:10].reshape(3, 3), om)
    np.testing.assert_allclose(row[15:21], lat)
    assert abs(row[26] - 0.55) < 1e-6


def test_adapter_picks_highest_completeness_per_voxel(tmp_path: Path):
    fb = tmp_path / "fitbest"; fb.mkdir()
    res = tmp_path / "Results"
    lat = [3.6, 3.6, 3.6, 90, 90, 90]
    # same voxel 3, two seed solutions: comp 0.3 and 0.8 -> keep 0.8
    _write_fitbest(fb / "FitBest_000003_000000000.csv", np.eye(3), lat, 0.30)
    _write_fitbest(fb / "FitBest_000003_000000009.csv", np.eye(3), lat, 0.80)

    n = fitbest_to_result_orientpos(fb, res)
    assert n == 1                                       # one voxel, best kept
    row = np.array((res / "Result_OrientPos_voxel_3.csv").read_text()
                   .splitlines()[1].split(), dtype=float)
    assert abs(row[26] - 0.80) < 1e-6


def test_adapter_ignores_non_fitbest_files(tmp_path: Path):
    fb = tmp_path / "fitbest"; fb.mkdir()
    (fb / "SpotDiagnostics.bin").write_bytes(b"\x00\x01")
    (fb / "notes.txt").write_text("ignore me")
    assert fitbest_to_result_orientpos(fb, tmp_path / "Results") == 0
