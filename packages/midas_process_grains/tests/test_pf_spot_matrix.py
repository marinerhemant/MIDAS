"""PF gets a SpotMatrix.csv — and it is the un-found rows that make it worth having.

PF never had one. `Result_OrientPos_voxel_*.csv` carries completeness as a
single number per voxel, so *which* reflections went missing, and on which ring,
was recorded nowhere. This file is built from `SpotDiagnostics.bin` alone: that
already holds observed AND predicted positions, the residuals, the matched flag
and the scan, so it needs no per-scan `InputAllExtraInfoFittingAll*.csv` join —
PF has one of those per scan and getting the join wrong is silent.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from midas_process_grains.io.spot_diag import (
    PF_SPOT_MATRIX_COLS,
    SPOT_DIAG_MAGIC,
    SPOT_DIAG_SENTINEL,
    SpotDiag,
    write_pf_spot_matrix,
)

NDIAG = 19
C = {n: i for i, n in enumerate(PF_SPOT_MATRIX_COLS)}


def _voxel(n_matched, n_unmatched, ring, scan):
    sp = np.zeros((n_matched + n_unmatched, NDIAG))
    m = slice(0, n_matched)
    sp[m, 10] = 1.0
    sp[m, 0], sp[m, 1], sp[m, 2] = 100.0, 200.0, 30.0     # theor Y/Z/omega
    sp[m, 3], sp[m, 4] = 11.5, ring                        # theorEta, ringNr
    sp[m, 5] = np.arange(1, n_matched + 1) * 2 + 1          # theorSpotID
    sp[m, 9] = scan                                        # theorScanNr
    sp[m, 11], sp[m, 12], sp[m, 13] = 101.0, 202.0, 30.5   # obs Y/Z/omega
    sp[m, 14] = np.arange(1, n_matched + 1) * 10            # obsSpotID
    sp[m, 15] = scan                                       # obsScanNr
    sp[m, 16], sp[m, 17], sp[m, 18] = 0.2, 5.5, 0.05        # IA, diffLen, diffOme
    u = slice(n_matched, None)
    sp[u, 10] = 0.0
    sp[u, 0], sp[u, 1], sp[u, 2] = 300.0, 400.0, 44.0
    sp[u, 3], sp[u, 4] = 77.5, ring
    sp[u, 5] = 900 + np.arange(n_unmatched)
    sp[u, 9] = scan
    sp[u, 11:19] = SPOT_DIAG_SENTINEL
    return sp


def _write_diag(path, voxels, version=2):
    with open(path, "wb") as f:
        f.write(struct.pack("<IIii", SPOT_DIAG_MAGIC, version, len(voxels), NDIAG))
        f.write(struct.pack("<d", SPOT_DIAG_SENTINEL))
        f.write(b"\x00" * 40)
        for nr, sp in voxels:
            f.write(np.array([nr, len(sp), int((sp[:, 10] > 0.5).sum())],
                             dtype=np.int32).tobytes())
        for nr, _ in voxels:
            f.write(np.array([nr] + [0.0] * 12, dtype=np.float64).tobytes())
        for _, sp in voxels:
            f.write(np.asarray(sp, dtype=np.float64).tobytes())
    return SpotDiag(path)


@pytest.fixture
def diag(tmp_path):
    return _write_diag(tmp_path / "SpotDiagnostics.bin",
                       [(7, _voxel(4, 2, ring=3, scan=5)),
                        (19, _voxel(3, 1, ring=2, scan=9))])


def _read(p):
    hdr = open(p).readline().lstrip("%").split()
    d = np.atleast_2d(np.genfromtxt(p, skip_header=1))
    return hdr, d


def test_header_matches_width(diag, tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    assert write_pf_spot_matrix(diag, p) == 10          # (4+2) + (3+1)
    hdr, d = _read(p)
    assert hdr == list(PF_SPOT_MATRIX_COLS)
    assert d.shape == (10, len(PF_SPOT_MATRIX_COLS))


def test_unfound_rows_carry_the_prediction_and_nothing_observed(diag, tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    write_pf_spot_matrix(diag, p)
    _, d = _read(p)
    u = d[:, C["Matched"]] == 0
    assert u.sum() == 3, "2 + 1 predicted-but-never-found"
    # integer columns cannot hold NaN -> -1
    assert np.all(d[u, C["SpotID"]] == -1)
    assert np.all(d[u, C["ScanNr"]] == -1)
    # observed floats and residuals are NaN, never 0.0
    for c in ("Omega", "YLab", "ZLab", "DiffLen", "DiffOme", "InternalAngle"):
        assert np.all(np.isnan(d[u, C[c]])), c
    # the prediction survives, including which ring lost the spot
    np.testing.assert_allclose(d[u, C["YExp"]], 300.0)
    np.testing.assert_allclose(d[u, C["OmegaExp"]], 44.0)
    assert set(d[u, C["RingNr"]].astype(int)) == {3, 2}
    assert np.all(d[u, C["theorSpotID"]] >= 900)


def test_matched_rows_carry_observed_and_predicted_and_residuals(diag, tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    write_pf_spot_matrix(diag, p)
    _, d = _read(p)
    m = d[:, C["Matched"]] == 1
    assert m.sum() == 7
    np.testing.assert_allclose(d[m, C["YLab"]], 101.0)     # observed
    np.testing.assert_allclose(d[m, C["YExp"]], 100.0)     # predicted
    np.testing.assert_allclose(d[m, C["DiffLen"]], 5.5)
    assert np.all(d[m, C["SpotID"]] > 0)


def test_voxel_id_not_grain_id(diag, tmp_path):
    """PF is keyed by voxel; a voxel is not a grain."""
    p = tmp_path / "SpotMatrix.csv"
    write_pf_spot_matrix(diag, p)
    _, d = _read(p)
    assert set(d[:, C["VoxelNr"]].astype(int)) == {7, 19}
    assert (d[:, C["VoxelNr"]] == 7).sum() == 6
    assert (d[:, C["VoxelNr"]] == 19).sum() == 4


def test_scan_columns_survive(diag, tmp_path):
    """The scan is PF's extra dimension and FF's SpotMatrix has no room for it."""
    p = tmp_path / "SpotMatrix.csv"
    write_pf_spot_matrix(diag, p)
    _, d = _read(p)
    v7 = d[:, C["VoxelNr"]] == 7
    assert set(d[v7, C["theorScanNr"]].astype(int)) == {5}
    m = v7 & (d[:, C["Matched"]] == 1)
    assert set(d[m, C["ScanNr"]].astype(int)) == {5}


def test_v1_blanks_theor_spot_id_on_matched_rows(tmp_path):
    """v1 wrote theorGx into col 5 on matched rows; shipping it would be wrong
    for exactly half the file, so it is blanked rather than emitted."""
    d1 = _write_diag(tmp_path / "SpotDiagnostics.bin",
                     [(1, _voxel(3, 2, ring=1, scan=0))], version=1)
    p = tmp_path / "SpotMatrix.csv"
    write_pf_spot_matrix(d1, p)
    _, d = _read(p)
    m = d[:, C["Matched"]] == 1
    assert np.all(np.isnan(d[m, C["theorSpotID"]])), "v1 matched rows must be blank"
    assert np.all(np.isfinite(d[~m, C["theorSpotID"]])), "unmatched are still valid"


def test_completeness_reconciles_with_the_diagnostics(diag, tmp_path):
    p = tmp_path / "SpotMatrix.csv"
    write_pf_spot_matrix(diag, p)
    _, d = _read(p)
    for i, v in enumerate(diag.voxel_nrs):
        rows = d[d[:, C["VoxelNr"]] == v]
        assert len(rows) == diag.n_theor[i]
        assert (rows[:, C["Matched"]] == 1).sum() == diag.n_matched[i]
