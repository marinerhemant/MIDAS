"""``SpotDiagnostics.bin`` reader — the only record of un-found expected spots.

Covers the v1/v2 distinction, because v1 files are already on disk everywhere
and their col 5 is untrustworthy on matched rows (writer stored ``theorGx``
there; measured col5==col6 on 41,118/41,118 matched rows of a real FF layer).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from midas_process_grains.io.spot_diag import (
    SPOT_DIAG_COLS,
    SPOT_DIAG_MAGIC,
    SPOT_DIAG_SENTINEL,
    SpotDiag,
    load_spot_diag,
)

NCOLS = len(SPOT_DIAG_COLS)


def _write(path, version, voxels):
    """voxels = [(voxNr, meta13, spots(nTheor, 19)), ...]"""
    with open(path, "wb") as f:
        f.write(struct.pack("<IIii", SPOT_DIAG_MAGIC, version,
                            len(voxels), NCOLS))
        f.write(struct.pack("<d", SPOT_DIAG_SENTINEL))
        f.write(b"\x00" * 40)
        for nr, _, sp in voxels:
            n_match = int((sp[:, 10] > 0.5).sum())
            f.write(np.array([nr, sp.shape[0], n_match],
                             dtype=np.int32).tobytes())
        for _, meta, _ in voxels:
            f.write(np.asarray(meta, dtype=np.float64).tobytes())
        for _, _, sp in voxels:
            f.write(np.asarray(sp, dtype=np.float64).tobytes())
    return path


def _voxel(nr, n_matched, n_unmatched, ring=1):
    n = n_matched + n_unmatched
    sp = np.zeros((n, NCOLS))
    sp[:, 4] = ring                                   # ringNr
    sp[:, 5] = np.arange(1, n + 1) * 2 + 1            # theorSpotID
    sp[:n_matched, 10] = 1.0                          # matched flag
    sp[:n_matched, 11:19] = 7.0                       # observed side present
    sp[n_matched:, 11:19] = SPOT_DIAG_SENTINEL        # un-found → sentinel
    meta = np.array([nr, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3,
                     3.6, 3.6, 3.6, 90.0, 90.0, 90.0])
    return (nr, meta, sp)


@pytest.fixture
def diag_v2(tmp_path):
    p = tmp_path / "Results"
    p.mkdir()
    _write(p / "SpotDiagnostics.bin", 2,
           [_voxel(0, 8, 2, ring=1), _voxel(5, 6, 4, ring=2)])
    return tmp_path


def test_reads_header_and_directory(diag_v2):
    d = load_spot_diag(diag_v2)
    assert d.version == 2
    assert d.n_voxels == 2
    assert d.n_cols == NCOLS
    assert d.sentinel == SPOT_DIAG_SENTINEL
    assert d.voxel_nrs.tolist() == [0, 5]
    assert d.n_theor.tolist() == [10, 10]
    assert d.n_matched.tolist() == [8, 6]


def test_unmatched_are_the_completeness_deficit(diag_v2):
    d = load_spot_diag(diag_v2)
    u = d.unmatched()
    assert u.shape[0] == 6                       # (10-8) + (10-6)
    # every observed column of an un-found spot is the sentinel
    assert np.all(u[:, 11:19] == SPOT_DIAG_SENTINEL)
    # ...and the prediction is still there
    assert np.all(u[:, 5] > 0)
    np.testing.assert_allclose(d.completeness(), [0.8, 0.6])


def test_per_ring_completeness(diag_v2):
    d = load_spot_diag(diag_v2)
    byring = d.completeness_by_ring()
    assert byring[1] == {"total": 10, "matched": 8, "frac": 0.8}
    assert byring[2] == {"total": 10, "matched": 6, "frac": 0.6}


def test_voxel_access_by_index_and_number(diag_v2):
    d = load_spot_diag(diag_v2)
    v = d.voxel_by_nr(5)
    assert v["voxelNr"] == 5 and v["nTheor"] == 10 and v["nMatched"] == 6
    np.testing.assert_allclose(v["position"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(v["latc"], [3.6, 3.6, 3.6, 90, 90, 90])
    assert v["spots"].shape == (10, NCOLS)
    assert d.voxel(1)["voxelNr"] == 5
    with pytest.raises(IndexError):
        d.voxel(2)
    with pytest.raises(KeyError):
        d.voxel_by_nr(999)


def test_named_column_access(diag_v2):
    d = load_spot_diag(diag_v2)
    assert d.col("matched").shape == (20,)
    with pytest.raises(KeyError, match="unknown column"):
        d.col("hklIndex")          # the old wrong name must not resolve


# ---------------------------------------------------------------------------
# Version awareness — v1 files are already on disk everywhere
# ---------------------------------------------------------------------------


def test_v1_flags_col5_as_untrustworthy(tmp_path):
    p = tmp_path / "Results"
    p.mkdir()
    _write(p / "SpotDiagnostics.bin", 1, [_voxel(0, 4, 1)])
    d = load_spot_diag(tmp_path)
    assert d.version == 1
    assert d.col5_is_theor_spot_id is False
    assert "WARNING" in d.summary() and "matched" in d.summary()


def test_v2_trusts_col5(diag_v2):
    assert load_spot_diag(diag_v2).col5_is_theor_spot_id is True


def test_rejects_unknown_version(tmp_path):
    p = tmp_path / "Results"
    p.mkdir()
    _write(p / "SpotDiagnostics.bin", 99, [_voxel(0, 1, 1)])
    with pytest.raises(ValueError, match="not supported"):
        load_spot_diag(tmp_path)


def test_rejects_bad_magic(tmp_path):
    f = tmp_path / "SpotDiagnostics.bin"
    f.write_bytes(struct.pack("<IIii", 0xDEADBEEF, 2, 0, NCOLS)
                  + struct.pack("<d", -999.0) + b"\x00" * 40)
    with pytest.raises(ValueError, match="bad magic"):
        SpotDiag(f)


def test_missing_file_names_the_writer(tmp_path):
    with pytest.raises(FileNotFoundError, match="DoSpotDiag"):
        load_spot_diag(tmp_path)


def test_finds_bare_file_not_only_results_subdir(tmp_path):
    _write(tmp_path / "SpotDiagnostics.bin", 2, [_voxel(0, 3, 1)])
    assert load_spot_diag(tmp_path).n_voxels == 1
