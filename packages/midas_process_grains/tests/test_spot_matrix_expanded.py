"""SpotMatrix.csv carries the prediction, the residuals, and the spots NEVER found.

The un-found expected spots are the completeness deficit itself: before this,
`Grains.csv` / `SpotMatrix.csv` / `FitBest.bin` all described matched spots
only, so completeness could be read as a number but never explained. They come
from `SpotDiagnostics.bin`, which the refiner has always written by default.

Two invariants worth their own tests:
  * cols 0-11 stay byte-identical to the legacy 12-column layout, so a parser
    taking the first 12 tab fields is unaffected;
  * un-found rows are honestly empty on the observed side — NaN, or -1 in the
    two `%d` columns that cannot hold NaN.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from midas_process_grains.compute.c_parity_emit import (
    SPOT_MATRIX_HEADER_EXPANDED,
    SPOT_MATRIX_NCOLS,
    write_spot_matrix_csv,
)
from midas_process_grains.compute.c_parity_run import CParityKeptGrain
from midas_process_grains.io.spot_diag import (
    SPOT_DIAG_MAGIC,
    SPOT_DIAG_SENTINEL,
    SpotDiag,
)

NDIAG = 19


def _grain(gid, rep, n=3):
    return CParityKeptGrain(
        grain_id=gid, rep_pos=rep,
        member_positions=np.array([rep]), member_ids=np.array([gid]),
        orient_mat=np.eye(3), position=np.zeros(3),
        lattice=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]),
        diff_pos=1.0, diff_ome=0.1, diff_angle=0.2,
        grain_radius=10.0, confidence=0.9,
    )


def _cache(spot_ids):
    n = len(spot_ids)
    return {
        "spot_ids": np.asarray(spot_ids, dtype=np.int64),
        "y": np.zeros(n), "z": np.zeros(n), "g": np.zeros((n, 3)),
        "ds_obs": np.ones(n), "ds_0": np.ones(n),
        # YExp, ZExp, OmegaExp   /   DiffLen, DiffOme, InternalAngle
        "exp3": np.tile([1000.0, 2000.0, 30.0], (n, 1)),
        "res3": np.tile([500.0, 0.20, 0.25], (n, 1)),
    }


def _input_matrix(n_rows):
    """(N, 10): Omega, SpotID, DetH, DetV, Eta, RingNr, YLab, ZLab, 2θ, OmeRaw."""
    im = np.zeros((n_rows, 10))
    im[:, 0] = 30.0
    im[:, 1] = np.arange(1, n_rows + 1)
    im[:, 5] = 3                    # RingNr
    im[:, 8] = 20.0                 # 2θ
    return im


def _write_diag(path, voxels):
    with open(path, "wb") as f:
        f.write(struct.pack("<IIii", SPOT_DIAG_MAGIC, 2, len(voxels), NDIAG))
        f.write(struct.pack("<d", SPOT_DIAG_SENTINEL))
        f.write(b"\x00" * 40)
        for nr, sp in voxels:
            f.write(np.array([nr, len(sp), int((sp[:, 10] > 0.5).sum())],
                             dtype=np.int32).tobytes())
        for nr, _ in voxels:
            f.write(np.array([nr] + [0.0] * 12, dtype=np.float64).tobytes())
        for _, sp in voxels:
            f.write(np.asarray(sp, dtype=np.float64).tobytes())
    return path


def _diag_block(n_matched, n_unmatched, obs_ids, ring=3):
    sp = np.zeros((n_matched + n_unmatched, NDIAG))
    sp[:n_matched, 10] = 1.0
    sp[:n_matched, 14] = obs_ids                    # obsSpotID
    sp[:n_matched, 5] = np.arange(1, n_matched + 1) * 2 + 1   # theorSpotID
    sp[n_matched:, 10] = 0.0
    sp[n_matched:, 5] = 900 + np.arange(n_unmatched)          # theorSpotID
    sp[n_matched:, 4] = ring                                  # ringNr
    sp[n_matched:, 3] = 12.5                                  # theorEta
    sp[n_matched:, 0] = 1111.0                                # theorY
    sp[n_matched:, 1] = 2222.0                                # theorZ
    sp[n_matched:, 2] = 44.0                                  # theorOmega
    sp[n_matched:, 11:19] = SPOT_DIAG_SENTINEL
    return sp


@pytest.fixture
def setup(tmp_path):
    grains = [_grain(1, 0), _grain(2, 1)]
    caches = [_cache([1, 2, 3]), _cache([4, 5])]
    im = _input_matrix(10)
    diag = SpotDiag(_write_diag(
        tmp_path / "SpotDiagnostics.bin",
        [(0, _diag_block(3, 2, [1, 2, 3])),
         (1, _diag_block(2, 1, [4, 5]))],
    ))
    return grains, caches, im, diag, tmp_path


def _read(path):
    hdr = open(path).readline().lstrip("%").split()
    d = np.genfromtxt(path, skip_header=1, usecols=range(SPOT_MATRIX_NCOLS))
    return hdr, np.atleast_2d(d)


def test_header_and_width_agree(setup):
    grains, caches, im, diag, tmp = setup
    p = tmp / "SpotMatrix.csv"
    write_spot_matrix_csv(out_path=p, kept_grains=grains, fb=object(),
                          input_matrix=im, spot_cache=caches,
                          spot_diag=diag, progress=False)
    hdr, d = _read(p)
    assert len(hdr) == SPOT_MATRIX_NCOLS == d.shape[1]
    assert hdr[12:16] == ["Matched", "theorSpotID", "theorRingNr", "theorEta"]
    assert SPOT_MATRIX_HEADER_EXPANDED.lstrip("%").split() == hdr


def test_unfound_rows_are_added_and_honestly_empty(setup):
    grains, caches, im, diag, tmp = setup
    p = tmp / "SpotMatrix.csv"
    write_spot_matrix_csv(out_path=p, kept_grains=grains, fb=object(),
                          input_matrix=im, spot_cache=caches,
                          spot_diag=diag, progress=False)
    _, d = _read(p)
    m, u = d[:, 12] > 0.5, d[:, 12] <= 0.5
    assert m.sum() == 5, "3 + 2 matched spots"
    assert u.sum() == 3, "2 + 1 un-found predictions"
    # the two %d columns cannot hold NaN, so they carry -1
    assert np.all(d[u, 1] == -1) and np.all(d[u, 7] == -1)
    # every other observed column is NaN, not 0.0
    assert np.all(np.isnan(d[u][:, [2, 3, 4, 5, 6, 8, 9, 10, 11]]))
    # ...and the prediction is present
    np.testing.assert_allclose(d[u][:, 16], 1111.0)
    np.testing.assert_allclose(d[u][:, 17], 2222.0)
    np.testing.assert_allclose(d[u][:, 18], 44.0)
    np.testing.assert_allclose(d[u][:, 14], 3.0)     # theorRingNr
    assert np.all(d[u, 13] >= 900)                   # theorSpotID
    # un-found rows are attributed to a real grain
    assert set(d[u, 0].astype(int)) <= {1, 2}


def test_matched_rows_keep_the_legacy_twelve(setup):
    """Cols 0-11 must not move: parsers read the first 12 tab fields."""
    grains, caches, im, diag, tmp = setup
    wide = tmp / "wide.csv"
    write_spot_matrix_csv(out_path=wide, kept_grains=grains, fb=object(),
                          input_matrix=im, spot_cache=caches,
                          spot_diag=diag, progress=False)
    narrow = tmp / "narrow.csv"
    write_spot_matrix_csv(out_path=narrow, kept_grains=grains, fb=object(),
                          input_matrix=im, spot_cache=caches,
                          spot_diag=None, progress=False)
    _, w = _read(wide)
    nh = open(narrow).readline()
    n = np.atleast_2d(np.genfromtxt(narrow, skip_header=1,
                                   usecols=range(SPOT_MATRIX_NCOLS)))
    wm = w[w[:, 12] > 0.5][:, :12]
    nm = n[n[:, 12] > 0.5][:, :12]
    np.testing.assert_array_equal(wm, nm)
    assert len(nh.lstrip("%").split()) == SPOT_MATRIX_NCOLS


def test_no_spot_diag_means_no_unfound_rows_not_a_crash(setup):
    """An older run has no SpotDiagnostics.bin; the file must still write."""
    grains, caches, im, _, tmp = setup
    p = tmp / "SpotMatrix.csv"
    write_spot_matrix_csv(out_path=p, kept_grains=grains, fb=object(),
                          input_matrix=im, spot_cache=caches,
                          spot_diag=None, progress=False)
    _, d = _read(p)
    assert np.all(d[:, 12] > 0.5), "no un-found rows without SpotDiagnostics"
    assert np.all(np.isnan(d[:, 13])), "theorSpotID unavailable -> NaN"
    # the prediction from FitBest is still there
    np.testing.assert_allclose(d[:, 16], 1000.0)
    np.testing.assert_allclose(d[:, 19], 500.0)


def test_post_fit_columns_nan_without_fitbestfinal(setup):
    grains, caches, im, diag, tmp = setup
    p = tmp / "SpotMatrix.csv"
    write_spot_matrix_csv(out_path=p, kept_grains=grains, fb=object(),
                          input_matrix=im, spot_cache=caches,
                          spot_diag=diag, fb_final=None, progress=False)
    _, d = _read(p)
    assert np.all(np.isnan(d[:, 22:28])), "post-fit must be NaN, never 0.0"


def test_theor_spot_id_joins_matched_rows(setup):
    grains, caches, im, diag, tmp = setup
    p = tmp / "SpotMatrix.csv"
    write_spot_matrix_csv(out_path=p, kept_grains=grains, fb=object(),
                          input_matrix=im, spot_cache=caches,
                          spot_diag=diag, progress=False)
    _, d = _read(p)
    m = d[:, 12] > 0.5
    # every matched row got a theorSpotID from the diagnostics join
    assert np.all(np.isfinite(d[m][:, 13]))
    assert np.all(d[m][:, 13] % 2 == 1), "theorSpotID is ih*2+1+within → odd/even ids"
