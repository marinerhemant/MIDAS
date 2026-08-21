"""``OrientPosFit.bin`` is 27 doubles wide before 2026-08-21 and 33 after.

A wrong width shifts every column silently — orientation becomes position,
lattice becomes errors — with no exception anywhere, so the sniffing has to be
content-based rather than arithmetic. Any multiple of 297 doubles divides by
both 27 and 33, which is a real ambiguity and not a hypothetical one: 11
grains x 27 == 9 grains x 33.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.io.binary import (
    ORIENT_POS_FIT_DOUBLES,
    ORIENT_POS_FIT_DOUBLES_V2,
    ORIENT_POS_FIT_LAYOUT,
    read_fit_best_final,
    read_orient_pos_fit,
)

_SENTINEL_COLS = (0, 10, 14, 21)


def _row(spot_id, ncols, seed=0):
    """One plausible OPF row: SpId repeated in the four sentinel columns."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(1.0, 100.0, ncols)
    for c in _SENTINEL_COLS:
        r[c] = spot_id
    return r


def _write_opf(tmp_path, rows, ncols, with_key=True):
    d = tmp_path / "Results"
    d.mkdir(exist_ok=True)
    arr = np.asarray(rows, dtype=np.float64).reshape(len(rows), ncols)
    arr.tofile(d / "OrientPosFit.bin")
    if with_key:
        key = np.zeros((len(rows), 2), dtype=np.int32)
        key[:, 0] = arr[:, 0].astype(np.int32)
        key[:, 1] = 5
        key.tofile(d / "Key.bin")
    return tmp_path


def test_reads_legacy_27(tmp_path):
    rd = _write_opf(tmp_path, [_row(7, 27), _row(9, 27, 1)], 27)
    a = read_orient_pos_fit(rd)
    assert a.shape == (2, ORIENT_POS_FIT_DOUBLES)
    assert a[0, 0] == 7 and a[1, 0] == 9


def test_reads_new_33_and_exposes_prepost(tmp_path):
    r0, r1 = _row(7, 33), _row(9, 33, 1)
    r0[27:30] = [600.0, 0.30, 0.40]        # pre  pos/ome/angle
    r0[30:33] = [350.0, 0.20, 0.25]        # post pos/ome/angle
    rd = _write_opf(tmp_path, [r0, r1], 33)
    a = read_orient_pos_fit(rd)
    assert a.shape == (2, ORIENT_POS_FIT_DOUBLES_V2)
    L = ORIENT_POS_FIT_LAYOUT
    assert a[0, L["pos_err_pre"]] == 600.0
    assert a[0, L["ome_err_pre"]] == 0.30
    assert a[0, L["internal_ang_pre"]] == 0.40
    assert a[0, L["pos_err_post"]] == 350.0
    # the whole point: post should be an improvement on pre
    assert a[0, L["pos_err_post"]] < a[0, L["pos_err_pre"]]


def test_the_297_ambiguity_is_resolved_not_guessed(tmp_path):
    """9 rows x 33 == 11 rows x 27 == 297 doubles. Must pick 33 correctly."""
    rows = [_row(i + 1, 33, i) for i in range(9)]
    rd = _write_opf(tmp_path, rows, 33)
    assert (tmp_path / "Results" / "OrientPosFit.bin").stat().st_size == 297 * 8
    a = read_orient_pos_fit(rd)
    assert a.shape == (9, 33), "must not fall back to 11x27"
    assert a[:, 0].tolist() == [float(i + 1) for i in range(9)]


def test_ambiguity_resolved_by_content_when_key_absent(tmp_path):
    """Same 297-double file with no Key.bin — the sentinel check must carry it."""
    rows = [_row(i + 1, 33, i) for i in range(9)]
    rd = _write_opf(tmp_path, rows, 33, with_key=False)
    a = read_orient_pos_fit(rd)
    assert a.shape == (9, 33)


def test_27_wide_297_double_file_reads_as_27(tmp_path):
    rows = [_row(i + 1, 27, i) for i in range(11)]
    rd = _write_opf(tmp_path, rows, 27, with_key=False)
    assert (tmp_path / "Results" / "OrientPosFit.bin").stat().st_size == 297 * 8
    a = read_orient_pos_fit(rd)
    assert a.shape == (11, 27), "sentinel columns should rule out 9x33 here"


def test_refuses_a_width_it_cannot_justify(tmp_path):
    d = tmp_path / "Results"
    d.mkdir()
    # 30 doubles: divisible by neither 27 nor 33.
    np.arange(30, dtype=np.float64).tofile(d / "OrientPosFit.bin")
    with pytest.raises(ValueError, match="multiple of neither"):
        read_orient_pos_fit(tmp_path)


def test_unambiguous_width_needs_no_sentinels(tmp_path):
    """Arithmetic settles it; the sentinel check must only break ties.

    Gating on sentinels instead broke the v4 pipeline smoke tests, whose
    synthetic 4x27 OrientPosFit leaves cols 0/10/14/21 unpopulated. 108
    doubles divides by 27 only, so there is nothing to disambiguate.
    """
    d = tmp_path / "Results"
    d.mkdir()
    np.zeros(4 * 27, dtype=np.float64).tofile(d / "OrientPosFit.bin")
    assert read_orient_pos_fit(tmp_path).shape == (4, 27)


def test_prepost_keys_absent_on_legacy_file_are_a_loud_failure(tmp_path):
    """Asking for a 33-only column on a 27-wide file must raise, not alias."""
    rd = _write_opf(tmp_path, [_row(7, 27)], 27)
    a = read_orient_pos_fit(rd)
    with pytest.raises(IndexError):
        _ = a[0, ORIENT_POS_FIT_LAYOUT["pos_err_post"]]


# ---------------------------------------------------------------------------
# FitBestFinal.bin
# ---------------------------------------------------------------------------


def test_fit_best_final_missing_names_the_version(tmp_path):
    with pytest.raises(FileNotFoundError, match="2026-08-21"):
        read_fit_best_final(tmp_path)


def test_fit_best_final_reads_and_pads_its_tail(tmp_path):
    from midas_process_grains.io.binary import MAX_N_HKLS, TailPaddedBinary

    out = tmp_path / "Output"
    out.mkdir()
    full = np.zeros((2, MAX_N_HKLS, 22))
    full[0, :3, 0] = [1, 2, 3]
    full[1, :2, 0] = [4, 5]
    tail = np.zeros((7, 22))
    tail[:, 0] = np.arange(10, 17)
    with open(out / "FitBestFinal.bin", "wb") as f:
        f.write(full.tobytes())
        f.write(tail.tobytes())

    fb = read_fit_best_final(tmp_path)
    assert isinstance(fb, TailPaddedBinary)
    assert fb.shape == (3, MAX_N_HKLS, 22)
    assert int((fb[2][:, 0] > 0).sum()) == 7
