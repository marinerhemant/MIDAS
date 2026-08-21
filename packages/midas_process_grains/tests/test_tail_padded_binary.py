"""The LAST seed must not vanish when the C writer leaves a short final slot.

``FitUnified.c`` pwrites only ``nSpotsComp`` records per seed at a full-slot
stride (``:2297`` for FitBest.bin, ``:2136`` for ProcessKey.bin). The final
seed therefore leaves the file short of a clean multiple, and the readers used
to floor-divide by the stride and truncate — silently deleting that seed.

Measured on the 56,125-seed Ni FF layer (2026-08-21 ``ff_refiner_prepost``):
FitBest.bin was 56,124 full slots + 87 rows, ProcessKey.bin 56,124 + 87 ints,
and the dropped seed 56,124 (SpotID 245283) had keep_flag set, NrIDsPerID 87
and completeness 0.777 — an ordinary live grain candidate. Every fixture in
``conftest.py`` writes full zero-padded slots, so nothing in the suite could
have caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.io.binary import (
    FIT_BEST_DOUBLES,
    MAX_N_HKLS,
    PROCESS_KEY_INTS,
    TailPaddedBinary,
    read_fit_best,
    read_process_key,
)

N_FULL = 3          # seeds written with a complete slot
N_TAIL_ROWS = 87    # rows in the short final slot (the real observed value)


@pytest.fixture
def run_short_tail(tmp_path):
    """A run dir whose FitBest/ProcessKey end in a genuinely short slot."""
    (tmp_path / "Output").mkdir()
    (tmp_path / "Results").mkdir()

    # FitBest: N_FULL complete slots, then N_TAIL_ROWS rows of a 4th seed.
    fb = np.zeros((N_FULL, MAX_N_HKLS, 22), dtype=np.float64)
    for s in range(N_FULL):
        fb[s, :5, 0] = np.arange(1, 6) + 100 * s        # SpotIDs
        fb[s, :5, 20] = 10.0 * (s + 1)                  # DiffLen
    tail = np.zeros((N_TAIL_ROWS, 22), dtype=np.float64)
    tail[:, 0] = np.arange(1, N_TAIL_ROWS + 1) + 900     # distinctive SpotIDs
    tail[:, 20] = 598.785                               # the real observed mean
    with open(tmp_path / "Output" / "FitBest.bin", "wb") as f:
        f.write(fb.tobytes())
        f.write(tail.tobytes())

    # ProcessKey: same shape of truncation, N_TAIL_ROWS ints in the last slot.
    pk = np.zeros((N_FULL, PROCESS_KEY_INTS), dtype=np.int32)
    for s in range(N_FULL):
        pk[s, :5] = np.arange(1, 6) + 100 * s
    pk_tail = (np.arange(1, N_TAIL_ROWS + 1) + 900).astype(np.int32)
    with open(tmp_path / "Results" / "ProcessKey.bin", "wb") as f:
        f.write(pk.tobytes())
        f.write(pk_tail.tobytes())
    return tmp_path


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------


def test_fit_best_keeps_the_short_final_seed(run_short_tail):
    fb = read_fit_best(run_short_tail)
    assert fb.shape == (N_FULL + 1, MAX_N_HKLS, 22), (
        "the short final slot must be presented as a seed, not truncated away"
    )
    last = fb[N_FULL]
    assert last.shape == (MAX_N_HKLS, 22)
    assert int((last[:, 0] > 0).sum()) == N_TAIL_ROWS
    np.testing.assert_allclose(last[:N_TAIL_ROWS, 20], 598.785)
    # everything past the written rows is zero-padded, so the SpotID>0 filter
    # every caller applies still selects exactly the real rows
    np.testing.assert_array_equal(last[N_TAIL_ROWS:], 0.0)


def test_process_key_keeps_the_short_final_seed(run_short_tail):
    pk = read_process_key(run_short_tail)
    assert pk.shape == (N_FULL + 1, PROCESS_KEY_INTS)
    last = pk[N_FULL]
    assert int((last > 0).sum()) == N_TAIL_ROWS
    np.testing.assert_array_equal(last[:N_TAIL_ROWS],
                                  np.arange(1, N_TAIL_ROWS + 1) + 900)
    np.testing.assert_array_equal(last[N_TAIL_ROWS:], 0)


def test_fit_best_and_process_key_agree_on_seed_count(run_short_tail):
    """The mismatch that made c_parity_run truncate every run."""
    assert read_fit_best(run_short_tail).shape[0] == \
        read_process_key(run_short_tail).shape[0]


# ---------------------------------------------------------------------------
# Indexing surface used by real callers
# ---------------------------------------------------------------------------


def test_full_slots_are_unchanged(run_short_tail):
    fb = read_fit_best(run_short_tail)
    for s in range(N_FULL):
        np.testing.assert_array_equal(fb[s][:5, 0],
                                      np.arange(1, 6) + 100 * s)


def test_negative_and_out_of_range_indexing(run_short_tail):
    fb = read_fit_best(run_short_tail)
    np.testing.assert_array_equal(fb[-1], fb[N_FULL])      # tail via -1
    np.testing.assert_array_equal(fb[-2], fb[N_FULL - 1])
    with pytest.raises(IndexError):
        _ = fb[N_FULL + 1]


def test_slicing_spans_the_tail(run_short_tail):
    """read_index_best_full chunks with fb[i0:i1]; it must reach the tail."""
    fb = read_fit_best(run_short_tail)
    everything = fb[0:N_FULL + 1]
    assert everything.shape == (N_FULL + 1, MAX_N_HKLS, 22)
    assert int((everything[N_FULL][:, 0] > 0).sum()) == N_TAIL_ROWS
    # a slice wholly inside the full slots stays a zero-copy memmap view
    inner = fb[0:2]
    assert inner.shape == (2, MAX_N_HKLS, 22)
    # empty and open-ended slices
    assert fb[2:2].shape == (0, MAX_N_HKLS, 22)
    assert fb[:].shape == (N_FULL + 1, MAX_N_HKLS, 22)
    assert fb[N_FULL:].shape == (1, MAX_N_HKLS, 22)


def test_refuses_whole_file_materialisation(run_short_tail):
    """np.asarray must not silently drop the tail nor copy the whole file."""
    fb = read_fit_best(run_short_tail)
    assert isinstance(fb, TailPaddedBinary)
    with pytest.raises(TypeError, match="refusing to materialise"):
        np.asarray(fb)
    # the explicit escape hatch works and includes the tail
    full = fb.to_numpy()
    assert full.shape == (N_FULL + 1, MAX_N_HKLS, 22)
    assert int((full[N_FULL][:, 0] > 0).sum()) == N_TAIL_ROWS


def test_rejects_a_torn_file(tmp_path):
    """A remainder that is not a whole number of records is corruption."""
    (tmp_path / "Output").mkdir()
    fb = np.zeros((1, MAX_N_HKLS, 22), dtype=np.float64)
    with open(tmp_path / "Output" / "FitBest.bin", "wb") as f:
        f.write(fb.tobytes())
        f.write(np.zeros(22 * 3 + 7, dtype=np.float64).tobytes())  # 7 orphans
    with pytest.raises(ValueError, match="torn"):
        read_fit_best(tmp_path)


# ---------------------------------------------------------------------------
# Exact-multiple files must be byte-identical to the old behaviour
# ---------------------------------------------------------------------------


def test_read_index_best_full_chunks_over_the_tail(run_short_tail):
    """The chunked-slice consumer must see the recovered seed.

    ``read_index_best_full`` walks FitBest in 512-seed slices, and separately
    has a ProcessKey fallback that calls ``.astype`` on the reader's return
    value — both break on a per-seed view unless handled.
    """
    from midas_process_grains.io.binary import read_index_best_full

    ibf = read_index_best_full(run_short_tail)      # FitBest path
    assert ibf.shape == (N_FULL + 1, MAX_N_HKLS, 2)
    assert int((ibf[N_FULL, :, 0] > 0).sum()) == N_TAIL_ROWS

    # ProcessKey fallback path: same run dir with FitBest.bin removed
    (run_short_tail / "Output" / "FitBest.bin").unlink()
    ibf2 = read_index_best_full(run_short_tail)
    assert ibf2.shape == (N_FULL + 1, MAX_N_HKLS, 2)
    assert int((ibf2[N_FULL, :, 0] > 0).sum()) == N_TAIL_ROWS


def test_materialize_on_a_padded_view(run_short_tail):
    # NB: run_short_tail and tiny_run_dir both build inside tmp_path, so a
    # single test cannot request both — their mkdir()s collide.
    from midas_process_grains.io.binary import materialize

    padded = materialize(read_process_key(run_short_tail))
    assert isinstance(padded, np.ndarray)
    assert padded.shape == (N_FULL + 1, PROCESS_KEY_INTS)
    assert int((padded[N_FULL] > 0).sum()) == N_TAIL_ROWS


def test_materialize_on_a_plain_array(tiny_run_dir):
    from midas_process_grains.io.binary import materialize

    plain = materialize(read_process_key(tiny_run_dir))
    assert isinstance(plain, np.ndarray)
    assert plain.shape == (3, PROCESS_KEY_INTS)


def test_exact_multiple_returns_a_plain_array(tiny_run_dir):
    """No wrapper, no behaviour change, for files with no short slot."""
    fb = read_fit_best(tiny_run_dir)
    pk = read_process_key(tiny_run_dir)
    assert not isinstance(fb, TailPaddedBinary)
    assert not isinstance(pk, TailPaddedBinary)
    # np.asarray still works and is zero-copy for these
    assert np.asarray(fb).shape == fb.shape
    assert np.asarray(pk).shape == pk.shape
