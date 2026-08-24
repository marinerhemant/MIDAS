"""The record layouts must match what their docstrings claim.

Two different IndexBest layouts coexist, both putting n_t_spots before
n_matches but at different offsets:

  legacy per-seed  (15 doubles): [0] avg_ia, [1..9] OM, [10..12] pos,
                                 [13] n_t_spots, [14] n_matches
  consolidated     (16 doubles): [0] SpotID, [1] avgIA, [2..10] OM,
                                 [11..13] pos, [14] n_t_spots, [15] n_matches

Swapping the two within either layout silently inverts every completeness —
``n_matches/n_t_spots`` becomes its reciprocal, which for a well-indexed grain
is a plausible-looking number greater than 1. The legacy docstring stated the
reverse order until 2026-08-23; no code followed it, but a new reader written
against it would have produced exactly that.
"""
from __future__ import annotations

import re

import numpy as np

from midas_index.io import output as out_mod
from midas_index.io.output import INDEX_BEST_RECORD_DOUBLES


def _docstring_index_of(mod, name: str) -> int:
    """Pull ``[NN]  name`` out of the module docstring."""
    for line in (mod.__doc__ or "").splitlines():
        m = re.match(r"\s*\[(\d+)\]\s+(\S+)", line)
        if m and m.group(2).rstrip(",") == name:
            return int(m.group(1))
    raise AssertionError(f"{name!r} not documented in the layout block")


def test_legacy_record_matches_its_own_docstring():
    """The written record must land where the docstring says it does."""
    class _Seed:
        avg_ia = 0.25
        best_or_mat = None
        best_pos = None
        n_t_spots = 61.0
        n_matches = 47.0

    import torch

    s = _Seed()
    s.best_or_mat = torch.eye(3, dtype=torch.float64)
    s.best_pos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    rec = out_mod._seed_record(s)

    assert rec.shape == (INDEX_BEST_RECORD_DOUBLES,)
    i_t = _docstring_index_of(out_mod, "n_t_spots")
    i_m = _docstring_index_of(out_mod, "n_matches")
    assert rec[i_t] == 61.0, f"docstring says n_t_spots at [{i_t}], record disagrees"
    assert rec[i_m] == 47.0, f"docstring says n_matches at [{i_m}], record disagrees"


def test_the_denominator_comes_before_the_numerator():
    """Both layouts order them this way; the ordering is the whole trap."""
    assert _docstring_index_of(out_mod, "n_t_spots") < \
           _docstring_index_of(out_mod, "n_matches")


def test_completeness_from_the_record_is_a_fraction():
    """Reading them in the documented order gives <= 1; swapped gives > 1."""
    import torch

    class _Seed:
        avg_ia = 0.1
        best_or_mat = torch.eye(3, dtype=torch.float64)
        best_pos = torch.zeros(3, dtype=torch.float64)
        n_t_spots = 61.0
        n_matches = 47.0

    rec = out_mod._seed_record(_Seed())
    i_t = _docstring_index_of(out_mod, "n_t_spots")
    i_m = _docstring_index_of(out_mod, "n_matches")
    assert 0.0 <= rec[i_m] / rec[i_t] <= 1.0
    # and the swap is exactly the failure mode: a plausible number above 1
    assert rec[i_t] / rec[i_m] > 1.0


def test_the_refiner_reads_the_same_offsets_it_is_written_at():
    """midas_fit_grain.driver maps consolidated -> legacy; keep them agreed."""
    import inspect

    from midas_fit_grain import driver

    src = inspect.getsource(driver)
    # the mapping lines, as written in _read_consolidated_as_index_best
    assert "index_best[v, 13] = rec[14]" in src, (
        "the refiner no longer maps consolidated[14] (n_t_spots) to "
        "legacy[13]; the two layouts have drifted"
    )
    assert "index_best[v, 14] = ids.size" in src
    assert driver.INDEX_BEST_RECORD_DOUBLES == INDEX_BEST_RECORD_DOUBLES
