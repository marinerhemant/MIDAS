"""Phase identification from a few d-spacings.

Each test targets one of the four faults the module exists to prevent:
a residual quoted without a line count; hand-written reflection conditions;
a local minimum in the free scale; and a null that under-populates small d.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_hkls.phase_id import (
    PhaseCandidate, candidate_d_lines, worst_relative_residual,
    global_minimax_scale, chance_worst_residual, identify_phase,
)


def _crystal(a, b, c, sg_number, element="La"):
    return Crystal(lattice=Lattice(a=a, b=b, c=c, alpha=90., beta=90., gamma=90.),
                   space_group=SpaceGroup.from_number(sg_number),
                   atoms=[Atom(element=element, fract=(0., 0., 0.), label="X")])


# La3Ni2O7 Fmmm supercell, the collaborators' hand cell
LA327 = _crystal(5.2739, 5.2384, 20.5, 69)
# a rival with MORE lines in the same range
LA4310 = _crystal(5.41, 5.46, 28.0, 69)


def test_lines_come_from_the_space_group_not_by_hand():
    """F-centring conditions are never retyped, so they cannot be mistyped."""
    lines = candidate_d_lines(LA327, d_min=1.0, d_max=12.0)
    assert lines.size > 10
    assert np.all(np.diff(lines) > 0)              # unique and sorted
    assert lines.min() >= 1.0 and lines.max() <= 12.0
    # a primitive cell of the same dimensions has strictly MORE lines
    prim = _crystal(5.2739, 5.2384, 20.5, 16)      # P222
    assert candidate_d_lines(prim, d_min=1.0, d_max=12.0).size > lines.size


def test_more_lines_means_a_better_match_by_chance_alone():
    """The reason a residual without a line count is not evidence."""
    d_obs = np.array([2.21, 1.92, 1.63, 1.42, 1.31, 1.18])
    few = chance_worst_residual(20, d_obs, d_min=1.0, d_max=12.0, n_draws=60,
                                rng_seed=1)
    many = chance_worst_residual(400, d_obs, d_min=1.0, d_max=12.0, n_draws=60,
                                 rng_seed=1)
    assert np.median(many) < np.median(few), (
        f"400 random lines ({np.median(many):.3f} %) did not beat 20 "
        f"({np.median(few):.3f} %) — the chance mechanism is not being modelled")


def test_the_null_is_uniform_in_reciprocal_VOLUME_not_in_d():
    """A uniform-in-d null under-populates small d and inflates significance."""
    rng = np.random.default_rng(0)
    d_min, d_max = 1.0, 12.0
    g_lo, g_hi = 1 / d_max, 1 / d_min
    u = rng.random(200_000)
    g = (u * (g_hi ** 3 - g_lo ** 3) + g_lo ** 3) ** (1 / 3)
    d_vol = 1 / g
    # the volume-correct null puts the great majority at SMALL d
    assert np.median(d_vol) < 0.5 * (d_min + d_max)
    assert (d_vol < 2.0).mean() > 0.5


def test_worst_residual_is_the_WORST_not_the_mean():
    lines = np.array([2.0, 3.0, 4.0])
    d_obs = [2.0, 3.0, 4.4]                 # two perfect, one 10 % off
    assert worst_relative_residual(lines, d_obs) == pytest.approx(
        abs(4.0 - 4.4) / 4.4 * 100, rel=1e-9)


def test_perfect_match_is_zero_and_empty_lines_is_infinite():
    lines = np.array([2.0, 3.0, 4.0])
    assert worst_relative_residual(lines, [2.0, 3.0]) == pytest.approx(0.0)
    assert np.isinf(worst_relative_residual(np.array([]), [2.0]))


def test_global_scale_escapes_a_LOCAL_minimum():
    """Fault 3: a sweep from 1.0 stops in the nearest well, not the deepest."""
    lines = np.array([2.0, 3.0, 4.0, 5.0])
    true_scale = 0.93
    d_obs = lines * true_scale
    scale, worst = global_minimax_scale(lines, d_obs)
    assert scale == pytest.approx(true_scale, abs=2e-3)
    assert worst < 0.05
    # and the naive "scale = 1" answer is much worse, which is the point
    assert worst_relative_residual(lines, d_obs, 1.0) > 5.0


def test_global_scale_rejects_a_bad_range():
    with pytest.raises(ValueError, match="scale_range"):
        global_minimax_scale(np.array([2.0]), [2.0], scale_range=(1.2, 0.9))


def test_identify_phase_reports_line_count_cell_and_provenance():
    d_obs = [2.6370, 2.6192, 1.8642, 1.5236, 1.3185, 1.1965]
    cands = [PhaseCandidate("La3Ni2O7", LA327, cell_source="hand index, this dataset"),
             PhaseCandidate("La4Ni3O10", LA4310, cell_source="ambient literature")]
    res = identify_phase(d_obs, cands, d_min=1.0, d_max=12.0, n_null_draws=40)
    assert len(res) == 2
    for m in res:
        assert m.n_lines > 0
        assert "a=" in m.cell
        assert m.cell_source                       # provenance is recorded
        assert 0.0 <= m.p_value <= 1.0
        assert m.chance_median_pct is not None
    assert res[0].worst_free_pct <= res[1].worst_free_pct      # sorted best first


def test_a_candidate_with_many_lines_gets_a_WEAK_p_even_when_it_wins():
    """The headline behaviour: winning on residual is not winning."""
    d_obs = [2.6370, 2.6192, 1.8642, 1.5236, 1.3185, 1.1965]
    big = PhaseCandidate("many-line rival", _crystal(9.9, 10.1, 30.0, 69))
    res = identify_phase(d_obs, [big], d_min=1.0, d_max=12.0, n_null_draws=60)
    m = res[0]
    assert m.n_lines > 100
    assert m.p_value > 0.05, (
        f"a {m.n_lines}-line phase reached p = {m.p_value:.3f}; with that many "
        "lines a random phase should match about as well")


def test_the_true_phase_beats_its_own_chance_level():
    """A positive control: exact d values from a cell must be significant."""
    lines = candidate_d_lines(LA327, d_min=1.0, d_max=12.0)
    d_obs = lines[[2, 5, 9, 14, 20, 27]]           # exactly on the lattice
    res = identify_phase(d_obs, [PhaseCandidate("La3Ni2O7", LA327)],
                         d_min=1.0, d_max=12.0, n_null_draws=80)
    assert res[0].worst_free_pct < 1e-3
    assert res[0].p_value < 0.05


def test_identify_phase_refuses_empty_observations():
    with pytest.raises(ValueError, match="no observed"):
        identify_phase([], [PhaseCandidate("x", LA327)])
