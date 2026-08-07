"""The gate must cut when there are two populations and refuse when there aren't.

The refusal is the point. An antimode can always be computed; on a smoothly
degrading dataset it is a number invented from noise, and shipping it would be
worse than shipping no gate at all.
"""

import numpy as np
import pytest

from midas_process_grains.compute.quality_gate import adaptive_quality_threshold


def _bimodal(n_good=3000, n_bad=800, seed=0):
    """Well-fitted grains near 100 µm, failures near 600 µm — the real shape."""
    rng = np.random.default_rng(seed)
    good = 10 ** rng.normal(np.log10(100.0), 0.10, n_good)
    bad = 10 ** rng.normal(np.log10(600.0), 0.12, n_bad)
    return np.concatenate([good, bad])


def test_cuts_between_the_two_populations():
    v = _bimodal()
    g = adaptive_quality_threshold(v)
    assert g.threshold is not None, g.reason
    assert 130.0 < g.threshold < 460.0, g.threshold
    # the cut must actually separate: nearly all good kept, nearly all bad gone
    keep = g.apply(v)
    assert keep[:3000].mean() > 0.98
    assert keep[3000:].mean() < 0.02


def test_declines_on_a_single_population():
    rng = np.random.default_rng(1)
    v = 10 ** rng.normal(np.log10(120.0), 0.18, 4000)
    g = adaptive_quality_threshold(v)
    assert g.threshold is None
    assert "declined" in g.reason
    assert g.apply(v).all()          # keep everything


def test_declines_on_a_smooth_heavy_tail():
    """A long tail is not two populations. This is the case that matters:
    quality degrades continuously, and there is no honest place to cut."""
    rng = np.random.default_rng(2)
    v = 100.0 * (1.0 + rng.pareto(1.8, 5000))
    g = adaptive_quality_threshold(v)
    assert g.threshold is None, f"invented a cut at {g.threshold}: {g.reason}"
    assert g.apply(v).all()


def test_declines_when_the_second_mode_is_a_handful_of_outliers():
    rng = np.random.default_rng(3)
    v = np.concatenate([10 ** rng.normal(np.log10(100.0), 0.10, 4000),
                        10 ** rng.normal(np.log10(800.0), 0.05, 6)])
    g = adaptive_quality_threshold(v)
    assert g.threshold is None
    assert "minority" in g.reason or "valley" in g.reason


def test_declines_on_degenerate_input():
    assert adaptive_quality_threshold(np.full(500, 3.0)).threshold is None
    assert adaptive_quality_threshold(np.array([1.0, 2.0, 3.0])).threshold is None
    g = adaptive_quality_threshold(np.array([1.0, 2.0, 3.0]))
    assert "too few" in g.reason


def test_threshold_is_dataset_specific_not_a_constant():
    """The whole reason this exists: two datasets with different geometry want
    different cuts, and the gate must follow rather than impose one."""
    a = _bimodal(seed=4)
    b = _bimodal(seed=5) * 1.9          # e.g. a longer sample-detector distance
    ga = adaptive_quality_threshold(a)
    gb = adaptive_quality_threshold(b)
    assert ga.threshold is not None and gb.threshold is not None
    assert gb.threshold > 1.5 * ga.threshold, (ga.threshold, gb.threshold)
    # and each still separates its own data
    assert ga.apply(a)[:3000].mean() > 0.98
    assert gb.apply(b)[:3000].mean() > 0.98


def test_score_like_metric_with_comparable_widths():
    """Confidence is bounded and score-like: higher is better, no log."""
    rng = np.random.default_rng(6)
    v = np.concatenate([rng.normal(0.985, 0.012, 3000),
                        rng.normal(0.86, 0.020, 900)]).clip(0, 1)
    g = adaptive_quality_threshold(v, lower_is_better=False, log_transform=False)
    assert g.threshold is not None, g.reason
    assert 0.88 < g.threshold < 0.99, g.threshold
    assert g.apply(v)[:3000].mean() > 0.95


def test_known_limitation_very_disparate_component_widths():
    """DOCUMENTED LIMITATION, not a wish: one smoothing width cannot resolve a
    very narrow population beside a very broad one. Here the good mode has
    sigma 0.004 and the bad one 0.03 -- 7x apart -- and the gate declines even
    though the mixture is obviously separated (Ashman D > 6).

    Declining is the safe direction (every grain is kept), so this is recorded
    rather than worked around. A variable-bandwidth KDE would fix it.
    """
    rng = np.random.default_rng(6)
    v = np.concatenate([rng.normal(0.995, 0.004, 3000),
                        rng.normal(0.86, 0.03, 900)]).clip(0, 1)
    g = adaptive_quality_threshold(v, lower_is_better=False, log_transform=False)
    assert g.ashman_d > 6.0            # the separation is real
    assert g.threshold is None         # and the gate still refuses
    assert g.apply(v).all()            # failing safe: nothing is discarded


def test_reports_its_evidence():
    g = adaptive_quality_threshold(_bimodal(seed=7))
    assert g.bimodality > 10.0
    assert g.valley_depth > 0.30
    assert g.modes is not None and g.modes[0] < g.modes[1]
    assert g.n_kept is not None and 0 < g.n_kept < g.n_total


@pytest.mark.parametrize("sep", [0.0, 0.05, 0.15])
def test_refuses_as_the_populations_merge(sep):
    """As the two populations slide together the gate must give up, not
    interpolate a meaningless boundary."""
    rng = np.random.default_rng(8)
    v = np.concatenate([10 ** rng.normal(2.0, 0.12, 3000),
                        10 ** rng.normal(2.0 + sep, 0.12, 900)])
    g = adaptive_quality_threshold(v)
    assert g.threshold is None, f"sep={sep} gave {g.threshold}: {g.reason}"
