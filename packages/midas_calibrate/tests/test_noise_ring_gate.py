"""``MinRingSNR`` is applied by v1 ``run_estep`` (it was parsed but never applied).

A ring whose MEDIAN baseline-referenced SNR over its fits is below ``MinRingSNR`` is dropped whole, with the same floor of
4 best-ranked rings as calibrate_v2's ``filter_fits_by_ring_quality`` so it can never empty the dataset. The default (0)
changes nothing.

What this is NOT: a claim that the gate improves the fitted geometry. A /verify of that claim (2026-10-14, claim
35366c1d3340) came back REFUTED: dropping a strong REAL ring gives the same shift on a weak real CCD frame, and on
known-truth synthetic images the gate does nothing. These tests only pin the rule's mechanics.
"""

from __future__ import annotations

import warnings

import pytest

from test_baseline_slope_bias import _params

from midas_calibrate import estep
from midas_calibrate.estep import _gate_noise_rings
from midas_calibrate.refine import FittedPoint
from midas_calibrate.rings import build_ring_table


def _fits(per_ring):
    """{ring_idx: (n, snr_baseline)} -> list of FittedPoint."""
    return [FittedPoint(Y_pix=0.0, Z_pix=0.0, ring_idx=r, snr=5.0, snr_baseline=s)
            for r, (n, s) in per_ring.items() for _ in range(n)]


RINGS = {0: (60, 12.0), 1: (60, 3.0), 2: (60, 0.9), 3: (40, 8.0), 4: (30, 1.4), 5: (50, 9.0)}


def test_noise_rings_are_dropped_whole():
    out = _gate_noise_rings(_fits(RINGS), 2.0)
    assert sorted({f.ring_idx for f in out}) == [0, 1, 3, 5]         # 2 (0.9) and 4 (1.4) are below 2
    assert sum(f.ring_idx == 1 for f in out) == 60                     # a weak REAL ring keeps every point


def test_gate_is_off_at_zero_and_silent():
    fits = _fits(RINGS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                 # no warning of any kind
        out = _gate_noise_rings(fits, 0.0)
    assert out == fits                                                 # nothing removed


def test_floor_keeps_the_best_ranked_rings():
    with pytest.warns(RuntimeWarning, match="fewer than 4"):
        out = _gate_noise_rings(_fits(RINGS), 50.0)                    # would drop all six
    assert sorted({f.ring_idx for f in out}) == [0, 1, 2, 5]           # the 4 with most points, ties by SNR


def test_a_ring_is_judged_by_its_median_not_its_best_point():
    """60 % of ring 2's points are at the noise level and 40 % are strong: median ~0.9, max ~12.

    An implementation that used the maximum (or the mean) would keep it."""
    fits = [f for f in _fits({0: (60, 12.0), 1: (60, 3.0), 3: (40, 8.0), 5: (50, 9.0)})]
    fits += _fits({2: (36, 0.9)}) + _fits({2: (24, 12.0)})
    out = _gate_noise_rings(fits, 2.0)
    assert 2 not in {f.ring_idx for f in out}                          # dropped on its median
    assert sorted({f.ring_idx for f in out}) == [0, 1, 3, 5]


def test_run_estep_applies_min_ring_snr_from_the_params(monkeypatch):
    """The gate runs inside run_estep (so v1 autocalibrate AND calibrate_v2's centroid path get it)."""
    p = _params()
    rt = build_ring_table(p)
    monkeypatch.setattr(estep, "integrate_cake", lambda *a, **k: None)
    monkeypatch.setattr(estep, "extract_fitted_points", lambda *a, **k: _fits(RINGS))
    p.MinRingSNR = 2.0
    _cake, fits = estep.run_estep(p, None, rt)
    assert sorted({f.ring_idx for f in fits}) == [0, 1, 3, 5]
    p.MinRingSNR = 0.0                                                 # default: unchanged
    _cake, fits = estep.run_estep(p, None, rt)
    assert sorted({f.ring_idx for f in fits}) == [0, 1, 2, 3, 4, 5]
