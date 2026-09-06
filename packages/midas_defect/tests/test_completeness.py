"""The four-way completeness audit.

The distinction MISSED vs ABSENT is the whole point: one indicts the analysis,
the other describes the sample. Every test here is built so that confusing them
would fail it.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from midas_defect.completeness import (
    audit_completeness, window_from_residuals, CompletenessAudit,
)

MASK = np.zeros((200, 200), bool)


def _call(pred, obs, assigned, mask=MASK, **kw):
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float).reshape(-1, 3)
    kw.setdefault("window_px", 5.0)
    kw.setdefault("window_omega_deg", 1.0)
    return audit_completeness(
        predicted_hkl=np.arange(len(pred) * 3).reshape(-1, 3) if False else kw.pop("hkl"),
        predicted_row=pred[:, 0], predicted_col=pred[:, 1],
        predicted_omega_deg=pred[:, 2],
        observed_row=obs[:, 0], observed_col=obs[:, 1],
        observed_omega_deg=obs[:, 2],
        assigned_hkl=assigned, mask=mask, **kw)


def test_indexed_missed_masked_absent_are_each_reached():
    hkl = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    pred = [(50, 50, 0.0),      # INDEXED: assigned
            (80, 80, 0.0),      # MISSED : a spot is there, not assigned
            (120, 120, 0.0),    # MASKED : behind the mask
            (160, 160, 0.0)]    # ABSENT : nothing there
    obs = [(50, 50, 0.0), (80.5, 80.2, 0.1)]
    mask = MASK.copy()
    mask[110:132, 110:132] = True
    a = _call(pred, obs, assigned=[(1, 0, 0)], mask=mask, hkl=hkl)
    assert a.counts == {"INDEXED": 1, "MISSED": 1, "MASKED": 1, "ABSENT": 1}
    assert list(a.verdicts) == ["INDEXED", "MISSED", "MASKED", "ABSENT"]
    assert a.n_predicted == 4
    assert a.observable == 3


def test_missed_is_not_reported_as_absent():
    """The failure mode this audit exists to catch."""
    hkl = [(2, 0, 0)]
    a = _call([(90, 90, 0.0)], [(91.0, 90.4, 0.2)], assigned=[], hkl=hkl)
    assert a.counts["MISSED"] == 1 and a.counts["ABSENT"] == 0
    assert a.missed[0]["hkl"] == (2, 0, 0)
    assert a.missed[0]["distance_px"] < 2.0


def test_absent_is_not_reported_as_missed():
    hkl = [(2, 0, 0)]
    a = _call([(90, 90, 0.0)], [(140.0, 140.0, 0.0)], assigned=[], hkl=hkl)
    assert a.counts["ABSENT"] == 1 and a.counts["MISSED"] == 0
    assert a.absent[0]["nearest_px"] > 5.0


def test_a_spot_at_the_right_place_but_wrong_omega_is_absent():
    hkl = [(2, 0, 0)]
    a = _call([(90, 90, 0.0)], [(90.0, 90.0, 9.0)], assigned=[], hkl=hkl)
    assert a.counts["ABSENT"] == 1


def test_an_assigned_reflection_is_never_masked():
    """Assigned means observed; calling it MASKED would be incoherent."""
    hkl = [(1, 0, 0)]
    mask = np.ones((200, 200), bool)
    a = _call([(50, 50, 0.0)], [(50, 50, 0.0)], assigned=[(1, 0, 0)],
              mask=mask, hkl=hkl)
    assert a.counts == {"INDEXED": 1, "MISSED": 0, "MASKED": 0, "ABSENT": 0}


def test_masked_beats_absent_so_the_detector_is_not_blamed_on_the_sample():
    hkl = [(3, 0, 0)]
    mask = MASK.copy()
    mask[40:62, 40:62] = True
    a = _call([(50, 50, 0.0)], [(180, 180, 0.0)], assigned=[], mask=mask, hkl=hkl)
    assert a.counts["MASKED"] == 1 and a.counts["ABSENT"] == 0


def test_partial_masking_below_the_fraction_still_searches():
    hkl = [(3, 0, 0)]
    mask = MASK.copy()
    mask[44:50, 44:56] = True                 # under half the box
    a = _call([(50, 50, 0.0)], [(180, 180, 0.0)], assigned=[], mask=mask, hkl=hkl)
    assert a.counts["ABSENT"] == 1 and a.counts["MASKED"] == 0


# ------------------------------------------------------------------ window

def test_window_comes_from_the_residuals():
    rp = np.arange(1.0, 101.0)
    ro = np.arange(0.01, 1.01, 0.01)
    wp, wo = window_from_residuals(rp, ro, percentile=90)
    assert wp == pytest.approx(np.percentile(rp, 90))
    assert wo == pytest.approx(np.percentile(ro, 90))


def test_window_refuses_to_be_invented():
    with pytest.raises(ValueError, match="must not be guessed"):
        window_from_residuals([], [])


def test_a_too_wide_window_manufactures_MISSED():
    """Why the window may not be a round number, demonstrated."""
    hkl = [(2, 0, 0)]
    far = [(120.0, 120.0, 0.0)]
    tight = _call([(90, 90, 0.0)], far, assigned=[], window_px=5.0, hkl=hkl)
    loose = _call([(90, 90, 0.0)], far, assigned=[], window_px=60.0, hkl=hkl)
    assert tight.counts["ABSENT"] == 1
    assert loose.counts["MISSED"] == 1          # the same data, a different verdict


def test_nonpositive_window_is_rejected():
    with pytest.raises(ValueError, match="window must be positive"):
        _call([(1, 1, 0.0)], [(1, 1, 0.0)], assigned=[], window_px=0.0,
              hkl=[(1, 0, 0)])


def test_empty_observation_list_is_all_absent():
    hkl = [(1, 0, 0), (0, 1, 0)]
    a = audit_completeness(
        predicted_hkl=np.array(hkl), predicted_row=np.array([10.0, 20.0]),
        predicted_col=np.array([10.0, 20.0]),
        predicted_omega_deg=np.array([0.0, 0.0]),
        observed_row=np.array([]), observed_col=np.array([]),
        observed_omega_deg=np.array([]), assigned_hkl=[], mask=MASK,
        window_px=5.0, window_omega_deg=1.0)
    assert a.counts["ABSENT"] == 2


def test_str_reports_all_four_and_the_window():
    hkl = [(1, 0, 0)]
    a = _call([(50, 50, 0.0)], [(50, 50, 0.0)], assigned=[(1, 0, 0)], hkl=hkl)
    s = str(a)
    for k in ("INDEXED", "MISSED", "MASKED", "ABSENT"):
        assert k in s
    assert "px" in s and "deg" in s
