"""Unit tests for compute.auto_phase."""

import numpy as np
import pytest

from midas_process_grains.compute.auto_phase import (
    detect_phase, COMMON_PHASES, PhaseCandidate,
)


def test_detect_phase_finds_au_from_au_d_spacings():
    """Feeding Au's own d-spacings should pick Au_FCC."""
    au_ds = np.array([2.355, 2.039, 1.442, 1.230, 1.178])
    res = detect_phase(au_ds)
    assert res.best.name == "Au_FCC"
    assert res.best.space_group == 225
    assert res.score < 0.01   # essentially zero


def test_detect_phase_finds_ti_hcp_from_ti_d_spacings():
    ti_ds = np.array([2.555, 2.342, 2.243, 1.726, 1.476])
    res = detect_phase(ti_ds)
    assert res.best.name == "alpha_Ti_HCP"
    assert res.best.space_group == 194


def test_detect_phase_returns_score_dict_for_all():
    au_ds = np.array([2.355, 2.039, 1.442])
    res = detect_phase(au_ds)
    assert len(res.all_scores) == len(COMMON_PHASES)
    # Best score == minimum
    assert res.score == min(res.all_scores.values())


def test_detect_phase_with_custom_candidate_list():
    custom = [
        PhaseCandidate(name="X", space_group=999, lattice=(1, 1, 1, 90, 90, 90),
                       d_spacings_A=np.array([5.0, 3.0, 2.0])),
        PhaseCandidate(name="Y", space_group=998, lattice=(1, 1, 1, 90, 90, 90),
                       d_spacings_A=np.array([1.0, 0.5, 0.25])),
    ]
    res = detect_phase(np.array([4.9, 3.05, 1.95]), candidates=custom)
    assert res.best.name == "X"
