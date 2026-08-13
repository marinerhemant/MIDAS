"""Ring indexing against candidate phases."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.index_rings import CEO2, PhaseCandidate, index_rings

pytest.importorskip("midas_hkls")


def test_ceo2_indexes_its_own_reflections_exactly():
    """Sanity: a cell must index d-spacings generated from itself.

    Fluorite CeO2, a = 5.41165: d(111) = a/sqrt(3), d(200) = a/2,
    d(220) = a/sqrt(8). If these do not come back at ~0 ppm the reflection
    generator or the matcher is wrong, and every other result here is noise.
    """
    a = 5.41165
    d = [a / np.sqrt(3), a / 2.0, a / np.sqrt(8)]
    res = index_rings(d, CEO2, tolerance_ppm=100.0)
    assert res.n_matched == 3, res.describe()
    assert res.rms_residual_ppm < 100.0
    assert {tuple(sorted(m.hkl, reverse=True)) for m in res.matches} == {
        (1, 1, 1), (2, 0, 0), (2, 2, 0)}


def test_wrong_phase_matches_nothing_at_a_tight_tolerance():
    """The negative control that makes a positive result mean something.

    Measured: CeO2 matches 0 of the 6 observed U3O8-sample rings. If a cell
    matched everything, match COUNT would carry no information.
    """
    observed = [4.157, 3.437, 2.640, 7.412, 1.767, 3.369]
    res = index_rings(observed, CEO2, tolerance_ppm=2_000.0)
    assert res.n_matched == 0, res.describe()


def test_tolerance_controls_matching():
    a = 5.41165
    d_off = [a / np.sqrt(3) * 1.01]        # 10 000 ppm off
    assert index_rings(d_off, CEO2, tolerance_ppm=1_000.0).n_matched == 0
    assert index_rings(d_off, CEO2, tolerance_ppm=20_000.0).n_matched == 1


def test_residual_sign_is_observed_minus_calculated():
    a = 5.41165
    res = index_rings([a / np.sqrt(3) * 1.001], CEO2, tolerance_ppm=5_000.0)
    m = res.matches[0]
    assert m.residual_ppm == pytest.approx(1000.0, rel=0.05)


def test_unmatched_rings_are_reported_not_dropped():
    """A ring with no reflection must appear as unmatched, not vanish."""
    res = index_rings([4.157, 99.0], CEO2, tolerance_ppm=2_000.0)
    assert len(res.matches) == 2
    assert not res.matches[1].matched
    assert "--" in res.describe()


def test_rms_is_infinite_when_nothing_matched():
    res = index_rings([99.0], CEO2, tolerance_ppm=10.0)
    assert res.rms_residual_ppm == float("inf")


def test_radii_are_carried_through_for_reporting():
    a = 5.41165
    res = index_rings([a / 2.0], CEO2, radii_px=[248.0], tolerance_ppm=500.0)
    assert res.matches[0].radius_px == pytest.approx(248.0)


def test_phase_candidate_is_hashable_and_immutable():
    p = PhaseCandidate(name="x", space_group=225, a=1.0, b=1.0, c=1.0)
    hash(p)
    with pytest.raises(Exception):
        p.a = 2.0
