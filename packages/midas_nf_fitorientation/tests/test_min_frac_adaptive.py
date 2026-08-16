"""Adaptive MinFracAccept: the threshold must adapt, and must stay OFF by default.

A screen threshold decides which candidate orientations reach the refine, so
raising it can drop a real voxel and silently change the microstructure. Two
things therefore have to hold:

  * default OFF -- MinFracAcceptSigma 0 leaves the parameter file's value
    untouched, byte for byte, so no existing reconstruction moves.
  * the formula reproduces the historical constant at the conditions that
    constant was tuned for, so switching it on is a no-op on a well-behaved
    reduction and only bites where 0.04 had stopped being selective.

Whether it changes a REAL map is not decidable here -- that needs the same data
fitted both ways and the .mic files compared voxel by voxel. See
recon/handoff/ab_min_frac.py.
"""
import logging

import pytest

from midas_nf_fitorientation.params import FitParams
from midas_nf_fitorientation.screen import resolve_min_frac_accept


def _p(**kw):
    p = FitParams()
    p.min_frac_accept = 0.04
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_off_by_default_returns_the_file_value_exactly():
    p = _p()
    assert p.min_frac_accept_sigma == 0.0
    got = resolve_min_frac_accept(p, p_lit=0.0251, n_spots=98.2)
    assert got == 0.04, "default must not perturb an existing reconstruction"


def test_reproduces_the_historical_constant_at_its_tuning_point():
    """0.81% lit, 98.2 spots/orientation -- the conditions 0.04 came from.

    If this drifts, enabling the adaptive rule silently re-tunes every legacy
    reconstruction instead of leaving well-behaved ones alone.
    """
    got = resolve_min_frac_accept(_p(min_frac_accept_sigma=3.5),
                                  p_lit=0.0081, n_spots=98.2)
    assert got == pytest.approx(0.0397, abs=5e-4)


def test_tightens_when_the_reduction_is_sensitive():
    """Same sample at 3.5 sigma: 2.51% lit -> 0.04 is only 1.6x chance."""
    got = resolve_min_frac_accept(_p(min_frac_accept_sigma=3.5),
                                  p_lit=0.0251, n_spots=98.2)
    assert got == pytest.approx(0.080, abs=2e-3)
    assert got > 0.04


def test_more_spots_per_orientation_allows_a_looser_threshold():
    """dhcp predicts 261 spots vs fcc's 98, so the chance estimate is tighter."""
    fcc = resolve_min_frac_accept(_p(min_frac_accept_sigma=3.5),
                                  p_lit=0.0251, n_spots=98.2)
    dhcp = resolve_min_frac_accept(_p(min_frac_accept_sigma=3.5),
                                   p_lit=0.0251, n_spots=261.0)
    assert dhcp < fcc
    assert dhcp == pytest.approx(0.059, abs=2e-3)


def test_always_above_the_chance_rate():
    for p_lit in (0.001, 0.008, 0.025, 0.05, 0.10):
        got = resolve_min_frac_accept(_p(min_frac_accept_sigma=3.5),
                                      p_lit=p_lit, n_spots=100.0)
        assert got > p_lit, "a threshold at or below chance accepts noise"


@pytest.mark.parametrize("p_lit,n_spots", [(0.0, 100.0), (1.0, 100.0),
                                           (-0.1, 100.0), (0.02, 0.0)])
def test_degenerate_inputs_fall_back_and_warn(p_lit, n_spots, caplog):
    with caplog.at_level(logging.WARNING):
        got = resolve_min_frac_accept(_p(min_frac_accept_sigma=3.5),
                                      p_lit=p_lit, n_spots=n_spots)
    assert got == 0.04
    assert "ignored" in " ".join(r.getMessage() for r in caplog.records)


def test_logs_what_it_chose_and_what_it_replaced(caplog):
    with caplog.at_level(logging.INFO):
        resolve_min_frac_accept(_p(min_frac_accept_sigma=3.5),
                                p_lit=0.0251, n_spots=98.2)
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "adaptive" in msg and "0.0400" in msg, (
        "the log must show BOTH the chosen value and the file's value, or a "
        "changed reconstruction is unattributable")
