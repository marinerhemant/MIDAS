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


def test_legacy_per_voxel_kernel_has_no_free_variables():
    """The mixed-grid-size screen path must not reference undefined names.

    ``_screen_per_voxel`` only fires when ``gs`` varies across the batch, so it
    is never exercised by the normal path -- which is exactly why two free
    variables (``spot_weight``, and later ``_min_frac``) survived in it
    unnoticed. It would raise NameError the first time a real mixed-grid run
    reached it. Checked statically rather than by calling it, because
    constructing that state needs the whole screen fixture.
    """
    import symtable
    from pathlib import Path
    import midas_nf_fitorientation.screen as S

    src = Path(S.__file__).read_text()
    st = symtable.symtable(src, "screen.py", "exec")
    fn = next(c for c in st.get_children() if c.get_name() == "_screen_per_voxel")
    free = sorted(s.get_name() for s in fn.get_symbols() if s.is_free())
    assert not free, f"free variables in _screen_per_voxel: {free}"


# ---------------------------------------------------------------------------
#  Refine early-exit (candidate-rank waves)
# ---------------------------------------------------------------------------

def test_early_exit_selects_the_same_winner_as_refine_all():
    """The writeback's early exit already decides the answer.

    Kept as documentation of a measured dead end: refining candidates lazily by
    rank so the exit could SAVE work made the fit 26-43% slower (thousands of
    small GPU launches, and per-chunk tensor rebuilds), so the eager pre-pass
    stays. The selection property below is what made the idea look safe, and is
    still true -- it is the throughput that killed it, not correctness.

    The writeback stops at the first candidate with hard_frac > 1 - 1e-4 (and
    the C FitOrientationOMP does the same), so candidates ranked after it never
    influence the result. Refining them was pure waste -- 653 per voxel on real
    data. This simulates both orders and asserts the recorded winner is
    identical, which is the property that makes the optimisation safe.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        fracs = rng.uniform(0.0, 1.0, n)
        if rng.random() < 0.6:                    # often a near-perfect hit
            fracs[int(rng.integers(0, n))] = 1.0 - rng.uniform(0, 5e-5)

        # (a) refine-all, then break in the writeback (current behaviour)
        best_all, chosen_all = -1.0, None
        for i, f in enumerate(fracs):
            if f > best_all:
                best_all, chosen_all = f, i
            if f > 1.0 - 1e-4:
                break

        # (b) wave-based: stop refining once a voxel is satisfied
        best_w, chosen_w, refined = -1.0, None, 0
        for i, f in enumerate(fracs):
            refined += 1
            if f > best_w:
                best_w, chosen_w = f, i
            if f > 1.0 - 1e-4:
                break

        assert chosen_w == chosen_all
        assert best_w == best_all
        assert refined <= n


