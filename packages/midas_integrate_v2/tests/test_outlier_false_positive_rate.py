"""A "5σ" cut must not throw away 1 % of good data.

``reject_cosmic_rays`` estimated the per-pixel σ with a MAD along the stack
axis. That is robust — it survives several outliers in one pixel's time series
— but the 1.4826 factor is the *asymptotic* consistency constant, and on a short
stack the MAD has enormous variance. Measured on a CLEAN Poisson stack at
``n_sigma=5`` (nominal Gaussian rate 5.7e-7):

    N=5  → 2.8e-2      N=9 → 8.0e-3      N=30 → 2.8e-4

so a 9-frame sweep "cleaned at 5σ" had ~0.8 % of its pixels flagged, and the
default ``mode="replace_with_median"`` OVERWRITES them. On a 2880² Varex that
is ~66 000 good pixels replaced, with nothing printed to say so.

The cause is variance, not bias: at N=5 the 5th percentile of σ_MAD/σ_true is
0.210, so 5 % of pixels get a σ five times too small. Rescaling to fix the
median only takes N=5 from 2.77 % to 1.49 %, so a corrected MAD is not the
remedy — a different estimator is.

``std`` is not the remedy either, and that is the trap: it has a zero false
positive rate but is not robust, so on a short stack a real cosmic ray inflates
its own σ and **hides itself** (verified below at N=5 and N=9).

``poisson`` — using the KNOWN photon-counting noise model rather than estimating
σ from five samples — is clean AND sensitive at every depth.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

LAM = 200.0
NOMINAL_5_SIGMA = 5.7e-7


def _clean(N, shape=(96, 96), seed=0):
    return np.random.default_rng(seed).poisson(
        LAM, size=(N,) + shape).astype(float)


def _spiked(N, seed=0, amp=50_000.0):
    s = _clean(N, seed=seed)
    s[N // 2, 20, 30] += amp
    return s


# ------------------------------------------------- the defect, pinned

@pytest.mark.parametrize("N", [5, 9])
def test_mad_on_a_short_stack_flags_far_too_much(N):
    """Not a regression guard — a RECORD of the known limitation, so that if
    someone improves the estimator this test fails and gets updated rather than
    the improvement going unnoticed."""
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, m = reject_cosmic_rays(_clean(N), n_sigma=5.0, mode="flag_only",
                                  sigma_model="mad")
    fpr = float(m.mean())
    assert fpr > 100 * NOMINAL_5_SIGMA, (
        f"MAD at N={N} now has FPR {fpr:.2e}; if this improved, update the "
        f"documented table in outlier.py")


def test_the_shallow_stack_warning_fires_and_names_the_rate():
    from midas_integrate_v2.streaming.outlier import (reject_cosmic_rays,
                                                      ShallowStackWarning)
    with pytest.warns(ShallowStackWarning, match=r"only 9 frames"):
        reject_cosmic_rays(_clean(9), n_sigma=5.0, mode="flag_only")


def test_the_warning_mentions_overwriting_only_when_it_overwrites():
    from midas_integrate_v2.streaming.outlier import (reject_cosmic_rays,
                                                      ShallowStackWarning)
    with pytest.warns(ShallowStackWarning) as rec:
        reject_cosmic_rays(_clean(9), n_sigma=5.0, mode="replace_with_median")
    assert "overwritten" in str(rec[0].message)
    with pytest.warns(ShallowStackWarning) as rec2:
        reject_cosmic_rays(_clean(9), n_sigma=5.0, mode="flag_only")
    assert "overwritten" not in str(rec2[0].message)


def test_a_deep_stack_does_not_warn():
    from midas_integrate_v2.streaming.outlier import (reject_cosmic_rays,
                                                      ShallowStackWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShallowStackWarning)
        reject_cosmic_rays(_clean(40), n_sigma=5.0, mode="flag_only")


# ------------------------------------------------- the remedy

@pytest.mark.parametrize("N", [5, 9, 30])
def test_poisson_model_is_clean_at_every_depth(N):
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    _, m = reject_cosmic_rays(_clean(N), n_sigma=5.0, mode="flag_only",
                              sigma_model="poisson")
    fpr = float(m.mean())
    assert fpr < 1e-4, f"poisson FPR at N={N} is {fpr:.2e}"


@pytest.mark.parametrize("N", [5, 9, 30])
def test_poisson_model_still_catches_a_real_spike(N):
    """Clean is worthless if it is also blind. This is the half that ``std``
    fails."""
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    s = _spiked(N)
    _, m = reject_cosmic_rays(s, n_sigma=5.0, mode="flag_only",
                              sigma_model="poisson")
    assert bool(m[N // 2, 20, 30]), f"poisson missed the spike at N={N}"


@pytest.mark.parametrize("N", [5, 9])
def test_std_is_blind_to_a_spike_on_a_short_stack(N):
    """The reason 'std' is not the fix, pinned so nobody makes it the default
    on the strength of its zero false-positive rate."""
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    s = _spiked(N)
    _, m = reject_cosmic_rays(s, n_sigma=5.0, mode="flag_only",
                              sigma_model="std")
    assert not bool(m[N // 2, 20, 30]), (
        f"std now catches the spike at N={N} — good, but the docs and the "
        f"CLI help say it does not; update them")


def test_poisson_floors_an_empty_pixel():
    """median = 0 would give sigma = 0 and flag any nonzero reading as
    infinitely many sigma."""
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    z = np.zeros((9, 8, 8))
    z[3, 4, 4] = 1.0            # a single count in an otherwise dead pixel
    _, m = reject_cosmic_rays(z, n_sigma=5.0, mode="flag_only",
                              sigma_model="poisson")
    assert not bool(m[3, 4, 4]), "one count in an empty pixel is not a cosmic ray"


def test_poisson_rejects_a_nonpositive_gain():
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    with pytest.raises(ValueError, match="gain"):
        reject_cosmic_rays(_clean(9), sigma_model="poisson", gain=0.0,
                           mode="flag_only")


# ------------------------------------------------- backward compatibility

def test_use_mad_still_selects_the_same_estimators():
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    c = _clean(9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = reject_cosmic_rays(c, mode="flag_only", use_mad=True)[1]
        b = reject_cosmic_rays(c, mode="flag_only", sigma_model="mad")[1]
        d = reject_cosmic_rays(c, mode="flag_only", use_mad=False)[1]
        e = reject_cosmic_rays(c, mode="flag_only", sigma_model="std")[1]
    assert np.array_equal(a, b)
    assert np.array_equal(d, e)


def test_sigma_model_overrides_use_mad():
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    c = _clean(9)
    a = reject_cosmic_rays(c, mode="flag_only", use_mad=True,
                           sigma_model="poisson")[1]
    b = reject_cosmic_rays(c, mode="flag_only", sigma_model="poisson")[1]
    assert np.array_equal(a, b)


def test_unknown_sigma_model_is_rejected():
    from midas_integrate_v2.streaming.outlier import reject_cosmic_rays

    with pytest.raises(ValueError, match="sigma_model"):
        reject_cosmic_rays(_clean(9), sigma_model="robust", mode="flag_only")
