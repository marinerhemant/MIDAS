"""The error bar on a centroid, and the specific way it goes wrong.

These encode a failure found on real ID03 data (LAB_NOTEBOOK §11b): the pipeline
propagated only the background-SUBTRACTED counts, which understated the per-pixel error
bar by ~5-6x. Subtracting a background removes its mean, never its variance -- and the
near-baseline bins carry the largest lever arms, so they dominate var(centroid).
"""
import numpy as np
import torch

from midas_dfxm import centroid_uncertainty


def _profile(n=25, step=0.04, width=0.05, amp=4e4, bg=1e3, seed=0):
    x = np.arange(n) * step
    c = x[n // 2]
    peak = amp * np.exp(-0.5 * ((x - c) / width) ** 2)
    return x, peak, np.full(n, float(bg))


def test_matches_analytic_for_a_pure_poisson_peak():
    x, peak, _ = _profile(bg=0.0)
    got = centroid_uncertainty(torch.tensor(peak)[None], x, gain=1.0)
    S = peak.sum()
    c = (peak * x).sum() / S
    want = np.sqrt(((x - c) ** 2 * peak).sum()) / S
    assert abs(float(got[0]) - want) < 1e-9 * max(want, 1e-12)


def test_subtracted_background_still_contributes_variance():
    """The regression test for the real bug. The centroid and its normalisation come from
    the SUBTRACTED profile -- that is the number you report -- but the variance must use the
    counts the detector actually recorded. Checked against the closed form."""
    x, peak, bg = _profile()
    got = float(centroid_uncertainty(torch.tensor(peak)[None], x,
                                     background_per_bin=torch.tensor(bg)[None], gain=1.0)[0])
    S = peak.sum()                      # subtracted sum: what normalises the centroid
    c = (peak * x).sum() / S
    want = np.sqrt(((x - c) ** 2 * (peak + bg)).sum()) / S     # recorded counts in the variance
    assert abs(got - want) < 1e-9 * want

    # and putting the background back into `counts` is a DIFFERENT estimator (it dilutes the
    # centroid as well), so it must NOT agree -- that distinction is the whole point.
    diluted = float(centroid_uncertainty(torch.tensor(peak + bg)[None], x, gain=1.0)[0])
    assert diluted != got


def test_ignoring_the_background_understates_the_error_bar():
    """And by a lot -- this is why the bug survived review: the wrong answer looks
    plausible, it is just far too small."""
    # baseline fraction matched to the real ID03 case: ~31.5% of the integrated weight
    x, peak, _ = _profile(bg=0.0)
    bg = np.full_like(peak, 0.315 * peak.sum() / (len(peak) * (1 - 0.315)))
    honest = float(centroid_uncertainty(torch.tensor(peak)[None], x,
                                        background_per_bin=torch.tensor(bg)[None], gain=1.0)[0])
    naive = float(centroid_uncertainty(torch.tensor(peak)[None], x, gain=1.0)[0])
    ratio = honest / naive
    # real-data value was ~3.6x in sigma (12.9x in variance); assert the direction and scale
    assert ratio > 2.5, f"expected a large understatement, got {ratio:.2f}x"


def test_gain_and_read_noise_scale_as_documented():
    x, peak, bg = _profile()
    a = float(centroid_uncertainty(torch.tensor(peak)[None], x,
                                   background_per_bin=torch.tensor(bg)[None], gain=1.0)[0])
    b = float(centroid_uncertainty(torch.tensor(peak)[None], x,
                                   background_per_bin=torch.tensor(bg)[None], gain=4.0)[0])
    assert abs(b / a - 2.0) < 1e-6, "sigma must scale as sqrt(gain)"
    c = float(centroid_uncertainty(torch.tensor(peak)[None], x,
                                   background_per_bin=torch.tensor(bg)[None],
                                   gain=1.0, read_var=500.0)[0])
    assert c > a


def test_numpy_and_torch_agree():
    x, peak, bg = _profile()
    t = float(centroid_uncertainty(torch.tensor(peak)[None], x,
                                   background_per_bin=torch.tensor(bg)[None])[0])
    n = centroid_uncertainty(peak[None], x, background_per_bin=bg[None])
    assert abs(float(n[0]) - t) < 1e-9 * t
