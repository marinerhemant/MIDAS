"""choose_margin: a plateau must be shown by a margin SMALLER than the reference; the reference margin compares with itself, so it can never certify convergence
(thiourea run 01: the sweep ran to its largest margin, 6, which then reported converged=True with the median ratio still rising 0.95 -> 1.0)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from midas_sx.integrate import ProfileModel, build_profile, choose_margin


def _volume(halo: float, seed=0, n_spots=14):
    rng = np.random.default_rng(seed)
    shape = (40, 160, 160)
    f, r, c = np.indices(shape).astype(float)
    vol = np.full(shape, 40.0)
    centres = []
    for i in range(n_spots):
        cf, cr, cc = 20.0, 20.0 + 11 * (i // 3) + 3 * (i % 3), 20.0 + 22 * (i % 7)
        d2 = ((f - cf) / 0.8) ** 2 + ((r - cr) / 1.3) ** 2 + ((c - cc) / 1.3) ** 2
        vol += 3000.0 * np.exp(-0.5 * d2)
        if halo:
            rho = np.sqrt(((f - cf) / 2.0) ** 2 + (r - cr) ** 2 + (c - cc) ** 2)
            vol += halo * 3000.0 / (1.0 + rho ** 2)            # slowly decaying tail: the integral keeps growing with the box
        centres.append((cf, cr, cc))
    return vol.astype(np.float32), np.array(centres)


MODEL = ProfileModel(sigma_px=1.3, sf0=0.3, sf1=0.0, k=3.0, ext_f0=2.4, ext_f1=0.0, ext_p=4.0, margin=0)


def test_compact_spots_reach_a_plateau_below_the_largest_margin():
    vol, centres = _volume(halo=0.0)
    m, info = choose_margin(vol, centres, np.full(len(centres), 3.0), MODEL)
    assert info["converged"] and info["chosen"] < 6, info["table"]
    assert m.margin == info["chosen"]


def test_growing_tails_are_reported_as_not_converged_and_warn_in_build_profile():
    vol, centres = _volume(halo=0.06)
    m, info = choose_margin(vol, centres, np.full(len(centres), 3.0), MODEL)
    assert not info["converged"] and info["chosen"] == 6, info["table"]
    # the same condition must reach the user through build_profile
    import midas_sx.integrate as I
    orig = I.measure_extents
    I.measure_extents = lambda *a, **k: (MODEL, {})
    try:
        with pytest.warns(RuntimeWarning, match="still growing"):
            build_profile(vol, centres, np.full(len(centres), 3.0))
    finally:
        I.measure_extents = orig
