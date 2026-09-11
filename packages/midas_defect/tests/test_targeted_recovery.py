"""Targeted extraction at predicted sites must beat a same-ring null (2026-09-10 fold-in).

Ported from the La3Ni2O7 project's repro/predicted_recovery.py. The two failure
modes pinned here are the ones the original was written against: a powder ring
scoring as "recovery" at every azimuth, and a flat photon-counter patch giving an
infinite signal-to-noise.
"""
import math

import numpy as np

from midas_defect.completeness import targeted_recovery, targeted_snr

NZ, NY, BC = 220, 240, (110.0, 120.0)


def _scene(seed=0, ring_amp=900.0, subtract_ring=True, spots=()):
    rng = np.random.default_rng(seed)
    zz, yy = np.mgrid[0:NZ, 0:NY]
    r = np.hypot(zz - BC[0], yy - BC[1])
    model = 200.0 + ring_amp * np.exp(-0.5 * ((r - 80.0) / 1.5) ** 2)
    sig = np.zeros((NZ, NY))
    for (pr, pc) in spots:
        sig += 4000.0 * np.exp(-0.5 * (((zz - pr) / 1.4) ** 2 + ((yy - pc) / 1.4) ** 2))
    raw = rng.poisson(model + sig, size=(2, NZ, NY)).astype(np.float32)
    sub = raw - (model if subtract_ring else 200.0)
    return sub, raw, np.zeros((NZ, NY), bool)


def _on_ring(az_deg, radius=80.0):
    a = np.radians(np.asarray(az_deg, float))
    r = np.broadcast_to(np.asarray(radius, float), a.shape)
    return BC[0] + r * np.sin(a), BC[1] + r * np.cos(a)


def test_planted_reflections_are_recovered_above_the_same_ring_null():
    """Predicted sites at DISTINCT radii, as real predictions are.

    (A first version put all eight sites on one radius. Random azimuths on that
    radius then kept landing on the four planted spots -- null rate 0.22, excess
    1.9 sigma. That is the null doing its job, not a failure of the method: a
    scene whose real reflections all share one |q| IS a ring, to this test.)
    """
    pr, pc = _on_ring([10, 100, 190, 280], radius=[50.0, 65.0, 95.0, 80.0])       # planted
    er, ec = _on_ring([55, 145, 235, 325], radius=[58.0, 72.0, 88.0, 102.0])      # nothing there
    sub, raw, mask = _scene(spots=list(zip(pr, pc)))
    res = targeted_recovery(sub, raw, mask, np.zeros(8, int), np.r_[pr, er], np.r_[pc, ec],
                            beam_centre=BC, n_null=40)
    assert res.n_scored == 8
    assert res.recovered[:4].all() and not res.recovered[4:].any(), res.snr
    assert res.null_rate < 0.1, str(res)
    assert res.excess_sigma > 3.0, str(res)


def test_an_unsubtracted_ring_is_not_recovery():
    """Every site on a bright ring clears SNR 5 -- and so does the null. Excess must stay small."""
    rr, cc = _on_ring(np.arange(0, 360, 30))
    sub, raw, mask = _scene(subtract_ring=False)
    res = targeted_recovery(sub, raw, mask, np.zeros(len(rr), int), rr, cc, beam_centre=BC, n_null=40)
    assert res.n_recovered >= 10, str(res)             # the raw count looks like a triumph
    assert res.null_rate > 0.8, str(res)               # the null sees the same ring
    assert res.excess_sigma < 2.0, str(res)            # and the excess says it is nothing


def test_a_flat_patch_is_floored_at_poisson_not_infinite():
    sub = np.zeros((1, 80, 80)); raw = np.zeros((1, 80, 80))
    sub[0, 40, 40] = 3.0
    mask = np.zeros((80, 80), bool)
    snr, _ = targeted_snr(sub, raw, mask, 0, 40, 40)
    assert math.isfinite(snr) and snr == 3.0           # noise = sqrt(0 + 1), not MAD = 0
    raw[:] = 1000.0
    snr, _ = targeted_snr(sub, raw, mask, 0, 40, 40)
    assert snr < 0.1


def test_a_masked_or_edge_site_is_not_scored():
    sub = np.ones((1, 80, 80)); raw = np.ones((1, 80, 80)); mask = np.zeros((80, 80), bool)
    mask[39:42, 39:42] = True
    assert targeted_snr(sub, raw, mask, 0, 40, 40) is None
    assert targeted_snr(sub, raw, np.zeros((80, 80), bool), 0, 5, 40) is None
    res = targeted_recovery(sub, raw, mask, [0, 0], [40, 5], [40, 40], beam_centre=(40, 40))
    assert res.n_scored == 0 and res.n_recovered == 0 and np.isnan(res.snr).all()
