"""Friedel-pair calibration: beam centre, the antipodal gate, and tx.

The tx estimator is validated the only way an estimator like this can be:
plant a known value in a simulator built from the diffraction condition, and
recover it. A test that only checks self-consistency would pass on a sign error.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from midas_calibrate_v2.friedel import (
    friedel_domega_floor, is_friedel, antipodal_pairs,
    beam_centre_from_pairs, predicted_eta, tx_from_pairs,
)

LAM = 0.42459          # Å
LSD = 349_622.0        # µm
PX = 172.0             # µm
CR, CC = 810.25, 737.96


# ------------------------------------------------------------------ simulator

def _omega_solutions(g_sample, lam):
    """ω (deg) at which G_sample satisfies Bragg, rotating about +z."""
    gx, gy, gz = g_sample
    rho = np.hypot(gx, gy)
    q2 = gx * gx + gy * gy + gz * gz
    c = lam * q2 / 2.0
    if rho < 1e-12 or abs(c / rho) > 1.0:
        return []
    phi = np.arctan2(gy, gx)
    a = np.arccos(-c / rho)
    return [np.degrees(a - phi), np.degrees(-a - phi)]


def _detector_xy(g_sample, omega_deg, lam, lsd, px, tx_deg=0.0):
    """Detector (row, col) for a reflection at this ω, with detector roll tx."""
    w = np.radians(omega_deg)
    R = np.array([[np.cos(w), -np.sin(w), 0.0],
                  [np.sin(w),  np.cos(w), 0.0],
                  [0.0, 0.0, 1.0]])
    gx, gy, gz = R @ np.asarray(g_sample, float)
    kfx = 1.0 / lam + gx
    if kfx <= 0:
        return None
    y = lsd * gy / kfx          # horizontal, µm
    z = lsd * gz / kfx          # vertical, µm
    t = np.radians(tx_deg)      # roll about the beam
    yr = y * np.cos(t) - z * np.sin(t)
    zr = y * np.sin(t) + z * np.cos(t)
    return (CR + zr / px, CC + yr / px)


def _wrap(d):
    """Wrap an omega difference into (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def _make_pairs(n_refl=140, tx_deg=0.0, seed=3, noise_px=0.0):
    """Friedel pairs (row1,col1,w1,row2,col2,w2) from random reflections.

    Each of +G and -G has TWO omega solutions, and only one of the four
    combinations is the Friedel pair — the one that lands antipodally on the
    detector. Picking a branch blindly gives two spots of the same |G| that are
    not a pair at all, which is precisely the confusion `is_friedel` exists to
    resolve. Select by antipodality here so the simulator is ground truth.
    """
    rng = np.random.default_rng(seed)
    out = []
    guard = 0
    while len(out) < n_refl and guard < 200 * n_refl:
        guard += 1
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        g = v * rng.uniform(0.12, 0.45)          # 1/Å
        w_pos = _omega_solutions(g, LAM)
        w_neg = _omega_solutions(-np.asarray(g), LAM)
        if not w_pos or not w_neg:
            continue
        best = None
        for w1 in w_pos:
            p1 = _detector_xy(g, w1, LAM, LSD, PX, tx_deg)
            if p1 is None:
                continue
            for w2 in w_neg:
                p2 = _detector_xy(-np.asarray(g), w2, LAM, LSD, PX, tx_deg)
                if p2 is None:
                    continue
                off = np.hypot(0.5 * (p1[0] + p2[0]) - CR,
                               0.5 * (p1[1] + p2[1]) - CC)
                if best is None or off < best[0]:
                    best = (off, p1, p2, w1, w2)
        if best is None or best[0] > 1e-6:
            continue
        _, p1, p2, w1, w2 = best
        if not (60 < np.hypot(p1[0] - CR, p1[1] - CC) < 800):
            continue
        if noise_px:
            p1 = (p1[0] + rng.normal(0, noise_px), p1[1] + rng.normal(0, noise_px))
            p2 = (p2[0] + rng.normal(0, noise_px), p2[1] + rng.normal(0, noise_px))
        out.append((p1[0], p1[1], 0.0, p2[0], p2[1], _wrap(w2 - w1)))
    return out


# ------------------------------------------------------------------ the floor

def test_domega_floor_matches_the_closed_form():
    q = np.array([0.1, 0.3, 0.5])
    got = friedel_domega_floor(q, LAM)
    want = 2.0 * np.degrees(np.arcsin(q * LAM / 2.0))
    assert np.allclose(got, want)


def test_domega_floor_is_nan_beyond_the_limiting_sphere():
    assert np.isnan(friedel_domega_floor(np.array([100.0]), LAM))[0]


def test_the_floor_rises_with_q():
    q = np.linspace(0.05, 0.6, 20)
    f = friedel_domega_floor(q, LAM)
    assert np.all(np.diff(f) > 0)


def test_is_friedel_rejects_a_pair_below_its_floor():
    """The whole point: an antipodal pair too close in omega is a different grain."""
    radius = 400.0
    q = 2 * np.sin(np.arctan2(radius * PX, LSD) / 2) / LAM
    floor = float(friedel_domega_floor(np.array([q]), LAM)[0])
    assert is_friedel(radius, floor + 0.5, wavelength_A=LAM, lsd_um=LSD, pixel_um=PX)
    assert not is_friedel(radius, floor - 0.5, wavelength_A=LAM, lsd_um=LSD, pixel_um=PX)


def test_simulated_pairs_all_pass_the_gate():
    """A gate that rejects genuine pairs would be worse than no gate."""
    pairs = _make_pairs(80)
    rad = np.array([np.hypot(p[0] - CR, p[1] - CC) for p in pairs])
    dw = np.array([p[5] - p[2] for p in pairs])
    ok = is_friedel(rad, dw, wavelength_A=LAM, lsd_um=LSD, pixel_um=PX,
                    tolerance_deg=1e-6)
    assert ok.all(), f"gate rejected {int((~ok).sum())} of {len(ok)} genuine pairs"


# ------------------------------------------------------------- beam centre

def test_beam_centre_recovered_from_planted_pairs():
    pairs = _make_pairs(120)
    pts = np.array([[p[0], p[1]] for p in pairs] + [[p[3], p[4]] for p in pairs])
    res = beam_centre_from_pairs(pts, seed=(CR + 9.0, CC - 7.0),
                                 search_px=20.0, step_px=1.0, tol_px=2.0,
                                 n_null=60)
    assert res.row == pytest.approx(CR, abs=0.5)
    assert res.col == pytest.approx(CC, abs=0.5)
    assert res.n_pairs >= 50
    assert res.p_value < 0.05
    assert np.median(res.null_counts) < res.n_matched


def test_null_keeps_the_radii_so_it_could_have_matched():
    """A null that cannot produce matches proves nothing."""
    pairs = _make_pairs(120)
    pts = np.array([[p[0], p[1]] for p in pairs] + [[p[3], p[4]] for p in pairs])
    res = beam_centre_from_pairs(pts, seed=(CR, CC), search_px=6.0,
                                 tol_px=4.0, n_null=120)
    assert res.null_counts.max() > 0, "the null never matched anything — it cannot fail"


def test_beam_centre_refuses_a_non_centrosymmetric_cloud():
    """No pairs must mean an error, never a confident number from nothing."""
    rng = np.random.default_rng(1)
    th = rng.uniform(0.3, 1.2, 150)                  # one quadrant only
    r = rng.uniform(100, 700, 150)
    pts = np.stack([CR + r * np.sin(th), CC + r * np.cos(th)], 1)
    with pytest.raises(RuntimeError, match="no antipodal pairs"):
        beam_centre_from_pairs(pts, seed=(CR, CC), search_px=6.0,
                               tol_px=3.0, n_null=20)


def test_beam_centre_rejects_bad_input():
    with pytest.raises(ValueError):
        beam_centre_from_pairs(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        beam_centre_from_pairs(np.zeros((10, 3)))


def test_antipodal_pairs_are_distinct_and_symmetric():
    pairs = _make_pairs(40)
    pts = np.array([[p[0], p[1]] for p in pairs] + [[p[3], p[4]] for p in pairs])
    pr = antipodal_pairs(pts, (CR, CC), tol_px=2.0)
    assert len(pr) > 0
    assert len({tuple(sorted(t)) for t in pr.tolist()}) == len(pr)
    assert np.all(pr[:, 0] != pr[:, 1])


# ---------------------------------------------------------------------- tx

def test_predicted_eta_has_no_free_parameter_and_brackets_the_truth():
    pairs = _make_pairs(40, tx_deg=0.0)
    hits = 0
    for (r1, c1, w1, r2, c2, w2) in pairs:
        radius = np.hypot(r1 - CR, c1 - CC)
        br = predicted_eta(radius, w2 - w1, wavelength_A=LAM, lsd_um=LSD, pixel_um=PX)
        if br is None:
            continue
        eta = np.degrees(np.arctan2(r1 - CR, c1 - CC))
        if min(abs((eta - b + 180) % 360 - 180) for b in br) < 1.0:
            hits += 1
    assert hits > 0.8 * len(pairs), f"only {hits}/{len(pairs)} pairs bracketed"


@pytest.mark.parametrize("planted", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
def test_tx_is_recovered_from_a_planted_value(planted):
    """THE positive control. Plant tx, recover it, 1:1."""
    pairs = _make_pairs(160, tx_deg=planted, seed=7)
    res = tx_from_pairs(pairs, (CR, CC), wavelength_A=LAM, lsd_um=LSD,
                        pixel_um=PX, n_bootstrap=400)
    assert res.tx_deg == pytest.approx(planted, abs=0.25), \
        f"planted {planted}, recovered {res.tx_deg:+.3f} from {res.n_pairs} pairs"


def test_tx_sign_is_not_inverted():
    """A sign error is the single most likely defect here; pin it."""
    pos = tx_from_pairs(_make_pairs(160, tx_deg=+3.0, seed=11), (CR, CC),
                        wavelength_A=LAM, lsd_um=LSD, pixel_um=PX, n_bootstrap=200)
    neg = tx_from_pairs(_make_pairs(160, tx_deg=-3.0, seed=11), (CR, CC),
                        wavelength_A=LAM, lsd_um=LSD, pixel_um=PX, n_bootstrap=200)
    assert pos.tx_deg > 2.0 and neg.tx_deg < -2.0


def test_tx_survives_realistic_positional_noise():
    """With 0.5 px scatter on every spot the CI must still bracket the truth."""
    res = tx_from_pairs(_make_pairs(200, tx_deg=1.0, seed=13, noise_px=0.5),
                        (CR, CC), wavelength_A=LAM, lsd_um=LSD, pixel_um=PX)
    assert res.ci_low < res.ci_high, "noiseless-looking CI on noisy input"
    assert res.ci_low <= 1.0 <= res.ci_high, \
        f"CI [{res.ci_low:.3f}, {res.ci_high:.3f}] misses the planted 1.0"
    assert res.tx_deg == pytest.approx(1.0, abs=0.3)


def test_tx_refuses_when_no_pair_is_usable():
    with pytest.raises(RuntimeError):
        tx_from_pairs([(CR + 10, CC, 0.0, CR - 10, CC, 0.0)], (CR, CC),
                      wavelength_A=LAM, lsd_um=LSD, pixel_um=PX)
