"""friedel_wedge: absolute Wedge + OmegaStep scale from Friedel pairs.

The synthetic here is generated FORWARD (crystal -> exact Bragg omegas about a
tilted axis -> ray from the grain's position to the detector -> raw omega with
a step-scale error), a different code path from the fitter, which goes
detector -> G -> mate. Truth is drawn off-grid; grain positions, omega noise and
pixel noise are all present so the fitter is not handed its own model.
"""
import math

import numpy as np
import pytest

from midas_joint_ff_calibrate.friedel_wedge import (
    corrected_params_text, find_friedel_pairs, fit_wedge_omega_scale,
    predict_mate_omega,
)

LSD = 1.0e6          # um
LAM = 0.1729787      # A
OME_REF = -179.75


def _rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def _fcc_hkls(nmax=4):
    out = []
    for h in range(-nmax, nmax + 1):
        for k in range(-nmax, nmax + 1):
            for l in range(-nmax, nmax + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if len({h % 2, k % 2, l % 2}) == 1:      # fcc: all even or all odd
                    out.append((h, k, l))
    hkl = np.array(out, float)
    s = (hkl ** 2).sum(1)
    keep = s <= 20                                       # rings up to (420)
    ring = np.unique(s[keep], return_inverse=True)[1] + 1
    return hkl[keep], ring


def simulate(n_grains, wedge_deg, eps, *, a=3.59, ome_noise=0.04, px_noise=15.0,
             pos_spread=400.0, seed=0):
    """Spots (grain, ring, raw omega, Y, Z) at a known wedge and step error.

    raw omega relates to true omega by  true = REF + (raw - REF) * (1 + eps).
    """
    rng = np.random.default_rng(seed)
    hkl, ring = _fcc_hkls()
    W = math.radians(wedge_deg); cW, sW = math.cos(W), math.sin(W)
    k = 1.0 / LAM
    G_, R_, O_, Y_, Z_ = [], [], [], [], []
    pos = {}
    for g in range(n_grains):
        U = _rand_rot(rng)
        p = rng.uniform(-pos_spread, pos_spread, 3) * np.array([1, 1, 0.12])
        pos[g] = p
        gs = (hkl / a) @ U.T                                   # stage-frame G, 1/A
        gn = np.linalg.norm(gs, axis=1)
        gz = gs[:, 2]; gh = np.hypot(gs[:, 0], gs[:, 1]); al = np.arctan2(gs[:, 1], gs[:, 0])
        # R_y(-W) R_z(w) g : x' = cW*mx - sW*gz must equal -|g|^2/(2k)
        c = (-gn ** 2 / (2 * k) + sW * gz) / (cW * gh)
        ok = np.abs(c) < 1
        for sgn in (1, -1):
            w = sgn * np.arccos(np.clip(c, -1, 1)) - al
            mx = gh * np.cos(w + al); my = gh * np.sin(w + al)
            Gx = cW * mx - sW * gz; Gy = my; Gz = sW * mx + cW * gz
            kout = np.stack([k + Gx, Gy, Gz], 1)
            # grain position in the lab at this omega, then the ray to the detector
            pmx = p[0] * np.cos(w) - p[1] * np.sin(w); pmy = p[0] * np.sin(w) + p[1] * np.cos(w)
            plx = cW * pmx - sW * p[2]; plz = sW * pmx + cW * p[2]
            t = (LSD - plx) / kout[:, 0]
            Y = pmy + t * kout[:, 1]; Z = plz + t * kout[:, 2]
            eta = np.degrees(np.arctan2(-Y, Z))
            om_true = (np.degrees(w) + 180.0) % 360.0 - 180.0
            m = ok & (np.abs(eta) > 6) & (np.abs(eta) < 174) & (np.hypot(Y, Z) < 2.0e5)
            om_raw = OME_REF + (om_true - OME_REF) / (1.0 + eps)
            om_raw = om_raw + rng.normal(0, ome_noise, om_raw.size)
            G_.append(np.full(m.sum(), g)); R_.append(ring[m]); O_.append(om_raw[m])
            Y_.append(Y[m] + rng.normal(0, px_noise, m.sum()))
            Z_.append(Z[m] + rng.normal(0, px_noise, m.sum()))
    cat = np.concatenate
    return cat(G_), cat(R_), cat(O_), cat(Y_), cat(Z_), pos


def _fit(wedge, eps, **kw):
    g, r, o, y, z, pos = simulate(250, wedge, eps, **kw)
    pairs = find_friedel_pairs(g, r, o, y, z, pos_by_grain=pos)
    x, s, n, boots = fit_wedge_omega_scale(pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=30)
    return x, boots.std(axis=0), n


def test_predict_mate_is_plus_180_without_wedge():
    rng = np.random.default_rng(1)
    y = rng.uniform(-1.5e5, 1.5e5, 50); z = rng.uniform(-1.5e5, 1.5e5, 50)
    oa = rng.uniform(-180, 0, 50)
    pred = predict_mate_omega(y, z, oa, LSD, 0.0, oa + 180.0)
    np.testing.assert_allclose(pred, oa + 180.0, atol=1e-9)


def test_recovers_wedge_and_omega_step_jointly():
    W_true, eps_true = 0.0173, 2.7e-4                    # off-grid truth
    x, se, n = _fit(W_true, eps_true, seed=3)
    assert n > 2000
    assert abs(x[0] - W_true) < max(4 * se[0], 0.0015), (x, se)
    assert abs(x[1] - eps_true) < max(4 * se[1], 2e-5), (x, se)


def test_negative_wedge_sign():
    x, se, _ = _fit(-0.0119, 0.0, seed=4)
    assert abs(x[0] + 0.0119) < max(4 * se[0], 0.0015), (x, se)


def test_null_wedge_with_step_error_does_not_leak():
    """A step-scale error alone must not show up as a wedge (it does in a
    per-pair fit without the scale term)."""
    x, se, _ = _fit(0.0, 5e-4, seed=5)
    assert abs(x[0]) < max(4 * se[0], 0.0015), (x, se)
    assert abs(x[1] - 5e-4) < max(4 * se[1], 2e-5)


def test_without_scale_term_the_step_error_biases_the_wedge_one_sidedly():
    """Documents WHY the scale term is fitted: hold eps at 0 on data that has a
    step error and the wedge from one side of the beam differs from the other."""
    g, r, o, y, z, pos = simulate(250, 0.0, 5e-4, seed=6)
    pairs = find_friedel_pairs(g, r, o, y, z, pos_by_grain=pos)
    cot = 1 / np.tan(np.radians(pairs.eta_a))
    up = fit_wedge_omega_scale(pairs.subset(cot > 0), lsd_um=LSD, ome_ref=OME_REF,
                               fit_omega_scale=False, n_boot=0)[0][0]
    dn = fit_wedge_omega_scale(pairs.subset(cot < 0), lsd_um=LSD, ome_ref=OME_REF,
                               fit_omega_scale=False, n_boot=0)[0][0]
    assert up * dn < 0 and abs(up - dn) > 0.01


def test_pairs_are_ordered_and_unique():
    g, r, o, y, z, pos = simulate(40, 0.01, 0.0, seed=7)
    pairs = find_friedel_pairs(g, r, o, y, z)
    assert len(pairs) > 100
    assert np.all(pairs.ome_b > pairs.ome_a)
    sep = (pairs.ome_b - pairs.ome_a - 180.0 + 180.0) % 360.0 - 180.0
    assert np.all(np.abs(sep) < 1.0)


def test_corrected_params_text_writes_wedge_and_optional_step():
    txt = "Lsd 1000000\nWedge 0\nOmegaStep 0.250514\n"
    out = corrected_params_text(txt, wedge_deg=0.0233)
    assert "Wedge 0.0233" in out and "OmegaStep 0.250514" in out
    out = corrected_params_text(txt, wedge_deg=0.0233, omega_step=0.25058)
    assert "OmegaStep 0.25058" in out
    out = corrected_params_text("Lsd 1\n", wedge_deg=-0.01)
    assert out.strip().endswith("Wedge -0.01")


# ── raw peak-list pairing (no grain labels) ─────────────────────────────────
from midas_joint_ff_calibrate.friedel_wedge import find_friedel_pairs_peaks  # noqa: E402


def _fit_peaks(wedge, eps, n_grains=800, seed=11):
    g, r, o, y, z, _ = simulate(n_grains, wedge, eps, seed=seed)
    pairs = find_friedel_pairs_peaks(r, o, y, z)
    x, s, n, boots = fit_wedge_omega_scale(pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=20)
    # fraction of pairs that are true Friedel mates (same simulated grain)
    return x, boots.std(axis=0), n, pairs


def test_peaks_pairing_recovers_wedge_and_step_in_a_crowded_field():
    x, se, n, pairs = _fit_peaks(0.0211, 3.1e-4)            # off-grid truth
    assert n > 5000
    assert abs(x[0] - 0.0211) < max(4 * se[0], 0.0015), (x, se)
    assert abs(x[1] - 3.1e-4) < max(4 * se[1], 2e-5), (x, se)


def test_peaks_pairing_null():
    x, se, n, _ = _fit_peaks(0.0, 0.0, seed=12)
    assert abs(x[0]) < max(4 * se[0], 0.0015) and abs(x[1]) < max(4 * se[1], 2e-5), (x, se)


def _drop(sim, frac, seed):
    g, r, o, y, z, pos = sim
    k = np.random.default_rng(seed).random(len(o)) >= frac
    return g[k], r[k], o[k], y[k], z[k], pos


def _wrong_fraction(g, o, y, pairs):
    lut = {(round(a, 6), round(b, 3)): gg for gg, a, b in zip(g, o, y)}
    lb = {round(a, 6): gg for gg, a in zip(g, o)}
    ga = np.array([lut[(round(a, 6), round(b, 3))] for a, b in zip(pairs.ome_a, pairs.y_a)])
    gb = np.array([lb[round(b, 6)] for b in pairs.ome_b])
    return float(np.mean(ga != gb))


@pytest.mark.parametrize("n_grains,drop", [(100, 0.0), (800, 0.0), (1500, 0.3), (1500, 0.6)])
def test_chance_pair_fraction_tracks_the_true_wrong_pair_rate(n_grains, drop):
    """Checked against synthetic truth, including missing mates (drop): a wrong
    pair needs BOTH true mates missing. The raw decoy count alone read 3-35 %
    where the truth was ~0."""
    from midas_joint_ff_calibrate.friedel_wedge import chance_pair_fraction
    g, r, o, y, z, _ = _drop(simulate(n_grains, 0.02, 0.0, seed=21), drop, 5)
    pairs = find_friedel_pairs_peaks(r, o, y, z)
    true = _wrong_fraction(g, o, y, pairs)
    est = chance_pair_fraction(r, o, y, z, len(pairs))
    assert abs(est - true) < max(0.01, 0.35 * true), (n_grains, drop, est, true)


def test_wrong_pairs_do_not_bias_the_wedge():
    """13 % wrong pairs (crowded, half the mates missing): the Huber fit holds."""
    g, r, o, y, z, _ = _drop(simulate(1500, 0.0211, 2.0e-4, seed=22), 0.6, 6)
    pairs = find_friedel_pairs_peaks(r, o, y, z)
    x, s, n, boots = fit_wedge_omega_scale(pairs, lsd_um=LSD, ome_ref=OME_REF, n_boot=20)
    se = boots.std(axis=0)
    assert abs(x[0] - 0.0211) < max(4 * se[0], 0.0015), (x, se)
    assert abs(x[1] - 2.0e-4) < max(4 * se[1], 3e-5), (x, se)
