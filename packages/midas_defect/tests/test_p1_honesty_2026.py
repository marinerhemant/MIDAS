"""Tests for P1 reporting fixes + the honesty layer (AUDIT_2026-06-23.md)."""

import math

import numpy as np
import pytest

from midas_defect.line_profile.modified_wh import modified_wh_per_grain, _cubic_H_squared
from midas_defect.gnd.nye_tensor import per_grain_nye_tensor
from midas_defect.variants.matched_pairs import find_sigma3_partners
from midas_defect.types import CrystalPhase
from midas_defect.variants.common_reference import build_sigma3_pair
from midas_defect.honesty import (systematic_uq, Probe, IndependenceError,
                                  assert_independent)


# --------------------------------------------------------------------------- #
# P1-1: modified_wh — q_U suppressed when ill-conditioned; rho F-band present
# --------------------------------------------------------------------------- #
def _hkls():
    # 11 FCC-allowed reflections — enough that the q_U scan cannot overfit pure noise
    return np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1], [2, 2, 2],
                     [4, 0, 0], [3, 3, 1], [4, 2, 0], [4, 2, 2], [5, 1, 1],
                     [4, 4, 0]], int)


def _entry(hkls, beta):
    return {"refl_indices": np.arange(len(hkls)),
            "G_magnitude": np.array([math.sqrt((h * h).sum()) for h in hkls], float),
            "fwhm": np.asarray(beta, float)}


def test_modified_wh_qU_suppressed_when_illconditioned():
    hkls = _hkls()
    G = np.array([math.sqrt(float((h * h).sum())) for h in hkls])
    H2 = np.array([_cubic_H_squared(*h) for h in hkls])
    qU_true = 2.2
    # clean grain: beta exactly linear in |G|sqrt(C) -> R^2=1 -> q_U kept
    beta_clean = 0.001 + 0.01 * G * np.sqrt(np.clip(1 - qU_true * H2, 1e-6, None))
    # noisy grain: random beta -> low R^2 -> q_U suppressed
    rng = np.random.default_rng(0)
    beta_noise = rng.uniform(0.005, 0.02, len(hkls))
    out = modified_wh_per_grain([_entry(hkls, beta_clean), _entry(hkls, beta_noise)],
                                hkls, burgers_length=2.55e-10, q_U_r2_min=0.5)
    assert np.isfinite(out["q_U_per_grain"][0])          # clean kept
    assert abs(out["q_U_per_grain"][0] - qU_true) < 0.1  # and ~correct
    assert np.isnan(out["q_U_per_grain"][1])             # noisy suppressed
    assert out["n_q_U_suppressed"] >= 1
    # rho F-band brackets and is ordered low<high
    band = out["rho_band_per_grain"][0]
    assert band[0] < band[1]
    assert band[0] <= out["rho_per_grain"][0] <= band[1] * 1.001


# --------------------------------------------------------------------------- #
# P1-2: layer-z guards — results must not depend on the (unreliable) refined Z
# --------------------------------------------------------------------------- #
def _two_variant_grains(rng, n=40):
    import midas_stress.orientation as ori
    P = np.eye(3)
    _, T = build_sigma3_pair(P, CrystalPhase.FCC)
    OM = []
    lab = []
    for _ in range(n):
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        OM.append(ori.axis_angle_to_orient_mat(a, rng.uniform(0, 5)) @ P); lab.append(0)
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        OM.append(ori.axis_angle_to_orient_mat(a, rng.uniform(0, 5)) @ T); lab.append(1)
    return np.array(OM), np.array(lab)


def test_nye_tensor_ignores_unreliable_z():
    rng = np.random.default_rng(1)
    OM, lab = _two_variant_grains(rng, 40)
    n = len(OM)
    pos = np.column_stack([rng.uniform(0, 200, n), rng.uniform(0, 200, n),
                           np.zeros(n)])          # layer-z = const
    a = per_grain_nye_tensor(OM, pos.copy(), 2.55e-10, lab, z_reliable=False)
    pos2 = pos.copy(); pos2[:, 2] = rng.uniform(-210, 210, n)   # garbage refined Z
    b = per_grain_nye_tensor(OM, pos2, 2.55e-10, lab, z_reliable=False)
    ra, rb = a["rho_GND_per_grain"], b["rho_GND_per_grain"]
    ok = np.isfinite(ra) & np.isfinite(rb)
    np.testing.assert_allclose(ra[ok], rb[ok], rtol=1e-9)      # z noise must not matter


def test_matched_pairs_ignores_unreliable_z():
    rng = np.random.default_rng(2)
    OM, lab = _two_variant_grains(rng, 30)
    n = len(OM)
    pos = np.column_stack([rng.uniform(0, 300, n), rng.uniform(0, 300, n), np.zeros(n)])
    p1 = find_sigma3_partners(OM, pos.copy(), lab, z_reliable=False)
    pos2 = pos.copy(); pos2[:, 2] = rng.uniform(-210, 210, n)
    p2 = find_sigma3_partners(OM, pos2, lab, z_reliable=False)
    assert np.array_equal(p1["pairs"], p2["pairs"])           # pairing independent of z


# --------------------------------------------------------------------------- #
# Honesty layer
# --------------------------------------------------------------------------- #
def test_systematic_uq_flags_sign_flip():
    # a quantity that flips sign across defensible relabelings is an artifact
    flip = systematic_uq([lambda: +0.4, lambda: -0.3, lambda: +0.1])
    assert flip["sign_stable"] is False
    assert flip["relative_spread"] > 1.0
    stable = systematic_uq([lambda: 1.0, lambda: 1.1, lambda: 0.95])
    assert stable["sign_stable"] is True
    assert stable["relative_spread"] < 0.3


def test_assert_independent_rejects_shared_axis_attribution():
    # L_9R (q-space) and dPDF (r-space) keyed to the same axis + attribution
    L9R = Probe("L_9R", axis_id="OM@111", attribution_id="per_grain_nn", space="q")
    dpdf = Probe("dPDF", axis_id="OM@111", attribution_id="per_grain_nn", space="r")
    with pytest.raises(IndependenceError):
        assert_independent([L9R, dpdf])
    # genuinely different inputs are allowed
    indep = Probe("strain", axis_id="bragg_fit", attribution_id="indexed_spot", space="q")
    assert_independent([L9R, indep])
