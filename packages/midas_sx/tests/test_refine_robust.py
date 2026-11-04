"""Outlier-aware refinement and the gate-free parameters (PREREGISTER A27)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_sx.refine import RefineModel, perturbed_crystal, refine   # noqa: E402
from test_refine import crystal_of, unique_hkl                      # noqa: E402


@pytest.fixture(scope="module")
def data():
    st = sim.structure_s2_bilayer()
    hkl, f2 = unique_hkl(st, 0.8)
    rng = np.random.default_rng(0)
    sig = 0.01 * f2.max() * np.ones(len(f2))
    obs = 3.0 * f2 + rng.normal(size=len(f2)) * sig
    return st, hkl, f2, obs, sig


def test_a_row_with_a_tiny_sigma_and_a_wrong_value_ruins_the_plain_fit_and_not_the_outlier_aware_one(data):
    st, hkl, f2, obs, sig = data
    obs2, sig2 = obs.copy(), sig.copy()
    j = int(np.argmax(f2))
    obs2[j] = 0.1 * obs2[j]; sig2[j] = 0.05 * sig2[j]                         # a box that missed its spot: a small Poisson sigma (seed 14 of A26: 7 against 150), 10 % of the intensity
    start = perturbed_crystal(crystal_of(st), np.random.default_rng(1), shift_A=0.05)
    plain = refine(RefineModel(start), hkl, obs2, sig2)
    robust = refine(RefineModel(start), hkl, obs2, sig2, outlier_sigma=5.0)
    clean = refine(RefineModel(start), hkl, obs, sig)
    assert plain.goof > 5 and robust.goof < 1.5
    assert list(robust.rejected) == [j] and robust.n_obs == len(hkl) - 1
    assert abs(robust.x[0] - clean.x[0]) < 0.02 * clean.x[0]                 # the scale is back where the clean data put it
    assert abs(plain.x[0] - clean.x[0]) > 0.02 * clean.x[0]


def test_outlier_aware_fit_of_clean_data_rejects_nothing_and_is_identical(data):
    st, hkl, f2, obs, sig = data
    start = perturbed_crystal(crystal_of(st), np.random.default_rng(1), shift_A=0.05)
    a = refine(RefineModel(start), hkl, obs, sig)
    b = refine(RefineModel(start), hkl, obs, sig, outlier_sigma=6.0)
    assert len(b.rejected) == 0 and np.allclose(a.x, b.x, rtol=1e-6, atol=1e-9) and abs(a.chi2 - b.chi2) < 1e-6 * max(a.chi2, 1.0)


def test_gate_free_parameters_report_a_bound_hit_without_failing_the_verdict(data):
    st, hkl, f2, obs, sig = data
    truth = crystal_of(st)
    # a hopeless starting U for an oxygen forces nothing here, so build the situation directly: lower the U bound above the truth
    m = RefineModel(perturbed_crystal(truth, np.random.default_rng(1), shift_A=0.05), u_bounds=(0.0115, 0.2))      # every U (0.005-0.010) sits below the bound
    r_fatal = refine(m, hkl, obs, sig)
    free = [n for n in m.names if n.endswith(".U")]
    r_free = refine(m, hkl, obs, sig, gate_free=free)
    assert not r_fatal.converged and "on bound" in r_fatal.reason
    assert set(r_free.at_bound_ignored) >= {n for n in free} and "on bound" not in r_free.reason
