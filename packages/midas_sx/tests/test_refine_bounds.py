"""Bound hits: a rail (the data demand a value outside the physical range) fails the convergence verdict, an UNDETERMINED parameter on its bound (the data do not
care) does not: decided by a likelihood-ratio test, not by a list of names to ignore (gate_free is only a manual override)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_hkls import Atom, Crystal                                # noqa: E402
from midas_sx.refine import EIGHT_PI2, EXT_E_MIN, RefineModel, perturbed_crystal, refine   # noqa: E402
from test_refine import crystal_of, unique_hkl                      # noqa: E402


@pytest.fixture(scope="module")
def s2():
    st = sim.structure_s2_bilayer()
    hkl, f2 = unique_hkl(st, 0.8)
    return st, hkl, f2, crystal_of(st)


def _with_u(crystal, label, u):
    atoms = [replace(a, B_iso=EIGHT_PI2 * u) if a.label == label else a for a in crystal.atoms]
    return Crystal(crystal.lattice, crystal.space_group, atoms, crystal.name)


def _f2_of(crystal, hkl):
    return np.asarray(refine.__globals__["RefineModel"](crystal).make_f2(hkl)(__import__("torch").tensor(RefineModel(crystal).x0())).detach().numpy())


def test_a_parameter_the_data_do_not_care_about_may_sit_on_its_bound_without_failing(s2):
    st, hkl, f2, truth = s2
    t0 = _with_u(truth, "O3", 1e-4)                                  # truth exactly at the lower bound: about half the noisy fits want a negative U
    model0 = RefineModel(t0)
    import torch
    clean = model0.make_f2(hkl)(torch.tensor(model0.x0())).detach().numpy()
    seen_harmless = 0
    for seed in range(14):
        rng = np.random.default_rng(seed)
        sig = 0.03 * clean + 1e-3 * clean.max()
        obs = clean + rng.normal(size=clean.shape) * sig
        res = refine(RefineModel(perturbed_crystal(t0, np.random.default_rng(100 + seed), shift_A=0.05)), hkl, obs, sig)
        if "O3.U" in res.at_bound_harmless:
            seen_harmless += 1
            assert res.converged, res.reason
            assert res.at_bound == [] and res.bound_delta_chi2 <= 2.71 + 1e-9
        elif "O3.U" in res.at_bound:                                   # the bound test is one-sided 5 %: a few draws are real rails, and must stay fatal
            assert not res.converged and res.bound_delta_chi2 > 2.71
    assert seen_harmless >= 1, "the construction should put O3.U on the bound in some noise draws"


def test_a_genuine_rail_still_fails_the_verdict(s2):
    st, hkl, f2, truth = s2
    t0 = _with_u(truth, "O3", -0.03)                                 # the data demand a physically impossible (negative) U
    model0 = RefineModel(t0)
    import torch
    clean = model0.make_f2(hkl)(torch.tensor(model0.x0(), dtype=torch.float64)).detach().numpy()
    sig = 0.01 * clean + 1e-3 * clean.max()
    res = refine(RefineModel(perturbed_crystal(truth, np.random.default_rng(1), shift_A=0.02)), hkl, clean, sig)
    assert "O3.U" in res.at_bound and not res.converged and "on bound: O3.U" in res.reason
    assert res.bound_delta_chi2 > 2.71 and res.at_bound_harmless == []


def test_bound_test_false_restores_the_old_rule_and_gate_free_is_still_an_override(s2):
    st, hkl, f2, truth = s2
    t0 = _with_u(truth, "O3", 1e-4)
    import torch
    m0 = RefineModel(t0)
    clean = m0.make_f2(hkl)(torch.tensor(m0.x0())).detach().numpy()
    for seed in range(14):
        rng = np.random.default_rng(seed)
        sig = 0.03 * clean + 1e-3 * clean.max()
        obs = clean + rng.normal(size=clean.shape) * sig
        start = perturbed_crystal(t0, np.random.default_rng(100 + seed), shift_A=0.05)
        a = refine(RefineModel(start), hkl, obs, sig)
        if "O3.U" in a.at_bound_harmless:
            b = refine(RefineModel(start), hkl, obs, sig, bound_test=False)
            assert not b.converged and "O3.U" in b.at_bound                         # the old rule: every bound hit fatal
            c = refine(RefineModel(start), hkl, obs, sig, bound_test=False, gate_free=("O3.U",))
            assert c.converged and c.at_bound_ignored == ["O3.U"]
            return
    pytest.fail("no noise draw put O3.U on its bound")


def test_extinction_started_at_zero_thickness_has_a_finite_jacobian(s2):
    import torch
    st, hkl, f2, truth = s2
    m = RefineModel(truth, refine_extinction=True, wavelength_A=0.4133, ext_t_start_um=0.0)
    x = m.x0()
    i = m.names.index("ext.t2")
    assert x[i] >= EXT_E_MIN
    f = m.make_f2(hkl)
    for e in (0.0, EXT_E_MIN, 1e-6):
        xe = x.copy(); xe[i] = e
        J = torch.autograd.functional.jacobian(f, torch.tensor(xe))
        assert torch.isfinite(J).all(), e


def test_a_rail_is_not_hidden_by_another_parameter_on_its_bound(s2):
    """Two parameters on bounds: O2.U truly negative (a rail the data demand), O3.U truly at the lower bound (undetermined). The test relaxes ONE bound at a
    time, so O2.U keeps its own chi2 drop; a joint test with threshold 2.71 x 2 let it hide (physics lens of /verify claim 97f96459b1e5)."""
    import torch
    st, hkl, f2, truth = s2
    t0 = _with_u(_with_u(truth, "O2", -0.002), "O3", 1e-4)
    m0 = RefineModel(t0)
    clean = m0.make_f2(hkl)(torch.tensor(m0.x0())).detach().numpy()
    sig = 0.01 * clean + 1e-3 * clean.max()
    start = perturbed_crystal(truth, np.random.default_rng(3), shift_A=0.02)
    both = refine(RefineModel(start), hkl, clean, sig)
    assert "O2.U" in both.at_bound and not both.converged, (both.at_bound, both.at_bound_harmless, both.bound_delta_chi2_by_param)
    d = both.bound_delta_chi2_by_param
    assert 2.71 < d["O2.U"] and sum(d.values()) <= 2 * 2.71, d              # the regime in which a joint 2.71 x k threshold hid the rail (O2.U alone: 3.17)
    if "O3.U" in both.bound_delta_chi2_by_param:
        assert both.bound_delta_chi2_by_param["O3.U"] <= 2.71 and "O3.U" in both.at_bound_harmless
    only = refine(RefineModel(start), hkl, clean, sig, bound_test=True, gate_free=("O2.U",))
    assert only.at_bound_ignored == ["O2.U"]


def test_a_rail_on_the_UPPER_bound_is_found_and_not_hidden(s2):
    """The first version of the per-parameter test only had lower-bound cases in its tests (a mutant that relaxed only lower bounds survived /verify); here O3.U truly sits
    ABOVE its upper bound (0.05), O2.U is harmlessly on its lower one."""
    import torch
    st, hkl, f2, truth = s2
    t0 = _with_u(_with_u(truth, "O3", 0.08), "O2", 1e-4)
    m0 = RefineModel(t0, u_bounds=(1e-4, 0.05))
    clean = m0.make_f2(hkl)(torch.tensor(m0.x0())).detach().numpy()
    sig = 0.01 * clean + 1e-3 * clean.max()
    r = refine(RefineModel(perturbed_crystal(truth, np.random.default_rng(5), shift_A=0.02), u_bounds=(1e-4, 0.05)), hkl, clean, sig)
    d = r.bound_delta_chi2_by_param
    assert "O3.U" in r.at_bound and not r.converged and d["O3.U"] > 2.71, (r.at_bound, r.reason, d)
    if "O2.U" in d:
        assert d["O2.U"] <= 2.71 and "O2.U" in r.at_bound_harmless
