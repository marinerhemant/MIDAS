"""Refinable primary-extinction thickness (PREREGISTER A21): machinery on exact and noisy F², no frames."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402
import extinction_truth as et                                       # noqa: E402

from midas_hkls import SpaceGroup                                    # noqa: E402
from midas_sx.refine import RefineModel, perturbed_crystal, refine  # noqa: E402
from test_refine import crystal_of, unique_hkl                      # noqa: E402

LAM = 0.4133


@pytest.fixture(scope="module")
def s2():
    st = sim.structure_s2_bilayer()
    hkl, f2 = unique_hkl(st, 0.8)
    return st, hkl, f2, crystal_of(st)


def _extinct(st, hkl, f2, t, sigma_ln=0.0):
    return f2 * et.make_factor(st, LAM, t, sigma_ln)(hkl, f2)


def test_without_the_flag_nothing_changes(s2):
    st, hkl, f2, truth = s2
    m = RefineModel(truth)
    assert "ext.t2" not in m.names and "ext" not in m.kinds
    assert RefineModel(truth, refine_extinction=True, wavelength_A=LAM).names[-1] == "ext.t2"
    with pytest.raises(ValueError):
        RefineModel(truth, refine_extinction=True)


def test_exact_extinct_f2_gives_t_b_and_every_other_parameter_back(s2):
    st, hkl, f2, truth = s2
    obs = _extinct(st, hkl, f2, 8.0)
    sigma = 0.01 * obs.max() * np.ones(len(obs))
    m = RefineModel(perturbed_crystal(truth, np.random.default_rng(3), shift_A=0.05), refine_extinction=True,
                    wavelength_A=LAM, ext_t_start_um=14.0)
    res = refine(m, hkl, obs, sigma)
    assert res.converged, res.reason
    # truth is gemmi's F2 times a table-interpolated factor: cross-code floor is ~1e-4 in t_b
    assert abs(res.t_b_um - 8.0) < 0.02
    cell = np.array(st.cell[:3])
    for atom, row in zip(truth.atoms, res.absolute.itertuples()):
        d_A = np.abs((np.array([row.x, row.y, row.z]) - np.array(atom.fract)) * cell)
        assert d_A.max() < 2e-3, (atom.label, d_A)                       # Angstrom
        assert abs(row.U_iso - atom.B_iso / (8 * np.pi ** 2)) < 2e-4, (atom.label, row.U_iso)
    assert abs(res.x[0] - 1.0) < 5e-3


def test_kinematic_fit_of_extinct_data_is_biased_and_the_extinction_fit_is_not(s2):
    st, hkl, f2, truth = s2
    obs = _extinct(st, hkl, f2, 8.0)
    sigma = 0.01 * obs.max() * np.ones(len(obs))
    start = perturbed_crystal(truth, np.random.default_rng(3), shift_A=0.05)
    kin = refine(RefineModel(start), hkl, obs, sigma)
    ext = refine(RefineModel(start, refine_extinction=True, wavelength_A=LAM, ext_t_start_um=14.0), hkl, obs, sigma)
    assert kin.goof > 5 * ext.goof and ext.goof < 0.5


def test_kinematic_data_returns_zero_thickness_without_a_rail_flag_or_a_rank_failure(s2):
    st, hkl, f2, truth = s2
    sigma = 0.01 * f2.max() * np.ones(len(f2))
    m = RefineModel(perturbed_crystal(truth, np.random.default_rng(4), shift_A=0.05), refine_extinction=True,
                    wavelength_A=LAM, ext_t_start_um=6.0)
    res = refine(m, hkl, f2, sigma)
    assert res.converged, res.reason
    assert res.t_b_um < 0.2 and "ext.t2" not in res.at_bound
    assert np.isfinite(res.sigma[res.names.index("ext.t2")])


def test_extinction_is_strongest_where_the_structure_factor_is(s2):
    st, hkl, f2, truth = s2
    y = et.make_factor(st, LAM, 8.0)(hkl, f2)
    assert y[np.argmax(f2)] < y[np.argmin(f2)]
    assert np.all(y <= 1.0 + 1e-9) and y.min() > 0.2


def test_correlation_matrix_is_symmetric_with_unit_diagonal(s2):
    st, hkl, f2, truth = s2
    obs = _extinct(st, hkl, f2, 6.0)
    m = RefineModel(perturbed_crystal(truth, np.random.default_rng(5), shift_A=0.05), refine_extinction=True,
                    wavelength_A=LAM, ext_t_start_um=9.0)
    res = refine(m, hkl, obs, 0.01 * obs.max() * np.ones(len(obs)))
    C = res.correlation()
    assert np.allclose(np.diag(C), 1.0) and np.allclose(C, C.T, equal_nan=True)
    assert np.nanmax(np.abs(C - np.eye(len(C)))) <= 1.0 + 1e-9


def test_extinction_acts_on_each_twin_domain_before_they_add():
    """A twin domain is its own block: F2_obs = (1 - m) F2(h) y(h) + m F2(T h) y(T h), each y from that domain's OWN structure factor. The refiner used to
    multiply the twin SUM by one factor computed from the sum (wrong whenever F2(h) and F2(T h) differ); exact data must give m, t_b and the structure back."""
    st = sim.structure_s2o()
    sg = SpaceGroup.from_hm(st.spacegroup)
    allh = sim.enumerate_hkl(st, 0.8)
    reps = np.array(sorted({sg.asu_representative(int(h), int(k), int(l)) for h, k, l in allh}))
    f2 = sim.f2_truth(st, reps)
    keep = f2 > 1e-6 * f2.max()
    hkl, f2 = reps[keep], f2[keep]
    part = np.stack([hkl[:, 1], -hkl[:, 0], hkl[:, 2]], axis=1)
    f2p = sim.f2_truth(st, part)
    fac = et.make_factor(st, LAM, 8.0)
    m_true = 0.3
    obs = (1 - m_true) * f2 * fac(hkl, f2) + m_true * f2p * fac(part, f2p)
    assert np.abs((f2 * fac(hkl, f2)) / f2 - 1).max() > 0.05 and np.abs(f2 - f2p).max() / f2.max() > 1e-3        # extinction and the twin split both matter here
    # the model at the TRUE parameters must reproduce exact data (this is what the ordering decides; a 1 % sigma hides it)
    import torch
    mt = RefineModel(crystal_of(st), refine_twin=True, twin_start=m_true, refine_extinction=True, wavelength_A=LAM)
    x = mt.x0().copy(); x[mt.names.index("ext.t2")] = 8.0 ** 2
    k_i = mt.names.index("scale"); x[k_i] = 1.0
    pred = mt.make_f2(hkl)(torch.as_tensor(x, dtype=torch.float64)).detach().numpy()
    # 2e-5: cross-code numerics floor (4.6e-6 even with no twin: kinematic F2 vs gemmi, extinction factor vs the exact Bessel integral); the old ordering is 2.83e-3 off
    assert np.abs(pred / obs - 1).max() < 2e-5, np.abs(pred / obs - 1).max()
    start = perturbed_crystal(crystal_of(st), np.random.default_rng(3), shift_A=0.03)
    res = refine(RefineModel(start, refine_twin=True, twin_start=0.15, refine_extinction=True, wavelength_A=LAM), hkl, obs, 0.01 * obs.max() * np.ones(len(obs)))
    assert res.converged, res.reason
    assert abs(res.twin_fraction - m_true) < 3e-3, res.twin_fraction
    assert abs(res.t_b_um - 8.0) < 0.3, res.t_b_um
    assert res.goof < 0.5


def test_the_partner_component_uses_its_own_bragg_angle_when_a_differs_from_b():
    """S2P (b/a = 1.03): T h = (k, -h, l) has a different d than h; the model must reproduce data built with each component's own theta (it used h's theta for both: 1.5e-4 off)."""
    import torch
    st = sim.structure_s2p(1.03)
    sg = SpaceGroup.from_hm(st.spacegroup)
    allh = sim.enumerate_hkl(st, 0.8)
    reps = np.array(sorted({sg.asu_representative(int(h), int(k), int(l)) for h, k, l in allh}))
    f2 = sim.f2_truth(st, reps)
    keep = f2 > 1e-6 * f2.max()
    hkl, f2 = reps[keep], f2[keep]
    part = np.stack([hkl[:, 1], -hkl[:, 0], hkl[:, 2]], axis=1)
    f2p = sim.f2_truth(st, part)
    fac = et.make_factor(st, LAM, 8.0)                               # the truth factor takes each reflection's own d (gemmi calculate_d)
    m = 0.3
    obs = (1 - m) * f2 * fac(hkl, f2) + m * f2p * fac(part, f2p)
    mt = RefineModel(crystal_of(st), refine_twin=True, twin_start=m, refine_extinction=True, wavelength_A=LAM)
    x = mt.x0().copy(); x[mt.names.index("ext.t2")] = 64.0; x[mt.names.index("scale")] = 1.0
    pred = mt.make_f2(hkl, partner=part)(torch.as_tensor(x, dtype=torch.float64)).detach().numpy()
    assert np.abs(pred / obs - 1).max() < 2e-5, np.abs(pred / obs - 1).max()
