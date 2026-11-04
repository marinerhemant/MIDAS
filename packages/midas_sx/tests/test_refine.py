"""The refiner against exact and synthetic-noise F², and its gates. The structure truth is gemmi's (validation/simulate.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup           # noqa: E402
from midas_hkls.tables import STBF                                  # noqa: E402
from midas_sx.refine import (EIGHT_PI2, RefineModel, perturbed_crystal, refine,   # noqa: E402
                             site_free_directions)


def crystal_of(st):
    atoms = [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites]
    return Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), atoms, st.name)


def unique_hkl(st, d_min):
    sg = SpaceGroup.from_hm(st.spacegroup)
    allh = sim.enumerate_hkl(st, d_min)
    reps = sorted({sg.asu_representative(int(h), int(k), int(l)) for h, k, l in allh})
    hkl = np.array(reps)
    f2 = sim.f2_truth(st, hkl)
    keep = f2 > 1e-6 * f2.max()
    return hkl[keep], f2[keep]


# ------------------------------------------------------------ site symmetry
def test_free_directions_follow_the_wyckoff_site():
    sg = SpaceGroup.from_hm("I 4/m m m")
    n = lambda x: len(site_free_directions(sg, x))
    assert n((0, 0, 0.5)) == 0 and n((0, 0, 0.0)) == 0          # 2b, 2a: fixed
    assert n((0, 0, 0.319)) == 1 and n((0, 0.5, 0.095)) == 1    # 4e, 8g: z only
    assert n((0.1, 0.2, 0.3)) == 3                              # general position
    fm3m = SpaceGroup.from_hm("F m -3 m")
    assert len(site_free_directions(fm3m, (0, 0, 0))) == 0
    d = site_free_directions(fm3m, (0.2, 0.2, 0.2))             # 32f: along (1,1,1)
    assert d.shape == (1, 3) and np.allclose(np.abs(d[0]), 1 / np.sqrt(3))


# ---------------------------------------------- the model is the simulator's
@pytest.mark.parametrize("make", [sim.structure_cu, sim.structure_s2_bilayer])
def test_midas_hkls_f2_equals_gemmi_f2(make):
    """If these disagreed, the refiner would fit a different model from the one that made the data."""
    import torch
    from midas_hkls import structure_factors
    st = make()
    hkl, f2_gemmi = unique_hkl(st, 0.75)
    ct = crystal_of(st).to_torch(dtype=torch.float64)
    F = structure_factors(ct, torch.as_tensor(hkl))
    f2 = (F.real ** 2 + F.imag ** 2).numpy()
    assert len(hkl) >= (7 if st.name.startswith('S1') else 100)
    # absolute agreement against the strongest reflection, and relative agreement where F2 is not a near-extinction
    # (a relative difference on F2/max ~ 1e-6 is a ~1e-11 absolute one: cancellation, not a model difference)
    assert np.max(np.abs(f2 - f2_gemmi)) / f2_gemmi.max() < 5e-7
    strong = f2_gemmi > 1e-3 * f2_gemmi.max()
    assert np.max(np.abs(f2[strong] / f2_gemmi[strong] - 1.0)) < 5e-6


# -------------------------------------------------------- Step-1 exact gate
def _model_f2(crystal, hkl):
    import torch
    from midas_hkls import structure_factors
    F = structure_factors(crystal.to_torch(dtype=torch.float64), torch.as_tensor(hkl))
    return (F.real ** 2 + F.imag ** 2).numpy()


def _compare_to_truth(res, truth, k_true, tol_x, tol_u, tol_k):
    ab = res.absolute
    at = RefineModel(truth).absolute(RefineModel(truth).x0())
    assert np.max(np.abs(ab[["x", "y", "z"]].to_numpy() - at[["x", "y", "z"]].to_numpy())) < tol_x
    assert np.max(np.abs(ab.U_iso.to_numpy() / at.U_iso.to_numpy() - 1.0)) < tol_u
    assert abs(res.x[0] / k_true - 1.0) < tol_k


@pytest.mark.parametrize("make", [sim.structure_cu, sim.structure_s2_bilayer])
def test_exact_f2_of_the_refiners_own_model_is_recovered_to_1e8(make):
    st = make()
    truth = crystal_of(st)
    hkl, _ = unique_hkl(st, 0.75)
    k_true = 3.7
    obs = k_true * _model_f2(truth, hkl)
    sigma = 0.01 * np.sqrt(obs * obs.max()) / 10 + 1e-3 * obs.max()          # arbitrary positive weights
    start = perturbed_crystal(truth, np.random.default_rng(0), shift_A=0.1)
    res = refine(RefineModel(start), hkl, obs, sigma)
    assert res.converged, res.reason
    _compare_to_truth(res, truth, k_true, 1e-8, 1e-8, 1e-8)
    assert res.chi2 < 1e-12 and np.all(np.isfinite(res.sigma))


@pytest.mark.parametrize("make", [sim.structure_cu, sim.structure_s2_bilayer])
def test_gemmi_f2_is_recovered_to_the_cross_code_floor(make):
    """Same data from the simulator's gemmi: the floor is the midas_hkls-vs-gemmi table difference (PREREGISTER A7)."""
    st = make()
    truth = crystal_of(st)
    hkl, f2 = unique_hkl(st, 0.75)
    k_true = 3.7
    obs = k_true * f2
    sigma = 0.01 * np.sqrt(obs * obs.max()) / 10 + 1e-3 * obs.max()
    res = refine(RefineModel(perturbed_crystal(truth, np.random.default_rng(1), shift_A=0.1)), hkl, obs, sigma)
    assert res.converged, res.reason
    _compare_to_truth(res, truth, k_true, 1e-6, 1e-5, 1e-5)


# ----------------------------------------- statistics of the refiner itself
def test_pulls_are_unbiased_with_unit_width_for_gaussian_noise_on_the_model():
    """No reduction involved: Gaussian noise of known sigma on exact F². Formal sigma must be the right size."""
    st = sim.structure_s2_bilayer()
    truth = crystal_of(st)
    hkl, f2 = unique_hkl(st, 0.9)
    k_true = 2.0e-3
    exact = k_true * f2
    sigma = 0.01 * exact + 2e-4 * exact.max()               # 1 % relative + a small floor: every reflection carries some information
    rng = np.random.default_rng(11)
    pulls = []
    m0 = RefineModel(truth)
    x_truth = m0.x0(); x_truth[0] = k_true
    for i in range(80):
        obs = exact + rng.normal(0, sigma)
        start = perturbed_crystal(truth, rng, shift_A=0.1)
        res = refine(RefineModel(start), hkl, obs, sigma)
        assert res.converged, (i, res.reason)
        # parameters relative to the truth: coordinates compared absolutely, via the absolute table
        t = res.absolute
        row = []
        for j, (nm, kind) in enumerate(zip(res.names, res.kinds)):
            if kind == "scale":
                row.append((res.x[j] - k_true) / res.sigma[j])
            elif kind == "U":
                row.append((res.x[j] - m0.x0()[j]) / res.sigma[j])
            else:
                ia = res.names.index(nm)
                label = nm.split(".")[0]
                idx = [a.label for a in truth.atoms].index(label)
                dz = (t.iloc[idx][["x", "y", "z"]].to_numpy() - np.array(truth.atoms[idx].fract))
                shift_A = float(dz @ (np.linalg.inv(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))) @ m0.basis_frac[j]) \
                    / float(m0.basis_frac[j] @ m0.basis_frac[j])
                row.append(shift_A / res.sigma[j])
        pulls.append(row)
    pulls = np.array(pulls)
    names = res.names
    for j, nm in enumerate(names):
        assert abs(pulls[:, j].mean()) < 0.5, (nm, pulls[:, j].mean())          # SE of the mean ~ 0.11
        assert 0.7 < pulls[:, j].std(ddof=1) < 1.4, (nm, pulls[:, j].std(ddof=1))


# --------------------------------------------------------------------- gates
def test_a_parameter_pushed_onto_its_bound_is_not_reported_as_converged():
    st = sim.structure_cu()
    truth = crystal_of(st)
    hkl, f2 = unique_hkl(st, 0.9)
    # data that want a NEGATIVE U (intensity rising with angle): the fit must rail, and say so
    s2 = np.array([(1 / (2 * truth.lattice.d_spacing(*map(int, h)))) ** 2 for h in hkl])
    obs = f2 * np.exp(+2 * EIGHT_PI2 * 0.02 * s2)
    res = refine(RefineModel(truth), hkl, obs, 0.01 * obs.max() * np.ones(len(obs)))
    assert not res.converged and any("U" in n for n in res.at_bound) and "bound" in res.reason


def test_too_few_reflections_is_flagged_rank_deficient_with_infinite_sigma():
    st = sim.structure_s2_bilayer()
    truth = crystal_of(st)
    hkl, f2 = unique_hkl(st, 1.2)
    hkl, f2 = hkl[:6], f2[:6]
    res = refine(RefineModel(truth), hkl, f2, 0.01 * f2.max() * np.ones(len(f2)))
    assert not res.converged and not res.full_rank and "rank" in res.reason
    assert np.any(np.isinf(res.sigma)) and len(res.null_params) > 0


def test_bad_inputs_are_refused():
    st = sim.structure_cu()
    truth = crystal_of(st)
    hkl, f2 = unique_hkl(st, 0.9)
    with pytest.raises(ValueError):
        refine(RefineModel(truth), hkl, f2, np.zeros(len(f2)))
    with pytest.raises(ValueError):
        refine(RefineModel(truth), hkl, f2[:-1], np.ones(len(f2)))


def test_a_start_on_the_chi2_ridge_can_land_in_a_false_minimum_and_the_result_says_so():
    """A8: 0.02 fractional on S2's 20.5 A axis is 0.41 A, on the chi2 ridge of the heavy z. Some sign patterns end in a
    false minimum. The refiner must not present that as a good fit: it is flagged (bound hit and/or poor fit)."""
    st = sim.structure_s2_bilayer()
    truth = crystal_of(st)
    hkl, _ = unique_hkl(st, 0.75)
    obs = 3.7 * _model_f2(truth, hkl)
    sigma = 0.01 * np.sqrt(obs * obs.max()) / 10 + 1e-3 * obs.max()
    outcomes = []
    for seed in range(12):
        res = refine(RefineModel(perturbed_crystal(truth, np.random.default_rng(seed), shift_frac=0.02)), hkl, obs, sigma)
        recovered = res.chi2 < 1e-6
        outcomes.append((recovered, res.converged and not res.poor_fit))
    n_false = sum(1 for rec, good in outcomes if not rec)
    assert n_false >= 1, "expected at least one false minimum from the ridge start; if none, the ridge finding no longer holds"
    # whenever it did NOT recover the truth, the result must not claim to be a good, converged fit
    assert all(not good for rec, good in outcomes if not rec)
