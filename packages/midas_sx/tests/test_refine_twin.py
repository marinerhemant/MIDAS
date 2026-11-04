"""Exactly merohedral twin fraction (PREREGISTER A24): the SHELX-style sum of a reflection and its twin partner."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup           # noqa: E402
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine   # noqa: E402


def crystal_of(st):
    return Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup),
                   [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)


@pytest.fixture(scope="module")
def data():
    st = sim.structure_s2o()
    sg = SpaceGroup.from_hm(st.spacegroup)
    allh = sim.enumerate_hkl(st, 0.8)
    reps = np.array(sorted({sg.asu_representative(int(h), int(k), int(l)) for h, k, l in allh}))
    f2 = sim.f2_truth(st, reps)
    keep = f2 > 1e-6 * f2.max()
    return st, reps[keep], f2[keep]


def test_twin_partner_is_a_symmetry_equivalent_of_a_representative_of_the_partner(data):
    """T normalises mmm, so F2_tw is constant on mmm-equivalent reflections (the condition for merging in Immm before the twin refinement)."""
    st, hkl, f2 = data
    sg = SpaceGroup.from_hm(st.spacegroup)
    T = lambda h: np.stack([h[:, 1], -h[:, 0], h[:, 2]], axis=1)
    ftw = sim.twin_f2(st, 0.3)
    for h in hkl[::7][:12]:
        eq = np.array([sg.asu_representative(int(a), int(b), int(c)) for a, b, c in [(h[0], h[1], h[2]), (-h[0], h[1], h[2]), (h[0], -h[1], h[2]), (h[1] * 0 + h[0], h[1], -h[2])]])
        v = ftw(eq, sim.f2_truth(st, eq))
        assert np.allclose(v, v[0], rtol=1e-9)


def test_twin_fraction_and_structure_are_recovered_from_exact_twinned_f2(data):
    st, hkl, f2 = data
    f2tw = sim.twin_f2(st, 0.3)(hkl, f2)
    sigma = 0.01 * f2tw.max() * np.ones(len(f2tw))
    start = perturbed_crystal(crystal_of(st), np.random.default_rng(1), shift_A=0.05)
    res = refine(RefineModel(start, refine_twin=True, twin_start=0.15), hkl, f2tw, sigma)
    assert res.converged, res.reason
    assert abs(res.twin_fraction - 0.3) < 2e-3, res.twin_fraction
    cell = np.array(st.cell[:3])
    for atom, row in zip(crystal_of(st).atoms, res.absolute.itertuples()):
        assert (np.abs((np.array([row.x, row.y, row.z]) - np.array(atom.fract)) * cell)).max() < 3e-3
    assert res.goof < 0.5 and np.isfinite(res.sigma_twin_fraction)


def test_untwinned_refinement_of_twinned_data_misfits(data):
    st, hkl, f2 = data
    f2tw = sim.twin_f2(st, 0.3)(hkl, f2)
    sigma = 0.01 * f2tw.max() * np.ones(len(f2tw))
    start = perturbed_crystal(crystal_of(st), np.random.default_rng(1), shift_A=0.05)
    u = refine(RefineModel(start), hkl, f2tw, sigma)
    t = refine(RefineModel(start, refine_twin=True), hkl, f2tw, sigma)
    assert u.goof > 3 * t.goof


def test_untwinned_data_returns_zero_fraction_without_a_rail_flag(data):
    st, hkl, f2 = data
    sigma = 0.01 * f2.max() * np.ones(len(f2))
    start = perturbed_crystal(crystal_of(st), np.random.default_rng(2), shift_A=0.05)
    res = refine(RefineModel(start, refine_twin=True, twin_start=0.2), hkl, f2, sigma)
    assert res.converged, res.reason
    assert res.twin_fraction < 1e-3 and "twin.m" not in res.at_bound


def test_identity_twin_law_leaves_the_intensities_unchanged(data):
    st, hkl, f2 = data
    sigma = 0.01 * f2.max() * np.ones(len(f2))
    start = perturbed_crystal(crystal_of(st), np.random.default_rng(2), shift_A=0.05)
    a = RefineModel(start, refine_twin=True, twin_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    f = a.make_f2(hkl)
    import torch
    x1, x2 = a.x0().copy(), a.x0().copy()
    x2[a.names.index("twin.m")] = 0.4
    assert np.allclose(f(torch.tensor(x1)).numpy(), f(torch.tensor(x2)).numpy(), rtol=1e-12)
