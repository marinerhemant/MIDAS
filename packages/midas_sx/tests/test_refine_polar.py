"""Polar groups: the origin along the polar axis is not determined by |F|, so the refinement must not carry it as a parameter; fixed atoms (hydrogens) still scatter
but have no parameters, and anchor the origin. Structure: thiourea at 130 K (P 2_1 m a, CIF of Zenodo 18910433, coordinates typed in)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gemmi")
import torch                                                          # noqa: E402

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup             # noqa: E402
from midas_sx.refine import (EIGHT_PI2, RefineModel, origin_gauge_directions, perturbed_crystal, refine)   # noqa: E402

CELL = (7.4962, 8.5211, 5.4695, 90, 90, 90)
SITES = [("S1", "S", (0.2032, 0.0, 0.152), 0.0229), ("C1", "C", (0.133, 0.0, 0.4491), 0.0211), ("N1", "N", (0.1059, 0.1329, 0.5687), 0.0296),
         ("H1A", "H", (0.13, 0.218, 0.499), 0.036), ("H1B", "H", (0.066, 0.132, 0.72), 0.036),
         ("S2", "S", (0.1844, 0.5, 0.3763), 0.0216), ("C2", "C", (0.3054, 0.5, 0.1114), 0.0205), ("N2", "N", (0.3546, 0.3667, 0.0057), 0.0269),
         ("H2A", "H", (0.321, 0.282, 0.062), 0.032), ("H2B", "H", (0.416, 0.37, -0.131), 0.032)]


def thiourea():
    atoms = [Atom(el, fr, 1.0, EIGHT_PI2 * u, label=lab) for lab, el, fr, u in SITES]
    return Crystal(Lattice(*CELL), SpaceGroup.from_hm("P 21 m a"), atoms, "thiourea130K")


def hkl_set(crystal, d_min=0.85):
    sg = crystal.space_group
    lat = crystal.lattice
    out = set()
    for h in range(-9, 10):
        for k in range(-10, 11):
            for l in range(-6, 7):
                if (h, k, l) != (0, 0, 0) and lat.d_spacing(h, k, l) >= d_min:
                    out.add(tuple(sg.asu_representative(h, k, l)))
    return np.array(sorted(out))


def test_gauge_directions():
    assert origin_gauge_directions(thiourea().space_group).shape == (1, 3)                     # P 2_1 m a: a polar axis
    assert abs(abs(origin_gauge_directions(thiourea().space_group)[0] @ [1, 0, 0]) - 1) < 1e-9
    assert origin_gauge_directions(SpaceGroup.from_hm("P 21 21 21")).shape == (0, 3)
    assert origin_gauge_directions(SpaceGroup.from_hm("P -1")).shape == (0, 3)
    assert origin_gauge_directions(SpaceGroup.from_hm("P 1")).shape == (3, 3)
    assert origin_gauge_directions(SpaceGroup.from_hm("P 1 21 1")).shape == (1, 3)


def test_the_polar_origin_parameter_is_removed_and_a_nonpolar_group_loses_nothing():
    full = RefineModel(thiourea(), fix_origin=False)
    fixed = RefineModel(thiourea())
    assert fixed.n_par == full.n_par - 1 and len(fixed.origin_fixed) == 1 and fixed.origin_fixed[0].endswith(".x")
    names = list(fixed.names)
    assert fixed.origin_fixed[0] not in names
    nonpolar = Crystal(Lattice(*CELL), SpaceGroup.from_hm("P n m a"), [Atom("S", (0.2, 0.25, 0.15), 1.0, 1.0, label="S1")], "x")
    assert RefineModel(nonpolar).origin_fixed == []


def test_refinement_converges_full_rank_with_the_origin_fixed_and_is_rank_deficient_without():
    t = thiourea()
    hkl = hkl_set(t)
    m0 = RefineModel(t, fix_origin=False)
    F2 = m0.make_f2(hkl)(torch.tensor(m0.x0())).detach().numpy()
    sig = 0.02 * F2 + 1e-3 * F2.max()
    start = perturbed_crystal(t, np.random.default_rng(1), shift_A=0.04)
    good = refine(RefineModel(start), hkl, F2, sig)
    assert good.full_rank and good.converged, (good.reason, good.null_params)
    bad = refine(RefineModel(start, fix_origin=False), hkl, F2, sig)
    assert not bad.full_rank and "rank deficient" in bad.reason


def test_fixed_atoms_scatter_but_have_no_parameters_and_the_polar_origin_is_still_pinned():
    t = thiourea()
    fixed = ("H1A", "H1B", "H2A", "H2B")
    m_all = RefineModel(t)
    m_fix = RefineModel(t, fixed_atoms=fixed)
    assert not any(n.split(".")[0] in fixed for n in m_fix.names)
    assert len(m_fix.origin_fixed) == 1 and m_fix.n_par == RefineModel(t, fix_origin=False, fixed_atoms=fixed).n_par - 1      # the origin is pinned on a refined atom even with fixed hydrogens
    hkl = hkl_set(t)
    f_all = m_all.make_f2(hkl)(torch.tensor(m_all.x0())).detach().numpy()
    f_fix = m_fix.make_f2(hkl)(torch.tensor(m_fix.x0())).detach().numpy()
    assert np.allclose(f_all, f_fix, rtol=1e-12)                          # at the start values the two models are the same structure
    with pytest.raises(ValueError, match="fixed_atoms"):
        RefineModel(t, fixed_atoms=("Zz",))
    sig = 0.02 * f_fix + 1e-3 * f_fix.max()
    start = perturbed_crystal(t, np.random.default_rng(2), shift_A=0.03)
    r = refine(RefineModel(start, fixed_atoms=fixed), hkl, f_fix, sig)
    assert r.full_rank and r.converged, (r.reason, r.null_params)
    ab = r.absolute                                                       # the fixed atoms keep the START values (here displaced from the truth), exactly
    for a_ in start.atoms:
        if a_.label in fixed:
            row = ab[ab.atom == a_.label].iloc[0]
            assert np.array_equal([row.x, row.y, row.z], np.array(a_.fract, float)) and abs(row.U_iso - a_.B_iso / EIGHT_PI2) < 1e-15
    moved = [lab for lab in ("S1", "C1", "N1", "S2", "C2", "N2") if not np.allclose(ab[ab.atom == lab].iloc[0][["x", "y", "z"]].to_numpy(float),
                                                                                    [a_.fract for a_ in start.atoms if a_.label == lab][0])]
    assert len(moved) >= 4                                                # the refined atoms did move


def test_hydrogens_do_not_anchor_the_origin_loosely():
    """Without the pin, a refinement with fixed H and noisy data lets the heavy atoms drift together along the polar axis (observed on the thiourea frames: 0.28 A, the same
    for all six atoms, GooF unchanged; here -0.16 / +0.14 / +0.38 Å for three noise draws) and the conditioning is ten times worse. With the pin the common shift stays
    below 0.08 A."""
    t = thiourea()
    fixed = ("H1A", "H1B", "H2A", "H2B")
    hkl = hkl_set(t, d_min=0.95)
    m0 = RefineModel(t, fixed_atoms=fixed)
    f = m0.make_f2(hkl)(torch.tensor(m0.x0())).detach().numpy()
    A = t.lattice.cartesian_vectors()
    heavy = [a for a in t.atoms if a.element != "H"]
    sig = 0.15 * f + 0.02 * f.max()
    for seed in (0, 1, 2):
        obs = f + np.random.default_rng(seed).normal(size=f.shape) * sig
        res = {}
        for pin in (True, False):
            start = perturbed_crystal(t, np.random.default_rng(10 + seed), shift_A=0.04)
            r = refine(RefineModel(start, fixed_atoms=fixed, fix_origin=pin), hkl, obs, sig, outlier_sigma=5.0)
            dx = [((np.array(r.absolute[r.absolute.atom == a.label].iloc[0][["x", "y", "z"]], float) - np.array(a.fract)) @ A)[0] for a in heavy]
            res[pin] = (float(np.mean(dx)), r.sv_ratio)
        assert abs(res[True][0]) < 0.08 and abs(res[False][0]) > 0.12, (seed, res)
        assert res[True][1] > 5 * res[False][1], (seed, res)
