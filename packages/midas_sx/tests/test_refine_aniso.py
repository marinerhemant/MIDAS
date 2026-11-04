"""Anisotropic displacement parameters (PREREGISTER A22): constraints, conversion to the CIF tensor, and the refinement on exact F²."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup           # noqa: E402
from midas_sx.refine import (EIGHT_PI2, RefineModel, aniso_constraints, perturbed_crystal, refine,   # noqa: E402
                             u_cart_of_cif)
from test_refine import crystal_of, unique_hkl                      # noqa: E402

I4 = SpaceGroup.from_hm("I 4/m m m")
TET = Lattice(3.83, 3.83, 20.5, 90, 90, 90)


@pytest.mark.parametrize("fract,names", [((0, 0, 0.319), ["Uxx", "Uzz"]),             # 4e  4mm
                                         ((0, 0.5, 0.095), ["Uxx", "Uyy", "Uzz"]),     # 8g  mm2
                                         ((0, 0, 0.0), ["Uxx", "Uzz"]),                # 2a  4/mmm
                                         ((0.1, 0.2, 0.3), ["Uxx", "Uyy", "Uzz", "Uxy", "Uxz", "Uyz"])])   # general
def test_wyckoff_sites_get_the_expected_free_tensor_entries(fract, names):
    assert aniso_constraints(TET, I4, fract)["names"] == names


def test_cubic_site_has_one_parameter_and_it_is_isotropic():
    c = aniso_constraints(Lattice(3.6, 3.6, 3.6, 90, 90, 90), SpaceGroup.from_hm("F m -3 m"), (0, 0, 0))
    assert c["names"] == ["Uxx"] and np.allclose(c["B_cart"][:, 0], [1, 1, 1, 0, 0, 0])


def test_cif_conversion_roundtrip_in_a_triclinic_cell():
    lat = Lattice(5.1, 6.3, 7.7, 83.0, 101.0, 112.0)
    c = aniso_constraints(lat, SpaceGroup.from_hm("P 1"), (0.1, 0.2, 0.3))
    p = np.array([0.004, 0.006, 0.012, 0.001, 0.002, -0.001])
    u6c = c["B_cart"] @ p
    u6_cif = c["to_cif"] @ u6c
    Um = np.array([[u6c[0], u6c[3], u6c[4]], [u6c[3], u6c[1], u6c[5]], [u6c[4], u6c[5], u6c[2]]])
    assert np.allclose(u_cart_of_cif(lat, u6_cif), Um, atol=1e-12)


def _gemmi_vs_midas_f2(lattice_cell, sg, fract, u6_cif, hkls):
    import gemmi
    import torch
    from midas_hkls import structure_factors
    ss = gemmi.SmallStructure(); ss.cell = gemmi.UnitCell(*lattice_cell); ss.spacegroup = gemmi.find_spacegroup_by_name(sg)
    g = gemmi.SmallStructure.Site(); g.label = "Ni"; g.element = gemmi.Element("Ni"); g.fract = gemmi.Fractional(*fract)
    g.occ = 1.0; g.u_iso = 0.01; g.aniso = gemmi.SMat33d(*u6_cif)
    ss.add_site(g); ss.setup_cell_images(); ss.change_occupancies_to_crystallographic()
    calc = gemmi.StructureFactorCalculatorX(ss.cell)
    fg = np.array([abs(calc.calculate_sf_from_small_structure(ss, [int(v) for v in h])) ** 2 for h in hkls])
    cr = Crystal(Lattice(*lattice_cell), SpaceGroup.from_hm(sg), [Atom("Ni", fract, 1.0, 0.8, U_aniso=tuple(u6_cif), label="Ni")], "t")
    F = structure_factors(cr.to_torch(dtype=torch.float64), torch.as_tensor(np.array(hkls)))
    return fg / (F.real ** 2 + F.imag ** 2).numpy()


HK = [(1, 0, 3), (1, 1, 4), (2, 0, 2), (0, 1, 5), (2, 1, 1), (1, 2, 3), (3, 1, 2)]


@pytest.mark.parametrize("cell,sg,fract", [((4.0, 4.0, 6.0, 90, 90, 120), "P 6/m m m", (1 / 3, 2 / 3, 0.5)),    # 3-fold: R mixes a and b
                                           ((3.83, 3.83, 20.5, 90, 90, 90), "I 4/m m m", (0, 0, 0.319))])
def test_constrained_tensor_gives_the_same_f2_in_gemmi_and_midas_hkls_and_a_wrong_tensor_does_not(cell, sg, fract):
    lat = Lattice(*cell)
    c = aniso_constraints(lat, SpaceGroup.from_hm(sg), fract)
    p = np.array([0.004, 0.012])
    u6 = c["to_cif"] @ (c["B_cart"] @ p)
    assert np.allclose(_gemmi_vs_midas_f2(cell, sg, fract, u6, HK), 1.0, atol=1e-6)
    bad = u6.copy(); bad[3] += 0.004                                       # U12 not allowed by the site symmetry
    assert np.abs(_gemmi_vs_midas_f2(cell, sg, fract, bad, HK) - 1.0).max() > 1e-3


def test_exact_f2_of_an_anisotropic_structure_is_recovered_from_a_displaced_isotropic_start():
    st = sim.structure_s2_bilayer_aniso()
    truth = crystal_of(sim.structure_s2_bilayer())              # iso starting values; aniso truth is in the DATA
    hkl, f2 = unique_hkl(st, 0.8)                                  # gemmi aniso F2 (the simulator's own code path)
    sigma = 0.01 * f2.max() * np.ones(len(f2))
    start = perturbed_crystal(truth, np.random.default_rng(2), shift_A=0.05)
    m = RefineModel(start, aniso_sites=("La1", "La2", "Ni"))
    assert [n for n in m.names if "." in n and n.split(".")[1].startswith("Uxx") or n.endswith("Uzz")] == \
        ["La1.Uxx", "La1.Uzz", "La2.Uxx", "La2.Uzz", "Ni.Uxx", "Ni.Uzz"]
    res = refine(m, hkl, f2, sigma)
    assert res.converged, res.reason
    got = dict(zip(res.names, res.x))
    want = {"La1.Uxx": 0.0040, "La1.Uzz": 0.0120, "La2.Uxx": 0.0045, "La2.Uzz": 0.0100, "Ni.Uxx": 0.0035, "Ni.Uzz": 0.0080}
    for k, v in want.items():
        assert abs(got[k] - v) < 5e-5, (k, got[k], v)             # cross-code floor (midas_hkls vs gemmi tables)
    cell = np.array(st.cell[:3])
    for atom, row in zip(crystal_of(st).atoms, res.absolute.itertuples()):
        assert (np.abs((np.array([row.x, row.y, row.z]) - np.array(atom.fract)) * cell)).max() < 2e-3
    assert res.goof < 0.5


def test_isotropic_refinement_of_anisotropic_data_does_not_fit():
    st = sim.structure_s2_bilayer_aniso()
    truth = crystal_of(sim.structure_s2_bilayer())
    hkl, f2 = unique_hkl(st, 0.8)
    sigma = 0.01 * f2.max() * np.ones(len(f2))
    start = perturbed_crystal(truth, np.random.default_rng(2), shift_A=0.05)
    iso = refine(RefineModel(start), hkl, f2, sigma)
    ani = refine(RefineModel(start, aniso_sites=("La1", "La2", "Ni")), hkl, f2, sigma)
    assert iso.goof > 3 * ani.goof


def test_unknown_site_label_is_refused():
    with pytest.raises(ValueError, match="aniso_sites"):
        RefineModel(crystal_of(sim.structure_s2_bilayer()), aniso_sites=("Pu",))
