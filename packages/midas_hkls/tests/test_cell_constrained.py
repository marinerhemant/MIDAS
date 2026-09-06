"""Constrained and joint cell refinement. Truth is planted, so a passing test
means the estimator recovered something real, not that it agreed with itself."""
from __future__ import annotations
import math
import numpy as np
import pytest

from midas_hkls import Lattice
from midas_hkls.cell_constrained import (FREE_PARAMS, DomainData,
                                         refine_cell_constrained,
                                         refine_cell_joint, split_with_error)

A0, C0 = 3.6116, 19.2516
SPLIT = 0.34


def _cell(a, b, c):
    return (a, b, c, 90., 90., 90.)


def _spots(cell, hkl, sigma=3e-4, seed=0):
    lat = Lattice(a=cell[0], b=cell[1], c=cell[2],
                  alpha=cell[3], beta=cell[4], gamma=cell[5])
    B = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
    return hkl @ B.T + np.random.default_rng(seed).normal(0, sigma, (len(hkl), 3))


def _hkl(mx=3, mz=6):
    return np.array([(i, j, k) for i in range(-mx, mx+1) for j in range(-mx, mx+1)
                     for k in range(-mz, mz+1)
                     if (i, j, k) != (0, 0, 0) and (i+j+k) % 2 == 0], float)


def test_a_planted_orthorhombic_split_is_recovered():
    b0 = A0 * (1 + SPLIT/100)
    h = _hkl()
    g = _spots(_cell(A0, b0, C0), h)
    f = refine_cell_constrained(h, g, system="orthorhombic",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    got, sig, z = split_with_error(f)
    assert abs(got - SPLIT) < 0.05, f"recovered {got}, planted {SPLIT}"
    assert z > 10


@pytest.mark.parametrize("system,n_free", [(k, len(v)) for k, v in FREE_PARAMS.items()])
def test_each_system_refines_only_its_free_parameters(system, n_free):
    h = _hkl()
    g = _spots(_cell(A0, A0, C0), h)
    f = refine_cell_constrained(h, g, system=system,
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert len(f.free_names) == n_free
    c = f.cell
    if system in ("cubic",):
        assert abs(c[0]-c[1]) < 1e-9 and abs(c[1]-c[2]) < 1e-9
    if system in ("tetragonal", "hexagonal"):
        assert abs(c[0]-c[1]) < 1e-9
    if system in ("orthorhombic", "tetragonal", "cubic"):
        assert all(abs(v-90.) < 1e-9 for v in c[3:])
    if system == "hexagonal":
        assert abs(c[5]-120.) < 1e-9


def test_the_symmetry_constraint_is_itself_a_discriminator():
    """A tetragonal fit forbids the split, so it must fit measurably worse."""
    b0 = A0 * (1 + SPLIT/100)
    h = _hkl(); g = _spots(_cell(A0, b0, C0), h)
    o = refine_cell_constrained(h, g, system="orthorhombic",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    t = refine_cell_constrained(h, g, system="tetragonal",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert t.rms > 2.0 * o.rms


def test_a_tetragonal_fit_reports_the_split_as_imposed_not_measured():
    h = _hkl(); g = _spots(_cell(A0, A0, C0), h)
    f = refine_cell_constrained(h, g, system="tetragonal",
                                cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    split, sig, z = split_with_error(f)
    assert split == 0.0
    assert math.isnan(sig), "a==b by symmetry is imposed; sigma must not be a number"


def test_joint_refinement_shares_one_cell_across_domains():
    b0 = A0 * (1 + SPLIT/100)
    cell = _cell(A0, b0, C0)
    h = _hkl()
    doms = []
    for d in range(3):
        idx = np.random.default_rng(d).choice(len(h), 60, replace=False)
        hh = h[idx]
        g = _spots(cell, hh, seed=100+d)
        R = np.linalg.qr(np.random.default_rng(200+d).normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0: R[:, 2] *= -1
        doms.append(DomainData(hh, g @ R.T))
    f = refine_cell_joint(doms, system="orthorhombic",
                          cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
    assert f.n_domains == 3 and f.n_reflections == 180
    assert len(f.rotvecs) == 3
    got, _, _ = split_with_error(f)
    assert abs(got - SPLIT) < 0.08, f"joint fit recovered {got}"


def test_more_domains_on_one_cell_beat_one_domain():
    b0 = A0 * (1 + SPLIT/100); cell = _cell(A0, b0, C0); h = _hkl()
    def err(nd, per, seed):
        doms = []
        for d in range(nd):
            idx = np.random.default_rng(seed+d).choice(len(h), per, replace=False)
            doms.append(DomainData(h[idx], _spots(cell, h[idx], sigma=3e-3,
                                                  seed=seed+50+d)))
        f = refine_cell_joint(doms, system="orthorhombic",
                              cell0=(3.6, 3.6, 19.2, 90., 90., 90.))
        return f.cell_sigma[0]
    assert err(4, 40, 0) < err(1, 40, 0)


def test_the_module_says_why_a_free_triclinic_cell_is_the_wrong_tool():
    import midas_hkls.cell_constrained as cc
    doc = " ".join(cc.__doc__.split())        # docstring wraps; compare flat
    assert "absorb noise into exactly the quantity being measured" in doc
    assert "bootstrap" in doc
