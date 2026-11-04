"""Crystal-frame consistency of the elastic bases and the reference-length (d0) model.

MIDAS orientation matrices (crystal -> lab) live in the ``midas_hkls`` B-matrix frame:
a* along x, b* in the xy plane (60 deg from x for hexagonal), c* along z; real-space a
at -30/150 deg, b at 90 deg. ``orientation.make_symmetries`` operators are expressed in
that frame. A stiffness basis (or a d0 basis) is only usable with those orientations if
it is invariant under the same operators / built from the real-space axes of that frame.
"""
import numpy as np
import pytest

from midas_stress import (
    stiffness_from_cij, symmetry_parameterisation, trigonal_setting_for_space_group,
)
from midas_stress.equilibrium import recover_d0_anisotropic
from midas_stress.orientation import make_symmetries, quat_to_orient_mat
from midas_stress.tensor import lattice_params_to_A_matrix, lattice_params_to_strain

MANDEL_ORDER = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]     # midas [xx yy zz xy xz yz]
MND = np.array([1, 1, 1, 2 ** 0.5, 2 ** 0.5, 2 ** 0.5])


def _to_tensor(C6):
    C = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(MANDEL_ORDER):
        for J, (k, l) in enumerate(MANDEL_ORDER):
            for a, b in {(i, j), (j, i)}:
                for c, d in {(k, l), (l, k)}:
                    C[a, b, c, d] = C6[I, J] / (MND[I] * MND[J])
    return C


def _ops(sg):
    n, sym = make_symmetries(sg)
    return [np.asarray(quat_to_orient_mat(list(q)), float).reshape(3, 3) for q in sym]


def _noninvariance(sg, symmetry):
    names, P = symmetry_parameterisation(symmetry)
    cij = np.random.default_rng(1).uniform(20, 120, len(names))
    C = _to_tensor(np.tensordot(cij, P, axes=1))
    worst = max(np.abs(np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, C) - C).max()
                for R in _ops(sg))
    return worst / np.abs(C).max()


@pytest.mark.parametrize("sg,symmetry", [
    (225, "cubic"), (194, "hexagonal"), (123, "tetragonal"), (62, "orthorhombic"),
    (14, "monoclinic"),
    (149, "trigonal"), (162, "trigonal"),                                  # 2-fold along a*
    (150, "trigonal_a"), (156, "trigonal_a"), (164, "trigonal_a"),
    (166, "trigonal_a"), (167, "trigonal_a"),                              # 2-fold along a
])
def test_stiffness_basis_is_invariant_under_the_midas_frame_operators(sg, symmetry):
    assert _noninvariance(sg, symmetry) < 1e-12


@pytest.mark.parametrize("sg,wrong", [(167, "trigonal"), (166, "trigonal"),
                                      (149, "trigonal_a"), (162, "trigonal_a")])
def test_the_other_trigonal_setting_is_not_invariant(sg, wrong):
    """The old single 'trigonal' basis was wrong for the P321/R-3c family (e.g. sapphire)."""
    assert _noninvariance(sg, wrong) > 0.1


def test_trigonal_setting_for_space_group():
    assert {sg: trigonal_setting_for_space_group(sg) for sg in (149, 162, 150, 164, 166, 167)} == {
        149: "trigonal", 162: "trigonal", 150: "trigonal_a", 164: "trigonal_a",
        166: "trigonal_a", 167: "trigonal_a"}


def test_trigonal_a_is_the_standard_basis_rotated_90_degrees_about_c():
    names, P = symmetry_parameterisation("trigonal")
    names_a, Pa = symmetry_parameterisation("trigonal_a")
    assert names == names_a
    # the hexagonal-like constants (no C14 shear coupling) are invariant under the rotation
    for k, n in enumerate(names):
        if n != "C14":
            np.testing.assert_allclose(Pa[k], P[k], atol=1e-12)
        else:
            assert np.abs(Pa[k] - P[k]).max() > 0.5
    c = {"C11": 150.0, "C12": 60.0, "C13": 50.0, "C14": 12.0, "C33": 170.0, "C44": 40.0}
    assert stiffness_from_cij(c, "trigonal").shape == (6, 6)
    assert stiffness_from_cij(c, "trigonal_a").shape == (6, 6)


# ---- d0 recovery for monoclinic cells ------------------------------------------------

def _rot(rng, spread_deg):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = np.radians(spread_deg) * rng.normal()
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def test_d0_recovery_is_exact_for_a_scaled_monoclinic_reference():
    """Reference error along real-space a (beta = 105): the exact strain basis recovers it;
    the old diag(1,0,0) approximation left a residual of several percent of the error."""
    rng = np.random.default_rng(5)
    true = np.array([4.0, 5.0, 6.0, 90.0, 105.0, 90.0])
    s = 2e-3
    assumed = true.copy()
    assumed[0] *= 1.0 + s
    n = 60
    base = _rot(rng, 180.0)
    OM = np.array([_rot(rng, 8.0) @ base for _ in range(n)])
    lp = np.tile(true, (n, 1))                        # stress-free grains: true cell everywhere
    Cm = np.diag([160.0, 150.0, 170.0, 60.0, 55.0, 50.0]) * 1.0
    Cm[0, 1] = Cm[1, 0] = 50.0
    Cm[0, 2] = Cm[2, 0] = 45.0
    Cm[1, 2] = Cm[2, 1] = 55.0
    out = recover_d0_anisotropic(lp, assumed, Cm, OM, crystal_system="monoclinic")
    np.testing.assert_allclose(out["reference_recovered"], true, rtol=2e-5)
    assert out["residual_norm_after"] < 1e-3 * out["residual_norm_before"]


def test_d0_basis_matches_the_cell_motion_it_models():
    """Scaling reference axis k by (1+d) moves the strain of a fixed cell by -d B_k to first
    order, with B_k = sym(A e_k e_k^T A^-1); checked against lattice_params_to_strain."""
    cell = np.array([4.0, 5.0, 6.0, 80.0, 95.0, 100.0])         # triclinic
    A = np.asarray(lattice_params_to_A_matrix(cell), float)
    for k in range(3):
        d = 1e-6
        scaled = cell.copy()
        scaled[k] *= 1.0 + d
        eps = np.asarray(lattice_params_to_strain(cell, scaled), float)
        e = np.zeros((3, 3))
        e[k, k] = 1.0
        M = A @ e @ np.linalg.inv(A)
        np.testing.assert_allclose(eps, -d * 0.5 * (M + M.T), atol=2e-11)
