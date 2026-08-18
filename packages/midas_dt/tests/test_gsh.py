"""Gates on the symmetry-adapted GSH operator in midas_dt.gsh.

The decisive test is :func:`test_fibre_integral_equals_closed_form`: the claim
that the operator needs no Rodrigues mesh rests entirely on the identity

    (1/2pi) int_0^{2pi} D^l_{mn}(R_y(psi) g_0) dpsi
        = (4 pi / (2l+1)) conj(Y_l^m(y)) Y_l^n(h)

so that identity is checked against brute-force quadrature of its own left-hand
side, mode by mode, to machine precision.

Nothing here assumes a Wigner-d sign or phase convention. The D-matrices are
built by exponentiating angular-momentum operators, and every property the
derivation leans on -- unitarity, the homomorphism, the D_{m0} <-> Y_l^m link --
is checked numerically first. That ordering matters: a convention error would
otherwise show up only as a plausibly-shaped, wrong pole figure.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_dt.gsh import (
    CubicGSH,
    SymGSH,
    cubic_rotations,
    hkl_family,
    invariant_basis,
    sph_harm_vec,
    wigner_D,
)

RNG = np.random.default_rng(20260815)

TI_ALPHA = (2.9505, 2.9505, 4.6826, 90.0, 90.0, 120.0)


def _rand_unit(n: int = 1) -> np.ndarray:
    v = RNG.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _Y(l: int, m: int, v: np.ndarray) -> complex:
    return complex(sph_harm_vec(l, v[None, :])[0, m + l])


# ------------------------------------------- the D-matrices are the irreps
@pytest.mark.parametrize("l", range(7))
def test_wigner_D_is_unitary_and_a_homomorphism(l):
    for _ in range(6):
        r1, r2 = Rotation.random(2, rng=RNG)
        d1, d2 = wigner_D(l, r1), wigner_D(l, r2)
        assert np.abs(d1.conj().T @ d1 - np.eye(2 * l + 1)).max() < 1e-10
        assert np.abs(wigner_D(l, r1 * r2) - d1 @ d2).max() < 1e-10


@pytest.mark.parametrize("l", range(7))
def test_D_m0_is_the_spherical_harmonic(l):
    """D^l_{m0}(R) == sqrt(4pi/(2l+1)) conj(Y_l^m(R zhat)) -- checked, not assumed."""
    pref = np.sqrt(4 * np.pi / (2 * l + 1))
    for _ in range(6):
        rot = Rotation.random(rng=RNG)
        n = rot.apply([0.0, 0.0, 1.0])
        d = wigner_D(l, rot)
        for m in range(-l, l + 1):
            assert d[m + l, l] == pytest.approx(pref * np.conj(_Y(l, m, n)),
                                                abs=1e-11)


def test_wigner_D_accepts_a_bare_matrix():
    rot = Rotation.random(rng=RNG)
    assert np.allclose(wigner_D(3, rot), wigner_D(3, rot.as_matrix()))


# ------------------------------------------------------- the closed form
def _fibre_average(l: int, h: np.ndarray, y: np.ndarray, n_psi: int) -> np.ndarray:
    """Brute-force (1/2pi) int D^l(g) dpsi over the fibre {g : g h = y}."""
    g0 = Rotation.align_vectors(y[None, :], h[None, :])[0]
    assert np.allclose(g0.apply(h), y, atol=1e-12)
    psi = 2 * np.pi * np.arange(n_psi) / n_psi     # periodic: midpoint == trapz
    acc = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
    for p in psi:
        acc += wigner_D(l, Rotation.from_rotvec(p * y) * g0)
    return acc / n_psi


def _closed_form(l: int, h: np.ndarray, y: np.ndarray) -> np.ndarray:
    ys = np.array([_Y(l, m, y) for m in range(-l, l + 1)])
    yh = np.array([_Y(l, n, h) for n in range(-l, l + 1)])
    return (4 * np.pi / (2 * l + 1)) * np.outer(np.conj(ys), yh)


@pytest.mark.parametrize("l", range(9))
def test_fibre_integral_equals_closed_form(l):
    """The whole operator rests on this. Machine precision or it is not true."""
    worst = 0.0
    for _ in range(4):
        h, y = _rand_unit()[0], _rand_unit()[0]
        brute = _fibre_average(l, h, y, n_psi=4 * l + 16)
        exact = _closed_form(l, h, y)
        worst = max(worst, np.abs(brute - exact).max() / np.abs(exact).max())
    assert worst < 1e-9, f"l={l}: relative error {worst:.3e}"


@pytest.mark.parametrize("l", (3, 6, 8))
def test_fibre_quadrature_is_exact_at_2l_plus_1_points(l):
    """The integrand is a trig polynomial of degree <= l, so it must be."""
    h, y = _rand_unit()[0], _rand_unit()[0]
    coarse = _fibre_average(l, h, y, n_psi=2 * l + 1)
    fine = _fibre_average(l, h, y, n_psi=400)
    assert np.abs(coarse - fine).max() < 1e-10


# ------------------------------------------------------------- symmetry
def test_cubic_rotations_is_a_closed_group_of_24():
    rots = cubic_rotations()
    assert len(rots) == 24
    keys = {tuple(np.round(m, 8).ravel() + 0.0) for m in rots.as_matrix()}
    assert len(keys) == 24
    for a in rots:
        for b in rots:
            assert tuple(np.round((a * b).as_matrix(), 8).ravel() + 0.0) in keys


def test_cubic_rotations_matches_midas_hkls():
    """The independent reference and the general machinery must agree exactly.

    `gsh.cubic_rotations` enumerates signed permutation matrices;
    `midas_hkls.point_group_rotations` closes a pair of generators. Every cubic
    validation in this module's history was measured against the former, so this
    is what transfers that validation to the latter.
    """
    pg = pytest.importorskip("midas_hkls.point_group")
    ours = {tuple(np.round(m, 8).ravel() + 0.0)
            for m in cubic_rotations().as_matrix()}
    theirs = {tuple(np.round(m, 8).ravel() + 0.0)
              for m in pg.point_group_rotations("432")}
    assert ours == theirs


def test_cubic_has_no_l2_invariant():
    """The classical result: M(2) == 0 for cubic, so l=2 carries no texture.

    A basis that reported an l=2 invariant would be a symmetry bug, and it would
    show up as an over-flexible fit rather than as an error.
    """
    assert invariant_basis(2, cubic_rotations()).shape[1] == 0
    assert invariant_basis(0, cubic_rotations()).shape[1] == 1
    assert invariant_basis(4, cubic_rotations()).shape[1] == 1


def test_invariant_basis_rejects_a_non_group():
    """Character count vs projector rank is a real gate, not decoration."""
    partial = cubic_rotations()[:5]           # not closed
    with pytest.raises(AssertionError, match="character count"):
        invariant_basis(4, partial)


def test_invariant_basis_is_orthonormal():
    for l in (0, 4, 6, 8):
        b = invariant_basis(l, cubic_rotations())
        if b.shape[1]:
            assert np.allclose(b.conj().T @ b, np.eye(b.shape[1]), atol=1e-10)


# --------------------------------------------------------- basis bookkeeping
@pytest.mark.parametrize("L", (4, 6, 10, 16))
def test_only_even_orders_are_carried(L):
    basis = CubicGSH(L)
    assert all(l % 2 == 0 for l, _ in basis.levels)
    assert all(l % 2 == 0 for l, _, _ in basis.index)
    assert basis.n_coef == sum(b.shape[1] * (2 * l + 1) for l, b in basis.levels)


def test_ghost_dimension_is_nonzero_and_reported():
    """Odd l is annihilated for every scan design; the size must be visible."""
    basis = CubicGSH(10)
    assert basis.ghost_dimension() > 0
    # the ghost subspace is genuinely large -- this is why positivity matters
    assert basis.ghost_dimension() >= 0.1 * basis.n_coef


def test_coefficient_count_grows_faster_than_linearly_in_L():
    """The identifiability trap: unknowns ~ L^3 while the row count is fixed."""
    counts = [CubicGSH(L).n_coef for L in (4, 6, 8, 10)]
    assert counts == sorted(counts)
    assert counts[-1] > 4 * counts[0]


def test_repr_mentions_the_ghost_dimension():
    assert "ghost" in repr(CubicGSH(6))


# -------------------------------------------------------------- pole figures
def test_uniform_odf_gives_a_constant_pole_figure():
    """Only the l=0 coefficient is populated => no azimuthal structure."""
    basis = CubicGSH(6)
    coef = np.zeros(basis.n_coef, dtype=complex)
    coef[0] = 1.0
    normals = hkl_family((1, 1, 1))
    vals = [basis.evaluate(coef, normals, y) for y in _rand_unit(12)]
    assert np.std(np.real(vals)) < 1e-12
    assert np.abs(np.imag(vals)).max() < 1e-12


@pytest.mark.parametrize("hkl,mult", [((1, 1, 1), 8), ((1, 0, 0), 6),
                                      ((1, 1, 0), 12), ((2, 1, 0), 24)])
def test_uniform_pole_density_equals_the_family_multiplicity(hkl, mult):
    """An analytic gate on the entire normalisation chain.

    With only ``a_0 = 1`` populated, the closed form collapses to
    ``4pi * kappa_0 * conj(Y_0^0) = 4pi * (m / sqrt(4pi)) * (1 / sqrt(4pi)) = m``,
    exactly the family multiplicity. Every factor in the operator -- the
    ``4pi/(2l+1)`` prefactor, the ``Y_0^0`` normalisation on both the crystal and
    sample sides, and the invariant basis -- has to be right for this to land on
    an integer, so it catches a dropped or doubled factor that a
    correlation-based test would absorb into its scale.
    """
    basis = CubicGSH(6)
    coef = np.zeros(basis.n_coef, dtype=complex)
    coef[0] = 1.0
    y = np.array([0.3, 0.4, np.sqrt(1.0 - 0.09 - 0.16)])
    val = basis.evaluate(coef, hkl_family(hkl), y)
    assert val.real == pytest.approx(float(mult), abs=1e-9)
    assert abs(val.imag) < 1e-9


@pytest.mark.parametrize("hkl,mult", [((1, 1, 1), 8), ((1, 0, 0), 6),
                                      ((1, 1, 0), 12), ((2, 1, 0), 24)])
def test_hkl_family_multiplicities(hkl, mult):
    fam = hkl_family(hkl)
    assert len(fam) == mult
    assert np.allclose(np.linalg.norm(fam, axis=1), 1.0)


def test_hkl_family_is_sign_closed():
    fam = hkl_family((1, 1, 1))
    keys = {tuple(np.round(v, 6) + 0.0) for v in fam}
    for v in fam:
        assert tuple(np.round(-v, 6) + 0.0) in keys


# ------------------------------------------------------ non-cubic symmetry
def test_hexagonal_basis_differs_from_cubic_and_has_an_l2_invariant():
    pg = pytest.importorskip("midas_hkls.point_group")
    group = pg.proper_rotations_from_space_group("P6_3/mmc", TI_ALPHA)
    assert len(group) == 12
    # 622 DOES have an l=2 invariant (the fibre term), unlike cubic -- which is
    # why a cubic-only model cannot express hcp basal texture at all.
    hexa = SymGSH(6, group=group, lattice=TI_ALPHA)
    assert invariant_basis(2, hexa.group).shape[1] == 1
    assert hexa.n_coef != CubicGSH(6).n_coef


def test_symgsh_accepts_matrices_or_a_rotation():
    pg = pytest.importorskip("midas_hkls.point_group")
    mats = pg.proper_rotations_from_space_group("P6_3/mmc", TI_ALPHA)
    a = SymGSH(4, group=mats, lattice=TI_ALPHA)
    b = SymGSH(4, group=Rotation.from_matrix(mats), lattice=TI_ALPHA)
    assert a.n_coef == b.n_coef


def test_symgsh_rejects_a_badly_shaped_group():
    with pytest.raises(ValueError, match=r"\(n, 3, 3\)"):
        SymGSH(4, group=np.zeros((4, 4)))


def test_families_uses_the_hexagonal_metric():
    """Same {hkl}, hexagonal lattice: must not equal the cubic shortcut."""
    pg = pytest.importorskip("midas_hkls.point_group")
    group = pg.proper_rotations_from_space_group("P6_3/mmc", TI_ALPHA)
    hexa = SymGSH(4, group=group, lattice=TI_ALPHA)
    fam = hexa.families((1, 0, 1))
    assert len(fam) == 12
    assert np.allclose(np.linalg.norm(fam, axis=1), 1.0)
    # the basal normal is in {0001} and perpendicular to every prism normal
    prism = hexa.families((1, 0, 0))
    basal = hexa.families((0, 0, 1))
    assert np.abs(prism @ basal.T).max() < 1e-12


def test_cubic_families_match_the_shortcut():
    """SymGSH.families with no lattice must reproduce hkl_family for cubic."""
    pytest.importorskip("midas_hkls.point_group")
    basis = CubicGSH(4)
    a = {tuple(np.round(v, 6) + 0.0) for v in basis.families((1, 1, 1))}
    b = {tuple(np.round(v, 6) + 0.0) for v in hkl_family((1, 1, 1))}
    assert a == b
