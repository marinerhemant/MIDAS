"""Symmetry-adapted generalised spherical harmonics: the orientation operator.

Texture tomography needs an operator that maps an orientation distribution in a
voxel to the azimuthal intensity its rings show. The classical route discretises
orientation space into a tetrahedral mesh of the Rodrigues fundamental zone and
integrates the ``n || q`` fibre against tetrahedra. That is unnecessary: for a
single GSH mode the fibre integral is **closed form**, the classical Bunge
pole-figure equation,

.. math::

    \\frac{1}{2\\pi}\\int_0^{2\\pi} D^l_{mn}(R_y(\\psi)\\, g_0)\\, d\\psi
        = \\frac{4\\pi}{2l+1}\\, \\overline{Y_l^m(y)}\\, Y_l^n(h)

so the operator is a *table of spherical harmonic values* evaluated at the
crystal direction ``h`` and the sample direction ``y``. No mesh, no tetrahedra,
no fibre intersection anywhere.

**Validation.** Three independent routes agree: the identity above against
brute-force numerical fibre integration (5.6e-15), our own Monte-Carlo pole
figure from sampled orientations (correlation 0.99968), and the third-party
TexTOM implementation (Frewein et al., IUCrJ 11, 2024; code Zenodo
10.5281/zenodo.12543638), which computes the same projection numerically over its
own orientation grid with its own Haar measure and fundamental zone
(correlation 0.999992). ``tests/test_gsh.py`` carries the first; the third is
``scripts/validate_gsh_vs_textom.py``, which needs a TexTOM checkout.

Two structural facts are baked in here rather than discovered downstream:

* **Only even l is measurable.** ``Y_l^n(-h) = (-1)^l Y_l^n(h)`` and diffraction
  cannot separate ``h`` from ``-h``, so odd ``l`` is annihilated by the forward
  operator for *every* scan design. That is the classical ghost subspace; it is
  excluded from the unknowns and its dimension is reported by
  :meth:`SymGSH.ghost_dimension` so a caller cannot mistake "not fitted" for
  "not there". No amount of extra data recovers it -- only a positivity
  constraint does (Matthies), which is what :mod:`midas_dt.texture_kernel` is
  for.
* **Crystal symmetry collapses the n index** to ``M(l)`` dimensions. The
  invariant subspace is built by explicit group averaging and its rank checked
  against the character-theory multiplicity, so two independent routes must agree
  before a basis is returned.

Symmetry comes from :mod:`midas_hkls.point_group` (all 230 space groups),
which is an optional dependency -- ``pip install midas-dt[texture]``.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm_y

__all__ = [
    "CubicGSH",
    "SymGSH",
    "cubic_rotations",
    "hkl_family",
    "invariant_basis",
    "sph_harm_vec",
    "wigner_D",
]


# ------------------------------------------------------------------ irreps
def _angular_momentum(l: int):
    """(J_y, J_z) in the |l, m> basis, m = -l .. +l."""
    m = np.arange(-l, l + 1, dtype=float)
    dim = 2 * l + 1
    up = np.zeros((dim, dim), dtype=complex)
    for i in range(dim - 1):
        up[i + 1, i] = np.sqrt(l * (l + 1) - m[i] * (m[i] + 1))
    jy = (up - up.conj().T) / 2j
    return jy, np.diag(m).astype(complex)


def wigner_D(l: int, rot) -> np.ndarray:
    """``D^l`` for a rotation, rows/cols ordered ``m = -l .. +l``.

    Built by exponentiating angular-momentum operators rather than from a
    Wigner-d recursion, so no sign or phase convention is assumed anywhere: every
    property the closed form leans on (unitarity, the homomorphism, the
    ``D_{m0}`` <-> ``Y_l^m`` link) is then checkable numerically, and
    ``tests/test_gsh.py`` checks them.

    Parameters
    ----------
    l
        Harmonic order.
    rot
        A :class:`scipy.spatial.transform.Rotation`, or a ``(3, 3)`` matrix.
    """
    if not isinstance(rot, Rotation):
        rot = Rotation.from_matrix(np.asarray(rot, dtype=float))
    # Gimbal lock (beta = 0 or pi) is EXPECTED here and harmless, so the warning
    # is suppressed rather than left to alarm a caller into "fixing" it. Every
    # symmetry group contains such elements -- the identity, and every 2-fold
    # about z. At gimbal lock the ZYZ decomposition is a one-parameter family
    # (only alpha + gamma is determined) and scipy returns one member of it. That
    # is sufficient: D^l is a homomorphism, so it depends on the *rotation*, not
    # on which decomposition names it, and every member of the family gives the
    # identical matrix. Verified by test_wigner_D_is_unitary_and_a_homomorphism,
    # which passes on exactly these elements.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        a, b, c = rot.as_euler("ZYZ")
    jy, jz = _angular_momentum(l)
    return expm(-1j * a * jz) @ expm(-1j * b * jy) @ expm(-1j * c * jz)


def sph_harm_vec(l: int, v: np.ndarray) -> np.ndarray:
    """``[Y_l^n(v)]`` for ``n = -l..l``; ``v`` is ``(..., 3)``, need not be unit."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
    phi = np.arctan2(v[:, 1], v[:, 0])
    return np.stack([sph_harm_y(l, n, theta, phi) for n in range(-l, l + 1)],
                    axis=-1)


# ------------------------------------------------------- crystal symmetry
def cubic_rotations() -> Rotation:
    """The 24 proper rotations of the octahedral group, as signed permutations.

    This duplicates ``midas_hkls.point_group_rotations('432')`` on purpose, and
    the duplication is load-bearing rather than an oversight:

    * it is an **independent reference implementation** -- built by enumerating
      signed permutation matrices, with no generators and no group closure -- so
      ``tests/test_gsh.py`` asserting the two agree element-for-element is
      evidence, not a tautology. Every cubic validation in this module's history
      was measured against *this* construction;
    * it keeps the cubic path working without the optional ``[texture]``
      dependency.

    For any other symmetry use :func:`midas_hkls.proper_rotations_from_space_group`.
    """
    mats = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    m = np.zeros((3, 3))
                    m[0, perm[0]], m[1, perm[1]], m[2, perm[2]] = sx, sy, sz
                    if abs(np.linalg.det(m) - 1.0) < 1e-9:
                        mats.append(m)
    mats = np.array(mats)
    assert len(mats) == 24, f"expected 24 proper cubic rotations, got {len(mats)}"
    return Rotation.from_matrix(mats)


def invariant_basis(l: int, group: Rotation) -> np.ndarray:
    """Orthonormal basis ``(2l+1, M(l))`` of the symmetry-invariant subspace.

    Built by averaging ``D^l`` over the group. The rank is checked against the
    character-theory multiplicity ``M(l) = |G|^-1 sum_g chi_l(g)``, which is the
    trace of the same projector -- so a convention error in ``wigner_D`` shows up
    as a rank/trace disagreement rather than as a silently wrong basis.
    """
    dim = 2 * l + 1
    proj = np.zeros((dim, dim), dtype=complex)
    for rot in group:
        proj += wigner_D(l, rot)
    proj /= len(group)
    m_char = int(round(proj.trace().real))
    u, s, _ = np.linalg.svd(proj)
    m_rank = int((s > 1e-8).sum())
    if m_char != m_rank:
        raise AssertionError(
            f"l={l}: character count {m_char} != projector rank {m_rank} -- "
            "the group passed is probably not closed")
    return u[:, :m_rank]


def _as_rotation(group) -> Rotation:
    """Accept a Rotation or an ``(n, 3, 3)`` array of matrices."""
    if group is None:
        return cubic_rotations()
    if isinstance(group, Rotation):
        return group
    arr = np.asarray(group, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(
            f"group must be a Rotation or an (n, 3, 3) array, got {arr.shape}")
    return Rotation.from_matrix(arr)


class SymGSH:
    """Symmetry-adapted, even-l-only GSH basis for any proper point group.

    Parameters
    ----------
    L
        Maximum harmonic order. Only even orders are carried.
    group
        The crystal's **Laue-group proper rotations**, as a
        :class:`~scipy.spatial.transform.Rotation` or an ``(n, 3, 3)`` array --
        i.e. the output of
        :func:`midas_hkls.proper_rotations_from_space_group`, which handles all
        230 space groups and the improper-to-``-R`` mapping Friedel's law
        requires. Defaults to cubic, so every caller written against the
        validated cubic path keeps its behaviour unchanged.
    lattice
        ``(a, b, c, alpha, beta, gamma)`` or a :class:`midas_hkls.Lattice`.
        Needed only by :meth:`families`: for cubic, (hkl) doubles as a Cartesian
        direction and the lattice may be omitted, but for every other system the
        reciprocal metric matters -- in hcp the angle between (10-10) and (0001)
        is not what the raw index triple suggests.

    Notes
    -----
    Choosing ``L`` is not free. Required ``L`` rises with the number of
    crystallites a voxel holds (measured: L=6 at ~1000, L=8 at ~3000, L=10 at
    ~1e4), while the unknown count grows roughly as ``L^3`` and the row count is
    fixed -- so the system goes underdetermined at high ``L`` and an
    underdetermined least-squares fit drives its residual toward zero for free.
    Check :attr:`n_coef` against the number of independent measurements before
    reading any residual as evidence. See ``manuals/xrd-ct/ENVELOPE.md``.
    """

    def __init__(self, L: int, group=None, lattice=None):
        self.L = int(L)
        self.group = _as_rotation(group)
        self.lattice = lattice
        self.levels: list[tuple[int, np.ndarray]] = []
        for l in range(0, self.L + 1, 2):     # even l only: odd l is the ghost
            b = invariant_basis(l, self.group)
            if b.shape[1]:
                self.levels.append((l, b))
        # flat index layout: (l, mu, m), mu-major and m-minor within a level
        self.index: list[tuple[int, int, int]] = []
        for l, b in self.levels:
            for mu in range(b.shape[1]):
                for m in range(-l, l + 1):
                    self.index.append((l, mu, m))
        self.n_coef = len(self.index)

    def __repr__(self) -> str:            # pragma: no cover - convenience
        return (f"SymGSH(L={self.L}, |G|={len(self.group)}, "
                f"n_coef={self.n_coef}, ghost={self.ghost_dimension()})")

    def ghost_dimension(self) -> int:
        """Coefficients annihilated by the forward operator at this L (odd l).

        Reported rather than hidden: these are unrecoverable from *any* amount of
        diffraction data, so a reconstruction is only ever determined up to this
        subspace unless a positivity constraint is imposed.
        """
        tot = 0
        for l in range(1, self.L + 1, 2):
            b = invariant_basis(l, self.group)
            tot += b.shape[1] * (2 * l + 1)
        return tot

    def crystal_side(self, l: int, basis: np.ndarray,
                     normals: np.ndarray) -> np.ndarray:
        """``kappa_l^mu`` = <basis_mu, sum over the {hkl} family of Y_l^n(h)>."""
        k = sph_harm_vec(l, normals).sum(axis=0)           # (2l+1,)
        return basis.conj().T @ k                          # (M,)

    def pole_row(self, normals: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Row of the operator giving ``P_h(y)`` as a functional of the coefficients.

        Parameters
        ----------
        normals
            ``(m, 3)`` symmetry-equivalent plane normals of one {hkl} family, in
            the crystal frame -- from :meth:`families` or
            :func:`midas_hkls.plane_normals`.
        y
            The sample direction, ``(3,)``.
        """
        row = np.zeros(self.n_coef, dtype=complex)
        off = 0
        for l, b in self.levels:
            kappa = self.crystal_side(l, b, normals)        # (M,)
            ym = sph_harm_vec(l, y)[0]                      # (2l+1,)
            pref = 4 * np.pi / (2 * l + 1)
            for mu in range(b.shape[1]):
                row[off:off + 2 * l + 1] = pref * kappa[mu] * np.conj(ym)
                off += 2 * l + 1
        return row

    def evaluate(self, coef: np.ndarray, normals: np.ndarray,
                 y: np.ndarray) -> complex:
        """``P_h(y)`` for a coefficient vector."""
        return complex(self.pole_row(normals, y) @ coef)

    def families(self, hkl) -> np.ndarray:
        """Symmetry-equivalent unit plane normals of {hkl} for THIS group.

        Uses the group and lattice this basis was built with, so a hexagonal
        basis gets hexagonal families. Requires ``midas-dt[texture]``;
        :func:`hkl_family` is the dependency-free cubic shortcut.
        """
        try:
            from midas_hkls.point_group import plane_normals
        except ImportError as exc:                          # pragma: no cover
            raise ImportError(
                "SymGSH.families needs midas-hkls: pip install midas-dt[texture]. "
                "For cubic only, midas_dt.gsh.hkl_family needs no extra."
            ) from exc
        return plane_normals(hkl, self.group.as_matrix(), self.lattice)


class CubicGSH(SymGSH):
    """Cubic specialisation. Behaviour identical to the validated original."""

    def __init__(self, L: int):
        super().__init__(L, cubic_rotations(), None)


def hkl_family(hkl) -> np.ndarray:
    """Unique unit plane normals of a **cubic** {hkl} family, both signs.

    The cubic shortcut: (hkl) already is a Cartesian direction. Wrong for every
    other crystal system -- use :meth:`SymGSH.families` there.
    """
    h = np.array(hkl, dtype=float)
    fam = {tuple(np.round(r.apply(h), 9) + 0.0) for r in cubic_rotations()}
    fam |= {tuple(np.round(-np.array(v), 9) + 0.0) for v in fam}
    arr = np.array(sorted(fam))
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)
