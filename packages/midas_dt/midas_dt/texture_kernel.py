"""Radial orientation kernels on SO(3): a texture **simulator** and test fixture.

A de la Vallee Poussin style kernel ``cos^(2 kappa)(omega/2)`` centred on an
orientation gives a texture with a controllable half-width that can be both
*sampled* (draw orientations from it) and *expanded* (turn it into exact GSH
coefficients through the addition theorem). Having both is what makes an
end-to-end operator gate possible: plant a kernel, sample real orientations from
it, bin their {hkl} normals into a Monte-Carlo pole figure, and compare against
what :mod:`midas_dt.gsh` predicts in closed form. The two paths share only the
geometry, so agreement tests the operator rather than the algebra restating
itself.

**Scope, stated because the obvious next step does not work.** This is a
simulator and a validation fixture. It is *not* an inversion basis here.

Non-negative radial-basis inversion is a real and published idea -- weights
``w_i >= 0`` on a grid of these kernels make the reconstructed ODF non-negative
pointwise, which is the classical Matthies escape from the odd-l ghost null
space, and Carlsen et al. 2025 (J. Appl. Cryst. 58, 10.1107/S1600576725001426)
argue the coefficients go sparse exactly when a voxel holds few grains. We tried
it against our own GSH result and the comparison was **INVALID**: the two arms
had different degrees of freedom, so the residuals were not comparable and no
conclusion about either basis follows. Recorded in
``manuals/xrd-ct/LAB_NOTEBOOK.md``.

One conceptual error from that attempt is worth carrying, because it is easy to
repeat: **a kernel does not have to be band-limited at the operator's L.** The
*operator* is truncated at ``L``, but non-negativity constrains the *full*
function, so a kernel used as a positivity-carrying basis element needs its own,
much higher, expansion order. :func:`radial_coeffs` takes ``L`` explicitly for
that reason.

How much a truncation costs, measured on the cubic-symmetrised kernel's peak
(``tests/test_texture_kernel.py`` pins these):

======================  ======  ======  ======  ======
half-width              L=6     L=10    L=16    L=22
======================  ======  ======  ======  ======
8 deg                   5.8 %   22.7 %  61.6 %  89.2 %
16 deg                  46.8 %  88.5 %  99.8 %  100 %
40 deg                  100 %   100 %   100 %   100 %
======================  ======  ======  ======  ======

So it is the **sharp** kernels that lose amplitude, not the wide ones -- a 40
degree kernel really is band-limited by l=6. (An earlier note in this project had
this backwards, naming a 40-degree kernel as the one that lost amplitude;
corrected here against the numbers above.) The mechanism is symmetry rather than
bandwidth: cubic symmetry has ``M(2) = 0``, so the l=2 term -- the *largest*
coefficient for any kernel sharper than ~30 degrees -- is annihilated outright,
and ``L = 6`` leaves a cubic kernel with only ``l = 0, 4, 6``.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .gsh import SymGSH, cubic_rotations, wigner_D

__all__ = [
    "halfwidth_deg",
    "kappa_for_halfwidth",
    "kernel_profile",
    "kernel_to_gsh",
    "radial_coeffs",
    "sample_kernel",
    "sample_kernel_angles",
]


# ------------------------------------------------------------ radial profile
def kernel_profile(omega: np.ndarray, kappa: float) -> np.ndarray:
    """de la Vallee Poussin profile ``cos^(2 kappa)(omega/2)``, non-negative always.

    ``omega`` is the rotation angle from the kernel centre, in radians. Pointwise
    non-negativity is the whole point: it is what a positivity-constrained
    reconstruction would inherit.
    """
    return np.cos(np.asarray(omega, dtype=float) / 2.0) ** (2.0 * kappa)


def halfwidth_deg(kappa: float) -> float:
    """Angle in degrees at which the profile falls to half its peak."""
    return float(np.degrees(2.0 * np.arccos(2.0 ** (-1.0 / (2.0 * kappa)))))


def kappa_for_halfwidth(target_deg: float) -> float:
    """Invert :func:`halfwidth_deg`: the concentration for a wanted half-width."""
    if not 0.0 < target_deg < 180.0:
        raise ValueError(f"half-width must be in (0, 180) degrees, got {target_deg}")
    c = np.cos(np.radians(target_deg) / 2.0)
    return float(-np.log(2.0) / (2.0 * np.log(c)))


def radial_coeffs(L: int, kappa: float, n_quad: int = 20001) -> np.ndarray:
    """``Ahat_l`` for ``l = 0..L``, by projection onto the SO(3) characters.

    Normalised Haar measure on SO(3) has rotation-angle density
    ``p(omega) = (1 - cos omega)/pi`` on ``[0, pi]`` (which integrates to 1), and
    the characters ``chi_l(omega) = sin((2l+1) omega/2) / sin(omega/2)`` are
    orthogonal under it. The profile is rescaled to Haar mean 1, so ``Ahat_0 == 1``
    exactly and a kernel carries the same ``l = 0`` coefficient as the uniform
    ODF -- which is what makes a pole-figure *ratio* against uniform a clean
    comparison, with every normalisation convention cancelling.

    ``L`` is the kernel's own expansion order and should be well above the
    operator's truncation; see the module docstring.
    """
    w = np.linspace(1e-9, np.pi - 1e-9, n_quad)
    p = (1.0 - np.cos(w)) / np.pi
    f = kernel_profile(w, kappa)
    f = f / np.trapezoid(f * p, w)                       # Haar mean 1
    out = np.empty(L + 1)
    for l in range(L + 1):
        chi = np.sin((2 * l + 1) * w / 2.0) / np.sin(w / 2.0)
        out[l] = np.trapezoid(f * chi * p, w)
    return out


def sample_kernel_angles(kappa: float, n: int, rng, n_quad: int = 4001):
    """Draw ``omega`` from the kernel's Haar-weighted angular density.

    The density is ``profile(omega) * (1 - cos omega)``, not the profile alone:
    forgetting the Haar factor over-weights small angles and produces a texture
    sharper than the ``kappa`` asked for.
    """
    w = np.linspace(0.0, np.pi, n_quad)
    dens = kernel_profile(w, kappa) * (1.0 - np.cos(w))
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, w)


def sample_kernel(centre: Rotation, kappa: float, n: int, rng,
                  group=None) -> Rotation:
    """Draw ``n`` orientations from the **symmetrised** kernel.

    A draw is ``g = g_centre . delta . S`` with ``delta`` from the radial profile
    and ``S`` uniform on the crystal's proper rotations.

    **Right multiplication is the correct side** and is not a free choice: a
    forward model computes sample-frame normals as ``n_sample = g h_crystal``, so
    crystal symmetry acts on the right. Putting ``S`` on the left symmetrises
    the *sample* frame instead, which is a different (and wrong) physical
    statement -- and one that a symmetry test cannot catch, because the sampled
    set is still symmetric, just about the wrong thing.
    """
    grp = cubic_rotations() if group is None else group
    if not isinstance(grp, Rotation):
        grp = Rotation.from_matrix(np.asarray(grp, dtype=float))
    ang = sample_kernel_angles(kappa, n, rng)
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    delta = Rotation.from_rotvec(ax * ang[:, None])
    pick = grp[rng.integers(0, len(grp), size=n)]
    return centre * delta * pick


def kernel_to_gsh(basis: SymGSH, centres: Rotation,
                  ahat: np.ndarray) -> np.ndarray:
    """Transform ``T`` whose column ``i`` is kernel ``i``'s GSH coefficient vector.

    Derivation, matched term by term against :meth:`SymGSH.pole_row`'s convention
    ``C[l,m,n] = sum_mu c[l,mu,m] conj(b[n,mu])``::

        C_sym[l,m,n] = Ahat_l (1/|G|) sum_S conj(D^l_mn(g_i S))
                     = Ahat_l sum_p conj(D^l_mp(g_i)) conj(P^l_pn),   P = b b^H
        c[l,mu,m]    = sum_n C_sym[l,m,n] b[n,mu]

    so the per-level block is ``Ahat_l * (conj(D) @ conj(P)) @ b`` indexed
    ``[m, mu]``, flattened mu-major / m-minor to match :attr:`SymGSH.index`.

    A conjugation error in that last line is **invisible to a symmetry test** --
    the result is still symmetric, just wrong -- which is why
    ``tests/test_texture_kernel.py`` gates it with a Monte-Carlo pole figure
    against sampled orientations instead.
    """
    if not isinstance(centres, Rotation):
        centres = Rotation.from_matrix(np.asarray(centres, dtype=float))
    if len(ahat) <= basis.L:
        raise ValueError(
            f"ahat has {len(ahat)} entries but the basis needs l up to "
            f"{basis.L}; call radial_coeffs(L>={basis.L}, ...)")
    T = np.zeros((basis.n_coef, len(centres)), dtype=complex)
    off = 0
    for l, b in basis.levels:
        dim, M = 2 * l + 1, b.shape[1]
        P = b @ b.conj().T
        for i, rot in enumerate(centres):
            D = wigner_D(l, rot)
            blk = ahat[l] * (D.conj() @ P.conj()) @ b      # (2l+1, M) as [m, mu]
            T[off:off + M * dim, i] = blk.T.ravel()        # mu-major, m-minor
        off += M * dim
    if off != basis.n_coef:                                # pragma: no cover
        raise AssertionError(f"coefficient layout mismatch: {off} != {basis.n_coef}")
    return T
