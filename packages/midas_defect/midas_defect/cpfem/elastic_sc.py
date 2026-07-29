"""Per-grain elastic stress / strain / energy prediction under macroscopic load.

This is the Tier-1 crystal-plasticity-adjacent prediction: take the indexed
grain orientations and the applied macroscopic stress, and predict the
per-grain elastic state in three bounds:

  * Reuss        — uniform stress (every grain sees σ_macro). Each grain's
                   strain follows from its orientation-dependent compliance.
                   Lower bound on aggregate stiffness; upper bound on the
                   orientation-driven variability in strain / energy.
  * Voigt        — uniform strain (every grain has the average ε). Each
                   grain's stress follows from its orientation-dependent
                   stiffness. Upper bound on aggregate stiffness.
  * Kroner–Eshelby self-consistent — iterates an effective stiffness C* so
                   that the volume-averaged stress and strain match the
                   macroscopic values, with each grain treated as an
                   inclusion in the homogeneous-effective-medium described
                   by C*. This is the canonical Hill-averaged prediction
                   used by the polycrystal-mechanics community.

The Kroner SC scheme used here is the isotropic-effective-medium
approximation: at each iteration, we extract scalar bulk and shear moduli
(K*, μ*) from C* via Voigt-Reuss-Hill, build the spherical-inclusion
Eshelby tensor for that isotropic medium, and apply it to each crystal.
This is the simplest closed-form SC scheme that captures orientation
asymmetry without requiring a non-trivial Eshelby calculation. It is
exact only for a statistically-isotropic polycrystal but is widely used
as a leading-order approximation for textured aggregates.

No fitting parameters. Inputs: orientations, single-crystal stiffness,
macroscopic stress. Outputs: per-grain σ, ε, and U.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..stress.voigt import (
    strain_tensor_to_voigt,
    stress_tensor_to_voigt,
    voigt_to_strain_tensor,
    voigt_to_stress_tensor,
)
from ..stress.cubic_anisotropic import cubic_stiffness_voigt


# ----- Bond / Voigt rotation matrices for stiffness ---------------------------

def _bond_rotation_matrices(OM: NDArray[np.floating]) -> NDArray[np.floating]:
    """Per-grain 6x6 Bond rotation matrix M such that
        C_lab = M @ C_xtal @ M^T  (Voigt notation, engineering shear convention)
    Reference: Auld, "Acoustic fields and waves in solids" Vol. 1, eq. (3.32).
    """
    OM = np.asarray(OM, dtype=float)
    if OM.ndim != 3 or OM.shape[-2:] != (3, 3):
        raise ValueError(f"OM must be (n,3,3); got {OM.shape}")
    n = OM.shape[0]
    M = np.zeros((n, 6, 6), dtype=float)
    a = OM
    # Upper-left 3x3 block: a[i,j]^2
    for i in range(3):
        for j in range(3):
            M[:, i, j] = a[:, i, j] ** 2
    # Upper-right 3x3 block: 2 a[i,j+1] a[i,j+2]    (j+1, j+2 mod 3)
    for i in range(3):
        M[:, i, 3] = 2 * a[:, i, 1] * a[:, i, 2]
        M[:, i, 4] = 2 * a[:, i, 0] * a[:, i, 2]
        M[:, i, 5] = 2 * a[:, i, 0] * a[:, i, 1]
    # Lower-left 3x3 block
    for j in range(3):
        M[:, 3, j] = a[:, 1, j] * a[:, 2, j]
        M[:, 4, j] = a[:, 0, j] * a[:, 2, j]
        M[:, 5, j] = a[:, 0, j] * a[:, 1, j]
    # Lower-right 3x3 block
    for I, (i1, i2) in enumerate([(1, 2), (0, 2), (0, 1)]):
        for J, (j1, j2) in enumerate([(1, 2), (0, 2), (0, 1)]):
            M[:, 3 + I, 3 + J] = (
                a[:, i1, j1] * a[:, i2, j2] + a[:, i1, j2] * a[:, i2, j1]
            )
    return M


def per_grain_lab_stiffness(
    OM: NDArray[np.floating],
    c11: float,
    c12: float,
    c44: float,
) -> NDArray[np.floating]:
    """Per-grain stiffness in lab frame, Voigt 6x6 form. Units: same as c11.

    Convention: OM rotates crystal → sample (the MIDAS Grains.csv convention).
    Returns C_lab such that σ_voigt_lab = C_lab @ ε_voigt_lab with the
    engineering-shear strain convention (γ_ij = 2 ε_ij for i ≠ j).
    """
    C_xtal = cubic_stiffness_voigt(c11, c12, c44)
    M = _bond_rotation_matrices(OM)             # (n, 6, 6)
    # C_lab_g = M_g @ C_xtal @ M_g^T
    return np.einsum("gij,jk,glk->gil", M, C_xtal, M)


# ----- Isotropic averages (Voigt, Reuss, Hill) --------------------------------

def _voigt_average_isotropic(C_voigt: NDArray[np.floating]) -> tuple[float, float]:
    """Voigt isotropic average from a single cubic stiffness in Voigt notation."""
    c11, c12, c44 = C_voigt[0, 0], C_voigt[0, 1], C_voigt[3, 3]
    K = (c11 + 2 * c12) / 3
    mu = (c11 - c12 + 3 * c44) / 5
    return K, mu


def _reuss_average_isotropic(C_voigt: NDArray[np.floating]) -> tuple[float, float]:
    """Reuss isotropic average from a single cubic stiffness."""
    c11, c12, c44 = C_voigt[0, 0], C_voigt[0, 1], C_voigt[3, 3]
    # Compliances: s11 = (c11+c12)/[(c11-c12)(c11+2c12)],
    #              s12 = -c12/[(c11-c12)(c11+2c12)], s44 = 1/c44.
    denom = (c11 - c12) * (c11 + 2 * c12)
    s11 = (c11 + c12) / denom
    s12 = -c12 / denom
    s44 = 1.0 / c44
    K = 1.0 / (3 * (s11 + 2 * s12))
    mu = 5.0 / (4 * (s11 - s12) + 3 * s44)
    return K, mu


def hill_average_isotropic(c11: float, c12: float, c44: float) -> tuple[float, float]:
    """Voigt-Reuss-Hill polycrystalline (K, μ) for a cubic single crystal."""
    C = cubic_stiffness_voigt(c11, c12, c44)
    Kv, mv = _voigt_average_isotropic(C)
    Kr, mr = _reuss_average_isotropic(C)
    return 0.5 * (Kv + Kr), 0.5 * (mv + mr)


def _isotropic_voigt_stiffness(K: float, mu: float) -> NDArray[np.floating]:
    """6x6 isotropic stiffness from (K, μ), Voigt notation (engineering shear)."""
    lam = K - (2.0 / 3.0) * mu
    C = np.zeros((6, 6), dtype=float)
    for i in range(3):
        for j in range(3):
            C[i, j] = lam if i != j else lam + 2 * mu
    for i in range(3, 6):
        C[i, i] = mu
    return C


# ----- Eshelby sphere in isotropic medium ------------------------------------

def _eshelby_tensor_sphere_isotropic(nu: float) -> NDArray[np.floating]:
    """Eshelby tensor S for a spherical inclusion in isotropic medium of
    Poisson ratio ν. 6x6 Voigt form with engineering-shear convention.
    Reference: Mura, Micromechanics of defects in solids, eq. (11.21).
    """
    S = np.zeros((6, 6), dtype=float)
    a = (7 - 5 * nu) / (15 * (1 - nu))
    b = (5 * nu - 1) / (15 * (1 - nu))
    c = (4 - 5 * nu) / (15 * (1 - nu))
    # S_iijj diagonal: a; S_iijj off-diag (i≠j): b
    for i in range(3):
        for j in range(3):
            S[i, j] = a if i == j else b
    for i in range(3, 6):
        S[i, i] = c
    return S


# ----- Reuss / Voigt / SC predictions ----------------------------------------

def reuss_per_grain(
    OM: NDArray[np.floating],
    sigma_macro: NDArray[np.floating],
    c11: float,
    c12: float,
    c44: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Reuss-bound per-grain (σ_g, ε_g) in the lab frame.

    Reuss = every grain sees the macroscopic stress; strain follows from each
    grain's compliance.

    Parameters
    ----------
    OM : (n, 3, 3)
        Per-grain orientation matrices (crystal → sample).
    sigma_macro : (3, 3)
        Applied macroscopic stress in the lab frame. Units same as c11.
    c11, c12, c44 : float
        Cubic single-crystal stiffness constants.

    Returns
    -------
    sigma : (n, 3, 3) — equal to sigma_macro broadcast over grains.
    eps   : (n, 3, 3) — per-grain elastic strain.
    """
    OM = np.asarray(OM, dtype=float)
    sigma_macro = np.asarray(sigma_macro, dtype=float)
    n = OM.shape[0]

    C_lab = per_grain_lab_stiffness(OM, c11, c12, c44)        # (n, 6, 6)
    S_lab = np.linalg.inv(C_lab)                              # (n, 6, 6)

    sigma = np.broadcast_to(sigma_macro, (n, 3, 3)).copy()
    sig_v = stress_tensor_to_voigt(sigma)                     # (n, 6)
    eps_v = np.einsum("gij,gj->gi", S_lab, sig_v)             # (n, 6)
    eps = voigt_to_strain_tensor(eps_v)
    return sigma, eps


def voigt_per_grain(
    OM: NDArray[np.floating],
    eps_macro: NDArray[np.floating],
    c11: float,
    c12: float,
    c44: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Voigt-bound per-grain (σ_g, ε_g) under uniform macroscopic strain."""
    OM = np.asarray(OM, dtype=float)
    eps_macro = np.asarray(eps_macro, dtype=float)
    n = OM.shape[0]

    C_lab = per_grain_lab_stiffness(OM, c11, c12, c44)
    eps = np.broadcast_to(eps_macro, (n, 3, 3)).copy()
    eps_v = strain_tensor_to_voigt(eps)
    sig_v = np.einsum("gij,gj->gi", C_lab, eps_v)
    sigma = voigt_to_stress_tensor(sig_v)
    return sigma, eps


def kroner_self_consistent(
    OM: NDArray[np.floating],
    sigma_macro: NDArray[np.floating],
    c11: float,
    c12: float,
    c44: float,
    *,
    weights: NDArray[np.floating] | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict:
    """Kröner-Eshelby self-consistent elastic per-grain prediction.

    For each grain (treated as a spherical inclusion in the effective
    medium):
        σ_g = σ_macro + (I - C* · S_E · C*^{-1}) · (C_g - C*) · ε_g
    iterated on C* until convergence.

    The effective medium is approximated as isotropic; (K*, μ*) are
    refreshed at each iteration from the volume-averaged stiffness.

    Parameters
    ----------
    OM : (n, 3, 3)
        Per-grain orientation matrices (crystal → sample).
    sigma_macro : (3, 3)
        Applied macroscopic stress (lab frame). Same units as c11.
    c11, c12, c44 : float
        Cubic single-crystal stiffness constants.
    weights : (n,), optional
        Per-grain volume weights. Default: uniform.
    max_iter, tol :
        Iteration limits / convergence on relative change in C*.

    Returns
    -------
    dict with keys
        sigma  : (n, 3, 3) per-grain stress (lab frame)
        eps    : (n, 3, 3) per-grain strain (lab frame)
        C_eff  : (6, 6) effective medium stiffness
        K_eff  : float effective bulk modulus
        mu_eff : float effective shear modulus
        n_iter : int converged iterations
        residual : float final relative change
    """
    OM = np.asarray(OM, dtype=float)
    sigma_macro = np.asarray(sigma_macro, dtype=float)
    n = OM.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=float)
    weights = np.asarray(weights, dtype=float)
    w = weights / weights.sum()

    C_lab = per_grain_lab_stiffness(OM, c11, c12, c44)      # (n, 6, 6)

    # Initial guess for effective C*: Voigt average
    C_eff = np.einsum("g,gij->ij", w, C_lab)
    K_eff, mu_eff = _voigt_average_isotropic(C_eff)
    sig_v_macro = stress_tensor_to_voigt(sigma_macro[None])[0]   # (6,)

    residual = np.inf
    for it in range(max_iter):
        nu_eff = (3 * K_eff - 2 * mu_eff) / (2 * (3 * K_eff + mu_eff))
        S_E = _eshelby_tensor_sphere_isotropic(nu_eff)            # (6,6)

        C_eff_iso = _isotropic_voigt_stiffness(K_eff, mu_eff)
        C_eff_iso_inv = np.linalg.inv(C_eff_iso)

        # Hill self-consistent: ε_g = (I + S_E · C_eff_iso^{-1} · (C_g - C_eff_iso))^{-1} · ε_macro
        ID6 = np.eye(6)
        dC = C_lab - C_eff_iso                                     # (n, 6, 6)
        T = ID6[None] + np.einsum("ij,jk,gkl->gil", S_E, C_eff_iso_inv, dC)
        eps_macro_v = C_eff_iso_inv @ sig_v_macro                  # (6,)
        # Batched per-grain strain concentration: ε_g = T_g^{-1} ε_macro
        A_g = np.linalg.inv(T)                                     # (n, 6, 6)
        eps_v = np.einsum("gij,j->gi", A_g, eps_macro_v)
        sig_v = np.einsum("gij,gj->gi", C_lab, eps_v)

        # Self-consistent update: C_eff = ⟨C_g · A_g⟩
        C_eff_new = np.einsum("g,gij,gjk->ik", w, C_lab, A_g)
        # Symmetrize
        C_eff_new = 0.5 * (C_eff_new + C_eff_new.T)
        K_new, mu_new = _voigt_average_isotropic(C_eff_new)

        residual = max(abs(K_new - K_eff) / K_eff, abs(mu_new - mu_eff) / mu_eff)
        C_eff = C_eff_new
        K_eff, mu_eff = K_new, mu_new
        if residual < tol:
            break

    sigma = voigt_to_stress_tensor(sig_v)
    eps = voigt_to_strain_tensor(eps_v)
    return dict(
        sigma=sigma, eps=eps, C_eff=C_eff,
        K_eff=K_eff, mu_eff=mu_eff,
        n_iter=it + 1, residual=residual,
    )


# ----- Per-grain elastic energy ----------------------------------------------

def per_grain_energy(sigma: NDArray[np.floating],
                     eps: NDArray[np.floating]) -> NDArray[np.floating]:
    """Elastic energy density U = (1/2) σ : ε per grain. Same units as σ·ε."""
    return 0.5 * np.einsum("gij,gij->g", sigma, eps)


__all__ = [
    "per_grain_lab_stiffness",
    "hill_average_isotropic",
    "reuss_per_grain",
    "voigt_per_grain",
    "kroner_self_consistent",
    "per_grain_energy",
]
