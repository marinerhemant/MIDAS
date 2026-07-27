"""Voigt-Mandel tensor conversions, lattice-parameter strain, and frame transforms.

All computations use Voigt-Mandel notation with sqrt(2) factors
on shear components, following the convention in MIDAS Paper I Appendix A.

Mandel ordering: [xx, yy, zz, sqrt(2)*xy, sqrt(2)*xz, sqrt(2)*yz].

As of 0.6.0, all functions accept torch.Tensor inputs transparently and
return torch tensors when given torch input — same dispatch pattern as
`orientation.py` / `frames.py`. NumPy callers see no API change.
"""


from __future__ import annotations
import math

import numpy as np
from ._optional import torch


_SQRT2 = math.sqrt(2.0)
_SQRT2_INV = 1.0 / _SQRT2


def _is_torch(*args) -> bool:
    return any(isinstance(a, torch.Tensor) for a in args)


# -------------------------------------------------------------------
#  Voigt notation conversions
# -------------------------------------------------------------------

def tensor_to_voigt(T):
    """Convert symmetric 3x3 tensor(s) to 6-vector Voigt-Mandel notation.

    Mandel convention: shear components scaled by sqrt(2) so that
    ||T||_F == ||v||_2 (Frobenius norm preserved).

    Parameters
    ----------
    T : ndarray or torch.Tensor (..., 3, 3)

    Returns
    -------
    Same backend as input, shape (..., 6).
    """
    if _is_torch(T):
        return torch.stack([
            T[..., 0, 0], T[..., 1, 1], T[..., 2, 2],
            _SQRT2 * T[..., 0, 1], _SQRT2 * T[..., 0, 2], _SQRT2 * T[..., 1, 2],
        ], dim=-1)
    return np.stack([
        T[..., 0, 0], T[..., 1, 1], T[..., 2, 2],
        _SQRT2 * T[..., 0, 1], _SQRT2 * T[..., 0, 2], _SQRT2 * T[..., 1, 2],
    ], axis=-1)


def voigt_to_tensor(v):
    """Convert 6-vector Voigt-Mandel notation to symmetric 3x3 tensor(s).

    Parameters
    ----------
    v : ndarray or torch.Tensor (..., 6)

    Returns
    -------
    Same backend as input, shape (..., 3, 3).
    """
    if _is_torch(v):
        xx, yy, zz = v[..., 0], v[..., 1], v[..., 2]
        xy = v[..., 3] * _SQRT2_INV
        xz = v[..., 4] * _SQRT2_INV
        yz = v[..., 5] * _SQRT2_INV
        row0 = torch.stack([xx, xy, xz], dim=-1)
        row1 = torch.stack([xy, yy, yz], dim=-1)
        row2 = torch.stack([xz, yz, zz], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)
    T = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
    T[..., 0, 0] = v[..., 0]
    T[..., 1, 1] = v[..., 1]
    T[..., 2, 2] = v[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = v[..., 3] * _SQRT2_INV
    T[..., 0, 2] = T[..., 2, 0] = v[..., 4] * _SQRT2_INV
    T[..., 1, 2] = T[..., 2, 1] = v[..., 5] * _SQRT2_INV
    return T


def tensor_to_voigt_engineering(T):
    """Convert to engineering Voigt notation (no sqrt(2) factor).

    Returns [T_xx, T_yy, T_zz, 2*T_xy, 2*T_xz, 2*T_yz].
    """
    if _is_torch(T):
        return torch.stack([
            T[..., 0, 0], T[..., 1, 1], T[..., 2, 2],
            2.0 * T[..., 0, 1], 2.0 * T[..., 0, 2], 2.0 * T[..., 1, 2],
        ], dim=-1)
    return np.stack([
        T[..., 0, 0], T[..., 1, 1], T[..., 2, 2],
        2.0 * T[..., 0, 1], 2.0 * T[..., 0, 2], 2.0 * T[..., 1, 2],
    ], axis=-1)


# -------------------------------------------------------------------
#  A-matrix: lattice parameters to orthonormal basis
# -------------------------------------------------------------------

def lattice_params_to_A_matrix(latc):
    """Build the A matrix (Paper I Eq. 6) from lattice parameters.

    Maps fractional crystal coordinates to Cartesian coordinates.

    Parameters
    ----------
    latc : ndarray or torch.Tensor (..., 6)
        [a, b, c, alpha_deg, beta_deg, gamma_deg]
    """
    if _is_torch(latc):
        return _lattice_params_to_A_matrix_torch(latc)
    d2r = math.pi / 180.0
    a = latc[..., 0]
    b = latc[..., 1]
    c = latc[..., 2]
    alpha = latc[..., 3] * d2r
    beta = latc[..., 4] * d2r
    gamma = latc[..., 5] * d2r

    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    cos_g = np.cos(gamma)

    cos_g_star = (cos_a * cos_b - cos_g) / (sin_a * sin_b)
    sin_g_star = np.sqrt(np.clip(1.0 - cos_g_star**2, 0, None))

    A = np.zeros(latc.shape[:-1] + (3, 3), dtype=np.float64)
    A[..., 0, 0] = a * sin_b * sin_g_star
    A[..., 1, 0] = -a * sin_b * cos_g_star
    A[..., 1, 1] = b * sin_a
    A[..., 2, 0] = a * cos_b
    A[..., 2, 1] = b * cos_a
    A[..., 2, 2] = c
    return A


def _lattice_params_to_A_matrix_torch(latc: torch.Tensor) -> torch.Tensor:
    d2r = math.pi / 180.0
    a = latc[..., 0]
    b = latc[..., 1]
    c = latc[..., 2]
    alpha = latc[..., 3] * d2r
    beta = latc[..., 4] * d2r
    gamma = latc[..., 5] * d2r
    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
    cos_b, sin_b = torch.cos(beta), torch.sin(beta)
    cos_g = torch.cos(gamma)

    cos_g_star = (cos_a * cos_b - cos_g) / (sin_a * sin_b)
    sin_g_star = torch.sqrt((1.0 - cos_g_star ** 2).clamp_min(0.0))

    zero = torch.zeros_like(a)
    row0 = torch.stack([a * sin_b * sin_g_star, zero, zero], dim=-1)
    row1 = torch.stack([-a * sin_b * cos_g_star, b * sin_a, zero], dim=-1)
    row2 = torch.stack([a * cos_b, b * cos_a, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


# -------------------------------------------------------------------
#  Strain tensor from lattice parameters (lattice-parameter / deformation-gradient method)
# -------------------------------------------------------------------

def lattice_params_to_strain(latc_strained, latc_unstrained):
    """Compute Green-Lagrange strain tensor in grain frame.

    Matches CalcStrainTensorFableBeaudoin in CalcStrains.c, which
    implements the lattice-parameter (deformation-gradient) form.

    Parameters
    ----------
    latc_strained : ndarray or torch.Tensor (..., 6)
    latc_unstrained : ndarray or torch.Tensor (..., 6) or (6,)
    """
    if _is_torch(latc_strained, latc_unstrained):
        A = lattice_params_to_A_matrix(latc_strained)
        A0 = lattice_params_to_A_matrix(latc_unstrained)
        A0_inv = torch.linalg.inv(A0)
        F = A @ A0_inv
        I = torch.eye(3, dtype=F.dtype, device=F.device)
        return 0.5 * (F + F.transpose(-1, -2)) - I
    A = lattice_params_to_A_matrix(latc_strained)
    A0 = lattice_params_to_A_matrix(latc_unstrained)
    A0_inv = np.linalg.inv(A0)
    F = A @ A0_inv
    I = np.eye(3)
    return 0.5 * (F + np.swapaxes(F, -1, -2)) - I


# -------------------------------------------------------------------
#  Coordinate frame transformations
# -------------------------------------------------------------------

def strain_grain_to_lab(strain_grain, orient):
    """Transform strain tensor from grain to lab frame.

    Paper I Eq. 4: epsilon_lab = U * epsilon_gr * U^T
    """
    if _is_torch(strain_grain, orient):
        return orient @ strain_grain @ orient.transpose(-1, -2)
    return orient @ strain_grain @ np.swapaxes(orient, -1, -2)


def strain_lab_to_grain(strain_lab, orient):
    """Transform strain tensor from lab to grain frame.

    epsilon_gr = U^T * epsilon_lab * U
    """
    if _is_torch(strain_lab, orient):
        return orient.transpose(-1, -2) @ strain_lab @ orient
    return np.swapaxes(orient, -1, -2) @ strain_lab @ orient


# -------------------------------------------------------------------
#  6x6 rotation matrix in Voigt space (Paper I Eq. 14)
# -------------------------------------------------------------------

def rotation_voigt_mandel(U):
    """Build the 6x6 Mandel rotation matrix (lab -> grain).

    Transforms vectorized symmetric tensors from the lab frame
    into the crystal (grain) frame:
        {epsilon_grain} = M @ {epsilon_lab}

    The lab-frame stiffness follows from Paper I Eq. stress_calc2:
        {sigma_lab} = M^T @ C_grain @ M @ {epsilon_lab}

    Mandel ordering: [xx, yy, zz, sqrt(2)*xy, sqrt(2)*xz, sqrt(2)*yz].

    Parameters
    ----------
    U : ndarray or torch.Tensor (..., 3, 3) — rotation/orientation matrix
    """
    if _is_torch(U):
        M = torch.zeros(*U.shape[:-2], 6, 6, dtype=U.dtype, device=U.device)
        for i in range(3):
            for j in range(3):
                M[..., i, j] = U[..., i, j] ** 2
        pairs = [(0, 1), (0, 2), (1, 2)]
        for col_idx, (p, q) in enumerate(pairs):
            for row in range(3):
                M[..., row, 3 + col_idx] = _SQRT2 * U[..., row, p] * U[..., row, q]
        for row_idx, (p, q) in enumerate(pairs):
            for col in range(3):
                M[..., 3 + row_idx, col] = _SQRT2 * U[..., p, col] * U[..., q, col]
        for row_idx, (r1, r2) in enumerate(pairs):
            for col_idx, (c1, c2) in enumerate(pairs):
                M[..., 3 + row_idx, 3 + col_idx] = (
                    U[..., r1, c1] * U[..., r2, c2]
                    + U[..., r1, c2] * U[..., r2, c1]
                )
        return M.transpose(-1, -2)
    M = np.zeros(U.shape[:-2] + (6, 6), dtype=U.dtype)
    for i in range(3):
        for j in range(3):
            M[..., i, j] = U[..., i, j] ** 2
    pairs = [(0, 1), (0, 2), (1, 2)]
    for col_idx, (p, q) in enumerate(pairs):
        for row in range(3):
            M[..., row, 3 + col_idx] = _SQRT2 * U[..., row, p] * U[..., row, q]
    for row_idx, (p, q) in enumerate(pairs):
        for col in range(3):
            M[..., 3 + row_idx, col] = _SQRT2 * U[..., p, col] * U[..., q, col]
    for row_idx, (r1, r2) in enumerate(pairs):
        for col_idx, (c1, c2) in enumerate(pairs):
            M[..., 3 + row_idx, 3 + col_idx] = (
                U[..., r1, c1] * U[..., r2, c2]
                + U[..., r1, c2] * U[..., r2, c1]
            )
    return np.swapaxes(M, -1, -2)


# -------------------------------------------------------------------
#  Scalar stress/strain invariants
# -------------------------------------------------------------------

def hydrostatic(T):
    """Hydrostatic (mean) component: tr(T)/3.

    Parameters
    ----------
    T : ndarray or torch.Tensor (..., 3, 3)
    """
    if _is_torch(T):
        return torch.diagonal(T, dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    return np.trace(T, axis1=-2, axis2=-1) / 3.0


def deviatoric(T):
    """Deviatoric component: T - (tr(T)/3) * I.

    Parameters
    ----------
    T : ndarray or torch.Tensor (..., 3, 3)
    """
    p = hydrostatic(T)
    if _is_torch(T):
        I = torch.eye(3, dtype=T.dtype, device=T.device)
        return T - p[..., None, None] * I
    return T - p[..., None, None] * np.eye(3)


def von_mises(T):
    """Von Mises equivalent stress (or strain).

    For stress: sigma_vm = sqrt(3/2 * s_ij * s_ij)
    where s is the deviatoric part.

    Parameters
    ----------
    T : ndarray or torch.Tensor (..., 3, 3)
    """
    s = deviatoric(T)
    if _is_torch(T):
        ss = (s * s).sum(dim=(-2, -1))
        return torch.sqrt(1.5 * ss)
    ss = np.sum(s * s, axis=(-2, -1))
    return np.sqrt(1.5 * ss)
