"""PyTorch backend for differentiable stress-strain analysis.

Mirrors the NumPy API in tensor.py, hooke.py, and equilibrium.py but
uses PyTorch tensors throughout, enabling gradient-based optimization
of crystallographic parameters.

Voigt-Mandel ordering: [xx, yy, zz, sqrt(2)*xy, sqrt(2)*xz, sqrt(2)*yz]
(consistent with the NumPy modules in this package and MIDAS Paper I).

Requires: ``pip install midas-stress[torch]``
"""

import math
from typing import Optional

import torch
import torch.nn as nn

_SQRT2 = math.sqrt(2.0)
_SQRT2_INV = 1.0 / _SQRT2


# -------------------------------------------------------------------
#  Voigt-Mandel conversions
# -------------------------------------------------------------------

def tensor_to_voigt(T: torch.Tensor) -> torch.Tensor:
    """3x3 symmetric tensor(s) to 6-vector Voigt-Mandel. Differentiable.

    Parameters
    ----------
    T : Tensor (..., 3, 3)

    Returns
    -------
    Tensor (..., 6) -- [xx, yy, zz, sqrt(2)*xy, sqrt(2)*xz, sqrt(2)*yz]
    """
    return torch.stack([
        T[..., 0, 0],
        T[..., 1, 1],
        T[..., 2, 2],
        _SQRT2 * T[..., 0, 1],
        _SQRT2 * T[..., 0, 2],
        _SQRT2 * T[..., 1, 2],
    ], dim=-1)


def voigt_to_tensor(v: torch.Tensor) -> torch.Tensor:
    """6-vector Voigt-Mandel to 3x3 symmetric tensor(s). Differentiable.

    Parameters
    ----------
    v : Tensor (..., 6)

    Returns
    -------
    Tensor (..., 3, 3)
    """
    xx, yy, zz = v[..., 0], v[..., 1], v[..., 2]
    xy = v[..., 3] * _SQRT2_INV
    xz = v[..., 4] * _SQRT2_INV
    yz = v[..., 5] * _SQRT2_INV
    row0 = torch.stack([xx, xy, xz], dim=-1)
    row1 = torch.stack([xy, yy, yz], dim=-1)
    row2 = torch.stack([xz, yz, zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


# -------------------------------------------------------------------
#  Stiffness
# -------------------------------------------------------------------

def cubic_stiffness(
    C11: float, C12: float, C44: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """6x6 stiffness for cubic crystal in Voigt-Mandel notation.

    Parameters
    ----------
    C11, C12, C44 : float
        Independent elastic constants in GPa.

    Returns
    -------
    Tensor (6, 6)
    """
    C = torch.zeros(6, 6, dtype=dtype, device=device)
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = C12
    C[3, 3] = C[4, 4] = C[5, 5] = 2.0 * C44
    return C


# -------------------------------------------------------------------
#  6x6 rotation in Voigt space
# -------------------------------------------------------------------

def rotation_voigt_mandel(U: torch.Tensor) -> torch.Tensor:
    """6x6 Mandel rotation matrix (lab -> grain). Differentiable.

    Ordering: [xx, yy, zz, sqrt(2)*xy, sqrt(2)*xz, sqrt(2)*yz].
    Shear pair indices: (0,1)=xy, (0,2)=xz, (1,2)=yz.

    Parameters
    ----------
    U : Tensor (..., 3, 3) rotation matrix

    Returns
    -------
    Tensor (..., 6, 6)
    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    M = torch.zeros(*U.shape[:-2], 6, 6, dtype=U.dtype, device=U.device)

    for i in range(3):
        for j in range(3):
            M[..., i, j] = U[..., i, j] ** 2

    for ci, (p, q) in enumerate(pairs):
        for r in range(3):
            M[..., r, 3 + ci] = _SQRT2 * U[..., r, p] * U[..., r, q]

    for ri, (p, q) in enumerate(pairs):
        for c in range(3):
            M[..., 3 + ri, c] = _SQRT2 * U[..., p, c] * U[..., q, c]

    for ri, (r1, r2) in enumerate(pairs):
        for ci, (c1, c2) in enumerate(pairs):
            M[..., 3 + ri, 3 + ci] = (
                U[..., r1, c1] * U[..., r2, c2]
                + U[..., r1, c2] * U[..., r2, c1]
            )

    return M.transpose(-1, -2)


# -------------------------------------------------------------------
#  Hooke's law
# -------------------------------------------------------------------

def hooke_stress(
    strain: torch.Tensor,
    stiffness: torch.Tensor,
    orient: Optional[torch.Tensor] = None,
    frame: str = "lab",
) -> torch.Tensor:
    """Differentiable Hooke's law: strain -> stress.

    Parameters
    ----------
    strain : Tensor (..., 3, 3) or (..., 6)
    stiffness : Tensor (6, 6)
        Single-crystal stiffness in Voigt-Mandel, crystal frame.
    orient : Tensor (..., 3, 3), optional
        Required for ``frame="lab"``.
    frame : str
        ``"grain"`` or ``"lab"``.

    Returns
    -------
    Tensor (..., 3, 3) stress tensor.
    """
    if strain.shape[-1] == 3 and strain.shape[-2] == 3:
        eps_v = tensor_to_voigt(strain)
    else:
        eps_v = strain

    if frame == "grain":
        sig_v = eps_v @ stiffness.T
        return voigt_to_tensor(sig_v)

    if orient is None:
        raise ValueError("orient required for lab-frame computation")

    M = rotation_voigt_mandel(orient)
    Mt = M.transpose(-1, -2)
    C_lab = Mt @ stiffness @ M
    sig_v = (C_lab @ eps_v.unsqueeze(-1)).squeeze(-1)
    return voigt_to_tensor(sig_v)


# -------------------------------------------------------------------
#  Equilibrium constraint
# -------------------------------------------------------------------

def volume_average_stress_constraint(
    stresses: torch.Tensor,
    volumes: torch.Tensor,
    applied_stress: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable volume-average stress constraint (FF-1).

    Parameters
    ----------
    stresses : Tensor (N, 3, 3)
    volumes : Tensor (N,)
    applied_stress : Tensor (3, 3), optional. Default: zero.

    Returns
    -------
    Tensor (N, 3, 3) corrected stresses.
    """
    if applied_stress is None:
        applied_stress = torch.zeros(3, 3, dtype=stresses.dtype,
                                     device=stresses.device)

    V_total = volumes.sum()
    w = volumes / V_total
    sig_avg = (w[:, None, None] * stresses).sum(dim=0)
    delta = applied_stress - sig_avg
    return stresses + delta.unsqueeze(0)
