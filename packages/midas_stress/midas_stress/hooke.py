"""Hooke's law: compute stress from strain using single-crystal stiffness.

Supports grain-frame and lab-frame computations using the 6x6 Mandel
rotation matrix from Paper I Eq. 14.
"""

import numpy as np

from .tensor import tensor_to_voigt, voigt_to_tensor, rotation_voigt_mandel


def hooke_stress(
    strain: np.ndarray,
    stiffness: np.ndarray,
    orient: np.ndarray = None,
    frame: str = "lab",
) -> np.ndarray:
    """Compute stress from strain using Hooke's law.

    Parameters
    ----------
    strain : ndarray (..., 3, 3) or (..., 6)
        Strain tensor. If 3x3, converted to Voigt-Mandel internally.
    stiffness : ndarray (6, 6)
        Single-crystal stiffness matrix C in Voigt-Mandel notation,
        defined in the crystal frame.
    orient : ndarray (..., 3, 3), optional
        Orientation matrix. Required if ``frame`` is ``"lab"``.
    frame : str
        ``"grain"``: strain is in grain frame, return stress in grain frame.
        ``"lab"``: strain is in lab frame, transform to grain, apply C,
        transform back.

    Returns
    -------
    ndarray (..., 3, 3) stress tensor in the requested frame.
    """
    if strain.shape[-1] == 3 and strain.shape[-2] == 3:
        eps_voigt = tensor_to_voigt(strain)
    else:
        eps_voigt = strain

    if frame == "grain":
        sig_voigt = eps_voigt @ stiffness.T
        return voigt_to_tensor(sig_voigt)

    if orient is None:
        raise ValueError("orient required for lab-frame computation")

    # Paper I: {sigma_lab} = M^T @ C_grain @ M @ {epsilon_lab}
    M = rotation_voigt_mandel(orient)       # lab -> grain
    Mt = np.swapaxes(M, -1, -2)            # grain -> lab
    C_lab = Mt @ stiffness @ M
    sig_voigt = (C_lab @ eps_voigt[..., None]).squeeze(-1)
    return voigt_to_tensor(sig_voigt)
