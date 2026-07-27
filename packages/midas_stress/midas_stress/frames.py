"""Coordinate frame conversions between MIDAS, APS, and sample frames.

MIDAS (ESRF) frame:
    X = along the X-ray beam
    Y = outboard (OB)
    Z = up

APS frame (Park convention):
    X = outboard (OB)
    Y = up
    Z = along the X-ray beam

The two frames are related by a cyclic permutation of axes:
    (X_APS, Y_APS, Z_APS) = (Y_MIDAS, Z_MIDAS, X_MIDAS)

Sample frame:
    The sample frame is the lab frame rotated by the omega angle about
    the rotation axis. When omega = 0, the sample frame coincides with
    the lab frame.

Reference: Park, J.-S., matlab_tools/hedm (2024),
https://github.com/junspark/matlab_tools

As of 0.6.0, all functions accept torch.Tensor inputs transparently and
return torch tensors when given torch input — same dispatch pattern as
`orientation.py`. Existing NumPy callers see no API change.
"""


from __future__ import annotations
import math

import numpy as np
from ._optional import torch


# -------------------------------------------------------------------
#  Rotation matrices between frames
# -------------------------------------------------------------------

#: 3x3 rotation matrix converting MIDAS (ESRF) coordinates to APS coordinates.
#: v_APS = R_MIDAS_TO_APS @ v_MIDAS
R_MIDAS_TO_APS = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)

#: Inverse: APS -> MIDAS. Since R is orthogonal, R^{-1} = R^T.
R_APS_TO_MIDAS = R_MIDAS_TO_APS.T.copy()


# -------------------------------------------------------------------
#  Backend dispatch helpers (torch / numpy)
# -------------------------------------------------------------------

def _is_torch(*args) -> bool:
    return any(isinstance(a, torch.Tensor) for a in args)


def _r_midas_to_aps(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(R_MIDAS_TO_APS, dtype=dtype, device=device)


def _r_aps_to_midas(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(R_APS_TO_MIDAS, dtype=dtype, device=device)


def lab_to_sample_rotation(omega_deg, frame: str = "midas"):
    """Build the lab-to-sample rotation matrix for a given omega angle.

    When omega = 0, the lab and sample frames coincide.

    Parameters
    ----------
    omega_deg : float or torch.Tensor (0-d)
        Omega angle in degrees.
    frame : str
        ``"midas"`` or ``"aps"`` — which lab frame convention to use.
        In MIDAS, the rotation axis is Z (up).
        In APS, the rotation axis is Y (up).

    Returns
    -------
    (3, 3) ndarray (NumPy backend) or torch.Tensor (torch backend).
    """
    if _is_torch(omega_deg):
        return _lab_to_sample_rotation_torch(omega_deg, frame)
    c = math.cos(math.radians(omega_deg))
    s = math.sin(math.radians(omega_deg))

    if frame.lower() == "aps":
        # Rotation about Y (up in APS)
        return np.array([
            [ c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [ s, 0.0,  c],
        ], dtype=np.float64)
    elif frame.lower() in ("midas", "esrf"):
        # Rotation about Z (up in MIDAS)
        return np.array([
            [ c,  s, 0.0],
            [-s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown frame '{frame}'. Use 'midas' or 'aps'.")


def _lab_to_sample_rotation_torch(omega_deg, frame: str) -> torch.Tensor:
    """Torch path for lab_to_sample_rotation."""
    omega = omega_deg if isinstance(omega_deg, torch.Tensor) else torch.tensor(omega_deg)
    omega_rad = omega * (math.pi / 180.0)
    c = torch.cos(omega_rad)
    s = torch.sin(omega_rad)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    if frame.lower() == "aps":
        # Rotation about Y (up in APS)
        return torch.stack([
            torch.stack([c, zero, -s], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([s, zero, c], dim=-1),
        ], dim=-2)
    if frame.lower() in ("midas", "esrf"):
        # Rotation about Z (up in MIDAS)
        return torch.stack([
            torch.stack([c, s, zero], dim=-1),
            torch.stack([-s, c, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ], dim=-2)
    raise ValueError(f"Unknown frame '{frame}'. Use 'midas' or 'aps'.")


# -------------------------------------------------------------------
#  Convert vectors (positions, etc.)
# -------------------------------------------------------------------

def vector_midas_to_aps(v):
    """Convert vector(s) from MIDAS to APS frame.

    Parameters
    ----------
    v : ndarray or torch.Tensor (..., 3)

    Returns
    -------
    Same backend as input, shape (..., 3).
    """
    if _is_torch(v):
        R = _r_midas_to_aps(v.dtype, v.device)
        return (R @ v.unsqueeze(-1)).squeeze(-1)
    return (R_MIDAS_TO_APS @ v[..., None]).squeeze(-1)


def vector_aps_to_midas(v):
    """Convert vector(s) from APS to MIDAS frame.

    Parameters
    ----------
    v : ndarray or torch.Tensor (..., 3)

    Returns
    -------
    Same backend as input, shape (..., 3).
    """
    if _is_torch(v):
        R = _r_aps_to_midas(v.dtype, v.device)
        return (R @ v.unsqueeze(-1)).squeeze(-1)
    return (R_APS_TO_MIDAS @ v[..., None]).squeeze(-1)


# -------------------------------------------------------------------
#  Convert orientation matrices
# -------------------------------------------------------------------

def orient_midas_to_aps(U):
    """Convert orientation matrix from MIDAS to APS frame.

    If U_midas takes crystal -> MIDAS lab, then
    U_aps = R_MIDAS_TO_APS @ U_midas takes crystal -> APS lab.

    Parameters
    ----------
    U : ndarray or torch.Tensor (..., 3, 3)
    """
    if _is_torch(U):
        return _r_midas_to_aps(U.dtype, U.device) @ U
    return R_MIDAS_TO_APS @ U


def orient_aps_to_midas(U):
    """Convert orientation matrix from APS to MIDAS frame.

    Parameters
    ----------
    U : ndarray or torch.Tensor (..., 3, 3)
    """
    if _is_torch(U):
        return _r_aps_to_midas(U.dtype, U.device) @ U
    return R_APS_TO_MIDAS @ U


# -------------------------------------------------------------------
#  Convert symmetric tensors (strain, stress)
# -------------------------------------------------------------------

def tensor_midas_to_aps(T):
    """Convert symmetric 3x3 tensor(s) from MIDAS to APS frame.

    Applies similarity transform: T_aps = R @ T_midas @ R^T.

    Parameters
    ----------
    T : ndarray or torch.Tensor (..., 3, 3)
    """
    if _is_torch(T):
        R = _r_midas_to_aps(T.dtype, T.device)
        return R @ T @ R.transpose(-1, -2)
    return R_MIDAS_TO_APS @ T @ R_MIDAS_TO_APS.T


def tensor_aps_to_midas(T):
    """Convert symmetric 3x3 tensor(s) from APS to MIDAS frame.

    Parameters
    ----------
    T : ndarray or torch.Tensor (..., 3, 3)
    """
    if _is_torch(T):
        R = _r_aps_to_midas(T.dtype, T.device)
        return R @ T @ R.transpose(-1, -2)
    return R_APS_TO_MIDAS @ T @ R_APS_TO_MIDAS.T


def tensor_lab_to_sample(T, omega_deg, frame: str = "midas"):
    """Convert symmetric tensor(s) from lab to sample frame.

    Parameters
    ----------
    T : ndarray or torch.Tensor (..., 3, 3)
    omega_deg : float or torch.Tensor (0-d)
    frame : str — ``"midas"`` or ``"aps"``.
    """
    R = lab_to_sample_rotation(omega_deg, frame)
    if _is_torch(T, omega_deg):
        if not isinstance(R, torch.Tensor):
            R = torch.as_tensor(R, dtype=T.dtype, device=T.device)
        return R @ T @ R.transpose(-1, -2)
    return R @ T @ R.T


# -------------------------------------------------------------------
#  Full conversion pipeline (MIDAS Grains.csv -> sample frame)
# -------------------------------------------------------------------

def grains_midas_to_sample(
    orientations,
    positions,
    strains,
    omega_deg: float = 0.0,
    target_frame: str = "aps",
) -> dict:
    """Convert MIDAS Grains.csv data to the APS sample frame.

    This replicates the pipeline in Park's ``parseGrainData_OneLayer_ff.m``:
    first apply the MIDAS->APS cyclic permutation, then the lab->sample
    rotation at the given omega.

    Parameters
    ----------
    orientations : ndarray (N, 3, 3)
        Orientation matrices from MIDAS (crystal -> MIDAS lab).
    positions : ndarray (N, 3)
        Grain center-of-mass positions in MIDAS frame (micrometers).
    strains : ndarray (N, 3, 3)
        Strain tensors in MIDAS lab frame.
    omega_deg : float
        Omega angle at which lab and sample coincide (default 0).
    target_frame : str
        ``"aps"`` (default) or ``"midas"``.

    Returns
    -------
    dict with keys:
        'orientations': ndarray (N, 3, 3) in sample frame
        'positions': ndarray (N, 3) in sample frame
        'strains': ndarray (N, 3, 3) in sample frame
    """
    is_torch = _is_torch(orientations, positions, strains, omega_deg)
    if is_torch:
        ref = orientations if isinstance(orientations, torch.Tensor) else (
            positions if isinstance(positions, torch.Tensor) else strains
        )
        if target_frame.lower() == "aps":
            R_frame = _r_midas_to_aps(ref.dtype, ref.device)
        elif target_frame.lower() in ("midas", "esrf"):
            R_frame = torch.eye(3, dtype=ref.dtype, device=ref.device)
        else:
            raise ValueError(f"Unknown target_frame '{target_frame}'.")
        R_lab2sam = lab_to_sample_rotation(omega_deg, target_frame)
        if not isinstance(R_lab2sam, torch.Tensor):
            R_lab2sam = torch.as_tensor(R_lab2sam, dtype=ref.dtype, device=ref.device)
        R_total = R_lab2sam @ R_frame

        orient_out = R_total @ orientations
        pos_out = (R_total @ positions.unsqueeze(-1)).squeeze(-1)
        strain_out = R_total @ strains @ R_total.transpose(-1, -2)
    else:
        if target_frame.lower() == "aps":
            R_frame = R_MIDAS_TO_APS.copy()
        elif target_frame.lower() in ("midas", "esrf"):
            R_frame = np.eye(3)
        else:
            raise ValueError(f"Unknown target_frame '{target_frame}'.")
        R_lab2sam = lab_to_sample_rotation(omega_deg, target_frame)
        R_total = R_lab2sam @ R_frame
        orient_out = R_total @ orientations
        pos_out = (R_total @ positions[..., None]).squeeze(-1)
        strain_out = R_total @ strains @ R_total.T

    return {
        'orientations': orient_out,
        'positions': pos_out,
        'strains': strain_out,
    }
