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
"""

import math

import numpy as np


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


def lab_to_sample_rotation(omega_deg: float, frame: str = "midas") -> np.ndarray:
    """Build the lab-to-sample rotation matrix for a given omega angle.

    When omega = 0, the lab and sample frames coincide.

    Parameters
    ----------
    omega_deg : float
        Omega angle in degrees.
    frame : str
        ``"midas"`` or ``"aps"`` — which lab frame convention to use.
        In MIDAS, the rotation axis is Z (up).
        In APS, the rotation axis is Y (up).

    Returns
    -------
    ndarray (3, 3) — rotation matrix such that v_sample = R @ v_lab.
    """
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


# -------------------------------------------------------------------
#  Convert vectors (positions, etc.)
# -------------------------------------------------------------------

def vector_midas_to_aps(v: np.ndarray) -> np.ndarray:
    """Convert vector(s) from MIDAS to APS frame.

    Parameters
    ----------
    v : ndarray (..., 3)

    Returns
    -------
    ndarray (..., 3)
    """
    return (R_MIDAS_TO_APS @ v[..., None]).squeeze(-1)


def vector_aps_to_midas(v: np.ndarray) -> np.ndarray:
    """Convert vector(s) from APS to MIDAS frame.

    Parameters
    ----------
    v : ndarray (..., 3)

    Returns
    -------
    ndarray (..., 3)
    """
    return (R_APS_TO_MIDAS @ v[..., None]).squeeze(-1)


# -------------------------------------------------------------------
#  Convert orientation matrices
# -------------------------------------------------------------------

def orient_midas_to_aps(U: np.ndarray) -> np.ndarray:
    """Convert orientation matrix from MIDAS to APS frame.

    If U_midas takes crystal -> MIDAS lab, then
    U_aps = R_MIDAS_TO_APS @ U_midas takes crystal -> APS lab.

    Parameters
    ----------
    U : ndarray (..., 3, 3)

    Returns
    -------
    ndarray (..., 3, 3)
    """
    return R_MIDAS_TO_APS @ U


def orient_aps_to_midas(U: np.ndarray) -> np.ndarray:
    """Convert orientation matrix from APS to MIDAS frame.

    Parameters
    ----------
    U : ndarray (..., 3, 3)

    Returns
    -------
    ndarray (..., 3, 3)
    """
    return R_APS_TO_MIDAS @ U


# -------------------------------------------------------------------
#  Convert symmetric tensors (strain, stress)
# -------------------------------------------------------------------

def tensor_midas_to_aps(T: np.ndarray) -> np.ndarray:
    """Convert symmetric 3x3 tensor(s) from MIDAS to APS frame.

    Applies similarity transform: T_aps = R @ T_midas @ R^T

    Parameters
    ----------
    T : ndarray (..., 3, 3)

    Returns
    -------
    ndarray (..., 3, 3)
    """
    return R_MIDAS_TO_APS @ T @ R_MIDAS_TO_APS.T


def tensor_aps_to_midas(T: np.ndarray) -> np.ndarray:
    """Convert symmetric 3x3 tensor(s) from APS to MIDAS frame.

    Parameters
    ----------
    T : ndarray (..., 3, 3)

    Returns
    -------
    ndarray (..., 3, 3)
    """
    return R_APS_TO_MIDAS @ T @ R_APS_TO_MIDAS.T


def tensor_lab_to_sample(
    T: np.ndarray,
    omega_deg: float,
    frame: str = "midas",
) -> np.ndarray:
    """Convert symmetric tensor(s) from lab to sample frame.

    Parameters
    ----------
    T : ndarray (..., 3, 3)
    omega_deg : float
        Omega angle in degrees.
    frame : str
        ``"midas"`` or ``"aps"``.

    Returns
    -------
    ndarray (..., 3, 3)
    """
    R = lab_to_sample_rotation(omega_deg, frame)
    return R @ T @ R.T


# -------------------------------------------------------------------
#  Full conversion pipeline (MIDAS Grains.csv -> sample frame)
# -------------------------------------------------------------------

def grains_midas_to_sample(
    orientations: np.ndarray,
    positions: np.ndarray,
    strains: np.ndarray,
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
    if target_frame.lower() == "aps":
        R_frame = R_MIDAS_TO_APS.copy()
    elif target_frame.lower() in ("midas", "esrf"):
        R_frame = np.eye(3)
    else:
        raise ValueError(f"Unknown target_frame '{target_frame}'.")

    R_lab2sam = lab_to_sample_rotation(omega_deg, target_frame)
    R_total = R_lab2sam @ R_frame

    # Orientations: U_sample = R_total @ U_midas
    orient_out = R_total @ orientations

    # Positions: p_sample = R_total @ p_midas
    pos_out = (R_total @ positions[..., None]).squeeze(-1)

    # Strains: eps_sample = R_total @ eps_midas @ R_total^T
    strain_out = R_total @ strains @ R_total.T

    return {
        'orientations': orient_out,
        'positions': pos_out,
        'strains': strain_out,
    }
