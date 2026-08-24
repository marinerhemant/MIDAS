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

Tomography reconstruction grid:
    (slice, iy, ix) voxel indices. `slice` runs along the rotation axis, so
    slice = MIDAS Z = APS Y, and the sample-stage vertical position is what
    registers a tomogram against an FF or NF layer. See
    :func:`tomo_grid_to_midas` and :func:`tomo_slice_for_z` at the end of this
    module; the in-plane handedness is NOT assumed.

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


# -------------------------------------------------------------------
#  Tomography reconstruction grid  <->  MIDAS lab
# -------------------------------------------------------------------
#
# A tomographic reconstruction is the third frame that has to line up with a
# diffraction experiment, and it is the one with no established convention in
# this repository: the reconstruction cube's shape lives in a filename, its
# pixel size lives in the acquisition config, and the rotation-axis position is
# an output of the reconstruction rather than an input.
#
# The frames, and why each index means what it means:
#
#   Detector (tomo camera) : (row, col). Rows run along the rotation axis
#                            (vertical); columns run across the beam.
#   Reconstruction grid    : (slice, iy, ix). One slice per detector row, so
#                            `slice` is vertical. Within a slice the two axes
#                            are the in-plane sample axes at omega = 0.
#   MIDAS lab              : x along the beam, y outboard, z up.
#   APS lab                : x outboard, y up, z along the beam.
#
# So the vertical axis is `slice` = MIDAS z = APS y, and THAT is what ties a
# tomogram to an FF or NF layer: the sample-stage vertical position is recorded
# per scan, so the registration is READ, not fitted. Fitting it and then
# validating the fit with the same data is circular.
#
# The in-plane pair is deliberately NOT hard-coded to a handedness here. A
# reconstruction's in-plane axes depend on the projection ordering, the sign of
# the rotation direction, and whether the reconstructor flips its output; those
# are properties of the tomography code and the acquisition, not of MIDAS.
# `tomo_grid_to_midas` therefore takes an explicit `in_plane` string and refuses
# to guess -- a wrong choice mirrors the sample, which is silent and produces a
# perfectly plausible reconstruction (the same failure class as the omega sign).

#: The four axis assignments a reconstruction's in-plane pair can take, as
#: (ix -> MIDAS axis, iy -> MIDAS axis) with signs. MIDAS in-plane axes are
#: x (beam) and y (outboard).
TOMO_IN_PLANE = {
    "xy":   ((1.0, 0.0), (0.0, 1.0)),    # ix -> +x, iy -> +y
    "yx":   ((0.0, 1.0), (1.0, 0.0)),    # ix -> +y, iy -> +x
    "-xy":  ((-1.0, 0.0), (0.0, 1.0)),
    "x-y":  ((1.0, 0.0), (0.0, -1.0)),
    "-x-y": ((-1.0, 0.0), (0.0, -1.0)),
    "-yx":  ((0.0, -1.0), (1.0, 0.0)),
    "y-x":  ((0.0, 1.0), (-1.0, 0.0)),
    "-y-x": ((0.0, -1.0), (-1.0, 0.0)),
}


def tomo_grid_to_midas(
    slice_idx, iy, ix,
    *,
    pixel_size_um: float,
    slice_pitch_um: float,
    rot_axis_ix: float,
    rot_axis_iy: float,
    slice0_z_um: float = 0.0,
    in_plane: str = "xy",
):
    """Reconstruction voxel indices -> MIDAS lab coordinates (µm).

    The sample-frame position at omega = 0, which is where a tomogram lives.

    Parameters
    ----------
    slice_idx, iy, ix
        Voxel indices; scalars or broadcastable arrays.
    pixel_size_um
        In-plane reconstruction pixel size. **Required, never defaulted** — it
        scales every path length and the illuminated volume, and no
        reconstruction file format in use here records it.
    slice_pitch_um
        Distance between slices along the rotation axis. Equals
        ``pixel_size_um`` for an isotropic reconstruction, but binning the
        detector vertically breaks that, so it is separate.
    rot_axis_ix, rot_axis_iy
        In-plane index of the rotation axis. This is an OUTPUT of the
        reconstruction (the shift sweep), not a property of the detector;
        ``n/2`` is a guess, not a default.
    slice0_z_um
        MIDAS z of slice 0 — the sample-stage vertical position. This is the
        number that registers the tomogram against an FF or NF layer.
    in_plane
        One of :data:`TOMO_IN_PLANE`. No default handedness is assumed; see the
        module notes.

    Returns
    -------
    (x, y, z) in MIDAS lab µm, same shape as the broadcast inputs.
    """
    if in_plane not in TOMO_IN_PLANE:
        raise ValueError(
            f"in_plane must be one of {sorted(TOMO_IN_PLANE)}; got {in_plane!r}. "
            "There is no safe default: the wrong choice mirrors the sample, "
            "which is silent and reconstructs perfectly."
        )
    if not (pixel_size_um > 0):
        raise ValueError(
            f"pixel_size_um must be > 0; got {pixel_size_um!r}. It is not "
            "recorded in any reconstruction file format used here, so it has "
            "to be supplied from the acquisition config."
        )
    if not (slice_pitch_um > 0):
        raise ValueError(f"slice_pitch_um must be > 0; got {slice_pitch_um!r}")

    (ax_x, ax_y), (ay_x, ay_y) = TOMO_IN_PLANE[in_plane]
    dx = (ix - rot_axis_ix) * pixel_size_um
    dy = (iy - rot_axis_iy) * pixel_size_um
    x = ax_x * dx + ay_x * dy
    y = ax_y * dx + ay_y * dy
    z = slice_idx * slice_pitch_um + slice0_z_um
    return x, y, z


def midas_to_tomo_grid(
    x, y, z,
    *,
    pixel_size_um: float,
    slice_pitch_um: float,
    rot_axis_ix: float,
    rot_axis_iy: float,
    slice0_z_um: float = 0.0,
    in_plane: str = "xy",
):
    """MIDAS lab coordinates (µm) -> fractional reconstruction voxel indices.

    Exact inverse of :func:`tomo_grid_to_midas`, and here rather than at the
    call site so the forward and backward maps cannot drift apart — the
    handedness convention has one home.

    Every entry of :data:`TOMO_IN_PLANE` is a signed axis permutation, so the
    in-plane inverse is the transpose; no matrix solve and no chance of an
    inverse that is only approximately orthogonal.

    Returns ``(slice_idx, iy, ix)`` as floats. They are deliberately NOT
    rounded: a caller asking "is this point in the sample" wants to control its
    own rounding, and ``rint`` here would quietly extend the mask by half a
    voxel in every direction.
    """
    if in_plane not in TOMO_IN_PLANE:
        raise ValueError(
            f"in_plane must be one of {sorted(TOMO_IN_PLANE)}; got {in_plane!r}"
        )
    if not (pixel_size_um > 0 and slice_pitch_um > 0):
        raise ValueError("pixel_size_um and slice_pitch_um must be > 0")

    (ax_x, ax_y), (ay_x, ay_y) = TOMO_IN_PLANE[in_plane]
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # A is orthonormal (a signed permutation), so A^-1 = A^T.
    dx = ax_x * x + ax_y * y
    dy = ay_x * x + ay_y * y
    ix = dx / pixel_size_um + rot_axis_ix
    iy = dy / pixel_size_um + rot_axis_iy
    s = (np.asarray(z, dtype=np.float64) - slice0_z_um) / slice_pitch_um
    return s, iy, ix


def tomo_slice_for_z(
    z_um, *, slice_pitch_um: float, slice0_z_um: float = 0.0, n_slices=None
):
    """MIDAS z (µm) -> nearest reconstruction slice index.

    The inverse of the vertical half of :func:`tomo_grid_to_midas`, and the
    practical way to ask "which tomo slice corresponds to this FF layer?".

    Raises when the requested z falls outside the reconstruction rather than
    clamping: silently returning the end slice would extrapolate the sample
    mask beyond the tomographic field of view, fabricating path length.
    """
    if not (slice_pitch_um > 0):
        raise ValueError(f"slice_pitch_um must be > 0; got {slice_pitch_um!r}")
    idx_f = (np.asarray(z_um, dtype=np.float64) - slice0_z_um) / slice_pitch_um
    idx = np.rint(idx_f).astype(np.int64)
    if n_slices is not None:
        bad = (idx < 0) | (idx >= int(n_slices))
        if np.any(bad):
            raise ValueError(
                f"z={z_um} µm maps to slice {idx} which is outside the "
                f"{int(n_slices)}-slice reconstruction. The tomogram does not "
                "cover this layer; extrapolating would fabricate path length."
            )
    return idx if idx.ndim else int(idx)
