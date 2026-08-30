"""Diffraction-geometry helpers used across MIDAS HEDM kernels.

Scope is intentionally narrow: small primitives that are NOT orientation
math but are shared between indexing, refinement, and reconstruction
modules. Larger detector-geometry routines live in their respective
packages (midas-transforms, midas-diffract).

Backends:
  - NumPy (default).
  - PyTorch when any input is a torch.Tensor — returns a tensor on the
    input's device/dtype, differentiable except where noted per function.

All angles returned in DEGREES (matches the legacy MIDAS convention).
"""

from __future__ import annotations

import math

import numpy as np
from ._optional import torch

_RAD2DEG = 180.0 / math.pi


def _is_torch(*args) -> bool:
    return any(isinstance(a, torch.Tensor) for a in args)


def calc_eta_angle_all(y, z):
    """Azimuthal eta angle on the detector (degrees).

    Returns the signed angle of the (y, z) point measured from the +z
    axis (positive z), with sign flipped where y > 0. Matches the legacy
    MIDAS convention used across indexer, refiner, and pf_MIDAS:

        eta = -sign(y) * arccos(z / sqrt(y^2 + z^2))      (in degrees)

    Parameters
    ----------
    y, z : float, ndarray, or torch.Tensor (any matching shape)

    Returns
    -------
    Same backend as inputs. Scalar in, scalar out; array in, array out.
    Output in degrees.

    Notes
    -----
    - Undefined at y = z = 0 (no azimuth exists there); the torch backend
      returns 0 with zero gradient rather than NaN, but callers should still
      filter.
    - The torch backend is differentiable everywhere except that origin.
      It previously claimed to be "autograd-safe" while returning a NaN
      gradient along the whole y = 0 axis -- every spot at eta = 0 or 180.
    """
    if _is_torch(y, z):
        return _calc_eta_angle_all_torch(y, z)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    r = np.sqrt(y_arr * y_arr + z_arr * z_arr)
    # arccos(z / r); guard r = 0 to avoid nan (caller responsibility but be defensive)
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = _RAD2DEG * np.arccos(np.where(r > 0, z_arr / r, 1.0))
    if y_arr.ndim == 0:
        return float(-alpha) if y_arr > 0 else float(alpha)
    out = alpha.copy()
    out[y_arr > 0] *= -1
    return out


def _calc_eta_angle_all_torch(y, z) -> torch.Tensor:
    dtype = y.dtype if isinstance(y, torch.Tensor) else z.dtype
    device = y.device if isinstance(y, torch.Tensor) else z.device
    y_t = torch.as_tensor(y, dtype=dtype, device=device)
    z_t = torch.as_tensor(z, dtype=dtype, device=device)
    # atan2, not acos(z/r) with a sign flip. They agree to ~5e-11 deg (the
    # difference is the acos form's own precision loss near |z/r| = 1), but the
    # acos form returned a NaN GRADIENT along the entire y = 0 axis -- every
    # spot at eta = 0 or 180, a line on the detector rather than a point. Two
    # separate causes, both of the guard-the-output-not-the-input kind:
    # sqrt'(0) at the origin, and acos'(1) = -inf wherever z/r reaches +-1.
    # MEASURED before the change: d(eta)/dy = d(eta)/dz = nan at (0, 1).
    #
    # ``0.0 - y`` rather than ``-y``: negating a positive zero gives NEGATIVE
    # zero, and atan2(-0.0, z<0) is -pi where atan2(+0.0, z<0) is +pi. That
    # would silently flip eta from +180 to -180 on the z < 0 half of the y = 0
    # axis. The subtraction yields +0.0 and keeps the legacy convention.
    #
    # At the origin eta does not exist at all, so both components are forced to
    # a constant: value 0 (matching the NumPy backend) and exactly zero
    # gradient. atan2's own backward is NaN at (0, 0).
    at_origin = (y_t * y_t + z_t * z_t) <= 0.0
    y_arg = torch.where(at_origin, torch.zeros_like(y_t), 0.0 - y_t)
    z_arg = torch.where(at_origin, torch.ones_like(z_t), z_t)
    return _RAD2DEG * torch.atan2(y_arg, z_arg)
