"""Diffraction-geometry helpers used across MIDAS HEDM kernels.

Scope is intentionally narrow: small primitives that are NOT orientation
math but are shared between indexing, refinement, and reconstruction
modules. Larger detector-geometry routines live in their respective
packages (midas-transforms, midas-diffract).

Backends:
  - NumPy (default).
  - PyTorch when any input is a torch.Tensor — returns a tensor on the
    input's device/dtype, autograd-safe.

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
    - Undefined at y = z = 0; callers must filter before calling.
    - For tensors, the function is autograd-safe; sign flipping uses a
      smooth `torch.where`, not an in-place masked assignment.
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
    r = torch.sqrt(y_t * y_t + z_t * z_t)
    safe_r = torch.where(r > 0, r, torch.ones_like(r))
    cos_arg = torch.where(r > 0, z_t / safe_r, torch.ones_like(z_t))
    alpha = (_RAD2DEG) * torch.arccos(cos_arg.clamp(-1.0, 1.0))
    return torch.where(y_t > 0, -alpha, alpha)
