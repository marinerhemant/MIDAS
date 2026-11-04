"""Device and dtype resolution for midas_defect: PRECISION FIRST.

The detector -> q geometry (pixel to lab vector and back, Ewald-crossing roots, orientation fits) is consistent only to ~1e-8 px in float64. The shared
``midas_transforms.device`` contract (cuda > mps > cpu, float32 on cuda/mps) is right for the C-parity transforms but wrong here: on a Mac it silently
ran this package's geometry on MPS in single precision (and MPS has no float64 at all, so asking for it crashed obscurely).

* device: argument > ``MIDAS_DEFECT_DEVICE`` > ``cuda`` if available > ``cpu``. MPS is never chosen implicitly; ask for it explicitly if you want it.
* dtype: argument > ``MIDAS_DEFECT_DTYPE`` > float64 on EVERY device.
* ``mps`` asked for explicitly with no dtype: float32, with a RuntimeWarning that the geometry loses precision (the explicit request is honoured, loudly).
* ``float64`` asked for explicitly on ``mps`` raises a clear ValueError (MPS cannot do it) instead of failing inside torch.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Union

import torch

__all__ = ["resolve_device", "resolve_dtype"]

_DTYPE_ALIAS = {"float32": torch.float32, "fp32": torch.float32, "single": torch.float32,
                "float64": torch.float64, "fp64": torch.float64, "double": torch.float64}


def resolve_device(user: Optional[Union[str, torch.device]] = None) -> torch.device:
    if isinstance(user, torch.device):
        return user
    if user is None:
        user = os.environ.get("MIDAS_DEFECT_DEVICE")
    if user is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(user)


def resolve_dtype(device: torch.device, user: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    if isinstance(user, torch.dtype):
        dt = user
    else:
        if user is None:
            user = os.environ.get("MIDAS_DEFECT_DTYPE")
        if user is None:
            if device.type == "mps":
                warnings.warn("midas_defect on MPS runs in float32 (MPS has no float64): the detector -> q geometry is good to ~1e-8 px only in float64; "
                              "use device='cpu' for precision work", RuntimeWarning, stacklevel=3)
                return torch.float32
            dt = torch.float64
        else:
            try:
                dt = _DTYPE_ALIAS[str(user).lower()]
            except KeyError as e:
                raise ValueError(f"Unsupported dtype '{user}'. Use one of {sorted(_DTYPE_ALIAS)}.") from e
    if device.type == "mps" and dt == torch.float64:
        raise ValueError("float64 is not available on the MPS backend: pass dtype='float32' explicitly (precision loss: the geometry is good to ~1e-8 px "
                         "only in float64) or use device='cpu'")
    return dt
