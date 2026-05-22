"""Analytical radial distortion basis for calibrate-v2.

The distortion *model* (layout tables, the v1↔v2 coefficient mapping, and the
backend-agnostic kernel) now lives in the shared :mod:`midas_distortion` leaf so
that calibration (here), :mod:`midas_peakfit` and :mod:`midas_transforms` all
evaluate one definition. This module re-exports those canonical symbols and adds
the two calibrate-v2-specific helpers (:func:`build_p_coeffs`,
:func:`coeffs_from_named`) that assemble a torch coefficient vector from an
unpacked-parameter dict.

See :mod:`midas_distortion.core` for the model description.
"""
from __future__ import annotations

from typing import List

import torch

# Single source of truth — the distortion layout + kernel + v1↔v2 maps.
from midas_distortion import (  # noqa: F401  (re-exported)
    HarmonicTerm,
    P_COEF_NAMES,
    PHASE_NAMES,
    ISO_NAMES,
    AMP_NAMES,
    v1_term_layout,
    v2_term_layout,
    extended_term_layout,
    extended_p_coef_names,
    V1_TO_V2_DISTORTION,
    V2_TO_V1_DISTORTION,
    v1_to_v2_coeffs,
    v2_to_v1_coeffs,
    distortion_factor,
    apply_distortion,
)

# v2 slot index for each v1 p-index (name → position in P_COEF_NAMES).
_V1_IDX_TO_V2_SLOT = {
    v1_idx: P_COEF_NAMES.index(name)
    for v1_idx, name in V1_TO_V2_DISTORTION.items()
}


# ----------------------------------------------------------- helpers

def build_p_coeffs(unpacked: dict, *, dtype=None, device=None) -> torch.Tensor:
    """Stack the 15 v2-named distortion params from an unpacked dict.

    Centralises the v2 name list — call sites no longer need to know the
    order.  Missing names default to 0.0 (e.g. when a spec fixes a coefficient
    at zero by omission).
    """
    pieces: List = []
    ref = None
    for nm in P_COEF_NAMES:
        v = unpacked.get(nm)
        if v is None:
            pieces.append(None)
            continue
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(
                v, dtype=dtype if dtype is not None else torch.float64,
                device=device if device is not None else "cpu")
        if ref is None:
            ref = v
        pieces.append(v)
    if ref is None:
        # Nothing supplied — return a 15-zero vector.
        return torch.zeros(
            15, dtype=dtype if dtype is not None else torch.float64,
            device=device if device is not None else "cpu")
    out_dtype = dtype if dtype is not None else ref.dtype
    out_device = device if device is not None else ref.device
    pieces = [
        (p.to(dtype=out_dtype, device=out_device)
         if p is not None else torch.zeros((), dtype=out_dtype, device=out_device))
        for p in pieces
    ]
    return torch.stack(pieces)


def coeffs_from_named(named: dict, default: float = 0.0, n: int = 15,
                      dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Build a p_coeffs[15] tensor from a v2-named dict.

    Recognised keys are :data:`P_COEF_NAMES`.  For backward compat, keys
    ``p0``..``p14`` are also accepted via the v1 → v2 mapping (so an old
    paramstest dict still works).
    """
    p = torch.full((n,), default, dtype=dtype, device=device)
    for k, v in named.items():
        if k in P_COEF_NAMES:
            i = P_COEF_NAMES.index(k)
        elif k.startswith("p") and k[1:].isdigit():
            v1_idx = int(k[1:])
            if 0 <= v1_idx < 15:
                i = _V1_IDX_TO_V2_SLOT[v1_idx]
            else:
                continue
        else:
            continue
        if isinstance(v, torch.Tensor):
            p[i] = v.to(dtype=dtype, device=device)
        else:
            p[i] = float(v)
    return p


__all__ = [
    "HarmonicTerm",
    "P_COEF_NAMES", "PHASE_NAMES", "ISO_NAMES", "AMP_NAMES",
    "v1_term_layout", "v2_term_layout",
    "extended_term_layout", "extended_p_coef_names",
    "V1_TO_V2_DISTORTION", "V2_TO_V1_DISTORTION",
    "v1_to_v2_coeffs", "v2_to_v1_coeffs",
    "distortion_factor", "apply_distortion",
    "build_p_coeffs", "coeffs_from_named",
]
