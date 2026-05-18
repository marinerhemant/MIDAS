"""Item 39 — Multi-modal data alignment helper.

Operando experiments often pair XRD with auxiliary streams (Raman,
DSC, electrochemistry, mass spec, …). This helper takes a list of
:class:`AuxiliaryStream` objects and aligns each to the per-frame
XRD timestamps via interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class AuxiliaryStream:
    name: str
    timestamps: np.ndarray   # seconds, monotonic
    values: np.ndarray
    units: str = ""


def align_to_xrd_frames(
    frame_timestamps: np.ndarray,
    aux_streams: List[AuxiliaryStream],
    *,
    interpolation: str = "linear",
) -> Dict[str, np.ndarray]:
    """Resample each auxiliary stream onto ``frame_timestamps``.

    ``interpolation`` ∈ {``"linear"``, ``"nearest"``, ``"previous"``}.
    Returns dict keyed by ``stream.name``.
    """
    if interpolation not in ("linear", "nearest", "previous"):
        raise ValueError(f"unknown interpolation {interpolation!r}")
    frame_t = np.asarray(frame_timestamps, dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    for stream in aux_streams:
        ts = np.asarray(stream.timestamps, dtype=np.float64)
        vs = np.asarray(stream.values, dtype=np.float64)
        sort_idx = np.argsort(ts)
        ts = ts[sort_idx]
        vs = vs[sort_idx]
        if interpolation == "linear":
            out[stream.name] = np.interp(frame_t, ts, vs)
        elif interpolation == "nearest":
            idx = np.argmin(
                np.abs(frame_t[:, None] - ts[None, :]), axis=1,
            )
            out[stream.name] = vs[idx]
        else:  # previous
            idx = np.searchsorted(ts, frame_t, side="right") - 1
            idx = np.clip(idx, 0, ts.size - 1)
            out[stream.name] = vs[idx]
    return out


__all__ = ["AuxiliaryStream", "align_to_xrd_frames"]
