"""Item 40 — Time-resolved pump-probe substrate.

This is the *infrastructure* surface for the separate pump-probe
analysis package — we provide trigger metadata + a wrapped FrameSource
that exposes pump/unpump labels per frame, plus a differential helper.
**Do not** build pump-probe analysis here; coordinate with the dedicated
workstream owner before extending.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from .frame_source import FrameSource


@dataclass
class TriggerMetadata:
    frame_id: str
    pump_state: str       # "pumped" / "unpumped" / "off"
    delay_ps: Optional[float] = None
    sequence_index: Optional[int] = None


class TriggerTaggedFrameSource(FrameSource):
    """Wrap any :class:`FrameSource` with per-frame trigger metadata."""

    def __init__(self, base: FrameSource, triggers: List[TriggerMetadata]):
        if len(triggers) != base.n_frames:
            raise ValueError(
                f"triggers length {len(triggers)} != base.n_frames "
                f"{base.n_frames}"
            )
        self._base = base
        self._triggers = list(triggers)

    @property
    def n_frames(self) -> int:
        return self._base.n_frames

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._base.frame_shape

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        return iter(self._base)

    def get(self, idx: int) -> Tuple[str, np.ndarray]:
        return self._base.get(idx)

    def trigger_for(self, idx: int) -> TriggerMetadata:
        return self._triggers[idx]


def differential_with_variance(
    pumped_profiles: torch.Tensor, pumped_sigma: torch.Tensor,
    unpumped_profiles: torch.Tensor, unpumped_sigma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Δ I(Q) = I_pumped − I_unpumped with σ²_diff = σ²_p + σ²_u."""
    diff = pumped_profiles - unpumped_profiles
    var_diff = pumped_sigma * pumped_sigma + unpumped_sigma * unpumped_sigma
    return diff, torch.sqrt(var_diff.clamp(min=0.0))


__all__ = [
    "TriggerMetadata", "TriggerTaggedFrameSource",
    "differential_with_variance",
]
