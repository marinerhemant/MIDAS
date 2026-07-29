"""Honesty layer: systematic-perturbation UQ + independence accounting.

Two failure modes from AUDIT_2026-06-23.md that ordinary bootstrap UQ cannot catch:

1. **Resampling-only UQ reports an artifact's *consistency* as *precision*.** The demk
   bootstrap resampled grains that all carried the same systematic (the projection axis,
   the parent/twin labeling), so it confidently returned "P=1.00 / 11.2 sigma" for a
   geometry artifact. UQ must perturb the SYSTEMATICS (relabel, re-choose axis, reseed),
   not just resample noise. ``systematic_uq`` does that.

2. **Fourier-conjugate / shared-input quantities sold as "independent probes."** L_9R (a
   q-space FWHM) and the 3D-dPDF "9R correlation" (its r-space transform) were keyed to
   the SAME OM@[1,1,1] axis and the SAME voxel attribution, so their agreement was a
   mathematical identity, not corroboration. ``assert_independent`` refuses to let two
   quantities that share their load-bearing inputs be treated as independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = ["systematic_uq", "Probe", "IndependenceError", "assert_independent"]


def systematic_uq(
    perturbations: Sequence[Callable[[], float]],
    *,
    labels: Optional[Sequence[str]] = None,
) -> dict:
    """Spread of a quantity across SYSTEMATIC perturbations (not noise resampling).

    Each entry of ``perturbations`` is a zero-arg callable that recomputes the quantity
    under a different defensible assumption — e.g. swapped parent/twin labels, a
    re-chosen activated axis, a different clustering seed. The spread across them is the
    systematic uncertainty the bootstrap misses. If the quantity flips sign or the
    spread swamps the value, the quantity is an assumption artifact, not a measurement.

    Returns
    -------
    dict: values, mean, std, min, max, spread (max-min), sign_stable (bool),
          relative_spread (spread / |mean|).
    """
    vals = np.array([float(p()) for p in perturbations], dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return dict(values=vals, mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                    spread=np.nan, sign_stable=False, relative_spread=np.nan,
                    labels=list(labels) if labels else None)
    mean = float(finite.mean())
    spread = float(finite.max() - finite.min())
    return dict(
        values=vals, mean=mean, std=float(finite.std()),
        min=float(finite.min()), max=float(finite.max()), spread=spread,
        sign_stable=bool(np.all(finite > 0) or np.all(finite < 0)),
        relative_spread=float(spread / abs(mean)) if mean != 0 else np.inf,
        labels=list(labels) if labels else None,
    )


@dataclass(frozen=True)
class Probe:
    """Provenance of a derived quantity, for independence checks.

    axis_id : identifier of the projection/sampling direction it depends on (e.g.
        "OM@111"). attribution_id : how voxels were assigned (e.g. "per_grain_nn").
    space : "q" or "r" — a quantity and its Fourier transform are NOT independent.
    """
    name: str
    axis_id: Optional[str] = None
    attribution_id: Optional[str] = None
    space: Optional[str] = None


class IndependenceError(RuntimeError):
    """Raised when quantities sharing load-bearing inputs are treated as independent."""


def assert_independent(probes: Sequence[Probe]) -> None:
    """Raise if any two probes share their load-bearing inputs (axis + attribution).

    Two quantities keyed to the same projection axis AND the same voxel attribution are
    the same measurement (possibly in conjugate spaces) — their agreement is not
    corroboration. Use before claiming "N independent probes confirm X".
    """
    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            a, b = probes[i], probes[j]
            same_axis = (a.axis_id is not None and a.axis_id == b.axis_id)
            same_attr = (a.attribution_id is not None
                         and a.attribution_id == b.attribution_id)
            if same_axis and same_attr:
                conj = (a.space is not None and b.space is not None
                        and a.space != b.space)
                raise IndependenceError(
                    f"'{a.name}' and '{b.name}' share axis '{a.axis_id}' and attribution "
                    f"'{a.attribution_id}'"
                    + (" (and are Fourier conjugates q<->r)" if conj else "")
                    + " — they are one measurement, not independent probes "
                      "(AUDIT_2026-06-23.md)."
                )
