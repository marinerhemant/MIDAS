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

__all__ = ["systematic_uq", "Probe", "IndependenceError", "assert_independent",
    "decoy_test", "inflated_cell", "feature_in_raw",
]


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


# ---------------------------------------------------------------------------
# 2026-09-10 -- the two controls the La3Ni2O7 port dispositioned into this module on
# 2026-09-01 (work/scripts/step17_domain2_check.py, step55_raw_check.py). They never
# landed; the port's FINAL STATE table dropped them without a word.
# ---------------------------------------------------------------------------

def inflated_cell(cell, fraction, axes=(0,)):
    """A deliberately wrong cell: lengths on ``axes`` scaled by ``1 + fraction``."""
    out = [float(x) for x in cell]
    for i in axes:
        out[i] *= 1.0 + float(fraction)
    return tuple(out)


def decoy_test(score_of, cell, decoys, *, threshold):
    """Does a deliberately WRONG model pass too? If so, passing means nothing.

    Beating a null is necessary and not sufficient. Run the same machinery on the real
    ``cell`` and on each decoy; ``score_of(cell)`` returns the statistic the acceptance
    rule thresholds (a matched-reflection count, a figure of merit).

    Measured on La3Ni2O7 (2026-08-26): a second-domain orientation beat its omega null
    (27 matched against a null maximum of 6, p = 0.000) while its cell floated +0.70 %
    from domain 1's at the same grid point -- and a decoy cell with ``a`` inflated 3 %
    also beat the null. Only a structural test settled it.

    Parameters
    ----------
    score_of : callable, ``cell -> float``
    cell : the model under test
    decoys : mapping ``label -> cell`` of deliberately wrong models, e.g.
        ``{"a +3 %": inflated_cell(cell, 0.03)}``
    threshold : the acceptance threshold the real analysis applies

    Returns
    -------
    dict with ``real``, ``decoys`` (label -> score), ``real_passes``,
    ``decoys_passing`` and ``verdict``: ``"informative"`` (the real model passes and
    no decoy does), ``"uninformative"`` (a decoy passes too, so the acceptance does not
    discriminate), or ``"real_fails"``.
    """
    real = float(score_of(cell))
    scores = {str(k): float(score_of(v)) for k, v in dict(decoys).items()}
    passing = [k for k, v in scores.items() if v >= threshold]
    real_passes = real >= threshold
    verdict = "real_fails" if not real_passes else ("uninformative" if passing else "informative")
    return dict(real=real, decoys=scores, threshold=float(threshold),
                real_passes=bool(real_passes), decoys_passing=passing, verdict=verdict)


def feature_in_raw(raw, processed, rows, cols, perp_rows, perp_cols, *,
                   half_width=20, core=2, min_sigma=5.0):
    """Is a feature in the RAW frames, or made by the processing?

    A smooth background estimate can both CREATE a ridge (by over-subtracting either
    side of it) and ERASE one (by absorbing it), so a feature measured on a
    background-subtracted image must be checked against pixels nothing was done to.
    Takes the median perpendicular profile across a path in both images and compares
    its core against its wings, in robust wing sigma (Poisson-floored on a flat patch,
    where a zero MAD means a flat patch, not certainty).

    Parameters
    ----------
    raw, processed : 2-D images on one pixel grid (e.g. max over frames)
    rows, cols : path points, pixels
    perp_rows, perp_cols : unit perpendicular at each path point, pixels
    half_width : profile half-length, pixels
    core : ``|offset| <= core`` is the feature
    min_sigma : core-minus-wing contrast needed to call it present

    Returns
    -------
    dict with ``offsets``, ``raw_profile``, ``processed_profile``, ``raw_sigma``,
    ``processed_sigma``, ``in_raw``, ``in_processed`` and ``verdict``: ``"in_raw"``,
    ``"processing_only"`` (present only after processing -- the step may have made it),
    or ``"absent"``.
    """
    import numpy as _np
    raw = _np.asarray(raw, float); processed = _np.asarray(processed, float)
    rows = _np.asarray(rows, float); cols = _np.asarray(cols, float)
    pr = _np.asarray(perp_rows, float); pc = _np.asarray(perp_cols, float)
    off = _np.arange(-half_width, half_width + 1, dtype=float)

    def _profile(img):
        rr = _np.clip(_np.rint(rows[:, None] + pr[:, None] * off[None, :]).astype(int), 0, img.shape[0] - 1)
        cc = _np.clip(_np.rint(cols[:, None] + pc[:, None] * off[None, :]).astype(int), 0, img.shape[1] - 1)
        return _np.nanmedian(img[rr, cc], axis=0)

    def _sigma(prof):
        wing = _np.abs(off) >= 0.5 * half_width
        base = _np.nanmedian(prof[wing])
        sd = 1.4826 * _np.nanmedian(_np.abs(prof[wing] - base))
        sd = sd if sd > 0 else _np.sqrt(max(abs(base), 1.0))
        return float((_np.nanmean(prof[_np.abs(off) <= core]) - base) / sd)

    rp, pp = _profile(raw), _profile(processed)
    rs, ps = _sigma(rp), _sigma(pp)
    in_raw, in_proc = rs >= min_sigma, ps >= min_sigma
    verdict = "in_raw" if in_raw else ("processing_only" if in_proc else "absent")
    return dict(offsets=off, raw_profile=rp, processed_profile=pp, raw_sigma=rs,
                processed_sigma=ps, in_raw=bool(in_raw), in_processed=bool(in_proc), verdict=verdict)
