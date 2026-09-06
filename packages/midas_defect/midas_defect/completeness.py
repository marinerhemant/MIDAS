"""Completeness audit: predict every reflection, then look at each one.

An index reported as "45 reflections" says nothing on its own. The number that
matters is what happened to the reflections the orientation predicts and did
*not* deliver, because the four possible reasons are not interchangeable:

===========  ==========================================================
INDEXED      a spot is there and the solver assigned it
MISSED       a spot is there and the solver did **not** take it
MASKED       predicted onto masked detector; never observable
ABSENT       no spot within tolerance — the data does not contain it
===========  ==========================================================

Only ABSENT is a statement about the **sample**. MISSED is a statement about
the **analysis**, and conflating the two is how a silent spot-filtering bug
survives: it removes real reflections and the index still looks clean because
nobody counted what the model expected and did not get. MASKED is a statement
about the **detector** — a prediction landing behind a module gap is not
evidence of absence any more than one landing off the edge is.

The search window
-----------------
It must come from the model's own residual distribution, not from a round
number. In the analysis this was extracted from, a hardcoded 8 px / 2° window
reported six MISSED reflections and one spurious positive; every one of them
vanished when the window was set to the 90th percentile of the residuals over
the reflections the model *did* index. :func:`window_from_residuals` does that,
and :func:`audit_completeness` requires the window explicitly rather than
defaulting to something convenient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CompletenessAudit", "window_from_residuals", "audit_completeness"]


@dataclass
class CompletenessAudit:
    """The four-way verdict on every predicted reflection."""
    counts: Dict[str, int]
    verdicts: np.ndarray               # (n_predicted,) of str
    missed: List[dict] = field(default_factory=list)
    absent: List[dict] = field(default_factory=list)
    window_px: float = 0.0
    window_omega_deg: float = 0.0

    @property
    def n_predicted(self) -> int:
        return int(sum(self.counts.values()))

    @property
    def observable(self) -> int:
        """Predicted reflections that were not behind the mask."""
        return self.n_predicted - self.counts["MASKED"]

    def __str__(self) -> str:
        t = self.n_predicted
        body = " / ".join(f"{self.counts[k]} {k}"
                          for k in ("INDEXED", "MISSED", "MASKED", "ABSENT"))
        return (f"{body} of {t} predicted "
                f"(window {self.window_px:.1f} px, "
                f"{self.window_omega_deg:.2f} deg)")


def window_from_residuals(residual_px: Sequence[float],
                          residual_omega_deg: Sequence[float],
                          percentile: float = 90.0) -> Tuple[float, float]:
    """Search window from the model's OWN residuals over what it indexed.

    Pass the per-reflection position and ω residuals of the *assigned*
    reflections. Using a window much wider than these manufactures matches;
    using one much narrower manufactures absences. Neither failure announces
    itself.
    """
    rp = np.asarray(residual_px, float)
    ro = np.asarray(residual_omega_deg, float)
    if rp.size == 0 or ro.size == 0:
        raise ValueError("no residuals supplied — the window cannot be "
                         "derived from the model, and must not be guessed")
    return (float(np.percentile(rp, percentile)),
            float(np.percentile(ro, percentile)))


def audit_completeness(*,
                       predicted_hkl: np.ndarray,
                       predicted_row: np.ndarray,
                       predicted_col: np.ndarray,
                       predicted_omega_deg: np.ndarray,
                       observed_row: np.ndarray,
                       observed_col: np.ndarray,
                       observed_omega_deg: np.ndarray,
                       assigned_hkl: Sequence[Sequence[int]],
                       mask: np.ndarray,
                       window_px: float,
                       window_omega_deg: float,
                       mask_box_half: int = 6,
                       mask_fraction: float = 0.5,
                       observed_intensity: Optional[np.ndarray] = None,
                       ) -> CompletenessAudit:
    """Classify every predicted reflection as INDEXED / MISSED / MASKED / ABSENT.

    Parameters
    ----------
    predicted_*
        One entry per predicted reflection: its hkl and where the model puts
        it. The caller is responsible for having already restricted these to
        reflections that reach Bragg inside the measured ω range and land on
        the detector — this function does not re-derive the geometry.
    observed_*
        The measured spot list, in the same coordinates.
    assigned_hkl
        The hkls the solver actually took. Membership is by hkl, so a
        reflection the solver assigned counts INDEXED even if its residual is
        larger than the window.
    mask
        Boolean detector mask, True = masked. A prediction whose surrounding
        box is more than ``mask_fraction`` masked is MASKED, not ABSENT.

    Notes
    -----
    The mask test is applied **before** the search, and only to reflections the
    solver did not assign: an assigned reflection is observed by definition, so
    calling it MASKED would be incoherent.
    """
    predicted_hkl = np.asarray(predicted_hkl)
    n = len(predicted_hkl)
    for nm, arr in (("row", predicted_row), ("col", predicted_col),
                    ("omega", predicted_omega_deg)):
        if len(arr) != n:
            raise ValueError(f"predicted_{nm} has {len(arr)} entries, "
                             f"predicted_hkl has {n}")
    obs_r = np.asarray(observed_row, float)
    obs_c = np.asarray(observed_col, float)
    obs_w = np.asarray(observed_omega_deg, float)
    if not (obs_r.shape == obs_c.shape == obs_w.shape):
        raise ValueError("observed arrays must have matching shapes")
    if window_px <= 0 or window_omega_deg <= 0:
        raise ValueError("window must be positive; derive it with "
                         "window_from_residuals()")
    mask = np.asarray(mask, bool)

    assigned = {tuple(int(v) for v in h) for h in assigned_hkl}
    counts = {"INDEXED": 0, "MISSED": 0, "MASKED": 0, "ABSENT": 0}
    verdicts = np.empty(n, dtype=object)
    missed: List[dict] = []
    absent: List[dict] = []

    nr, nc = mask.shape
    for i in range(n):
        key = tuple(int(v) for v in predicted_hkl[i])
        pr = float(predicted_row[i])
        pc = float(predicted_col[i])
        pw = float(predicted_omega_deg[i])
        is_assigned = key in assigned

        if not is_assigned:
            ri, ci = int(round(pr)), int(round(pc))
            box = mask[max(ri - mask_box_half, 0):ri + mask_box_half + 1,
                       max(ci - mask_box_half, 0):ci + mask_box_half + 1]
            if box.size and box.mean() > mask_fraction:
                counts["MASKED"] += 1
                verdicts[i] = "MASKED"
                continue

        if is_assigned:
            counts["INDEXED"] += 1
            verdicts[i] = "INDEXED"
            continue

        if obs_r.size == 0:
            counts["ABSENT"] += 1
            verdicts[i] = "ABSENT"
            absent.append({"hkl": key, "omega": pw, "row": pr, "col": pc,
                           "nearest_px": float("inf")})
            continue

        d_px = np.hypot(obs_r - pr, obs_c - pc)
        near = (d_px < window_px) & (np.abs(obs_w - pw) <= window_omega_deg)
        if near.any():
            j = int(np.flatnonzero(near)[np.argmin(d_px[near])])
            counts["MISSED"] += 1
            verdicts[i] = "MISSED"
            rec = {"hkl": key, "omega": pw, "row": pr, "col": pc,
                   "spot": j, "distance_px": float(d_px[j])}
            if observed_intensity is not None:
                rec["intensity"] = float(np.asarray(observed_intensity)[j])
            missed.append(rec)
        else:
            counts["ABSENT"] += 1
            verdicts[i] = "ABSENT"
            absent.append({"hkl": key, "omega": pw, "row": pr, "col": pc,
                           "nearest_px": float(d_px.min())})

    return CompletenessAudit(counts=counts, verdicts=verdicts,
                             missed=missed, absent=absent,
                             window_px=float(window_px),
                             window_omega_deg=float(window_omega_deg))
