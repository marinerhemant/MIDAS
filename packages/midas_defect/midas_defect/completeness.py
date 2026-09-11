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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CompletenessAudit", "window_from_residuals", "audit_completeness",
           "TargetedRecovery", "targeted_snr", "targeted_recovery"]


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


# ── Targeted extraction at predicted sites ──────────────────────────────────
#
# Blind detection is the wrong tool once an orientation is known. On La3Ni2O7
# 2604 the labelled frames showed bright nodes marching along (00L) and other
# rows with no blob found for them at all; predicting every reflection and
# sampling the frames there took three raster positions from 40 / 26 / 17 to
# 47 / 44 / 32 DISTINCT reflections (2026-09-04, provisional -- both nulls were
# the author's own). Ported from the project's repro/predicted_recovery.py.

def targeted_snr(sub, raw, mask, frame, row, col, *, box=3, ann_in=6, ann_out=12,
                 min_annulus_px=30):
    """Signal-to-noise of a predicted site: a box against its own annulus.

    ``sub`` and ``raw`` are frame stacks in the SAME order (the background-
    subtracted and the raw counts of the live frames), and ``frame`` indexes
    that stack -- map a raw frame number through the live-frame index first.
    Returns ``(snr, integrated)`` or ``None`` when the site cannot be scored:
    too near an edge, any masked pixel in the box, or fewer than
    ``min_annulus_px`` unmasked annulus pixels.

    **Noise is floored at the Poisson width of the local RAW counts.** On a
    photon counter a flat patch of the subtracted stack has MAD = 0, and an
    unfloored MAD gave SNR 2.7e11 on 1140 counts. A flat patch is not certainty.
    """
    if not (0 < box < ann_in < ann_out):
        raise ValueError(f"need 0 < box < ann_in < ann_out, got {box}, {ann_in}, {ann_out}")
    mask = np.asarray(mask, bool)
    nz, ny = mask.shape
    r0, c0 = int(round(float(row))), int(round(float(col)))
    if not (ann_out <= r0 < nz - ann_out and ann_out <= c0 < ny - ann_out):
        return None
    if mask[r0 - box:r0 + box + 1, c0 - box:c0 + box + 1].any():
        return None
    sl = (slice(r0 - ann_out, r0 + ann_out + 1), slice(c0 - ann_out, c0 + ann_out + 1))
    ann = np.asarray(sub[frame][sl], float)
    rawann = np.asarray(raw[frame][sl], float)
    out = np.array(mask[sl], bool, copy=True)
    n = ann_out
    out[n - ann_in:n + ann_in + 1, n - ann_in:n + ann_in + 1] = True   # punch out the signal region
    if int((~out).sum()) < min_annulus_px:
        return None
    vals = ann[~out]
    bg = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - bg)))
    poisson = math.sqrt(max(float(np.median(rawann[~out])), 0.0) + 1.0)
    noise = max(mad, poisson)
    sig = np.asarray(sub[frame][r0 - box:r0 + box + 1, c0 - box:c0 + box + 1], float)
    return (float(sig.max()) - bg) / noise, float(sig.sum() - bg * sig.size)


@dataclass
class TargetedRecovery:
    """Per-site scores at predicted positions, and the same-ring null they must beat."""
    snr: np.ndarray                    # (n_sites,) NaN where the site could not be scored
    integrated: np.ndarray             # (n_sites,) NaN where the site could not be scored
    recovered: np.ndarray              # (n_sites,) scored AND snr >= snr_min
    snr_min: float
    null_hits: int
    null_trials: int

    @property
    def n_scored(self) -> int:
        return int(np.isfinite(self.snr).sum())

    @property
    def n_recovered(self) -> int:
        return int(self.recovered.sum())

    @property
    def null_rate(self) -> float:
        return self.null_hits / max(self.null_trials, 1)

    @property
    def expected(self) -> float:
        """Recoveries expected by chance at the same radii and frames."""
        return self.null_rate * self.n_scored

    @property
    def expected_sd(self) -> float:
        r = self.null_rate
        return math.sqrt(max(self.n_scored * r * (1.0 - r), 1e-9))

    @property
    def excess_sigma(self) -> float:
        return (self.n_recovered - self.expected) / self.expected_sd

    def __str__(self) -> str:
        return (f"{self.n_recovered} of {self.n_scored} scored sites at SNR >= {self.snr_min:g}; "
                f"same-ring null {self.null_hits}/{self.null_trials} = {self.null_rate:.3f} -> "
                f"expected {self.expected:.1f} +- {self.expected_sd:.1f}; "
                f"excess {self.excess_sigma:.1f} sigma")


def targeted_recovery(sub, raw, mask, frame, row, col, *, beam_centre, snr_min=5.0,
                      n_null=24, rng_seed=0, box=3, ann_in=6, ann_out=12):
    """Score predicted reflection sites, against a null drawn on the SAME RING.

    ``frame``, ``row``, ``col`` give each predicted site (frame indexes the
    ``sub``/``raw`` stack, as in :func:`targeted_snr`). ``beam_centre`` is
    ``(row, col)``. Pass only sites the blob finder did NOT already find, and
    -- for several domains -- not sites another domain has taken.

    **The trap: predicted sites lie on powder rings, and a ring has intensity at
    every azimuth.** So every scored site is re-scored ``n_null`` times at the
    SAME radius and SAME frame but a random azimuth: identical radial
    background, identical ring, only the orientation information removed. A
    recovery count that does not exceed that expectation is the ring, not
    recovery -- read :attr:`TargetedRecovery.excess_sigma`, never the raw count.

    This is the weaker of the two nulls the method needs. The decisive one keeps
    the cell, replaces the orientation by a RANDOM rotation, predicts, and
    harvests identically; on 2604 zero of 40 scrambles beat the real rate at any
    of three positions. It needs the caller's predictor, so it is not here --
    run it before quoting a recovery.
    """
    frame = np.asarray(frame).astype(int)
    row = np.asarray(row, float)
    col = np.asarray(col, float)
    if not (frame.shape == row.shape == col.shape) or frame.ndim != 1:
        raise ValueError("frame, row and col must be 1-D arrays of equal length")
    bcr, bcc = (float(v) for v in beam_centre)
    rng = np.random.default_rng(rng_seed)
    n = len(row)
    snr = np.full(n, np.nan)
    integ = np.full(n, np.nan)
    hits = trials = 0
    kw = dict(box=box, ann_in=ann_in, ann_out=ann_out)
    for i in range(n):
        s = targeted_snr(sub, raw, mask, frame[i], row[i], col[i], **kw)
        if s is None:
            continue
        snr[i], integ[i] = s
        rr = math.hypot(row[i] - bcr, col[i] - bcc)
        for _ in range(n_null):
            az = rng.uniform(0.0, 2.0 * math.pi)
            sn = targeted_snr(sub, raw, mask, frame[i], bcr + rr * math.sin(az),
                              bcc + rr * math.cos(az), **kw)
            if sn is None:
                continue
            trials += 1
            hits += int(sn[0] >= snr_min)
    recovered = np.isfinite(snr) & (np.nan_to_num(snr, nan=-np.inf) >= snr_min)
    return TargetedRecovery(snr=snr, integrated=integ, recovered=recovered, snr_min=float(snr_min),
                            null_hits=int(hits), null_trials=int(trials))
