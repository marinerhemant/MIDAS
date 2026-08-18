"""Friedel pairing and spot-to-grain assignment for DCT.

Phase 2 of ``implementation_plan.md``.

Friedel pairs
-------------
``G`` and ``-G`` flash exactly 180 deg apart in omega -- proved in
``tests/test_scan.py``, and the reason is simple: rotating a further half turn
negates the in-plane part of ``G``, which is the only part omega touches. So the
two members of a pair are the *same lattice plane* seen from opposite sides, and

* in the **lab** frame the partner's scattering vector is the *mirror* of the
  original through the horizontal plane containing the beam -- same ``(x, y)``,
  negated ``z`` -- **not** its negative. Verified numerically; the lab-frame
  vectors have ``cos = +0.79`` for a typical reflection, so testing lab-frame
  antiparallelism finds no pairs at all. The antiparallel relation holds in the
  **sample** frame, which is why :class:`~midas_dct_tt.scan.BraggFlash` carries
  ``G_sample`` and this module matches on it;
* consequently the two beams are mirrored vertically about the direct beam,
  which is what lets a pair localise the grain along the beam by triangulation
  -- the standard DCT trick;
* their **missing cones point in opposite directions**, so a Friedel-paired TT
  measurement recovers part of what one setting cannot see (Section 2 of the
  plan).

Pairing is a **discrete** operation. It is deliberately off-gradient (plain
Python, integer indices) per design principle 3: an assignment that changed
smoothly with the data would be neither correct nor interpretable.

Spot assignment
---------------
:func:`assign_spots` is nearest-prediction matching under explicit tolerances,
greedy on match quality. It is not a solved indexing problem and does not pretend
to be: it is the piece needed to score a synthetic reconstruction, and it reports
what it could not assign rather than silently dropping it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["Spot", "assign_spots", "friedel_partner_omega", "friedel_pairs", "unassigned"]


def friedel_partner_omega(omega_deg: float) -> float:
    """The omega at which ``-G`` flashes, given where ``G`` does: ``omega + 180``."""
    return (float(omega_deg) + 180.0) % 360.0


def friedel_pairs(flashes, *, omega_tol_deg: float = 0.5, direction_tol: float = 1e-3):
    """Index pairs ``(i, j)`` of flashes that are Friedel partners.

    Two flashes pair when their omegas differ by 180 deg (within
    ``omega_tol_deg``) **and** their **sample-frame** scattering vectors are
    antiparallel (within ``direction_tol``, as ``1 + cos`` between the unit
    vectors). Both conditions are required: the omega test alone would pair
    unrelated reflections that happen to flash half a turn apart, which in a real
    dataset is common.

    Matching in the sample frame is not a detail. In the lab frame a Friedel
    partner is a *mirror* through the horizontal plane, not an inversion, so a
    lab-frame antiparallel test silently returns no pairs at all.

    Each flash appears in at most one pair; ties are broken by the smaller
    combined mismatch. Returns a list of ``(i, j)`` with ``i < j``.
    """
    n = len(flashes)
    if any(f.G_sample is None for f in flashes):
        raise ValueError(
            "friedel_pairs needs G_sample on every flash (populated by "
            "bragg_flashes); lab-frame vectors cannot express the relation"
        )
    candidates = []
    for i in range(n):
        gi = flashes[i].G_sample
        gi_hat = gi / torch.linalg.vector_norm(gi)
        for j in range(i + 1, n):
            gj = flashes[j].G_sample
            d_om = abs((flashes[j].omega_deg - flashes[i].omega_deg) % 360.0 - 180.0)
            if d_om > omega_tol_deg:
                continue
            gj_hat = gj / torch.linalg.vector_norm(gj)
            anti = float(1.0 + torch.dot(gi_hat, gj_hat))     # 0 when antiparallel
            if anti > direction_tol:
                continue
            candidates.append((d_om / max(omega_tol_deg, 1e-12) + anti, i, j))

    candidates.sort()
    used, pairs = set(), []
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((i, j))
    return sorted(pairs)


@dataclass
class Spot:
    """One observed or predicted DCT spot.

    ``grain`` and ``hkl`` are populated for predictions and are what an
    assignment resolves for an observation.
    """

    omega_deg: float
    u_px: float
    v_px: float
    intensity: float = 1.0
    grain: int = None
    hkl: tuple = None


def assign_spots(observed, predicted, *, omega_tol_deg: float = 1.0,
                 pixel_tol: float = 5.0):
    """Greedily match observed spots to predictions within explicit tolerances.

    Returns a list, one entry per observed spot: the index into ``predicted``, or
    ``None`` where nothing matched inside tolerance. Matching is greedy on a
    normalised distance (omega and pixel residuals each scaled by their own
    tolerance), and each prediction is used at most once.

    Unmatched spots are reported as ``None`` rather than forced to their nearest
    neighbour. In DCT a real frame contains spots from grains that were never in
    the candidate list, and an assignment that always succeeds is an assignment
    that cannot tell you the grain map is incomplete -- use :func:`unassigned` to
    count them.
    """
    scored = []
    for oi, o in enumerate(observed):
        for pi, p in enumerate(predicted):
            d_om = abs((o.omega_deg - p.omega_deg + 180.0) % 360.0 - 180.0)
            if d_om > omega_tol_deg:
                continue
            d_px = math.hypot(o.u_px - p.u_px, o.v_px - p.v_px)
            if d_px > pixel_tol:
                continue
            scored.append((d_om / omega_tol_deg + d_px / pixel_tol, oi, pi))

    scored.sort()
    out = [None] * len(observed)
    taken = set()
    for _, oi, pi in scored:
        if out[oi] is not None or pi in taken:
            continue
        out[oi] = pi
        taken.add(pi)
    return out


def unassigned(assignment) -> int:
    """How many observed spots found no prediction. Report this; do not hide it."""
    return sum(1 for a in assignment if a is None)
