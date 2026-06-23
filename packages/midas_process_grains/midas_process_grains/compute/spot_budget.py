"""Spot-budget enforcement — the volume-independent grain-count check.

Motivation
----------
The Stage-8.5 *volume* budget (``drop_policy``) compares ``ΣV_grain`` to the
illuminated sample volume. It is the right idea but the wrong observable for
counting: the NNLS per-grain volumes are systematically inflated (a ~1.6×
radius bias pushes the packing fraction to 4–4.5× even when the *count* is
plausible), so the volume budget cannot cleanly separate "too many grains"
from "each grain's volume overestimated".

The diffracted-spot population does not have that ambiguity. Every observed
reflection carries a fixed, measured intensity and can be *spent* by at most
one physical grain — **except** between grains that legitimately share
reflections: twin variants (Σ3/Σ9/… share a common-plane set) and spatially
coincident grains. So the spot budget is the clean, volume-independent test
of whether a grain population is real or inflated by coincidental/over-split
groupings:

* if the population is real, almost every observed spot is claimed once
  (or co-claimed only *within* a twin family);
* if it is inflated, spots are claimed by multiple *unrelated* grains —
  the "we randomly assigned spots together to form orientations" signature.

This module assigns every observed spot to the single highest-quality
grain (or twin family) that claims it, then

1. **decomposes** the per-spot multiplicity into

   * claimed once (clean),
   * co-claimed *within* one twin family (legitimate twin sharing), and
   * contested *between* unrelated families (over-subscription / over-split);

2. **enforces** the budget by keeping only grains that retain a sufficient
   fraction of *owned* (won or twin-co-owned) spots; and

3. reports the leaf- and family-level **spot overcount ratios**
   (``Σ spots-per-grain / N_observed`` and ``Σ family-union / N_observed``)
   which are the spot analogue of the volume packing fraction.

All quantities are computed from the matched-spot sets (``Results/
ProcessKey.bin`` → one ``set[int]`` of SpotIDs per grain), the per-grain
quality score, and the Stage-5 twin-family labels. No geometry, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class SpotBudgetResult:
    """Result of :func:`enforce_spot_budget`.

    Attributes
    ----------
    keep_mask : (N,) bool
        ``True`` for grains that survive spot-budget enforcement.
    owned_spots : (N,) int64
        Per-grain count of spots the grain won outright or co-owns through
        its twin family.
    claimed_spots : (N,) int64
        Per-grain count of spots claimed (= ``len(spot_set)``).
    owned_fraction : (N,) float
        ``owned_spots / claimed_spots`` (0 where a grain claims nothing).
    family_id : (N,) int64
        Dense family index used internally (singletons each unique).
    n_observed_spots : int
        Number of distinct SpotIDs claimed by *any* grain (the budget).
    n_claimed_once : int
        Spots claimed by exactly one grain.
    n_shared_within_family : int
        Spots claimed by >1 grain, all in the *same* twin family
        (legitimate twin/overlap sharing).
    n_contested_between_families : int
        Spots claimed by grains spanning >=2 *unrelated* families — the
        over-subscription / over-split signature.
    mean_multiplicity_grain : float
        ``Σ claimed_spots / n_observed_spots`` — leaf spot overcount.
    mean_multiplicity_family : float
        ``Σ (per-family union size) / n_observed_spots`` — twin-credited
        spot overcount (collapsing twin families to one claimant each).
    n_kept : int
    spot_budget_count : int
        Twin-credited spot-budget *capacity*: ``n_observed_spots`` divided
        by the mean family-union size — the number of grain-families the
        detector can support if each spot is spent once. A scale reference,
        independent of the keep policy.
    """

    keep_mask: np.ndarray
    owned_spots: np.ndarray
    claimed_spots: np.ndarray
    owned_fraction: np.ndarray
    family_id: np.ndarray
    n_observed_spots: int
    n_claimed_once: int
    n_shared_within_family: int
    n_contested_between_families: int
    mean_multiplicity_grain: float
    mean_multiplicity_family: float
    n_kept: int
    spot_budget_count: int


@dataclass
class SpotCoverageResult:
    """Result of :func:`enforce_spot_budget_coverage`.

    Attributes
    ----------
    keep_mask : (N,) bool
    n_kept : int
    n_observed_spots : int
        Distinct SpotIDs claimed by *any* grain (the full observed set).
    n_covered_spots : int
        Distinct SpotIDs covered by the *kept* grains.
    coverage_fraction : float
        ``n_covered / n_observed``.
    mean_multiplicity_kept : float
        ``Σ claimed_by_kept / n_covered`` — the realised per-spot sharing of
        the kept population (compare to the physical ~1.5× Friedel/twin rate).
    multiplicity_cap : int
        The per-spot claim ceiling that was used.
    """

    keep_mask: np.ndarray
    n_kept: int
    n_observed_spots: int
    n_covered_spots: int
    coverage_fraction: float
    mean_multiplicity_kept: float
    multiplicity_cap: int


def enforce_spot_budget_coverage(
    *,
    spot_sets: Sequence[set],
    quality_score: np.ndarray,
    multiplicity_cap: int = 2,
    min_new_spots: int = 5,
    min_new_fraction: float = 0.5,
) -> SpotCoverageResult:
    """Coverage-greedy spot-budget enforcement with a physical sharing allowance.

    Unlike :func:`enforce_spot_budget` (winner-take-all, one grain per spot),
    this admits the legitimate Friedel/twin/overlap sharing seen in real data
    (C-pipeline SpotMatrix shows ~1.5× per-spot multiplicity). Grains are
    visited in descending quality; each maintains a per-spot claim counter. A
    grain is **kept** iff a sufficient share of its spots are still
    *unsaturated* (claimed fewer than ``multiplicity_cap`` times); otherwise it
    is a redundant over-split copy and is dropped.

    The kept population covers ~all observed spots while holding per-spot
    multiplicity near the physical sharing rate — the count of *distinct*
    grains the diffraction pattern actually supports.

    Parameters
    ----------
    spot_sets : length-N sequence of ``set[int]``
    quality_score : (N,) float
        Higher = better; visited first. NaN ranks lowest.
    multiplicity_cap : int, default 2
        Max number of kept grains permitted to claim a single spot. ``2``
        admits Friedel/twin pairs; raise for heavily-overlapped data.
    min_new_spots : int, default 5
        Absolute floor of unsaturated spots a grain must add to be kept.
    min_new_fraction : float, default 0.5
        A grain must have at least this fraction of its spots unsaturated.
    """
    N = len(spot_sets)
    Q = np.asarray(quality_score, dtype=np.float64)
    if N == 0:
        return SpotCoverageResult(
            keep_mask=np.zeros(0, dtype=bool), n_kept=0, n_observed_spots=0,
            n_covered_spots=0, coverage_fraction=0.0,
            mean_multiplicity_kept=0.0, multiplicity_cap=multiplicity_cap,
        )
    qfill = np.where(np.isfinite(Q), Q, -np.inf)
    order = np.argsort(-qfill, kind="stable")

    spot_claims: dict[int, int] = {}
    keep = np.zeros(N, dtype=bool)
    for g in order:
        ss = spot_sets[g]
        if not ss:
            continue
        unsat = sum(1 for s in ss if spot_claims.get(s, 0) < multiplicity_cap)
        frac = unsat / len(ss)
        if unsat >= min_new_spots and frac >= min_new_fraction:
            keep[g] = True
            for s in ss:
                spot_claims[s] = spot_claims.get(s, 0) + 1

    # observed = union of ALL spot sets; covered = spots a kept grain touches
    all_spots: set = set()
    for ss in spot_sets:
        all_spots |= ss
    n_observed = len(all_spots)
    n_covered = len(spot_claims)
    sum_kept_claims = sum(spot_claims.values())
    return SpotCoverageResult(
        keep_mask=keep,
        n_kept=int(keep.sum()),
        n_observed_spots=n_observed,
        n_covered_spots=n_covered,
        coverage_fraction=(n_covered / n_observed) if n_observed else 0.0,
        mean_multiplicity_kept=(sum_kept_claims / n_covered) if n_covered else 0.0,
        multiplicity_cap=multiplicity_cap,
    )


def _dense_family_ids(twin_family_id: np.ndarray) -> np.ndarray:
    """Map raw twin-family labels to dense ids; singletons (<0 or NaN) each
    get their own unique family."""
    fam_raw = np.asarray(twin_family_id, dtype=np.float64)
    N = fam_raw.shape[0]
    is_single = ~np.isfinite(fam_raw) | (fam_raw < 0)
    multi_ids = np.unique(fam_raw[~is_single].astype(np.int64))
    id_map = {int(g): k for k, g in enumerate(multi_ids)}
    n_multi = len(multi_ids)
    fam_dense = np.empty(N, dtype=np.int64)
    nxt = n_multi
    for i in range(N):
        if is_single[i]:
            fam_dense[i] = nxt
            nxt += 1
        else:
            fam_dense[i] = id_map[int(fam_raw[i])]
    return fam_dense


def enforce_spot_budget(
    *,
    spot_sets: Sequence[set],
    quality_score: np.ndarray,
    twin_family_id: np.ndarray,
    min_owned_spots: int = 5,
    min_owned_fraction: float = 0.5,
) -> SpotBudgetResult:
    """Assign every observed spot to its best (highest-quality) claimant,
    crediting twin families, and enforce the spot budget.

    Parameters
    ----------
    spot_sets : length-N sequence of ``set[int]``
        Matched SpotIDs per grain (final-grain union sets).
    quality_score : (N,) float
        Per-grain quality; higher = better. A grain wins a contested spot
        if its family holds the highest score among claimants. NaN ranks
        lowest.
    twin_family_id : (N,) int
        Stage-5 twin-family labels. ``< 0``/NaN → singleton.
    min_owned_spots : int, default 5
        A grain must retain at least this many owned spots to survive.
    min_owned_fraction : float, default 0.5
        A grain must retain at least this fraction of its *claimed* spots as
        owned (won or twin-co-owned) to survive. This is the over-attribution
        guard: a grain whose spots were mostly won by unrelated, higher-quality
        grains is over-split and is dropped.

    Returns
    -------
    SpotBudgetResult
    """
    N = len(spot_sets)
    Q = np.asarray(quality_score, dtype=np.float64)
    if N == 0:
        z = np.zeros(0)
        return SpotBudgetResult(
            keep_mask=z.astype(bool), owned_spots=z.astype(np.int64),
            claimed_spots=z.astype(np.int64), owned_fraction=z.astype(np.float64),
            family_id=z.astype(np.int64), n_observed_spots=0, n_claimed_once=0,
            n_shared_within_family=0, n_contested_between_families=0,
            mean_multiplicity_grain=0.0, mean_multiplicity_family=0.0,
            n_kept=0, spot_budget_count=0,
        )

    fam = _dense_family_ids(twin_family_id)
    qfill = np.where(np.isfinite(Q), Q, -np.inf)

    # spot -> list of grain indices claiming it
    claims: dict[int, list[int]] = {}
    for g, ss in enumerate(spot_sets):
        for s in ss:
            claims.setdefault(s, []).append(g)

    n_observed = len(claims)
    owned = np.zeros(N, dtype=np.int64)
    claimed = np.array([len(ss) for ss in spot_sets], dtype=np.int64)

    n_once = 0
    n_within = 0
    n_between = 0
    # per-family union spot counter (for twin-credited overcount)
    family_union: dict[int, int] = {}

    for s, gs in claims.items():
        fams_here = {fam[g] for g in gs}
        for f in fams_here:
            family_union[f] = family_union.get(f, 0) + 1

        if len(gs) == 1:
            n_once += 1
            owned[gs[0]] += 1
            continue
        if len(fams_here) == 1:
            # legitimate twin/overlap sharing: all co-own
            n_within += 1
            for g in gs:
                owned[g] += 1
            continue
        # contested between >=2 unrelated families: award to best family
        n_between += 1
        # best family = the one whose best member has the highest quality
        best_fam = max(fams_here,
                       key=lambda f: max(qfill[g] for g in gs if fam[g] == f))
        for g in gs:
            if fam[g] == best_fam:
                owned[g] += 1

    owned_frac = np.where(claimed > 0, owned / np.maximum(claimed, 1), 0.0)
    keep = (owned >= min_owned_spots) & (owned_frac >= min_owned_fraction)

    sum_claimed = int(claimed.sum())
    sum_family_union = int(sum(family_union.values()))
    mean_mult_grain = sum_claimed / n_observed if n_observed else 0.0
    mean_mult_family = sum_family_union / n_observed if n_observed else 0.0
    # twin-credited capacity: how many families the detector supports if each
    # spot is spent once = n_observed / (mean family-union size).
    n_families = len(set(fam.tolist()))
    mean_union_size = sum_family_union / n_families if n_families else 0.0
    spot_budget_count = int(round(n_observed / mean_union_size)) if mean_union_size else 0

    return SpotBudgetResult(
        keep_mask=keep,
        owned_spots=owned,
        claimed_spots=claimed,
        owned_fraction=owned_frac,
        family_id=fam,
        n_observed_spots=n_observed,
        n_claimed_once=n_once,
        n_shared_within_family=n_within,
        n_contested_between_families=n_between,
        mean_multiplicity_grain=float(mean_mult_grain),
        mean_multiplicity_family=float(mean_mult_family),
        n_kept=int(keep.sum()),
        spot_budget_count=spot_budget_count,
    )
