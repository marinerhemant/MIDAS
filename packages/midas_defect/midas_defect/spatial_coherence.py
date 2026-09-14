"""Spatial coherence across a raster: use ONE position's found orientation to
recover the same grain at ANOTHER position where it was too weak to survive
that position's own free search -- and gate the recovery correctly.

The design (2026-09-14) came out of a direct back-and-forth about what
"properly gated" means for THIS question, which is easy to get wrong by
importing the wrong gate from next door:

**This module answers "does this orientation exist here", not "is this
domain's cell precise".** Those are different questions with very different
statistical power on real raster data. Orientation is a HIGH-LEVERAGE
parameter -- a few degrees of misorientation moves every predicted spot by
many pixels, so a wrong orientation matches essentially nothing (a
random-orientation null on real 2604 data: 0 successes in 55 draws across 5
positions -- see the project notes). Cell length is comparatively LOW-LEVERAGE
here -- forcing a genuinely-planted split to zero moved predictions by ~1.85 px
on synthetic ground truth but only ~0.1-0.3 px on real data, i.e. a few
percent of cell length is close to the noise floor. A cell-precision gate
(``honesty.decoy_test`` with inflated-cell decoys) is the RIGHT tool for
:mod:`midas_hkls.cell_constrained` / a/b-splitting work, and the WRONG tool
here: it fails almost everything on this raster, INCLUDING pass-1's own
native, already-accepted domains (a 60-of-60-reflection near-duplicate cross
-position match still fails it) -- so failing it is not informative about
whether an orientation is real, only about whether this raster's cell lengths
are precisely determined (they mostly are not, for reasons already documented
elsewhere in this project). Do not resurrect it here; if you want a
cell-precision check on a recovered domain's OWN fitted cell, that is a
separate, later analysis using :mod:`midas_hkls.cell_constrained` machinery,
run on the DomainData this module hands you.

**Orientation dedup is symmetry-aware and reuses the existing fast path**
(:func:`midas_defect.rows._misorientation_422`, already validated against
``midas_stress.misorientation_om`` to 9e-14 degrees for space group 139, with
a documented ~8x speed advantage there) rather than reimplementing it or
calling the slower canonical routine directly.

**Mode is "seed-and-refit", not "forced check on unclaimed spots only".**
Every candidate is seeded against a target position's FULL spot cloud via
:func:`midas_defect.rows.refine_to_convergence`, always starting from the SAME
fixed nominal cell (never a neighbour's own fitted cell -- gate to a fixed
nominal reference, `refine_to_convergence`'s own documented lesson). Checked
directly on real data: the converged orientation stays within ~1-1.5 degrees
of the SEED, not the position's own locally-dominant domain, even when the
seed started 20-30 degrees from anything free search ever found there -- so
this is genuinely testing the candidate, not just basin-hopping to whatever's
locally strong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import Geometry, qlab_to_qsample
from .raster import reduce_one_position, PositionResult, _ingest_position
from .rows import refine_to_convergence, _misorientation_422
from midas_hkls.cell_constrained import DomainData

__all__ = ["Candidate", "RecoveredDomain", "PositionCoherence", "RasterCoherenceResult",
          "recover_domains_across_raster"]


@dataclass
class Candidate:
    """One deduplicated candidate orientation, with provenance to its best
    (highest-n) pass-1 origin."""
    U: np.ndarray
    origin_point: int
    n_origin: int
    branch: str
    members: List[Tuple[int, int]]      # (point, dom_idx) of every pass-1 domain folded into this cluster


@dataclass
class RecoveredDomain:
    """A candidate confirmed at a position where it was NOT already native
    (nearest pass-1 domain at that position was >= dedup_threshold_deg away),
    via seed-and-refit, accepted on the random-orientation null only."""
    point: int
    origin_point: int
    n: int
    U: np.ndarray
    cell: Tuple[float, float, float]
    hkl: np.ndarray            # (n, 3) -- claimed subset only
    g: np.ndarray               # (n, 3) -- claimed subset only, q-sample frame
    nearest_pass1_deg: float
    null_threshold: float


@dataclass
class PositionCoherence:
    point: int
    res: PositionResult
    q_all: np.ndarray
    null_counts: List[int]
    null_threshold: float
    recovered: List[RecoveredDomain] = field(default_factory=list)

    @property
    def null_is_suspect(self) -> bool:
        """True when this position's own random-orientation null threshold is
        implausibly close to a real domain's size -- a failure mode found and
        reproduced exactly 2026-09-14 (notebook 06's synthetic demo, a
        307-reflection native domain): 2 of 20 random-orientation null draws
        there basin-hopped all the way to the FULL true domain via
        `refine_to_convergence`'s own iterate-refit loop -- a ~10%
        false-full-convergence rate at that position, not a one-off fluke
        (replayed with the exact RNG state and confirmed). It is RARE and
        position-specific, not a universal property of dense domains -- most
        positions in the same run had a null threshold near 0 -- which is
        exactly what makes it dangerous: a low `n_null_draws` (the default
        here trades speed for this) can let one or two such lucky draws
        dominate the empirical 95th percentile and push it to whatever that
        draw happened to converge onto. It did not produce a false RECOVERY
        in the case found (nothing else could exceed a 307 threshold either),
        but the gate had gone inert -- check this before trusting either a
        recovery OR an absence of one at a flagged position; do not trust the
        auto-computed threshold blind, and consider raising `n_null_draws`
        for a position that flags.
        """
        if not self.res.domains.domains:
            return False
        largest_native = max(d.n for d in self.res.domains.domains)
        return largest_native > 0 and self.null_threshold > 0.3 * largest_native


@dataclass
class RasterCoherenceResult:
    dedup_threshold_deg: float
    candidates: List[Candidate]
    per_position: Dict[int, PositionCoherence]

    def to_domain_data(self, *, include_native: bool = True,
                       include_recovered: bool = True) -> List[DomainData]:
        """Flatten native + accepted-recovered domains across the WHOLE raster
        into one list, ready for :func:`midas_hkls.cell_constrained.refine_cell_joint`
        or ``refine_cell_radial`` (and their ``_robust`` variants) -- both
        already jointly fit several ``DomainData`` against one shared cell,
        which is the whole point: one recovery pipeline serves both fits.

        The ``g`` in every returned ``DomainData`` is in this project's
        ``q = 2*pi/d`` convention (``geometry.qlab_to_qsample``'s own units,
        unchanged) -- pass ``two_pi=True`` to whichever ``refine_cell_*``
        call consumes this list, or the fit will silently converge on a
        wrong cell rather than raise (caught once, in this module's own
        tests: an omitted ``two_pi=True`` fit ``a`` off by a factor of ~6,
        no exception).
        """
        out = []
        for p, pc in self.per_position.items():
            if include_native:
                for i, dom in enumerate(pc.res.domains.domains):
                    out.append(DomainData(hkl=np.asarray(dom.hkl, float),
                                          g=pc.q_all[dom.claim],
                                          label=f"p{p}_native{i}"))
            if include_recovered:
                for rec in pc.recovered:
                    out.append(DomainData(hkl=np.asarray(rec.hkl, float), g=rec.g,
                                          label=f"p{p}_from_p{rec.origin_point}"))
        return out

    def summary(self) -> str:
        lines = [f"{len(self.candidates)} candidate orientation(s) after "
                f"{self.dedup_threshold_deg}deg dedup"]
        for p, pc in self.per_position.items():
            n_native = len(pc.res.domains.domains)
            flag = "  ** SUSPECT NULL -- see .null_is_suspect docstring, do not trust blind **" \
                if pc.null_is_suspect else ""
            lines.append(f"  p{p}: {n_native} native + {len(pc.recovered)} recovered "
                        f"(null threshold={pc.null_threshold:.1f} from {len(pc.null_counts)} draws){flag}")
        return "\n".join(lines)


def _misorient_deg(U1: np.ndarray, U2: np.ndarray, space_group_number: int) -> float:
    return _misorientation_422(np.asarray(U1, float), np.asarray(U2, float), space_group_number)


def _dedup(all_candidates: List[dict], threshold_deg: float, space_group_number: int) -> List[List[dict]]:
    remaining = sorted(all_candidates, key=lambda c: -c["n"])
    clusters: List[List[dict]] = []
    for c in remaining:
        placed = False
        for cluster in clusters:
            if _misorient_deg(c["U"], cluster[0]["U"], space_group_number) < threshold_deg:
                cluster.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])
    return clusters


def recover_domains_across_raster(
    loader: Callable[[int], np.ndarray], geom: Geometry, point_indices: Sequence[int], *,
    a: float, c: float, space_group_number: int,
    sigma_rtn: Tuple[float, float, float],
    dedup_threshold_deg: float = 1.0,
    n_null_draws: int = 40,
    null_alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    reduce_kwargs: Optional[dict] = None,
) -> RasterCoherenceResult:
    """Pass 1 (free search, every position) -> dedup orientations (symmetry-
    aware, ORIENTATION ONLY -- cell is deliberately not part of dedup) -> pass
    2 (seed-and-refit every candidate at every OTHER position, nominal-cell
    seeded) -> accept on a random-orientation null specific to that position
    (NOT a fixed count -- noise-capture rate can differ by position).

    ``n_null_draws`` random orientations (``scipy.spatial.transform.Rotation
    .random``) are seeded at each position with the SAME real spot cloud (not
    scrambled -- scrambling destroys the Bragg-condition structure entirely
    and makes the null trivially pass; see this module's docstring). Match
    ``n_null_draws`` to roughly how many real candidates you are actually
    testing per position -- this is a look-elsewhere-corrected threshold, not
    a fixed significance level, so it should scale with how many looks you
    take, the same principle as :func:`midas_defect.rows.search_null`.

    Returns a :class:`RasterCoherenceResult`; call ``.to_domain_data()`` to
    get the combined list ready for `midas_hkls.cell_constrained.refine_cell_joint`
    / ``refine_cell_radial``.
    """
    from scipy.spatial.transform import Rotation

    rng = rng or np.random.default_rng(0)
    reduce_kwargs = dict(reduce_kwargs or {})

    per_position: Dict[int, PositionCoherence] = {}
    all_candidates: List[dict] = []
    for p in point_indices:
        frames = loader(p)
        res = reduce_one_position(frames, geom, a=a, c=c, space_group_number=space_group_number,
                                  sigma_rtn=sigma_rtn, point=p, **reduce_kwargs)
        spots, qlab, omega_deg, mask, sub, _ = _ingest_position(frames, geom)
        import torch
        q_all = qlab_to_qsample(qlab, torch.deg2rad(torch.as_tensor(
            res.omega_sign.chosen_sign * omega_deg, dtype=qlab.dtype))).detach().cpu().numpy()
        per_position[p] = PositionCoherence(point=p, res=res, q_all=q_all,
                                            null_counts=[], null_threshold=float("nan"))
        for i, dom in enumerate(res.domains.domains):
            all_candidates.append(dict(point=p, dom_idx=i, U=np.asarray(dom.U, float),
                                       n=dom.n, branch=dom.branch))

    clusters = _dedup(all_candidates, dedup_threshold_deg, space_group_number)
    candidates = [Candidate(U=(best := max(cl, key=lambda c: c["n"]))["U"],
                            origin_point=best["point"], n_origin=best["n"], branch=best["branch"],
                            members=[(m["point"], m["dom_idx"]) for m in cl])
                 for cl in clusters]

    for p, pc in per_position.items():
        null_counts = []
        for rmat in Rotation.random(n_null_draws, random_state=rng.integers(0, 2**31)).as_matrix():
            r = refine_to_convergence(pc.q_all, rmat, a0=a, c0=c,
                                      space_group_number=space_group_number, sigma_rtn=sigma_rtn)
            null_counts.append(int(r.claim.sum()) if r is not None else 0)
        pc.null_counts = null_counts
        pc.null_threshold = float(np.quantile(null_counts, 1.0 - null_alpha))

        for cand in candidates:
            nearest_deg = min(_misorient_deg(cand.U, d.U, space_group_number)
                              for d in pc.res.domains.domains) if pc.res.domains.domains else float("inf")
            if nearest_deg < dedup_threshold_deg:
                continue          # already native here -- nothing to recover
            result = refine_to_convergence(pc.q_all, cand.U, a0=a, c0=c,
                                           space_group_number=space_group_number, sigma_rtn=sigma_rtn)
            if result is None:
                continue
            n_found = int(result.claim.sum())
            if n_found <= pc.null_threshold:
                continue
            pc.recovered.append(RecoveredDomain(
                point=p, origin_point=cand.origin_point, n=n_found, U=result.lat.U,
                cell=(result.lat.a, result.lat.b, result.lat.c),
                hkl=result.hkl[result.claim], g=pc.q_all[result.claim],
                nearest_pass1_deg=nearest_deg, null_threshold=pc.null_threshold))

    return RasterCoherenceResult(dedup_threshold_deg=dedup_threshold_deg,
                                 candidates=candidates, per_position=per_position)
