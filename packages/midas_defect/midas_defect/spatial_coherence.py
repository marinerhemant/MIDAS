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
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import Geometry, qlab_to_qsample
from .raster import reduce_one_position, PositionResult, IngestBundle, _ingest_position
from .rows import refine_to_convergence, _misorientation_422
from midas_hkls import ab_separable, partner_multiplicity, index_asymmetry
from midas_hkls.cell_constrained import DomainData, refine_cell_radial_robust, split_with_error

__all__ = ["Candidate", "RecoveredDomain", "PositionCoherence", "RasterCoherenceResult",
          "recover_domains_across_raster",
          "morans_i", "PermutationResult", "permutation_null_morans_i",
          "cluster_permutation_null_morans_i", "exact_rank_p_value",
          "planted_identical_cell_delta",
          "best_domain_per_position", "grid_quantity", "values_to_grid",
          "SpatialCoherenceReport", "spatial_coherence_report", "grid_coherence_report"]


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
    via seed-and-refit, accepted on the random-orientation null only.

    ``ab_gate`` is the SAME ``ab_separable`` / ``partner_multiplicity`` /
    ``index_asymmetry`` gate `raster.py`'s ``reduce_one_position`` already
    runs on every native domain (``ENVELOPE.md`` #3, #14) -- computed here too
    (2026-09-15) because the random-orientation null answers a different
    question ("is this the right orientation at all") than index_asymmetry
    answers ("is the matched-reflection subset systematically biased"), and a
    real DAC raster found NEITHER question screened off by the other: a
    horizontal-vs-vertical recovery-count asymmetry that looked like real
    spatial-coherence evidence (33 vs 20 recoveries, 2604_25K) turned out to
    be driven by index_asymmetry present raster-wide (56-80% of domains,
    BOTH directions) with every null_threshold near 0 and every recovery
    passing it trivially -- `/verify` REFUTED (claim 538cfb72ae49). The null
    gate was working exactly as designed; it was simply never going to catch
    this failure mode, because it isn't the one it tests for."""
    point: int
    origin_point: int
    n: int
    U: np.ndarray
    cell: Tuple[float, float, float]
    hkl: np.ndarray            # (n, 3) -- claimed subset only
    g: np.ndarray               # (n, 3) -- claimed subset only, q-sample frame
    nearest_pass1_deg: float
    null_threshold: float
    ab_gate: dict = field(default_factory=dict)

    @property
    def index_asymmetry_is_suspect(self) -> bool:
        """True when this recovery's own ``ab_gate`` shows the extreme-skew
        signature (ratio far from 1, or too few reflections to even score) --
        the same 0.2/5.0 threshold `raster.py` already uses, at which the
        2604 raster's own documented systematic sits (ratio 0.0/inf almost
        everywhere it was checked). Does NOT mean the recovery is fake -- a
        real crystal on a limited omega wedge can carry this too (ENVELOPE.md
        #3) -- it means the recovery is not evidence of spatial coherence on
        its own, and needs the raster-wide check before it is quoted as one.
        """
        return _ab_gate_is_suspect(self.ab_gate)


def _ab_gate_is_suspect(ab_gate: dict) -> bool:
    """Shared logic behind `RecoveredDomain.index_asymmetry_is_suspect`, factored out
    (2026-09-16, /verify claim b8f1772140ae's physics lens) so a NATIVE domain's own
    index_asymmetry can be checked with the identical rule, not skipped. Found directly: a
    real `/verify` refutation attempt on the S5 raster noted `best_domain_per_position` filtered
    `index_asymmetry_is_suspect` on recovered domains only, silently letting an equally-suspect
    NATIVE domain win at a position and count as "clean" -- the exact failure mode this whole
    gate exists to catch, just on the other branch."""
    stage = ab_gate.get("stage")
    if stage in ("ab_separable", "partner_multiplicity"):
        return True
    ratio = ab_gate.get("reason", {}).get("ratio") if isinstance(
        ab_gate.get("reason"), dict) else None
    return ratio is not None and (ratio < 0.2 or ratio > 5.0)


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
            n_asym = sum(r.index_asymmetry_is_suspect for r in pc.recovered)
            if n_asym:
                flag += (f"  ** {n_asym}/{len(pc.recovered)} recovered domain(s) have suspect "
                        "index_asymmetry -- see .index_asymmetry_is_suspect docstring, not "
                        "evidence of spatial coherence on their own **")
            lines.append(f"  p{p}: {n_native} native + {len(pc.recovered)} recovered "
                        f"(null threshold={pc.null_threshold:.1f} from {len(pc.null_counts)} draws){flag}")
        return "\n".join(lines)


def _misorient_deg(U1: np.ndarray, U2: np.ndarray, space_group_number: int) -> float:
    return _misorientation_422(np.asarray(U1, float), np.asarray(U2, float), space_group_number)


def _ab_gate(hkl: np.ndarray) -> dict:
    """Same three-stage gate as `raster.py`'s native-domain path (rank, then
    partner multiplicity, then index_asymmetry) -- factored out so a
    recovered domain gets the identical check, not a reimplementation that
    could drift from it."""
    if not ab_separable(hkl):
        return {"stage": "ab_separable",
               "reason": "rank < 2: this domain cannot separate a from b"}
    pm = partner_multiplicity(hkl)
    if pm and all(min(nf, nr) <= 1 for nf, nr in pm.values()):
        return {"stage": "partner_multiplicity",
               "reason": f"every a/b pair rests on <= 1 spot on one side: {pm}"}
    return {"stage": "index_asymmetry", "reason": index_asymmetry(hkl)}


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


def _pass1_one_position(p: int, loader: Callable[[int], np.ndarray], geom: Geometry,
                        a: float, c: float, space_group_number: int,
                        sigma_rtn: Tuple[float, float, float], reduce_kwargs: dict,
                        cache_dir: Optional[str], save_dense: bool):
    """One position's ingest + free search -- the unit of work pass 1 dispatches, either
    directly in a for-loop (``n_workers`` unset) or via ``ProcessPoolExecutor`` (``n_workers``
    set). Returns ``(p, q_all, res, candidate_entries)`` -- deliberately NOT the ingest bundle
    or dense array, which pass 2 never touches and would be an expensive, pointless pickle
    across a process boundary.

    Module-level (not a closure) so it is picklable under the ``spawn`` start method the
    parallel path uses -- a local closure over e.g. a captured ``frames`` dict, fine for the
    tests' own in-memory ``loader``, would NOT survive ``spawn`` in the parallel path; a real
    ``loader`` for parallel use must itself be a plain, importable, module-level callable
    (a per-position file read keyed only on ``p``, not a closure over local state).
    """
    import torch
    frames = loader(p)
    # ingest ONCE, feed the SAME bundle to reduce_one_position -- this used to call
    # _ingest_position a second time on the same frames purely to get qlab/omega_deg,
    # redundantly redoing the expensive background-subtraction + 3-D blob-finding step
    # reduce_one_position had just done internally. Found 2026-09-14.
    cache_path = Path(cache_dir) / f"position_{p}.h5" if cache_dir else None
    if cache_path is not None and cache_path.exists():
        from .persistence import load_ingest_hdf5
        bundle = load_ingest_hdf5(cache_path)
        if bundle.sub is None:   # a sparse-only cache -- reduce_one_position needs sub too
            spots, qlab, omega_deg, mask, sub, ingest_counts = _ingest_position(frames, geom)
            bundle = IngestBundle(spots=spots, qlab=qlab, omega_deg=omega_deg, mask=mask,
                                  sub=sub, ingest_counts=ingest_counts)
    else:
        spots, qlab, omega_deg, mask, sub, ingest_counts = _ingest_position(frames, geom)
        bundle = IngestBundle(spots=spots, qlab=qlab, omega_deg=omega_deg, mask=mask,
                              sub=sub, ingest_counts=ingest_counts)
    qlab, omega_deg = bundle.qlab, bundle.omega_deg
    res = reduce_one_position(frames, geom, a=a, c=c, space_group_number=space_group_number,
                              sigma_rtn=sigma_rtn, point=p, ingested=bundle, **reduce_kwargs)
    if cache_path is not None and not cache_path.exists():
        from .persistence import save_position_hdf5
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_position_hdf5(cache_path, bundle=bundle, res=res, save_dense=save_dense)
    q_all = qlab_to_qsample(qlab, torch.deg2rad(torch.as_tensor(
        res.omega_sign.chosen_sign * omega_deg, dtype=qlab.dtype))).detach().cpu().numpy()
    candidate_entries = [dict(point=p, dom_idx=i, U=np.asarray(dom.U, float), n=dom.n,
                             branch=dom.branch)
                        for i, dom in enumerate(res.domains.domains)]
    return p, q_all, res, candidate_entries


def _pass2_one_position(p: int, q_all: np.ndarray, native_Us: List[np.ndarray],
                        candidates_payload: List[dict], a: float, c: float,
                        space_group_number: int, sigma_rtn: Tuple[float, float, float],
                        n_null_draws: int, null_alpha: float, dedup_threshold_deg: float,
                        seed_int: int):
    """One position's null threshold + cross-position recovery attempts -- the unit of work
    pass 2 dispatches, sequentially or via ``ProcessPoolExecutor``. Module-level for the same
    picklability reason as :func:`_pass1_one_position`; ``seed_int`` is drawn by the CALLER
    (once per position, in ``point_indices`` order, regardless of dispatch order) so the RNG
    draw sequence is identical whether this runs sequentially or in parallel."""
    from scipy.spatial.transform import Rotation
    null_counts = []
    for rmat in Rotation.random(n_null_draws, random_state=seed_int).as_matrix():
        r = refine_to_convergence(q_all, rmat, a0=a, c0=c,
                                  space_group_number=space_group_number, sigma_rtn=sigma_rtn)
        null_counts.append(int(r.claim.sum()) if r is not None else 0)
    null_threshold = float(np.quantile(null_counts, 1.0 - null_alpha))

    recovered = []
    for cand in candidates_payload:
        nearest_deg = (min(_misorient_deg(cand["U"], u, space_group_number) for u in native_Us)
                      if native_Us else float("inf"))
        if nearest_deg < dedup_threshold_deg:
            continue          # already native here -- nothing to recover
        result = refine_to_convergence(q_all, cand["U"], a0=a, c0=c,
                                       space_group_number=space_group_number, sigma_rtn=sigma_rtn)
        if result is None:
            continue
        n_found = int(result.claim.sum())
        if n_found <= null_threshold:
            continue
        recovered_hkl = result.hkl[result.claim]
        recovered.append(dict(point=p, origin_point=cand["origin_point"], n=n_found,
                              U=result.lat.U, cell=(result.lat.a, result.lat.b, result.lat.c),
                              hkl=recovered_hkl, g=q_all[result.claim],
                              nearest_pass1_deg=nearest_deg, null_threshold=null_threshold,
                              ab_gate=_ab_gate(recovered_hkl)))
    return p, null_counts, null_threshold, recovered


def recover_domains_across_raster(
    loader: Callable[[int], np.ndarray], geom: Geometry, point_indices: Sequence[int], *,
    a: float, c: float, space_group_number: int,
    sigma_rtn: Tuple[float, float, float],
    dedup_threshold_deg: float = 1.0,
    n_null_draws: int = 40,
    null_alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    reduce_kwargs: Optional[dict] = None,
    cache_dir: Optional[str] = None,
    save_dense: bool = True,
    n_workers: Optional[int] = None,
) -> RasterCoherenceResult:
    """Pass 1 (free search, every position) -> dedup orientations (symmetry-
    aware, ORIENTATION ONLY -- cell is deliberately not part of dedup) -> pass
    2 (seed-and-refit every candidate at every OTHER position, nominal-cell
    seeded) -> accept on a random-orientation null specific to that position
    (NOT a fixed count -- noise-capture rate can differ by position).

    ``cache_dir``, if given, makes pass 1 write-through: for each position,
    ``<cache_dir>/position_{p}.h5`` is read (via ``midas_defect.persistence
    .load_ingest_hdf5``) instead of re-ingesting when it already exists, and
    written (ingest bundle + full results, ``save_dense`` controlling whether
    the dense background-subtracted array is included) when it does not.
    ``loader(p)`` still runs either way -- raw frames stay needed for ring
    detection and the later honesty checks even when ingest itself is
    cached; only the expensive background-subtraction + 3-D blob-finding
    step is skipped on a cache hit.

    ``n_null_draws`` random orientations (``scipy.spatial.transform.Rotation
    .random``) are seeded at each position with the SAME real spot cloud (not
    scrambled -- scrambling destroys the Bragg-condition structure entirely
    and makes the null trivially pass; see this module's docstring). Match
    ``n_null_draws`` to roughly how many real candidates you are actually
    testing per position -- this is a look-elsewhere-corrected threshold, not
    a fixed significance level, so it should scale with how many looks you
    take, the same principle as :func:`midas_defect.rows.search_null`.

    ``n_workers``, if an integer > 1, parallelizes BOTH pass 1 and pass 2 across positions via
    ``ProcessPoolExecutor`` (``spawn`` context -- a real hang was found and fixed 2026-09-16:
    fork-after-thread-init in a numpy/torch-linked worker deadlocked with the platform default
    ``fork``; ``spawn`` re-imports fresh in each child instead). Default (``None``/``1``)
    preserves the EXACT original sequential behaviour and RNG draw order -- every existing
    caller is unaffected byte-for-byte. Measured on a real 900-position raster (S5,
    2026-09-16): 48 workers, 15.3 minutes total, where the sequential path was extrapolating to
    3.5-4 HOURS for pass 1 alone (~15s/position). Validated bit-for-bit identical
    native+recovered counts against the sequential path on an 18-position mixed sample before
    trusting it at full scale. Requires ``loader`` to be a plain, picklable, module-level
    callable under ``spawn`` -- see :func:`_pass1_one_position`'s own docstring.
    """
    rng = rng or np.random.default_rng(0)
    reduce_kwargs = dict(reduce_kwargs or {})
    point_indices = list(point_indices)
    parallel = n_workers is not None and n_workers > 1

    per_position: Dict[int, PositionCoherence] = {}
    all_candidates: List[dict] = []

    if not parallel:
        for p in point_indices:
            p, q_all, res, candidate_entries = _pass1_one_position(
                p, loader, geom, a, c, space_group_number, sigma_rtn, reduce_kwargs,
                cache_dir, save_dense)
            per_position[p] = PositionCoherence(point=p, res=res, q_all=q_all,
                                                null_counts=[], null_threshold=float("nan"))
            all_candidates.extend(candidate_entries)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_pass1_one_position, p, loader, geom, a, c, space_group_number,
                              sigma_rtn, reduce_kwargs, cache_dir, save_dense): p
                   for p in point_indices}
            for fut in as_completed(futs):
                p, q_all, res, candidate_entries = fut.result()
                per_position[p] = PositionCoherence(point=p, res=res, q_all=q_all,
                                                    null_counts=[], null_threshold=float("nan"))
                all_candidates.extend(candidate_entries)

    clusters = _dedup(all_candidates, dedup_threshold_deg, space_group_number)
    candidates = [Candidate(U=(best := max(cl, key=lambda c: c["n"]))["U"],
                            origin_point=best["point"], n_origin=best["n"], branch=best["branch"],
                            members=[(m["point"], m["dom_idx"]) for m in cl])
                 for cl in clusters]
    candidates_payload = [dict(U=c.U, origin_point=c.origin_point) for c in candidates]

    # Seeds drawn sequentially in point_indices order, REGARDLESS of n_workers/dispatch order
    # (as_completed order is non-deterministic under parallel pass 1) -- this is what keeps the
    # RNG draw sequence, and therefore the sequential path's exact output, unchanged by this
    # refactor.
    seeds = {p: int(rng.integers(0, 2**31)) for p in point_indices}

    if not parallel:
        for p in point_indices:
            pc = per_position[p]
            native_Us = [np.asarray(d.U, float) for d in pc.res.domains.domains]
            _, null_counts, null_threshold, recovered_payload = _pass2_one_position(
                p, pc.q_all, native_Us, candidates_payload, a, c, space_group_number,
                sigma_rtn, n_null_draws, null_alpha, dedup_threshold_deg, seeds[p])
            pc.null_counts = null_counts
            pc.null_threshold = null_threshold
            pc.recovered = [RecoveredDomain(**r) for r in recovered_payload]
    else:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {}
            for p in point_indices:
                pc = per_position[p]
                native_Us = [np.asarray(d.U, float) for d in pc.res.domains.domains]
                futs[ex.submit(_pass2_one_position, p, pc.q_all, native_Us, candidates_payload,
                              a, c, space_group_number, sigma_rtn, n_null_draws, null_alpha,
                              dedup_threshold_deg, seeds[p])] = p
            for fut in as_completed(futs):
                p, null_counts, null_threshold, recovered_payload = fut.result()
                pc = per_position[p]
                pc.null_counts = null_counts
                pc.null_threshold = null_threshold
                pc.recovered = [RecoveredDomain(**r) for r in recovered_payload]

    return RasterCoherenceResult(dedup_threshold_deg=dedup_threshold_deg,
                                 candidates=candidates, per_position=per_position)


# ---------------------------------------------------------------------------
# Spatial coherence as EVIDENCE, not assumption: does an a/b-split map cluster
# beyond what a shuffle of its own values would give -- and does a matched,
# zero-real-splitting control (real orientations, real claimed hkl, one common
# planted cell) cluster too? Only "observed clusters, control does not" is
# evidence of a real spatial pattern; if the control ALSO clusters, that is
# explained by the same mechanism `06`'s Step 3b already found for the SIGN of
# the split (index-population skew tracking smoothly-varying real orientation,
# `/verify` claim d93ca3c0a6e2, REFUTED) -- just applied to magnitude/spatial
# clustering instead of sign. Ported and generalized from the S5 project's own
# `s5/repro/s5_maps_make.py` (`morans_I`, `planted_control`), which already
# validated the algorithm but only ever ran it on the REAL map, never on its
# own planted control's map, and returned the control as a flat array with no
# position identity -- both fixed here.
# ---------------------------------------------------------------------------

def morans_i(grid: np.ndarray) -> float:
    """Moran's I, rook (4-neighbor) contiguity, on a 2-D grid with NaN for
    missing/excluded cells. `NaN` if fewer than 20 finite cells (not enough to
    say anything) or if every neighbor pair happens to be excluded.

    Verbatim port of `s5_maps_make.morans_I` -- same algorithm, now tested and
    reusable instead of stuck in a one-off analysis script.
    """
    grid = np.asarray(grid, float)
    ok = np.isfinite(grid)
    if ok.sum() < 20:
        return float("nan")
    z = grid - np.nanmean(grid)
    num = 0.0
    W = 0.0
    idx = np.argwhere(ok)
    for i, j in idx:
        for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            a, b = i + di, j + dj
            if 0 <= a < grid.shape[0] and 0 <= b < grid.shape[1] and ok[a, b]:
                num += z[i, j] * z[a, b]
                W += 1
    den = np.nansum(z[ok] ** 2)
    return (ok.sum() / W) * (num / den) if W and den else float("nan")


@dataclass
class PermutationResult:
    """Moran's I against its own permutation null: shuffle the finite values
    across the same finite-cell positions, recompute Moran's I each time --
    tests CLUSTERING (does spatial arrangement matter), not magnitude (a
    magnitude-spread test cannot tell a real trend from the same artifact at
    different severity per position -- see `06` Step 3's own docstring)."""
    observed_i: float
    null_median: float
    null_p95: float
    null_distribution: np.ndarray
    clustered: Optional[bool]   # None means "could not be tested" (too few finite cells),
                                # NOT "tested and found not clustered" -- keep these distinct;
                                # found directly while running this on real S5 data, where a
                                # naive `bool(...)` (False whenever observed_i is NaN) silently
                                # reported an untested grid as a negative result.


def permutation_null_morans_i(grid: np.ndarray, *, n_perm: int = 200,
                              rng: Optional[np.random.Generator] = None) -> PermutationResult:
    """`morans_i(grid)` against a permutation null built by shuffling the
    grid's own finite values across the same finite-cell positions.

    `clustered = observed_i > null_p95`, or `None` if `morans_i` itself returned NaN (fewer
    than 20 finite cells -- its own floor) -- "could not be tested" must never collapse into
    "tested, not clustered". Generalizes the shuffle loop already inline in
    `s5_maps_make.main()` into a reusable, tested function -- same algorithm, `n_perm=200`
    matching that script's own default.
    """
    rng = rng or np.random.default_rng(0)
    grid = np.asarray(grid, float)
    observed_i = morans_i(grid)
    finite = np.isfinite(grid)
    pos = np.argwhere(finite)
    vals = grid[finite].copy()
    nulls = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = grid.copy()
        perm = rng.permutation(vals)
        for (i, j), v in zip(pos, perm):
            shuffled[i, j] = v
        nulls[k] = morans_i(shuffled)
    null_p95 = float(np.nanpercentile(nulls, 95)) if np.isfinite(nulls).any() else float("nan")
    clustered = None if not np.isfinite(observed_i) else bool(observed_i > null_p95)
    return PermutationResult(
        observed_i=observed_i,
        null_median=float(np.nanmedian(nulls)) if np.isfinite(nulls).any() else float("nan"),
        null_p95=null_p95,
        null_distribution=nulls,
        clustered=clustered,
    )


def cluster_permutation_null_morans_i(grid: np.ndarray, cluster_ids: np.ndarray, *,
                                      n_perm: int = 200,
                                      rng: Optional[np.random.Generator] = None) -> PermutationResult:
    """The right null when many grid cells got their value from the SAME underlying source --
    e.g. a raster where a cross-position recovery propagates one domain's fitted delta to
    every position that recovery touched, so a large contiguous patch can share one identical
    value by construction, not by any spatial physics. `permutation_null_morans_i`'s plain
    null shuffles individual finite CELL values -- it does reshuffle duplicate-heavy value
    pools, but it does not test the more specific, more skeptical question this function does:
    "holding the OBSERVED partition of cells into clusters (and each cluster's actual shape)
    fixed, is the SPECIFIC assignment of fitted values to particular cluster footprints more
    spatially coherent than a random relabelling of values to footprints?" Found necessary
    2026-09-16 (S5, `/verify` claim `6ada36c60a11`'s artifact lens): 93.7% of a "clean" 526-cell
    raster's winning domains were cross-position recoveries, and just 5 origin domains supplied
    65% of the whole map -- the nominal cell count badly overstates independent information,
    and this is the test that checks whether that alone explains the observed Moran's I excess.

    ``cluster_ids`` is the SAME SHAPE as ``grid`` -- an integer (or any hashable) id per cell,
    identifying which underlying domain/origin gave that cell its value (e.g. the position
    itself for a native domain, or the recovery's ``origin_point`` for a recovered one). NaN
    cells in ``grid`` are ignored regardless of their id.

    ``clustered`` here answers a stricter question than the plain permutation null's -- treat a
    positive result on BOTH as considerably more convincing than either alone, and a positive
    result on the plain null but not this one as a real reason to suspect the effective sample
    size, not the crystallography, is doing the work.
    """
    rng = rng or np.random.default_rng(0)
    grid = np.asarray(grid, float)
    observed_i = morans_i(grid)
    finite = np.isfinite(grid)
    pos = np.argwhere(finite)

    # one representative value per unique cluster (all cells sharing a cluster share the
    # identical value by construction -- take the first, they must agree)
    cluster_of_cell = {tuple(ij): cluster_ids[tuple(ij)] for ij in pos}
    unique_clusters = sorted(set(cluster_of_cell.values()), key=lambda c: str(c))
    cluster_value = {}
    for ij in pos:
        c = cluster_of_cell[tuple(ij)]
        cluster_value.setdefault(c, grid[tuple(ij)])

    nulls = np.empty(n_perm)
    for k in range(n_perm):
        shuffled_clusters = list(unique_clusters)
        rng.shuffle(shuffled_clusters)
        relabel = dict(zip(unique_clusters, shuffled_clusters))
        shuffled = grid.copy()
        for ij in pos:
            c = cluster_of_cell[tuple(ij)]
            shuffled[tuple(ij)] = cluster_value[relabel[c]]
        nulls[k] = morans_i(shuffled)

    null_p95 = float(np.nanpercentile(nulls, 95)) if np.isfinite(nulls).any() else float("nan")
    clustered = None if not np.isfinite(observed_i) else bool(observed_i > null_p95)
    return PermutationResult(
        observed_i=observed_i,
        null_median=float(np.nanmedian(nulls)) if np.isfinite(nulls).any() else float("nan"),
        null_p95=null_p95,
        null_distribution=nulls,
        clustered=clustered,
    )


def exact_rank_p_value(observed: float, null_draws: Sequence[float]) -> float:
    """Distribution-free, exact one-sided p-value for "does `observed` exceed a small sample
    of independent null/control draws" -- the standard exceedance-rank formula
    ``(#draws >= observed + 1) / (n + 1)``, valid for ANY n including the very small n a control
    sweep like this project's typically has (11 draws). Prefer this over a parametric z-score
    ("N sigma above the mean") when n is small and/or the draws are not obviously normal --
    found necessary 2026-09-16 (S5, `/verify` claim `6ada36c60a11`'s statistics lens): a
    "+7.25 sigma" framing on 11 bounded, right-skewed control draws was false precision; the
    honest number is the exact rank p-value, here 1/12 = 0.083 -- suggestive, not significant
    at conventional thresholds.
    """
    null_draws = np.asarray(list(null_draws), float)
    n = len(null_draws)
    if n == 0:
        return float("nan")
    n_exceed_or_equal = int(np.sum(null_draws >= observed))
    return (n_exceed_or_equal + 1) / (n + 1)


def _domain_field(d, name: str):
    """Read `name` off a domain-like entry that may be a live `Domain` object
    (attribute access -- `spatial_coherence`'s own in-memory pass 1) OR a plain
    dict (`PositionResult.to_dict()["domains"][i]`, what `assemble_raster_results`
    hands back after a round trip through JSON -- what `06`'s own Step 2/5 cells
    already work with). Accepting both here, once, avoids every caller needing
    its own adapter just to bridge live vs. serialized domains.
    """
    return getattr(d, name) if hasattr(d, name) else d[name]


def _domain_n_claim(d) -> int:
    if hasattr(d, "claim"):
        return int(np.asarray(d.claim).sum())
    if isinstance(d, dict) and "n_claim" in d:
        return int(d["n_claim"])
    return len(_domain_field(d, "hkl"))


def planted_identical_cell_delta(
    domains_by_position: Dict[int, List["object"]], *, a: float, c: float,
    system: str = "orthorhombic", noise: float, rng: np.random.Generator,
    fit_fn=refine_cell_radial_robust, min_reflections: int = 6,
) -> Dict[int, float]:
    """The planted-identical-cell control, generalized from `s5_maps_make
    .planted_control`: for each position's DOMINANT domain (most claimed
    reflections -- same convention `06`'s Step 5 orientation map already
    uses), plant ONE common cell (`a == b` exactly) on that domain's REAL
    orientation and REAL claimed hkl set, refit, and report the delta a
    perfectly identical crystal would manufacture through this raster's own
    real geometry, real index-population skew, and real fit noise.

    Returns `{point: delta_pct}` -- keyed by position (unlike the original,
    which returned a flat array and lost position identity), so the result
    can be placed on a grid and tested for spatial clustering at all, which
    is the entire point of this function existing.

    `fit_fn` defaults to `refine_cell_radial_robust` (this project's explicit
    choice for Step 8) but is pluggable -- this function does not hard-wire
    itself to one estimator. `noise` is a single isotropic sigma in the same
    q-space units as `g` (matching the original's own simplification of
    `SIGMA_RTN`'s three components down to one scalar) -- pass your own
    measured value, there is no default.

    Each entry in `domains_by_position[p]` may be a live `midas_defect.domains
    .Domain` (attribute access) or a plain dict as `PositionResult.to_dict()`/
    `assemble_raster_results` produce (dict access) -- see `_domain_field`.
    """
    from midas_hkls import Lattice

    lat = Lattice(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=90.0)
    B_fixed = np.asarray(lat.reciprocal_cartesian_vectors(), float).T * 2 * math.pi

    out: Dict[int, float] = {}
    for p, domains in domains_by_position.items():
        usable = [d for d in domains if _domain_n_claim(d) >= min_reflections]
        if not usable:
            continue
        dom = max(usable, key=_domain_n_claim)
        hkl = np.asarray(_domain_field(dom, "hkl"), float)
        U = np.asarray(_domain_field(dom, "U"), float)
        q = (U @ (B_fixed @ hkl.T)).T
        q_noisy = q + rng.normal(0.0, noise, q.shape)
        dd = [DomainData(hkl=hkl, g=q_noisy)]
        try:
            fit = fit_fn(dd, system=system, cell0=(a, a, c, 90.0, 90.0, 90.0), two_pi=True)
        except Exception:
            continue   # a degenerate planted refit is skipped, not zero-filled
        delta, _sigma, _z = split_with_error(fit)
        if np.isfinite(delta):
            out[p] = delta
    return out


def best_domain_per_position(
    result: RasterCoherenceResult, *,
    exclude_index_asymmetry_suspect: bool = True,
    min_reflections: int = 1,
) -> Dict[int, dict]:
    """**The proper way to turn a `RasterCoherenceResult` into "one number per position"** --
    generalized out of the S5 raster run (2026-09-16) rather than left as bespoke driver-script
    logic. For each position, picks the largest-by-claimed-reflection-count domain among native
    + recovered, and BY DEFAULT excludes ANY domain (native or recovered) with the extreme-skew
    index_asymmetry signature, via the SAME `_ab_gate_is_suspect` rule `RecoveredDomain
    .index_asymmetry_is_suspect` uses.

    **Do not skip the default, and note this now covers native domains too.** An earlier version
    of this function (2026-09-16) only checked recovered domains -- a `/verify` physics-lens
    attack on claim `b8f1772140ae` caught it directly: a position's WINNING domain could be
    native, carry the identical extreme-skew signature, and count as "clean" anyway, because
    nothing computed `_ab_gate` on native domains at all. Fixed same day. Measured directly on
    the real 900-position S5 raster: 90.7 % of 15104 cross-position recovery attempts were
    index_asymmetry-suspect (see `manuals/solve-cell/ENVELOPE.md` #3 for the mechanism and the
    real 2604_25K incident this guards against -- a raw recovery-count comparison there looked
    like real spatial-coherence evidence and was `/verify`-REFUTED once checked). Including
    suspect recoveries inflated the S5 observed Moran's I from 0.147 (clean, pre-fix) to 0.239
    (all) -- the "all" number must never be quoted as evidence on its own, and the "clean"
    number itself needs re-measuring now that native domains are also gated (it may move).

    `.res.domains.domains` entries here are real `midas_defect.domains.Domain` objects, always
    (this function is only ever called on an in-memory `RasterCoherenceResult`, never a
    JSON round trip) -- unlike `planted_identical_cell_delta`'s `_domain_field` bridge, no dict
    adapter is needed on that side.

    Returns ``{point: {"hkl", "g", "U", "n", "source"}}`` for every position with at least one
    usable domain (``n >= min_reflections``) -- feed straight to :func:`grid_quantity` for any
    scalar you can compute from one domain dict, or wrap each value in a one-element list
    (``{p: [d] for p, d in best.items()}``) to feed :func:`planted_identical_cell_delta` or
    ``midas_hkls.cell_constrained.refine_cell_joint``/``refine_cell_radial``.
    """
    out: Dict[int, dict] = {}
    for p, pc in result.per_position.items():
        cands = []
        for dom in pc.res.domains.domains:
            dom_hkl = np.asarray(dom.hkl, float)
            if exclude_index_asymmetry_suspect and _ab_gate_is_suspect(_ab_gate(dom_hkl)):
                continue
            # origin=p for a native domain -- it IS its own source, not borrowed from
            # elsewhere. Lets a caller group positions by which single fitted domain actually
            # produced their value (see cluster_permutation_null_morans_i) without parsing
            # the human-readable `source` string.
            cands.append(dict(hkl=dom_hkl, g=pc.q_all[np.asarray(dom.claim, bool)],
                              U=np.asarray(dom.U, float), n=int(dom.n), source="native",
                              origin=p))
        for rec in pc.recovered:
            if exclude_index_asymmetry_suspect and rec.index_asymmetry_is_suspect:
                continue
            cands.append(dict(hkl=rec.hkl, g=rec.g, U=rec.U, n=int(rec.n),
                              source=f"recovered_from_p{rec.origin_point}",
                              origin=rec.origin_point))
        usable = [c for c in cands if c["n"] >= min_reflections]
        if usable:
            out[p] = max(usable, key=lambda c: c["n"])
    return out


def grid_quantity(
    domains_by_position: Dict[int, dict], quantity_fn: Callable[[dict], Optional[float]], *,
    n_fast: int, n_pos: int,
) -> np.ndarray:
    """Lay ANY per-domain scalar onto the raster's own ``(row, col)`` grid -- the shared shape
    :func:`morans_i` / :func:`permutation_null_morans_i` expect. Not specific to the a/b split:
    ``quantity_fn`` can return ``delta_pct`` (via your own fit on ``dom["hkl"]``/``dom["g"]``),
    ``dom["n"]`` itself (a coverage/reflection-count map), a mosaicity, an aspect ratio -- any
    function of one :func:`best_domain_per_position` entry. Return ``None`` from ``quantity_fn``
    to leave a position's cell ``NaN`` (missing), the same convention every grid in this module
    already uses.

    ``point = row * n_fast + col`` (fast axis = CenY, matching ``s5_config.Scan.rowcol`` and
    every other raster convention in this project) -- pass your raster's own ``n_fast``/``n_pos``,
    there is no default.
    """
    n_rows = n_pos // n_fast
    grid = np.full((n_rows, n_fast), np.nan)
    for p, dom in domains_by_position.items():
        v = quantity_fn(dom)
        if v is None:
            continue
        ri, ci = divmod(p, n_fast)
        if 0 <= ri < n_rows and 0 <= ci < n_fast:
            grid[ri, ci] = float(v)
    return grid


def values_to_grid(values: Dict[int, Optional[float]], *, n_fast: int, n_pos: int) -> np.ndarray:
    """Lay a plain ``{point: value}`` mapping onto the raster's own grid -- for a caller who
    already has one scalar per position computed their OWN way (a cell length, an a/b split, a
    mosaicity -- anything from a pipeline other than this module's own domain-recovery
    machinery) and has no `midas_defect` domain dict to run a :func:`grid_quantity`
    ``quantity_fn`` on. A one-line specialization of :func:`grid_quantity` with the identity
    function as ``quantity_fn`` -- same "missing point stays out of the dict" and "point = row *
    n_fast + col" conventions.
    """
    return grid_quantity(values, lambda v: v, n_fast=n_fast, n_pos=n_pos)


@dataclass
class SpatialCoherenceReport:
    """Bundles the whole "is this raster-wide pattern real" question for ONE quantity.

    ``control_sweep`` is a LIST, not one number, on purpose: measured directly on the S5 raster
    (2026-09-16), a SINGLE planted-identical-cell control draw is not decisive evidence the
    control "genuinely does not cluster" -- across 11 independent noise draws, 2 (18 %) crossed
    their own permutation null, against the ~5 % a well-behaved negative control's own null
    construction would predict. Report the control's spread, not one point estimate, exactly
    the same "single draw vs distribution" discipline this project already applies to a
    bootstrap CI.
    """
    observed_grid: np.ndarray
    observed: "PermutationResult"
    control_grids: List[np.ndarray] = field(default_factory=list)
    control_sweep: List["PermutationResult"] = field(default_factory=list)
    n_positions: int = 0
    n_effective_clusters: Optional[int] = None
    cluster_observed: Optional["PermutationResult"] = None

    @property
    def control_exceeded_own_null_fraction(self) -> Optional[float]:
        """Fraction of `control_sweep` draws that crossed THEIR OWN null -- compare against
        the ~5 % (`null_alpha` used to build them) a well-behaved negative control should show.
        `None` if no control was run at all."""
        if not self.control_sweep:
            return None
        return sum(bool(r.clustered) for r in self.control_sweep) / len(self.control_sweep)

    def observed_exceeds_all_controls(self) -> Optional[bool]:
        """True when the observed Moran's I is larger than EVERY control draw's own I -- a
        more direct, more robust comparison than "beats its own null" alone, since it is not
        sensitive to any one control draw's particular permutation-null threshold."""
        finite = [r.observed_i for r in self.control_sweep if np.isfinite(r.observed_i)]
        if not finite or not np.isfinite(self.observed.observed_i):
            return None
        return bool(self.observed.observed_i > max(finite))

    @property
    def exceeds_all_controls_p_value(self) -> Optional[float]:
        """Exact, distribution-free one-sided p-value for "observed exceeds every control
        draw" -- via :func:`exact_rank_p_value`. Prefer this over a parametric "N sigma above
        the control mean" claim: found necessary 2026-09-16 (`/verify` claim `6ada36c60a11`'s
        statistics lens) after a "+7.25 sigma" framing on 11 small, bounded, right-skewed
        control draws was flagged as false precision. `None` if no control was run."""
        finite = [r.observed_i for r in self.control_sweep if np.isfinite(r.observed_i)]
        if not finite or not np.isfinite(self.observed.observed_i):
            return None
        return exact_rank_p_value(self.observed.observed_i, finite)

    def summary(self) -> str:
        lines = [f"observed: I={self.observed.observed_i:+.3f}  null_p95={self.observed.null_p95:+.3f}  "
                f"clustered={self.observed.clustered}"]
        if self.n_effective_clusters is not None:
            lines.append(f"coverage: {self.n_positions} raster position(s), but only "
                        f"{self.n_effective_clusters} INDEPENDENT domain(s)/origin(s) behind "
                        f"them -- report this as the effective sample size, not {self.n_positions}.")
        if self.cluster_observed is not None:
            lines.append(f"cluster-block permutation (holds each domain's own footprint fixed, "
                        f"reshuffles which VALUE lands on which footprint): "
                        f"I={self.cluster_observed.observed_i:+.3f}  "
                        f"null_p95={self.cluster_observed.null_p95:+.3f}  "
                        f"clustered={self.cluster_observed.clustered} -- a positive result here "
                        f"is not explainable by shared-origin footprints alone.")
        if self.control_sweep:
            frac = self.control_exceeded_own_null_fraction
            exceeds_all = self.observed_exceeds_all_controls()
            p_exact = self.exceeds_all_controls_p_value
            lines.append(f"control: {len(self.control_sweep)} draw(s), "
                        f"{frac*100:.0f}% crossed their own null (expect ~5% if well-behaved), "
                        f"observed exceeds all draws' own I: {exceeds_all} "
                        f"(exact rank p={p_exact:.3g})")
            for i, r in enumerate(self.control_sweep):
                lines.append(f"  draw {i}: I={r.observed_i:+.3f}  null_p95={r.null_p95:+.3f}  "
                            f"clustered={r.clustered}")
        return "\n".join(lines)


def grid_coherence_report(
    observed_grid: np.ndarray, *,
    control_grids: Optional[Sequence[np.ndarray]] = None,
    cluster_id_grid: Optional[np.ndarray] = None,
    n_perm: int = 200,
) -> SpatialCoherenceReport:
    """The observed-vs-null-vs-control(s) test :func:`spatial_coherence_report` runs, starting
    from grids you already built YOUR OWN way -- for anyone whose a/b/c spatial maps come from a
    different pipeline than this module's own `recover_domains_across_raster`/
    `best_domain_per_position`, who should not have to adopt this project's `Domain`/
    `RasterCoherenceResult` machinery just to ask "does my map cluster more than chance, and more
    than a matched control?" This is the lower-level, grid-only entry point;
    :func:`spatial_coherence_report` is the higher-level one that builds these same grids FROM a
    `RasterCoherenceResult` and calls this underneath.

    All of ``observed_grid``, each of ``control_grids``, and ``cluster_id_grid`` must share the
    SAME ``(n_rows, n_fast)`` shape and the same NaN-for-missing convention every other grid in
    this module uses -- build them with :func:`values_to_grid` (from a ``{point: value}``
    mapping) or your own code.

    ``control_grids`` -- zero or more grids of the SAME quantity, built however you construct a
    matched null (an identical planted cell, a shuffled-label control, a synthetic dataset with
    zero real effect -- whatever is the right control for your own measurement). Unlike
    :func:`spatial_coherence_report`'s ``control_fn``, these are plain, already-computed arrays,
    not a function this call re-invokes several times -- pass as many independent draws as you
    have. :class:`SpatialCoherenceReport` reports their spread, not one point estimate: on the
    S5 raster this project studied, 2 of 11 independent control draws crossed their OWN
    permutation null on their own, against the ~5% a well-behaved negative control predicts, so
    one draw alone is not decisive evidence the control "doesn't cluster."

    ``cluster_id_grid`` -- optional, same shape as ``observed_grid``, one hashable id per cell
    identifying which independent measurement actually produced that cell's value (e.g. which
    physical grain/domain, if your own pipeline can also let one measurement supply several
    raster cells). Supply this whenever that can happen -- omitting it when it applies is exactly
    the gap that let the S5 raster's naive per-cell test read as "clustered" when the number of
    genuinely independent measurements behind it (75, out of 526 raster cells) did not support
    that verdict at all (see :func:`cluster_permutation_null_morans_i`'s own docstring for the
    full story). Leave it out if every cell in your grid is its own independent measurement.

    Returns the same :class:`SpatialCoherenceReport` :func:`spatial_coherence_report` does --
    call ``.summary()`` for a plain-text report of all of it at once.

    Examples
    --------
    >>> a_grid = values_to_grid(my_own_a_fits, n_fast=30, n_pos=900)     # doctest: +SKIP
    >>> control_grid = values_to_grid(my_own_control_a_fits, n_fast=30, n_pos=900)  # doctest: +SKIP
    >>> report = grid_coherence_report(a_grid, control_grids=[control_grid])  # doctest: +SKIP
    >>> print(report.summary())                                          # doctest: +SKIP
    """
    observed_grid = np.asarray(observed_grid, float)
    observed = permutation_null_morans_i(observed_grid, n_perm=n_perm, rng=np.random.default_rng(0))

    control_grids_out: List[np.ndarray] = []
    control_sweep: List[PermutationResult] = []
    for cg in (control_grids or []):
        cg = np.asarray(cg, float)
        control_grids_out.append(cg)
        control_sweep.append(permutation_null_morans_i(cg, n_perm=n_perm, rng=np.random.default_rng(0)))

    n_effective_clusters = None
    cluster_observed = None
    if cluster_id_grid is not None:
        cluster_id_grid = np.asarray(cluster_id_grid, dtype=object)
        finite_origins = {cluster_id_grid[i, j] for i, j in np.argwhere(np.isfinite(observed_grid))}
        n_effective_clusters = len(finite_origins)
        cluster_observed = cluster_permutation_null_morans_i(
            observed_grid, cluster_id_grid, n_perm=n_perm, rng=np.random.default_rng(0))

    return SpatialCoherenceReport(observed_grid=observed_grid, observed=observed,
                                  control_grids=control_grids_out, control_sweep=control_sweep,
                                  n_positions=int(np.isfinite(observed_grid).sum()),
                                  n_effective_clusters=n_effective_clusters,
                                  cluster_observed=cluster_observed)


def spatial_coherence_report(
    domains_by_position: Dict[int, dict], quantity_fn: Callable[[dict], Optional[float]], *,
    n_fast: int, n_pos: int,
    control_fn: Optional[Callable[[Dict[int, dict], np.random.Generator], Dict[int, float]]] = None,
    n_control_draws: int = 1,
    n_perm: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> SpatialCoherenceReport:
    """The proper end-to-end raster-wide spatial-coherence test, for ANY quantity -- generalized
    from the S5 Step 8 analysis (2026-09-16) into a single reusable call instead of bespoke
    driver-script logic every time. Grids ``quantity_fn`` over ``domains_by_position`` (typically
    :func:`best_domain_per_position`'s own output -- already index_asymmetry-filtered), tests it
    against a permutation null (:func:`permutation_null_morans_i`), and -- if ``control_fn`` is
    given -- runs it ``n_control_draws`` times with independent RNG state and reports the WHOLE
    spread, not one draw (see :class:`SpatialCoherenceReport`'s own docstring for why one draw
    is not decisive).

    ``control_fn`` is deliberately generic: it must accept ``(domains_by_position, rng)`` and
    return ``{point: value}`` on the SAME quantity `quantity_fn` computes. For the a/b-split
    case this project uses throughout, that is
    ``functools.partial(planted_identical_cell_delta, a=..., c=..., noise=..., min_reflections=...)``
    wrapped once more to accept positional ``(domains_by_position, rng)`` in that order (
    ``planted_identical_cell_delta`` itself takes ``rng`` as a keyword) -- there is no single
    generic control construction for an arbitrary quantity (a mosaicity or a reflection count
    has no obvious "planted identical" analogue), so this function does not invent one; supply
    your own or omit ``control_fn`` entirely for an observed-only report.

    This is a thin wrapper over :func:`grid_coherence_report`: it builds ``observed_grid``,
    ``control_grids`` (from ``control_fn``, via :func:`values_to_grid`), and ``cluster_id_grid``
    (from every domain dict's ``"origin"``, when present) out of ``domains_by_position``, then
    hands them to that lower-level, grid-only function for the actual statistics. If your own
    maps do not come from a `RasterCoherenceResult` at all, call :func:`grid_coherence_report`
    directly instead of building a fake ``domains_by_position`` just to reach it.

    Examples
    --------
    >>> best = best_domain_per_position(result)   # doctest: +SKIP
    >>> def delta_of(dom):
    ...     dd = [DomainData(hkl=dom["hkl"], g=dom["g"])]
    ...     fit = refine_cell_radial_robust(dd, system="orthorhombic",
    ...                                    cell0=(a, a, c, 90., 90., 90.), two_pi=True)
    ...     delta, sigma, z = split_with_error(fit)
    ...     return delta if np.isfinite(delta) else None
    >>> import functools
    >>> control_fn = lambda dbp, rng: planted_identical_cell_delta(
    ...     {p: [d] for p, d in dbp.items()}, a=a, c=c, noise=noise, rng=rng,
    ...     min_reflections=6)
    >>> report = spatial_coherence_report(best, delta_of, n_fast=30, n_pos=900,
    ...                                  control_fn=control_fn, n_control_draws=11)
    >>> print(report.summary())
    """
    rng = rng or np.random.default_rng(0)
    observed_grid = grid_quantity(domains_by_position, quantity_fn, n_fast=n_fast, n_pos=n_pos)

    control_grids: List[np.ndarray] = []
    if control_fn is not None:
        for _ in range(max(1, n_control_draws)):
            control_values = control_fn(domains_by_position, rng)
            control_grids.append(values_to_grid(control_values, n_fast=n_fast, n_pos=n_pos))

    # effective sample size + cluster-block permutation: only possible when every domain dict
    # carries an "origin" (best_domain_per_position's own output does) -- skip silently
    # otherwise rather than force every quantity_fn/domains_by_position shape to support it.
    cluster_id_grid = None
    if all(isinstance(d, dict) and "origin" in d for d in domains_by_position.values()):
        cluster_id_grid = np.full(observed_grid.shape, np.nan, dtype=object)
        for p, d in domains_by_position.items():
            ri, ci = divmod(p, n_fast)
            if 0 <= ri < cluster_id_grid.shape[0] and 0 <= ci < n_fast:
                cluster_id_grid[ri, ci] = d["origin"]

    return grid_coherence_report(observed_grid, control_grids=control_grids,
                                 cluster_id_grid=cluster_id_grid, n_perm=n_perm)
