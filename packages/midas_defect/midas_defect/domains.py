"""Every domain at one raster position, in one call.

Ported 2026-09-10 from the La3Ni2O7 project's ``repro/reduce_v4.py`` (``reduce_one``), the per-position
driver behind the 2604 raster. The primitives were already here; the COMPOSITION was not, and the
composition is where the expensive bugs lived -- seed-cell anchoring and a reported cell never fitted to
its own reflections were both composition bugs no primitive test could catch (LAB_NOTEBOOK R15). Every rule
below is a measured fix, numbered as in the original.

Input is an ingested, sample-frame spot cloud (q = 2 pi / d) -- the defect ``phase-1-ingest.md`` contract.
Frames, powder rings and stationary anvil/gasket cells are the caller's; say which spots may SEED with
``seedable`` and which may be CLAIMED at all with ``live``.

**The cell and the space group are required.** Six ``midas_defect.rows`` functions default to La3Ni2O7
(a = 3.6116, c = 19.2516, sg 139). A search that silently inherited those would apply I-centring to any
crystal; this one passes yours to every call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .rows import (cell_from_row, find_lattice_rows, index_from_pairs, index_from_row, match_mask,
                   omega_smear_duplicates, refine_lattice, refine_to_convergence, search_null,
                   RefinedLattice)

__all__ = ["Domain", "DomainSearch", "find_domains"]


@dataclass
class Domain:
    """One accepted domain. ``lat`` IS the least-squares fit to ``q[claim]`` / ``hkl``."""
    U: np.ndarray
    lat: object                       # rows.RefinedLattice
    claim: np.ndarray                 # one spot per reflection: count and fit with this
    frag: np.ndarray                  # every spot the orientation explains: consume with this
    hkl: np.ndarray                   # (n_claim, 3), in np.flatnonzero(claim) order
    branch: str                       # "row" or "pair"
    seeded: bool                      # cell borrowed from an earlier domain
    n_match: int
    margin: float
    n_rungs: int = 0
    step: Optional[Tuple[int, int, int]] = None
    seed_source: str = ""          # "own row", "earlier domain" or "nominal" (seed_from_nominal)

    @property
    def n(self) -> int:
        return int(self.claim.sum())


@dataclass
class DomainSearch:
    domains: List[Domain]
    explained: np.ndarray             # union of every domain's frag
    null_threshold: Optional[float]   # search_null threshold, None if the pair branch never ran
    n_rows_tried: int

    def __str__(self) -> str:
        body = ", ".join(f"{d.branch}{'*' if d.seeded else ''}{'/nominal' if d.seed_source == 'nominal' else ''} "
                        f"n={d.n} c={d.lat.c:.4f}" for d in self.domains)
        return (f"{len(self.domains)} domain(s): {body or 'none'}; explained {int(self.explained.sum())} "
                f"spots; rows tried {self.n_rows_tried}; pair null {self.null_threshold}")


def find_domains(q, intensity, row, col, frame, *, a: float, c: float, space_group_number: int,
                 live=None, seedable=None, nominal_c: Optional[float] = None, cell_tol_c: float = 0.015,
                 cell_tol_gamma_deg: float = 1.5, tol_sigma: float = 7.0,
                 sigma_rtn=(0.0071, 0.0145, 0.0094), tol_q: float = 0.05, max_domains: int = 8,
                 null_reps: int = 15, pair_retries: int = 8, min_row_match: int = 8,
                 min_row_margin: int = 3, min_seed_spots: int = 12, n_seed: int = 150,
                 duplicate_max_fraction: float = 0.34, rng_seed: int = 0,
                 seed_from_nominal: bool = False) -> DomainSearch:
    """Find every domain at one position: row-seeded first, then pair-seeded on the known cell.

    ``row``, ``col``, ``frame`` locate each spot on the detector (``frame`` fractional), used only to
    recognise an omega-smeared duplicate of an earlier domain's reflection.

    The rules, each a measured fix in the original driver:

    * FIX 2 -- TWO MASKS. ``frag`` (every spot the orientation explains) is consumed and counts as
      explained; ``claim`` (one survivor per reflection, ``unique_by_hkl``) is counted and fitted. Consuming
      only ``claim`` left discarded fragments for the next domain to claim -- a spurious domain made of the
      first one's streaks.
    * FIX 4 -- POOL the rows. ``find_lattice_rows`` seeds on the brightest spots and the dominant domain owns
      them: at 2604 p=329 the full set gave one usable (0,0,1) row, the residual after domain 1 gave four.
      Re-seeding alone lost a domain at p=113. The pool is seeded from the full set and EXTENDED from the
      residual after every acceptance -- a superset of both.
    * FIX 5 -- ANY unambiguous row seeds, not only (0,0,L): at p=66 the leftovers formed eight further rows
      nothing could index. Because a row's step now fixes the cell too, a row whose runner-up spacing fits
      nearly as well is refused.
    * FIX 6/8/9 -- ``refine_to_convergence``: iterate refine -> re-match on the domain's own cell, keep a
      round only on a strict gain, refit on the returned set.
    * FIX 9b -- REFIT AFTER THE DUPLICATE DROP. Dropping omega-smeared duplicates after the refit left the
      postcondition false for 3 of 4 non-first row domains, |dc| up to 0.0271 A -- the size of the effect
      under study. Here the drop comes after convergence on BOTH branches, then the refit.
    * FIX 7 -- the pair branch's floor is a WHOLE-SEARCH NULL (``search_null``), not a margin and not a
      hand-picked 8. A runner-up margin collapses once a second real grain exists (a planted 3-grain control
      returned 1 of 3; the null returns all three at <= 0.29 deg); the fixed 8 discarded real 5-7 reflection
      domains. Rejected pair candidates RETIRE their anchors and the search retries.
    * The pair branch's cell gate is referenced to a FIXED NOMINAL c (``nominal_c``, default ``c``), never to
      the seed: a seed-referenced gate manufactures the correlation it would be read as evidence of. Ungated,
      the search accepted c = 19.736 (+2.5 %) and gamma = 87.13. **The gamma bound censors gamma -- never
      quote gamma from gated domains.**

    ``seed_from_nominal=True`` lets the pair branch start from the DECLARED cell ``(a, a, c, 90, 90, 90)`` when
    no lattice row has fixed one. Off by default: the 2604 raster never did this. It is for positions where
    c* lies near the beam and no row of three rungs exists (S5 at 30 K: every position), where the search
    otherwise returns nothing without saying why. The declared cell stays the seed for every pair domain at
    that position (a seven-reflection fit is not a better seed than the declared cell), the cell gate still
    references ``nominal_c``, and each Domain's ``seed_source`` says "nominal". Found by the fresh-context
    dry run of 2026-09-10 on a held-out S5 position.

    ``tol_sigma`` 7 sat on a plateau at three 2604 positions (4 collapsed p=66 to 15 reflections; 9 kept
    adding domains where spurious ones appear); ``sigma_rtn`` is that sample's measured (radial, transverse,
    normal) residual. Both are per-dataset values: re-measure them.
    """
    q = np.asarray(q, float)
    I = np.asarray(intensity, float)
    n = len(q)
    row = np.asarray(row, float)
    col = np.asarray(col, float)
    frame = np.asarray(frame, float)
    if not (len(I) == len(row) == len(col) == len(frame) == n):
        raise ValueError("q, intensity, row, col and frame must have one entry per spot")
    live = np.ones(n, bool) if live is None else np.asarray(live, bool).copy()
    seedable = live.copy() if seedable is None else (np.asarray(seedable, bool) & live)
    sg = int(space_group_number)
    c_nom = float(c if nominal_c is None else nominal_c)
    kw = dict(tol_sigma=tol_sigma, sigma_rtn=sigma_rtn)   # the space group is passed EXPLICITLY at every call

    doms: List[Domain] = []
    work = live.copy()
    seed_lat = None
    nominal_lat = RefinedLattice(U=np.eye(3), B=np.diag([2.0*math.pi/a, 2.0*math.pi/a, 2.0*math.pi/c]),
                                 a=float(a), b=float(a), c=float(c), alpha=90.0, beta=90.0, gamma=90.0)
    null_thr: Optional[float] = None

    def rows_of(sel):
        sel = sel & seedable
        if int(sel.sum()) < min_seed_spots:
            return []
        out = []
        for r in find_lattice_rows(q[sel], I[sel], n_seed=n_seed, min_rungs=3, a=a, c=c,
                                   space_group_number=sg):
            if r.hkl_step is None:
                continue
            if r.match_rel_2 <= max(2.0 * (r.match_rel or 0.0), 0.02):
                continue                                  # two steps fit equally well
            out.append(r)
        return out

    def is_new(L, seen):
        for dirn, sp in seen:
            if abs(float(np.dot(L.direction, dirn))) > math.cos(math.radians(2.0)) and abs(L.spacing - sp) < 0.01 * sp:
                return False
        return True

    def drop_smeared(claim, frag, hkl_full):
        """Remove omega-smeared copies of earlier domains' reflections. Old arrays built per domain, in one order."""
        if not doms:
            return claim, frag, True
        idx_old = [np.flatnonzero(d.claim) for d in doms]
        dup = omega_smear_duplicates(hkl_full[claim], np.c_[row[claim], col[claim]], frame[claim],
                                     np.concatenate([d.hkl for d in doms]),
                                     np.concatenate([np.c_[row[i], col[i]] for i in idx_old]),
                                     np.concatenate([frame[i] for i in idx_old]))
        dup = np.asarray(dup, bool)
        if dup.size and dup.mean() > duplicate_max_fraction:
            return claim, frag, False
        drop = np.flatnonzero(claim)[dup]
        claim, frag = claim.copy(), frag.copy()
        claim[drop] = False
        frag[drop] = False
        return claim, frag, True

    def converge(U, a0, c0, B0, floor):
        res = refine_to_convergence(q, U, a0=a0, c0=c0, B0=B0, avail=work, min_reflections=floor,
                                    space_group_number=sg, **kw)
        if res is None:
            return None
        claim, frag, ok = drop_smeared(res.claim, res.frag, res.hkl)
        if not ok or int(claim.sum()) < floor:
            return None
        lat = res.lat
        if int(claim.sum()) != int(res.claim.sum()):          # FIX 9b
            lat = refine_lattice(q[claim], res.hkl[claim], a0=lat.a, c0=lat.c, sigma_rtn=sigma_rtn,
                                 min_reflections=min(int(claim.sum()), 6))
            if lat is None:
                return None
        return lat, claim, frag, res.hkl

    pool = rows_of(live)
    seen = [(L.direction, L.spacing) for L in pool]
    tried = set()
    while len(doms) < max_domains:
        accepted = False
        for L in pool:
            if id(L) in tried:
                continue
            tried.add(id(L))
            if seed_lat is not None:
                a_s, c_s, seeded = seed_lat.a, seed_lat.c, True
            else:
                a_s, c_s = cell_from_row(L.hkl_step, L.spacing, a0=a, c0=c, space_group_number=sg)
                seeded = False
            U, nm, _phi, margin = index_from_row(q, I, L.direction, L.hkl_step, a=a_s, c=c_s, exclude=~work,
                                                 space_group_number=sg, return_margin=True)
            if U is None or nm < min_row_match or margin < min_row_margin:
                continue
            got = converge(U, a_s, c_s, None, min_row_match)
            if got is None:
                continue
            lat, claim, frag, hkl_full = got
            if seed_lat is None:
                seed_lat = lat
            doms.append(Domain(U=lat.U, lat=lat, claim=claim, frag=frag, hkl=hkl_full[claim].copy(),
                               branch="row", seeded=seeded, n_match=int(nm), margin=float(margin),
                               n_rungs=int(L.n_rungs), step=tuple(int(x) for x in L.hkl_step),
                               seed_source="earlier domain" if seeded else "own row"))
            work &= ~frag
            accepted = True
            break

        pair_seed = seed_lat if seed_lat is not None else (nominal_lat if seed_from_nominal else None)
        if not accepted and pair_seed is not None:
            if null_thr is None:
                sd = work & seedable
                if int(sd.sum()) >= min_seed_spots:
                    def _search(qq, II, BB, **k2):
                        return index_from_pairs(qq, II, BB, a=pair_seed.a, c=pair_seed.c, tol_q=tol_q,
                                                min_reflections=4, space_group_number=sg, **k2)[1]
                    null_thr, _ = search_null(q[sd], I[sd], pair_seed.B, _search, n_rep=null_reps, seed=rng_seed)
                    null_thr = float(null_thr)
                else:
                    null_thr = float("inf")
            if math.isfinite(null_thr):
                floor = max(5, int(math.ceil(null_thr)) + 1)
                for _ in range(pair_retries):
                    Up, npair, _pair, pmargin = index_from_pairs(q, I, pair_seed.B, a=pair_seed.a, c=pair_seed.c,
                                                                 exclude=~(work & seedable), tol_q=tol_q,
                                                                 min_reflections=4, space_group_number=sg)
                    if Up is None:
                        break
                    first, _h, _r = match_mask(q, Up, B=pair_seed.B, a=pair_seed.a, c=pair_seed.c,
                                               return_residual=True, space_group_number=sg, **kw)
                    anchors = first & work
                    got = converge(Up, pair_seed.a, pair_seed.c, pair_seed.B, floor)
                    ok = got is not None
                    if ok:
                        lat, claim, frag, hkl_full = got
                        ok = (int(claim.sum()) > null_thr
                              and abs(lat.c / c_nom - 1.0) <= cell_tol_c
                              and abs(lat.gamma - 90.0) <= cell_tol_gamma_deg)
                    if ok:
                        doms.append(Domain(U=lat.U, lat=lat, claim=claim, frag=frag, hkl=hkl_full[claim].copy(),
                                           branch="pair", seeded=True, n_match=int(npair), margin=float(pmargin),
                                           seed_source="earlier domain" if seed_lat is not None else "nominal"))
                        work &= ~frag
                        accepted = True
                        break
                    if not (anchors & seedable).any():
                        break                                  # nothing left to retire: stop, do not spin
                    seedable = seedable & ~anchors             # retire these anchors, try elsewhere
        if not accepted:
            break
        for L in rows_of(work):                               # extend the pool, never replace it
            if is_new(L, seen):
                seen.append((L.direction, L.spacing))
                pool.append(L)

    explained = np.zeros(n, bool)
    for d in doms:
        explained |= d.frag
    return DomainSearch(domains=doms, explained=explained, null_threshold=null_thr, n_rows_tried=len(tried))
