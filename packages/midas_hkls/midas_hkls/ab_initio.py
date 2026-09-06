"""Ab-initio lattice finding: g-vectors in, unit cell out, no cell supplied.

The method is the Patterson / difference-vector one. Written from the published
algorithm — ImageD11 implements the same idea but is GPL-2 and MIDAS is
BSD-3, so **no code is shared**; only the physics, which is not anyone's.

How it works
------------
A set of scattering vectors from one crystal lies on a reciprocal lattice, so
``v·g`` is an integer for every ``g`` exactly when ``v`` is a **real-space
lattice vector**. Sum ``exp(2πi r·g)`` over the observed g and the magnitude
peaks precisely at those ``r``. That sum is a 3-D Fourier transform of the
g-vector density — one FFT gives every candidate lattice vector at once, with
no direction scan.

    1. bin the g-vectors onto a grid spanning ±q_max
    2. FFT → the Patterson; its peaks away from the origin are candidate
       real-space lattice vectors
    3. score each candidate by how many g give a near-integer ``v·g``
    4. refine the survivors (linear least squares on the integers)
    5. pick three short, independent, high-scoring vectors as a basis
    6. reduce the basis, and check it is neither a supercell nor a sublattice

What this implementation insists on
-----------------------------------
**A chance level for every score.** A random direction of length L already
indexes about ``2·tol`` of the reflections by accident, and long vectors score
better than short ones for no physical reason. Every candidate is reported
against that expectation, and :func:`chance_score` gives it in closed form —
so a "good" score is one that beats chance, not one that is merely large.

**Explicit resolution limits.** The grid fixes both the longest vector findable
(``R_max = n_grid/(2·q_max)``) and the finest distinguishable one
(``Δr = 1/(2·q_max)``). Both are reported. A cell axis longer than ``R_max``
cannot be found, and the failure is silent unless you look.

**A refusal, rather than a cell, when the data cannot support one.** With too
few reflections the Patterson has no peaks above chance. Say so.

The supercell test is RELATIVE, and here is why
-----------------------------------------------
An absolute "predicted vs observed reflection count" is invalid whenever the
measurement does not sample the full sphere. On a 36° ω wedge the true
La₃Ni₂O₇ subcell (V = 251 Å³) predicts ~1052 reflections inside |g| < 1 and
only 51 are observed — a ratio of **20.6 for the correct answer**. Any absolute
threshold tight enough to catch a supercell would refuse the truth.

So the ranking is relative: among bases that explain comparable numbers of
spots, the **smallest cell volume** wins. Every candidate shares the same
coverage, so the comparison between them is fair even when the absolute
expectation is not. ``supercell_ratio`` is still reported, as a diagnostic to
read against your own coverage, and never as a gate.

Supercells
----------
Two mechanisms, because one is not enough. **Axis division** catches supercells
aligned with the basis (verified at x8 and x27). **A sublattice search** over
fractional combinations ``(n₁a+n₂b+n₃c)/m`` up to ``m = 4`` catches skewed ones,
which axis division misses entirely — measured cases of x2, x3 and x7 all
survived axis division alone.

Beyond ``m = 4`` the search stops, so a very high-index skewed supercell can
still get through. ``supercell_ratio`` is reported for that case, but only as a
loose last resort: on a 36° ω wedge the **true** 2604 cell scores 20.6, so any
threshold tight enough to catch a modest supercell also refuses the truth. The
real protection is the relative ranking — smallest cell among those explaining
comparably much. Check the volume against chemistry before believing it.

Conventions
-----------
g-vectors in **1/d** (not 2π/d), so the recovered vectors and cell come out in
Å directly. Pass ``two_pi=True`` if yours carry the 2π.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["LatticeCandidate", "AbInitioResult", "patterson", "chance_score",
           "score_vector", "refine_vector", "find_candidate_vectors",
           "select_basis", "reduce_basis", "index_ab_initio"]


# ---------------------------------------------------------------------------
# scoring, and what a score is worth
# ---------------------------------------------------------------------------

def score_vector(v: np.ndarray, g: np.ndarray, tol: float = 0.15
                 ) -> Tuple[int, float]:
    """How many g-vectors does this candidate index, and how well?

    Returns ``(n_inliers, rms_of_inlier_residuals)`` where the residual is the
    distance of ``v·g`` from the nearest integer.
    """
    p = np.asarray(g, float) @ np.asarray(v, float)
    r = np.abs(p - np.round(p))
    inl = r < tol
    n = int(inl.sum())
    return n, float(np.sqrt((r[inl] ** 2).mean())) if n else float("inf")


def chance_score(n_g: int, tol: float = 0.15) -> float:
    """Reflections a **random** direction indexes by accident.

    For a vector long enough that ``v·g`` wanders over many integers, the
    fractional parts are near-uniform, so a fraction ``2·tol`` land within
    ``tol`` of an integer regardless of any lattice. That is the level a
    candidate must beat, and it is why a raw inlier count means nothing on its
    own — a long vector scores well for no physical reason.
    """
    return float(n_g) * min(2.0 * tol, 1.0)


def refine_vector(v: np.ndarray, g: np.ndarray, tol: float = 0.15,
                  n_cycles: int = 12) -> Tuple[np.ndarray, int, float]:
    """Refine a candidate so its projections sit on integers.

    With the integers ``n_i = round(v·g_i)`` fixed, ``v`` solves a linear least
    squares — ``min Σ (v·gᵢ − nᵢ)²`` — so each cycle is exact and only the
    integer assignment iterates. Converges in a handful of cycles or not at all.
    """
    v = np.array(v, float)
    g = np.asarray(g, float)
    last = None
    for _ in range(n_cycles):
        p = g @ v
        n = np.round(p)
        inl = np.abs(p - n) < tol
        if inl.sum() < 3:
            return v, int(inl.sum()), float("inf")
        A = g[inl]
        v_new, *_ = np.linalg.lstsq(A, n[inl], rcond=None)
        if last is not None and np.allclose(v_new, last, rtol=1e-12, atol=1e-12):
            v = v_new
            break
        last, v = v_new, v_new
    n_in, rms = score_vector(v, g, tol)
    return v, n_in, rms


@dataclass
class LatticeCandidate:
    """One candidate real-space lattice vector."""
    vector: np.ndarray
    score: int
    rms: float
    chance: float
    length: float

    @property
    def excess(self) -> float:
        """Score above what a random direction would get. This is the evidence."""
        return self.score - self.chance

    def __str__(self) -> str:
        return (f"|v|={self.length:7.3f} A  score {self.score:4d} "
                f"(chance {self.chance:.1f}, excess {self.excess:+.1f})  "
                f"rms {self.rms:.4f}")


# ---------------------------------------------------------------------------
# the Patterson
# ---------------------------------------------------------------------------

def patterson(g: np.ndarray, *, n_grid: int = 128,
              q_max: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """3-D Patterson of the g-vector set.

    Returns ``(magnitude, dr, r_max)`` — the FFT magnitude on an ``n_grid³``
    array, the real-space sample spacing ``Δr = 1/(2·q_max)`` in Å, and the
    longest findable vector ``R_max = n_grid·Δr``.

    A lattice vector longer than ``R_max`` is **not findable** and nothing about
    the output will say so, which is why both numbers come back.
    """
    g = np.asarray(g, float)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"g must be (N, 3), got {g.shape}")
    if n_grid < 16 or n_grid & (n_grid - 1):
        raise ValueError("n_grid should be a power of two and >= 16")
    if q_max is None:
        q_max = float(np.abs(g).max()) * 1.05
    if q_max <= 0:
        raise ValueError("q_max must be positive")

    dq = 2.0 * q_max / n_grid
    idx = np.rint(g / dq).astype(int) + n_grid // 2
    keep = np.all((idx >= 0) & (idx < n_grid), axis=1)
    grid = np.zeros((n_grid,) * 3, float)
    np.add.at(grid, (idx[keep, 0], idx[keep, 1], idx[keep, 2]), 1.0)

    mag = np.abs(np.fft.fftn(grid))
    dr = 1.0 / (2.0 * q_max)
    return mag, dr, n_grid * dr


def find_candidate_vectors(g: np.ndarray, *, n_grid: int = 128,
                           q_max: Optional[float] = None,
                           tol: float = 0.15,
                           n_peaks: int = 60,
                           min_length: float = 1.5,
                           min_excess_sigma: float = 3.0
                           ) -> Tuple[List[LatticeCandidate], Dict]:
    """Patterson peaks → scored, refined candidate lattice vectors.

    Candidates are kept only if their score beats :func:`chance_score` by
    ``min_excess_sigma`` Poisson sigmas of the chance level — a long vector that
    merely indexes ``2·tol`` of the reflections is not a lattice vector.

    Returns ``(candidates_sorted_by_excess, diagnostics)``.
    """
    g = np.asarray(g, float)
    mag, dr, r_max = patterson(g, n_grid=n_grid, q_max=q_max)

    # peak positions, origin excluded; wrap indices to signed offsets
    flat = np.argsort(mag.ravel())[::-1]
    chance = chance_score(len(g), tol)
    floor = chance + min_excess_sigma * math.sqrt(max(chance, 1.0))

    seen: List[LatticeCandidate] = []
    tried = 0
    for f in flat:
        if len(seen) >= n_peaks or tried > 40 * n_peaks:
            break
        tried += 1
        i, j, k = np.unravel_index(f, mag.shape)
        off = np.array([i, j, k], float)
        off[off > n_grid / 2] -= n_grid
        v0 = off * dr
        L = float(np.linalg.norm(v0))
        if L < min_length or L > r_max / 2:
            continue
        v, n_in, rms = refine_vector(v0, g, tol)
        L = float(np.linalg.norm(v))
        if not np.isfinite(rms) or L < min_length:
            continue
        if n_in < floor:
            continue
        if any(np.linalg.norm(v - c.vector) < 0.5 * dr or
               np.linalg.norm(v + c.vector) < 0.5 * dr for c in seen):
            continue
        seen.append(LatticeCandidate(vector=v, score=n_in, rms=rms,
                                     chance=chance, length=L))

    seen.sort(key=lambda c: (-c.excess, c.length))
    diag = {"dr": dr, "r_max": r_max, "chance": chance, "score_floor": floor,
            "n_peaks_examined": tried, "n_candidates": len(seen),
            "q_max": q_max if q_max is not None else float(np.abs(g).max()) * 1.05}
    return seen, diag


# ---------------------------------------------------------------------------
# from candidate vectors to a basis
# ---------------------------------------------------------------------------

def reduce_basis(A: np.ndarray, n_cycles: int = 100) -> np.ndarray:
    """Buerger-reduce a real-space basis to short, near-orthogonal vectors.

    Columns are the basis vectors. Repeatedly replaces the longest vector by the
    shortest of its combinations with the others (``v ± w``), which is Gauss
    reduction lifted to three dimensions. Terminates when nothing gets shorter.

    This is the step where an ab-initio indexer usually goes wrong, so what it
    does **not** do is worth stating: it does not standardise to a Niggli
    setting, and it does not choose between the reduced cell and any
    symmetry-implied conventional cell. It returns a short basis for the same
    lattice — nothing more.
    """
    A = np.array(A, float)
    if A.shape != (3, 3):
        raise ValueError("A must be 3x3 with basis vectors as columns")
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("basis is degenerate (zero volume)")

    for _ in range(n_cycles):
        changed = False
        order = np.argsort([np.linalg.norm(A[:, i]) for i in range(3)])
        A = A[:, order]
        for i in (2, 1):
            best = A[:, i].copy()
            bl = np.linalg.norm(best)
            for j in range(3):
                if j == i:
                    continue
                for s in (+1, -1):
                    cand = A[:, i] + s * A[:, j]
                    if np.linalg.norm(cand) < bl - 1e-12:
                        best, bl = cand, np.linalg.norm(cand)
            if not np.allclose(best, A[:, i]):
                A[:, i] = best
                changed = True
        if not changed:
            break
    if np.linalg.det(A) < 0:
        A[:, 2] = -A[:, 2]
    return A


def _index_fraction(A: np.ndarray, g: np.ndarray, tol: float) -> Tuple[float, np.ndarray]:
    """Fraction of g indexed by the basis, and the integer hkl."""
    h = g @ A                      # h_i = a_i . g  (A columns = basis vectors)
    r = np.abs(h - np.round(h))
    ok = np.all(r < tol, axis=1)
    return float(ok.mean()), np.round(h)


def _predicted_reflection_count(A: np.ndarray, q_max: float) -> float:
    """How many reflections a cell of this volume should put inside |g| < q_max.

    A **supercell** has too large a real-space volume, hence too dense a
    reciprocal lattice, and predicts far more reflections than were observed.
    That mismatch is the cleanest supercell detector there is, and it costs one
    determinant.
    """
    v_real = abs(float(np.linalg.det(A)))
    return (4.0 / 3.0) * math.pi * q_max ** 3 * v_real


def _noncoplanarity(A: np.ndarray) -> float:
    """|det| / (|a||b||c|) — 1 for orthogonal axes, → 0 for a flat basis.

    The guard that matters. A degenerate basis "indexes" a large fraction of any
    spot list, because a collapsed metric sends many projections to near-integer
    values for free, so any ranking that rewards subset size will choose it. On
    real DAC data this silently returned bases of **zero volume** owning 159-224
    of 442 spots.
    """
    A = np.asarray(A, float)
    d = np.prod([np.linalg.norm(A[:, i]) for i in range(3)])
    return abs(float(np.linalg.det(A))) / d if d > 0 else 0.0


def _find_finer_lattice(A: np.ndarray, g: np.ndarray, tol: float,
                        max_index: int = 4, min_volume: float = 5.0,
                        min_base_fraction: float = 0.15) -> Optional[np.ndarray]:
    """Is the true lattice a **superlattice** of this basis? Search properly.

    If the found basis spans a sublattice of index *m*, the true lattice contains
    vectors ``(n₁a + n₂b + n₃c)/m`` that are not in it. Axis-only division finds
    these only when the supercell happens to be axis-aligned; a **skewed** one
    survives it untouched, and measured cases of x2, x3 and x7 did exactly that.

    So enumerate: for each ``m`` up to ``max_index`` and each residue triple
    ``(n₁,n₂,n₃) mod m``, test whether that fractional combination indexes the
    spots as well as the basis does. If one does, it belongs to the true lattice
    — swap it in for whichever basis vector keeps the volume smallest, and the
    caller re-reduces.

    Bounded and cheap: 7 combinations at m=2, 26 at m=3, 63 at m=4. **A skewed
    supercell of prime index above ``max_index`` is therefore still missed** — a
    x504 construction reduced only to x7, because 7 exceeds the default 4. Raise
    ``max_index`` if you suspect one; the cost is m³ tests per cycle.

    ``min_base_fraction`` is deliberately low. On multi-domain data a correct
    single-grain basis legitimately owns a small share of the whole spot list,
    and a 0.5 floor silently disabled this search on exactly the data it is
    most needed for.
    """
    base_frac, _ = _index_fraction(A, g, tol)
    if base_frac < min_base_fraction:
        return None
    best = None
    for m in range(2, int(max_index) + 1):
        for n1 in range(m):
            for n2 in range(m):
                for n3 in range(m):
                    if n1 == n2 == n3 == 0:
                        continue
                    w = (n1 * A[:, 0] + n2 * A[:, 1] + n3 * A[:, 2]) / m
                    if np.linalg.norm(w) < 1e-6:
                        continue
                    # try w in place of each basis vector
                    for i in range(3):
                        B = A.copy()
                        B[:, i] = w
                        v = abs(float(np.linalg.det(B)))
                        if v < min_volume or _noncoplanarity(B) < 0.05:
                            continue
                        f, _ = _index_fraction(B, g, tol)
                        if f >= base_frac - 1e-9 and (best is None or v < best[0]):
                            best = (v, B)
    return None if best is None else best[1]


def _try_shorter(A: np.ndarray, g: np.ndarray, tol: float,
                 divisors: Sequence[int] = (2, 3),
                 min_base_fraction: float = 0.5,
                 min_length: float = 1.0,
                 min_volume: float = 5.0,
                 min_noncoplanarity: float = 0.05) -> Optional[np.ndarray]:
    """Is the true lattice finer than this basis? Try dividing each axis.

    If ``a/2`` still indexes everything then ``a`` was a supercell axis and the
    reported cell would be twice too long — a silent, plausible, wrong answer.

    **This only catches supercells aligned with the basis axes.** A general
    supercell — one skewed relative to its own reduced basis — survives axis
    division untouched; the reflection-count check in :func:`select_basis` is
    the guard for that, and it is the one that must be enforced.

    Two guards, both learned the hard way. The basis must already index a real
    fraction (``min_base_fraction``) before dividing means anything: on a basis
    that indexes almost nothing, *every* division "maintains" the fraction and
    the search shrinks the cell to zero volume. And no axis is divided below
    ``min_length``, which is shorter than any real lattice translation.
    """
    base_frac, _ = _index_fraction(A, g, tol)
    if base_frac < min_base_fraction:
        return None
    for i in range(3):
        for d in divisors:
            B = A.copy()
            B[:, i] = B[:, i] / d
            if np.linalg.norm(B[:, i]) < min_length:
                continue
            if abs(np.linalg.det(B)) < min_volume:
                continue
            if _noncoplanarity(B) < min_noncoplanarity:
                continue
            f, _ = _index_fraction(B, g, tol)
            if f >= base_frac - 1e-9:
                return B
    return None


def select_basis(candidates: Sequence[LatticeCandidate], g: np.ndarray, *,
                 tol: float = 0.15, q_max: Optional[float] = None,
                 max_try: int = 12,
                 min_subset: int = 20,
                 min_subset_fraction: float = 0.0,
                 subset_tolerance: float = 0.75,
                 max_supercell_ratio: float = 100.0,
                 hkl_condition_max: float = 1e6,
                 min_volume: float = 5.0,
                 min_noncoplanarity: float = 0.05
                 ) -> Tuple[Optional[np.ndarray], Dict]:
    """Choose three candidates spanning **one** lattice, and say which spots it owns.

    The important change from a textbook indexer: a real spot list from a
    diamond-anvil cell is not one crystal. It carries gasket powder, anvil
    reflections, a pressure marker and several domains, so **demanding that a
    basis index most of the list guarantees failure**. What a correct basis does
    is own a *self-consistent subset*.

    The criterion, and the order matters: among bases whose subset is at least
    ``min_subset`` **and** within ``subset_tolerance`` of the largest subset
    found, take the one with the **smallest cell volume**.

    Ranking by subset size first is wrong and it fails in a specific, measurable
    way: a denser reciprocal lattice has a node near more things, so a supercell
    always out-owns the true cell, and the effect grows with ``tol``. Measured on
    2604, largest-subset-first returned V/V_true of 2.0, 14.1, 1.0, 14.3, 1.0,
    7.3, 7.3, 7.2, 7.1, 7.0 as tol went 0.06 → 0.18 — the answer tracked the
    tolerance rather than the crystal. Smallest-volume-among-comparable is the
    standard criterion and it is stable.
    """
    g = np.asarray(g, float)
    if q_max is None:
        q_max = float(np.linalg.norm(g, axis=1).max())
    notes: List[str] = []
    pool = list(candidates)[:max_try]
    if len(pool) < 3:
        return None, {"notes": [f"only {len(pool)} candidate vectors survived "
                                "the chance-level cut; a lattice needs three "
                                "independent ones"]}

    floor = max(min_subset, int(round(min_subset_fraction * len(g))))
    best = None
    viable: List[Tuple[int, float, np.ndarray]] = []
    n_rejected_larger = 0
    considered = 0
    n_degenerate = 0
    n_rank_deficient = 0
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            for k in range(j + 1, len(pool)):
                A = np.stack([pool[i].vector, pool[j].vector, pool[k].vector], 1)
                if abs(float(np.linalg.det(A))) < 1e-6:
                    continue
                try:
                    A = reduce_basis(A)
                    for _ in range(8):
                        finer = _find_finer_lattice(A, g, tol)
                        if finer is None:
                            finer = _try_shorter(A, g, tol)
                        if finer is None:
                            break
                        A = reduce_basis(finer)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                considered += 1
                # Re-check AFTER reduction and division: the division loop can
                # walk a good basis into a degenerate one, and a degenerate
                # basis wins on subset size against every real cell.
                if abs(float(np.linalg.det(A))) < min_volume:
                    n_degenerate += 1
                    continue
                if _noncoplanarity(A) < min_noncoplanarity:
                    n_degenerate += 1
                    continue
                h = g @ A
                inl = np.all(np.abs(h - np.round(h)) < tol, axis=1)
                n_sub = int(inl.sum())
                if n_sub < floor:
                    continue
                # The indexed SET must span three dimensions, not merely the
                # basis. A basis can be perfectly non-coplanar while the
                # reflections it happens to own all lie in one plane of hkl
                # space -- and then the cell is not determined in every
                # direction. This check used to live in the refinement, which
                # runs AFTER selection has committed, so a whole tolerance
                # would come back refused with a good cell available.
                hi = np.round(h[inl])
                H = hi.T @ hi
                ev = np.linalg.eigvalsh(H)
                if ev[0] <= 0 or ev[-1] / max(ev[0], 1e-30) > hkl_condition_max:
                    n_rank_deficient += 1
                    continue
                vol = abs(float(np.linalg.det(A)))
                viable.append((n_sub, vol, A))

    if viable:
        best_sub = max(v[0] for v in viable)
        # among bases explaining comparably much, the SMALLEST cell wins
        comparable = [v for v in viable if v[0] >= subset_tolerance * best_sub]
        n_sub, vol, A = min(comparable, key=lambda v: v[1])
        best = ((0,), A, n_sub, vol)
        n_rejected_larger = len(viable) - 1

    if best is None:
        return None, {"considered": considered,
                      "n_degenerate": n_degenerate,
                      "n_rank_deficient": n_rank_deficient, "notes": [
            f"no triple of candidates owned a self-consistent subset of at "
            f"least {floor} spots whose hkl span three dimensions "
            f"({n_degenerate} bases were degenerate, {n_rank_deficient} owned "
            "a coplanar set). The Patterson found periodicities but none of "
            "them organises enough of the pattern into a 3-D lattice."]}

    _, A, n_sub, vol = best
    pred = _predicted_reflection_count(A, q_max)
    ratio = pred / max(n_sub, 1)
    if ratio > max_supercell_ratio:
        # A LAST-RESORT guard, deliberately loose. The relative ranking above is
        # the real protection and works whenever the true vectors are among the
        # candidates; this only catches the case where they are not. Calibrated
        # on 2604: the TRUE cell scores 20.6 on a 36 deg wedge, while the gross
        # supercells that slipped through an earlier design scored 205 and 232.
        # Anything tighter than ~100 starts refusing correct answers on
        # narrow-wedge data.
        return None, {"supercell_ratio": ratio, "considered": considered,
                      "notes": notes + [
                          f"REJECTED: predicts ~{pred:.0f} reflections but the "
                          f"cell owns only {n_sub}, a ratio of {ratio:.0f}. Even "
                          "allowing for a narrow wedge that is not a plausible "
                          "cell. No smaller candidate was available to rank "
                          "against, so the Patterson likely missed the true "
                          "lattice vectors entirely."]}
    notes.append(
        f"this cell owns {n_sub} of {len(g)} spots ({n_sub/len(g):.0%}); the "
        "rest belong to other domains, the gasket, the anvils or noise. "
        "A subset is the expected outcome on real DAC data, not a failure.")
    return A, {"subset_size": n_sub, "subset_fraction": n_sub / len(g),
               "volume": vol, "predicted_reflections": pred,
               "supercell_ratio": pred / max(n_sub, 1), "considered": considered,
               "n_degenerate": n_degenerate, "n_viable": len(viable),
               "n_rank_deficient": n_rank_deficient,
               "n_rejected_larger": n_rejected_larger,
               "best_subset_seen": max((v[0] for v in viable), default=0),
               "noncoplanarity": _noncoplanarity(A), "notes": notes}


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

@dataclass
class AbInitioResult:
    """A cell found without one being supplied — and what it rests on."""
    success: bool
    cell: Optional[Tuple[float, ...]] = None
    cell_sigma: Optional[Tuple[float, ...]] = None
    UB: Optional[np.ndarray] = None
    UBI: Optional[np.ndarray] = None
    hkl: Optional[np.ndarray] = None
    indexed_mask: Optional[np.ndarray] = None    # (N,) bool into the input g
    indexed_fraction: float = 0.0
    n_indexed: int = 0
    n_reflections: int = 0
    candidates: List[LatticeCandidate] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.success:
            return "ab-initio FAILED: " + ("; ".join(self.notes) or "no reason recorded")
        c, s = self.cell, self.cell_sigma
        body = ", ".join(f"{v:.4f}±{e:.4f}" for v, e in zip(c[:3], s[:3]))
        ang = ", ".join(f"{v:.3f}±{e:.3f}" for v, e in zip(c[3:], s[3:]))
        head = (f"cell {body} | {ang} | indexed {self.n_indexed}/"
                f"{self.n_reflections} ({self.indexed_fraction:.0%})")
        return head + (("\n  " + "\n  ".join(self.notes)) if self.notes else "")


def index_ab_initio(g: np.ndarray, *, tol: float = 0.15, n_grid: int = 128,
                    q_max: Optional[float] = None, two_pi: bool = False,
                    min_reflections: int = 20,
                    min_subset: int = 20,
                    sigma_g: Optional[float] = None,
                    min_excess_sigma: float = 3.0) -> AbInitioResult:
    """Find a unit cell from g-vectors alone. No cell, no symmetry, no seed.

    Parameters
    ----------
    g : (N, 3)
        Scattering vectors in the **sample** frame, in 1/d (Å⁻¹) unless
        ``two_pi``.
    two_pi : bool
        Set if your g carry the 2π. The cell is returned in Å either way.
    min_reflections : int
        Refuse below this. The Patterson needs enough difference vectors to
        build a peak; a dozen reflections will not do it, and returning a cell
        from too few is worse than returning nothing.

    Returns an :class:`AbInitioResult` whose ``success`` may be False — that is
    a normal outcome, not an exception.

    Notes
    -----
    Every quantitative claim carries its own check: candidate scores are
    reported against :func:`chance_score`, the accepted cell is tested for being
    a supercell both by dividing its axes and by comparing its predicted
    reflection count against the observed one, and the final cell comes from
    :func:`~midas_hkls.ub_refine.refine_ub_from_gvectors` so it arrives with a
    covariance rather than bare.
    """
    from .ub_refine import refine_ub_from_gvectors

    g = np.asarray(g, float)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"g must be (N, 3), got {g.shape}")
    g_work = g / (2.0 * math.pi) if two_pi else g
    n = len(g_work)
    notes: List[str] = []

    if n < min_reflections:
        return AbInitioResult(
            success=False, n_reflections=n,
            notes=[f"{n} reflections is below the floor of {min_reflections}. "
                   "The Patterson builds its peaks from difference vectors, so "
                   "too few reflections give no peak above chance and any cell "
                   "it returned would be an artifact. This is a refusal, not a "
                   "failure to converge."])

    cands, diag = find_candidate_vectors(
        g_work, n_grid=n_grid, q_max=q_max, tol=tol,
        min_excess_sigma=min_excess_sigma)
    if len(cands) < 3:
        return AbInitioResult(
            success=False, n_reflections=n, candidates=cands, diagnostics=diag,
            notes=[f"only {len(cands)} candidate vectors beat the chance level "
                   f"of {diag['chance']:.1f} indexed reflections by "
                   f"{min_excess_sigma} sigma. A lattice needs three "
                   "independent ones.",
                   f"grid limits: shortest resolvable {diag['dr']:.3f} A, "
                   f"longest findable {diag['r_max']:.1f} A -- a longer axis "
                   "than that cannot be found at this q_max and n_grid."])

    A, sel = select_basis(cands, g_work, tol=tol, q_max=q_max,
                          min_subset=min_subset)
    notes += sel.get("notes", [])
    if A is None:
        return AbInitioResult(success=False, n_reflections=n, candidates=cands,
                              diagnostics={**diag, **sel}, notes=notes)

    frac, hkl = _index_fraction(A, g_work, tol)
    inl = np.all(np.abs(g_work @ A - hkl) < tol, axis=1)
    n_in = int(inl.sum())
    if n_in < 6:
        return AbInitioResult(
            success=False, n_reflections=n, candidates=cands,
            diagnostics={**diag, **sel},
            notes=notes + [f"only {n_in} reflections index on the accepted "
                           "basis — too few to refine a triclinic cell."])

    try:
        fit = refine_ub_from_gvectors(hkl[inl], g_work[inl], sigma_g=sigma_g)
    except ValueError as exc:
        # a refusal, not a crash: the accepted basis indexed a set of
        # reflections that does not span three dimensions
        return AbInitioResult(
            success=False, n_reflections=n, candidates=cands,
            diagnostics={**diag, **sel},
            notes=notes + [f"the {n_in} indexed reflections do not determine a "
                           f"cell in every direction: {exc}"])
    notes.append(f"grid: shortest resolvable {diag['dr']:.3f} A, longest "
                 f"findable {diag['r_max']:.1f} A.")
    if fit.undetermined:
        notes.append("NOT DETERMINED by these reflections: "
                     + ", ".join(fit.undetermined))
    return AbInitioResult(
        success=True, cell=fit.cell, cell_sigma=fit.cell_sigma, UB=fit.UB,
        UBI=fit.UBI, hkl=hkl[inl], indexed_mask=inl,
        indexed_fraction=frac, n_indexed=n_in,
        n_reflections=n, candidates=cands,
        diagnostics={**diag, **sel, "rms_drlv": fit.rms_drlv}, notes=notes)
