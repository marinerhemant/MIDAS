"""Find the one sample-frame rotation that carries one grain set onto another.

Use case: the same specimen measured twice with the orientation frame changed
in between -- a remount, a sample flipped for a second mounting, an NF layer
against an FF layer taken on a different mount, two beamtimes. No grain
correspondence is given. The model is

    O_b = R . O_a . S        for every grain the two sets share,

with ``O`` the MIDAS orientation matrix (crystal -> sample), ``S`` a crystal
symmetry operator (right side, as in :func:`misorientation_om_batch`) and ``R``
the unknown sample-frame rotation (left side).

Method (vote, score, refine, test):

0. **Distinct orientations.** Both sets are collapsed to one entry per cluster
   of orientations within ``tol_deg`` (:func:`distinct_orientations`), so a
   voxel map's repeated grain orientations count once.
1. **Vote.** Every seed grain ``a`` from set A, paired with every grain ``b`` of
   set B under every symmetry operator ``S``, proposes ``R = O_b S^T O_a^T``.
   Candidates are binned in quaternion space and each bin counts DISTINCT
   seeds: a real frame change collects one vote per seed whose grain is in both
   sets; unrelated sets spread their votes thinly.
2. **Score.** The best bins' mean rotations are scored by the fraction of A
   that lands within ``tol_deg`` of some grain of B after rotation.
3. **Refine.** The matched pairs re-estimate R (chordal quaternion mean) a few
   times.
4. **Test.** The null is the SAME vote/score/refine search run ``n_null`` times
   with A replaced by uniformly random orientations, so it carries the same
   best-of-candidates selection and refinement as the real answer. A result is
   ``significant`` only when the refined score clears that null maximum by a
   margin. Voting uses 16 half-cell-offset quaternion grids so a true cluster
   cannot be split across bin edges.

Status: PROVISIONAL
-------------------
The claim that this method finds the frame rotation is logged as ``c4d3fbb9876c`` and is PROVISIONAL (2026-09-29): not established. The shared-texture false positives and the power limits below are the caveats recorded with it. Treat a ``significant`` result as a lead to check, not as a finding.

Limits (measured in the amendment verify, 2026-09-29)
------------------------------------------------------
* **Texture.** The null draws A from UNIFORM random orientations. Two sets of
  DIFFERENT grains drawn from the same texture (4 components, 8-20 deg spread) were
  called significant in 21 of 28 trials (unrelated sets of 300-3000 grains): the
  search then finds the rotation that registers the two textures, not shared
  grains. ``significant`` therefore certifies "some rotation matches far more
  orientations than random ones do", not "the two sets share grains". For textured
  material, also require (a) the matched orientations to be tight (report
  ``frame_match_fraction`` at 0.5 deg as well as 1 deg: texture matches spread over
  0.5-1.5 deg, shared grains sit below 0.5 deg) and (b) a control pair known to be
  unrelated. A texture-aware null is future work.
* **Power.** With clumped voxel-style A (0.2 deg noise): 0 of 8 detected at 5 %
  shared, 3 of 8 at 10 %, 8 of 8 at 20 % and above. At exactly 30 % shared R is
  correct but ``significant`` may be False (the score 0.30 is below
  ``min_ratio`` x null max).
* **Cost.** About 110 s CPU for 300 x 300; ~1.5 GB and ~20 min CPU for 1500 voxels
  (250 distinct) against 3000 grains.
* **Distinct-orientation collapse** splits a grain with > ~0.3 deg intragranular
  spread into up to 3 representatives and merges grains < ~0.9 deg apart; the null
  is unaffected.
* **Conventions it cannot recover:** a different Euler convention or transposed
  orientation matrix, a different crystal-axis choice (rhombohedral vs hexagonal
  axes), or an improper R (mirror). For cubic crystals a mirror is equivalent to a
  proper rotation and is found.

Numbers are in DEGREES for tolerances and the returned angle (the rotation's
axis-angle), orientation matrices as (N, 9) or (N, 3, 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .orientation import (
    _orient_mat_to_quat_batch,
    make_symmetries,
    misorientation_om_batch,
    quat_to_orient_mat,
)

__all__ = ["FrameRotationResult", "find_frame_rotation", "frame_match_fraction", "distinct_orientations"]


@dataclass
class FrameRotationResult:
    """Outcome of :func:`find_frame_rotation`.

    ``R`` maps set-A orientations onto set B: ``O_b ~ R @ O_a @ S``.
    ``angle_deg``/``axis`` are its axis-angle form. ``score`` is the fraction of
    (sampled) A grains within ``tol_deg`` of a B grain after applying ``R``;
    ``null_scores`` the best score of the SAME search run with A replaced by random
    orientations (selection-matched null); ``identity_score`` for R = I.
    ``significant`` is True only if ``score`` exceeds ``max(null_scores)`` by at
    least ``margin`` (absolute) AND is at least ``min_ratio`` times it.
    """
    R: np.ndarray
    angle_deg: float
    axis: np.ndarray
    score: float
    identity_score: float
    null_scores: np.ndarray
    significant: bool
    n_matched: int
    n_scored: int
    top_votes: list = field(default_factory=list)
    rotation_null_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_distinct_a: int = 0
    n_distinct_b: int = 0

    def summary(self) -> str:
        return (f"R: {self.angle_deg:.3f} deg about [{self.axis[0]:+.4f}, {self.axis[1]:+.4f}, "
                f"{self.axis[2]:+.4f}]; match {self.score:.3f} ({self.n_matched}/{self.n_scored}); "
                f"identity {self.identity_score:.3f}; null median {np.median(self.null_scores):.3f} "
                f"max {np.max(self.null_scores):.3f}; significant={self.significant}; "
                f"distinct A/B {self.n_distinct_a}/{self.n_distinct_b}")


def _stream(seed: int, salt: int) -> np.random.SeedSequence:
    """A generator stream decorrelated from ``default_rng(seed)``.

    A plain ``default_rng(seed)`` here collides with the common pattern of simulating the data
    with ``default_rng(seed)`` and passing the same ``seed`` to the search: the search's "random"
    null orientations then reproduce the simulated set, one null draw scores 1.0 and a real
    rotation is reported as not significant (found by a verify refuter, then in the test suite).
    """
    return np.random.SeedSequence([int(seed), int(salt)])


def _om33(oms) -> np.ndarray:
    a = np.asarray(oms, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 9:
        a = a.reshape(-1, 3, 3)
    if a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError(f"orientation matrices must be (N, 9) or (N, 3, 3), got {np.shape(oms)}")
    return a


def _sym_mats(space_group: int) -> np.ndarray:
    _, sym = make_symmetries(space_group)
    return np.array([quat_to_orient_mat(q) for q in sym], dtype=np.float64).reshape(-1, 3, 3)


def _quat(R: np.ndarray) -> np.ndarray:
    """(n, 3, 3) -> (n, 4) unit quaternions with w >= 0."""
    return _orient_mat_to_quat_batch(np.ascontiguousarray(R.reshape(-1, 9)))


def _axis_angle(R: np.ndarray) -> tuple[float, np.ndarray]:
    q = _quat(R[None])[0]
    ang = 2.0 * np.degrees(np.arccos(np.clip(q[0], -1.0, 1.0)))
    s = np.linalg.norm(q[1:])
    axis = q[1:] / s if s > 1e-12 else np.array([0.0, 0.0, 1.0])
    return float(ang), axis


def _quat_mean_R(q: np.ndarray) -> np.ndarray:
    """Chordal mean of quaternions (sign-aligned to the first) -> 3x3."""
    q = q * np.sign(q @ q[0])[:, None]
    w, v = np.linalg.eigh(q.T @ q)
    m = v[:, -1]
    m = m if m[0] >= 0 else -m
    return np.array(quat_to_orient_mat(m / np.linalg.norm(m)), dtype=np.float64).reshape(3, 3)


def _random_rotations(n: int, rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.array([quat_to_orient_mat(x) for x in q], dtype=np.float64).reshape(-1, 3, 3)


def _best_match(A: np.ndarray, B9: np.ndarray, space_group: int, chunk: int = 64):
    """For each row of A (n, 3, 3): min misorientation (deg) to B and its index."""
    A9 = A.reshape(-1, 9)
    nb = B9.shape[0]
    best = np.empty(len(A9)); arg = np.empty(len(A9), dtype=int)
    for s in range(0, len(A9), chunk):
        a = A9[s:s + chunk]
        d = np.degrees(np.asarray(misorientation_om_batch(
            np.repeat(a, nb, axis=0), np.tile(B9, (len(a), 1)), space_group))).reshape(len(a), nb)
        best[s:s + chunk] = d.min(1); arg[s:s + chunk] = d.argmin(1)
    return best, arg


def distinct_orientations(oms, space_group: int, tol_deg: float = 1.0, seed: int = 0) -> np.ndarray:
    """Indices of one representative per cluster of orientations within ``tol_deg``.

    Greedy: visit orientations in random order; each unassigned one becomes a
    representative and absorbs every unassigned orientation within ``tol_deg``.
    Voxel maps repeat a grain's orientation dozens of times; counting them once
    each keeps a single matched grain from inflating a match fraction.
    """
    O9 = _om33(oms).reshape(-1, 9)
    order = np.random.default_rng(_stream(seed, 0x44494F)).permutation(len(O9))
    free = np.ones(len(O9), bool); reps = []
    for i in order:
        if not free[i]:
            continue
        reps.append(i)
        idx = np.flatnonzero(free)
        d = np.degrees(np.asarray(misorientation_om_batch(np.repeat(O9[[i]], len(idx), 0), O9[idx], space_group)))
        free[idx[d <= tol_deg]] = False
    return np.array(sorted(reps), dtype=int)


def frame_match_fraction(R, om_a, om_b, space_group: int, tol_deg: float = 1.0) -> float:
    """Fraction of ``om_a`` within ``tol_deg`` of some ``om_b`` after ``R @ O_a``."""
    A = _om33(om_a); B9 = _om33(om_b).reshape(-1, 9)
    m, _ = _best_match(np.asarray(R, float).reshape(3, 3)[None] @ A, B9, space_group)
    return float(np.mean(m <= tol_deg))


def _bin_keys(q: np.ndarray, cell: float, offset: np.ndarray) -> np.ndarray:
    """Quaternion -> one int64 bin key on a grid shifted by ``offset`` (in cells)."""
    k = np.floor(q / cell + offset).astype(np.int64) + 512           # |q|<=1 -> |k| < 512 for cell > 0.002
    return ((k[:, 0] * 1024 + k[:, 1]) * 1024 + k[:, 2]) * 1024 + k[:, 3]


def _vote(A, BS, ia, cell, n_candidates):
    """Candidate rotations from seed votes on 16 half-cell-offset grids.

    A true frame rotation's candidates form a tight cluster; on one fixed grid the
    cluster can straddle bin edges in several quaternion components at once and
    fragment its votes down to the noise floor (verify, 2026-09-28: 169.99 deg
    about x, 2 of 7 seeds lost). With every component shifted by 0 or 1/2 cell
    independently, some grid holds the cluster at least 1/4 cell from all edges.
    Returns up to ``n_candidates`` (quaternion, votes), de-duplicated.
    """
    q_all = np.concatenate([_quat(BS @ A[i].T) for i in ia])
    seed_id = np.repeat(np.arange(len(ia)), len(BS))
    cands = []
    for bits in range(16):
        off = np.array([(bits >> d) & 1 for d in range(4)], dtype=np.float64) * 0.5
        key = _bin_keys(q_all, cell, off)
        uk = np.unique(key * 4096 + seed_id)                        # distinct (bin, seed)
        bins, votes = np.unique(uk // 4096, return_counts=True)
        for o in np.argsort(-votes)[:n_candidates]:
            sel = key == bins[o]
            qm = q_all[sel] * np.sign(q_all[sel] @ q_all[sel][0])[:, None]
            m = qm.mean(0); m /= np.linalg.norm(m)
            cands.append((m if m[0] >= 0 else -m, int(votes[o])))
    cands.sort(key=lambda c: -c[1])
    out = []
    for q, v in cands:                                              # merge near-duplicates across grids
        if all(abs(q @ p) < np.cos(cell) for p, _ in out):
            out.append((q, v))
        if len(out) == n_candidates:
            break
    return out


def _search(A, As, B, B9, S, space_group, tol_deg, n_seeds, n_candidates, n_refine, rng):
    """vote -> score top candidates on ``As`` -> refine. Returns (score, R, votes)."""
    ia = rng.choice(len(A), size=min(n_seeds, len(A)), replace=False)
    BS = np.einsum("bij,skj->bsik", B, S).reshape(-1, 3, 3)          # O_b S^T
    cell = np.sin(np.radians(tol_deg) / 2.0) * 2.0                   # quaternion distance ~ angle/2
    cands = _vote(A, BS, ia, cell, n_candidates)
    score, R = -1.0, np.eye(3)
    for q, _ in cands:
        Rm = np.array(quat_to_orient_mat(q), dtype=np.float64).reshape(3, 3)
        sc = float(np.mean(_best_match(Rm[None] @ As, B9, space_group)[0] <= tol_deg))
        if sc > score:
            score, R = sc, Rm
    for _ in range(n_refine):
        m, j = _best_match(R[None] @ As, B9, space_group)
        ok = m <= tol_deg
        if ok.sum() < 3:
            break
        cand = np.einsum("nij,skj->nsik", B[j[ok]], S) @ np.swapaxes(As[ok], 1, 2)[:, None]
        qc = _quat(cand.reshape(-1, 3, 3)).reshape(ok.sum(), len(S), 4)
        q0 = _quat(R[None])[0]
        pick = np.argmax(np.abs(qc @ q0), axis=1)
        R_new = _quat_mean_R(qc[np.arange(len(qc)), pick])
        sc2 = float(np.mean(_best_match(R_new[None] @ As, B9, space_group)[0] <= tol_deg))
        if sc2 < score:
            break
        R, score = R_new, sc2
    return score, R, [v for _, v in cands]


def find_frame_rotation(om_a, om_b, space_group: int, *, tol_deg: float = 1.0,
                        n_seeds: int = 40, n_score: int = 300, n_candidates: int = 12,
                        n_null: int = 8, n_refine: int = 3, margin: float = 0.05,
                        min_ratio: float = 3.0, seed: int = 0) -> FrameRotationResult:
    """Find R with ``O_b = R . O_a . S`` for the grains two orientation sets share.

    Parameters
    ----------
    om_a, om_b : (N, 9) or (N, 3, 3) orientation matrices (MIDAS, crystal -> sample).
    space_group : int, used for the crystal symmetry operators.
    tol_deg : match tolerance (degrees) for scoring and refinement; also sets the
        voting bin size.
    n_seeds : grains of A used to vote. Votes scale as n_seeds x len(B) x n_sym.
    n_score : grains of A (randomly sampled) used to score candidates and nulls.
    n_candidates : candidates scored per search (from 16 offset voting grids).
    n_null : SELECTION-MATCHED null draws. Each replaces A by uniformly random
        orientations of the same size and runs the identical vote/score/refine
        search against B, so the null gets the same best-of-candidates selection
        and refinement as the real answer (a plain random-rotation null does not,
        and is reported only for information as ``rotation_null_scores``).
    margin, min_ratio : significance thresholds against the null maximum.

    Returns
    -------
    FrameRotationResult. ``R`` is the best estimate even when not significant;
    always read ``significant`` before using it.
    """
    rng = np.random.default_rng(_stream(seed, 0x52464D))
    A_in = _om33(om_a); B_in = _om33(om_b)
    # one entry per distinct orientation: voxel maps repeat each grain's orientation,
    # which lets one lucky rotation match a whole clump while the random null has
    # no clumps (HAO NF vs FF 2026-09-28: unrelated sets 12-15 % vs a ~4.5 % null)
    A = A_in[distinct_orientations(A_in, space_group, tol_deg, seed)]
    B = B_in[distinct_orientations(B_in, space_group, tol_deg, seed)]
    B9 = B.reshape(-1, 9)
    S = _sym_mats(space_group)
    isc = rng.choice(len(A), size=min(n_score, len(A)), replace=False)
    As = A[isc]
    score, R, votes = _search(A, As, B, B9, S, space_group, tol_deg, n_seeds, n_candidates, n_refine, rng)

    m_id, _ = _best_match(As, B9, space_group)
    identity_score = float(np.mean(m_id <= tol_deg))
    null = []
    for _ in range(n_null):
        An = _random_rotations(len(A), rng)
        null.append(_search(An, An[:len(As)], B, B9, S, space_group, tol_deg, n_seeds,
                            n_candidates, n_refine, rng)[0])
    null = np.array(null)
    rot_null = np.array([float(np.mean(_best_match(Rr[None] @ As, B9, space_group)[0] <= tol_deg))
                         for Rr in _random_rotations(10, rng)])
    nmax = float(null.max()) if len(null) else 0.0
    significant = bool(score >= nmax + margin and score >= min_ratio * max(nmax, 1e-9))
    ang, axis = _axis_angle(R)
    m_fin, _ = _best_match(R[None] @ As, B9, space_group)
    return FrameRotationResult(R=R, angle_deg=ang, axis=axis, score=score,
                               identity_score=identity_score, null_scores=null,
                               significant=significant, n_matched=int((m_fin <= tol_deg).sum()),
                               n_scored=len(As), top_votes=votes, rotation_null_scores=rot_null,
                               n_distinct_a=len(A), n_distinct_b=len(B))
