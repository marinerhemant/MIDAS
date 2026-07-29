"""Indexing from the sparse merged spot cloud.

The determinability and reconstruction elsewhere assume the spot->reflection
correspondence is known.  Real data does not: we get an unlabelled spot cloud and
must find the grains.  This module answers the go/no-go question -- *can the
sparse four-wedge, two/three-mounting data uniquely index grain orientations?* --
with a forward-matching scorer and two tools:

* :func:`orientation_uniqueness` -- for each true grain, compare the match score
  at the true orientation against many random orientations.  A large margin means
  the sparse spots pin the orientation (indexable); a small margin means aliasing.
* :func:`grid_index` -- an actual from-scratch indexer: score a coarse SO(3)
  grid against the measured spots, keep peaks, refine, and report recall vs truth.

Grain position barely shifts far-field spot positions (micron offset over a
metre), so orientation scoring is effectively position-independent -- which is
why orientation indexes cleanly even though position is only weakly constrained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import XAFConfig
from .forward import XAFForwardModel
from .sample import GrainPopulation


def predict_grain_spots(fwd: XAFForwardModel, euler: np.ndarray,
                        pos: np.ndarray) -> Dict[int, np.ndarray]:
    """Accessible (y, z, omega_deg) predicted for one candidate grain, per mounting."""
    dtype = fwd._latc0.dtype
    g = GrainPopulation(
        euler=torch.as_tensor(euler, dtype=dtype).view(1, 3),
        position=torch.as_tensor(pos, dtype=dtype).view(1, 3),
        strain=torch.zeros(1, 6, dtype=dtype))
    with torch.no_grad():
        sim = fwd.simulate(g)
    t = sim.table
    out = {}
    mid = t.mounting_id.cpu().numpy()
    yy = t.y_pixel.cpu().numpy(); zz = t.z_pixel.cpu().numpy()
    om = np.degrees(t.omega.cpu().numpy())
    for m in range(fwd.cfg.n_mountings):
        s = mid == m
        out[m] = np.stack([yy[s], zz[s], om[s]], axis=1)
    return out


def _build_trees(measured, tol_px: float, tol_omega: float, n_mountings: int):
    from scipy.spatial import cKDTree
    trees = {}
    for m in range(n_mountings):
        ms = measured.for_mounting(m)
        if len(ms) == 0:
            trees[m] = None
            continue
        pts = np.stack([ms.y_pixel / tol_px, ms.z_pixel / tol_px,
                        _wrap(ms.omega_deg) / tol_omega], axis=1)
        trees[m] = cKDTree(pts)
    return trees


def _wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def score_orientation(fwd, euler, pos, trees, tol_px, tol_omega):
    """Fraction (and count) of a candidate's predicted spots matched by a measured
    spot within tolerance."""
    pred = predict_grain_spots(fwd, euler, pos)
    n_pred = matched = 0
    for m, P in pred.items():
        if P.shape[0] == 0 or trees[m] is None:
            continue
        q = np.stack([P[:, 0] / tol_px, P[:, 1] / tol_px, _wrap(P[:, 2]) / tol_omega], axis=1)
        d, _ = trees[m].query(q, k=1)
        matched += int((d <= 1.0).sum())
        n_pred += P.shape[0]
    return matched, n_pred


@dataclass
class UniquenessResult:
    grain_id: int
    n_pred: int
    true_matched: int
    best_random_matched: int
    margin: int              # true_matched - best_random_matched
    indexable: bool


def orientation_uniqueness(
    fwd: XAFForwardModel,
    grains: GrainPopulation,
    measured,
    *,
    n_random: int = 300,
    tol_px: float = 3.0,
    tol_omega: float = 1.0,
    min_margin: int = 6,
    seed: int = 0,
) -> Dict[str, object]:
    """Per-grain true-vs-random match scores: is the orientation pinned?"""
    from .sample import _random_euler
    trees = _build_trees(measured, tol_px, tol_omega, fwd.cfg.n_mountings)
    rng = np.random.default_rng(seed)
    rand_euler = _random_euler(rng, n_random)

    results: List[UniquenessResult] = []
    for gi in range(grains.n_grains):
        e = grains.euler[gi].cpu().numpy()
        p = grains.position[gi].cpu().numpy()
        tm, npd = score_orientation(fwd, e, p, trees, tol_px, tol_omega)
        best_rand = 0
        for re in rand_euler:
            rm, _ = score_orientation(fwd, re, p, trees, tol_px, tol_omega)
            if rm > best_rand:
                best_rand = rm
        margin = tm - best_rand
        results.append(UniquenessResult(gi, npd, tm, best_rand, margin,
                                        margin >= min_margin))
    idx = np.array([r.indexable for r in results])
    return {
        "per_grain": results,
        "frac_indexable": float(idx.mean()),
        "median_true_matched": float(np.median([r.true_matched for r in results])),
        "median_best_random": float(np.median([r.best_random_matched for r in results])),
        "median_margin": float(np.median([r.margin for r in results])),
    }


def sample_scattering_vectors(measured, cfg):
    """Grain-fixed (common-frame) scattering vector G for each measured spot.

    G_common = R_mount^T . R_z(-omega) . (k_out - k_in), in 1/Angstrom.  This is
    what an orientation must map crystal reflections onto -- the basis of
    vector-space (Friedel-seeded) indexing.  Opposite-face Friedel mates give
    +G and -G, so the set is centrosymmetric and robust.
    """
    from . import geometry as geo
    wl = cfg.wavelength_A
    tt = np.radians(measured.two_theta_deg)
    eta = np.radians(measured.eta_deg)
    om = np.radians(measured.omega_deg)
    # scattered unit dir (forward-model convention) -> Q_lab = (1/wl)(d_hat - x)
    s = np.sin(tt)
    dhat = np.stack([np.cos(tt), -s * np.sin(eta), s * np.cos(eta)], axis=1)
    Q = (dhat - np.array([1.0, 0, 0])) / wl                      # (S,3), 1/A
    co, so = np.cos(-om), np.sin(-om)                            # R_z(-omega)
    Qx = Q[:, 0] * co - Q[:, 1] * so
    Qy = Q[:, 0] * so + Q[:, 1] * co
    Gsamp = np.stack([Qx, Qy, Q[:, 2]], axis=1)
    G = np.empty_like(Gsamp)
    for m in range(cfg.n_mountings):
        sel = measured.mounting_id == m
        R = np.asarray(geo.mounting_matrix(cfg, m), float)
        G[sel] = Gsamp[sel] @ R                                  # R^T applied (row-vec)
    return G


def friedel_seeded_index(
    fwd: XAFForwardModel,
    measured,
    *,
    n_seed_spots: int = 40,
    mag_tol: float = 0.02,
    angle_tol_deg: float = 1.5,
    match_tol_deg: float = 1.5,
    min_matched: int = 8,
    dedupe_deg: float = 3.0,
    adapt_frac: float = 0.5,
) -> Dict[str, object]:
    """Vector-space indexer seeded by measured scattering vectors.

    Returns orientation-matrix seeds. Recall (fraction of grains found) is the
    goal; false-positive seeds are cheaply rejected downstream by the
    forward-matching verify/refine step. An adaptive threshold at
    ``adapt_frac`` of the best score removes weak partial-match seeds."""
    from scipy.spatial import cKDTree
    from .crystal import build_reflections
    from midas_stress.orientation import euler_to_orient_mat_batch  # noqa: F401
    cfg = fwd.cfg
    G = sample_scattering_vectors(measured, cfg)
    gmag = np.linalg.norm(G, axis=1)
    ghat = G / gmag[:, None]
    tree = cKDTree(ghat)                       # for scoring (unit directions)
    match_chord = 2.0 * math.sin(math.radians(match_tol_deg) / 2.0)

    hkls_cart, _, _ = build_reflections(cfg.material, cfg.wavelength_A, cfg.tth_max_deg)
    H = hkls_cart.cpu().numpy()
    hmag = np.linalg.norm(H, axis=1)
    hhat = H / hmag[:, None]

    def candidates(gm):                         # crystal vecs with matching |G|
        return np.where(np.abs(hmag - gm) < mag_tol * gm)[0]

    def score(R):                               # crystal vecs matched in dir AND |G|
        pred = hhat @ R.T
        nbrs = tree.query_ball_point(pred, r=match_chord)
        matched = 0
        for h_i, js in enumerate(nbrs):
            if js and np.any(np.abs(gmag[js] - hmag[h_i]) < mag_tol * hmag[h_i]):
                matched += 1
        return matched

    # Single-reflection seeding with a rotation scan: for each bright measured
    # vector g and candidate crystal vector h (|h|~|g|), align h->g and scan the
    # remaining 1-DOF rotation about g.  Only ONE correct (g,h) assignment is
    # needed to hit a grain -- far more robust than two-vector seeding.
    n_phi = 24
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    order = np.argsort(-measured.intensity)[:n_seed_spots]
    kept_R, kept_score = [], []
    for ig in order:
        g = ghat[ig]
        for hi in candidates(gmag[ig])[:16]:
            R0 = _align(hhat[hi], g)
            if R0 is None:
                continue
            best_R, best_s = None, 0
            for phi in phis:
                R = _rot_about(g, phi) @ R0
                s = score(R)
                if s > best_s:
                    best_s, best_R = s, R
            if best_R is not None and best_s >= min_matched:
                _dedupe_add(kept_R, kept_score, best_R, best_s, dedupe_deg)
    if kept_score:                             # adaptive threshold vs best seed
        thr = max(min_matched, adapt_frac * max(kept_score))
        keep = [i for i, s in enumerate(kept_score) if s >= thr]
        kept_R = [kept_R[i] for i in keep]
        kept_score = [kept_score[i] for i in keep]
    return {"orientations_mat": kept_R, "matched": kept_score,
            "n_found": len(kept_R)}


def _align(h, g):
    """Minimal rotation R with R h = g (unit vectors)."""
    v = np.cross(h, g)
    s = np.linalg.norm(v)
    c = float(np.clip(h @ g, -1, 1))
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _rot_about(axis, angle):
    a = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    ax = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * ax + (1 - c) * np.outer(a, a)


def _orient_from_two_vec(h1, h2, g1, g2):
    """Rotation R with R h1 ~ g1, R h2 ~ g2 (orthonormal-frame method)."""
    def frame(a, b):
        e1 = a / np.linalg.norm(a)
        e3 = np.cross(a, b)
        n = np.linalg.norm(e3)
        if n < 1e-6:
            return None
        e3 /= n
        e2 = np.cross(e3, e1)
        return np.stack([e1, e2, e3], axis=1)
    Fh, Fg = frame(h1, h2), frame(g1, g2)
    if Fh is None or Fg is None:
        return None
    return Fg @ Fh.T


def _dedupe_add(kept_R, kept_score, R, s, tol_deg):
    for i, K in enumerate(kept_R):
        dR = R @ K.T
        ang = math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        if ang < tol_deg:
            if s > kept_score[i]:
                kept_R[i], kept_score[i] = R, s
            return
    kept_R.append(R); kept_score.append(s)


def grid_index(
    fwd: XAFForwardModel,
    measured,
    *,
    n_candidates: int = 20000,
    tol_px: float = 3.0,
    tol_omega: float = 1.0,
    min_completeness: float = 0.5,
    min_matched: int = 8,
    seed: int = 0,
) -> Dict[str, object]:
    """From-scratch indexer: score random SO(3) candidates, keep high-scoring
    orientations as grain hypotheses (deduplicated by misorientation)."""
    from .sample import _random_euler
    from midas_stress.orientation import euler_to_orient_mat_batch
    trees = _build_trees(measured, tol_px, tol_omega, fwd.cfg.n_mountings)
    rng = np.random.default_rng(seed)
    cand = _random_euler(rng, n_candidates)
    zero = np.zeros(3)

    hyp = []
    for e in cand:
        m, npd = score_orientation(fwd, e, zero, trees, tol_px, tol_omega)
        if npd > 0 and m >= min_matched and m / npd >= min_completeness:
            hyp.append((m, e))
    hyp.sort(key=lambda x: -x[0])
    # dedupe by misorientation (>3 deg apart => distinct grain)
    kept_e, kept_m = [], []
    oms = [euler_to_orient_mat_batch(np.array([e]))[0].reshape(3, 3) for _, e in hyp]
    for (score, e), om in zip(hyp, oms):
        new = True
        for ke in kept_e:
            ok = euler_to_orient_mat_batch(np.array([ke]))[0].reshape(3, 3)
            dR = om @ ok.T
            ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            if ang < 3.0:
                new = False
                break
        if new:
            kept_e.append(e); kept_m.append(score)
    return {"orientations": np.array(kept_e), "matched": np.array(kept_m),
            "n_found": len(kept_e), "n_candidates": n_candidates}
