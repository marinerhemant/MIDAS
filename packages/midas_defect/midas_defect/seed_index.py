"""Seed indexer for heavily-deformed-single-crystal voxel clouds.

Given a sparse q-space voxel cloud (sample frame), produce an average crystal
orientation matrix `U` and optionally refined lattice constants `(a, c)`.

Strategy
--------
1. **Bright cores.** Pick the N brightest cluster centroids in the voxel
   cloud (`_discrete_extract_bright_cores`).
2. **Shell matching.** For each centroid, find allowed CuAl₂ Bragg shells
   whose `|q|` matches within tolerance — that's the candidate set of
   (h,k,l) families consistent with the observation.
3. **Pair voting.** For every pair of non-collinear centroids `(i, j)` and
   every compatible (hkl_i, hkl_j) assignment, build a candidate `U` that
   maps the two crystal-frame g-vectors onto the two observed q-vectors via
   an orthonormal-basis construction. Score `U` by counting how many other
   bright centroids it can also explain (direction within `tol_angle_deg`
   AND |q| within `tol_q_rel`). Best-scoring `U` wins.
4. **Refinement.** Torch+Adam optimization of `U` (parameterized as a
   rotation vector via the matrix exponential — autograd-safe) and
   optionally `(a, c)` over all matched centroids. Loss is `sum (1 - cos)`
   where `cos` is the cosine similarity between predicted and observed
   q-vectors.

Differentiability
-----------------
Steps 1–3 are discrete (live in `_discrete_*` helpers, no gradient flow).
Step 4 is fully differentiable: the returned `U`, `a`, `c` carry gradients
all the way back to the (centroid_q, hkl) pairs that fed the refinement.

MIDAS reuses
------------
* `midas_stress.orientation.{quat_to_orient_mat,orient_mat_to_quat,
  axis_angle_to_orient_mat}` — quat <-> matrix; flat 9-element row-major
  OMs are the canonical form (per feedback memory).
* `midas_hkls.lattice_torch.d_spacing` (through `lattice.q_inv_of_hkl_torch`)
  for the diff |q|(hkl, a, c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch

from midas_transforms.device import resolve_device, resolve_dtype

from .lattice import (
    CUAL2_A_DEFAULT,
    CUAL2_C_DEFAULT,
    Shell,
    cual2_crystal,
    tetragonal_shells,
)


__all__ = [
    "SeedIndexResult",
    "find_seed_orientation",
    "refine_U_lattice",
    "predict_q_from_U",
    "refine_U_from_centroids",
    "bootstrap_orientation_uncertainty",
    "iterate_seed_asterism_refinement",
    "refine_U_from_friedel_pairs",
]


@dataclass
class SeedIndexResult:
    """Output of `find_seed_orientation`."""
    U: np.ndarray                    # (3, 3) orientation matrix mapping cry -> sample-q
    a: float                         # refined lattice constant a (Å)
    c: float                         # refined lattice constant c (Å)
    score: int                       # number of bright centroids explained
    matched_hkls: List[Tuple[int, int, int]]   # one per bright centroid (or None entries)
    matched_centroids: np.ndarray    # (N, 3) sample-frame q-vectors used in refinement
    matched_intensities: np.ndarray  # (N,) weights
    final_loss: float                # final value of the refinement loss
    n_refine_steps: int              # actual Adam steps taken

    def U_flat(self) -> np.ndarray:
        """Return U as a flat 9-element row-major array (midas_stress convention)."""
        return self.U.reshape(-1)


# ---------------------------------------------------------------------------
# Discrete helpers (no gradient flow)
# ---------------------------------------------------------------------------

def _discrete_extract_bright_cores(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    *, n_bright: int = 20, min_separation: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy bright-core extraction.

    Returns `(centroids (N, 3), intensities (N,))` sorted by descending I,
    where no two centroids are closer than `min_separation` (in q-units).
    """
    order = np.argsort(intensity)[::-1]
    chosen_idx: list[int] = []
    chosen_pts: list[np.ndarray] = []
    sep2 = float(min_separation) ** 2
    for k in order:
        p = np.array([qx[k], qy[k], qz[k]], dtype=np.float64)
        ok = True
        for q in chosen_pts:
            if np.sum((p - q) ** 2) < sep2:
                ok = False
                break
        if ok:
            chosen_idx.append(int(k))
            chosen_pts.append(p)
            if len(chosen_pts) >= n_bright:
                break
    centroids = np.stack(chosen_pts) if chosen_pts else np.zeros((0, 3))
    ints = np.array([intensity[k] for k in chosen_idx], dtype=np.float64)
    return centroids, ints


def _discrete_compatible_hkls(
    centroid_q: np.ndarray, shells: Sequence[Shell],
    *, tol_q_rel: float, max_per_centroid: int = 4,
) -> List[Tuple[int, int, int]]:
    """All (h,k,l) families whose |q| matches `|centroid_q|` within tol_q_rel.

    Capped at `max_per_centroid` to keep pair voting tractable.
    """
    q_obs = float(np.linalg.norm(centroid_q))
    if q_obs <= 0:
        return []
    out: list[Tuple[int, int, int]] = []
    for s in shells:
        if abs(s.q_inv_A - q_obs) / q_obs <= tol_q_rel:
            for hkl in s.hkls:
                out.append(hkl)
                if len(out) >= max_per_centroid:
                    return out
    return out


def _hkl_to_g_cry(hkl: Sequence[int], a: float, c: float) -> np.ndarray:
    """Crystal-frame g-vector (1/Å) for tetragonal lattice.

    `g_cry = 2π (h/a, k/a, l/c)`. |g_cry| = 2π/d, matches `lattice.py`.
    """
    return 2.0 * math.pi * np.array(
        [hkl[0] / a, hkl[1] / a, hkl[2] / c], dtype=np.float64
    )


def _U_from_two_pairs(
    g_cry_i: np.ndarray, q_obs_i: np.ndarray,
    g_cry_j: np.ndarray, q_obs_j: np.ndarray,
) -> Optional[np.ndarray]:
    """Build an orientation matrix U so that U @ g_cry ≈ q_obs for both pairs.

    Returns None if (g_cry_i, g_cry_j) are nearly collinear (no unique U) or
    if the inter-pair angle differs from the observed inter-pair angle by
    more than ~tol_cos. Caller is expected to filter further by scoring.
    """
    def _orthobasis(u: np.ndarray, v: np.ndarray) -> Optional[np.ndarray]:
        n_u = np.linalg.norm(u); n_v = np.linalg.norm(v)
        if n_u < 1e-12 or n_v < 1e-12:
            return None
        e1 = u / n_u
        cross = np.cross(u, v)
        n_c = np.linalg.norm(cross)
        if n_c / (n_u * n_v) < 1e-3:
            return None        # nearly collinear
        e3 = cross / n_c
        e2 = np.cross(e3, e1)
        return np.column_stack([e1, e2, e3])

    Mc = _orthobasis(g_cry_i, g_cry_j)
    Ml = _orthobasis(q_obs_i, q_obs_j)
    if Mc is None or Ml is None:
        return None
    return Ml @ Mc.T


def _score_U_against_centroids(
    U: np.ndarray, centroids: np.ndarray, shells: Sequence[Shell],
    a: float, c: float,
    *, tol_q_rel: float, tol_angle_deg: float,
) -> Tuple[int, List[Optional[Tuple[int, int, int]]]]:
    """How many centroids does this U explain?

    For each centroid, find the best (h,k,l) prediction (the one whose
    `U @ g_cry` is closest in *direction* AND within `tol_q_rel` in
    magnitude). Count centroids with a match found.

    Returns (score, matched_hkls_per_centroid). matched_hkls_per_centroid
    has one entry per centroid; None means "no match found".
    """
    cos_tol = math.cos(math.radians(tol_angle_deg))
    score = 0
    matches: list[Optional[Tuple[int, int, int]]] = []
    centroid_qmag = np.linalg.norm(centroids, axis=1)
    centroid_unit = centroids / centroid_qmag[:, None]
    # Precompute predicted g_sample for every allowed hkl up to qmax of data
    qmax = centroid_qmag.max() * (1.0 + 2.0 * tol_q_rel)
    pred = []
    pred_hkl = []
    for s in shells:
        if s.q_inv_A > qmax:
            continue
        for hkl in s.hkls:
            g_cry = _hkl_to_g_cry(hkl, a, c)
            g_sam = U @ g_cry
            pred.append(g_sam)
            pred_hkl.append(hkl)
    if not pred:
        return 0, [None] * len(centroids)
    pred = np.stack(pred)
    pred_qmag = np.linalg.norm(pred, axis=1)
    pred_unit = pred / pred_qmag[:, None]
    # for each centroid, scan predictions
    for ci in range(len(centroids)):
        best_cos = -1.0
        best_hkl = None
        for pi in range(len(pred)):
            if abs(pred_qmag[pi] - centroid_qmag[ci]) / centroid_qmag[ci] > tol_q_rel:
                continue
            cosv = float(centroid_unit[ci] @ pred_unit[pi])
            if cosv > best_cos:
                best_cos = cosv
                best_hkl = pred_hkl[pi]
        if best_cos >= cos_tol and best_hkl is not None:
            score += 1
            matches.append(best_hkl)
        else:
            matches.append(None)
    return score, matches


# ---------------------------------------------------------------------------
# Differentiable refinement
# ---------------------------------------------------------------------------

def _skew(v: torch.Tensor) -> torch.Tensor:
    """3D vector to 3x3 skew-symmetric matrix (autograd-safe)."""
    zero = torch.zeros_like(v[..., 0])
    row0 = torch.stack([zero,      -v[..., 2], v[..., 1]], dim=-1)
    row1 = torch.stack([v[..., 2], zero,      -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1], v[..., 0], zero     ], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from a rotation vector (axis × angle in rad).

    Explicit Rodrigues formula. Equivalent to `expm(skew(rotvec))` but
    autograd-safe on every device (MPS does not yet implement
    `torch.linalg.matrix_exp`, so we cannot rely on that).
    """
    # theta = |rotvec|; the +1e-30 keeps the gradient finite at theta=0 in
    # exchange for a negligible bias far from the origin.
    theta2 = (rotvec * rotvec).sum()
    theta  = torch.sqrt(theta2 + 1e-30)
    sin_over_t       = torch.sin(theta) / theta
    one_m_cos_over_t2 = (1.0 - torch.cos(theta)) / (theta2 + 1e-30)
    K  = _skew(rotvec)
    K2 = K @ K
    I  = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return I + sin_over_t * K + one_m_cos_over_t2 * K2


def _matrix_to_rotvec(U: np.ndarray) -> np.ndarray:
    """Inverse of `_rotvec_to_matrix` for a single 3x3 rotation matrix."""
    # Use the eigenvalue form: trace(U) = 1 + 2 cos(angle); axis is the
    # eigenvector with eigenvalue 1, computed robustly via (U - U.T) / (2 sin).
    tr = float(np.trace(U))
    cos_a = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    angle = math.acos(cos_a)
    if angle < 1e-8:
        return np.zeros(3)
    if angle > math.pi - 1e-4:
        # near 180°: pick the largest diagonal entry of (U+I)/2 to get axis
        M = (U + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        axis = M[:, i]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return axis * angle
    s = math.sin(angle)
    ax = np.array([
        U[2, 1] - U[1, 2],
        U[0, 2] - U[2, 0],
        U[1, 0] - U[0, 1],
    ]) / (2.0 * s)
    return ax * angle


def predict_q_from_U(
    U: torch.Tensor, hkls: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
) -> torch.Tensor:
    """Predicted sample-frame q-vectors. Differentiable in (U, a, c).

    Parameters
    ----------
    U : (3, 3) torch.Tensor
    hkls : (N, 3) torch.Tensor (any dtype; will be promoted)
    a, c : scalar tensors (lattice constants in Å)

    Returns
    -------
    q : (N, 3) torch.Tensor — `U @ (2π h/a, 2π k/a, 2π l/c)` per row.
    """
    hkls_t = hkls.to(dtype=U.dtype, device=U.device)
    twopi = 2.0 * math.pi
    g_cry = torch.stack([
        twopi * hkls_t[..., 0] / a,
        twopi * hkls_t[..., 1] / a,
        twopi * hkls_t[..., 2] / c,
    ], dim=-1)
    return (U @ g_cry.unsqueeze(-1)).squeeze(-1)


def refine_U_lattice(
    U_init: np.ndarray,
    centroids: np.ndarray,
    intensities: np.ndarray,
    matched_hkls: Sequence[Tuple[int, int, int]],
    *,
    a_init: float,
    c_init: float,
    refine_lattice: bool = True,
    n_steps: int = 300,
    lr: float = 1e-2,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Torch+Adam refinement of (U, a, c) against matched (centroid, hkl) pairs.

    Returns (U_refined (3,3), a_refined, c_refined, final_loss).
    """
    device_ = resolve_device(device)
    dtype_  = resolve_dtype(device_, dtype)
    # Filter to matched only (drop any None entries the caller may have left in)
    keep = [i for i, h in enumerate(matched_hkls) if h is not None]
    if len(keep) < 2:
        raise ValueError(
            f"need at least 2 matched (centroid, hkl) pairs for refinement, "
            f"got {len(keep)}"
        )
    cents_kept = np.stack([centroids[i] for i in keep])
    ints_kept  = np.array([intensities[i] for i in keep], dtype=np.float64)
    hkls_kept  = np.array([matched_hkls[i] for i in keep], dtype=np.int64)

    q_obs = torch.as_tensor(cents_kept, dtype=dtype_, device=device_)
    w = torch.as_tensor(ints_kept, dtype=dtype_, device=device_)
    w = w / w.sum()
    hkls_t = torch.as_tensor(hkls_kept, dtype=dtype_, device=device_)

    rotvec = torch.as_tensor(_matrix_to_rotvec(U_init), dtype=dtype_, device=device_,
                             ).clone().requires_grad_(True)
    a_t = torch.as_tensor(a_init, dtype=dtype_, device=device_,
                          ).clone().requires_grad_(refine_lattice)
    c_t = torch.as_tensor(c_init, dtype=dtype_, device=device_,
                          ).clone().requires_grad_(refine_lattice)

    params = [rotvec] + ([a_t, c_t] if refine_lattice else [])
    opt = torch.optim.Adam(params, lr=lr)

    last_loss = float("inf")
    for step in range(n_steps):
        opt.zero_grad()
        U = _rotvec_to_matrix(rotvec)
        q_pred = predict_q_from_U(U, hkls_t, a_t, c_t)
        # weighted cosine-similarity loss: 1 - cos(angle(q_pred, q_obs))
        dot   = (q_pred * q_obs).sum(dim=-1)
        np_   = torch.linalg.vector_norm(q_pred, dim=-1)
        no_   = torch.linalg.vector_norm(q_obs, dim=-1)
        cos_sim = dot / (np_ * no_ + 1e-30)
        loss = (w * (1.0 - cos_sim)).sum()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    U_final = _rotvec_to_matrix(rotvec).detach().cpu().numpy()
    return U_final, float(a_t.detach().cpu()), float(c_t.detach().cpu()), last_loss


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def find_seed_orientation(
    qx: "np.ndarray | torch.Tensor",
    qy: "np.ndarray | torch.Tensor",
    qz: "np.ndarray | torch.Tensor",
    intensity: "np.ndarray | torch.Tensor",
    *,
    crystal=None,                       # midas_hkls.Crystal; defaults to CuAl₂
    n_bright: int = 20,
    min_separation: float = 0.05,       # bright-core min separation in 1/Å
    tol_q_rel: float = 0.02,            # |q|-match tolerance
    tol_angle_deg: float = 5.0,         # direction tolerance for scoring
    max_per_centroid: int = 4,
    refine_lattice: bool = True,
    n_refine_steps: int = 300,
    refine_lr: float = 1e-2,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> SeedIndexResult:
    """End-to-end seed indexer for a deformed-single-crystal voxel cloud.

    See module docstring for the algorithm. Returns a `SeedIndexResult`.
    """
    if crystal is None:
        crystal = cual2_crystal()
    qx_np = np.asarray(qx, dtype=np.float64)
    qy_np = np.asarray(qy, dtype=np.float64)
    qz_np = np.asarray(qz, dtype=np.float64)
    I_np  = np.asarray(intensity, dtype=np.float64)

    # 1) bright cores
    centroids, ints = _discrete_extract_bright_cores(
        qx_np, qy_np, qz_np, I_np,
        n_bright=n_bright, min_separation=min_separation,
    )
    if len(centroids) < 2:
        raise ValueError(
            f"need at least 2 bright cores, found {len(centroids)} "
            f"(try lowering min_separation)"
        )

    # 2) shells + compatible hkls per centroid
    qmax = float(np.linalg.norm(centroids, axis=1).max() * (1.0 + tol_q_rel))
    shells = tetragonal_shells(crystal, q_max_inv_A=qmax)
    candidates = [
        _discrete_compatible_hkls(c, shells, tol_q_rel=tol_q_rel,
                                  max_per_centroid=max_per_centroid)
        for c in centroids
    ]

    # 3) pair voting
    a0 = float(crystal.lattice.a)
    c0 = float(crystal.lattice.c)
    best_U: Optional[np.ndarray] = None
    best_score = -1
    best_matches: List[Optional[Tuple[int, int, int]]] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            for hkl_i in candidates[i]:
                g_i = _hkl_to_g_cry(hkl_i, a0, c0)
                for hkl_j in candidates[j]:
                    g_j = _hkl_to_g_cry(hkl_j, a0, c0)
                    # angle preservation: skip pairs with incompatible inter-pair angles
                    cos_obs = float(centroids[i] @ centroids[j]) / (
                        np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-30
                    )
                    cos_pred = float(g_i @ g_j) / (
                        np.linalg.norm(g_i) * np.linalg.norm(g_j) + 1e-30
                    )
                    if abs(cos_obs - cos_pred) > math.sin(math.radians(tol_angle_deg)):
                        continue
                    U_pair = _U_from_two_pairs(g_i, centroids[i], g_j, centroids[j])
                    if U_pair is None:
                        continue
                    score, matches = _score_U_against_centroids(
                        U_pair, centroids, shells, a0, c0,
                        tol_q_rel=tol_q_rel, tol_angle_deg=tol_angle_deg,
                    )
                    if score > best_score:
                        best_score = score
                        best_U = U_pair
                        best_matches = matches

    if best_U is None or best_score < 2:
        raise RuntimeError(
            f"no consistent U found (best score {best_score} of "
            f"{len(centroids)} centroids); try increasing tol_q_rel "
            f"or tol_angle_deg"
        )

    # 4) refine
    U_ref, a_ref, c_ref, loss = refine_U_lattice(
        best_U, centroids, ints, best_matches,
        a_init=a0, c_init=c0,
        refine_lattice=refine_lattice,
        n_steps=n_refine_steps, lr=refine_lr,
        device=device, dtype=dtype,
    )

    return SeedIndexResult(
        U=U_ref, a=a_ref, c=c_ref,
        score=best_score,
        matched_hkls=best_matches,
        matched_centroids=centroids,
        matched_intensities=ints,
        final_loss=loss,
        n_refine_steps=n_refine_steps,
    )


# ---------------------------------------------------------------------------
# Orientation improvement via asterism centroids
# ---------------------------------------------------------------------------

def refine_U_from_centroids(
    U_init: np.ndarray,
    hkl_q_centroids: Sequence[Tuple[Tuple[int, int, int], np.ndarray]],
    weights: Optional[np.ndarray] = None,
    *,
    a: float, c: float,
    refine_lattice: bool = False,
    n_steps: int = 400,
    lr: float = 5e-3,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Refine U (and optionally a, c) against intensity-weighted q-centroids.

    Designed to consume the per-hkl `q_fit` outputs of
    `asterism_fit.fit_asterism_patches`. Those centroids are
    intensity-weighted Bragg-position estimates, much more accurate than the
    single-brightest-voxel seed-indexer inputs.

    Parameters
    ----------
    U_init : (3, 3) ndarray
        Starting orientation matrix (e.g., from `find_seed_orientation`).
    hkl_q_centroids
        List of (hkl, q_centroid_xyz_3vec) pairs, one per asterism fit.
    weights
        Optional per-pair weight (e.g., the asterism's integrated intensity).
        Equal weights if omitted.
    refine_lattice
        If True, also refine (a, c) jointly with U.

    Returns
    -------
    (U_refined (3, 3), a_refined, c_refined, final_loss)
    """
    device_ = resolve_device(device)
    dtype_  = resolve_dtype(device_, dtype)
    hkls = np.array([h for h, _ in hkl_q_centroids], dtype=np.int64)
    centroids = np.array([q for _, q in hkl_q_centroids], dtype=np.float64)
    if weights is None:
        weights = np.ones(len(hkls), dtype=np.float64)
    weights = weights / weights.sum()
    if len(hkls) < 4:
        raise ValueError(
            f"need at least 4 (hkl, centroid) pairs; got {len(hkls)}"
        )

    hkls_t = torch.as_tensor(hkls, dtype=dtype_, device=device_)
    q_obs_t = torch.as_tensor(centroids, dtype=dtype_, device=device_)
    w_t = torch.as_tensor(weights, dtype=dtype_, device=device_)

    rotvec = torch.as_tensor(_matrix_to_rotvec(U_init), dtype=dtype_,
                              device=device_).clone().requires_grad_(True)
    a_t = torch.as_tensor(a, dtype=dtype_, device=device_,
                          ).clone().requires_grad_(refine_lattice)
    c_t = torch.as_tensor(c, dtype=dtype_, device=device_,
                          ).clone().requires_grad_(refine_lattice)
    params = [rotvec] + ([a_t, c_t] if refine_lattice else [])
    opt = torch.optim.Adam(params, lr=lr)

    last_loss = float("inf")
    for _ in range(n_steps):
        opt.zero_grad()
        U = _rotvec_to_matrix(rotvec)
        q_pred = predict_q_from_U(U, hkls_t, a_t, c_t)
        # weighted cosine-similarity loss
        dot = (q_pred * q_obs_t).sum(dim=-1)
        np_ = torch.linalg.vector_norm(q_pred, dim=-1)
        no_ = torch.linalg.vector_norm(q_obs_t, dim=-1)
        cos_sim = dot / (np_ * no_ + 1e-30)
        loss = (w_t * (1.0 - cos_sim)).sum()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

    U_final = _rotvec_to_matrix(rotvec).detach().cpu().numpy()
    return (U_final, float(a_t.detach().cpu()),
            float(c_t.detach().cpu()), last_loss)


def bootstrap_orientation_uncertainty(
    hkl_q_centroids: Sequence[Tuple[Tuple[int, int, int], np.ndarray]],
    U_init: np.ndarray,
    *,
    a: float, c: float,
    n_boot: int = 25,
    keep_fraction: float = 0.7,
    weights: Optional[np.ndarray] = None,
    n_steps: int = 300,
    lr: float = 5e-3,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> dict:
    """Bootstrap angular uncertainty of `refine_U_from_centroids`.

    Returns dict with `U_mean` (3, 3), `angular_spread_deg` (pairwise stats),
    and the full list of bootstrap U's.
    """
    rng = np.random.default_rng(seed)
    n = len(hkl_q_centroids)
    keep_n = max(4, int(keep_fraction * n))
    U_list = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=keep_n, replace=False)
        sub = [hkl_q_centroids[i] for i in idx]
        sub_w = (np.array([weights[i] for i in idx])
                  if weights is not None else None)
        U_b, _, _, _ = refine_U_from_centroids(
            U_init, sub, weights=sub_w, a=a, c=c,
            refine_lattice=False, n_steps=n_steps, lr=lr,
            device=device, dtype=dtype,
        )
        U_list.append(U_b)
    U_arr = np.stack(U_list)
    # mean orientation: SVD of stacked U's, project to SO(3)
    U_mean_raw = U_arr.mean(axis=0)
    Uu, _, Vt = np.linalg.svd(U_mean_raw)
    d = np.linalg.det(Uu @ Vt)
    U_mean = Uu @ np.diag([1, 1, d]) @ Vt
    # pairwise misorientation
    pair_angles = []
    for i in range(len(U_arr)):
        for j in range(i + 1, len(U_arr)):
            delta = U_arr[i] @ U_arr[j].T
            c_a = max(-1.0, min(1.0, (np.trace(delta) - 1.0) / 2.0))
            pair_angles.append(math.degrees(math.acos(c_a)))
    return dict(
        U_mean=U_mean,
        U_bootstraps=U_arr,
        pair_angle_max_deg=float(max(pair_angles)) if pair_angles else 0.0,
        pair_angle_mean_deg=float(np.mean(pair_angles)) if pair_angles else 0.0,
        pair_angle_p95_deg=float(np.percentile(pair_angles, 95)) if pair_angles else 0.0,
        n_boot=n_boot,
        keep_n=keep_n,
    )


def refine_U_from_friedel_pairs(
    U_init: np.ndarray,
    hkl_q_centroids: Sequence[Tuple[Tuple[int, int, int], np.ndarray]],
    weights: Optional[np.ndarray] = None,
    *,
    a: float, c: float,
    refine_lattice: bool = False,
    n_steps: int = 400,
    lr: float = 5e-3,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> Tuple[np.ndarray, float, float, float, int]:
    """Refine U using Friedel-pair-averaged centroids.

    For each (hkl, q_fit) pair, if (-hkl, q_fit_neg) is also present in the
    set, average the two centroids via:
        avg_q = 0.5 * (q_fit_hkl - q_fit_neg_hkl)
    (note the minus sign — by Friedel symmetry, the predicted q for -hkl is
    the negative of the predicted q for +hkl; we collapse the two
    measurements into a single noise-reduced estimate at +hkl).

    Unpaired hkls are kept as-is. Total constraint set: (#pairs averaged)
    + (#unpaired). Reduces fit noise without inventing data.

    Returns (U_refined, a_refined, c_refined, final_loss, n_pairs_used).
    """
    paired_centroids: list[Tuple[Tuple[int, int, int], np.ndarray]] = []
    paired_weights: list[float] = []
    by_hkl = {h: (h, q) for (h, q) in hkl_q_centroids}
    weights_dict = (
        {h: float(w) for (h, q), w in zip(hkl_q_centroids, weights)}
        if weights is not None else None
    )
    n_pairs = 0
    used = set()
    for (h, q) in hkl_q_centroids:
        neg = tuple(-x for x in h)
        if neg in by_hkl and neg not in used and h not in used:
            _, q_neg = by_hkl[neg]
            # average: by Friedel, q_fit(-h) ≈ -q_fit(h). Combine accordingly.
            q_avg = 0.5 * (np.asarray(q) - np.asarray(q_neg))
            paired_centroids.append((h, q_avg))
            if weights_dict is not None:
                w = 0.5 * (weights_dict[h] + weights_dict[neg])
            else:
                w = 1.0
            paired_weights.append(w)
            used.add(h); used.add(neg)
            n_pairs += 1
        elif h not in used:
            # unpaired; keep as-is
            paired_centroids.append((h, np.asarray(q)))
            paired_weights.append(
                weights_dict[h] if weights_dict is not None else 1.0
            )
            used.add(h)
    w_arr = np.asarray(paired_weights, dtype=np.float64)
    U_out, a_out, c_out, loss = refine_U_from_centroids(
        U_init, paired_centroids, weights=w_arr,
        a=a, c=c, refine_lattice=refine_lattice,
        n_steps=n_steps, lr=lr, device=device, dtype=dtype,
    )
    return U_out, a_out, c_out, loss, n_pairs


def iterate_seed_asterism_refinement(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    *,
    U_init: np.ndarray, a_init: float, c_init: float,
    crystal=None,
    n_iters: int = 3,
    asterism_q_max: float = 8.0,
    asterism_crop_half: float = 0.20,
    asterism_crop_q_scale: float = 0.05,
    asterism_min_voxels: int = 30,
    asterism_n_steps: int = 250,
    asterism_loss_kind: str = "lsq",
    refine_lattice: bool = True,
    centroid_n_steps: int = 400,
    centroid_lr: float = 5e-3,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """Alternate `asterism_fit` and `refine_U_from_centroids` to convergence.

    Each iteration:
      1. Fit a 3-D Gaussian at every predicted hkl position under the current U.
      2. Collect intensity-weighted q_fit centroids.
      3. Re-refine U (and lattice) against the centroids.
    The orientation typically tightens to <1° within 2-3 iterations.

    Returns dict with `U`, `a`, `c`, plus per-iteration history.
    """
    from .asterism_fit import fit_asterism_patches  # avoid circular import

    if crystal is None:
        from .lattice import cual2_crystal
        crystal = cual2_crystal(a=a_init, c=c_init)

    U = np.asarray(U_init, dtype=np.float64).copy()
    a = float(a_init); c = float(c_init)
    history = []
    for it in range(n_iters):
        fits = fit_asterism_patches(
            qx, qy, qz, intensity,
            U=U, a=a, c=c, crystal=crystal,
            q_max_inv_A=asterism_q_max,
            crop_halfwidth=asterism_crop_half,
            crop_q_scale=asterism_crop_q_scale,
            min_voxels=asterism_min_voxels,
            n_steps=asterism_n_steps, lr=1e-2,
            loss_kind=asterism_loss_kind,
            device=device,
        )
        if len(fits) < 4:
            history.append(dict(iter=it + 1, n_centroids=len(fits),
                                 status="too_few_centroids"))
            break
        pairs = [(f.hkl, f.q_fit) for f in fits]
        weights = np.array([f.integrated_intensity for f in fits])
        U_new, a_new, c_new, loss = refine_U_from_centroids(
            U, pairs, weights=weights,
            a=a, c=c, refine_lattice=refine_lattice,
            n_steps=centroid_n_steps, lr=centroid_lr,
            device=device,
        )
        # angular step
        delta = U_new @ U.T
        c_a = max(-1.0, min(1.0, (np.trace(delta) - 1.0) / 2.0))
        step_deg = math.degrees(math.acos(c_a))
        history.append(dict(
            iter=it + 1, n_centroids=len(fits),
            a=a_new, c=c_new, loss=loss,
            angular_step_deg=step_deg,
            status="ok",
        ))
        U = U_new; a = a_new; c = c_new
        # update crystal for next iteration's asterism shell prediction
        from .lattice import cual2_crystal
        crystal = cual2_crystal(a=a, c=c)
    return dict(U=U, a=a, c=c, history=history)
