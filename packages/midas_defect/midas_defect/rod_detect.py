"""P1 — 3D rod (q-space line) detection.

Detect 1-D streaks of intensity in a sparse reciprocal-space voxel cloud.
Each detected rod is the q-space signature of a planar defect family in
the crystal (stacking faults, twin walls, dense dislocation walls); the
rod direction is the *defect-plane normal*.

Pipeline
--------
1. **Bright cores** — `_discrete_extract_bright_cores` (greedy, no gradient).
2. **Pair-seeded RANSAC** — for each pair of bright cores, propose a line
   through them; score by counting bright voxels within a tube of radius
   `r_tube` (scoring tensor is torch + diff in tube-radius).
3. **NMS** — group near-duplicate lines.
4. **Refinement** — torch+Adam on a soft-tube score `Σ I · σ(...)`; both
   direction (unit-vector via normalize) and pivot are diff.
5. **Reporting** — per rod, compute integrated intensity, line-projected
   profile, list of CuAl₂ shells crossed, and (if `U` is given) the
   crystal-frame `defect_normal_hkl` interpretation.

Differentiability
-----------------
* `tube_score(direction, pivot, q, intensity, r_tube)` is a fully diff
  torch routine; we *refine* the line geometry by maximising it.
* `pair_propose` / `nms` / `extract_bright_cores` are inherently discrete
  and live in `_discrete_*` helpers; the discrete output is just an
  *initialization* for the diff refinement, which polishes it.
* Reporting (shell-crossing, hkl) is post-hoc and discrete.

MIDAS reuses
------------
* `midas_transforms.device` — canonical device/dtype resolution.
* `midas_stress.orientation` (when `U` is provided) — for lab/sample → crystal
  frame conversion of rod directions.
* `lattice.tetragonal_shells` — shell-crossing lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch

from midas_transforms.device import resolve_device, resolve_dtype

from .lattice import Shell, tetragonal_shells


__all__ = [
    "QRod",
    "find_rods",
    "tube_score",
    "soft_tube_score",
    "refine_rod",
    "find_rods_iterative_residual",
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class QRod:
    """One detected rod in q-space."""
    direction: np.ndarray            # unit vector, sample/lab frame (3,)
    pivot:     np.ndarray            # one point on the line, sample/lab frame (3,)
    length:    float                 # extent along direction, |t_max - t_min|, 1/Å
    t_min:     float                 # smallest inlier projection (signed)
    t_max:     float                 # largest inlier projection (signed)
    n_inliers: int                   # voxels within tube
    integrated_intensity: float      # Σ I over inliers
    score:     float                 # tube-score (= integrated_intensity, by default)
    shells_crossed: List[Tuple[float, Tuple[int, int, int]]] = field(default_factory=list)
                                       # list of (|q| crossed, representative hkl)
    defect_normal_hkl: Optional[Tuple[int, int, int]] = None
                                       # nearest low-index crystal axis to U.T @ direction
    profile_t: Optional[np.ndarray] = None  # 1-D projected positions
    profile_I: Optional[np.ndarray] = None  # 1-D projected intensities

    def as_dict(self) -> dict:
        return dict(
            direction=self.direction.tolist(),
            pivot=self.pivot.tolist(),
            length=float(self.length),
            t_min=float(self.t_min),
            t_max=float(self.t_max),
            n_inliers=int(self.n_inliers),
            integrated_intensity=float(self.integrated_intensity),
            score=float(self.score),
            shells_crossed=[(float(q), [int(x) for x in h]) for q, h in self.shells_crossed],
            defect_normal_hkl=(
                [int(x) for x in self.defect_normal_hkl]
                if self.defect_normal_hkl is not None else None
            ),
        )


# ---------------------------------------------------------------------------
# Differentiable scoring primitives
# ---------------------------------------------------------------------------

def _perp_dist(q: torch.Tensor, pivot: torch.Tensor, direction: torch.Tensor
               ) -> torch.Tensor:
    """Perpendicular distance from points `q (N, 3)` to a line through `pivot`
    with unit `direction (3,)`. Returns (N,)."""
    d_unit = direction / torch.linalg.vector_norm(direction)
    rel = q - pivot
    along = (rel * d_unit).sum(dim=-1, keepdim=True)
    perp = rel - along * d_unit
    return torch.linalg.vector_norm(perp, dim=-1)


def tube_score(
    q: torch.Tensor, intensity: torch.Tensor,
    pivot: torch.Tensor, direction: torch.Tensor, *, r_tube: float,
) -> torch.Tensor:
    """Hard tube score: sum of intensities of voxels within radius `r_tube`.

    Non-differentiable in `r_tube` (hard threshold). Diff in (pivot, direction)
    only as long as the inlier set does not change — used for evaluation
    and reporting, not for refinement. Use `soft_tube_score` to optimize.
    """
    perp = _perp_dist(q, pivot, direction)
    inside = (perp <= r_tube).to(intensity.dtype)
    return (intensity * inside).sum()


def soft_tube_score(
    q: torch.Tensor, intensity: torch.Tensor,
    pivot: torch.Tensor, direction: torch.Tensor,
    *, r_tube: float, sharpness: float = 5.0,
) -> torch.Tensor:
    """Differentiable analogue of `tube_score`.

    Replaces the hard tube mask with a sigmoid window:
        weight(q) = σ( sharpness · (r_tube - d_perp) / r_tube )
    so the inlier weight smoothly transitions from 1 (inside) to 0 (outside)
    over a band of width ~ r_tube / sharpness.

    Differentiable in `(pivot, direction)`; bias toward concentrating
    high-intensity voxels in the tube → suitable for max-via-Adam refinement.
    """
    perp = _perp_dist(q, pivot, direction)
    z = sharpness * (r_tube - perp) / r_tube
    weight = torch.sigmoid(z)
    return (intensity * weight).sum()


def refine_rod(
    q: torch.Tensor, intensity: torch.Tensor,
    pivot_init: torch.Tensor, direction_init: torch.Tensor,
    *, r_tube: float, sharpness: float = 5.0,
    n_steps: int = 80, lr: float = 5e-3,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Refine `(pivot, direction)` by maximizing `soft_tube_score` via Adam.

    Returns the refined `(pivot, direction_unit, final_score)`.
    """
    pivot = pivot_init.detach().clone().requires_grad_(True)
    direction = direction_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([pivot, direction], lr=lr)
    last_score = -math.inf
    for _ in range(n_steps):
        opt.zero_grad()
        S = soft_tube_score(q, intensity, pivot, direction,
                            r_tube=r_tube, sharpness=sharpness)
        loss = -S
        loss.backward()
        opt.step()
        last_score = float(S.detach().cpu())
    direction_unit = direction.detach() / torch.linalg.vector_norm(direction.detach())
    return pivot.detach(), direction_unit, last_score


# ---------------------------------------------------------------------------
# Discrete helpers
# ---------------------------------------------------------------------------

def _discrete_extract_bright_cores(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    *, n_cores: int = 200, min_separation: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy bright-core extraction; same algorithm as in `seed_index`."""
    order = np.argsort(intensity)[::-1]
    chosen: list[int] = []
    pts: list[np.ndarray] = []
    sep2 = float(min_separation) ** 2
    for k in order:
        p = np.array([qx[k], qy[k], qz[k]], dtype=np.float64)
        ok = True
        for q in pts:
            if np.sum((p - q) ** 2) < sep2:
                ok = False
                break
        if ok:
            chosen.append(int(k))
            pts.append(p)
            if len(pts) >= n_cores:
                break
    centroids = np.stack(pts) if pts else np.zeros((0, 3))
    ints = np.array([intensity[k] for k in chosen], dtype=np.float64)
    return centroids, ints


def _discrete_nms_rods(
    rod_specs: list[tuple[float, np.ndarray, np.ndarray]],
    *, direction_tol_deg: float = 5.0, pivot_perp_tol: float = 0.3,
) -> list[int]:
    """NMS over (score, pivot, direction) tuples; returns indices to keep.

    Two rods are considered duplicates if their directions agree within
    `direction_tol_deg`. We optionally also require the rods to be near
    each other in space (perp distance from `piv_i` to line(piv_j, dir_j)
    less than `pivot_perp_tol`) — but for the typical "rod in q-space"
    use-case, two rods with the same direction are physically the same
    defect family regardless of where along the rod we sampled, so the
    default `pivot_perp_tol = 0.3` is generous and most often the
    direction check alone decides.
    """
    order = sorted(range(len(rod_specs)), key=lambda i: -rod_specs[i][0])
    cos_tol = math.cos(math.radians(direction_tol_deg))
    keep: list[int] = []
    for i in order:
        _S_i, piv_i, dir_i = rod_specs[i]
        suppressed = False
        for j in keep:
            _, piv_j, dir_j = rod_specs[j]
            cos_ij = abs(float(dir_i @ dir_j))   # sign-insensitive
            if cos_ij < cos_tol:
                continue
            rel = piv_i - piv_j
            along = float(rel @ dir_j)
            perp = rel - along * dir_j
            if float(np.linalg.norm(perp)) < pivot_perp_tol:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return keep


def _hkl_shells_crossed(
    pivot: np.ndarray, direction: np.ndarray, t_min: float, t_max: float,
    shells: Sequence[Shell],
) -> list[tuple[float, Tuple[int, int, int]]]:
    """Which CuAl₂ shells does the line-segment cross?

    Line: q(t) = pivot + t * direction, |direction|=1. Shell |q|=R intersects
    the line at t roots of |pivot|² + 2t (pivot·direction) + t² = R².
    """
    out = []
    pdot = float(pivot @ direction)
    p2 = float(pivot @ pivot)
    for s in shells:
        R2 = s.q_inv_A * s.q_inv_A
        disc = pdot * pdot - (p2 - R2)
        if disc < 0:
            continue
        sd = math.sqrt(disc)
        for t in (-pdot - sd, -pdot + sd):
            if t_min - 1e-6 <= t <= t_max + 1e-6:
                out.append((float(s.q_inv_A), s.hkls[0]))
                break    # one root per shell is enough for reporting
    return out


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def find_rods(
    qx: "np.ndarray | torch.Tensor",
    qy: "np.ndarray | torch.Tensor",
    qz: "np.ndarray | torch.Tensor",
    intensity: "np.ndarray | torch.Tensor",
    *,
    # discrete-pass knobs
    n_cores: int = 150,
    core_min_separation: float = 0.05,
    pair_min_dist: float = 0.3,        # skip pairs inside one asterism patch
    pair_max_dist: float = 20.0,       # skip pairs too far apart
    # scoring knobs
    r_tube: float = 0.03,              # 1/Å
    sharpness: float = 5.0,
    N_min_inliers: int = 30,
    L_min: float = 0.3,                # 1/Å minimum rod length
    L_max: Optional[float] = None,     # 1/Å maximum rod length (None = unlimited)
    cloud_min_intensity: Optional[float] = None,  # pre-filter haze
    max_voxels_for_scoring: int = 100_000,
    # continuity check (rejects "rods" that are really discrete Bragg-cluster lines)
    continuity_max_gap_frac: float = 0.15,
    uniform_n_bins: int = 4,
    uniform_min_per_bin: int = 2,
    # NMS knobs
    nms_direction_tol_deg: float = 5.0,
    nms_pivot_perp_tol: float = 0.3,
    # refinement
    refine_steps: int = 80,
    refine_lr: float = 5e-3,
    # reporting knobs
    crystal=None,                       # midas_hkls.Crystal for shell crossings
    U: Optional[np.ndarray] = None,     # 3x3 orientation matrix; if given,
                                        # rods get a `defect_normal_hkl` annotation
    max_hkl_for_normal: int = 4,        # consider hkl with max(|h|,|k|,|l|) ≤ this
    # device / dtype
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> List[QRod]:
    """End-to-end P1 pipeline. Returns a list of `QRod` ordered by score desc."""
    device_ = resolve_device(device)
    dtype_  = resolve_dtype(device_, dtype)

    qx_np = np.asarray(qx, dtype=np.float64)
    qy_np = np.asarray(qy, dtype=np.float64)
    qz_np = np.asarray(qz, dtype=np.float64)
    I_np  = np.asarray(intensity, dtype=np.float64)

    if cloud_min_intensity is not None:
        keep_mask = I_np >= cloud_min_intensity
        qx_np = qx_np[keep_mask]
        qy_np = qy_np[keep_mask]
        qz_np = qz_np[keep_mask]
        I_np  = I_np[keep_mask]
        if len(qx_np) == 0:
            return []

    # 1) bright cores
    cores, _core_ints = _discrete_extract_bright_cores(
        qx_np, qy_np, qz_np, I_np,
        n_cores=n_cores, min_separation=core_min_separation,
    )
    n_cores_actual = len(cores)
    if n_cores_actual < 2:
        return []

    # 2) prepare voxel tensors for scoring (subsample for speed)
    n_total = len(qx_np)
    if n_total > max_voxels_for_scoring:
        order = np.argsort(I_np)[::-1][:max_voxels_for_scoring]
        qx_s, qy_s, qz_s, I_s = qx_np[order], qy_np[order], qz_np[order], I_np[order]
    else:
        qx_s, qy_s, qz_s, I_s = qx_np, qy_np, qz_np, I_np
    q_t = torch.tensor(np.stack([qx_s, qy_s, qz_s], axis=1),
                       dtype=dtype_, device=device_)
    I_t = torch.tensor(I_s, dtype=dtype_, device=device_)

    # 3) pair-seeded RANSAC
    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    # each entry: (hard_score, pivot (3,), direction_unit (3,), inlier_indices)
    seen_pair_signature: set[tuple[float, float, float, float]] = set()
    for i in range(n_cores_actual):
        for j in range(i + 1, n_cores_actual):
            v = cores[j] - cores[i]
            d2 = float(v @ v)
            if d2 < pair_min_dist * pair_min_dist:
                continue
            if d2 > pair_max_dist * pair_max_dist:
                continue
            dist = math.sqrt(d2)
            direction = v / dist
            pivot = (cores[i] + cores[j]) / 2.0
            # dedup signatures so two nearly-identical lines don't compete
            sig = (round(direction[0], 2), round(direction[1], 2),
                   round(direction[2], 2), round(np.linalg.norm(pivot), 2))
            if sig in seen_pair_signature:
                continue
            seen_pair_signature.add(sig)
            piv_t = torch.tensor(pivot, dtype=dtype_, device=device_)
            dir_t = torch.tensor(direction, dtype=dtype_, device=device_)
            score = float(tube_score(q_t, I_t, piv_t, dir_t, r_tube=r_tube).cpu())
            # inlier indices for the hard tube on the SCORING subsample
            perp = _perp_dist(q_t, piv_t, dir_t).cpu().numpy()
            inliers_local = np.where(perp <= r_tube)[0]
            if len(inliers_local) < N_min_inliers:
                continue
            # length check: span of inlier projections
            rel = (np.stack([qx_s, qy_s, qz_s], axis=1)[inliers_local]
                   - pivot[None, :])
            along = rel @ direction
            t_min, t_max = float(along.min()), float(along.max())
            if (t_max - t_min) < L_min:
                continue
            if L_max is not None and (t_max - t_min) > L_max:
                continue
            # continuity check: reject candidates with discrete-cluster patterns
            if not _inliers_continuous(
                along, t_min, t_max,
                max_gap_frac=continuity_max_gap_frac,
                n_bins=uniform_n_bins, min_per_bin=uniform_min_per_bin,
            ):
                continue
            candidates.append((score, pivot, direction, inliers_local))

    if not candidates:
        return []

    # 4) NMS
    keep_idx = _discrete_nms_rods(
        [(s, p, d) for (s, p, d, _) in candidates],
        direction_tol_deg=nms_direction_tol_deg,
        pivot_perp_tol=nms_pivot_perp_tol,
    )
    candidates = [candidates[i] for i in keep_idx]

    # 5) per-rod refinement + reporting
    shells: list[Shell] = []
    if crystal is not None:
        qmax = float(np.linalg.norm(np.stack([qx_np, qy_np, qz_np], axis=1), axis=1).max())
        shells = tetragonal_shells(crystal, q_max_inv_A=qmax * 1.05)

    rods: list[QRod] = []
    for (score0, pivot0, direction0, inliers0) in candidates:
        piv_t = torch.tensor(pivot0, dtype=dtype_, device=device_)
        dir_t = torch.tensor(direction0, dtype=dtype_, device=device_)
        piv_r, dir_r, _ = refine_rod(
            q_t, I_t, piv_t, dir_t,
            r_tube=r_tube, sharpness=sharpness,
            n_steps=refine_steps, lr=refine_lr,
        )
        pivot_np = piv_r.cpu().numpy()
        direction_np = dir_r.cpu().numpy()
        # recompute inliers + integrated I + length on the FULL voxel set
        rel = (np.stack([qx_np, qy_np, qz_np], axis=1) - pivot_np[None, :])
        along = rel @ direction_np
        perp_vec = rel - along[:, None] * direction_np[None, :]
        perp_full = np.linalg.norm(perp_vec, axis=1)
        inlier_full = np.where(perp_full <= r_tube)[0]
        if len(inlier_full) < N_min_inliers:
            continue
        sum_I = float(I_np[inlier_full].sum())
        t_in = along[inlier_full]
        t_min, t_max = float(t_in.min()), float(t_in.max())
        length = t_max - t_min
        if length < L_min:
            continue
        if L_max is not None and length > L_max:
            continue
        if not _inliers_continuous(
            t_in, t_min, t_max,
            max_gap_frac=continuity_max_gap_frac,
            n_bins=uniform_n_bins, min_per_bin=uniform_min_per_bin,
        ):
            continue

        # shell crossings
        shells_crossed = _hkl_shells_crossed(pivot_np, direction_np,
                                             t_min, t_max, shells) if shells else []

        # defect plane normal in crystal frame
        defect_hkl: Optional[Tuple[int, int, int]] = None
        if U is not None:
            defect_hkl = _nearest_low_hkl_to_direction(
                U.T @ direction_np, max_hkl=max_hkl_for_normal,
            )

        # profile: 1-D projection (sorted by t)
        sort_idx = np.argsort(t_in)
        profile_t = t_in[sort_idx]
        profile_I = I_np[inlier_full][sort_idx]

        rods.append(QRod(
            direction=direction_np,
            pivot=pivot_np,
            length=length,
            t_min=t_min,
            t_max=t_max,
            n_inliers=int(len(inlier_full)),
            integrated_intensity=sum_I,
            score=float(sum_I),
            shells_crossed=shells_crossed,
            defect_normal_hkl=defect_hkl,
            profile_t=profile_t.astype(np.float32),
            profile_I=profile_I.astype(np.float32),
        ))

    rods.sort(key=lambda r: -r.score)
    # second NMS pass after refinement, in case refinement merged previously-
    # distinct candidates onto the same line.
    if len(rods) > 1:
        keep2 = _discrete_nms_rods(
            [(r.score, r.pivot, r.direction) for r in rods],
            direction_tol_deg=nms_direction_tol_deg,
            pivot_perp_tol=nms_pivot_perp_tol,
        )
        rods = [rods[i] for i in keep2]
    return rods


def find_rods_iterative_residual(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    *,
    n_iter: int = 5,
    suppress_perp: float = 0.06,
    suppress_along_pad: float = 0.5,
    suppress_floor: float = 0.05,
    **find_rods_kwargs,
) -> List[List[QRod]]:
    """Iteratively find rods by subtracting (zeroing) the previously-detected ones.

    Strategy:
      1. Run `find_rods` on the current cloud, keep the top-1 rod (after NMS).
      2. Zero out the cloud's intensity inside that rod's tube (perp < suppress_perp,
         |along - rod_axis| < rod_length/2 + suppress_along_pad).
         A small `suppress_floor` is left so the cloud doesn't collapse.
      3. Repeat up to `n_iter` times or until no rod is detected.

    Returns a list-of-lists: outer index = iteration, inner = rods returned that round.
    Outer length is len <= n_iter; iteration stops early if no rod found.

    `**find_rods_kwargs` forwards to `find_rods` (use it to set NMS, L_min, etc.).
    """
    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)
    qz = np.asarray(qz, dtype=np.float64)
    I  = np.asarray(intensity, dtype=np.float64).copy()
    q_all = np.stack([qx, qy, qz], axis=1)

    rounds: list[list[QRod]] = []
    for it in range(n_iter):
        rods = find_rods(qx, qy, qz, I, **find_rods_kwargs)
        if not rods:
            break
        rounds.append(rods)
        top = rods[0]
        # Suppress the dominant rod's tube
        rel = q_all - top.pivot[None, :]
        along = rel @ top.direction
        perp = np.linalg.norm(rel - along[:, None] * top.direction[None, :], axis=1)
        in_tube = (
            (perp < suppress_perp)
            & (np.abs(along) < top.length / 2.0 + suppress_along_pad)
        )
        # multiplicative suppression rather than hard zero — keeps a residual
        # so subsequent rods that cross the suppressed region are still visible
        I[in_tube] *= suppress_floor
    return rounds


def _nearest_low_hkl_to_direction(d_cry: np.ndarray, *, max_hkl: int = 4
                                  ) -> Tuple[int, int, int]:
    """Closest low-index (h,k,l) direction in the crystal frame.

    Sign-insensitive (rod direction `d` and `-d` describe the same line); we
    return the positive-first form: first non-zero entry positive.
    """
    d = d_cry / max(np.linalg.norm(d_cry), 1e-30)
    best = (1, 0, 0)
    best_cos = -1.0
    # Only enumerate non-negative-first hkls to avoid sign ambiguity.
    for h in range(0, max_hkl + 1):
        for k in range(-max_hkl, max_hkl + 1):
            for l in range(-max_hkl, max_hkl + 1):
                if h == 0 and k < 0:
                    continue
                if h == 0 and k == 0 and l <= 0:
                    continue
                v = np.array([h, k, l], dtype=np.float64)
                n = np.linalg.norm(v)
                cosv = abs(float(d @ v) / n)
                if cosv > best_cos:
                    best_cos = cosv
                    best = (h, k, l)
    # reduce to lowest-form (e.g., (2,0,0) → (1,0,0))
    g = math.gcd(math.gcd(abs(best[0]), abs(best[1])), abs(best[2]))
    if g > 1:
        best = (best[0] // g, best[1] // g, best[2] // g)
    return best


def _inliers_continuous(t_values: np.ndarray, t_min: float, t_max: float,
                        *, max_gap_frac: float = 0.15,
                        n_bins: int = 4, min_per_bin: int = 2) -> bool:
    """Reject rods whose inliers are not continuously distributed along the line.

    A true rod has small gaps between successive inlier projections (the rod
    is filled with intensity along its whole length). A "phantom rod" through
    a few distant Bragg clusters has big inter-cluster gaps. Two checks:

      * the largest gap between consecutive sorted `t_values` is less than
        `max_gap_frac` of the total rod length, AND
      * each of `n_bins` equal-width bins contains at least `min_per_bin`
        inliers (catches the rare case where one short rod hides inside one bin).
    """
    if t_max <= t_min or len(t_values) < 4:
        return False
    length = t_max - t_min
    sorted_t = np.sort(t_values)
    if float(np.diff(sorted_t).max()) > max_gap_frac * length:
        return False
    edges = np.linspace(t_min, t_max, n_bins + 1)
    counts, _ = np.histogram(t_values, bins=edges)
    return bool((counts >= min_per_bin).all())
