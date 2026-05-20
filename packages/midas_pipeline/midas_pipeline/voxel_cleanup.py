"""Missing-spot directionality voxel-grid cleanup (PF-HEDM).

Removes / reassigns mis-indexed voxels (lone jutting voxels, corner orphans,
small fragments) using the *directional* missing-spot signal: a voxel assigned
to grain G whose predicted spots are unmatched specifically in the rotation
directions where G's sinogram cell is empty is most likely mis-assigned.

Validated regime (see dev/paper/MISSING_SPOT_DIRECTIONALITY_CLEANUP.md):
small / compact / tightly-supported grains. On large spatially-spread grains
the directional signal vanishes (every voxel's predictions land in occupied
sinogram cells) and this stage is a near no-op — which is the safe behaviour.

Design choices that make it safe (each validated in the prototypes):
  * Position-aware matching against the *observed* spots (real data), so there
    is no grain-shape circularity in deciding "matched".
  * Geometry-miss requires the grain's sinogram CELL to be empty (true
    occupancy, not convex extent), with an occupancy count threshold that acts
    as a built-in leave-one-out (a lone voxel cannot self-occupy its cell).
  * A spatial connectivity gate: only act on voxels that are also isolated /
    weakly-connected, so legitimate edge/tip voxels of irregular grains (which
    can be sole extremal projectors) are never removed.
  * Iteration to a fixed point: rebuild occupancy from the shrinking map.

The forward model (orientation -> predicted (omega, eta, ring) + observability)
is injected as ``predict_fn`` so the core is testable without the indexer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

DEG = np.pi / 180.0

# predict_fn(grain_id, voxel_indices) -> (omega_deg, eta_deg, ring, valid)
#   each array shaped (n_vox, n_pred); `valid` is the observability mask.
PredictFn = Callable[[int, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


@dataclass
class CleanupResult:
    new_grain: np.ndarray            # (Nv,) int64 — cleaned voxel->grain map
    flagged: np.ndarray              # (Nv,) bool   — voxels acted on
    directional: np.ndarray          # (Nv,) float  — final geometry-miss score
    scalar: np.ndarray               # (Nv,) float  — position-aware incompleteness
    n_passes: int
    per_pass_flagged: List[int] = field(default_factory=list)


def _occupancy(vx, vy, vgr, grains, omegas_rad, s0, sbin, ns):
    """Per-grain sinogram occupancy count over (omega-bin, s-bin)."""
    sw, cw = np.sin(omegas_rad), np.cos(omegas_rad)
    occ: Dict[int, np.ndarray] = {}
    for g in grains:
        vid = np.where(vgr == g)[0]
        cnt = np.zeros((omegas_rad.size, ns), np.int32)
        if vid.size:
            pr = vx[vid][:, None] * sw[None] + vy[vid][:, None] * cw[None]
            sidx = np.clip(((pr - s0) / sbin).round().astype(int), 0, ns - 1)
            for o in range(omegas_rad.size):
                np.add.at(cnt[o], sidx[:, o], 1)
        occ[g] = cnt
    return occ


def _neighbour_count(vgr, n):
    """4-connected same-grain neighbour count per voxel (grid is n x n)."""
    g2 = vgr.reshape(n, n)
    cnt = np.zeros((n, n), int)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cnt += (np.roll(np.roll(g2, di, 0), dj, 1) == g2) & (g2 >= 0)
    return cnt.ravel()


def _majority_neighbour(vgr, n, v):
    """Most common valid grain among 4-neighbours of voxel v (or -1)."""
    i, j = divmod(v, n)
    vals = []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii, jj = i + di, j + dj
        if 0 <= ii < n and 0 <= jj < n:
            g = vgr[ii * n + jj]
            if g >= 0 and g != vgr[v]:
                vals.append(int(g))
    if not vals:
        return -1
    u, c = np.unique(vals, return_counts=True)
    return int(u[c.argmax()])


def _score_grain(predict_fn, g, vox_ids, vx, vy, obs_by_ring,
                 occ, omegas_rad, s0, sbin, ns,
                 margin_ome, margin_eta, scan_tol, occ_min_count):
    """Return (directional, scalar) score arrays for vox_ids (assigned to g)."""
    om, et, rg, valid = predict_fn(g, vox_ids)
    nvox = len(vox_ids)
    directional = np.zeros(nvox)
    scalar = np.zeros(nvox)
    cnt = occ[g]
    for k in range(nvox):
        v = vox_ids[k]
        obsv = valid[k].astype(bool)
        nobs = int(obsv.sum())
        if nobs == 0:
            continue
        sv = vx[v] * np.sin(om[k] * DEG) + vy[v] * np.cos(om[k] * DEG)
        matched = np.zeros(obsv.shape, bool)
        for t in np.where(obsv)[0]:
            r = int(rg[k, t])
            ent = obs_by_ring.get(r)
            if ent is None:
                continue
            oo, oe, osp = ent
            if ((np.abs(oo - om[k, t]) < margin_ome)
                    & (np.abs(oe - et[k, t]) < margin_eta)
                    & (np.abs(osp - sv[t]) < scan_tol)).any():
                matched[t] = True
        unmatched = obsv & ~matched
        scalar[k] = unmatched.sum() / nobs
        # directional: unmatched spot landing in an EMPTY grain sinogram cell
        ob = (np.round(om[k]) % 360).astype(int) % omegas_rad.size
        sb = np.clip(((sv - s0) / sbin).round().astype(int), 1, ns - 2)
        cell = cnt[ob, sb] + cnt[ob, sb - 1] + cnt[ob, sb + 1]
        empty = cell < occ_min_count
        directional[k] = (unmatched & empty).sum() / nobs
    return directional, scalar


def cleanup_voxel_grid(
    *,
    predict_fn: PredictFn,
    vx: np.ndarray,
    vy: np.ndarray,
    grain: np.ndarray,
    grains: List[int],
    obs_by_ring: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pitch: float,
    margin_ome: float,
    margin_eta: float,
    scan_tol: float,
    score_threshold: float = 0.30,
    max_same_neighbours: int = 1,
    max_iters: int = 8,
    occ_min_count: int = 2,
    action: str = "reassign",     # "reassign" | "remove"
    omega_step_deg: float = 1.0,
) -> CleanupResult:
    """Iterative, connectivity-gated, directional voxel cleanup.

    Each pass: rebuild sinogram occupancy from the current (shrinking) map,
    score every still-active voxel, and act on those with
    ``directional > score_threshold`` AND ``<= max_same_neighbours`` same-grain
    neighbours. ``action="reassign"`` moves a flagged voxel to its majority
    neighbouring grain (falling back to -1); ``"remove"`` sets it to -1.
    """
    n = int(round(np.sqrt(grain.size)))
    if n * n != grain.size:
        raise ValueError(f"grain map size {grain.size} is not a square grid")
    omegas_rad = np.deg2rad(np.arange(0.0, 360.0, omega_step_deg))
    s0 = float(min(vx.min(), vy.min())) - pitch
    sbin = pitch
    ns = int((max(vx.max(), vy.max()) - s0) / sbin) + 3

    cur = grain.copy().astype(np.int64)
    flagged = np.zeros(grain.size, bool)
    directional = np.zeros(grain.size)
    scalar = np.zeros(grain.size)
    per_pass: List[int] = []

    for it in range(max_iters):
        occ = _occupancy(vx, vy, cur, grains, omegas_rad, s0, sbin, ns)
        nb = _neighbour_count(cur, n)
        directional[:] = 0.0
        scalar[:] = 0.0
        for g in grains:
            vox = np.where(cur == g)[0]
            if vox.size == 0:
                continue
            d, s = _score_grain(
                predict_fn, g, vox, vx, vy, obs_by_ring, occ,
                omegas_rad, s0, sbin, ns,
                margin_ome, margin_eta, scan_tol, occ_min_count,
            )
            directional[vox] = d
            scalar[vox] = s
        new_flag = (directional > score_threshold) & (nb <= max_same_neighbours) \
            & (cur >= 0) & ~flagged
        idx = np.where(new_flag)[0]
        per_pass.append(int(idx.size))
        if idx.size == 0:
            break
        for v in idx:
            if action == "reassign":
                cur[v] = _majority_neighbour(cur, n, int(v))
            else:
                cur[v] = -1
            flagged[v] = True

    return CleanupResult(
        new_grain=cur, flagged=flagged, directional=directional,
        scalar=scalar, n_passes=it + 1, per_pass_flagged=per_pass,
    )
