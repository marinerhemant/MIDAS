"""Stage 6: grain-size recomputation after physics-bounded split.

The legacy ProcessGrains computes ``GrainRadius`` as
``mean(per-spot radii)`` where per-spot radius is the (I_obs / K_ring)^(1/3)
back-out. The Stage 6 v4 rule:

* **For grains emitted directly from a clean Pass-1 cluster** (no split),
  the legacy formula is consistent with the physics — emit it unchanged.

* **For grains emitted from a SPLIT cluster**, the per-spot radii were
  computed before we knew which split product each spot belonged to.
  Each shared spot contributes to all K split products. We re-run the
  joint NNLS volume correction (Stage 6 of v3) restricted to the
  candidates within the original Pass-1 cluster, then attribute the
  shared-spot intensity correctly.

For ``twin family`` parent entries (Stage 5 post-hoc labels), the
parent volume is the SUM of the child grain volumes — twin partners
are physically distinct grains that share reflections, so summing
their individually-attributed volumes is exact.

This module is a thin wrapper around the existing
:func:`compute.volume_nnls.compute_nnls_volumes`, with the bookkeeping
that maps "Pass-1 cluster" → "set of grains the split produced" → "joint
spot lists" → "NNLS-attributed volumes".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    from .volume_nnls import compute_nnls_volumes, NnlsVolumeResult, physical_ring_K
except ImportError:
    compute_nnls_volumes = None
    NnlsVolumeResult = None
    physical_ring_K = None


@dataclass
class GrainSizeRecomputeResult:
    """Per-final-grain volume + radius after split-aware NNLS attribution."""

    grain_id:           np.ndarray   # (N_g,) int64
    volume_naive_um3:   np.ndarray   # (N_g,) float64 — mean-of-per-spot legacy
    volume_nnls_um3:    np.ndarray   # (N_g,) float64 — split-aware NNLS
    radius_naive_um:    np.ndarray   # (N_g,) float64
    radius_nnls_um:     np.ndarray   # (N_g,) float64
    n_shared_spots:     np.ndarray   # (N_g,) int32
    deflation_factor:   np.ndarray   # (N_g,) float64 — V_nnls / V_naive
    # Per-grain σ on R_nnls from the linearised NNLS covariance. NaN
    # for grains in clean clusters that didn't go through NNLS (just
    # inherits the legacy R_naive).
    sigma_R_nnls_um:    np.ndarray | None = None


def recompute_grain_sizes(
    *,
    final_grain_id_per_candidate: np.ndarray,  # (N_cand,) int64
    pass1_cluster_id:             np.ndarray,  # (N_cand,) int64
    spot_sets_per_candidate:      List[set],   # length N_cand
    per_spot_intensity:           Dict[int, float],  # SpotID → integrated intensity
    per_spot_ring:                Dict[int, int],    # SpotID → ring
    ring_K:                       Dict[int, float],  # ring → kinematic factor
    rep_radius_naive_um_per_grain: np.ndarray,  # (N_g,) — legacy GrainRadius
) -> GrainSizeRecomputeResult:
    """Re-attribute shared-spot intensities to split-product grains via
    sparse NNLS within each Pass-1 cluster, then return per-grain volumes.

    For grains in clusters that did NOT split (one final grain per Pass-1
    cluster), the NNLS solution is trivially the naive-attribution result.
    For split clusters, this redistributes intensity according to the
    NNLS objective:

        min ‖I_obs(s) / K(ring(s))  −  Σ_g M[s, g] · α_g‖²,  α_g ≥ 0

    where M[s, g] = 1 iff grain g claims spot s. Each grain's volume
    α_g is then converted to radius via R = (3·α/4π)^(1/3) after a
    global volume-conserving rescale.

    For grains in **clean** clusters, the legacy ``rep_radius_naive`` is
    returned as both V_naive and V_nnls (so coverage downstream is uniform).
    """
    n_g = int(rep_radius_naive_um_per_grain.shape[0])

    radius_naive = rep_radius_naive_um_per_grain.copy()
    volume_naive = (4.0 / 3.0) * np.pi * radius_naive ** 3
    radius_nnls  = radius_naive.copy()
    volume_nnls  = volume_naive.copy()
    n_shared     = np.zeros(n_g, dtype=np.int32)
    sigma_R      = np.full(n_g, np.nan, dtype=np.float64)

    # Group candidates by Pass-1 cluster
    p1_to_cands: Dict[int, List[int]] = {}
    for c, p in enumerate(pass1_cluster_id):
        p = int(p)
        if final_grain_id_per_candidate[c] < 0:
            continue
        p1_to_cands.setdefault(p, []).append(c)

    for p1, cands in p1_to_cands.items():
        # Which final grains does this Pass-1 cluster produce?
        grains_in_p1 = sorted(set(int(final_grain_id_per_candidate[c]) for c in cands))
        if len(grains_in_p1) <= 1:
            # No split; legacy volume is correct (NNLS = naive trivially)
            continue

        # Build the local NNLS problem
        # All spots claimed by any candidate in this Pass-1 cluster
        all_spots = set()
        spots_per_grain: Dict[int, set] = {g: set() for g in grains_in_p1}
        for c in cands:
            g = int(final_grain_id_per_candidate[c])
            all_spots |= spot_sets_per_candidate[c]
            spots_per_grain[g] |= spot_sets_per_candidate[c]

        # Build (I/K) vector + grain-membership matrix
        spot_list = sorted(all_spots)
        spot_index = {s: i for i, s in enumerate(spot_list)}
        n_s = len(spot_list)
        n_local_g = len(grains_in_p1)
        y = np.zeros(n_s, dtype=np.float64)
        for i, s in enumerate(spot_list):
            ring = per_spot_ring.get(int(s))
            if ring is None: continue
            K = ring_K.get(int(ring))
            if K is None or K <= 0: continue
            I = per_spot_intensity.get(int(s), 0.0)
            y[i] = I / K

        # M is sparse (n_s × n_local_g), mostly 0/1
        M = np.zeros((n_s, n_local_g), dtype=np.float64)
        for j, g in enumerate(grains_in_p1):
            for s in spots_per_grain[g]:
                if s in spot_index:
                    M[spot_index[s], j] = 1.0

        # Solve NNLS via scipy.optimize.lsq_linear (active-set is fine for this scale)
        from scipy.optimize import lsq_linear
        try:
            sol = lsq_linear(M, y, bounds=(0, np.inf), method="trf",
                             lsq_solver="lsmr", max_iter=200)
            alpha = sol.x
        except Exception:
            continue

        # Linearised covariance: per-grain σ_α from residual + (M^T M)^-1.
        # Active-set only (V > 0). Boundary grains report NaN.
        residual = M @ alpha - y
        dof = max(int((alpha > 1e-12).sum()), 1)
        sigma_r2 = float((residual ** 2).sum() / max(n_s - dof, 1))
        active = alpha > 1e-12
        sigma_alpha = np.full(n_local_g, np.nan)
        if active.any() and sigma_r2 > 0:
            M_a = M[:, active]
            MTM = M_a.T @ M_a + 1e-9 * np.eye(int(active.sum()))
            try:
                C = np.linalg.inv(MTM) * sigma_r2
                sigma_alpha[active] = np.sqrt(np.diag(C))
            except np.linalg.LinAlgError:
                pass

        # Rescale so the total volume is conserved vs naive
        total_naive = float(sum(volume_naive[g] for g in grains_in_p1))
        total_alpha = float(alpha.sum())
        if total_alpha <= 0:
            continue
        scale = total_naive / total_alpha

        for j, g in enumerate(grains_in_p1):
            v = float(alpha[j]) * scale
            volume_nnls[g] = v
            r = (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0) if v > 0 else 0.0
            radius_nnls[g] = r
            # Convert σ_α → σ_R via dR/dV = R/(3V): σ_R = σ_α · scale · R/(3V)
            if active[j] and v > 0:
                sigma_R[g] = float(sigma_alpha[j]) * scale * r / (3.0 * v)
            shared = 0
            for s in spots_per_grain[g]:
                claim_count = sum(1 for gg in grains_in_p1 if s in spots_per_grain[gg])
                if claim_count > 1: shared += 1
            n_shared[g] = shared

    deflation = np.where(volume_naive > 0, volume_nnls / volume_naive, 1.0)
    return GrainSizeRecomputeResult(
        grain_id=np.arange(n_g, dtype=np.int64),
        volume_naive_um3=volume_naive,
        volume_nnls_um3=volume_nnls,
        radius_naive_um=radius_naive,
        radius_nnls_um=radius_nnls,
        n_shared_spots=n_shared,
        deflation_factor=deflation,
        sigma_R_nnls_um=sigma_R,
    )


def merge_twin_family_volumes(
    *,
    volumes_um3: np.ndarray,
    twin_family_id: np.ndarray,
) -> Dict[int, float]:
    """For each twin-family parent ID, return the SUM of child grain volumes.

    Twin partners are physically distinct grains; parent-level "twin
    family" volume is the sum of children's individually-attributed
    volumes (no double counting because NNLS already attributed shared
    spots once per child).
    """
    fams: Dict[int, float] = {}
    for g, fam in enumerate(twin_family_id):
        if fam < 0: continue
        fams[int(fam)] = fams.get(int(fam), 0.0) + float(volumes_um3[g])
    return fams
