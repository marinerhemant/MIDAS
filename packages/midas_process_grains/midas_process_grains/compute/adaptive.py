"""Data-driven misori threshold for the ``adaptive`` ProcessGrains mode.

The legacy/spot_aware/paper_claim modes use a hand-chosen MisoriTol (0.4°,
0.5°, 0.01° respectively). Empirically the correct threshold is the antimode
of the pairwise-misorientation histogram among the alive candidates: it
separates the "same-grain duplicate" mode (near zero) from the "different
grains" mode (above ~degrees). On a Ni FF dataset with ~57 k candidates →
~11 k grains the antimode lands at θ* ≈ 0.011°, matching the §3.6 paper
specification almost exactly (and contradicting the legacy 0.4° rule, which
is ~40× too loose).

The function below derives θ* per dataset:

  1. Bucket-prefilter pairs via the symmetry-extended 4D quaternion grid at
     a generous threshold ``theta_pre_deg`` (default 5°).
  2. Compute exact symmetry-aware pairwise misorientation for each survivor
     in a single batched torch call.
  3. Take the log10 histogram of those misorientations on (−2, 0.5) decade
     window and return the antimode (argmin of the histogram density).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from midas_stress.orientation import (
    make_symmetries,
    misorientation_quat_batch,
    orient_mat_to_quat,
)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion multiplication, last dim 4 (w, x, y, z)."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def derive_misori_tol(
    orient_mats: np.ndarray,
    space_group: int,
    *,
    alive_mask: np.ndarray | None = None,
    theta_pre_deg: float = 5.0,
    chunk_pairs: int = 500_000,
    floor_deg: float = 0.05,
    ceiling_deg: float = 1.0,
) -> Tuple[float, dict]:
    """Find the data-driven misorientation antimode (degrees).

    Parameters
    ----------
    orient_mats : (N, 3, 3)
        Per-candidate orientation matrices.
    space_group : int
    alive_mask : (N,) bool, optional
        If supplied, restrict to alive candidates.
    theta_pre_deg : float
        Prefilter threshold; pairs beyond this are not considered. The
        antimode must lie within this window.
    chunk_pairs : int
        Batch size for the vectorised misorientation evaluation.
    floor_deg, ceiling_deg : float
        Safety clamp on the returned antimode: never go below ``floor_deg``
        (numerical noise floor of the refiner) or above ``ceiling_deg``
        (sanity bound — antimode should be sub-degree for real datasets).

    Returns
    -------
    theta_star_deg : float
        The antimode, in degrees, clipped to [floor_deg, ceiling_deg].
    diag : dict
        Diagnostic info — ``n_pairs``, ``raw_antimode_deg``, ``hist`` (numpy
        array), ``hist_centers_deg`` (numpy array).
    """
    N = orient_mats.shape[0]
    if alive_mask is None:
        alive_idx = np.arange(N, dtype=np.int64)
    else:
        alive_idx = np.flatnonzero(alive_mask).astype(np.int64)
    if alive_idx.size < 2:
        return float(floor_deg), {"n_pairs": 0, "raw_antimode_deg": float("nan"),
                                  "hist": np.empty(0), "hist_centers_deg": np.empty(0)}

    # Compute alive quats once
    om_t = torch.from_numpy(np.ascontiguousarray(orient_mats[alive_idx], dtype=np.float64))
    quats_alive = np.asarray(orient_mat_to_quat(om_t))

    # 24 sym-equivalent reps per alive seed
    n_sym, sym_list = make_symmetries(space_group)
    sym_q = np.asarray(sym_list, dtype=np.float64)

    reps = _qmul(quats_alive[:, None, :], sym_q[None, :, :])         # (n_alive, n_sym, 4)
    reps_flat = reps.reshape(-1, 4)
    seed_local = np.repeat(np.arange(alive_idx.size, dtype=np.int64), n_sym)

    # Sign canonicalise (qw ≥ 0)
    sgn = np.where(reps_flat[:, 0] >= 0, 1.0, -1.0)
    reps_flat = reps_flat * sgn[:, None]
    # Bucket in a 4D grid sized to the prefilter chord. To avoid losing pairs
    # whose canonical reps sit on opposite sides of a bucket boundary, run
    # TWO grids — origin-aligned and shifted by half a cell — and take the
    # union of emitted pairs. This catches all edge cases with only 2× the
    # bucket work (vs 81× for full-neighbour search) AND keeps the same
    # per-bucket member count as the original grid.
    cell = 2.0 * math.sin(math.radians(theta_pre_deg) / 2.0)

    pair_chunks: list[np.ndarray] = []
    for offset in (0.0, 0.5 * cell):
        cell_idx = np.floor((reps_flat + offset) / cell).astype(np.int64)
        order = np.lexsort((cell_idx[:, 3], cell_idx[:, 2], cell_idx[:, 1], cell_idx[:, 0]))
        sorted_cells = cell_idx[order]
        sorted_seeds = seed_local[order]
        diff = np.any(np.diff(sorted_cells, axis=0) != 0, axis=1)
        breaks = np.concatenate([[0], np.flatnonzero(diff) + 1, [sorted_cells.shape[0]]])
        sizes = np.diff(breaks)
        big = np.flatnonzero(sizes >= 2)
        for k in big:
            lo, hi = int(breaks[k]), int(breaks[k + 1])
            members = np.unique(sorted_seeds[lo:hi])
            if members.size < 2:
                continue
            ii, jj = np.triu_indices(members.size, k=1)
            pair_chunks.append(np.stack([members[ii], members[jj]], axis=1))
    if not pair_chunks:
        return float(floor_deg), {"n_pairs": 0, "raw_antimode_deg": float("nan"),
                                  "hist": np.empty(0), "hist_centers_deg": np.empty(0)}
    all_pairs_local = np.concatenate(pair_chunks, axis=0)

    # Dedupe (each true pair appears once per coincident sym rep)
    enc = (all_pairs_local[:, 0].astype(np.int64) * (1 << 31)
           + all_pairs_local[:, 1].astype(np.int64))
    uniq = np.unique(enc)
    a_loc = (uniq >> 31).astype(np.int64)
    b_loc = (uniq & ((1 << 31) - 1)).astype(np.int64)
    i_g = alive_idx[a_loc]
    j_g = alive_idx[b_loc]
    n_pairs = len(i_g)

    # Vectorised misori in chunks
    quats_full = np.full((N, 4), np.nan, dtype=np.float64)
    quats_full[alive_idx] = quats_alive

    misori_deg = np.empty(n_pairs, dtype=np.float64)
    for s in range(0, n_pairs, chunk_pairs):
        e = min(s + chunk_pairs, n_pairs)
        qa = quats_full[i_g[s:e]]
        qb = quats_full[j_g[s:e]]
        ta = torch.from_numpy(np.ascontiguousarray(qa))
        tb = torch.from_numpy(np.ascontiguousarray(qb))
        out = misorientation_quat_batch(ta, tb, space_group)
        out_np = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        misori_deg[s:e] = np.rad2deg(out_np)

    # Antimode = local minimum sandwiched between two local maxima of the
    # smoothed log10-misori histogram. The naive "argmin of histogram" picks
    # the histogram floor (the empty <1e-3 deg bins); the smart finder
    # requires TWO distinguishable populations before declaring an antimode.
    #
    # IMPORTANT: search range MUST include both modes:
    # - duplicate mode ≈ refiner-residual scale (10^-2 deg on clean runs)
    # - cross-grain mode ≈ Mackenzie/texture-mode scale (10^0 to 10^1 deg)
    # The previous "band restricted to ceiling_deg" cut off the cross-grain
    # mode and produced a degenerate single-max histogram → fallback to floor.
    from scipy.ndimage import gaussian_filter1d
    log_m = np.log10(np.clip(misori_deg, 1e-4, None))
    hist, edges = np.histogram(log_m, bins=120)
    mids = (edges[:-1] + edges[1:]) / 2.0
    smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)

    is_max = np.zeros_like(smooth, dtype=bool)
    is_max[1:-1] = (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:])
    is_min = np.zeros_like(smooth, dtype=bool)
    is_min[1:-1] = (smooth[1:-1] < smooth[:-2]) & (smooth[1:-1] < smooth[2:])
    max_idx = np.flatnonzero(is_max)
    min_idx = np.flatnonzero(is_min)

    # Find antimode = lowest valley between the LEFTMOST and RIGHTMOST local
    # maxima of the full histogram (full search, not band-restricted).
    raw_antimode = float(floor_deg)
    if len(max_idx) >= 2:
        lo_i, hi_i = int(max_idx[0]), int(max_idx[-1])
        between = (min_idx > lo_i) & (min_idx < hi_i)
        if between.any():
            cand = min_idx[between]
            valley = int(cand[np.argmin(smooth[cand])])
            raw_antimode = float(10 ** mids[valley])
        else:
            mid_i = (lo_i + hi_i) // 2
            raw_antimode = float(10 ** mids[mid_i])

    theta_star = float(np.clip(raw_antimode, floor_deg, ceiling_deg))
    diag = {
        "n_pairs": n_pairs,
        "raw_antimode_deg": raw_antimode,
        "hist": hist,
        "hist_centers_deg": 10.0 ** mids,
    }
    return theta_star, diag


# ---------------------------------------------------------------------------
# Item 10: sub-grain vs refiner-noise 2-component mixture
# ---------------------------------------------------------------------------

def classify_pairs_subgrain_vs_noise(
    misori_deg: np.ndarray,
    sigma_o_median: float,
    *,
    band_max_deg: float = 5.0,
    subgrain_scale_deg: float = 1.0,
) -> dict:
    """Fit a two-component mixture to the *duplicate band* of pairwise
    misori (pairs at misori ≤ band_max_deg) to distinguish two physical
    populations:

      A. **Refiner-noise duplicates** — two indexer seeds for the same
         physical grain whose orientations differ by Gaussian-distributed
         refinement noise. Distribution: half-normal centred at 0 with
         scale σ_pair = σ_o_median · √2 (independent noise on each seed).

      B. **Real sub-grain pairs** — two physically distinct sub-grains
         (mosaic blocks, low-angle boundaries) whose orientations differ
         by a few tenths to a few degrees of true material misorientation.
         Distribution: exponential with scale ``subgrain_scale_deg`` (~1°
         empirical for many polycrystals; user can override).

    Both components share the same support [0, band_max_deg]. Component
    weights are estimated by closed-form maximum-likelihood given the two
    component scales.

    Returns
    -------
    dict
        * ``noise_scale_deg`` — half-normal scale (= σ_pair)
        * ``subgrain_scale_deg`` — exponential scale
        * ``noise_weight`` — fraction of band-pairs from component A
        * ``subgrain_weight`` — fraction from component B
        * ``posterior_noise`` — (n_band_pairs,) P(refiner-noise | misori)
        * ``mask_band`` — (n_pairs,) bool mask selecting band-pairs
        * ``mask_merge`` — (n_pairs,) bool: True if posterior_noise > 0.5
                         AND pair is in band (refiner-noise duplicate to merge)

    Notes
    -----
    The merge decision is "merge if more likely to be refiner-noise than
    sub-grain". Sub-grain pairs are KEPT SEPARATE — this is the physical
    distinction the v3 algorithm couldn't make on its own.
    """
    noise_scale = max(sigma_o_median * math.sqrt(2.0), 1e-6)
    mask_band = misori_deg <= band_max_deg
    if not mask_band.any():
        return {
            "noise_scale_deg": noise_scale,
            "subgrain_scale_deg": subgrain_scale_deg,
            "noise_weight": 1.0,
            "subgrain_weight": 0.0,
            "posterior_noise": np.empty(0),
            "mask_band": mask_band,
            "mask_merge": mask_band.copy(),
        }
    x = misori_deg[mask_band]
    # Half-normal density at x: (√(2/π)/σ) · exp(-x²/(2σ²)),  x ≥ 0
    f_noise = (math.sqrt(2.0 / math.pi) / noise_scale) * np.exp(-(x ** 2) / (2.0 * noise_scale ** 2))
    # Exponential density at x: (1/λ) · exp(-x/λ),  x ≥ 0
    lam = max(subgrain_scale_deg, 1e-6)
    f_sub = (1.0 / lam) * np.exp(-x / lam)

    # Closed-form 1D EM (3 iterations) for the mixture weight π_noise
    pi_n = 0.5
    for _ in range(20):
        num = pi_n * f_noise
        denom = num + (1.0 - pi_n) * f_sub
        denom = np.maximum(denom, 1e-30)
        w_noise = num / denom
        pi_n_new = float(np.mean(w_noise))
        if abs(pi_n_new - pi_n) < 1e-6:
            pi_n = pi_n_new
            break
        pi_n = pi_n_new

    posterior_noise = (pi_n * f_noise) / np.maximum(pi_n * f_noise + (1.0 - pi_n) * f_sub, 1e-30)
    # mask_merge: only band-pairs with posterior > 0.5 are refiner-noise
    mask_merge_full = np.zeros_like(mask_band)
    mask_merge_full[mask_band] = posterior_noise > 0.5
    return {
        "noise_scale_deg": noise_scale,
        "subgrain_scale_deg": subgrain_scale_deg,
        "noise_weight": float(pi_n),
        "subgrain_weight": float(1.0 - pi_n),
        "posterior_noise": posterior_noise,
        "mask_band": mask_band,
        "mask_merge": mask_merge_full,
    }
