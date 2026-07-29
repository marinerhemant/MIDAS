"""RMC analysis utilities: partial g(r), coordination numbers, ergodicity.

Day-3 building blocks:

  * :func:`partial_g_r`             — per-species-pair g_{ij}(r).
  * :func:`coordination_number`     — mean coordination Z_i within a shell.
  * :func:`ergodicity_diagnostics`  — autocorrelation of χ² trace + acceptance
    ratio + effective sample size.
  * :class:`CoordinationBias`       — soft constraint (added to χ²) that
    biases the ensemble toward a target coordination number for a given
    species pair.

All utilities operate on either a single :class:`Supercell` or a list of
snapshots (for the ensemble analysis of Day 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_FOUR_PI = 4.0 * float(np.pi)


# ---------------------------------------------------------------------------
# Partial g_{ij}(r)
# ---------------------------------------------------------------------------

def partial_g_r(
    supercell,
    r_grid: torch.Tensor,
    *,
    r_max: Optional[float] = None,
    bin_width: Optional[float] = None,
    min_image: bool = True,
) -> Dict[Tuple[str, str], torch.Tensor]:
    """Per-species-pair radial distribution g_{ij}(r).

    Returns a dict keyed by ``(species_i, species_j)`` (canonical ordering:
    ``species_i <= species_j`` alphabetically) with values shape
    ``(len(r_grid),)`` — the unnormalised g_{ij}(r) is normalised by

        g_{ij}(r) = n_{ij}(r) / (4π r² · Δr · ρ_j · N_i)

    for i == j, factor 2 on the pair count (each unordered same-species
    pair contributes twice). For i != j, no factor 2.
    """
    r_t = torch.as_tensor(r_grid, dtype=torch.float64)
    if r_max is None:
        r_max = float(r_t.max()) + 0.5
    if bin_width is None:
        bin_width = float(r_t[1] - r_t[0])

    dists, i_idx, j_idx = supercell.pair_distances(
        r_max=r_max, min_image=min_image, return_indices=True)
    dists_np = dists.numpy()
    species = np.array(supercell.species, dtype=object)
    N_of = {s: int(np.sum(species == s)) for s in supercell.species_set()}
    V = supercell.volume

    result: Dict[Tuple[str, str], torch.Tensor] = {}
    all_species = supercell.species_set()

    # Bin every pair by species and distance
    r_edges = torch.cat([r_t - bin_width / 2,
                          r_t[-1:] + bin_width / 2])
    r_edges_np = r_edges.numpy()

    for a in all_species:
        for b in all_species:
            if b < a:
                continue                    # canonical (i, j) with i <= j
            # Mask pairs of species (a, b) — remember pair_distances returns
            # each unordered pair once.
            sp_i = species[i_idx.numpy()]
            sp_j = species[j_idx.numpy()]
            mask = (((sp_i == a) & (sp_j == b))
                    | ((sp_i == b) & (sp_j == a)))
            d_ab = dists_np[mask]
            counts, _ = np.histogram(d_ab, bins=r_edges_np)
            counts = counts.astype(np.float64)
            # Same-species pairs: each unordered pair contributes 2 to
            # g_{aa} (i -> j and j -> i).
            if a == b:
                counts = counts * 2.0
            # Normalise: g_ij(r) = n_ij(r) / (4π r² Δr · ρ_j · N_i)
            rho_j = N_of[b] / V
            N_i = N_of[a]
            shell_vol = _FOUR_PI * (r_t.numpy() ** 2) * bin_width
            g = counts / (shell_vol * rho_j * max(N_i, 1))
            result[(a, b)] = torch.as_tensor(g, dtype=torch.float64)
    return result


# ---------------------------------------------------------------------------
# Coordination number
# ---------------------------------------------------------------------------

def coordination_number(
    supercell,
    *,
    r_shell: Tuple[float, float],
    species_i: Optional[str] = None,
    species_j: Optional[str] = None,
    min_image: bool = True,
) -> Dict[str, float]:
    """Mean coordination Z of species_i by species_j within (r_min, r_max).

    Returns a dict with keys:
        ``"Z_mean"``, ``"Z_std"``, ``"n_center_atoms"``,
        ``"r_min"``, ``"r_max"``.

    If ``species_i`` / ``species_j`` are None, sums over all species (total
    coordination).
    """
    r_min, r_max = float(r_shell[0]), float(r_shell[1])
    dists, i_idx, j_idx = supercell.pair_distances(
        r_max=r_max, min_image=min_image, return_indices=True)
    species = np.array(supercell.species, dtype=object)
    sp_i = species[i_idx.numpy()]
    sp_j = species[j_idx.numpy()]
    d = dists.numpy()

    # Mask species combo
    if species_i is None:
        mask_i = np.ones(d.shape, dtype=bool)
    else:
        mask_i = (sp_i == species_i) | (sp_j == species_i)
    if species_j is None:
        mask_j = np.ones(d.shape, dtype=bool)
    else:
        mask_j = (sp_i == species_j) | (sp_j == species_j)
    # Distance mask
    mask_r = (d >= r_min) & (d <= r_max)
    mask = mask_i & mask_j & mask_r

    # Count coordination per center atom
    # Every unordered pair contributes to BOTH endpoints' coordination
    center_species = species_i if species_i is not None else None
    if center_species is None:
        centers = np.arange(supercell.n_atoms)
    else:
        centers = np.where(species == center_species)[0]
    if centers.size == 0:
        return {"Z_mean": 0.0, "Z_std": 0.0,
                 "n_center_atoms": 0, "r_min": r_min, "r_max": r_max}

    per_atom = np.zeros(supercell.n_atoms, dtype=np.float64)
    keep_i = i_idx.numpy()[mask]
    keep_j = j_idx.numpy()[mask]
    # If species_j is not None, only count neighbours of the matching species
    if species_j is not None:
        sp_i_kept = species[keep_i]
        sp_j_kept = species[keep_j]
        # For each pair, add 1 to center if the OTHER side is species_j
        for i_atom, j_atom, s_i, s_j in zip(keep_i, keep_j, sp_i_kept, sp_j_kept):
            if s_j == species_j:
                per_atom[i_atom] += 1.0
            if s_i == species_j:
                per_atom[j_atom] += 1.0
    else:
        # No species_j filter → both endpoints count
        for i_atom, j_atom in zip(keep_i, keep_j):
            per_atom[i_atom] += 1.0
            per_atom[j_atom] += 1.0

    per_center = per_atom[centers]
    return {
        "Z_mean": float(per_center.mean()),
        "Z_std": float(per_center.std()),
        "n_center_atoms": int(centers.size),
        "r_min": r_min,
        "r_max": r_max,
    }


# ---------------------------------------------------------------------------
# Ergodicity diagnostics
# ---------------------------------------------------------------------------

def ergodicity_diagnostics(rmc_result) -> Dict[str, float]:
    """Simple diagnostics on an RMCResult: acceptance ratio,
    χ² autocorrelation time, effective sample size.
    """
    chi2 = np.asarray(rmc_result.chi2_trace, dtype=np.float64)
    n = chi2.size
    if n < 4:
        return {"acceptance_ratio": rmc_result.acceptance_ratio,
                 "n_moves": rmc_result.n_moves, "autocorr_time": float("nan"),
                 "effective_sample_size": float("nan")}
    # De-mean before autocorrelation
    chi2_c = chi2 - chi2.mean()
    var = float(chi2_c.var())
    if var <= 0:
        return {"acceptance_ratio": rmc_result.acceptance_ratio,
                 "n_moves": rmc_result.n_moves, "autocorr_time": 0.0,
                 "effective_sample_size": float(n)}
    # ACF via FFT (fast for long traces)
    f = np.fft.fft(chi2_c, n=2 * n)
    acf = np.fft.ifft(f * f.conj()).real[:n]
    acf = acf / acf[0]
    # Integrated autocorrelation time — first crossing of 0.05
    crossings = np.where(acf < 0.05)[0]
    tau = int(crossings[0]) if crossings.size > 0 else n
    ess = max(1.0, n / max(2.0 * tau, 1.0))
    return {
        "acceptance_ratio": rmc_result.acceptance_ratio,
        "n_moves": rmc_result.n_moves,
        "autocorr_time": float(tau),
        "effective_sample_size": float(ess),
    }


# ---------------------------------------------------------------------------
# CoordinationBias — soft constraint added to χ²
# ---------------------------------------------------------------------------

@dataclass
class CoordinationBias:
    """Soft coordination-number constraint added to χ² as::

        chi2_bias = weight · (Z_observed − Z_target)²

    where Z_observed is computed within ``r_shell`` for the given
    species pair. Use to keep RMC ensembles physically sensible
    (e.g. Si has coordination 4 in silica; steel is Fe-Fe ~ 8).
    """
    r_shell: Tuple[float, float]
    Z_target: float
    weight: float = 100.0
    species_i: Optional[str] = None
    species_j: Optional[str] = None

    def penalty(self, supercell) -> float:
        info = coordination_number(
            supercell, r_shell=self.r_shell,
            species_i=self.species_i, species_j=self.species_j)
        return self.weight * (info["Z_mean"] - self.Z_target) ** 2


__all__ = [
    "partial_g_r",
    "coordination_number",
    "ergodicity_diagnostics",
    "CoordinationBias",
]
