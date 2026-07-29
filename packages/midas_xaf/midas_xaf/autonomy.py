"""Autonomy driver v1: information-optimal acquisition scheduling.

The decide->acquire->update loop at the heart of self-driving diffraction,
demonstrated on the digital twin: given a menu of candidate acquisitions (omega
wedges, optionally across mountings), greedily choose the sequence that reduces
the strain-tensor uncertainty fastest (D-optimal / Fisher-information design),
and benchmark against uniform and random schedules.

v1 scope: candidates are (mounting, omega-wedge) pairs with uniform cost; the
information state is the accumulated Fisher matrix of the grain population's
strain observables (the same pixel/frame Jacobians used by the CRLB metrics).
Extensions (cost-aware budgets, event-driven re-tasking, online reconstruction
in the loop) layer on top without changing the interface.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import XAFConfig
from .forward import XAFForwardModel
from .sample import GrainPopulation
from .metrics import _noise_weights


@dataclass
class CandidateBank:
    """Per-candidate, per-grain strain-Jacobian blocks.

    ``J[c][g]`` is the (n_obs_cg, 6) noise-weighted Jacobian contributed by
    candidate acquisition ``c`` for grain ``g`` (rows = y/z/frame observables of
    the spots that candidate captures).
    """
    labels: List[str]
    J: List[List[torch.Tensor]]     # [n_candidates][n_grains]
    n_grains: int


def build_candidate_bank(
    cfg: XAFConfig,
    grains: GrainPopulation,
    *,
    wedge_centers_deg: Sequence[float],
    wedge_half_deg: float = 8.0,
    mountings: Optional[Sequence[int]] = None,
) -> CandidateBank:
    """Compute each candidate wedge's strain-information contribution.

    One full Jacobian per (grain, mounting) is computed with the wedge gate
    opened; rows are then assigned to candidate wedges by their omega, so the
    bank costs no more than the ordinary CRLB analysis.
    """
    from .metrics import _frozen_indices
    from .reconstruct import _predict

    if mountings is None:
        mountings = list(range(cfg.n_mountings))
    # open the omega gate: any omega passes; exit-cone/detector gates stay on
    cfg_open = replace(cfg, wedge_centers_deg=(0.0,), wedge_halfwidth_deg=180.0)
    fwd = XAFForwardModel(cfg_open)
    dtype = fwd._latc0.dtype

    labels = [f"m{m}:w{c:+.0f}" for m in mountings for c in wedge_centers_deg]
    n_c = len(labels)
    bank: List[List[torch.Tensor]] = [[torch.zeros(0, 6, dtype=dtype)
                                       for _ in range(grains.n_grains)]
                                      for _ in range(n_c)]
    centers = np.asarray(wedge_centers_deg, float)

    for g in range(grains.n_grains):
        e_g = grains.euler[g:g + 1].to(dtype)
        p_g = grains.position[g:g + 1].to(dtype)
        s_g = grains.strain[g].to(dtype)
        frozen = _frozen_indices(fwd, e_g, p_g, s_g, list(mountings))
        for mi, (euler_m, pos_m, kk, hh) in enumerate(frozen):
            if kk.numel() == 0:
                continue
            def obs(strain6, _kk=kk, _hh=hh, _mi=mountings[mi]):
                pred = _predict(fwd, e_g.view(3), p_g.view(3), strain6,
                                _mi, _kk, _hh)
                return pred.reshape(-1)
            J = torch.func.jacfwd(obs)(s_g.view(6)).reshape(-1, 3, 6)  # (P,3,6)
            w = _noise_weights(fwd, kk.numel(), J.device, J.dtype).reshape(-1, 3)
            Jw = (J * w[..., None])                                     # (P,3,6)
            # omega of each spot (deg) for wedge assignment
            with torch.no_grad():
                latc = fwd._latc0.unsqueeze(0)
                sd = fwd.model(euler_m, pos_m, lattice_params=latc,
                               strain=s_g.view(1, 6))
                om = torch.rad2deg(sd.omega[0])[kk, hh].cpu().numpy()
            d = np.abs(((om[:, None] - centers[None, :]) + 180.0) % 360.0 - 180.0)
            wedge_of_spot = np.argmin(d, axis=1)
            in_any = d[np.arange(len(om)), wedge_of_spot] <= wedge_half_deg
            for ci, c in enumerate(centers):
                sel = in_any & (wedge_of_spot == ci)
                if sel.any():
                    idx = mi * len(centers) + ci
                    bank[idx][g] = torch.cat(
                        [bank[idx][g], Jw[torch.as_tensor(sel)].reshape(-1, 6)])
    return CandidateBank(labels=labels, J=bank, n_grains=grains.n_grains)


def _worst_precision_ue(F: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(F)
    lo = float(ev.min())
    return 1e6 / np.sqrt(lo) if lo > 1e-12 else float("inf")


def _median_precision(Fs: List[torch.Tensor]) -> float:
    vals = [_worst_precision_ue(F) for F in Fs]
    return float(np.median(vals))


def run_schedule(bank: CandidateBank, order: Sequence[int]) -> List[float]:
    """Median worst-direction strain precision (µε) after each acquisition."""
    Fs = [torch.zeros(6, 6, dtype=bank.J[0][0].dtype)
          for _ in range(bank.n_grains)]
    out = []
    for c in order:
        for g in range(bank.n_grains):
            Jc = bank.J[c][g]
            if Jc.numel():
                Fs[g] = Fs[g] + Jc.T @ Jc
        out.append(_median_precision(Fs))
    return out


def greedy_schedule(bank: CandidateBank, k: int,
                    ridge: float = 1e-8) -> List[int]:
    """D-optimal greedy: pick the candidate maximizing the summed log-det gain
    of the per-grain Fisher matrices."""
    n_c = len(bank.labels)
    dtype = bank.J[0][0].dtype
    Fs = [ridge * torch.eye(6, dtype=dtype) for _ in range(bank.n_grains)]
    chosen: List[int] = []
    remaining = set(range(n_c))
    for _ in range(min(k, n_c)):
        best_c, best_gain = None, -np.inf
        for c in remaining:
            gain = 0.0
            for g in range(bank.n_grains):
                Jc = bank.J[c][g]
                if not Jc.numel():
                    continue
                Fn = Fs[g] + Jc.T @ Jc
                gain += float(torch.logdet(Fn) - torch.logdet(Fs[g]))
            if gain > best_gain:
                best_gain, best_c = gain, c
        chosen.append(best_c)
        remaining.discard(best_c)
        for g in range(bank.n_grains):
            Jc = bank.J[best_c][g]
            if Jc.numel():
                Fs[g] = Fs[g] + Jc.T @ Jc
    return chosen


def benchmark(
    cfg: XAFConfig,
    grains: GrainPopulation,
    *,
    wedge_centers_deg: Sequence[float],
    wedge_half_deg: float = 8.0,
    k: Optional[int] = None,
    n_random: int = 30,
    seed: int = 0,
) -> Dict[str, object]:
    """Active (greedy D-optimal) vs uniform vs random acquisition schedules.

    Returns per-step median strain precision for each policy, plus the number
    of acquisitions each needs to reach the precision that the *full* menu
    ultimately achieves x1.25 (a target-reaching comparison).
    """
    bank = build_candidate_bank(cfg, grains,
                                wedge_centers_deg=wedge_centers_deg,
                                wedge_half_deg=wedge_half_deg)
    n_c = len(bank.labels)
    k = k or n_c
    active_order = greedy_schedule(bank, k)
    active = run_schedule(bank, active_order)

    # uniform: evenly spread through the (mounting, wedge) menu
    uniform_order = [int(round(i * (n_c - 1) / max(k - 1, 1))) for i in range(k)]
    seen: List[int] = []
    for c in uniform_order:                      # dedupe while keeping order
        if c not in seen:
            seen.append(c)
    uniform = run_schedule(bank, seen)

    rng = np.random.default_rng(seed)
    rand_curves = [run_schedule(bank, rng.permutation(n_c)[:k])
                   for _ in range(n_random)]
    random_med = np.median(np.array(rand_curves), axis=0).tolist()

    target = active[-1] * 1.25
    def n_to_reach(curve):
        for i, v in enumerate(curve):
            if v <= target:
                return i + 1
        return None
    return {
        "labels": bank.labels,
        "active_order": [bank.labels[c] for c in active_order],
        "active": active,
        "uniform": uniform,
        "random_median": random_med,
        "target_ue": target,
        "n_active": n_to_reach(active),
        "n_uniform": n_to_reach(uniform),
        "n_random": n_to_reach(random_med),
    }
