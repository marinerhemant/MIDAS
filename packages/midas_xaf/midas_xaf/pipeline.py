"""End-to-end analysis pipeline on the digital twin.

Ties the pieces together on realistic *measured* (unlabelled, noisy, spurious-
contaminated) spots: assign measured spots to a seed grain's predicted spots,
then robustly refine orientation/position/strain -- the real reconstruction that
day-1 beam-time data will need.  ``run_pipeline`` scores the whole chain against
ground truth (indexing is seeded from the uniqueness-verified orientation, i.e.
we already showed the orientation is recoverable; see :mod:`midas_xaf.indexing`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import XAFConfig
from .forward import XAFForwardModel
from .sample import GrainPopulation
from .metrics import _frozen_indices
from .reconstruct import _predict, mat2euler_zxz, _misorientation_deg, Recovery


_OMEGA_START_DEG = -180.0   # matches geometry.build_hedm_geometry


def _frame(omega_deg, cfg):
    return (np.asarray(omega_deg) - _OMEGA_START_DEG) / cfg.omega_step_deg


def _assign(fwd, euler_seed, pos_seed, measured, tol_px, tol_omega):
    """Match a seed grain's predicted spots to measured spots.

    Returns per-mounting ``(kk, hh, obs(P,3))`` of matched predicted-index ->
    measured (y, z, frame), plus assignment stats.
    """
    from scipy.spatial import cKDTree
    cfg = fwd.cfg
    dtype = fwd._latc0.dtype
    frozen = _frozen_indices(
        fwd, torch.as_tensor(euler_seed, dtype=dtype).view(1, 3),
        torch.as_tensor(pos_seed, dtype=dtype).view(1, 3),
        torch.zeros(6, dtype=dtype), list(range(cfg.n_mountings)))
    latc = fwd._latc0.unsqueeze(0)
    tol_frame = tol_omega / abs(cfg.omega_step_deg)
    scale = np.array([tol_px, tol_px, tol_frame])
    obs_out, n_matched, n_pred = [], 0, 0
    for m, (euler_m, pos_m, kk, hh) in enumerate(frozen):
        ms = measured.for_mounting(m)
        if len(ms) == 0 or kk.numel() == 0:
            obs_out.append((kk[:0], hh[:0], torch.zeros(0, 3, dtype=dtype)))
            continue
        with torch.no_grad():
            sd = fwd.model(euler_m, pos_m, lattice_params=latc,
                           strain=torch.zeros(1, 6, dtype=dtype))
            pred = torch.stack([sd.y_pixel[0], sd.z_pixel[0], sd.frame_nr[0]],
                               -1)[kk, hh].cpu().numpy()
        meas = np.stack([ms.y_pixel, ms.z_pixel, _frame(ms.omega_deg, cfg)], axis=1)
        d, j = cKDTree(meas / scale).query(pred / scale, k=1)
        ok = torch.as_tensor(d <= 1.0)
        n_pred += len(pred); n_matched += int(ok.sum())
        obs_out.append((kk[ok], hh[ok],
                        torch.as_tensor(meas[j[ok.numpy()]], dtype=dtype)))
    return obs_out, {"n_pred": n_pred, "n_matched": n_matched}


def refine_from_measured(fwd, euler_seed, pos_seed, measured, *, tol_px=3.0,
                         tol_omega=1.0, huber_delta=5.0, max_iter=50):
    """LM refine (euler, pos, strain) against assigned measured spots."""
    cfg = fwd.cfg
    dtype = fwd._latc0.dtype
    obs, stats = _assign(fwd, euler_seed, pos_seed, measured, tol_px, tol_omega)
    n = sum(int(kk.numel()) for (kk, _, _) in obs)
    if n < 8:
        return None, stats
    wv = torch.tensor([1.0 / cfg.sigma_det_px, 1.0 / cfg.sigma_det_px,
                       1.0 / cfg.sigma_omega_steps], dtype=dtype)
    mountings = list(range(cfg.n_mountings))

    def residual(params):
        parts = []
        for (mi, (kk, hh, o)) in zip(mountings, obs):
            if kk.numel() == 0:
                continue
            pred = _predict(fwd, params[:3], params[3:6], params[6:12], mi, kk, hh)
            parts.append(((pred - o) * wv).reshape(-1))
        return torch.cat(parts) if parts else torch.zeros(0, dtype=dtype)

    params = torch.cat([torch.as_tensor(euler_seed, dtype=dtype).view(3),
                        torch.as_tensor(pos_seed, dtype=dtype).view(3),
                        torch.zeros(6, dtype=dtype)])

    def hcost(r):
        a = r.abs()
        return float(torch.where(a <= huber_delta, r * r, 2 * huber_delta * a - huber_delta ** 2).sum())

    def hw(r):
        a = r.abs()
        return torch.where(a <= huber_delta, torch.ones_like(a), torch.sqrt(huber_delta / a.clamp(min=1e-12)))

    lam = 1e-3
    for _ in range(max_iter):
        r = residual(params); cost = hcost(r); w = hw(r)
        J = torch.func.jacfwd(residual)(params)
        Jw = J * w[:, None]
        JTJ = Jw.T @ Jw; JTr = Jw.T @ (r * w)
        diag = torch.diagonal(JTJ).clamp(min=1e-30)
        stepped = False
        for _i in range(30):
            try:
                delta = torch.linalg.solve(JTJ + lam * torch.diag(diag), -JTr)
            except Exception:
                lam *= 10; continue
            if hcost(residual(params + delta)) < cost:
                params = params + delta; lam = max(lam / 3, 1e-12); stepped = True; break
            lam *= 3
            if lam > 1e14:
                break
        if not stepped or float(delta.abs().max()) < 1e-10:
            break
    return params.detach(), {**stats, "n_used": n}


@dataclass
class PipelineResult:
    median_misorientation_mdeg: float
    median_strain_err_ue: float
    median_assignment_purity: float
    frac_recovered: float
    n_grains: int


def run_pipeline(cfg: XAFConfig, grains: GrainPopulation, *, seed_perturb_deg=0.05,
                 detect_frac=0.6, spurious_frac=0.1, pos_noise_px=1.0,
                 omega_noise_deg=0.25, seed=0) -> PipelineResult:
    """Digital twin -> (uniqueness-seeded) index -> assign+refine -> score."""
    from . import synth
    fwd = XAFForwardModel(cfg)
    d = synth.make_measured_spots(cfg, grains, fwd=fwd, detect_frac=detect_frac,
                                  spurious_frac=spurious_frac, pos_noise_px=pos_noise_px,
                                  omega_noise_deg=omega_noise_deg, seed=seed)
    measured = d["spots"]
    rng = np.random.default_rng(seed + 1)
    dtype = fwd._latc0.dtype

    misos, strain_errs, purities, recovered = [], [], [], 0
    for gi in range(grains.n_grains):
        e_true = grains.euler[gi].cpu().numpy()
        p_true = grains.position[gi].cpu().numpy()
        # seed = truth + small perturbation (indexing shown to recover orientation)
        e_seed = e_true + rng.normal(scale=np.radians(seed_perturb_deg), size=3)
        params, stats = refine_from_measured(fwd, e_seed, np.zeros(3), measured)
        # assignment purity: fraction of assigned measured spots truly from this grain
        obs_pur = _assignment_purity(fwd, e_seed, measured, gi)
        purities.append(obs_pur)
        if params is None:
            continue
        recovered += 1
        miso = _misorientation_deg(torch.as_tensor(e_true, dtype=dtype),
                                   params[:3])
        s_true = grains.strain[gi].to(dtype)
        strain_err = float((params[6:12] - s_true).pow(2).mean().sqrt() * 1e6)
        misos.append(miso * 1000.0)
        strain_errs.append(strain_err)
    return PipelineResult(
        median_misorientation_mdeg=float(np.median(misos)) if misos else float("nan"),
        median_strain_err_ue=float(np.median(strain_errs)) if strain_errs else float("nan"),
        median_assignment_purity=float(np.median(purities)) if purities else float("nan"),
        frac_recovered=recovered / grains.n_grains, n_grains=grains.n_grains)


def _assignment_purity(fwd, euler_seed, measured, true_gid, tol_px=3.0, tol_omega=1.0):
    """Fraction of assigned measured spots that truly belong to this grain."""
    from scipy.spatial import cKDTree
    cfg = fwd.cfg; dtype = fwd._latc0.dtype
    frozen = _frozen_indices(fwd, torch.as_tensor(euler_seed, dtype=dtype).view(1, 3),
                             torch.zeros(1, 3, dtype=dtype), torch.zeros(6, dtype=dtype),
                             list(range(cfg.n_mountings)))
    latc = fwd._latc0.unsqueeze(0)
    tol_frame = tol_omega / abs(cfg.omega_step_deg)
    tot = good = 0
    for m, (euler_m, pos_m, kk, hh) in enumerate(frozen):
        ms = measured.for_mounting(m)
        if len(ms) == 0 or kk.numel() == 0:
            continue
        with torch.no_grad():
            sd = fwd.model(euler_m, pos_m, lattice_params=latc, strain=torch.zeros(1, 6, dtype=dtype))
            pred = torch.stack([sd.y_pixel[0], sd.z_pixel[0], sd.frame_nr[0]], -1)[kk, hh].cpu().numpy()
        meas = np.stack([ms.y_pixel, ms.z_pixel, _frame(ms.omega_deg, cfg)], axis=1)
        tree = cKDTree(meas / np.array([tol_px, tol_px, tol_frame]))
        dd, jj = tree.query(pred / np.array([tol_px, tol_px, tol_frame]), k=1)
        ok = dd <= 1.0
        tot += int(ok.sum())
        good += int((ms.true_grain_id[jj[ok]] == true_gid).sum())
    return good / tot if tot else float("nan")
