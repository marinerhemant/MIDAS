"""Merged-reconstruction validation for XAF-HEDM.

A correspondence-known joint refinement of one grain's (orientation, position,
strain) against its accessible spots across **both** mountings, in a common
crystal frame.  In simulation the reflection behind each spot is known, so no
(non-differentiable) spot assignment is needed -- this isolates the question
Phase 2 must answer: *given the merged geometry, are the grain parameters
recoverable, and does merging beat a single mounting?*  It is a local refine
(started near truth); basin-of-convergence / global indexing is a separate
concern handled by the production indexer.

Orientation is carried across the remount differentiably: mounting-m orientation
matrix is ``R_mount @ euler2mat(euler_1)``, converted back to Euler with a torch
ZXZ inverse so autograd flows to the single fitted ``euler_1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .forward import XAFForwardModel
from .metrics import _frozen_indices, _noise_weights


# --------------------------------------------------------------------------- #
#  Differentiable ZXZ (Bunge) matrix -> Euler                                 #
# --------------------------------------------------------------------------- #
def mat2euler_zxz(R: torch.Tensor) -> torch.Tensor:
    """Inverse of ``euler2mat`` (R = Rz(phi1) Rx(Phi) Rz(phi2)); differentiable."""
    Phi = torch.acos(R[2, 2].clamp(-1.0, 1.0))
    phi1 = torch.atan2(R[0, 2], -R[1, 2])
    phi2 = torch.atan2(R[2, 0], R[2, 1])
    return torch.stack([phi1, Phi, phi2])


def _misorientation_deg(e1: torch.Tensor, e2: torch.Tensor) -> float:
    R1 = XAFForwardModel_euler2mat(e1)
    R2 = XAFForwardModel_euler2mat(e2)
    tr = torch.trace(R1.T @ R2)
    return float(torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)) * 180.0 / np.pi)


def XAFForwardModel_euler2mat(e: torch.Tensor) -> torch.Tensor:
    import midas_diffract as md
    return md.HEDMForwardModel.euler2mat(e)


# --------------------------------------------------------------------------- #
#  Observed spots (frozen correspondence)                                     #
# --------------------------------------------------------------------------- #
def _make_observed(fwd, euler_g, pos_g, strain_g, mountings, noise_sigma, seed):
    """Per-mounting ``(kk, hh, obs(P,3))`` at the true parameters (+ noise)."""
    frozen = _frozen_indices(fwd, euler_g, pos_g, strain_g, mountings)
    latc = fwd._latc0.unsqueeze(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = []
    for (euler_m, pos_m, kk, hh) in frozen:
        sd = fwd.model(euler_m, pos_m, lattice_params=latc,
                       strain=strain_g.view(1, 6))
        o = torch.stack([sd.y_pixel[0], sd.z_pixel[0], sd.frame_nr[0]],
                        dim=-1)[kk, hh].detach()               # (P, 3)
        if noise_sigma is not None:
            s = torch.tensor([fwd.cfg.sigma_det_px, fwd.cfg.sigma_det_px,
                              fwd.cfg.sigma_omega_steps], dtype=o.dtype)
            o = o + torch.randn(o.shape, generator=g).to(o.dtype) * s
        obs.append((kk, hh, o))
    return obs


def _predict(fwd, euler1, pos, strain, mounting, kk, hh):
    """Predicted (y, z, frame) for mounting ``mounting`` at fit params."""
    if mounting == 0:
        euler_m = euler1
    else:
        R = fwd._Rmounts_t[mounting].to(euler1.dtype)
        euler_m = mat2euler_zxz(R @ XAFForwardModel_euler2mat(euler1))
    pos_m = fwd.mounting_position(pos.view(1, 3), mounting)
    latc = fwd._latc0.unsqueeze(0)
    sd = fwd.model(euler_m.view(1, 3), pos_m, lattice_params=latc,
                   strain=strain.view(1, 6))
    o = torch.stack([sd.y_pixel[0], sd.z_pixel[0], sd.frame_nr[0]], dim=-1)
    return o[kk, hh]


@dataclass
class Recovery:
    misorientation_deg: float
    position_error_um: float
    strain_error_ue: float          # RMS over the 6 Voigt components
    n_spots: int
    converged: bool
    recovered_euler: Optional[np.ndarray] = None    # (3,) rad
    recovered_position: Optional[np.ndarray] = None  # (3,) um
    recovered_strain: Optional[np.ndarray] = None    # (6,) crystal Voigt


def reconstruct_grain(
    fwd: XAFForwardModel,
    euler_g: torch.Tensor,   # (1,3) truth
    pos_g: torch.Tensor,     # (1,3) truth
    strain_g: torch.Tensor,  # (6,) truth
    *,
    mountings: Optional[Sequence[int]] = None,
    perturb_deg: float = 1.0,
    perturb_um: float = 3.0,
    perturb_strain: float = 5.0e-4,
    noise_sigma: Optional[bool] = True,
    seed: int = 0,
    max_iter: int = 60,
    huber_delta: float = 5.0,
) -> Recovery:
    """Local joint refine of one grain from its merged spots; returns errors."""
    if mountings is None:
        mountings = list(range(fwd.cfg.n_mountings))
    dtype = fwd._latc0.dtype
    euler_g = euler_g.view(3).to(dtype)
    pos_g = pos_g.view(3).to(dtype)
    strain_g = strain_g.view(6).to(dtype)

    obs = _make_observed(fwd, euler_g.view(1, 3), pos_g.view(1, 3), strain_g,
                         mountings, noise_sigma, seed)
    n_spots = sum(int(kk.numel()) for (kk, _, _) in obs)
    if n_spots < 6:
        return Recovery(float("nan"), float("nan"), float("nan"), n_spots, False)

    # Inverse-noise weights per observable (y, z, frame), broadcast per spot.
    wv = torch.tensor([1.0 / fwd.cfg.sigma_det_px, 1.0 / fwd.cfg.sigma_det_px,
                       1.0 / fwd.cfg.sigma_omega_steps], dtype=dtype)

    def residual(params: torch.Tensor) -> torch.Tensor:
        eu, pos, strain = params[:3], params[3:6], params[6:12]
        parts = []
        for (mi, (kk, hh, o)) in zip(mountings, obs):
            pred = _predict(fwd, eu, pos, strain, mi, kk, hh)   # (P, 3)
            parts.append(((pred - o) * wv).reshape(-1))
        return torch.cat(parts) if parts else torch.zeros(0, dtype=dtype)

    rng = np.random.default_rng(seed + 7)
    params = torch.cat([
        euler_g + torch.tensor(rng.normal(scale=np.radians(perturb_deg), size=3), dtype=dtype),
        pos_g + torch.tensor(rng.normal(scale=perturb_um, size=3), dtype=dtype),
        strain_g + torch.tensor(rng.normal(scale=perturb_strain, size=6), dtype=dtype),
    ])

    # Robust Levenberg-Marquardt with Huber IRLS.
    #  * Marquardt's diag(J^T J) damping is scale-invariant -- essential because
    #    the Hessian spans ~7 orders of magnitude (orientation/strain move spots
    #    strongly, far-field position weakly).
    #  * Huber reweighting caps the influence of near-Ewald-tangency reflections
    #    (near-infinite d(omega)/d(orientation)) that otherwise produce a few
    #    huge residuals off-truth and destabilise the Gauss-Newton step -- this
    #    widens the convergence basin substantially.
    def huber_cost(r):
        a = r.abs()
        d = huber_delta
        return float(torch.where(a <= d, r * r, 2 * d * a - d * d).sum())

    def huber_sqrt_w(r):
        a = r.abs()
        return torch.where(a <= huber_delta, torch.ones_like(a),
                           torch.sqrt(huber_delta / a.clamp(min=1e-12)))

    lam = 1e-3
    converged = False
    for _ in range(max_iter):
        r = residual(params)
        cost = huber_cost(r)
        wh = huber_sqrt_w(r)
        J = torch.func.jacfwd(residual)(params)          # (3P, 12)
        Jw = J * wh[:, None]
        JTJ = Jw.T @ Jw
        JTr = Jw.T @ (r * wh)
        diag = torch.diagonal(JTJ).clamp(min=1e-30)
        stepped = False
        for _inner in range(30):
            A = JTJ + lam * torch.diag(diag)
            try:
                delta = torch.linalg.solve(A, -JTr)
            except Exception:
                lam *= 10.0
                continue
            if huber_cost(residual(params + delta)) < cost:
                params = params + delta
                lam = max(lam / 3.0, 1e-12)
                stepped = True
                break
            lam *= 3.0
            if lam > 1e14:
                break
        if not stepped or float(delta.abs().max()) < 1e-10:
            converged = True
            break

    with torch.no_grad():
        eu, pos, strain = params[:3], params[3:6], params[6:12]
        misori = _misorientation_deg(euler_g, eu)
        pos_err = float((pos - pos_g).norm())
        strain_err = float(((strain - strain_g) ** 2).mean().sqrt() * 1e6)
    return Recovery(misori, pos_err, strain_err, n_spots, converged,
                    recovered_euler=eu.detach().cpu().numpy(),
                    recovered_position=pos.detach().cpu().numpy(),
                    recovered_strain=strain.detach().cpu().numpy())


def recovery_study(
    fwd: XAFForwardModel,
    grains,
    *,
    n_grains: Optional[int] = None,
    noise: bool = True,
    seed: int = 0,
    perturb_deg: float = 0.02,
    perturb_um: float = 0.5,
    perturb_strain: float = 5.0e-5,
) -> Dict[str, object]:
    """Compare single-mounting vs merged recovery over a few grains.

    Seeded near truth (post-indexing regime): this refiner has a narrow basin
    because spot positions are extremely stiff in orientation; global indexing
    that supplies the seed is a separate, production-indexer concern.  With a
    valid seed the fit reaches the CRLB, so the noisy single-vs-merged gap
    reflects the true information gain of the cross-axis merge.
    """
    n = grains.n_grains if n_grains is None else min(n_grains, grains.n_grains)
    out = {"single": [], "merged": []}
    kw = dict(noise_sigma=noise, perturb_deg=perturb_deg, perturb_um=perturb_um,
              perturb_strain=perturb_strain)
    for g in range(n):
        e = grains.euler[g:g + 1]
        p = grains.position[g:g + 1]
        s = grains.strain[g]
        out["single"].append(reconstruct_grain(
            fwd, e, p, s, mountings=[0], seed=seed + g, **kw))
        out["merged"].append(reconstruct_grain(
            fwd, e, p, s, mountings=list(range(fwd.cfg.n_mountings)),
            seed=seed + g, **kw))

    def agg(key, field):
        vals = [getattr(r, field) for r in out[key] if r.converged
                and np.isfinite(getattr(r, field))]
        return float(np.median(vals)) if vals else float("nan")

    return {
        "median_misori_deg_single": agg("single", "misorientation_deg"),
        "median_misori_deg_merged": agg("merged", "misorientation_deg"),
        "median_pos_err_um_single": agg("single", "position_error_um"),
        "median_pos_err_um_merged": agg("merged", "position_error_um"),
        "median_strain_err_ue_single": agg("single", "strain_error_ue"),
        "median_strain_err_ue_merged": agg("merged", "strain_error_ue"),
        "n_grains": n,
    }
