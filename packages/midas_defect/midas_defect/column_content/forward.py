"""Differentiable reflection prediction for the column-content fit (monochromatic rotation geometry).

Uses exactly the geometry primitives the DAC ingest chain uses (:mod:`midas_defect.geometry`):
``q_sample = U @ B @ hkl``, the Ewald condition ``q_lab_x(omega) = -|q|^2 / (2 k0)``, ``q_lab = R_y(-W) R_z(omega) q_sample``
and :func:`~midas_defect.geometry.qlab_to_pixel` (tilts + full 15-term distortion, Newton inverse). The only new piece is
:func:`ewald_omega_torch`, the closed-form crossing of :func:`~midas_defect.geometry.ewald_crossing_omegas` written in torch,
so a small orientation offset can be differentiated through omega, row and col.

The crossing BRANCH (which of the two roots, and the 2*pi wrap) is fixed per reflection at the seed orientation by
:func:`observable_reflections`; offsets small enough not to change the branch (well under a degree near grazing
crossings) are what the joint fit uses. :func:`guard` checks the torch path against the numpy reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from midas_defect.geometry import Geometry, ewald_crossing_omegas, qlab_to_pixel, qsample_to_qlab

DT = torch.float64
TOL_PX = 1e-6          # Newton tolerance for qlab_to_pixel in BOTH paths (the 1e-3 default made the guard sit at its tolerance)


@dataclass
class Observable:
    """Every reflection of one orientation that is predicted on the detector inside the swept omega range."""
    hkl: np.ndarray          # (n, 3) int
    branch: np.ndarray       # (n,) +1 / -1: omega = phi + branch * d (unwrapped)
    wrap: np.ndarray         # (n,) int: + 2*pi*wrap
    frame: np.ndarray        # (n,) fractional frame index (stack-local)
    row: np.ndarray          # (n,)
    col: np.ndarray          # (n,)
    two_theta_deg: np.ndarray


def _phi_d(q, wavelength_A, wedge_rad):
    k0 = 2.0 * math.pi / wavelength_A
    cW, sW = math.cos(wedge_rad), math.sin(wedge_rad)
    A = q[0] * cW
    Bc = -q[1] * cW
    C = -(q @ q) / (2.0 * k0) + sW * q[2]
    R = math.hypot(A, Bc)
    if R == 0.0 or abs(C) > R * (1.0 + 1e-12):
        return None
    return math.atan2(Bc, A), math.acos(max(-1.0, min(1.0, C / R)))


def observable_reflections(U: np.ndarray, B: np.ndarray, hkl_all: np.ndarray, geom: Geometry, *,
                           omega_sign: int = 1, two_theta_min_deg: float = 0.0) -> Observable:
    """Numpy reference prediction (the same loop as :func:`midas_defect.synthetic.synthetic_position_frames`), also
    recording each crossing's branch and wrap so :func:`predict_torch` can follow it under small rotations."""
    U = np.asarray(U, float); B = np.asarray(B, float)
    wedge = math.radians(geom.wedge_deg)
    lo = math.radians(geom.omega_first_deg - 0.5 * geom.omega_step_deg)
    hi = math.radians(geom.omega_first_deg + (geom.n_frames - 0.5) * geom.omega_step_deg)
    lo, hi = min(lo, hi), max(lo, hi)
    out = dict(hkl=[], branch=[], wrap=[], frame=[], row=[], col=[], tth=[])
    for h in np.asarray(hkl_all):
        q = U @ B @ np.asarray(h, float)
        qm = float(np.linalg.norm(q))
        if qm < 1e-9:
            continue
        pd = _phi_d(q, geom.wavelength_A, wedge)
        if pd is None:
            continue
        phi, d = pd
        tth = math.degrees(2.0 * math.asin(min(1.0, qm * geom.wavelength_A / (4.0 * math.pi))))
        if tth < two_theta_min_deg:
            continue
        for br in (-1, 1):
            for n in (-1, 0, 1):
                w = phi + br * d + 2.0 * math.pi * n
                wr = omega_sign * w
                if not (lo <= wr <= hi):
                    continue
                qlab = qsample_to_qlab(torch.as_tensor(q, dtype=DT), w, wedge)
                try:
                    r, c = qlab_to_pixel(qlab.reshape(1, 3), geom, device="cpu", dtype=DT, tol_px=TOL_PX)
                except RuntimeError:
                    continue
                r, c = float(r[0]), float(c[0])
                if not (np.isfinite(r) and np.isfinite(c) and 0 <= r < geom.n_pix_z and 0 <= c < geom.n_pix_y):
                    continue
                f = (math.degrees(wr) - geom.omega_first_deg) / geom.omega_step_deg
                out["hkl"].append(np.asarray(h, int)); out["branch"].append(br); out["wrap"].append(n)
                out["frame"].append(f); out["row"].append(r); out["col"].append(c); out["tth"].append(tth)
    if not out["hkl"]:
        z = np.zeros(0)
        return Observable(np.zeros((0, 3), int), z.astype(int), z.astype(int), z, z, z, z)
    return Observable(np.asarray(out["hkl"], int), np.asarray(out["branch"], int), np.asarray(out["wrap"], int),
                      np.asarray(out["frame"]), np.asarray(out["row"]), np.asarray(out["col"]), np.asarray(out["tth"]))


def rotvec_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """(..., 3) rotation vectors (radians) -> (..., 3, 3); differentiable, safe at w = 0."""
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def ewald_omega_torch(q: torch.Tensor, branch: torch.Tensor, wrap: torch.Tensor, wavelength_A: float,
                      wedge_rad: float) -> torch.Tensor:
    """omega (radians, unwrapped branch) for sample-frame q (..., 3); torch twin of ewald_crossing_omegas."""
    k0 = 2.0 * math.pi / wavelength_A
    cW, sW = math.cos(wedge_rad), math.sin(wedge_rad)
    A = q[..., 0] * cW
    Bc = -q[..., 1] * cW
    C = -(q * q).sum(-1) / (2.0 * k0) + sW * q[..., 2]
    R = torch.sqrt(A * A + Bc * Bc).clamp_min(1e-12)
    d = torch.acos(torch.clamp(C / R, -1.0 + 1e-12, 1.0 - 1e-12))
    return torch.atan2(Bc, A) + branch * d + 2.0 * math.pi * wrap


def predict_torch(U0: np.ndarray, B: np.ndarray, obs: Observable, rotvecs: torch.Tensor, geom: Geometry, *,
                  omega_sign: int = 1):
    """frame, row, col of every observable reflection for K orientations U_k = R(rotvec_k) @ U0.

    rotvecs (K, 3) radians, sample frame. Returns three (K, n) tensors (differentiable in rotvecs)."""
    Um = rotvec_to_matrix(rotvecs) @ torch.as_tensor(np.asarray(U0, float) @ np.asarray(B, float), dtype=DT)  # (K,3,3)
    h = torch.as_tensor(obs.hkl, dtype=DT)                                           # (n,3)
    q = torch.einsum("kij,nj->kni", Um, h)                                           # (K,n,3)
    br = torch.as_tensor(obs.branch, dtype=DT); wr = torch.as_tensor(obs.wrap, dtype=DT)
    wedge = math.radians(geom.wedge_deg)
    w = ewald_omega_torch(q, br, wr, geom.wavelength_A, wedge)                       # (K,n)
    qlab = qsample_to_qlab(q, w, wedge)
    r, c = qlab_to_pixel(qlab.reshape(-1, 3), geom, device="cpu", dtype=DT, tol_px=TOL_PX)
    frame = (torch.rad2deg(omega_sign * w) - geom.omega_first_deg) / geom.omega_step_deg
    return frame, r.reshape(w.shape), c.reshape(w.shape)


def guard(U0: np.ndarray, B: np.ndarray, hkl_all: np.ndarray, geom: Geometry, *, omega_sign: int = 1,
          tol_px: float = 1e-4) -> float:
    """Max |torch - numpy| (px or frames) at zero offset; raises if above tol_px. Run before any fit."""
    obs = observable_reflections(U0, B, hkl_all, geom, omega_sign=omega_sign)
    if not len(obs.hkl):
        return 0.0
    f, r, c = predict_torch(U0, B, obs, torch.zeros(1, 3, dtype=DT), geom, omega_sign=omega_sign)
    err = max(float(np.max(np.abs(f[0].detach().numpy() - obs.frame))),
              float(np.max(np.abs(r[0].detach().numpy() - obs.row))),
              float(np.max(np.abs(c[0].detach().numpy() - obs.col))))
    if err > tol_px:
        raise RuntimeError(f"column_content forward guard failed: torch vs numpy differ by {err:.3g}")
    return err


def crossing_check(U0, B, hkl_all, geom):
    """Cross-check observable_reflections against geometry.ewald_crossing_omegas (the package reference)."""
    wedge = math.radians(geom.wedge_deg)
    obs = observable_reflections(U0, B, hkl_all, geom)
    worst = 0.0
    for h, br, n in zip(obs.hkl, obs.branch, obs.wrap):
        q = np.asarray(U0, float) @ np.asarray(B, float) @ h.astype(float)
        phi, d = _phi_d(q, geom.wavelength_A, wedge)
        w = (phi + br * d + math.pi) % (2 * math.pi) - math.pi
        ref = ewald_crossing_omegas(q, geom.wavelength_A, wedge)
        worst = max(worst, float(np.min(np.abs(ref - w))))
    return worst
