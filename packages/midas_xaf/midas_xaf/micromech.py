"""Micromechanics forward+inverse hook for XAF-HEDM.

Applies a **macroscopic stress state** (compression / tension / shear — whatever
the DAC loading imposes) to the grain population, forward-simulates, runs the
merged reconstruction, and backs out the stress from the recovered per-grain
strains.  This is the actual measurement the cell is being built for: grain-
resolved response under a controlled multiaxial load.

Elasticity is isotropic here (parameterised by E, nu) as a clean, plug-in
framework; a single-crystal stiffness tensor can replace :func:`isotropic_strain`
once the material/loading is finalised by the design team.  Stresses are in MPa,
strains dimensionless (reported in microstrain), E in GPa.

Voigt convention matches the forward model's ``strain``:
``[e11, e12, e13, e22, e23, e33]`` (plain, not engineering).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import torch

from .config import XAFConfig
from .forward import XAFForwardModel
from .sample import GrainPopulation, make_sample
from . import reconstruct as _R


def voigt6_to_mat(v) -> np.ndarray:
    v = np.asarray(v, float)
    return np.array([[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]])


def mat_to_voigt6(m) -> np.ndarray:
    m = np.asarray(m, float)
    return np.array([m[0, 0], m[0, 1], m[0, 2], m[1, 1], m[1, 2], m[2, 2]])


def isotropic_strain(stress3x3, E_MPa: float, nu: float) -> np.ndarray:
    """Sample-frame strain from stress via isotropic compliance."""
    s = np.asarray(stress3x3, float)
    return ((1.0 + nu) * s - nu * np.trace(s) * np.eye(3)) / E_MPa


def isotropic_stress(strain3x3, E_MPa: float, nu: float) -> np.ndarray:
    """Inverse: stress from strain (isotropic stiffness)."""
    e = np.asarray(strain3x3, float)
    lam = E_MPa * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E_MPa / (2 * (1 + nu))
    return lam * np.trace(e) * np.eye(3) + 2 * mu * e


def corundum_stiffness_GPa() -> np.ndarray:
    """Single-crystal corundum (sapphire, trigonal 3m) stiffness, 6x6 Voigt (GPa).

    Voigt order [11,22,33,23,13,12].  Constants: C11=497.3, C33=500.9, C44=146.8,
    C12=162.8, C13=116.0, C14=-21.9 GPa."""
    C11, C33, C44, C12, C13, C14 = 497.3, 500.9, 146.8, 162.8, 116.0, -21.9
    C66 = 0.5 * (C11 - C12)
    return np.array([
        [C11, C12, C13,  C14, 0,   0],
        [C12, C11, C13, -C14, 0,   0],
        [C13, C13, C33,  0,   0,   0],
        [C14, -C14, 0,   C44, 0,   0],
        [0,   0,   0,    0,   C44, C14],
        [0,   0,   0,    0,   C14, C66]])


def _stress_to_voigt(s):                 # 3x3 -> [11,22,33,23,13,12]
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def _eng_voigt_strain_to_mat(ev):        # engineering Voigt strain -> 3x3 tensor
    e = np.zeros((3, 3))
    e[0, 0], e[1, 1], e[2, 2] = ev[0], ev[1], ev[2]
    e[1, 2] = e[2, 1] = ev[3] / 2
    e[0, 2] = e[2, 0] = ev[4] / 2
    e[0, 1] = e[1, 0] = ev[5] / 2
    return e


def grain_strains_from_stress_aniso(euler: torch.Tensor, stress3x3,
                                    C6x6_GPa) -> torch.Tensor:
    """Per-grain crystal-frame strain (N,6 plain-Voigt) using single-crystal
    compliance.  Stress in sample frame is rotated into each crystal frame,
    the anisotropic compliance is applied, and the result is returned in the
    forward model's plain-Voigt order [11,12,13,22,23,33]."""
    from midas_stress.orientation import euler_to_orient_mat_batch
    S6 = np.linalg.inv(np.asarray(C6x6_GPa, float) * 1e3)     # 1/MPa
    stress3x3 = np.asarray(stress3x3, float)
    oms = euler_to_orient_mat_batch(euler.detach().cpu().numpy()).reshape(-1, 3, 3)
    out = []
    for R in oms:
        sc = R.T @ stress3x3 @ R                              # crystal-frame stress
        ec = _eng_voigt_strain_to_mat(S6 @ _stress_to_voigt(sc))
        out.append(mat_to_voigt6(ec))
    return torch.as_tensor(np.stack(out), dtype=euler.dtype)


def grain_strains_from_stress(euler: torch.Tensor, stress3x3, E_MPa: float,
                              nu: float) -> torch.Tensor:
    """Per-grain crystal-frame strain (N,6 Voigt) for a sample-frame stress.

    ``eps_crystal = R^T eps_sample R`` with ``R`` the crystal->sample matrix.
    """
    from midas_stress.orientation import euler_to_orient_mat_batch
    eps_s = isotropic_strain(stress3x3, E_MPa, nu)
    oms = euler_to_orient_mat_batch(euler.detach().cpu().numpy()).reshape(-1, 3, 3)
    out = np.stack([mat_to_voigt6(R.T @ eps_s @ R) for R in oms], axis=0)
    return torch.as_tensor(out, dtype=euler.dtype)


def micromech_study(
    cfg: XAFConfig,
    stress3x3,                       # applied macroscopic stress (MPa), sample frame
    *,
    E_GPa: float = 400.0,            # ruby ~400 GPa
    nu: float = 0.23,
    n_grains: int = 10,
    seed: int = 0,
    noise: bool = True,
) -> Dict[str, object]:
    """Apply a stress state, reconstruct (merged), and recover the stress."""
    cfg = replace(cfg, n_grains=n_grains, seed=seed)
    g0 = make_sample(cfg)
    E = E_GPa * 1e3                  # MPa
    stress3x3 = np.asarray(stress3x3, float)

    strains = grain_strains_from_stress(g0.euler, stress3x3, E, nu)
    grains = GrainPopulation(euler=g0.euler, position=g0.position, strain=strains,
                             fiducial_euler=g0.fiducial_euler,
                             fiducial_position=g0.fiducial_position)
    fwd = XAFForwardModel(cfg)

    from midas_stress.orientation import euler_to_orient_mat_batch
    oms = euler_to_orient_mat_batch(g0.euler.detach().cpu().numpy()).reshape(-1, 3, 3)

    eps_s_rec, errs = [], []
    for gi in range(n_grains):
        rec = _R.reconstruct_grain(
            fwd, grains.euler[gi:gi + 1], grains.position[gi:gi + 1],
            grains.strain[gi], noise_sigma=noise, perturb_deg=0.02,
            perturb_um=0.5, perturb_strain=5e-5, seed=seed + gi)
        if not rec.converged or rec.recovered_strain is None:
            continue
        eps_c = voigt6_to_mat(rec.recovered_strain)
        R = oms[gi]
        eps_s_rec.append(R @ eps_c @ R.T)      # back to sample frame
        errs.append(rec.strain_error_ue)

    eps_s_applied = isotropic_strain(stress3x3, E, nu)
    eps_s_rec_mean = np.mean(eps_s_rec, axis=0) if eps_s_rec else np.full((3, 3), np.nan)
    stress_rec = isotropic_stress(eps_s_rec_mean, E, nu)

    def _dev(m):
        return m - np.trace(m) / 3.0 * np.eye(3)
    dev_err = float(np.sqrt(np.nanmean((_dev(stress_rec) - _dev(stress3x3)) ** 2)))
    hyd_err = abs(np.trace(stress_rec) - np.trace(stress3x3)) / 3.0
    return {
        "applied_stress_MPa": stress3x3,
        "recovered_stress_MPa": stress_rec,
        "applied_strain_ue": eps_s_applied * 1e6,
        "recovered_strain_ue": eps_s_rec_mean * 1e6,
        "stress_error_MPa": float(np.sqrt(np.nanmean((stress_rec - stress3x3) ** 2))),
        # split: deviatoric (shear-relevant) is far better recovered than the
        # worst-constrained hydrostatic/volumetric component.
        "deviatoric_stress_error_MPa": dev_err,
        "hydrostatic_stress_error_MPa": float(hyd_err),
        "median_per_grain_strain_err_ue": float(np.median(errs)) if errs else float("nan"),
        "n_recovered": len(eps_s_rec),
    }


def deviatoric_load(sigma_axial_MPa: float, axis: str = "x") -> np.ndarray:
    """Convenience: uniaxial stress (tension +, compression -) along an axis.

    Combine two (e.g. +x tension and -z compression) to make a shear-inducing
    deviatoric state, matching the cell's compression/tension design intent.
    """
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    s = np.zeros((3, 3))
    s[idx, idx] = sigma_axial_MPa
    return s
