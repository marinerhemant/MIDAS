"""Robustness studies: how much registration error and sample non-rigidity the
merged reconstruction tolerates. These set the experimental requirements -- how
precisely the remount must be recovered (fiducials) and how rigid the sample must
stay across the disassemble/remount cycle.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

import numpy as np
import torch

from .config import XAFConfig
from .forward import XAFForwardModel
from .sample import GrainPopulation
from . import synth
from .pipeline import refine_from_measured
from .reconstruct import _misorientation_deg

_ORTHO = (((1., 0, 0), 90.0), ((0, 1., 0), 90.0))


def registration_robustness(
    cfg: XAFConfig,
    grains: GrainPopulation,
    reg_errors_deg: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    *,
    n_grains: int = 8,
    seed: int = 1,
) -> List[Dict[str, float]]:
    """Degradation vs an error in the recovered remount transform.

    The digital twin is generated with the *true* remount; reconstruction uses a
    remount perturbed by ``reg_error`` (as fiducial registration would). Mounting
    1 is unaffected, so orientation survives, but the mis-registered second
    mounting stops contributing -- strain conditioning degrades toward the
    single-mounting value.
    """
    n_mount = max(cfg.n_mountings, 2)
    twin_cfg = replace(cfg, n_mountings=n_mount, remount_specs=_ORTHO,
                       n_grains=grains.n_grains)
    fwd_twin = XAFForwardModel(twin_cfg)
    measured = synth.make_measured_spots(twin_cfg, grains, fwd=fwd_twin,
                                         seed=seed)["spots"]
    rng = np.random.default_rng(seed + 5)
    dtype = fwd_twin._latc0.dtype

    rows = []
    for err in reg_errors_deg:
        ortho_err = (((1., 0, 0), 90.0 + err), ((0, 1., 0), 90.0 + err))
        refine_cfg = replace(cfg, n_mountings=n_mount, remount_specs=ortho_err,
                             n_grains=grains.n_grains)
        fwd_ref = XAFForwardModel(refine_cfg)
        misos, strains = [], []
        for gi in range(min(n_grains, grains.n_grains)):
            e_true = grains.euler[gi].cpu().numpy()
            e_seed = e_true + rng.normal(scale=np.radians(0.05), size=3)
            params, _ = refine_from_measured(fwd_ref, e_seed, np.zeros(3), measured)
            if params is None:
                continue
            misos.append(_misorientation_deg(
                torch.as_tensor(e_true, dtype=dtype), params[:3]) * 1000.0)
            strains.append(float((params[6:12] - grains.strain[gi].to(dtype)
                                  ).pow(2).mean().sqrt() * 1e6))
        rows.append({
            "reg_error_deg": float(err),
            "median_misori_mdeg": float(np.median(misos)) if misos else float("nan"),
            "median_strain_err_ue": float(np.median(strains)) if strains else float("nan"),
        })
    return rows


def calibration_sensitivity(
    cfg: XAFConfig,
    grains: GrainPopulation,
    lsd_errors_frac: Sequence[float] = (0.0, 1e-4, 3e-4, 1e-3),
    *,
    n_grains: int = 8,
    seed: int = 1,
) -> List[Dict[str, float]]:
    """Strain bias from a sample--detector-distance calibration error.

    A fractional Lsd error scales all 2theta and imprints as an apparent
    (mostly hydrostatic) strain -- this sets how well Lsd must be calibrated.
    """
    base_lsd = cfg.resolved_Lsd_um()
    twin_cfg = replace(cfg, Lsd_um=base_lsd, n_grains=grains.n_grains)
    fwd_twin = XAFForwardModel(twin_cfg)
    measured = synth.make_measured_spots(twin_cfg, grains, fwd=fwd_twin,
                                         seed=seed)["spots"]
    rng = np.random.default_rng(seed + 9)
    dtype = fwd_twin._latc0.dtype
    rows = []
    for ef in lsd_errors_frac:
        fwd_ref = XAFForwardModel(replace(cfg, Lsd_um=base_lsd * (1.0 + ef),
                                          n_grains=grains.n_grains))
        biases, errs = [], []
        for gi in range(min(n_grains, grains.n_grains)):
            e_true = grains.euler[gi].cpu().numpy()
            e_seed = e_true + rng.normal(scale=np.radians(0.05), size=3)
            params, _ = refine_from_measured(fwd_ref, e_seed, np.zeros(3), measured)
            if params is None:
                continue
            s = params[6:12].cpu().numpy()
            hydro = (s[0] + s[3] + s[5]) / 3.0        # mean of e11,e22,e33
            biases.append(hydro * 1e6)
            errs.append(float(((params[6:12] - grains.strain[gi].to(dtype))
                               .pow(2).mean().sqrt()) * 1e6))
        rows.append({
            "lsd_error_frac": float(ef),
            "median_hydrostatic_bias_ue": float(np.median(biases)) if biases else float("nan"),
            "median_strain_err_ue": float(np.median(errs)) if errs else float("nan"),
        })
    return rows
