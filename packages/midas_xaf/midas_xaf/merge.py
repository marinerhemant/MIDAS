"""Registration of the two mountings into one common frame.

Two independent handles recover the rigid remount transform ``R`` (and
translation ``t``):

1. **Fiducials** -- 2-3 embedded high-Z markers, localised in each mounting,
   solved by Kabsch/Procrustes.  This is the primary handle the user embeds for
   exactly this purpose ("track them when we re-mount").  Three non-collinear
   markers are the minimum (two leave the rotation about their axis free).
2. **Shared-face grains** -- grains seen in both mountings give an independent
   over-determined check (not required, but validates rigidity).

Because the sample is rigid, the residual after registration measures how well
the merge holds -- i.e. the noise floor the merged reconstruction inherits.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import XAFConfig
from .forward import XAFForwardModel
from .sample import GrainPopulation
from . import geometry as geo


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple:
    """Rigid transform (R, t) minimising ||Q - (R P + t)|| (proper rotation)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t


def _rot_angle_deg(dR: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))


def register_fiducials(
    true_R: np.ndarray,
    fiducial_positions: np.ndarray,   # (F, 3) mounting-1 positions (um)
    sigma_um: float,
    *,
    seed: int = 0,
    n_use: Optional[int] = None,
) -> Dict[str, float]:
    """Recover the remount rotation from noisy fiducial positions via Kabsch.

    ``sigma_um`` is the per-marker localisation precision (from
    :func:`midas_xaf.metrics.position_localization`: box-Friedel or scanning).
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(fiducial_positions, float)
    if n_use is not None:
        P = P[:n_use]
    Q = (true_R @ P.T).T
    Pn = P + rng.normal(scale=sigma_um, size=P.shape)
    Qn = Q + rng.normal(scale=sigma_um, size=Q.shape)
    R_est, t_est = kabsch(Pn, Qn)
    return {
        "n_fiducials": P.shape[0],
        "sigma_um": sigma_um,
        "angle_error_deg": _rot_angle_deg(R_est @ true_R.T),
        "t_error_um": float(np.linalg.norm(t_est)),
        "degenerate": P.shape[0] < 3,
    }


def fiducial_registration_study(
    fwd: XAFForwardModel,
    grains: GrainPopulation,
    *,
    sigma_um: Optional[float] = None,
    n_fiducials_list=(2, 3, 4, 6),
    trials: int = 20,
) -> Dict[str, object]:
    """How remount-registration accuracy scales with #fiducials and localisation.

    Uses the true remount transform and, if ``sigma_um`` is None, the box-Friedel
    position precision from the current geometry as the marker-localisation noise.
    """
    from . import metrics
    true_R = np.asarray(geo.mounting_matrix(fwd.cfg, 1), float)
    if sigma_um is None:
        sigma_um = metrics.position_localization(fwd, grains)["effective_position_um"]

    # A generous pool of random marker positions inside the sample.
    rng = np.random.default_rng(fwd.cfg.seed + 4242)
    pool = rng.normal(size=(max(n_fiducials_list), 3))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    pool *= 0.8 * fwd.cfg.sample_radius_um * np.cbrt(
        rng.uniform(0, 1, size=(pool.shape[0], 1)))

    rows = []
    for nf in n_fiducials_list:
        errs = [register_fiducials(true_R, pool[:nf], sigma_um, seed=1000 + nf * 100 + k
                                   )["angle_error_deg"] for k in range(trials)]
        rows.append({"n_fiducials": nf, "sigma_um": sigma_um,
                     "median_angle_error_deg": float(np.median(errs)),
                     "degenerate": nf < 3})
    return {"sigma_um": sigma_um, "rows": rows}
