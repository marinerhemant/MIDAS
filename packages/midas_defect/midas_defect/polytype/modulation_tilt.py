"""Quantify the twin-polarity doublet: modulation tilt (split vs order) + Sigma3 landing.

Two geometry-only diagnostics on a resolved satellite doublet (see
:mod:`midas_defect.polytype.satellite_doublet`):

* :func:`fit_modulation_tilt` -- the transverse split Delta_perp(n) between the two
  members grows linearly with satellite order n. The slope is a *modulation tilt*:
  each polarity's modulation wavevector is tilted by +-beta from the exact <111>, so
  Delta_perp = 2 n (G/3) tan(beta). A split that grows with order is the signature of
  a real modulation (two q_mod directions), NOT a static Ewald/relrod asymmetry
  (which is order-independent) and NOT a lattice misorientation (which would split
  the *fundamentals* most, by |q|).

* :func:`sigma3_landing_residual` -- apply the Sigma3 twin operation (180 deg about
  the shared <111>, which for cubic == 60 deg about <111> modulo the 3-fold) to one
  member's q-vector and measure how close it lands on the other. A residual of ~1
  voxel confirms the two members are twin-related reflections.

HONESTY. These confirm the two members are *twin-related polarity variants*; they do
NOT assign parent-grain vs twin-lamella identity (a spatial/volume statement FF
cannot make -- see :mod:`midas_defect.attribution`).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["fit_modulation_tilt", "sigma3_landing_residual"]


def _rot_about(axis: NDArray[np.floating], deg: float) -> NDArray[np.floating]:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(a, a)


def fit_modulation_tilt(
    orders: NDArray[np.floating],
    splits_inv_A: NDArray[np.floating],
    g3_inv_A: float,
) -> dict:
    """Fit Delta_perp = slope * |n| (through origin) -> modulation tilt beta.

    Parameters
    ----------
    orders : (k,) satellite orders n (sign ignored; |n| used).
    splits_inv_A : (k,) measured transverse split |Delta_perp| at each order.
    g3_inv_A : the satellite spacing G/3 (1/A).

    Returns
    -------
    dict with beta_deg, slope_inv_A_per_order, r2 (through-origin), n_points.
    beta from slope = 2 (G/3) tan(beta)  =>  beta = atan(slope / (2 G/3)).
    """
    n = np.abs(np.asarray(orders, dtype=np.float64))
    y = np.asarray(splits_inv_A, dtype=np.float64)
    ok = np.isfinite(n) & np.isfinite(y) & (y >= 0)
    n, y = n[ok], y[ok]
    if n.size < 2 or np.allclose(n, 0):
        return {"beta_deg": float("nan"), "slope_inv_A_per_order": float("nan"),
                "r2": float("nan"), "n_points": int(n.size)}
    slope = float(np.sum(n * y) / np.sum(n * n))      # least squares through origin
    resid = y - slope * n
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    beta = math.degrees(math.atan(slope / (2.0 * g3_inv_A))) if g3_inv_A > 0 else float("nan")
    return {"beta_deg": beta, "slope_inv_A_per_order": slope, "r2": r2,
            "n_points": int(n.size)}


def sigma3_landing_residual(
    member_a_q: NDArray[np.floating],
    member_b_q: NDArray[np.floating],
    axis_dir: NDArray[np.floating],
    *,
    voxel_inv_A: float = 0.013,
) -> dict:
    """Apply the Sigma3 (180 deg about ``axis_dir``) to A and measure landing on B.

    Returns residual_inv_A (|R180 A - B|), raw_inv_A (|A - B|), improvement
    (raw/residual), residual_voxels (residual / ``voxel_inv_A``), and is_twin_mapped
    (residual < 0.5 * raw and within a few voxels). The 180-about-<111> operator is
    the cubic symmetry-equivalent of the 60-deg Sigma3 twin for axial features.
    """
    a = np.asarray(member_a_q, dtype=np.float64)
    b = np.asarray(member_b_q, dtype=np.float64)
    R = _rot_about(axis_dir, 180.0)
    pred = R @ a
    residual = float(np.linalg.norm(pred - b))
    raw = float(np.linalg.norm(a - b))
    improvement = raw / residual if residual > 0 else float("inf")
    return {
        "residual_inv_A": residual,
        "raw_inv_A": raw,
        "improvement": improvement,
        "residual_voxels": residual / voxel_inv_A if voxel_inv_A > 0 else float("nan"),
        "is_twin_mapped": bool(residual < 0.5 * raw and residual < 5 * voxel_inv_A),
    }
