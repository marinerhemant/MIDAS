"""Circle and multi-ring (BC, Lsd) fits for the manual ring-picker GUI.

Three routines, increasing in cost and accuracy:

  * ``kasa_circle_fit``       — algebraic least-squares (closed form,
    minimises algebraic distance). Cheap, slightly biased on partial
    arcs, but always returns *something*. Used as a seed.

  * ``geometric_lm_refine``   — Levenberg-Marquardt on the *geometric*
    residual r_i - R. Sub-pixel even on noisy partial arcs.

  * ``joint_bc_lsd_fit``      — simultaneous (BC, Lsd) fit across
    multiple rings whose 2θ values are known a-priori, locking the
    centre to a single (y, z) and solving for one Lsd that explains
    every ring radius.

All three are pure-numpy, no GUI dependencies, importable on their own,
and covered by the regression suite in ``tests/test_gui_circle_fit.py``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Algebraic Kåsa fit
# ---------------------------------------------------------------------------

def kasa_circle_fit(
    xs: Sequence[float],
    ys: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Algebraic (Kåsa) least-squares circle fit.

    Solves ``2 cx x + 2 cy y + (R² - cx² - cy²) = x² + y²`` linearly
    for (cx, cy, c0) and returns ``cx, cy, R`` with the geometric RMS
    residual.  Needs at least 3 non-collinear points.

    Returns
    -------
    (cx, cy, R, rms) : tuple of floats
        Centre coordinates, radius, and RMS of ``r_i − R``.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3:
        raise ValueError("kasa_circle_fit needs at least 3 points")
    A = np.column_stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = sol
    r2 = c0 + cx ** 2 + cy ** 2
    if r2 <= 0:
        raise ValueError("kasa_circle_fit: negative R² (points likely collinear)")
    R = float(np.sqrt(r2))
    rms = float(np.sqrt(np.mean((np.hypot(xs - cx, ys - cy) - R) ** 2)))
    return float(cx), float(cy), R, rms


# ---------------------------------------------------------------------------
# 2. Geometric Levenberg-Marquardt refinement
# ---------------------------------------------------------------------------

def geometric_lm_refine(
    xs: Sequence[float],
    ys: Sequence[float],
    cx0: float,
    cy0: float,
    R0: float,
    *,
    max_iter: int = 50,
    tol: float = 1e-9,
    damping_init: float = 1e-3,
) -> Tuple[float, float, float, float]:
    """LM iterations on the geometric residual ``r_i − R``.

    Algebraic Kåsa minimises (r_i² − R²); the bias of that residual
    scales with the radius itself, which matters on partial arcs.
    Geometric LM minimises (r_i − R) directly — one Gauss-Newton step
    typically halves the RMS, two more push it to machine precision
    on a clean circle.

    Parameters
    ----------
    cx0, cy0, R0 : initial guess (typically from :func:`kasa_circle_fit`).
    max_iter     : safety cap; convergence is usually <10 iterations.
    tol          : stop when ``‖Δp‖∞`` falls below this.
    damping_init : initial Levenberg damping; auto-adjusted each step.

    Returns
    -------
    (cx, cy, R, rms) : refined parameters and the final RMS residual.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.size < 3:
        raise ValueError("geometric_lm_refine needs at least 3 points")
    cx, cy, R = float(cx0), float(cy0), float(R0)
    lam = float(damping_init)
    prev_cost = np.inf

    for _ in range(max_iter):
        dx = xs - cx
        dy = ys - cy
        r = np.hypot(dx, dy)
        # Guard against r==0 (point right at centre) — would give NaN partials.
        r = np.where(r < 1e-12, 1e-12, r)
        residual = r - R                                    # shape (N,)
        cost = float(np.sum(residual * residual))

        # Jacobian w.r.t. (cx, cy, R)
        # ∂(r − R)/∂cx = −dx/r,  ∂/∂cy = −dy/r,  ∂/∂R = −1
        J = np.column_stack([-dx / r, -dy / r, -np.ones_like(r)])
        JtJ = J.T @ J
        Jtr = J.T @ residual

        # Levenberg damping
        diag = np.diag(np.diag(JtJ))
        try:
            step = np.linalg.solve(JtJ + lam * diag, -Jtr)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        cx_new, cy_new, R_new = cx + step[0], cy + step[1], R + step[2]
        new_residual = np.hypot(xs - cx_new, ys - cy_new) - R_new
        new_cost = float(np.sum(new_residual * new_residual))

        if new_cost < cost:
            cx, cy, R = cx_new, cy_new, R_new
            lam = max(lam / 3.0, 1e-12)
            if abs(prev_cost - new_cost) / max(prev_cost, 1e-30) < tol:
                prev_cost = new_cost
                break
            prev_cost = new_cost
        else:
            lam *= 5.0
            if lam > 1e10:
                break

    rms = float(np.sqrt(np.mean(
        (np.hypot(xs - cx, ys - cy) - R) ** 2)))
    return float(cx), float(cy), float(R), rms


# ---------------------------------------------------------------------------
# 3. Joint multi-ring (BC, Lsd) fit
# ---------------------------------------------------------------------------

def joint_bc_lsd_fit(
    rings: Sequence[Tuple[Sequence[float], Sequence[float], float]],
    pixel_size_um: float,
    *,
    cx0: Optional[float] = None,
    cy0: Optional[float] = None,
    lsd0_um: Optional[float] = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    damping_init: float = 1e-3,
) -> dict:
    """Joint LM fit of ``(BC_y, BC_z, Lsd)`` across multiple picked rings.

    The forward model:

        R_k_pred = Lsd / pixel_size · tan(2θ_k)

    so for every picked point (x_i, y_i) on ring ``k`` with known
    ``2θ_k``,

        residual_i = √((x_i − BC_y)² + (y_i − BC_z)²) − R_k_pred(Lsd).

    Locking the centre across rings drives the BC several × tighter
    than a per-ring fit, and the algebraic-Lsd estimate from each
    ring averages out into a single sub-px / sub-mm number.

    Parameters
    ----------
    rings : sequence of ``(xs_k, ys_k, two_theta_rad_k)`` tuples
        At least one ring with ≥3 points. Mixing different ring
        cardinalities is fine.
    pixel_size_um : detector pixel size in µm.
    cx0, cy0, lsd0_um : optional initial guesses. If omitted, the
        routine seeds from a per-ring Kåsa fit (median centre,
        median per-ring Lsd from ``R_k_kasa · px / tan(2θ_k)``).

    Returns
    -------
    dict with keys
        cx, cy, lsd_um   — refined parameters
        rms_total_px     — RMS over all points, all rings
        rms_per_ring_px  — list of per-ring RMS
        n_total          — total number of points
        n_rings          — number of rings
        per_ring_R_pred  — predicted R for each ring at the fit Lsd
    """
    if not rings:
        raise ValueError("joint_bc_lsd_fit: at least one ring required")
    cleaned = []
    for k, (xs, ys, tth) in enumerate(rings):
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if xs.size != ys.size:
            raise ValueError(f"ring {k}: len(xs) != len(ys)")
        if xs.size < 3:
            raise ValueError(f"ring {k}: needs ≥3 points (got {xs.size})")
        if tth <= 0:
            raise ValueError(f"ring {k}: two_theta must be > 0, got {tth}")
        cleaned.append((xs, ys, float(tth)))

    # ----------------------- seed (cx, cy, lsd) ---------------------
    if cx0 is None or cy0 is None or lsd0_um is None:
        per_ring_seeds = []
        for xs, ys, tth in cleaned:
            cx_k, cy_k, R_k, _ = kasa_circle_fit(xs, ys)
            lsd_k = R_k * pixel_size_um / np.tan(tth)
            per_ring_seeds.append((cx_k, cy_k, lsd_k))
        seeds = np.asarray(per_ring_seeds, dtype=np.float64)
        if cx0 is None:
            cx0 = float(np.median(seeds[:, 0]))
        if cy0 is None:
            cy0 = float(np.median(seeds[:, 1]))
        if lsd0_um is None:
            lsd0_um = float(np.median(seeds[:, 2]))

    cx, cy, lsd = float(cx0), float(cy0), float(lsd0_um)
    lam = float(damping_init)
    prev_cost = np.inf

    def _resid_and_jac(cx, cy, lsd):
        """Build a stacked residual + Jacobian over all rings."""
        rows_resid = []
        rows_J = []
        for xs, ys, tth in cleaned:
            dx = xs - cx
            dy = ys - cy
            r = np.hypot(dx, dy)
            r = np.where(r < 1e-12, 1e-12, r)
            R_pred = lsd / pixel_size_um * np.tan(tth)
            res = r - R_pred                                # (N_k,)
            # ∂r/∂cx = −dx/r,  ∂r/∂cy = −dy/r,
            # ∂R_pred/∂lsd = tan(tth)/pixel_size  (R_pred indep. of cx, cy)
            j_cx = -dx / r
            j_cy = -dy / r
            j_lsd = -np.full_like(r, np.tan(tth) / pixel_size_um)
            rows_resid.append(res)
            rows_J.append(np.column_stack([j_cx, j_cy, j_lsd]))
        return np.concatenate(rows_resid), np.vstack(rows_J)

    for _ in range(max_iter):
        residual, J = _resid_and_jac(cx, cy, lsd)
        cost = float(np.sum(residual * residual))
        JtJ = J.T @ J
        Jtr = J.T @ residual
        diag = np.diag(np.diag(JtJ))
        try:
            step = np.linalg.solve(JtJ + lam * diag, -Jtr)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        cx_new, cy_new, lsd_new = cx + step[0], cy + step[1], lsd + step[2]
        if lsd_new <= 0:
            # never let Lsd flip sign; back off
            lam *= 5.0
            continue
        new_residual, _ = _resid_and_jac(cx_new, cy_new, lsd_new)
        new_cost = float(np.sum(new_residual * new_residual))

        if new_cost < cost:
            cx, cy, lsd = cx_new, cy_new, lsd_new
            lam = max(lam / 3.0, 1e-12)
            if abs(prev_cost - new_cost) / max(prev_cost, 1e-30) < tol:
                prev_cost = new_cost
                break
            prev_cost = new_cost
        else:
            lam *= 5.0
            if lam > 1e10:
                break

    # ----------------------- per-ring diagnostics -------------------
    per_ring_rms = []
    per_ring_R_pred = []
    n_total = 0
    for xs, ys, tth in cleaned:
        r = np.hypot(xs - cx, ys - cy)
        R_pred = lsd / pixel_size_um * np.tan(tth)
        per_ring_rms.append(float(np.sqrt(np.mean((r - R_pred) ** 2))))
        per_ring_R_pred.append(float(R_pred))
        n_total += xs.size
    full_residual, _ = _resid_and_jac(cx, cy, lsd)
    rms_total = float(np.sqrt(np.mean(full_residual * full_residual)))

    return {
        "cx": float(cx),
        "cy": float(cy),
        "lsd_um": float(lsd),
        "rms_total_px": rms_total,
        "rms_per_ring_px": per_ring_rms,
        "per_ring_R_pred_px": per_ring_R_pred,
        "n_total": int(n_total),
        "n_rings": len(cleaned),
    }


__all__ = [
    "kasa_circle_fit",
    "geometric_lm_refine",
    "joint_bc_lsd_fit",
]
