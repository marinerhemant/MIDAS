"""How much ω does an experiment actually need? A number, not an opinion.

A narrow rotation wedge is the binding constraint on any single-crystal
measurement made through a constrained aperture — a diamond-anvil cell, a
cryostat with limited access, a fly scan cut short to save beamtime. When each
grain puts only a handful of reflections into the window, every route to a
lattice-parameter difference fails, and it fails for a reason that no amount of
counting statistics repairs.

This module turns "we need more ω" into a table the beamline can act on: for a
given half-range, how many reflections a grain gives, how many of them can
**see** the quantity of interest, and what precision that buys.

Why the sensitive count is reported separately
----------------------------------------------
Reflections with h = k are blind to an a≠b splitting: swapping a and b leaves
their d-spacing unchanged. They still constrain the *orientation*, and so they
still help, but they carry no direct information about the splitting. A budget
quoted as a raw reflection count therefore overstates what the measurement can
do. Both numbers are returned.

Why the precision is Fisher and not 1/sqrt(N)
---------------------------------------------
The cell and the orientation are fitted **together**, so the three rotation
angles and the two lattice parameters compete for the same reflections. With
few reflections the orientation soaks up much of the signal and σ(δ) is far
worse than a naive count suggests. :func:`sigma_delta` builds the Jacobian of
the full parameter set and propagates the measured per-reflection error, which
is the honest calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["RotationBudget", "random_orientations", "accepted_reflections",
           "sigma_delta", "rotation_budget"]


def random_orientations(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` rotation matrices, uniform on SO(3) (unit quaternions)."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def _b_matrix(a: float, b: float, c: float) -> np.ndarray:
    return np.diag([1.0 / a, 1.0 / b, 1.0 / c])


def accepted_reflections(hkl: np.ndarray, U: np.ndarray, B: np.ndarray, *,
                         half_range_deg: float,
                         wavelength_A: float,
                         lsd_um: float,
                         pixel_um: float,
                         det_rows: int,
                         det_cols: int,
                         beam_row: float,
                         beam_col: float,
                         beamstop_px: float = 0.0) -> np.ndarray:
    """Boolean mask: which reflections reach Bragg in ±half_range AND land.

    Bragg is solved exactly for both ω roots, rotating about the vertical axis.
    A reflection counts only if some root falls in the window, the diffracted
    beam goes forward, and the spot lands on the active detector outside the
    beamstop.
    """
    hkl = np.asarray(hkl, float)
    Gs = (hkl @ np.asarray(B).T) @ np.asarray(U).T
    gn = np.linalg.norm(Gs, axis=1)
    D = -wavelength_A * gn ** 2 / 2.0
    A_, Bc = Gs[:, 0], -Gs[:, 1]
    R = np.hypot(A_, Bc)
    feasible = (R > 1e-12) & (np.abs(D) <= R) & (gn > 1e-9)

    keep = np.zeros(len(hkl), bool)
    for i in np.flatnonzero(feasible):
        base = np.arctan2(Bc[i], A_[i])
        dw = np.arccos(np.clip(D[i] / R[i], -1.0, 1.0))
        for w in (base + dw, base - dw):
            ome = (np.degrees(w) + 180.0) % 360.0 - 180.0
            if abs(ome) > half_range_deg:
                continue
            cw, sw = np.cos(w), np.sin(w)
            gl = np.array([cw * Gs[i, 0] - sw * Gs[i, 1],
                           sw * Gs[i, 0] + cw * Gs[i, 1], Gs[i, 2]])
            kf_x = 1.0 / wavelength_A + gl[0]
            if kf_x <= 0:
                continue
            t = lsd_um / kf_x
            col = gl[1] * t / pixel_um + beam_col
            row = gl[2] * t / pixel_um + beam_row
            if not (0 <= row < det_rows and 0 <= col < det_cols):
                continue
            if np.hypot(row - beam_row, col - beam_col) < beamstop_px:
                continue
            keep[i] = True
            break
    return keep


def sigma_delta(hkl_accepted: np.ndarray, U: np.ndarray, *,
                a: float, b: float, c: float,
                sigma_g_inv_A: float) -> float:
    """Fisher σ on δ = (a−b)/(a+b), in **percent**, fitting 3 rotations + a + b.

    Returns ``inf`` when the accepted set cannot constrain the parameters.
    """
    hkl_accepted = np.asarray(hkl_accepted, float)
    n = len(hkl_accepted)
    if n < 6:
        return float("inf")
    eps = 1e-6

    def predict(p):
        rx, ry, rz, aa, bb = p
        W = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
        R = np.eye(3) + W + 0.5 * W @ W
        return ((hkl_accepted @ _b_matrix(aa, bb, c).T) @ (R @ U).T).ravel()

    p0 = np.array([0.0, 0.0, 0.0, a, b])
    f0 = predict(p0)
    J = np.empty((f0.size, 5))
    for j in range(5):
        pp = p0.copy()
        pp[j] += eps
        J[:, j] = (predict(pp) - f0) / eps
    try:
        cov = np.linalg.inv(J.T @ J) * sigma_g_inv_A ** 2
    except np.linalg.LinAlgError:
        return float("inf")
    s = a + b
    grad = np.array([0.0, 0.0, 0.0, 2 * b / s ** 2, -2 * a / s ** 2])
    return float(np.sqrt(max(grad @ cov @ grad, 0.0)) * 100.0)


@dataclass
class RotationBudget:
    """One row per ω half-range, plus the orientations it was averaged over."""
    table: pd.DataFrame
    n_orientations: int
    target_delta_pct: float

    def required_half_range(self, fraction: float = 0.95) -> Optional[float]:
        """Smallest tabulated half-range reaching the target for ``fraction``."""
        ok = self.table[self.table["frac_reaching_target"] >= fraction]
        return float(ok["half_range_deg"].iloc[0]) if len(ok) else None

    def __str__(self) -> str:
        return self.table.to_string(index=False)


def rotation_budget(hkl: np.ndarray, *,
                    a: float, b: float, c: float,
                    wavelength_A: float, lsd_um: float, pixel_um: float,
                    det_rows: int, det_cols: int,
                    beam_row: float, beam_col: float,
                    half_ranges_deg: Sequence[float] = (6, 15, 22.5, 30),
                    sigma_g_inv_A: float = 0.0026,
                    target_delta_pct: float = 0.1,
                    beamstop_px: float = 0.0,
                    n_orientations: int = 200,
                    sensitive: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                    rng_seed: int = 17) -> RotationBudget:
    """Forward-model the reflection and precision budget versus ω range.

    ``half_ranges_deg`` are HALF ranges: a ±22.5° entry is a 45° scan.

    ``sensitive`` maps an (n, 3) hkl array to a boolean array marking the
    reflections that carry information about the quantity being measured.
    The default marks ``h != k``, which is the a/b-splitting case; pass your own
    for a different question.

    ``sigma_g_inv_A`` must be **measured** on comparable data — it is the
    per-reflection rms \\|dG\\|, and the whole table scales with it.
    """
    hkl = np.asarray(hkl)
    if sensitive is None:
        def sensitive(h):
            return h[:, 0] != h[:, 1]

    rng = np.random.default_rng(rng_seed)
    Us = random_orientations(n_orientations, rng)
    B = _b_matrix(a, b, c)

    rows = []
    for half in half_ranges_deg:
        n_ref, n_sens, sig = [], [], []
        for U in Us:
            keep = accepted_reflections(
                hkl, U, B, half_range_deg=float(half),
                wavelength_A=wavelength_A, lsd_um=lsd_um, pixel_um=pixel_um,
                det_rows=det_rows, det_cols=det_cols,
                beam_row=beam_row, beam_col=beam_col, beamstop_px=beamstop_px)
            acc = hkl[keep]
            n_ref.append(int(keep.sum()))
            n_sens.append(int(sensitive(np.asarray(acc)).sum()) if len(acc) else 0)
            sig.append(sigma_delta(acc, U, a=a, b=b, c=c,
                                   sigma_g_inv_A=sigma_g_inv_A))
        sig = np.asarray(sig)
        rows.append({
            "half_range_deg": float(half),
            "full_range_deg": 2.0 * float(half),
            "reflections_per_grain": float(np.median(n_ref)),
            "sensitive_per_grain": float(np.median(n_sens)),
            "sigma_delta_pct": float(np.median(sig)),
            "frac_reaching_target": float(np.mean(sig <= target_delta_pct)),
        })
    return RotationBudget(table=pd.DataFrame(rows),
                          n_orientations=n_orientations,
                          target_delta_pct=target_delta_pct)
