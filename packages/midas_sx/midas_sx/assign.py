"""hkl assignment of observed spots, given an orientation.

``q_sample = U B h`` with B the crystallographic reciprocal basis (``|B h| = 1/d``, no 2π) from
``midas_hkls.Lattice`` and U the orientation (sample frame -> crystal-fixed Cartesian). A spot's
q_sample comes from its direction and ω through ``midas_defect.geometry.qlab_to_qsample`` (the MIDAS
rotation convention), so the sign and wedge conventions are the production ones.

The residual ``max |h_float - h_int|`` is the diagnostic that matters: a wrong ω sign, a flipped
detector axis or a wrong beam centre all show up as a collapse of the fraction of spots with a small
residual, not as a quiet bias.
"""
from __future__ import annotations

import numpy as np

from midas_hkls import Lattice

__all__ = ["reciprocal_basis", "q_sample_of_spots", "assign_hkl", "qsample_of_hkl", "calculated_crossing", "ewald_crossings"]


def ewald_crossings(q_s: np.ndarray, wavelength_A: float, wedge_deg: float = 0.0) -> np.ndarray:
    """Omega [rad] at which a sample-frame q (WITH the 2 pi, shape (3,)) crosses the Ewald sphere, for ``q_lab = R_y(-W) R_z(omega) q_sample`` (W = ``wedge_deg``, the MIDAS wedge: the rotation
    axis tilted by W about the lab y axis). ``W = 0`` is ``midas_defect.geometry.ewald_crossing_omegas`` itself (bit-identical); otherwise the same equation solved analytically:
    cos W (q0 cos w - q1 sin w) - sin W q2 = -|q|^2 / (2k), k = 2 pi / lambda. Returns 0 or 2 solutions."""
    if not wedge_deg:
        from midas_defect.geometry import ewald_crossing_omegas
        return ewald_crossing_omegas(q_s, wavelength_A)
    q = np.asarray(q_s, float)
    W = np.radians(wedge_deg)
    k = 2.0 * np.pi / wavelength_A
    A, B, C = np.cos(W) * q[0], -np.cos(W) * q[1], -float(q @ q) / (2.0 * k) + np.sin(W) * q[2]
    R = float(np.hypot(A, B))
    if R == 0.0 or abs(C) > R:
        return np.empty(0)
    phi, d = np.arctan2(B, A), np.arccos(C / R)
    return np.array([phi + d, phi - d])


def reciprocal_basis(lattice: Lattice) -> np.ndarray:
    """B with ``q_cartesian = B @ hkl`` (1/Å, no 2π); columns are the reciprocal vectors."""
    return lattice.reciprocal_cartesian_vectors().T


def q_sample_of_spots(s: np.ndarray, omega_deg, wavelength_A: float, wedge_deg: float = 0.0) -> np.ndarray:
    """Sample-frame q [1/Å, no 2π] of spots with lab directions ``s`` (N, 3) observed at ω [deg]."""
    import torch
    from midas_defect.geometry import qlab_to_qsample
    k0 = 2.0 * np.pi / wavelength_A
    q_lab = k0 * (np.asarray(s, float) - np.array([1.0, 0.0, 0.0]))
    q = qlab_to_qsample(torch.as_tensor(q_lab, dtype=torch.float64),
                        torch.as_tensor(np.radians(np.asarray(omega_deg, float)), dtype=torch.float64),
                        np.radians(wedge_deg))
    return q.numpy() / (2.0 * np.pi)


def qsample_of_hkl(hkl, lattice: Lattice, U: np.ndarray) -> np.ndarray:
    """Sample-frame q [1/Å, no 2π] of reflections ``hkl`` (N, 3) for orientation U: ``U B h``."""
    return (np.asarray(U, float) @ reciprocal_basis(lattice) @ np.asarray(hkl, float).T).T


def assign_hkl(s: np.ndarray, omega_deg, lattice: Lattice, U: np.ndarray, wavelength_A: float,
               wedge_deg: float = 0.0):
    """Nearest-integer hkl for each spot.

    Returns ``(hkl_int (N,3) int, hkl_float (N,3), residual (N,))`` with
    ``residual = max_i |h_float_i - h_int_i|``.
    """
    q = q_sample_of_spots(s, omega_deg, wavelength_A, wedge_deg)
    M = np.asarray(U, float) @ reciprocal_basis(lattice)
    h = np.linalg.solve(M, q.T).T
    hi = np.rint(h).astype(int)
    return hi, h, np.max(np.abs(h - hi), axis=1)


def calculated_crossing(q_sample: np.ndarray, omega_deg: float, wavelength_A: float, *, window_deg: float = 2.0,
                        step_deg: float = 0.02, wedge_deg: float = 0.0):
    """Ewald crossing of a reflection with sample-frame ``q_sample`` (1/A, no 2 pi) nearest to ``omega_deg``.

    Solves ``f(w) = |k s0 + R_z(w) q| - k = 0`` (k = 1/lambda, s0 = +X, right-handed R_z as ``midas_defect``) by a scan over
    ``omega_deg +- window_deg`` and a bracketed root polish. Returns ``(omega_star_deg, s_star)`` with ``s_star`` the unit diffracted
    direction AT the crossing, or ``None`` when no sign change lies inside the window.

    The corrections (Lorentz, polarization, absorption) belong to THIS direction. The measured centroid of an extended or skewed spot
    is not it: measured on the simulated S2 spots the two Lorentz factors differ by 0.57 % rms (max 3.8 %), which moved refined
    parameters by up to 1.3 sigma (PREREGISTER A15).
    """
    from scipy.optimize import brentq
    k = 1.0 / wavelength_A
    q = np.asarray(q_sample, float)
    Wr = np.radians(wedge_deg)
    cw, sw = np.cos(Wr), np.sin(Wr)

    def lab(w):
        c, sn = np.cos(np.radians(w)), np.sin(np.radians(w))
        x, y, z = c * q[0] - sn * q[1], sn * q[0] + c * q[1], q[2]
        return np.array([cw * x - sw * z, y, sw * x + cw * z]) if wedge_deg else np.array([x, y, z])        # R_y(-W) R_z(w) q

    def f(w):
        v = lab(w) + np.array([k, 0.0, 0.0])
        return float(np.linalg.norm(v) - k)

    grid = np.arange(omega_deg - window_deg, omega_deg + window_deg + 0.5 * step_deg, step_deg)
    vals = np.array([f(g) for g in grid])
    # a bracket: strictly negative on one side and >= 0 (or the reverse) on the other. A strict ``product < 0`` MISSES a root that sits exactly on a
    # grid node (|v| - k rounds to exactly 0.0 when the start omega IS the crossing, as for a predicted target: 7 of 46 in the A39-FLOOR simulator check).
    idx = np.flatnonzero(((vals[:-1] < 0) & (vals[1:] >= 0)) | ((vals[:-1] > 0) & (vals[1:] <= 0)))
    if idx.size == 0:
        return None
    j = idx[np.argmin(np.abs(0.5 * (grid[idx] + grid[idx + 1]) - omega_deg))]
    w = brentq(f, grid[j], grid[j + 1], xtol=1e-12)
    v = lab(w) + np.array([k, 0.0, 0.0])
    return float(w), v / np.linalg.norm(v)
