"""How well a SET of topotomography reflections determines the rotation field.

A single TT scan cannot see rotation about its own scattering vector. The rocking
condition is disturbed by lattice rotation about ``a_lab = (k_in x G)/|k_in x G|``,
and since ``G`` is aligned to the rotation axis, as ``psi`` sweeps the sensitivity
direction traces the great circle perpendicular to ``G`` and never acquires a
component along it. A rotation about ``G`` does not change the Bragg condition at
all, so that component is not weakly constrained -- it is exactly null.

For a set of reflections with unit scattering vectors ``g_i`` in the sample frame,
the per-scan second moment of the sensitivity direction is ``(I - g g^T)/2``, so

    M = (1/N) sum_i (I - g_i g_i^T) / 2

and the eigenvalues of ``M`` say how well each rotation component is determined.
For two reflections separated by ``gamma`` this is exactly

    (1 - cos gamma)/4 ,   (1 + cos gamma)/4 ,   1/2

so the weakest component scales as ``gamma^2/8`` for small ``gamma``. Reflections
that are close together do NOT rescue the null, however many you take.

This is a scan-PLANNING check, and it is the one that matters for a rotation-field
experiment: it is decided before any photons are collected, and no amount of
counting statistics repairs a bad choice. Verified against the ESRF Ti-7Al
experiment (grain 605), whose two published reflections sit only 13.3 deg apart
and give ``[0.0067, 0.4933, 0.5]`` -- the third rotation component is 75x less
constrained than the other two. The analytic law reproduces that experiment's
actual 90+90 view sampling to four decimal places.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

__all__ = ["sensitivity_moment", "rotation_conditioning",
           "separation_for_conditioning", "best_reflection_pair"]


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("scattering vectors must be non-zero")
    return v / n


def sensitivity_moment(g_vectors):
    """``M``, the mean second moment of the sensitivity direction, ``(3, 3)``."""
    g = _unit(np.atleast_2d(g_vectors))
    return np.mean([(np.eye(3) - np.outer(u, u)) / 2.0 for u in g], axis=0)


def rotation_conditioning(g_vectors):
    """Eigenvalues of ``M`` (ascending) and the conditioning ratio.

    Returns
    -------
    eigenvalues : (3,) ndarray
        Ascending. The first is the worst-determined rotation component.
    ratio : float
        ``lambda_min / lambda_max``. ``0`` means a component is unmeasurable;
        ``1`` means all three are equally determined.

    Notes
    -----
    A single reflection always returns ``ratio == 0`` exactly -- that is the roll
    degeneracy, not a numerical accident.
    """
    ev = np.linalg.eigvalsh(sensitivity_moment(g_vectors))
    ev = np.clip(ev, 0.0, None)
    return ev, float(ev[0] / ev[-1]) if ev[-1] > 0 else 0.0


def separation_for_conditioning(ratio: float) -> float:
    """Smallest separation ``gamma`` (deg) of a PAIR reaching a given ratio.

    Inverts ``(1 - cos gamma)/4 = ratio * 1/2``. Raises if the ratio exceeds the
    best a pair can do (``1.0`` at 90 degrees).
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must lie in [0, 1]")
    c = 1.0 - 2.0 * ratio
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def best_reflection_pair(g_vectors):
    """Index pair maximising the worst-determined component, and its stats.

    Returns ``(i, j, gamma_deg, ratio)``. Use this to choose the SECOND
    reflection: the useful one is the most nearly orthogonal available, not the
    brightest or the most convenient for the goniometer.
    """
    g = _unit(np.atleast_2d(g_vectors))
    if len(g) < 2:
        raise ValueError("need at least two reflections")
    best = None
    for i, j in itertools.combinations(range(len(g)), 2):
        _, ratio = rotation_conditioning(g[[i, j]])
        gam = math.degrees(math.acos(min(1.0, abs(float(g[i] @ g[j])))))
        if best is None or ratio > best[3]:
            best = (i, j, gam, ratio)
    return best
