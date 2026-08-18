"""Goniometer tilts for topotomography, and what a given stage can reach.

:mod:`midas_dct_tt.rotation_coverage` answers *which pair of reflections would
determine the rotation field* -- a question about the crystal alone. This module
answers the question that decides whether that plan is executable: **can this
goniometer actually get there?** The two together turn a conditioning law into an
experiment.

The tilt solution
-----------------
A TT scan requires ``G`` parallel to the tomographic rotation axis. With the
instrument transformation ``T`` folded out, that is solved by two stage angles::

    Gs = T^T G_sample
    ut = atan(Gs_y / Gs_z)
    lt = atan(-Gs_x / (Gs_y sin(ut) + Gs_z cos(ut)))

matching the ESRF ID11 ``samrx``/``samry`` pair. Validated against 74 real TT
scans of \\citet{stinville2021}: median residual 0.043 deg (samrx) and 0.050 deg
(samry), against a random-grain null of 25-40 deg -- see
``dev/real_data/validate_tilts.py``.

Two degeneracies, both real
---------------------------
1. **Branch.** ``(ut, lt)`` and ``(ut + 180, -lt)`` describe the same alignment,
   because flipping ``ut`` by 180 deg negates the denominator of ``lt``. Since
   ``atan`` returns ``ut`` in ``(-90, 90)``, the sibling branch always lands in
   ``|ut| > 90`` and is therefore *unreachable on any stage with an envelope
   below 90 deg*. :func:`topotomo_tilts` returns the principal branch and
   :func:`tilt_branches` both; reachability tests both regardless, so the
   conclusion never depends on which one a caller happened to look at.
2. **Friedel.** ``G`` and ``-G`` give *identical* tilts (both numerator and
   denominator negate). Enumerating ``(h,k,l)`` and ``(-h,-k,-l)`` separately is
   harmless but redundant.

Orientation input
-----------------
Rodrigues vectors in DCT grain maps written by ``pymicro`` (and the ESRF
deposits built with it) use a convention that is **not**
:func:`midas_stress.orientation.rodrigues_to_orient_mat`; use
:func:`midas_dct_tt.esrf.rodrigues_to_crystal_to_sample` to read them. That
function documents the discrepancy and the evidence for it.

NumPy, not torch, to sit alongside :mod:`~midas_dct_tt.rotation_coverage`: this
is scan planning, evaluated once before an experiment, never inside a fit.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .rotation_coverage import rotation_conditioning

__all__ = [
    "INSTRUMENT_OFFSETS_ID11",
    "best_reachable_pair",
    "instrument_transformation",
    "reachable_reflections",
    "reciprocal_basis",
    "tilt_branches",
    "topotomo_tilts",
]

# ESRF ID11 nano/3DXRD, as used for the Ti-7Al TT campaign (their notebook, and
# the values that reproduce their 74 logged scans to 0.05 deg).
INSTRUMENT_OFFSETS_ID11 = (-1.2, 0.7, 90.0)   # samrx, samry, omega, degrees

_EPS = 1e-12


def _rx(t):
    t = math.radians(t)
    return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)],
                     [0, math.sin(t), math.cos(t)]])


def _ry(t):
    t = math.radians(t)
    return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0],
                     [-math.sin(t), 0, math.cos(t)]])


def _rz(t):
    t = math.radians(t)
    return np.array([[math.cos(t), -math.sin(t), 0],
                     [math.sin(t), math.cos(t), 0], [0, 0, 1]])


def instrument_transformation(samrx_offset, samry_offset, omega_offset):
    """``T = Rz(omega) @ Ry(samry) @ Rx(samrx)`` from stage offsets, in degrees.

    Applied as ``Gs = T.T @ G_sample``. This composition order is the ESRF
    ``compute_instrument_transformation_matrix`` convention; it is *not*
    symmetric, so the argument order matters.
    """
    return _rz(omega_offset) @ _ry(samry_offset) @ _rx(samrx_offset)


def reciprocal_basis(a_A, b_A, c_A, alpha=90.0, beta=90.0, gamma=90.0):
    """Reciprocal basis as a COLUMN operator: ``G_crystal = B @ [h, k, l]``.

    Delegates to :meth:`midas_hkls.lattice.Lattice.reciprocal_cartesian_vectors`
    (which returns rows, without the ``2*pi``) and transposes. No lattice
    algebra is re-derived here.

    The ``2*pi`` factor is *omitted*, so ``|B @ hkl| = 1/d``. Every consumer in
    this module uses only the direction of ``G``, where the factor cancels
    identically; :func:`reachable_reflections` uses ``d`` itself for the Bragg
    test, in the same units.
    """
    from midas_hkls.lattice import Lattice
    return Lattice(a_A, b_A, c_A, alpha, beta, gamma).reciprocal_cartesian_vectors().T


def topotomo_tilts(G_sample, T):
    """Stage tilts ``(up, low)`` in degrees aligning ``G`` to the rotation axis.

    Returns the principal branch, ``|up| < 90``. See :func:`tilt_branches` for
    the sibling solution.
    """
    Gs = np.asarray(T, float).T @ np.asarray(G_sample, float)
    if abs(Gs[2]) < _EPS and abs(Gs[1]) < _EPS:
        raise ValueError("G is along the beam; the tilt solution is degenerate")
    ut = math.atan2(Gs[1], Gs[2]) if abs(Gs[2]) > _EPS else math.copysign(math.pi / 2, Gs[1])
    # atan2 spans (-180, 180]; fold to the principal branch so the sibling is
    # always the one outside +/-90 and the two are never silently swapped.
    if abs(ut) > math.pi / 2:
        ut -= math.copysign(math.pi, ut)
    den = Gs[1] * math.sin(ut) + Gs[2] * math.cos(ut)
    lt = math.atan(-Gs[0] / den) if abs(den) > _EPS else math.copysign(math.pi / 2, -Gs[0])
    return math.degrees(ut), math.degrees(lt)


def tilt_branches(G_sample, T):
    """Both equivalent tilt solutions: ``[(up, low), (up + 180, -low)]``.

    Both align ``G``. The second is reachable only on a stage with a ``|up|``
    envelope beyond 90 deg, which is why it is usually irrelevant -- but that is
    a fact about stages, not about the mathematics, so it is exposed rather than
    assumed.
    """
    ut, lt = topotomo_tilts(G_sample, T)
    return [(ut, lt), (ut + 180.0 if ut <= 0 else ut - 180.0, -lt)]


def reachable_reflections(orientation, B, wavelength_A, *, envelope,
                          T=None, hkl_max=3, unique=True):
    """Reflections a stage can bring into TT alignment.

    Parameters
    ----------
    orientation : (3, 3) array
        Crystal-to-sample matrix, so ``G_sample = orientation @ (B @ hkl)``.
    B : (3, 3) array
        Reciprocal basis as a column operator, e.g. from :func:`reciprocal_basis`.
    wavelength_A : float
        Used only to drop reflections with ``d < lambda / 2``, which cannot
        diffract at any angle.
    envelope : float or (float, float)
        Stage limit in degrees, ``|up|`` and ``|low|``. A scalar applies to both.
    T : (3, 3) array, optional
        Instrument transformation; defaults to ID11's
        (:data:`INSTRUMENT_OFFSETS_ID11`).
    hkl_max : int
        Enumeration bound on ``|h|, |k|, |l|``.
    unique : bool
        Drop ``-hkl`` when ``+hkl`` is present. They are the same TT scan
        (Friedel degeneracy above), so keeping both double-counts.

    Returns
    -------
    list of ``(hkl, up_deg, low_deg)``, the tilts being the branch that fits.

    Notes
    -----
    **Systematic absences are NOT applied.** Every geometrically reachable
    reflection is counted, so any reachability figure from this function is an
    optimistic bound: filtering absences can only shrink the set. Pass the result
    through :func:`midas_dct_tt.planning.accessible_reflections` with a
    ``crystal=`` to impose structure factors.
    """
    orientation = np.asarray(orientation, float)
    B = np.asarray(B, float)
    T = instrument_transformation(*INSTRUMENT_OFFSETS_ID11) if T is None else np.asarray(T, float)
    env_u, env_l = (envelope, envelope) if np.isscalar(envelope) else envelope

    out, seen = [], set()
    rng = range(-hkl_max, hkl_max + 1)
    for hkl in itertools.product(rng, rng, rng):
        if hkl == (0, 0, 0):
            continue
        if unique and tuple(-x for x in hkl) in seen:
            continue
        G = orientation @ (B @ np.array(hkl, float))
        n = float(np.linalg.norm(G))
        if n < _EPS or 1.0 / n < wavelength_A / 2.0:      # |B @ hkl| = 1/d
            continue
        for ut, lt in tilt_branches(G, T):
            if abs(ut) <= env_u and abs(lt) <= env_l:
                out.append((hkl, ut, lt))
                seen.add(hkl)
                break
    return out


def best_reachable_pair(orientation, B, wavelength_A, *, envelope, T=None,
                        hkl_max=3):
    """Best-conditioned pair of reflections this stage can actually reach.

    Composes :func:`reachable_reflections` with
    :func:`midas_dct_tt.rotation_coverage.rotation_conditioning`: of everything
    in the envelope, choose the pair whose worst-determined rotation component is
    strongest.

    Returns ``(hkl_a, hkl_b, separation_deg, ratio)``, or ``None`` if fewer than
    two reflections are reachable.

    ``ratio`` is ``lambda_min / lambda_max``, exactly as
    :func:`~midas_dct_tt.rotation_coverage.rotation_conditioning` defines it, and
    for a PAIR it saturates at **0.5**, not 1. The eigenvalues of a pair are
    ``[(1 - cos g)/4, (1 + cos g)/4, 1/2]``, so the largest is always ``1/2`` and

        ``lambda_min / lambda_max = (1 - cos g) / 2``.

    Reaching 1 needs a third reflection out of the plane of the first two.

    .. warning::
       This is **not** the "conditioning ratio" tabulated in the paper's Sec. 8,
       which uses ``lambda_min / lambda_mid = (1 - cos g)/(1 + cos g)`` -- a
       pair-specific normalisation that does reach 1 at 90 deg. The two differ by
       a factor ``(1 + cos g)/2``: a 66.1 deg pair is 0.423 in the paper's
       normalisation and 0.297 here. Quote the separation, which is unambiguous,
       rather than a bare ratio.
    """
    reach = reachable_reflections(orientation, B, wavelength_A,
                                  envelope=envelope, T=T, hkl_max=hkl_max)
    if len(reach) < 2:
        return None
    orientation, B = np.asarray(orientation, float), np.asarray(B, float)
    gs = [orientation @ (B @ np.array(h, float)) for h, _, _ in reach]
    gs = [g / np.linalg.norm(g) for g in gs]

    best = None
    for i, j in itertools.combinations(range(len(gs)), 2):
        _, ratio = rotation_conditioning([gs[i], gs[j]])
        if best is None or ratio > best[3]:
            gam = math.degrees(math.acos(min(1.0, abs(float(gs[i] @ gs[j])))))
            best = (reach[i][0], reach[j][0], gam, ratio)
    return best
