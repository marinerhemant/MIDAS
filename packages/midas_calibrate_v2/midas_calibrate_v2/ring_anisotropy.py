"""How much apparent a/b splitting the GEOMETRY alone makes: the cos(2 eta) of a ring.

A ring that should be a circle in |q| is modulated in azimuth by anything that shears the
detector model. For an in-plane metric anisotropy delta, ``|q| = q0 (1 - delta cos 2 psi)``,
so the relative cos(2 eta) amplitude ``A2/q0`` of a ring whose lattice is known to be
isotropic IS an equivalent delta: the splitting the geometry manufactures on its own. The
cos(eta) term ``A1/q0`` is a beam-centre error.

**Measure it on a ring on the SAME frames as the reflections you care about.** The
distortion model refined by :func:`midas_calibrate_v2.calibrate` (``a1..a6``) absorbs the
calibrant's own cos(2 eta), so a separate calibrant exposure bounds the calibration, not the
sample frames. Measured on La3Ni2O7 in a diamond anvil cell (2026-09-08): the ON-FRAME anvil
rings gave A2/q0 = 0.142 % at q ~ 3.05 1/A (0.243 % at 4.96, 0.994 % at 5.89, all above an
azimuth-permutation null) -- the size of the 0.108-0.149 % a/b splitting being reported, which
was refuted as a material property on that evidence -- while the separate CeO2 exposure gave
only ~0.015-0.022 %.

A geometric systematic is coherent across every subset of the data, so a sign test or a
disjoint-subset test cannot catch it. This one can.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["RingHarmonics", "ring_harmonics"]


@dataclass(frozen=True)
class RingHarmonics:
    """Harmonic content of one ring's ``q`` against azimuth."""

    q0: float
    a1_rel: float            #: cos(eta) amplitude / q0 -- a beam-centre error
    a2_rel: float            #: cos(2 eta) amplitude / q0 -- the EQUIVALENT delta
    phase1_deg: float
    phase2_deg: float        #: long axis of the apparent distortion, in [0, 180)
    p_a1: float              #: azimuth-permutation p-value of A1
    p_a2: float              #: azimuth-permutation p-value of A2
    n: int
    n_azimuth_sectors: int   #: occupied 30-degree sectors, of 12
    determined: bool         #: False when the azimuths cannot separate cos 2eta from q0


def _fit(e: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    X = np.column_stack([np.ones_like(e), np.cos(e), np.sin(e), np.cos(2 * e), np.sin(2 * e)])
    if w is None:
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        sw = np.sqrt(w)
        c, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return c


def ring_harmonics(eta_deg: Sequence[float], q: Sequence[float], *,
                   weights: Optional[Sequence[float]] = None, n_null: int = 500,
                   rng_seed: int = 0) -> RingHarmonics:
    """Fit ``q = q0 (1 + A1 cos(eta - p1) + A2 cos 2(eta - p2))`` and test it.

    Parameters
    ----------
    eta_deg
        Azimuth of each point, degrees, in any consistent convention.
    q
        |q|, or a radius, of each point. ``A1`` and ``A2`` are RELATIVE, so the unit cancels.
    weights
        Optional per-point weights (e.g. ``1/sigma**2``).
    n_null
        Azimuth permutations. The null keeps every point's ``q`` and shuffles only which
        azimuth it sits at -- the same radial support, so it could match -- and the p-values
        are ``(k + 1)/(n_null + 1)``.

    ``determined`` is False when fewer than 4 of 12 thirty-degree sectors are occupied: the
    cos 2eta term is then degenerate with ``q0`` and the amplitude means nothing, however small
    its p-value. The anvil spots at one raster position are often like this; pooled over a
    raster they are not.
    """
    e = np.radians(np.asarray(eta_deg, float))
    y = np.asarray(q, float)
    w = None if weights is None else np.asarray(weights, float)
    if e.shape != y.shape or e.size < 6:
        raise ValueError("need matching eta_deg and q with at least 6 points")
    c = _fit(e, y, w)
    q0 = float(c[0])
    a1 = math.hypot(c[1], c[2]) / q0
    a2 = math.hypot(c[3], c[4]) / q0
    rng = np.random.default_rng(rng_seed)
    k1 = k2 = 0
    for _ in range(int(n_null)):
        cp = _fit(rng.permutation(e), y, w)
        k1 += math.hypot(cp[1], cp[2]) / cp[0] >= a1
        k2 += math.hypot(cp[3], cp[4]) / cp[0] >= a2
    sectors = int(np.unique(np.floor((np.degrees(e) % 360.0) / 30.0)).size)
    return RingHarmonics(
        q0=q0, a1_rel=float(a1), a2_rel=float(a2),
        phase1_deg=float(math.degrees(math.atan2(c[2], c[1])) % 360.0),
        phase2_deg=float((math.degrees(math.atan2(c[4], c[3])) / 2.0) % 180.0),
        p_a1=float((k1 + 1) / (n_null + 1)), p_a2=float((k2 + 1) / (n_null + 1)),
        n=int(e.size), n_azimuth_sectors=sectors, determined=bool(sectors >= 4))
