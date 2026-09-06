"""Measuring a small a/b splitting: which reflections can, and how far the
detector actually reaches.

Two small functions, both of which exist because getting them wrong destroyed a
real measurement rather than raising an error.

**Sensitivity needs RANK, not a partner.** (Corrected 2026-09-04 after a
verify lens refuted the partner criterion.) For a tetragonal-parent cell,
``|G|^2 = h^2/a^2 + k^2/b^2 + l^2/c^2``. A family with ``|h| != |k|`` looks
sensitive to a/b, but a LONE (2,1,L) family gives ``4/a^2 + 1/b^2`` -- one
equation in two unknowns. Only when the a<->b partner ``(k,h,L)`` is also
observed can a and b be separated. Measured across the full 2604 raster (486 domains): the dominant |h| != |k|
families are (0,1) with 1818 spots, (0,2) with 1349 and (1,2) with 1021, while
(2,1) is minor at 532 and (2,0) has just 17. Of 104 domains whose |h| != |k| reflections lie in a
single family -- the configuration a partner test calls "blind" -- **104 of 104
are rank-2 and fully separable.** The partner criterion had zero true
positives; what it actually measured was omega coverage, P(partner in a 39 deg
wedge) = 0.174.

**The index box must come from the geometry.** A hand-picked ``lmax=16``
silently deleted the (1,0,L) family -- the partner of (0,1,L) -- and took the
one domain carrying the splitting from 6-7 sensitive reflections to ZERO, with
no error and every other diagnostic still healthy. The detector's own 2theta
limit gives ``l_max = c/d_min = 21`` for that geometry.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

__all__ = ["hkl_box_from_geometry", "distortion_rank",
           "distortion_condition", "ab_separable", "shear_separable",
           "partner_multiplicity", "index_asymmetry"]


def hkl_box_from_geometry(a: float, c: float, *, wavelength_A: float,
                          tth_max_deg: float) -> Tuple[int, int]:
    """Largest ``|h|`` and ``|l|`` the detector can reach. Never guess these.

    ``d_min = lambda / (2 sin(tth_max/2))``, then ``h_max = a/d_min`` and
    ``l_max = c/d_min``, rounded up.

    A box that is too tight fails SILENTLY and in the direction of a cleaner
    result: fewer reflections, no error, and a quietly empty measurement.
    """
    if not (0 < tth_max_deg < 180):
        raise ValueError(f"tth_max_deg must be in (0, 180), got {tth_max_deg}")
    d_min = wavelength_A / (2.0 * math.sin(math.radians(tth_max_deg) / 2.0))
    return int(math.ceil(a / d_min)), int(math.ceil(c / d_min))


def distortion_rank(hkl: np.ndarray) -> int:
    """Rank of the SIGNED distortion design matrix -- what a reflection set can
    actually determine.

    For a tetragonal parent, write ``u = 1/a^2``, ``v = 1/b^2`` and let ``eps``
    be an in-plane gamma shear. Then

        |G|^2 = h^2 u + k^2 v + l^2 w - 2 eps h k / a0^2

    so the design columns are ``(h^2, k^2, h*k)``. Rank 2 means a and b are
    separable; rank 3 means a, b AND the shear are.

    **Rank 2 does NOT require the (k,h) partner.** A lone (2,-1,L) family is
    rank 1, but (2,-1) together with (1,1) is rank 2 -- and (1,1) is exactly the
    kind of |h| = |k| reflection a naive test discards as "blind". Measured on
    La3Ni2O7 2604: a partner-based gate passed 171 of 621 positions while the
    rank test passes **364**, and every one of the 104 domains whose |h| != |k|
    reflections lie in a single family is rank-2 regardless. The partner buys a
    ~2.9x condition-number improvement, not identifiability.

    **The signs matter, so do not take absolute values.** ``hk`` is what carries
    the gamma shear, and ``|hk|`` destroys it: (1,1) and (1,-1) have hk of +1 and
    -1 and are the pair that measures the shear, while (0,1)/(1,0) have hk = 0
    and are exactly BLIND to it. For a Ruddlesden-Popper subcell the Fmmm
    distortion IS a gamma shear, so a diagonal a != b fit converts a pure
    0.197 deg shear into a spurious splitting of median 0.28 % -- at or above the
    0.29 % signal being sought.
    """
    hkl = np.asarray(hkl)
    if hkl.ndim != 2 or hkl.shape[1] != 3:
        raise ValueError(f"hkl must be (N, 3), got {hkl.shape}")
    h, k = hkl[:, 0].astype(float), hkl[:, 1].astype(float)
    A = np.column_stack([h*h, k*k, h*k])
    if len(A) < 2:
        return int(np.linalg.matrix_rank(A))
    return int(np.linalg.matrix_rank(A, tol=1e-9))


def distortion_condition(hkl: np.ndarray) -> float:
    """Condition number of the (h^2, k^2) design -- how WELL a and b separate.

    Finite means separable; large means poorly conditioned. This is the quantity
    a partner improves (measured: 5.75 without, 1.67 with), and reporting it is
    more honest than a pass/fail gate.
    """
    hkl = np.asarray(hkl)
    h, k = hkl[:, 0].astype(float), hkl[:, 1].astype(float)
    A = np.column_stack([h*h, k*k])
    sv = np.linalg.svd(A, compute_uv=False)
    if len(sv) < 2 or sv[-1] <= 0:
        return float("inf")
    return float(sv[0]/sv[-1])


def ab_separable(hkl: np.ndarray) -> bool:
    """Can this reflection set separate a from b at all? (rank >= 2)"""
    return distortion_rank(hkl) >= 2


def shear_separable(hkl: np.ndarray) -> bool:
    """Can it separate a, b AND the gamma shear? (rank == 3)

    Required whenever the material's real distortion may be a shear -- fitting
    a diagonal cell to sheared data manufactures a splitting.
    """
    return distortion_rank(hkl) >= 3


def partner_multiplicity(hkl: np.ndarray) -> dict:
    """How many reflections support each a<->b sensitive pair.

    The count that matters, and the one a bare "N sensitive reflections" hides.
    Measured on 2604 p=329 domain 2: 6 reflections were called sensitive, but
    the (1,0) side of the (0,1)/(1,0) pair was a SINGLE spot -- remove it and
    all six stop being sensitive. A pair supported on one spot is one blob-
    finding decision away from nothing.

    Returns ``{(min(|h|,|k|), max(|h|,|k|)): (n_forward, n_reverse)}`` for every
    pair present in both orientations.
    """
    hkl = np.asarray(hkl)
    h, k = np.abs(hkl[:, 0]), np.abs(hkl[:, 1])
    counts: dict = {}
    for x, y in zip(h, k):
        counts[(int(x), int(y))] = counts.get((int(x), int(y)), 0) + 1
    out = {}
    for (x, y), n in counts.items():
        if x == y or (y, x) not in counts:
            continue
        key = (min(x, y), max(x, y))
        if key not in out:
            out[key] = (counts[(key[0], key[1])], counts[(key[1], key[0])])
    return out


def index_asymmetry(hkl: np.ndarray) -> dict:
    """How unevenly a and b are constrained by the observed index distribution.

    **This is the systematic that produces a fake splitting of a consistent
    sign.** In a limited omega wedge the observed reflections are not symmetric
    in h and k: measured across the full 2604 raster, ``|k| > |h|`` in 5004
    reflections against ``|h| > |k|`` in 1009 -- a 5x asymmetry, with (0,1)
    at 1818 spots and (2,0) at 17.

    The consequence is that ``1/a^2`` rides on a sparse design column:
    ``sigma(1/a^2)/sigma(1/b^2)`` had median 5.46 and exceeded 2 in 71.8 % of
    domains. Any radial systematic then pushes ``a`` one way, and the fitted
    splitting comes out with a CONSISTENT SIGN across domains -- 19 of 25 in
    that data. Since a<->b labelling is a per-domain gauge, a common sign is
    impossible physically and is the signature of this artifact.

    Returns ``n_k_gt_h``, ``n_h_gt_k``, ``ratio`` and ``sigma_ratio`` (the
    relative uncertainty of 1/a^2 against 1/b^2 from the design alone). Report
    it alongside any per-domain splitting; a large ``sigma_ratio`` means the
    sign is not trustworthy.
    """
    hkl = np.asarray(hkl)
    h, k = np.abs(hkl[:, 0]), np.abs(hkl[:, 1])
    nk, nh = int((k > h).sum()), int((h > k).sum())
    A = np.column_stack([hkl[:, 0].astype(float)**2, hkl[:, 1].astype(float)**2])
    try:
        cov = np.linalg.inv(A.T @ A)
        sr = float(np.sqrt(cov[0, 0]/cov[1, 1]))
    except np.linalg.LinAlgError:
        sr = float("inf")
    return dict(n_k_gt_h=nk, n_h_gt_k=nh,
                ratio=(nk/nh if nh else float("inf")), sigma_ratio=sr)
