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
           "ab_sensitive_mask", "partner_multiplicity", "index_asymmetry",
           "asymmetry_sign_test"]


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
    """Can this reflection set separate a from b at all? (rank >= 2)

    **Necessary, not sufficient.** Rank 2 says the ``(h^2, k^2)`` design has two
    independent columns; it does not say the a - b difference is protected. Only
    an ``(h,k)/(k,h)`` PARTNER compares a with b at the same nominal ``|G|``, where
    a radial (2-theta-dependent) systematic cancels; without one, each family's
    radial error enters the splitting directly. So also check
    :func:`ab_sensitive_mask` / :func:`partner_multiplicity` and
    :func:`index_asymmetry`. Measured on La3Ni2O7 2604 domain 1 (2026-09-10):
    True with ZERO partners and every ``h != k`` reflection on the ``|h| > |k|``
    side (12 : 0) -- and a joint orthorhombic fit on that set returned
    delta = 1.8 % with a bootstrap interval [0.17, 10.4] that excludes zero. An
    interval excluding zero on an unprotected set is the artifact, not a result.
    """
    return distortion_rank(hkl) >= 2


def shear_separable(hkl: np.ndarray) -> bool:
    """Can it separate a, b AND the gamma shear? (rank == 3)

    Required whenever the material's real distortion may be a shear -- fitting
    a diagonal cell to sheared data manufactures a splitting.
    """
    return distortion_rank(hkl) >= 3


def ab_sensitive_mask(hkl: np.ndarray) -> np.ndarray:
    """Per-reflection: can THIS reflection help separate ``a`` from ``b``?

    The per-reflection form of :func:`partner_multiplicity`. A reflection is
    sensitive when both hold:

    1. ``|h| != |k|`` -- otherwise ``|G|`` is invariant under ``a <-> b`` and the
       reflection carries no information about the splitting at all;
    2. its partner ``(|k|, |h|)`` is present somewhere in ``hkl`` -- without the
       partner, a shift in ``|G|`` is degenerate with a shift in the scale
       (distance, wavelength) and cannot be attributed to ``a`` versus ``b``.

    **Scope, and it is narrower than it looks.** This is the right rule for a
    ``|G|``-only (d-spacing) analysis. A pipeline holding full 3-D q VECTORS has
    a second, stronger handle this mask does not credit: the in-plane azimuth of
    ``G(h,k,l)`` is ``atan2(k/b, h/a)``, which depends on ``a/b`` for ANY
    ``h, k`` both non-zero, with no partner required. Do not use this mask to
    argue a vector pipeline is insensitive.

    **It is also blind to a gamma shear.** Under the Fmmm-supercell distortion
    mode the splitting pair is ``(1,1)/(1,-1)``, which this mask calls NOT
    sensitive because ``|h| == |k|`` -- while ``(1,0)/(0,1)``, which it does
    credit, does not split under that mode at all. If the real distortion may be
    a shear, use :func:`shear_separable`, not this.
    See ``reference_rp_subcell_supercell_distortion_mode`` in the MIDAS notes.

    ``l`` is IGNORED when matching partners, matching
    :func:`partner_multiplicity`. A partner at a different ``l`` still pins the
    in-plane ratio once ``c`` is known; if you need the stricter same-``l``
    pairing, filter before calling.

    Parameters
    ----------
    hkl
        ``(N, 3)`` integer Miller indices.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``N``.

    Examples
    --------
    >>> import numpy as np
    >>> ab_sensitive_mask(np.array([[1,0,3],[0,1,3],[1,1,2],[1,-1,2]]))
    array([ True,  True, False, False])
    """
    hkl = np.asarray(hkl)
    if hkl.size == 0:
        return np.zeros(0, dtype=bool)
    h, k = np.abs(hkl[:, 0]), np.abs(hkl[:, 1])
    present = {(int(x), int(y)) for x, y in zip(h, k)}
    return np.array([x != y and (int(y), int(x)) in present
                     for x, y in zip(h, k)], dtype=bool)


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
    domains. A ``|q|``- or ``2theta``-DEPENDENT systematic (detector distance,
    beam centre, a distortion residual -- an ordinary geometry-calibration
    error) then pushes ``a`` one way, and the fitted splitting comes out with
    a CONSISTENT SIGN across domains -- 19 of 25 in that data. Since a<->b
    labelling is a per-domain gauge, a common sign is impossible physically
    and is the signature of this artifact.

    **Not any systematic.** A uniform, ISOTROPIC error (e.g. a wavelength
    miscalibration, which scales every ``q`` by the same fraction) is exactly
    degenerate with an overall cell-scale change and produces NO splitting at
    all, however sparse the design (verified 2026-09-13 by a /verify physics
    lens: zero-noise Monte Carlo, flat ~50 % agreement up to a fractional
    dilation 20x the plausible size). The mechanism needs the systematic's
    SIZE to depend on ``|q|``/2theta, not just be present.

    **This diagnosis is itself not yet decisive.** A /verify artifact lens
    (2026-09-13, claim ``d93ca3c0a6e2``) showed a synthetic control with
    ``a == b`` planted EXACTLY, this SAME index-population skew, and one small
    SHARED (not domain-specific) quadratic radial distortion reproduces the
    raster's own measured sign-consistency (agreement, significance) with
    zero real crystallographic splitting. Detecting this pattern is evidence
    of SOME non-physical contribution, not proof of which one, and not proof
    the sample carries no real splitting at all -- see
    :func:`asymmetry_sign_test` and ``midas_defect.raster``'s module
    docstring for the full finding.

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


def asymmetry_sign_test(domains: Sequence[Tuple[np.ndarray, float, float]]) -> dict:
    """Does the SIGN of a domain's own a-vs-b fit track which of h, k dominates its own hkl set?

    ``index_asymmetry``'s own finding is that a sparse, one-sided (h, k) design manufactures
    "a fake splitting of a CONSISTENT SIGN" (its docstring; measured raster-wide as a consistent
    sign in 19 of 25 domains). A raster-wide check that only compares the MAGNITUDE of an a/b
    split against an identical-crystal null (as ``06_raster_lattice_batch.ipynb``'s Step 3 did
    until 2026-09-12, ``manuals/solve-cell/ENVELOPE.md`` #14 caveat) is blind to this: the
    artifact and a real crystal-to-crystal trend can produce the SAME spread of ``|a-b|``. This
    tests the thing the mechanism actually predicts instead: whether ``sign(a-b)`` correlates
    with whether ``|h|>|k|`` or ``|k|>|h|`` dominates that SAME domain's own reflections.

    This is gauge-safe where a magnitude comparison across domains is not: which axis a domain's
    fit calls "a" and which its own reflections call "h" share ONE arbitrary per-domain
    labelling (a 90 deg rotation about c* swaps both together), so correlating a domain's own
    ``sign(a-b)`` against its own ``sign(n_h_gt_k - n_k_gt_h)`` needs no cross-domain convention
    fixed first -- unlike comparing "a" between two different domains, which does.

    ``domains``: one ``(hkl, a, b)`` triple per domain, from an INDEPENDENT (not
    tetragonal-constrained) per-domain fit -- e.g. ``Domain.hkl``, ``Domain.lat.a``,
    ``Domain.lat.b`` from :func:`midas_defect.domains.find_domains`. A domain with a tied index
    count (as many ``|h|>|k|`` as ``|k|>|h|``) or ``a == b`` exactly carries no directional
    information and is excluded; ``n_used`` says how many remain.

    Each domain sorts into one of two GAUGE-EQUIVALENT buckets, not a literal pairing: "same
    sense" lumps together ``|h|>|k| with a>b`` AND ``|k|>|h| with a<b`` (both say the same
    thing -- "the more-populous index's own axis came out shorter" -- under the OTHER domain's
    own h<->k<->a<->b relabelling), and "opposite sense" lumps the other two combinations. This
    lumping is deliberate, not a bug (/verify's reproduction lens flagged 2026-09-13 that
    field names naming a literal ``h_gt_k``/``a_gt_b`` pairing would misdescribe it): reported
    as ``n_same_sense``/``n_opposite_sense`` below, never as a single literal combination.

    Returns ``n_used``, ``n_same_sense``, ``n_opposite_sense``, the larger fraction as
    ``agree_fraction``, ``p_value`` (two-sided EXACT binomial test against 0.5 -- no Gaussian
    assumption), and ``direction`` (``+1`` if "same sense" dominates -- the more-populous index
    tends to come out with the SHORTER axis -- ``-1`` for the other way, ``0`` if ``n_used`` is
    0).

    **A small ``p_value`` here is evidence of SOME non-physical contribution -- it does not by
    itself say which one, or that the sample carries no real splitting at all.** /verify
    REFUTED that stronger reading on the 2604 raster (2026-09-13, claim ``d93ca3c0a6e2``): a
    synthetic control with ``a == b`` planted EXACTLY, the SAME (already-measured) index
    population skew, and one small SHARED quadratic radial distortion -- no domain-specific
    physics at all -- reproduced this raster's own agreement fraction and significance almost
    exactly. This function is real progress over comparing MAGNITUDE spread to a null (which
    cannot see a sign at all), not a settled verdict: read a strong result here as "treat any
    a/b split from this data with more suspicion," not as confirmation of which mechanism, real
    or artifactual, is responsible.
    """
    from scipy.stats import binomtest

    n_same, n_opposite = 0, 0
    for hkl, a, b in domains:
        asym = index_asymmetry(hkl)
        if asym["n_h_gt_k"] == asym["n_k_gt_h"] or a == b:
            continue
        h_dominant = asym["n_h_gt_k"] > asym["n_k_gt_h"]
        a_bigger = a > b
        # "same sense": the more-populous index's OWN axis is the shorter one -- h-dominant
        # with a<b, or (the gauge-mirror of that) k-dominant with b<a. Both read the same way
        # once a domain's own h<->k<->a<->b labelling is allowed to flip together.
        if h_dominant != a_bigger:
            n_same += 1
        else:
            n_opposite += 1
    n_used = n_same + n_opposite
    if n_used == 0:
        return dict(n_used=0, n_same_sense=0, n_opposite_sense=0,
                    agree_fraction=float("nan"), p_value=float("nan"), direction=0)
    p = float(binomtest(n_same, n_used, 0.5, alternative="two-sided").pvalue)
    return dict(n_used=n_used, n_same_sense=n_same, n_opposite_sense=n_opposite,
                agree_fraction=max(n_same, n_opposite) / n_used, p_value=p,
                direction=(1 if n_same >= n_opposite else -1))
