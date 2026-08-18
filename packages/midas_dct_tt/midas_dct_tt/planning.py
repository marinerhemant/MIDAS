"""Scan planning: which reflections should a TT or DCT experiment actually use?

Phase 4 of ``implementation_plan.md``. This module exists because the choice of
reflection set is not a detail — it controls three things at once, and they do
not all pull the same way:

1. **Tomographic coverage.** TT's missing cone has half-angle ``theta``
   (:func:`~midas_dct_tt.geometry.missing_cone_half_angle_deg`), so *low* theta —
   low-index reflections, high energy — gives better spatial reconstruction. This
   is also the recommendation of Liu *et al.* (2025), independently derived.
2. **Identifiability.** One reflection constrains 3 of the 9 components of ``F``;
   three *non-coplanar* ones give all 9, and collinear ones add nothing
   (:func:`~midas_dct_tt.inverse.identifiability_rank`).
3. **Strain-to-orientation leakage.** If a reconstruction models rotation only —
   which is what current DCT/TT codes do — elastic strain leaks into the recovered
   orientation. The size of that leak is

       spurious rotation  =  -½ A⁻¹ · axial([M, E]),
       M = Σᵢ αᵢ Gᵢ Gᵢᵀ,   αᵢ = (1 + sin²θᵢ)/2

   i.e. the **commutator of the strain with the reflection set's second-moment
   tensor**. So it vanishes identically — for *every* strain direction — whenever
   ``M ∝ I``, which is exactly a symmetry-closed reflection set.

Point 3 is the actionable one and it is why this module is worth having:
**leakage is a property of the reflection set you choose, not a fixed penalty of
the technique.** Choosing a symmetry-closed set cuts total spurious orientation by
**~5.3x** under realistic conditions (0.0224 -> 0.0043 deg, acceptance ON, 0.1%
noise, averaged over 6 seeds; C2). The idealised numbers are far larger — 120x with
acceptance ON and noiseless, exactly zero with acceptance OFF — but they are not
what an experiment sees, because at 0.1% noise the residual leak is **24x below the
noise floor**. Rank on leakage; expect ~5x, not 120x.

Status: pre-registered as C2, REFUTED as stated (2026-08-03)
------------------------------------------------------------
Registered in ``dev/paper/PREREGISTER.md`` addendum C2. Noiseless, the registered
criteria pass (0.0001824 deg, ratio 0.0085). **Under the registered noise protocol
they do not**: across 8 strain directions at 0.1% Gaussian noise, **0/8 confirm,
4/8 refute, 4/8 indeterminate** (``dev/paper/RESULTS_C2.md``).

The reason is a threshold-design failure, not a failure of the physics: the
residual leak of a symmetry-closed set (1.8e-4 deg) is **24x below** the
noise-induced orientation error (4.3e-3 +/- 2.1e-3 deg), so an absolute 0.0057 deg
bar is crossed by noise alone in about half of draws. The *benefit* is
undiminished and still worth ranking on — 5.3x less total spurious orientation —
it simply cannot be demonstrated at the effect size that was registered.

The refinement verification produced, and the trap it exposed:

* The null is **exact** only with the acceptance off (gradient 1.5e-7). With it on
  and noiseless the advantage is **120.6x** (0.0174228 vs 0.0001445 deg); with
  0.1% noise the achievable advantage is **5.3x**, noise-limited.
* ``axial([M, E])`` is a theorem about the **peak-shift** channel, and that is the
  channel the ranking is for.
* **The integrated-intensity channel is separately unidentifiable, for BOTH sets,
  and it is not leakage.** With the acceptance on, the TT scan rotates about
  ``G_lab`` and the resolution function is transversely isotropic, so ``m0`` is
  psi-independent to ~1e-13: the whole intensity channel collapses to *three
  scalars* (one per reflection) fitted by *three* rotation parameters. A
  rotation-only fit to it is exactly determined and will absorb **any** dimming —
  absorption, extinction, Debye-Waller, a wrong ``sf2``, beam decay — into a
  spurious rotation of order 0.02 deg. That is a property of the observable, not
  of the reflection set, and it is symmetric across sets. The intensity loss is
  also exactly *even* in the rotation (verified to 1e-13), so its gradient at zero
  vanishes identically and the "leak" it reports has no determined sign.

Practical consequence: rank on leakage (as this module does), but do **not** infer
orientation from integrated intensity alone under a rotation-only model."""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass

import torch

from midas_dfxm.field_inverse import crlb_trace, deformation_identifiability

from .geometry import bragg_angle_deg, wavevector_magnitude
from .inverse import identifiability_rank

__all__ = [
    "ReflectionSetReport",
    "accessible_reflections",
    "physical_crlb_trace",
    "rank_reflection_sets",
    "second_moment_tensor",
    "strain_leakage_index",
    "strain_leakage_for_strain",
]


def _alpha(theta_deg, dtype=torch.float64):
    """Per-reflection weight ``(1 + sin²θ)/2`` from the leakage derivation."""
    s = torch.sin(torch.deg2rad(torch.as_tensor(theta_deg, dtype=dtype)))
    return 0.5 * (1.0 + s ** 2)


def second_moment_tensor(g_list, theta_deg_list=None) -> torch.Tensor:
    """``M = Σᵢ αᵢ Gᵢ Gᵢᵀ`` for a reflection set — the object that controls leakage.

    ``g_list`` is a sequence of sample-frame reciprocal vectors ``(3,)``. If
    ``theta_deg_list`` is omitted the weights are set to 1 (the geometry-only
    form); pass the Bragg angles for the exact form.

    Normalised by its trace so that the scale of ``|G|`` drops out and only the
    *shape* of the set matters — which is what leakage depends on.
    """
    G = torch.stack([g if isinstance(g, torch.Tensor) else torch.as_tensor(g) for g in g_list])
    if not torch.is_floating_point(G):
        G = G.to(torch.float64)
    if theta_deg_list is None:
        a = torch.ones(G.shape[0], dtype=G.dtype, device=G.device)
    else:
        a = torch.stack([_alpha(t, dtype=G.dtype) for t in theta_deg_list])
    M = torch.einsum("i,ij,ik->jk", a, G, G)
    return M / torch.diagonal(M).sum()


def strain_leakage_index(g_list, theta_deg_list=None) -> float:
    """Position-channel leakage: ``‖dev(M)‖_F / ‖M‖_F``. **Zero is perfect.**

    .. note::
       This governs the **peak-shift** channel, which is what a TT/DCT
       reconstruction should fit. Zero here is exact with the acceptance off and
       gives a 120x reduction with it on (C2). It does *not* protect a fit made to
       integrated intensity alone — that observable is unidentifiable under a
       rotation-only model for every reflection set (module docstring).

    Because the leak is ``axial([M, E])``, it vanishes for every strain ``E``
    exactly when ``M ∝ I``, i.e. when the deviatoric part of ``M`` is zero. This
    index is therefore the single number to minimise when choosing a reflection
    set, and it needs no assumption about the strain state.

    A symmetry-closed set such as ``(200),(020),(002)`` or a full ``{111}``
    family scores ~0; ``(111),(200),(220)`` scores ~0.3.
    """
    M = second_moment_tensor(g_list, theta_deg_list)
    dev = M - torch.eye(3, dtype=M.dtype, device=M.device) * torch.diagonal(M).sum() / 3.0
    return float(torch.linalg.matrix_norm(dev) / torch.linalg.matrix_norm(M))


def strain_leakage_for_strain(g_list, E, theta_deg_list=None) -> float:
    """‖axial([M, E])‖ for one specific strain tensor — the direction-resolved leak.

    Use :func:`strain_leakage_index` to choose a set; use this to check a set
    against a strain state you actually expect. ``E`` should be symmetric.
    """
    M = second_moment_tensor(g_list, theta_deg_list)
    E = E if isinstance(E, torch.Tensor) else torch.as_tensor(E)
    E = E.to(dtype=M.dtype, device=M.device)
    C = M @ E - E @ M
    axial = torch.stack([C[2, 1] - C[1, 2], C[0, 2] - C[2, 0], C[1, 0] - C[0, 1]]) * 0.5
    return float(torch.linalg.vector_norm(axial))


@dataclass
class ReflectionSetReport:
    """Everything that should drive the choice of a TT/DCT reflection set.

    Attributes
    ----------
    hkls : tuple
    rank : int
        Identifiable components of ``F`` (out of 9). Below 9 means the set cannot
        determine the deformation gradient however good the data is.
    condition : float
        ``σ_max/σ_min`` of the identifiability Jacobian over its non-null
        directions. Lower is better; 1.0 is ideal.
    crlb_trace : float
        ``tr[(A^T A)^-1] sigma^2`` from :func:`midas_dfxm.field_inverse.crlb_trace`
        -- the **minimum achievable sum of variances** on the nine components of
        ``F`` at unit noise. This is the A-optimality criterion and it is what the
        ranking sorts on: lower is strictly better, it is a property of the
        estimator rather than of a hypothesis, and unlike ``leakage`` it has never
        been refuted. Infinite for a rank-deficient set.
    leakage : float
        :func:`strain_leakage_index` -- systematic **bias**, where ``crlb_trace`` is
        **variance**. Both matter and they are not redundant. Reported rather than
        ranked on, because pre-registered test C2 refuted the strong form: with the
        registered noise the benefit of a symmetry-closed set is ~5x, not the
        120x the noiseless idealisation suggests.
    max_missing_cone_deg : float
        Largest Bragg angle in the set = worst tomographic missing cone.
    thetas_deg : tuple
    """

    hkls: tuple
    rank: int
    condition: float
    crlb_trace: float
    leakage: float
    max_missing_cone_deg: float
    thetas_deg: tuple

    def summary(self) -> str:
        h = " ".join(str(tuple(x)) for x in self.hkls)
        return (f"{h:34s} rank {self.rank}/9  CRLB {self.crlb_trace:7.4f}  "
                f"cond {self.condition:6.3f}  leak {self.leakage:.4f}  "
                f"cone {self.max_missing_cone_deg:5.2f} deg")


def accessible_reflections(B, wavelength_A, *, hkl_max: int = 3, orientation=None,
                           min_theta_deg: float = 0.0, max_theta_deg: float = 90.0,
                           crystal=None, f2_rel_tol: float = 1e-6):
    """Reflections reachable at this wavelength, as ``[(hkl, G_sample, theta), ...]``.

    .. warning::
       **Systematic absences are filtered only if you pass ``crystal``.** Geometry
       alone cannot tell you whether a reflection exists: ``d >= lambda/2`` admits
       ``(100)``, which is *forbidden* in fcc (mixed parity gives ``|F| = 0``).
       Without the structure-factor filter this function returns reflections that
       do not diffract, and because ``(100)`` has the smallest Bragg angle it then
       *wins* a missing-cone ranking and is recommended as optimal. Pass the
       crystal.

    ``B`` is the reciprocal basis (``midas_dfxm.field.reciprocal_basis``), so
    ``G_crystal = B @ hkl`` with the ``2π`` convention. ``orientation`` is the
    optional crystal→sample matrix; omit it to work in the crystal frame.

    Reflections with ``d < λ/2`` are inaccessible at any goniometer setting and
    are dropped silently — that is geometry, not a threshold.
    """
    B = B if isinstance(B, torch.Tensor) else torch.as_tensor(B)
    if not torch.is_floating_point(B):
        B = B.to(torch.float64)
    k = wavevector_magnitude(wavelength_A).to(B.dtype)
    OM = None if orientation is None else torch.as_tensor(orientation, dtype=B.dtype)

    allowed = None
    if crystal is None:
        warnings.warn(
            "accessible_reflections called without `crystal`: systematic "
            "absences are NOT filtered. This is not a small omission here -- "
            "the missing cone has half-angle theta, so the forbidden "
            "low-angle reflections are the ones a coverage ranking prefers, "
            "and they win. For fcc at HEXM energies the geometry-only optimum "
            "is the orthogonal {100} triplet, which has the smallest cone AND "
            "zero strain leakage AND produces no photons. Pass `crystal=` to "
            "get a plan that can actually diffract.",
            RuntimeWarning, stacklevel=2)
    else:
        from midas_dfxm.takagi_taupin import susceptibility_fourier

        def allowed(h):
            _, chih, _ = susceptibility_fourier(crystal, tuple(int(v) for v in h),
                                                wavelength_A=float(wavelength_A),
                                                absorption=False)
            return abs(complex(chih))

    ref_amp, out = 0.0, []
    rng = range(-hkl_max, hkl_max + 1)
    for hkl in itertools.product(rng, rng, rng):
        if hkl == (0, 0, 0):
            continue
        G = B @ torch.as_tensor(hkl, dtype=B.dtype)
        g_mag = torch.linalg.vector_norm(G)
        if float(g_mag) > 2.0 * float(k):
            continue                                  # d < lambda/2
        theta = float(bragg_angle_deg(g_mag, k))
        if not (min_theta_deg <= theta <= max_theta_deg):
            continue
        if allowed is not None:
            amp = allowed(hkl)
            ref_amp = max(ref_amp, amp)
            out.append((hkl, (OM @ G) if OM is not None else G, theta, amp))
        else:
            out.append((hkl, (OM @ G) if OM is not None else G, theta))

    if allowed is None:
        return out
    # Drop systematic absences: |F| below a relative tolerance of the strongest.
    keep = [(h, g, t) for (h, g, t, a) in out if a > f2_rel_tol * max(ref_amp, 1e-300)]
    return keep


def rank_reflection_sets(reflections, *, n_reflections: int = 3, top: int = 10,
                         require_full_rank: bool = True, max_candidates: int = 20000,
                         sort_by: str = "cone", crlb_kw=None):
    """Score and rank candidate reflection sets. Best = zero leakage, low cone.

    ``reflections`` is the output of :func:`accessible_reflections`. Returns a
    list of :class:`ReflectionSetReport`. **The three criteria genuinely conflict,
    so the sort is an explicit choice** (``sort_by``), not something this function
    decides for you. All three are reported whichever you pick.

    ``"cone"`` *(default)* -- smallest missing cone first, then CRLB. TT's
    distinguishing capability is tomographic; if spatial coverage is not the
    priority there is usually a cheaper way to measure strain.

    ``"crlb"`` -- lowest Cramer-Rao bound on ``F`` first: the minimum achievable
    sum of variances under noise, from
    :func:`midas_dfxm.field_inverse.crlb_trace`. The principled *estimator* design
    criterion, and unlike the leakage rule it is not a hypothesis that can be
    refuted.

    ``"leakage"`` -- least strain-to-orientation bias first.

    Why this cannot be collapsed into one number
    --------------------------------------------
    ``crlb_trace`` falls as ``1/|Q|^2``, so it climbs monotonically to the highest
    accessible index -- which is exactly where the missing cone is **worst**:

    ======================  ========  ============  ==========
    set                     ``|Q|``   CRLB          theta
    ======================  ========  ============  ==========
    ``(100),(010),(001)``   1.00      3.0132        1.363 deg
    ``(200),(020),(002)``   2.00      0.7533        2.726 deg
    ``(300),(030),(003)``   3.00      0.3348        4.091 deg
    ======================  ========  ============  ==========

    A pure CRLB sort therefore buys strain precision by giving away exactly the
    spatial coverage TT was chosen for. ``leakage`` is a third, independent axis
    (systematic **bias**, where CRLB is **variance**). Read all three.

    ``require_full_rank`` drops sets that cannot determine all 9 components of
    ``F``; leave it on unless you are deliberately planning an
    orientation-only experiment.

    Raises
    ------
    ValueError
        If the candidate count would exceed ``max_candidates`` — enumerate a
        smaller ``hkl_max`` or pre-filter by theta rather than waiting on a
        combinatorial blow-up.
    """
    n_comb = math.comb(len(reflections), n_reflections) if len(reflections) >= n_reflections else 0
    if n_comb == 0:
        return []
    if n_comb > max_candidates:
        raise ValueError(
            f"{n_comb} candidate sets from {len(reflections)} reflections exceeds "
            f"max_candidates={max_candidates}. Reduce hkl_max or narrow the theta "
            "range first."
        )

    reports = []
    for combo in itertools.combinations(reflections, n_reflections):
        hkls = tuple(c[0] for c in combo)
        gs = [c[1] for c in combo]
        thetas = tuple(c[2] for c in combo)
        rank, sv = identifiability_rank(torch.eye(3, dtype=gs[0].dtype), gs)
        if require_full_rank and rank < 9:
            continue
        nz = sv[sv > 1e-8 * sv[0]]
        cond = float(nz[0] / nz[-1])
        try:
            crlb = float(crlb_trace(list(hkls), noise_std=1.0, **(crlb_kw or {})))
        except Exception:
            crlb = float("inf")
        reports.append(ReflectionSetReport(
            hkls=hkls, rank=rank, condition=cond, crlb_trace=crlb,
            leakage=strain_leakage_index(gs, thetas),
            max_missing_cone_deg=max(thetas), thetas_deg=thetas,
        ))
    keys = {
        "cone": lambda r: (r.max_missing_cone_deg, r.crlb_trace),
        "crlb": lambda r: (r.crlb_trace, r.max_missing_cone_deg),
        "leakage": lambda r: (round(r.leakage, 9), r.max_missing_cone_deg, r.crlb_trace),
    }
    if sort_by not in keys:
        raise ValueError(f"sort_by must be one of {sorted(keys)}, got {sort_by!r}")
    reports.sort(key=keys[sort_by])
    return reports[:top]


def physical_crlb_trace(reflections, *, wavelength_A, crystal=None,
                        anisotropic: bool = True, photon_weighted: bool = True,
                        lorentz: bool = True, acceptance_kw=None,
                        reference_count=None):
    """A-optimality using the noise the instrument actually has.

    :func:`~midas_dfxm.field_inverse.crlb_trace` assumes the measurement error on
    ``dQ`` is **isotropic and identical for every reflection**. Under that
    assumption ``A^T A = I_3 (x) M`` with ``M = sum_g Q_g Q_g^T``, so the trace is
    homogeneous of degree -2 in ``|Q|`` and the bound falls as ``|Q|^-2`` for
    *any* three vectors. That is an identity of the error model, not a statement
    about reflections, and it is the sole reason a naive A-optimal search climbs
    to the highest accessible index.

    Both halves of the assumption are false here, and this package already
    contains the corrections:

    * **The acceptance is wildly anisotropic.** For (200) at
      \\SI{71.7}{keV} the principal widths are 1.83e-3, 3.80e-2 and 3.65e-1
      1/A -- a 199:1 range (:mod:`midas_dct_tt.acceptance`). The narrow
      (rocking) width scales *proportionally to* ``|Q|``, so it cancels the
      ``|Q|^-2`` gain exactly in that channel; only the divergence-limited
      longitudinal channel keeps it. Anisotropy alone turns ``|Q|^-2`` into
      roughly ``|Q|^-1``.
    * **High-index reflections are weaker.** Photon count scales as ``|F|^2``
      times the rotation-scan Lorentz factor ``1/sin 2theta``. For fcc Cu,
      ``|F|^2`` alone is 6909 / 2512 / 1133 for 200/400/600.

    With both applied the ordering can **invert**: the highest-index set becomes
    the worst. Use this, not the isotropic form, to justify any design claim.

    ``reflections`` is the output of :func:`accessible_reflections`, i.e.
    ``[(hkl, G_sample, theta_deg), ...]``. Counts are absolute -- equal exposure
    *time* per reflection -- so only ratios between sets are meaningful; pass
    ``reference_count`` to fix a common scale. Returns ``tr[(A^T W A)^-1]``, or
    ``inf`` if the set is rank deficient.
    """
    from .acceptance import ObjectiveFreeAcceptance, tt_resolution_aniso
    from .conventions import tt_alignment

    refl = list(reflections)
    m = len(refl)
    if m == 0:
        return float("inf")
    dtype = torch.float64

    counts, precisions, q0s = [], [], []
    for hkl, G_sample, theta_deg in refl:
        Q0 = torch.as_tensor(G_sample, dtype=dtype).reshape(3)
        q0s.append(Q0)

        n = 1.0
        if photon_weighted:
            if crystal is None:
                raise ValueError(
                    "photon_weighted=True needs `crystal`: the count scales as "
                    "|F|^2 and there is no |F| without a structure. Pass the "
                    "crystal, or set photon_weighted=False to get the "
                    "equal-photon idealisation."
                )
            from midas_dfxm.takagi_taupin import susceptibility_fourier
            _, chih, _ = susceptibility_fourier(
                crystal, tuple(int(v) for v in hkl),
                wavelength_A=float(wavelength_A), absorption=False)
            n *= abs(complex(chih)) ** 2
        if lorentz:
            n /= max(math.sin(2.0 * math.radians(float(theta_deg))), 1e-12)
        counts.append(n)

        if anisotropic:
            al = tt_alignment(Q0, float(wavelength_A))
            aniso = tt_resolution_aniso(al)
            acc = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab,
                                          **(acceptance_kw or {}))
            R = al.sample_to_lab.to(dtype)          # sample -> lab
            P = torch.zeros(3, 3, dtype=dtype)
            for e_lab in aniso.frame():
                e_lab = e_lab.to(dtype)
                sig = float(acc.effective_sigma(al.G_lab, e_lab))
                e_s = R.transpose(-1, -2) @ e_lab   # back to the sample frame
                P = P + torch.outer(e_s, e_s) / (sig * sig)
        else:
            P = torch.eye(3, dtype=dtype)
        precisions.append(P)

    # NOT normalised per set. Dividing by the set's own mean would force every
    # set to unit mean and cancel the photon weighting exactly whenever the set
    # is symmetry-equivalent (an orthogonal <n00> triplet), which is precisely
    # the comparison the weighting exists to make. Counts are therefore left in
    # absolute units of |chi_h|^2 / sin(2 theta), i.e. equal exposure TIME per
    # reflection, and only ratios between sets are meaningful.
    if reference_count is not None:
        counts = [c / float(reference_count) for c in counts]

    fisher = torch.zeros(9, 9, dtype=dtype)
    for Q0, P, n in zip(q0s, precisions, counts):
        A_g = torch.zeros(3, 9, dtype=dtype)
        for i in range(3):
            A_g[i, 3 * i:3 * i + 3] = Q0
        fisher = fisher + n * (A_g.transpose(-1, -2) @ P @ A_g)

    if int(torch.linalg.matrix_rank(fisher)) < 9:
        return float("inf")
    return float(torch.diagonal(torch.linalg.inv(fisher)).sum())
