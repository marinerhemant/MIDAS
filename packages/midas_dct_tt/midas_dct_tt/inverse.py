"""Deformation inverse: exact finite strain vs the linearised approximation.

Phase 3 of ``implementation_plan.md``. Pre-registered in
``dev/paper/PREREGISTER.md`` before any result was computed.

The two models
--------------
The local scattering vector of a deformed sub-volume is exactly

    Q = F^-T G0

The incumbent treatment linearises. With ``F = I + H``,

    F^-T = I - H^T + (H^T)^2 - ...        so    Q_lin = (I - H^T) G0

and the omitted term is ``O(|H|^2 |G0|)``. :func:`local_Q` implements both, so
the *only* difference between an exact and a linearised inversion is this one
function -- which is what makes the comparison a controlled experiment rather
than two codebases.

Identifiability, stated up front
--------------------------------
``Q = F^-T G0`` is 3 equations in the 9 unknowns of ``F^-T``. **One reflection
therefore constrains exactly 3 components per voxel**, in the specific
combination ``F^-T G0``, and leaves a 6-dimensional null space. Three
*non-coplanar* reflections determine all 9. Collinear reflections (``(111)``,
``(222)``) add nothing. :func:`identifiability_rank` measures this rather than
assuming it.

Do not extrapolate this to orientation *distributions*: recovering a per-voxel
ODF from a single setting is covered by a hard negative in the known-limits
ledger and is not attempted here.
"""
from __future__ import annotations

import torch
from midas_dfxm.field import deform_reflection
from midas_invert import fit

from .forward import topograph_stack

__all__ = [
    "fit_uniform_deformation",
    "identifiability_rank",
    "intensity_scales",
    "local_Q",
    "profiled_intensity_residual",
]


def local_Q(F: torch.Tensor, G0: torch.Tensor, *, model: str = "exact") -> torch.Tensor:
    """Local scattering vector for a deformation gradient ``F`` and reference ``G0``.

    Parameters
    ----------
    F : (..., 3, 3) tensor
    G0 : (3,) or (..., 3) tensor
    model : {"exact", "linear"}
        ``"exact"`` solves ``F^T Q = G0`` (delegated to
        :func:`midas_dfxm.field.deform_reflection`, no explicit inverse).
        ``"linear"`` applies the small-strain form ``(I - H^T) G0``, ``H = F - I``
        -- the incumbent approximation, implemented here **only** so it can be
        compared against on identical data.
    """
    if model == "exact":
        return deform_reflection(F, G0)
    if model != "linear":
        raise ValueError(f"model must be 'exact' or 'linear', got {model!r}")

    eye = torch.eye(3, device=F.device, dtype=F.dtype).expand_as(F)
    H = F - eye
    g = G0.to(device=F.device, dtype=F.dtype)
    g = g.expand(*F.shape[:-2], 3) if g.dim() == 1 else g
    return g - torch.einsum("...ji,...j->...i", H, g)


def identifiability_rank(F, G0_list, *, rtol: float = 1e-8):
    """Rank and singular values of ``d(Q_1..Q_n)/dF`` for a set of reflections.

    Returns ``(rank, singular_values)``. The rank is how many of the 9 components
    of ``F`` the given reflections constrain at all; the singular-value spectrum
    says how well. A reflection set that looks adequate by count can still be
    rank-deficient if its ``G0`` are coplanar -- which is the whole point of
    measuring this rather than counting reflections.
    """
    F = F.detach().clone().requires_grad_(True)
    rows = []
    for G0 in G0_list:
        Q = local_Q(F, torch.as_tensor(G0, dtype=F.dtype, device=F.device))
        for i in range(3):
            (g,) = torch.autograd.grad(Q[i], F, retain_graph=True)
            rows.append(g.reshape(-1))
    J = torch.stack(rows)                       # (3n, 9)
    sv = torch.linalg.svdvals(J)
    # Report the spectrum over the full 9-dimensional parameter space. svdvals of
    # a (3n, 9) Jacobian returns only min(3n, 9) values, so for a single
    # reflection the six unconstrained directions would simply be absent rather
    # than visibly zero -- which reads as "no null space" when the null space is
    # the entire point.
    if sv.numel() < 9:
        sv = torch.cat([sv, torch.zeros(9 - sv.numel(), dtype=sv.dtype, device=sv.device)])
    rank = int((sv > rtol * sv[0]).sum())
    return rank, sv


def fit_uniform_deformation(
    observed,
    grain,
    alignments,
    psi_deg,
    *,
    detector,
    hkls,
    resolutions,
    model: str = "exact",
    init_F=None,
    steps: int = 300,
    lr: float = 1e-3,
    sf2=1.0,
    supersample: int = 1,
):
    """Fit a single uniform ``F`` to observed TT sinograms under a chosen model.

    One ``F`` for the whole grain, so the exact-vs-linearised comparison is not
    entangled with the ill-posedness and regularisation of a per-voxel field
    inverse. This is the pre-registered decisive test (Hypothesis A).

    ``alignments``, ``hkls`` and ``resolutions`` are **lists**, one entry per
    reflection, and ``observed`` is a list of sinograms to match. Multiple
    reflections are not optional for a 9-component fit:
    :func:`identifiability_rank` shows one reflection constrains only 3 of 9, so
    a single-reflection fit leaves six directions free and its "recovered F" is
    whatever the optimiser wandered to in that null space. Three non-coplanar
    reflections give full rank.

    Returns ``(F, info)``.

    ``fit`` defaults to a cosine ``lr`` schedule to stop Adam ending a run worse
    than a point it already visited; that overshoot does **not** occur for this
    9-parameter fit (measured ``info["final_over_min"] = 0.0`` at 200 steps), so
    the schedule is off.  It is a truncated fit, not an oscillating one
    (``tail_improvement`` 0.84), and annealing on top costs accuracy: recovering
    a planted ``F`` at ``steps=200, lr=1e-3``, ``max|F - F_true|`` went
    7.03e-4 -> 9.57e-4.  ``return_best`` stays on and is free here.  Read
    ``info["settled"]`` -- it is False, and the fix for that is more ``steps``.
    """
    if not (len(alignments) == len(hkls) == len(resolutions) == len(observed)):
        raise ValueError(
            "alignments, hkls, resolutions and observed must be the same length "
            "(one entry per reflection)"
        )
    G0s = [grain.field.reference_G(h) for h in hkls]
    dev, dt = G0s[0].device, G0s[0].dtype
    n = grain.n_voxels

    if init_F is None:
        init_F = torch.eye(3, dtype=dt, device=dev)
    F = init_F.detach().clone().to(device=dev, dtype=dt).requires_grad_(True)
    obs = [o.to(device=dev, dtype=dt) for o in observed]

    def loss_fn():
        total = 0.0
        for G0, al, hkl, res, ob in zip(G0s, alignments, hkls, resolutions, obs):
            Q = local_Q(F, G0, model=model).expand(n, 3)
            pred = topograph_stack(
                grain, al, psi_deg, detector=detector, hkl=hkl,
                resolution=res, sf2=sf2, supersample=supersample, Q_sample=Q,
            )
            total = total + ((pred - ob) ** 2).sum()
        return total

    info = fit([F], loss_fn, steps=steps, lr=lr, optimizer="adam",
               lr_schedule="none", log_every=1)
    return F.detach(), info


# --- nuisance intensity scales ---------------------------------------------
def intensity_scales(pred, obs, *, weights=None, eps: float = 1e-30):
    """Per-reflection intensity scales that best match ``pred`` to ``obs``.

    ``pred`` and ``obs`` are ``(R, S)`` -- R reflections, S scan points. Returns
    ``(R, 1)`` scales ``a_r`` minimising ``sum_s w (a_r * pred - obs)^2``, i.e. the
    closed form ``a_r = sum(w * pred * obs) / sum(w * pred^2)``.

    **Why this must exist.** No real TT or DCT experiment has absolute
    cross-reflection intensity calibration: structure factors, absorption path,
    detector gain, beam decay and exposure all differ per reflection and none is
    known to better than a few percent. A fit that treats predicted intensity as
    absolute will therefore spend any mismatch on whatever parameter can absorb
    it, and in a rotation-only model that parameter is **orientation**. Measured:
    a pure 4.72% flux error with *zero strain* produced **0.0265 deg** of entirely
    spurious rotation -- larger than the strain signal it was being compared to.

    Profiling the scale out analytically (rather than adding it to the optimiser)
    keeps the parameter count unchanged and is differentiable, so it composes with
    autograd fits.

    .. warning::
       **It does not cost nothing.** In a TT scan the sample rotates about
       ``G_lab``, so a *longitudinal* (lattice-parameter) strain shifts ``m0`` by a
       factor that is constant in ``psi`` -- measured spread **4.2e-15** under a
       pure dilation, i.e. exactly a per-reflection scalar. Profiling therefore
       **annihilates the entire longitudinal-strain sensitivity of the intensity
       channel**: the profiled residual is flat at machine zero (~1e-30) for every
       trial dilation, against a proper minimum (8.8e-04 at zero, 7.8e-03 at 2x
       truth) unprofiled.

       This is worst for exactly the sets :mod:`midas_dct_tt.planning` recommends:
       for ``(200),(020),(002)`` every ``G`` is a cube axis, so the **whole normal-
       strain block** is psi-invariant and is removed. Retained Fisher trace ~0.79
       (mixed) and ~0.83 (cube). Position/peak-shift sensitivity is unaffected --
       this is a statement about the *intensity* channel only.
    """
    w = torch.ones_like(pred) if weights is None else weights
    num = (w * pred * obs).sum(dim=-1, keepdim=True)
    den = (w * pred * pred).sum(dim=-1, keepdim=True).clamp_min(eps)
    return num / den


def profiled_intensity_residual(pred, obs, *, weights=None, normalise: bool = True):
    """Intensity residual **immune to a per-reflection flux error**.

    Scales are profiled out by :func:`intensity_scales` before the residual is
    formed. Two precise statements, because the loose one is false:

    * The residual **at the true parameters is zero** for any per-reflection
      constant -- so the *noiseless* argmin is preserved exactly. Measured 7.1e-32
      profiled vs **2.032e-03** unprofiled at a 4.72% flux error (the unprofiled
      value is analytically ``a^2/(1+a)^2``).
    * The residual **value** is invariant only when the per-reflection factors are
      **equal**, because ``normalise=True`` divides by a *global* sum. With
      unequal factors the relative deviation is 3.0e-03 (for [0.9, 1.0, 1.1]) and
      4.6e-02 (for [0.5, 1.0, 2.0]). With noise the argmin shifts too.

    .. warning::
       **A per-``psi`` flux error is NOT absorbed** -- and that is the realistic
       one. Beam decay, or a psi-dependent absorption path as the sample rotates,
       varies within a scan, and a single scalar per reflection cannot follow it.
       Measured: a **1.0% linear drift over one psi scan** produces 0.00107 deg of
       spurious rotation and fails the 1e-3 deg bar, where a per-reflection
       scan-constant error of 4.72% gives 0.00021 deg and passes.

    What survives is the *shape* of the intensity variation across the scan, which
    is where the physics is. Note the interaction with the acceptance model: under
    a transversely isotropic acceptance the TT intensity is exactly
    ``psi``-independent, so R scales absorb all R numbers and this residual is
    **identically zero** -- correctly reporting that the channel is empty. Use
    :class:`~midas_dct_tt.acceptance.AnisotropicTTResolution`, where the true thin
    plate gives a 180-deg-period modulation, and the residual becomes informative.

    ``normalise`` divides by the observed scale so the result is dimensionless.

    .. note::
       This is an **unweighted** sum of squares, i.e. it assumes independent
       pixels. A detector PSF correlates them, which leaves the fitted values
       unbiased but makes chi-squared and error bars optimistic -- by **~3.9x** at
       a 1.5 px PSF for a global parameter. Correct with
       :func:`~midas_dct_tt.detector.effective_dof` (``mode="mean"`` for a fitted
       parameter): multiply the standard error by ``1/sqrt(effective_dof)``.
    """
    a = intensity_scales(pred, obs, weights=weights)
    r = a * pred - obs
    w = torch.ones_like(pred) if weights is None else weights
    out = (w * r * r).sum()
    if normalise:
        den = (w * obs * obs).sum().clamp_min(1e-30)
        out = out / den
    return out
