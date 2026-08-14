"""Branch C: differentiable direct inversion.

Branches A and B both invert and fit as two separate steps. This one does
neither separately: it builds a differentiable forward map

    voxel peak parameters -> per-voxel pattern -> line integral over the ray
                          -> the measured (omega, translation, R) sinogram

and solves for the voxel parameters by gradient descent on the residual.

**Why it can be better than either.** Branch A back-projects fitted
parameters, which is only valid for the three additive outputs (see
:mod:`midas_dt.conventions`). Branch B is exact but reconstructs every bin and
then fits every voxel independently, so a voxel's fit cannot use the fact that
its neighbours are constrained by the same rays. Here the peak model is
enforced *inside* the inversion, so every measurement constrains every
parameter it actually touches, and sigma comes from the curvature of the loss
rather than from repeated reconstructions.

**Why it might not be.** Gradient descent on a non-convex objective, seeded
from Branch B, is not guaranteed to improve on its seed. Nothing here asserts
that it does.

    NO PERFORMANCE CLAIM IS MADE OR IMPLIED.

    Whether this beats Branch B on accuracy at matched compute is an open
    question that has NOT been tested. Per the implementation plan the claim
    is gated on a preregistered comparison (fix the null and the effect size
    BEFORE running) followed by an adversarial verification pass. Until that
    happens, treat this as a third method that produces plausible maps, not as
    an improvement on the other two. If it loses, that is a result and it gets
    reported.

    What HAS been verified is correctness, on synthetic data with known
    ground truth: the projector matches an independent reference, its adjoint
    passes the dot-product test, the analytic gradients match finite
    differences, and the solver recovers planted parameters from noiseless
    data. See ``tests/test_direct.py``.

**Reuse.** The optimiser and the Laplace covariance come from
``midas_invert`` (``fit``, ``laplace_uncertainty``); the peak model is the one
in :mod:`midas_dt.peakfit`, reproduced in torch so it is differentiable, with
the same fixed ``mu = 0.5`` and shared width -- otherwise the branches would
not be comparable channel for channel. The projector is local rather than from
``midas_pipeline.recon``: that package is a pipeline orchestrator and making it
a dependency of a leaf package would invert the dependency direction. The two
are checked against each other in the tests instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .branches import BranchResult
from .conventions import FIT_OUTPUT_NAMES, fit_output_index
from .sinogram import SinogramStack

__all__ = [
    "DirectResult",
    "MIX_FACTOR",
    "projection_matrix",
    "forward_project",
    "run_direct",
    "laplace_sigma",
]

log = logging.getLogger(__name__)

#: The pseudo-Voigt mixing fraction. Fixed at 0.5, not fitted -- ``PeakFit.c``
#: hard-codes it and :func:`midas_dt.peakfit.fit_lineout` reproduces that. A
#: fitted mix here would make Branch C's ``MixFactor`` map incomparable with
#: the other two branches, which always report 0.5.
MIX_FACTOR = 0.5


def _torch():
    try:
        import torch
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "direct inversion needs midas-invert. Install with "
            "`pip install midas-dt[direct]`, or use run_recon_then_fit, which "
            "is exact and needs only numpy."
        ) from exc
    return torch


# --------------------------------------------------------------- projector
def projection_matrix(size: int, angles_deg, n_det: int, *,
                      dtype=None, device=None):
    """Sparse Radon operator: ``(n_angles * n_det, size * size)``.

    One matrix, applied to every R bin, because the line integral is the same
    geometry regardless of which part of the pattern is being integrated. That
    is also what makes the forward map cheap: the nonlinearity is per-voxel and
    the projection is a single sparse mat-mul.

    **Convention.** ``t = x sin(theta) + y cos(theta)``, matching
    ``midas_tomo``'s phantom projector. Not ``x cos + y sin``: the two differ
    by a transpose of the image, which is a mistake that reconstructs to
    something that looks fine and correlates ~0 with the truth.

    Intensity is splatted linearly into the two neighbouring detector bins.
    Nearest-neighbour splatting aliases badly enough to cap the achievable
    correlation well below 1 regardless of the solver.
    """
    torch = _torch()
    dtype = dtype or torch.float64
    ang = torch.as_tensor(np.asarray(angles_deg, dtype=np.float64),
                          dtype=dtype, device=device).ravel()
    n_ang = int(ang.numel())

    c = (size - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=dtype, device=device) - c,
        torch.arange(size, dtype=dtype, device=device) - c,
        indexing="ij",
    )
    x = xx.reshape(-1)
    y = yy.reshape(-1)
    vox = torch.arange(size * size, device=device)

    th = torch.deg2rad(ang)
    # (n_ang, n_vox) detector coordinate of every voxel at every angle
    t = x[None, :] * torch.sin(th)[:, None] + y[None, :] * torch.cos(th)[:, None]
    t = t + (n_det - 1) / 2.0

    lo = torch.floor(t)
    frac = t - lo
    lo = lo.long()

    rows, cols, vals = [], [], []
    for shift, weight in ((0, 1.0 - frac), (1, frac)):
        det = lo + shift
        ok = (det >= 0) & (det < n_det)
        a_idx = torch.arange(n_ang, device=device)[:, None].expand_as(det)
        rows.append((a_idx[ok] * n_det + det[ok]))
        cols.append(vox[None, :].expand_as(det)[ok])
        vals.append(weight[ok])

    idx = torch.stack([torch.cat(rows), torch.cat(cols)])
    return torch.sparse_coo_tensor(
        idx, torch.cat(vals), (n_ang * n_det, size * size),
        dtype=dtype, device=device,
    ).coalesce()


def forward_project(images, A):
    """Apply the sparse Radon operator to ``(n_r, size, size)`` images.

    Returns ``(n_r, n_angles * n_det)``. Autograd flows through, which is the
    entire point: ``A`` is constant, so the gradient of the loss with respect
    to the voxel parameters is ``A^T`` applied to the projection residual.
    """
    torch = _torch()
    flat = images.reshape(images.shape[0], -1)          # (n_r, n_vox)
    return torch.sparse.mm(A, flat.T).T


# -------------------------------------------------------------- peak model
def _profile(radii, centre, sigma):
    """Pseudo-Voigt shape, peak value 1, matching ``peakfit.fit_lineout``.

    ``mu * lorentzian + (1 - mu) * gaussian`` with ``mu`` fixed and the two
    widths shared -- the legacy model, reproduced so the branches compare.
    """
    torch = _torch()
    d = (radii[None, :] - centre[:, None]) / sigma[:, None]
    g = torch.exp(-0.5 * d * d)
    lo = 1.0 / (d * d + 1.0)
    return MIX_FACTOR * lo + (1.0 - MIX_FACTOR) * g


def _patterns(params, radii):
    """Per-voxel patterns ``(n_active, n_r)`` from the constrained parameters."""
    amp, centre, sigma, bg = params
    return bg[:, None] + amp[:, None] * _profile(radii, centre, sigma)


@dataclass(frozen=True)
class Scales:
    """Units the raw parameters are expressed in, so all of them are O(1).

    This is not cosmetic. Adam takes a step of roughly ``lr`` per iteration in
    RAW parameter units, almost independently of the gradient magnitude. If the
    raw amplitude is measured in counts and the seed is off by 6000, reaching
    the answer needs ~6000/lr steps -- 75,000 at the default. Measured before
    this was introduced: the solver sat 1.5 px from a planted peak centre after
    2000 steps and got monotonically *worse* with a larger learning rate, which
    reads exactly like non-identifiability and is not.

    With amplitude in units of ``amp`` and width in units of the window, every
    raw parameter is order one and the default learning rate means the same
    thing for all four.
    """

    amp: float
    r_min: float
    r_max: float

    @property
    def width(self) -> float:
        return max(self.r_max - self.r_min, 1e-9)


def _constrain(raw, sc: Scales):
    """Raw unconstrained tensors -> physical parameters.

    Softplus rather than clamping for the positive quantities: a clamp has zero
    gradient once it binds, so a voxel driven negative early would never
    recover. The centre is squashed into the channel's own radius window,
    because a peak fitted outside the range it was extracted from is
    meaningless -- the same bound ``peakfit.fit_lineout`` applies.
    """
    torch = _torch()
    raw_amp, raw_cen, raw_sig, raw_bg = raw
    sp = torch.nn.functional.softplus
    amp = sc.amp * sp(raw_amp)
    bg = sc.amp * sp(raw_bg)
    sigma = sc.width * sp(raw_sig) + 1e-3
    centre = sc.r_min + sc.width * torch.sigmoid(raw_cen)
    return amp, centre, sigma, bg


def _unconstrain(amp, centre, sigma, bg, sc: Scales):
    """Inverse of :func:`_constrain`, for seeding from Branch B."""
    torch = _torch()

    def inv_softplus(v):
        v = torch.clamp(v, min=1e-6)
        return v + torch.log(-torch.expm1(-v))

    frac = torch.clamp((centre - sc.r_min) / sc.width, 1e-4, 1 - 1e-4)
    return [
        inv_softplus(torch.clamp(amp / sc.amp, min=1e-6)),
        torch.log(frac / (1 - frac)),
        inv_softplus(torch.clamp((sigma - 1e-3) / sc.width, min=1e-6)),
        inv_softplus(torch.clamp(bg / sc.amp, min=1e-6)),
    ]


# ------------------------------------------------------------------ result
@dataclass
class DirectResult:
    """Fitted voxel parameters plus the diagnostics needed to judge them."""

    maps: dict[str, np.ndarray]
    loss: float
    n_active: int
    n_steps: int
    converged: bool
    channel: object
    limits: object
    residual_rel: float
    sigma: dict[str, np.ndarray] = field(default_factory=dict)
    history: list = field(default_factory=list)

    def as_branch_result(self) -> BranchResult:
        """Adapt to :class:`~midas_dt.branches.BranchResult` so ``compare()``
        works across all three branches."""
        return BranchResult(
            maps=self.maps, branch="direct", channel=self.channel,
            limits=self.limits, sigma=self.sigma,
            linearity={n: _LINEARITY.get(n, "exact") for n in self.maps},
        )

    def describe(self) -> str:
        return (f"direct: {self.n_active} active voxels, {self.n_steps} steps, "
                f"loss {self.loss:.6g}, relative residual {self.residual_rel:.4f}"
                f"{'' if self.converged else '  [NOT CONVERGED]'}")


#: Outputs whose Branch C value is not the same measurement as the other
#: branches'. Surfaced through ``BranchResult.approximate_outputs()`` rather
#: than quietly filled with a lookalike number.
_LINEARITY = {
    # There is no observed per-voxel pattern in direct inversion -- nothing is
    # ever reconstructed -- so the "observed" maximum can only be the model's.
    "MaxIntensityObs": "model peak value, not an observation",
    # The residual lives in projection space; it does not decompose per voxel.
    "MeanError": "not defined per voxel (residual is in projection space)",
    # No separate end-window estimate exists; this is the fitted background.
    "BGSimple": "equals BGFit; no separate end-window estimate",
}


# -------------------------------------------------------------------- solve
def run_direct(
    stack: SinogramStack,
    *,
    size: int | None = None,
    steps: int = 400,
    lr: float = 0.05,
    optimizer: str = "adam",
    seed_from: BranchResult | None = None,
    mask_threshold: float | None = None,
    weighted: bool = True,
    attenuation=None,
    device=None,
    dtype=None,
    log_every: int = 0,
) -> DirectResult:
    """Solve for voxel peak parameters directly against the sinograms.

    Parameters
    ----------
    size : int, optional
        Reconstruction grid edge. Defaults to the number of translations,
        which is the sampling the scan actually provides. Asking for more
        voxels than translations does not add information.
    seed_from : BranchResult, optional
        A Branch B result to start from -- the intended use, and what the plan
        means by "built on Branch B's data path". Without it the solver starts
        from the data's own moments, which is slower and more prone to local
        minima.
    mask_threshold : float, optional
        Voxels whose seed total falls below this are held at zero and not
        optimised. Defaults to the 60th percentile, matching
        :func:`~midas_dt.branches.run_recon_then_fit`, so the two agree on what
        counts as sample.
    weighted : bool
        Weight the residual by 1/variance. The variance is carried from
        integration, so this is a real Poisson weighting and not a guess.
        Turning it off weights empty detector bins as heavily as the peak.
    attenuation : (n_omega, size, size) array, optional
        Self-absorption factors from
        :func:`midas_dt.absorption.attenuation_factors`. Folded into the
        forward operator, which is the **exact** way to handle absorption:
        the model then describes what the instrument measured, so nothing has
        to be undone afterwards. Branches A and B can only apply the
        approximate rotation-averaged correction, because they reconstruct
        before they know about it.

    Returns
    -------
    DirectResult
        Call ``.as_branch_result()`` to compare against branches A and B.
    """
    torch = _torch()
    from midas_invert import fit

    dtype = dtype or torch.float64
    ch = stack.channel
    n_eta, n_r = stack.bin_shape
    n_bins, n_omega, n_trans = stack.intensity.shape
    size = int(size or n_trans)

    # Collapse eta, exactly as Branch B does before fitting, so the two are
    # fitting the same 1-D lineout and any difference is the method.
    inten = stack.intensity.reshape(n_eta, n_r, n_omega, n_trans).sum(axis=0)
    var = stack.variance.reshape(n_eta, n_r, n_omega, n_trans).sum(axis=0)

    data = torch.as_tensor(inten, dtype=dtype, device=device).reshape(n_r, -1)
    if weighted:
        w = 1.0 / torch.clamp(
            torch.as_tensor(var, dtype=dtype, device=device).reshape(n_r, -1),
            min=1e-12)
    else:
        w = torch.ones_like(data)

    radii = torch.as_tensor(
        np.linspace(ch.r_min, ch.r_max, n_r), dtype=dtype, device=device)
    if attenuation is None:
        A = projection_matrix(size, stack.omega_deg, n_trans,
                              dtype=dtype, device=device)
    else:
        from .absorption import attenuated_projection_matrix
        A = attenuated_projection_matrix(
            size, stack.omega_deg, n_trans, attenuation,
            dtype=dtype, device=device)
        log.info("direct: solving with self-absorption in the forward operator "
                 "(mean factor %.4f)", float(np.mean(attenuation)))

    amp0, cen0, sig0, bg0, active, amp_scale = _seed(
        stack, size, n_r, seed_from, mask_threshold, dtype, device)
    sc = Scales(amp=amp_scale, r_min=float(ch.r_min), r_max=float(ch.r_max))
    n_active = int(active.sum())
    if n_active == 0:
        raise ValueError(
            "no voxels are above the mask threshold, so there is nothing to "
            "solve for. Lower mask_threshold, or check that the sinograms "
            "carry signal."
        )
    log.info("direct: %d of %d voxels active on a %dx%d grid",
             n_active, size * size, size, size)

    raw = [t.clone().requires_grad_(True) for t in
           _unconstrain(amp0, cen0, sig0, bg0, sc)]

    act_idx = torch.as_tensor(np.flatnonzero(active.reshape(-1)), device=device)

    def predict():
        amp, centre, sigma, bg = _constrain(raw, sc)
        pat = _patterns((amp, centre, sigma, bg), radii)      # (n_active, n_r)
        img = torch.zeros((n_r, size * size), dtype=dtype, device=device)
        img = img.index_copy(1, act_idx, pat.T)
        return torch.sparse.mm(A, img.T).T                    # (n_r, n_rays)

    def loss_fn():
        resid = predict() - data
        return torch.mean(w * resid * resid)

    info = fit(raw, loss_fn, steps=steps, lr=lr, optimizer=optimizer,
               log_every=log_every)

    with torch.no_grad():
        pred = predict()
        num = torch.linalg.vector_norm(pred - data)
        den = torch.clamp(torch.linalg.vector_norm(data), min=1e-30)
        residual_rel = float(num / den)
        amp, centre, sigma, bg = _constrain(raw, sc)
        maps = _to_maps(amp, centre, sigma, bg, radii, active, size)

    return DirectResult(
        maps=maps, loss=float(info["loss"]), n_active=n_active, n_steps=steps,
        converged=bool(info.get("loss") is not None), channel=ch,
        limits=stack.limits, residual_rel=residual_rel,
        history=list(info.get("history", [])),
    )


def _seed(stack, size, n_r, seed_from, mask_threshold, dtype, device):
    """Initial parameters and the active mask.

    From a Branch B result when given one, else from the data's own moments.
    """
    torch = _torch()
    ch = stack.channel

    if seed_from is not None:
        m = seed_from.maps
        got = _resample(m.get("RMEAN"), size)
        amp_m = _resample(m.get("MaxInt"), size)
        sig_m = _resample(m.get("SigmaG"), size)
        bg_m = _resample(m.get("BGFit"), size)
        active = np.isfinite(got) & np.isfinite(amp_m) & (np.nan_to_num(amp_m) > 0)
        centre = np.where(np.isfinite(got), got, 0.5 * (ch.r_min + ch.r_max))
        amp = np.where(np.isfinite(amp_m), amp_m, 0.0)
        sigma = np.where(np.isfinite(sig_m) & (sig_m > 0), sig_m,
                         0.05 * (ch.r_max - ch.r_min))
        bg = np.where(np.isfinite(bg_m), bg_m, 0.0)
    else:
        # Moments of the summed projections: crude and isotropic, but it puts
        # the centre in the right part of the window and -- critically -- the
        # amplitude on the right SCALE.
        cube = stack.intensity.reshape(
            stack.bin_shape[0], n_r, *stack.intensity.shape[1:]).sum(axis=0)
        lineout = cube.sum(axis=(1, 2))
        r = np.linspace(ch.r_min, ch.r_max, n_r)
        tot = float(lineout.sum())
        c0 = float((r * lineout).sum() / tot) if tot > 0 else float(r.mean())
        s0 = 0.05 * (ch.r_max - ch.r_min)

        # Per-RAY peak, not the sum over rays. A ray crossing the sample
        # integrates ~size voxels, so a voxel amplitude is the ray's peak
        # divided by that path length. Summing over every ray first (which an
        # earlier version did) overestimates the seed by the number of rays --
        # here 384x -- and Adam, whose step size is fixed in raw units, then
        # needs tens of thousands of steps to walk back down.
        ray_peak = cube.max(axis=0)
        hot = ray_peak[ray_peak > 0]
        peak = float(np.median(hot)) if hot.size else 1.0
        a0 = max(peak / max(size, 1), 1e-9)

        centre = np.full((size, size), c0)
        amp = np.full((size, size), a0)
        sigma = np.full((size, size), s0)
        bg = np.zeros((size, size))
        active = np.ones((size, size), dtype=bool)

    if mask_threshold is not None:
        active = active & (np.nan_to_num(amp) > mask_threshold)
    elif seed_from is not None:
        finite = np.nan_to_num(amp)[active] if active.any() else np.array([0.0])
        active = active & (np.nan_to_num(amp) > np.percentile(finite, 0))

    sel = active.reshape(-1)
    t = lambda a: torch.as_tensor(  # noqa: E731
        np.asarray(a, dtype=np.float64).reshape(-1)[sel], dtype=dtype,
        device=device)
    # The amplitude unit: the typical active-voxel amplitude, so the raw
    # parameter starts near 1 rather than near its value in counts.
    live = np.asarray(amp, dtype=np.float64).reshape(-1)[sel]
    live = live[np.isfinite(live) & (live > 0)]
    amp_scale = float(np.median(live)) if live.size else 1.0
    if not np.isfinite(amp_scale) or amp_scale <= 0:
        amp_scale = 1.0
    return t(amp), t(centre), t(sigma), t(bg), active, amp_scale


def _resample(arr, size):
    """Nearest-neighbour resample a Branch B map onto the direct grid.

    Branch B reconstructs onto a power-of-two grid (``recon_size``), which is
    usually larger than the translation count. Nearest neighbour is deliberate:
    this is a seed, and interpolating a map that already contains NaN holes
    spreads the holes.
    """
    if arr is None:
        return np.full((size, size), np.nan)
    a = np.asarray(arr, dtype=np.float64)
    if a.shape == (size, size):
        return a
    iy = np.clip((np.arange(size) * a.shape[0] / size).astype(int),
                 0, a.shape[0] - 1)
    ix = np.clip((np.arange(size) * a.shape[1] / size).astype(int),
                 0, a.shape[1] - 1)
    return a[np.ix_(iy, ix)]


def _to_maps(amp, centre, sigma, bg, radii, active, size):
    """The 12 canonical outputs, derived analytically from the fitted model."""
    torch = _torch()
    with torch.no_grad():
        pat = _patterns((amp, centre, sigma, bg), radii)
        total = pat.sum(dim=1)
        n_r = radii.numel()
        vals = {
            "RMEAN": centre,
            "MixFactor": torch.full_like(centre, MIX_FACTOR),
            "SigmaG": sigma,
            "SigmaL": sigma,                      # shared width, as in the C
            "MaxInt": amp,
            "MaxIntensityObs": bg + amp,          # the model's peak; see _LINEARITY
            "BGFit": bg,
            "BGSimple": bg,
            "MeanError": torch.full_like(centre, float("nan")),
            "FitIntegratedIntensity": amp * sigma * float(np.sqrt(2 * np.pi)),
            "TotalIntensity": total,
            "TotalIntensityBackgroundCorr": total - bg * n_r,
        }
    out = {}
    sel = active.reshape(-1)
    for name in FIT_OUTPUT_NAMES:
        flat = np.full(size * size, np.nan)
        flat[sel] = vals[name].detach().cpu().numpy()
        out[name] = flat.reshape(size, size)
    assert set(out) == set(FIT_OUTPUT_NAMES), "output set drifted from the C order"
    assert len(out) == 12
    for i, n in enumerate(FIT_OUTPUT_NAMES):
        assert fit_output_index(n) == i
    return out


# ----------------------------------------------------------------- Laplace
def laplace_sigma(
    stack: SinogramStack,
    result: DirectResult,
    voxels,
    *,
    size: int | None = None,
    noise_var: float | None = None,
    device=None,
    dtype=None,
) -> dict[str, np.ndarray]:
    """Per-voxel sigma from the curvature of the loss, for selected voxels.

    This is the payoff the plan wanted from a differentiable model: an error
    bar out of the Laplace approximation rather than out of repeated
    reconstructions.

    **It is block-diagonal, and that is an approximation with a known sign.**
    The exact Laplace covariance is over ALL parameters at once -- 4 per active
    voxel, so a 32x32 sample with 600 active voxels is a 2400x2400 Hessian, and
    ``torch.autograd.functional.hessian`` builds it densely. Instead this holds
    every other voxel fixed and computes the 4x4 block for each requested
    voxel. Neighbouring voxels in tomography are strongly correlated through
    the shared rays, so ignoring the off-diagonal blocks **understates** the
    uncertainty. Use it to rank voxels by confidence, not as a calibrated
    interval.

    Parameters
    ----------
    voxels : sequence of (iy, ix)
        Which voxels to evaluate. There is no "all" option on purpose: the cost
        is one Hessian per voxel and an accidental full-map call is a very long
        wait with no output.
    noise_var : float, optional
        Scales MSE curvature to log-likelihood curvature. **Defaults to
        ``1/N``**, N being the number of data points -- NOT to the converged
        loss.

        The converged loss is the right plug-in when the loss is an unweighted
        MSE and the noise level is unknown. It is wrong here, because this loss
        is already weighted by ``1/variance`` and the variance is known -- it
        was propagated from integration. Passing the loss as well counts the
        noise twice. The only remaining conversion is mean-to-sum, hence 1/N.

        Measured on a planted phantom: with the loss (20204) the reported
        sigma on a peak centre was 446 px, for a parameter confined to a 20 px
        window and recovered to 0.02 px. With 1/N it is 0.035 px. The
        inflation factor is exactly ``sqrt(loss * N)``.

    Returns
    -------
    dict
        ``{"RMEAN": (X, X), "MaxInt": ..., "SigmaG": ..., "BGFit": ...}``, NaN
        everywhere not requested.
    """
    torch = _torch()
    from midas_invert import laplace_uncertainty

    dtype = dtype or torch.float64
    ch = stack.channel
    n_eta, n_r = stack.bin_shape
    _, n_omega, n_trans = stack.intensity.shape
    size = int(size or n_trans)

    inten = stack.intensity.reshape(n_eta, n_r, n_omega, n_trans).sum(axis=0)
    var = stack.variance.reshape(n_eta, n_r, n_omega, n_trans).sum(axis=0)
    data = torch.as_tensor(inten, dtype=dtype, device=device).reshape(n_r, -1)
    w = 1.0 / torch.clamp(
        torch.as_tensor(var, dtype=dtype, device=device).reshape(n_r, -1),
        min=1e-12)
    radii = torch.as_tensor(np.linspace(ch.r_min, ch.r_max, n_r),
                            dtype=dtype, device=device)
    A = projection_matrix(size, stack.omega_deg, n_trans,
                          dtype=dtype, device=device)

    names = ("MaxInt", "RMEAN", "SigmaG", "BGFit")
    _amp_live = np.asarray(result.maps["MaxInt"], dtype=np.float64)
    _amp_live = _amp_live[np.isfinite(_amp_live) & (_amp_live > 0)]
    sc = Scales(amp=float(np.median(_amp_live)) if _amp_live.size else 1.0,
                r_min=float(ch.r_min), r_max=float(ch.r_max))
    full = {n: np.asarray(result.maps[n], dtype=np.float64) for n in names}
    base = np.stack([np.nan_to_num(full[n]) for n in names])     # (4, X, X)
    base_t = torch.as_tensor(base.reshape(4, -1), dtype=dtype, device=device)

    out = {n: np.full((size, size), np.nan) for n in names}
    n_data = int(data.numel())
    nv = float(noise_var if noise_var is not None else 1.0 / n_data)
    rank_deficient = 0

    for (iy, ix) in voxels:
        v = int(iy) * size + int(ix)
        if not np.isfinite(full["MaxInt"][iy, ix]):
            continue

        def loss_for(theta):
            sp = torch.nn.functional.softplus
            amp = sc.amp * sp(theta[0])
            centre = sc.r_min + sc.width * torch.sigmoid(theta[1])
            sigma = sc.width * sp(theta[2]) + 1e-3
            bg = sc.amp * sp(theta[3])
            pat = _patterns((amp[None], centre[None], sigma[None], bg[None]),
                            radii)[0]
            img = base_t.new_zeros((n_r, size * size))
            # Every other voxel contributes its fitted pattern, held fixed.
            img = img + _fixed_field(base_t, radii, v)
            img[:, v] = pat
            pred = torch.sparse.mm(A, img.T).T
            resid = pred - data
            return torch.mean(w * resid * resid)

        theta0 = _unconstrain(
            base_t[0, v][None], base_t[1, v][None], base_t[2, v][None],
            base_t[3, v][None], sc)
        theta0 = torch.cat([t for t in theta0])
        lap = laplace_uncertainty(loss_for, theta0, noise_var=nv)
        # Delta method: sigma on the raw parameter -> sigma on the physical one.
        jac = _constrain_jacobian(theta0, sc)
        std = lap["std"].detach().cpu().numpy() * jac
        if int(lap["rank_eff"]) < 4:
            rank_deficient += 1
        for k, n in enumerate(names):
            out[n][iy, ix] = float(std[k])

    if rank_deficient:
        # Worth saying out loud rather than burying: a rank-deficient block
        # means one parameter combination is unconstrained by the data at that
        # voxel, and pinv silently returns a large-but-finite sigma for it
        # instead of the infinity that would be honest. Commonly the flat
        # background, which a single voxel barely constrains.
        log.warning(
            "%d of %d voxels had a rank-deficient 4x4 block; their sigma on "
            "the unconstrained direction is a pinv artefact, not a bound",
            rank_deficient, len(list(voxels)) if hasattr(voxels, '__len__') else -1)
    return out


def _fixed_field(base_t, radii, exclude):
    """Patterns of every voxel except *exclude*, as an ``(n_r, n_vox)`` field."""
    torch = _torch()
    with torch.no_grad():
        amp, centre, sigma, bg = base_t[0], base_t[1], base_t[2], base_t[3]
        pat = _patterns((amp, centre, sigma, bg), radii).T       # (n_r, n_vox)
        pat = pat.clone()
        pat[:, exclude] = 0.0
    return pat


def _constrain_jacobian(theta, sc: Scales):
    """d(physical)/d(raw) at *theta*, for the delta method.

    Carries the Scales factors: without them the reported sigma would be in
    raw units and silently wrong by the amplitude scale.
    """
    torch = _torch()
    with torch.no_grad():
        s = torch.sigmoid(theta)
        return np.array([
            sc.amp * float(torch.sigmoid(theta[0])),           # softplus'
            sc.width * float(s[1] * (1 - s[1])),               # scaled sigmoid'
            sc.width * float(torch.sigmoid(theta[2])),
            sc.amp * float(torch.sigmoid(theta[3])),
        ])
