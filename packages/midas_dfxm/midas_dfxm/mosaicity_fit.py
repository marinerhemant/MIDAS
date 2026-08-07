"""Physics-forward orientation + mosaicity fitting for DFXM mosaicity scans.

Moment analysis (per-pixel centre of mass) and phenomenological Gaussian fits both
report the *measured* orientation spread, which is the intrinsic sample mosaicity
convolved with the instrument resolution. This module instead fits the physical
forward model: a local orientation with an intrinsic mosaic covariance, convolved with
the (known/calibrated) instrument resolution, evaluated on the motor grid. Fitting it

  1. **deconvolves the instrument resolution** -> the recovered mosaic is the intrinsic
     sample property, not the instrument-broadened one (moments cannot do this);
  2. supports **multiple orientation components** (sub-grains) per pixel;
  3. is **spatially regularizable** -- all pixels can be fit jointly with a smoothness
     prior on the orientation field, turning an ill-posed per-pixel fit into a
     well-posed global inversion (moments are strictly per-pixel);
  4. is fully **differentiable** and provides per-pixel uncertainty.

For Gaussian intrinsic mosaic and Gaussian resolution the convolution is Gaussian with
covariance ``Sigma_total = Sigma_mosaic + Sigma_res``; the fit recovers
``Sigma_mosaic`` by construction. The peak *position* gives the local orientation.

Everything is torch-differentiable and device-portable.
"""
from __future__ import annotations

import torch

from midas_invert.optimize import fit as _invert_fit  # noqa: F401 (available for callers)


def _spd_from_cholesky(l11, l21, l22):
    """Lower-Cholesky (l11,l21,l22) -> SPD 2x2 covariance (batched)."""
    L11 = torch.nn.functional.softplus(l11)
    L22 = torch.nn.functional.softplus(l22)
    s00 = L11 ** 2
    s01 = L11 * l21
    s11 = l21 ** 2 + L22 ** 2
    return s00, s01, s11  # entries of Sigma = L L^T


def _inv2(a, b, d):
    """Inverse of SPD 2x2 [[a,b],[b,d]] -> (ia,ib,id_), det."""
    det = a * d - b * b + 1e-30
    return d / det, -b / det, a / det, det


def moment_orientation(data, chi, phi):
    """Per-pixel first-moment (centre-of-mass) orientation -- the baseline estimator.

    ``data`` ``(P, M)`` non-negative intensities, ``chi``/``phi`` ``(M,)`` motor coords.
    Returns ``(P, 2)``.
    """
    w = data.clamp_min(0)
    wsum = w.sum(-1, keepdim=True) + 1e-30
    return torch.stack([(w * chi).sum(-1) / wsum.squeeze(-1),
                        (w * phi).sum(-1) / wsum.squeeze(-1)], dim=-1)


def fit_orientation_mosaicity(
    data: torch.Tensor,
    chi: torch.Tensor,
    phi: torch.Tensor,
    res_cov,
    *,
    n_components: int = 1,
    lambda_smooth: float = 0.0,
    shape=None,
    steps: int = 500,
    lr: float = 0.02,
    max_offset: float = 1.5,
) -> dict:
    """Fit local orientation + intrinsic mosaic from a DFXM mosaicity scan.

    Parameters
    ----------
    data : (P, M) tensor
        Per-pixel intensities over ``M`` motor settings (normalise externally or not;
        each pixel is peak-normalised internally).
    chi, phi : (M,) tensor
        Motor coordinates of the two rocking axes.
    res_cov : (2, 2) array
        Instrument resolution covariance (calibrated, e.g. from a near-perfect crystal),
        which is deconvolved from the fitted total width.
    n_components : int
        Number of orientation components per pixel (>=2 resolves sub-grains).
    lambda_smooth : float
        Spatial curvature-smoothness weight on the (main-component) orientation field;
        requires ``shape=(ny,nx)``. ``0`` -> independent per-pixel fits.
    max_offset : float
        Bound (deg) on secondary components relative to the main -- prevents runaway.

    Returns
    -------
    dict with ``orientation`` (P, K, 2), ``mosaic_cov`` (P, K, 2, 2) *intrinsic*
    (resolution-deconvolved), ``amplitude`` (P, K), ``background`` (P,), ``loss``,
    and ``orientation_std`` (P, 2) a per-pixel uncertainty on the main orientation.
    """
    device, dtype = data.device, data.dtype
    P, M = data.shape
    K = n_components
    D = data / (data.amax(-1, keepdim=True) + 1e-30)
    R = torch.as_tensor(res_cov, device=device, dtype=dtype)          # (2,2)
    chi = chi.to(device, dtype); phi = phi.to(device, dtype)

    # init main component at the per-pixel argmax mode; moment as reference
    am = D.argmax(-1)
    c0 = chi[am].clone(); p0 = phi[am].clone()

    # parameters (batched over pixels, components)
    cmain = c0.clone().requires_grad_(True)
    pmain = p0.clone().requires_grad_(True)
    off_c = torch.zeros(P, K, device=device, dtype=dtype, requires_grad=True)  # secondary offsets
    off_p = torch.zeros(P, K, device=device, dtype=dtype, requires_grad=True)
    amp = torch.zeros(P, K, device=device, dtype=dtype, requires_grad=True)
    amp.data[:, 0] = 2.0                                              # main dominant
    if K > 1:
        amp.data[:, 1:] = -1.5
    l11 = torch.full((P, K), -1.5, device=device, dtype=dtype, requires_grad=True)
    l21 = torch.zeros(P, K, device=device, dtype=dtype, requires_grad=True)
    l22 = torch.full((P, K), -1.5, device=device, dtype=dtype, requires_grad=True)
    bg = torch.full((P,), -2.0, device=device, dtype=dtype, requires_grad=True)
    params = [cmain, pmain, off_c, off_p, amp, l11, l21, l22, bg]
    opt = torch.optim.Adam(params, lr=lr)

    def component_centers():
        cc = cmain[:, None] + max_offset * torch.tanh(off_c)         # (P,K)
        pp = pmain[:, None] + max_offset * torch.tanh(off_p)
        cc = torch.cat([cmain[:, None], cc[:, 1:]], dim=1) if K > 1 else cmain[:, None]
        pp = torch.cat([pmain[:, None], pp[:, 1:]], dim=1) if K > 1 else pmain[:, None]
        return cc, pp

    ny_nx = shape
    for _ in range(steps):
        opt.zero_grad()
        cc, pp = component_centers()
        A = torch.sigmoid(amp)                                        # (P,K)
        s00, s01, s11 = _spd_from_cholesky(l11, l21, l22)             # intrinsic mosaic (P,K)
        # total = mosaic + resolution
        t00 = s00 + R[0, 0]; t01 = s01 + R[0, 1]; t11 = s11 + R[1, 1]
        ia, ib, idd, _ = _inv2(t00, t01, t11)                        # (P,K)
        dc = chi[None, None, :] - cc[:, :, None]                     # (P,K,M)
        dp = phi[None, None, :] - pp[:, :, None]
        q = ia[:, :, None] * dc**2 + 2 * ib[:, :, None] * dc * dp + idd[:, :, None] * dp**2
        model = (A[:, :, None] * torch.exp(-0.5 * q)).sum(1) + torch.sigmoid(bg)[:, None] * 0.1
        data_loss = ((model - D) ** 2).mean()
        loss = data_loss
        if lambda_smooth > 0 and ny_nx is not None:
            ny, nx = ny_nx
            cf = cmain.reshape(ny, nx); pf = pmain.reshape(ny, nx)
            def curv(f):
                c = torch.zeros((), device=device, dtype=dtype)
                if nx > 2:
                    c = c + (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]).pow(2).mean()
                if ny > 2:
                    c = c + (f[2:] - 2 * f[1:-1] + f[:-2]).pow(2).mean()
                return c
            loss = loss + lambda_smooth * (curv(cf) + curv(pf))
        loss.backward(); opt.step()

    with torch.no_grad():
        cc, pp = component_centers()
        A = torch.sigmoid(amp)
        s00, s01, s11 = _spd_from_cholesky(l11, l21, l22)
        orientation = torch.stack([cc, pp], dim=-1)                  # (P,K,2)
        mosaic_cov = torch.stack([torch.stack([s00, s01], -1),
                                  torch.stack([s01, s11], -1)], -2)  # (P,K,2,2) intrinsic
        # crude per-pixel orientation uncertainty ~ total width / sqrt(SNR-ish signal)
        tot_w = torch.sqrt(s00 + R[0, 0] + s11 + R[1, 1])[:, 0]
        signal = D.sum(-1).clamp_min(1.0)
        ostd = (tot_w / torch.sqrt(signal))[:, None].expand(P, 2)
    return {
        "orientation": orientation, "mosaic_cov": mosaic_cov,
        "amplitude": A, "background": torch.sigmoid(bg),
        "loss": float(loss.detach()), "orientation_std": ostd,
        "moment": moment_orientation(data, chi, phi),
    }


# ---------------------------------------------------------------------------
# Single-axis rocking curves
#
# `fit_orientation_mosaicity` above is inherently two-axis (chi, phi) and scores a
# peak-normalised MSE. A theta rocking scan gives one axis, and asking whether a
# pixel holds one orientation or two is a *model-selection* question -- which needs a
# calibrated likelihood, not an MSE. Hence a separate 1-D fit with an explicit
# Gaussian-or-Poisson negative log-likelihood, so a likelihood ratio between the
# K=1 and K=2 fits is meaningful and can be referred to a parametric-bootstrap null.
# ---------------------------------------------------------------------------

def _pseudo_voigt(dtheta, width, eta):
    """Unit-height pseudo-Voigt: ``(1-eta)*Gaussian + eta*Lorentzian``, matched FWHM.

    ``width`` is the FWHM. ``eta=0`` is exactly Gaussian, ``eta=1`` exactly Lorentzian.
    """
    hw = 0.5 * width
    g = torch.exp(-torch.log(torch.tensor(2.0, dtype=dtheta.dtype,
                                          device=dtheta.device)) * (dtheta / hw) ** 2)
    l = 1.0 / (1.0 + (dtheta / hw) ** 2)
    return (1.0 - eta) * g + eta * l


def rocking_nll(model, data, sigma=None):
    """Negative log-likelihood of ``data`` under ``model``, summed over the scan axis.

    ``sigma=None`` -> Poisson (photon counting, the honest choice for raw counts);
    otherwise Gaussian with the supplied standard deviation, which may be a scalar, one
    value per pixel ``(P,)``, or **per scan point** ``(P, M)``. Per-point is the right
    choice for background-subtracted detector data, where photon noise grows with the
    local count rate: a constant sigma over-weights the top of the peak, which is
    precisely where two-component structure lives, and would inflate a likelihood ratio.

    Constant terms that do not depend on the parameters are dropped -- valid because the
    statistic is only ever used as a *difference* between two fits to the same data.
    """
    if sigma is None:
        mu = model.clamp_min(1e-9)
        return (mu - data * torch.log(mu)).sum(-1)
    if sigma.ndim == 0:
        s = sigma.expand_as(data)
    elif sigma.shape == data.shape:
        s = sigma
    elif sigma.shape == data.shape[:-1]:
        s = sigma[..., None].expand_as(data)
    else:
        raise ValueError(
            f"sigma shape {tuple(sigma.shape)} matches neither data {tuple(data.shape)} "
            f"nor its pixel axis {tuple(data.shape[:-1])}")
    return (0.5 * ((model - data) / s.clamp_min(1e-12)) ** 2).sum(-1)


def fit_rocking_curve(
    data: torch.Tensor,
    theta: torch.Tensor,
    *,
    n_components: int = 1,
    sigma=None,
    max_offset: float | None = None,
    min_width: float | None = None,
    max_width: float | None = None,
    fit_eta: bool = False,
    steps: int = 100,
    lr: float = 0.05,
    polish: int = 150,
    jacobian: str = "analytic",
    init_from: dict | None = None,
) -> dict:
    """Fit ``n_components`` pseudo-Voigt peaks per pixel to a single-axis rocking scan.

    Built for the model-selection question "does this pixel hold one lattice
    orientation or two?", so every parameterisation choice is about keeping the K=2 fit
    from winning for a spurious reason:

    * secondary centres are **bounded** at ``main +- max_offset`` through a ``tanh``, so
      a component cannot escape to the scan edge (unconstrained mixtures demonstrably
      run to the range edges on real DFXM peaks);
    * widths are **floored** at ``min_width``, so a second component cannot collapse onto
      a single noisy point -- that degeneracy is what makes an unfloored K=2 fit always
      win;
    * amplitudes are ``softplus`` (positive), and the fit works in **counts** rather than
      peak-normalised units so the likelihood is calibrated;
    * the main centre is initialised at the per-pixel argmax (mode), the secondary at the
      largest residual after the main -- not at random.

    Parameters
    ----------
    data : (P, M) tensor
        Background-subtracted intensities in counts, ``P`` pixels x ``M`` scan points.
    theta : (M,) tensor
        Scan-axis coordinate (deg).
    sigma : None | float | (P,) tensor
        ``None`` -> Poisson likelihood. Otherwise Gaussian with this noise level.
    max_offset : float, optional
        Bound (deg) on a secondary centre relative to the main. Default ``0.5 * range``.
    min_width, max_width : float, optional
        FWHM bounds (deg). Defaults: one scan step, and the full scan range.
    fit_eta : bool
        Fit the Gaussian/Lorentzian mixing per pixel. Default ``False`` = pure Gaussian,
        which prior DFXM rocking curves are well described by; enable to test that.
    polish : int
        Levenberg-Marquardt iterations after the Adam stage (``0`` disables). Required
        for a trustworthy likelihood ratio -- see the note at the polish step.
    jacobian : {"analytic", "autodiff"}
        How the LM Jacobian is formed. ``"analytic"`` is the default and is what makes
        full maps affordable; ``"autodiff"`` is the slower reference the analytic form is
        tested against.
    init_from : dict, optional
        A previous fit with fewer components. The shared components start at that
        solution and the new ones start at ~zero amplitude, so the optimiser begins
        *at* the simpler model's likelihood. Strongly recommended whenever the two fits
        will be compared: the models are nested, so K=2 can only be better than K=1, and
        a cold-started K=2 that lands worse is an optimiser failure masquerading as
        evidence against a second component.

    Returns
    -------
    dict with ``center`` (P,K) sorted ascending, ``width`` (P,K) FWHM, ``amplitude``
    (P,K), ``background`` (P,), ``eta`` (P,K), ``nll`` (P,) the per-pixel negative
    log-likelihood at the optimum, ``n_params`` the parameter count per pixel, and
    ``model`` (P,M) the fitted curve.
    """
    device, dtype = data.device, data.dtype
    P, M = data.shape
    K = int(n_components)
    theta = theta.to(device, dtype)
    rng = float(theta.max() - theta.min())
    step = rng / max(M - 1, 1)
    if max_offset is None:
        max_offset = 0.5 * rng
    if min_width is None:
        min_width = step
    if max_width is None:
        max_width = rng
    if sigma is not None and not torch.is_tensor(sigma):
        sigma = torch.as_tensor(float(sigma), device=device, dtype=dtype)
    if sigma is not None:
        sigma = sigma.to(device, dtype)

    peak = data.amax(-1, keepdim=True).clamp_min(1e-9)
    # init: main at the mode; secondary at the largest point away from the main
    am = data.argmax(-1)
    c_main = theta[am].clone()
    span = max_width - min_width

    def _inv_softplus(x):
        """Stable inverse of softplus: ``y = x + log(1 - exp(-x))``.

        The naive ``log(expm1(x))`` overflows by x~700, and detector amplitudes are
        routinely in the thousands of counts.
        """
        x = torch.as_tensor(x, device=device, dtype=dtype).clamp_min(1e-6)
        return x + torch.log(-torch.expm1(-x))

    def _inv_sigmoid(p):
        p = torch.as_tensor(p, device=device, dtype=dtype).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    n_e = K if fit_eta else 0

    def _pack(c_main_, c_new_, w_init_, a_init_, b_init_, eta_raw=-4.0):
        offs = torch.zeros(P, K, device=device, dtype=dtype)
        if K > 1:
            d0 = ((c_new_ - c_main_) / max_offset).clamp(-0.95, 0.95)
            offs[:, 1:] = torch.atanh(d0)[:, None]
        wc = w_init_.clamp(min_width + 1e-9, max_width - 1e-9)
        parts = [c_main_[:, None], offs, _inv_sigmoid((wc - min_width) / span),
                 _inv_softplus(a_init_.clamp_min(1e-6)), b_init_[:, None]]
        if fit_eta:
            # eta = sigmoid(raw), so raw=0 would start at eta=0.5 -- a HALF LORENTZIAN.
            # The pseudo-Voigt nests the Gaussian at eta=0 and a nested model must be
            # startable AT the simpler model, or the extra parameter biases away from it
            # (starting at 0.5 made 14.5% of exactly-Gaussian curves return eta>0.1).
            #
            # But there is a genuine tension, and it needs TWO starts rather than one
            # compromise value: a start at eta ~ 0 reproduces the Gaussian exactly, which
            # is what guarantees nesting -- yet sigmoid'(large negative) ~ 0, so eta can
            # never move from there and a real tail would be missed. So the caller supplies
            # eta_raw per candidate: one very negative (nesting guard, stays put) and one
            # moderate (free to find a tail). Best-of-per-pixel then has both properties.
            parts.append(torch.full((P, K), float(eta_raw), device=device, dtype=dtype))
        return torch.cat(parts, dim=1).detach().clone()

    cold_w0 = min(max(0.25 * rng, min_width * 1.5), max_width * 0.9)
    far = (theta[None, :] - c_main[:, None]).abs() * data.clamp_min(0)
    a_cold = peak.expand(P, K).clone()
    if K > 1:
        a_cold[:, 1:] = 0.3 * peak
    # background init = MEDIAN over the scan axis. The minimum is a biased estimator --
    # it is pulled down by noise by roughly one sigma -- whereas for any peak occupying
    # less than half the scan the median is the baseline, the same logic as the dark frame.
    b_cold = data.median(-1).values
    inits = [_pack(c_main.clone(), theta[far.argmax(-1)],
                   torch.full((P, K), cold_w0, device=device, dtype=dtype),
                   a_cold, b_cold)]

    if init_from is not None:
        # Two warm starts, because neither alone is sufficient:
        #  (a) the new component at ~zero amplitude reproduces the simpler model exactly,
        #      which is what guarantees LRT >= 0 -- but it is a SADDLE: with amplitude 0
        #      the new component's position has no gradient, so a genuine doublet never
        #      gets found from there (measured: nll 934 vs 61 for the cold start);
        #  (b) the new component seeded at the largest residual with real amplitude finds
        #      genuine doublets, but can land worse than the simpler model.
        # Running both and keeping the better fit PER PIXEL gives both properties.
        prev_K = init_from["center"].shape[1]
        if prev_K > K:
            raise ValueError(f"init_from has {prev_K} components, more than K={K}")
        c0 = init_from["center"][:, 0].to(device, dtype).clone()
        b0 = init_from["background"].to(device, dtype).clone()
        w0_ = torch.empty(P, K, device=device, dtype=dtype)
        w0_[:, :prev_K] = init_from["width"].to(device, dtype)
        w0_[:, prev_K:] = init_from["width"][:, :1].to(device, dtype)
        resid = data - init_from["model"].to(device, dtype)
        c_res = theta[resid.argmax(-1)]
        a_tiny = torch.empty(P, K, device=device, dtype=dtype)
        a_tiny[:, :prev_K] = init_from["amplitude"].to(device, dtype)
        a_tiny[:, prev_K:] = 1e-4 * peak
        a_seed = a_tiny.clone()
        a_seed[:, :prev_K] = 0.7 * init_from["amplitude"].to(device, dtype)
        a_seed[:, prev_K:] = 0.3 * peak
        # eta_raw = -20 -> eta = 2e-9, i.e. the Gaussian to machine level: this candidate
        # exists purely so the K=2 / free-eta fit can never be worse than what it nests.
        # eta_raw = -1 -> eta = 0.27, free to move and find a genuine tail.
        inits = [_pack(c0.clone(), c_res, w0_, a_tiny, b0, eta_raw=-20.0),
                 _pack(c0.clone(), c_res, w0_, a_tiny, b0, eta_raw=-1.0),
                 _pack(c0.clone(), c_res, w0_, a_seed, b0, eta_raw=-1.0)] + inits

    n_p = inits[0].shape[1]

    def unpack(q):
        i = 0
        cm = q[:, i]; i += 1
        of = q[:, i:i + K]; i += K
        wr = q[:, i:i + K]; i += K
        ar = q[:, i:i + K]; i += K
        br = q[:, i]; i += 1
        er = q[:, i:i + n_e] if fit_eta else torch.zeros(q.shape[0], K,
                                                         device=q.device, dtype=q.dtype)
        return cm, of, wr, ar, br, er

    def evaluate(q):
        cm, of, wr, ar, br, er = unpack(q)
        # Component 0 IS the main centre, always -- never cm + offset[0]. Letting
        # offset[0] also shift it makes the pair (cm, offset[0]) a gauge freedom, i.e. an
        # exactly singular direction in the normal equations that only the LM damping
        # hides.
        cc = cm[:, None]
        if K > 1:
            cc = torch.cat([cc, cm[:, None] + max_offset * torch.tanh(of[:, 1:])], dim=1)
        w = min_width + span * torch.sigmoid(wr)                    # (P,K) FWHM
        A = torch.nn.functional.softplus(ar)
        eta = torch.sigmoid(er) if fit_eta else torch.zeros_like(er)
        d = theta[None, None, :] - cc[:, :, None]                   # (P,K,M)
        prof = _pseudo_voigt(d, w[:, :, None], eta[:, :, None])
        # The background is UNCONSTRAINED, deliberately. A softplus here is a trap: on
        # background-subtracted data the baseline sits at ~0, so softplus' ~ 1e-3, the
        # Jacobian column for the background is effectively dead, and Levenberg-Marquardt
        # damping pins it at its starting value. That produced a stable but WRONG optimum
        # (nll 143.4 against 94.7 from an independent scipy fit of the same curve) -- and
        # it converged there reliably, which is exactly what makes it dangerous. A
        # slightly negative background is also physically legitimate after subtraction.
        model = (A[:, :, None] * prof).sum(1) + br[:, None]
        return cc, w, A, eta, model

    def optimise(q0, use_adam=True):
        q = q0.clone()
        if use_adam:
            q = q.requires_grad_(True)
            opt = torch.optim.Adam([q], lr=lr * max(rng, 1e-6))
            for _ in range(steps):
                opt.zero_grad()
                rocking_nll(evaluate(q)[4], data, sigma).sum().backward()
                opt.step()
            q = q.detach()
        return _polish(q)

    # Adam gets close but stops short on this smooth, low-dimensional problem: it leaves
    # the fitted width ~1% narrow, and on a bright peak a 1% width error is a large
    # likelihood penalty that a second component will happily mop up -- turning
    # under-convergence into fake evidence for a second component. So the polish is not
    # cosmetic; it is what makes the likelihood ratio mean anything.
    #
    # It must be PER PIXEL. A batched L-BFGS on the summed loss shares one line-search
    # step length and one curvature history across every pixel, so pixels converge
    # unevenly -- measured: it left a true single Gaussian with LRT ~ +97 (against ~1
    # from an independent scipy fit of the same curve). Levenberg-Marquardt solves each
    # pixel's own 7x7 normal equations, which is both exact for this model class and
    # cheap, and it reproduces scipy to ~1e-3 in the likelihood.
    sig = None
    if sigma is not None:
        sig = sigma if sigma.shape == data.shape else (
            sigma.expand_as(data) if sigma.ndim == 0
            else sigma[:, None].expand_as(data))

    def residual(qq):
        model = evaluate(qq)[4]
        if sig is None:                           # Poisson -> variance-stabilised weight
            return (model - data) / model.detach().clamp_min(1e-6).sqrt()
        return (model - data) / sig.clamp_min(1e-12)

    def _model_one(q_row):
        """Single-pixel forward, so the Jacobian can be taken per pixel."""
        i = 0
        cm = q_row[i]; i += 1
        of = q_row[i:i + K]; i += K
        wr = q_row[i:i + K]; i += K
        ar = q_row[i:i + K]; i += K
        br = q_row[i]; i += 1
        er = q_row[i:i + n_e] if fit_eta else torch.zeros(K, device=q_row.device,
                                                          dtype=q_row.dtype)
        cc = cm.reshape(1)
        if K > 1:
            cc = torch.cat([cc, cm + max_offset * torch.tanh(of[1:])])
        w = min_width + span * torch.sigmoid(wr)
        A = torch.nn.functional.softplus(ar)
        eta = torch.sigmoid(er) if fit_eta else torch.zeros_like(er)
        d = theta[None, :] - cc[:, None]                            # (K,M)
        prof = _pseudo_voigt(d, w[:, None], eta[:, None])
        return (A[:, None] * prof).sum(0) + br                      # (M,)

    def _resid_one(q_row, d_row, s_row):
        return (_model_one(q_row) - d_row) / s_row

    # Per-pixel Jacobian, available two ways.
    #
    # autograd.functional.jacobian(..., vectorize=True) on the pixel-summed residual
    # replicates the whole graph across the M scan points -- 28 GB resident at 25k pixels,
    # enough to push a 68 GB machine into swap. vmap(jacrev) is the correct per-sample
    # form and is lean, but still walks the autograd graph M times.
    #
    # This model's derivatives are elementary, so the analytic Jacobian is exact, ~O(1)
    # passes, and allocates only the (P, M, n_p) result. It is validated against
    # vmap(jacrev) to 1e-9 in tests/test_rocking_fit.py -- hand-derived derivatives are
    # exactly the kind of thing that is silently wrong, so the autodiff path is kept as
    # the reference rather than deleted.
    _jac_autodiff = torch.func.vmap(torch.func.jacrev(_resid_one), in_dims=(0, 0, 0))

    _LN2 = float(np.log(2.0)) if False else 0.6931471805599453

    def _jac_analytic(q, dat, s_row):
        cm, of, wr, ar, br, er = unpack(q)
        sw = torch.sigmoid(wr)
        w = min_width + span * sw                                   # (P,K)
        hw = 0.5 * w
        A = torch.nn.functional.softplus(ar)
        cc = cm[:, None]
        if K > 1:
            tnh = torch.tanh(of[:, 1:])
            cc = torch.cat([cc, cm[:, None] + max_offset * tnh], dim=1)
        eta = torch.sigmoid(er) if fit_eta else torch.zeros_like(er)
        d = theta[None, None, :] - cc[:, :, None]                    # (P,K,M)
        u = d / hw[:, :, None]
        G = torch.exp(-_LN2 * u ** 2)
        L = 1.0 / (1.0 + u ** 2)
        prof = (1.0 - eta[:, :, None]) * G + eta[:, :, None] * L
        dprof_du = (1.0 - eta[:, :, None]) * (-2.0 * _LN2 * u * G) \
            + eta[:, :, None] * (-2.0 * u * L ** 2)
        # d(model)/d(centre of component k) = -A_k * dprof/du * (1/hw)
        dm_dc = -A[:, :, None] * dprof_du / hw[:, :, None]           # (P,K,M)
        cols = [dm_dc.sum(1)]                                        # cm: shared by all k
        zero = torch.zeros_like(cols[0])
        off_cols = [zero]                                            # offset[0] is unused
        if K > 1:
            sech2 = 1.0 - tnh ** 2                                   # (P,K-1)
            off_cols += [dm_dc[:, k] * max_offset * sech2[:, k - 1][:, None]
                         for k in range(1, K)]
        cols += off_cols
        # d(model)/d(w_raw): du/dw = -u/w, dw/dw_raw = span*sw*(1-sw)
        dw_draw = (span * sw * (1.0 - sw))[:, :, None]
        cols += [(A[:, :, None] * dprof_du * (-u / w[:, :, None]) * dw_draw)[:, k]
                 for k in range(K)]
        # d(model)/d(a_raw): softplus'(x) = sigmoid(x)
        dA = torch.sigmoid(ar)[:, :, None]
        cols += [(prof * dA)[:, k] for k in range(K)]
        cols.append(torch.ones_like(zero))                           # background
        if fit_eta:
            de = (torch.sigmoid(er) * (1.0 - torch.sigmoid(er)))[:, :, None]
            cols += [(A[:, :, None] * (L - G) * de)[:, k] for k in range(K)]
        J = torch.stack(cols, dim=-1)                                # (P,M,n_p)
        return J / s_row[:, :, None]

    _jac = _jac_analytic if jacobian == "analytic" else (
        lambda q, dat, s: _jac_autodiff(q, dat, s))

    def _polish(q):
        if not polish:
            return q
        lam = torch.full((P, 1, 1), 1e-3, device=device, dtype=dtype)
        prev = (residual(q) ** 2).sum(-1)
        eye = torch.eye(n_p, device=device, dtype=dtype)[None]
        stall = 0
        for _ in range(polish):
            s_now = sig if sig is not None else \
                evaluate(q)[4].detach().clamp_min(1e-6).sqrt()
            J = _jac(q, data, s_now)                                # (P, M, n_p)
            r = residual(q)                                         # (P, M)
            JtJ = J.transpose(1, 2) @ J                             # (P,n_p,n_p)
            Jtr = (J.transpose(1, 2) @ r[:, :, None])               # (P,n_p,1)
            diag = torch.diagonal(JtJ, dim1=1, dim2=2)
            damp = lam * torch.diag_embed(diag.clamp_min(1e-12))
            # scale-aware ridge: an absolute 1e-12 is meaningless when the columns carry
            # counts-squared units, and component 0's offset column is identically zero
            # by the gauge fix above, so a genuine zero direction is always present
            ridge = (1e-10 * diag.sum(-1, keepdim=True) / n_p).clamp_min(1e-30)
            try:
                delta = torch.linalg.solve(
                    JtJ + damp + ridge[:, :, None] * eye, -Jtr)[:, :, 0]
            except Exception:
                break
            trial = q + delta
            new = (residual(trial) ** 2).sum(-1)
            better = new < prev
            q = torch.where(better[:, None], trial, q)
            gain = (prev - new).clamp_min(0.0) / prev.abs().clamp_min(1e-12)
            prev = torch.where(better, new, prev)
            lam = torch.where(better[:, None, None], (lam * 0.3).clamp_min(1e-9),
                              (lam * 10.0).clamp_max(1e9))
            # LM converges quadratically here, so stopping early is worthwhile -- but a
            # REJECTED step also produces zero gain, and rejection is how LM works: it
            # raises the damping and tries again. Treating one zero-gain iteration as
            # convergence aborts exactly the escalation that was about to rescue the
            # fit (measured: it left the width at 24.17 mdeg instead of 23.99, and the
            # null LRT at 129 instead of 2.6). Require a run of stalled iterations.
            stall = stall + 1 if float(gain.max()) < 1e-12 else 0
            if stall >= 6:
                break
        return q

    # Candidate set = each start optimised, PLUS each raw start evaluated as-is. Including
    # the un-optimised starts is what makes the nested-model guarantee hold exactly: a warm
    # start reproduces the simpler model's likelihood, so if the optimiser ever wanders
    # somewhere worse the raw start wins and the result can never be worse than the model it
    # nests. Adam runs BEFORE the LM polish and is not monotone, so it genuinely can walk
    # away from a good warm start -- measured as 2.8% nesting violations with fit_eta=True,
    # against 0% for an independent MLE. Warm starts additionally skip Adam, since they
    # begin near the optimum and only need the quadratically-convergent polish.
    candidates = []
    n_warm = len(inits) - 1 if init_from is not None else 0
    for i, q0 in enumerate(inits):
        candidates.append(q0)                                  # raw start
        candidates.append(optimise(q0, use_adam=(i >= n_warm)))

    q = None
    best_nll = None
    for cand in candidates:
        with torch.no_grad():
            n = rocking_nll(evaluate(cand)[4], data, sigma)
        if q is None:
            q, best_nll = cand, n
        else:
            take = n < best_nll
            q = torch.where(take[:, None], cand, q)
            best_nll = torch.where(take, n, best_nll)

    with torch.no_grad():
        cc, w, A, eta, model = evaluate(q)
        cm_, off_, w_raw_, a_raw_, b_raw, _ = unpack(q)
        nll = rocking_nll(model, data, sigma)
        order = cc.argsort(dim=-1)
        cc = cc.gather(-1, order); w = w.gather(-1, order)
        A = A.gather(-1, order); eta = eta.gather(-1, order)
    # per pixel: K*(centre,width,amp) + background, minus the main centre being shared
    n_params = 3 * K + 1
    return {"center": cc, "width": w, "amplitude": A, "eta": eta,
            "background": b_raw.detach(),
            "nll": nll, "n_params": n_params, "model": model,
            "theta_step": step, "theta_range": rng}


def rocking_lrt(fit1: dict, fit2: dict) -> torch.Tensor:
    """Likelihood-ratio statistic ``2*(nll_1 - nll_2)`` for K=2 over K=1, per pixel.

    Large means the second component earned its parameters. Deliberately NOT converted
    to a p-value through an asymptotic chi-square: the mixture null sits on a parameter
    boundary (amplitude 0), so the asymptotic distribution does not hold. Refer this
    statistic to a parametric-bootstrap null instead.
    """
    return 2.0 * (fit1["nll"] - fit2["nll"])
