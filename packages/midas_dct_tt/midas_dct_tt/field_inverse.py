"""Per-voxel deformation-gradient inversion from TT projections -- the 12-D problem.

``fit_uniform_deformation`` recovers **one** ``F`` for a whole grain: 9 unknowns.
This recovers ``F(r)`` per voxel -- 3 spatial dimensions times 9 deformation
components, the "12-D" reconstruction that Liu's 2024 thesis (Sec. 6.3) names as
the extension the field needs, in his own words:

    "the reconstruction framework would require an extension to a **12D (3D
    position and 9D deformation gradient tensor) space** to capture all degrees
    of freedom."

Why this is tractable here and was declared infeasible before
------------------------------------------------------------
Vigano *et al.* (2014) rule 12-D out on a specific ground: their 6-D DCT
reconstruction discretises orientation space and adds a sampled dimension per
degree of freedom, so cost grows exponentially and 12-D is unreachable. That
argument is about **discrete sampling**, and it does not apply to a continuous
parameterisation. Here every voxel simply carries a free ``F`` and the gradient
comes from autograd, so cost is **linear** in voxels and in components: one
adjoint sweep supplies all ``9N`` derivatives at once, and nothing is enumerated.

That is an implementation contribution, not new physics, and it should be
described as one. The physics of the forward operator is unchanged.

The problem is ill-posed, and the regularisation is not optional
---------------------------------------------------------------
TT is laminography with a missing cone of half-angle ``theta``
(:func:`~midas_dct_tt.geometry.missing_cone_half_angle_deg`), so some spatial
frequencies are simply not measured -- suppressed ~10x on the operator, verified
directly rather than inferred from a reconstruction. On top of that, one
reflection constrains only 3 of the 9 components of ``F``
(:func:`~midas_dct_tt.inverse.identifiability_rank`). A per-voxel fit therefore has
a large null space, and an unregularised solution will happily fill it with
structure that no data supports. ``lambda_smooth`` penalises the finite-difference
gradient of ``H = F - I`` across the voxel grid; it is a **prior**, and results
should always be reported with the value used and a sensitivity to it.

Composition with the rest of the package
----------------------------------------
* ``profile_scales=True`` removes a per-reflection, scan-constant flux error via
  :func:`~midas_dct_tt.inverse.intensity_scales` -- worth having, since an
  uncalibrated 4.72% flux error alone fakes 0.0265 deg of rotation. **But it is
  not free**: in TT a longitudinal strain is itself psi-constant, so profiling
  also annihilates the intensity channel's entire lattice-parameter sensitivity
  (and, for cube-axis reflection sets, the whole normal-strain block). It also
  does **not** absorb a per-``psi`` drift such as beam decay. See the warnings on
  :func:`~midas_dct_tt.inverse.profiled_intensity_residual`.
* ``psf_px`` puts the detector blur inside the *model*, not just the data. If the
  data were simulated with a PSF and the model has none, the fit is solving a
  different operator than the one that produced the data.
* Pass :class:`~midas_dct_tt.acceptance.ObjectiveFreeAcceptance`, which is the
  physically correct acceptance for a bare detector. The older ``na = 0`` forms
  are the *opposite* limit (an infinitely selective objective) and understate the
  roll:rock anisotropy by ~10x.
"""
from __future__ import annotations

import torch

from .detector import apply_psf
from .forward import topograph_stack
from .inverse import intensity_scales, local_Q

__all__ = [
    "affine_basis",
    "cross_validate_lambda",
    "decompose_affine",
    "fit_deformation_field",
    "select_lambda_discrepancy",
    "smoothness_penalty",
    "weighted_corr",
]


def smoothness_penalty(H: torch.Tensor, shape) -> torch.Tensor:
    """Sum of squared first differences of ``H`` over a regular voxel grid.

    ``H`` is ``(N, 3, 3)`` and ``shape`` is ``(nx, ny, nz)`` with
    ``nx*ny*nz == N``. Returns a scalar, differentiable.

    Deliberately a first-difference (not second) penalty: it prefers a *uniform*
    ``F``, which is the honest null hypothesis for a grain with no measured
    intragranular structure, rather than merely a smooth one.
    """
    if shape is None:
        raise ValueError(
            "smoothness regularisation needs a regular grid; this grain has "
            "shape=None (unstructured point cloud). Use lambda_smooth=0 or "
            "supply a gridded grain."
        )
    nx, ny, nz = shape
    if nx * ny * nz != H.shape[0]:
        raise ValueError(f"grid {shape} does not match {H.shape[0]} voxels")
    g = H.reshape(nx, ny, nz, 3, 3)
    out = H.new_zeros(())
    if nx > 1:
        out = out + ((g[1:] - g[:-1]) ** 2).sum()
    if ny > 1:
        out = out + ((g[:, 1:] - g[:, :-1]) ** 2).sum()
    if nz > 1:
        out = out + ((g[:, :, 1:] - g[:, :, :-1]) ** 2).sum()
    return out


def _model_stack(grain, H, alignment, hkl, resolution, psi_deg, detector,
                 *, supersample, psf_px, model):
    eye = torch.eye(3, dtype=H.dtype, device=H.device).expand_as(H)
    Q = local_Q(eye + H, grain.field.reference_G(hkl), model=model)
    st = topograph_stack(grain, alignment, psi_deg, detector=detector, hkl=hkl,
                         resolution=resolution, supersample=supersample,
                         Q_sample=Q)
    return apply_psf(st, psf_px) if psf_px else st


def fit_deformation_field(
    observed,
    grain,
    alignments,
    psi_deg,
    *,
    detector,
    hkls,
    resolutions,
    model: str = "exact",
    init_H=None,
    lambda_smooth: float = 1e-2,
    profile_scales: bool = True,
    psf_px: float = 0.0,
    steps: int = 300,
    lr: float = 1e-3,
    lr_schedule: str = "cosine",
    return_best: bool = True,
    supersample: int = 1,
    optimizer: str = "adam",
    log_every: int = 0,
):
    """Recover per-voxel ``F(r)`` from TT topograph stacks. Returns ``(F, info)``.

    ``observed``, ``alignments``, ``hkls`` and ``resolutions`` are lists of equal
    length, one entry per reflection; each ``observed[r]`` is a stack
    ``(n_psi, nu, nv)``.

    ``F = I + H`` with ``H`` free per voxel (``(N, 3, 3)``), so the reference
    lattice is the natural zero and ``lambda_smooth`` pulls towards a uniform
    grain rather than towards ``F = 0``, which would be a nonsense state.

    The data term is an **unweighted** sum of squares. With ``psf_px`` set the
    noise is spatially correlated, so the fit stays unbiased but its chi-squared
    and any error bar derived from it are optimistic -- scale by
    ``1/sqrt(midas_dct_tt.effective_dof(shape, psf_px, mode="mean"))``.

    ``info`` carries ``loss``, ``data``, ``reg`` and ``history``. Report
    ``lambda_smooth`` alongside any result: with a missing cone and only 3 of 9
    components per reflection, the prior is doing real work and a number quoted
    without it is not reproducible.

    **Convergence.** At a fixed learning rate Adam does not settle: measured on
    the 12-D campaign (``dev/paper/runs/c2/verify_p10a/probe9.log``) the iterate
    returned after 600 steps was **19-62% worse in loss than the best iterate the
    same run had already visited**, with the argmin as early as step 207 of 600.
    Returning a point worse than one already evaluated is not a modelling
    choice, so two defaults changed:

    * ``lr_schedule="cosine"`` anneals ``lr`` to zero over ``steps``
      (``"none"`` restores the old fixed rate). A floor of ``lr/100`` was tried
      first and is not enough: it leaves the last steps moving at the campaign's
      full ``3e-4`` and the overshoot survives;
    * ``return_best=True`` returns the lowest-loss iterate actually evaluated
      rather than whatever the last update landed on.

    Both are needed. The schedule stops the oscillation; ``return_best`` is the
    guarantee, since no schedule proves the final step was downhill. Note the
    old code could not even report the returned point's loss: with Adam the
    closure is evaluated *before* ``opt.step()``, so the returned ``H`` had never
    been scored and ``info["loss"]`` belonged to the previous iterate.

    **The cosine default is a trade-off, not a free win, and it is
    regime-dependent.** Annealing halves the effective learning rate over the
    run, so it helps a fit that is OSCILLATING around its minimum and *hurts*
    one that is merely TRUNCATED and still descending. Measured on the sibling
    primitive ``midas_invert.fit``, switching it on cost
    ``reconstruct_differentiable`` **0.973 -> 0.916** in dice and
    ``fit_uniform_deformation`` **7.03e-4 -> 9.57e-4** in ``max|F - F_true|``;
    both call sites are pinned to ``lr_schedule="none"`` for that reason. It is
    the right default *here* because this fit is squarely in the oscillating
    regime -- at fixed ``lr`` the 600-step iterate sat 0.27-3.3x above its own
    best (``dev/paper/runs/c2/logs_P16.txt``).

    ``tail_improvement`` is the number that says which regime you are in: large
    means truncated (lengthen the run, and prefer ``"none"``), ~0 alongside a
    large ``final_over_min`` at fixed ``lr`` means oscillating (anneal). Check it
    before assuming this default suits a new problem.

    ``info`` therefore also carries ``loss_min``, ``argmin``, ``final_loss``,
    ``final_over_min`` (relative excess of the last iterate over the best),
    ``tail_improvement`` (relative loss decrease over the last 10% of steps),
    ``grad_max_final`` and ``settled``.

    ``settled`` is **not** a convergence certificate and is deliberately not
    named one. It is ``True`` when ``final_over_min`` and ``tail_improvement``
    are both below ``1e-3``, i.e. *the run stopped moving inside its own
    budget*. Under a cosine schedule that is partly manufactured: annealing
    ``lr`` to zero drives ``tail_improvement`` to zero whether or not the
    iterate is near a stationary point, so a truncated run can be ``settled``.
    Read it with ``grad_max_final`` (the gradient at the last iterate, not
    at the returned one), and treat the physical test as the one that
    matters -- whether the data residual reaches the measured noise floor,
    which is what :func:`select_lambda_discrepancy` asks. On the 12-D campaign
    it *refuses*, and that refusal is a result, not a nuisance.
    """
    if lr_schedule not in ("cosine", "none"):
        raise ValueError(
            f"lr_schedule must be 'cosine' or 'none', got {lr_schedule!r}"
        )
    if not (len(alignments) == len(hkls) == len(resolutions) == len(observed)):
        raise ValueError(
            "alignments, hkls, resolutions and observed must be the same length "
            f"(got {len(alignments)}, {len(hkls)}, {len(resolutions)}, "
            f"{len(observed)})"
        )
    if grain.field is None:
        raise ValueError(
            "grain has no DeformationField; attach one (attach_uniform_field) so "
            "the reference lattice and reference_G are defined"
        )

    n = grain.n_voxels
    dt = grain.positions.dtype
    dev = grain.positions.device
    H = (torch.zeros(n, 3, 3, dtype=dt, device=dev) if init_H is None
         else init_H.clone().to(dtype=dt, device=dev))
    H.requires_grad_(True)

    obs = [o.to(dtype=dt, device=dev) for o in observed]
    scale = [o.abs().amax().clamp_min(1e-30) for o in obs]

    opt = (torch.optim.Adam([H], lr=lr) if optimizer == "adam"
           else torch.optim.LBFGS([H], lr=lr, max_iter=20))
    # eta_min=0: anneal the whole way. A floor of lr/100 still leaves the last
    # steps moving at the campaign's full 3e-4 and the overshoot survives.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=max(steps, 1), eta_min=0.0)
             if lr_schedule == "cosine" else None)

    history = []

    def closure():
        opt.zero_grad(set_to_none=True)
        data = H.new_zeros(())
        for r, (al, hkl, res, o) in enumerate(zip(alignments, hkls, resolutions, obs)):
            m = _model_stack(grain, H, al, hkl, res, psi_deg, detector,
                             supersample=supersample, psf_px=psf_px, model=model)
            if profile_scales:
                a = intensity_scales(m.reshape(1, -1), o.reshape(1, -1))
                m = m * a.reshape(())
            data = data + (((m - o) / scale[r]) ** 2).mean()
        reg = (lambda_smooth * smoothness_penalty(H, grain.shape)
               if lambda_smooth else H.new_zeros(()))
        loss = data + reg
        loss.backward()
        closure.data = float(data.detach())
        closure.reg = float(reg.detach())
        # Snapshot inside the closure, not in the loop: this is the only place
        # where the loss and the H that produced it are guaranteed to be the
        # same point. LBFGS evaluates the closure several times per step and
        # Adam evaluates it before the update, so a loop-level snapshot pairs a
        # loss with the wrong iterate under both.
        lv = float(loss.detach())
        if lv < closure.best_loss:
            closure.best_loss = lv
            closure.best_data = closure.data
            closure.best_reg = closure.reg
            closure.best_step = closure.step
            closure.best_H = H.detach().clone()
        return loss

    closure.best_loss = float("inf")
    closure.best_data = closure.best_reg = float("nan")
    closure.best_step = -1
    closure.best_H = None
    closure.step = 0

    for i in range(steps):
        closure.step = i
        loss = opt.step(closure) if optimizer == "lbfgs" else None
        if optimizer == "adam":
            loss = closure()
            opt.step()
        if sched is not None:
            sched.step()
        history.append(float(loss.detach()))
        if log_every and i % log_every == 0:
            print(f"  step {i:4d}  loss {history[-1]:.6e}  "
                  f"data {closure.data:.6e}  reg {closure.reg:.6e}")

    best_loss = closure.best_loss
    use_best = return_best and closure.best_H is not None
    if use_best:
        H_out = closure.best_H
        loss_out, data_out, reg_out = best_loss, closure.best_data, closure.best_reg
    else:
        H_out = H.detach()
        loss_out = history[-1] if history else float("nan")
        data_out = getattr(closure, "data", float("nan"))
        reg_out = getattr(closure, "reg", float("nan"))

    final = history[-1] if history else float("nan")
    denom = abs(best_loss) if history and best_loss != 0.0 else float("nan")
    final_over_min = (final - best_loss) / denom if history else float("nan")
    # Loss decrease over the last 10% of steps, relative to the best. A run that
    # is still improving here was truncated, however smooth its tail looks.
    tail = float("nan")
    if len(history) >= 20:
        k = max(1, len(history) // 10)
        tail = (history[-k - 1] - final) / denom
    settled = bool(history) and (final_over_min <= 1e-3) and (
        tail <= 1e-3 if len(history) >= 20 else False)
    # Named for the point it is measured at. H.grad holds the gradient of the
    # LAST closure evaluation, which with return_best is not the iterate handed
    # back -- calling it `grad_max` would invite exactly the misreading
    # ("gradient at the solution") this whole change exists to stop.
    grad_max_final = (float(H.grad.abs().max()) if H.grad is not None
                      else float("nan"))

    eye = torch.eye(3, dtype=dt, device=dev).expand(n, 3, 3)
    return (eye + H_out).detach(), {
        "loss": loss_out,
        "data": data_out,
        "reg": reg_out,
        "history": history,
        "lambda_smooth": lambda_smooth,
        "loss_min": best_loss if history else float("nan"),
        "argmin": closure.best_step,
        "final_loss": final,
        "final_over_min": final_over_min,
        "tail_improvement": tail,
        "grad_max_final": grad_max_final,
        "settled": settled,
        "lr_schedule": lr_schedule,
        "returned_best": use_best,
    }


def cross_validate_lambda(
    observed,
    grain,
    alignments,
    psi_deg,
    *,
    detector,
    hkls,
    resolutions,
    lambdas,
    n_folds: int = 3,
    **fit_kw,
):
    """Choose ``lambda_smooth`` by **held-out-view** cross-validation.

    Returns ``(best_lambda, table)`` where ``table`` is a list of
    ``{"lambda", "train", "test"}`` -- mean data residual on the fitted and the
    held-out ``psi`` views.

    Why held-out *views* and not held-out voxels. The regulariser is a prior on
    spatial smoothness, so holding out voxels would test the prior against
    itself: a smoother field trivially predicts its own neighbours better and CV
    would pick the largest ``lambda`` every time. Views are the honest split,
    because ``H`` is not a function of ``psi`` -- a field fitted to one set of
    projection angles either predicts the others or it does not, and over-smoothing
    is penalised there just as over-fitting is.

    This exists because selecting ``lambda`` by correlation against a known
    planted field -- which is how ``RESULTS_field_inverse.md`` reported it -- is
    **not available on real data**, and a hyperparameter tuned on truth is not a
    result. The interesting question is whether this selector, which never sees
    the truth, lands on the same value.

    Folds are strided (``i::n_folds``) rather than contiguous blocks: a TT scan is
    periodic in ``psi``, so a contiguous held-out block is a *missing wedge* and
    would measure extrapolation across a gap instead of interpolation, conflating
    the choice of ``lambda`` with the missing-cone problem.

    Per-reflection intensity scales are profiled on the **training** views and
    applied unchanged to the held-out ones; re-profiling on the test fold would
    let the nuisance parameter absorb genuine prediction error.
    """
    psi = torch.as_tensor(psi_deg)
    n = psi.shape[0]
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"n_folds must be in [2, {n}], got {n_folds}")

    table = []
    for lam in lambdas:
        tr_acc, te_acc = 0.0, 0.0
        for f in range(n_folds):
            te = torch.arange(f, n, n_folds)
            tr = torch.tensor([i for i in range(n) if (i - f) % n_folds != 0])
            obs_tr = [o[tr] for o in observed]
            obs_te = [o[te] for o in observed]

            H_fit, info = fit_deformation_field(
                obs_tr, grain, alignments, psi[tr], detector=detector, hkls=hkls,
                resolutions=resolutions, lambda_smooth=lam, **fit_kw)
            H = H_fit - torch.eye(3, dtype=H_fit.dtype,
                                  device=H_fit.device).expand_as(H_fit)

            profile = fit_kw.get("profile_scales", True)
            psf = fit_kw.get("psf_px", 0.0)
            model = fit_kw.get("model", "exact")
            ss = fit_kw.get("supersample", 1)
            with torch.no_grad():
                te_loss = 0.0
                for al, hkl, res, o_tr, o_te in zip(alignments, hkls, resolutions,
                                                    obs_tr, obs_te):
                    if profile:
                        m_tr = _model_stack(grain, H, al, hkl, res, psi[tr], detector,
                                            supersample=ss, psf_px=psf, model=model)
                        a = intensity_scales(m_tr.reshape(1, -1), o_tr.reshape(1, -1))
                    m_te = _model_stack(grain, H, al, hkl, res, psi[te], detector,
                                        supersample=ss, psf_px=psf, model=model)
                    if profile:
                        m_te = m_te * a.reshape(())
                    sc = o_te.abs().amax().clamp_min(1e-30)
                    te_loss += float((((m_te - o_te) / sc) ** 2).mean())
            tr_acc += info["data"]
            te_acc += te_loss
        table.append({"lambda": lam, "train": tr_acc / n_folds,
                      "test": te_acc / n_folds})

    best = min(table, key=lambda r: r["test"])["lambda"]
    return best, table


def select_lambda_discrepancy(lambdas, data_residuals, noise_floor, *, tol: float = 0.10):
    """Morozov discrepancy: the **largest** ``lambda`` still fitting to the noise.

    Returns the selected ``lambda``. Rationale: below the noise floor there is
    nothing left to fit, so among all ``lambda`` that reach it, the largest is the
    least over-fitted; any ``lambda`` whose residual sits *above* the floor is
    over-smoothed and is discarding real signal.

    **This is the selector that works, and it beats the obvious alternatives by a
    lot.** Measured against the oracle (the ``lambda`` maximising correlation with
    a known planted field), on the same data:

    ===============================  ==========  ==========
    selector                         18 views    36 views
    ===============================  ==========  ==========
    held-out-view cross-validation   66%         50%
    L-curve corner                   66%         50%
    **discrepancy, pure-noise floor  100%        100%**
    ===============================  ==========  ==========

    .. warning::
       **``noise_floor`` must be PURE noise.** The tempting estimate -- the data
       residual evaluated at the true field -- also contains forward-model
       discretisation mismatch, measured here as **36%** of that quantity
       (9.37e-07 mismatch vs 1.68e-06 noise). Inflating the floor by that much
       makes every ``lambda`` "consistent with noise", the criterion becomes
       vacuous, and it silently selects the *most* over-smoothed candidate --
       which is exactly the failure that made this look useless on the first
       attempt (55% of oracle, worse than doing nothing clever).

       On real data estimate it from **photon counts** or **repeat frames**.
       Do **not** estimate it from the residual of your own best fit: that is
       circular and will reproduce the inflated-floor failure.

    Why this succeeds where cross-validation fails: ``lambda`` moves the solution
    largely within the **null space** of the operator, which by construction does
    not change the data fit -- so a criterion built on comparing data fits
    (CV, L-curve) is blind to precisely the thing being chosen. The discrepancy
    principle instead injects independent information (how well the data *could*
    be fitted), and that is what breaks the degeneracy.

    ``tol`` is the fractional slack on the floor; 10% recovered the oracle at both
    view counts tested, 5% recovered it at 18 views and 79% of it at 36. The
    result is sensitive to this, so state the value used.
    """
    lam = list(lambdas)
    res = list(data_residuals)
    if len(lam) != len(res):
        raise ValueError(f"lambdas ({len(lam)}) and data_residuals ({len(res)}) differ")
    if not lam:
        raise ValueError("no candidates")
    if noise_floor <= 0:
        raise ValueError(f"noise_floor must be > 0, got {noise_floor}")
    thresh = (1.0 + tol) * noise_floor
    ok = [l for l, r in zip(lam, res) if r <= thresh]
    if not ok:
        raise ValueError(
            f"no lambda reaches the noise floor {noise_floor:.4e} within {tol:.0%}; "
            f"best residual is {min(res):.4e}. Either the floor is underestimated "
            "or the model cannot fit these data."
        )
    return max(ok)


# --- scoring a field inverse without fooling yourself -----------------------
def affine_basis(positions: torch.Tensor) -> torch.Tensor:
    """``(N, 4)`` basis ``[1, x, y, z]`` for the affine part of a per-voxel field.

    Combined with 9 tensor components this spans **36 degrees of freedom**: a
    uniform ``F`` plus a linear gradient of ``F``.
    """
    ones = torch.ones(positions.shape[0], 1, dtype=positions.dtype,
                      device=positions.device)
    return torch.cat([ones, positions], dim=1)


def decompose_affine(H: torch.Tensor, positions: torch.Tensor, weights=None):
    """Split a per-voxel field into ``(affine_part, non_affine_residual)``.

    **Use this before claiming a per-voxel reconstruction.** A field inverse that
    recovers only the mean and linear gradient of ``F`` has recovered **36
    numbers**, not ``9N`` -- and because a smooth planted field is usually
    dominated by its affine part, the overall correlation can look high while
    nothing beyond affine has been retrieved. Measured on this package's own first
    12-D demonstration: overall correlation 0.567, affine parts 0.662,
    **non-affine residuals 0.0056 (p = 0.83)**. The headline number was the affine
    part wearing a per-voxel costume.

    ``weights`` (e.g. voxel occupancy ``chi``) restricts the fit to where the
    grain actually is. Strongly recommended for soft-boundary grains: in that same
    demonstration **62% of the correlation came from voxels with chi < 0.01** --
    vacuum, where the recovered field is pure prior extrapolation.

    ``H`` is ``(N, 3, 3)``, ``positions`` ``(N, 3)``. Returns two ``(N, 3, 3)``
    tensors summing to ``H``.
    """
    A = affine_basis(positions)
    w = (torch.ones(H.shape[0], dtype=H.dtype, device=H.device)
         if weights is None else weights.to(H.dtype))
    sw = torch.sqrt(w.clamp_min(0)).unsqueeze(-1)
    Aw = A * sw
    Y = H.reshape(H.shape[0], 9) * sw
    coef = torch.linalg.lstsq(Aw, Y).solution              # (4, 9)
    fit = (A @ coef).reshape(-1, 3, 3)
    return fit, H - fit


def weighted_corr(a: torch.Tensor, b: torch.Tensor, weights=None) -> float:
    """Weighted Pearson correlation between two fields, flattened.

    ``weights`` is per-voxel and is broadcast over tensor components, so passing
    occupancy ``chi`` scores the grain rather than the surrounding vacuum.
    """
    x, y = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    if weights is None:
        w = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
    else:
        w = weights.to(x.dtype).clamp_min(0)
    W = w.unsqueeze(-1).expand_as(x).reshape(-1)
    xf, yf = x.reshape(-1), y.reshape(-1)
    sw = W.sum().clamp_min(1e-30)
    xm, ym = (W * xf).sum() / sw, (W * yf).sum() / sw
    dx, dy = xf - xm, yf - ym
    num = (W * dx * dy).sum()
    den = torch.sqrt((W * dx * dx).sum() * (W * dy * dy).sum()).clamp_min(1e-30)
    return float(num / den)
