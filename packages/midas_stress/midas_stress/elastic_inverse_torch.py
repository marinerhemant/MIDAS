"""Torch-native joint fit of single-crystal stiffness and isotropic d_0 bias.

The non-torch :func:`midas_stress.elastic_inverse.fit_single_crystal_stiffness`
handles the ``d_0`` bias by alternating: fix ``C``, project the unloaded-stage
residual onto the isotropic direction to update ``eps_iso``; subtract it from
every stage's strain; re-fit ``C`` by linear least-squares; repeat.  The two
sub-fits commute approximately because ``eps_iso`` enters every stage as a
projection along ``<C_lab> @ {I}`` and ``C`` is otherwise linear in the
corrected strains.

The joint problem is *bilinear* in ``(c, eps_iso)``:

    r^(i)(c, eps_iso) = (A^(i) - eps_iso * Q^(i)) @ c - b^(i)

where ``A^(i)`` is the Hill stage matrix from measured strains, ``Q^(i)`` is
the per-stage matrix whose column ``k`` is ``(sum_g w_g U_g^T P_k U_g) @ {I}``,
and ``b^(i)`` is the Voigt-flattened applied stress.  Torch L-BFGS on the
stacked sum-of-squares loss is the natural one-shot solution and exposes the
joint Hessian (and hence a joint ``(c, eps_iso)`` covariance) at the optimum.

Compared to alternating:
  * agrees to numerical precision when the orthogonality argument holds
    (the common case);
  * gives a joint covariance instead of two marginal covariances;
  * remains stable when ``eps_iso`` and the principal Cij are not nearly
    orthogonal (highly textured samples, strong anisotropy ratio).

The ``eps_iso`` correction is applied to **every** stage's residual — this
matches the alternating fit, which subtracts ``eps_iso`` from every stage's
strain after estimating it from the unloaded stage.  ``fit_eps_iso=False``
pins ``eps_iso=0`` for cases without an unloaded reference.

Runs on CPU, CUDA and MPS; autograd-friendly (used by the LOO routines).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from .elastic_inverse import (
    _aggregate_basis_rotated,
    _basis_dual,
    _initial_stiffness,
    _prep_stages,
    build_stage_matrix,
    fit_single_crystal_stiffness,
    symmetry_parameterisation,
)
from .tensor import tensor_to_voigt


_I_VOIGT = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def _resolve_device(stages_prepped: list[dict], device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    for s in stages_prepped:
        for key in ("orient", "strain", "applied_stress"):
            v = s.get(key)
            if isinstance(v, torch.Tensor):
                return v.device
    return torch.device("cpu")


def _build_stage_terms(
    stages_prepped: list[dict],
    P_stack_t: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict]:
    """Build (A, q, b, is_unloaded) for every stage as torch tensors.

    A     : (6, N_c)  Hill stage matrix from measured per-grain strain
    q     : (6, N_c)  d_0 response matrix; column k = aggregate(P_k) @ {I}
    b     : (6,)      Voigt-flattened applied stress
    """
    I_voigt = torch.tensor(_I_VOIGT, dtype=dtype, device=device)
    terms = []
    for s in stages_prepped:
        orient = torch.as_tensor(s["orient"], dtype=dtype, device=device)
        strain = torch.as_tensor(s["strain"], dtype=dtype, device=device)
        weights = torch.as_tensor(s["weights"], dtype=dtype, device=device)
        applied = torch.as_tensor(s["applied_stress"], dtype=dtype, device=device)

        A = build_stage_matrix(orient, strain, weights, P_stack_t)      # (6, N_c)
        Pbar = _aggregate_basis_rotated(orient, weights, P_stack_t)     # (N_c, 6, 6)
        q = torch.einsum("kij,j->ik", Pbar, I_voigt)                    # (6, N_c)
        b = tensor_to_voigt(applied)                                    # (6,)
        terms.append({
            "A": A, "q": q, "b": b,
            "is_unloaded": s["is_unloaded"],
        })
    return terms


def _joint_loss(
    theta: torch.Tensor,
    terms: list[dict],
    N_c: int,
    eps_iso_stage_mask: Sequence[bool],
    component_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum-of-squares residual stacked over stages.

    theta = concat([c (N_c,), eps_iso (1,)]).  eps_iso_stage_mask says
    whether ``eps_iso`` contributes to a given stage's residual (True) or
    not (False).  Optional ``component_weights`` is a (6,) tensor that
    multiplies the per-Mandel-component squared residual; defaults to
    uniform.  Used by the IRLS path to weight each equilibrium-equation
    component by its inverse-variance from the data.
    """
    c = theta[:N_c]
    eps_iso = theta[N_c]
    total = theta.new_zeros(())
    for use_eps, term in zip(eps_iso_stage_mask, terms):
        Aeff = term["A"] - eps_iso * term["q"] if use_eps else term["A"]
        r = Aeff @ c - term["b"]
        if component_weights is None:
            total = total + (r * r).sum()
        else:
            total = total + (component_weights * r * r).sum()
    return total


def _joint_loss_single_stage(
    theta: torch.Tensor,
    term: dict,
    N_c: int,
    use_eps_iso: bool,
) -> torch.Tensor:
    """Per-stage contribution to the joint loss (for LOO bookkeeping)."""
    c = theta[:N_c]
    eps_iso = theta[N_c]
    Aeff = term["A"] - eps_iso * term["q"] if use_eps_iso else term["A"]
    r = Aeff @ c - term["b"]
    return (r * r).sum()


def _pd_penalty(
    c: torch.Tensor,
    P_stack_t: torch.Tensor,
    pd_floor: float,
    pd_weight: float,
) -> torch.Tensor:
    """Soft barrier on the spectrum of ``C = sum_k c_k * P_k``.

    Penalty is ``pd_weight * sum_i max(0, pd_floor - lambda_i)^2`` where
    the eigenvalues are computed via :func:`torch.linalg.eigvalsh` on the
    symmetric Mandel stiffness.  Returns a zero tensor when ``pd_weight``
    is zero so the optimiser sees a clean no-op.

    PD-ness of the 6x6 Mandel stiffness is the Born stability criterion
    for every crystal system (Mouhat & Coudert, 2014), so this single
    penalty subsumes the system-specific Cauchy inequalities
    (e.g.cubic ``C11 > |C12|``).
    """
    if pd_weight <= 0.0:
        return c.new_zeros(())
    C = torch.einsum("k,kij->ij", c, P_stack_t)
    C_sym = 0.5 * (C + C.transpose(-1, -2))
    eigs = torch.linalg.eigvalsh(C_sym)
    violation = torch.clamp(pd_floor - eigs, min=0.0)
    return pd_weight * (violation * violation).sum()


def fit_joint_d0_stiffness(
    stages: list[dict],
    symmetry: str,
    *,
    fit_eps_iso: bool = True,
    enforce_pd: bool = False,
    pd_floor: float = 0.0,
    pd_weight: float = 1e6,
    irls_weights: bool = False,
    irls_max_iter: int = 30,
    irls_tol: float = 1e-6,
    initial_stiffness: Optional[np.ndarray] = None,
    initial_eps_iso: float = 0.0,
    material_hint: Optional[str] = None,
    min_confidence: float = 0.0,
    cond_threshold: float = 1e3,
    max_iter: int = 50,
    tol: float = 1e-10,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Joint d_0 + C_ij fit via torch L-BFGS.

    Parameters
    ----------
    stages, symmetry, initial_stiffness, material_hint, min_confidence,
    cond_threshold : see :func:`midas_stress.elastic_inverse.fit_single_crystal_stiffness`.
    fit_eps_iso : bool
        If True and at least one stage carries ``is_unloaded=True``, fit
        ``eps_iso`` jointly with ``C``.  The ``eps_iso`` correction is
        applied to **every** stage's residual (matching the alternating
        fit's "subtract from every stage" semantics).  If False, pin
        ``eps_iso=0`` and fit ``C`` only.
    enforce_pd : bool
        If True, add a soft penalty
        ``pd_weight * sum_i max(0, pd_floor - lambda_i)^2`` to the loss,
        where ``lambda_i`` are eigenvalues of the symmetric Mandel
        stiffness ``C = sum_k c_k * P_k``.  PD-ness of the 6x6 Mandel
        form is the Born stability criterion (Mouhat & Coudert, 2014),
        so this subsumes the system-specific Cauchy inequalities
        (e.g.\\ cubic ``C11 > |C12|``).  Default off; enable when the
        unconstrained joint fit returns a non-PD stiffness, typically
        on ill-conditioned load palettes or noise-dominated data.
    pd_floor : float
        Minimum allowed eigenvalue when ``enforce_pd=True``.  Default 0
        (strictly non-negative); set to a small positive value (e.g.
        ``1e-3`` in GPa units) to keep the fit strictly inside the PD
        cone for downstream operations such as compliance inversion.
    pd_weight : float
        Weight multiplying the squared PD violation.  Default 1e6 is
        large enough to enforce the constraint to within ``~1e-3``
        eigenvalue units against typical residual-noise magnitudes; the
        fit reports ``pd_violation`` so the caller can verify the
        penalty was strong enough.
    irls_weights : bool
        If True, after the uniform-weight joint fit converges, run an
        outer iteratively-reweighted least-squares loop that estimates
        per-Mandel-component equilibrium-equation noise from the
        residual std across stages and refits with weights ``1/sigma^2``.
        This is a Gauss-Newton maximum-likelihood estimator under a
        per-component Gaussian noise model whose noise is data-estimated;
        it is the principled mitigation for the within-rank conditioning
        ceiling of single-axis load campaigns
        (see Paper~III \\S\\ref{sec:exp-conditioning}).  Default off.
    irls_max_iter, irls_tol : int, float
        Maximum outer IRLS iterations and relative-change tolerance on
        ``c``.  Defaults converge inside 30 iterations on the synthetic
        and experimental cases in Paper~III.
    initial_eps_iso : float
        Starting value for the ``eps_iso`` parameter.  Default 0.
    max_iter : int
        Maximum outer iterations of the L-BFGS driver (each calls the
        L-BFGS optimiser with its own inner ``max_iter=20``).
    tol : float
        Relative change in ``theta`` below which the joint fit is
        considered converged.
    device : str, optional
        Torch device.  Inferred from input tensors if any are torch
        tensors; falls back to CPU.
    dtype : torch.dtype
        Float precision; float64 strongly recommended for stiffness fits.

    Returns
    -------
    dict with the same keys as ``fit_single_crystal_stiffness`` plus:
        - ``joint_covariance`` : (N_c+1, N_c+1) covariance of ``[c, eps_iso]``
        - ``eps_iso_se``       : float
        - ``joint_lbfgs_iters``: int (outer iterations)
        - ``device``           : str
    """
    names, P_stack_np = symmetry_parameterisation(symmetry)
    N_c = len(names)
    prepped = _prep_stages(stages, min_confidence)
    if not prepped:
        raise ValueError("stages must contain at least one entry")

    dev = _resolve_device(prepped, device)
    P_stack_t = torch.as_tensor(P_stack_np, dtype=dtype, device=dev)
    terms = _build_stage_terms(prepped, P_stack_t, dtype, dev)

    unloaded_present = any(s["is_unloaded"] for s in prepped)
    # eps_iso is identified only when (a) we asked to fit it and (b) there
    # is at least one unloaded stage to pin against.  Without an unloaded
    # stage the joint problem can still be solved with eps_iso fitted from
    # the loaded stages alone, but it then absorbs any load-cell offset
    # into eps_iso — we conservatively pin eps_iso=0 in that case.
    eps_iso_identified = bool(fit_eps_iso) and unloaded_present
    if eps_iso_identified:
        # Apply eps_iso to every stage's residual (matches alternating).
        mask = [True] * len(prepped)
    else:
        mask = [False] * len(prepped)

    # Seed (c, eps_iso) from the closed-form alternating fit.  This is
    # essential: the joint loss has gradient magnitudes that differ by
    # ~6 orders between c (~1e-4) and eps_iso (~1e+2), so L-BFGS from a
    # cold start would optimise eps_iso and freeze c in place.  The
    # alternating fit lands in the correct basin in O(few) iterations
    # and is cheap; L-BFGS then polishes.
    A_rows = [t["A"] for t in terms]
    b_rows = [t["b"] for t in terms]
    A_stacked = torch.cat(A_rows, dim=0)               # (6 * Nstage, N_c)
    b_stacked = torch.cat(b_rows, dim=0)               # (6 * Nstage,)
    if A_stacked.shape[0] < N_c:
        raise ValueError(
            f"Under-determined: {A_stacked.shape[0]} equations for {N_c} "
            f"unknowns ({names}). Add more load stages."
        )

    seed = fit_single_crystal_stiffness(
        stages, symmetry=symmetry,
        fit_eps_iso=eps_iso_identified,
        initial_stiffness=initial_stiffness,
        material_hint=material_hint,
        min_confidence=min_confidence,
        cond_threshold=cond_threshold,
        max_iter=20,
        tol=1e-8,
        method="hill",
    )
    c_init_np = np.array([seed["cij"][n] for n in names])
    eps_iso_init_val = float(seed["eps_iso"]) if eps_iso_identified else float(initial_eps_iso)
    c_init = torch.as_tensor(c_init_np, dtype=dtype, device=dev)

    # Preconditioning: scale parameters to comparable magnitudes so the
    # BFGS Hessian estimate is well-behaved.  c is O(100 GPa); eps_iso is
    # O(1e-4).  Internally we work in (c_scaled, eps_iso_scaled) with
    # c_scaled = c / c_scale, eps_iso_scaled = eps_iso / eps_iso_scale.
    c_scale = max(float(np.linalg.norm(c_init_np, ord=np.inf)), 1.0)
    eps_iso_scale = 1e-4
    c_init_s = c_init / c_scale
    eps_init_s = torch.tensor([eps_iso_init_val / eps_iso_scale],
                              dtype=dtype, device=dev)
    theta = torch.cat([c_init_s, eps_init_s]).requires_grad_(True)

    scale_vec = torch.cat([
        torch.full((N_c,), c_scale, dtype=dtype, device=dev),
        torch.tensor([eps_iso_scale], dtype=dtype, device=dev),
    ])

    active_pd_weight = float(pd_weight) if enforce_pd else 0.0

    def total_loss(th_phys):
        L = _joint_loss(th_phys, terms, N_c, mask)
        if active_pd_weight > 0:
            L = L + _pd_penalty(
                th_phys[:N_c], P_stack_t, pd_floor, active_pd_weight,
            )
        return L

    def scaled_loss(th):
        return total_loss(th * scale_vec)

    optimiser = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=50,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
        history_size=50,
    )

    def closure():
        optimiser.zero_grad()
        loss = scaled_loss(theta)
        loss.backward()
        return loss

    prev = theta.detach().clone()
    n_outer = 0
    for n_outer in range(1, max_iter + 1):
        optimiser.step(closure)
        delta = theta.detach() - prev
        denom = max(prev.norm().item(), 1e-30)
        if delta.norm().item() / denom < tol:
            break
        prev = theta.detach().clone()

    # -----------------------------------------------------------------
    # Optional IRLS: re-fit with per-component inverse-variance weights
    # estimated from the residual std across stages.  Iterate until c
    # stabilises.  This is the principled mitigation for the within-
    # rank conditioning ceiling under single-axis loading.
    # -----------------------------------------------------------------
    irls_iters_used = 0
    irls_weights_final = None
    if irls_weights:
        irls_floor = 1e-12          # numerical floor on per-component sigma
        with torch.no_grad():
            c_prev = (theta.detach() * scale_vec)[:N_c].clone()
        for irls_it in range(1, irls_max_iter + 1):
            # Per-component residual std across stages, at current theta.
            with torch.no_grad():
                theta_phys_now = theta.detach() * scale_vec
                c_now = theta_phys_now[:N_c]
                eps_now = theta_phys_now[N_c]
                resid_rows = []
                for use_eps, term in zip(mask, terms):
                    Aeff = term["A"] - eps_now * term["q"] if use_eps else term["A"]
                    r = Aeff @ c_now - term["b"]
                    resid_rows.append(r)
                resid = torch.stack(resid_rows, dim=0)              # (N_stage, 6)
                sigma_per_comp = resid.std(dim=0)                    # (6,)
                sigma_per_comp = torch.clamp(sigma_per_comp, min=irls_floor)
                comp_w = 1.0 / (sigma_per_comp * sigma_per_comp)    # (6,)
            irls_weights_final = comp_w.detach()

            # Re-define the loss with the new weights and re-run L-BFGS
            # from the current theta.  Re-create the optimiser because
            # the weight change invalidates the BFGS Hessian estimate.
            def total_loss_w(th_phys, _w=comp_w):
                L = _joint_loss(th_phys, terms, N_c, mask, component_weights=_w)
                if active_pd_weight > 0:
                    L = L + _pd_penalty(th_phys[:N_c], P_stack_t, pd_floor,
                                         active_pd_weight)
                return L

            def scaled_loss_w(th):
                return total_loss_w(th * scale_vec)

            opt_w = torch.optim.LBFGS(
                [theta],
                lr=1.0, max_iter=50,
                tolerance_grad=1e-14, tolerance_change=1e-16,
                line_search_fn="strong_wolfe", history_size=50,
            )

            def closure_w():
                opt_w.zero_grad()
                L = scaled_loss_w(theta)
                L.backward()
                return L
            prev_inner = theta.detach().clone()
            for inner in range(1, max_iter + 1):
                opt_w.step(closure_w)
                dthi = theta.detach() - prev_inner
                if dthi.norm().item() / max(prev_inner.norm().item(), 1e-30) < tol:
                    break
                prev_inner = theta.detach().clone()

            with torch.no_grad():
                c_new = (theta.detach() * scale_vec)[:N_c]
                rel = (c_new - c_prev).norm().item() / max(c_prev.norm().item(), 1e-30)
            irls_iters_used = irls_it
            if rel < irls_tol:
                break
            c_prev = c_new.clone()

    with torch.no_grad():
        theta_phys = theta.detach() * scale_vec
        c_vec_t = theta_phys[:N_c]
        eps_iso_t = theta_phys[N_c]
        # Data-only residual (the PD penalty is *not* a data residual; its
        # squared norm should not contribute to the variance estimate
        # because PD enforcement is a prior, not a measurement).
        residual_sq = _joint_loss(theta_phys, terms, N_c, mask).item()
        if active_pd_weight > 0:
            C_fit_t = torch.einsum("k,kij->ij", c_vec_t, P_stack_t)
            C_fit_sym = 0.5 * (C_fit_t + C_fit_t.transpose(-1, -2))
            eigs_fit = torch.linalg.eigvalsh(C_fit_sym)
            pd_violation = float(
                torch.clamp(pd_floor - eigs_fit, min=0.0).max().item()
            )
            min_eigenvalue = float(eigs_fit.min().item())
        else:
            pd_violation = 0.0
            C_fit_t = torch.einsum("k,kij->ij", c_vec_t, P_stack_t)
            C_fit_sym = 0.5 * (C_fit_t + C_fit_t.transpose(-1, -2))
            min_eigenvalue = float(torch.linalg.eigvalsh(C_fit_sym).min().item())

    # Joint Hessian and covariance.  N_eq = 6 * N_stage, N_params = N_c (+1 if id'd)
    N_eq = A_stacked.shape[0]
    if eps_iso_identified:
        N_params = N_c + 1
    else:
        N_params = N_c

    # Compute the Hessian in *physical* coordinates so the covariance is
    # in the units the user expects.  Scaling at the optimum is just a
    # change of variables; we evaluate H at theta_phys directly.
    def loss_for_hessian(th_phys):
        return _joint_loss(th_phys, terms, N_c, mask)

    H_full = torch.autograd.functional.hessian(loss_for_hessian, theta_phys)
    H_full = 0.5 * (H_full + H_full.transpose(-1, -2))

    if eps_iso_identified:
        H = H_full
    else:
        H = H_full[:N_c, :N_c]

    dof = max(N_eq - N_params, 1)
    sigma_sq = residual_sq / dof
    try:
        H_inv = torch.linalg.inv(H)
    except RuntimeError:
        H_inv = torch.linalg.pinv(H)
    # Loss is sum of squared residuals; the Gauss-Newton Hessian = 2 A_eff^T A_eff,
    # so cov = sigma_sq * inv(A_eff^T A_eff) = 2 * sigma_sq * inv(H).
    joint_cov_small = 2.0 * sigma_sq * H_inv

    if eps_iso_identified:
        joint_cov = joint_cov_small
    else:
        joint_cov = theta.new_zeros((N_c + 1, N_c + 1))
        joint_cov[:N_c, :N_c] = joint_cov_small

    # Marginal covariance of c is the top-left N_c x N_c block.
    cov_c = joint_cov[:N_c, :N_c]

    # Condition number of the *measured-strain* design matrix (matches the
    # paper's cond(A) reported by the alternating fit; useful for paper
    # parity and for the conditioning diagnostic in §sec:conditioning).
    cond_num = float(torch.linalg.cond(A_stacked).item())

    c_vec = c_vec_t.cpu().numpy()
    eps_iso = float(eps_iso_t.item())
    cov_c_np = cov_c.cpu().numpy()
    joint_cov_np = joint_cov.cpu().numpy()
    residual_norm = float(np.sqrt(max(residual_sq, 0.0)))

    C_fit = np.einsum("k,kij->ij", c_vec, P_stack_np)

    cij = {n: float(c_vec[i]) for i, n in enumerate(names)}
    cij_se = {
        n: float(np.sqrt(max(cov_c_np[i, i], 0.0)))
        for i, n in enumerate(names)
    }
    eps_iso_se = float(np.sqrt(max(joint_cov_np[N_c, N_c], 0.0)))

    return {
        "stiffness":          C_fit,
        "cij":                cij,
        "cij_se":             cij_se,
        "cij_names":          names,
        "covariance":         cov_c_np,
        "joint_covariance":   joint_cov_np,
        "compliance":         None,
        "sij":                None,
        "condition_number":   cond_num,
        "well_conditioned":   bool(cond_num < cond_threshold),
        "eps_iso":            eps_iso,
        "eps_iso_se":         eps_iso_se,
        "residual_norm":      residual_norm,
        "n_iter":             1,            # alternating count (single joint solve)
        "joint_lbfgs_iters":  int(n_outer),
        "symmetry":           symmetry.lower(),
        "method":             "hill",
        "device":             str(dev),
        "pd_enforced":        bool(enforce_pd),
        "pd_violation":       pd_violation,
        "min_eigenvalue":     min_eigenvalue,
        "irls_enabled":       bool(irls_weights),
        "irls_iters":         int(irls_iters_used),
        "irls_weights_final": (
            irls_weights_final.cpu().numpy() if irls_weights_final is not None else None
        ),
    }


def loo_influence_stages(
    fit_result: dict,
    stages: list[dict],
    symmetry: str,
    *,
    min_confidence: float = 0.0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """One-step Newton leave-one-stage-out estimates from the joint fit.

    For each load stage ``i``, returns the approximate ``(c, eps_iso)``
    that the joint fit would have produced if that stage were excluded:

        theta_loo^{(-i)} = theta_hat + (H - H_i)^{-1} @ g_i

    where ``H`` is the data Gauss-Newton Hessian (no PD penalty), ``H_i``
    is stage ``i``'s contribution and ``g_i`` is the per-stage gradient,
    all evaluated at the joint optimum ``theta_hat``.  This one-step
    Newton approximation is exact for a linear least-squares problem
    and tight at the optimum of the bilinear joint loss.  Useful for
    quoting a per-Cij leave-one-out standard error as part of the
    paper's operational uncertainty bracket
    (see Paper~III \\S\\ref{sec:exp-conditioning}).

    Cost is ``O(N_stages * (N_c + 1)^3)``, replacing ``N_stages`` full
    refits that each spend tens of L-BFGS iterations.

    Parameters
    ----------
    fit_result : dict
        Output of :func:`fit_joint_d0_stiffness`.
    stages : list[dict]
        Same stages list that produced ``fit_result``.
    symmetry : str
        Same symmetry passed to the fit.
    min_confidence, device, dtype : as in :func:`fit_joint_d0_stiffness`.

    Returns
    -------
    dict with keys:
        - ``cij_loo``        : (N_stage, N_c) per-LOO Cij vectors
        - ``eps_iso_loo``    : (N_stage,) per-LOO eps_iso
        - ``cij_loo_std``    : (N_c,) per-Cij standard deviation across LOO fits
        - ``eps_iso_loo_std``: float, std of eps_iso across LOO fits
        - ``cij_names``      : list[str]
        - ``stage_indices``  : list[int]
    """
    names, P_stack_np = symmetry_parameterisation(symmetry)
    N_c = len(names)
    prepped = _prep_stages(stages, min_confidence)
    if not prepped:
        raise ValueError("stages must contain at least one entry")
    if len(prepped) < 2:
        raise ValueError("LOO requires at least 2 stages")

    dev = _resolve_device(prepped, device)
    P_stack_t = torch.as_tensor(P_stack_np, dtype=dtype, device=dev)
    terms = _build_stage_terms(prepped, P_stack_t, dtype, dev)

    # Reconstruct theta_hat from the fit_result dict.
    c_hat = torch.as_tensor(
        [fit_result["cij"][n] for n in names], dtype=dtype, device=dev,
    )
    eps_iso_hat = torch.tensor(
        float(fit_result["eps_iso"]), dtype=dtype, device=dev,
    ).unsqueeze(0)
    theta_hat = torch.cat([c_hat, eps_iso_hat])

    unloaded_present = any(s["is_unloaded"] for s in prepped)
    eps_iso_identified = unloaded_present and abs(float(fit_result["eps_iso"])) > 0.0
    mask = [True] * len(prepped) if eps_iso_identified else [False] * len(prepped)

    # Per-stage Hessian and gradient at the joint optimum.
    H_per = []
    g_per = []
    for use_eps, term in zip(mask, terms):
        def fn(th, term=term, use_eps=use_eps):
            return _joint_loss_single_stage(th, term, N_c, use_eps)
        H_i = torch.autograd.functional.hessian(fn, theta_hat)
        H_i = 0.5 * (H_i + H_i.transpose(-1, -2))
        theta_req = theta_hat.detach().clone().requires_grad_(True)
        L_i = fn(theta_req)
        g_i = torch.autograd.grad(L_i, theta_req)[0]
        H_per.append(H_i)
        g_per.append(g_i)
    H_total = sum(H_per)

    # Active parameter set: drop the eps_iso row/col when it's not
    # identified, so the kept Hessian is invertible.
    if eps_iso_identified:
        active = slice(None)
    else:
        active = slice(0, N_c)

    cij_loo = np.zeros((len(prepped), N_c))
    eps_iso_loo = np.zeros(len(prepped))
    for i in range(len(prepped)):
        H_kept = H_total - H_per[i]
        try:
            dtheta = torch.linalg.solve(H_kept[active, active], g_per[i][active])
        except RuntimeError:
            dtheta = torch.linalg.pinv(H_kept[active, active]) @ g_per[i][active]
        theta_loo = theta_hat.clone()
        theta_loo[active] = theta_hat[active] + dtheta
        cij_loo[i] = theta_loo[:N_c].cpu().numpy()
        eps_iso_loo[i] = float(theta_loo[N_c].item())

    cij_loo_std = cij_loo.std(axis=0)
    eps_iso_loo_std = float(eps_iso_loo.std())

    return {
        "cij_loo":         cij_loo,
        "eps_iso_loo":     eps_iso_loo,
        "cij_loo_std":     cij_loo_std,
        "eps_iso_loo_std": eps_iso_loo_std,
        "cij_names":       names,
        "stage_indices":   list(range(len(prepped))),
    }
