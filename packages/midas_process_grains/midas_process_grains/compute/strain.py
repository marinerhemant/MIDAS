"""Strain solvers — Kenesei (per-spot lstsq) and Fable-Beaudoin (lattice).

The MIDAS paper §3.7 documents two strain-tensor methods:

  1. **Fable-Beaudoin** (paper Eq. 5–7): map the refined lattice parameters
     ``(a, b, c, α, β, γ)`` to a strain tensor via the orthogonalisation
     matrix ``A``.  Implemented as :func:`solve_strain_fable_beaudoin`,
     thin wrapper over ``midas_stress.tensor.lattice_params_to_strain``.

  2. **Kenesei** (paper Eq. 8–11): solve the per-spot linear system
     ``G ε = b`` where each row encodes one diffraction peak's d-spacing
     change. The C reference uses NLOPT Nelder-Mead simplex with bounds
     ±0.01 ([CalcStrains.c:156](FF_HEDM/src/CalcStrains.c#L156),
     line 211–213). We expose two variants:

       * :func:`solve_strain_kenesei_bounded` — uses
         ``scipy.optimize.lsq_linear`` with the same ±0.01 bounds. Matches
         the C reference numerically. **Default**.
       * :func:`solve_strain_kenesei_unbounded` — closed-form
         ``torch.linalg.lstsq`` with optional Tikhonov regularization.
         Fully autograd-differentiable. The pure backslash, with the
         caveat that FF-HEDM geometry undercontrains ε_xx (every g-vector
         has g_x ≈ -sin θ ≈ 0.05–0.15, nearly constant per ring), so on
         real data ε_xx blows up under noise without bounds. Use only
         when (a) gradients matter and (b) bounds are not a substitute.
         Neither ``lstsq`` nor its ``pinv`` fallback check rank/condition
         before solving, so this near-degeneracy passes through silently;
         :func:`design_matrix_conditioning` attaches a rank/condition-number
         diagnostic (``PerSpotStrainResult.condition_number``/``degenerate``)
         on both paths so callers can flag rather than trust a blown-up fit.

The math: for each indexed spot ``i``,

.. math::

    \\hat{\\mathbf{g}}_i^\\top\\, \\boldsymbol\\varepsilon_{\\mathrm{lab}}\\,
    \\hat{\\mathbf{g}}_i = \\frac{ds^{\\,obs}_i - ds^0_i}{ds^0_i}

which expanded into the 6 unique strain components reads

.. math::

    [\\hat g_x^2,\\, \\hat g_y^2,\\, \\hat g_z^2,\\, 2\\hat g_x\\hat g_y,
     \\, 2\\hat g_x\\hat g_z,\\, 2\\hat g_y\\hat g_z]\\,
    [\\varepsilon_{xx},\\, \\varepsilon_{yy},\\, \\varepsilon_{zz},
     \\, \\varepsilon_{xy},\\, \\varepsilon_{xz},\\, \\varepsilon_{yz}]^\\top
    = b_i.

Stacking gives a linear system :math:`G\\boldsymbol\\varepsilon = b` solved via
``torch.linalg.lstsq``. This is the user's "backslash operator" (MATLAB
``A \\\\ b``).

The solver is fully differentiable — autograd flows back to ``g_obs`` and
``ds_obs`` so end-to-end calibration loops can backprop through grain strain.

References
----------
- Sharma 2012a §3 / paper Eq. 8–11
- ``midas_stress.tensor.lattice_params_to_strain`` for the alternative
  lattice-parameters → strain (paper Eq. 5–7) which we expose as a secondary
  output method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


__all__ = [
    "build_design_matrix",
    "design_matrix_conditioning",
    "solve_strain_kenesei_bounded",
    "solve_strain_kenesei_unbounded",
    "solve_strain_kenesei_prior_anchored",
    "solve_strain_fable_beaudoin",
    # Backwards-compatible aliases (kept until v0.2)
    "solve_strain_lstsq",
    "solve_strain_lattice",
    "voigt6_to_tensor",
    "PerSpotStrainResult",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PerSpotStrainResult:
    """Output of :func:`solve_strain_lstsq` for one grain.

    Attributes
    ----------
    epsilon_voigt : torch.Tensor
        ``(6,)`` strain in Voigt order ``(ε_xx, ε_yy, ε_zz, ε_xy, ε_xz, ε_yz)``.
    epsilon_tensor : torch.Tensor
        ``(3, 3)`` symmetric strain tensor in lab frame.
    residual_norm : torch.Tensor
        Scalar L2 norm of ``G ε - b`` (the "StrainError" written into
        SpotMatrix.csv).
    n_spots : int
        Number of spots that fed into the solve. Diagnostic; may differ from
        the input row count if any rows were filtered (zero-magnitude g).
    condition_number : float
        SVD-based condition number (``s_max / s_min``) of the design matrix
        actually solved (see :func:`design_matrix_conditioning`); ``inf`` if
        rank-deficient. ``nan`` for solvers that do not attach this
        diagnostic (default, kept for backwards compatibility).
    degenerate : bool
        ``True`` iff the design matrix was rank-deficient or ill-conditioned
        per :func:`design_matrix_conditioning`. This is a FLAG, not a
        rejection: near-degenerate reflection geometry (e.g. the chronically
        small ε_xx singular value in FF-HEDM, see :func:`solve_strain_lstsq`)
        is the norm for this data, not a rare edge case, so a degenerate fit
        is still returned -- callers filter/reweight/trust-flag downstream.
        ``False`` (unflagged) for solvers that do not attach this diagnostic.
    """

    epsilon_voigt: torch.Tensor
    epsilon_tensor: torch.Tensor
    residual_norm: torch.Tensor
    n_spots: int
    condition_number: float = float("nan")
    degenerate: bool = False


# ---------------------------------------------------------------------------
# Pure tensor kernels
# ---------------------------------------------------------------------------


def build_design_matrix(g_obs: torch.Tensor) -> torch.Tensor:
    """Build the (n, 6) design matrix from unit g-vectors.

    Each row is ``[g_x², g_y², g_z², 2 g_x g_y, 2 g_x g_z, 2 g_y g_z]``
    matching paper Eq. 9.

    Parameters
    ----------
    g_obs : torch.Tensor
        ``(n, 3)``. Will be normalised internally; caller may pass raw
        observed g-vectors of any magnitude.
    """
    if g_obs.dim() != 2 or g_obs.shape[1] != 3:
        raise ValueError(f"g_obs must be (n, 3); got {g_obs.shape}")
    norms = torch.linalg.norm(g_obs, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-30)
    g_hat = g_obs / norms                                                   # (n, 3)
    gx, gy, gz = g_hat[:, 0], g_hat[:, 1], g_hat[:, 2]
    G = torch.stack(
        [gx * gx, gy * gy, gz * gz, 2 * gx * gy, 2 * gx * gz, 2 * gy * gz],
        dim=1,
    )
    return G


def design_matrix_conditioning(
    G: torch.Tensor,
    *,
    rank_rtol: float = 1e-10,
    cond_threshold: float = 1e3,
) -> Tuple[int, float, bool]:
    """SVD-based rank / condition-number diagnostic for a strain design matrix.

    ``torch.linalg.lstsq`` (and its ``pinv`` fallback) return *a*
    least-squares solution even when ``G`` is rank-deficient or merely
    ill-conditioned -- neither checks rank or condition before solving. FF-
    HEDM g-vector geometry chronically starves the ε_xx direction of a small
    singular value even at ``n >= 6`` spots (module docstring, and the ``g_x
    ≈ -sinθ`` per-ring geometry exercised in
    ``test_kenesei_bounded_clamps_blow_up_at_box_edge``), so this is the norm
    for real data, not a rare edge case. We therefore **flag, not reject**:
    the caller decides whether to filter, reweight, or trust-flag a
    degenerate grain downstream.

    Parameters
    ----------
    G : torch.Tensor
        ``(n, 6)`` design matrix (see :func:`build_design_matrix`), i.e. the
        matrix actually being solved (post-weighting, if any).
    rank_rtol : float
        Singular values below ``rank_rtol * s_max`` count as zero for rank
        (mirrors ``midas_dfxm.inverse.strain_identifiability``'s ``1e-10``).
    cond_threshold : float
        Flag as degenerate when ``s_max / s_min`` exceeds this (mirrors
        ``midas_stress.elastic_inverse``'s ``cond_threshold`` default, 1e3).

    Returns
    -------
    rank : int
    condition_number : float
        ``s_max / s_min``; ``inf`` when the matrix is rank-deficient.
    degenerate : bool
        ``True`` iff ``rank < G.shape[1]`` or ``condition_number >
        cond_threshold``.
    """
    s = torch.linalg.svdvals(G.detach())
    s_max = float(s.max())
    n_cols = G.shape[-1]
    tol = s_max * rank_rtol
    rank = int((s > tol).sum())
    if rank < n_cols:
        condition_number = float("inf")
    else:
        s_min = float(s.min())
        condition_number = (s_max / s_min) if s_min > 0.0 else float("inf")
    degenerate = (rank < n_cols) or (condition_number > cond_threshold)
    return rank, condition_number, degenerate


def voigt6_to_tensor(eps: torch.Tensor) -> torch.Tensor:
    """Convert ``(6,)`` Voigt strain to ``(3, 3)`` symmetric tensor.

    Voigt order here is ``(xx, yy, zz, xy, xz, yz)`` matching
    :func:`build_design_matrix` rows. ``ε_xy`` etc. are tensor (NOT engineering)
    shears: a ``2 g_x g_y`` factor in the design matrix means the Voigt entry
    already represents ``ε_xy`` (the off-diagonal of the symmetric tensor).
    """
    if eps.shape[-1] != 6:
        raise ValueError(f"voigt vector must end with size 6; got {eps.shape}")
    out = torch.zeros(eps.shape[:-1] + (3, 3), dtype=eps.dtype, device=eps.device)
    out[..., 0, 0] = eps[..., 0]
    out[..., 1, 1] = eps[..., 1]
    out[..., 2, 2] = eps[..., 2]
    out[..., 0, 1] = out[..., 1, 0] = eps[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = eps[..., 4]
    out[..., 1, 2] = out[..., 2, 1] = eps[..., 5]
    return out


def solve_strain_lstsq(
    g_obs: torch.Tensor,
    ds_obs: torch.Tensor,
    ds_0: torch.Tensor,
    *,
    weights: Optional[torch.Tensor] = None,
    rcond: Optional[float] = None,
    regularization: float = 0.0,
    cond_threshold: float = 1e3,
) -> PerSpotStrainResult:
    """Solve ``G ε = b`` for one grain's six strain components.

    Parameters
    ----------
    g_obs : torch.Tensor
        ``(n, 3)`` observed g-vectors in lab frame. Magnitudes are ignored
        internally (we normalise).
    ds_obs : torch.Tensor
        ``(n,)`` observed d-spacings.
    ds_0 : torch.Tensor
        ``(n,)`` reference (unstrained) d-spacings for the same theoretical
        hkls. Same length as ``ds_obs``.
    weights : torch.Tensor, optional
        ``(n,)`` per-spot weights (e.g. inverse residuals). When provided
        we solve ``(W G) ε = (W b)``.
    rcond : float, optional
        Cut-off for small singular values, passed to ``torch.linalg.lstsq``.
        ``None`` uses the backend default.
    regularization : float, optional
        Tikhonov regularization strength. When > 0, solves
        ``ε = (GᵀG + α I)⁻¹ Gᵀ b`` (closed-form, differentiable). This is the
        recommended default for FF-HEDM data: the geometry constrains ε_yy
        and ε_zz well (g_y, g_z magnitudes ≈ 0.7) but barely constrains ε_xx
        (g_x ≈ 0.07 because the beam is along x), so unbounded lstsq can
        return huge ε_xx values. A small α (~1e-6) anchors ε_xx without
        biasing the well-constrained components. The C reference solver
        achieves the same effect via NLOPT bounds ±0.01 inside the simplex.
    cond_threshold : float, optional
        Passed to :func:`design_matrix_conditioning` to set the
        ``degenerate`` flag on the returned result (default 1e3). This is a
        FLAG, not a rejection -- see :func:`design_matrix_conditioning` and
        the ``degenerate`` field on :class:`PerSpotStrainResult`. Fires
        whether the primary ``lstsq`` solve succeeds or falls back to
        ``pinv``.

    Returns
    -------
    PerSpotStrainResult
    """
    if g_obs.shape[0] != ds_obs.shape[0] or g_obs.shape[0] != ds_0.shape[0]:
        raise ValueError(
            f"row-count mismatch: g_obs {g_obs.shape}, ds_obs {ds_obs.shape}, "
            f"ds_0 {ds_0.shape}"
        )
    n = g_obs.shape[0]
    if n < 6:
        raise ValueError(
            f"Strain lstsq needs ≥ 6 indexed spots; got {n}. "
            "Drop the grain or fall back to the lattice-params method."
        )

    # Promote to a single common dtype so linalg.lstsq doesn't choke.
    common_dtype = torch.promote_types(
        g_obs.dtype,
        torch.promote_types(ds_obs.dtype, ds_0.dtype),
    )
    g_obs = g_obs.to(dtype=common_dtype)
    ds_obs = ds_obs.to(dtype=common_dtype)
    ds_0 = ds_0.to(dtype=common_dtype)
    if weights is not None:
        weights = weights.to(dtype=common_dtype)

    G = build_design_matrix(g_obs)                                          # (n, 6)
    b = (ds_obs - ds_0) / torch.clamp(ds_0, min=1e-30)                      # (n,)

    if weights is not None:
        if weights.shape != (n,):
            raise ValueError(f"weights must be ({n},); got {weights.shape}")
        w = weights.unsqueeze(1)
        G = G * w
        b = b * weights

    # Rank/condition diagnostic on the design matrix actually being solved
    # (post-weighting). Neither `lstsq` nor `pinv` check this themselves --
    # both silently return *a* answer for a rank-deficient or ill-conditioned
    # `G`, which is the FF-HEDM norm (chronic near-degenerate ε_xx direction),
    # not a rare edge case. We flag, not reject: see `design_matrix_conditioning`.
    _rank, condition_number, degenerate = design_matrix_conditioning(
        G, cond_threshold=cond_threshold,
    )

    if regularization > 0.0:
        # Column-normalised Tikhonov: rescale each column of G to unit L2 norm
        # so the regularisation α has the same physical effect on every
        # component regardless of how well that component is constrained.
        #
        # Math: solve (Gn^T Gn + α I) εn = Gn^T b where Gn = G / D, then
        # back out ε = εn / D. Closed-form, differentiable, numerically
        # stable for the FF-HEDM ε_xx pathology (see docstring).
        col_norm = torch.linalg.norm(G, dim=0)
        col_norm = torch.clamp(col_norm, min=1e-30)
        Gn = G / col_norm
        gnt_gn = Gn.T @ Gn
        eye = torch.eye(
            gnt_gn.shape[0], dtype=gnt_gn.dtype, device=gnt_gn.device,
        )
        eps_n = torch.linalg.solve(gnt_gn + regularization * eye, Gn.T @ b)
        eps = eps_n / col_norm
    else:
        # Pure backslash. ``torch.linalg.lstsq`` is the differentiable form.
        try:
            sol = torch.linalg.lstsq(G, b.unsqueeze(1), rcond=rcond)
            eps = sol.solution.squeeze(1)                                   # (6,)
        except RuntimeError:
            # MPS / older torch can choke on lstsq with autograd; pinv is
            # equivalent and also differentiable. This fallback used to be
            # silent -- the rank/condition diagnostic above fires here too,
            # since it was computed on `G` before the branch, not skipped.
            eps = torch.linalg.pinv(G) @ b

    residual = G @ eps - b
    residual_norm = torch.linalg.norm(residual)
    return PerSpotStrainResult(
        epsilon_voigt=eps,
        epsilon_tensor=voigt6_to_tensor(eps),
        residual_norm=residual_norm,
        n_spots=int(n),
        condition_number=condition_number,
        degenerate=degenerate,
    )


# ---------------------------------------------------------------------------
# Bounded Kenesei lstsq (paper Eq. 8-11 with NLOPT-style bounds)
# ---------------------------------------------------------------------------


def solve_strain_kenesei_bounded(
    g_obs: torch.Tensor,
    ds_obs: torch.Tensor,
    ds_0: torch.Tensor,
    *,
    bounds: Tuple[float, float] = (-0.01, 0.01),
    weights: Optional[torch.Tensor] = None,
) -> PerSpotStrainResult:
    """Bounded Kenesei per-spot lstsq — matches the C reference output.

    Solves ``min ‖Gε - b‖²`` subject to ``bounds[0] ≤ ε_i ≤ bounds[1]`` for
    each of the six Voigt components, via ``scipy.optimize.lsq_linear``.
    The bounds (default ±0.01) match the C ``StrainTensorKenesei`` NLOPT
    simplex bounds at [CalcStrains.c:211-213](FF_HEDM/src/CalcStrains.c#L211).

    The bounded solver runs under ``torch.no_grad()`` (it's a scipy iterative
    optimiser, not autograd-aware); use :func:`solve_strain_kenesei_unbounded`
    if you need gradients. For physical FF-HEDM data the bounds are tight
    enough that production strains ε ≪ 0.01, so the bound is not actually
    active in practice — it just clamps the ε_xx ill-conditioning blow-up
    that affects FF-HEDM geometry.

    Parameters
    ----------
    g_obs, ds_obs, ds_0 : torch.Tensor
        Same as :func:`solve_strain_kenesei_unbounded`.
    bounds : (low, high)
        Per-component ε bounds. Default ``(-0.01, 0.01)`` matches C reference.
    weights : torch.Tensor, optional
        Per-spot weights as for the unbounded solver.

    Returns
    -------
    PerSpotStrainResult
    """
    from scipy.optimize import lsq_linear

    if g_obs.shape[0] != ds_obs.shape[0] or g_obs.shape[0] != ds_0.shape[0]:
        raise ValueError(
            f"row-count mismatch: g_obs {g_obs.shape}, ds_obs {ds_obs.shape}, "
            f"ds_0 {ds_0.shape}"
        )
    n = g_obs.shape[0]
    if n < 6:
        raise ValueError(
            f"Kenesei lstsq needs ≥ 6 indexed spots; got {n}. Drop the grain "
            "or fall back to Fable-Beaudoin."
        )

    common_dtype = torch.promote_types(
        g_obs.dtype,
        torch.promote_types(ds_obs.dtype, ds_0.dtype),
    )
    g = g_obs.to(dtype=common_dtype).detach().cpu().numpy()
    do = ds_obs.to(dtype=common_dtype).detach().cpu().numpy()
    d0 = ds_0.to(dtype=common_dtype).detach().cpu().numpy()
    G = build_design_matrix(torch.from_numpy(g)).numpy()
    b = (do - d0) / np.clip(d0, 1e-30, None)

    if weights is not None:
        w = weights.detach().cpu().numpy()
        G = G * w[:, None]
        b = b * w

    # scipy lsq_linear: bounded LSQ via trust-region reflective.
    res_scipy = lsq_linear(
        G, b,
        bounds=(np.full(6, bounds[0]), np.full(6, bounds[1])),
        method="trf", tol=1e-10,
    )
    eps_np = res_scipy.x
    residual = G @ eps_np - b
    residual_norm = float(np.linalg.norm(residual))

    eps_t = torch.from_numpy(eps_np).to(
        device=g_obs.device, dtype=common_dtype,
    )
    return PerSpotStrainResult(
        epsilon_voigt=eps_t,
        epsilon_tensor=voigt6_to_tensor(eps_t),
        residual_norm=torch.tensor(residual_norm, device=g_obs.device, dtype=common_dtype),
        n_spots=int(n),
    )


# Alias for backwards compatibility — pure-torch unbounded lstsq is still
# available under the original name.
solve_strain_kenesei_unbounded = solve_strain_lstsq


# ---------------------------------------------------------------------------
# Batched Kenesei — solve B grains in one tensor op (GPU-friendly)
# ---------------------------------------------------------------------------


def solve_strain_kenesei_batched(
    g_obs_list: List[np.ndarray],          # B grains, each (n_i, 3)
    ds_obs_list: List[np.ndarray],         # B grains, each (n_i,)
    ds_0_list: List[np.ndarray],           # B grains, each (n_i,)
    *,
    bounds: Tuple[float, float] = (-0.01, 0.01),
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched bit-equivalent Kenesei solve over B grains in one tensor op.

    Builds padded ``(B, max_n, 6)`` design matrix and ``(B, max_n)`` target
    vector, masks padded rows, and solves
    ``ε_b = (G_b^T G_b + λI)^{-1} G_b^T b_b`` for every grain in parallel via
    ``torch.linalg.solve`` on a ``(B, 6, 6)`` stack.

    Bounds enforcement: post-clamp to ``[bounds[0], bounds[1]]``. For
    physical FF-HEDM data the bounds are loose enough that production
    strains never approach them in practice (same as the per-grain bounded
    solver); the C reference's NLOPT bounds are similarly inactive on
    real grains. So unbounded + clamp ≈ bounded NLOPT to within solver
    tolerance.

    Returns
    -------
    eps_voigt : torch.Tensor
        ``(B, 6)`` strain in Voigt order on the requested device.
    rmse_per_spot : torch.Tensor
        ``(B,)`` residual RMSE per spot (before any unit scaling).
    """
    B = len(g_obs_list)
    if B == 0:
        return (torch.empty(0, 6, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=dtype))

    n_per = np.asarray([g.shape[0] for g in g_obs_list], dtype=np.int64)
    max_n = int(n_per.max())
    if max_n == 0:
        return (torch.zeros(B, 6, device=device, dtype=dtype),
                torch.full((B,), float("nan"), device=device, dtype=dtype))

    G_pad = np.zeros((B, max_n, 6), dtype=np.float64)
    b_pad = np.zeros((B, max_n), dtype=np.float64)
    mask = np.zeros((B, max_n), dtype=bool)
    for i, (g_obs, ds_obs, ds_0) in enumerate(
            zip(g_obs_list, ds_obs_list, ds_0_list)):
        n_i = g_obs.shape[0]
        if n_i == 0:
            continue
        # Normalise g
        norms = np.linalg.norm(g_obs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-30)
        g_hat = g_obs / norms
        gx, gy, gz = g_hat[:, 0], g_hat[:, 1], g_hat[:, 2]
        G_i = np.stack(
            [gx * gx, gy * gy, gz * gz, 2 * gx * gy, 2 * gx * gz, 2 * gy * gz],
            axis=1,
        )
        b_i = (ds_obs - ds_0) / np.maximum(ds_0, 1e-30)
        G_pad[i, :n_i] = G_i
        b_pad[i, :n_i] = b_i
        mask[i, :n_i] = True

    G_t = torch.from_numpy(G_pad).to(device=device, dtype=dtype)
    b_t = torch.from_numpy(b_pad).to(device=device, dtype=dtype)
    mask_t = torch.from_numpy(mask).to(device=device, dtype=dtype)

    # Zero out padded rows (already zero from np.zeros, but keep explicit).
    G_t = G_t * mask_t.unsqueeze(-1)
    b_t = b_t * mask_t

    # G^T G : (B, 6, 6); G^T b : (B, 6, 1)
    GTG = G_t.transpose(-2, -1) @ G_t
    GTb = (G_t.transpose(-2, -1) @ b_t.unsqueeze(-1))

    # Tiny Tikhonov for grains with rank < 6 (e.g. <6 valid spots).
    eye = torch.eye(6, device=device, dtype=dtype).unsqueeze(0)
    GTG_reg = GTG + 1e-12 * eye

    eps_voigt = torch.linalg.solve(GTG_reg, GTb).squeeze(-1)             # (B, 6)
    eps_voigt = torch.clamp(eps_voigt, min=bounds[0], max=bounds[1])

    # Per-grain RMSE = sqrt(mean((G ε - b)² over valid rows))
    pred = (G_t @ eps_voigt.unsqueeze(-1)).squeeze(-1)                   # (B, max_n)
    resid = (pred - b_t) * mask_t                                        # (B, max_n)
    n_valid = mask_t.sum(dim=-1).clamp(min=1)
    rmse = torch.sqrt((resid * resid).sum(dim=-1) / n_valid)             # (B,)
    return eps_voigt, rmse


# ---------------------------------------------------------------------------
# Prior-anchored Kenesei  (the closest equivalent to the C reference)
# ---------------------------------------------------------------------------


def solve_strain_kenesei_prior_anchored(
    g_obs: torch.Tensor,
    ds_obs: torch.Tensor,
    ds_0: torch.Tensor,
    eps_prior_voigt: torch.Tensor,
    *,
    anchor_strength: float = 0.1,
) -> PerSpotStrainResult:
    """Prior-anchored Kenesei strain — the closest closed-form equivalent
    to the C ``StrainTensorKenesei`` Nelder-Mead behaviour.

    The C reference initialises the simplex at the **Fable-Beaudoin** strain
    tensor (``ProcessGrains.c`` calls ``CalcStrainTensorFableBeaudoin`` first
    and passes its result as ``StrainTensorInput`` to ``StrainTensorKenesei``,
    [CalcStrains.c:205-210](FF_HEDM/src/CalcStrains.c#L205)). Because Nelder-
    Mead is a local search and the cost gradient along ε_xx is tiny in
    FF-HEDM (g_x ≈ -sin θ ≈ 0.05–0.15), the optimiser barely moves ε_xx away
    from its initial value, while ε_yy / ε_zz are pulled toward the
    least-squares minimum.

    This function reproduces that effect via Bayesian-prior Tikhonov:

    .. math::

        \\varepsilon^* = \\arg\\min_\\varepsilon
            \\|G \\varepsilon - b\\|^2
          + \\alpha\\, (\\varepsilon - \\varepsilon_{\\rm FB})^\\top
            \\hat D\\,(\\varepsilon - \\varepsilon_{\\rm FB})

    with closed-form solution
    ``ε* = (GᵀG + α D̂)⁻¹ (Gᵀb + α D̂ ε_FB)``,
    where ``D̂ = diag(GᵀG)`` makes the prior weight scale inversely with how
    well each component is constrained by the data — strong on ε_xx, weak
    on ε_yy / ε_zz. Closed-form, differentiable, no scipy dependency.

    Parameters
    ----------
    g_obs, ds_obs, ds_0 : torch.Tensor
        As in :func:`solve_strain_kenesei_unbounded`.
    eps_prior_voigt : torch.Tensor
        ``(6,)`` Fable-Beaudoin (or any prior) strain in lab frame, Voigt
        layout ``(xx, yy, zz, xy, xz, yz)``.
    anchor_strength : float
        ``α`` in the equation above. Default 0.1 — well-conditioned
        components inherit ~all data, ε_xx inherits ~all prior.

    Returns
    -------
    PerSpotStrainResult
    """
    if g_obs.shape[0] != ds_obs.shape[0] or g_obs.shape[0] != ds_0.shape[0]:
        raise ValueError(
            f"row-count mismatch: g_obs {g_obs.shape}, ds_obs {ds_obs.shape}, "
            f"ds_0 {ds_0.shape}"
        )
    n = g_obs.shape[0]
    if n < 6:
        raise ValueError(
            f"Prior-anchored Kenesei needs ≥ 6 indexed spots; got {n}."
        )
    if eps_prior_voigt.shape != (6,):
        raise ValueError(
            f"eps_prior_voigt must be shape (6,); got {eps_prior_voigt.shape}"
        )

    common_dtype = torch.promote_types(
        g_obs.dtype,
        torch.promote_types(
            torch.promote_types(ds_obs.dtype, ds_0.dtype),
            eps_prior_voigt.dtype,
        ),
    )
    g_obs = g_obs.to(dtype=common_dtype)
    ds_obs = ds_obs.to(dtype=common_dtype)
    ds_0 = ds_0.to(dtype=common_dtype)
    eps_prior = eps_prior_voigt.to(dtype=common_dtype, device=g_obs.device)

    G = build_design_matrix(g_obs)                                          # (n, 6)
    b = (ds_obs - ds_0) / torch.clamp(ds_0, min=1e-30)                      # (n,)

    GtG = G.T @ G                                                            # (6, 6)
    Gtb = G.T @ b                                                            # (6,)
    eye = torch.eye(6, dtype=GtG.dtype, device=GtG.device)

    # Bayesian prior: ε ~ N(ε_prior, σ_prior² I). MAP solution is
    #   (GᵀG + α I) ε = Gᵀb + α ε_prior
    # where α controls how much the prior dominates: with α set to be small
    # compared to well-constrained eigenvalues of GᵀG (~tens), data wins for
    # ε_yy / ε_zz; with α much larger than poorly-constrained eigenvalues
    # (~1e-2 for ε_xx in FF-HEDM), prior wins for ε_xx.
    A = GtG + anchor_strength * eye
    rhs = Gtb + anchor_strength * eps_prior
    eps = torch.linalg.solve(A, rhs)

    residual = G @ eps - b
    residual_norm = torch.linalg.norm(residual)
    return PerSpotStrainResult(
        epsilon_voigt=eps,
        epsilon_tensor=voigt6_to_tensor(eps),
        residual_norm=residual_norm,
        n_spots=int(n),
    )


# ---------------------------------------------------------------------------
# Fable-Beaudoin lattice-parameter method (paper Eq. 5-7)
# ---------------------------------------------------------------------------


def solve_strain_fable_beaudoin(
    lattice_strained: torch.Tensor,
    lattice_unstrained: torch.Tensor,
) -> torch.Tensor:
    """Fable-Beaudoin lattice-parameter strain (paper Eq. 5-7).

    Maps the refined lattice parameters ``(a, b, c, α, β, γ)`` to the
    grain-frame strain tensor ``ε_gr`` via the orthogonalisation matrix
    ``A`` (paper Eq. 7) and the Green-Lagrange form
    ``ε = ½(F + F^T) - I`` with ``F = A A_0^{-1}``.

    Implemented as a thin wrapper over
    ``midas_stress.tensor.lattice_params_to_strain``.

    Parameters
    ----------
    lattice_strained : torch.Tensor
        ``(6,)`` (a, b, c, α, β, γ) — angles in degrees.
    lattice_unstrained : torch.Tensor
        ``(6,)`` reference lattice parameters.

    Returns
    -------
    torch.Tensor
        ``(3, 3)`` strain tensor in grain frame.
    """
    from midas_stress.tensor import lattice_params_to_strain
    return lattice_params_to_strain(lattice_strained, lattice_unstrained)


# Backwards-compatible alias.
solve_strain_lattice = solve_strain_fable_beaudoin
