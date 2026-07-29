"""Differentiable small-box PDF structure modelling (PDFfit-style).

Given a crystal structure (space group + lattice + asymmetric-unit atoms, via
``midas_hkls``) this computes the model reduced PDF ``G_calc(r)`` as a sum of
Gaussian-broadened interatomic-distance contributions — the same real-space
"PDFfit" model PDFgui uses — but as a torch graph that is differentiable in the
refinable parameters (lattice, atomic coordinates, displacement parameters,
scale, correlated-motion δ1/δ2, instrument Qdamp/Qbroad).

    G_calc(r) = (s / r) (1/N) Σ_i Σ_{j≠i} (Z_i Z_j / <Z>²)
                  · exp(-(r - r_ij)² / 2σ_ij²) / (√(2π) σ_ij)
                - 4π r ρ0 s,   then × exp(-(r Qdamp)²/2),
    σ_ij² = (U_i + U_j)(1 - δ1/r_ij - δ2/r_ij²) + (Qbroad r_ij)².

Distances ``r_ij`` come from the metric tensor (``Δf·G·Δf``), so they are
differentiable in the lattice parameters; the unit-cell atoms come from the
symmetry expansion, so refining the asymmetric unit automatically respects the
space group. The pair weighting uses atomic numbers (the f(Q→0)=Z real-space
approximation standard in PDFfit).

The point of doing this in midas-pdf rather than calling PDFgui: the refinement
is **error-aware** (a χ² against the propagated σ on G(r), with parameter
uncertainties from the autograd Hessian) and **differentiable end to end** (the
structure can be co-refined with geometry / normalization / multiple scattering).
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np
import torch

from midas_hkls.absorption import Z_for
from midas_hkls.lattice_torch import cell_volume, metric_tensor

__all__ = [
    "pdffit_gr",
    "PairList",
    "build_pair_list",
    "refine_structure",
    "orthogonalization_matrix",
]

_FOUR_PI = 4.0 * float(np.pi)
_B_TO_U = 1.0 / (8.0 * float(np.pi) ** 2)   # B = 8π²U


def orthogonalization_matrix(lattice_params: torch.Tensor) -> torch.Tensor:
    """Crystallographic orthogonalization matrix M(a, b, c, α, β, γ), row-vector
    convention (a-along-x, b in xy plane, c completes right-handed frame).

    Differentiable in every lattice entry.  Satisfies ``M @ M.T == metric_tensor``.
    A row-vector fractional coordinate ``f`` maps to Cartesian as
    ``r_cart_row = f_row @ M``; this is the standard PDB / International-Tables
    convention that anisotropic U_ij tensors from CIF files are defined in.

    Distinct from ``torch.linalg.cholesky(G)`` which is lower-triangular in a
    Cholesky-of-metric frame — for triclinic cells the two differ by an
    orthogonal rotation and give different ``bhat^T U bhat`` results.
    """
    lat = torch.as_tensor(lattice_params, dtype=torch.float64)
    a, b, c = lat[0], lat[1], lat[2]
    alpha = lat[3] * (float(np.pi) / 180.0)
    beta = lat[4] * (float(np.pi) / 180.0)
    gamma = lat[5] * (float(np.pi) / 180.0)
    ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sg = torch.sin(gamma).clamp(min=1e-12)
    vol_factor = torch.sqrt(
        (1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg).clamp(min=1e-30)
    )

    z0 = torch.zeros((), dtype=lat.dtype)
    row0 = torch.stack([a,     z0,       z0])
    row1 = torch.stack([b * cg, b * sg,   z0])
    row2 = torch.stack([c * cb,
                        c * (ca - cb * cg) / sg,
                        c * vol_factor / sg])
    return torch.stack([row0, row1, row2], dim=0)


class PairList:
    """Precomputed, parameter-independent pair bookkeeping for a crystal.

    Holds, for every ordered pair (unit-cell atom i, unit-cell atom j) and every
    integer lattice translation ``n`` within the cutoff: the fractional
    separation ``Δf = f_i - f_j + n`` and the Z-weight ``Z_i Z_j``. Distances are
    recomputed differentiably from the metric tensor each forward pass; only the
    integer ``n`` set and the atom indices are fixed here.
    """

    def __init__(self, dfrac: torch.Tensor, zweight: torch.Tensor,
                 n_uc: int, z_mean: float,
                 i_idx: Optional[torch.Tensor] = None,
                 j_idx: Optional[torch.Tensor] = None):
        self.dfrac = dfrac          # (P, 3) fractional separations
        self.zweight = zweight      # (P,) Z_i Z_j
        self.n_uc = n_uc
        self.z_mean = z_mean
        self.i_idx = i_idx          # (P,) unit-cell-atom index i (Rev 15)
        self.j_idx = j_idx          # (P,) unit-cell-atom index j (Rev 15)


def build_pair_list(crystal_tensor, *, r_max: float = 10.0, margin: float = 1.0) -> PairList:
    """Enumerate all atom pairs (with periodic images) out to ``r_max``."""
    frac, _occ, _B, _U = crystal_tensor.unit_cell_view()
    frac = frac.detach()                                 # (n_uc, 3)
    n_uc = frac.shape[0]
    elements = list(crystal_tensor.elements)             # (n_uc,) unit-cell elements
    Z = np.array([Z_for(e) for e in elements], dtype=np.float64)
    z_mean = float(Z.mean())

    # how many lattice translations to span r_max: use the shortest cell vector
    G = metric_tensor(crystal_tensor.lattice_params).detach().numpy()
    cell_lengths = np.sqrt(np.diag(G))
    Nmax = int(np.ceil((r_max + margin) / cell_lengths.min())) + 1
    shifts = list(itertools.product(range(-Nmax, Nmax + 1), repeat=3))
    n_shift = torch.tensor(shifts, dtype=torch.float64)   # (S, 3)

    # build all (i, j, n) with a coarse distance pre-filter using the initial G
    Gt = torch.as_tensor(G, dtype=torch.float64)
    df_list, zw_list, ii_list, jj_list = [], [], [], []
    for i in range(n_uc):
        for j in range(n_uc):
            d = frac[i][None, :] - frac[j][None, :] + n_shift   # (S, 3)
            r2 = torch.einsum("sa,ab,sb->s", d, Gt, d)
            keep = (r2 > 1e-6) & (r2 <= (r_max + margin) ** 2)
            if keep.any():
                n_keep = int(keep.sum())
                df_list.append(d[keep])
                zw_list.append(torch.full((n_keep,), Z[i] * Z[j], dtype=torch.float64))
                ii_list.append(torch.full((n_keep,), i, dtype=torch.long))
                jj_list.append(torch.full((n_keep,), j, dtype=torch.long))
    dfrac = torch.cat(df_list, dim=0)
    zweight = torch.cat(zw_list, dim=0)
    i_idx = torch.cat(ii_list, dim=0)
    j_idx = torch.cat(jj_list, dim=0)
    return PairList(dfrac, zweight, n_uc, z_mean, i_idx=i_idx, j_idx=j_idx)


def pdffit_gr(
    crystal_tensor,
    r: torch.Tensor | np.ndarray,
    pairs: PairList,
    *,
    scale: torch.Tensor | float = 1.0,
    u_iso: Optional[torch.Tensor | float] = None,
    u_aniso: Optional[torch.Tensor] = None,
    occupancy: Optional[torch.Tensor] = None,
    delta1: torch.Tensor | float = 0.0,
    delta2: torch.Tensor | float = 0.0,
    q_damp: torch.Tensor | float = 0.0,
    q_broad: torch.Tensor | float = 0.0,
    lattice_params: Optional[torch.Tensor] = None,
    r_max: float = 10.0,
) -> torch.Tensor:
    """Model reduced PDF ``G_calc(r)`` for the crystal. Differentiable in the
    lattice, displacement, and profile parameters.

    ``lattice_params`` (6,) overrides ``crystal_tensor.lattice_params`` — pass a
    leaf tensor with ``requires_grad`` to refine the cell. ``u_iso`` (Å²)
    overrides the isotropic displacement (single shared value for the MVP);
    if ``None`` it is taken from the atoms' B factors.

    Rev 15 additions:

    * ``u_aniso`` — (n_uc, 3, 3) Cartesian anisotropic displacement tensors,
      one per unit-cell atom.  Overrides ``u_iso`` when provided.  The pair
      broadening becomes σ²_ij = bhat^T (U_i + U_j) bhat where bhat is the
      unit Cartesian bond vector — fully differentiable in every entry.
    * ``occupancy`` — (n_uc,) per-site occupancies (0 ≤ occ ≤ 1).  Multiplies
      each pair weight by occ_i · occ_j and rescales ρ0 = Σ occ / V, so
      partial occupancy propagates to both the pair term and the linear
      -4π r ρ0 baseline consistently.
    """
    r_t = torch.as_tensor(r, dtype=torch.float64)
    lat = (crystal_tensor.lattice_params if lattice_params is None
           else torch.as_tensor(lattice_params, dtype=torch.float64))
    G = metric_tensor(lat)                                # (3,3), diff in lattice
    V = cell_volume(lat)
    if occupancy is not None:
        occ_t = torch.as_tensor(occupancy, dtype=torch.float64)
        rho0 = occ_t.sum() / V                            # partial-occ number density
    else:
        rho0 = pairs.n_uc / V                             # atoms / Å³

    d = pairs.dfrac                                       # (P,3)
    r_ij = torch.sqrt(torch.einsum("pa,ab,pb->p", d, G, d).clamp(min=1e-12))

    if u_aniso is not None:
        if pairs.i_idx is None or pairs.j_idx is None:
            raise ValueError(
                "u_aniso requires a PairList built with i_idx/j_idx — "
                "rebuild via build_pair_list().")
        U = torch.as_tensor(u_aniso, dtype=torch.float64)         # (n_uc,3,3)
        if U.shape != (pairs.n_uc, 3, 3):
            raise ValueError(
                f"u_aniso must be shape ({pairs.n_uc}, 3, 3), got {tuple(U.shape)}")
        # Cartesian bond direction in the *crystallographic* frame (a along x,
        # b in xy plane) — this is the frame CIF anisotropic U_ij tensors are
        # defined in.  Do NOT use the Cholesky of G here: cubic accidentally
        # agrees, but triclinic differs by an orthogonal rotation and yields
        # wrong σ² = bhat^T U bhat.
        M = orthogonalization_matrix(lat)                          # (3,3)
        r_cart = pairs.dfrac @ M                                   # (P,3)
        bhat = r_cart / r_ij.unsqueeze(-1).clamp(min=1e-12)        # (P,3)
        U_i = U[pairs.i_idx]                                       # (P,3,3)
        U_j = U[pairs.j_idx]
        U_pair = U_i + U_j                                         # (P,3,3)
        u_pair = torch.einsum("pa,pab,pb->p", bhat, U_pair, bhat)
    elif u_iso is None:
        _f, _occ, B_iso, _U = crystal_tensor.unit_cell_view()
        u_pair = 2.0 * B_iso.mean() * _B_TO_U             # U_i + U_j with mean B
    else:
        u_pair = 2.0 * torch.as_tensor(u_iso, dtype=torch.float64)   # U_i + U_j

    d1 = torch.as_tensor(delta1, dtype=torch.float64)
    d2 = torch.as_tensor(delta2, dtype=torch.float64)
    qb = torch.as_tensor(q_broad, dtype=torch.float64)
    sharpen = (1.0 - d1 / r_ij - d2 / (r_ij * r_ij)).clamp(min=0.05)
    sigma2 = (u_pair * sharpen + (qb * r_ij) ** 2).clamp(min=1e-6)
    sigma = torch.sqrt(sigma2)

    w = pairs.zweight / (pairs.z_mean ** 2)               # (Z_i Z_j)/<Z>²
    if occupancy is not None:
        if pairs.i_idx is None or pairs.j_idx is None:
            raise ValueError(
                "occupancy requires a PairList built with i_idx/j_idx.")
        occ = torch.as_tensor(occupancy, dtype=torch.float64)
        if occ.shape != (pairs.n_uc,):
            raise ValueError(
                f"occupancy must have length {pairs.n_uc}, got {tuple(occ.shape)}")
        w = w * occ[pairs.i_idx] * occ[pairs.j_idx]

    # Gaussian-broadened pair density on the r grid: sum over pairs
    # contribution_p(r) = w_p / (sqrt(2π) σ_p) exp(-(r - r_ij)²/2σ²) / r
    rr = r_t[:, None]                                     # (R,1)
    gauss = torch.exp(-0.5 * ((rr - r_ij[None, :]) / sigma[None, :]) ** 2) \
        / (np.sqrt(2 * np.pi) * sigma[None, :])
    # Normalise the pair sum by the *number of scatterers actually present*
    # (= Σ occ_i for partial occupancy; = n_uc when every site is fully
    # occupied — the limit that recovers the standard PDFfit form).
    if occupancy is not None:
        n_norm = torch.as_tensor(occupancy, dtype=torch.float64).sum().clamp(min=1e-12)
    else:
        n_norm = torch.tensor(float(pairs.n_uc), dtype=torch.float64)
    pair_term = (w[None, :] * gauss).sum(dim=1) / n_norm          # (R,)
    s = torch.as_tensor(scale, dtype=torch.float64)
    G_calc = s * (pair_term / r_t.clamp(min=1e-6) - _FOUR_PI * r_t * rho0)

    qd = torch.as_tensor(q_damp, dtype=torch.float64)
    # Qdamp instrument envelope (= 1 when qd = 0); always applied so it stays
    # differentiable when q_damp is a refined parameter.
    G_calc = G_calc * torch.exp(-0.5 * (r_t * qd) ** 2)
    return G_calc


class _RefineResult(dict):
    def __getattr__(self, name):
        try: return self[name]
        except KeyError: raise AttributeError(name) from None


def refine_structure(
    crystal_tensor,
    r: torch.Tensor | np.ndarray,
    G_obs: torch.Tensor | np.ndarray,
    pairs: PairList,
    *,
    sigma_obs: Optional[torch.Tensor | np.ndarray] = None,
    init_a: Optional[float] = None,
    init_u_iso: float = 0.005,
    init_scale: float = 1.0,
    bg_order: Optional[int] = None,
    steps: int = 120,
    lr: float = 0.05,
    n_posterior_samples: int = 0,
) -> _RefineResult:
    """Error-aware small-box refinement of (cubic) lattice ``a``, isotropic
    displacement ``u_iso``, and ``scale`` against an observed ``G_obs``.

    With ``bg_order`` set, a smooth polynomial nuisance background
    ``Σ_j c_j (r/r_max)^j`` is **jointly refined** with the structure — the r-space
    stand-in for a residual normalization / multiple-scattering term. Refining it
    alongside the structure removes the bias it would otherwise induce, and the
    reported structure-parameter uncertainties are correctly *marginalised* over
    it (the nuisance is in the joint Hessian).

    Minimises χ² ``Σ ((G_obs - G_calc)/σ_obs)²`` by L-BFGS, then reports
    **parameter uncertainties from the autograd Hessian** at the minimum
    (``cov = 2 H⁻¹``). Returns fitted values, 1σ uncertainties, the fit, χ²/ndof,
    and the loss history. (MVP: cubic cell, single shared U; extends to full DOF.)
    """
    r_t = torch.as_tensor(r, dtype=torch.float64)
    G_t = torch.as_tensor(G_obs, dtype=torch.float64)
    w = (torch.ones_like(G_t) if sigma_obs is None
         else 1.0 / torch.as_tensor(sigma_obs, dtype=torch.float64).clamp(min=1e-12) ** 2)

    lat0 = crystal_tensor.lattice_params.detach()
    angles = lat0[3:]
    a0 = float(lat0[0]) if init_a is None else float(init_a)
    r_scale = float(r_t.max())

    # flat parameter vector: [a, u_iso, scale, (bg coeffs...)]
    names = ["a", "u_iso", "scale"]
    init = [a0, init_u_iso, init_scale]
    n_bg = 0 if bg_order is None else bg_order + 1
    init += [0.0] * n_bg
    theta = torch.tensor(init, dtype=torch.float64, requires_grad=True)

    def model(th):
        lp = torch.cat([th[0].reshape(1).expand(3), angles])
        G = pdffit_gr(crystal_tensor, r_t, pairs, scale=th[2], u_iso=th[1],
                      lattice_params=lp)
        if n_bg:
            powers = torch.stack([(r_t / r_scale) ** j for j in range(n_bg)], dim=1)
            G = G + powers @ th[3:]
        return G

    def chi2(th):
        res = G_t - model(th)
        return (w * res * res).sum()

    opt = torch.optim.LBFGS([theta], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")
    history: list[float] = []

    def closure():
        opt.zero_grad()
        loss = chi2(theta)
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    opt.step(closure)

    fitted = {k: float(theta[i].detach()) for i, k in enumerate(names)}
    if n_bg:
        fitted["bg_coef"] = [float(c) for c in theta[3:].detach()]

    # parameter uncertainties from the Hessian of chi2 (cov = 2 H^-1); the
    # structure-param sigmas are correctly marginalised over the nuisance bg.
    cov = None
    try:
        th_leaf = theta.detach().clone().requires_grad_(True)
        H = torch.autograd.functional.hessian(chi2, th_leaf)
        cov = 2.0 * torch.linalg.inv(H)
        sig = torch.sqrt(torch.diagonal(cov).clamp(min=0.0))
        uncert = {k: float(sig[i]) for i, k in enumerate(names)}
    except Exception:
        uncert = {k: float("nan") for k in names}

    with torch.no_grad():
        G_fit = model(theta)
        ndof = max(int(G_t.numel()) - theta.numel(), 1)
        chi2_red = float((w * (G_t - G_fit) ** 2).sum()) / ndof

    # ------------ Laplace posterior samples ------------------------------
    # Sample from N(θ_MAP, cov) and evaluate the model at each sample so
    # the user gets a proper posterior band on the refined G(r), rather
    # than the Gaussian assumption baked into a diagonal Jacobian.
    posterior = None
    if n_posterior_samples > 0 and cov is not None:
        try:
            mean = theta.detach()
            L = torch.linalg.cholesky(cov + 1e-12 * torch.eye(cov.shape[0],
                                                                dtype=cov.dtype))
            z = torch.randn((n_posterior_samples, cov.shape[0]),
                             dtype=cov.dtype)
            samples = mean.unsqueeze(0) + z @ L.T                # (N, P)
            with torch.no_grad():
                G_samples = torch.stack(
                    [model(samples[i]) for i in range(n_posterior_samples)])
            posterior = {
                "theta_samples": samples,                        # (N, P)
                "theta_names":   names,
                "G_samples":     G_samples,                      # (N, R)
                "G_mean":        G_samples.mean(dim=0),
                "G_std":         G_samples.std(dim=0),
                "a_samples":     samples[:, 0],
                "u_iso_samples": samples[:, 1],
                "scale_samples": samples[:, 2],
            }
        except Exception as e:                                    # pragma: no cover
            posterior = {"error": f"posterior sampling failed: {e}"}

    return _RefineResult(
        fitted=fitted, uncertainty=uncert, G_calc=G_fit.detach(),
        chi2_reduced=chi2_red, loss=history[-1] if history else float("nan"),
        history=history, loss_history=history,
        posterior=posterior, cov=cov,
    )
