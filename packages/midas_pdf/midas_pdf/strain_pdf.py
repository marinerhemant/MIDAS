"""Strain-sensitive (directional / azimuthally-sliced) PDF forward model.

Physics
-------
For a powder or amorphous sample under a homogeneous macroscopic elastic strain
``eps`` (sample frame), pair vectors are isotropically distributed across the
grain ensemble, and a pair along unit direction ``n`` has its length scaled by

    s(n) = 1 + n^T eps n .

An azimuthally-sliced PDF -- G(r) computed from one azimuthal wedge of the 2D
detector, optionally at a sample tilt -- preferentially probes pairs aligned
with the scattering-vector direction ``q_hat`` of that wedge.  With an axial
selection kernel ``W(t) = |n . q_hat|^(2m)`` (m = 0 recovers the isotropic
PDF; larger m = sharper azimuthal selectivity), the sliced PDF is exactly the
kernel-weighted average of the *unstrained* model PDF with all distances
scaled by ``s(n)``:

    G(r; q_hat, eps) = sum_k  w_k(q_hat) * G0(r; lattice lengths * s(n_k))

evaluated by spherical quadrature over directions ``n_k``.  Distance scaling
is implemented by scaling the cell lengths passed to ``pdffit_gr`` -- exact
for any crystal system, and the number density rho0 rescales consistently.

Consequences (the point of the module):

* the isotropic PDF (m = 0) sees only the volumetric strain tr(eps)/3 and is
  exactly blind to the 5 deviatoric components;
* azimuthal slicing on one detector restores the in-detector-plane components
  (e22, e33, e23 for beam along x);
* sample tilts (or large scattering angles) bring the beam-direction
  components in -- the full tensor becomes determinable, and the Fisher /
  CRLB machinery here prices each component before any measurement exists.

Strain convention: Voigt order ``(e11, e22, e33, e23, e13, e12)``, sample
frame, beam along +x, detector plane y-z (eta measured from +z toward +y).
All tensors are float64 (package convention).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .structure import PairList, build_pair_list, pdffit_gr

__all__ = [
    "strain_voigt_to_matrix", "fibonacci_directions", "probe_directions",
    "sliced_gr_stack", "strain_fisher", "strain_crlb", "recover_strain",
]

_DT = torch.float64


def strain_voigt_to_matrix(strain6: torch.Tensor) -> torch.Tensor:
    """(6,) Voigt ``(e11,e22,e33,e23,e13,e12)`` -> (3,3) symmetric tensor."""
    e = strain6
    row0 = torch.stack([e[0], e[5], e[4]])
    row1 = torch.stack([e[5], e[1], e[3]])
    row2 = torch.stack([e[4], e[3], e[2]])
    return torch.stack([row0, row1, row2])


def fibonacci_directions(n: int, dtype=_DT) -> torch.Tensor:
    """(n, 3) near-uniform unit directions on the sphere."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    z = 1.0 - 2.0 * i / n
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    d = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)
    return torch.as_tensor(d, dtype=dtype)


def probe_directions(
    etas_deg: Sequence[float],
    *,
    theta_deg: float = 0.0,
    tilts_deg: Sequence[float] = (0.0,),
) -> torch.Tensor:
    """Sample-frame scattering-vector directions probed by azimuthal wedges.

    Beam along +x; detector azimuth ``eta`` measured from +z toward +y; at
    Bragg angle ``theta`` the scattering vector is tipped upstream by theta.
    A sample tilt ``tau`` about the vertical (z) axis maps the lab direction
    into the sample frame with Rz(-tau).  Returns ``(n_tilts*n_etas, 3)``.
    """
    out = []
    th = np.deg2rad(theta_deg)
    for tau_deg in tilts_deg:
        tau = np.deg2rad(tau_deg)
        Rz = np.array([[np.cos(tau), np.sin(tau), 0.0],
                       [-np.sin(tau), np.cos(tau), 0.0],
                       [0.0, 0.0, 1.0]])
        for eta_deg in etas_deg:
            eta = np.deg2rad(eta_deg)
            q_lab = np.array([-np.sin(th),
                              np.cos(th) * np.sin(eta),
                              np.cos(th) * np.cos(eta)])
            out.append(Rz @ q_lab)
    return torch.as_tensor(np.array(out), dtype=_DT)


def _scaled_lattice(lat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Scale cell lengths by ``s`` (angles unchanged) -- scales all pair
    distances by exactly ``s`` in any crystal system."""
    return torch.cat([lat[:3] * s, lat[3:]])


def sliced_gr_stack(
    crystal_tensor,
    r: torch.Tensor,
    pairs: PairList,
    qhats: torch.Tensor,          # (D, 3) probe directions
    strain6: torch.Tensor,        # (6,) Voigt (e11,e22,e33,e23,e13,e12)
    *,
    kernel_m: int = 8,
    n_quad: int = 128,
    lattice_params: Optional[torch.Tensor] = None,
    **pdf_kwargs,
) -> torch.Tensor:
    """Azimuthally-sliced model PDFs ``(D, len(r))`` under sample-frame strain.

    The quadrature PDFs ``G0(r; s_k)`` are shared across probe directions --
    only the kernel weights differ -- so the cost is ``n_quad`` calls to
    :func:`pdffit_gr` regardless of how many slices are requested.
    Differentiable in ``strain6`` (and everything ``pdffit_gr`` supports).
    ``kernel_m = 0`` gives the isotropic PDF for every direction.
    """
    lat = (crystal_tensor.lattice_params if lattice_params is None
           else torch.as_tensor(lattice_params, dtype=_DT))
    dirs = fibonacci_directions(n_quad)                    # (K, 3)
    eps = strain_voigt_to_matrix(torch.as_tensor(strain6, dtype=_DT))
    s = 1.0 + torch.einsum("ka,ab,kb->k", dirs, eps, dirs)  # (K,)

    Gk = torch.stack([
        pdffit_gr(crystal_tensor, r, pairs,
                  lattice_params=_scaled_lattice(lat, s[k]), **pdf_kwargs)
        for k in range(n_quad)
    ])                                                      # (K, R)

    qh = torch.as_tensor(qhats, dtype=_DT)
    qh = qh / qh.norm(dim=-1, keepdim=True)
    if kernel_m == 0:
        W = torch.full((qh.shape[0], n_quad), 1.0 / n_quad, dtype=_DT)
    else:
        t = (qh @ dirs.T).abs().clamp(min=1e-12)            # (D, K)
        W = t ** (2 * kernel_m)
        W = W / W.sum(dim=1, keepdim=True)
    return W @ Gk                                           # (D, R)


def strain_fisher(
    crystal_tensor,
    r: torch.Tensor,
    pairs: PairList,
    qhats: torch.Tensor,
    *,
    sigma_G: float = 0.02,
    strain0: Optional[torch.Tensor] = None,
    kernel_m: int = 8,
    n_quad: int = 128,
    **pdf_kwargs,
) -> torch.Tensor:
    """(6, 6) Fisher information of the Voigt strain from the sliced-PDF stack
    (iid Gaussian noise ``sigma_G`` per G(r) point)."""
    eps0 = (torch.zeros(6, dtype=_DT) if strain0 is None
            else torch.as_tensor(strain0, dtype=_DT))

    def f(e6):
        return sliced_gr_stack(crystal_tensor, r, pairs, qhats, e6,
                               kernel_m=kernel_m, n_quad=n_quad,
                               **pdf_kwargs).reshape(-1)

    J = torch.func.jacfwd(f)(eps0)                          # (D*R, 6)
    return (J.T @ J) / (sigma_G ** 2)


def strain_crlb(
    crystal_tensor,
    r: torch.Tensor,
    pairs: PairList,
    qhats: torch.Tensor,
    *,
    sigma_G: float = 0.02,
    rank_rtol: float = 1e-9,
    **kwargs,
) -> Dict[str, object]:
    """Cramer--Rao analysis of the 6 strain components.

    Returns eigenvalues of the Fisher matrix, its effective rank (number of
    determinable strain directions), per-component CRLB in microstrain for the
    determinable subspace (pseudo-inverse), and the blind directions.
    """
    F = strain_fisher(crystal_tensor, r, pairs, qhats, sigma_G=sigma_G,
                      **kwargs)
    evals, evecs = torch.linalg.eigh(F)
    rank = int((evals > evals.max() * rank_rtol).sum())
    Finv = torch.linalg.pinv(F, rtol=rank_rtol)
    per_comp_ue = 1e6 * torch.sqrt(torch.diag(Finv).clamp(min=0.0))
    blind = evecs[:, evals <= evals.max() * rank_rtol]
    # a component with any weight in the null space is NOT determinable on its
    # own -- the pinv diagonal would misleadingly report a small (subspace-
    # restricted) value, so mark it unbounded instead.
    null_proj = (blind ** 2).sum(dim=1)
    per_comp_ue = torch.where(null_proj > 1e-6,
                              torch.full_like(per_comp_ue, float("inf")),
                              per_comp_ue)
    return {
        "fisher": F,
        "eigenvalues": evals,
        "rank": rank,
        "per_component_ue": per_comp_ue,     # inf-like large values = blind
        "blind_directions": blind,           # (6, 6-rank) Voigt-space basis
    }


def recover_strain(
    crystal_tensor,
    r: torch.Tensor,
    pairs: PairList,
    qhats: torch.Tensor,
    y_obs: torch.Tensor,          # (D, R) measured sliced PDFs
    *,
    strain_init: Optional[torch.Tensor] = None,
    n_iter: int = 4,
    kernel_m: int = 8,
    n_quad: int = 128,
    rank_rtol: float = 1e-9,
    **pdf_kwargs,
) -> Dict[str, object]:
    """Gauss--Newton fit of the 6 Voigt strain components to sliced PDFs.

    The forward is nearly linear in strain at the 1e-3 level, so a few GN
    steps converge; the pseudo-inverse confines updates to the determinable
    subspace (blind directions stay at their initial value).
    """
    eps = (torch.zeros(6, dtype=_DT) if strain_init is None
           else torch.as_tensor(strain_init, dtype=_DT).clone())
    y = torch.as_tensor(y_obs, dtype=_DT).reshape(-1)

    def f(e6):
        return sliced_gr_stack(crystal_tensor, r, pairs, qhats, e6,
                               kernel_m=kernel_m, n_quad=n_quad,
                               **pdf_kwargs).reshape(-1)

    for _ in range(n_iter):
        J = torch.func.jacfwd(f)(eps)
        resid = y - f(eps)
        eps = eps + torch.linalg.pinv(J, rtol=rank_rtol) @ resid
    resid = y - f(eps)
    return {
        "strain": eps,
        "residual_rms": float(resid.pow(2).mean().sqrt()),
    }
