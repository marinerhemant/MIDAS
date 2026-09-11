"""Real-space distortion field of a dislocation network (finite segments).

The third face of the same physics. :mod:`midas_ddd.fourier` gives ``u~(q)`` for
small-angle and near-Bragg scattering; this gives ``beta(r) = grad u`` in real
space, which is what DFXM images.

Why not just superpose the Stroh solutions
-------------------------------------------
:func:`midas_ddd.elasticity` solves an **infinite straight** dislocation exactly,
in full anisotropy. A DDD network is made of **finite segments**, and superposing
infinite lines through their endpoints is not an approximation of it -- it is a
different configuration. For a closed loop it is qualitatively wrong: the loop's
entire relaxation volume lives in the closure, which a set of infinite lines does
not have. The DFXM contrast of an ExaDiS loop computed that way would be the
contrast of something else.

Mura's formula
--------------
For a dislocation loop C with Burgers vector b, the elastic distortion is a line
integral over the loop of the Green's-function gradient (Mura, *Micromechanics
of Defects in Solids*, 1987):

    beta_ij(x) = line-integral_C  eps_jnh C_pqmn b_m G_ip,q(x - x') dl_h(x')

This module evaluates it by Gauss-Legendre quadrature along each segment, with
the **isotropic** Green's function

    G_ij(R)   = [ (3-4nu) delta_ij / R + R_i R_j / R^3 ] / (16 pi mu (1-nu))

and a non-singular core radius ``a`` entering as ``R -> sqrt(R^2 + a^2)`` (Cai,
Arsenlis, Weinberger & Bulatov, JMPS 2006), which keeps the field finite on the
line itself.

Isotropic, deliberately: there is no closed-form finite-segment solution in full
anisotropy, which is exactly why production DDD codes use the non-singular
isotropic kernel. The anisotropic Stroh path remains available for infinite
straight lines, where it is exact.

ELASTIC distortion, not total -- and which one you want depends on the modality
------------------------------------------------------------------------------
Mura's formula returns the **elastic** distortion ``beta^e``. The total is
``beta = beta^e + beta^p``, where the plastic eigendistortion ``beta^p`` is a
delta function on the cut surface. :mod:`midas_ddd.fourier` works with the
total; this module gives the elastic part. They are both physical and they are
not the same thing:

* **DFXM images the elastic distortion.** The lattice is continuous across the
  cut for a perfect dislocation -- the cut is bookkeeping, not a real
  discontinuity in the crystal -- so ``beta^e`` is what sets the contrast.
* **The scattering kernels use the total**, because the eigendistortion is how
  the loop's inserted or removed material enters.

The two are tied together quantitatively, and the relation is gated in
``test_realspace.py::test_elastic_and_total_differ_by_exactly_the_relaxation_volume``.
Integrating the trace over a large ball centred on a prismatic loop,

    real-space (elastic):  int tr(beta^e) dV / dV  =  (2/3)(1-2nu)/(1-nu)
    Fourier    (total)  :  sphere average of q.u~ / dV  =  (1+nu)/(3(1-nu))

and those two sum to exactly 1, i.e. they differ by the plastic term's
``int tr(beta^p) dV = -dV``. Verified across four Poisson ratios.

This is worth knowing because the discrepancy looks like a bug the first time
you meet it: the two kernels disagreed by a clean functional factor, and the
resolution was that they were computing different (both correct) quantities.

Known limit: the core regularisation biases the DILATATIONAL part
------------------------------------------------------------------
Substituting ``R -> sqrt(R^2 + a^2)`` in G adds a spurious term to ``tr(beta)``
whose weight scales as ``lambda``, i.e. as ``1/(1-2nu)``, while the physical
term scales as ``(1-2nu)``. The relative error therefore grows sharply as
``nu -> 1/2``: measured against the closed form for the ball trace integral it
is +0.1 % at nu = 0.29 but **+32 % at nu = 0.47**, and refining the grid does
not remove it (it converges to the wrong value; only ``a -> 0`` or a larger ball
recovers the closed form). The rotational part -- what the Burgers circuit and
the infinite-screw comparison test, and what dominates DFXM contrast -- is
unaffected. Treat dilatational quantities from this kernel with suspicion above
nu ~ 0.4.

Everything is torch-differentiable and batched over field points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

__all__ = [
    "SegmentDislocation",
    "green_gradient_isotropic",
    "lame_from_voigt",
    "network_distortion",
    "segment_dislocations",
]

# 8-point Gauss-Legendre on [-1, 1]. The Green's gradient falls as 1/R^2, so a
# high-order rule matters once a field point comes within a few segment lengths.
_GL8_X = (-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
          -0.1834346424956498, 0.1834346424956498, 0.5255324099163290,
          0.7966664774136267, 0.9602898564975363)
_GL8_W = (0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
          0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
          0.2223810344533745, 0.1012285362903763)


def lame_from_voigt(C6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best-fit isotropic Lame constants ``(lambda, mu)`` from a Voigt stiffness.

    For a genuinely isotropic ``C6`` this is exact (``lambda = C12``,
    ``mu = C44``). For an anisotropic one it is the Voigt average, and the caller
    should know that is what they are getting -- see the module docstring on why
    the real-space finite-segment kernel is isotropic at all.
    """
    C11, C12, C44 = C6[0, 0], C6[0, 1], C6[3, 3]
    # Voigt averages; reduce to (C12, C44) exactly when the crystal is isotropic.
    mu = (C11 - C12 + 3.0 * C44) / 5.0
    lam = C11 - 2.0 * mu
    return lam, mu


def green_gradient_isotropic(
    R: torch.Tensor, lam: torch.Tensor, mu: torch.Tensor, a: float = 0.0
) -> torch.Tensor:
    """``dG_ij/dx_k`` for isotropic elasticity. ``R`` is ``(..., 3)``; out ``(..., 3, 3, 3)``.

    ``G_ij(R) = [ (3-4nu) delta_ij / Ra + R_i R_j / Ra^3 ] / (16 pi mu (1-nu))``
    with ``Ra = sqrt(|R|^2 + a^2)``, so

    ``G_ij,k = [ -(3-4nu) delta_ij R_k / Ra^3
                 + (delta_ik R_j + delta_jk R_i) / Ra^3
                 - 3 R_i R_j R_k / Ra^5 ] / (16 pi mu (1-nu))``.

    The ``a`` regularisation is what keeps the field finite on the dislocation
    line; with ``a = 0`` this diverges there, as the classical solution does.
    """
    nu = lam / (2.0 * (lam + mu))
    pref = 1.0 / (16.0 * math.pi * mu * (1.0 - nu))

    # Explicit None-indexing throughout: chained unsqueeze() silently puts axes
    # in the wrong place once R carries two batch dimensions (field point,
    # quadrature node), and the resulting broadcast error is far from the cause.
    R2 = (R * R).sum(dim=-1) + a * a                    # (...)
    Ra = torch.sqrt(R2)
    Ra3 = (Ra ** 3)[..., None, None, None]              # (..., 1, 1, 1)
    Ra5 = (Ra ** 5)[..., None, None, None]

    Ri = R[..., :, None, None]                          # (..., 3, 1, 1)
    Rj = R[..., None, :, None]                          # (..., 1, 3, 1)
    Rk = R[..., None, None, :]                          # (..., 1, 1, 3)

    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    d_ij = eye[:, :, None]                              # (3, 3, 1)
    d_ik = eye[:, None, :]                              # (3, 1, 3)
    d_jk = eye[None, :, :]                              # (1, 3, 3)

    term1 = -(3.0 - 4.0 * nu) * d_ij * Rk / Ra3
    term2 = (d_ik * Rj + d_jk * Ri) / Ra3
    term3 = -3.0 * Ri * Rj * Rk / Ra5
    return pref * (term1 + term2 + term3)


@dataclass
class SegmentDislocation:
    """A finite straight dislocation segment, with the Stroh duck type.

    Exposes ``displacement_gradient(positions)`` with the same signature as
    :class:`midas_ddd.elasticity`-backed ``StrohDislocation``, so a list of these
    drops straight into
    :func:`midas_dfxm.dislocation.dislocation_deformation_field` with no change
    to that function.
    """

    start_um: torch.Tensor           # (3,)
    end_um: torch.Tensor             # (3,)
    burgers_um: torch.Tensor         # (3,) physical, micrometers
    lam: torch.Tensor
    mu: torch.Tensor
    core_radius_um: float = 2.556e-4
    n_quad: int = 8

    def displacement_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """``beta_ij`` at each field point. ``positions`` ``(N, 3)`` -> ``(N, 3, 3)``."""
        return _segment_distortion(
            positions, self.start_um, self.end_um, self.burgers_um,
            self.lam, self.mu, self.core_radius_um, self.n_quad)


def _point_segment_distance(positions, x1, x2):
    """Closest approach from each field point to the segment. ``(N,)``."""
    seg = x2 - x1
    L2 = (seg * seg).sum()
    if float(L2.detach()) == 0.0:
        return torch.linalg.vector_norm(positions - x1, dim=-1)
    t = ((positions - x1) @ seg) / L2
    t = t.clamp(0.0, 1.0)
    closest = x1.unsqueeze(0) + t.unsqueeze(-1) * seg.unsqueeze(0)
    return torch.linalg.vector_norm(positions - closest, dim=-1)


def _segment_distortion(positions, x1, x2, b, lam, mu, a, n_quad,
                        panels_per_distance: float = 4.0, max_panels: int = 512):
    """Mura line integral over one straight segment. ``(N, 3, 3)``.

    The integrand falls as ``1/R^2``, so a fixed Gauss rule spread over the whole
    segment misses the peak entirely whenever the segment is long compared with
    the distance to the field point: an 8-point rule on a 400 um segment probed
    at 0.5 um came out ~1000x low and *rising* with r instead of falling. The
    segment is therefore split into panels no longer than
    ``d_min / panels_per_distance``, with ``d_min`` the closest approach of any
    field point (floored at the core radius so a point on the line cannot demand
    infinitely many panels).
    """
    dtype, device = positions.dtype, positions.device
    seg = x2 - x1
    L = torch.linalg.norm(seg)
    if float(L.detach()) == 0.0:
        return torch.zeros(positions.shape[0], 3, 3, dtype=dtype, device=device)
    t_hat = seg / L

    d_min = float(_point_segment_distance(positions, x1, x2).min().detach())
    d_min = max(d_min, float(a), 1e-12)
    n_panels = int(min(max_panels,
                       max(1, math.ceil(float(L.detach()) /
                                        (d_min / panels_per_distance)))))

    xs = torch.tensor(_GL8_X, dtype=dtype, device=device)
    ws = torch.tensor(_GL8_W, dtype=dtype, device=device)

    # Panel edges in [0, 1] along the segment; nodes inside each panel.
    edges = torch.linspace(0.0, 1.0, n_panels + 1, dtype=dtype, device=device)
    lo, hi = edges[:-1].unsqueeze(-1), edges[1:].unsqueeze(-1)        # (P, 1)
    s_nodes = (lo + hi) / 2.0 + (hi - lo) / 2.0 * xs.unsqueeze(0)      # (P, Q)
    w_nodes = ((hi - lo) / 2.0) * ws.unsqueeze(0)                      # (P, Q)
    s_flat = s_nodes.reshape(-1)                                       # (P*Q,)
    w_flat = w_nodes.reshape(-1)

    src = x1.unsqueeze(0) + s_flat.unsqueeze(-1) * seg.unsqueeze(0)    # (PQ, 3)
    R = positions.unsqueeze(1) - src.unsqueeze(0)                      # (N, PQ, 3)

    dG = green_gradient_isotropic(R, lam, mu, a)                       # (N, PQ, 3,3,3)

    eye = torch.eye(3, dtype=dtype, device=device)
    C = (lam * torch.einsum("pq,mn->pqmn", eye, eye)
         + mu * (torch.einsum("pm,qn->pqmn", eye, eye)
                 + torch.einsum("pn,qm->pqmn", eye, eye)))

    # T_in = C_pqmn b_m G_ip,q   (leading axes: field point, quadrature node)
    T = torch.einsum("pqmn,m,abipq->abin", C, b, dG)                   # (N, PQ, 3, 3)

    # beta_ij = eps_jnh T_in t_h, with dl_h = t_h L ds
    eps = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    for i, j, k, v in ((0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0),
                       (0, 2, 1, -1.0), (2, 1, 0, -1.0), (1, 0, 2, -1.0)):
        eps[i, j, k] = v
    integrand = torch.einsum("jnh,abin,h->abij", eps, T, t_hat)        # (N, PQ, 3, 3)
    weights = (L * w_flat).reshape(1, -1, 1, 1)
    return (integrand * weights).sum(dim=1)


def segment_dislocations(
    net,
    C6: torch.Tensor,
    *,
    core_radius_um: Optional[float] = None,
    n_quad: int = 8,
) -> list:
    """Every segment of ``net`` as a :class:`SegmentDislocation`.

    Open lines are included: the real-space superposition of segments is well
    defined for any network (it is just the field of those segments), so a
    deformation structure images perfectly well in DFXM. Its small-angle
    signature is a separate matter: no relaxation volume, so nothing as
    ``q -> 0``, but a finite-q sheet per edge or mixed line
    (:func:`midas_ddd.line_small_angle_amplitude`).
    """
    lam, mu = lame_from_voigt(C6.to(dtype=net.nodes_um.dtype,
                                    device=net.nodes_um.device))
    a = core_radius_um if core_radius_um is not None else net.b_magnitude_um
    burgers = net.burgers_um()
    out = []
    for s, (i, j) in enumerate(net.segments.tolist()):
        out.append(SegmentDislocation(
            start_um=net.nodes_um[i], end_um=net.nodes_um[j],
            burgers_um=burgers[s], lam=lam, mu=mu,
            core_radius_um=float(a), n_quad=n_quad))
    return out


def network_distortion(
    positions_um: torch.Tensor,
    net,
    C6: torch.Tensor,
    *,
    core_radius_um: Optional[float] = None,
    n_quad: int = 8,
    chunk: int = 4096,
) -> torch.Tensor:
    """``beta(r)`` from every segment in the network. ``(N, 3, 3)``.

    Distortions superpose in linear elasticity, so this is a plain sum over
    segments. ``chunk`` bounds peak memory: the intermediate is
    ``(chunk, n_quad, 3, 3, 3)``.
    """
    positions_um = torch.as_tensor(positions_um, dtype=net.nodes_um.dtype,
                                   device=net.nodes_um.device)
    segs = segment_dislocations(net, C6, core_radius_um=core_radius_um,
                                n_quad=n_quad)
    out = torch.zeros(positions_um.shape[0], 3, 3,
                      dtype=positions_um.dtype, device=positions_um.device)
    for lo in range(0, positions_um.shape[0], chunk):
        p = positions_um[lo:lo + chunk]
        acc = torch.zeros(p.shape[0], 3, 3, dtype=p.dtype, device=p.device)
        for sd in segs:
            acc = acc + sd.displacement_gradient(p)
        out[lo:lo + chunk] = acc
    return out
