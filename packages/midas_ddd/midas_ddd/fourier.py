"""Fourier-space displacement field of a dislocation network.

One kernel, three evaluation points::

    q . u~(q)        -> small-angle scattering        (midas_saxs)
    (G+q) . u~(q)    -> near-Bragg diffuse / Huang    (midas_defect)
    inverse FT       -> real-space beta(r)            (midas_dfxm)

Physics
-------
A dislocation loop bounding a surface ``S`` with Burgers vector ``b`` carries the
plastic eigen-distortion ``beta^p_ij(r) = -b_i n_j delta_S(r)``, whose transform
is ``beta~^p_ij(q) = -b_i I_j(q)`` with the **surface form factor**

    I_j(q) = int_S n_j exp(-i q.r) dS.

Mechanical equilibrium ``C_ijkl d_j d_l u_k = C_ijkl d_j beta^p_kl`` becomes, in
Fourier space with the acoustic (Christoffel) tensor ``K_ik(q) = C_ijkl q_j q_l``,

    u~_k(q) = i [K^-1(q)]_ki C_ijmn q_j b_m I_n(q).

Everything is torch-differentiable in node positions, Burgers vectors and the
elastic constants.

The cut surface is physical, not a gauge
----------------------------------------
Two surfaces spanning the same loop differ by ``dI = lambda q``, which changes
``u~`` by ``i lambda b`` -- a displacement jump of ``b`` across the cut. It is
tempting to dodge the choice by keeping only the part of ``I`` perpendicular to
``q``, which Stokes gives for free from a pure line integral
(``I_perp = i (q x J)/q^2`` with ``J = closed-integral dl exp(-i q.r)``).

**Do not.** Measured directly: for a prismatic loop probed along its own normal
-- the direction where the loop scatters most strongly -- the transverse
projection gives *exactly zero* while the true surface integral gives the full
signal. The q-parallel part of ``I`` is where the relaxation volume lives. So
this module takes an explicit cut surface: the fan triangulation of each closed
loop from its centroid, documented, reproducible, and shared with the FFT
reference implementation so the two can be compared.

Scope: closed loops
-------------------
A finite cut surface exists only for a **closed circuit**. Open lines -- the
deformation population -- have no bounding surface, carry no relaxation volume,
and contribute nothing as ``q -> 0``; their small-angle signature is a weak
transverse streak that this kernel does not model. :func:`u_tilde` therefore
consumes the loop inventory from :func:`midas_ddd.find_loops` and tells you, in
:class:`FourierResult`, how much line length it ignored.

The gate
--------
For isotropic elasticity and a prismatic loop (``b || A || n``), the small-q
limit has an exact closed form, verified here to 2e-15 over 1200 random
directions and four different moduli::

    q . u~(q -> 0) = i dV [ kappa + (1 - kappa) (n.qhat)^2 ],
        dV = b . A,   kappa = lambda / (lambda + 2 mu).

Note what that says: the limit is **direction dependent**. A loop does not
scatter like a compact particle of volume ``dV`` -- it varies between
``kappa dV`` in its own plane and ``dV`` along its normal, a contrast ratio of
``(lambda + 2 mu)/lambda``. A void, by contrast, is isotropic. That anisotropy
*is* the loop-versus-void discriminator, and it is present at ``q -> 0``, not
only at finite q. See :func:`prismatic_loop_small_q_limit`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .elasticity import _voigt_to_tensor

__all__ = [
    "FourierResult",
    "acoustic_tensor",
    "isotropic_stiffness",
    "loop_cut_surface",
    "loop_vertices",
    "loop_small_q_limit",
    "prismatic_loop_small_q_limit",
    "q_dot_u_tilde",
    "small_angle_amplitude",
    "prismatic_loop_small_q_limit_total",
    "q_dot_u_tilde_per_loop",
    "surface_form_factor",
    "u_tilde",
]


# Degree-5 symmetric Gauss rule on the unit triangle. Exact for polynomials up
# to order 5; for exp(-i q.r) the error is O((q h)^6), which
# `_quadrature_error_estimate` reports against.
#
# The published coefficients are truncated to 10 digits, so their weights sum to
# 0.9999999996 and the barycentric triples to 0.9999999999. Left alone, that
# 4e-10 deficit is inherited *exactly* by the q -> 0 limit -- the surface form
# factor returns 0.9999999996 A instead of A -- and it showed up as a constant
# 4e-10 relative error against the closed-form gate, independent of q, which is
# precisely the signature of a normalisation bug rather than a physics one.
# Renormalising both makes the q = 0 limit exact to machine precision, which is
# the property the dV gate depends on.
def _normalised_tri_rule(rows):
    wsum = sum(r[3] for r in rows)
    out = []
    for l0, l1, l2, w in rows:
        lsum = l0 + l1 + l2
        out.append((l0 / lsum, l1 / lsum, l2 / lsum, w / wsum))
    return tuple(out)


_TRI_BARY = _normalised_tri_rule((
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.225),
    (0.0597158717, 0.4701420641, 0.4701420641, 0.1323941527),
    (0.4701420641, 0.0597158717, 0.4701420641, 0.1323941527),
    (0.4701420641, 0.4701420641, 0.0597158717, 0.1323941527),
    (0.7974269853, 0.1012865073, 0.1012865073, 0.1259391805),
    (0.1012865073, 0.7974269853, 0.1012865073, 0.1259391805),
    (0.1012865073, 0.1012865073, 0.7974269853, 0.1259391805),
))


def isotropic_stiffness(lam: float, mu: float, *, dtype=torch.float64, device=None):
    """6x6 Voigt stiffness for an isotropic solid, from Lame constants.

    Provided because the analytic small-q gate is an isotropic result, and
    because :func:`midas_ddd.cubic_stiffness` deliberately refuses the isotropic
    limit (the Stroh sextic solution is degenerate there) -- but the Fourier
    kernel does not use Stroh, so isotropic is perfectly well posed here.
    """
    lam_t = torch.as_tensor(lam, dtype=dtype, device=device)
    mu_t = torch.as_tensor(mu, dtype=dtype, device=device)
    z = torch.zeros((), dtype=dtype, device=device)
    c11 = lam_t + 2.0 * mu_t
    rows = [
        torch.stack([c11, lam_t, lam_t, z, z, z]),
        torch.stack([lam_t, c11, lam_t, z, z, z]),
        torch.stack([lam_t, lam_t, c11, z, z, z]),
        torch.stack([z, z, z, mu_t, z, z]),
        torch.stack([z, z, z, z, mu_t, z]),
        torch.stack([z, z, z, z, z, mu_t]),
    ]
    return torch.stack(rows, dim=0)


def acoustic_tensor(C4: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Christoffel tensor ``K_ik(q) = C_ijkl q_j q_l``. ``(..., 3, 3)``.

    Singular at ``q = 0`` (it scales as ``q^2``), which is why the small-q limit
    is taken as a *limit* and never evaluated at exactly zero.
    """
    return torch.einsum("ijkl,...j,...l->...ik", C4, q, q)


# ---------------------------------------------------------------------------
# Cut surface
# ---------------------------------------------------------------------------

def loop_vertices(net, loop) -> torch.Tensor:
    """The loop's vertices, unwrapped. ``(n, 3)`` micrometers.

    Rebuilt by walking **minimum-image** segment vectors, so a loop straddling a
    periodic boundary is reconstructed unwrapped rather than folded into a
    near-zero-area sliver, then anchored at its first node's true position.
    """
    segvec = net.segment_vectors_um()
    r = torch.zeros(3, dtype=net.nodes_um.dtype, device=net.nodes_um.device)
    verts = [r]
    for s, o in zip(loop.segment_indices, loop.orientations):
        r = r + segvec[s] * o
        verts.append(r)
    V = torch.stack(verts[:-1])                     # drop the repeated closure point
    return V + net.nodes_um[loop.node_indices[0]]


def loop_cut_surface(net, loop, *, n_rings: int = 1
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triangulate one closed loop's cut surface from its centroid.

    Returns ``(v0, v1, v2)``, each ``(T, 3)`` in micrometers, oriented so their
    area vectors sum to the loop's area vector.

    ``n_rings`` subdivides **radially**, in concentric rings from the centroid
    out to the boundary. A plain fan (``n_rings = 1``) is tempting and wrong for
    SAXS: its triangles have an edge of length R running from the centroid to
    each vertex, so the quadrature parameter ``|q| h`` is set by the loop RADIUS
    no matter how finely the polygon itself is discretised. Subdividing the
    circumference alone does nothing for it. Since SAXS wants ``qR`` of order
    several, radial subdivision is what makes the interesting q range usable at
    all -- :func:`u_tilde` picks ``n_rings`` automatically from ``q_max``.
    """
    V = loop_vertices(net, loop)
    c = V.mean(dim=0, keepdim=True)
    n_rings = max(1, int(n_rings))

    if n_rings == 1:
        v1 = V
        v2 = torch.roll(V, -1, dims=0)
        return c.expand_as(v1), v1, v2

    # Ring k sits at radial fraction k / n_rings between the centroid and V.
    fracs = [k / n_rings for k in range(n_rings + 1)]
    rings = [c + f * (V - c) for f in fracs]        # rings[0] is the centroid

    v0s, v1s, v2s = [], [], []
    # Innermost: a fan of n triangles from the centroid to ring 1.
    r1 = rings[1]
    v0s.append(c.expand_as(r1)); v1s.append(r1); v2s.append(torch.roll(r1, -1, dims=0))
    # Outer annuli: each quad split into two triangles, same circulation sense.
    for k in range(1, n_rings):
        a, b = rings[k], rings[k + 1]
        a_n, b_n = torch.roll(a, -1, dims=0), torch.roll(b, -1, dims=0)
        v0s.append(a);   v1s.append(b);   v2s.append(b_n)
        v0s.append(a);   v1s.append(b_n); v2s.append(a_n)
    return torch.cat(v0s), torch.cat(v1s), torch.cat(v2s)


def _rings_for_q(net, loop, q, *, target_qh: float = 0.4, max_rings: int = 64) -> int:
    """How many radial rings keep ``|q| h`` under ``target_qh``."""
    V = loop_vertices(net, loop)
    R = float(torch.linalg.vector_norm(
        V - V.mean(dim=0, keepdim=True), dim=-1).max().detach())
    qmax = float(torch.linalg.vector_norm(q, dim=-1).max().detach())
    if R <= 0 or qmax <= 0:
        return 1
    return int(min(max_rings, max(1, math.ceil(R * qmax / target_qh))))


def surface_form_factor(
    v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """``I_j(q) = int_S n_j exp(-i q.r) dS`` over a triangulated surface.

    Parameters
    ----------
    v0, v1, v2 : (T, 3)
        Triangle vertices, micrometers.
    q : (Q, 3)
        Scattering vectors, inverse micrometers (so ``q.r`` is dimensionless).

    Returns
    -------
    (Q, 3) complex tensor, units of µm².

    At ``q = 0`` the quadrature weights sum to 1, so this returns the exact
    signed area vector -- which is what makes the ``dV`` gate pass at machine
    precision rather than at quadrature precision.
    """
    e1 = v1 - v0
    e2 = v2 - v0
    # Area vector per triangle; its direction is the surface normal, and its
    # magnitude the area, so n dS = 0.5 (e1 x e2) needs no separate normalisation.
    a_vec = 0.5 * torch.linalg.cross(e1, e2)                     # (T, 3)

    total = torch.zeros(q.shape[0], 3, dtype=torch.complex128 if q.dtype ==
                        torch.float64 else torch.complex64, device=q.device)
    a_c = a_vec.to(total.dtype)
    for l0, l1, l2, w in _TRI_BARY:
        pts = l0 * v0 + l1 * v1 + l2 * v2                        # (T, 3)
        ang = q @ pts.transpose(0, 1)                            # (Q, T), real
        # exp(-i ang) from cos/sin, rather than promoting `ang` to complex and
        # calling complex exp. This inner loop runs
        # (7 quadrature nodes x T triangles x Q pixels x L loops) times -- ~4e8
        # for a modest frame -- so it is essentially the whole cost of a render.
        phase = torch.complex(torch.cos(ang), -torch.sin(ang))   # (Q, T)
        total = total + w * (phase @ a_c)                        # (Q, 3)
    return total


def _quadrature_error_estimate(v0, v1, v2, q) -> float:
    """Worst-case ``|q| h`` over the triangulation; the rule's error is O((qh)^6)."""
    h = torch.maximum(
        torch.linalg.vector_norm(v1 - v0, dim=-1),
        torch.maximum(torch.linalg.vector_norm(v2 - v0, dim=-1),
                      torch.linalg.vector_norm(v2 - v1, dim=-1)),
    ).max()
    qmax = torch.linalg.vector_norm(q, dim=-1).max()
    # Diagnostic only -- detach so it never sits on the autograd graph.
    return float((h * qmax).detach())


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

def _loop_contributions(net, q, C4, Kinv, loops, cdt, rings) -> torch.Tensor:
    """``u~`` per loop, sharing one inverted acoustic tensor. ``(L, Q, 3)``.

    ``K^-1(q)`` depends only on q, never on which loop is being summed, so
    inverting it once and reusing it across loops turns an O(L x Q) stack of 3x3
    inversions into a single one. With 40 loops on a 256^2 frame that is the
    difference between 8 s and well under 1 s.
    """
    if not loops:
        return torch.zeros(0, q.shape[0], 3, dtype=cdt, device=q.device)
    out = []
    qc = q.to(cdt)
    for lp, nr in zip(loops, rings):
        v0, v1, v2 = loop_cut_surface(net, lp, n_rings=nr)
        I = surface_form_factor(v0, v1, v2, q)                    # (Q, 3) complex
        b = (lp.burgers_b * net.b_magnitude_um).to(cdt)           # µm
        # u~_k = i [K^-1]_ki C_ijmn q_j b_m I_n
        #
        # Contract the q-INDEPENDENT part first: D_ijn = C_ijmn b_m is a fixed
        # (3,3,3) per loop. The other order builds C_ijmn q_j as a (Q,3,3,3)
        # intermediate -- 432 MB of complex128 for a 1024^2 detector, i.e. the
        # difference between rendering a frame and running out of memory.
        D = torch.einsum("ijmn,m->ijn", C4.to(cdt), b)            # (3, 3, 3)
        rhs = torch.einsum("ijn,...j,...n->...i", D, qc, I)       # (Q, 3)
        out.append(1j * torch.einsum("...ki,...i->...k", Kinv, rhs))
    return torch.stack(out)


def q_dot_u_tilde_per_loop(net, q, C6, *, loops=None) -> torch.Tensor:
    """``q . u~`` for each loop separately. ``(L, Q)`` complex.

    The incoherent (ensemble-averaged) sum needs ``|A_loop|^2`` loop by loop,
    which would otherwise mean one full :func:`u_tilde` call per loop and one
    redundant acoustic-tensor inversion per loop with it.
    """
    from .validate import find_loops

    if loops is None:
        loops = find_loops(net)
    q = torch.as_tensor(q, dtype=net.nodes_um.dtype, device=net.nodes_um.device)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if bool((torch.linalg.vector_norm(q, dim=-1) == 0).any()):
        raise ValueError("q contains exactly zero, where K(q) is singular; "
                         "the q -> 0 behaviour is a limit, not a value.")
    C4 = _voigt_to_tensor(C6.to(dtype=net.nodes_um.dtype, device=net.nodes_um.device))
    cdt = torch.complex128 if net.nodes_um.dtype == torch.float64 else torch.complex64
    Kinv = torch.linalg.inv(acoustic_tensor(C4, q)).to(cdt)
    rings = [_rings_for_q(net, lp, q) for lp in loops]
    contrib = _loop_contributions(net, q, C4, Kinv, loops, cdt, rings)   # (L, Q, 3)
    return torch.einsum("lqi,qi->lq", contrib, q.to(cdt))


@dataclass
class FourierResult:
    """``u~(q)`` plus what the kernel could and could not account for."""

    u_tilde: torch.Tensor            # (Q, 3) complex, µm * µm^3 -> µm^4
    q: torch.Tensor                  # (Q, 3) inverse µm
    n_loops: int
    n_segments_in_loops: int
    n_segments_ignored: int
    line_length_ignored_um: float
    quadrature_qh: float
    warnings: List[str]
    #: Per-loop contributions, ``(L, Q, 3)``, populated only when
    #: ``per_loop=True``. The incoherent (ensemble) sum needs these, and
    #: computing them in the same pass as the total is what keeps every surface
    #: form factor from being evaluated twice.
    u_tilde_per_loop: Optional[torch.Tensor] = None

    def __repr__(self) -> str:       # pragma: no cover - cosmetic
        return (f"FourierResult({self.q.shape[0]} q-points, {self.n_loops} loops, "
                f"{self.n_segments_ignored} open segments ignored, "
                f"qh={self.quadrature_qh:.3g})")


def u_tilde(
    net,
    q: torch.Tensor,
    C6: torch.Tensor,
    *,
    loops: Optional[Sequence] = None,
    qh_warn: float = 0.5,
    per_loop: bool = False,
    n_rings: Optional[int] = None,
) -> FourierResult:
    """Fourier displacement field of every closed loop in ``net``.

    Parameters
    ----------
    net : DislocationNetwork
    q : (Q, 3) tensor
        Scattering vectors in **inverse micrometers**. Must not contain exactly
        zero -- the acoustic tensor is singular there; take a limit instead.
    C6 : (6, 6) tensor
        Voigt stiffness, any consistent unit (only ratios enter ``u~``).
    loops
        Loop inventory. Defaults to :func:`midas_ddd.find_loops(net)`.
    n_rings
        Radial subdivisions of each loop's cut surface. ``None`` (default) picks
        one per loop so the quadrature parameter ``|q| h`` stays near 0.4 over
        the requested q range. Pass an integer to pin it.

    Returns
    -------
    FourierResult
        ``u_tilde`` has shape ``(Q, 3)``, complex.

    Notes
    -----
    Open (non-loop) segments are **ignored**, and the result reports how much
    line length that was. They have no bounding surface and no relaxation
    volume; see the module docstring.
    """
    from .validate import find_loops

    if loops is None:
        loops = find_loops(net)

    q = torch.as_tensor(q, dtype=net.nodes_um.dtype, device=net.nodes_um.device)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    qn = torch.linalg.vector_norm(q, dim=-1)
    if bool((qn == 0).any()):
        raise ValueError(
            "q contains exactly zero, where the acoustic tensor K(q) = C q q is "
            "singular. The q -> 0 behaviour is a limit, not a value: evaluate at "
            "small finite q, or use prismatic_loop_small_q_limit() for the "
            "closed form.")

    C4 = _voigt_to_tensor(C6.to(dtype=net.nodes_um.dtype, device=net.nodes_um.device))
    cdt = torch.complex128 if net.nodes_um.dtype == torch.float64 else torch.complex64

    K = acoustic_tensor(C4, q)                                    # (Q, 3, 3)
    Kinv = torch.linalg.inv(K).to(cdt)                            # (Q, 3, 3)

    # Radial subdivision, chosen per loop so the quadrature parameter |q|h stays
    # small across the requested q range. Without it a plain fan caps usable q at
    # roughly 0.5/R, which for a 5 nm loop is well below the interesting range.
    rings = ([max(1, int(n_rings))] * len(loops) if n_rings is not None
             else [_rings_for_q(net, lp, q) for lp in loops])
    contrib = _loop_contributions(net, q, C4, Kinv, loops, cdt, rings)
    total = contrib.sum(dim=0) if contrib.shape[0] else torch.zeros(
        q.shape[0], 3, dtype=cdt, device=q.device)
    n_in_loops = sum(len(lp.segment_indices) for lp in loops)
    worst_qh = max((_quadrature_error_estimate(
                        *loop_cut_surface(net, lp, n_rings=nr), q)
                    for lp, nr in zip(loops, rings)), default=0.0)

    seg_len = net.segment_lengths_um()
    in_loop = torch.zeros(net.n_segments, dtype=torch.bool, device=seg_len.device)
    for lp in loops:
        in_loop[torch.tensor(lp.segment_indices, dtype=torch.int64,
                             device=seg_len.device)] = True
    ignored_len = float(seg_len[~in_loop].sum().detach())

    warnings: List[str] = []
    if worst_qh > qh_warn:
        warnings.append(
            f"triangle quadrature is being pushed: max |q|h = {worst_qh:.2f}. The "
            f"degree-5 rule errs as O((qh)^6); subdivide the loops or lower q_max.")
    n_ignored = int((~in_loop).sum())
    if n_ignored:
        warnings.append(
            f"{n_ignored} open segment(s) ({ignored_len:.3g} um of line) are not "
            f"in any closed loop and contribute nothing here. Open lines have no "
            f"cut surface and no relaxation volume; their small-angle signature "
            f"is a transverse streak this kernel does not model.")

    return FourierResult(
        u_tilde=total, q=q, n_loops=len(loops),
        n_segments_in_loops=n_in_loops, n_segments_ignored=n_ignored,
        line_length_ignored_um=ignored_len, quadrature_qh=worst_qh,
        warnings=warnings,
        u_tilde_per_loop=contrib if per_loop else None,
    )


def q_dot_u_tilde(net, q, C6, **kw) -> torch.Tensor:
    """``q . u~(q)``, the small-angle scattering amplitude factor. ``(Q,)`` complex.

    The SAXS amplitude is ``A(q) = -i rho_e q.u~(q)``; this returns the
    ``q.u~`` factor so the electron-density scaling stays in
    :mod:`midas_saxs` where the material lives.
    """
    res = u_tilde(net, q, C6, **kw)
    return torch.einsum("...i,...i->...", res.q.to(res.u_tilde.dtype), res.u_tilde)


# ---------------------------------------------------------------------------
# The analytic gate
# ---------------------------------------------------------------------------

def small_angle_amplitude(net, q, C6, *, loops=None, per_loop=False):
    """TOTAL small-angle amplitude of the closed loops. ``(Q,)`` complex, µm^3.

    **Use this, not** :func:`q_dot_u_tilde`, **for small-angle scattering.**

    At small angle the scattering vector IS the deviation (``K = q``), and the
    Laue term -- the scattering from the loop's own extra or missing atoms --
    is the same order as the distortion term. Ehrhart, Trinkaus & Larson,
    *Diffuse scattering from dislocation loops*, Phys. Rev. B **25** (1982) 834,
    Eq. (8a):

        I ~ (1/v^2) | i q.s~(q)  +  b A~(q) |^2

    with only the first term being what :func:`q_dot_u_tilde` returns. Their
    Eq. (8b) recasts the sum in a form that is manifestly gauge-clean and is
    what is implemented here:

        amplitude = [q x (b - b')] . [q x A~(q)] / q^2,
        b'_m = q_j [K^-1]_jk C_klmn q_l b_n,
        A~(q) = surface form factor of the loop area = int dA exp(-i q.r)

    **The Laue term is not a correction, it inverts the answer.** For ``q``
    along the loop normal the cross product ``q x A~`` vanishes identically, so
    the total is EXACTLY ZERO -- while the distortion term alone is at its
    maximum there, ``dV``. The two cancel. The small-q law is

        q.u~_total -> i dV (1 - kappa) sin^2(theta),   kappa = lambda/(lambda+2mu)

    (verified to 7e-16 over 1500 directions and 5 moduli), against the
    distortion-only ``dV [kappa + (1-kappa) cos^2(theta)]``. The two sum to
    ``dV`` identically. A loop's small-angle scattering VANISHES along its
    normal and peaks in its plane -- the opposite of what the distortion term
    alone says.

    Near a Bragg peak the distortion term carries a factor ``G`` that the Laue
    term does not, so ``G/q >> 1`` and :mod:`midas_defect.huang` is unaffected;
    this correction is specific to small angle.
    """
    from .validate import find_loops

    if loops is None:
        loops = find_loops(net)
    q = torch.as_tensor(q, dtype=net.nodes_um.dtype, device=net.nodes_um.device)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    q2 = (q * q).sum(dim=-1)
    if bool((q2 == 0).any()):
        raise ValueError("q contains exactly zero; the small-angle limit is a "
                         "limit, not a value. See prismatic_loop_small_q_limit_total.")

    C4 = _voigt_to_tensor(C6.to(dtype=net.nodes_um.dtype, device=net.nodes_um.device))
    cdt = torch.complex128 if net.nodes_um.dtype == torch.float64 else torch.complex64
    Kinv = torch.linalg.inv(acoustic_tensor(C4, q))

    out = []
    rings = [_rings_for_q(net, lp, q) for lp in loops]
    for lp, nr in zip(loops, rings):
        b = lp.burgers_b * net.b_magnitude_um                     # (3,) µm
        # b'_m = q_j K^-1_jk C_klmn q_l b_n   -- the elastic "screening" of b
        bp = torch.einsum("qj,qjk,klmn,ql,n->qm", q, Kinv, C4, q, b)
        v0, v1, v2 = loop_cut_surface(net, lp, n_rings=nr)
        Astar = surface_form_factor(v0, v1, v2, q)                # (Q,3) complex µm^2
        lhs = torch.linalg.cross(q, (b.unsqueeze(0) - bp))        # (Q,3) real
        rhs = torch.linalg.cross(q.to(cdt), Astar)                # (Q,3) complex
        # The 1j matches q_dot_u_tilde's phase convention. Eq. (8a) writes
        # "i q.s~ + b A~", putting the i on the DISTORTION term only, so the
        # Eq. (8b) combination comes out real where q_dot_u_tilde is imaginary.
        # Only |.|^2 is observable so this is cosmetic, but two functions in one
        # module returning the same quantity with a relative factor of i is a
        # trap for anyone who combines them.
        out.append(1j * (lhs.to(cdt) * rhs).sum(dim=-1) / q2.to(cdt))
    if not out:
        z = torch.zeros(q.shape[0], dtype=cdt, device=q.device)
        return z.unsqueeze(0) if per_loop else z
    stack = torch.stack(out)
    return stack if per_loop else stack.sum(dim=0)


def prismatic_loop_small_q_limit_total(
    dV_um3: float,
    normal: Sequence[float],
    q_hat: torch.Tensor,
    lam: float,
    mu: float,
) -> torch.Tensor:
    """TOTAL small-angle limit for a prismatic loop: ``i dV (1-kappa) sin^2(theta)``.

    Distortion PLUS Laue (Ehrhart/Trinkaus/Larson 1982, Eq. 8b). Compare
    :func:`prismatic_loop_small_q_limit`, which is the distortion term alone and
    gives ``dV [kappa + (1-kappa) cos^2]`` -- the two sum to ``dV`` identically.

    The physically important difference: this one is **zero along the loop
    normal** and maximal in the loop plane. An exact null is a far better
    signature than a 2.5:1 contrast, because a void has no null at all -- but
    note the null survives variant averaging no better than the old contrast
    did: ``<sin^2>`` over isotropic normals is 2/3, direction-independent, so an
    unbiased population is still exactly degenerate with a void.
    """
    n = torch.as_tensor(normal, dtype=q_hat.dtype, device=q_hat.device)
    n = n / torch.linalg.norm(n)
    kappa = lam / (lam + 2.0 * mu)
    s2 = 1.0 - (q_hat @ n) ** 2
    return 1j * dV_um3 * (1.0 - kappa) * s2


def loop_small_q_limit(
    b_um: Sequence[float],
    A_um2: Sequence[float],
    q_hat: torch.Tensor,
    lam: float,
    mu: float,
) -> torch.Tensor:
    """Closed-form ``q . u~`` as ``q -> 0`` for ANY loop, isotropic elasticity.

        q . u~  ->  i [ lam (b.A) + 2 mu (qhat.b)(qhat.A) ] / (lam + 2 mu)

    This is the general statement; :func:`prismatic_loop_small_q_limit` is its
    ``b || A`` special case. Use this one for shear or mixed loops, where the
    prismatic form is not merely imprecise but qualitatively wrong: a pure shear
    loop has ``dV = b.A = 0``, so the prismatic form returns exactly zero while
    the true amplitude is ``2 mu (qhat.b)(qhat.A)/(lam+2mu)``, which is not.

    Equivalently ``q.u~ = i qhat.P.qhat / (lam + 2 mu)`` with the elastic dipole
    tensor ``P_ij = C_ijkl b_k A_l`` -- the standard object, not something new
    here. See Clouet, Varvenne & Jourdan, "Elastic modeling of point-defects and
    their interaction" (arXiv:1802.04062), Eq. 26, and Dederichs, J. Phys. F 3
    (1973) 471 for the statement that the symmetry of the long-range displacement
    field is what diffuse scattering measures.

    Returns ``(Q,)`` complex.
    """
    b = torch.as_tensor(b_um, dtype=q_hat.dtype, device=q_hat.device)
    A = torch.as_tensor(A_um2, dtype=q_hat.dtype, device=q_hat.device)
    dV = float(b @ A)
    qb = q_hat @ b
    qA = q_hat @ A
    return 1j * (lam * dV + 2.0 * mu * qb * qA) / (lam + 2.0 * mu)


def prismatic_loop_small_q_limit(
    dV_um3: float,
    normal: Sequence[float],
    q_hat: torch.Tensor,
    lam: float,
    mu: float,
    *,
    check: bool = True,
) -> torch.Tensor:
    """``q . u~`` as ``q -> 0`` for a PRISMATIC loop (``b || A``), isotropic.

    ``q . u~ -> i dV [ kappa + (1 - kappa) (n.qhat)^2 ]``, ``kappa = lam/(lam+2mu)``.

    **Only valid for b parallel to A.** For anything else use
    :func:`loop_small_q_limit`; this form is wrong by 12-100 % for mixed loops
    and returns exactly zero for a pure shear loop. ``check`` only guards the
    obvious misuse of passing ``dV = 0``.

    Provenance and scope -- read before quoting this
    ------------------------------------------------
    * **Not novel.** This is the isotropic, prismatic specialisation of the
      elastic dipole tensor ``P_ij = C_ijkl b_k A_l``; the general relation is in
      Clouet et al. (arXiv:1802.04062) Eq. 26 and the scattering statement in
      Dederichs (1973).
    * **"Limit" is loose.** No single limit exists at ``q = 0``; there is a
      family of ray limits, one per direction, each stable to ~1e-12.
    * **It is NOT a loop-versus-void discriminator.** At ``q -> 0`` the amplitude
      depends only on ``(dV, n)``. Every loop-specific attribute -- radius, line
      character, b separately from A -- has already been discarded. A uniaxial
      plate precipitate or lenticular void with the same relaxation volume gives
      an *identical* signature. What this separates is a uniaxial dipole from an
      isotropic one.
    * **A randomly oriented loop population is exactly degenerate with a void.**
      For any cubic family (<111>, <100>, <110>) the variant average of ``n n``
      is exactly ``I/3``, so the population dipole is exactly isotropic and the
      angular contrast is exactly 1. Observing the anisotropy needs a single
      crystal AND variant selection (e.g. an applied stress). In a powder there
      is nothing.
    * **The contrast is a matrix property.** ``1 - kappa = (1-2nu)/(1-nu)``: 0.50
      at nu = 1/3, 0.18 at nu = 0.45, and 0 as nu -> 0.5. It says as much about
      the host as about the defect.

    Returns ``(Q,)`` complex.
    """
    n = torch.as_tensor(normal, dtype=q_hat.dtype, device=q_hat.device)
    n = n / torch.linalg.norm(n)
    if check and dV_um3 == 0.0:
        raise ValueError(
            "dV = 0 means b is perpendicular to A -- a shear loop, which this "
            "prismatic form cannot represent (it would return 0 while the true "
            "amplitude is non-zero). Use loop_small_q_limit(b, A, ...).")
    kappa = lam / (lam + 2.0 * mu)
    c2 = (q_hat @ n) ** 2
    return 1j * dV_um3 * (kappa + (1.0 - kappa) * c2)
