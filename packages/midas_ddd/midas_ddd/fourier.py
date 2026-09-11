"""Fourier-space displacement field of a dislocation network.

One kernel, three evaluation points::

    q . u~(q) + Laue term  -> small-angle scattering      (midas_saxs)
    (G+q) . u~(q)          -> near-Bragg diffuse / Huang  (midas_defect)
    inverse FT             -> real-space beta(r)          (midas_dfxm)

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
-- the direction where ``q . u~`` is largest -- the transverse projection gives
*exactly zero* while the true surface integral gives the full value. The
q-parallel part of ``I`` is where the relaxation volume lives. So
this module takes an explicit cut surface: the fan triangulation of each closed
loop from its centroid, documented, reproducible, and shared with the FFT
reference implementation so the two can be compared.

Scope: ``u_tilde`` is closed loops only; the small-angle amplitude is not
------------------------------------------------------------------------
A finite cut surface exists only for a **closed circuit**, so :func:`u_tilde`
(the distortion term, ``q . u~``, what :mod:`midas_defect`'s near-Bragg Huang
kernel consumes) handles closed loops only. It consumes the loop inventory from
:func:`midas_ddd.find_loops` and tells you, in :class:`FourierResult`, how much
line length it ignored.

**The total small-angle amplitude has no such limit.** By Stokes it is a pure
line integral of ``[q x (b - b')].d exp(-i q.m) sinc(q.d/2) / q^2`` (Seeger &
Kroner 1959), so it is defined for open lines too:
:func:`line_small_angle_amplitude` covers finite lines (open, or a junction
network that does not wind) at every ``q``, where an edge or mixed line
scatters into a sheet perpendicular to itself and an isotropic screw scatters
nothing. A line that closes only through a periodic cell's boundary has an
amplitude only on the cell's reciprocal lattice
(:func:`lattice_small_angle_amplitude`); between lattice points
:func:`periodic_small_angle_intensity` gives the intensity at a stated
resolution, exact on the lattice and invariant to which cell stores the
structure (Poisson-summed window average, Paddison arXiv:1809.07088 Eq. 44).
Registered and gated in ``packages/midas_saxs/dev/paper/PREREGISTER_periodic_lines.md``;
confirmed as amended A1-A6, and under adversarial `/verify` as claim
``887b8869a9c4``, not yet established.

The gate
--------
For isotropic elasticity and a prismatic loop (``b || A || n``), the small-q
limit of the DISTORTION term has an exact closed form, verified here to 2e-15
over 1200 random directions and four different moduli::

    q . u~(q -> 0) = i dV [ kappa + (1 - kappa) (n.qhat)^2 ],
        dV = b . A,   kappa = lambda / (lambda + 2 mu).

That is ``q . u~`` alone, and it is **not** the small-angle scattering
amplitude. At small angle the Laue term is the same order and must be added
(Ehrhart, Trinkaus & Larson, Phys. Rev. B 25 (1982) 834, Eq. 8a); the total,
:func:`small_angle_amplitude`, tends to ``i dV (1 - kappa) sin^2(theta)`` and
VANISHES along the normal, exactly where the distortion term alone is largest.
See :func:`prismatic_loop_small_q_limit` (distortion) and
:func:`prismatic_loop_small_q_limit_total` (total). Neither is a
loop-versus-void discriminator; the caveats are on
:func:`prismatic_loop_small_q_limit`.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .elasticity import _voigt_to_tensor

__all__ = [
    "FourierResult",
    "acoustic_tensor",
    "isotropic_stiffness",
    "lattice_small_angle_amplitude",
    "line_small_angle_amplitude",
    "loop_cut_surface",
    "loop_vertices",
    "loop_small_q_limit",
    "periodic_small_angle_intensity",
    "prismatic_loop_small_q_limit",
    "q_dot_u_tilde",
    "segment_components",
    "small_angle_amplitude",
    "prismatic_loop_small_q_limit_total",
    "q_dot_u_tilde_per_loop",
    "surface_form_factor",
    "u_tilde",
    "winding_components",
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
    """``q . u~(q)``, the DISTORTION term only. ``(Q,)`` complex.

    **Not** the small-angle scattering amplitude. At small angle the Laue term
    is the same order and reverses the answer along the loop normal, so use
    :func:`small_angle_amplitude` there. The displacement field behind this,
    :func:`u_tilde`, is what the near-Bragg forward uses as ``(G+q) . u~``,
    where the distortion term carries a factor ``G`` the Laue term does not.
    """
    res = u_tilde(net, q, C6, **kw)
    return torch.einsum("...i,...i->...", res.q.to(res.u_tilde.dtype), res.u_tilde)


# ---------------------------------------------------------------------------
# The analytic gate
# ---------------------------------------------------------------------------

def small_angle_amplitude(net, q, C6, *, loops=None, per_loop=False,
                          method="line", n_rings=None):
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

    ``method="line"`` (default) evaluates the amplitude as a line integral over
    each loop's segments, :func:`line_small_angle_amplitude`: exact per straight
    segment, no triangulation, no ring count. ``method="surface"`` integrates each
    loop's explicit cut surface with ``n_rings`` radial subdivisions (``None``
    picks them from q) and is kept as the reference. By Stokes' theorem the two
    are equal for a closed loop; measured 2026-09-10 to 1.7e-14 over prismatic,
    shear, mixed and non-planar loops, isotropic and cubic, qR up to 10
    (``midas_saxs/dev/paper/RESULTS_line_term.md``, G1), with the residual
    shrinking as ``(qh)^6`` -- the surface quadrature.

    **Periodic cells, several loops, per_loop=False.** Each loop's position is
    fixed only up to a lattice vector, namely which image its nodes are stored in.
    Off the cell's reciprocal lattice the coherent total therefore depends on that
    choice, and a warning says so. Sum ``|A|^2`` per loop (``per_loop=True``), or
    use :func:`periodic_small_angle_intensity` for the coherent case.
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
    if not per_loop and len(loops) > 1 and any(bool(p) for p in net.pbc):
        warnings.warn(
            "small_angle_amplitude(per_loop=False) sums several loops coherently in a "
            "periodic cell. Each loop's position is fixed only up to a lattice vector "
            "(which image its nodes are stored in), so off the cell's reciprocal lattice "
            "that coherent total is not defined by the network alone. Sum |A|^2 per loop "
            "(per_loop=True), or use periodic_small_angle_intensity.",
            stacklevel=2)
    if method == "line":
        return _loops_by_line_integral(net, q, C6, loops, per_loop)
    if method != "surface":
        raise ValueError(f"method must be 'surface' or 'line', got {method!r}")

    C4 = _voigt_to_tensor(C6.to(dtype=net.nodes_um.dtype, device=net.nodes_um.device))
    cdt = torch.complex128 if net.nodes_um.dtype == torch.float64 else torch.complex64
    Kinv = torch.linalg.inv(acoustic_tensor(C4, q))

    out = []
    rings = ([max(1, int(n_rings))] * len(loops) if n_rings is not None
             else [_rings_for_q(net, lp, q) for lp in loops])
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


# ---------------------------------------------------------------------------
# Line-integral form of the total small-angle amplitude: loops AND open lines
# ---------------------------------------------------------------------------

def _skew(q: torch.Tensor) -> torch.Tensor:
    """``[q]_x`` with ``[q]_x w = q x w``. ``(..., 3, 3)``."""
    z = torch.zeros_like(q[..., 0])
    return torch.stack([
        torch.stack([z, -q[..., 2], q[..., 1]], dim=-1),
        torch.stack([q[..., 2], z, -q[..., 0]], dim=-1),
        torch.stack([-q[..., 1], q[..., 0], z], dim=-1),
    ], dim=-2)


def segment_components(net) -> Tuple[torch.Tensor, int]:
    """Connected-component label of every SEGMENT: ``(labels (M,), n_components)``.

    Segments that share a node are one component. A simple loop is its own
    component; a network joined through junctions is one component however many
    arms it has. Labels run 0..C-1 in order of first appearance.
    """
    parent = list(range(net.n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = [(int(i), int(j)) for i, j in net.segments.tolist()]
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    seen: Dict[int, int] = {}
    labels = [seen.setdefault(find(i), len(seen)) for i, _ in pairs]
    return (torch.tensor(labels, dtype=torch.int64, device=net.nodes_um.device),
            len(seen))


def _refuse_long_segments(net, max_fraction: float = 0.25, *, keep=None) -> None:
    """Refuse a segment longer than ``max_fraction`` of the cell along a periodic axis.

    Minimum-image storage cannot represent a segment of half a cell or longer: it
    folds to ``L - d`` and is read as its mirror, or as non-winding at exactly
    ``L/2``, with no trace left in the stored data. After folding it looks like a
    short segment, so the guard sits at a quarter cell. DDD codes keep segments at
    1-5 % of the cell (the ExaDiS networks shipped with the notebooks: 2 %), so a
    physical network never trips this; a hand-built or mis-scaled one does. That
    5x-typical margin is a property of the generator, not of the physics: a
    nanoconfined cell (tens of nm) discretised at the same ABSOLUTE segment length
    a larger cell would use can sit much closer to this limit, so a hand-built
    network at that scale should check its own segment fractions
    (``net.segment_vectors_um().abs() / net.cell_size_um``) rather than assume the
    typical margin (/verify claim 3d56329b9706, artifact lens).

    ``keep`` : ``(M,)`` bool or ``None``. ``None`` (the default) checks every
    segment in ``net`` -- the right choice when the caller has not yet resolved
    which segments a downstream computation will use (:func:`winding_components`,
    and the topology walk :func:`_image_shifts` performs internally). When a
    caller already knows which segments (or, for a shift-propagating walk, which
    whole CONNECTED COMPONENTS) are in scope, passing that mask here avoids
    refusing on a segment that has no way to affect the requested output --
    findings 2/5 of the same /verify: the guard used to fire on the whole network
    even when the caller's ``labels`` excluded the offending segment's entire
    component.
    """
    d = net.segment_vectors_um().detach()
    L = net.cell_size_um.detach()
    sel = torch.ones(net.n_segments, dtype=torch.bool, device=d.device) if keep is None else keep
    if not bool(sel.any()):
        return
    for ax in range(3):
        if bool(net.pbc[ax]) and float(L[ax]) > 0:
            frac = torch.where(sel, d[:, ax].abs() / L[ax], torch.zeros((), dtype=d.dtype))
            if bool((frac > max_fraction).any()):
                s = int(frac.argmax())
                raise ValueError(
                    f"segment {s} spans {float(frac[s]):.3f} of the cell along periodic axis "
                    f"{ax} (limit {max_fraction}). A segment of half a cell or longer cannot "
                    f"be stored under the minimum-image convention: it is folded and read as "
                    f"its mirror image, silently. Refine the discretisation or enlarge the "
                    f"cell.")


def _image_shifts(net, *, refuse_keep=None):
    """Integer image shift per node, component labels, and which components wind.

    A periodic DDD cell stores every node folded into the box, so an object that
    crosses a face is split into pieces one lattice vector apart. Each connected
    component is walked from its lowest-index node along minimum-image segment
    vectors, and every node gets an integer image shift that makes the walk
    continuous (``nodes_um + shift * cell_size_um``). For a component with no
    cycle through the boundary that is exact up to one global lattice vector.

    A component WINDS when some segment disagrees with the walk: a cycle whose
    minimum-image steps add up to a non-zero lattice vector, i.e. a line (or a
    junction network) that closes only through the periodic boundary. The test is
    topological, so it does not depend on node labels, segment order or which
    image a node is stored in.

    ``refuse_keep`` scopes the long-segment guard (see :func:`_refuse_long_segments`);
    ``None`` (the default, used by :func:`winding_components`, which has no caller
    intent to scope to) checks every segment in ``net``. A caller that already
    knows which whole connected COMPONENTS are in scope should pass a segment mask
    covering exactly those components, not just the segments it will ultimately
    sum: the shift walk below propagates across shared nodes within a component,
    so a mis-folded segment anywhere in a component can corrupt the shift of every
    other node in that same component, even ones on segments the caller keeps.

    Returns ``(shift (N, 3) numpy, labels (M,) int64 tensor, n_components,
    winds (C,) numpy bool)``.
    """
    _refuse_long_segments(net, keep=refuse_keep)
    pos = net.nodes_um.detach().cpu().numpy()
    L = net.cell_size_um.detach().cpu().numpy()
    periodic = np.array([bool(net.pbc[a]) and L[a] > 0 for a in range(3)])
    L_safe = np.where(periodic, L, 1.0)
    segs = net.segments.detach().cpu().numpy().astype(np.int64).reshape(-1, 2)

    adj: List[List[int]] = [[] for _ in range(net.n_nodes)]
    for i, j in segs.tolist():
        adj[i].append(j)
        adj[j].append(i)

    shift = np.zeros((net.n_nodes, 3))
    seen = np.zeros(net.n_nodes, dtype=bool)
    for root in range(net.n_nodes):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if seen[v]:
                    continue
                k = np.where(periodic, np.round((pos[v] - pos[u]) / L_safe), 0.0)
                shift[v] = shift[u] - k
                seen[v] = True
                stack.append(v)

    labels, n = segment_components(net)
    winds = np.zeros(n, dtype=bool)
    if len(segs):
        k = np.where(periodic, np.round((pos[segs[:, 1]] - pos[segs[:, 0]]) / L_safe), 0.0)
        broken = np.any(shift[segs[:, 1]] - shift[segs[:, 0]] + k != 0.0, axis=1)
        winds[labels.cpu().numpy()[broken]] = True
    return shift, labels, n, winds


def winding_components(net) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Connected components, and which of them close only through the periodic boundary.

    Returns ``(labels (M,), n_components, winds (C,) bool)``; ``labels`` are those of
    :func:`segment_components`. A component winds when it contains a cycle whose
    minimum-image segment vectors add up to a non-zero lattice vector: a periodic
    line crossing the whole cell, or a junction network that does. A loop folded
    across a face does not wind. The test is topological, so node labels, segment
    order and stored images do not change it.

    A winding component has a scattering amplitude only on the cell's reciprocal
    lattice: see :func:`lattice_small_angle_amplitude` and
    :func:`periodic_small_angle_intensity`.
    """
    _, labels, n, winds = _image_shifts(net)
    return labels, n, torch.as_tensor(winds, dtype=torch.bool, device=labels.device)


def line_small_angle_amplitude(
    net,
    q,
    C6,
    *,
    per_component: bool = False,
    labels: Optional[torch.Tensor] = None,
    n_groups: Optional[int] = None,
    chunk_elements: int = 4_000_000,
) -> torch.Tensor:
    """TOTAL small-angle amplitude as a line integral over every segment. µm^3.

    By Stokes, ``closed-int exp(-i q.r) dl = i q x A~(q)``, so the
    Ehrhart-Trinkaus-Larson Eq. (8b) amplitude that :func:`small_angle_amplitude`
    integrates over a cut surface is a pure line integral::

        A(q) = sum_s [q x (b_s - b'_s(q))] . J_s(q) / q^2
        J_s  = d_s exp(-i q.m_s) sinc(q.d_s / 2)     d_s segment vector, m_s midpoint

    This is the form of Seeger & Kroner, Z. Naturforsch. A 14, 74 (1959). For a
    closed loop it equals the surface form and needs no cut surface. It is also
    defined for OPEN lines, which the surface form is not. For a straight edge line
    in isotropic elasticity it gives the Fourier transform of the Volterra
    dilatation on the line's reciprocal-space sheet (Thomson, Levine & Long, Acta
    Cryst. A 55, 433 (1999), Eq. 6), and an isotropic screw gives exactly zero.

    Status: the claim that this form is correct for open AND periodic lines was
    REFUTED by /verify on its periodic clause (claim 8305670629a7, 2026-09-10,
    ``packages/midas_saxs/dev/paper/RESULTS_line_term.md``). The lenses reproduced
    independently the closed-loop identity and the straight-line sheet values
    (Thomson Eq. 6; an anisotropic Stroh field). Those parts have not been verified
    as a claim of their own. Winding lines are refused here (below).

    Parameters
    ----------
    q : (Q, 3)
        Scattering vectors, **inverse micrometers**. Must not contain exactly zero.
    per_component
        ``False``: the coherent sum over all included segments, ``(Q,)``.
        ``True``: one coherent amplitude per group, ``(G, Q)``.
    labels, n_groups
        Group label per segment, ``(M,)``; segments labelled ``< 0`` are left out.
        Default: connected components (:func:`segment_components`). Segments
        joined through nodes interfere; separate objects are for the caller to
        sum incoherently.
    chunk_elements
        Bound on ``Q x segments`` per evaluation block, for memory.

    Periodic cells
    --------------
    An object folded across the cell faces is made continuous by walking its
    minimum-image segment vectors. For anything that does not wind (a loop, an
    open line, a junction cluster) that is exact up to one global lattice vector,
    so its ``|A|^2`` does not depend on node labels or stored images.

    A component that closes only through the periodic boundary
    (:func:`winding_components`) is REFUSED. One period of it has no amplitude off
    the cell's reciprocal lattice that the network alone defines: where the chain
    is cut changes ``|A|^2`` pixel by pixel (/verify claim 8305670629a7). Use
    :func:`lattice_small_angle_amplitude` on the lattice, or
    :func:`periodic_small_angle_intensity` at pixel q.

    A coherent sum over SEVERAL objects in a periodic cell has the same problem in
    a milder form, because each object's global lattice vector is arbitrary. In a
    periodic cell, sum ``|A|^2`` per object (``per_component=True``) or use
    :func:`periodic_small_angle_intensity`.

    Scope
    -----
    Linear elasticity, infinite medium, no core: not valid beyond q of a few
    inverse Burgers vectors (see :func:`midas_ddd.resolution_report`). A line
    that ENDS inside the medium is not an elastic solution -- Burgers vector is
    not conserved there -- and its amplitude includes those ends' unphysical
    sources. Networks that conserve Burgers vector at every free node, and lines
    crossing the whole cell, are the intended input.
    """
    dt, dev = net.nodes_um.dtype, net.nodes_um.device
    q = torch.as_tensor(q, dtype=dt, device=dev)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if bool(((q * q).sum(dim=-1) == 0).any()):
        raise ValueError("q contains exactly zero; the small-angle amplitude is a "
                         "limit there, not a value.")
    cdt = torch.complex128 if dt == torch.float64 else torch.complex64

    # Scope the long-segment guard to the connected components `labels` actually
    # touches, not the whole network: an unrelated component's fold ambiguity
    # cannot corrupt this call's output, since the shift walk below never crosses
    # between components (/verify claim 3d56329b9706, artifact lens). Resolved
    # before the walk so _image_shifts can use it. `labels=None` touches every
    # component anyway, so no scoping is possible there and none is attempted.
    refuse_keep = None
    resolved_labels = resolved_n = None
    if labels is not None:
        resolved_labels, resolved_n = _resolve_labels(net, labels, n_groups)
        pre_components, pre_n = segment_components(net)
        if pre_n > 0:
            touched = torch.zeros(pre_n, dtype=torch.bool, device=dev)
            pre_keep = resolved_labels >= 0
            if bool(pre_keep.any()):
                touched[pre_components[pre_keep].unique()] = True
            refuse_keep = touched[pre_components]

    shift, components, n_components, winds = _image_shifts(net, refuse_keep=refuse_keep)
    if labels is None:
        labels, n_groups = components, n_components
    else:
        labels, n_groups = resolved_labels, resolved_n
    keep = labels >= 0
    Q = q.shape[0]
    if n_groups == 0 or not bool(keep.any()):
        z = torch.zeros(max(int(n_groups or 0), 0), Q, dtype=cdt, device=dev)
        return z if per_component else torch.zeros(Q, dtype=cdt, device=dev)
    if winds.any():
        winding = keep & torch.as_tensor(winds, device=dev)[components]
        if bool(winding.any()):
            raise ValueError(
                f"{int(winding.sum())} segment(s) belong to a line that closes only "
                f"through the periodic boundary. One period of such a line has no "
                f"small-angle amplitude off the cell's reciprocal lattice that the "
                f"network alone defines: the answer depends on where its chain is "
                f"cut, i.e. on node numbering (/verify claim 8305670629a7). Use "
                f"lattice_small_angle_amplitude on the lattice, or "
                f"periodic_small_angle_intensity at arbitrary q.")

    C4 = _voigt_to_tensor(C6.to(dtype=dt, device=dev))
    pos = net.nodes_um + torch.as_tensor(shift, dtype=dt, device=dev) * net.cell_size_um
    d_all = net.segment_vectors_um()
    mid_all = pos[net.segments[:, 0]] + 0.5 * d_all
    out = _segment_sum(q, d_all[keep], mid_all[keep], net.burgers_um()[keep],
                       labels[keep], n_groups, C4, chunk_elements)
    return out if per_component else out.sum(dim=0)


def _resolve_labels(net, labels, n_groups) -> Tuple[torch.Tensor, int]:
    """Group label per segment (``< 0`` = left out) and the number of groups."""
    if labels is None:
        labels, n_groups = segment_components(net)
        return labels, int(n_groups)
    labels = torch.as_tensor(labels, dtype=torch.int64, device=net.nodes_um.device)
    if n_groups is None:
        n_groups = int(labels.max()) + 1 if labels.numel() else 0
    return labels, int(n_groups)


def _segment_sum(q, d, mid, b, lab, n_groups, C4, chunk_elements) -> torch.Tensor:
    """``sum_s [q x (b_s - b'_s)] . d_s sinc(q.d_s/2) exp(-i q.m_s) / q^2`` per group.

    ``(n_groups, Q)`` complex, µm^3. The one kernel behind the line, lattice and
    periodic functions; they differ only in where the midpoints ``mid`` sit and
    which ``q`` they evaluate.
    """
    dt, dev = q.dtype, q.device
    cdt = torch.complex128 if dt == torch.float64 else torch.complex64
    Q, S = int(q.shape[0]), int(d.shape[0])
    if Q == 0 or S == 0:
        return torch.zeros(n_groups, Q, dtype=cdt, device=dev)
    q2 = (q * q).sum(dim=-1)
    eye = torch.eye(3, dtype=dt, device=dev)
    step = max(1, int(chunk_elements) // S)
    blocks = []
    for lo in range(0, Q, step):
        hi = min(Q, lo + step)
        qq = q[lo:hi]
        Kinv = torch.linalg.inv(acoustic_tensor(C4, qq))
        # b'_m = q_j Kinv_jk C_klmn q_l b_n = M_mn b_n, and d.P.b = [q x (b - b')].d.
        M = torch.einsum("qj,qjk,klmn,ql->qmn", qq, Kinv, C4, qq)
        P = _skew(qq) @ (eye - M)
        strength = torch.einsum("si,qij,sj->qs", d, P, b)              # (Qc, S)
        shape = torch.sinc(0.5 * (qq @ d.T) / math.pi)                 # sin(x)/x
        arg = qq @ mid.T
        re = strength * shape / q2[lo:hi, None]
        a = torch.complex(re * torch.cos(arg), -re * torch.sin(arg))   # (Qc, S)
        blocks.append(torch.zeros(n_groups, hi - lo, dtype=cdt, device=dev)
                      .index_add(0, lab, a.transpose(0, 1)))
    return torch.cat(blocks, dim=1)


# ---------------------------------------------------------------------------
# Periodic cells: the reciprocal lattice, and intensity between its points
# ---------------------------------------------------------------------------

#: Intensity FWHM of a Gaussian in units of its standard deviation.
_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _reciprocal_lattice_um(net) -> torch.Tensor:
    """``2 pi / L_i`` per axis, inverse micrometers, for a cell periodic along all three axes."""
    L = net.cell_size_um.detach()
    if not all(bool(p) for p in net.pbc) or bool((L <= 0).any()):
        raise NotImplementedError(
            f"the cell's reciprocal lattice is used only for a cell periodic along all "
            f"three axes with positive size; got pbc={tuple(bool(p) for p in net.pbc)}, "
            f"cell size {L.tolist()} um.")
    return (2.0 * math.pi) / L


def lattice_small_angle_amplitude(
    net,
    G,
    C6,
    *,
    per_component: bool = False,
    labels: Optional[torch.Tensor] = None,
    n_groups: Optional[int] = None,
    chunk_elements: int = 4_000_000,
    lattice_tol: float = 1e-9,
) -> torch.Tensor:
    """TOTAL small-angle amplitude of a periodic cell ON ITS RECIPROCAL LATTICE. µm^3.

    A cell periodic along all three axes stores one period of an infinite
    structure, which scatters only at ``G = 2 pi (h/Lx, k/Ly, l/Lz)``. At each such
    point the amplitude is the line integral of :func:`line_small_angle_amplitude`
    evaluated at ``G``, over every included segment exactly as stored::

        A(G) = sum_s [G x (b_s - b'_s(G))] . d_s sinc(G.d_s/2) exp(-i G.m_s) / G^2

    Because ``exp(-i G.nL) = 1``, this does not depend on node labels, segment
    orientation, which image a node is stored in, or where a periodic line's chain
    is cut. That is exactly what fails between lattice points (Paddison,
    arXiv:1809.07088, p. 3). For a network that conserves Burgers vector, winding
    lines included, it is the Fourier coefficient of the periodic structure at
    ``G``.

    Status: preregistered 2026-09-10 and under test,
    ``packages/midas_saxs/dev/paper/PREREGISTER_periodic_lines.md``. Not yet verified.

    Parameters
    ----------
    G : (K, 3)
        Inverse micrometers. Each must be a non-zero integer combination of
        ``2 pi / L_i`` to ``lattice_tol`` (relative). Anything else is refused, not
        rounded; the exact lattice vector is what gets evaluated.
    per_component, labels, n_groups
        As :func:`line_small_angle_amplitude`. Groups default to connected
        components and are summed coherently unless ``per_component``.
    """
    dt, dev = net.nodes_um.dtype, net.nodes_um.device
    cdt = torch.complex128 if dt == torch.float64 else torch.complex64
    astar = _reciprocal_lattice_um(net).to(dtype=dt, device=dev)
    G = torch.as_tensor(G, dtype=dt, device=dev)
    if G.ndim == 1:
        G = G.unsqueeze(0)
    frac = G.detach() / astar
    hkl = torch.round(frac)
    off = (frac - hkl).abs()
    if bool((off > lattice_tol * hkl.abs().clamp(min=1.0)).any()):
        raise ValueError(
            f"G is not on the cell's reciprocal lattice (largest fractional index offset "
            f"{float(off.max()):.3g}). Between lattice points a periodic cell's amplitude "
            f"is not defined by the network alone; use periodic_small_angle_intensity "
            f"for arbitrary q.")
    if bool((hkl.abs().sum(dim=-1) == 0).any()):
        raise ValueError(
            "G = 0 is excluded: the forward amplitude of a periodic cell is a limit that "
            "depends on the boundary condition, and in SAXS it is the direct beam.")
    labels, n_groups = _resolve_labels(net, labels, n_groups)
    keep = labels >= 0
    # Scoped to `keep`: every segment's contribution here is self-contained (no
    # shift-propagating walk, unlike line_small_angle_amplitude), so an excluded
    # segment's fold ambiguity cannot affect an included one (/verify claim
    # 3d56329b9706, artifact lens).
    _refuse_long_segments(net, keep=keep)
    if n_groups == 0 or not bool(keep.any()):
        z = torch.zeros(max(n_groups, 0), G.shape[0], dtype=cdt, device=dev)
        return z if per_component else torch.zeros(G.shape[0], dtype=cdt, device=dev)
    C4 = _voigt_to_tensor(C6.to(dtype=dt, device=dev))
    d_all = net.segment_vectors_um()
    mid_all = net.nodes_um[net.segments[:, 0]] + 0.5 * d_all     # as stored: exp(-iG.nL) = 1
    out = _segment_sum(hkl * astar, d_all[keep], mid_all[keep], net.burgers_um()[keep],
                       labels[keep], n_groups, C4, chunk_elements)
    return out if per_component else out.sum(dim=0)


def _poisson_ripple_bound(s: float, a) -> float:
    """3-D bound on ``|sum_G W(q-G) / <sum W> - 1|`` for a Gaussian of width ``s`` on a
    lattice of spacings ``a``: ``prod_i (1 + r_i) - 1`` with ``r_i = 2 sum_m exp(-2 pi^2 m^2
    s^2 / a_i^2)`` (Poisson summation). 3.9e-6 at FWHM = 2 a*, 18 % at FWHM = a*."""
    rho = 1.0
    for ai in (float(x) for x in a):
        rho *= 1.0 + 2.0 * sum(math.exp(-2.0 * math.pi ** 2 * m * m * s * s / (ai * ai))
                               for m in range(1, 5))
    return rho - 1.0


def _lattice_average(q, astar, fwhm_inv_um, values_at, *, cutoff_sigmas: float = 12.0,
                     chunk_elements: int = 4_000_000, normalised: bool = False):
    """Gaussian window average of lattice values at arbitrary q, per cell.

    Poisson summation of the replicated cell seen through a window ``w(r)`` with
    ``|w^(k)|^2 = W(k)``, averaged over where the window sits::

        I(q) = sum_{G != 0} W(q - G) V(G) / <sum_G W>
        W(k) = exp(-|k|^2 / 2 s^2),   s = fwhm / (2 sqrt(2 ln 2))
        <sum_G W> = (2 pi s^2)^{3/2} / (a*_x a*_y a*_z)      (Parseval)

    The constant is the Gaussian's integral per reciprocal cell, so the same
    structure stored in an n-fold supercell gives exactly n times the per-cell
    value, and a constant field ``c`` comes back as ``c (1 +/- rho)`` with ``rho``
    the Poisson ripple of the lattice sum (``info["ripple_bound"]``; 3.9e-6 at the
    default FWHM = 2 a*). ``normalised=True`` divides by ``sum_{G != 0} W(q - G)``
    instead (Paddison, arXiv:1809.07088, Eq. 44): no ripple, but then the value
    depends on which cell stores the structure, by that same ripple (/verify claim
    887b8869a9c4). Kept only as the planted control of the registered gates.

    ``values_at(hkl)`` maps ``(K, 3)`` int64 indices to ``(n, K)`` real values. It is
    called once, on every lattice point that any q needs. A point enters when
    ``|q-G|^2 - d_min^2 <= (cutoff_sigmas s)^2``, ``d_min`` being the distance to the
    nearest non-zero lattice point. Every neglected weight is therefore below
    ``exp(-cutoff_sigmas^2 / 2)`` of the largest, and no neighbourhood is empty.
    Weights are formed relative to that nearest point, so a narrow resolution does
    not underflow. Returns ``(I (n, Q), info)``.
    """
    dt, dev = q.dtype, q.device
    a = astar.to(dtype=dt, device=dev)
    a_max = float(a.max())
    fwhm = 2.0 * a_max if fwhm_inv_um is None else float(fwhm_inv_um)
    if not (0.25 * a_max * (1 - 1e-12) <= fwhm <= 10.0 * a_max * (1 + 1e-12)):
        raise ValueError(
            f"resolution FWHM {fwhm:.6g} 1/um is outside [0.25, 10] x the coarsest "
            f"reciprocal-lattice spacing {a_max:.6g} 1/um. Narrower cannot interpolate "
            f"between lattice points; wider averages the q dependence away.")
    s = fwhm / _FWHM_PER_SIGMA
    d_far = 0.5 * float(torch.linalg.vector_norm(a))
    # Every included point lies within `reach` of q (the nearest non-zero point is within
    # d_far + a_max, the rest within cutoff_sigmas * s of that distance), and q lies within
    # d_far of the lattice point it rounds to, so offsets beyond reach + d_far never enter.
    reach = cutoff_sigmas * s + d_far + a_max
    ball = reach + d_far
    nbox = torch.tensor([int(math.ceil(ball / float(a[i]))) for i in range(3)],
                        dtype=torch.int64, device=dev)
    offsets = torch.cartesian_prod(*[torch.arange(-int(n), int(n) + 1, device=dev) for n in nbox])
    offsets = offsets[torch.linalg.vector_norm(offsets.to(dt) * a, dim=1) <= ball]
    B = int(offsets.shape[0])
    qd = q.detach()
    Q = int(qd.shape[0])
    base = torch.round(qd / a).to(torch.int64)
    zero3 = torch.zeros(3, dtype=torch.int64, device=dev)
    lo = (base.min(dim=0).values if Q else zero3) - nbox
    span = (base.max(dim=0).values if Q else zero3) + nbox - lo + 1
    cut2 = (cutoff_sigmas * s) ** 2

    def key(hkl):
        r = hkl - lo
        return (r[..., 0] * span[1] + r[..., 1]) * span[2] + r[..., 2]

    def around(i0, i1):
        hkl = base[i0:i1, None, :] + offsets[None]                      # (Qc, B, 3)
        diff = qd[i0:i1, None, :] - hkl.to(dt) * a
        d2 = (diff * diff).sum(dim=-1)
        d2 = torch.where((hkl != 0).any(dim=-1), d2, torch.full_like(d2, math.inf))
        d2min = d2.min(dim=1, keepdim=True).values
        rel = d2 - d2min
        return hkl, rel, rel <= cut2, d2min[:, 0]

    step = max(1, int(chunk_elements) // B)
    needed = []
    for i0 in range(0, Q, step):
        hkl, _, inside, _ = around(i0, min(Q, i0 + step))
        needed.append(torch.unique(key(hkl)[inside]))
    ukeys = (torch.unique(torch.cat(needed)) if needed
             else torch.zeros(0, dtype=torch.int64, device=dev))
    rest = ukeys // span[2]
    hkl_u = torch.stack([rest // span[1], rest % span[1], ukeys % span[2]], dim=1) + lo
    values = values_at(hkl_u)                                           # (n, K)
    n_out, K = int(values.shape[0]), int(values.shape[1])

    # Weights are formed relative to the nearest point (rel = d^2 - d_min^2), so the
    # Parseval constant carries the same factor: exp(-d_min^2 / 2 s^2) per pixel.
    norm = (2.0 * math.pi * s * s) ** 1.5 / float(torch.prod(a))
    rows = max(1, int(chunk_elements) // (B * step))       # value rows gathered at once
    out, n_eff = [], []
    for i0 in range(0, Q, step):
        hkl, rel, inside, d2min = around(i0, min(Q, i0 + step))
        w = torch.where(inside, torch.exp(-rel / (2.0 * s * s)), torch.zeros_like(rel))
        idx = torch.searchsorted(ukeys, key(hkl)).clamp(max=max(K - 1, 0))
        wsum = w.sum(dim=1)
        denom = wsum if normalised else norm * torch.exp(d2min / (2.0 * s * s))
        summed = [(values[r0:r0 + rows][:, idx] * w).sum(dim=-1) for r0 in range(0, n_out, rows)]
        out.append((torch.cat(summed, dim=0) if summed
                    else torch.zeros(0, w.shape[0], dtype=dt, device=dev)) / denom)
        n_eff.append(wsum ** 2 / (w * w).sum(dim=1))
    I = torch.cat(out, dim=1) if out else torch.zeros(n_out, 0, dtype=dt, device=dev)
    info = dict(
        fwhm_inv_um=fwhm,
        sigma_inv_um=s,
        reciprocal_lattice_inv_um=a.tolist(),
        q_floor_inv_um=max(3.0 * a_max, math.sqrt(d_far ** 2 + 2.0 * math.log(1e3) * s * s)),
        n_lattice_points=K,
        cutoff_sigmas=float(cutoff_sigmas),
        n_eff=torch.cat(n_eff) if n_eff else torch.zeros(0, dtype=dt, device=dev),
        ripple_bound=0.0 if normalised else _poisson_ripple_bound(s, a),
        parseval_norm=norm,
        normalised=bool(normalised),
    )
    return I, info


def periodic_small_angle_intensity(
    net,
    q,
    C6,
    *,
    fwhm_inv_um: Optional[float] = None,
    per_component: bool = False,
    labels: Optional[torch.Tensor] = None,
    n_groups: Optional[int] = None,
    cutoff_sigmas: float = 12.0,
    chunk_elements: int = 4_000_000,
):
    """Small-angle INTENSITY of a periodic cell at arbitrary q, at a stated resolution.

    Returns ``(I, info)``. ``I`` is ``(Q,)``, or ``(groups, Q)`` with
    ``per_component``, in the units of ``|A|^2`` (µm^6) per cell, so it adds
    directly to a per-loop incoherent sum.

    A cell periodic along all three axes has a scattering amplitude only on its
    reciprocal lattice (:func:`lattice_small_angle_amplitude`). Between lattice
    points this returns the replicated cell seen through a smooth window, averaged
    over where the window sits, per cell (Poisson summation)::

        I(q) = sum_{G != 0} W(q - G) |A(G)|^2 / <sum_G W>
        W(k) = exp(-|k|^2 / 2 s^2),   FWHM = 2 sqrt(2 ln 2) s
        <sum_G W> = (2 pi s^2)^{3/2} V_cell / (2 pi)^3

    - It does not depend on node labels, segment orientation, stored images or a
      rigid translation of the whole network, because ``|A(G)|^2`` does not; nor on
      which cell stores the structure (an n-fold supercell gives n times the
      per-cell value).
    - ``A(G)`` is coherent over every included segment, so a dipole keeps its
      screening.
    - A constant ``|A(G)|^2 = c`` returns ``c (1 +/- rho)``, ``rho`` the Poisson
      ripple of the lattice sum (``info["ripple_bound"]``): 3.9e-6 at the default
      resolution, 18 % at FWHM = a*, and at narrow resolution the estimator shows
      the cell's own lattice peaks, which is what a narrow window sees.

    What it is not: a value the network defines between lattice points. Each
    ``|A(G)|^2`` is one realisation, so for a random network the estimate scatters by
    about ``n_eff^-1/2`` (``info["n_eff"]``, per q). Below
    ``info["q_floor_inv_um"]`` the excluded ``G = 0`` term and the cell's own
    periodicity dominate; values there are returned but are not representative.

    Status: preregistered 2026-09-10, amended A5 on 2026-09-11 after /verify claim
    887b8869a9c4 found the earlier normalised form cell-dependent; under test,
    ``packages/midas_saxs/dev/paper/PREREGISTER_periodic_lines.md``. Not yet verified.

    Parameters
    ----------
    q : (Q, 3)
        Inverse micrometers. ``q = 0`` is allowed: only lattice points are evaluated.
    fwhm_inv_um
        Intensity FWHM of the Gaussian weight. Default: twice the coarsest
        reciprocal-lattice spacing, ``4 pi / L_min``. Allowed: 0.25 to 10 times
        ``2 pi / L_min``.
    per_component, labels, n_groups
        Which segments, grouped how. Default: every segment, one coherent sum.
        ``per_component=True`` averages each group's own ``|A_g(G)|^2``. That is the
        incoherent-across-groups diagnostic, and it loses screening between groups.
    cutoff_sigmas
        Lattice points whose weight is below ``exp(-cutoff_sigmas^2 / 2)`` of the
        largest are left out.

    ``info`` holds ``fwhm_inv_um``, ``sigma_inv_um``, ``reciprocal_lattice_inv_um``,
    ``q_floor_inv_um``, ``n_lattice_points``, ``cutoff_sigmas``, ``ripple_bound``,
    ``n_eff`` ``(Q,)``, ``roundoff_floor`` ``(Q,)`` and ``burgers``.

    - ``roundoff_floor``: the float64 residue of the lattice sums, averaged like the
      intensity, ``(eps sum_s |b_s||d_s| / |G|)^2`` per lattice point. A value below
      about ``1e6 x`` its floor is roundoff, not signal: a pure screw network in
      isotropic or cubic elasticity comes out at about ``1 x`` the floor.
    - ``burgers``: ``n_violating`` free nodes of the included segments where the
      Burgers vector is not conserved, and the largest residual in units of ``b``.
      Such a network is not an elastic solution; a warning is issued.

    A segment longer than a quarter cell along a periodic axis is refused, because
    minimum-image storage would have folded anything longer than half a cell into
    its mirror image without a trace.
    """
    dt, dev = net.nodes_um.dtype, net.nodes_um.device
    astar = _reciprocal_lattice_um(net).to(dtype=dt, device=dev)
    q = torch.as_tensor(q, dtype=dt, device=dev)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    labels, n_groups = _resolve_labels(net, labels, n_groups)
    keep = labels >= 0
    # Scoped to `keep`, same reasoning as lattice_small_angle_amplitude: no
    # shift-propagating walk here, so an excluded segment cannot corrupt an
    # included one (/verify claim 3d56329b9706, artifact lens).
    _refuse_long_segments(net, keep=keep)
    C4 = _voigt_to_tensor(C6.to(dtype=dt, device=dev))
    d_all = net.segment_vectors_um()
    mid_all = net.nodes_um[net.segments[:, 0]] + 0.5 * d_all
    d, mid, b, lab = d_all[keep], mid_all[keep], net.burgers_um()[keep], labels[keep]
    n_out = n_groups if per_component else 1
    # Implementation-independent float64 residue of the lattice sums (see docstring).
    floor_scale = (float(torch.finfo(dt).eps)
                   * float((torch.linalg.vector_norm(b, dim=1)
                            * torch.linalg.vector_norm(d, dim=1)).sum().detach())) ** 2

    def lattice_intensity(hkl):
        G = hkl.to(dt) * astar
        floor = (floor_scale / (G * G).sum(dim=1)).unsqueeze(0)
        if n_groups == 0 or not bool(keep.any()):
            return torch.cat([torch.zeros(n_out, hkl.shape[0], dtype=dt, device=dev), floor])
        A = _segment_sum(G, d, mid, b, lab, n_groups, C4, chunk_elements)
        if not per_component:
            A = A.sum(dim=0, keepdim=True)
        return torch.cat([A.real ** 2 + A.imag ** 2, floor])

    I, info = _lattice_average(q, astar, fwhm_inv_um, lattice_intensity,
                               cutoff_sigmas=cutoff_sigmas, chunk_elements=chunk_elements)
    info["roundoff_floor"] = I[-1]
    I = I[:-1]
    info["burgers"] = _burgers_at_included_nodes(net, keep)
    if info["burgers"]["n_violating"]:
        warnings.warn(
            f"Burgers vector is not conserved at {info['burgers']['n_violating']} free node(s) "
            f"of the included segments (largest residual {info['burgers']['max_residual_b']:.3g} "
            f"b). A periodic line whose Burgers vector changes along it is not an elastic "
            f"solution, and the lattice intensity of such a network is representation-"
            f"invariant but unphysical.", stacklevel=2)
    return (I if per_component else I[0]), info


def _burgers_at_included_nodes(net, keep, *, tol: float = 1e-6) -> Dict[str, float]:
    """Burgers-sum residual, over ALL arms, at the free nodes the ``keep`` segments touch."""
    resid = torch.zeros((net.n_nodes, 3), dtype=net.burgers_b.dtype, device=net.burgers_b.device)
    resid.index_add_(0, net.segments[:, 0], net.burgers_b)
    resid.index_add_(0, net.segments[:, 1], -net.burgers_b)
    mag = torch.linalg.vector_norm(resid, dim=-1).detach()
    touched = torch.unique(net.segments[keep].reshape(-1))
    if net.constraints is not None:
        touched = touched[net.constraints.to(touched.device)[touched] == 0]
    bad = mag[touched] > tol
    return dict(n_checked=int(touched.numel()), n_violating=int(bad.sum()),
                max_residual_b=float(mag[touched].max()) if touched.numel() else 0.0)


def _loops_by_line_integral(net, q, C6, loops, per_loop):
    """:func:`small_angle_amplitude` with ``method="line"``: one group per loop."""
    cdt = torch.complex128 if net.nodes_um.dtype == torch.float64 else torch.complex64
    if not loops:
        z = torch.zeros(q.shape[0], dtype=cdt, device=q.device)
        return z.unsqueeze(0) if per_loop else z
    labels = torch.full((net.n_segments,), -1, dtype=torch.int64, device=net.nodes_um.device)
    for k, lp in enumerate(loops):
        labels[torch.as_tensor(lp.segment_indices, dtype=torch.int64,
                               device=labels.device)] = k
    per = line_small_angle_amplitude(net, q, C6, per_component=True,
                                     labels=labels, n_groups=len(loops))
    return per if per_loop else per.sum(dim=0)
