"""Per-dislocation forward: anisotropic-elasticity displacement field -> F(r).

Phase 4 of ``implementation_plan.md`` — the novel defect-typing core.

A straight dislocation in an anisotropically-elastic crystal has a displacement
field given by the Stroh sextic solution. In the slip frame (``e1`` in-plane, ``e2``
= slip-plane normal, ``e3`` = line), the displacement gradient (distortion) is

    beta_ij(x1, x2) = (1/pi) Im[ sum_a A_ia D_a k_j(a) / (x1 + p_a x2) ]

with ``k = (1, p_a, 0)``, ``p_a`` the three sextic roots (Im>0), ``A`` the Stroh
amplitude matrix, and ``D_a = sum_l b_l B_la`` the Burgers amplitude. This is the
*same* angular quantity that ``midas_defect.contrast_factor.single_contrast_factor``
integrates for line-profile analysis — here we evaluate it at field points and
scale by ``1/r`` to get the real-space distortion. We **reuse** the Stroh solver
and geometry helpers from ``midas_defect`` rather than re-deriving elasticity.

The deformation gradient is ``F(r) = I + beta(r)`` (crystal frame); feeding it to
:class:`midas_dfxm.field.DeformationField` gives the DFXM contrast of an individual
dislocation, and the ``g.b`` structure of ``beta`` drives defect typing
(:mod:`midas_dfxm.typing`).

Distortions superpose in linear elasticity, so a list of dislocations builds walls,
dipoles, and low-angle boundaries by summation.

Units: positions and Burgers magnitude in micrometers (Burgers passed in Angstrom
and converted); stiffness in any consistent unit (only ratios enter the roots).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

# Reuse the anisotropic-elasticity (Stroh) machinery from midas_defect. These are
# the single source of truth for the sextic solution and crystal->slip geometry;
# do NOT re-port them. Centralised here so an upstream signature change is caught
# in one place.
from midas_defect.contrast_factor import (  # noqa: F401
    _crystal_A_matrix,
    _rotate_tensor,
    _slip_frame,
    _stroh_eig,
    _to_cartesian,
    _voigt_to_tensor,
    cubic_stiffness,
    fcc_slip_systems,
)

from .field import DeformationField

_INV_2PI = 1.0 / (2.0 * math.pi)
# Stroh prefactor. We sum one root per conjugate pair (3 of the 6 eigenvalues), which
# recovers the physical field only as 2*Im{...} -- equivalently a 1/pi prefactor, not
# 1/(2 pi). Fixed 2026-07: the previous 1/(2 pi) made every distortion (and hence every
# absolute strain magnitude) a factor 2 too small. The convention-free check is the
# Burgers circuit -- the closed-loop integral of beta.dl must equal b -- which now holds
# to 1e-6 for edge and screw at any radius, and is gated by a test.
_STROH_PREFACTOR = 1.0 / math.pi
_ANGSTROM_PER_UM = 1e4  # 1 um = 1e4 Angstrom


@dataclass
class StrohDislocation:
    """Solved Stroh state for one straight dislocation (crystal frame).

    Build with :func:`stroh_dislocation`. Holds the slip-frame rotation ``M``
    (crystal->slip), sextic roots ``p``, amplitude matrix ``A``, Burgers amplitude
    ``D`` (carrying the physical Burgers magnitude, micrometers), the Cartesian line
    direction, and a Lorentzian core radius (micrometers).
    """

    M: torch.Tensor          # (3, 3) crystal -> slip
    p: torch.Tensor          # (3,) complex sextic roots
    A: torch.Tensor          # (3, 3) complex amplitude matrix
    D: torch.Tensor          # (3,) complex Burgers amplitude (um)
    line_cart: torch.Tensor  # (3,) unit line direction, crystal frame
    core_position: torch.Tensor  # (3,) crystal-frame core point (um)
    core_radius_um: float = 0.05
    core_model: str = "lorentzian"   # "lorentzian" (default) or "compact"

    def displacement_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """Distortion tensor ``beta(r) = grad u`` at crystal-frame ``positions``.

        ``positions`` is ``(N, 3)`` (micrometers). Returns ``(N, 3, 3)`` in the
        crystal frame. Plane-strain: the field is invariant along the line and
        ``beta[:, :, e3-column]`` vanishes in the slip frame before rotation.

        ``core_model`` selects the core regularization:

        ``"lorentzian"`` (default, back-compatible)
            cutoff ``r^2/(r^2 + rc^2)``: smooth everywhere but with a long tail, so it
            still suppresses the field by ~3% at ``r = 6 rc``. With this model
            ``grad(u)`` and :meth:`displacement` differ by that tail near the core.
        ``"compact"``
            cutoff ``min(1, r^2/rc^2)``: exactly 1 for ``r >= rc``, so beta is the
            *exact* singular solution outside the core and therefore agrees with
            ``grad(displacement())`` to machine precision there. Use this whenever u and
            beta are consumed together -- notably coherent imaging, where the exit-wave
            phase is ``g.u`` while the contrast comes from beta.

        Inside ``r < rc`` neither model is physical (linear elasticity has failed); mask
        that region rather than interpreting it.
        """
        rel = positions - self.core_position
        # Project to slip-frame in-plane coordinates (x1 along e1, x2 along e2).
        rel_slip = rel @ self.M.transpose(-1, -2)  # (N, 3): components on e1,e2,e3
        x1 = rel_slip[:, 0]
        x2 = rel_slip[:, 1]
        p = self.p
        denom = x1[:, None] + p[None, :] * x2[:, None]  # (N, 3) complex
        # Core cutoff, folded into a numerically safe reciprocal (conjugate form) so the
        # r->0 singularity gives 0, not inf*0=nan.
        r2 = x1 * x1 + x2 * x2
        rc2 = self.core_radius_um ** 2
        if self.core_model == "compact":
            cutoff = torch.clamp(r2 / rc2, max=1.0)[:, None]
        elif self.core_model == "lorentzian":
            cutoff = (r2 / (r2 + rc2))[:, None]
        else:
            raise ValueError(f"core_model must be 'lorentzian' or 'compact', got {self.core_model!r}")
        denom2 = denom.real ** 2 + denom.imag ** 2 + 1e-300
        inv = denom.conj() * cutoff / denom2
        AD = self.A * self.D[None, :]  # (3, 3): [i, alpha]
        S1 = (AD[None, :, :] * inv[:, None, :]).sum(-1)                       # j=1
        S2 = (AD[None, :, :] * (p[None, None, :] * inv[:, None, :])).sum(-1)  # j=2
        beta_slip = torch.zeros(
            positions.shape[0], 3, 3, device=positions.device, dtype=positions.dtype
        )
        beta_slip[:, :, 0] = _STROH_PREFACTOR * S1.imag
        beta_slip[:, :, 1] = _STROH_PREFACTOR * S2.imag
        # Rotate slip-frame distortion back to crystal frame: beta_c = M^T beta_s M.
        M = self.M
        return M.transpose(-1, -2) @ beta_slip @ M

    def displacement(self, positions: torch.Tensor) -> torch.Tensor:
        """Anisotropic displacement field ``u(r)`` at crystal-frame ``positions``.

        The Stroh displacement is the integral of :meth:`displacement_gradient`:

            ``u_i = (1/pi) Im{ sum_alpha A[i,alpha] D[alpha] ln(x1 + p_alpha x2) }``

        since ``d/dx1 ln(x1 + p x2) = 1/(x1 + p x2)`` and ``d/dx2 ln(...) = p/(...)``,
        which are exactly the terms summed in the distortion. Returns ``(N, 3)`` in the
        crystal frame, in micrometres.

        The complex logarithm is multivalued, and that is physical: ``u`` jumps by the
        Burgers vector across the cut plane, so ``exp(i g.u)`` is single-valued only when
        ``g.b`` is an integer -- the coherent form of the ``g.b`` invisibility criterion.
        Required for coherent imaging, where the exit-wave phase is ``g.u``.
        """
        rel = positions - self.core_position
        rel_slip = rel @ self.M.transpose(-1, -2)
        x1, x2 = rel_slip[:, 0], rel_slip[:, 1]
        # Regularize by clamping the in-plane RADIUS to the core radius, rather than by
        # perturbing ``denom``. This keeps u exactly the singular Stroh solution wherever
        # r >= rc -- so grad(u) reproduces beta there to machine precision once beta's
        # Lorentzian cutoff is off (see ``core_model``). Inside the core both fields are
        # regularized and neither is physical; mask that region.
        rc = self.core_radius_um
        r = torch.sqrt(x1 * x1 + x2 * x2)
        scale = torch.clamp(rc / r.clamp_min(1e-30), max=1.0)         # 1 outside, rc/r inside
        x1c, x2c = x1 * torch.where(r > rc, torch.ones_like(scale), scale), \
                   x2 * torch.where(r > rc, torch.ones_like(scale), scale)
        denom = x1c[:, None] + self.p[None, :] * x2c[:, None]         # (N, 3) complex
        AD = self.A * self.D[None, :]                                 # (3, 3): [i, alpha]
        S = (AD[None, :, :] * torch.log(denom)[:, None, :]).sum(-1)   # (N, 3) complex
        u_slip = _STROH_PREFACTOR * S.imag
        return u_slip @ self.M                                        # slip -> crystal

    def deformation_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """``F(r) = I + beta(r)`` at ``positions`` ``(N, 3)`` -> ``(N, 3, 3)``."""
        beta = self.displacement_gradient(positions)
        eye = torch.eye(3, device=positions.device, dtype=positions.dtype)
        return eye + beta


def _cartesian_geometry(burgers, slip_normal, line, *, crystal, dtype, device):
    """Resolve Miller inputs to Cartesian ``(b, n, l)`` (reuse midas_defect maps)."""
    def _vec(x):
        return torch.as_tensor(x, dtype=dtype, device=device)

    b, n, l = _vec(burgers), _vec(slip_normal), _vec(line)
    if crystal is not None:
        A = _crystal_A_matrix(crystal, dtype=dtype, device=device)
        b = _to_cartesian(b, A, "direct")
        l = _to_cartesian(l, A, "direct")
        n = _to_cartesian(n, A, "reciprocal")
    return b, n, l


def stroh_dislocation(
    C6: torch.Tensor,
    *,
    burgers: Sequence[float],
    slip_normal: Sequence[float],
    character: str = "edge",
    line: Sequence[float] | None = None,
    burgers_length_A: float = 2.556,
    core_position=(0.0, 0.0, 0.0),
    core_radius_um: float = 0.05,
    core_model: str = "lorentzian",
    crystal=None,
) -> StrohDislocation:
    """Solve the Stroh state for one dislocation.

    ``burgers``/``line`` are real-space ``[uvw]``; ``slip_normal`` is reciprocal
    ``(hkl)``. ``character`` selects the line when ``line`` is not given: ``"screw"``
    (line ∥ b) or ``"edge"`` (line = n x b). ``burgers_length_A`` sets the physical
    Burgers magnitude (Angstrom). Differentiable w.r.t. ``C6`` (and, via ``crystal``,
    the lattice).
    """
    dtype, device = C6.dtype, C6.device
    b_cart, n_cart, l_in = _cartesian_geometry(
        burgers, slip_normal, line if line is not None else burgers,
        crystal=crystal, dtype=dtype, device=device,
    )
    if line is not None:
        l_cart = l_in
    elif character == "screw":
        l_cart = b_cart
    elif character == "edge":
        l_cart = torch.linalg.cross(n_cart, b_cart)
    else:
        raise ValueError(f"character must be 'screw' or 'edge', got {character!r}")

    M = _slip_frame(l_cart, n_cart)
    C4 = _rotate_tensor(_voigt_to_tensor(C6), M)
    p, A, B = _stroh_eig(C4)

    b_len_um = burgers_length_A / _ANGSTROM_PER_UM
    b_slip = M @ (b_cart / torch.linalg.norm(b_cart)) * b_len_um
    D = b_slip.to(p.dtype) @ B

    return StrohDislocation(
        M=M, p=p, A=A, D=D,
        line_cart=l_cart / torch.linalg.norm(l_cart),
        core_position=torch.as_tensor(core_position, dtype=dtype, device=device),
        core_radius_um=core_radius_um, core_model=core_model,
    )


def edge_dislocation_wall(
    C6: torch.Tensor,
    *,
    burgers: Sequence[float],
    slip_normal: Sequence[float],
    n_dislocations: int = 5,
    spacing_um: float = 2.0,
    along=(0.0, 1.0, 0.0),
    burgers_length_A: float = 2.556,
    core_radius_um: float = 0.05,
    crystal=None,
) -> list[StrohDislocation]:
    """A low-angle tilt boundary: a row of identical edge dislocations.

    ``n_dislocations`` edges stacked along ``along`` at ``spacing_um``. The wall
    produces a discrete tilt of ``theta ~ |b| / spacing`` across the boundary (Frank's
    relation) — a clean synthetic target for the mosaicity scan and for GND recovery.
    Returns a list of :class:`StrohDislocation` (superpose via
    :func:`dislocation_deformation_field`).
    """
    dtype, device = C6.dtype, C6.device
    axis = torch.as_tensor(along, dtype=dtype, device=device)
    axis = axis / torch.linalg.norm(axis)
    c0 = (n_dislocations - 1) / 2.0
    out = []
    for k in range(n_dislocations):
        pos = (k - c0) * spacing_um * axis
        out.append(stroh_dislocation(
            C6, burgers=burgers, slip_normal=slip_normal, character="edge",
            burgers_length_A=burgers_length_A, core_position=pos,
            core_radius_um=core_radius_um, crystal=crystal,
        ))
    return out


def dislocation_dipole(
    C6: torch.Tensor,
    *,
    burgers: Sequence[float],
    slip_normal: Sequence[float],
    character: str = "edge",
    separation_um: float = 3.0,
    along=(1.0, 0.0, 0.0),
    burgers_length_A: float = 2.556,
    core_radius_um: float = 0.05,
    crystal=None,
) -> list[StrohDislocation]:
    """A dipole: two opposite-sign dislocations separated by ``separation_um``.

    The far field cancels (screened), leaving a localized contrast feature — the
    canonical weak-beam pair. Opposite sign is realised by negating the Burgers vector.
    """
    b = torch.as_tensor(burgers, dtype=C6.dtype, device=C6.device)
    axis = torch.as_tensor(along, dtype=C6.dtype, device=C6.device)
    axis = axis / torch.linalg.norm(axis)
    half = 0.5 * separation_um * axis
    # Fix the line direction from the +b dislocation and pass it EXPLICITLY to both,
    # so the -b partner is a genuinely opposite-sign dislocation (negating b alone
    # would also flip the auto-derived line = the SAME physical dislocation).
    plus = stroh_dislocation(
        C6, burgers=b, slip_normal=slip_normal, character=character,
        burgers_length_A=burgers_length_A, core_position=half,
        core_radius_um=core_radius_um, crystal=crystal)
    minus = stroh_dislocation(
        C6, burgers=-b, slip_normal=slip_normal, line=plus.line_cart,
        burgers_length_A=burgers_length_A, core_position=-half,
        core_radius_um=core_radius_um, crystal=crystal)
    return [plus, minus]


def dislocation_deformation_field(
    positions: torch.Tensor,
    dislocations: StrohDislocation | Sequence[StrohDislocation],
    *,
    reference_orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    shape: tuple[int, int, int] | None = None,
) -> DeformationField:
    """Build a :class:`DeformationField` from one or more dislocations by superposition.

    Distortions add in linear elasticity: ``F = I + sum_k beta_k``. ``positions`` is
    ``(N, 3)`` (micrometers). Returns a field ready for the DFXM forward.
    """
    if isinstance(dislocations, StrohDislocation):
        dislocations = [dislocations]
    device, dtype = positions.device, positions.dtype
    beta = torch.zeros(positions.shape[0], 3, 3, device=device, dtype=dtype)
    for d in dislocations:
        beta = beta + d.displacement_gradient(positions)
    eye = torch.eye(3, device=device, dtype=dtype)
    F = eye + beta
    if reference_orientation is None:
        reference_orientation = torch.eye(3, device=device, dtype=dtype)
    else:
        reference_orientation = torch.as_tensor(reference_orientation, device=device, dtype=dtype)
    latc = torch.as_tensor(lattice_params, device=device, dtype=dtype)
    return DeformationField(
        positions=positions, F=F,
        reference_orientation=reference_orientation,
        lattice_params=latc, shape=shape,
    )
