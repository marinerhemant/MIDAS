"""Dislocation diffraction contrast factors via anisotropic elasticity (Stroh).

Implements the calculation performed by the ANIZC program — Borbély,
Dragomir-Cernatescu, Ribárik & Ungár, *J. Appl. Cryst.* **36** (2003) 160-162 —
the diffraction contrast factor ``C`` of a straight dislocation in an elastically
anisotropic crystal. ``C`` is obtained by integrating the squared angular
distortion ``F(φ)`` over the slip plane (paper eqns 3-4); the angular distortion
``β_ij(φ)`` (paper eqn 6) comes from the sextic / Stroh solution of the mechanical
equilibrium of a single dislocation in an infinite anisotropic medium (Teodosiu
1982; Stroh formalism, cf. Ting, *Anisotropic Elasticity*).

Where the elasticity lives
--------------------------
The Stroh sextic solver and the stiffness builders are **not** in this file
any more -- they are in :mod:`midas_ddd.elasticity`, and re-exported below so
existing imports keep resolving. What stays here is the contrast factor
itself, which is line-profile analysis rather than general elasticity.

Why the contrast factor lives in midas_defect
---------------------------------------------
:mod:`midas_defect.williamson_hall` reduces per-hkl asterism breadth to a single
strain ε and a dislocation density ρ = prefactor·ε²/b², treating every reflection
as equally strain-sensitive. That is the *elastically isotropic* approximation. In
a real crystal the strain broadening is hkl-dependent — "strain anisotropy" — and
the dislocation contrast factor :math:`\\bar C_{hkl}` is precisely the quantity that
captures it. The *modified* Williamson-Hall / Warren-Averbach methods replace
``|q|`` with ``|q|·C̄^{1/2}`` to collapse the anisotropic scatter onto one line
(Ungár & Borbély, *Appl. Phys. Lett.* **69** (1996) 3173). This module computes
:math:`\\bar C_{hkl}` so :func:`williamson_hall.modified_williamson_hall` can use it.

Conventions
-----------
* Elastic constants: full 6×6 Voigt stiffness matrix (any consistent unit — ``C``
  is dimensionless). :func:`cubic_stiffness` builds it from ``(c11, c12, c44)``.
* Slip coordinate system (paper Fig. 1): ``e3`` = dislocation line, ``e2`` =
  slip-plane normal, ``e1 = e2 × e3``. Character ψ = ∠(line, Burgers): 0° = screw,
  90° = edge.
* Burgers vector, slip-plane normal, line and diffraction ``g`` are given in
  crystal Cartesian axes; for a cubic cell these coincide with the integer
  ``(hkl)`` / ``[uvw]`` axes. ``C`` depends only on directions, not magnitudes.

Differentiability / device (engineering contract)
--------------------------------------------------
Pure torch: the Voigt→4-tensor expansion, the slip-frame rotation, the 6×6 Stroh
eigenproblem (:func:`torch.linalg.eig`) and the φ-quadrature are all differentiable
w.r.t. the elastic constants. ``complex128`` eig is unsupported on the MPS backend,
so this path runs on CPU / CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

__all__ = [
    "cubic_stiffness",
    "single_contrast_factor",
    "average_contrast_factor",
    "cubic_invariant_H2",
    "fit_cbar_h00_q",
    "fcc_slip_systems",
    "bcc_slip_systems",
    "CbarModel",
]


# ---------------------------------------------------------------------------
# Anisotropic-elasticity primitives -- MOVED to midas_ddd.elasticity
# ---------------------------------------------------------------------------
#
# The Stroh sextic solver, the stiffness builders, the slip-frame rotation and
# the crystal->Cartesian maps now live in `midas_ddd.elasticity`. They moved
# when a third and fourth consumer appeared (midas_saxs and the near-Bragg
# diffuse forward) and a SAXS package could not reasonably depend on an FF-HEDM
# metrology package just to build a stiffness matrix.
#
# They are re-exported here, unchanged, so every historical import path keeps
# working -- `from midas_defect.contrast_factor import cubic_stiffness, _stroh_eig`
# resolves exactly as before. New code should import from `midas_ddd` directly.
# Do NOT re-port them into this file.

from midas_ddd.elasticity import (  # noqa: F401
    _VOIGT_IDX,
    _crystal_A_matrix,
    _gen_slip_systems,
    _rotate_tensor,
    _slip_frame,
    _stroh_eig,
    _to_cartesian,
    _voigt_to_tensor,
    bcc_slip_systems,
    cubic_stiffness,
    fcc_slip_systems,
    hexagonal_stiffness,
)


def single_contrast_factor(
    C6: torch.Tensor,
    *,
    burgers: Sequence[float],
    slip_normal: Sequence[float],
    line: Sequence[float],
    g: Sequence[float],
    crystal=None,
    n_phi: int = 720,
) -> torch.Tensor:
    """Contrast factor ``C`` of one straight dislocation for one reflection.

    This is the ANIZC single-dislocation calculation (paper §2): rotate the
    stiffness into the slip frame, solve the sextic, build the angular distortion
    ``β_ij(φ)`` (eqn 6), form ``F(φ) = Σ_i Σ_{j=1,2} γ_i γ_j β_ij`` (eqn 4) with
    ``γ`` the direction cosines of ``g`` in the slip frame, and return
    ``C = (1/π)∫₀²ᐩ F² dφ = 2·⟨F²⟩_φ`` (eqn 3).

    Parameters
    ----------
    C6
        6×6 Voigt stiffness tensor (torch, e.g. from :func:`cubic_stiffness`).
    burgers, line
        Real-space directions ``[uvw]``.
    slip_normal, g
        Reciprocal directions ``(hkl)`` — slip-plane normal and diffraction vector.
    crystal
        Optional `midas_hkls.Crystal`. When given, ``burgers``/``line`` are
        interpreted as ``[uvw]`` and ``slip_normal``/``g`` as ``(hkl)`` and mapped
        to Cartesian via the cell's orthogonalisation matrix — enabling any
        (tetragonal, hexagonal, …) cell. When ``None`` the inputs are taken as
        Cartesian directions directly (the cubic / Miller-≡-Cartesian shortcut).
        Only directions matter, never magnitudes.
    n_phi
        Number of φ quadrature points over ``[0, 2π)``.

    Returns
    -------
    torch scalar (dtype of ``C6``). Differentiable w.r.t. ``C6`` (and, via
    ``crystal``, the lattice parameters).
    """
    dtype, device = C6.dtype, C6.device

    def _vec(x):
        return torch.as_tensor(x, dtype=dtype, device=device)

    b_vec, n_vec, l_vec, g_vec = (_vec(burgers), _vec(slip_normal),
                                  _vec(line), _vec(g))

    if crystal is not None:
        A = _crystal_A_matrix(crystal, dtype=dtype, device=device)
        b_vec = _to_cartesian(b_vec, A, "direct")
        l_vec = _to_cartesian(l_vec, A, "direct")
        n_vec = _to_cartesian(n_vec, A, "reciprocal")
        g_vec = _to_cartesian(g_vec, A, "reciprocal")

    M = _slip_frame(l_vec, n_vec)
    C4 = _rotate_tensor(_voigt_to_tensor(C6), M)
    p, A, B = _stroh_eig(C4)

    b_slip = M @ (b_vec / torch.linalg.norm(b_vec))      # unit Burgers, slip frame
    gamma = M @ (g_vec / torch.linalg.norm(g_vec))       # direction cosines of g
    D = b_slip.to(p.dtype) @ B                            # D_α = Σ_l b_l B_{lα}

    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1,
                         dtype=dtype, device=device)[:-1]
    cphi = torch.cos(phi)[:, None]                        # (n_phi, 1)
    sphi = torch.sin(phi)[:, None]
    inv = 1.0 / (cphi + p[None, :] * sphi)                # (n_phi, 3) complex
    AD = A * D[None, :]                                   # (3, 3): [i, α]

    # β_{i,1} (j=1) and β_{i,2} (j=2) over φ; β = 2 Im{ Σ_α A_iα D_α (·)/denom }
    S1 = (AD[None, :, :] * inv[:, None, :]).sum(-1)               # j=1
    S2 = (AD[None, :, :] * (p[None, None, :] * inv[:, None, :])).sum(-1)  # j=2
    beta1 = 2.0 * S1.imag                                 # (n_phi, 3)
    beta2 = 2.0 * S2.imag

    F = (gamma[None, :] * (gamma[0] * beta1 + gamma[1] * beta2)).sum(-1)
    # ANIZC / Klimanek-Kužel normalisation C = (1/π)∫₀²ᐩ F² dφ = 2·⟨F²⟩_φ.
    # The factor 2 is EQUATION (3) OF THE PAPER, not a fitted constant: Borbély,
    # Dragomir-Cernatescu, Ribárik & Ungár, J. Appl. Cryst. 36 (2003) 160-162,
    # writes C = (1/π)∫F²dφ, which is twice the bare φ-mean. It is corroborated
    # three ways that do not use the silver example: the Burgers circuit closes
    # to 1e-14, the isotropic limit converges on the analytic sin²Ψcos²Ψ, and the
    # resulting C̄h00 reproduces Ungár et al., J. Appl. Cryst. 32 (1999) 992.
    # DO NOT "fix" this to match a bare mean. Halving it silently halves every
    # contrast factor and puts modified_williamson_hall's ρ out by 2x, and only
    # the anchor tests would fail.
    return 2.0 * (F * F).mean()


# ---------------------------------------------------------------------------
# Averaged contrast factor over a slip-system family
# ---------------------------------------------------------------------------


def average_contrast_factor(
    C6: torch.Tensor,
    g: Sequence[float],
    *,
    slip_systems=None,
    family: str = "fcc",
    character: str = "screw",
    crystal=None,
    n_phi: int = 720,
) -> torch.Tensor:
    """Slip-system-averaged contrast factor ``C̄`` for a reflection ``g``.

    Averages :func:`single_contrast_factor` over all slip systems of the family
    (equal dislocation densities, paper §3), for a chosen dislocation character.
    For an untextured aggregate this equals the average over the symmetry-
    equivalent ``(hkl)`` permutations (Ungár & Tichy 1999).

    All geometry is resolved to Cartesian here: slip-plane normals and ``g`` are
    reciprocal ``(hkl)`` directions, Burgers vectors are real-space ``[uvw]``, and
    the edge-character line ``= n × b`` is built in Cartesian — correct for any
    crystal system once ``crystal`` is supplied.

    Parameters
    ----------
    C6
        6×6 Voigt stiffness tensor.
    g
        Diffraction vector as ``(hkl)``.
    slip_systems
        Explicit ``[(normal, burgers), ...]`` list; overrides ``family``.
    family
        ``"fcc"`` ({111}⟨110⟩) or ``"bcc"`` ({110}⟨111⟩) when ``slip_systems``
        is not given.
    character
        ``"screw"`` (line ∥ b) or ``"edge"`` (line ⊥ b, in slip plane).
    crystal
        Optional `midas_hkls.Crystal`. Maps Miller indices to Cartesian via the
        cell's orthogonalisation matrix (any system). ``None`` ⇒ cubic shortcut.

    Notes
    -----
    The average is currently unweighted (equal Burgers magnitude per family, true
    for cubic ⟨110⟩/⟨111⟩). Multi-Burgers families (e.g. hexagonal ⟨a⟩ vs ⟨c+a⟩)
    will need the ``b²``-weighting of paper eqn 2 — added with the hexagonal slip
    catalog.
    """
    if character not in ("screw", "edge"):
        raise ValueError(f"character must be 'screw' or 'edge', got {character!r}")
    if slip_systems is None:
        systems = {"fcc": fcc_slip_systems, "bcc": bcc_slip_systems}[family]()
    else:
        systems = slip_systems

    dtype, device = C6.dtype, C6.device
    A = (_crystal_A_matrix(crystal, dtype=dtype, device=device)
         if crystal is not None else None)

    def _cart(v, kind):
        t = torch.as_tensor(v, dtype=dtype, device=device)
        return _to_cartesian(t, A, kind) if A is not None else t

    g_cart = _cart(g, "reciprocal")
    vals = []
    for normal, burgers in systems:
        n_cart = _cart(normal, "reciprocal")
        b_cart = _cart(burgers, "direct")
        if character == "screw":
            line_cart = b_cart
        else:                                   # edge: ⊥ b, in slip plane
            line_cart = torch.linalg.cross(n_cart, b_cart)
        vals.append(single_contrast_factor(
            C6, burgers=b_cart, slip_normal=n_cart, line=line_cart,
            g=g_cart, n_phi=n_phi))             # already Cartesian ⇒ crystal=None
    return torch.stack(vals).mean()


# ---------------------------------------------------------------------------
# Cubic averaged-contrast-factor model:  C̄ = C̄_h00 (1 - q H²)
# ---------------------------------------------------------------------------

def cubic_invariant_H2(hkl: Sequence[int]) -> float:
    """Fourth-order cubic invariant ``H² = (h²k²+h²l²+k²l²)/(h²+k²+l²)²``.

    This is the variable in the linear strain-anisotropy model
    ``C̄_hkl = C̄_h00 (1 - q H²)`` (Ungár & Tichy, *Phys. Status Solidi A* **171**
    (1999) 425). ``H² = 0`` for ⟨h00⟩, ``1/3`` for ⟨hhh⟩.
    """
    h, k, l = (float(x) for x in hkl)
    s = h * h + k * k + l * l
    if s == 0:
        raise ValueError("hkl must be non-zero")
    return (h * h * k * k + h * h * l * l + k * k * l * l) / (s * s)


@dataclass
class CbarModel:
    """Fitted cubic strain-anisotropy model ``C̄ = C̄_h00 (1 - q H²)``."""
    cbar_h00: float
    q: float
    residual_norm: float
    n_hkls: int

    def __call__(self, hkl: Sequence[int]) -> float:
        return self.cbar_h00 * (1.0 - self.q * cubic_invariant_H2(hkl))


def fit_cbar_h00_q(
    hkls: Sequence[Sequence[int]],
    cbars: Sequence[float],
) -> CbarModel:
    """Fit ``C̄ = C̄_h00 (1 - q H²)`` to averaged contrast factors.

    Linear in ``H²``: ``C̄ = C̄_h00 - (C̄_h00 q) H²``, so the intercept is
    ``C̄_h00`` and ``q = -slope / intercept``.
    """
    H2 = np.array([cubic_invariant_H2(h) for h in hkls], dtype=float)
    y = np.asarray(cbars, dtype=float)
    if len(y) < 2:
        raise ValueError("need at least 2 reflections to fit C̄_h00 and q")
    A = np.column_stack([np.ones_like(H2), H2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    q = -slope / intercept if intercept != 0 else float("nan")
    resid = float(np.linalg.norm(y - A @ coef))
    return CbarModel(cbar_h00=intercept, q=q, residual_norm=resid, n_hkls=len(y))
