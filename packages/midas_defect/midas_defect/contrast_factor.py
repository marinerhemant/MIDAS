"""Dislocation diffraction contrast factors via anisotropic elasticity (Stroh).

Implements the calculation performed by the ANIZC program — Borbély,
Dragomir-Cernatescu, Ribárik & Ungár, *J. Appl. Cryst.* **36** (2003) 160-162 —
the diffraction contrast factor ``C`` of a straight dislocation in an elastically
anisotropic crystal. ``C`` is obtained by integrating the squared angular
distortion ``F(φ)`` over the slip plane (paper eqns 3-4); the angular distortion
``β_ij(φ)`` (paper eqn 6) comes from the sextic / Stroh solution of the mechanical
equilibrium of a single dislocation in an infinite anisotropic medium (Teodosiu
1982; Stroh formalism, cf. Ting, *Anisotropic Elasticity*).

Why this lives in midas_defect
------------------------------
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
from typing import List, Sequence, Tuple

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
# Voigt index map and stiffness construction
# ---------------------------------------------------------------------------

#: Voigt index for each (i, j) Cartesian pair: 11→0 22→1 33→2 23→3 13→4 12→5.
_VOIGT_IDX = torch.tensor([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def cubic_stiffness(
    c11: float, c12: float, c44: float, *,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Return the 6×6 Voigt stiffness matrix of a cubic crystal.

    Parameters
    ----------
    c11, c12, c44
        Cubic elastic constants (GPa, or any consistent unit). ``C`` is
        dimensionless so only their ratios matter.
    """
    device = torch.device("cpu") if device is None else device
    c11 = torch.as_tensor(c11, dtype=dtype, device=device)
    c12 = torch.as_tensor(c12, dtype=dtype, device=device)
    c44 = torch.as_tensor(c44, dtype=dtype, device=device)
    z = torch.zeros((), dtype=dtype, device=device)
    rows = [
        torch.stack([c11, c12, c12, z, z, z]),
        torch.stack([c12, c11, c12, z, z, z]),
        torch.stack([c12, c12, c11, z, z, z]),
        torch.stack([z, z, z, c44, z, z]),
        torch.stack([z, z, z, z, c44, z]),
        torch.stack([z, z, z, z, z, c44]),
    ]
    return torch.stack(rows, dim=0)


def _voigt_to_tensor(C6: torch.Tensor) -> torch.Tensor:
    """Expand a 6×6 Voigt stiffness matrix to the full ``(3,3,3,3)`` tensor."""
    idx = _VOIGT_IDX.to(C6.device)
    return C6[idx[:, :, None, None], idx[None, None, :, :]]


def _rotate_tensor(C4: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Rotate a 4th-rank tensor: ``C'_{ijkl} = M_ia M_jb M_kc M_ld C_{abcd}``."""
    return torch.einsum("ia,jb,kc,ld,abcd->ijkl", M, M, M, M, C4)


def _slip_frame(line: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (rows = e1, e2, e3) from crystal to slip coordinates.

    ``e3`` = unit line, ``e2`` = unit slip-plane normal, ``e1 = e2 × e3``. The
    returned ``M`` maps a crystal-frame vector ``v`` to the slip frame via
    ``M @ v``. ``normal`` is expected ⊥ ``line`` (a physical slip system); a
    Gram-Schmidt step removes any residual non-orthogonality.
    """
    e3 = line / torch.linalg.norm(line)
    e2 = normal - (normal @ e3) * e3
    e2 = e2 / torch.linalg.norm(e2)
    e1 = torch.linalg.cross(e2, e3)
    e1 = e1 / torch.linalg.norm(e1)
    return torch.stack([e1, e2, e3], dim=0)


def _stroh_eig(
    C4: torch.Tensor,
    *,
    degeneracy_tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stroh eigensolution: 3 roots ``p`` (Im>0) and amplitude/stress vectors A,B.

    Solves the 6×6 fundamental eigenproblem ``N ξ = p ξ`` with
    ``ξ = (a, b)``, where for a dislocation line along ``e3``::

        Q_ik = C_{i1k1},  R_ik = C_{i1k2},  T_ik = C_{i2k2}
        N = [[ -T⁻¹Rᵀ ,        T⁻¹  ],
             [ R T⁻¹Rᵀ - Q ,  -R T⁻¹ ]]

    Returns ``(p, A, B)`` for the three roots with positive imaginary part,
    with columns normalised so ``2 aₐᵀ bₐ = 1`` (Stroh orthonormality).
    """
    Q = C4[:, 0, :, 0]
    R = C4[:, 0, :, 1]
    T = C4[:, 1, :, 1]
    Tinv = torch.linalg.inv(T)
    RT = R.transpose(-1, -2)
    N1 = -Tinv @ RT
    N3 = R @ Tinv @ RT - Q
    top = torch.cat([N1, Tinv], dim=1)
    bot = torch.cat([N3, N1.transpose(-1, -2)], dim=1)
    N = torch.cat([top, bot], dim=0)

    p_all, _ = torch.linalg.eig(N)
    mask = p_all.imag > 0
    if int(mask.sum()) != 3:
        raise ValueError(
            f"expected 3 sextic roots with Im(p)>0, got {int(mask.sum())}; "
            "the stiffness may be elastically isotropic (degenerate roots) or "
            "ill-conditioned."
        )
    p = p_all[mask]

    # Near-degenerate roots ⇒ the matrix is defective and the simple-eigenvector
    # Stroh path returns unreliable amplitude vectors. This happens in the
    # elastically isotropic limit (Zener ratio → 1), where the roots collapse to
    # p = i. Refuse rather than silently return garbage — a dislocation-contrast
    # analysis is only meaningful for an elastically anisotropic crystal anyway.
    pdiff = torch.abs(p[:, None] - p[None, :])
    pdiff = pdiff + torch.eye(3, dtype=pdiff.dtype, device=pdiff.device) * 9.0
    if float(pdiff.min().detach()) < degeneracy_tol:
        raise ValueError(
            "near-degenerate sextic roots: the crystal is at/near the elastically "
            "isotropic limit (Zener ratio ≈ 1), where the simple-eigenvector Stroh "
            "solution is unreliable. Contrast factors require an anisotropic crystal."
        )

    # Amplitude vectors a from the 3×3 null space of (Q + p(R+Rᵀ) + p²T):
    Qc, Rc, Tc, RTc = (Q.to(p.dtype), R.to(p.dtype),
                       T.to(p.dtype), RT.to(p.dtype))
    eye3 = torch.eye(3, dtype=p.dtype, device=C4.device)
    a_cols, b_cols = [], []
    for k in range(3):
        Mk = Qc + p[k] * (Rc + RTc) + p[k] * p[k] * Tc
        # null vector = singular vector with smallest singular value
        _, _, Vh = torch.linalg.svd(Mk)
        a = Vh[-1].conj()
        b = (RTc + p[k] * Tc) @ a              # Stroh stress vector
        s = 1.0 / torch.sqrt(2.0 * (a @ b))    # bilinear (no conjugate)
        a_cols.append(a * s)
        b_cols.append(b * s)
    A = torch.stack(a_cols, dim=1)
    B = torch.stack(b_cols, dim=1)
    return p, A, B


# ---------------------------------------------------------------------------
# Crystal → Cartesian geometry (any crystal system)
# ---------------------------------------------------------------------------
#
# The Stroh core works in a Cartesian slip frame. For a *cubic* cell the integer
# Miller indices double as Cartesian directions, but for any lower symmetry they
# do not. We reuse the canonical orthogonalisation ("A") matrix from
# `midas_stress.tensor.lattice_params_to_A_matrix` (Busing-Levy convention, the
# same matrix the Fable-Beaudoin strain solver uses) — never a re-ported B
# matrix. ``A`` maps fractional crystal coordinates to Cartesian, so:
#   * a real-space direction  [uvw]  (Burgers vector, dislocation line)  → A · [uvw]
#   * a reciprocal direction   (hkl)  (plane normal, diffraction vector)  → A⁻ᵀ · (hkl)
# For cubic, A = a·I, so both maps are the identity up to scale and every result
# is byte-identical to passing raw Miller indices (see the cubic-identity test).


def _crystal_A_matrix(crystal, *, dtype, device) -> torch.Tensor:
    """Orthogonalisation matrix A (fractional → Cartesian) for a `Crystal`."""
    from midas_stress.tensor import lattice_params_to_A_matrix

    lat = crystal.lattice
    latc = torch.tensor(
        [lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma],
        dtype=dtype, device=device)
    return lattice_params_to_A_matrix(latc)


def _to_cartesian(v: torch.Tensor, A: torch.Tensor, kind: str) -> torch.Tensor:
    """Map a Miller vector to Cartesian: ``direct`` → A·v, ``reciprocal`` → A⁻ᵀ·v."""
    if kind == "direct":
        return A @ v
    if kind == "reciprocal":
        return torch.linalg.solve(A.transpose(-1, -2), v)   # A⁻ᵀ · v
    raise ValueError(f"kind must be 'direct' or 'reciprocal', got {kind!r}")


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
    ``C = ⟨F²⟩_φ = (1/2π)∫₀²ᐩ F² dφ`` (eqn 3).

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
    # The factor 2 (vs the bare φ-mean) is pinned to the paper's silver worked
    # example C(2̄20, 60° mixed) = 0.3843 — see tests/test_contrast_factor.py.
    return 2.0 * (F * F).mean()


# ---------------------------------------------------------------------------
# Slip-system tables and averaged contrast factor
# ---------------------------------------------------------------------------

def _gen_slip_systems(
    planes: Sequence[Tuple[int, int, int]],
    burgers_pool: Sequence[Tuple[int, int, int]],
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """All (normal, burgers) pairs with burgers ⊥ normal, ± deduplicated."""
    systems = []
    seen = set()
    for n in planes:
        nv = np.array(n, float)
        for b in burgers_pool:
            bv = np.array(b, float)
            if abs(float(nv @ bv)) > 1e-9:
                continue
            key = (n, tuple(sorted((b, tuple(-x for x in b)))))
            if key in seen:
                continue
            seen.add(key)
            systems.append((n, b))
    return systems


def fcc_slip_systems() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """The 12 FCC ``{111}⟨110⟩`` slip systems as ``(plane_normal, burgers)``."""
    planes = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]   # 4 distinct {111}
    burgers = [(1, -1, 0), (1, 0, -1), (0, 1, -1),
               (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    return _gen_slip_systems(planes, burgers)


def bcc_slip_systems() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """The 12 BCC ``{110}⟨111⟩`` slip systems as ``(plane_normal, burgers)``."""
    planes = [(1, 1, 0), (1, -1, 0), (1, 0, 1),                 # 6 distinct {110}
              (1, 0, -1), (0, 1, 1), (0, 1, -1)]
    burgers = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    return _gen_slip_systems(planes, burgers)


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
