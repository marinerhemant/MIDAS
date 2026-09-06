"""Anisotropic-elasticity primitives for dislocations (Stroh sextic solution).

**Single source of truth.** These primitives used to live in
``midas_defect.contrast_factor`` / ``midas_defect.contrast_factor_hex``, which
already carried a "do NOT re-port" contract because ``midas_dfxm.dislocation``
imports them. They moved down here when a third and fourth consumer appeared
(``midas_saxs`` and the near-Bragg diffuse forward), so that dislocation
elasticity has exactly one home and nothing has to depend on an FF-HEDM
metrology package to get a stiffness matrix. ``midas_defect.contrast_factor``
re-exports every name below, so every historical import path still works.

What is here
------------
* Stiffness construction: :func:`cubic_stiffness`, :func:`hexagonal_stiffness`.
* Voigt ↔ 4-tensor and frame rotation: :func:`_voigt_to_tensor`,
  :func:`_rotate_tensor`, :func:`_slip_frame`.
* The Stroh sextic eigensolution :func:`_stroh_eig` — 3 roots ``p`` with
  ``Im(p) > 0`` plus the amplitude/stress matrices ``A``, ``B``.
* Crystal → Cartesian geometry for non-cubic cells: :func:`_crystal_A_matrix`,
  :func:`_to_cartesian`.
* Slip-system tables: :func:`fcc_slip_systems`, :func:`bcc_slip_systems`.

The physics reference is unchanged: Teodosiu 1982 / Ting, *Anisotropic
Elasticity*, as implemented by ANIZC (Borbély, Dragomir-Cernatescu, Ribárik &
Ungár, *J. Appl. Cryst.* **36** (2003) 160-162).

Conventions
-----------
* Elastic constants: full 6×6 Voigt stiffness matrix, any consistent unit.
* Slip coordinate system: ``e3`` = dislocation line, ``e2`` = slip-plane normal,
  ``e1 = e2 × e3``. Character ψ = ∠(line, Burgers): 0° = screw, 90° = edge.
* Burgers vector, slip-plane normal, line and diffraction ``g`` in crystal
  Cartesian axes; for a cubic cell these coincide with the integer ``(hkl)`` /
  ``[uvw]`` axes.

Differentiability / device
--------------------------
Pure torch: the Voigt→4-tensor expansion, the slip-frame rotation and the 6×6
Stroh eigenproblem (:func:`torch.linalg.eig`) are all differentiable w.r.t. the
elastic constants. ``complex128`` eig is unsupported on the MPS backend, so this
path runs on CPU / CUDA.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "cubic_stiffness",
    "hexagonal_stiffness",
    "fcc_slip_systems",
    "bcc_slip_systems",
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
# Hexagonal stiffness (moved from midas_defect.contrast_factor_hex)
# ---------------------------------------------------------------------------

def hexagonal_stiffness(
    c11: float, c12: float, c13: float, c33: float, c44: float, *,
    dtype: torch.dtype = torch.float64, device=None,
) -> torch.Tensor:
    """6×6 Voigt stiffness of a hexagonal crystal (``C66 = (C11-C12)/2``)."""
    device = torch.device("cpu") if device is None else device
    t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    c11, c12, c13, c33, c44 = t(c11), t(c12), t(c13), t(c33), t(c44)
    c66 = (c11 - c12) / 2.0
    z = torch.zeros((), dtype=dtype, device=device)
    rows = [
        torch.stack([c11, c12, c13, z, z, z]),
        torch.stack([c12, c11, c13, z, z, z]),
        torch.stack([c13, c13, c33, z, z, z]),
        torch.stack([z, z, z, c44, z, z]),
        torch.stack([z, z, z, z, c44, z]),
        torch.stack([z, z, z, z, z, c66]),
    ]
    return torch.stack(rows, dim=0)


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


# ---------------------------------------------------------------------------
# Slip-system tables
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

