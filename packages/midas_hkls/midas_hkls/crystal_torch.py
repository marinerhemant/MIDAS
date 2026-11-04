"""Torch packing of a Crystal — produces a ``CrystalTensor`` with selectable
``requires_grad`` flags on each parameter group.

Symmetry expansion is done in pure-python integer arithmetic (in
``Crystal.unit_cell_atoms``); the torch path operates on the already-expanded
unit-cell atom list.  Asymmetric-unit gradients flow because the symmetry
expansion is a constant linear map that is rebuilt from torch operations
when ``link_to_asu=True`` is used (default for fitting workflows).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from .crystal import Atom, Crystal
from .tables import STBF

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = ["CrystalTensor", "crystal_to_tensor"]


@dataclass
class CrystalTensor:
    """Torch-backed view of a Crystal, suitable for ``structure_factors``.

    Two modes:
      * ``link_to_asu=True`` (default): unit-cell tensors are *derived* from
        the ASU handles via the cached ``sym_*`` map every time
        ``unit_cell_view()`` is called.  This keeps the autograd graph fresh
        across optimizer iterations.
      * ``link_to_asu=False``: ``fract``/``occ``/``B_iso``/``U_aniso`` are
        snapshot tensors with no link back to the ASU.

    Use ``unit_cell_view()`` to obtain the live (re-expanded) tensors.
    """
    lattice_params: "torch.Tensor"          # (6,)
    elements: List[str]                     # parallel to N (unit-cell atoms)
    asu_elements: List[str]                 # length n_asu
    space_group_number: int
    n_atoms: int
    link_to_asu: bool
    # Asymmetric-unit handles (always populated; carry the requires_grad flags):
    fract_asu: "torch.Tensor"               # (n_asu, 3)
    occ_asu: "torch.Tensor"                 # (n_asu,)
    B_iso_asu: "torch.Tensor"               # (n_asu,)
    U_aniso_asu: Optional["torch.Tensor"]   # (n_asu, 6) or None
    # Which atoms are anisotropic: an EXPLICIT per-atom flag, built from ``Atom.U_aniso is not None``. The Debye-Waller factor used to decide this from the
    # data (an all-zero row meant "isotropic"), which made it discontinuous at U = 0 (zero gradient for an atom refined from U = 0) and misread a genuine
    # all-zero tensor. ``None`` (tensors built outside ``crystal_to_tensor``) falls back to the zero-row test.
    aniso_mask_asu: Optional["torch.Tensor"] = None          # (n_asu,) bool
    _aniso_mask_snapshot: Optional["torch.Tensor"] = None    # (N,) bool, link_to_asu=False
    # Snapshot unit-cell tensors (only used when link_to_asu=False):
    _fract_snapshot: Optional["torch.Tensor"] = None
    _occ_snapshot: Optional["torch.Tensor"] = None
    _B_iso_snapshot: Optional["torch.Tensor"] = None
    _U_aniso_snapshot: Optional["torch.Tensor"] = None
    # Symmetry expansion map (constant tensors, OK to reuse across iterations):
    sym_R: Optional["torch.Tensor"] = None         # (N, 3, 3)
    sym_t: Optional["torch.Tensor"] = None         # (N, 3)
    sym_atom_idx: Optional["torch.Tensor"] = None  # (N,)
    sym_op_idx: Optional["torch.Tensor"] = None    # (N,)

    def unit_cell_view(self):
        """Return ``(fract, occ, B_iso, U_aniso)`` with a fresh autograd graph
        attached when ``link_to_asu=True``, else the snapshots."""
        if self.link_to_asu:
            fract = _expand_fract(self.fract_asu, self.sym_R, self.sym_t,
                                  self.sym_atom_idx, self.sym_op_idx)
            occ = self.occ_asu[self.sym_atom_idx]
            B_iso = self.B_iso_asu[self.sym_atom_idx]
            U_aniso = (
                _expand_U_aniso(self.U_aniso_asu, self.sym_R,
                                self.sym_atom_idx, self.sym_op_idx)
                if self.U_aniso_asu is not None else None
            )
            return fract, occ, B_iso, U_aniso
        return (
            self._fract_snapshot,
            self._occ_snapshot,
            self._B_iso_snapshot,
            self._U_aniso_snapshot,
        )

    def aniso_mask(self):
        """(N,) bool over the unit-cell atoms: True for anisotropic atoms; ``None`` when the tensor has no mask (legacy construction)."""
        if self.aniso_mask_asu is None:
            return None
        if self.link_to_asu:
            return self.aniso_mask_asu[self.sym_atom_idx]
        return self._aniso_mask_snapshot

    # ----------------------------- backward-compatible attribute shims

    @property
    def fract(self):  # pragma: no cover - convenience
        return self.unit_cell_view()[0]

    @property
    def occ(self):  # pragma: no cover - convenience
        return self.unit_cell_view()[1]

    @property
    def B_iso(self):  # pragma: no cover - convenience
        return self.unit_cell_view()[2]

    @property
    def U_aniso(self):  # pragma: no cover - convenience
        return self.unit_cell_view()[3]


def crystal_to_tensor(
    crystal: Crystal,
    *,
    device=None,
    dtype=None,
    requires_grad: dict | None = None,
    link_to_asu: bool = True,
    dedupe_tol: float = 1e-4,
) -> CrystalTensor:
    """Pack a Crystal into a ``CrystalTensor``.

    ``requires_grad`` keys: 'fract', 'occ', 'B_iso', 'U_aniso', 'lattice'.

    When ``link_to_asu=True`` (default), the unit-cell tensors are *derived*
    from the asymmetric-unit tensors via the cached symmetry-expansion map,
    so ``backward()`` propagates gradients to the ASU handles.
    """
    import torch

    if dtype is None:
        dtype = torch.float64
    rg = requires_grad or {}

    L = crystal.lattice
    lattice_params = torch.tensor(
        [L.a, L.b, L.c, L.alpha, L.beta, L.gamma], dtype=dtype, device=device
    )
    if rg.get("lattice", False):
        lattice_params = lattice_params.clone().detach().requires_grad_(True)

    # Expansion map: build (sym_R, sym_t, asu_idx, op_idx) once, in numpy.
    expansion = _build_expansion(crystal, dedupe_tol=dedupe_tol)
    sym_R = torch.tensor(expansion["R"], dtype=dtype, device=device)         # (M, 3, 3)
    sym_t = torch.tensor(expansion["t"], dtype=dtype, device=device)         # (M, 3)
    asu_idx = torch.tensor(expansion["asu_idx"], dtype=torch.long, device=device)
    op_idx = torch.tensor(expansion["op_idx"], dtype=torch.long, device=device)

    # ASU handles
    asu_fract = torch.tensor(
        np.array([list(a.fract) for a in crystal.atoms], dtype=np.float64),
        dtype=dtype, device=device,
    )
    asu_occ = torch.tensor(
        np.array([a.occupancy for a in crystal.atoms], dtype=np.float64),
        dtype=dtype, device=device,
    )
    asu_B = torch.tensor(
        np.array([a.B_iso for a in crystal.atoms], dtype=np.float64),
        dtype=dtype, device=device,
    )
    has_aniso = any(a.U_aniso is not None for a in crystal.atoms)
    if has_aniso:
        asu_U = torch.tensor(
            np.array([list(a.U_aniso) if a.U_aniso else [0.0] * 6
                      for a in crystal.atoms], dtype=np.float64),
            dtype=dtype, device=device,
        )
    else:
        asu_U = None
    asu_mask = torch.tensor([a.U_aniso is not None for a in crystal.atoms], dtype=torch.bool, device=device) if has_aniso else None

    if rg.get("fract", False):
        asu_fract = asu_fract.clone().detach().requires_grad_(True)
    if rg.get("occ", False):
        asu_occ = asu_occ.clone().detach().requires_grad_(True)
    if rg.get("B_iso", False):
        asu_B = asu_B.clone().detach().requires_grad_(True)
    if asu_U is not None and rg.get("U_aniso", False):
        asu_U = asu_U.clone().detach().requires_grad_(True)

    elements = [crystal.atoms[i].element for i in expansion["asu_idx"]]
    asu_elements = [a.element for a in crystal.atoms]

    fract_snap = occ_snap = B_snap = U_snap = mask_snap = None
    if not link_to_asu:
        uc_atoms = crystal.unit_cell_atoms(dedupe_tol=dedupe_tol)
        fract_snap = torch.tensor(
            np.array([list(a.fract) for a in uc_atoms], dtype=np.float64),
            dtype=dtype, device=device,
        )
        occ_snap = torch.tensor(
            np.array([a.occupancy for a in uc_atoms], dtype=np.float64),
            dtype=dtype, device=device,
        )
        B_snap = torch.tensor(
            np.array([a.B_iso for a in uc_atoms], dtype=np.float64),
            dtype=dtype, device=device,
        )
        if has_aniso:
            U_snap = torch.tensor(
                np.array([list(a.U_aniso) if a.U_aniso else [0.0] * 6 for a in uc_atoms]),
                dtype=dtype, device=device,
            )
            mask_snap = torch.tensor([a.U_aniso is not None for a in uc_atoms], dtype=torch.bool, device=device)

    return CrystalTensor(
        lattice_params=lattice_params,
        elements=elements,
        asu_elements=asu_elements,
        space_group_number=crystal.space_group.number,
        n_atoms=int(asu_idx.shape[0]),
        link_to_asu=link_to_asu,
        fract_asu=asu_fract,
        occ_asu=asu_occ,
        B_iso_asu=asu_B,
        U_aniso_asu=asu_U,
        aniso_mask_asu=asu_mask,
        _aniso_mask_snapshot=mask_snap,
        _fract_snapshot=fract_snap,
        _occ_snapshot=occ_snap,
        _B_iso_snapshot=B_snap,
        _U_aniso_snapshot=U_snap,
        sym_R=sym_R,
        sym_t=sym_t,
        sym_atom_idx=asu_idx,
        sym_op_idx=op_idx,
    )


# ---------------------------------------------------------- expansion helpers

def _build_expansion(crystal: Crystal, *, dedupe_tol: float) -> dict:
    """For each (asu_atom, symop) pair that survives dedupe, record (R, t, asu_idx, op_idx)."""
    ops = crystal.space_group.operations
    out_R: list[np.ndarray] = []
    out_t: list[np.ndarray] = []
    out_asu_idx: list[int] = []
    out_op_idx: list[int] = []
    seen: dict[tuple[str, tuple[int, int, int]], None] = {}
    inv_tol = 1.0 / max(dedupe_tol, 1e-6)
    for j, atom in enumerate(crystal.atoms):
        x0 = np.array(atom.fract, dtype=float)
        for k, op in enumerate(ops):
            R = np.array(op.R, dtype=float).reshape(3, 3)
            t = np.array(op.t, dtype=float) / float(STBF)
            xnew = (R @ x0 + t) % 1.0
            key_grid = tuple(int(round(v * inv_tol)) % int(round(inv_tol)) for v in xnew)
            key = (atom.element, key_grid)
            if key in seen:
                continue
            seen[key] = None
            out_R.append(R)
            out_t.append(t)
            out_asu_idx.append(j)
            out_op_idx.append(k)
    return {
        "R": np.stack(out_R) if out_R else np.zeros((0, 3, 3)),
        "t": np.stack(out_t) if out_t else np.zeros((0, 3)),
        "asu_idx": np.array(out_asu_idx, dtype=int),
        "op_idx": np.array(out_op_idx, dtype=int),
    }


def _expand_fract(asu_fract, sym_R, sym_t, asu_idx, op_idx):
    """fract_uc[m] = R[m] @ fract_asu[asu_idx[m]] + t[m], wrapped to [0, 1)."""
    import torch
    x0 = asu_fract[asu_idx]                              # (M, 3)
    R = sym_R                                             # already aligned (M,3,3)
    t = sym_t                                             # (M, 3)
    x_new = torch.einsum('mij,mj->mi', R, x0) + t
    x_new = x_new - torch.floor(x_new)
    return x_new


def _expand_U_aniso(asu_U, sym_R, asu_idx, op_idx):
    """U_uc = R · U_asu · Rᵀ in fractional basis, packed back to 6 components."""
    import torch
    U6 = asu_U[asu_idx]                                   # (M, 6)
    U_mat = _u6_to_mat3_torch(U6)                         # (M, 3, 3)
    U_new = torch.einsum('mij,mjk,mlk->mil', sym_R, U_mat, sym_R)  # R U Rᵀ
    return _mat3_to_u6_torch(U_new)


def _u6_to_mat3_torch(u6):
    import torch
    u11, u22, u33, u12, u13, u23 = u6.unbind(dim=-1)
    row0 = torch.stack([u11, u12, u13], dim=-1)
    row1 = torch.stack([u12, u22, u23], dim=-1)
    row2 = torch.stack([u13, u23, u33], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _mat3_to_u6_torch(M):
    import torch
    return torch.stack([
        M[..., 0, 0], M[..., 1, 1], M[..., 2, 2],
        M[..., 0, 1], M[..., 0, 2], M[..., 1, 2],
    ], dim=-1)
