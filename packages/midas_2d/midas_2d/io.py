"""Structure builders and loaders for 2D / few-layer materials.

Phase 1 ships a zinc-blende CdSe builder (the colloidal-nanoplatelet workhorse
in the Schaller/Flanders line).  More builders / CIF and MD-snapshot loaders
arrive in later phases.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup

if TYPE_CHECKING:  # pragma: no cover
    from midas_hkls.crystal_torch import CrystalTensor

__all__ = [
    "cdse_zincblende",
    "build_crystal_tensor",
    "cdse_supercell",
    "supercell_from_crystal",
    "load_xyz_frames",
]

# Zinc-blende CdSe lattice constant (Angstrom).  Bulk value; NPLs sit close.
CDSE_ZB_A = 6.077


def cdse_zincblende(a=CDSE_ZB_A) -> Crystal:
    """Zinc-blende (F-43m, SG 216) CdSe asymmetric unit.

    Cd at (0,0,0), Se at (1/4,1/4,1/4); the space group expands these to the
    full fcc + tetrahedral basis.
    """
    lattice = Lattice(a, a, a, 90.0, 90.0, 90.0)
    sg = SpaceGroup.from_number(216)
    atoms = [
        Atom(element="Cd", fract=(0.0, 0.0, 0.0)),
        Atom(element="Se", fract=(0.25, 0.25, 0.25)),
    ]
    return Crystal(lattice=lattice, space_group=sg, atoms=atoms, name="CdSe-zincblende")


def build_crystal_tensor(crystal=None, *, device=None, dtype=None,
                         requires_grad=None):
    """Pack a ``Crystal`` (default: zinc-blende CdSe) into a ``CrystalTensor``.

    ``requires_grad`` is a dict with keys among
    {'fract','occ','B_iso','U_aniso','lattice'} -- forwarded to
    ``midas_hkls.crystal_torch.crystal_to_tensor``.
    """
    from midas_hkls.crystal_torch import crystal_to_tensor

    if crystal is None:
        crystal = cdse_zincblende()
    return crystal_to_tensor(crystal, device=device, dtype=dtype,
                             requires_grad=requires_grad)


# ---------------------------------------------------------------- supercells

def supercell_from_crystal(crystal, n_cells, *, dtype=None, device=None):
    """Tile a ``Crystal`` into an explicit Cartesian atom list (an MD-ready
    structure).

    Parameters
    ----------
    crystal : Crystal
        Provides the conventional-cell atoms (via ``unit_cell_atoms``) and the
        (orthogonal or general) lattice vectors.
    n_cells : (int, int, int)
        Number of unit cells along a, b, c.

    Returns
    -------
    coords : tensor (M, 3)
        Cartesian coordinates in Angstrom.
    elements : list[str]
        Length-M element symbols (parallel to ``coords``).
    cell : tensor (3, 3)
        The 3x3 matrix whose ROWS are the real-space lattice vectors (A).
    """
    import torch

    if dtype is None:
        dtype = torch.float64
    nx, ny, nz = (int(v) for v in n_cells)

    # Real-space lattice vectors (rows) from the metric: for an orthorhombic /
    # cubic cell this is diag(a, b, c); use the general direct-cell builder.
    cell = _direct_cell_matrix(crystal.lattice, dtype=dtype, device=device)  # (3,3)

    uc_atoms = crystal.unit_cell_atoms()
    base_frac = torch.tensor([list(a.fract) for a in uc_atoms], dtype=dtype, device=device)
    base_elem = [a.element for a in uc_atoms]

    coords = []
    elements: list[str] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = torch.tensor([i, j, k], dtype=dtype, device=device)
                frac = base_frac + shift                       # (nbasis, 3)
                cart = frac @ cell                             # rows are lattice vecs
                coords.append(cart)
                elements.extend(base_elem)
    coords = torch.cat(coords, dim=0)                          # (M, 3)
    return coords, elements, cell


def cdse_supercell(n_cells=(8, 8, 4), *, a=CDSE_ZB_A, dtype=None, device=None):
    """Explicit Cartesian zinc-blende CdSe nanoplatelet supercell.

    Default (8, 8, 4) -> a few-monolayer platelet: large in-plane, 4 cells
    thick out of plane.  Returns ``(coords_A, elements, cell)``.
    """
    return supercell_from_crystal(cdse_zincblende(a=a), n_cells,
                                  dtype=dtype, device=device)


def load_xyz_frames(path, *, dtype=None, device=None):
    """Load an MD trajectory from an (ext)XYZ file into a stack of frames.

    Returns ``(coords, elements)`` where ``coords`` is (F, M, 3) Angstrom for
    F frames of M atoms and ``elements`` is the length-M symbol list (assumed
    constant across frames).  Minimal parser: standard XYZ
    (count / comment / 'El x y z' lines), multi-frame concatenated.
    """
    import numpy as np
    import torch

    if dtype is None:
        dtype = torch.float64

    frames: list[np.ndarray] = []
    elements: list[str] = []
    with open(path, "r") as fh:
        lines = fh.read().splitlines()
    idx = 0
    n_lines = len(lines)
    while idx < n_lines:
        if not lines[idx].strip():
            idx += 1
            continue
        natoms = int(lines[idx].strip())
        idx += 2  # skip count + comment
        elem_this: list[str] = []
        xyz = np.empty((natoms, 3), dtype=np.float64)
        for a in range(natoms):
            parts = lines[idx + a].split()
            elem_this.append(parts[0])
            xyz[a] = [float(parts[1]), float(parts[2]), float(parts[3])]
        idx += natoms
        if not elements:
            elements = elem_this
        frames.append(xyz)
    coords = torch.tensor(np.stack(frames), dtype=dtype, device=device)  # (F, M, 3)
    return coords, elements


def _direct_cell_matrix(lattice, *, dtype, device):
    """3x3 matrix with ROWS = real-space lattice vectors (Angstrom), built from
    the standard crystallographic convention (a along x, b in xy-plane)."""
    import math

    import torch

    a, b, c = lattice.a, lattice.b, lattice.c
    al = math.radians(lattice.alpha)
    be = math.radians(lattice.beta)
    ga = math.radians(lattice.gamma)
    ca, cb, cg, sg = math.cos(al), math.cos(be), math.cos(ga), math.sin(ga)
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    M = torch.tensor([
        [a, 0.0, 0.0],
        [b * cg, b * sg, 0.0],
        [cx, cy, cz],
    ], dtype=dtype, device=device)
    return M
