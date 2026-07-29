"""Lattice / crystal helpers for midas_defect.

This module is intentionally thin and **phase-agnostic**. The math (metric
tensor, d-spacing, 2θ, structure factor, HKL enumeration) lives in midas_hkls.
What we add here is:

  * `fcc_cu_crystal(a)`        — demk-default FCC Cu(-rich SS) Crystal (SG 225)
  * `cual2_crystal(a, c)`      — θ-Al₂Cu (I4/mcm); first-class, retained for a
                                  future genuine-CuAl₂ sample with superstructure
  * `bragg_shells(crystal,...)`— unique |q| Bragg shells aggregated by symmetry
                                  family (works for any `Crystal`); used by the
                                  q-space overlay viz and the rod hkl-crossing
                                  lookup. `tetragonal_shells` is a back-compat
                                  alias.

Differentiability
-----------------
The torch d-spacing path is preserved by reusing midas_hkls.lattice_torch.
Both `(a, c)` flow through as differentiable tensors when supplied that way;
the discrete HKL enumeration is wrapped in `_discrete_*` helpers off the
gradient path (which is fine — the integer hkl set is fixed once the cell
is committed; what flows is |q|(hkl) = 1 / d(hkl, a, c)).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np

from midas_hkls.crystal import Atom, Crystal
from midas_hkls.lattice import Lattice
from midas_hkls.space_group import SpaceGroup


# ---------------------------------------------------------------------------
# Demk default cell — FCC Cu(-rich solid solution), Fm-3m (#225)
# ---------------------------------------------------------------------------

#: demk FCC cell from the powder line-out (d(111)=2.099 Å). ~0.6 % expanded over
#: pure Cu (3.6149 Å) — Al in solid solution.
FCC_CU_A_DEFAULT: float = 3.6356


def fcc_cu_crystal(a: float = FCC_CU_A_DEFAULT) -> Crystal:
    """Return the FCC Cu(-rich solid solution) `Crystal` (Fm-3m, #225).

    The demk default phase. A single Cu site at (0,0,0); the face-centring is
    supplied by space group 225, so the structure factor reproduces the FCC
    all-even / all-odd selection rule used by `defect_tests`.

    Parameters
    ----------
    a
        Cubic lattice constant in Å. Default 3.6356 (demk line-out).
    """
    lattice = Lattice(a=a, b=a, c=a, alpha=90.0, beta=90.0, gamma=90.0)
    sg = SpaceGroup.from_number(225)
    atoms = [Atom(element="Cu", fract=(0.0, 0.0, 0.0), occupancy=1.0,
                  label="Cu1")]
    return Crystal(lattice=lattice, space_group=sg, atoms=atoms)


# ---------------------------------------------------------------------------
# θ-Al₂Cu (I4/mcm, #140) — retained, first-class (future genuine-CuAl₂ data)
# ---------------------------------------------------------------------------

#: Canonical published cell for θ-Al₂Cu (Meetsma et al.; ICSD #150596).
CUAL2_A_DEFAULT: float = 6.066
CUAL2_C_DEFAULT: float = 4.874

# Wyckoff positions in I4/mcm:
#   Cu  4a  (0, 0, 1/4)
#   Al  8h  (x, x+1/2, 0)  with x ≈ 0.1583
CUAL2_AL_X_8H: float = 0.1583


def cual2_crystal(a: float = CUAL2_A_DEFAULT,
                  c: float = CUAL2_C_DEFAULT,
                  *,
                  al_x_8h: float = CUAL2_AL_X_8H) -> Crystal:
    """Return the θ-Al₂Cu `Crystal` (I4/mcm) with the supplied cell.

    Parameters
    ----------
    a, c
        Tetragonal lattice constants in Å. Defaults match the canonical
        published cell (Meetsma et al., Acta Cryst. C45 (1989)).
    al_x_8h
        Free x for the Al 8h site. Default 0.1583.
    """
    lattice = Lattice(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=90.0)
    sg = SpaceGroup.from_number(140)
    atoms = [
        Atom(element="Cu", fract=(0.0, 0.0, 0.25), occupancy=1.0,
             label="Cu1"),
        Atom(element="Al", fract=(al_x_8h, al_x_8h + 0.5, 0.0),
             occupancy=1.0, label="Al1"),
    ]
    return Crystal(lattice=lattice, space_group=sg, atoms=atoms)


# ---------------------------------------------------------------------------
# Shell aggregation — |q|-unique with symmetry-family representatives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shell:
    """A unique |q| value plus the (hkl) families that contribute to it."""
    q_inv_A: float                       # |q| in 1/Å (= 2π/d)
    d_A: float                           # d-spacing in Å
    hkls: Tuple[Tuple[int, int, int], ...]


def bragg_shells(
    crystal: Crystal,
    *,
    q_max_inv_A: float = 10.0,
    q_tol_inv_A: float = 1e-4,
) -> List[Shell]:
    """All allowed Bragg shells of any `Crystal` up to `q_max_inv_A`.

    Aggregates symmetry-equivalent HKLs (same |q|) into a single `Shell`
    entry. Uses `midas_hkls.generate_hkls` for the underlying enumeration so
    the systematic absences and ASU representatives come from one place.
    Phase-agnostic: works for FCC (`fcc_cu_crystal()`), tetragonal CuAl₂
    (`cual2_crystal()`), or any other cell.

    Parameters
    ----------
    crystal
        `midas_hkls.Crystal` (e.g. `fcc_cu_crystal()` for the demk default).
    q_max_inv_A
        Upper bound on |q| in 1/Å. Reflections with |q| > q_max are dropped.
    q_tol_inv_A
        Two reflections within this |q| tolerance are merged into one shell.

    Returns
    -------
    list of `Shell`, sorted by increasing |q|.
    """
    from midas_hkls.hkl_gen import generate_hkls

    d_min = 2.0 * math.pi / q_max_inv_A
    refs = generate_hkls(crystal.space_group, crystal.lattice, d_min=d_min)
    by_q: dict[float, list[Tuple[int, int, int]]] = defaultdict(list)
    for r in refs:
        q = 2.0 * math.pi / r.d_spacing
        if q > q_max_inv_A:
            continue
        # round to a multiple of q_tol_inv_A so float-equal reflections group
        key = round(q / q_tol_inv_A) * q_tol_inv_A
        by_q[key].append((r.h, r.k, r.l))
    out: List[Shell] = []
    for q in sorted(by_q):
        hkls = tuple(sorted(set(by_q[q])))
        out.append(Shell(q_inv_A=float(q),
                         d_A=float(2.0 * math.pi / q),
                         hkls=hkls))
    return out


#: Back-compat alias — `bragg_shells` is phase-agnostic and supersedes this name.
tetragonal_shells = bragg_shells


# ---------------------------------------------------------------------------
# Differentiable |q|(hkl, a, c) — torch path
# ---------------------------------------------------------------------------

def q_inv_of_hkl_torch(
    hkl: "Sequence[Sequence[int]] | np.ndarray",
    a: "float | object",
    c: "float | object",
    *,
    device: "object | None" = None,
    dtype: "object | None" = None,
):
    """`|q|` in 1/Å for a tetragonal cell. Differentiable through `(a, c)`.

    Inputs may be numpy arrays, lists, or torch tensors. Returns a torch
    tensor of shape (..., ) matching the leading dims of `hkl`.
    """
    import torch
    from midas_hkls.lattice_torch import d_spacing

    if isinstance(a, torch.Tensor) and dtype is None:
        dtype = a.dtype
    if isinstance(a, torch.Tensor) and device is None:
        device = a.device
    if dtype is None:
        dtype = torch.float64
    if device is None:
        device = torch.device("cpu")
    a_t = torch.as_tensor(a, dtype=dtype, device=device)
    c_t = torch.as_tensor(c, dtype=dtype, device=device)
    # tetragonal lattice params with α=β=γ=90 (cubic is the c=a special case)
    angle90 = torch.full((), 90.0, dtype=dtype, device=device)
    lat = torch.stack([a_t, a_t, c_t, angle90, angle90, angle90], dim=-1)
    d = d_spacing(hkl, lat, dtype=dtype, device=device)
    return 2.0 * math.pi / d


def q_inv_of_hkl_cubic_torch(hkl, a, *, device=None, dtype=None):
    """`|q|` in 1/Å for a cubic cell (FCC default path). Differentiable through `a`.

    Thin wrapper over `q_inv_of_hkl_torch` with c = a.
    """
    return q_inv_of_hkl_torch(hkl, a, a, device=device, dtype=dtype)
