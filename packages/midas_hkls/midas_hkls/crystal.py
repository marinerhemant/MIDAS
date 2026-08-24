"""Crystal / Atom data types.

A ``Crystal`` is the asymmetric-unit description (atoms + lattice + space
group).  ``unit_cell_atoms()`` expands by symmetry + centering and dedupes
special positions to produce the per-cell atom list consumed by the
structure-factor kernel.

ADP convention: Å² (Uᵢⱼ entries).  B = 8π²U.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .lattice import Lattice
from .space_group import SpaceGroup
from .tables import STBF


__all__ = ["Atom", "Crystal", "B_to_U", "U_to_B", "parse_phase_atoms"]


_8PI2 = 8.0 * (np.pi ** 2)


def B_to_U(B: float) -> float:
    """Convert a Debye-Waller B factor (Å²) to mean-square displacement U (Å²)."""
    return float(B) / _8PI2


def U_to_B(U: float) -> float:
    """Convert U (Å²) to B (Å²)."""
    return float(U) * _8PI2


@dataclass(frozen=True)
class Atom:
    """One site in the asymmetric unit (or, after expansion, in the unit cell).

    Coordinates are *fractional* w.r.t. the lattice vectors.  Anisotropic ADPs
    are stored as the 6 unique components U11, U22, U33, U12, U13, U23 (Å²,
    fractional convention as in CIF).  When ``U_aniso`` is None, ``B_iso`` is
    used.
    """
    element: str                                  # e.g. "Fe", "O", "Si"
    fract: Tuple[float, float, float]             # asymmetric-unit fractional coords
    occupancy: float = 1.0
    B_iso: float = 0.0                            # Å²
    U_aniso: Optional[Tuple[float, float, float, float, float, float]] = None
    label: str = ""                               # CIF site label (optional)

    def __post_init__(self) -> None:
        if len(self.fract) != 3:
            raise ValueError("fract must have 3 components")
        if not (0.0 <= self.occupancy <= 1.0 + 1e-9):
            raise ValueError(f"occupancy out of [0, 1]: {self.occupancy}")
        if self.U_aniso is not None and len(self.U_aniso) != 6:
            raise ValueError("U_aniso must have 6 components (U11,U22,U33,U12,U13,U23)")

    def U_iso_equivalent(self) -> float:
        """If anisotropic, return the isotropic-equivalent U = (U11+U22+U33)/3."""
        if self.U_aniso is None:
            return B_to_U(self.B_iso)
        return (self.U_aniso[0] + self.U_aniso[1] + self.U_aniso[2]) / 3.0


def parse_phase_atoms(rows):
    """``PhaseAtom`` parameter lines -> ``list[Atom]``, or None if there are none.

    One shared parser for every pipeline. The *collection* of the lines stays
    with each pipeline (NF, FF and PF read parameter files differently), but
    the field order and the defaults must not: a basis parsed two ways is a
    structure factor computed two ways.

    Line format::

        PhaseAtom <element> <x> <y> <z> [occupancy] [B_iso]

    Fractional coordinates; ``occupancy`` defaults to 1.0 and ``B_iso`` to 0.0
    (Å²). Returning None rather than an empty list when nothing is declared is
    deliberate — it is what keeps ``hkls.csv`` byte-identical to its historical
    form, since the F2 column is only appended when a basis exists.

    A short line raises rather than defaulting: silently discarding a
    malformed row would put every atom at the origin, which produces a
    perfectly plausible F2 column for the wrong structure.
    """
    if not rows:
        return None
    atoms = []
    for r in rows:
        toks = str(r).split()
        if len(toks) < 4:
            raise ValueError(
                f"PhaseAtom needs '<element> <x> <y> <z> [occ] [B_iso]', "
                f"got {r!r}"
            )
        atoms.append(Atom(
            str(toks[0]),
            (float(toks[1]), float(toks[2]), float(toks[3])),
            occupancy=float(toks[4]) if len(toks) > 4 else 1.0,
            B_iso=float(toks[5]) if len(toks) > 5 else 0.0,
        ))
    return atoms


@dataclass
class Crystal:
    """Crystal structure (asymmetric unit) + symmetry."""
    lattice: Lattice
    space_group: SpaceGroup
    atoms: List[Atom] = field(default_factory=list)
    name: str = ""

    # ------------------------------------------------------------- expansion

    def unit_cell_atoms(self, *, dedupe_tol: float = 1e-4) -> List[Atom]:
        """Apply space-group ops + lattice centering, dedupe special positions.

        Returns a flat list of atoms covering the full unit cell.  Each entry's
        occupancy is the original asymmetric-unit value (NOT scaled — the kernel
        handles multiplicity through the explicit atom list).  Sites that
        coincide under symmetry (within ``dedupe_tol`` fractional distance) are
        kept once, with anisotropic ADPs averaged.
        """
        out: list[Atom] = []
        for atom in self.atoms:
            x0 = np.array(atom.fract, dtype=float)
            U_aniso = np.array(atom.U_aniso, dtype=float) if atom.U_aniso else None
            for op in self.space_group.operations:
                R = np.array(op.R, dtype=float).reshape(3, 3)
                t = np.array(op.t, dtype=float) / float(STBF)
                xnew = (R @ x0 + t) % 1.0
                # transform anisotropic ADPs: U' = R · U · Rᵀ in fractional basis
                Unew = (R @ _u6_to_mat(U_aniso) @ R.T) if U_aniso is not None else None
                cand = Atom(
                    element=atom.element,
                    fract=tuple(float(v) for v in xnew),
                    occupancy=atom.occupancy,
                    B_iso=atom.B_iso,
                    U_aniso=tuple(_mat_to_u6(Unew)) if Unew is not None else None,
                    label=atom.label,
                )
                if not _is_duplicate(cand, out, dedupe_tol):
                    out.append(cand)
        return out

    # --------------------------------------------------------------- helpers

    def add_atom(self, atom: Atom) -> "Crystal":
        return replace(self, atoms=self.atoms + [atom])

    @property
    def n_atoms_asu(self) -> int:
        return len(self.atoms)

    def to_torch(
        self,
        *,
        device: "object | None" = None,
        dtype: "object | None" = None,
        requires_grad: dict | None = None,
    ) -> "CrystalTensor":
        """Pack into torch tensors; defers import so torch stays optional.

        ``requires_grad`` is a dict of param name → bool; recognized keys are
        ``"fract"``, ``"occ"``, ``"B_iso"``, ``"U_aniso"``, ``"lattice"``.
        """
        from .crystal_torch import CrystalTensor, crystal_to_tensor
        return crystal_to_tensor(self, device=device, dtype=dtype, requires_grad=requires_grad)


# ------------------------------------------------------------ small helpers

def _u6_to_mat(u6: Sequence[float] | None) -> np.ndarray:
    if u6 is None:
        return np.zeros((3, 3))
    u11, u22, u33, u12, u13, u23 = u6
    return np.array([[u11, u12, u13],
                     [u12, u22, u23],
                     [u13, u23, u33]], dtype=float)


def _mat_to_u6(m: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    return (float(m[0, 0]), float(m[1, 1]), float(m[2, 2]),
            float(m[0, 1]), float(m[0, 2]), float(m[1, 2]))


def _is_duplicate(cand: Atom, existing: Iterable[Atom], tol: float) -> bool:
    for a in existing:
        if a.element != cand.element:
            continue
        d = np.array(a.fract) - np.array(cand.fract)
        # wrap to [-0.5, 0.5)
        d -= np.round(d)
        if float(np.linalg.norm(d)) < tol:
            return True
    return False
