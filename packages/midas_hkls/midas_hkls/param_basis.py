"""Read an atom basis out of a MIDAS parameter file.

One reader for FF and PF. NF collects its own lines (its parameter handling
predates this and has its own multi-line machinery) but shares the *parsing*
through :func:`midas_hkls.parse_phase_atoms`, so there is still exactly one
definition of what a ``PhaseAtom`` line means.

Recognised keys::

    PhaseAtom <element> <x> <y> <z> [occupancy] [B_iso]   (repeatable)
    PhaseCIF  <path>
    DropForbiddenReflections <0|1>
    ForbiddenF2Threshold <float>

Returns kwargs for :func:`midas_hkls.zarr_compat.generate_hkls_from_zarr`, so
a caller that declares no basis gets ``{}`` and the historical 11-column
``hkls.csv`` byte-for-byte.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

__all__ = ["read_phase_basis"]


def _iter_kv(path: Path):
    """Yield ``(key, rest)`` for every non-comment line.

    MIDAS parameter files are ``Key value...`` with ``#`` comments; some
    writers append a trailing ``;`` (paramstest.txt does), which is stripped
    so the same reader works on both flavours.
    """
    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip().rstrip(";").strip()
            if not line:
                continue
            parts = line.split(None, 1)
            yield parts[0], (parts[1].strip() if len(parts) > 1 else "")


def read_phase_basis(param_file: Union[str, Path]) -> Dict[str, Any]:
    """Collect the atom-basis keys from ``param_file``.

    Empty dict when nothing is declared — which is what keeps ``hkls.csv``
    unchanged for every existing run.

    ``PhaseAtom`` and ``PhaseCIF`` are mutually exclusive and both being
    present raises. Preferring one silently would let a stale ``PhaseAtom``
    block shadow a CIF, with nothing in the output saying which structure
    produced the numbers.
    """
    p = Path(param_file)
    if not p.is_file():
        return {}

    atom_lines: list[str] = []
    cif_path: str | None = None
    drop = 0
    thr = 1e-6

    for key, rest in _iter_kv(p):
        if key == "PhaseAtom":
            atom_lines.append(rest)
        elif key == "PhaseCIF":
            cif_path = rest
        elif key == "DropForbiddenReflections":
            try:
                drop = int(float(rest.split()[0]))
            except (ValueError, IndexError):
                drop = 0
        elif key == "ForbiddenF2Threshold":
            try:
                thr = float(rest.split()[0])
            except (ValueError, IndexError):
                pass

    if atom_lines and cif_path:
        raise ValueError(
            f"{p}: both PhaseAtom and PhaseCIF are declared. They are "
            "mutually exclusive — remove one, so the output says "
            "unambiguously which structure produced |F|^2."
        )

    out: Dict[str, Any] = {}
    if atom_lines:
        from .crystal import parse_phase_atoms
        out["atoms"] = parse_phase_atoms(atom_lines)
    elif cif_path:
        out["cif_path"] = cif_path
    else:
        # No basis: DropForbiddenReflections cannot mean anything, and
        # silently ignoring it would hide a parameter file that asked for
        # filtering and did not get it.
        if drop:
            raise ValueError(
                f"{p}: DropForbiddenReflections is set but no PhaseAtom or "
                "PhaseCIF is declared, so no |F|^2 exists to call forbidden."
            )
        return {}

    if drop:
        out["drop_forbidden"] = True
        out["forbidden_f2_threshold"] = thr
    return out
