"""Minimal CIF file reader — hydrates a MIDAS Crystal from a .cif.

The full CIF spec is large; this reader covers the subset needed to
build a ``midas_hkls.Crystal`` for PDF refinement:

  * cell parameters (``_cell_length_a`` etc.)
  * space group number (``_symmetry_int_tables_number`` or aliases)
  * atomic positions via the standard loop
    (``_atom_site_type_symbol``, ``_atom_site_fract_x/y/z``,
    optional ``_atom_site_occupancy``, ``_atom_site_B_iso_or_equiv``)

For richer CIF ingestion (partial occupancies with aliases, multiple
loops, symmetry operators), delegate to :mod:`pymatgen` or
:mod:`gemmi`; we deliberately stay dependency-light here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

_ESD_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\(\d+\))?$")


def _strip_esd(value: str) -> str:
    """Strip the parenthesised uncertainty from a CIF numeric string.

    ``"3.524(1)"`` → ``"3.524"``.
    """
    m = _ESD_RE.match(value.strip())
    return m.group(1) if m else value.strip()


def _try_float(value: str) -> Union[float, str]:
    try:
        return float(_strip_esd(value))
    except ValueError:
        return value


def _unquote(s: str) -> str:
    """Strip matching leading/trailing single or double quotes."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def _tokenise_line(line: str) -> List[str]:
    """Simple tokeniser: split on whitespace, honouring '..' / ".." quotes."""
    tokens: List[str] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
            continue
        if c in "'\"":
            quote = c
            j = i + 1
            while j < n and line[j] != quote:
                j += 1
            tokens.append(line[i + 1: j])
            i = j + 1
        else:
            j = i
            while j < n and line[j] not in " \t":
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


@dataclass
class CIFData:
    """Structured contents of a single CIF ``data_...`` block."""
    keys: Dict[str, str]                          # simple key → value strings
    loops: List[Tuple[List[str], List[List[str]]]]  # (keys, rows-of-values)

    def get_num(self, key: str, default: Optional[float] = None) -> Optional[float]:
        raw = self.keys.get(key)
        if raw is None:
            return default
        v = _try_float(raw)
        return float(v) if isinstance(v, float) else default


def parse_cif(path: Union[str, Path]) -> CIFData:
    """Parse ``path`` and return one :class:`CIFData` (the first data block).

    Follows the CIF 1.1 core: comments start with ``#``; blocks start with
    ``data_``. Multi-line semicolon-delimited text fields are NOT
    supported (rare in PDF-facing CIFs).
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    # Strip comments and blank lines
    stripped: List[str] = []
    for ln in lines:
        code = ln.split("#", 1)[0].rstrip()
        if code.strip():
            stripped.append(code)

    keys: Dict[str, str] = {}
    loops: List[Tuple[List[str], List[List[str]]]] = []
    i = 0
    while i < len(stripped):
        line = stripped[i].strip()
        if line.startswith("data_"):
            i += 1
            continue
        if line.lower().startswith("loop_"):
            # collect loop keys
            loop_keys: List[str] = []
            i += 1
            while i < len(stripped) and stripped[i].strip().startswith("_"):
                loop_keys.append(stripped[i].strip().lower())
                i += 1
            # collect data rows until next key/loop/EOF
            rows: List[List[str]] = []
            while i < len(stripped):
                row_line = stripped[i].strip()
                if row_line.startswith("_") or row_line.lower().startswith("loop_"):
                    break
                row_tokens = _tokenise_line(row_line)
                if row_tokens:
                    rows.append(row_tokens)
                i += 1
            # Some CIFs put multiple rows on one line; merge into groups of len(loop_keys)
            flat: List[str] = [tok for row in rows for tok in row]
            k = len(loop_keys)
            if k > 0 and len(flat) % k == 0:
                rows = [flat[j : j + k] for j in range(0, len(flat), k)]
            loops.append((loop_keys, rows))
        elif line.startswith("_"):
            # simple key / value pair — may span next line
            parts = _tokenise_line(line)
            key = parts[0].lower()
            if len(parts) > 1:
                keys[key] = " ".join(parts[1:])
                i += 1
            else:
                # value on next line
                i += 1
                if i < len(stripped):
                    keys[key] = _unquote(stripped[i].strip())
                    i += 1
        else:
            i += 1

    return CIFData(keys=keys, loops=loops)


def _first_loop_with(cif: CIFData, must_have: List[str]):
    for loop_keys, rows in cif.loops:
        if all(k in loop_keys for k in must_have):
            return loop_keys, rows
    return None


def read_cif_to_crystal(path: Union[str, Path]):
    """Build a ``midas_hkls.Crystal`` from a CIF file.

    Required CIF keys: ``_cell_length_a/b/c``, ``_cell_angle_alpha/beta/gamma``,
    a space-group number under one of the standard aliases, and an atom-site
    loop with ``_atom_site_fract_x/y/z``.  Element inferred from
    ``_atom_site_type_symbol`` (preferred) or ``_atom_site_label`` (stripped
    of digits).

    Returns the plain ``Crystal`` (not the torch version). Call
    ``.to_torch()`` on it if you want the differentiable form.
    """
    from midas_hkls import Crystal, Atom, Lattice, SpaceGroup

    cif = parse_cif(path)

    a = cif.get_num("_cell_length_a")
    b = cif.get_num("_cell_length_b")
    c = cif.get_num("_cell_length_c")
    alpha = cif.get_num("_cell_angle_alpha")
    beta  = cif.get_num("_cell_angle_beta")
    gamma = cif.get_num("_cell_angle_gamma")
    for name, val in [("_cell_length_a", a), ("_cell_length_b", b),
                       ("_cell_length_c", c), ("_cell_angle_alpha", alpha),
                       ("_cell_angle_beta", beta), ("_cell_angle_gamma", gamma)]:
        if val is None:
            raise ValueError(f"CIF missing required key {name!r}")

    # Space group: try IT number aliases first, then H-M / Hall via midas-hkls
    sg_obj = None
    # 1. Numeric IT number aliases
    for k in ("_symmetry_int_tables_number", "_space_group_it_number"):
        if k in cif.keys:
            try:
                sg_obj = SpaceGroup.from_number(int(_strip_esd(cif.keys[k])))
                break
            except (ValueError, KeyError):
                pass
    # 2. Hermann-Mauguin symbol aliases (via midas-hkls resolver)
    if sg_obj is None:
        for k in ("_symmetry_space_group_name_h-m",
                  "_space_group_name_h-m_alt",
                  "_space_group_name_h-m"):
            if k in cif.keys:
                hm = _unquote(cif.keys[k]).strip()
                # Fallback ladder: (a) verbatim → (b) shorthand normaliser →
                # (c) curated alias table
                for candidate in (hm,
                                    _normalise_hm_shorthand(hm),
                                    _HM_ALIASES.get(hm, hm),
                                    _HM_ALIASES.get(
                                        _normalise_hm_shorthand(hm), hm)):
                    try:
                        sg_obj = SpaceGroup.from_hm(candidate)
                        break
                    except (ValueError, KeyError):
                        pass
                if sg_obj is not None:
                    break
    # 3. Hall symbol
    if sg_obj is None and "_space_group_name_hall" in cif.keys:
        try:
            sg_obj = SpaceGroup.from_hall(_unquote(cif.keys["_space_group_name_hall"]))
        except (ValueError, KeyError):
            pass
    if sg_obj is None:
        raise ValueError(
            "CIF has no resolvable space-group identifier "
            "(_symmetry_Int_Tables_number, _symmetry_space_group_name_H-M, "
            "or _space_group_name_Hall).")

    # Atom-site loop
    loop = _first_loop_with(cif, ["_atom_site_fract_x",
                                    "_atom_site_fract_y",
                                    "_atom_site_fract_z"])
    if loop is None:
        raise ValueError("CIF missing _atom_site_fract_x/y/z loop")
    loop_keys, rows = loop
    col = {k: i for i, k in enumerate(loop_keys)}

    def _elem_from_row(row: List[str]) -> str:
        if "_atom_site_type_symbol" in col:
            sym = row[col["_atom_site_type_symbol"]]
            # Some CIFs use "Ni2+" style — strip trailing charge
            return re.sub(r"[0-9+\-]+$", "", sym) or sym
        elif "_atom_site_label" in col:
            lbl = row[col["_atom_site_label"]]
            return re.sub(r"[0-9]+$", "", lbl)
        else:
            raise ValueError("CIF loop lacks _atom_site_type_symbol / _atom_site_label")

    atoms = []
    for row in rows:
        elem = _elem_from_row(row)
        fx = float(_strip_esd(row[col["_atom_site_fract_x"]]))
        fy = float(_strip_esd(row[col["_atom_site_fract_y"]]))
        fz = float(_strip_esd(row[col["_atom_site_fract_z"]]))
        # Optional per-atom occupancy + B_iso — fields already present on Atom
        occ_key = "_atom_site_occupancy"
        biso_key = "_atom_site_b_iso_or_equiv"
        occupancy = (float(_strip_esd(row[col[occ_key]]))
                      if occ_key in col else 1.0)
        B_iso = (float(_strip_esd(row[col[biso_key]]))
                  if biso_key in col else 0.0)
        atoms.append(Atom(element=elem, fract=(fx, fy, fz),
                           occupancy=occupancy, B_iso=B_iso))

    crystal = Crystal(
        lattice=Lattice(a, b, c, alpha, beta, gamma),
        space_group=sg_obj,
        atoms=atoms,
        name=Path(path).stem,
    )
    return crystal


def _normalise_hm_shorthand(hm: str) -> str:
    """Insert missing bars in H-M shorthand.

    Common CIFs (particularly older ones) omit the overbar on
    centrosymmetric elements of the point group. This normaliser handles
    the two most common shorthand patterns:

      1. **Cubic bar-omitting** for centrosymmetric cubic groups where
         the ``3`` in the middle position denotes ``-3``.  Both compact
         (``Fm3m``) and space-separated (``F m 3 m``) inputs are
         handled.  Examples:

            Fm3m → Fm-3m       Pm3n → Pm-3n
            Ia3d → Ia-3d       Fd3m → Fd-3m
            F m 3 m → F m -3 m   Im3m → Im-3m

      2. **Whitespace collapse** for CIFs that fully space-separate the
         symbol; we normalise multiple spaces to single spaces before
         trying midas-hkls.

    ``SpaceGroup.from_hm`` is still the source of truth --- this
    function only massages the input for the fallback pass.
    """
    import re
    # First: collapse any repeated whitespace
    s = re.sub(r"\s+", " ", hm.strip())

    # Cubic bar-omit patterns (compact form)
    # letter '3' letter  where the first letter is one of the cubic centrings
    # (m / n / d) or the point-group symbol letter after the centring letter.
    # Match both compact and space-separated variants.
    #
    # Compact: 'Fm3m' → 'Fm-3m'; 'Fd3' → 'Fd-3'
    s = re.sub(r"([abmnd])(3)([mnhrcd])", r"\1-\2\3", s)     # centred at end
    s = re.sub(r"([abmnd])(3)(?![-\dmnhrcd])", r"\1-\2", s)  # trailing 3
    # Space-separated: 'F m 3 m' → 'F m -3 m'
    s = re.sub(r"([abmnd])\s+3\s+([mnhrcd])", r"\1 -3 \2", s)
    s = re.sub(r"([abmnd])\s+3\s*$", r"\1 -3", s)
    return s


# Curated fallback: some CIFs use variants that neither the string nor the
# normalised string resolve. These map to the canonical H-M via aliases.
_HM_ALIASES: Dict[str, str] = {
    # Trigonal / rhombohedral extended settings sometimes written with H
    "R -3 m H": "R-3m",
    "R -3 c H": "R-3c",
    "R 3 m H": "R3m",
    "R 3 c H": "R3c",
    # Cubic Wyckoff-only shorthand seen in older CIFs
    "F d -3 m S": "Fd-3m",
    "F d -3 m Z": "Fd-3m",
}


def write_crystal_to_cif(crystal, path: Union[str, Path], *,
                          name: Optional[str] = None) -> Path:
    """Serialise a ``midas_hkls.Crystal`` to a CIF file.

    Writes the asymmetric-unit atom list — reading it back with
    :func:`read_cif_to_crystal` and expanding via
    ``Crystal.unit_cell_atoms()`` recovers the full cell.

    Includes ``_atom_site_occupancy`` and ``_atom_site_B_iso_or_equiv``
    for every atom so partial-occupancy structures round-trip cleanly.
    """
    path = Path(path)
    block = name or crystal.name or path.stem
    lat = crystal.lattice
    sg = crystal.space_group
    lines: List[str] = []
    lines.append(f"data_{block}")
    lines.append(f"_cell_length_a       {lat.a:.6f}")
    lines.append(f"_cell_length_b       {lat.b:.6f}")
    lines.append(f"_cell_length_c       {lat.c:.6f}")
    lines.append(f"_cell_angle_alpha    {lat.alpha:.6f}")
    lines.append(f"_cell_angle_beta     {lat.beta:.6f}")
    lines.append(f"_cell_angle_gamma    {lat.gamma:.6f}")
    lines.append(f"_symmetry_Int_Tables_number   {int(sg.number)}")
    hm = getattr(sg, "hm_symbol", None)
    if hm:
        lines.append(f"_symmetry_space_group_name_H-M   '{hm}'")
    lines.append("")
    lines.append("loop_")
    lines.append("_atom_site_label")
    lines.append("_atom_site_type_symbol")
    lines.append("_atom_site_fract_x")
    lines.append("_atom_site_fract_y")
    lines.append("_atom_site_fract_z")
    lines.append("_atom_site_occupancy")
    lines.append("_atom_site_B_iso_or_equiv")
    counters: Dict[str, int] = {}
    for atom in crystal.atoms:
        counters[atom.element] = counters.get(atom.element, 0) + 1
        label = atom.label or f"{atom.element}{counters[atom.element]}"
        fx, fy, fz = atom.fract
        lines.append(f"{label:8s}  {atom.element:2s}  "
                     f"{float(fx):.6f}  {float(fy):.6f}  {float(fz):.6f}  "
                     f"{float(atom.occupancy):.4f}  "
                     f"{float(atom.B_iso):.4f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_supercell_to_cif(supercell, path: Union[str, Path], *,
                            name: Optional[str] = None) -> Path:
    """Serialise a :class:`midas_pdf.rmc.Supercell` to a CIF as space
    group P1 (every atom explicit — no symmetry equivalents).

    This is the canonical export path for RMC-refined configurations
    (which lose the parent crystal's symmetry).
    """
    from midas_hkls import Lattice
    path = Path(path)
    block = name or path.stem
    # Extract lattice params from the 3×3 cell matrix
    A = supercell.cell.detach().cpu().numpy()
    a_len = float(np.linalg.norm(A[0]))
    b_len = float(np.linalg.norm(A[1]))
    c_len = float(np.linalg.norm(A[2]))
    alpha = float(np.degrees(np.arccos(np.clip(
        np.dot(A[1], A[2]) / (b_len * c_len), -1, 1))))
    beta = float(np.degrees(np.arccos(np.clip(
        np.dot(A[0], A[2]) / (a_len * c_len), -1, 1))))
    gamma = float(np.degrees(np.arccos(np.clip(
        np.dot(A[0], A[1]) / (a_len * b_len), -1, 1))))

    cell_inv = np.linalg.inv(A)
    fract = supercell.positions.detach().cpu().numpy() @ cell_inv

    lines: List[str] = []
    lines.append(f"data_{block}")
    lines.append(f"_cell_length_a       {a_len:.6f}")
    lines.append(f"_cell_length_b       {b_len:.6f}")
    lines.append(f"_cell_length_c       {c_len:.6f}")
    lines.append(f"_cell_angle_alpha    {alpha:.6f}")
    lines.append(f"_cell_angle_beta     {beta:.6f}")
    lines.append(f"_cell_angle_gamma    {gamma:.6f}")
    lines.append("_symmetry_Int_Tables_number   1     # P1 — every atom explicit")
    lines.append("_symmetry_space_group_name_H-M   'P 1'")
    lines.append("")
    lines.append("loop_")
    lines.append("_atom_site_label")
    lines.append("_atom_site_type_symbol")
    lines.append("_atom_site_fract_x")
    lines.append("_atom_site_fract_y")
    lines.append("_atom_site_fract_z")
    counters: Dict[str, int] = {}
    for i, elem in enumerate(supercell.species):
        counters[elem] = counters.get(elem, 0) + 1
        label = f"{elem}{counters[elem]}"
        fx, fy, fz = float(fract[i, 0]), float(fract[i, 1]), float(fract[i, 2])
        lines.append(f"{label:10s}  {elem:2s}  "
                     f"{fx:.6f}  {fy:.6f}  {fz:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


__all__ = [
    "CIFData",
    "parse_cif",
    "read_cif_to_crystal",
    "write_crystal_to_cif",
    "write_supercell_to_cif",
]
