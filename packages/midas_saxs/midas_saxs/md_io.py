"""Read MD atomic configurations (LAMMPS dumps) for the atomistic SAXS source.

The scattering code needs four things from a snapshot: positions, the species of
every atom, the periodic cell, and which directions are periodic. It needs them
unambiguously, so this reader refuses to guess:

* A LAMMPS ``type`` column is an integer, not an element. It becomes a species
  only through an explicit ``type_map`` (or an ``element`` column in the dump).
  There is no default element.
* Units must be Angstrom. LAMMPS ``metal`` and ``real`` units are; ``lj``,
  ``si``, ``cgs`` and the rest are refused. A dump with no ``ITEM: UNITS``
  header (older LAMMPS) is taken as Angstrom, which is the caller's to check.
* Triclinic boxes are converted with the LAMMPS rule (bounding box to ``lo`` /
  ``hi``) and returned as three lattice row vectors ``a, b, c``.

Coordinates come back as the file stores them (Cartesian, Angstrom); scaled
columns (``xs ys zs``) are converted. Nothing is wrapped into the cell here,
because ``|sum_i f_i exp(i G.r_i)|^2`` at a reciprocal-lattice point G does not
depend on which periodic image an atom is stored in. :meth:`AtomicFrame.wrapped`
exists for callers that want positions inside the cell.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "AtomicFrame",
    "iter_lammps_dump",
    "read_lammps_dump",
    "read_lammps_trajectory",
    "write_lammps_dump",
]

#: LAMMPS unit styles whose length unit is the Angstrom.
_ANGSTROM_UNITS = ("metal", "real")

TypeMap = Union[Mapping[int, str], Sequence[str]]


@dataclass
class AtomicFrame:
    """One MD snapshot.

    Attributes
    ----------
    positions : (N, 3) float64, Angstrom, Cartesian, as stored in the file.
    species : (N,) array of element symbols (``"Cu"``, ``"He"``, ...).
    cell : (3, 3) float64, Angstrom. Rows are the lattice vectors ``a, b, c``
        (lower-triangular, the LAMMPS convention: ``a`` along x, ``b`` in the xy plane).
    origin : (3,) float64, Angstrom. The cell's lower corner.
    pbc : which of the three directions are periodic.
    timestep : the dump's ``ITEM: TIMESTEP`` value, if present.
    ids : (N,) int atom ids, if the dump has an ``id`` column.
    columns : any other per-atom columns, keyed by header name.
    """

    positions: np.ndarray
    species: np.ndarray
    cell: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pbc: Tuple[bool, bool, bool] = (True, True, True)
    timestep: Optional[int] = None
    ids: Optional[np.ndarray] = None
    columns: Dict[str, np.ndarray] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.species = np.asarray(self.species)
        self.cell = np.asarray(self.cell, dtype=np.float64)
        self.origin = np.asarray(self.origin, dtype=np.float64)
        n = self.positions.shape[0]
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")
        if self.species.shape != (n,):
            raise ValueError(f"species must be ({n},), got {self.species.shape}")
        if self.cell.shape != (3, 3):
            raise ValueError(f"cell must be (3, 3), got {self.cell.shape}")
        if not np.isfinite(self.positions).all():
            raise ValueError("positions contain NaN or inf")
        if abs(np.linalg.det(self.cell)) < 1e-12:
            raise ValueError("the cell is singular (zero volume)")

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]

    @property
    def volume(self) -> float:
        """Cell volume, A^3."""
        return float(abs(np.linalg.det(self.cell)))

    def fractional(self) -> np.ndarray:
        """Fractional coordinates ``s`` with ``r = origin + s @ cell``. Not wrapped."""
        return np.linalg.solve(self.cell.T, (self.positions - self.origin).T).T

    def wrapped(self) -> np.ndarray:
        """Cartesian positions with every periodic direction folded into [0, 1) of the cell."""
        s = self.fractional()
        for k, periodic in enumerate(self.pbc):
            if periodic:
                s[:, k] -= np.floor(s[:, k])
        return self.origin + s @ self.cell

    def species_list(self) -> List[str]:
        """Distinct species, in order of first appearance."""
        _, first = np.unique(self.species, return_index=True)
        return [str(self.species[i]) for i in sorted(first)]


# ----------------------------------------------------------------------------- helpers

def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _type_lookup(type_map: Optional[TypeMap]) -> Optional[Dict[int, str]]:
    if type_map is None:
        return None
    if isinstance(type_map, Mapping):
        return {int(k): str(v) for k, v in type_map.items()}
    return {i + 1: str(v) for i, v in enumerate(type_map)}      # LAMMPS types start at 1


def _check_element(sym: str, where: str) -> str:
    from midas_hkls.form_factors import available_elements

    s = sym.strip()
    s = s[:1].upper() + s[1:].lower()
    if s not in set(available_elements()):
        raise ValueError(
            f"{where}: {sym!r} is not an element in the Cromer-Mann table. "
            f"Pass an element symbol per LAMMPS type through `type_map`.")
    return s


def _cell_from_bounds(rows: List[List[float]], triclinic: bool):
    """LAMMPS box bounds -> (origin, cell rows). See the LAMMPS dump documentation."""
    if not triclinic:
        lo = np.array([r[0] for r in rows])
        hi = np.array([r[1] for r in rows])
        return lo, np.diag(hi - lo)
    (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo, zhi, yz) = rows
    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
    xhi = xhi_b - max(0.0, xy, xz, xy + xz)
    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)
    cell = np.array([[xhi - xlo, 0.0, 0.0],
                     [xy, yhi - ylo, 0.0],
                     [xz, yz, zhi - zlo]])
    return np.array([xlo, ylo, zlo]), cell


_PBC_FLAGS = {"pp": True}


def _parse_pbc(tokens: Sequence[str]) -> Tuple[bool, bool, bool]:
    flags = tokens[-3:]
    if len(flags) != 3:
        raise ValueError(f"cannot read the boundary flags from {tokens!r}")
    return tuple(_PBC_FLAGS.get(f, False) for f in flags)       # type: ignore[return-value]


# ----------------------------------------------------------------------------- reader

def iter_lammps_dump(
    path: str,
    *,
    type_map: Optional[TypeMap] = None,
) -> Iterator[AtomicFrame]:
    """Yield every snapshot of a LAMMPS text dump (plain or ``.gz``), one at a time.

    Streams the file, so a long trajectory is never held in memory at once.

    Parameters
    ----------
    path
        ``dump ... custom`` / ``atom`` text file.
    type_map
        ``{1: "Cu", 2: "He"}`` or ``["Cu", "He"]`` (index 0 is LAMMPS type 1).
        Required unless the dump has an ``element`` column.
    """
    tmap = _type_lookup(type_map)
    with _open(path) as fh:
        lines = iter(fh)
        units: Optional[str] = None
        timestep: Optional[int] = None
        n_atoms: Optional[int] = None
        origin = cell = None
        pbc = (True, True, True)
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if not line.startswith("ITEM:"):
                raise ValueError(f"{path}: expected an 'ITEM:' line, got {line[:60]!r}. "
                                 f"Not a LAMMPS dump?")
            item = line[5:].strip()
            if item.startswith("UNITS"):
                units = next(lines).strip()
                if units not in _ANGSTROM_UNITS:
                    raise ValueError(
                        f"{path}: dump units are {units!r}; the SAXS source needs "
                        f"Angstrom ({', '.join(_ANGSTROM_UNITS)}). Re-run with metal "
                        f"or real units, or convert the dump first.")
            elif item.startswith("TIME") and not item.startswith("TIMESTEP"):
                next(lines)
            elif item.startswith("TIMESTEP"):
                timestep = int(next(lines))
            elif item.startswith("NUMBER OF ATOMS"):
                n_atoms = int(next(lines))
            elif item.startswith("BOX BOUNDS"):
                toks = item.split()[2:]
                triclinic = "xy" in toks
                pbc = _parse_pbc(toks)
                rows = [[float(x) for x in next(lines).split()] for _ in range(3)]
                width = 3 if triclinic else 2
                if any(len(r) < width for r in rows):
                    raise ValueError(f"{path}: malformed BOX BOUNDS block")
                origin, cell = _cell_from_bounds([r[:width] for r in rows], triclinic)
            elif item.startswith("ATOMS"):
                if n_atoms is None or cell is None:
                    raise ValueError(f"{path}: ATOMS block before NUMBER OF ATOMS / BOX BOUNDS")
                cols = item.split()[1:]
                buf = []
                for _ in range(n_atoms):
                    try:
                        buf.append(next(lines))
                    except StopIteration:
                        raise ValueError(
                            f"{path}: truncated, timestep {timestep} promises {n_atoms} "
                            f"atoms and the file ends after {len(buf)}") from None
                yield _build_frame(buf, cols, n_atoms, timestep, origin, cell, pbc, tmap,
                                   str(path))
                timestep = n_atoms = None
                origin = cell = None
            else:
                raise ValueError(f"{path}: unrecognised dump section {line[:60]!r}")


def _build_frame(buf, cols, n_atoms, timestep, origin, cell, pbc, tmap, where) -> AtomicFrame:
    if n_atoms == 0:
        raise ValueError(f"{where}: timestep {timestep} has zero atoms")
    idx = {c: i for i, c in enumerate(cols)}
    numeric = [i for i, c in enumerate(cols) if c != "element"]
    data = np.loadtxt(buf, usecols=numeric, ndmin=2)
    if data.shape != (n_atoms, len(numeric)):
        raise ValueError(f"{where}: atom rows do not match the header columns {cols}")
    col = {cols[i]: data[:, j] for j, i in enumerate(numeric)}

    # coordinates
    for names, scaled in ((("x", "y", "z"), False), (("xu", "yu", "zu"), False),
                          (("xs", "ys", "zs"), True), (("xsu", "ysu", "zsu"), True)):
        if all(n in col for n in names):
            xyz = np.stack([col[n] for n in names], axis=1)
            if scaled:
                xyz = origin + xyz @ cell
            used = set(names)
            break
    else:
        raise ValueError(f"{where}: dump columns {cols} have no x y z (or xu, xs, xsu) triple")

    # species
    if "element" in idx:
        k = idx["element"]
        names = [ln.split()[k] for ln in buf]
        species = np.array([_check_element(s, where) for s in names])
        used.add("element")
    elif "type" in col:
        if tmap is None:
            raise ValueError(
                f"{where}: the dump has integer atom types and no element column. "
                f"Pass type_map={{1: 'Cu', ...}}; there is no default element.")
        types = col["type"].astype(int)
        missing = sorted(set(types.tolist()) - set(tmap))
        if missing:
            raise ValueError(f"{where}: type_map has no entry for type(s) {missing}")
        lut = {t: _check_element(tmap[t], where) for t in set(types.tolist())}
        species = np.array([lut[t] for t in types.tolist()])
        used.add("type")
    else:
        raise ValueError(f"{where}: dump columns {cols} have neither 'element' nor 'type'")

    ids = col["id"].astype(np.int64) if "id" in col else None
    used.add("id")
    extra = {c: col[c] for c in col if c not in used}
    return AtomicFrame(positions=xyz, species=species, cell=cell, origin=origin, pbc=pbc,
                       timestep=timestep, ids=ids, columns=extra, source=where)


def read_lammps_dump(path: str, *, type_map: Optional[TypeMap] = None,
                     frame: int = 0) -> AtomicFrame:
    """Read one snapshot (the ``frame``-th, 0-based) of a LAMMPS dump."""
    if frame < 0:
        raise ValueError("frame must be >= 0; use read_lammps_trajectory for the whole file")
    n_seen = 0
    for f in iter_lammps_dump(path, type_map=type_map):
        if n_seen == frame:
            return f
        n_seen += 1
    raise ValueError(f"{path}: has only {n_seen} frames, asked for frame {frame}")


def read_lammps_trajectory(path: str, *, type_map: Optional[TypeMap] = None,
                           frames: Optional[slice] = None) -> List[AtomicFrame]:
    """Read every snapshot (or ``frames``, a slice over frame indices) into a list.

    Raises if the cell or the species list changes between the selected frames:
    averaging intensities across them would otherwise silently mix different
    q-lattices. A fixed-cell NVT / NVE trajectory passes.
    """
    sl = frames if frames is not None else slice(None)
    out = [f for k, f in enumerate(iter_lammps_dump(path, type_map=type_map))
           if _in_slice(k, sl)]
    if not out:
        raise ValueError(f"{path}: no frames selected")
    for f in out[1:]:
        if not np.allclose(f.cell, out[0].cell, rtol=0, atol=1e-9) or \
                not np.allclose(f.origin, out[0].origin, rtol=0, atol=1e-9):
            raise ValueError(f"{path}: the cell changes between frames (timesteps "
                             f"{out[0].timestep} and {f.timestep}); average only fixed-cell frames")
        if f.n_atoms != out[0].n_atoms or not np.array_equal(f.species, out[0].species):
            raise ValueError(f"{path}: atom count or species order changes between frames")
    return out


def _in_slice(k: int, sl: slice) -> bool:
    start = sl.start or 0
    step = sl.step or 1
    if sl.stop is not None and k >= sl.stop:
        return False
    return k >= start and (k - start) % step == 0


# ----------------------------------------------------------------------------- writer

def write_lammps_dump(
    path: str,
    frames: Union[AtomicFrame, Iterable[AtomicFrame]],
    *,
    type_map: Optional[TypeMap] = None,
    element_column: bool = False,
    scaled: bool = False,
    units: Optional[str] = "metal",
) -> None:
    """Write snapshots as a LAMMPS custom dump. Used by the tests and for making example inputs.

    The cell must be lower-triangular (LAMMPS' own convention). With ``type_map`` the
    species are written as integer types; with ``element_column=True`` as symbols.
    """
    fr = [frames] if isinstance(frames, AtomicFrame) else list(frames)
    if not element_column and type_map is None:
        raise ValueError("give a type_map (integer types) or element_column=True")
    inv: Optional[Dict[str, int]] = None
    if not element_column:
        tm = _type_lookup(type_map)
        inv = {v: k for k, v in tm.items()}                     # type: ignore[union-attr]
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as fh:
        for k, f in enumerate(fr):
            a, b, c = f.cell
            if abs(a[1]) + abs(a[2]) + abs(b[2]) > 1e-9:
                raise ValueError("cell is not lower-triangular; LAMMPS cannot represent it")
            xy, xz, yz = float(b[0]), float(c[0]), float(c[1])
            tri = max(abs(xy), abs(xz), abs(yz)) > 0
            lo = [float(v) for v in f.origin]
            hi = [float(v) for v in f.origin + np.array([a[0], b[1], c[2]])]
            flags = " ".join("pp" if p else "ff" for p in f.pbc)
            if units:
                fh.write(f"ITEM: UNITS\n{units}\n")
            fh.write(f"ITEM: TIMESTEP\n{f.timestep if f.timestep is not None else k}\n")
            fh.write(f"ITEM: NUMBER OF ATOMS\n{f.n_atoms}\n")
            if tri:
                xlo_b = lo[0] + min(0.0, xy, xz, xy + xz)
                xhi_b = hi[0] + max(0.0, xy, xz, xy + xz)
                ylo_b = lo[1] + min(0.0, yz)
                yhi_b = hi[1] + max(0.0, yz)
                fh.write(f"ITEM: BOX BOUNDS xy xz yz {flags}\n")
                fh.write(f"{xlo_b!r} {xhi_b!r} {xy!r}\n{ylo_b!r} {yhi_b!r} {xz!r}\n"
                         f"{lo[2]!r} {hi[2]!r} {yz!r}\n")
            else:
                fh.write(f"ITEM: BOX BOUNDS {flags}\n")
                fh.write(f"{lo[0]!r} {hi[0]!r}\n{lo[1]!r} {hi[1]!r}\n{lo[2]!r} {hi[2]!r}\n")
            ids = f.ids if f.ids is not None else np.arange(1, f.n_atoms + 1)
            xyz = (f.fractional() if scaled else f.positions).tolist()
            names = ("xs ys zs" if scaled else "x y z")
            if element_column:
                fh.write(f"ITEM: ATOMS id element {names}\n")
                for i in range(f.n_atoms):
                    fh.write(f"{int(ids[i])} {f.species[i]} "
                             f"{xyz[i][0]!r} {xyz[i][1]!r} {xyz[i][2]!r}\n")
            else:
                fh.write(f"ITEM: ATOMS id type {names}\n")
                for i in range(f.n_atoms):
                    fh.write(f"{int(ids[i])} {inv[str(f.species[i])]} "      # type: ignore[index]
                             f"{xyz[i][0]!r} {xyz[i][1]!r} {xyz[i][2]!r}\n")
