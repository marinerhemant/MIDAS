"""Reading MIDAS far-field ``Grains.csv``.

The FF analogue of :mod:`midas_plotting.mic`: one place that knows the column
layout so analysis scripts stop re-deriving it.

Columns are looked up **by name** from the ``%ID ...`` header line, never by
position. That matters more here than it looks: ``Grains.csv`` has grown to 47
columns, and `midas-fit-grain` 0.5.6 shipped a cyclic rotation of the
``DiffPos`` / ``DiffOme`` / ``DiffAngle`` columns (fixed in 0.5.7) that a
positional reader would silently inherit -- one grain's ω residual read 223.87°
where the true value was 0.054°.

As a further guard, :func:`read_grains` recomputes the orientation matrix from
the Euler angles and compares it against the ``O11..O33`` columns. Both describe
the same orientation, so any disagreement means the row is being sliced wrong
(or the file was written by a broken version), and it is far better to hear
about that than to plot it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["GrainList", "read_grains"]

_OM_NAMES = [f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
_EUL_NAMES = ["Eul0", "Eul1", "Eul2"]
_LATTICE_NAMES = ["a", "b", "c", "alpha", "beta", "gamma"]
_FAB = [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
_KEN = [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]

#: Euler/orientation-matrix agreement above this is treated as a real
#: inconsistency. ``Grains.csv`` is written at ~6 significant figures, so
#: round-trip differences of ~1e-6 are expected and harmless.
OM_EULER_TOL = 1e-3


@dataclass
class GrainList:
    """A parsed FF ``Grains.csv``.

    Attributes
    ----------
    ids : (N,) int
    pos : (N, 3) float
        Grain centre-of-mass X, Y, Z in **micrometres**. Trustworthy to about
        ~100 µm on a typical reconstruction -- do not read the six decimals the
        file prints (``FF_HEDM_Lab_Notebook.md`` §2d).
    euler : (N, 3) float
        Bunge ZXZ Euler angles in **radians**, matching the ``.mic`` convention
        so :func:`midas_plotting.ipf_rgb` accepts them directly.
    orient_mat : (N, 3, 3) float
    lattice : (N, 6) float
        a, b, c (Å) and alpha, beta, gamma (degrees).
    radius : (N,) float
        ``GrainRadius`` in µm. Correct only with ``midas-process-grains``
        >= 0.6.1; older versions report ~the sample-wide mean for every grain.
    completeness : (N,) float
    diff_pos, diff_ome, diff_angle : (N,) float
        Residuals. Cyclically mislabeled by `midas-fit-grain` 0.5.6.
    strain_fab, strain_ken : (N, 3, 3) float or None
    rmse_strain : (N,) float or None
    phase : (N,) int or None
    header : dict
        The ``%key value`` preamble (NumGrains, BeamCenter, ...).
    columns : list of str
    raw : (N, C) float
    path : Path
    """

    ids: np.ndarray
    pos: np.ndarray
    euler: np.ndarray
    orient_mat: np.ndarray
    lattice: np.ndarray
    radius: np.ndarray
    completeness: np.ndarray
    diff_pos: np.ndarray
    diff_ome: np.ndarray
    diff_angle: np.ndarray
    strain_fab: Optional[np.ndarray]
    strain_ken: Optional[np.ndarray]
    rmse_strain: Optional[np.ndarray]
    phase: Optional[np.ndarray]
    header: dict
    columns: list
    raw: np.ndarray
    path: Path

    def __len__(self) -> int:
        return int(self.ids.size)

    @property
    def space_group(self) -> Optional[int]:
        """Space group from the file's own ``%\tSpaceGroup:`` line, if present.

        Plot functions default to this rather than to a hard-coded 225, so a
        hexagonal or tetragonal sample is not silently coloured with the cubic
        IPF triangle -- which produces a plausible-looking figure that is
        simply wrong.
        """
        v = self.header.get("SpaceGroup")
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return None

    @property
    def lattice_parameter(self) -> Optional[np.ndarray]:
        """The header's reference lattice parameter (a, b, c, al, be, ga)."""
        v = self.header.get("Lattice Parameter")
        if not v:
            return None
        try:
            arr = np.array([float(x) for x in str(v).split()], dtype=float)
        except ValueError:
            return None
        return arr if arr.size == 6 else None

    @property
    def n_grains(self) -> int:
        return len(self)

    def strain(self, convention: str = "fab") -> np.ndarray:
        """``(N, 3, 3)`` strain tensor in the requested convention."""
        c = convention.lower()
        if c in ("fab", "fable", "efab"):
            s = self.strain_fab
        elif c in ("ken", "kenesei", "eken"):
            s = self.strain_ken
        else:
            raise ValueError(
                f"unknown strain convention {convention!r}; use 'fab' or 'ken'")
        if s is None:
            raise ValueError(
                f"{self.path.name} has no {c} strain columns "
                "(ProcessGrains may have been run without strain output)")
        return s


def _parse(path: Path):
    """Split a Grains.csv into (header dict, column names, data rows).

    The preamble is not uniform. Real files contain all of:

        %NumGrains 2
        %PhaseInfo                      <- key with no value
        %\tSpaceGroup:225               <- TAB-indented `key:value` continuation
        %\tLattice Parameter:4.0782 ... <- key with spaces, colon-separated
        %ID\tO11\t...                    <- the column header

    so a naive `body.split()[0]` raises on the indented lines. The column
    header is identified by content (it names the orientation-matrix columns)
    rather than by position, since the number of preamble lines varies with
    the number of phases.
    """
    header, columns, rows = {}, None, []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if not line.startswith("%"):
            rows.append(line.split("\t"))
            continue

        body = line[1:]
        fields = [f.strip() for f in body.split("\t")]
        # Column header: names O11 (and therefore the whole grain record).
        if "O11" in fields or (fields and fields[0] in ("ID", "GrainID")
                               and len(fields) > 5):
            columns = fields
            continue

        stripped = body.strip()
        if not stripped:
            continue
        if ":" in stripped:                      # `SpaceGroup:225`
            k, _, v = stripped.partition(":")
            header[k.strip()] = v.strip()
        else:                                    # `NumGrains 2` / `PhaseInfo`
            parts = stripped.split()
            header[parts[0]] = " ".join(parts[1:])
    return header, columns, rows


def read_grains(path, *, check_orientation: bool = True) -> GrainList:
    """Parse an FF ``Grains.csv``.

    Parameters
    ----------
    path : str or Path
    check_orientation : bool
        Recompute the orientation matrix from the Euler columns and compare
        against ``O11..O33``. A mismatch means the columns are being read
        wrong; a warning is emitted rather than an exception, so a file written
        by an old or unusual version can still be inspected -- but do not
        trust orientation-derived output (including IPF colour) when it fires.
    """
    path = Path(path)
    header, columns, rows = _parse(path)
    if columns is None:
        raise ValueError(
            f"{path}: no column header found. Expected a line beginning "
            "'%ID' or '%GrainID' listing tab-separated column names.")
    if not rows:
        raise ValueError(f"{path}: header present but no grain rows.")

    idx = {name: i for i, name in enumerate(columns)}
    arr = np.array([[float(v) for v in r] for r in rows], dtype=float)
    if arr.shape[1] != len(columns):
        raise ValueError(
            f"{path}: {arr.shape[1]} data columns but {len(columns)} header "
            "names -- the file is malformed or tab/space separated "
            "inconsistently.")

    def col(name, required=True):
        if name not in idx:
            if required:
                raise ValueError(
                    f"{path}: required column {name!r} not found. "
                    f"Columns present: {columns}")
            return None
        return arr[:, idx[name]]

    def block(names):
        if not all(n in idx for n in names):
            return None
        return np.stack([arr[:, idx[n]] for n in names], axis=1)

    id_name = "ID" if "ID" in idx else "GrainID"
    ids = col(id_name).astype(int)
    pos = np.stack([col("X"), col("Y"), col("Z")], axis=1)

    om_flat = block(_OM_NAMES)
    if om_flat is None:
        raise ValueError(f"{path}: orientation matrix columns O11..O33 missing.")
    orient_mat = om_flat.reshape(-1, 3, 3)

    eul = block(_EUL_NAMES)
    if eul is None:
        # Older writers omitted the Euler columns; derive them so downstream
        # (and ipf_rgb) has a single, consistent source.
        from midas_stress.orientation import orient_mat_to_euler

        eul = np.array([np.asarray(orient_mat_to_euler(m.reshape(-1).tolist()),
                                   dtype=float).reshape(3)
                        for m in orient_mat])

    if check_orientation:
        _check_orientation(path, eul, orient_mat)

    lattice = block(_LATTICE_NAMES)
    if lattice is None:
        lattice = np.full((len(ids), 6), np.nan)

    fab = block(_FAB)
    ken = block(_KEN)
    return GrainList(
        ids=ids,
        pos=pos,
        euler=eul,
        orient_mat=orient_mat,
        lattice=lattice,
        radius=col("GrainRadius", required=False),
        completeness=col("Confidence", required=False),
        diff_pos=col("DiffPos", required=False),
        diff_ome=col("DiffOme", required=False),
        diff_angle=col("DiffAngle", required=False),
        strain_fab=None if fab is None else fab.reshape(-1, 3, 3),
        strain_ken=None if ken is None else ken.reshape(-1, 3, 3),
        rmse_strain=col("RMSErrorStrain", required=False),
        phase=None if "PhaseNr" not in idx else col("PhaseNr").astype(int),
        header=header,
        columns=columns,
        raw=arr,
        path=path,
    )


def _check_orientation(path, euler, orient_mat) -> None:
    """Euler and O11..O33 must describe the same orientation."""
    try:
        from midas_stress.orientation import euler_to_orient_mat_batch
    except Exception:                                      # noqa: BLE001
        return
    rec = np.asarray(euler_to_orient_mat_batch(euler)).reshape(-1, 3, 3)
    dev = float(np.abs(rec - orient_mat).max())
    if dev > OM_EULER_TOL:
        warnings.warn(
            f"{path.name}: Euler angles and the O11..O33 matrix disagree by "
            f"{dev:.3g} (tolerance {OM_EULER_TOL:g}). The columns are probably "
            "being read wrong, or the file was written by a version with a "
            "column-ordering bug -- `midas-fit-grain` 0.5.6 shipped one. Do "
            "NOT trust orientation-derived output (IPF colour, pole figures, "
            "misorientation) from this file until it is resolved.",
            RuntimeWarning,
            stacklevel=3,
        )
