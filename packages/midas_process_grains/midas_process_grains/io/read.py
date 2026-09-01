"""Canonical readers for ``Grains.csv`` and ``SpotMatrix.csv``.

These files have been widened repeatedly (Grains 19 -> 21 -> 47 -> 53 columns;
SpotMatrix 12 -> 28) and are written under two different header tokens.  Every
positional reader in the tree encoded one snapshot of that and silently drifted.
The single rule here is: **resolve every column by NAME, never by position.**

Three traps this module exists to close
---------------------------------------
1. **Two header spellings.**  ``compute/c_parity_emit.py`` writes
   ``%GrainID...`` (the ``c_parity`` default, i.e. what the CLI produces) while
   ``io/csv.py`` writes ``%ID...`` (every other mode).  Both are real and both
   are on disk.  Readers keyed on one see *zero grains* on the other, usually
   without raising.  We anchor on ``O11`` instead, which every version has.

2. **Columns 13-18 mean different things at different widths.**  On a 21-column
   legacy file they are ``E11 E22 E33 E12 E13 E23`` -- a Voigt STRAIN.  On 47
   and 53 they are ``a b c alpha beta gamma`` -- a LATTICE.  Reading position
   13:19 and calling it "strain" is the single most common bug in the tree and
   it never raises.  Here they are separate attributes and only one is ever
   populated.

3. **SpotMatrix carries predicted-but-never-found reflections.**  Rows with
   ``Matched == 0`` are reflections the grain was predicted to produce and which
   were never observed; they hold ``-1`` in ``SpotID``/``RingNr`` and NaN in the
   observed columns.  They are ~3.3 % of rows on real data.  ``matched_only``
   therefore defaults to **True**: seeing them is opt-in, not opt-out.
   NaN alone is *not* a usable marker -- on matched rows ``theorSpotID`` and the
   post-fit columns are also NaN whenever the diagnostics sidecars are absent.
   Only column 12 is authoritative.

Units, because they are not obvious and are not in the column names
-------------------------------------------------------------------
* ``strain_fab`` / ``strain_ken`` / ``rms_error_strain`` are **microstrain**.
* ``euler`` is **radians** (``compute/c_parity_emit.py``'s module docstring
  says degrees; it is wrong -- see ``orient_mat_to_euler_rad`` at its line 388).
* positions and ``diff_pos`` are micrometres, ``diff_ome``/``diff_angle`` degrees.
* ``strain_voigt`` (legacy 21-column files only) is in whatever units that
  generation wrote; it is **not** converted here and ``strain_units`` says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "GrainsTable",
    "SpotMatrixTable",
    "read_grains_csv",
    "read_spot_matrix",
    "GrainsFormatError",
]

#: Column names that identify the grain-id column, in preference order.
_ID_NAMES = ("GrainID", "ID")
#: The nine orientation-matrix columns. Present in every version, which is why
#: the header line is located by looking for O11 rather than by the ID token.
_OM_NAMES = ("O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33")
_POS_NAMES = ("X", "Y", "Z")
_LATTICE_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")
#: Legacy 21-column Voigt strain, at the SAME positions the lattice occupies on
#: wider files. Distinguished by name only.
_VOIGT_NAMES = ("E11", "E22", "E33", "E12", "E13", "E23")
_FAB_NAMES = tuple(f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3))
_KEN_NAMES = tuple(f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3))
_EULER_NAMES = ("Eul0", "Eul1", "Eul2")


class GrainsFormatError(ValueError):
    """Raised when a file is not a recognisable ProcessGrains output."""


@dataclass
class GrainsTable:
    """One ``Grains.csv``, resolved by column name.

    Optional blocks are ``None`` when the file's width predates them, so a
    caller can test presence instead of guessing from a column count.
    """

    ids: np.ndarray                                   # (n,) int64
    orient_mat: np.ndarray                            # (n, 3, 3) float64
    positions: np.ndarray                             # (n, 3) float64, um
    lattice: Optional[np.ndarray] = None              # (n, 6) a,b,c,al,be,ga
    strain_voigt: Optional[np.ndarray] = None         # (n, 6) legacy only
    diff_pos: Optional[np.ndarray] = None             # (n,) um
    diff_ome: Optional[np.ndarray] = None             # (n,) deg
    diff_angle: Optional[np.ndarray] = None           # (n,) deg
    grain_radius: Optional[np.ndarray] = None         # (n,) um
    confidence: Optional[np.ndarray] = None           # (n,)
    strain_fab: Optional[np.ndarray] = None           # (n, 3, 3) microstrain
    strain_ken: Optional[np.ndarray] = None           # (n, 3, 3) microstrain
    rms_error_strain: Optional[np.ndarray] = None     # (n,) microstrain
    phase_nr: Optional[np.ndarray] = None             # (n,) int64
    euler: Optional[np.ndarray] = None                # (n, 3) RADIANS
    diff_pos_pre: Optional[np.ndarray] = None
    diff_ome_pre: Optional[np.ndarray] = None
    diff_angle_pre: Optional[np.ndarray] = None
    diff_pos_post: Optional[np.ndarray] = None
    diff_ome_post: Optional[np.ndarray] = None
    diff_angle_post: Optional[np.ndarray] = None
    columns: List[str] = field(default_factory=list)  # header, '%' stripped
    n_columns: int = 0
    header_token: str = ""                            # 'ID' or 'GrainID'
    space_group: Optional[int] = None
    lattice_parameter: Optional[Sequence[float]] = None
    num_phases: Optional[int] = None
    strain_units: str = "microstrain"
    path: Optional[Path] = None

    @property
    def n_grains(self) -> int:
        return int(self.ids.shape[0])

    def column(self, name: str) -> np.ndarray:
        """Raw column by header name -- the escape hatch for anything not
        promoted to an attribute. Raises rather than returning the wrong one."""
        if name not in self._raw:
            raise KeyError(
                f"{self.path}: no column {name!r}; header has {self.columns}")
        return self._raw[name]

    _raw: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)


@dataclass
class SpotMatrixTable:
    """One ``SpotMatrix.csv``, resolved by column name."""

    grain_id: np.ndarray
    spot_id: np.ndarray
    omega: np.ndarray
    detector_hor: np.ndarray
    detector_vert: np.ndarray
    ome_raw: np.ndarray
    eta: np.ndarray
    ring_nr: np.ndarray
    y_lab: np.ndarray
    z_lab: np.ndarray
    theta: np.ndarray
    strain_error: np.ndarray
    matched: Optional[np.ndarray] = None
    columns: List[str] = field(default_factory=list)
    n_columns: int = 0
    n_rows_total: int = 0
    n_rows_unmatched: int = 0
    matched_only: bool = True
    path: Optional[Path] = None
    _raw: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def column(self, name: str) -> np.ndarray:
        if name not in self._raw:
            raise KeyError(
                f"{self.path}: no column {name!r}; header has {self.columns}")
        return self._raw[name]


def _split_row(raw: str) -> List[str]:
    """Split a data row.

    ``SpotMatrix.csv`` is written with ``newline='\\t\\n'`` so every row ends in
    a tab; a naive ``split('\\t')`` yields a trailing empty field and
    ``float('')`` raises. ``Grains.csv`` has no trailing tab. Stripping first
    handles both, and whitespace-splitting handles the space-separated
    generations too.
    """
    return raw.rstrip().rstrip("\t").split()


def _find_header(lines: Sequence[str], path: Path) -> int:
    """Index of the column-header line.

    Anchored on ``O11``, never on the ID token: the ID column is spelled
    ``GrainID`` by ``c_parity_emit`` and ``ID`` by ``io/csv``, and keying on
    either one silently rejects half the corpus.
    """
    for i, line in enumerate(lines):
        if line.startswith("%") and "O11" in line.lstrip("%").split():
            return i
    raise GrainsFormatError(
        f"{path}: no '%' header line containing the orientation columns "
        f"{_OM_NAMES[0]}..{_OM_NAMES[-1]} — not a ProcessGrains Grains.csv? "
        f"(NF's mic2grains.py writes a prose header and is not readable here.)"
    )


def _meta(lines: Sequence[str]) -> dict:
    """Parse the '%' preamble. Never assume a fixed line count: it is 9 today
    but is written per-phase-agnostically and has changed."""
    out: dict = {}
    for line in lines:
        s = line.strip()
        if not s.startswith("%"):
            continue
        body = s.lstrip("%").strip()
        if body.startswith("NumGrains"):
            try:
                out["num_grains"] = int(body.split()[1])
            except (IndexError, ValueError):
                pass
        elif body.startswith("NumPhases"):
            try:
                out["num_phases"] = int(body.split()[1])
            except (IndexError, ValueError):
                pass
        elif body.startswith("SpaceGroup"):
            try:
                out["space_group"] = int(body.split(":")[1].strip())
            except (IndexError, ValueError):
                pass
        elif body.startswith("Lattice"):
            try:
                out["lattice_parameter"] = tuple(
                    float(x) for x in body.split(":")[1].strip().split())
            except (IndexError, ValueError):
                pass
    return out


def _stack(raw: Dict[str, np.ndarray], names: Sequence[str]
           ) -> Optional[np.ndarray]:
    """Column-stack a named block, or None if any member is absent."""
    if not all(n in raw for n in names):
        return None
    return np.column_stack([raw[n] for n in names])


def read_grains_csv(path, *, require: Sequence[str] = ()) -> GrainsTable:
    """Read a ``Grains.csv`` of any known width, by column name.

    Handles 19, 21, 23, 47 and 53-column files and both ``%ID`` and
    ``%GrainID`` headers. Optional blocks come back ``None`` rather than
    silently reading a neighbouring column.

    Parameters
    ----------
    path : file to read.
    require : column names that must be present; raises if any is missing.
        Use this instead of checking a column count.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    hidx = _find_header(lines, path)
    cols = lines[hidx].lstrip("%").split()

    id_name = next((n for n in _ID_NAMES if n in cols), None)
    if id_name is None:
        raise GrainsFormatError(
            f"{path}: header has neither {_ID_NAMES[0]} nor {_ID_NAMES[1]}: {cols!r}")

    missing = [n for n in require if n not in cols]
    if missing:
        raise GrainsFormatError(
            f"{path}: required column(s) {missing} absent; header has {cols!r}")

    rows: List[List[str]] = []
    for raw in lines[hidx + 1:]:
        if not raw.strip() or raw.lstrip().startswith("%"):
            continue
        toks = _split_row(raw)
        if len(toks) < len(cols):
            continue        # short/truncated row: skip rather than mis-align
        rows.append(toks[:len(cols)])

    n = len(rows)
    data: Dict[str, np.ndarray] = {}
    if n:
        arr = np.array(rows, dtype=object)
        for j, name in enumerate(cols):
            try:
                data[name] = arr[:, j].astype(np.float64)
            except (TypeError, ValueError):
                data[name] = arr[:, j]
    else:
        for name in cols:
            data[name] = np.zeros(0, dtype=np.float64)

    om = _stack(data, _OM_NAMES)
    if om is None:
        raise GrainsFormatError(f"{path}: header is missing OM columns: {cols!r}")

    t = GrainsTable(
        ids=data[id_name].astype(np.int64),
        orient_mat=om.reshape(n, 3, 3) if n else om.reshape(0, 3, 3),
        positions=_stack(data, _POS_NAMES),
        lattice=_stack(data, _LATTICE_NAMES),
        strain_voigt=_stack(data, _VOIGT_NAMES),
        columns=cols,
        n_columns=len(cols),
        header_token=id_name,
        path=path,
    )
    for attr, name in (
        ("diff_pos", "DiffPos"), ("diff_ome", "DiffOme"),
        ("diff_angle", "DiffAngle"), ("grain_radius", "GrainRadius"),
        ("confidence", "Confidence"), ("rms_error_strain", "RMSErrorStrain"),
        ("diff_pos_pre", "DiffPosPre"), ("diff_ome_pre", "DiffOmePre"),
        ("diff_angle_pre", "DiffAnglePre"), ("diff_pos_post", "DiffPosPost"),
        ("diff_ome_post", "DiffOmePost"), ("diff_angle_post", "DiffAnglePost"),
    ):
        if name in data:
            setattr(t, attr, data[name])
    if "PhaseNr" in data:
        t.phase_nr = data["PhaseNr"].astype(np.int64)
    fab = _stack(data, _FAB_NAMES)
    ken = _stack(data, _KEN_NAMES)
    t.strain_fab = fab.reshape(n, 3, 3) if fab is not None else None
    t.strain_ken = ken.reshape(n, 3, 3) if ken is not None else None
    t.euler = _stack(data, _EULER_NAMES)
    if t.strain_voigt is not None and t.strain_fab is None:
        # legacy generation: units never recorded, so do not claim they are ue
        t.strain_units = "unspecified (legacy E11..E23 block)"
    t._raw = data

    meta = _meta(lines[:hidx])
    t.space_group = meta.get("space_group")
    t.lattice_parameter = meta.get("lattice_parameter")
    t.num_phases = meta.get("num_phases")
    declared = meta.get("num_grains")
    if declared is not None and declared != n:
        import warnings
        warnings.warn(
            f"{path}: header says %NumGrains {declared} but {n} data rows were "
            f"parsed. Rows shorter than the header are skipped; check the file.",
            RuntimeWarning, stacklevel=2)
    return t


def read_spot_matrix(path, *, matched_only: bool = True) -> SpotMatrixTable:
    """Read a ``SpotMatrix.csv`` of either width, by column name.

    Parameters
    ----------
    matched_only : default **True**. Drops rows with ``Matched == 0`` -- the
        reflections that were predicted but never found. They carry ``-1`` in
        ``SpotID``/``RingNr`` and NaN in the observed columns, and on real data
        are ~3.3 % of rows. Feeding them to a fit gives NaN residuals; feeding
        them to a spot-overlap statistic makes every pair of grains share
        spot ``-1``. Pass False only if you specifically want the un-found
        population (e.g. the per-ring completeness deficit).

        A 12-column legacy file has no ``Matched`` column; every row is by
        definition an observation, so nothing is dropped and ``matched`` is None.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    hidx = None
    for i, line in enumerate(lines):
        if line.startswith("%") and "SpotID" in line.lstrip("%").split():
            hidx = i
            break
    if hidx is None:
        raise GrainsFormatError(
            f"{path}: no '%' header line containing SpotID — not a SpotMatrix.csv?")
    cols = lines[hidx].lstrip("%").split()

    rows: List[List[str]] = []
    for raw in lines[hidx + 1:]:
        if not raw.strip() or raw.lstrip().startswith("%"):
            continue
        toks = _split_row(raw)
        if len(toks) < len(cols):
            continue
        rows.append(toks[:len(cols)])

    n_total = len(rows)
    data: Dict[str, np.ndarray] = {}
    if n_total:
        arr = np.array(rows, dtype=object)
        for j, name in enumerate(cols):
            try:
                data[name] = arr[:, j].astype(np.float64)
            except (TypeError, ValueError):
                data[name] = arr[:, j]
    else:
        for name in cols:
            data[name] = np.zeros(0, dtype=np.float64)

    matched = data.get("Matched")
    n_unmatched = int(np.sum(matched < 0.5)) if matched is not None else 0
    if matched_only and matched is not None and n_unmatched:
        keep = matched > 0.5
        data = {k: v[keep] for k, v in data.items()}
        matched = data["Matched"]

    gid_name = next((c for c in ("GrainID", "ID") if c in cols), cols[0])

    def _col(name, alt=None):
        if name in data:
            return data[name]
        if alt and alt in data:
            return data[alt]
        return np.zeros(len(data[gid_name]), dtype=np.float64)

    t = SpotMatrixTable(
        grain_id=data[gid_name].astype(np.int64),
        spot_id=_col("SpotID").astype(np.int64),
        omega=_col("Omega"),
        detector_hor=_col("DetectorHor"),
        detector_vert=_col("DetectorVert"),
        ome_raw=_col("OmeRaw"),
        eta=_col("Eta"),
        ring_nr=_col("RingNr").astype(np.int64),
        y_lab=_col("YLab"),
        z_lab=_col("ZLab"),
        theta=_col("Theta"),
        strain_error=_col("StrainError"),
        matched=matched,
        columns=cols,
        n_columns=len(cols),
        n_rows_total=n_total,
        n_rows_unmatched=n_unmatched,
        matched_only=matched_only,
        path=path,
    )
    t._raw = data
    return t
