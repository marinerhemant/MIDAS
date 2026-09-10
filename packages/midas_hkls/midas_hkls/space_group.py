"""SpaceGroup class with sginfo-equivalent functionality.

Backed by the Hall-symbol parser and the canonical 530-entry Hall table
extracted from sginfo.h.  Public API:

    sg = SpaceGroup.from_number(225)
    sg = SpaceGroup.from_hm("Fm-3m")
    sg = SpaceGroup.from_hall("-F 4 2 3")
    sg.symmetry_operations()
    sg.is_systematically_absent(h, k, l)
    sg.equivalent_hkls(h, k, l)
    sg.multiplicity(h, k, l)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Iterable, List, Optional, Tuple

from .hall import parse_hall
from .symops import SymOp, expand_group
from .tables import LATTICE_TRANSLATIONS, STBF, crystal_system_for, laue_class_for


@dataclass
class _SgEntry:
    hall_symbol: str
    sg_number: int
    extension: str
    labels: str
    crystal_system: str
    laue_class: str
    centering: str


@lru_cache(maxsize=1)
def _load_table() -> List[_SgEntry]:
    raw = json.loads(files("midas_hkls").joinpath("data/space_groups.json").read_text())
    return [_SgEntry(**r) for r in raw]


@lru_cache(maxsize=1)
def _by_number() -> dict[int, _SgEntry]:
    """First entry per SG number is the standard setting (sginfo convention)."""
    by_num: dict[int, _SgEntry] = {}
    for e in _load_table():
        by_num.setdefault(e.sg_number, e)
    return by_num


def _normalize_hm(s: str) -> str:
    """Aggressively normalize a Hermann-Mauguin name for table lookup."""
    return re.sub(r"[\s_/]+", "", s).lower()


@lru_cache(maxsize=1)
def _by_hm() -> dict[str, _SgEntry]:
    """Map normalized HM symbol → first entry having that name in `labels`."""
    out: dict[str, _SgEntry] = {}
    for e in _load_table():
        # `labels` looks like "C_2/m = C_1_2/m_1" — first '=' is the canonical short form.
        first = e.labels.split("=")[0]
        key = _normalize_hm(first)
        out.setdefault(key, e)
        # Also index any of the alternate names in the labels field.
        for alt in e.labels.split("="):
            out.setdefault(_normalize_hm(alt), e)
    return out


@lru_cache(maxsize=1)
def _by_hall() -> dict[str, _SgEntry]:
    return {e.hall_symbol.strip(): e for e in _load_table()}


@dataclass
class SpaceGroup:
    number: int
    hall_symbol: str
    hm_symbol: str            # the "canonical short" HM name, e.g. "P_m_-3_m"
    extension: str            # sginfo's setting code, may be ""
    crystal_system: str
    laue_class: str
    centering: str            # 'P' / 'A' / ... / 'F'
    operations: Tuple[SymOp, ...] = field(repr=False)

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_number(cls, number: int, extension: str = "") -> "SpaceGroup":
        if not (1 <= number <= 230):
            raise ValueError(f"space group number out of range: {number}")
        if extension:
            for e in _load_table():
                if e.sg_number == number and e.extension == extension:
                    entry = e
                    break
            else:
                raise ValueError(f"no entry for SG {number} extension {extension!r}")
        else:
            entry = _by_number()[number]
        return cls._from_entry(entry)

    @classmethod
    def from_hm(cls, hm: str) -> "SpaceGroup":
        key = _normalize_hm(hm)
        if key not in _by_hm():
            raise ValueError(f"unknown Hermann-Mauguin symbol: {hm!r}")
        return cls._from_entry(_by_hm()[key])

    @classmethod
    def from_hall(cls, hall: str) -> "SpaceGroup":
        key = hall.strip()
        if key not in _by_hall():
            # Support custom Hall symbols (not in sginfo table) as well.
            return cls._build_from_hall(key, sg_number=0)
        return cls._from_entry(_by_hall()[key])

    @classmethod
    def _from_entry(cls, entry: _SgEntry) -> "SpaceGroup":
        sg = cls._build_from_hall(entry.hall_symbol, sg_number=entry.sg_number)
        sg.hm_symbol = entry.labels.split("=")[0].strip()
        sg.extension = entry.extension
        sg.crystal_system = entry.crystal_system
        sg.laue_class = entry.laue_class
        return sg

    @classmethod
    def _build_from_hall(cls, hall: str, sg_number: int) -> "SpaceGroup":
        centering, generators, _origin = parse_hall(hall)
        ops = expand_group(generators, LATTICE_TRANSLATIONS[centering])
        # Canonical sort for reproducibility
        ops_sorted = tuple(sorted(ops, key=lambda o: (o.R, o.t)))
        return cls(
            number=sg_number,
            hall_symbol=hall.strip(),
            hm_symbol="",
            extension="",
            crystal_system=crystal_system_for(sg_number) if sg_number else "",
            laue_class=laue_class_for(sg_number) if sg_number else "",
            centering=centering,
            operations=ops_sorted,
        )

    # ----------------------------------------------------------------- queries
    @property
    def order(self) -> int:
        return len(self.operations)

    def symmetry_operations(self) -> Tuple[SymOp, ...]:
        return self.operations

    def is_centrosymmetric(self) -> bool:
        inv_R = (-1, 0, 0, 0, -1, 0, 0, 0, -1)
        return any(op.R == inv_R for op in self.operations)

    def lattice_translations(self) -> Tuple[Tuple[int, int, int], ...]:
        return LATTICE_TRANSLATIONS[self.centering]

    # -------------------------------------------------- reciprocal-space tools
    def is_systematically_absent(self, h: int, k: int, l: int) -> bool:
        """Return True if (hkl) is systematically absent under this space group.

        Implementation: a reflection is absent iff there exists a symmetry op
        (R,t) such that R^T (hkl) == (hkl) but exp(-2πi (hkl)·t) != 1.  We
        evaluate (hkl)·t modulo STBF using integer arithmetic.
        """
        if (h, k, l) == (0, 0, 0):
            return False
        for op in self.operations:
            if op.apply_hkl(h, k, l) == (h, k, l):
                phase = (h * op.t[0] + k * op.t[1] + l * op.t[2]) % STBF
                if phase != 0:
                    return True
        return False

    def equivalent_hkls(self, h: int, k: int, l: int) -> List[Tuple[int, int, int]]:
        """All Friedel-included Laue-equivalent reflections (set, sorted)."""
        seen: set[Tuple[int, int, int]] = set()
        for op in self.operations:
            seen.add(op.apply_hkl(h, k, l))
        # Friedel pair always belongs in the X-ray Laue class (centric structure factor).
        for hkl in list(seen):
            seen.add((-hkl[0], -hkl[1], -hkl[2]))
        return sorted(seen)

    def multiplicity(self, h: int, k: int, l: int) -> int:
        return len(self.equivalent_hkls(h, k, l))

    def asu_representative(self, h: int, k: int, l: int) -> Tuple[int, int, int]:
        """Pick a canonical representative for the Laue equivalence class."""
        return max(self.equivalent_hkls(h, k, l))


def list_space_groups() -> List[Tuple[int, str, str]]:
    """Return [(number, hall_symbol, hm_short)] for every entry in the table."""
    out = []
    for e in _load_table():
        hm = e.labels.split("=")[0].strip()
        out.append((e.sg_number, e.hall_symbol, hm))
    return out


def centring_allowed(hkl, space_group) -> "np.ndarray":
    """Which reflections survive the LATTICE CENTRING absences.

    Returns a boolean mask over ``hkl`` (an ``(n, 3)`` integer array). This is
    the centring rule ONLY -- the absences that follow from the lattice's
    centring translations, independent of any atomic basis. Glide- and
    screw-absences are not included; for those, and for basis-driven weakness,
    compute ``|F|^2`` from a :class:`~midas_hkls.Crystal`.

    The rule is DERIVED, not tabulated: a reflection survives a centring
    translation ``t`` exactly when ``h . t`` is an integer, so with the table
    held in units of 1/12 the test is ``(h . t) % 12 == 0`` for every vector in
    ``LATTICE_TRANSLATIONS[centering]``. That reproduces the familiar forms --
    F: h, k, l all the same parity; I: h + k + l even; C: h + k even;
    R (obverse hexagonal): -h + k + l = 3n -- and, unlike a hand-written table,
    it cannot be right for some centrings and wrong for others.

    WHY THIS IS HERE. ``midas_defect.rows`` carried its own hand-written version
    keyed on lists of space-group numbers. It applied the **I** rule (h+k+l even)
    to an **F**-centred cell, which is wrong in BOTH directions: on La3Ni2O7 S5
    it credited 726 of 2161 reflections (33.6 %) that F forbids, while being
    blind to the all-odd families F allows -- 0 of 2161 claimed reflections were
    all-odd, against 436 of 805 once fixed. Three further call sites gated on
    ``space_group_number == 139`` and so applied no centring rule at all to any
    other group. A rule this easy to get wrong belongs in one place, derived.

    Parameters
    ----------
    hkl : (n, 3) array_like of int
    space_group : int, str or SpaceGroup
        A number (1-230), a Hermann-Mauguin symbol, or a built
        :class:`SpaceGroup`.

    Examples
    --------
    >>> import numpy as np
    >>> h = np.array([[1, 1, 1], [1, 1, 0], [2, 0, 0], [2, 1, 0]])
    >>> centring_allowed(h, 225).tolist()        # F, all same parity
    [True, False, True, False]
    >>> centring_allowed(h, 229).tolist()        # I, h+k+l even
    [False, True, True, False]
    >>> centring_allowed(h, 221).tolist()        # P, nothing forbidden
    [True, True, True, True]
    """
    import numpy as np

    if isinstance(space_group, SpaceGroup):
        sg = space_group
    elif isinstance(space_group, str):
        sg = SpaceGroup.from_hm(space_group)
    else:
        sg = SpaceGroup.from_number(int(space_group))

    h = np.asarray(hkl, dtype=np.int64)
    if h.ndim != 2 or h.shape[1] != 3:
        raise ValueError(f"hkl must be (n, 3); got {h.shape}")
    ok = np.ones(len(h), dtype=bool)
    for t in LATTICE_TRANSLATIONS[sg.centering]:
        if not any(t):
            continue
        ok &= (h @ np.asarray(t, dtype=np.int64)) % STBF == 0
    return ok
