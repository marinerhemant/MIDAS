"""Ring table builder — thin wrapper over midas_hkls + per-detector geometry.

Supports one calibrant or several on the same exposure (e.g. a CeO2 + LaB6
mixture).  A multi-phase table is just the per-phase tables concatenated and
re-sorted by radius, with a ``phase_idx`` column recording which calibrant
produced each row.  Nothing downstream of the ring table needs to know about
phases: the calibration residual only ever asks a fitted point for its
expected 2θ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from midas_hkls import Lattice, SpaceGroup, generate_hkls

from .params import CalibrationParams


# Two reflections of the SAME phase can have identical d-spacings without being
# symmetry-equivalent -- LaB6 (300)/(221) (9 = 9), (410)/(322) (17 = 17),
# CeO2 (511)/(333) (27 = 27), (600)/(442) (36 = 36).  midas_hkls gives them
# different ring_nr because they are different families, but they land on ONE
# Debye-Scherrer ring at exactly one radius.  Left unmerged they look like a
# zero-separation "doublet" to any blend-detection rule and get excluded, which
# throws away perfectly good rings.  Merge them by d-spacing instead.
DEFAULT_D_DEDUP_REL_TOL = 1e-6


@dataclass
class RingTable:
    ring_nr: np.ndarray         # [n_rings] int
    h: np.ndarray
    k: np.ndarray
    l: np.ndarray
    d_spacing: np.ndarray       # Å
    two_theta_deg: np.ndarray
    multiplicity: np.ndarray    # int
    r_ideal_px: np.ndarray      # px
    # Multi-phase bookkeeping.  ``phase_idx[i]`` indexes ``phase_names``.
    # A single-calibrant table leaves phase_idx as all-zeros and phase_names
    # as a 1-tuple, so single-phase callers need no changes.
    phase_idx: Optional[np.ndarray] = None
    phase_names: Tuple[str, ...] = ()
    # Extra (h, k, l) that were merged into each row by the d-spacing dedup.
    hkl_aliases: List[List[Tuple[int, int, int]]] = field(default_factory=list)

    ARRAY_FIELDS = ("ring_nr", "h", "k", "l", "d_spacing", "two_theta_deg",
                    "multiplicity", "r_ideal_px", "phase_idx")

    def __len__(self) -> int:
        return len(self.ring_nr)

    @property
    def n_phases(self) -> int:
        return max(len(self.phase_names), 1)

    def phase_of(self, i: int) -> str:
        """Name of the calibrant that produced row ``i``."""
        if self.phase_idx is None or not self.phase_names:
            return self.phase_names[0] if self.phase_names else "phase0"
        return self.phase_names[int(self.phase_idx[i])]

    def phase_mask(self, name: str) -> np.ndarray:
        """Boolean mask selecting the rows belonging to calibrant ``name``."""
        if not self.phase_names:
            raise ValueError("ring table carries no phase names")
        try:
            idx = self.phase_names.index(name)
        except ValueError:
            raise KeyError(
                f"unknown phase {name!r}; table has {list(self.phase_names)}"
            ) from None
        if self.phase_idx is None:
            return np.ones(len(self), dtype=bool) if idx == 0 \
                else np.zeros(len(self), dtype=bool)
        return self.phase_idx == idx

    def select(self, keep: np.ndarray) -> "RingTable":
        """Return a new table with only the rows where ``keep`` is True."""
        keep = np.asarray(keep, dtype=bool)
        kw = {}
        for f in self.ARRAY_FIELDS:
            v = getattr(self, f)
            kw[f] = None if v is None else v[keep]
        aliases = ([a for a, kp in zip(self.hkl_aliases, keep) if kp]
                   if self.hkl_aliases else [])
        return RingTable(phase_names=self.phase_names, hkl_aliases=aliases, **kw)


def _phase_specs(params: CalibrationParams) -> List[Dict]:
    """Normalise a params object into a list of phase specs.

    Uses ``params.Phases`` when present (list of dicts with ``sg`` and
    ``lattice``, optionally ``name``); otherwise falls back to the scalar
    ``SpaceGroup`` / ``LatticeConstant`` pair, which is the single-calibrant
    case and stays bit-identical to the pre-multi-phase behaviour.
    """
    phases = getattr(params, "Phases", None)
    if not phases:
        return [{"name": f"sg{params.SpaceGroup}",
                 "sg": int(params.SpaceGroup),
                 "lattice": tuple(params.LatticeConstant)}]
    out = []
    for i, p in enumerate(phases):
        if not isinstance(p, dict):
            raise TypeError(
                f"params.Phases[{i}] must be a dict with 'sg' and 'lattice'; "
                f"got {type(p)}")
        try:
            sg = int(p["sg"])
            lat = tuple(float(x) for x in p["lattice"])
        except KeyError as e:
            raise ValueError(
                f"params.Phases[{i}] is missing required key {e}; "
                "required: 'sg' (space-group number), 'lattice' "
                "(a, b, c, alpha, beta, gamma)") from None
        if len(lat) != 6:
            raise ValueError(
                f"params.Phases[{i}]['lattice'] must have 6 entries "
                f"(a, b, c, alpha, beta, gamma); got {len(lat)}")
        out.append({"name": str(p.get("name", f"phase{i}")),
                    "sg": sg, "lattice": lat})
    return out


def _dedup_by_d(rows: List[dict], rel_tol: float) -> List[dict]:
    """Merge same-phase rows whose d-spacings agree to ``rel_tol``.

    Multiplicities add (both families scatter into the one ring); the extra
    (h, k, l) are kept as aliases so the provenance is not lost.
    """
    if rel_tol <= 0 or not rows:
        return rows
    rows = sorted(rows, key=lambda r: (r["phase"], -r["d"]))
    out: List[dict] = []
    for r in rows:
        prev = out[-1] if out else None
        if (prev is not None and prev["phase"] == r["phase"]
                and abs(r["d"] - prev["d"]) <= rel_tol * max(prev["d"], 1e-12)):
            prev["multiplicity"] += r["multiplicity"]
            prev["aliases"].append(r["hkl"])
            continue
        out.append(dict(r, aliases=list(r.get("aliases", []))))
    return out


def build_ring_table(
    params: CalibrationParams,
    *,
    dedup_d_rel_tol: float = DEFAULT_D_DEDUP_REL_TOL,
) -> RingTable:
    """Generate the calibrant ring table at the current detector geometry.

    Multi-phase: set ``params.Phases`` to a list of
    ``{"name": ..., "sg": ..., "lattice": (a, b, c, al, be, ga)}`` and the
    returned table carries every phase's rings, sorted by radius, tagged with
    ``phase_idx``.  With ``Phases`` unset the behaviour is unchanged.
    """
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    max_R_um = params.MaxRingRad * px
    # A ray scattered through 2θ lands at radius R = Lsd·tan(2θ), so the
    # inverse is 2θ = arctan(R/Lsd) — the factor 2 is already inside "2θ".
    two_theta_max = np.degrees(np.arctan(max_R_um / params.Lsd))

    specs = _phase_specs(params)
    rows: List[dict] = []
    for pi, spec in enumerate(specs):
        refs = generate_hkls(
            SpaceGroup.from_number(spec["sg"]), Lattice(*spec["lattice"]),
            wavelength_A=params.Wavelength,
            two_theta_max_deg=two_theta_max,
        )
        for r in refs:
            rows.append(dict(phase=pi, ring_nr=int(r.ring_nr),
                             hkl=(int(r.h), int(r.k), int(r.l)),
                             d=float(r.d_spacing),
                             tt=float(r.two_theta_deg),
                             multiplicity=int(r.multiplicity), aliases=[]))
    if not rows:
        raise RuntimeError(
            "No reflections within max 2θ — check geometry / lattice / wavelength")

    rows = _dedup_by_d(rows, dedup_d_rel_tol)
    rows.sort(key=lambda r: r["tt"])

    n = len(rows)
    rt = RingTable(
        ring_nr=np.array([r["ring_nr"] for r in rows], dtype=np.int32),
        h=np.array([r["hkl"][0] for r in rows], dtype=np.int32),
        k=np.array([r["hkl"][1] for r in rows], dtype=np.int32),
        l=np.array([r["hkl"][2] for r in rows], dtype=np.int32),
        d_spacing=np.array([r["d"] for r in rows], dtype=np.float64),
        two_theta_deg=np.array([r["tt"] for r in rows], dtype=np.float64),
        multiplicity=np.array([r["multiplicity"] for r in rows], dtype=np.int32),
        r_ideal_px=np.empty(n, dtype=np.float64),
        phase_idx=np.array([r["phase"] for r in rows], dtype=np.int64),
        phase_names=tuple(s["name"] for s in specs),
        hkl_aliases=[list(r["aliases"]) for r in rows],
    )
    rt.r_ideal_px[:] = params.Lsd * np.tan(np.radians(rt.two_theta_deg)) / px

    # Radius filters.  MaxRingRad is enforced here as well as through
    # two_theta_max: the 2θ cap is evaluated at the *current* Lsd, so as the
    # refinement moves Lsd the admitted radius drifts.  Filtering r_ideal_px
    # directly makes the bound exact and mirrors the MinRingRad treatment.
    keep = np.ones(n, dtype=bool)
    if params.MinRingRad > 0:
        keep &= rt.r_ideal_px >= params.MinRingRad
    if params.MaxRingRad > 0:
        keep &= rt.r_ideal_px <= params.MaxRingRad
    if not keep.all():
        rt = rt.select(keep)
    return rt


def flag_blended_rings(
    rt: RingTable,
    *,
    min_separation_px: float = 0.0,
    cross_phase_only: bool = False,
) -> np.ndarray:
    """Per-ring mask: True where a ring has a neighbour closer than the cut.

    Unlike :func:`max_resolvable_ring_radius_px`, which returns a single
    *radial cutoff* and therefore discards every ring beyond the first close
    pair, this flags the offending rings individually.  That is what a
    multi-phase table needs: a CeO2/LaB6 collision at mid-radius should cost
    two rings, not every ring outside it.

    ``cross_phase_only`` restricts the flag to collisions between DIFFERENT
    calibrants, leaving genuine same-phase doublets to the doublet co-fitter
    (which can model them) rather than excluding them.
    """
    n = len(rt)
    flagged = np.zeros(n, dtype=bool)
    if n < 2 or min_separation_px <= 0:
        return flagged
    order = np.argsort(rt.r_ideal_px)
    R = rt.r_ideal_px[order]
    pidx = (rt.phase_idx[order] if rt.phase_idx is not None
            else np.zeros(n, dtype=np.int64))
    for a in range(n - 1):
        b = a + 1
        if R[b] - R[a] >= min_separation_px:
            continue
        if cross_phase_only and pidx[a] == pidx[b]:
            continue
        flagged[order[a]] = True
        flagged[order[b]] = True
    return flagged


def drop_blended_rings(
    rt: RingTable,
    *,
    min_separation_px: float = 0.0,
    cross_phase_only: bool = False,
) -> Tuple[RingTable, int]:
    """Return ``(filtered_table, n_dropped)`` using :func:`flag_blended_rings`."""
    flagged = flag_blended_rings(rt, min_separation_px=min_separation_px,
                                 cross_phase_only=cross_phase_only)
    if not flagged.any():
        return rt, 0
    return rt.select(~flagged), int(flagged.sum())


def max_resolvable_ring_radius_px(
    rt: RingTable,
    *,
    min_separation_px: float = 8.0,
    r_min_px: float = 0.0,
) -> tuple:
    """Largest ring radius out to which the rings are actually separable.

    Returns ``(radius_px, n_rings)``: the radius of the outermost ring such
    that every adjacent pair inside ``[r_min_px, radius_px]`` is at least
    ``min_separation_px`` apart, and how many rings that leaves.  Returns
    ``(None, n)`` when fewer than two rings are in range.

    Why this exists
    ---------------
    A ring table is generated from crystallography and says nothing about
    whether the detector can tell two rings apart.  For a dense calibrant at
    short wavelength the table can be far denser than the pixel pitch
    supports, and the peak fitter then fits unresolved blends: the fit
    "succeeds", the residuals are structureless, and the geometry is quietly
    biased.  Concretely, CeO2 at 0.116 A (107 keV) on a 150 um detector at
    Lsd = 330 mm puts **546 rings inside 1420 px -- one every ~2 px**, and
    fitting them gives +/-2000 ue residuals.  Capping at the radius this
    returns took that dataset from 300-600 ue to 13-36 ue per image.

    Because ring separation in pixels scales with Lsd, the binding pair is
    set by the SHORTEST distance in a scan.  For CeO2 at 107 keV it is
    (331)/(420) at 2theta = 5.353/5.492 deg, which needs Lsd >~ 490 mm just
    to separate by 8 px -- so the usable set caps near 2theta ~ 6.4 deg.

    Note this is a necessary, not sufficient, condition: it uses ideal ring
    radii and a fixed separation, and does not know peak widths or structure
    factors.  Weak rings inside the returned radius may still be unusable.

    For a mixed-calibrant exposure prefer :func:`drop_blended_rings`, which
    removes the colliding rings instead of truncating at the first collision.
    """
    R = np.asarray(rt.r_ideal_px, dtype=float)
    order = np.argsort(R)
    R = R[order]
    inside = R[R >= float(r_min_px)]
    if inside.size < 2:
        return None, int(inside.size)
    gaps = np.diff(inside)
    bad = np.nonzero(gaps < float(min_separation_px))[0]
    cut = float(inside[bad[0]]) if bad.size else float(inside[-1])
    return cut, int((inside <= cut).sum())


__all__ = [
    "RingTable",
    "build_ring_table",
    "flag_blended_rings",
    "drop_blended_rings",
    "max_resolvable_ring_radius_px",
    "DEFAULT_D_DEDUP_REL_TOL",
]
