"""Assigning observed rings to reflections of a candidate phase.

The step between "there is a peak at 115 px" and "that peak is U3O8 (hkl)".
Without it, a d-spacing map is a picture of a number rather than of a lattice.

Uses ``midas_hkls.generate_hkls`` for the reflection list rather than
computing d-spacings from a cell here: space-group extinctions are exactly
where a hand-rolled version quietly predicts rings that cannot exist, and then
matches an observed peak to one of them.

The output is a match table with a residual per ring and an overall score, so
a phase that does not fit says so numerically rather than being argued about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

__all__ = ["PhaseCandidate", "RingMatch", "IndexResult", "index_rings",
           "ALPHA_U3O8", "CEO2"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseCandidate:
    """A phase to test against the observed rings."""

    name: str
    space_group: int
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    reference: str = ""


#: alpha-U3O8, orthorhombic C2mm (#38). Loong et al.; the common room-T form.
ALPHA_U3O8 = PhaseCandidate(
    name="alpha-U3O8", space_group=38,
    a=6.716, b=11.960, c=4.147,
    reference="orthorhombic C2mm, a=6.716 b=11.960 c=4.147 A",
)

#: CeO2, fluorite Fm-3m (#225). The CALIBRANT -- the 2022 ps_dt.txt files give
#: its cell, not the sample's, which is why the sample must be tested
#: separately rather than assumed from the parameter file.
CEO2 = PhaseCandidate(
    name="CeO2 (calibrant)", space_group=225,
    a=5.41165, b=5.41165, c=5.41165,
    reference="fluorite Fm-3m, a=5.41165 A (from ps_dt.txt)",
)


@dataclass
class RingMatch:
    """One observed ring, and the best reflection found for it."""

    d_obs_a: float
    radius_px: float
    hkl: tuple[int, int, int] | None
    d_calc_a: float | None
    residual_ppm: float | None

    @property
    def matched(self) -> bool:
        return self.hkl is not None


@dataclass
class IndexResult:
    """Match table for one phase."""

    phase: PhaseCandidate
    matches: list[RingMatch] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return sum(m.matched for m in self.matches)

    @property
    def rms_residual_ppm(self) -> float:
        r = [abs(m.residual_ppm) for m in self.matches if m.matched]
        return float(np.sqrt(np.mean(np.square(r)))) if r else float("inf")

    def describe(self) -> str:
        lines = [f"{self.phase.name}  ({self.phase.reference})",
                 f"  matched {self.n_matched}/{len(self.matches)} rings, "
                 f"rms residual {self.rms_residual_ppm:.0f} ppm",
                 f"  {'R (px)':>8} {'d_obs':>8} {'hkl':>12} {'d_calc':>8} {'resid':>10}"]
        for m in self.matches:
            if m.matched:
                hkl = f"({m.hkl[0]} {m.hkl[1]} {m.hkl[2]})"
                lines.append(f"  {m.radius_px:8.2f} {m.d_obs_a:8.3f} {hkl:>12} "
                             f"{m.d_calc_a:8.3f} {m.residual_ppm:9.0f}p")
            else:
                lines.append(f"  {m.radius_px:8.2f} {m.d_obs_a:8.3f} "
                             f"{'--':>12} {'--':>8} {'--':>10}")
        return "\n".join(lines)


def index_rings(
    d_obs_a,
    phase: PhaseCandidate,
    *,
    radii_px=None,
    tolerance_ppm: float = 20_000.0,
    wavelength_a: float = 0.136994,
) -> IndexResult:
    """Match observed d-spacings to a phase's allowed reflections.

    Parameters
    ----------
    tolerance_ppm : float
        Maximum |d_obs - d_calc| / d_calc for a match, in parts per million.
        The default 20 000 ppm (2%) is deliberately loose: this is asking
        "could this be that phase at all", not measuring strain. Tighten it
        once a phase is established.

    Notes
    -----
    A high match count is necessary, not sufficient. A cell with many allowed
    reflections will match almost any peak list by chance, so compare the
    *residuals* and the number of PREDICTED strong rings that were NOT observed
    before believing an assignment.
    """
    try:
        from midas_hkls import Lattice, SpaceGroup, generate_hkls
    except ImportError as exc:
        raise ImportError(
            "ring indexing needs midas-hkls. Install with "
            "`pip install midas-dt[indexing]`."
        ) from exc

    d_obs = np.atleast_1d(np.asarray(d_obs_a, dtype=np.float64))
    radii = (np.atleast_1d(np.asarray(radii_px, dtype=np.float64))
             if radii_px is not None else np.full_like(d_obs, np.nan))

    lat = Lattice(phase.a, phase.b, phase.c, phase.alpha, phase.beta, phase.gamma)
    sg = SpaceGroup.from_number(phase.space_group)
    refl = generate_hkls(sg, lat, wavelength_A=wavelength_a,
                         d_min=float(np.min(d_obs)) * 0.8,
                         d_max=float(np.max(d_obs)) * 1.2)
    if not refl:
        log.warning("%s: no reflections in the observed d range", phase.name)
        return IndexResult(phase=phase, matches=[
            RingMatch(float(d), float(r), None, None, None)
            for d, r in zip(d_obs, radii)])

    d_calc = np.array([r.d_spacing for r in refl], dtype=np.float64)
    hkls = [(int(r.h), int(r.k), int(r.l)) for r in refl]

    matches = []
    for d, rad in zip(d_obs, radii):
        resid = (d - d_calc) / d_calc * 1e6
        j = int(np.argmin(np.abs(resid)))
        if abs(resid[j]) <= tolerance_ppm:
            matches.append(RingMatch(float(d), float(rad), hkls[j],
                                     float(d_calc[j]), float(resid[j])))
        else:
            matches.append(RingMatch(float(d), float(rad), None, None, None))
    return IndexResult(phase=phase, matches=matches)
