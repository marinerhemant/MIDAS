"""Ring table builder — thin wrapper over midas_hkls + per-detector geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from midas_hkls import Lattice, SpaceGroup, generate_hkls

from .params import CalibrationParams


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

    def __len__(self) -> int:
        return len(self.ring_nr)


def build_ring_table(params: CalibrationParams) -> RingTable:
    """Generate the calibrant ring table at the current detector geometry."""
    sg = SpaceGroup.from_number(params.SpaceGroup)
    lat = Lattice(*params.LatticeConstant)

    # Maximum 2θ from MaxRingRad and Lsd (use mean pixel size).
    # A ray scattered through 2θ lands at radius R = Lsd·tan(2θ), so the
    # inverse is 2θ = arctan(R/Lsd) — the factor 2 is already inside "2θ".
    # Multiplying by 2 here admitted rings out to ~2× MaxRingRad (2.09× at
    # MaxRingRad=1420 px on a 2880² 150 µm detector at Lsd=1 m), i.e. well
    # into the detector corners the caller asked to exclude.
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    max_R_um = params.MaxRingRad * px
    two_theta_max = np.degrees(np.arctan(max_R_um / params.Lsd))

    refs = generate_hkls(
        sg, lat,
        wavelength_A=params.Wavelength,
        two_theta_max_deg=two_theta_max,
    )
    if not refs:
        raise RuntimeError("No reflections within max 2θ — check geometry / lattice / wavelength")

    n = len(refs)
    rt = RingTable(
        ring_nr=np.array([r.ring_nr for r in refs], dtype=np.int32),
        h=np.array([r.h for r in refs], dtype=np.int32),
        k=np.array([r.k for r in refs], dtype=np.int32),
        l=np.array([r.l for r in refs], dtype=np.int32),
        d_spacing=np.array([r.d_spacing for r in refs], dtype=np.float64),
        two_theta_deg=np.array([r.two_theta_deg for r in refs], dtype=np.float64),
        multiplicity=np.array([r.multiplicity for r in refs], dtype=np.int32),
        r_ideal_px=np.empty(n, dtype=np.float64),
    )
    rt.r_ideal_px[:] = params.Lsd * np.tan(np.radians(rt.two_theta_deg)) / px

    # Radius filters.  MaxRingRad is enforced here as well as through
    # two_theta_max: the 2θ cap is evaluated at the *current* Lsd, so as the
    # refinement moves Lsd the admitted radius drifts.  Filtering r_ideal_px
    # directly makes the bound exact and mirrors the MinRingRad treatment.
    keep = np.ones(len(rt.r_ideal_px), dtype=bool)
    if params.MinRingRad > 0:
        keep &= rt.r_ideal_px >= params.MinRingRad
    if params.MaxRingRad > 0:
        keep &= rt.r_ideal_px <= params.MaxRingRad
    if not keep.all():
        for f in ("ring_nr", "h", "k", "l", "d_spacing", "two_theta_deg", "multiplicity", "r_ideal_px"):
            setattr(rt, f, getattr(rt, f)[keep])
    return rt


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
