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
