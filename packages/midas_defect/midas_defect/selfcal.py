"""Self-calibrate the detector from indexed crystals. No calibrant exposure.

Ported 2026-09-10 from the La3Ni2O7 project's ``step28_selfcal.py``, rebuilt on
:func:`midas_defect.geometry.pixel_to_qlab` (the MIDAS tilt + distortion model)
instead of the project's own flat-panel rotation.

Two constraints are already in the data, and no external standard improves on
either:

**The crystal's own symmetry.** In a layered (Ruddlesden-Popper) crystal c is
perpendicular to the ab plane in the tetragonal parent AND in the orthorhombic
child, so alpha = beta = 90 EXACTLY, whatever the in-plane distortion is. That is
two exact constraints per domain, for free. **gamma stays FREE**, because gamma
is the observable: an Fmmm distortion of the subcell IS a gamma shear. Imposing
alpha = beta = 90 is not circular; imposing a = b would be.

**One detector, several crystals.** Domains recorded in one exposure share one
geometry and have independent orientations and cells, so a joint fit
over-determines the geometry with data no single-crystal fit can use.

What is refined: shared beam centre (2), distance and two tilts (``ty``, ``tz``);
per domain the orientation (3), a, b and gamma, with c held at the value passed
in -- c pins the length scale, which is otherwise degenerate with the distance.
``tx`` is NOT refined by default: over a short rotation it is absorbed by the
crystal orientation (to 0.021 deg over 12 deg of omega on S5), so it is not
measurable this way.

**The check is the residual, never the angles** -- alpha and beta are pinned by
construction. Measured on 2604 (two domains, project record 2026-08-25): total
tilt 0.402 deg recovered from the crystals alone against the PONI's independently
calibrated 0.412 deg; rms |dG| 0.00194 -> 0.00105 1/A; family-level radial
systematic on a/b-BLIND families 0.344 % -> 0.129 %; domain 1 indexed 42 -> 46.
A residual that does not improve means geometry is not the explanation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import Geometry, pixel_to_qlab, qlab_to_qsample

__all__ = ["CrystalSpots", "SelfCalResult", "selfcalibrate_from_crystals"]

_GEOM_FIELDS = ("bcy_px", "bcz_px", "lsd_um", "tx_deg", "ty_deg", "tz_deg")


@dataclass
class CrystalSpots:
    """One indexed domain: its spots, their hkl, and a starting orientation and cell."""
    row: np.ndarray
    col: np.ndarray
    omega_deg: np.ndarray
    hkl: np.ndarray                    # (n, 3)
    U: np.ndarray                      # (3, 3) sample-frame orientation
    a: float
    b: float
    c: float                           # HELD: it sets the length scale
    gamma: float = 90.0


@dataclass
class SelfCalResult:
    geometry: Geometry
    cells: List[dict]                  # per domain: U, a, b, c, alpha, beta, gamma
    rms_before: float                  # |dG|, 1/A (q = 2 pi / d)
    rms_after: float
    per_domain_rms: List[Tuple[float, float]]
    n_spots: int
    free: Tuple[str, ...]
    success: bool
    message: str = ""
    residual: List[np.ndarray] = field(default_factory=list)

    def __str__(self) -> str:
        g = self.geometry
        return (f"self-calibration on {len(self.cells)} domain(s), {self.n_spots} spots: rms |dG| "
                f"{self.rms_before:.5f} -> {self.rms_after:.5f} 1/A; bc (row {g.bcz_px:.3f}, col "
                f"{g.bcy_px:.3f}), Lsd {g.lsd_um:.1f} um, tilts ty {g.ty_deg:+.4f} tz {g.tz_deg:+.4f} deg")


def _B(a, b, c, alpha, beta, gamma):
    from midas_hkls import Lattice
    lat = Lattice(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    return np.asarray(lat.reciprocal_cartesian_vectors(), float).T * 2.0 * math.pi


def _q_sample(geom, d, omega_sign):
    ql = pixel_to_qlab(np.asarray(d.row, float), np.asarray(d.col, float), geom, device="cpu", dtype="float64")
    w = torch.deg2rad(torch.as_tensor(omega_sign * np.asarray(d.omega_deg, float), dtype=ql.dtype))
    return qlab_to_qsample(ql, w).detach().cpu().numpy().astype(np.float64)


def selfcalibrate_from_crystals(domains: Sequence[CrystalSpots], geom: Geometry, *,
                                free: Sequence[str] = ("bcy_px", "bcz_px", "lsd_um", "ty_deg", "tz_deg"),
                                omega_sign: int = +1, max_nfev: int = 2000) -> SelfCalResult:
    """Refine shared detector geometry and per-domain (U, a, b, gamma) against indexed spots.

    ``domains`` carry FIXED hkl assignments -- index first (``index_from_cloud``,
    ``match_mask``), refine here, re-match on the refined geometry, and repeat
    until the assignment stops changing, as the original did for six rounds.
    Residual: ``q_sample(spot; geometry) - U B(a, b, c, 90, 90, gamma) hkl`` in
    1/A with q = 2 pi / d. ``omega_sign`` must be the one ``resolve_conventions``
    decided.
    """
    free = tuple(free)
    bad = [f for f in free if f not in _GEOM_FIELDS]
    if bad:
        raise ValueError(f"free geometry fields must be among {_GEOM_FIELDS}; got {bad}")
    if not domains:
        raise ValueError("need at least one indexed domain")
    doms = list(domains)
    for i, d in enumerate(doms):
        n = len(np.asarray(d.row))
        if not (len(np.asarray(d.col)) == len(np.asarray(d.omega_deg)) == np.asarray(d.hkl).shape[0] == n):
            raise ValueError(f"domain {i}: row, col, omega_deg and hkl lengths differ")
        if n < 6:
            raise ValueError(f"domain {i}: {n} spots cannot support a 6-parameter per-domain fit")
    n_free = len(free)
    # scale each parameter to an O(1) step: px, um, deg, rad, A, deg
    gscale = {"bcy_px": 1.0, "bcz_px": 1.0, "lsd_um": 100.0, "tx_deg": 0.1, "ty_deg": 0.1, "tz_deg": 0.1}
    x_scale = [gscale[f] for f in free] + [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 0.05] * len(doms)
    x0 = [float(getattr(geom, f)) for f in free]
    for d in doms:
        x0 += [0.0, 0.0, 0.0, float(d.a), float(d.b), float(d.gamma)]
    x0 = np.asarray(x0, float)

    def unpack(x):
        g = replace(geom, **{f: float(v) for f, v in zip(free, x[:n_free])})
        per = []
        for i, d in enumerate(doms):
            k = n_free + 6 * i
            U = Rotation.from_rotvec(x[k:k + 3]).as_matrix() @ np.asarray(d.U, float)
            per.append((U, x[k + 3], x[k + 4], x[k + 5]))
        return g, per

    def resid_list(x):
        g, per = unpack(x)
        out = []
        for d, (U, a, b, ga) in zip(doms, per):
            qs = _q_sample(g, d, omega_sign)
            pred = (U @ _B(a, b, float(d.c), 90.0, 90.0, ga) @ np.asarray(d.hkl, float).T).T
            out.append(qs - pred)
        return out

    def rms(rl):
        allr = np.concatenate([np.linalg.norm(r, axis=1) for r in rl])
        return float(np.sqrt(np.mean(allr ** 2)))

    r0 = resid_list(x0)
    sol = least_squares(lambda x: np.concatenate([r.ravel() for r in resid_list(x)]), x0,
                        x_scale=np.asarray(x_scale), method="trf", max_nfev=max_nfev)
    r1 = resid_list(sol.x)
    g, per = unpack(sol.x)
    cells = [dict(U=U, a=float(a), b=float(b), c=float(d.c), alpha=90.0, beta=90.0, gamma=float(ga))
             for d, (U, a, b, ga) in zip(doms, per)]
    per_rms = [(rms([a0]), rms([a1])) for a0, a1 in zip(r0, r1)]
    return SelfCalResult(geometry=g, cells=cells, rms_before=rms(r0), rms_after=rms(r1),
                         per_domain_rms=per_rms, n_spots=int(sum(len(r) for r in r1)), free=free,
                         success=bool(sol.success), message=str(sol.message), residual=r1)
