"""Resolve a 9R satellite into discrete twin-related members + Ewald-artifact test.

``satellite_radial_excess`` treats the satellite as a FIELD (is there excess at
G/3 vs same-|q| controls?). This module RESOLVES it into discrete reflections and
asks whether those are two real, twin-related reflections -- the key control
against the "it's just an ISF <111>* relrod sampled twice by the Ewald sphere"
objection (Cayron, Scripta Mater. 194:113629, 2021; Miller et al.).

A single relrod swept by the rotating Ewald sphere deposits intensity
CONTINUOUSLY over omega (the rocking angle), correlated with position along the
rod. Two genuine reflections (e.g. the two twin polarities of a 9R) each give a
COMPACT rocking curve at its OWN distinct omega. Hence:

    two compact rocking curves at distinct omega  ->  two real reflections
    one broad omega-continuous streak             ->  Ewald-sampled relrod artifact

The transverse azimuth between the two members is diagnostic of WHAT they are:
~180 deg (the two +-b polarities of one <112> partial) == the twin relation;
~120 deg would instead be two independent <112> shear variants.

HONESTY / attribution (AUDIT_2026-06-23.md, :mod:`midas_defect.attribution`).
The two resolved members are labeled by POLARITY / FAULT CHARACTER only
(transverse-offset sign; relrod-bridged vs isolated). They are deliberately NOT
labeled parent-grain vs twin-lamella: host-vs-lamella is a volume/topology
(spatial) statement, and FF-HEDM has no spatial resolution inside the diffracting
volume (it needs pf-/NF-HEDM or DFXM). This module therefore emits no
``parent``/``twin`` keys -- only ``member_a``/``member_b`` with their measured
character. Use :func:`midas_defect.attribution.assert_variant_attributable`
before attaching any host/lamella identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

__all__ = ["SatelliteDoublet", "resolve_satellite_doublet"]


@dataclass
class SatelliteDoublet:
    """Result of resolving one satellite rung into discrete members.

    members : list of per-member dicts, each with
        ``q_centroid`` (3,), ``perp_offset`` (float, |q_perp|),
        ``along`` (signed projection on axis), ``along_fwhm``,
        ``omega_center``, ``omega_fwhm``, ``integrated_intensity``, ``n_voxels``.
    n_members : 0, 1 or 2.
    azimuth_deg : angle between the two members' transverse offset vectors
        (NaN if < 2 members). ~180 => twin polarities; ~120 => two shear variants.
    is_twin_polarity : azimuth within ``azimuth_tol_deg`` of 180.
    verdict : "two-reflections" | "relrod-ewald-artifact" | "single" | "ambiguous".
    """

    n_members: int
    members: list
    azimuth_deg: float
    is_twin_polarity: bool
    verdict: str
    metadata: dict = field(default_factory=dict)


def _frame(axis: NDArray[np.floating]) -> tuple:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e2 = seed - (seed @ axis) * axis
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(axis, e2)
    return axis, e2, e3


def _wfwhm(x: NDArray[np.floating], w: NDArray[np.floating], nbins: int = 40) -> float:
    """Intensity-weighted FWHM of a 1-D distribution (histogram half-max width)."""
    if x.size < 5:
        return float("nan")
    lo, hi = np.percentile(x, [1.0, 99.0])
    if hi <= lo:
        return float("nan")
    h, edges = np.histogram(x, bins=nbins, range=(lo, hi), weights=w)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    if h.max() <= 0:
        return float("nan")
    above = ctr[h >= 0.5 * h.max()]
    return float(above.max() - above.min()) if above.size >= 2 else float("nan")


def _weighted_2means_1d(u: NDArray[np.floating], w: NDArray[np.floating],
                        n_iter: int = 50) -> NDArray[np.intp]:
    """Deterministic weighted 1-D 2-means; init at the 10th/90th weighted pctiles."""
    order = np.argsort(u)
    cw = np.cumsum(w[order])
    if cw[-1] <= 0:
        return np.zeros(u.shape, dtype=np.intp)
    c = np.array([
        u[order][np.searchsorted(cw, 0.10 * cw[-1])],
        u[order][np.searchsorted(cw, 0.90 * cw[-1])],
    ], dtype=np.float64)
    lab = np.zeros(u.shape, dtype=np.intp)
    for _ in range(n_iter):
        new = (np.abs(u - c[1]) < np.abs(u - c[0])).astype(np.intp)
        if np.array_equal(new, lab):
            break
        lab = new
        for k in (0, 1):
            m = lab == k
            if w[m].sum() > 0:
                c[k] = np.average(u[m], weights=w[m])
    # order labels so member 0 is the more negative transverse position
    if c[0] > c[1]:
        lab = 1 - lab
    return lab


def resolve_satellite_doublet(
    qs: NDArray[np.floating],
    vals: NDArray[np.floating],
    omega: NDArray[np.floating],
    axis_dir: NDArray[np.floating],
    q_target: float,
    *,
    tube_parallel: float = 0.15,
    perp_max: float = 0.30,
    min_voxels: int = 30,
    compact_omega_deg: float = 5.0,
    min_omega_sep_deg: float = 3.0,
    min_member_frac: float = 0.15,
    azimuth_tol_deg: float = 25.0,
) -> SatelliteDoublet:
    """Resolve the satellite at ``q_target`` along ``axis_dir`` into <=2 members.

    Parameters
    ----------
    qs, vals : (N,3), (N,)
        Sample-frame q-vectors and per-voxel intensity (the diffuse field).
    omega : (N,)
        Per-voxel rotation angle (degrees) -- the rocking coordinate. This is the
        load-bearing input for the Ewald-artifact discriminator.
    axis_dir : (3,)
        The activated <111> direction in the sample frame (e.g. from
        :func:`detect_activated_111_axis`).
    q_target : float
        Satellite radius along the axis, e.g. ``G_111 / 3`` or ``2 G_111 / 3``.
    tube_parallel, perp_max : float
        Selection tube: |proj - q_target| < tube_parallel and |perp| < perp_max.
    compact_omega_deg, min_omega_sep_deg : float
        A member is "compact" if its omega FWHM < ``compact_omega_deg``; two
        members are "distinct" if their omega centers differ by more than
        ``min_omega_sep_deg``.
    min_member_frac : float
        A second cluster must hold at least this fraction of the tube intensity
        to count as a resolved member (else the rung is "single").
    azimuth_tol_deg : float
        Tolerance for calling the transverse azimuth a twin (180 deg) signature.

    Returns
    -------
    SatelliteDoublet
    """
    qs = np.asarray(qs, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    axis, e2, e3 = _frame(axis_dir)

    proj = qs @ axis
    perp_vec = qs - proj[:, None] * axis
    perp_mag = np.linalg.norm(perp_vec, axis=1)
    sel = (
        (np.abs(np.abs(proj) - q_target) < tube_parallel)
        & (perp_mag < perp_max)
        & (vals > 0)
        & np.isfinite(vals)
    )
    if int(sel.sum()) < min_voxels:
        return SatelliteDoublet(0, [], float("nan"), False, "single",
                                metadata={"n_in_tube": int(sel.sum())})

    q = qs[sel]
    w = vals[sel]
    om = omega[sel]
    a = q @ e2
    b = q @ e3

    # dominant transverse split axis (intensity-weighted PCA in the (e2,e3) plane)
    c0 = np.array([np.average(a, weights=w), np.average(b, weights=w)])
    X = np.stack([a - c0[0], b - c0[1]], axis=1)
    cov = (X.T * w) @ X / w.sum()
    evals, evecs = np.linalg.eigh(cov)
    u1 = evecs[:, -1]                     # major transverse direction
    u = X @ u1

    labels = _weighted_2means_1d(u, w)

    def _member(mask) -> dict:
        qm, wm, omm = q[mask], w[mask], om[mask]
        cen = np.average(qm, axis=0, weights=wm)
        pr = cen @ axis
        pv = cen - pr * axis
        return {
            "q_centroid": cen,
            "along": float(pr),
            "perp_offset": float(np.linalg.norm(pv)),
            "perp_vec": pv,
            "along_fwhm": _wfwhm(qm @ axis, wm),
            "omega_center": float(np.average(omm, weights=wm)),
            "omega_fwhm": _wfwhm(omm, wm),
            "integrated_intensity": float(wm.sum()),
            "n_voxels": int(mask.sum()),
        }

    w0, w1 = w[labels == 0].sum(), w[labels == 1].sum()
    wtot = w0 + w1
    # is the split real? second cluster must hold >= min_member_frac of the intensity
    if min(w0, w1) < min_member_frac * wtot:
        m = _member(np.ones(len(q), dtype=bool))
        return SatelliteDoublet(1, [m], float("nan"), False, "single",
                                metadata={"n_in_tube": int(sel.sum()),
                                          "minor_frac": float(min(w0, w1) / wtot)})

    ma, mb = _member(labels == 0), _member(labels == 1)

    # transverse azimuth between the two members
    va = ma["perp_vec"][None, :] @ np.stack([e2, e3]).T
    vb = mb["perp_vec"][None, :] @ np.stack([e2, e3]).T
    va, vb = va.ravel(), vb.ravel()
    if np.linalg.norm(va) > 1e-9 and np.linalg.norm(vb) > 1e-9:
        cosang = (va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
        azimuth = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    else:
        azimuth = float("nan")
    is_twin = bool(np.isfinite(azimuth) and abs(azimuth - 180.0) <= azimuth_tol_deg)

    # Ewald-artifact verdict
    compact = (
        np.isfinite(ma["omega_fwhm"]) and np.isfinite(mb["omega_fwhm"])
        and ma["omega_fwhm"] < compact_omega_deg
        and mb["omega_fwhm"] < compact_omega_deg
    )
    distinct = abs(ma["omega_center"] - mb["omega_center"]) > min_omega_sep_deg
    if compact and distinct:
        verdict = "two-reflections"
    elif (not compact) and (not distinct):
        verdict = "relrod-ewald-artifact"
    else:
        verdict = "ambiguous"

    return SatelliteDoublet(
        n_members=2,
        members=[ma, mb],
        azimuth_deg=float(azimuth),
        is_twin_polarity=is_twin,
        verdict=verdict,
        metadata={
            "n_in_tube": int(sel.sum()),
            "split_dir_e2e3": u1.tolist(),
            "omega_sep_deg": float(abs(ma["omega_center"] - mb["omega_center"])),
            "note": ("members labeled by polarity/character only; NOT parent/twin "
                     "(FF has no spatial resolution -- AUDIT_2026-06-23.md)"),
        },
    )
