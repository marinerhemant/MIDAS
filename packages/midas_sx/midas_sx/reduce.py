"""Turn integrated, hkl-assigned spots into corrected F² observations, and say what was thrown away.

``F²_obs = I / (L · P · T)``, with L, P and T from :mod:`midas_hkls.corrections` (no formula lives
here). The scale is arbitrary; merging and refinement carry their own.

Every spot ends in exactly one of two states: an observation, or a row in the rejection log with ONE
reason. The counts are returned beside the table because a table without them hides the worst failure
mode of this kind of reduction: ``keep the good ones`` silently turning a rate over 45 spots into a rate
over 2 (see ``midas_defect.ingest`` for the case that taught that).

Factor variation across the spot's own ω-profile
-------------------------------------------------
Evaluating P and T at the spot centre assumes they are constant across the rocking curve. That fails for
(i) spots near the edge of the DAC opening cone, whose tail is clipped before the nominal edge, and (ii)
high-L spots, whose rocking curve is wide and whose diffracted direction sweeps during the crossing.
Both were measured in the simulator (``validation/DESIGN.md``). The check evaluates P·T where the support
ends in ω, along the Ewald-crossing direction ``s(ω) = (k s0 + R_z(ω) q) / |.|`` of the ASSIGNED
reflection, and rejects the spot when it moves by more than ``max_factor_variation``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import math

import numpy as np
import pandas as pd

from midas_hkls import Lattice, corrections as C

from .assign import assign_hkl, calculated_crossing, qsample_of_hkl
from .geometry import Geometry, Scan, spot_directions
from .integrate import ProfileModel, integrate_box

__all__ = ["DacModel", "PolarizationModel", "Switches", "reduce_spots", "REJECT_REASONS"]

REJECT_REASONS = ("low_angle", "no_hkl", "no_crossing", "frames_edge", "detector_edge", "masked_pixels", "no_background",
                  "near_axis", "outside_opening", "factor_variation", "crowded")


@dataclass(frozen=True)
class DacModel:
    """Diamond-anvil-cell absorption model (flat slabs, sample at the culet).

    ``axis0`` is the cell axis in the lab frame at ω = 0 (normal to the culets); it rotates with the
    sample. ``t_plus_mm`` / ``t_minus_mm`` are the effective anvil thicknesses on the +axis / -axis side.
    Misspecifying any of these is exactly the probe C3 of the preregistration uses.
    """
    axis0: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    t_plus_mm: float = 2.5
    t_minus_mm: float = 2.5
    mu_per_cm: float = 0.0
    half_opening_deg: Optional[float] = None


@dataclass(frozen=True)
class PolarizationModel:
    fraction: float = 0.99
    plane_eta_deg: float = 90.0              # horizontal


@dataclass(frozen=True)
class Switches:
    """Which corrections to apply. All on is the method; switching one off is an ablation."""
    lorentz: bool = True
    polarization: bool = True
    absorption: bool = True


def _transmission(s: np.ndarray, omega_deg, dac: DacModel, axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    c = C.rotate_about_axis(np.asarray(dac.axis0, float), axis, omega_deg)
    sgn = np.sign(c[..., 0])                                     # c . s0, s0 = +X
    t_up = np.where(sgn > 0, dac.t_minus_mm, dac.t_plus_mm)      # upstream anvil: the -sign(c.s0) side
    t_dn = np.where(sgn > 0, dac.t_plus_mm, dac.t_minus_mm)
    return C.diamond_anvil_transmission(s, c, mu_per_cm=dac.mu_per_cm, t_up_mm=t_up, t_down_mm=t_dn,
                                        half_opening_deg=dac.half_opening_deg)


def _pt(s, omega_deg, dac, pol, sw: Switches):
    """P·T (the part of the correction that depends on ω and the direction)."""
    eta = np.degrees(np.arctan2(-s[..., 1], s[..., 2]))
    tth = np.degrees(np.arccos(np.clip(s[..., 0], -1.0, 1.0)))
    P = C.polarization_factor(tth, eta, pol_fraction=pol.fraction, pol_plane_eta_deg=pol.plane_eta_deg) \
        if sw.polarization else np.ones(np.shape(tth))
    T = _transmission(s, omega_deg, dac) if sw.absorption else np.ones(np.shape(tth))
    return P * T


def _profile_variation(q_sample: np.ndarray, omega_deg: float, half_width_deg: float, wavelength_A: float,
                       dac: DacModel, pol: PolarizationModel, sw: Switches) -> float:
    """max over the two support ends of |P·T(ω±hw) / P·T(ω) - 1| along the Ewald-crossing direction."""
    k = 1.0 / wavelength_A
    s0 = np.array([1.0, 0.0, 0.0])
    om = np.array([omega_deg, omega_deg - half_width_deg, omega_deg + half_width_deg])
    q = C.rotate_about_axis(np.repeat(q_sample[None, :], 3, axis=0), (0.0, 0.0, 1.0), om)
    v = k * s0 + q
    s = v / np.linalg.norm(v, axis=1, keepdims=True)
    pt = _pt(s, om, dac, pol, sw)
    if pt[0] <= 0:
        return np.inf
    return float(np.max(np.abs(pt[1:] / pt[0] - 1.0)))


def reduce_spots(spots: pd.DataFrame, raw: np.ndarray, geom: Geometry, scan: Scan, lattice: Lattice,
                 U: np.ndarray, profile: ProfileModel, *, dac: DacModel = DacModel(),
                 pol: PolarizationModel = PolarizationModel(), switches: Switches = Switches(),
                 mask: Optional[np.ndarray] = None, hkl_tol: float = 0.15,
                 max_factor_variation: float = 0.02, min_axis: float = 1e-3,
                 crowd_px: float = 12.0, min_tth_deg: float = 2.0, eval_at: str = "calculated",
                 neighbours: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Reduce found spots to F² observations.

    ``spots`` needs ``frame, row, col`` (fractional array indices, as ``find_blobs_3d`` returns them) and
    optionally ``integrated`` (kept as ``blob_sum`` for the label-sum comparison arm).
    ``spots`` may carry ``extra_px`` and ``extra_frames`` columns: the integration box of that spot is widened by them (a blended twin pair is
    one measurement whose support must cover both components; PREREGISTER A26).
    ``neighbours``: the spot list used for the crowding test (default ``spots`` itself); pass every found spot when ``spots`` is a subset
    (one crystal domain), otherwise the other domain's spots are invisible to the crowding guard.
    ``eval_at``: where L, P and T are evaluated. ``"calculated"`` (default, PREREGISTER A15) uses the Ewald crossing computed from the
    assigned hkl, U and lattice (:func:`midas_sx.assign.calculated_crossing`); ``"measured"`` uses the spot's measured centroid
    direction and omega (the v2 behaviour, kept so the two can be compared). A spot with no crossing within 2 deg of its measured omega
    is rejected as ``no_crossing`` in the calculated mode.

    Returns ``(obs, rejected, counts)``. ``counts`` has one entry per reason plus ``input`` and ``kept``;
    ``input == kept + sum(reasons)`` always.
    """
    if eval_at not in ("calculated", "measured"):
        raise ValueError("eval_at must be 'calculated' or 'measured'")
    n = len(spots)
    fr = spots["frame"].to_numpy(float)
    ro = spots["row"].to_numpy(float)
    co = spots["col"].to_numpy(float)
    omega = scan.omega_of(fr)
    s, tth, eta = spot_directions(ro, co, geom)
    hkl, hfloat, resid = assign_hkl(s, omega, lattice, U, geom.wavelength_A, geom.wedge_deg)
    omega_meas = omega.copy()
    s_meas = s.copy()
    no_cross = np.zeros(n, bool)
    if eval_at == "calculated":
        qs_all = qsample_of_hkl(hkl, lattice, U)
        for i in range(n):
            cr_ = calculated_crossing(qs_all[i], float(omega[i]), geom.wavelength_A, wedge_deg=geom.wedge_deg) if resid[i] <= hkl_tol and np.any(hkl[i]) else None
            if cr_ is None:
                no_cross[i] = True
            else:
                omega[i], s[i] = cr_
    L = C.rotation_lorentz(s, min_abs=min_axis)
    # crowding: another spot in the same frames within crowd_px contaminates the shell
    if neighbours is None:
        nfr, nro, nco = fr, ro, co
    else:                                    # crowding against ALL found spots when ``spots`` is one domain's subset (PREREGISTER A23)
        nfr, nro, nco = (neighbours[c].to_numpy(float) for c in ("frame", "row", "col"))
    dist = np.hypot(ro[:, None] - nro[None, :], co[:, None] - nco[None, :])
    dfr = np.abs(fr[:, None] - nfr[None, :])
    dist[(dist < 1e-9) & (dfr < 1e-9)] = np.inf          # a spot is not its own neighbour (identical coordinates)
    crowded = np.any((dist < crowd_px) & (dfr < 2 * (profile.half_widths(10.0)[0] + 1)), axis=1)

    extra_px = spots["extra_px"].to_numpy(float) if "extra_px" in spots else None
    extra_fr = spots["extra_frames"].to_numpy(float) if "extra_frames" in spots else np.zeros(n)
    rows, rej = [], []
    counts = {r: 0 for r in REJECT_REASONS}
    for i in range(n):
        reason = None
        if tth[i] < min_tth_deg or not np.any(hkl[i]):
            # q ~ 0: a direct-beam / beamstop-edge artefact assigns to (0,0,0) with ZERO residual
            reason = "low_angle"
        elif resid[i] > hkl_tol:
            reason = "no_hkl"
        elif no_cross[i]:
            reason = "no_crossing"
        elif not np.isfinite(L[i]):
            reason = "near_axis"
        elif crowded[i]:
            reason = "crowded"
        res = None
        if reason is None:
            hw = profile.half_widths(L[i])
            if extra_px is not None:
                hw = (int(hw[0] + math.ceil(float(extra_fr[i]))), int(hw[1] + math.ceil(float(extra_px[i]))))      # integer half-widths, as half_widths returns
            res = integrate_box(raw, (fr[i], ro[i], co[i]), hw, mask=mask,
                                shell=(profile.shell_frames, profile.shell_px), background=profile.background)
            if not res.ok:
                reason = "no_background"
            elif res.clipped_frames:
                reason = "frames_edge"
            elif res.clipped_detector:
                reason = "detector_edge"
            elif res.n_masked:
                reason = "masked_pixels"
        if reason is None:
            qs = qsample_of_hkl(hkl[i:i + 1], lattice, U)[0]
            PT0 = _pt(s[i:i + 1], np.array([omega[i]]), dac, pol, switches)[0]
            if switches.absorption and PT0 <= 0.0:
                reason = "outside_opening"
            else:
                var = _profile_variation(qs, float(omega[i]), abs(scan.omega_step) * hw[0],
                                         geom.wavelength_A, dac, pol, switches)
                if var > max_factor_variation:
                    reason = "factor_variation"
        if reason is not None:
            counts[reason] += 1
            rej.append(dict(i=i, reason=reason, h=int(hkl[i, 0]), k=int(hkl[i, 1]), l=int(hkl[i, 2]),
                            frame=fr[i], row=ro[i], col=co[i], hkl_residual=float(resid[i]), L=float(L[i])))
            continue
        Lf = L[i] if switches.lorentz else 1.0
        corr = Lf * PT0
        rows.append(dict(
            h=int(hkl[i, 0]), k=int(hkl[i, 1]), l=int(hkl[i, 2]), frame=fr[i], row=ro[i], col=co[i],
            omega=float(omega[i]), omega_meas=float(omega_meas[i]), tth=float(tth[i]), eta=float(eta[i]), L=float(L[i]), PT=float(PT0),
            I=res.intensity, sigI=res.sigma, bg=res.background, n_box=res.n_box,
            F2=res.intensity / corr, sigF2=res.sigma / corr,
            blob_sum=float(spots["integrated"].iloc[i]) if "integrated" in spots else np.nan,
            hkl_residual=float(resid[i])))
    obs = pd.DataFrame(rows)
    counts["input"] = n
    counts["kept"] = len(obs)
    assert counts["input"] == counts["kept"] + sum(counts[r] for r in REJECT_REASONS)
    return obs, pd.DataFrame(rej), counts
