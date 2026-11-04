"""Orientation and geometry refinement from the measured spot positions.

The structure refinement needs the orientation matrix U good to a few hundredths of a degree (PREREGISTER A17-A19:
0.1 deg already moves refined parameters beyond their formal sigma, and the effect depends on the rotation axis). This
module refines U together with the detector geometry that turns a pixel into a direction, instead of taking either as given.

Model
-----
For an assigned reflection h the sample-frame vector is ``q = 2 pi U B h`` (``B`` the reciprocal basis of the lattice,
:func:`midas_sx.assign.reciprocal_basis`). The predicted spot is
  * the Ewald crossing of ``q`` nearest the MEASURED omega (``midas_defect.geometry.ewald_crossing_omegas``, analytic),
  * carried to the lab by ``q_lab = R_z(omega) q`` and to a pixel by ``midas_defect.geometry.qlab_to_pixel`` (the production map,
    including tilt and distortion), and omega to a frame index by the scan.
Residuals are ``(row_pred - row_obs)/sigma_px``, ``(col_pred - col_obs)/sigma_px`` and ``(frame_pred - frame_obs)/sigma_frame`` with EQUAL
weights per spot (weighting by intensity was 4-5x worse in the exploratory probe: bright spots carry a small centroid bias), a soft-L1
loss, and the parameters are a rotation vector applied on the left of the start U, the beam-centre row and column, and the relative
distance. The cell is NOT refined here (it trades off against the distance; state it as a limit when using this).

What it cannot see
------------------
A rotation of U about the rotation axis is an omega zero-point: indistinguishable from a shifted omega origin and harmless for the
structure, but a half-frame error in the frame-to-omega convention (``Scan``) appears as exactly that rotation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from midas_defect.geometry import Geometry, ewald_crossing_omegas, qlab_to_pixel
from midas_hkls import Lattice

from .assign import assign_hkl, ewald_crossings, reciprocal_basis
from .geometry import Scan, spot_directions

__all__ = ["OrientationResult", "predict_positions", "refine_orientation", "index_and_refine"]


@dataclass
class OrientationResult:
    U: np.ndarray
    geom: Geometry
    sigma_rot_deg: np.ndarray        # formal 1-sigma of the three rotation components, deg
    rms_px: float                    # ROBUST rms of the (row, col) residuals, pixels (1.4826 x median |r|): the gate uses this
    rms_frame: float                 # ROBUST rms of the frame residuals, frames
    n_used: int
    n_spots: int
    converged: bool
    reason: str
    nfev: int
    rounds: List[dict]
    rms_px_all: float = float("nan")      # plain rms including outlier spots (the gate does NOT use this)
    rms_frame_all: float = float("nan")
    n_outliers: int = 0                   # spots with a residual beyond 4 robust sigma in row, col or frame
    lambda_cell_scale: float = 0.0        # refine_wavelength only: relative scale eps of the wavelength, lambda = lambda0 (1 + eps) = geom.wavelength_A. JOINT with the cell: see index_and_refine
    sigma_lambda_cell_scale: float = float("nan")   # formal 1-sigma of eps (inf if the problem is rank deficient); nan when the parameter was not refined


def _newton_pixels(q_lab: np.ndarray, geom: Geometry, *, tol_px: float = 1e-8, max_iter: int = 60):
    """Pixel of each lab-frame q by damped Newton on the forward detector map (``apply_tilt_distortion``, the same map ``pixel_to_qlab`` uses), seeded with the flat-detector inverse.
    ``midas_defect.qlab_to_pixel`` iterates with the FLAT Jacobian, which diverges once a tilt is large (a 20 deg swung detector: 79 of 108 rays between 2theta 30 and 120 deg, including
    ones that land on the detector). Rays that do not go forward, or that do not converge, come back NaN."""
    import torch
    from midas_defect.geometry import _g_to_tensors
    from midas_transforms.fit_setup.transform import apply_tilt_distortion
    dt = torch.float64
    g = _g_to_tensors(geom, dtype=dt, device=torch.device("cpu"))
    pc = torch.as_tensor(geom.p_coeffs, dtype=dt)
    rho = torch.as_tensor(geom.rho_d_um, dtype=dt)
    px = g["px"]
    k0 = 2.0 * math.pi / g["lamb"]
    kf = torch.as_tensor(q_lab, dtype=dt) / k0 + torch.tensor([1.0, 0.0, 0.0], dtype=dt)
    fwd = kf[:, 0] > 1e-6
    sx = torch.where(fwd, kf[:, 0], torch.ones_like(kf[:, 0]))
    Yt, Zt = kf[:, 1] * g["lsd"] / sx, kf[:, 2] * g["lsd"] / sx
    cols, rows = g["bcy"] - Yt / px, g["bcz"] + Zt / px

    def fmap(c, r):
        return apply_tilt_distortion(c, r, Lsd=g["lsd"], BC_y=g["bcy"], BC_z=g["bcz"], tx=g["tx"], ty=g["ty"], tz=g["tz"], p_coeffs=pc, px=px, rho_d=rho)

    h = 0.25
    ok = torch.zeros_like(fwd)
    for _ in range(int(max_iter)):
        Yg, Zg = fmap(cols, rows)
        rY, rZ = Yt - Yg, Zt - Zg
        ok = fwd & (torch.maximum(rY.abs(), rZ.abs()) < tol_px * float(px))
        if bool(ok.all() | (~fwd).all()):
            break
        Yc, Zc = fmap(cols + h, rows); Yr, Zr = fmap(cols, rows + h)
        a11, a12, a21, a22 = (Yc - Yg) / h, (Yr - Yg) / h, (Zc - Zg) / h, (Zr - Zg) / h
        det = a11 * a22 - a12 * a21
        det = torch.where(det.abs() < 1e-12, torch.full_like(det, float("nan")), det)
        dc = (a22 * rY - a12 * rZ) / det
        dr = (-a21 * rY + a11 * rZ) / det
        step = torch.sqrt(dc ** 2 + dr ** 2)
        damp = torch.clamp(500.0 / torch.clamp(step, min=1e-30), max=1.0)               # at most 500 px per step
        upd = fwd & ~ok & torch.isfinite(dc) & torch.isfinite(dr)
        cols = torch.where(upd, cols + damp * dc, cols)
        rows = torch.where(upd, rows + damp * dr, rows)
    else:
        Yg, Zg = fmap(cols, rows)
        ok = fwd & (torch.maximum((Yt - Yg).abs(), (Zt - Zg).abs()) < tol_px * float(px))
    nan = torch.full_like(rows, float("nan"))
    return torch.where(ok, rows, nan).numpy(), torch.where(ok, cols, nan).numpy()


def _safe_pixels(q_lab: np.ndarray, geom: Geometry):
    """Pixels of lab-frame q vectors, never raising: the legacy ``midas_defect.qlab_to_pixel`` first (bit-identical wherever it converges), the Newton solver for the whole block if it
    does not; rays that cannot be placed come back NaN (the reflection is then simply not predicted on the detector)."""
    import torch
    try:
        r, c_ = qlab_to_pixel(q_lab, geom, device="cpu", dtype=torch.float64, tol_px=1e-8, max_iter=60)
        return r.numpy(), c_.numpy()
    except RuntimeError:
        return _newton_pixels(q_lab, geom)


def predict_positions(U: np.ndarray, lattice: Lattice, hkl: np.ndarray, omega_hint_deg: np.ndarray, geom: Geometry,
                      scan: Scan):
    """Predicted (frame, row, col) of each reflection at the Ewald crossing nearest ``omega_hint_deg``; ``valid`` marks reflections that
    cross at all and land on the forward hemisphere."""
    import torch
    hkl = np.asarray(hkl, float)
    q_s = 2.0 * np.pi * (np.asarray(U, float) @ reciprocal_basis(lattice) @ hkl.T).T
    n = len(q_s)
    om = np.full(n, np.nan)
    q_lab = np.full((n, 3), np.nan)
    hint = np.radians(np.asarray(omega_hint_deg, float))
    cw, sw = math.cos(math.radians(geom.wedge_deg)), math.sin(math.radians(geom.wedge_deg))
    for i in range(n):
        sols = ewald_crossings(q_s[i], geom.wavelength_A, geom.wedge_deg)
        if sols.size == 0:
            continue
        d = np.abs((sols - hint[i] + np.pi) % (2 * np.pi) - np.pi)
        w = float(sols[int(np.argmin(d))])
        c, s_ = math.cos(w), math.sin(w)
        x_, y_, z_ = c * q_s[i, 0] - s_ * q_s[i, 1], s_ * q_s[i, 0] + c * q_s[i, 1], q_s[i, 2]
        q_lab[i] = (cw * x_ - sw * z_, y_, sw * x_ + cw * z_) if geom.wedge_deg else (x_, y_, z_)                  # R_y(-W) R_z(omega) q_sample
        hint_deg = float(np.asarray(omega_hint_deg, float)[i])
        om[i] = hint_deg + (math.degrees(w) - hint_deg + 180.0) % 360.0 - 180.0         # the representative nearest the hint: a scan that runs through 360 (295 -> 372) needs omega > 180
    valid = np.isfinite(om)
    rows = np.full(n, np.nan)
    cols = np.full(n, np.nan)
    if valid.any():
        # tol_px: the default (1e-3 px) is coarser than a finite-difference step moves a prediction (5e-4 px for 1e-6 rad), which turns the
        # Jacobian into noise and the covariance singular; float64 can deliver 1e-8
        rows[valid], cols[valid] = _safe_pixels(q_lab[valid], geom)
    valid &= np.isfinite(rows)
    frame = (om - scan.omega_start) / scan.omega_step - 0.5
    return frame, rows, cols, valid


#: parameter vector: rotation vector (3, rad), beam centre row / col (px), relative distance, detector tilts tx ty tz (deg), omega-axis wedge (deg), relative wavelength / cell scale eps
_X_SCALE = np.array([0.01, 0.01, 0.01, 1.0, 1.0, 0.003, 0.3, 0.3, 0.3, 0.3, 1e-3])
_NPAR = len(_X_SCALE)


def _unpack(p, U0: np.ndarray, geom0: Geometry):
    p = np.asarray(p, float)
    if len(p) < _NPAR:
        p = np.concatenate([p, np.zeros(_NPAR - len(p))])
    U = Rotation.from_rotvec(p[:3]).as_matrix() @ U0
    g = replace(geom0, bcz_px=geom0.bcz_px + p[3], bcy_px=geom0.bcy_px + p[4], lsd_um=geom0.lsd_um * (1.0 + p[5]))
    if p[6] or p[7] or p[8] or p[9]:
        g = replace(g, tx_deg=geom0.tx_deg + p[6], ty_deg=geom0.ty_deg + p[7], tz_deg=geom0.tz_deg + p[8], wedge_deg=geom0.wedge_deg + p[9])
    if p[10]:
        g = replace(g, wavelength_A=geom0.wavelength_A * (1.0 + p[10]))
    return U, g


def refine_orientation(U0: np.ndarray, geom0: Geometry, lattice: Lattice, scan: Scan, hkl: np.ndarray,
                       frame_obs: np.ndarray, row_obs: np.ndarray, col_obs: np.ndarray, *, sigma_px: float = 0.25,
                       sigma_frame: float = 0.3, refine_geometry: bool = True, refine_tilts: bool = False, refine_wedge: bool = False,
                       f_scale: float = 3.0, max_nfev: int = 100, refine_wavelength: bool = False):
    """One joint least-squares fit for the given, already assigned, reflections. Returns ``(U, geom, info)``.

    ``refine_geometry``: beam centre and distance. ``refine_tilts``: the three detector tilts (tx, ty, tz) about the beam position, as offsets from ``geom0``. ``refine_wedge``: the tilt of the
    omega axis about the lab y axis (the MIDAS wedge, deg). A real goniometer is aligned to a few tenths of a degree; on the L-alanine / Diamond I19-1 frames a wedge of 0.2-0.3 deg
    made the fitted orientation of 100-frame windows creep by 0.6 deg over 170 deg of rotation.
    ``refine_wavelength``: one relative scale eps, lambda = ``geom0.wavelength_A`` (1 + eps), returned in ``info['lambda_cell_scale']`` (relative to ``geom0``) with ``info['sigma_lambda_cell_scale']``
    and carried in the returned geometry. It is a JOINT wavelength / cell-scale parameter, not an energy calibration: the cell is not refined here, and with the cell fixed a wavelength scale
    +eps is exactly equivalent to replacing the reference cell by one scaled by (1 - eps) (only lambda |q| enters): eps > 0 means the spots show a smaller cell than the reference. It absorbs
    any error of the reference cell, of the temperature, or of the header energy."""
    hkl = np.asarray(hkl)
    n = len(hkl)
    if n < 12:
        raise ValueError(f"only {n} assigned reflections; need >= 12 to refine orientation and geometry")
    hint = scan.omega_of(frame_obs)
    idx = [0, 1, 2] + ([3, 4, 5] if refine_geometry else []) + ([6, 7, 8] if refine_tilts else []) + ([9] if refine_wedge else []) + ([10] if refine_wavelength else [])
    npar = len(idx)

    def resid(p):
        pf = np.zeros(_NPAR)
        pf[idx] = p
        U, g = _unpack(pf, U0, geom0)
        fr, ro, co, ok = predict_positions(U, lattice, hkl, hint, g, scan)
        r = np.concatenate([(ro - row_obs) / sigma_px, (co - col_obs) / sigma_px, (fr - frame_obs) / sigma_frame])
        bad = ~np.tile(ok, 3)
        r[bad] = 10.0 * f_scale                      # a reflection that no longer crosses is a gross outlier, not a free pass
        return r

    sol = least_squares(resid, np.zeros(npar), loss="soft_l1", f_scale=f_scale, method="trf", x_scale=_X_SCALE[idx],
                        diff_step=1e-5, max_nfev=max_nfev)
    pf = np.zeros(_NPAR)
    pf[idx] = sol.x
    U, g = _unpack(pf, U0, geom0)
    # The formal covariance needs the Gauss-Newton Jacobian of the sigma-normalised residuals. scipy's `sol.jac` is a loss-modified
    # Jacobian that came back identically zero for soft-L1 in some converged fits, so build it directly by central differences.
    h = 1e-5
    J = np.empty((len(resid(sol.x)), npar))
    for j in range(npar):
        xp, xm = sol.x.copy(), sol.x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (resid(xp) - resid(xm)) / (2 * h)
    cn = np.linalg.norm(J, axis=0)
    sv = np.linalg.svd(J / np.where(cn > 0, cn, 1.0), compute_uv=False) if np.all(np.isfinite(J)) else np.array([0.0])
    if not np.all(np.isfinite(J)) or np.any(cn == 0) or sv[-1] <= 1e-9 * sv[0]:
        sig_rot = np.full(3, np.inf)                 # rank deficient (column-scaled test): say so, do not invent a sigma
        sig_lam = float("inf") if refine_wavelength else float("nan")
    else:
        cov = np.linalg.inv(J.T @ J)
        sig_rot = np.degrees(np.sqrt(np.maximum(np.diag(cov)[:3], 0.0)))
        sig_lam = float(np.sqrt(max(cov[idx.index(10), idx.index(10)], 0.0))) if refine_wavelength else float("nan")
    r = resid(sol.x)
    n_pix = 2 * n
    rp, rf = r[:n_pix] * sigma_px, r[n_pix:] * sigma_frame
    rms_all_px, rms_all_fr = float(np.sqrt(np.mean(rp ** 2))), float(np.sqrt(np.mean(rf ** 2)))
    # ROBUST rms (1.4826 x median |r|) for the gate: a handful of mis-centroided spots (a split or merged blob) can put several frames into
    # the plain rms of a fit whose soft-L1 loss has already ignored them; seen on 4 of 50 simulated seeds with orientation error 0.005-0.012 deg
    rob_px, rob_fr = float(1.4826 * np.median(np.abs(rp))), float(1.4826 * np.median(np.abs(rf)))
    n_out = int(np.sum((np.abs(r[:n]) * sigma_px > 4 * max(rob_px, 1e-9)) | (np.abs(r[n:n_pix]) * sigma_px > 4 * max(rob_px, 1e-9))
                       | (np.abs(r[n_pix:]) * sigma_frame > 4 * max(rob_fr, 1e-9))))
    return U, g, dict(sigma_rot_deg=sig_rot, rms_px=rob_px, rms_frame=rob_fr, rms_px_all=rms_all_px, rms_frame_all=rms_all_fr, n_outliers=n_out,
                      success=bool(sol.success), nfev=int(sol.nfev), status=int(sol.status),
                      lambda_cell_scale=float(pf[10]), sigma_lambda_cell_scale=sig_lam)


def index_and_refine(spots, lattice: Lattice, U0: np.ndarray, geom0: Geometry, scan: Scan, *, tols: Sequence[float] = (0.25, 0.20, 0.15),
                     refine_geometry: bool = True, refine_tilts: bool = False, refine_wedge: bool = False, min_tth_deg: float = 2.0, sigma_px: float = 0.25, sigma_frame: float = 0.3,
                     max_rms_px: float = 2.0, max_rms_frame: float = 1.0, refine_wavelength: bool = False, max_sigma_lambda: float = 5e-4) -> OrientationResult:
    """Assign hkl from the current model, refine, re-assign with a tighter tolerance, repeat.

    ``spots`` needs ``frame``, ``row``, ``col`` (fractional array indices, as ``find_blobs_3d`` returns them). Spots at low angle or with no
    integer hkl within the tolerance are left out of the fit, not forced. ``converged`` is False if the last round's optimiser failed or
    the ROBUST rms position residual is above ``max_rms_px`` pixels or ``max_rms_frame`` frames (outlier spots are counted in ``n_outliers`` and
    shown in ``rms_*_all``, not gated on); the caller must not use a non-converged orientation
    silently. Both are checked because a wrong cell is largely absorbed into the distance in pixels (0.19 px for a 2 % cell error) but not in omega (1.7 frames).

    ``refine_wavelength`` (default off, which leaves every output bit-identical) adds ONE relative scale eps, lambda = lambda0 (1 + eps), to the refinement. It enters every step that uses the
    wavelength (the Ewald crossings, the pixel -> q map, the hkl assignment of the following rounds) and the returned ``geom.wavelength_A`` is lambda, so ``assign_hkl``, ``calculated_crossing``,
    the corrections and ``reduce_spots`` all use it. Read it as a JOINT wavelength / cell-scale parameter (``OrientationResult.lambda_cell_scale``, relative to ``geom0``): the cell is fixed
    here, and with a fixed cell a wavelength scale +eps is exactly equivalent to an isotropic scale (1 - eps) of the reference cell (eps > 0: the spots show a smaller cell than the reference),
    so eps absorbs an error of the reference cell (temperature, a literature cell from another crystal) as much as an error of the header energy. It is NOT an energy calibration unless the cell is trusted. The result is not converged when eps is not determined
    (formal sigma above ``max_sigma_lambda`` or rank deficient). Why it exists: on the L-alanine / Diamond I19-1 frames a header-vs-COD-cell inconsistency of 0.19 % left a one-sided omega
    offset of 0.3-0.5 frame on one half of the detector (PREREGISTER A39, fork A) that the distance, tilts and wedge cannot absorb: the two Ewald crossings of a reflection move in opposite
    senses in omega under a wavelength error.
    """
    fr = spots["frame"].to_numpy(float)
    ro = spots["row"].to_numpy(float)
    co = spots["col"].to_numpy(float)
    U, geom = np.asarray(U0, float), geom0
    rounds, info, n_used, reason = [], {}, 0, "ok"
    for tol in tols:
        s, tth, _ = spot_directions(ro, co, geom)
        omega = scan.omega_of(fr)
        hkl, _, resid = assign_hkl(s, omega, lattice, U, geom.wavelength_A, geom.wedge_deg)
        use = (tth > min_tth_deg) & (resid < tol) & np.any(hkl != 0, axis=1)
        n_used = int(use.sum())
        if n_used < 12:
            reason = f"only {n_used} spots assigned at tolerance {tol}"
            rounds.append(dict(tol=tol, n_used=n_used))
            return OrientationResult(U, geom, np.full(3, np.inf), float("nan"), float("nan"), n_used, len(fr), False, reason, 0, rounds)
        U, geom, info = refine_orientation(U, geom, lattice, scan, hkl[use], fr[use], ro[use], co[use], sigma_px=sigma_px,
                                           sigma_frame=sigma_frame, refine_geometry=refine_geometry, refine_tilts=refine_tilts, refine_wedge=refine_wedge,
                                           refine_wavelength=refine_wavelength)
        rounds.append(dict(tol=tol, n_used=n_used, rms_px=info["rms_px"], rms_frame=info["rms_frame"], nfev=info["nfev"]))
    ok = info["success"] and info["rms_px"] <= max_rms_px and info["rms_frame"] <= max_rms_frame
    if not info["success"]:
        reason = "optimiser did not succeed"
    elif info["rms_px"] > max_rms_px:
        reason = f"rms position residual {info['rms_px']:.2f} px above {max_rms_px}"
    elif info["rms_frame"] > max_rms_frame:
        reason = f"rms frame residual {info['rms_frame']:.2f} frames above {max_rms_frame} (wrong cell or omega convention?)"
    if refine_wavelength:
        sig_l = info["sigma_lambda_cell_scale"]
        if ok and not (np.isfinite(sig_l) and sig_l <= max_sigma_lambda):
            ok, reason = False, f"wavelength / cell scale not determined: formal sigma {sig_l:.2g} above {max_sigma_lambda:g} (too few spots, too narrow an omega range, or degenerate with the other parameters)"
    return OrientationResult(U, geom, info["sigma_rot_deg"], info["rms_px"], info["rms_frame"], n_used, len(fr), bool(ok), reason,
                             info["nfev"], rounds, rms_px_all=info["rms_px_all"], rms_frame_all=info["rms_frame_all"],
                             n_outliers=info["n_outliers"], lambda_cell_scale=float(geom.wavelength_A / geom0.wavelength_A - 1.0),
                             sigma_lambda_cell_scale=info["sigma_lambda_cell_scale"])
