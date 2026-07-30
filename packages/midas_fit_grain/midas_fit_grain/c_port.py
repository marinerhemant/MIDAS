"""Faithful Python port of the C output-stage routines from
``FF_HEDM/src/FitPosOrStrainsOMP.c`` (and its callees in
``MIDAS_Math.c`` / ``CalcDiffractionSpots.c``).

The forward-model + optimizer side of midas_fit_grain runs in PyTorch.
After refinement converges, this module computes the per-spot residuals
exactly as ``FitPosOrStrainsOMP.c::CalcAngleErrors`` does, so the
on-disk ``FitBest.bin`` / ``OrientPosFit.bin`` rows are bit-compatible
with what ``ProcessGrains`` / ``CalcStrains`` expects.

Conventions
-----------
* All angles in **degrees** unless suffixed ``_rad``. (Matches C.)
* Lengths in **micrometers** for spatial, **angstroms** for
  reciprocal-space. (Matches C.)
* OrientMat is row-major 3×3 with ``g_lab = OM @ g_crystal``. (Matches
  ``CalcDiffractionSpots.c::CalcDiffrSpots_Furnace`` line 175.)

References (line numbers in C source as of MIDAS v11.0):
  CorrectHKLsLatC          : FF_HEDM/src/FitPosOrStrainsOMP.c:165-203
  CalcEtaAngle             : FF_HEDM/src/FitPosOrStrainsOMP.c:205-210
  CorrectForOme            : FF_HEDM/src/FitPosOrStrainsOMP.c:212-333
  SpotToGv                 : FF_HEDM/src/FitPosOrStrainsOMP.c:335-355
  DisplacementInTheSpot    : FF_HEDM/src/MIDAS_Math.c:301-324
  CalcSpotPosition         : FF_HEDM/src/CalcDiffractionSpots.c:59-64
  CalcOmega                : FF_HEDM/src/CalcDiffractionSpots.c:66-148
  CalcDiffrSpots_Furnace   : FF_HEDM/src/CalcDiffractionSpots.c:150-231
  CalcAngleErrors          : FF_HEDM/src/FitPosOrStrainsOMP.c:470-718
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# -------------------------------------------------------------------------
# Low-level geometry helpers
# -------------------------------------------------------------------------


def _eta(y: float, z: float) -> float:
    """``CalcEtaAngleLocal`` — eta in degrees from (y, z) lab coords."""
    r = math.sqrt(y * y + z * z)
    if r == 0.0:
        return 0.0
    a = RAD2DEG * math.acos(max(-1.0, min(1.0, z / r)))
    return -a if y > 0 else a


def _displacement_in_the_spot(
    pos_a: float, pos_b: float, pos_c: float,
    lsd: float, y_orig: float, z_orig: float, omega_deg: float,
    wedge_deg: float, chi_deg: float,
) -> Tuple[float, float]:
    """``DisplacementInTheSpot`` — grain-position-induced spot offset.

    Returns (DisplY, DisplZ). When ``IK[0]`` underflows, the C code leaves
    the outputs uninitialized; we mirror that by returning (0, 0).
    """
    s_om = math.sin(DEG2RAD * omega_deg)
    c_om = math.cos(DEG2RAD * omega_deg)
    XNoW = pos_a * c_om - pos_b * s_om
    YNoW = pos_a * s_om + pos_b * c_om
    ZNoW = pos_c
    cw = math.cos(DEG2RAD * wedge_deg)
    sw = math.sin(DEG2RAD * wedge_deg)
    XW = XNoW * cw - ZNoW * sw
    YW = YNoW
    ZW = XNoW * sw + ZNoW * cw
    cc = math.cos(DEG2RAD * chi_deg)
    sc = math.sin(DEG2RAD * chi_deg)
    XC = XW
    YC = cc * YW - sc * ZW
    ZC = sc * YW + cc * ZW
    ikx = y_orig - XC      # NOTE: C uses xi=y_orig, yi=z_orig??? read comments below
    iky = y_orig - YC
    ikz = z_orig - ZC
    # Re-checking C signature:
    #   DisplacementInTheSpot(double a, double b, double c,
    #                          double xi, double yi, double zi,
    #                          double omega, ...)
    # In FitPosOrStrainsOMP.c:548 it's called with xi=Lsd, yi=YOrig, zi=ZOrig.
    # So xi = Lsd, yi = YOrig, zi = ZOrig. Fix the indices:
    ikx = lsd - XC
    iky = y_orig - YC
    ikz = z_orig - ZC
    n = math.sqrt(ikx * ikx + iky * iky + ikz * ikz)
    if n == 0.0:
        return 0.0, 0.0
    ikx /= n
    iky /= n
    ikz /= n
    if abs(ikx) > 1e-12:
        return YC - XC * iky / ikx, ZC - XC * ikz / ikx
    return 0.0, 0.0


def _correct_for_ome(
    yc: float, zc: float, lsd: float,
    omega_ini_deg: float, wavelength: float, wedge_deg: float,
) -> Tuple[float, float, float, float, float, float]:
    """``CorrectForOme`` + a modified ``SpotToGv`` for the corrected omega.

    Returns ``(ys, zs, omega_corr_deg, g1, g2, g3)`` where
    ``(ys, zs)`` is the wedge-corrected detector spot position and
    ``(g1, g2, g3)`` is the lab-frame G-vector at the corrected omega
    (matches ``SpotsYZOGCorr[sp][0..5]`` in C).
    """
    eta = _eta(yc, zc)
    rrad = math.sqrt(yc * yc + zc * zc)
    tth = RAD2DEG * math.atan(rrad / lsd)
    theta = tth / 2.0
    s_th = math.sin(DEG2RAD * theta)
    c_th = math.cos(DEG2RAD * theta)
    ds = 2.0 * s_th / wavelength
    cw = math.cos(DEG2RAD * wedge_deg)
    sw = math.sin(DEG2RAD * wedge_deg)
    s_eta = math.sin(DEG2RAD * eta)
    c_eta = math.cos(DEG2RAD * eta)
    k1 = -ds * s_th
    k2 = -ds * c_th * s_eta
    k3 = ds * c_th * c_eta
    if eta == 90.0:
        k3 = 0.0
        k2 = -c_th
    elif eta == -90.0:
        k3 = 0.0
        k2 = c_th
    k1f = k1 * cw + k3 * sw
    k2f = k2
    k3f = k3 * cw - k1 * sw
    s_omI = math.sin(DEG2RAD * omega_ini_deg)
    c_omI = math.cos(DEG2RAD * omega_ini_deg)
    G1a = k1f * c_omI + k2f * s_omI
    G2a = k2f * c_omI - k1f * s_omI
    G3a = k3f
    lenGa = math.sqrt(G1a * G1a + G2a * G2a + G3a * G3a)
    if lenGa == 0.0:
        return 0.0, 0.0, omega_ini_deg, 0.0, 0.0, 0.0
    g1 = G1a * ds / lenGa
    g2 = G2a * ds / lenGa
    g3 = G3a * ds / lenGa
    # The C code overrides wedge to zero from here on (line 249-250).
    sw, cw = 0.0, 1.0
    lenG = math.sqrt(g1 * g1 + g2 * g2 + g3 * g3)
    k1i = -(lenG * lenG * wavelength) / 2.0
    tth = 2.0 * RAD2DEG * math.asin(max(-1.0, min(1.0, wavelength * lenG / 2.0)))
    rrad = lsd * math.tan(DEG2RAD * tth)
    A = (k1i + g3 * sw) / cw
    a_sin = g1 * g1 + g2 * g2
    b_sin = 2.0 * A * g2
    c_sin = A * A - g1 * g1
    a_cos = a_sin
    b_cos = -2.0 * A * g1
    c_cos = A * A - g2 * g2

    def _safe_sqrt(x):
        return math.sqrt(x) if x >= 0 else None

    p_sin = _safe_sqrt(b_sin * b_sin - 4 * a_sin * c_sin)
    p_cos = _safe_sqrt(b_cos * b_cos - 4 * a_cos * c_cos)
    p_check_sin = p_sin is None
    p_check_cos = p_cos is None
    if p_sin is None:
        p_sin = 0.0
    if p_cos is None:
        p_cos = 0.0

    def _bound(x):
        return 0.0 if (x < -1.0 or x > 1.0) else x

    sin1 = _bound((-b_sin - p_sin) / (2.0 * a_sin)) if a_sin != 0 else 0.0
    sin2 = _bound((-b_sin + p_sin) / (2.0 * a_sin)) if a_sin != 0 else 0.0
    cos1 = _bound((-b_cos - p_cos) / (2.0 * a_cos)) if a_cos != 0 else 0.0
    cos2 = _bound((-b_cos + p_cos) / (2.0 * a_cos)) if a_cos != 0 else 0.0
    if p_check_sin:
        sin1 = sin2 = 0.0
    if p_check_cos:
        cos1 = cos2 = 0.0
    opt1 = abs(sin1 * sin1 + cos1 * cos1 - 1.0)
    opt2 = abs(sin1 * sin1 + cos2 * cos2 - 1.0)
    if opt1 < opt2:
        omg1 = RAD2DEG * math.atan2(sin1, cos1)
        omg2 = RAD2DEG * math.atan2(sin2, cos2)
    else:
        omg1 = RAD2DEG * math.atan2(sin1, cos2)
        omg2 = RAD2DEG * math.atan2(sin2, cos1)
    if abs(omg1 - omega_ini_deg) < abs(omg2 - omega_ini_deg):
        omega_corr = omg1
    else:
        omega_corr = omg2
    eta_out = _eta(k2, k3)
    s_eta_o = math.sin(DEG2RAD * eta_out)
    c_eta_o = math.cos(DEG2RAD * eta_out)
    ys_out = -rrad * s_eta_o
    zs_out = rrad * c_eta_o
    return ys_out, zs_out, omega_corr, g1, g2, g3


# -------------------------------------------------------------------------
# CorrectHKLsLatC — apply lattice (with strain) to integer (h, k, l)
# -------------------------------------------------------------------------


def correct_hkls_latc(
    lat_c: Sequence[float],
    hkls_int: np.ndarray,            # (M, 3) int Miller indices
    ring_nr_per_hkl: np.ndarray,     # (M,) int ring numbers
    lsd: float, wavelength: float,
) -> np.ndarray:
    """Port of ``CorrectHKLsLatC``.

    Returns a ``(M, 7)`` float64 array shaped exactly like the C ``hkls``
    matrix consumed by ``CalcDiffrSpots_Furnace``:
      [0..2] = G_cart  (Cartesian reciprocal-space, 1/Å)
      [3]    = D-spacing (Å)
      [4]    = Theta (deg)
      [5]    = Ring radius on detector (um) at distance ``lsd``
      [6]    = RingNr (passed through)
    """
    a, b, c, alpha, beta, gamma = (float(v) for v in lat_c)
    sin_a = math.sin(DEG2RAD * alpha); cos_a = math.cos(DEG2RAD * alpha)
    sin_b = math.sin(DEG2RAD * beta);  cos_b = math.cos(DEG2RAD * beta)
    sin_g = math.sin(DEG2RAD * gamma); cos_g = math.cos(DEG2RAD * gamma)
    eps = 1e-30
    gamma_pr = math.acos(max(-1.0, min(1.0, (cos_a * cos_b - cos_g) / (sin_a * sin_b + eps))))
    beta_pr  = math.acos(max(-1.0, min(1.0, (cos_g * cos_a - cos_b) / (sin_g * sin_a + eps))))
    sin_beta_pr = math.sin(beta_pr)
    vol = a * b * c * sin_a * sin_beta_pr * sin_g
    a_pr = b * c * sin_a / vol
    b_pr = c * a * sin_b / vol
    c_pr = a * b * sin_g / vol
    B = np.array([
        [a_pr, b_pr * math.cos(gamma_pr), c_pr * math.cos(beta_pr)],
        [0.0,  b_pr * math.sin(gamma_pr), -c_pr * sin_beta_pr * cos_a],
        [0.0,  0.0,                        c_pr * sin_beta_pr * sin_a],
    ], dtype=np.float64)

    n = hkls_int.shape[0]
    out = np.zeros((n, 7), dtype=np.float64)
    GCart = (B @ hkls_int.astype(np.float64).T).T   # (n, 3)
    Ds = 1.0 / np.linalg.norm(GCart, axis=1).clip(min=eps)
    sin_theta = wavelength / (2.0 * Ds)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    Theta = RAD2DEG * np.arcsin(sin_theta)
    Rad = lsd * np.tan(DEG2RAD * 2.0 * Theta)
    out[:, 0:3] = GCart
    out[:, 3]   = Ds
    out[:, 4]   = Theta
    out[:, 5]   = Rad
    out[:, 6]   = ring_nr_per_hkl.astype(np.float64)
    return out


# -------------------------------------------------------------------------
# CalcDiffrSpots_Furnace — predicted spots for the current orientation
# -------------------------------------------------------------------------


def _calc_omega(x: float, y: float, z: float, theta_deg: float) -> Tuple[list, list]:
    """``CalcOmega`` — returns ``(omegas, etas)`` of length ``nsol`` each."""
    eps = 1e-12
    omegas: list = []
    etas: list = []
    length = math.sqrt(x * x + y * y + z * z)
    v = math.sin(theta_deg * DEG2RAD) * length
    if abs(y) < eps:
        if x != 0:
            cosome = -v / x
            if abs(cosome) <= 1.0:
                ome = RAD2DEG * math.acos(cosome)
                omegas.append(ome)
                omegas.append(-ome)
    else:
        y2 = y * y
        a = 1 + (x * x) / y2
        b = 2 * v * x / y2
        c = v * v / y2 - 1
        discr = b * b - 4 * a * c
        if discr >= 0:
            sd = math.sqrt(discr)
            cosome1 = (-b + sd) / (2 * a)
            if abs(cosome1) <= 1:
                o1a = math.acos(cosome1)
                o1b = -o1a
                eqa = -x * math.cos(o1a) + y * math.sin(o1a)
                eqb = -x * math.cos(o1b) + y * math.sin(o1b)
                if abs(eqa - v) < abs(eqb - v):
                    omegas.append(o1a * RAD2DEG)
                else:
                    omegas.append(o1b * RAD2DEG)
            cosome2 = (-b - sd) / (2 * a)
            if abs(cosome2) <= 1:
                o2a = math.acos(cosome2)
                o2b = -o2a
                eqa = -x * math.cos(o2a) + y * math.sin(o2a)
                eqb = -x * math.cos(o2b) + y * math.sin(o2b)
                if abs(eqa - v) < abs(eqb - v):
                    omegas.append(o2a * RAD2DEG)
                else:
                    omegas.append(o2b * RAD2DEG)
    for ome in omegas:
        s = math.sin(ome * DEG2RAD)
        c = math.cos(ome * DEG2RAD)
        # rotate (x, y) by -ome about z to get the omega-frame components.
        xn =  c * x - s * y
        yn =  s * x + c * y
        # eta in lab using (yn, z) as detector intersection:
        etas.append(_eta(yn, z))
    return omegas, etas


def calc_diffr_spots_furnace(
    orient_mat: np.ndarray,           # (3, 3) row-major
    lsd: float,
    omega_ranges: np.ndarray,         # (nR, 2)
    box_sizes: np.ndarray,            # (nR, 4)
    hkls_full: np.ndarray,            # (M, 7) from correct_hkls_latc
    exclude_pole_angle: float,        # MinEta in deg
) -> np.ndarray:
    """Port of ``CalcDiffrSpots_Furnace``.

    Returns a ``(N, 9)`` float64 array of theoretical spots:
      [0]=Y_um, [1]=Z_um, [2]=Omega_deg, [3..5]=G_cart_per_d-spacing,
      [6]=lsd, [7]=RingNr, [8]=2*indexhkl+1+i (== ``nrhkls`` in C, used
      as a unique theoretical-spot identifier).
    """
    # NOTE: the C `etas` returned by CalcOmega is computed in the
    # eta-after-rotation frame (rotated G); we mirror that.
    nR = omega_ranges.shape[0]
    out_rows: list = []
    for indexhkl in range(hkls_full.shape[0]):
        Ghkl = hkls_full[indexhkl, 0:3]
        ring_radius = float(hkls_full[indexhkl, 5])
        ring_nr = float(hkls_full[indexhkl, 6])
        ds = float(hkls_full[indexhkl, 3])
        theta = float(hkls_full[indexhkl, 4])
        Gc = orient_mat @ Ghkl                              # (3,)
        omegas, etas = _calc_omega(Gc[0], Gc[1], Gc[2], theta)
        norm_gc = math.sqrt(Gc[0] ** 2 + Gc[1] ** 2 + Gc[2] ** 2)
        if norm_gc == 0:
            continue
        GCr = ds * Gc / norm_gc
        nrhkls_base = indexhkl * 2 + 1
        for i, (omega, eta) in enumerate(zip(omegas, etas)):
            if math.isnan(omega) or math.isnan(eta):
                continue
            eta_abs = abs(eta)
            if (eta_abs < exclude_pole_angle) or (180 - eta_abs < exclude_pole_angle):
                continue
            yl = -math.sin(eta * DEG2RAD) * ring_radius
            zl =  math.cos(eta * DEG2RAD) * ring_radius
            keep = False
            for r in range(nR):
                lo, hi = omega_ranges[r]
                yl_lo, yl_hi, zl_lo, zl_hi = box_sizes[r]
                if (omega > lo and omega < hi
                        and yl > yl_lo and yl < yl_hi
                        and zl > zl_lo and zl < zl_hi):
                    keep = True
                    break
            if not keep:
                continue
            out_rows.append([
                yl, zl, omega,
                float(GCr[0]), float(GCr[1]), float(GCr[2]),
                lsd, ring_nr, float(nrhkls_base + i),
            ])
    return (np.array(out_rows, dtype=np.float64)
            if out_rows else np.zeros((0, 9), dtype=np.float64))


# -------------------------------------------------------------------------
# CalcAngleErrors — the core obs↔theor matching + SpotsComp builder
# -------------------------------------------------------------------------


def calc_angle_errors(
    *,
    pos:        Sequence[float],     # (3,) refined grain position (um)
    orient_mat: np.ndarray,          # (3, 3) refined orientation matrix
    lat_c:      Sequence[float],     # (6,) refined lattice [a,b,c,α,β,γ]
    spots_yzo:  np.ndarray,          # (S, ≥10) per-obs-spot view, see below
    hkls_int:   np.ndarray,          # (M, 3) integer Miller indices
    ring_nr_per_hkl: np.ndarray,     # (M,)
    lsd: float, wavelength: float,
    omega_ranges: np.ndarray,        # (nR, 2)
    box_sizes:    np.ndarray,        # (nR, 4)
    min_eta:      float,             # ExcludePoleAngle (deg)
    wedge_deg:    float,
    chi_deg:      float = 0.0,
    weight_mask:  float = 1.0,
    weight_fit_rmse: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], int]:
    """C-faithful port of ``CalcAngleErrors`` (FitPosOrStrainsOMP.c:470-718).

    ``spots_yzo`` columns mirror C's ``spotsYZO[sp][0..9]``:
      [0] YLab          (wedge+det-corr lab Y, um)
      [1] ZLab          (lab Z, um)
      [2] Omega         (wedge-corr omega, deg)
      [3] SpotID        (1-based row in ExtraInfo.bin)
      [4] OmegaIni      (raw omega, no wedge corr, deg)
      [5] YOrig         (raw lab Y, no wedge corr, um)
      [6] ZOrig         (raw lab Z, no wedge corr, um)
      [7] RingNr        (1-based MIDAS ring number)
      [8] maskTouched   (0 or 1; weighted by weight_mask if 1)
      [9] FitRMSE       (per-spot peak-fit RMSE; weighted by weight_fit_rmse)

    Returns ``(spots_comp, spots_yzog_corr, error_ini, n_matched)`` where
    ``spots_comp`` is the ``(n_matched, 22)`` FitBest row buffer and
    ``error_ini`` is ``(mean_angle_deg, mean_pos_um, mean_ome_deg)`` — the
    per-matched-spot means of ``(min_angle, diff_len, diff_ome)`` in that
    order (matches C's ``MatchDiff[i][0..2]`` and ``Error[0..2]`` mapping
    in ``FitPosOrStrainsOMP.c:686-719``). ``mean_angle_deg`` is bounded
    ``< 1.0`` and ``mean_ome_deg`` is bounded ``< 5.0`` by the matching
    filters above; ``mean_pos_um`` is unbounded. Callers must not assume
    this tuple is already in OrientPosFit.bin column order (22=pos,
    23=ome, 24=angle) — unpack by name, not position.
    """
    S = int(spots_yzo.shape[0])

    # 1. Apply lattice strain to (h, k, l) → full hkls table for the model.
    hkls_full = correct_hkls_latc(
        lat_c, hkls_int, ring_nr_per_hkl, lsd, wavelength,
    )

    # 2. Generate theoretical spots from current orientation.
    theor_spots = calc_diffr_spots_furnace(
        orient_mat, lsd, omega_ranges, box_sizes, hkls_full, min_eta,
    )

    # 3. Compute SpotsYZOGCorr per observed spot.
    spots_yzog = np.zeros((S, 7), dtype=np.float64)
    for sp in range(S):
        y_orig = spots_yzo[sp, 5]
        z_orig = spots_yzo[sp, 6]
        omega_ini = spots_yzo[sp, 4]
        ring_nr = spots_yzo[sp, 7]
        dispY, dispZ = _displacement_in_the_spot(
            pos[0], pos[1], pos[2], lsd, y_orig, z_orig, omega_ini,
            wedge_deg, chi_deg,
        )
        yt = y_orig - dispY
        zt = z_orig - dispZ
        ys, zs, omega_corr, g1, g2, g3 = _correct_for_ome(
            yt, zt, lsd, omega_ini, wavelength, wedge_deg,
        )
        spots_yzog[sp] = (ys, zs, omega_corr, g1, g2, g3, ring_nr)

    # 4. Match obs → theor by ring + |Δω| < 5 + min internal angle <1.
    spots_comp = np.zeros((S, 22), dtype=np.float64)
    match_diff = np.zeros((S, 3), dtype=np.float64)
    n_matched = 0
    if theor_spots.shape[0] == 0:
        return (spots_comp[:0], spots_yzog, (0.0, 0.0, 0.0), 0)

    th_y = theor_spots[:, 0]
    th_z = theor_spots[:, 1]
    th_om = theor_spots[:, 2]
    th_g  = theor_spots[:, 3:6]
    th_rn = theor_spots[:, 7]

    for sp in range(S):
        ys, zs, omega_corr, g1, g2, g3, ring_nr = spots_yzog[sp]
        # Theoretical candidates: same ring, |Δω| < 5°.
        same_ring = th_rn.astype(int) == int(ring_nr)
        d_om = np.abs(th_om - omega_corr)
        in_window = same_ring & (d_om < 5.0)
        idxs = np.nonzero(in_window)[0]
        if idxs.size == 0:
            continue
        Gobs = np.array([g1, g2, g3])
        norm_gobs = np.linalg.norm(Gobs)
        if norm_gobs == 0:
            continue
        norms_gth = np.linalg.norm(th_g[idxs], axis=1)
        dots = th_g[idxs] @ Gobs
        cos_ang = np.clip(dots / (norms_gth * norm_gobs).clip(min=1e-30),
                          -1.0, 1.0)
        angles = np.abs(np.degrees(np.arccos(cos_ang)))
        best_local = int(np.argmin(angles))
        min_angle = float(angles[best_local])
        if min_angle >= 1.0:
            continue
        best = int(idxs[best_local])
        diff_len = math.hypot(ys - th_y[best], zs - th_z[best])
        diff_ome = abs(omega_corr - th_om[best])

        weight = 1.0
        if spots_yzo[sp, 8] > 0.5:
            weight *= weight_mask
        rmse = spots_yzo[sp, 9] if spots_yzo.shape[1] > 9 else 0.0
        if weight_fit_rmse > 0.0 and math.isfinite(rmse):
            weight *= math.exp(-rmse * weight_fit_rmse)

        match_diff[n_matched] = (min_angle * weight,
                                 diff_len * weight,
                                 diff_ome * weight)
        # SpotsComp 22-column row layout (FitPosOrStrainsOMP.c:686-708).
        spots_comp[n_matched, 0] = spots_yzo[sp, 3]                  # SpotID
        # cols 1..6 = SpotsYZOGCorr[sp][0..5]
        for i in range(6):
            spots_comp[n_matched, i + 1] = spots_yzog[sp, i]
        # cols 7..12 = TheorSpotsYZWE[best][0..5]  (= theor_spots cols 0..5)
        for i in range(6):
            spots_comp[n_matched, i + 7] = theor_spots[best, i]
        # cols 13..15 = raw obs (YLab, ZLab, Omega)
        spots_comp[n_matched, 13] = spots_yzo[sp, 0]
        spots_comp[n_matched, 14] = spots_yzo[sp, 1]
        spots_comp[n_matched, 15] = spots_yzo[sp, 2]
        spots_comp[n_matched, 16] = spots_yzo[sp, 4]                 # OmegaIni
        spots_comp[n_matched, 17] = spots_yzo[sp, 5]                 # YOrig
        spots_comp[n_matched, 18] = spots_yzo[sp, 6]                 # ZOrig
        spots_comp[n_matched, 19] = min_angle
        spots_comp[n_matched, 20] = diff_len
        spots_comp[n_matched, 21] = diff_ome
        n_matched += 1

    if n_matched == 0:
        return (spots_comp[:0], spots_yzog,
                (0.0, 0.0, 0.0), 0)

    err_ini = (
        float(match_diff[:n_matched, 0].sum() / n_matched),
        float(match_diff[:n_matched, 1].sum() / n_matched),
        float(match_diff[:n_matched, 2].sum() / n_matched),
    )
    return spots_comp[:n_matched], spots_yzog, err_ini, n_matched
