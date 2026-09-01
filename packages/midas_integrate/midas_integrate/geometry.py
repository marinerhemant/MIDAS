"""Detector geometry primitives.

Pure-Python (numpy) port of FF_HEDM/src/DetectorGeometry.c.

Functions are written in two flavors where it matters:

- Scalar: used by ``detector_mapper.build_map`` inside a per-pixel loop.
  These mirror the C function signatures one-for-one.
- Vectorized: used wherever a whole detector or a large array of pixels can
  be processed at once (e.g. the initial pass to compute (R, Eta) for every
  pixel). These accept numpy arrays and broadcast.

The tilt + 15-parameter distortion model is the canonical MIDAS forward
transform. Bins can be equi-spaced in radial pixels (default), in
2θ via :func:`build_tth_bin_edges_in_R`, or in momentum transfer Q
via :func:`build_q_bin_edges_in_R`.
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
EPS = 1e-6

# Pixel corner offsets (matches DG_PosMatrix).
#
# ``QUAD_ORDER`` is the sequence in which ``point_in_quad`` and
# ``pixel_bin_intersect`` walk the pixel's edges: 0->1, 1->3, 3->2, 2->0. That
# is a closed boundary ONLY if the corners are stored in Z-order, i.e. index
#
#     0 = (-,-)   1 = (-,+)   2 = (+,-)   3 = (+,+)
#
# so the walk becomes (-,-) -> (-,+) -> (+,+) -> (+,-) -> back.
#
# These offsets used to be [(-,-), (-,+), (+,+), (+,-)] -- boundary order, not
# Z-order -- under which the same walk traces two sides and two DIAGONALS: a
# bowtie. ``point_in_quad`` then answered incorrectly for points inside the
# pixel, and ``pixel_bin_intersect`` searched the wrong segments for arc and
# ray crossings, so vertices were missed and the intersection polygon came out
# incomplete or empty.
#
# MEASURED consequence, before the fix, on a pixel straddling both an R and an
# eta bin boundary (true total = the pixel's area = 1.0):
#
#     r=24 eta=7: 0.007739
#     r=24 eta=8: 0.083343
#     r=25 eta=7: 0.000000   <- the pixel does overlap this cell
#     r=25 eta=8: 0.404529
#     TOTAL       0.495611
#
# Across a synthetic 64x64 detector (RBinSize 1 px, EtaBinSize 30 deg) interior
# pixels retained a MEAN of 0.85 of their area, worst case 0.50. With Z-order
# every interior pixel conserves exactly: mean 1.00000000, min = max = 1.0.
#
# Nothing else depends on the index order -- the other consumers take
# ``min``/``max`` over the corner axis -- so this is safe to correct here.
QUAD_ORDER = (0, 1, 3, 2)
PIXEL_CORNER_OFFSETS = np.array(
    [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
    dtype=np.float64,
)


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def calc_eta_angle(y: float, z: float) -> float:
    """Mirror dg_calc_eta_angle: atan2(-y, z) in degrees."""
    return RAD2DEG * math.atan2(-y, z)


def calc_eta_angle_array(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return RAD2DEG * np.arctan2(-y, z)


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def between(val: float, lo: float, hi: float) -> bool:
    return (val - EPS <= hi) and (val + EPS >= lo)


# ─────────────────────────────────────────────────────────────────────────────
# Tilt rotation
# ─────────────────────────────────────────────────────────────────────────────
def build_tilt_matrix(tx_deg: float, ty_deg: float, tz_deg: float) -> np.ndarray:
    """TRs = Rx(tx) · Ry(ty) · Rz(tz) — angles in degrees, returns 3x3 float64."""
    tx, ty, tz = (DEG2RAD * a for a in (tx_deg, ty_deg, tz_deg))
    Rx = np.array([
        [1, 0,           0          ],
        [0, math.cos(tx), -math.sin(tx)],
        [0, math.sin(tx),  math.cos(tx)],
    ], dtype=np.float64)
    Ry = np.array([
        [ math.cos(ty), 0, math.sin(ty)],
        [ 0,            1, 0           ],
        [-math.sin(ty), 0, math.cos(ty)],
    ], dtype=np.float64)
    Rz = np.array([
        [math.cos(tz), -math.sin(tz), 0],
        [math.sin(tz),  math.cos(tz), 0],
        [0,             0,            1],
    ], dtype=np.float64)
    return Rx @ (Ry @ Rz)


# ─────────────────────────────────────────────────────────────────────────────
# Forward transform: (Y, Z) pixel → (R [px], Eta [deg])
# ─────────────────────────────────────────────────────────────────────────────
def pixel_to_REta(
    Y, Z,
    *,
    Ycen: float, Zcen: float,
    TRs: np.ndarray,
    Lsd: float, RhoD: float, px: float,
    p0: float = 0.0, p1: float = 0.0, p2: float = 0.0, p3: float = 0.0,
    p4: float = 0.0, p5: float = 0.0, p6: float = 0.0, p7: float = 0.0,
    p8: float = 0.0, p9: float = 0.0, p10: float = 0.0, p11: float = 0.0,
    p12: float = 0.0, p13: float = 0.0, p14: float = 0.0,
    dLsd: float = 0.0, dP2: float = 0.0,
    parallax: float = 0.0,
    residual_corr_map: Optional[np.ndarray] = None,
    return_untilted: bool = False,
):
    """Vectorized forward transform from pixel coords to (R_pixels, Eta_deg).

    ``Y`` and ``Z`` may be scalars or numpy arrays — broadcasting applies.
    Mirrors ``dg_pixel_to_REta_corr`` in C ``DetectorGeometry.c`` including
    per-panel ``dLsd / dP2`` corrections and an optional residual ΔR(Y, Z)
    bilinear-interpolation lookup.
    """
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    panelLsd = Lsd + dLsd
    panelP2 = p2 + dP2
    Yc = (-Y + Ycen) * px
    Zc = ( Z - Zcen) * px
    # ABC = (0, Yc, Zc); apply TRs
    abcpr_x = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    abcpr_y = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    abcpr_z = TRs[2, 1] * Yc + TRs[2, 2] * Zc
    EtaUntilted = RAD2DEG * np.arctan2(-Yc, Zc) if return_untilted else None
    XYZ_x = panelLsd + abcpr_x
    XYZ_y = abcpr_y
    XYZ_z = abcpr_z
    # Avoid div-by-zero (XYZ_x ~ 0 means pixel is at the sample plane)
    safe_x = np.where(np.abs(XYZ_x) < 1e-30, 1e-30, XYZ_x)
    Rad = (panelLsd / safe_x) * np.sqrt(XYZ_y * XYZ_y + XYZ_z * XYZ_z)
    EtaTilted = RAD2DEG * np.arctan2(-XYZ_y, XYZ_z)
    RNorm = Rad / RhoD if RhoD > 0 else np.zeros_like(Rad)
    EtaT = 90.0 - EtaTilted
    EtaT_rad = DEG2RAD * EtaT
    # 15-param distortion (uses panelP2 = p2 + dP2)
    dist = (
        p0 * np.power(RNorm, 2.0) * np.cos(2 * EtaT_rad + DEG2RAD * p6)
        + p1 * np.power(RNorm, 4.0) * np.cos(4 * EtaT_rad + DEG2RAD * p3)
        + panelP2 * np.power(RNorm, 2.0)
        + p4 * np.power(RNorm, 6.0)
        + p5 * np.power(RNorm, 4.0)
        + p7 * np.power(RNorm, 4.0) * np.cos(EtaT_rad + DEG2RAD * p8)
        + p9 * np.power(RNorm, 3.0) * np.cos(3 * EtaT_rad + DEG2RAD * p10)
        + p11 * np.power(RNorm, 5.0) * np.cos(5 * EtaT_rad + DEG2RAD * p12)
        + p13 * np.power(RNorm, 6.0) * np.cos(6 * EtaT_rad + DEG2RAD * p14)
        + 1.0
    )
    Rt = Rad * dist / px      # in pixels (still on panel-local Lsd plane)
    # Re-project to global Lsd plane so radii from different panels share a
    # common scale (matches `Rt = Rt * (Lsd / panelLsd)` in the C version).
    if dLsd != 0.0:
        Rt = Rt * (Lsd / panelLsd)
    if parallax != 0.0:
        twoTheta = np.arctan(Rad / panelLsd)
        Rt = Rt + parallax * np.sin(twoTheta) / px
    if residual_corr_map is not None and residual_corr_map.size > 0:
        Rt = Rt + _residual_corr_lookup_array(residual_corr_map, Y, Z)
    if return_untilted:
        return Rt, EtaTilted, EtaUntilted
    return Rt, EtaTilted


def _residual_corr_lookup_array(map_arr: np.ndarray,
                                Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sample of ΔR(Y, Z); mirrors dg_residual_corr_lookup."""
    NZ, NY = map_arr.shape
    if NY == 0 or NZ == 0:
        return np.zeros_like(np.broadcast_to(Y, np.broadcast(Y, Z).shape),
                             dtype=np.float64)
    y = np.clip(np.asarray(Y, dtype=np.float64), 0.0, NY - 1.001)
    z = np.clip(np.asarray(Z, dtype=np.float64), 0.0, NZ - 1.001)
    y0 = y.astype(np.int64)
    z0 = z.astype(np.int64)
    fy = y - y0
    fz = z - z0
    v00 = map_arr[z0,     y0]
    v10 = map_arr[z0,     y0 + 1]
    v01 = map_arr[z0 + 1, y0]
    v11 = map_arr[z0 + 1, y0 + 1]
    return (v00 * (1 - fy) * (1 - fz)
            + v10 * fy       * (1 - fz)
            + v01 * (1 - fy) * fz
            + v11 * fy       * fz)


def solid_angle_factor(
    Y, Z,
    *,
    Ycen: float, Zcen: float,
    TRs: np.ndarray,
    Lsd: float, px: float,
):
    """Tilt-aware solid-angle correction factor (vectorized).

    Returns the dimensionless quantity ``Ω_pix / Ω_ref`` where
    ``Ω_pix = A_pix · |n̂·r̂| / r²`` is the solid angle subtended by a
    pixel at the sample, ``A_pix = px²``, and ``Ω_ref = A_pix / Lsd²`` is
    the on-axis (zero-tilt, zero-2θ) reference. Equivalently:

        Ω_pix / Ω_ref = Lsd² · (n̂ · r) / |r|³

    where ``r`` is the lab-frame vector from sample to pixel and ``n̂``
    is the lab-frame detector normal (the rotation of (1,0,0) by the
    same tilt matrix used for the (Y, Z) → (R, η) projection).

    For a flat detector aligned with the beam, ``n̂ = (1, 0, 0)`` and
    ``r = (Lsd, y, z)`` with ``|r| = Lsd / cos(2θ)``, recovering the
    flat-detector form ``cos³(2θ)``. For tilted detectors the local
    incidence angle of the diffracted ray with respect to the *tilted*
    detector normal is captured exactly.

    The intensity correction divides the recorded counts by this
    quantity (``corrected = area / SA_factor``), matching the convention
    used in ``DetectorMapper`` so that the on-axis pixel has correction
    1.0.
    """
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    Yc = (-Y + Ycen) * px
    Zc = ( Z - Zcen) * px
    abcpr_x = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    abcpr_y = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    abcpr_z = TRs[2, 1] * Yc + TRs[2, 2] * Zc
    XYZ_x = Lsd + abcpr_x
    XYZ_y = abcpr_y
    XYZ_z = abcpr_z
    # n_hat = TRs · (1, 0, 0)^T = first column of TRs.
    nx = TRs[0, 0]
    ny = TRs[1, 0]
    nz = TRs[2, 0]
    n_dot_r = nx * XYZ_x + ny * XYZ_y + nz * XYZ_z
    r_mag = np.sqrt(XYZ_x * XYZ_x + XYZ_y * XYZ_y + XYZ_z * XYZ_z)
    r3 = np.maximum(r_mag * r_mag * r_mag, 1e-30)
    return Lsd * Lsd * n_dot_r / r3


def REta_to_YZ(R, Eta_deg):
    """Inverse polar: (R, Eta_deg) → (Y, Z) centered at beam (vectorized)."""
    Eta_rad = np.asarray(Eta_deg) * DEG2RAD
    Y = -R * np.sin(Eta_rad)
    Z = R * np.cos(Eta_rad)
    return Y, Z


def REta_to_YZ_scalar(R: float, Eta_deg: float) -> Tuple[float, float]:
    """Scalar variant for inner loops."""
    e = Eta_deg * DEG2RAD
    return -R * math.sin(e), R * math.cos(e)


# ─────────────────────────────────────────────────────────────────────────────
# Newton-Raphson inversion
#
# UNITS. Both inverters take ``R_target`` in **detector pixels**, the same unit
# ``pixel_to_REta`` returns, and return raw pixel (Y, Z). The name reads as
# "convert R,eta TO pixels", which invites the opposite reading -- ring radii
# are micrometres nearly everywhere else in MIDAS -- and nothing rejected a
# micrometre radius: the seed line ``Y = Ycen + R_target*sin(eta)`` just put the
# start point tens of thousands of pixels off the panel, Newton walked it a
# little, and the caller got coordinates ~50 000 with no error at all. Measured
# consequence: the mandatory ring-overlay figure drew every ring off-frame and
# looked blank. Hence _warn_if_R_looks_like_micrometres below.
# ─────────────────────────────────────────────────────────────────────────────

#: How far past the detector's own scale ``R_target`` must sit before the
#: pixels-vs-micrometres warning fires. A real pixel radius cannot exceed ~1x
#: that scale, and the micrometre mistake overshoots it by the pixel pitch
#: (75-200x), so 10x separates the two with a wide margin either way.
_R_PIXEL_SANITY_FACTOR = 10.0


class RadiusUnitWarning(UserWarning):
    """``R_target`` is too large to be a pixel radius on this detector.

    A warning, not an error: the inverters are also used for synthetic and
    deliberately extrapolated geometries. But an unheeded one turns into an
    empty overlay figure or an empty ring mask, with no other symptom.
    """


def _detector_scale_px(Ycen: float, Zcen: float, RhoD: float, px: float):
    """Rough largest plausible pixel radius, or ``None`` if unknowable.

    Two independent estimates, and the LARGER wins so neither can cry wolf on
    its own:

    * ``RhoD / px`` — RhoD is the beam-centre-to-farthest-corner distance times
      the pitch (``midas_calibrate_v2.pipelines.ff_calibrate.rho_d_for``), so
      the quotient is that distance back in pixels. Some synthetic callers pass
      a token RhoD, which would make this absurdly small.
    * ``hypot(Ycen, Zcen)`` — the beam centre is inside the panel, so the
      panel is at least this big. Zero for a geometry centred on the origin.
    """
    scales = []
    if RhoD > 0.0 and px > 0.0:
        scales.append(RhoD / px)
    centre = math.hypot(float(Ycen), float(Zcen))
    if centre > 0.0:
        scales.append(centre)
    return max(scales) if scales else None


def _warn_if_R_looks_like_micrometres(R_max: float, Ycen: float, Zcen: float,
                                      RhoD: float, px: float,
                                      *, stacklevel: int = 3) -> None:
    """Cheap plausibility check on the radial input unit."""
    scale = _detector_scale_px(Ycen, Zcen, RhoD, px)
    if scale is None or R_max <= _R_PIXEL_SANITY_FACTOR * scale:
        return
    hint = (f" Dividing by the pixel pitch ({px:g} um) gives "
            f"{R_max / px:.1f} px.") if px > 0.0 else ""
    warnings.warn(
        f"R_target reaches {R_max:.4g}, more than "
        f"{_R_PIXEL_SANITY_FACTOR:g}x this detector's own scale "
        f"({scale:.1f} px). R_target is in DETECTOR PIXELS, not "
        f"micrometres.{hint} As given, the returned (Y, Z) land far off the "
        "panel and nothing downstream will complain.",
        RadiusUnitWarning, stacklevel=stacklevel)


def invert_REta_to_pixel(
    R_target: float, Eta_target: float, *,
    Ycen: float, Zcen: float,
    TRs: np.ndarray, Lsd: float, RhoD: float, px: float,
    p0: float = 0.0, p1: float = 0.0, p2: float = 0.0, p3: float = 0.0,
    p4: float = 0.0, p5: float = 0.0, p6: float = 0.0, p7: float = 0.0,
    p8: float = 0.0, p9: float = 0.0, p10: float = 0.0, p11: float = 0.0,
    p12: float = 0.0, p13: float = 0.0, p14: float = 0.0,
    parallax: float = 0.0,
    max_iter: int = 10,
    tol_R: float = 1e-8,
    tol_eta: float = 1e-8,
) -> Tuple[float, float]:
    """Newton-Raphson inversion: find raw (Y, Z) such that the forward
    transform reproduces ``(R_target, Eta_target)``.

    Mirror of dg_invert_REta_to_pixel_corr in DetectorGeometry.c.

    Parameters
    ----------
    R_target : radial distance in **detector pixels** — the same unit
        :func:`pixel_to_REta` returns, NOT micrometres. Multiply a micrometre
        radius by ``1/px`` first. See the units note above this section.
    Eta_target : azimuth in **degrees**, MIDAS convention.
    Ycen, Zcen : beam centre in **pixels**.
    Lsd, RhoD, px : sample-detector distance, distortion normalisation and
        pixel pitch, all in **micrometres**.

    Returns
    -------
    (Y, Z) : raw detector coordinates in **pixels**, unrounded.
    """
    _warn_if_R_looks_like_micrometres(abs(float(R_target)), Ycen, Zcen,
                                      RhoD, px)
    Y = Ycen + R_target * math.sin(Eta_target * DEG2RAD)
    Z = Zcen + R_target * math.cos(Eta_target * DEG2RAD)
    h = 0.01
    for _ in range(max_iter):
        R_eval, Eta_eval = pixel_to_REta(
            Y, Z,
            Ycen=Ycen, Zcen=Zcen, TRs=TRs, Lsd=Lsd, RhoD=RhoD, px=px,
            p0=p0, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7,
            p8=p8, p9=p9, p10=p10, p11=p11, p12=p12, p13=p13, p14=p14,
            parallax=parallax,
        )
        dR = R_target - float(R_eval)
        dEta = Eta_target - float(Eta_eval)
        if dEta > 180.0:
            dEta -= 360.0
        if dEta < -180.0:
            dEta += 360.0
        if abs(dR) < tol_R and abs(dEta) < tol_eta:
            break
        # Numerical Jacobian
        R_dY, Eta_dY = pixel_to_REta(
            Y + h, Z,
            Ycen=Ycen, Zcen=Zcen, TRs=TRs, Lsd=Lsd, RhoD=RhoD, px=px,
            p0=p0, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7,
            p8=p8, p9=p9, p10=p10, p11=p11, p12=p12, p13=p13, p14=p14,
            parallax=parallax,
        )
        R_dZ, Eta_dZ = pixel_to_REta(
            Y, Z + h,
            Ycen=Ycen, Zcen=Zcen, TRs=TRs, Lsd=Lsd, RhoD=RhoD, px=px,
            p0=p0, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7,
            p8=p8, p9=p9, p10=p10, p11=p11, p12=p12, p13=p13, p14=p14,
            parallax=parallax,
        )
        dRdY = (float(R_dY) - float(R_eval)) / h
        dRdZ = (float(R_dZ) - float(R_eval)) / h
        dEdY = (float(Eta_dY) - float(Eta_eval)) / h
        dEdZ = (float(Eta_dZ) - float(Eta_eval)) / h
        det = dRdY * dEdZ - dRdZ * dEdY
        if abs(det) < 1e-30:
            break
        deltaY = (dEdZ * dR - dRdZ * dEta) / det
        deltaZ = (dRdY * dEta - dEdY * dR) / det
        Y += deltaY
        Z += deltaZ
    return Y, Z


def invert_REta_to_pixel_batch(
    R_targets, Eta_targets, *,
    Ycen: float, Zcen: float,
    TRs: np.ndarray, Lsd: float, RhoD: float, px: float,
    p0: float = 0.0, p1: float = 0.0, p2: float = 0.0, p3: float = 0.0,
    p4: float = 0.0, p5: float = 0.0, p6: float = 0.0, p7: float = 0.0,
    p8: float = 0.0, p9: float = 0.0, p10: float = 0.0, p11: float = 0.0,
    p12: float = 0.0, p13: float = 0.0, p14: float = 0.0,
    parallax: float = 0.0,
    max_iter: int = 10,
    tol_R: float = 1e-8,
    tol_eta: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Newton-Raphson inversion over N targets in lock-step.

    Same physics as :func:`invert_REta_to_pixel` but iterates the whole
    batch at once: each Newton iteration evaluates ``pixel_to_REta``
    three times on the full batch (centre, +h Y perturb, +h Z perturb)
    rather than three times per point. For the ~1000-point E-step in
    midas-calibrate this collapses ~30 000 scalar Python calls into 30
    array calls.

    Convergence is per-point: once a point hits the ``(tol_R, tol_eta)``
    tolerance window, subsequent iterations leave its (Y, Z) unchanged.
    Points whose Jacobian becomes singular freeze in place (matches the
    scalar version's ``if abs(det) < 1e-30: break`` semantics).

    Parameters mirror :func:`invert_REta_to_pixel` exactly, units included:
    ``R_targets`` is in **detector pixels** (what :func:`pixel_to_REta`
    returns), ``Eta_targets`` in **degrees**, ``Ycen``/``Zcen`` in **pixels**,
    ``Lsd``/``RhoD``/``px`` in **micrometres**. Inputs may be Python scalars,
    lists, or numpy arrays; output is always a pair of numpy float64 arrays
    with shape ``(N,)`` (or ``()`` for scalar input).

    Returns
    -------
    Y, Z : np.ndarray of **detector pixels**, unrounded.
        Same shape as broadcast of ``R_targets`` and ``Eta_targets``.
    """
    R_targets = np.asarray(R_targets, dtype=np.float64)
    Eta_targets = np.asarray(Eta_targets, dtype=np.float64)
    # Broadcast to a common shape.
    R_targets, Eta_targets = np.broadcast_arrays(R_targets, Eta_targets)
    R_targets = np.ascontiguousarray(R_targets)
    Eta_targets = np.ascontiguousarray(Eta_targets)
    if R_targets.size:
        _warn_if_R_looks_like_micrometres(
            float(np.max(np.abs(R_targets))), Ycen, Zcen, RhoD, px)

    Y = Ycen + R_targets * np.sin(Eta_targets * DEG2RAD)
    Z = Zcen + R_targets * np.cos(Eta_targets * DEG2RAD)
    h = 0.01

    fwd_kwargs = dict(
        Ycen=Ycen, Zcen=Zcen, TRs=TRs, Lsd=Lsd, RhoD=RhoD, px=px,
        p0=p0, p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7,
        p8=p8, p9=p9, p10=p10, p11=p11, p12=p12, p13=p13, p14=p14,
        parallax=parallax,
    )

    for _ in range(max_iter):
        R_eval, Eta_eval = pixel_to_REta(Y, Z, **fwd_kwargs)
        R_eval = np.asarray(R_eval, dtype=np.float64)
        Eta_eval = np.asarray(Eta_eval, dtype=np.float64)
        dR = R_targets - R_eval
        dEta = Eta_targets - Eta_eval
        # Wrap the angular residual to (-180, 180]; matches scalar branch.
        dEta = np.where(dEta > 180.0, dEta - 360.0, dEta)
        dEta = np.where(dEta < -180.0, dEta + 360.0, dEta)

        active = (np.abs(dR) >= tol_R) | (np.abs(dEta) >= tol_eta)
        if not active.any():
            break

        # Vectorized two-sided numerical Jacobian: three evaluations per
        # iteration over the whole batch instead of three per point.
        R_dY, Eta_dY = pixel_to_REta(Y + h, Z, **fwd_kwargs)
        R_dZ, Eta_dZ = pixel_to_REta(Y, Z + h, **fwd_kwargs)
        R_dY = np.asarray(R_dY, dtype=np.float64)
        R_dZ = np.asarray(R_dZ, dtype=np.float64)
        Eta_dY = np.asarray(Eta_dY, dtype=np.float64)
        Eta_dZ = np.asarray(Eta_dZ, dtype=np.float64)
        dRdY = (R_dY - R_eval) / h
        dRdZ = (R_dZ - R_eval) / h
        dEdY = (Eta_dY - Eta_eval) / h
        dEdZ = (Eta_dZ - Eta_eval) / h
        det = dRdY * dEdZ - dRdZ * dEdY
        # Mask out singular Jacobians (matches scalar ``break`` -- those
        # points freeze at their current (Y, Z)).
        non_singular = np.abs(det) >= 1e-30
        update_mask = active & non_singular
        # Safe divisor to avoid runtime warnings; result on frozen points
        # is discarded by the mask below.
        safe_det = np.where(non_singular, det, 1.0)
        deltaY = (dEdZ * dR - dRdZ * dEta) / safe_det
        deltaZ = (dRdY * dEta - dEdY * dR) / safe_det
        Y = np.where(update_mask, Y + deltaY, Y)
        Z = np.where(update_mask, Z + deltaZ, Z)
    return Y, Z


# ─────────────────────────────────────────────────────────────────────────────
# Bin edges
# ─────────────────────────────────────────────────────────────────────────────
def build_bin_edges(
    RMin: float, EtaMin: float,
    n_r_bins: int, n_eta_bins: int,
    RBinSize: float, EtaBinSize: float,
):
    """Uniform R and Eta bin edges in numpy arrays."""
    eta_lo = EtaMin + EtaBinSize * np.arange(n_eta_bins, dtype=np.float64)
    eta_hi = eta_lo + EtaBinSize
    r_lo = RMin + RBinSize * np.arange(n_r_bins, dtype=np.float64)
    r_hi = r_lo + RBinSize
    return r_lo, r_hi, eta_lo, eta_hi


def build_q_bin_edges_in_R(
    QMin: float, QMax: float, QBinSize: float,
    Lsd: float, px: float, wavelength_A: float,
):
    """Compute non-uniform R bin edges that correspond to a uniform Q grid.

    Uses the Bragg formula::

        Q = (4π / λ) sin(θ),    R = (Lsd / px) tan(2θ)

    Returns (r_lo, r_hi) in pixel units, plus the matching n_r_bins.
    """
    n_r = int(math.ceil((QMax - QMin) / QBinSize))
    q_lo = QMin + QBinSize * np.arange(n_r, dtype=np.float64)
    q_hi = q_lo + QBinSize
    # 2θ = 2 arcsin(Qλ / 4π)
    two_theta_lo = 2.0 * np.arcsin(np.clip(q_lo * wavelength_A / (4.0 * math.pi), -1, 1))
    two_theta_hi = 2.0 * np.arcsin(np.clip(q_hi * wavelength_A / (4.0 * math.pi), -1, 1))
    r_lo = (Lsd / px) * np.tan(two_theta_lo)
    r_hi = (Lsd / px) * np.tan(two_theta_hi)
    return r_lo, r_hi, n_r


def build_tth_bin_edges_in_R(
    TthMin: float, TthMax: float, TthBinSize: float,
    Lsd: float, px: float,
):
    """Compute non-uniform R bin edges that correspond to a uniform 2θ grid.

    The (R, 2θ) relation is the standard powder-diffraction projection
    onto a planar detector::

        R = (Lsd / px) tan(2θ)

    Inputs are in degrees; returned ``r_lo``, ``r_hi`` are in pixel
    units, alongside the matching ``n_r_bins``.  Wavelength-independent
    by construction --- a calibrant-free choice when the user wants to
    bin in scattering angle rather than radial position or momentum
    transfer.
    """
    n_r = int(math.ceil((TthMax - TthMin) / TthBinSize))
    tth_lo = TthMin + TthBinSize * np.arange(n_r, dtype=np.float64)
    tth_hi = tth_lo + TthBinSize
    r_lo = (Lsd / px) * np.tan(np.radians(tth_lo))
    r_hi = (Lsd / px) * np.tan(np.radians(tth_hi))
    return r_lo, r_hi, n_r


# ─────────────────────────────────────────────────────────────────────────────
# Polygon-arc intersection geometry (Green's theorem)
# ─────────────────────────────────────────────────────────────────────────────
def circle_seg_intersect(y1: float, z1: float,
                         y2: float, z2: float,
                         R: float) -> List[Tuple[float, float]]:
    """Intersections of circle y²+z² = R² with segment P1→P2.

    Returns 0, 1, or 2 (y, z) points clipped to segment endpoints.
    """
    dy = y2 - y1
    dz = z2 - z1
    a = dy * dy + dz * dz
    if a < 1e-30:
        return []
    b = 2.0 * (y1 * dy + z1 * dz)
    c = y1 * y1 + z1 * z1 - R * R
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return []
    sqrt_disc = math.sqrt(disc)
    inv2a = 0.5 / a
    out: List[Tuple[float, float]] = []
    t1 = (-b - sqrt_disc) * inv2a
    if -EPS <= t1 <= 1.0 + EPS:
        tc = max(0.0, min(1.0, t1))
        out.append((y1 + tc * dy, z1 + tc * dz))
    t2 = (-b + sqrt_disc) * inv2a
    if -EPS <= t2 <= 1.0 + EPS and abs(t2 - t1) > 1e-12:
        tc = max(0.0, min(1.0, t2))
        out.append((y1 + tc * dy, z1 + tc * dz))
    return out


def ray_seg_intersect(y1: float, z1: float,
                      y2: float, z2: float,
                      eta_deg: float) -> Optional[Tuple[float, float]]:
    """Intersection of an η-ray from origin with segment P1→P2.

    Returns (y, z) or None. Only points on the *positive* ray direction count.
    """
    eta_rad = eta_deg * DEG2RAD
    ce = math.cos(eta_rad)
    se = math.sin(eta_rad)
    dy = y2 - y1
    dz = z2 - z1
    denom = dy * ce + dz * se
    if abs(denom) < 1e-30:
        return None
    t = -(y1 * ce + z1 * se) / denom
    if t < -EPS or t > 1.0 + EPS:
        return None
    t = max(0.0, min(1.0, t))
    hy = y1 + t * dy
    hz = z1 + t * dz
    if (-se) * hy + ce * hz < 0:
        return None
    return (hy, hz)


def point_in_quad(py: float, pz: float, quad: np.ndarray) -> bool:
    """Convex point-in-quadrilateral test (matches dg_point_in_quad).

    quad: shape (4, 2), corner-indexed by QUAD_ORDER.
    """
    pos = neg = 0
    for e in range(4):
        i0 = QUAD_ORDER[e]
        i1 = QUAD_ORDER[(e + 1) % 4]
        ey = quad[i1, 0] - quad[i0, 0]
        ez = quad[i1, 1] - quad[i0, 1]
        cross = ey * (pz - quad[i0, 1]) - ez * (py - quad[i0, 0])
        if cross > 0:
            pos += 1
        elif cross < 0:
            neg += 1
    return pos == 0 or neg == 0


def find_unique_vertices(edges_in: List[Tuple[float, float]],
                         RMin: float, RMax: float,
                         EtaMin: float, EtaMax: float
                         ) -> List[Tuple[float, float]]:
    """Drop duplicate vertices and clip to the (R, Eta) bin.

    Mirror of dg_find_unique_vertices.
    """
    out: List[Tuple[float, float]] = []
    n = len(edges_in)
    for i in range(n):
        y, z = edges_in[i]
        # dedup vs later entries (matches the C O(n²) scheme)
        is_dup = False
        for j in range(i + 1, n):
            yj, zj = edges_in[j]
            if math.hypot(y - yj, z - zj) == 0.0:
                is_dup = True
                break
        if is_dup:
            continue
        RT = math.hypot(y, z)
        ET = calc_eta_angle(y, z)
        if not between(ET, EtaMin, EtaMax):
            if between(ET + 360.0, EtaMin, EtaMax):
                ET += 360.0
            elif between(ET - 360.0, EtaMin, EtaMax):
                ET -= 360.0
        if not between(RT, RMin, RMax):
            continue
        if not between(ET, EtaMin, EtaMax):
            continue
        out.append((y, z))
    return out


def polygon_area(edges: List[Tuple[float, float]],
                 RMin: float, RMax: float) -> float:
    """Green's-theorem area of a polygon with mixed straight + arc edges.

    Vertices are sorted counterclockwise around the centroid before applying
    the line integral. Edges where both endpoints lie on R = RMin or R = RMax
    are integrated as circular arcs (R²/2)·dθ; all other edges use the
    Shoelace term.

    Mirror of dg_polygon_area in DetectorGeometry.c.
    """
    n = len(edges)
    if n < 3:
        return 0.0
    pts = np.asarray(edges, dtype=np.float64)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    # Sort by angle around centroid (counterclockwise = increasing atan2)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    # Match the C comparator: counterclockwise from +x axis, with stable
    # ordering for collinear points by distance.  np.lexsort handles it.
    dists = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
    order = np.lexsort((dists, -angles))   # primary sort = -angle (clockwise→counterclockwise) … see note
    # The C code's comparator effectively sorts counterclockwise. The tightest
    # match: sort by atan2 ascending (which is counterclockwise from -π to +π).
    order = np.argsort(angles, kind="stable")
    pts = pts[order]

    RMin2 = RMin * RMin
    RMax2 = RMax * RMax
    tol = 1e-6
    area = 0.0
    for i in range(n):
        y1, z1 = pts[i]
        y2, z2 = pts[(i + 1) % n]
        r1sq = y1 * y1 + z1 * z1
        r2sq = y2 * y2 + z2 * z2
        on_rmin = (abs(r1sq - RMin2) < tol * max(RMin2, 1e-30) and
                   abs(r2sq - RMin2) < tol * max(RMin2, 1e-30))
        on_rmax = (abs(r1sq - RMax2) < tol * max(RMax2, 1e-30) and
                   abs(r2sq - RMax2) < tol * max(RMax2, 1e-30))
        if on_rmin or on_rmax:
            R = RMin if on_rmin else RMax
            a1 = math.atan2(z1, y1)
            a2 = math.atan2(z2, y2)
            d = a2 - a1
            if d > math.pi:
                d -= 2.0 * math.pi
            if d < -math.pi:
                d += 2.0 * math.pi
            area += (R * R * 0.5) * d
        else:
            area += 0.5 * (y1 * z2 - y2 * z1)
    return abs(area)


def pixel_bin_intersect(
    cornerYZ: np.ndarray,            # shape (4, 2) of pixel corners in (Y, Z)
    RMin: float, RMax: float, EtaMin: float, EtaMax: float,
    bin_corners: Optional[np.ndarray] = None,   # shape (4, 2) bin corners
) -> float:
    """Compute intersection area of a pixel quad with one (R, Eta) bin.

    This is the inner kernel called by ``detector_mapper.build_map`` for each
    candidate (pixel, bin) pair. It mirrors the Green's-theorem section of
    ``mapper_build_map`` in MapperCore.c.

    ``bin_corners`` are the (Y,Z) of the bin's four (R, Eta) corners; if not
    provided, they are computed from RMin/RMax/EtaMin/EtaMax.
    """
    if bin_corners is None:
        bin_corners = np.empty((4, 2), dtype=np.float64)
        bin_corners[0] = REta_to_YZ_scalar(RMin, EtaMin)
        bin_corners[1] = REta_to_YZ_scalar(RMin, EtaMax)
        bin_corners[2] = REta_to_YZ_scalar(RMax, EtaMin)
        bin_corners[3] = REta_to_YZ_scalar(RMax, EtaMax)

    edges: List[Tuple[float, float]] = []

    # (1) pixel corners that lie inside the (R, Eta) bin
    for m in range(4):
        cy, cz = cornerYZ[m, 0], cornerYZ[m, 1]
        RT = math.hypot(cy, cz)
        ET = calc_eta_angle(cy, cz)
        if EtaMin < -180.0 and _sign(ET) != _sign(EtaMin):
            ET -= 360.0
        if EtaMax > 180.0 and _sign(ET) != _sign(EtaMax):
            ET += 360.0
        if RMin <= RT <= RMax and EtaMin <= ET <= EtaMax:
            edges.append((cy, cz))

    # (2) bin corners that lie inside the pixel quad
    for m in range(4):
        by, bz = bin_corners[m, 0], bin_corners[m, 1]
        if point_in_quad(by, bz, cornerYZ):
            edges.append((by, bz))

    # (3) edge-vs-edge intersections (only when we don't already have ≥4)
    if len(edges) < 4:
        for e in range(4):
            i0 = QUAD_ORDER[e]
            i1 = QUAD_ORDER[(e + 1) % 4]
            py1, pz1 = cornerYZ[i0, 0], cornerYZ[i0, 1]
            py2, pz2 = cornerYZ[i1, 0], cornerYZ[i1, 1]

            for hit in circle_seg_intersect(py1, pz1, py2, pz2, RMin):
                hy, hz = hit
                EtaH = calc_eta_angle(hy, hz)
                if EtaMin < -180.0 and _sign(EtaH) != _sign(EtaMin):
                    EtaH -= 360.0
                if EtaMax > 180.0 and _sign(EtaH) != _sign(EtaMax):
                    EtaH += 360.0
                if EtaMin - EPS <= EtaH <= EtaMax + EPS:
                    edges.append((hy, hz))
            for hit in circle_seg_intersect(py1, pz1, py2, pz2, RMax):
                hy, hz = hit
                EtaH = calc_eta_angle(hy, hz)
                if EtaMin < -180.0 and _sign(EtaH) != _sign(EtaMin):
                    EtaH -= 360.0
                if EtaMax > 180.0 and _sign(EtaH) != _sign(EtaMax):
                    EtaH += 360.0
                if EtaMin - EPS <= EtaH <= EtaMax + EPS:
                    edges.append((hy, hz))

            hit = ray_seg_intersect(py1, pz1, py2, pz2, EtaMin)
            if hit is not None:
                hy, hz = hit
                RH = math.hypot(hy, hz)
                if RMin - EPS <= RH <= RMax + EPS:
                    edges.append((hy, hz))
            hit = ray_seg_intersect(py1, pz1, py2, pz2, EtaMax)
            if hit is not None:
                hy, hz = hit
                RH = math.hypot(hy, hz)
                if RMin - EPS <= RH <= RMax + EPS:
                    edges.append((hy, hz))

    if len(edges) < 3:
        return 0.0
    edges = find_unique_vertices(edges, RMin, RMax, EtaMin, EtaMax)
    if len(edges) < 3:
        return 0.0
    return polygon_area(edges, RMin, RMax)
