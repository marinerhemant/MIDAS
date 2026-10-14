"""Numba-JIT'd implementation of the DetectorMapper inner loop.

Replaces the slow pure-Python ``pixel_bin_intersect`` + per-row Python loop
in ``detector_mapper.py`` with a single ``@njit(parallel=True)`` kernel
that mirrors ``mapper_build_map`` in MapperCore.c. Performance target:
match the C OpenMP version (typically ~2 s for PILATUS3 2M on 8 cores;
should be similar on the same hardware here).

Numba is an optional dependency. If unavailable, ``detector_mapper.py``
falls back to the pure-Python implementation (slow, single-threaded).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# macOS: pick numba's own threadpool before numba initialises its threading
# layer.
#
# The kernels below are @njit(parallel=True). numba's default layer on macOS is
# OpenMP, and these kernels run inside processes that have ALREADY loaded a
# different OpenMP runtime -- torch and numpy each ship one, which is why
# KMP_DUPLICATE_LIB_OK=TRUE is needed to import them together at all. Two
# OpenMP runtimes in one process then SEGFAULT when the parallel region opens.
#
# Measured: midas_calibrate_v2.calibrate() on an Apple-silicon Mac died with
# SIGSEGV inside map_kernel, in-process AND in a child process, so it was not a
# Jupyter-state problem. With NUMBA_THREADING_LAYER=workqueue the same call
# completes and returns the same answer as Linux to every digit printed
# (Lsd 940085.0 um, BC 1037.499/1014.497 px, 22.55 ue).
#
# workqueue is numba's built-in threadpool: still parallel, no external OpenMP,
# typically a little slower than omp. That is the right trade for a crash.
# Only macOS is touched, and only when the user has not chosen a layer, so a
# deliberate setting is always respected.
if sys.platform == "darwin":
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

try:
    import numba
    from numba import njit, prange
    if sys.platform == "darwin":
        # Belt and braces: the env var is read when the layer first
        # initialises, which may already have happened if something else
        # imported numba first. Setting config too covers that case.
        try:
            numba.config.THREADING_LAYER = os.environ["NUMBA_THREADING_LAYER"]
        except Exception:            # pragma: no cover - never fatal
            pass
    HAVE_NUMBA = True
except ImportError:                  # pragma: no cover
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def deco(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return deco
    def prange(*args, **kwargs):     # type: ignore
        return range(*args, **kwargs)

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
QUAD_ORDER = (0, 1, 3, 2)

#: Smallest (pixel, bin) overlap area kept, in pixel^2. Must match
#: ``detector_mapper.MIN_PIXEL_BIN_AREA`` -- the numba and pure-python map
#: builders have to produce the same map, and the parity tests compare them
#: entry by entry. Duplicated rather than imported because an njit kernel
#: cannot close over another module's value at compile time.
#:
#: Was 1e-5, which dropped the slivers that make the map area-conserving:
#: max |sum over bins - pixel area| goes 9.6e-06 -> 2.9e-14 at 1e-9, for 0.2%
#: more entries. See detector_mapper for why not lower than 1e-9.
_MIN_PIXEL_BIN_AREA = 1e-9

QUAD0 = 0
QUAD1 = 1
QUAD2 = 3
QUAD3 = 2

# Pixel corner offsets, used by the per-pixel quad construction.
_DY = (-0.5, 0.5)
_DZ = (-0.5, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (scalar @njit)
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True)
def _quad_min_r(qy: np.ndarray, qz: np.ndarray) -> float:
    """Smallest distance from the origin to the quad (0 if the origin is inside).

    numba twin of ``geometry.quad_min_radius``.

    The candidate radial bins of a pixel must come from this, not from the corners'
    radii: a straight edge can pass closer to the origin than either of its corners,
    by up to L^2 / (8 R). When a bin boundary falls in that window the pixel overlaps
    the bin below it in a thin sliver that a corners-only range never visits (up to
    ~0.4 percent of a pixel, on the pixels along the cardinal axes). The largest
    radius is always at a corner: a chord lies inside the circle through its ends.
    """
    best = 1.0e300
    pos = 0
    neg = 0
    for e in range(4):
        i1 = e + 1
        if i1 == 4:
            i1 = 0
        py = qy[e]
        pz = qz[e]
        dy = qy[i1] - py
        dz = qz[i1] - pz
        a = dy * dy + dz * dz
        t = 0.0
        if a > 0.0:
            t = -(py * dy + pz * dz) / a
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        ry = py + t * dy
        rz = pz + t * dz
        d = math.sqrt(ry * ry + rz * rz)
        if d < best:
            best = d
        cr = py * dz - pz * dy
        if cr > 0.0:
            pos += 1
        elif cr < 0.0:
            neg += 1
    if pos == 0 or neg == 0:       # all edges wind the same way round the origin: it is inside
        return 0.0
    return best


@njit(cache=True)
def _clip_half(buf: np.ndarray, s: int, n: int, d: int,
               uy: float, uz: float, sgn: float) -> int:
    """Sutherland-Hodgman: clip polygon buf[s:s+n] to sgn*(uy*z - uz*y) >= 0, write the
    result to buf[d:...] and return its vertex count. A vertex exactly on the line
    counts as inside."""
    m = 0
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        py = buf[s + i, 0]
        pz = buf[s + i, 1]
        qy = buf[s + j, 0]
        qz = buf[s + j, 1]
        fp = sgn * (uy * pz - uz * py)
        fq = sgn * (uy * qz - uz * qy)
        pin = fp >= 0.0
        qin = fq >= 0.0
        if pin:
            buf[d + m, 0] = py
            buf[d + m, 1] = pz
            m += 1
        if pin != qin:
            t = fp / (fp - fq)
            buf[d + m, 0] = py + t * (qy - py)
            buf[d + m, 1] = pz + t * (qz - pz)
            m += 1
    return m


@njit(cache=True, inline="always")
def _disk_piece(py: float, pz: float, dy: float, dz: float,
                ta: float, tb: float, R2: float) -> float:
    """Signed area of (triangle origin-A-B) intersect (disk R2) for the sub-segment
    A = P + ta*d, B = P + tb*d, which lies wholly inside or wholly outside the disk."""
    ay = py + ta * dy
    az = pz + ta * dz
    by = py + tb * dy
    bz = pz + tb * dz
    tm = 0.5 * (ta + tb)
    my = py + tm * dy
    mz = pz + tm * dz
    cr = ay * bz - az * by
    if my * my + mz * mz <= R2:
        return 0.5 * cr
    return 0.5 * R2 * math.atan2(cr, ay * by + az * bz)


@njit(cache=True)
def _disk_area(buf: np.ndarray, s: int, n: int, R: float) -> float:
    """Exact area of (counterclockwise polygon buf[s:s+n]) intersect (disk of radius R
    at the origin): per edge, a triangle for the part inside the disk and a circular
    sector for the parts outside, split at the (at most two) circle crossings."""
    if n < 3 or R <= 0.0:
        return 0.0
    R2 = R * R
    total = 0.0
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        py = buf[s + i, 0]
        pz = buf[s + i, 1]
        dy = buf[s + j, 0] - py
        dz = buf[s + j, 1] - pz
        a = dy * dy + dz * dz
        if a == 0.0:
            continue
        b = 2.0 * (py * dy + pz * dz)
        c = py * py + pz * pz - R2
        disc = b * b - 4.0 * a * c
        ta = 0.0
        if disc > 0.0:
            sq = math.sqrt(disc)
            t1 = (-b - sq) / (2.0 * a)
            t2 = (-b + sq) / (2.0 * a)
            if 0.0 < t1 < 1.0:
                total += _disk_piece(py, pz, dy, dz, ta, t1, R2)
                ta = t1
            if 0.0 < t2 < 1.0:
                total += _disk_piece(py, pz, dy, dz, ta, t2, R2)
                ta = t2
        total += _disk_piece(py, pz, dy, dz, ta, 1.0, R2)
    return total


@njit(cache=True)
def _pixel_bin_area(quad_y: np.ndarray, quad_z: np.ndarray,
                    bin_y: np.ndarray, bin_z: np.ndarray,
                    RMin: float, RMax: float,
                    EtaMin: float, EtaMax: float,
                    edges: np.ndarray) -> float:
    """Exact area of (pixel quad) intersect (annular sector RMin..RMax, EtaMin..EtaMax).

    quad_y/quad_z[4]   in QUAD_ORDER traversal (boundary) order
    bin_y/bin_z[4]     unused; kept so the call sites do not change
    edges              scratch (50, 2); rows 0-7, 8-15, 16-23, 24-31 hold polygons

    area = A(RMax) - A(RMin), A(R) = area(pixel clipped to the wedge, inside the disk
    of radius R); a wedge wider than 180 degrees is the whole-pixel disk area minus
    the complementary wedge. Same algorithm as ``geometry.pixel_bin_intersect``; see
    there for why it replaced the vertex enumeration.
    """
    if not (EtaMax > EtaMin) or not (RMax > RMin):
        return 0.0
    # load the quad counterclockwise
    s = 0.0
    for i in range(4):
        j = (i + 1) % 4
        s += quad_y[i] * quad_z[j] - quad_y[j] * quad_z[i]
    for i in range(4):
        k = i if s >= 0.0 else 3 - i
        edges[i, 0] = quad_y[k]
        edges[i, 1] = quad_z[k]

    w = EtaMax - EtaMin
    if w >= 360.0:
        a = _disk_area(edges, 0, 4, RMax) - _disk_area(edges, 0, 4, RMin)
        return a if a > 0.0 else 0.0
    e0 = EtaMin * DEG2RAD
    e1 = EtaMax * DEG2RAD
    d0y = -math.sin(e0)
    d0z = math.cos(e0)
    d1y = -math.sin(e1)
    d1z = math.cos(e1)
    if w <= 180.0:
        n1 = _clip_half(edges, 0, 4, 8, d0y, d0z, 1.0)
        n2 = _clip_half(edges, 8, n1, 16, d1y, d1z, -1.0)
        a = _disk_area(edges, 16, n2, RMax) - _disk_area(edges, 16, n2, RMin)
        return a if a > 0.0 else 0.0
    n1 = _clip_half(edges, 0, 4, 8, d1y, d1z, 1.0)
    n2 = _clip_half(edges, 8, n1, 16, d0y, d0z, -1.0)
    full = _disk_area(edges, 0, 4, RMax) - _disk_area(edges, 0, 4, RMin)
    comp = _disk_area(edges, 16, n2, RMax) - _disk_area(edges, 16, n2, RMin)
    a = full - comp
    return a if a > 0.0 else 0.0


@njit(cache=True, parallel=True)
def map_kernel(
    R_corners: np.ndarray,        # (NZ, NY, 4)
    Eta_corners: np.ndarray,      # (NZ, NY, 4)
    cornerYZ: np.ndarray,         # (NZ, NY, 4, 2)
    Rt_center: np.ndarray,        # (NZ, NY)
    Eta_center: np.ndarray,       # (NZ, NY)
    r_lo: np.ndarray, r_hi: np.ndarray,
    eta_lo: np.ndarray, eta_hi: np.ndarray,
    n_r_bins: int, n_eta_bins: int,
    NrPixelsY: int, NrPixelsZ: int,
    Lsd: float, px: float,
    solid_angle: int, polarization: int, pol_fraction: float,
    pol_plane_eta_rad: float,
    sa_factor: np.ndarray,        # (NZ, NY) tilt-aware solid-angle factor Ω/Ω_ref
    mask: np.ndarray,             # (NZ, NY); 0 = unmasked, 1 = masked. Empty array if no mask.
    mask_present: int,
    flat: np.ndarray,             # (NZ, NY); per-pixel relative sensitivity. Empty array if absent.
    flat_present: int,
    raw_y_arr: np.ndarray,        # (NZ, NY) — pre-applied inverse-transform pixel Y
    raw_z_arr: np.ndarray,        # (NZ, NY) — pre-applied inverse-transform pixel Z
    per_row_max: int,
):
    """Returns:
      out_arr      shape (NZ, per_row_max, 6) — (bin_idx, raw_y, raw_z, frac, deltaR, area) per entry
                   Only the first `per_row_count[j]` entries of each row are valid.
      per_row_count (NZ,) int64
      bin_mask_local (n_r_bins * n_eta_bins,) int32 — 1 where any masked pixel touched bin
    """
    out_arr = np.zeros((NrPixelsZ, per_row_max, 6), dtype=np.float64)
    per_row_count = np.zeros(NrPixelsZ, dtype=np.int64)
    bin_mask = np.zeros(n_r_bins * n_eta_bins, dtype=np.int32)

    for j in prange(NrPixelsZ):
        # Per-thread scratch (re-allocated inside the loop iteration; fine because
        # numba parallel allocates these in private storage).
        edges_scratch = np.empty((50, 2), dtype=np.float64)
        bin_y = np.empty(4, dtype=np.float64)
        bin_z = np.empty(4, dtype=np.float64)
        quad_y = np.empty(4, dtype=np.float64)
        quad_z = np.empty(4, dtype=np.float64)
        local = 0
        for i in range(NrPixelsY):
            # Permute the stored corners into perimeter order.
            #
            # ``cornerYZ`` is indexed by geometry.PIXEL_CORNER_OFFSETS, which is
            # Z-ORDER: [(-,-), (-,+), (+,-), (+,+)]. QUAD_ORDER = (0, 1, 3, 2)
            # is what turns that into a closed boundary walk. This kernel's
            # ``_pixel_bin_intersect_njit`` walks its quad SEQUENTIALLY, so the
            # permutation has to happen here rather than inside it.
            #
            # Copying k -> k straight through, as this used to, hands the njit
            # kernel a Z-ordered quad that it then traces as a bowtie: two sides
            # and two diagonals. That is the same defect the pure-python path
            # had, and it costs a mean 16% of every straddling pixel's area.
            for k in range(4):
                src = QUAD_ORDER[k]
                quad_y[k] = cornerYZ[j, i, src, 0]
                quad_z[k] = cornerYZ[j, i, src, 1]

            # bounding box in (R, Eta)
            rmi = R_corners[j, i, 0]; rma = R_corners[j, i, 0]
            emi = Eta_corners[j, i, 0]; ema = Eta_corners[j, i, 0]
            for k in range(1, 4):
                v = R_corners[j, i, k]
                if v < rmi: rmi = v
                if v > rma: rma = v
                v = Eta_corners[j, i, k]
                if v < emi: emi = v
                if v > ema: ema = v
            rq = _quad_min_r(quad_y, quad_z)
            if rq < rmi: rmi = rq

            # masked pixel handling
            is_masked = (mask_present == 1) and (mask[j, i] == 1.0)

            # candidate R bins
            for kr in range(n_r_bins):
                if r_hi[kr] < rmi or r_lo[kr] > rma:
                    continue
                # candidate Eta bins (handle ±360 wrap)
                eta_lo_s = emi
                eta_hi_s = ema
                # A pixel that crosses the eta = +-180 seam (or contains the beam centre) has
                # corners on both sides, so its (min, max) eta spans > 180 deg and says nothing
                # about which bins it touches. Visit EVERY eta bin; the exact intersection below
                # returns zero for the ones it misses. The previous unwrap built the interval
                # [max corner, 360 + min corner], the INNER part of the true extent, so a bin
                # that only the outermost corner reaches was never visited and its area was
                # silently dropped (up to ~30% of a pixel at R ~ 4 px with 10 deg bins).
                if (eta_hi_s - eta_lo_s) > 180.0:
                    eta_lo_s = -1.0e30
                    eta_hi_s = 1.0e30
                for ke in range(n_eta_bins):
                    in_range = False
                    if eta_hi[ke] >= eta_lo_s and eta_lo[ke] <= eta_hi_s:
                        in_range = True
                    elif eta_hi[ke] >= eta_lo_s + 360.0 and eta_lo[ke] <= eta_hi_s + 360.0:
                        in_range = True
                    elif eta_hi[ke] >= eta_lo_s - 360.0 and eta_lo[ke] <= eta_hi_s - 360.0:
                        in_range = True
                    if not in_range:
                        continue

                    bin_idx = kr * n_eta_bins + ke
                    if is_masked:
                        bin_mask[bin_idx] = 1
                        continue

                    RMin = r_lo[kr]
                    RMax = r_hi[kr]
                    EtaMin = eta_lo[ke]
                    EtaMax = eta_hi[ke]

                    # bin corners in (Y, Z)
                    bin_y[0] = -RMin * math.sin(EtaMin * DEG2RAD)
                    bin_z[0] =  RMin * math.cos(EtaMin * DEG2RAD)
                    bin_y[1] = -RMin * math.sin(EtaMax * DEG2RAD)
                    bin_z[1] =  RMin * math.cos(EtaMax * DEG2RAD)
                    bin_y[2] = -RMax * math.sin(EtaMin * DEG2RAD)
                    bin_z[2] =  RMax * math.cos(EtaMin * DEG2RAD)
                    bin_y[3] = -RMax * math.sin(EtaMax * DEG2RAD)
                    bin_z[3] =  RMax * math.cos(EtaMax * DEG2RAD)

                    area = _pixel_bin_area(quad_y, quad_z, bin_y, bin_z,
                                           RMin, RMax, EtaMin, EtaMax,
                                           edges_scratch)
                    if area < _MIN_PIXEL_BIN_AREA:
                        continue

                    Rt_c = Rt_center[j, i]
                    corrected = area
                    if solid_angle == 1:
                        sa = sa_factor[j, i]
                        if sa > 1e-12:
                            corrected /= sa
                    if polarization == 1:
                        twoTheta = math.atan(Rt_c * px / Lsd)
                        # Standard partially-polarized factor (Kahn 1982) --
                        # see detector_mapper for why this replaced
                        # 1 - PF sin^2(2th) cos^2(deta). Both map kernels and
                        # midas_integrate_v2 must use the SAME model or the
                        # parity tests break.
                        s2t = math.sin(twoTheta)
                        c2t = math.cos(twoTheta)
                        eta_mid_rad = ((EtaMin + EtaMax) * 0.5) * DEG2RAD
                        c2e = math.cos(2.0 * (eta_mid_rad - pol_plane_eta_rad))
                        polFactor = 0.5 * (1.0 + c2t * c2t
                                           - pol_fraction * c2e * s2t * s2t)
                        if polFactor > 1e-6:
                            corrected /= polFactor
                    if flat_present == 1:
                        f = flat[j, i]
                        if f > 1e-12:
                            corrected /= f

                    R_bin_center = (RMin + RMax) * 0.5
                    deltaR = Rt_c - R_bin_center

                    if local < per_row_max:
                        out_arr[j, local, 0] = float(bin_idx)
                        out_arr[j, local, 1] = raw_y_arr[j, i]
                        out_arr[j, local, 2] = raw_z_arr[j, i]
                        out_arr[j, local, 3] = corrected
                        out_arr[j, local, 4] = deltaR
                        out_arr[j, local, 5] = area
                        local += 1
        per_row_count[j] = local

    return out_arr, per_row_count, bin_mask


# ─────────────────────────────────────────────────────────────────────────────
# Panel + residual-correction helpers (numba scalar)
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, inline="always")
def _get_panel_index_njit(y: float, z: float,
                          panels: np.ndarray, n_panels: int) -> int:
    """Linear search through (n_panels, 11) panel array.

    Layout (matches midas_integrate.panel.panels_to_array):
      col 0: yMin   col 4: dY        col 8:  dP2
      col 1: yMax   col 5: dZ        col 9:  centerY
      col 2: zMin   col 6: dTheta    col 10: centerZ
      col 3: zMax   col 7: dLsd
    Returns -1 if (y, z) is outside every panel.
    """
    for k in range(n_panels):
        if (y >= panels[k, 0] and y <= panels[k, 1]
                and z >= panels[k, 2] and z <= panels[k, 3]):
            return k
    return -1


@njit(cache=True, inline="always")
def _apply_panel_correction_njit(y: float, z: float,
                                 panels: np.ndarray, idx: int):
    """Mirror of Panel.h ApplyPanelCorrection: rotate then translate."""
    cy = panels[idx, 9]
    cz = panels[idx, 10]
    dy = y - cy
    dz = z - cz
    dTheta = panels[idx, 6]
    dY = panels[idx, 4]
    dZ = panels[idx, 5]
    if dTheta != 0.0:
        rad = DEG2RAD * dTheta
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        y_out = cy + dy * cos_t - dz * sin_t + dY
        z_out = cz + dy * sin_t + dz * cos_t + dZ
    else:
        y_out = y + dY
        z_out = z + dZ
    return y_out, z_out


@njit(cache=True, inline="always")
def _residual_corr_lookup_njit(corr_map: np.ndarray,
                               corr_n_y: int, corr_n_z: int,
                               Y: float, Z: float) -> float:
    """Bilinear sample of ΔR(Y, Z); mirrors dg_residual_corr_lookup in C.

    ``corr_map`` has shape ``(corr_n_z, corr_n_y)``; returns 0 when either
    dim is 0 (the sentinel layout used when no residual correction map is
    configured).
    """
    if corr_n_y == 0 or corr_n_z == 0:
        return 0.0
    y = Y
    z = Z
    if y < 0.0:
        y = 0.0
    if z < 0.0:
        z = 0.0
    if y >= corr_n_y - 1.0:
        y = corr_n_y - 1.001
    if z >= corr_n_z - 1.0:
        z = corr_n_z - 1.001
    y0 = int(y)
    z0 = int(z)
    fy = y - y0
    fz = z - z0
    v00 = corr_map[z0,     y0]
    v10 = corr_map[z0,     y0 + 1]
    v01 = corr_map[z0 + 1, y0]
    v11 = corr_map[z0 + 1, y0 + 1]
    return (v00 * (1.0 - fy) * (1.0 - fz)
            + v10 * fy        * (1.0 - fz)
            + v01 * (1.0 - fy) * fz
            + v11 * fy        * fz)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar pixel→(R, Eta) — numba-compatible mirror of geometry.pixel_to_REta
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, inline="always")
def _pixel_to_REta_scalar(
    Y: float, Z: float,
    Ycen: float, Zcen: float,
    TRs: np.ndarray,                    # (3, 3)
    Lsd: float, RhoD: float, px: float,
    p0: float, p1: float, p2: float, p3: float, p4: float,
    p5: float, p6: float, p7: float, p8: float, p9: float,
    p10: float, p11: float, p12: float, p13: float, p14: float,
    parallax: float,
    dLsd: float, dP2: float,
    corr_map: np.ndarray, corr_n_y: int, corr_n_z: int,
):
    """Scalar mirror of ``midas_integrate.geometry.pixel_to_REta`` for numba.

    Returns ``(Rt_pixels, EtaTilted_deg)``. Bit-equivalent to the
    ``dg_pixel_to_REta_corr`` function used by C ``MapperCore.c`` —
    includes per-panel ``dLsd / dP2`` corrections and an optional
    residual ΔR(Y, Z) lookup. ``corr_n_y == 0`` (or ``corr_n_z == 0``)
    disables the residual correction.
    """
    panelLsd = Lsd + dLsd
    panelP2 = p2 + dP2
    Yc = (-Y + Ycen) * px
    Zc = (Z - Zcen) * px
    abcpr_x = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    abcpr_y = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    abcpr_z = TRs[2, 1] * Yc + TRs[2, 2] * Zc
    XYZ_x = panelLsd + abcpr_x
    XYZ_y = abcpr_y
    XYZ_z = abcpr_z
    if abs(XYZ_x) < 1e-30:
        XYZ_x = 1e-30 if XYZ_x >= 0.0 else -1e-30
    Rad = (panelLsd / XYZ_x) * math.sqrt(XYZ_y * XYZ_y + XYZ_z * XYZ_z)
    EtaTilted = RAD2DEG * math.atan2(-XYZ_y, XYZ_z)
    if RhoD > 0.0:
        RNorm = Rad / RhoD
    else:
        RNorm = 0.0
    EtaT = 90.0 - EtaTilted
    EtaT_rad = DEG2RAD * EtaT
    R2 = RNorm * RNorm
    R3 = R2 * RNorm
    R4 = R2 * R2
    R5 = R4 * RNorm
    R6 = R4 * R2
    cos1 = math.cos(EtaT_rad + DEG2RAD * p8)
    cos2 = math.cos(2.0 * EtaT_rad + DEG2RAD * p6)
    cos3 = math.cos(3.0 * EtaT_rad + DEG2RAD * p10)
    cos4 = math.cos(4.0 * EtaT_rad + DEG2RAD * p3)
    cos5 = math.cos(5.0 * EtaT_rad + DEG2RAD * p12)
    cos6 = math.cos(6.0 * EtaT_rad + DEG2RAD * p14)
    dist = (
        p0 * R2 * cos2
        + p1 * R4 * cos4
        + panelP2 * R2
        + p4 * R6
        + p5 * R4
        + p7 * R4 * cos1
        + p9 * R3 * cos3
        + p11 * R5 * cos5
        + p13 * R6 * cos6
        + 1.0
    )
    Rt = Rad * dist / px
    Rt = Rt * (Lsd / panelLsd)
    if parallax != 0.0:
        twoTheta = math.atan(Rad / panelLsd)
        Rt = Rt + parallax * math.sin(twoTheta) / px
    Rt = Rt + _residual_corr_lookup_njit(corr_map, corr_n_y, corr_n_z, Y, Z)
    return Rt, EtaTilted


# ─────────────────────────────────────────────────────────────────────────────
# Sub-pixel-aware kernel
#
# Mirrors the C ``mapper_build_map`` in MapperCore.c including the
# ``SubPixelLevel`` + ``SubPixelCardinalWidth`` adaptive cardinal-angle
# splitting that anti-aliases bins where η ∈ {0°, ±90°, ±180°}.
#
# When SubPixelLevel == 1 (or the pixel center is outside the cardinal
# bands) the loop reduces to a single (si=0, sj=0) iteration whose
# behaviour is identical to ``map_kernel`` — but corners are computed
# in-line instead of pre-computed, so the kernel is self-contained.
# ─────────────────────────────────────────────────────────────────────────────
@njit(cache=True, parallel=True)
def map_kernel_subpixel(
    r_lo: np.ndarray, r_hi: np.ndarray,
    eta_lo: np.ndarray, eta_hi: np.ndarray,
    n_r_bins: int, n_eta_bins: int,
    NrPixelsY: int, NrPixelsZ: int,
    Ycen: float, Zcen: float, Lsd: float, RhoD: float, px: float,
    TRs: np.ndarray,                    # (3, 3) tilt matrix
    p0: float, p1: float, p2: float, p3: float, p4: float,
    p5: float, p6: float, p7: float, p8: float, p9: float,
    p10: float, p11: float, p12: float, p13: float, p14: float,
    parallax: float,
    SubPixelLevel: int, SubPixelCardinalWidth: float,
    solid_angle: int, polarization: int, pol_fraction: float,
    pol_plane_eta_rad: float,
    mask: np.ndarray, mask_present: int,
    flat: np.ndarray, flat_present: int,
    raw_y_arr: np.ndarray, raw_z_arr: np.ndarray,
    distortion_y: np.ndarray,           # (NrPixelsZ, NrPixelsY) — Δy per pixel
    distortion_z: np.ndarray,           # (NrPixelsZ, NrPixelsY) — Δz per pixel
    distortion_present: int,
    panels: np.ndarray,                 # (n_panels, 11) flat panel array
    n_panels: int,
    corr_map: np.ndarray,               # (corr_n_z, corr_n_y) ΔR map
    corr_n_y: int, corr_n_z: int,
    per_row_max: int,
):
    """Returns ``(out_arr, per_row_count, bin_mask)``.

    Output entry layout (out_arr[..., :, :]):
        col 0: bin_idx           (float, but holds an integer)
        col 1: pxList.y          (raw_y + sp_cy — non-integer for sub-pixels)
        col 2: pxList.z          (raw_z + sp_cz)
        col 3: corrected area    (frac, with solid/pol/flat applied)
        col 4: deltaR            (Rt_sub_center − R_bin_center)
        col 5: raw area          (areaWeight, geometric)
    """
    out_arr = np.zeros((NrPixelsZ, per_row_max, 6), dtype=np.float64)
    per_row_count = np.zeros(NrPixelsZ, dtype=np.int64)
    bin_mask = np.zeros(n_r_bins * n_eta_bins, dtype=np.int32)

    # Pixel-corner offsets in (dy, dz) — match geometry.PIXEL_CORNER_OFFSETS
    #   c=0 → (-0.5, -0.5)
    #   c=1 → (-0.5, +0.5)
    #   c=2 → (+0.5, +0.5)
    #   c=3 → (+0.5, -0.5)
    # Already in QUAD_ORDER traversal order.

    for j in prange(NrPixelsZ):
        edges_scratch = np.empty((50, 2), dtype=np.float64)
        bin_y = np.empty(4, dtype=np.float64)
        bin_z = np.empty(4, dtype=np.float64)
        quad_y = np.empty(4, dtype=np.float64)
        quad_z = np.empty(4, dtype=np.float64)
        local = 0
        for i in range(NrPixelsY):
            # Per-pixel panel correction (mirrors MapperCore.c lines 94-105)
            pdY = 0.0
            pdZ = 0.0
            dLsd_pix = 0.0
            dP2_pix = 0.0
            if n_panels > 0:
                pIdx = _get_panel_index_njit(float(i), float(j),
                                             panels, n_panels)
                if pIdx >= 0:
                    py_out, pz_out = _apply_panel_correction_njit(
                        float(i), float(j), panels, pIdx)
                    pdY = py_out - float(i)
                    pdZ = pz_out - float(j)
                    dLsd_pix = panels[pIdx, 7]
                    dP2_pix = panels[pIdx, 8]
            if distortion_present == 1:
                ypr = float(i) + distortion_y[j, i] + pdY
                zpr = float(j) + distortion_z[j, i] + pdZ
            else:
                ypr = float(i) + pdY
                zpr = float(j) + pdZ
            is_masked = (mask_present == 1) and (mask[j, i] == 1.0)

            # Decide spLevel: split only near cardinals when requested
            spLevel = 1
            if SubPixelLevel > 1:
                _Rt_c0, _Eta_c0 = _pixel_to_REta_scalar(
                    ypr, zpr, Ycen, Zcen, TRs, Lsd, RhoD, px,
                    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                    p10, p11, p12, p13, p14, parallax,
                    dLsd_pix, dP2_pix,
                    corr_map, corr_n_y, corr_n_z,
                )
                absEta = abs(_Eta_c0)
                if (absEta <= SubPixelCardinalWidth
                    or abs(absEta - 90.0) <= SubPixelCardinalWidth
                    or abs(absEta - 180.0) <= SubPixelCardinalWidth):
                    spLevel = SubPixelLevel

            inv_sp = 1.0 / spLevel
            for si in range(spLevel):
                for sj in range(spLevel):
                    sp_dy_lo = si * inv_sp - 0.5
                    sp_dy_hi = (si + 1) * inv_sp - 0.5
                    sp_dz_lo = sj * inv_sp - 0.5
                    sp_dz_hi = (sj + 1) * inv_sp - 0.5
                    sp_cy = (2 * si + 1) * 0.5 * inv_sp - 0.5
                    sp_cz = (2 * sj + 1) * 0.5 * inv_sp - 0.5

                    # Compute 4 corners at sub-pixel resolution.
                    # Corner indices in QUAD_ORDER traversal (matches C):
                    #   k=0 → (sp_dy_lo, sp_dz_lo)
                    #   k=1 → (sp_dy_lo, sp_dz_hi)
                    #   k=2 → (sp_dy_hi, sp_dz_hi)
                    #   k=3 → (sp_dy_hi, sp_dz_lo)
                    #
                    # BOUNDARY order, and that is correct HERE.
                    # ``_pixel_bin_intersect_njit`` walks its quad sequentially
                    # (i0 = e, i1 = e+1), so it wants the perimeter already
                    # permuted -- unlike the pure-python
                    # ``geometry.pixel_bin_intersect``, which applies
                    # QUAD_ORDER = (0, 1, 3, 2) itself and therefore wants
                    # Z-ordered corners. The two kernels take DIFFERENT
                    # conventions; do not "unify" them without changing the
                    # traversal to match.
                    rmi = 1.0e30
                    rma = -1.0e30
                    emi = 1.0e30
                    ema = -1.0e30
                    for k in range(4):
                        if k == 0:
                            cdy = sp_dy_lo; cdz = sp_dz_lo
                        elif k == 1:
                            cdy = sp_dy_lo; cdz = sp_dz_hi
                        elif k == 2:
                            cdy = sp_dy_hi; cdz = sp_dz_hi
                        else:
                            cdy = sp_dy_hi; cdz = sp_dz_lo
                        Y = ypr + cdy
                        Z = zpr + cdz
                        Rt_c, Eta_c = _pixel_to_REta_scalar(
                            Y, Z, Ycen, Zcen, TRs, Lsd, RhoD, px,
                            p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                            p10, p11, p12, p13, p14, parallax,
                            dLsd_pix, dP2_pix,
                            corr_map, corr_n_y, corr_n_z,
                        )
                        if Rt_c < rmi:
                            rmi = Rt_c
                        if Rt_c > rma:
                            rma = Rt_c
                        if Eta_c < emi:
                            emi = Eta_c
                        if Eta_c > ema:
                            ema = Eta_c
                        # cornerYZ = REta_to_YZ(Rt, Eta)
                        e_rad = Eta_c * DEG2RAD
                        quad_y[k] = -Rt_c * math.sin(e_rad)
                        quad_z[k] =  Rt_c * math.cos(e_rad)

                    rq = _quad_min_r(quad_y, quad_z)
                    if rq < rmi:
                        rmi = rq

                    # Sub-pixel center for deltaR
                    ypr_sub = ypr + sp_cy
                    zpr_sub = zpr + sp_cz
                    Rt_sub, _Eta_sub = _pixel_to_REta_scalar(
                        ypr_sub, zpr_sub, Ycen, Zcen, TRs, Lsd, RhoD, px,
                        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9,
                        p10, p11, p12, p13, p14, parallax,
                        dLsd_pix, dP2_pix,
                        corr_map, corr_n_y, corr_n_z,
                    )

                    # Find candidate R bins
                    for kr in range(n_r_bins):
                        if r_hi[kr] < rmi or r_lo[kr] > rma:
                            continue
                        eta_lo_s = emi
                        eta_hi_s = ema
                        # A pixel that crosses the eta = +-180 seam (or contains the beam centre) has
                        # corners on both sides, so its (min, max) eta spans > 180 deg and says nothing
                        # about which bins it touches. Visit EVERY eta bin; the exact intersection below
                        # returns zero for the ones it misses. The previous unwrap built the interval
                        # [max corner, 360 + min corner], the INNER part of the true extent, so a bin
                        # that only the outermost corner reaches was never visited and its area was
                        # silently dropped (up to ~30% of a pixel at R ~ 4 px with 10 deg bins).
                        if (eta_hi_s - eta_lo_s) > 180.0:
                            eta_lo_s = -1.0e30
                            eta_hi_s = 1.0e30
                        for ke in range(n_eta_bins):
                            in_range = False
                            if eta_hi[ke] >= eta_lo_s and eta_lo[ke] <= eta_hi_s:
                                in_range = True
                            elif eta_hi[ke] >= eta_lo_s + 360.0 and eta_lo[ke] <= eta_hi_s + 360.0:
                                in_range = True
                            elif eta_hi[ke] >= eta_lo_s - 360.0 and eta_lo[ke] <= eta_hi_s - 360.0:
                                in_range = True
                            if not in_range:
                                continue

                            bin_idx = kr * n_eta_bins + ke
                            if is_masked:
                                # Flag once per pixel (only on first sub-pixel)
                                if si == 0 and sj == 0:
                                    bin_mask[bin_idx] = 1
                                continue

                            RMin = r_lo[kr]
                            RMax = r_hi[kr]
                            EtaMin = eta_lo[ke]
                            EtaMax = eta_hi[ke]

                            bin_y[0] = -RMin * math.sin(EtaMin * DEG2RAD)
                            bin_z[0] =  RMin * math.cos(EtaMin * DEG2RAD)
                            bin_y[1] = -RMin * math.sin(EtaMax * DEG2RAD)
                            bin_z[1] =  RMin * math.cos(EtaMax * DEG2RAD)
                            bin_y[2] = -RMax * math.sin(EtaMin * DEG2RAD)
                            bin_z[2] =  RMax * math.cos(EtaMin * DEG2RAD)
                            bin_y[3] = -RMax * math.sin(EtaMax * DEG2RAD)
                            bin_z[3] =  RMax * math.cos(EtaMax * DEG2RAD)

                            area = _pixel_bin_area(
                                quad_y, quad_z, bin_y, bin_z,
                                RMin, RMax, EtaMin, EtaMax,
                                edges_scratch,
                            )
                            if area < _MIN_PIXEL_BIN_AREA:
                                continue

                            corrected = area
                            if solid_angle == 1:
                                # Tilt-aware solid angle: use lab-frame
                                # geometric position (panelLsd + tilt-rotated
                                # detector-frame offset). n_hat is the first
                                # column of TRs (rotation of (1,0,0)).
                                Yc_sa = (-ypr_sub + Ycen) * px
                                Zc_sa = ( zpr_sub - Zcen) * px
                                ax = TRs[0, 1] * Yc_sa + TRs[0, 2] * Zc_sa
                                ay = TRs[1, 1] * Yc_sa + TRs[1, 2] * Zc_sa
                                az = TRs[2, 1] * Yc_sa + TRs[2, 2] * Zc_sa
                                rX = (Lsd + dLsd_pix) + ax
                                rY = ay
                                rZ = az
                                ndotr = (TRs[0, 0] * rX
                                         + TRs[1, 0] * rY
                                         + TRs[2, 0] * rZ)
                                r2 = rX * rX + rY * rY + rZ * rZ
                                if r2 > 1e-30 and ndotr > 0.0:
                                    rmag = math.sqrt(r2)
                                    sa = Lsd * Lsd * ndotr / (r2 * rmag)
                                    if sa > 1e-12:
                                        corrected /= sa
                            if polarization == 1:
                                twoTheta = math.atan(Rt_sub * px / Lsd)
                                # Standard partially-polarized factor
                                # (Kahn 1982) -- see detector_mapper.
                                s2t = math.sin(twoTheta)
                                c2t = math.cos(twoTheta)
                                eta_mid_rad = ((EtaMin + EtaMax) * 0.5) * DEG2RAD
                                c2e = math.cos(2.0 * (eta_mid_rad
                                                      - pol_plane_eta_rad))
                                polFactor = 0.5 * (1.0 + c2t * c2t
                                                   - pol_fraction * c2e * s2t * s2t)
                                if polFactor > 1e-6:
                                    corrected /= polFactor
                            if flat_present == 1:
                                f = flat[j, i]
                                if f > 1e-12:
                                    corrected /= f

                            R_bin_center = (RMin + RMax) * 0.5
                            deltaR = Rt_sub - R_bin_center

                            if local < per_row_max:
                                out_arr[j, local, 0] = float(bin_idx)
                                out_arr[j, local, 1] = raw_y_arr[j, i] + sp_cy
                                out_arr[j, local, 2] = raw_z_arr[j, i] + sp_cz
                                out_arr[j, local, 3] = corrected
                                out_arr[j, local, 4] = deltaR
                                out_arr[j, local, 5] = area
                                local += 1
        per_row_count[j] = local

    return out_arr, per_row_count, bin_mask
