"""Fault probability from the INTEGRATED gap fraction along a rod, and the frame/detector-space
plumbing (a voxel "tube", background, empty-control-site search) that measures it.

Every width-based fault-probability estimator in this package (:mod:`._width`,
:func:`..rod_profile.transverse_width`) needs a floor deconvolution: the measured width is
``sqrt(true_width**2 + instrument_width**2)``, and a mismeasured floor propagates 1:1 into the
answer. This module instead uses that a Hendricks-Teller-faulted rod CONSERVES intensity across
one period -- faults only move it from the node into the gap, they never destroy it -- so the
FRACTION of one period's integrated intensity sitting outside the node window is a function of
the fault probability alone, with no floor to subtract (resolution broadens the node, but the
broadened tails are still inside the same period and still get integrated).

**Two independent pieces:**

- :func:`gap_fraction_from_f` / :func:`f_from_gap_fraction` / :func:`ht_density_convolved` /
  :func:`gap_fraction_numeric` / :func:`ht_density_fn` are the closed-form Hendricks-Teller math
  -- pure functions of L, no detector or geometry involved. They implement the SAME physics as
  :func:`midas_defect.forward_sim.hendricks_teller` (the package's differentiable, multi-parameter
  rod-fitting model); ``tests/test_polytype_gap_fraction.py`` cross-validates the two directly.
  This module's closed form exists because the differentiable model has no analytic integral or
  inverse -- what is needed here -- while being restricted to a single fault probability with no
  perpendicular/size terms, which is all the gap-fraction argument needs.
- :func:`gap_fraction_window` measures the gap fraction on REAL frames: it builds a voxel "tube"
  around a rod path, assigns each voxel to (h, k, L) via the package's tilt- and distortion-aware
  geometry, subtracts a background, weights by reciprocal volume/solid angle/polarisation, and
  integrates. :func:`empty_site_sum` / :func:`find_empty_site` / :func:`inject_gaussian` /
  :func:`gaussian_unit_total` build and validate a genuinely empty control site to test that
  measurement against, the same way a synthetic null control validates any other estimator in
  this package.

**Two backgrounds, and why there are two.** A rod's own frames span only a few degrees of omega,
so a natural background estimate reads OTHER frames at the same pixel (``bg_mode="temporal"``).
On a real, crowded, near-beam-affected detector this can leave a residual that is constant across
the transverse direction but not truly frame-independent -- something that varies smoothly with
omega on a scale comparable to the tube's own frame window (general diffuse scattering, a slow
intensity drift across the scan) will not show up correctly in far-away frames, and summed over a
growing tube radius that residual never converges. ``bg_mode="local_t"`` instead reads the
background from THE SAME frames the signal is measured in, at a larger transverse offset --
correct for exactly that failure mode, wrong if a real, extended feature (unrelated to the rod
being measured) sits in the chosen offset band. Which mode is right depends on the data; both
are provided, and ``tests/test_polytype_gap_fraction.py`` demonstrates the specific failure mode
``local_t`` fixes and the specific one it cannot (an unexcluded bright feature in its own
background band) directly.

**What this module does NOT establish.** These are estimators, validated against known synthetic
inputs. Whether a real integrated gap fraction on a real rod yields a trustworthy fault
probability depends on real-data conditions this module cannot check for you: whether the tube
radius needed to capture the full node saturates before running into other reflections, whether a
genuinely empty control site exists near the rod at all, and whether the crystal structure factor
correction (needed because it varies within one period -- see the caller's own structure-factor
source) is known accurately enough. See the notebooks for a worked real-data attempt where these
conditions were NOT all met.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import brentq

from ..geometry import pixel_to_qlab
from ..rod_profile import rod_path_geometry

__all__ = [
    "PERIOD_L",
    "Window",
    "phase_halfwidth",
    "gap_fraction_from_f",
    "f_from_gap_fraction",
    "ht_density_convolved",
    "ht_density_fn",
    "gap_fraction_numeric",
    "omega_of_frame",
    "hkl_of",
    "voxel_weights",
    "gap_fraction_window",
    "g_from_bins",
    "empty_site_sum",
    "find_empty_site",
    "gaussian_unit_total",
    "inject_gaussian",
    "plant_rod",
]

# ---------------------------------------------------------------------------------------------
# Closed-form Hendricks-Teller gap fraction. Model (the one forward_sim.hendricks_teller
# implements): I(phi) = (1-a**2) / (1 - 2*a*cos(phi) + a**2), phi = 2*pi*(L/period_L). ``a`` is
# the layer-to-layer phase correlation: a=1 a perfect stack, a=0 no correlation. For faults that
# fully randomise the stacking phase with probability ``f`` per block, the correlation after n
# blocks is (1-f)**n, so a = 1-f. For an I4/mmm (0,0,L) rod only even L are nodes, so the repeat
# probed is c/2 (one RP block) and period_L = 2 by convention here; f is therefore PER BLOCK, and
# EFFECTIVE -- a fault that shifts the phase by less than a fully random amount gives an
# L-dependent f. The integral of I over one period is 2*pi for every a (faults move intensity
# from the node to the gap, never destroy it); the fraction inside |phi| <= Phi is the closed form
# below.
# ---------------------------------------------------------------------------------------------

PERIOD_L = 2.0


def phase_halfwidth(node_half_L: float, period_L: float = PERIOD_L) -> float:
    """``Phi``: the node half-window in HT phase units (``2*pi*node_half_L/period_L``)."""
    return 2.0 * math.pi * node_half_L / period_L


def gap_fraction_from_f(f, node_half_L: float, period_L: float = PERIOD_L):
    """G for fault probability ``f`` (0 <= f < 1) per block, node window ``|L - L0| <= node_half_L``.

    ``F_in = (2/pi) * arctan((1+a)/(1-a) * tan(Phi/2))``, ``G = 1 - F_in`` -- the closed-form
    integral of the Hendricks-Teller lineshape inside the node window. Small f: the node HWHM in L
    is ``f/pi`` (``FWHM_L = 2*f/pi``, i.e. ``f = 1.5708 * FWHM_L``), and ``G ~ 2*f/(pi*Phi)`` once
    ``Phi >> f``.
    """
    f = np.asarray(f, float)
    a = 1.0 - f
    Phi = phase_halfwidth(node_half_L, period_L)
    with np.errstate(divide="ignore"):
        ratio = np.where(f > 0, (1.0 + a) / np.maximum(f, 1e-300), np.inf)
    F_in = (2.0 / math.pi) * np.arctan(ratio * math.tan(Phi / 2.0))
    return 1.0 - F_in


def f_from_gap_fraction(G: float, node_half_L: float, period_L: float = PERIOD_L) -> float:
    """Inverse of :func:`gap_fraction_from_f`. ``G`` at or below 0 returns 0; ``G`` at or above the
    uniform-rod limit (``1 - 2*node_half_L/period_L``, the gap fraction of a fully random, f=1
    rod under this node window) returns 1 -- values above that limit are not reachable by any
    f < 1 and signal a problem with the measurement, not a fault probability near 1."""
    G_uniform = 1.0 - 2.0 * node_half_L / period_L
    if not np.isfinite(G):
        return float("nan")
    if G <= 0.0:
        return 0.0
    if G >= G_uniform:
        return 1.0
    return float(brentq(lambda f: float(gap_fraction_from_f(f, node_half_L, period_L)) - G, 1e-12, 1.0 - 1e-12,
                        xtol=1e-12))


def ht_density_convolved(f: float, sigma_L: float, period_L: float = PERIOD_L, n_grid: int = 1 << 16):
    """Periodic HT profile convolved with a Gaussian of ``sigma_L`` (in L), on one period.

    Returns ``(L_grid, density)`` with ``L_grid`` in ``[-period/2, period/2)`` about a node at 0
    and the density normalised to mean 1 over the period (so its integral is ``period_L``). For
    ``f == 0`` the HT part is a delta and the result is the Gaussian comb itself. Built in Fourier
    space: the HT series coefficients are ``a**|n|`` (``a = 1 - f``), the Gaussian's are
    ``exp(-(2*pi*n*sigma/period)**2 / 2)`` -- the same Fourier content
    :func:`midas_defect.forward_sim.hendricks_teller` sums in closed form real-space.
    """
    L = (np.arange(n_grid) / n_grid - 0.5) * period_L
    n = np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    a = 1.0 - f
    coeff = a ** np.abs(n) * np.exp(-0.5 * (2.0 * math.pi * n * sigma_L / period_L) ** 2)
    phase = np.exp(-2j * math.pi * n * 0.5)                       # grid starts at -period/2
    dens = np.fft.ifft(coeff * phase).real * n_grid
    return L, dens


def ht_density_fn(f: float, sigma_L: float, L0: float = 0.0, scale: float = 1.0, period_L: float = PERIOD_L):
    """Callable ``density(L)`` for a node comb at ``L0 + period_L * j``, for :func:`plant_rod`."""
    Lg, dens = ht_density_convolved(f, sigma_L, period_L)

    def fn(L):
        x = (np.asarray(L, float) - L0 + period_L / 2.0) % period_L - period_L / 2.0
        return scale * np.interp(x, Lg, dens, period=period_L)
    return fn


def gap_fraction_numeric(f: float, sigma_L: float, node_half_L: float, period_L: float = PERIOD_L) -> float:
    """G of the resolution-convolved profile, by direct integration on the Fourier grid -- the
    number a real measurement (with a real, finite resolution) should be compared to, unlike
    :func:`gap_fraction_from_f`'s zero-resolution ideal."""
    Lg, dens = ht_density_convolved(f, sigma_L, period_L)
    return float(dens[np.abs(Lg) > node_half_L].sum() / dens.sum())


# ---------------------------------------------------------------------------------------------
# Measuring the gap fraction on real frames: a voxel tube around a rod path, background, weights.
# ---------------------------------------------------------------------------------------------

@dataclass
class Window:
    """One node's measurement: per-L-bin sums (``S``/``var``) and the resulting gap fraction."""
    L0: int
    ok: bool
    reason: str = ""
    L_centres: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None            # weighted background-subtracted sum per bin
    var: Optional[np.ndarray] = None          # its Poisson variance
    coverage: Optional[np.ndarray] = None
    interpolated: Optional[np.ndarray] = None
    G: float = float("nan")
    G_err: float = float("nan")
    node_sum: float = float("nan")
    gap_sum: float = float("nan")
    n_node_vox_over_censor: int = 0
    node_share_over_censor: float = 0.0
    annulus_mean_per_vox: float = float("nan")
    annulus_sd_per_vox: float = float("nan")
    n_vox: int = 0
    notes: List[str] = field(default_factory=list)


def omega_of_frame(geom, k):
    """Reported omega (deg) of frame index(es) ``k``."""
    return geom.omega_first_deg + geom.omega_step_deg * np.asarray(k, float)


def hkl_of(rows, cols, omega_rep_deg, geom, sign, UB_inv, qlab_cache=None):
    """(h, k, L) of voxels, through the full tilt- and distortion-aware geometry.
    ``rows``/``cols`` are pixel centres, ``omega_rep_deg`` the reported omega per voxel."""
    if qlab_cache is None:
        q_lab = pixel_to_qlab(np.asarray(rows, float), np.asarray(cols, float), geom, device="cpu",
                              dtype=torch.float64).numpy()
    else:
        q_lab = qlab_cache
    ww = np.radians(sign * np.asarray(omega_rep_deg, float))
    c, s = np.cos(ww), np.sin(ww)
    qs = np.stack([c * q_lab[..., 0] + s * q_lab[..., 1], -s * q_lab[..., 0] + c * q_lab[..., 1], q_lab[..., 2]],
                  axis=-1)
    return qs @ UB_inv.T, q_lab


def voxel_weights(rows, cols, k, geom, sign, UB_inv):
    """``V / (dOmega * P)`` per voxel, up to a constant -- reciprocal-space volume element over
    solid angle and polarisation, so a voxel sum approximates an intensity integral rather than
    a raw pixel count. ``V`` is ``|det d(hkl)/d(frame,row,col)|`` by finite differences,
    ``dOmega ~ cos**3(2theta)`` (flat-detector form), ``P = 1 - (q_lab_y/k0)**2`` (horizontal
    polarisation along lab Y)."""
    r = np.asarray(rows, float); c = np.asarray(cols, float); kk = np.asarray(k, float)
    om = omega_of_frame(geom, kk)
    d = geom.omega_step_deg
    cols_d = []
    for (dr, dc, dk) in ((0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0), (0, -0.5, 0), (0, 0, 0.5), (0, 0, -0.5)):
        h, _ = hkl_of(r + dr, c + dc, om + dk * d, geom, sign, UB_inv)
        cols_d.append(h)
    J = np.stack([cols_d[0] - cols_d[1], cols_d[2] - cols_d[3], cols_d[4] - cols_d[5]], axis=-1)
    V = np.abs(np.linalg.det(J))
    _, q_lab = hkl_of(r, c, om, geom, sign, UB_inv)
    k0 = 2.0 * math.pi / geom.wavelength_A
    kf = q_lab.copy(); kf[..., 0] += k0
    cos2t = kf[..., 0] / np.linalg.norm(kf, axis=-1)
    dOmega = cos2t ** 3
    P = 1.0 - (q_lab[..., 1] / k0) ** 2
    return V / (dOmega * P)


def _dist_to_points(rr, cc, pr, pc):
    """Distance of each (rr, cc) to the nearest of points (pr, pc), and that point's index."""
    d2 = (rr[:, None] - pr[None, :]) ** 2 + (cc[:, None] - pc[None, :]) ** 2
    j = np.argmin(d2, axis=1)
    return np.sqrt(d2[np.arange(rr.size), j]), j


def _exclusion_valid(r, c, k, mask, exclusions, other_rod_paths, r_lo, r_hi, c_lo, c_hi, T_other_rods, K):
    """Shared validity mask (unmasked, not on a predicted reflection, not on another rod's tube) for
    a generic ``(r, c, k)`` voxel set -- used for both the signal tube and, in
    ``bg_mode="local_t"``, the local background annulus, so both see the identical exclusion list:
    a bright unexcluded feature must not leak into the background estimate either."""
    valid = ~mask[r, c]
    for (er, ec, ek, erad, ehalf) in exclusions:
        if not (r_lo - erad <= er <= r_hi + erad and c_lo - erad <= ec <= c_hi + erad):
            continue
        valid &= ~((np.hypot(r - er, c - ec) <= erad) & (np.abs(k - ek) <= ehalf))
    for (orow, ocol, ofr) in other_rod_paths:
        orow = np.asarray(orow, float); ocol = np.asarray(ocol, float); ofr = np.asarray(ofr, float)
        box = (orow >= r_lo - T_other_rods - 2) & (orow <= r_hi + T_other_rods + 2) \
            & (ocol >= c_lo - T_other_rods - 2) & (ocol <= c_hi + T_other_rods + 2)
        if not box.any():
            continue
        orow, ocol, ofr = orow[box], ocol[box], ofr[box]
        od, oj = _dist_to_points(r.astype(float), c.astype(float), orow, ocol)
        valid &= ~((od <= T_other_rods) & (np.abs(k - ofr[oj]) <= K + 0.5))
    return valid


def gap_fraction_window(frames: np.ndarray, mask: np.ndarray, geom, dom_U: np.ndarray, dom_B: np.ndarray,
                        sign: int, L0: int, *, T_px: float = 20.0, K: int = 4, K_bg: int = 6,
                        dL: float = 0.05, node_half_L: float = 0.25, censor: float = 119000.0,
                        node_min_cov: float = 0.95, gap_min_cov: float = 0.8, min_bg_frames: int = 12,
                        exclusions: Sequence[Tuple[float, float, float, float, float]] = (),
                        other_rod_paths: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]] = (),
                        annulus_px: Tuple[float, float] = (0.0, 0.0),
                        hk: Tuple[float, float] = (0.0, 0.0),
                        bg_mode: str = "temporal",
                        bg_margin: float = 5.0, bg_width: float = 10.0,
                        min_bg_px: int = 20) -> Window:
    """Measured gap fraction G for one node window, around one even node ``L0`` of rod ``hk``.

    **Tube.** Voxels (frame, row, col) whose pixel lies within ``T_px`` of the rod's detector
    path (L in ``[L0-1.1, L0+1.1]``) and whose frame lies within ``K`` of the frame where the
    nearest path point crosses the Ewald sphere.

    **Coordinates.** Every voxel is mapped to (h, k, L) with :func:`hkl_of`, then binned in L
    (``dL``); no pixel-to-L interpolation.

    **Background.** See the module docstring for ``bg_mode``. ``"temporal"``: per pixel, the
    median raw count over frames more than ``K_bg`` from every rod crossing near that pixel.
    ``"local_t"``: per frame in the tube's own K window, the median raw count over
    unmasked/unexcluded pixels at transverse offset ``[T_px+bg_margin, T_px+bg_margin+bg_width]``
    from the path IN THAT SAME FRAME.

    **Weights.** :func:`voxel_weights` -- only relative weights matter for G.

    **Validity.** Masked pixels, voxels near a predicted reflection of any domain
    (``exclusions``: ``(row, col, frame, radius_px, half_frames)``), and voxels near another
    diffuse rod's path (``other_rod_paths``: ``(rows, cols, frames)``) are invalid. A bin's
    coverage is the valid share of its tube reciprocal volume. Node bins need coverage
    ``>= node_min_cov``; gap bins below ``gap_min_cov`` are linearly interpolated along L from
    the nearest valid bins.

    **Linearity.** Any node-window voxel with raw counts above ``censor`` is counted in
    ``n_node_vox_over_censor``; the caller decides what to do, nothing is dropped silently.

    ``annulus_px`` (t_in, t_out) > 0 adds a residual-background diagnostic over
    ``T_px < t_in <= t <= t_out`` (not used in G).

    ``G = sum(gap bins) / sum(all bins)``, ``node = |L - L0| <= node_half_L``, ``gap`` = the rest
    out to +/-1.
    """
    n_f, n_r, n_c = frames.shape
    UB_inv = np.linalg.inv(dom_U @ dom_B)
    Ls = np.round(np.arange(L0 - 1.1, L0 + 1.1 + 1e-9, 0.01), 4)
    path = rod_path_geometry(dom_U, dom_B, hk[0], hk[1], Ls, geom, omega_sign=sign)
    if len(path) < len(Ls) - 2:
        return Window(L0, False, f"rod path incomplete ({len(path)}/{len(Ls)} points)")
    pk = (path.omega_deg - geom.omega_first_deg) / geom.omega_step_deg
    if pk.min() - K < -0.5 or pk.max() + K > n_f - 0.5:
        return Window(L0, False, "frame window off stack edge")
    bg_hi = T_px + bg_margin + bg_width if bg_mode == "local_t" else 0.0
    t_out = max(T_px, annulus_px[1], bg_hi)
    r_lo = int(math.floor(path.row.min() - t_out - 2)); r_hi = int(math.ceil(path.row.max() + t_out + 2))
    c_lo = int(math.floor(path.col.min() - t_out - 2)); c_hi = int(math.ceil(path.col.max() + t_out + 2))
    r_lo, c_lo = max(r_lo, 0), max(c_lo, 0)
    r_hi, c_hi = min(r_hi, n_r - 1), min(c_hi, n_c - 1)
    rr, cc = np.mgrid[r_lo:r_hi + 1, c_lo:c_hi + 1]
    rr, cc = rr.ravel(), cc.ravel()
    dist, jn = _dist_to_points(rr.astype(float), cc.astype(float), path.row, path.col)
    near = dist <= t_out
    rr, cc, dist, jn = rr[near], cc[near], dist[near], jn[near]
    k_near = pk[jn]

    # every voxel within t_out of the path, K frames either side of its own nearest crossing -- the
    # tube (dist<=T_px), the annulus_px diagnostic band, and (bg_mode="local_t") the background band
    # are all subsets of this ONE voxel set, so they see identical exclusion handling.
    kc = np.round(k_near).astype(int)
    vox_p, vox_k = [], []
    for off in range(-K, K + 1):
        kk = kc + off
        inside = (kk >= 0) & (kk < n_f) & (np.abs(kk - k_near) <= K + 0.5)
        vox_p.append(np.flatnonzero(inside)); vox_k.append(kk[inside])
    vp = np.concatenate(vox_p); vk = np.concatenate(vox_k)
    vr, vc, vt = rr[vp], cc[vp], dist[vp]
    in_tube = vt <= T_px
    valid = _exclusion_valid(vr, vc, vk, mask, exclusions, other_rod_paths, r_lo, r_hi, c_lo, c_hi, T_px, K)

    if bg_mode == "temporal":
        # per pixel: exclude frames within K_bg of ANY path point within t_out + 3 px, median the rest
        excl = np.zeros((rr.size, n_f), bool)
        kidx = np.arange(n_f)
        for j in range(len(path)):
            close = np.hypot(rr - path.row[j], cc - path.col[j]) <= t_out + 3.0
            if close.any():
                excl[np.ix_(close, np.abs(kidx - pk[j]) <= K_bg)] = True
        n_bg = (~excl).sum(axis=1)
        pix = frames[:, rr, cc].T                               # (n_pix, n_f)
        bg = np.where(excl, np.nan, pix)
        with np.errstate(all="ignore"):
            bg_med = np.nanmedian(bg, axis=1)
        bg_ok = n_bg >= min_bg_frames
        bg_med_v, bg_ok_v, bg_n_v = bg_med[vp], bg_ok[vp], n_bg[vp]
    elif bg_mode == "local_t":
        # per frame present in the K window: median raw count of valid pixels at transverse offset
        # [T_px+bg_margin, T_px+bg_margin+bg_width] from the path IN THAT SAME FRAME (a subset of
        # the shared voxel set above, own exclusion mask `valid`)
        band = (vt >= T_px + bg_margin) & (vt <= T_px + bg_margin + bg_width)
        bg_level = np.full(n_f, np.nan); bg_n = np.zeros(n_f, int)
        for kk in np.unique(vk[band]):
            sel = band & valid & (vk == kk)
            n = int(sel.sum())
            bg_n[kk] = n
            if n >= min_bg_px:
                bg_level[kk] = float(np.median(frames[kk, vr[sel], vc[sel]]))
        bg_med_v = bg_level[vk]
        bg_ok_v = np.isfinite(bg_med_v)
        bg_n_v = bg_n[vk]
    else:
        raise ValueError(f"unknown bg_mode {bg_mode!r}")

    hkl, _ = hkl_of(vr, vc, omega_of_frame(geom, vk), geom, sign, UB_inv)
    L = hkl[:, 2]
    w = voxel_weights(vr, vc, vk, geom, sign, UB_inv)
    w = w / np.median(w)
    raw = frames[vk, vr, vc].astype(float)
    val = raw - np.where(bg_ok_v, bg_med_v, 0.0)
    valid = valid & bg_ok_v

    edges = np.round(np.arange(L0 - 1.0, L0 + 1.0 + 1e-9, dL), 6)
    centres = 0.5 * (edges[:-1] + edges[1:])
    nb = centres.size
    b = np.digitize(L, edges) - 1
    inb = (b >= 0) & (b < nb)
    tube = in_tube & inb
    S = np.zeros(nb); var = np.zeros(nb); Vall = np.zeros(nb); Vok = np.zeros(nb)
    np.add.at(Vall, b[tube], w[tube])
    ok_t = tube & valid
    np.add.at(Vok, b[ok_t], w[ok_t])
    np.add.at(S, b[ok_t], (w * val)[ok_t])
    bg_var = 1.5708 * np.maximum(np.where(bg_ok_v, bg_med_v, 0.0), 0.0) / np.maximum(bg_n_v, 1)
    np.add.at(var, b[ok_t], (w * w * (np.maximum(raw, 0.0) + bg_var))[ok_t])
    with np.errstate(all="ignore"):
        cov = np.where(Vall > 0, Vok / Vall, 0.0)
    node = np.abs(centres - L0) <= node_half_L + 1e-9
    win = Window(L0, True, L_centres=centres, coverage=cov, n_vox=int(ok_t.sum()))
    if np.any(cov[node] < node_min_cov):
        win.ok, win.reason = False, f"node coverage {cov[node].min():.3f} < {node_min_cov}"
    # scale partially covered bins, interpolate starved gap bins
    with np.errstate(all="ignore"):
        S_c = np.where(cov > 0, S / cov, np.nan)
        var_c = np.where(cov > 0, var / cov ** 2, np.nan)
    interp = (~node) & (cov < gap_min_cov)
    if interp.any():
        good = ~interp & np.isfinite(S_c)
        if good.sum() < 2:
            win.ok, win.reason = False, "too few valid gap bins"
        else:
            S_c[interp] = np.interp(centres[interp], centres[good], S_c[good])
            var_c[interp] = np.interp(centres[interp], centres[good], var_c[good])
    win.S, win.var, win.interpolated = S_c, var_c, interp

    node_vox = tube & (np.abs(L - L0) <= node_half_L) & valid
    over = node_vox & (raw > censor)
    win.n_node_vox_over_censor = int(over.sum())
    tot_node = float(np.sum((w * val)[node_vox]))
    win.node_share_over_censor = float(np.sum((w * val)[over]) / tot_node) if tot_node > 0 else float("nan")

    if annulus_px[1] > 0:
        ann = (vt > annulus_px[0]) & (vt <= annulus_px[1]) & inb & valid
        if ann.any():
            win.annulus_mean_per_vox = float(np.mean(val[ann]))
            win.annulus_sd_per_vox = float(np.std(val[ann]))

    if win.ok:
        node_sum = float(np.nansum(S_c[node])); gap_sum = float(np.nansum(S_c[~node]))
        tot = node_sum + gap_sum
        win.node_sum, win.gap_sum = node_sum, gap_sum
        if tot <= 0:
            win.ok, win.reason = False, "non-positive total"
        else:
            win.G = gap_sum / tot
            vn, vg = float(np.nansum(var_c[node])), float(np.nansum(var_c[~node]))
            # G = g / (g + n): dG/dg = n / tot^2, dG/dn = -g / tot^2
            win.G_err = math.sqrt((node_sum / tot ** 2) ** 2 * vg + (gap_sum / tot ** 2) ** 2 * vn)
    return win


def g_from_bins(win: Window, node_half_L: float, f2: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
    """``(G, G_err, node_sum, gap_sum)`` from a window's bins, optionally dividing each bin by a
    structure-factor-squared array ``f2`` at its L centres (needed because ``|F_block(L)|**2``
    varies within one period, so a raw sum is not directly a gap fraction of the fault-only
    density), for a different ``node_half_L`` than the window was built with."""
    S, var, Lc = win.S, win.var, win.L_centres
    if f2 is not None:
        S = S / f2; var = var / f2 ** 2
    node = np.abs(Lc - win.L0) <= node_half_L + 1e-9
    n, g = float(np.nansum(S[node])), float(np.nansum(S[~node]))
    tot = n + g
    if not np.isfinite(tot) or tot <= 0:
        return float("nan"), float("nan"), n, g
    vn, vg = float(np.nansum(var[node])), float(np.nansum(var[~node]))
    return g / tot, math.sqrt((n / tot ** 2) ** 2 * vg + (g / tot ** 2) ** 2 * vn), n, g


# ---------------------------------------------------------------------------------------------
# Empty-control-site search and synthetic injection -- the null/positive control this estimator
# needs, at a genuinely blank patch of the SAME detector rather than a guessed reciprocal-space
# offset (which, near a real crystal's beam, may not be blank at all).
# ---------------------------------------------------------------------------------------------

def empty_site_sum(frames: np.ndarray, mask: np.ndarray, r0: float, c0: float, kf: int, *, T_px: float,
                   K: int = 4, K_bg: int = 6, bg_mode: str = "temporal", bg_margin: float = 5.0,
                   bg_width: float = 10.0, min_bg_frames: int = 12, min_bg_px: int = 20,
                   exclusions: Sequence[Tuple[float, float, float, float, float]] = (),
                   other_rod_paths: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]] = ()
                   ) -> Tuple[Optional[float], str]:
    """Background-subtracted sum of raw counts in a disk of radius ``T_px`` about a DETECTOR
    position (``r0``, ``c0``, frame ``kf``), frames ``kf +/- K``, using the identical background
    definition (``bg_mode``) as :func:`gap_fraction_window`'s tube -- but with no rod path, no hkl
    mapping, no HT model. This is the building block of the empty-control-site search: a
    candidate site is scored by this sum alone, which should be near zero at a genuinely empty
    site. Returns ``(sum, "")`` or ``(None, reason)``.
    """
    n_f, n_r, n_c = frames.shape
    if kf - K < 0 or kf + K >= n_f:
        return None, "frame window off stack edge"
    t_out = T_px + (bg_margin + bg_width if bg_mode == "local_t" else 0.0)
    ri, ci = int(round(r0)), int(round(c0))
    R = int(math.ceil(t_out)) + 2
    if ri - R < 0 or ri + R >= n_r or ci - R < 0 or ci + R >= n_c:
        return None, "site window off detector edge"
    rr, cc = np.mgrid[ri - R:ri + R + 1, ci - R:ci + R + 1]
    rr, cc = rr.ravel(), cc.ravel()
    t = np.hypot(rr - r0, cc - c0)
    near = t <= t_out
    rr, cc, t = rr[near], cc[near], t[near]
    r_lo, r_hi, c_lo, c_hi = ri - R, ri + R, ci - R, ci + R

    vox_p, vox_k = [], []
    for off in range(-K, K + 1):
        vox_p.append(np.arange(rr.size)); vox_k.append(np.full(rr.size, kf + off))
    vp = np.concatenate(vox_p); vk = np.concatenate(vox_k)
    vr, vc, vt = rr[vp], cc[vp], t[vp]
    in_tube = vt <= T_px
    valid = _exclusion_valid(vr, vc, vk, mask, exclusions, other_rod_paths, r_lo, r_hi, c_lo, c_hi, T_px, K)

    if bg_mode == "temporal":
        excl = np.abs(np.arange(n_f)[None, :] - kf) <= K_bg
        excl = np.repeat(excl, rr.size, axis=0)
        pix = frames[:, rr, cc].T
        bg = np.where(excl, np.nan, pix)
        with np.errstate(all="ignore"):
            bg_med = np.nanmedian(bg, axis=1)
        n_bg = (~excl).sum(axis=1)
        bg_ok = n_bg >= min_bg_frames
        bg_med_v, bg_ok_v = bg_med[vp], bg_ok[vp]     # vp indexes the (rr, cc) pixel list directly
    elif bg_mode == "local_t":
        band = (vt >= T_px + bg_margin) & (vt <= T_px + bg_margin + bg_width)
        bg_level = np.full(n_f, np.nan); bg_n = np.zeros(n_f, int)
        for kk in np.unique(vk[band]):
            sel = band & valid & (vk == kk)
            n = int(sel.sum())
            bg_n[kk] = n
            if n >= min_bg_px:
                bg_level[kk] = float(np.median(frames[kk, vr[sel], vc[sel]]))
        bg_med_v = bg_level[vk]
        bg_ok_v = np.isfinite(bg_med_v)
    else:
        raise ValueError(f"unknown bg_mode {bg_mode!r}")

    raw = frames[vk, vr, vc].astype(float)
    val = raw - np.where(bg_ok_v, bg_med_v, 0.0)
    ok = in_tube & valid & bg_ok_v
    if ok.sum() < 0.5 * in_tube.sum():
        return None, f"too few valid tube pixels ({int(ok.sum())}/{int(in_tube.sum())})"
    return float(np.sum(val[ok])), ""


def find_empty_site(frames: np.ndarray, mask: np.ndarray, r0: float, c0: float, kf: int, bc_row: float,
                    bc_col: float, *, n_angles: int = 24, min_angle_deg: float = 20.0, **kw
                    ) -> Tuple[Optional[dict], list]:
    """Rotate ``(r0, c0)`` about the beam centre through ``n_angles`` positions at the SAME radius
    and frame, skip any within ``min_angle_deg`` of the real site, and return the candidate with
    the smallest ``|sum|`` via :func:`empty_site_sum`, plus the full ranked list (for logging).
    ``**kw`` is passed through to :func:`empty_site_sum` (``T_px``, ``K``, ``bg_mode``,
    ``exclusions``, ``other_rod_paths``, ...)."""
    radius = math.hypot(r0 - bc_row, c0 - bc_col)
    theta0 = math.atan2(r0 - bc_row, c0 - bc_col)
    results = []
    for i in range(n_angles):
        theta = theta0 + 2 * math.pi * i / n_angles
        dtheta = math.degrees(abs(((theta - theta0 + math.pi) % (2 * math.pi)) - math.pi))
        if dtheta < min_angle_deg:
            continue
        rc = bc_row + radius * math.sin(theta); cc_ = bc_col + radius * math.cos(theta)
        s, reason = empty_site_sum(frames, mask, rc, cc_, kf, **kw)
        results.append(dict(theta_deg=math.degrees(theta), row=rc, col=cc_, sum=s, reason=reason))
    ok = [r for r in results if r["sum"] is not None]
    best = min(ok, key=lambda r: abs(r["sum"])) if ok else None
    return best, results


def gaussian_unit_total(sigma_t_px: float, sigma_k_frames: float, K: int = 4) -> float:
    """Total counts :func:`inject_gaussian` would add for ``amp=1``, ``poisson=False`` -- the
    exact grid sum it uses, so a caller can solve for the ``amp`` that injects a chosen total
    without a throwaway calibration call."""
    R = int(math.ceil(4 * sigma_t_px)) + 2
    rr, cc = np.mgrid[-R:R + 1, -R:R + 1]
    space = np.exp(-0.5 * (rr ** 2 + cc ** 2) / sigma_t_px ** 2).sum()
    frame = sum(math.exp(-0.5 * (off / sigma_k_frames) ** 2) for off in range(-K, K + 1))
    return float(space * frame)


def inject_gaussian(frames: np.ndarray, r0: float, c0: float, kf: int, amp: float, sigma_t_px: float, *,
                    K: int = 4, sigma_k_frames: float = 1.0, rng=None, poisson: bool = True
                    ) -> Tuple[float, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    """Add a Gaussian bump (space: ``sigma_t_px``; frame: ``sigma_k_frames``) centred at
    ``(r0, c0, kf)``, for the empty-control-site injection test (paired with
    :func:`empty_site_sum`). Returns ``(total_added, applied)`` where ``applied`` is
    ``[(kk, rr, cc, add), ...]`` -- enough for a caller to exactly UNDO the injection afterwards
    (``frames[kk, rr, cc] -= add`` for each entry), which a single scalar total cannot do."""
    n_f = frames.shape[0]
    R = int(math.ceil(4 * sigma_t_px)) + 2
    ri, ci = int(round(r0)), int(round(c0))
    rr, cc = np.mgrid[ri - R:ri + R + 1, ci - R:ci + R + 1]
    total = 0.0
    applied = []
    for off in range(-K, K + 1):
        kk = kf + off
        if not (0 <= kk < n_f):
            continue
        lam = amp * math.exp(-0.5 * (off / sigma_k_frames) ** 2) \
            * np.exp(-0.5 * (((rr - r0) ** 2 + (cc - c0) ** 2) / sigma_t_px ** 2))
        lam = np.clip(lam, 0.0, None)
        add = rng.poisson(lam).astype(frames.dtype) if poisson else lam.astype(frames.dtype)
        frames[kk, rr, cc] += add
        total += float(add.sum())
        applied.append((kk, rr, cc, add))
    return total, applied


def plant_rod(frames: np.ndarray, geom, dom_U, dom_B, sign, hk, L_lo, L_hi, density_fn, *, T_px: float,
              sigma_t_px: float, sigma_k_frames: float, K: int, rng=None, poisson: bool = True) -> np.ndarray:
    """Add a synthetic rod along ``(hk[0], hk[1], L)`` in place of real data, same geometry as
    :func:`gap_fraction_window`. Expected counts per voxel =
    ``density_fn(L) * gauss(t/sigma_t) * gauss(dk/sigma_k) / weight``, so that the weighted
    estimator sees ``density_fn(L)`` times a constant. Returns the added counts."""
    n_f, n_r, n_c = frames.shape
    UB_inv = np.linalg.inv(dom_U @ dom_B)
    Ls = np.round(np.arange(L_lo, L_hi + 1e-9, 0.01), 4)
    path = rod_path_geometry(dom_U, dom_B, hk[0], hk[1], Ls, geom, omega_sign=sign)
    pk = (path.omega_deg - geom.omega_first_deg) / geom.omega_step_deg
    r_lo = max(int(math.floor(path.row.min() - T_px - 2)), 0); r_hi = min(int(math.ceil(path.row.max() + T_px + 2)), n_r - 1)
    c_lo = max(int(math.floor(path.col.min() - T_px - 2)), 0); c_hi = min(int(math.ceil(path.col.max() + T_px + 2)), n_c - 1)
    rr, cc = np.mgrid[r_lo:r_hi + 1, c_lo:c_hi + 1]
    rr, cc = rr.ravel(), cc.ravel()
    dist, jn = _dist_to_points(rr.astype(float), cc.astype(float), path.row, path.col)
    near = dist <= T_px
    rr, cc, dist, jn = rr[near], cc[near], dist[near], jn[near]
    k_near = pk[jn]
    parts = []
    for off in range(-K, K + 1):
        kk = np.round(k_near).astype(int) + off
        ok = (kk >= 0) & (kk < n_f)
        if ok.any():
            parts.append((rr[ok], cc[ok], kk[ok], dist[ok], k_near[ok]))
    r_ = np.concatenate([p[0] for p in parts]); c_ = np.concatenate([p[1] for p in parts])
    k_ = np.concatenate([p[2] for p in parts]); t_ = np.concatenate([p[3] for p in parts])
    kn = np.concatenate([p[4] for p in parts])
    hkl, _ = hkl_of(r_, c_, omega_of_frame(geom, k_), geom, sign, UB_inv)
    w = voxel_weights(r_, c_, k_, geom, sign, UB_inv)
    w = w / np.median(w)                      # so density_fn is in counts per median voxel
    lam = density_fn(hkl[:, 2]) * np.exp(-0.5 * (t_ / sigma_t_px) ** 2) \
        * np.exp(-0.5 * ((k_ - kn) / sigma_k_frames) ** 2) / w
    lam = np.clip(lam, 0.0, None)
    cnt = rng.poisson(lam).astype(frames.dtype) if poisson else lam.astype(frames.dtype)
    added = np.zeros_like(frames)
    np.add.at(added, (k_, r_, c_), cnt)
    frames += added
    return added
