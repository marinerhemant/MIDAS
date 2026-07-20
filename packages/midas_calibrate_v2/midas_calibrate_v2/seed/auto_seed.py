"""Robust initial-seed generator for midas_calibrate_v2.calibrate().

The built-in seeder fails on several common detector configurations
because its simple pedestal-based σ estimator is fragile:

* **Flat-field-corrected images** (e.g. some area detectors): large
  negative pixels get clipped to 0, collapsing the noise estimate and
  making the threshold too permissive — inner ring arcs fragment into
  noise.
* **Eiger / Pilatus with UINT32_MAX dead-pixel sentinels**: those
  4.29 × 10⁹-count values inflate the σ estimator by several orders of
  magnitude, causing the threshold to reject every real ring pixel.
* **Dense-ring geometries** (short-wavelength or long Lsd): the fixed
  ``100·(1+std//100)`` formula is too permissive, flooding the detector
  with apparent arcs and producing an ambiguous ring assignment.

``make_seed`` fixes all three by combining:

* **Sentinel-aware statistics** — UINT32_MAX, NaN, ±Inf, negatives, and
  pixels above 1 × 10⁹ counts are masked before any σ estimate or
  median filter is computed.
* **Median-background subtract** — iterative diplib MedianFilter (falls
  back to a 4× downsample + scipy path) removes slowly-varying
  background without absorbing ring signal.
* **Dual-criterion threshold** — ``max(SNR·σ, top-1% of residual)``
  keeps the bright-pixel fraction bounded even when σ is tiny.
* **Adaptive min-arc length** — ``0.10 × min(image.shape)`` scales
  automatically from a 512-px Eiger to a 2880-px large-area detector.
* **midas_hkls ring catalog** — ``generate_hkls`` covers all 230 space
  groups via proper Hall-symbol symmetry operations.
* **Composite assignment score** — penalises missing inner rings, index
  gaps, and high RMS; Lsd is refined by linear LSQ over all matched
  (R, tan 2θ) pairs rather than by the single anchor ring.

Usage::

    from midas_calibrate_v2.seed.auto_seed import make_seed

    seed = make_seed(img, wavelength_A=0.116, px_um=150.0, calibrant="CeO2")

    # Pass the seed into calibrate() to bypass the built-in seeder:
    from midas_calibrate_v2 import calibrate
    result = calibrate(
        img,
        wavelength   = seed.wavelength_A,
        pxY          = seed.px_um,
        calibrant    = seed.calibrant_name,
        initial_BC_y = seed.BC_y,
        initial_BC_z = seed.BC_z,
        initial_Lsd  = seed.Lsd_um,
    )
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

# ── optional fast background estimator ──────────────────────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS OpenMP guard
try:
    import diplib as _DIPLIB
    _HAS_DIPLIB = True
except Exception:
    _DIPLIB = None
    _HAS_DIPLIB = False


# ══════════════════════════════════════════════════════════════════════════════
# Public result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Seed:
    """Output of make_seed() — everything calibrate() needs as a seed."""
    BC_y: float        # beam-centre column (pixels, MIDAS convention)
    BC_z: float        # beam-centre row
    Lsd_um: float      # sample-to-detector distance (µm)
    wavelength_A: float
    px_um: float
    calibrant_name: str   # canonical name for calibrate() e.g. "CeO2"
    first_ring: int    # 1-based index of innermost matched ring
    n_measured: int    # number of ring radii used in the assignment
    rms_px: float      # per-ring Lsd assignment residual (px)
    elapsed_s: float = 0.0
    notes: str = ""

    def __str__(self):
        return (
            f"Seed(BC=({self.BC_y:.2f}, {self.BC_z:.2f}) px, "
            f"Lsd={self.Lsd_um/1000:.3f} mm, "
            f"first_ring={self.first_ring}, "
            f"n={self.n_measured}, rms={self.rms_px:.3f} px, "
            f"t={self.elapsed_s:.1f}s)"
        )

    def __repr__(self):
        return self.__str__()


# ══════════════════════════════════════════════════════════════════════════════
# Calibrant registry + ring table via midas_hkls
# ══════════════════════════════════════════════════════════════════════════════

# NIST SRM lattice constants.  All entries recognised by midas_calibrate_v2's
# CALIBRANTS dict can be added here — any material midas_hkls supports works.
_CALIBRANTS = {
    "CeO2":  {"sg": 225, "a": 5.4116, "b": 5.4116, "c": 5.4116,
               "alpha": 90.0, "beta": 90.0, "gamma": 90.0},  # SRM 674b  Fm-3m
    "LaB6":  {"sg": 221, "a": 4.1569, "b": 4.1569, "c": 4.1569,
               "alpha": 90.0, "beta": 90.0, "gamma": 90.0},  # SRM 660c  Pm-3m
    "Si":    {"sg": 227, "a": 5.4307, "b": 5.4307, "c": 5.4307,
               "alpha": 90.0, "beta": 90.0, "gamma": 90.0},  # SRM 640d  Fd-3m
    "Al2O3": {"sg": 167, "a": 4.7589, "b": 4.7589, "c": 12.9920,
               "alpha": 90.0, "beta": 90.0, "gamma": 120.0}, # SRM 676a  R-3c
}


def _canonical_name(calibrant: str) -> str:
    """Case-insensitive lookup: 'ceo2' / 'CeO2' / 'CEO2' → 'CeO2'."""
    lc = calibrant.lower().replace("-", "").replace(" ", "")
    for key in _CALIBRANTS:
        if key.lower().replace("-", "") == lc:
            return key
    raise KeyError(f"Unknown calibrant {calibrant!r}. Known: {list(_CALIBRANTS)}")


def _resolve_calibrant(calibrant: Union[str, Dict]) -> dict:
    """Normalize a calibrant to a full lattice spec for ring-table generation.

    Accepts either a registered name (``"CeO2"``/``"LaB6"``/``"Si"``/``"Al2O3"``)
    or a dict describing an arbitrary powder calibrant.  Mirrors the
    ``Union[str, Dict]`` contract of :func:`midas_calibrate_v2.calibrate`.

    Dict form: ``a`` and ``sg`` are required; ``b`` defaults to ``a``, ``c`` to
    ``a``, ``alpha`` to 90, ``beta`` to ``alpha``, ``gamma`` to 90.

    Returns
    -------
    dict with keys ``name, sg, a, b, c, alpha, beta, gamma`` (``name`` is the
    canonical name for registered calibrants, else ``"<custom>"``).
    """
    if isinstance(calibrant, str):
        name = _canonical_name(calibrant)
        return {"name": name, **_CALIBRANTS[name]}
    if isinstance(calibrant, dict):
        try:
            a = float(calibrant["a"])
            sg = int(calibrant["sg"])
        except KeyError as e:
            raise ValueError(
                f"custom calibrant dict is missing required key {e}; "
                "required: 'a' (Å), 'sg' (space-group number). "
                "optional: 'b', 'c' (default a), 'alpha', 'beta' (default "
                "alpha), 'gamma' (default 90)."
            ) from None
        alpha = float(calibrant.get("alpha", 90.0))
        return {
            "name": "<custom>", "sg": sg, "a": a,
            "b": float(calibrant.get("b", a)),
            "c": float(calibrant.get("c", a)),
            "alpha": alpha,
            "beta": float(calibrant.get("beta", alpha)),
            "gamma": float(calibrant.get("gamma", 90.0)),
        }
    raise TypeError(
        f"calibrant must be a str name or a lattice dict, got {type(calibrant)}"
    )


def _ring_table(cal_spec: dict, wavelength_A: float,
                n_rings: int = 30) -> list:
    """Return unique powder rings sorted by 2θ, via midas_hkls.generate_hkls.

    Returns
    -------
    list of (hkl_tuple, d_A, two_theta_deg, ring_index_1based)

    midas_hkls.generate_hkls covers all 230 space groups through proper
    Hall-symbol symmetry operations, correctly handling systematic absences
    for trigonal/hexagonal calibrants (e.g. Al₂O₃) as well as cubic ones.
    ``ring_nr`` in the Reflection object is the unique-ring deduplication
    key — reflections with the same ring_nr are crystallographically
    equivalent and contribute to the same Debye-Scherrer ring.
    """
    from midas_hkls import SpaceGroup, Lattice, generate_hkls

    cal = cal_spec
    sg  = SpaceGroup.from_number(cal["sg"])
    lat = Lattice(a=cal["a"], b=cal["b"], c=cal["c"],
                  alpha=cal["alpha"], beta=cal["beta"], gamma=cal["gamma"])
    # two_theta_max_deg=60° captures every ring that could land on any
    # real detector at any Lsd > ~50 mm; n_rings then trims to the
    # innermost families needed for seeding.
    refs = generate_hkls(sg, lat, wavelength_A=wavelength_A,
                         two_theta_max_deg=60.0)

    # Deduplicate to one representative per unique ring (lowest h,k,l),
    # sort by 2θ, and truncate to n_rings.
    seen: dict = {}
    for r in refs:
        if r.ring_nr not in seen:
            seen[r.ring_nr] = r
    unique = sorted(seen.values(), key=lambda r: r.two_theta_deg)[:n_rings]
    return [((r.h, r.k, r.l), r.d_spacing, r.two_theta_deg, r.ring_nr)
            for r in unique]


# ══════════════════════════════════════════════════════════════════════════════
# Bad-pixel detection
# ══════════════════════════════════════════════════════════════════════════════

def _detect_bad_pixels(img: np.ndarray,
                       finite_max: float = 1e9) -> np.ndarray:
    """Boolean mask of dead/sentinel pixels.

    Catches UINT32_MAX (Eiger/Pilatus), negative pixels (HDF5/GE
    pipelines), NaN/±Inf, and any pixel above ``finite_max`` counts.
    Must run BEFORE any statistics so a single sentinel does not corrupt
    the noise estimate.
    """
    bad = ~np.isfinite(img)
    bad |= img >= 4_294_967_295.0   # UINT32_MAX
    bad |= img < 0
    bad |= img > finite_max
    return bad


# ══════════════════════════════════════════════════════════════════════════════
# Background subtract
# ══════════════════════════════════════════════════════════════════════════════

def _median_background(img: np.ndarray, *,
                        kernel_size: int = 101,
                        n_iters: int = 3,
                        use_diplib: bool = True) -> np.ndarray:
    """Smooth background via iterative median filter.

    Default path: downsample 4× → small-kernel scipy median → upsample.
    This is ~100× faster than a full-resolution median on a 2880² image
    and avoids the "kernel spans multiple rings" problem that occurs when
    the true kernel=101 reaches across closely-spaced ring families on
    Pilatus2M geometries.

    diplib path: used when available and use_diplib=True, applied on the
    downsampled image for further speedup.  Falls back silently on any
    exception (macOS OpenMP segfaults have been documented in the package).
    """
    from scipy import ndimage
    img = img.astype(np.float64)
    down = 4 if min(img.shape) >= 1024 else 1
    if down > 1:
        k_down = max(3, int(round(kernel_size / down)))
        if k_down % 2 == 0:
            k_down += 1
        small = ndimage.zoom(img, 1.0 / down, order=1)
        if use_diplib and _HAS_DIPLIB:
            try:
                dip_img = _DIPLIB.Image(small)
                for _ in range(n_iters):
                    dip_img = _DIPLIB.MedianFilter(dip_img, [k_down, k_down])
                small = np.asarray(dip_img).astype(np.float64)
            except Exception:
                for _ in range(n_iters):
                    small = ndimage.median_filter(small, size=k_down)
        else:
            for _ in range(n_iters):
                small = ndimage.median_filter(small, size=k_down)
        out = ndimage.zoom(small, (img.shape[0] / small.shape[0],
                                   img.shape[1] / small.shape[1]), order=1)
        if out.shape != img.shape:
            out = out[:img.shape[0], :img.shape[1]]
        return out.astype(np.float64)
    # tiny image — full-resolution fallback
    if use_diplib and _HAS_DIPLIB:
        try:
            dip_img = _DIPLIB.Image(img)
            for _ in range(n_iters):
                dip_img = _DIPLIB.MedianFilter(dip_img, [kernel_size, kernel_size])
            return np.asarray(dip_img).astype(np.float64)
        except Exception:
            pass
    out = img.copy()
    for _ in range(n_iters):
        out = ndimage.median_filter(out, size=kernel_size)
    return out


def _background_subtract(img: np.ndarray, *,
                          kernel_size: int = 101,
                          n_iters: int = 3,
                          use_diplib: bool = True) -> tuple:
    """Median-background subtract + MAD-based σ on the residual.

    Returns ``(residual, bad_mask, sigma)``:
      * ``residual`` = max(img − bg, 0) with bad pixels zeroed.
      * ``bad_mask`` = boolean mask of sentinel/dead pixels.
      * ``sigma``    = 1.4826 × MAD on residual over GOOD pixels only.

    Computing σ over good pixels is what makes this sentinel-safe: a
    single UINT32_MAX pixel on an Eiger frame would otherwise push σ to
    ~36 million counts and make the threshold reject every real ring.
    """
    img = img.astype(np.float64)
    bad = _detect_bad_pixels(img)
    img_for_bg = img.copy()
    if bad.any():
        good_med = float(np.median(img[~bad])) if (~bad).any() else 0.0
        img_for_bg[bad] = good_med
    bg = _median_background(img_for_bg, kernel_size=kernel_size,
                             n_iters=n_iters, use_diplib=use_diplib)
    diff = img - bg
    diff[diff < 0] = 0.0
    diff[bad] = 0.0
    good = ~bad
    if good.sum() > 100:
        gr = diff[good]
        mad = float(np.median(np.abs(gr - np.median(gr))))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(gr))
    else:
        sigma = float(np.std(diff))
    return diff, bad, float(sigma)


# ══════════════════════════════════════════════════════════════════════════════
# Ring-radius → (first_ring, Lsd) composite-score assignment
# ══════════════════════════════════════════════════════════════════════════════

def _visible_arc_length(R_px, BC_y, BC_z, ny, nz, n=360):
    """Fraction of a ring circle that lies inside the detector × circumference."""
    eta = np.linspace(0, 2*math.pi, n, endpoint=False)
    inside = ((BC_y + R_px * np.cos(eta) >= 0) &
              (BC_y + R_px * np.cos(eta) < ny) &
              (BC_z + R_px * np.sin(eta) >= 0) &
              (BC_z + R_px * np.sin(eta) < nz))
    return float(inside.mean() * 2 * math.pi * R_px)


def _assign_rings(measured_R_px, *, wavelength_A, px_um,
                  cal_spec, BC_y, BC_z, detector_ny, detector_nz,
                  max_first_ring=25, max_rel_residual=0.03,
                  missing_inner_penalty=0.5, gap_penalty=0.1, rms_penalty=0.1,
                  missing_inner_min_arc_px=80.0):
    """Score every (k = which ring is the smallest measured one) hypothesis;
    return the highest-scoring RingAssignment.

    Score = n_match − 0.5·n_missing_inner − 0.1·gaps − 0.1·rms_px.
    Linear-LSQ Lsd refit over all matched (R, tan 2θ) pairs.
    """
    R = np.sort(np.asarray(measured_R_px, dtype=np.float64))
    if R.size == 0:
        raise ValueError("no measured radii")
    table = _ring_table(cal_spec, wavelength_A, n_rings=50)

    best = None
    for k in range(min(max_first_ring, len(table))):
        _hkl_k, _d_k, tt_k_deg, ring_k = table[k]
        tt_k = math.radians(tt_k_deg)
        if tt_k <= 0:
            continue
        lsd_k = R[0] * px_um / math.tan(tt_k)
        if not (math.isfinite(lsd_k) and lsd_k > 0):
            continue
        # Predict radii at this candidate Lsd
        predicted = []
        for j in range(k, min(len(table), k + 25)):
            hkl_j, _d_j, tt_j_deg, ring_j = table[j]
            Rj = lsd_k * math.tan(math.radians(tt_j_deg)) / px_um
            if math.isfinite(Rj):
                predicted.append((ring_j, Rj, hkl_j))
        # Greedy nearest-match (each measured radius → closest unmatched predicted)
        used, matched = set(), []
        for mi, rm in enumerate(R):
            best_j, best_err = -1, np.inf
            for jj, (_, Rj, _) in enumerate(predicted):
                if jj in used:
                    continue
                rel = abs(Rj - rm) / max(rm, 1e-9)
                if rel < best_err:
                    best_err, best_j = rel, jj
            if best_j >= 0 and best_err < max_rel_residual:
                ri, Rj, hkl = predicted[best_j]
                matched.append((int(mi), int(ri), float(Rj), tuple(hkl)))
                used.add(best_j)
        if not matched:
            continue
        # Linear-LSQ Lsd refit
        ring_to_tt = {ri: math.radians(tt_deg) for _hkl, _d, tt_deg, ri in table}
        tans = np.array([math.tan(ring_to_tt[m[1]]) for m in matched])
        rs   = np.array([R[m[0]] for m in matched])
        lsd_ref = float(px_um * np.sum(rs * tans) / np.sum(tans ** 2))
        # Gaps in matched ring-index sequence
        idxs = sorted(m[1] for m in matched)
        gaps = (idxs[-1] - idxs[0] + 1) - len(matched)
        # Missing-inner-ring penalty
        n_miss = 0
        if k > 0:
            for j_in in range(k):
                _hkl_in, _d_in, tt_in_deg, _ri = table[j_in]
                Rj_in = lsd_ref * math.tan(math.radians(tt_in_deg)) / px_um
                arc = _visible_arc_length(Rj_in, BC_y, BC_z, detector_ny, detector_nz)
                if arc > missing_inner_min_arc_px:
                    n_miss += 1
        # Per-match RMS at refined Lsd
        resid = [(lsd_ref * math.tan(ring_to_tt[m[1]]) / px_um) - R[m[0]]
                 for m in matched]
        rms = float(np.sqrt(np.mean(np.array(resid) ** 2)))
        score = len(matched) - missing_inner_penalty*n_miss - gap_penalty*gaps - rms_penalty*rms
        if best is None or score > best["score"]:
            best = {"first_ring": int(ring_k), "Lsd_um": float(lsd_ref),
                    "n_match": len(matched), "rms_px": float(rms),
                    "score": float(score)}
    if best is None:
        raise RuntimeError("No ring assignment matched within tolerance")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Taubin circle fit + LM refine  (for arc-based BC recovery)
# ══════════════════════════════════════════════════════════════════════════════

def _taubin(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        raise ValueError("need ≥ 3 points")
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    Suu = (u*u).sum(); Svv = (v*v).sum(); Suv = (u*v).sum()
    Suuu = (u*u*u).sum(); Svvv = (v*v*v).sum()
    Suvv = (u*v*v).sum(); Svuu = (v*u*u).sum()
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        uc, vc = np.linalg.lstsq(A, b, rcond=None)[0]
    R = float(np.sqrt(uc*uc + vc*vc + (Suu + Svv) / max(x.size, 1)))
    return float(uc + xm), float(vc + ym), R


def _refine_circle(x, y, cx0, cy0, R0):
    from scipy.optimize import least_squares
    x, y = np.asarray(x, float), np.asarray(y, float)
    sol = least_squares(lambda p: np.hypot(x-p[0], y-p[1])-p[2],
                        x0=[cx0, cy0, R0], method="lm", max_nfev=200)
    cx, cy, R = (float(v) for v in sol.x)
    return cx, cy, R, float(np.sqrt(np.mean(sol.fun**2)))


def _ransac_bc(arcs, inlier_tol_px=25.0):
    if not arcs:
        return [], float("nan"), float("nan")
    centres = np.array([[a["cx"], a["cy"]] for a in arcs])
    best_n, best_idx = 0, None
    for i in range(len(arcs)):
        idx = np.where(np.linalg.norm(centres - centres[i], axis=1) < inlier_tol_px)[0]
        if idx.size > best_n:
            best_n, best_idx = idx.size, idx
    if best_idx is None or best_n < 3:
        best_idx = np.arange(len(arcs))
    inliers = [arcs[i] for i in best_idx]
    return inliers, float(np.median([a["cx"] for a in inliers])), float(np.median([a["cy"] for a in inliers]))


def _joint_bc_refine(arcs):
    from scipy.optimize import least_squares
    xs = [np.asarray(a["x"], float) for a in arcs]
    ys = [np.asarray(a["y"], float) for a in arcs]
    R0 = [float(a["R"]) for a in arcs]
    cx0 = float(np.median([a["cx"] for a in arcs]))
    cy0 = float(np.median([a["cy"] for a in arcs]))
    def fun(p):
        return np.concatenate([np.hypot(xi-p[0], yi-p[1])-r
                               for xi, yi, r in zip(xs, ys, p[2:])])
    sol = least_squares(fun, x0=np.concatenate([[cx0, cy0], R0]),
                        method="lm", max_nfev=400)
    cx, cy = float(sol.x[0]), float(sol.x[1])
    Rs = [float(r) for r in sol.x[2:]]
    pos, rms_per = 0, []
    for xi in xs:
        n = xi.size
        rms_per.append(float(np.sqrt(np.mean(sol.fun[pos:pos+n]**2))))
        pos += n
    return cx, cy, Rs, rms_per


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def make_seed(img: np.ndarray, *,
              wavelength_A: float,
              px_um: float,
              calibrant: Union[str, Dict] = "CeO2",
              snr_threshold: float = 4.0,
              bright_fraction_cap: float = 0.01,
              dilation_radius: int = 0,
              min_arc_pixels: Optional[int] = None,
              min_arc_fraction: float = 0.10,
              max_rms_px: float = 1.5,
              median_kernel: int = 101,
              median_iters: int = 3,
              use_diplib: bool = True,
              ) -> Seed:
    """Generate a robust (BC, Lsd) seed for midas_calibrate_v2.calibrate().

    Parameters
    ----------
    img : np.ndarray
        2-D detector image (any dtype; negatives treated as bad pixels).
        MIDAS convention: shape = (NrPixelsZ, NrPixelsY).
    wavelength_A : float
        X-ray wavelength in Ångström.
    px_um : float
        Detector pixel pitch in µm (square pixel assumed).
    calibrant : str or dict
        Calibrant material. Either a registered name (``"CeO2"``, ``"LaB6"``,
        ``"Si"``, ``"Al2O3"``) or a dict describing an arbitrary powder
        calibrant: required ``a`` (Å) and ``sg`` (space-group number); optional
        ``b``, ``c`` (default ``a``), ``alpha``, ``beta`` (default ``alpha``),
        ``gamma`` (default 90). Any material midas_hkls supports works.
        For a dict, the returned ``Seed.calibrant_name`` is ``"<custom>"`` —
        do not feed that back into ``calibrate()`` as a name; pass the original
        dict instead.
    snr_threshold : float
        Threshold = max(snr_threshold × σ, bright_fraction_cap percentile).
        Default 4 (standard 4σ).
    bright_fraction_cap : float
        Caps the bright-pixel fraction at this value (default 0.01 = 1 %).
        Prevents the threshold from being too permissive on dense-ring
        detectors where the noise σ is small but diffuse scatter covers
        the whole image.
    dilation_radius : int
        Disk dilation radius before skeletonise. 0 (default) works for all
        tested CeO2 calibrants. Increase to 1–2 only if arcs are very
        fragmented and not detected. WARNING: too large merges closely-
        spaced rings on PerkinElmer / short-Lsd Pilatus geometries.
    min_arc_pixels : int or None
        Minimum skeleton pixels per connected component. None → auto-derive
        from min_arc_fraction × min(image.shape).
    min_arc_fraction : float
        Fraction of the smallest detector dimension used as adaptive min-arc
        length (default 0.10). Ensures the threshold scales with detector
        size: 51 px on a 512-px Eiger, 288 px on a 2880-px Varex.
    max_rms_px : float
        Per-arc circle-fit RMS cutoff (default 1.5 px). Arcs that fit
        poorly to a circle are dropped.
    median_kernel : int
        Background-estimate median filter kernel size (default 101 px).
        Applied after 4× downsampling.
    median_iters : int
        Number of iterative median filter passes (default 3).
    use_diplib : bool
        Try diplib's MedianFilter first (faster); fall back to scipy.

    Returns
    -------
    Seed
        Dataclass with BC_y, BC_z, Lsd_um, wavelength_A, px_um,
        calibrant_name, first_ring, n_measured, rms_px, elapsed_s, notes.
        Pass BC and Lsd directly to calibrate() via its initial_BC_y /
        initial_BC_z / initial_Lsd kwargs (available on the
        calibrate_v2_seed branch).

    Algorithm outline
    -----------------
    A  Sentinel-aware median-bg subtract → residual + MAD σ (good px only)
    B  Dual threshold: max(SNR·σ, percentile-cap) → binary bright mask
    C  Optional dilation (default off) → skeletonize
    D  Connected components → adaptive-min-area filter → per-arc Taubin+LM
       circle fit → RMS+range filter → sort by radius
    E  RANSAC on arc centres → joint multi-circle LM (one shared BC,
       per-arc R) → 5×median-RMS outlier purge → re-extract radii at
       locked BC → 4-px duplicate collapse
    F  Composite-score ring-index hypothesis search → linear-LSQ Lsd refit
    """
    from skimage import measure, morphology

    print(" |   MIDAS auto-seeder for BC and Lsd has been launched.", flush=True)
    t0 = time.time()
    cal_spec = _resolve_calibrant(calibrant)
    calibrant_name = cal_spec["name"]
    nz, ny = img.shape

    # ── A  background subtract ──────────────────────────────────────────────
    diff, _bad, sigma = _background_subtract(
        img.astype(np.float32), kernel_size=median_kernel,
        n_iters=median_iters, use_diplib=use_diplib)

    # ── B  threshold ────────────────────────────────────────────────────────
    thr_sigma = snr_threshold * sigma
    if 0 < bright_fraction_cap < 1:
        thr_pct = float(np.percentile(diff, 100.0 * (1.0 - bright_fraction_cap)))
        thr = max(thr_sigma, thr_pct)
    else:
        thr = thr_sigma
    bright = diff > thr

    # ── C  (optional) dilation → skeletonize ────────────────────────────────
    if dilation_radius and dilation_radius > 0:
        bright = morphology.binary_dilation(
            bright, footprint=morphology.disk(int(dilation_radius)))
    thin = morphology.skeletonize(bright)

    # ── D  connected components → per-arc fits ──────────────────────────────
    if min_arc_pixels is None:
        min_arc_pixels = max(30, int(min_arc_fraction * min(img.shape)))
    label = measure.label(thin, connectivity=2)
    arcs = []
    for region in measure.regionprops(label):
        if region.area < min_arc_pixels:
            continue
        rows = region.coords[:, 0].astype(float)
        cols = region.coords[:, 1].astype(float)
        try:
            cx0, cy0, R0 = _taubin(cols, rows)
            cx, cy, R, rms = _refine_circle(cols, rows, cx0, cy0, R0)
        except Exception:
            continue
        if not np.isfinite(R) or R <= 5 or R > 4000 or rms > max_rms_px:
            continue
        arcs.append({"cx": cx, "cy": cy, "R": R, "rms": rms,
                     "n_pts": int(rows.size), "x": cols, "y": rows})
    if not arcs:
        raise RuntimeError(
            "make_seed: no arcs detected. "
            "Try lowering snr_threshold or min_arc_fraction.")
    arcs.sort(key=lambda a: a["R"])

    # ── E  RANSAC → joint LM → purge → collapse ─────────────────────────────
    inliers, _, _ = _ransac_bc(arcs)
    if len(inliers) >= 2:
        cx, cy, Rs, rms_per = _joint_bc_refine(inliers)
        if len(rms_per) >= 4:
            med = float(np.median(rms_per))
            good = [i for i, r in enumerate(rms_per) if r < 5*med + 0.5]
            if 0 < len(good) < len(rms_per):
                inliers = [inliers[i] for i in good]
                cx, cy, Rs, rms_per = _joint_bc_refine(inliers)
        Rs_all = list(Rs)
        in_set = {id(a) for a in inliers}
        for a in arcs:
            if id(a) in in_set:
                continue
            R_from = float(np.mean(np.hypot(a["x"] - cx, a["y"] - cy)))
            if 0 < R_from < 1e5:
                Rs_all.append(R_from)
    else:
        cx = float(np.median([a["cx"] for a in inliers])) if inliers else ny/2
        cy = float(np.median([a["cy"] for a in inliers])) if inliers else nz/2
        Rs_all = [a["R"] for a in arcs]

    R_sorted = np.sort(np.asarray(Rs_all, float))
    R_collapsed = [float(R_sorted[0])]
    for r in R_sorted[1:]:
        if r - R_collapsed[-1] > 4.0:
            R_collapsed.append(float(r))

    # ── F  ring assignment → Lsd ─────────────────────────────────────────────
    a = _assign_rings(
        R_collapsed, wavelength_A=wavelength_A, px_um=px_um,
        cal_spec=cal_spec, BC_y=cx, BC_z=cy,
        detector_ny=ny, detector_nz=nz)

    return Seed(
        BC_y=cx, BC_z=cy, Lsd_um=a["Lsd_um"],
        wavelength_A=wavelength_A, px_um=px_um,
        calibrant_name=calibrant_name,
        first_ring=a["first_ring"], n_measured=len(R_collapsed),
        rms_px=a["rms_px"], elapsed_s=time.time() - t0,
        notes=(f"{len(arcs)} arcs, {len(inliers)} BC inliers, "
               f"{len(R_collapsed)} collapsed radii, score={a['score']:.2f}"))
