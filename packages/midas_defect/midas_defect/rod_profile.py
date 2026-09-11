"""Rods measured on the FRAME STACK — the complement to `rod_detect`.

:mod:`midas_defect.rod_detect` finds rods as lines in a q-space voxel cloud.
That is the right tool once a cloud exists. It cannot answer the questions
below, which need the frames themselves:

- how does intensity run **along** a known rod, against continuous L?
- is the rod real, or is it a powder ring, a detector artefact, or the walk
  dragging a bright pixel?
- how **wide** is the rod transverse to itself, and how much of that width is
  just the instrument?

Walk the rod where it is straight
---------------------------------
A c\\* rod is a straight line in reciprocal space. The map from q to a flat
detector goes through the Ewald sphere and is **nonlinear**, so that straight
rod is a *curve* on the detector. Walking a straight line in detector space
drifts off the rod, worst at large \\|L\\| — which is exactly where the profile is
usually being read. So parameterise it where it is straight,
``G_sample(L) = U · B · (h, k, L)`` with L continuous, solve ω for each L, and
project. Two consequences, both improvements: the sampled path is the true
curved image of the rod, and each L is read from **its own frame** — the one
where that part of the rod is in diffraction condition — rather than from a max
projection that piles 36 frames of background under every point.

Quote sigma, never a ratio
--------------------------
Reading each point from its own background-subtracted frame gives data centred
on **zero**. A "ratio to control" then divides by noise about zero and is
meaningless (in the source analysis it returned −1550×). :func:`rod_significance`
reports ``(rod − median(control)) / sigma(control)`` with a robust sigma
(1.4826 × MAD), which is the correct scale for zero-centred data.

The control has to be able to fail
----------------------------------
The matched control is the identical walk at ``(h + ½, k + ½, L)``: not a
lattice rod, but the same \\|q\\| range, the same curvature, the same ω range and
the same detector regions. Anything the geometry does to the rod it does to the
control.

An earlier control — the azimuthal median at the same radius — returned exactly
zero for every row, because a polar-median-subtracted stack *is* an azimuthal
median per (2θ, sector), so its azimuthal median is zero by construction. **A
control that cannot fail is not a control.** :func:`rod_significance` refuses a
control whose scatter is identically zero for this reason.

Not-observable is not not-there
-------------------------------
Three distinct things make a point drop out, none of which is "no intensity":
no ω solution inside the delivered rocking range; the projection lands off the
detector; the transverse slice is mostly masked. All three are recorded
separately and returned as NaN, never as zero. A silent gap looks exactly like
a real minimum in the rod, which is the quantity being measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import math
import numpy as np

__all__ = [
    "RodPath", "RodProfile", "WidthResult",
    "rod_path", "matched_control_path", "profile_along",
    "rod_significance", "ring_L_marks", "transverse_width",
    "centred_L_nodes", "diffuse_to_bragg",
]

#: Reasons a point on the rod is not observable. Never conflate with zero.
DROP_NO_OMEGA = "no_omega_solution"
DROP_OFF_DETECTOR = "off_detector"
DROP_MASKED = "masked"


@dataclass
class RodPath:
    """The detector image of a reciprocal-space rod, point by point."""
    L: np.ndarray                    # continuous L actually realised
    row: np.ndarray
    col: np.ndarray
    omega_deg: np.ndarray
    q_mag: np.ndarray
    hk: Tuple[float, float]
    dropped: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.L)

    def __str__(self) -> str:
        d = ", ".join(f"{k} {v}" for k, v in sorted(self.dropped.items())) or "none"
        return (f"rod (h,k) = {self.hk}: {len(self)} points, "
                f"L {self.L.min():+.2f}..{self.L.max():+.2f}; dropped: {d}")


def rod_path(U: np.ndarray, B: np.ndarray, h: float, k: float,
             L_values: Sequence[float], *,
             wavelength_A: float, lsd_um: float, pixel_um: float,
             bc_row: float, bc_col: float, n_rows: int, n_cols: int,
             omega_lo_deg: float, omega_hi_deg: float,
             omega_sign: int = 1,
             q_convention: str = "1/d") -> RodPath:
    """Walk ``(h, k, L)`` through q-space, solving ω, and project to pixels.

    ``B`` must be in the same convention as ``q_convention``: ``"1/d"`` for a
    plain ``diag(1/a, 1/b, 1/c)``, or ``"2pi/d"`` for one scaled by 2π (the
    convention the rest of :mod:`midas_defect` uses). The Bragg condition is
    written for whichever is supplied. **The default is ``"1/d"`` although the rest
    of :mod:`midas_defect` is 2π/d.** A 2π-scaled ``B`` left on the default does not
    raise: the Bragg condition then fails for most ``L`` and the points land in
    ``RodPath.dropped`` instead. An unexpectedly short path is the symptom.

    **The detector is FLAT.** Points are projected onto a plane normal to the beam
    at ``lsd_um``, with ``Y = -(col - bc_col)·px`` and ``Z = +(row - bc_row)·px``: no
    tilt, no ``tx``, no radial distortion. On a tilted detector a predicted point is
    off by up to about ``lsd_um·tan(tilt)/pixel_um`` px -- ~14 px at 0.4° and
    Lsd ≈ 350 mm -- so every rod walked this way carries that error.

    **``omega_sign`` changes only the REPORTED ω.** The rotation that places a point
    on the detector is always the physical angle ``w`` solving the Bragg condition;
    ``omega_sign`` enters only ``omega = omega_sign·w``, which is both the value
    returned and the value tested against ``[omega_lo_deg, omega_hi_deg]``. It turns
    a physical rotation into a motor reading. It does **not** mirror reciprocal space:
    if the sample-frame vectors behind ``U`` were built with the opposite rotation
    sense, fix that where they were built. With a window symmetric about zero,
    ``+1`` and ``-1`` give identical pixels and negated ω (pinned by a test).

    Points that cannot be observed are **counted, not fabricated** — see
    ``RodPath.dropped``.
    """
    if q_convention not in ("1/d", "2pi/d"):
        raise ValueError("q_convention must be '1/d' or '2pi/d'")
    # Bragg: G_lab_x = -lambda|G|^2/2 in 1/d units; with q = 2pi/d the same
    # condition is q_lab_x = -|q|^2 lambda / (4 pi).
    bragg_scale = (wavelength_A / 2.0 if q_convention == "1/d"
                   else wavelength_A / (4.0 * math.pi))
    U = np.asarray(U, float)
    B = np.asarray(B, float)
    out_L, out_r, out_c, out_w, out_q = [], [], [], [], []
    dropped = {DROP_NO_OMEGA: 0, DROP_OFF_DETECTOR: 0}

    for Lc in np.asarray(L_values, float):
        g_s = U @ (B @ np.array([h, k, Lc], float))
        gn = float(np.linalg.norm(g_s))
        if gn < 1e-12:
            dropped[DROP_NO_OMEGA] += 1
            continue
        D = -bragg_scale * gn * gn
        A_, Bc = g_s[0], -g_s[1]
        R = math.hypot(A_, Bc)
        if R < 1e-12 or abs(D) > R:
            dropped[DROP_NO_OMEGA] += 1
            continue
        base = math.atan2(Bc, A_)
        dw = math.acos(max(-1.0, min(1.0, D / R)))
        chosen = None
        for w in (base + dw, base - dw):
            ome = (omega_sign * math.degrees(w) + 180.0) % 360.0 - 180.0
            if omega_lo_deg <= ome <= omega_hi_deg:
                chosen = (w, ome)
                break
        if chosen is None:
            dropped[DROP_NO_OMEGA] += 1
            continue
        w, ome = chosen
        cw, sw = math.cos(w), math.sin(w)
        g_lab = np.array([cw * g_s[0] - sw * g_s[1],
                          sw * g_s[0] + cw * g_s[1], g_s[2]])
        k_i = (1.0 / wavelength_A if q_convention == "1/d"
               else 2.0 * math.pi / wavelength_A)
        kf = np.array([k_i, 0.0, 0.0]) + g_lab
        if kf[0] <= 0:
            dropped[DROP_OFF_DETECTOR] += 1
            continue
        t = lsd_um / kf[0]
        # signs match midas_defect.geometry.pixel_to_qlab: Y = -(col-BCy)*px,
        # Z = +(row-BCz)*px
        r_p = bc_row + kf[2] * t / pixel_um
        c_p = bc_col - kf[1] * t / pixel_um
        if not (0 <= r_p < n_rows and 0 <= c_p < n_cols):
            dropped[DROP_OFF_DETECTOR] += 1
            continue
        out_L.append(Lc); out_r.append(r_p); out_c.append(c_p)
        out_w.append(ome); out_q.append(gn)

    return RodPath(L=np.asarray(out_L), row=np.asarray(out_r),
                   col=np.asarray(out_c), omega_deg=np.asarray(out_w),
                   q_mag=np.asarray(out_q), hk=(h, k), dropped=dropped)


def matched_control_path(U, B, h: float, k: float, L_values, **kw) -> RodPath:
    """The identical walk at ``(h + ½, k + ½, L)`` — the matched control.

    Not a reciprocal-lattice rod, but the same |q| range, curvature, ω range and
    detector regions. If intensity is elevated only at integer (h, k), the rod
    is real.
    """
    return rod_path(U, B, h + 0.5, k + 0.5, L_values, **kw)


@dataclass
class RodProfile:
    """Intensity along a rod, with non-observable points kept as NaN."""
    L: np.ndarray
    intensity: np.ndarray            # NaN where not observable
    n_valid_px: np.ndarray
    omega_deg: np.ndarray
    dropped: Dict[str, int]

    @property
    def observed(self) -> np.ndarray:
        return np.isfinite(self.intensity)

    def __str__(self) -> str:
        n = int(self.observed.sum())
        return (f"rod profile: {n} of {len(self.L)} points observable "
                f"({100*n/max(len(self.L),1):.0f} %); dropped "
                + ", ".join(f"{k} {v}" for k, v in sorted(self.dropped.items())))


def profile_along(stack: np.ndarray, mask: np.ndarray, path: RodPath, *,
                  omega_first_deg: float, omega_step_deg: float,
                  half_width_px: float = 6.0,
                  min_valid_fraction: float = 0.5,
                  reducer: str = "mean") -> RodProfile:
    """Transverse-slice intensity along ``path``, each point from its own frame.

    ``reducer`` is ``"mean"`` or ``"max"``. Prefer the mean: a max over a slice
    is positively biased, and a max over frames is worse — the maximum of many
    noisy samples sits well above the median of one.

    A slice more than ``1 - min_valid_fraction`` masked yields **NaN**, and is
    counted under ``masked``. It is never returned as zero.
    """
    stack = np.asarray(stack)
    mask = np.asarray(mask, bool)
    if stack.ndim != 3:
        raise ValueError(f"stack must be (n_frames, rows, cols), got {stack.shape}")
    if reducer not in ("mean", "max"):
        raise ValueError("reducer must be 'mean' or 'max'")
    n_f, n_r, n_c = stack.shape

    # local perpendicular from the path's own tangent
    if len(path) < 3:
        raise ValueError("path too short to define a tangent")
    dr = np.gradient(path.row)
    dc = np.gradient(path.col)
    norm = np.hypot(dr, dc)
    norm[norm == 0] = 1.0
    pr, pc = -dc / norm, dr / norm
    offs = np.arange(-half_width_px, half_width_px + 1e-9, 1.0)

    inten = np.full(len(path), np.nan)
    nvalid = np.zeros(len(path), int)
    dropped = dict(path.dropped)
    dropped.setdefault(DROP_MASKED, 0)

    for i in range(len(path)):
        kf = int(round((path.omega_deg[i] - omega_first_deg) / omega_step_deg))
        if not (0 <= kf < n_f):
            dropped[DROP_NO_OMEGA] = dropped.get(DROP_NO_OMEGA, 0) + 1
            continue
        rr = np.round(path.row[i] + offs * pr[i]).astype(int)
        cc = np.round(path.col[i] + offs * pc[i]).astype(int)
        inside = (rr >= 0) & (rr < n_r) & (cc >= 0) & (cc < n_c)
        if not inside.any():
            dropped[DROP_OFF_DETECTOR] = dropped.get(DROP_OFF_DETECTOR, 0) + 1
            continue
        rr, cc = rr[inside], cc[inside]
        good = ~mask[rr, cc]
        nvalid[i] = int(good.sum())
        if good.sum() < min_valid_fraction * len(offs):
            dropped[DROP_MASKED] += 1
            continue
        vals = stack[kf, rr[good], cc[good]]
        inten[i] = float(vals.mean() if reducer == "mean" else vals.max())

    return RodProfile(L=path.L, intensity=inten, n_valid_px=nvalid,
                      omega_deg=path.omega_deg, dropped=dropped)


def rod_significance(rod: RodProfile, control: RodProfile) -> dict:
    """Rod intensity in SIGMA above a matched control. Never a ratio.

    Refuses a control with zero scatter — that is the signature of a control
    that cannot fail (e.g. taking the azimuthal median of data that has already
    had its azimuthal median subtracted).
    """
    r = rod.intensity[np.isfinite(rod.intensity)]
    c = control.intensity[np.isfinite(control.intensity)]
    if r.size == 0 or c.size == 0:
        raise ValueError("rod or control has no observable points")
    med = float(np.median(c))
    sigma = float(1.4826 * np.median(np.abs(c - med)))
    if sigma <= 0:
        raise ValueError(
            "the control has zero scatter, so it cannot fail. This is what "
            "happens when the control is the azimuthal median of data whose "
            "azimuthal median has already been subtracted. Use a control that "
            "samples the same geometry but not the lattice — "
            "matched_control_path().")
    return {"median_sigma": float((np.median(r) - med) / sigma),
            "mean_sigma": float((r.mean() - med) / sigma),
            "control_median": med, "control_sigma": sigma,
            "n_rod": int(r.size), "n_control": int(c.size)}


def ring_L_marks(path: RodPath, ring_radii_px: Sequence[float], *,
                 bc_row: float, bc_col: float,
                 tolerance_px: float = 3.0) -> np.ndarray:
    """L values at which the rod crosses a powder ring — mark, do not delete.

    A ring puts a bump at one |q|, and therefore at one L, imitating exactly the
    modulation that would encode fault statistics. Get ``ring_radii_px`` from
    the RAW data (``midas_defect.ingest.detect_powder_rings``), not from a
    background-subtracted stack where the rings have already been removed.
    """
    if len(path) == 0 or len(ring_radii_px) == 0:
        return np.empty(0)
    rad = np.hypot(path.row - bc_row, path.col - bc_col)
    hit = np.zeros(len(path), bool)
    for r0 in ring_radii_px:
        hit |= np.abs(rad - r0) <= tolerance_px
    return path.L[hit]


@dataclass
class WidthResult:
    """Transverse rod width against the Bragg width on the SAME rod."""
    diffuse_fwhm_inv_A: float
    bragg_fwhm_inv_A: float
    excess_fwhm_inv_A: Optional[float]      # None when resolution-limited
    coherence_length_A: Optional[float]     # None when resolution-limited
    lower_bound_A: Optional[float]          # set when resolution-limited
    resolution_limited: bool

    def __str__(self) -> str:
        if self.resolution_limited:
            return (f"RESOLUTION-LIMITED: diffuse FWHM "
                    f"{self.diffuse_fwhm_inv_A:.5f} <= Bragg "
                    f"{self.bragg_fwhm_inv_A:.5f} 1/A. Lateral coherence "
                    f"> {self.lower_bound_A:.0f} A (a LOWER BOUND, not a value)")
        return (f"diffuse {self.diffuse_fwhm_inv_A:.5f} vs Bragg "
                f"{self.bragg_fwhm_inv_A:.5f} 1/A -> excess "
                f"{self.excess_fwhm_inv_A:.5f} -> lateral coherence "
                f"~{self.coherence_length_A:.0f} A")


def transverse_width(diffuse_fwhm_inv_A: float, bragg_fwhm_inv_A: float
                     ) -> WidthResult:
    """Deconvolve the instrument from a rod's transverse width.

    The Bragg peaks **on the same rod** carry exactly the resolution function —
    same beam, optics, mosaic and detector PSF — so::

        excess² = width(diffuse)² − width(Bragg)²

    If the two are equal the rod is **resolution-limited** and the honest output
    is a lower bound on the lateral coherence length, not a number. That is a
    real possible result, not a failure, and this function returns it as one
    rather than reporting a spuriously large length.

    With |G| = 1/d in 1/Å, a lateral domain of size D gives FWHM ~ 1/D, so
    D ~ 1/ΔG. No Scherrer constant is applied — the number is an
    order-of-magnitude coherence length.
    """
    if diffuse_fwhm_inv_A <= 0 or bragg_fwhm_inv_A <= 0:
        raise ValueError("widths must be positive")
    if diffuse_fwhm_inv_A <= bragg_fwhm_inv_A:
        return WidthResult(diffuse_fwhm_inv_A, bragg_fwhm_inv_A, None, None,
                           lower_bound_A=1.0 / bragg_fwhm_inv_A,
                           resolution_limited=True)
    excess = math.sqrt(diffuse_fwhm_inv_A ** 2 - bragg_fwhm_inv_A ** 2)
    return WidthResult(diffuse_fwhm_inv_A, bragg_fwhm_inv_A, excess,
                       1.0 / excess, None, False)


# ── The (h,k) dependence of a rod: what encodes an in-plane fault vector ────
#
# Ported 2026-09-10 from the La3Ni2O7 project's step23_hk_map.py. For planar
# disorder with an in-plane displacement R between faulted blocks the diffuse
# intensity on the (h,k) rod carries a factor 1 - cos(2 pi (h,k).R): it vanishes
# where (h,k).R is an integer and peaks where it is a half-integer, so WHICH rods
# carry diffuse intensity measures R. A Ruddlesden-Popper offset R = (1/2, 1/2)
# predicts rods where h + k is ODD and none where it is EVEN.

def centred_L_nodes(h: int, k: int, L_min: float, L_max: float, centring: str = "P") -> np.ndarray:
    """Integer L at which the (h, k) rod has an ALLOWED Bragg node, by lattice centring.

    ``"P"`` every L; ``"I"`` h + k + L even; ``"F"`` h, k, L all the same parity
    (none at all if h and k differ in parity); ``"C"`` h + k even (then every L).
    The between-node exclusion of :func:`diffuse_to_bragg` must come from THIS,
    not from a fixed integer grid: under I-centring even and odd (h + k) rows have
    their nodes at L of opposite parity.
    """
    Ls = np.arange(int(math.ceil(L_min)), int(math.floor(L_max)) + 1)
    c = centring.upper()
    if c == "P":
        keep = np.ones(len(Ls), bool)
    elif c == "I":
        keep = (h + k + Ls) % 2 == 0
    elif c == "F":
        keep = ((h % 2) == (k % 2)) & ((Ls % 2) == (h % 2))
    elif c == "C":
        keep = np.full(len(Ls), (h + k) % 2 == 0)
    else:
        raise ValueError(f"centring must be one of P, I, F, C; got {centring!r}")
    return Ls[keep].astype(float)


def diffuse_to_bragg(rod: RodProfile, control: RodProfile, bragg_L, *, near_bragg: float = 0.35,
                     bragg_core: float = 0.12, exclude=None, min_points: int = 60,
                     min_bragg_points: int = 10, min_peak_sigma: float = 10.0) -> dict:
    """Diffuse level between the Bragg nodes, over the node height, on ONE rod.

    **The normalisation is the whole point.** Raw diffuse intensity also scales
    with the parent structure factor ``|F(h,k)|^2``, so a row with weak Bragg
    peaks has a weak rod for a trivial reason, and ranking rods by raw level
    recovers "rods where the reflections are bright" -- not a selection rule.
    Compare this ratio across (h, k): flat means no rule, a switch with the
    parity of (h, k) . R is a fault vector.

    ``rod`` and ``control`` come from :func:`profile_along` on :func:`rod_path`
    and :func:`matched_control_path`. ``bragg_L`` are the allowed nodes
    (:func:`centred_L_nodes`). ``exclude`` is an optional boolean mask aligned
    with ``rod.L`` for ring crossings. Points within ``near_bragg`` of a node are
    kept out of the diffuse level; points within ``bragg_core`` of one give the
    peak (95th percentile). The control's median and robust sigma give the
    significance of the diffuse level.

    Returns a dict with ``usable`` and ``reason``, and when usable ``diffuse``,
    ``peak``, ``ratio``, ``sigma``, ``n_diffuse``, ``n_bragg``.

    **A row with no normaliser is not evidence.** If no allowed node reaches the
    Ewald sphere inside the measured omega range the ratio is noise over noise;
    the row comes back ``usable=False`` and must be reported as unusable, never
    as a zero. A node counts only if its peak stands ``min_peak_sigma`` control
    sigmas above THIS rod's own between-node level -- not above the control: a
    real diffuse floor already clears the control, which is the quantity being
    measured, so comparing the node to the control would pass a row with no node
    at all. (The original required 1e3 counts; this form transfers between
    exposures.)
    """
    L = np.asarray(rod.L, float)
    I = np.asarray(rod.intensity, float)
    nodes = np.asarray(bragg_L, float).ravel()
    out = dict(usable=False, reason="")
    if nodes.size == 0:
        out["reason"] = "no allowed Bragg node on this rod"
        return out
    ctl = np.asarray(control.intensity, float)
    ctl = ctl[np.isfinite(ctl)]
    if ctl.size < min_points:
        out["reason"] = f"control has {ctl.size} observable points (< {min_points})"
        return out
    cb = float(np.median(ctl))
    cs = 1.4826 * float(np.median(np.abs(ctl - cb)))
    if not np.isfinite(cs) or cs <= 0:
        out["reason"] = "control noise is zero or undefined"
        return out
    dist = np.min(np.abs(L[:, None] - nodes[None, :]), axis=1)
    ok = np.isfinite(I)
    if exclude is not None:
        ok &= ~np.asarray(exclude, bool)
    dif = ok & (dist >= near_bragg)
    brg = ok & (dist < bragg_core)
    if int(dif.sum()) < min_points:
        out["reason"] = f"{int(dif.sum())} observable between-node points (< {min_points})"
        return out
    if int(brg.sum()) < min_bragg_points:
        out["reason"] = f"{int(brg.sum())} observable points at Bragg nodes (< {min_bragg_points})"
        return out
    peak = float(np.percentile(I[brg], 95))
    d = float(np.median(I[dif]))
    if (peak - d) / cs < min_peak_sigma:
        out["reason"] = ("no Bragg node reaches the Ewald sphere in this omega range -- no "
                         "normaliser; not evidence of anything")
        return out
    out.update(usable=True, diffuse=d, peak=peak, ratio=d / peak, sigma=(d - cb) / cs,
               n_diffuse=int(dif.sum()), n_bragg=int(brg.sum()))
    return out
