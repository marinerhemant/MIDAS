"""Measuring the detector roll — the angle between detector rows and the lab.

Why this exists
---------------
``midas_tomo.hdf5`` **consumes** a detector roll: ``RotationAngle`` is read at
``hdf5.py:154`` and applied to the projections, the dark and the whites by
``_rotate_stack``. Nothing in ``packages/`` ever measured it, so in practice it
is 0 in every scan and the correction is dead code.

An uncorrected roll is not cosmetic. The reconstruction assumes each detector
row is one slice through the specimen at a fixed height; if the detector is
rolled by ``theta``, a "row" is a line that climbs by ``W tan(theta)`` across
the frame, so each reconstructed slice mixes heights. At 0.5 degrees over a
2320-pixel frame that is 20 pixels of vertical smearing.

Three routes against two references
-----------------------------------
:func:`tilt_from_beam_box`
    The beam-defining aperture is a rectangle, so its image on the detector is
    a rectangle rotated by the roll. Fit all four edges. Uses **flat fields**
    and no sample. Reference: **the slits**.

:func:`tilt_from_slice_shifts`
    The best rotation-axis shift, found independently for a low slice and a
    high one. A rolled detector puts the axis off vertical, so
    ``tan(roll) = d(best shift) / d(row)``. Reference: **the rotation axis**.

:func:`tilt_from_rotation_axis`
    Over a full 360 degrees the centre of mass of each row's sinogram averages
    to that row's rotation-axis position; its drift with row is the roll. Same
    reference as the previous one, different estimator.

The split that matters is the **reference**, not the count. The beam box
measures the detector against the slits; the other two against the rotation
axis. They agree only if the slits are square to the rotation axis, which is
the usual assumption and exactly what a disagreement refutes — and
:func:`compare_tilt_estimates` says which number to use when they part.

Of the two axis-referenced estimators, prefer
:func:`tilt_from_slice_shifts`. The centre of mass needs a clean attenuation
baseline and is dragged around by background and truncation; the best-shift
criterion is a sharpness optimum, which is both what the reconstruction
actually depends on and far more robust on a weakly absorbing specimen. Its
limit is quantisation — the sweep step over the row span sets the finest
angle visible, and below that it reports an upper bound rather than a value.

What can fail, and does
-----------------------
:func:`tilt_from_beam_box` carries two nulls that can come back the other way:

* **parallelism** — the left and right edges must agree, and the top and
  bottom. A disagreement means one "edge" is not an aperture edge (a shadow, a
  dead region, the specimen holder).
* **orthogonality** — the vertical and horizontal edge families must imply the
  *same* rotation. If they do not, the illuminated region is not a rectangle
  and the premise of the whole method is wrong.

Neither is decoration: without them the routine happily returns a confident
angle for a trapezoid.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EdgeFit",
    "TiltResult",
    "compare_tilt_estimates",
    "locate_edge_subpixel",
    "tilt_from_beam_box",
    "tilt_from_rotation_axis",
    "tilt_from_slice_shifts",
]

log = logging.getLogger(__name__)


@dataclass
class EdgeFit:
    """One aperture edge, fitted as a straight line."""

    name: str                # left | right | top | bottom
    angle_deg: float         # box rotation implied by this edge alone
    rms_px: float            # residual of the straight-line fit
    n_points: int
    span_px: float           # lever arm; precision scales with this

    def __str__(self) -> str:
        return (f"{self.name:<6} {self.angle_deg:+8.4f} deg   "
                f"rms {self.rms_px:5.2f} px   n={self.n_points:4d}  "
                f"span {self.span_px:.0f} px")


@dataclass
class TiltResult:
    """A detector-roll estimate and whether it can be believed.

    ``angle_deg`` is the **correcting** angle: pass it to
    ``scipy.ndimage.rotate`` (or as ``RotationAngle``) and the roll is removed.
    """

    angle_deg: float
    uncertainty_deg: float
    method: str
    trustworthy: bool
    reason: str
    edges: List[EdgeFit] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.trustworthy

    def summary(self) -> str:
        lines = [f"detector roll [{self.method}]: {self.angle_deg:+.4f} "
                 f"+/- {self.uncertainty_deg:.4f} deg"]
        lines += ["  " + str(e) for e in self.edges]
        lines.append(f"  trustworthy={self.trustworthy}: {self.reason}")
        return "\n".join(lines)


# --------------------------------------------------------------- edge finding

def locate_edge_subpixel(
    profile: np.ndarray, *, rising: bool, window: int = 12, smooth: int = 3
) -> Optional[float]:
    """Sub-pixel position of a single step edge in a 1-D profile.

    Found as the centroid of ``|d profile / dx|`` inside a window around the
    strongest gradient, which is unbiased for a symmetric edge and needs no
    model of the penumbra. Returns None when there is no usable edge.

    ``smooth`` boxcar-filters the profile first. This is not cosmetic: without
    it the *coarse* step — ``argmax`` of the raw gradient — latches onto noise
    and returns the same pixel for many consecutive lines, quantising the edge
    positions and **shrinking the fitted slope toward zero**. Measured on a
    0.25 deg synthetic roll with 1.2 % flat-field noise: 0.208 deg without
    smoothing, 0.25 with. The bias is toward reporting no tilt, which is the
    direction that would quietly leave a real roll uncorrected.
    """
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 2 * window + 3:
        return None
    if smooth and smooth > 1:
        n = int(smooth) | 1                      # odd, so the filter is centred
        k = np.ones(n) / float(n)
        # Replicate the ends rather than zero-pad: a 'same' convolution pads
        # with zeros, which manufactures a step at each boundary and makes a
        # perfectly flat profile look like it has an edge in it.
        p = np.convolve(np.pad(p, n // 2, mode="edge"), k, mode="valid")
    g = np.diff(p)
    signed = g if rising else -g
    i = int(np.argmax(signed))
    if signed[i] <= 0:
        return None
    lo, hi = max(0, i - window), min(g.size, i + window + 1)
    w = np.clip(signed[lo:hi], 0.0, None)
    if w.sum() <= 0:
        return None
    x = np.arange(lo, hi, dtype=np.float64)
    # +0.5 because diff() lives between samples.
    return float((x * w).sum() / w.sum()) + 0.5


def _fit_edge(
    img: np.ndarray, name: str, *, axis: str, rising: bool,
    scan_range: Tuple[int, int], search: Tuple[int, int],
    window: int, smooth: int, max_rms_px: float,
) -> Optional[EdgeFit]:
    """Fit one edge: locate it on every line, then fit a straight line."""
    a0, a1 = scan_range
    s0, s1 = search
    coords, positions = [], []
    for k in range(a0, a1):
        line = img[k, s0:s1] if axis == "row" else img[s0:s1, k]
        pos = locate_edge_subpixel(line, rising=rising, window=window,
                                   smooth=smooth)
        if pos is not None:
            coords.append(float(k))
            positions.append(pos + s0)
    if len(coords) < 20:
        return None

    c = np.array(coords)
    p = np.array(positions)
    # One robust pass: drop points more than 3 sigma off a first fit, which
    # removes the occasional line whose edge was eaten by a dead column.
    for _ in range(2):
        A = np.vstack([c, np.ones_like(c)]).T
        sol, *_ = np.linalg.lstsq(A, p, rcond=None)
        resid = p - A @ sol
        s = resid.std()
        if s == 0:
            break
        keep = np.abs(resid) <= 3.0 * s
        if keep.all() or keep.sum() < 20:
            break
        c, p = c[keep], p[keep]

    A = np.vstack([c, np.ones_like(c)]).T
    sol, *_ = np.linalg.lstsq(A, p, rcond=None)
    slope = float(sol[0])
    rms = float(np.sqrt(np.mean((p - A @ sol) ** 2)))
    if rms > max_rms_px:
        log.warning("%s edge fit rms %.2f px exceeds %.2f; discarding",
                    name, rms, max_rms_px)
        return None

    # Sign convention: the returned angle is the CORRECTING one -- the value
    # to pass to scipy.ndimage.rotate (and so to RotationAngle, which
    # hdf5.py:181 feeds straight into _rotate_stack) to remove the roll. That
    # is the operationally useful definition and it is what
    # test_applying_the_measured_angle_squares_the_box verifies, rather than
    # asserting a hand-picked sign.
    angle = math.degrees(math.atan(slope))
    if axis == "row":            # horizontal edge: position is a row, vs column
        angle = -angle
    return EdgeFit(name=name, angle_deg=angle, rms_px=rms,
                   n_points=int(c.size), span_px=float(c.max() - c.min()))


def tilt_from_beam_box(
    flat: np.ndarray,
    dark: Optional[np.ndarray] = None,
    *,
    margin_frac: float = 0.15,
    window: int = 12,
    smooth: int = 3,
    max_rms_px: float = 3.0,
    tol_deg: float = 0.05,
    min_span_px: int = 50,
) -> TiltResult:
    """Detector roll from the rectangular beam footprint in a flat field.

    ``margin_frac`` trims the ends of each edge before fitting, keeping the
    corners — where two edges meet and the penumbra is two-dimensional — out of
    the straight-line fits.

    ``tol_deg`` bounds both nulls: left-vs-right and top-vs-bottom parallelism,
    and vertical-vs-horizontal orthogonality.
    """
    f = np.asarray(flat, dtype=np.float64)
    if dark is not None:
        f = f - np.asarray(dark, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"flat must be 2-D; got shape {f.shape}")

    ny, nx = f.shape
    col = f.mean(axis=0)
    row = f.mean(axis=1)

    def lit(profile):
        thr = 0.5 * (float(profile.max()) + float(profile.min()))
        idx = np.nonzero(profile > thr)[0]
        return (int(idx.min()), int(idx.max())) if idx.size else None

    lit_x, lit_y = lit(col), lit(row)
    if lit_x is None or lit_y is None:
        raise ValueError(
            "no illuminated region found in the flat field; the beam may "
            "overfill the detector, in which case there are no aperture edges "
            "to fit and this method has no power."
        )
    x0, x1 = lit_x
    y0, y1 = lit_y
    detail: Dict[str, Any] = {"lit_columns": [x0, x1], "lit_rows": [y0, y1],
                              "shape": [ny, nx]}

    if x0 <= 2 or x1 >= nx - 3 or y0 <= 2 or y1 >= ny - 3:
        raise ValueError(
            f"the illuminated region (columns {x0}-{x1}, rows {y0}-{y1} of a "
            f"{ny}x{nx} frame) reaches the edge of the detector, so at least "
            "one aperture edge is outside the field of view. Without a full "
            "box this method cannot check orthogonality."
        )

    my = int(margin_frac * (y1 - y0))
    mx = int(margin_frac * (x1 - x0))
    rows = (y0 + my, y1 - my)
    cols = (x0 + mx, x1 - mx)
    if rows[1] - rows[0] < min_span_px or cols[1] - cols[0] < min_span_px:
        raise ValueError(
            f"the illuminated box is too small to fit edges over: usable rows "
            f"{rows}, columns {cols}, minimum span {min_span_px} px"
        )

    pad = max(30, window * 2)
    edges: List[EdgeFit] = []
    for name, axis, rising, scan, search in (
        ("left",   "row",    True,  rows, (max(0, x0 - pad), x0 + pad)),
        ("right",  "row",    False, rows, (x1 - pad, min(nx, x1 + pad))),
        ("top",    "column", True,  cols, (max(0, y0 - pad), y0 + pad)),
        ("bottom", "column", False, cols, (y1 - pad, min(ny, y1 + pad))),
    ):
        e = _fit_edge(f, name, axis=axis, rising=rising, scan_range=scan,
                      search=search, window=window, smooth=smooth,
                      max_rms_px=max_rms_px)
        if e is not None:
            edges.append(e)

    by = {e.name: e for e in edges}
    reasons: List[str] = []

    vert = [by[n] for n in ("left", "right") if n in by]
    horiz = [by[n] for n in ("top", "bottom") if n in by]
    if not vert and not horiz:
        raise ValueError("no aperture edge could be fitted in either direction")

    if len(vert) == 2 and abs(vert[0].angle_deg - vert[1].angle_deg) > tol_deg:
        reasons.append(
            f"left ({vert[0].angle_deg:+.4f}) and right ({vert[1].angle_deg:+.4f}) "
            f"edges are not parallel to within {tol_deg} deg - one of them is "
            "not an aperture edge"
        )
    if len(horiz) == 2 and abs(horiz[0].angle_deg - horiz[1].angle_deg) > tol_deg:
        reasons.append(
            f"top ({horiz[0].angle_deg:+.4f}) and bottom ({horiz[1].angle_deg:+.4f}) "
            f"edges are not parallel to within {tol_deg} deg"
        )
    if vert and horiz:
        av = float(np.mean([e.angle_deg for e in vert]))
        ah = float(np.mean([e.angle_deg for e in horiz]))
        detail["vertical_family_deg"] = av
        detail["horizontal_family_deg"] = ah
        detail["orthogonality_error_deg"] = av - ah
        if abs(av - ah) > tol_deg:
            reasons.append(
                f"the vertical edges imply {av:+.4f} deg and the horizontal "
                f"edges {ah:+.4f} deg, a {av - ah:+.4f} deg discrepancy. The "
                "illuminated region is not a rectangle, so 'the beam box is "
                "square to the lab' is false and this method does not apply."
            )
    else:
        reasons.append(
            f"only the {'vertical' if vert else 'horizontal'} edge family was "
            "fitted, so orthogonality could not be checked"
        )

    angles = np.array([e.angle_deg for e in edges])
    angle = float(angles.mean())
    unc = float(angles.std(ddof=1) / math.sqrt(angles.size)) if angles.size > 1 else float("nan")

    return TiltResult(
        angle_deg=angle, uncertainty_deg=unc, method="beam-box edges",
        trustworthy=not reasons,
        reason="; ".join(reasons) if reasons else
               "all fitted edges parallel and orthogonal within tolerance",
        edges=edges, detail=detail,
    )


# ------------------------------------------------- the independent cross-check

def tilt_from_rotation_axis(
    data: np.ndarray,
    dark: np.ndarray,
    whites: np.ndarray,
    angles_deg: Sequence[float],
    *,
    min_contrast: float = 0.01,
    min_air_fraction: float = 0.03,
    max_rms_px: float = 2.0,
) -> TiltResult:
    """Detector roll from the rotation axis's projected position per row.

    Over a full turn the centre of mass of a row's sinogram is
    ``c_r + A cos(theta + phi)``, so **averaging over 360 degrees leaves
    ``c_r``**, that row's rotation-axis position, with the sample's own
    asymmetry cancelling rather than being modelled. The axis is a straight
    line in the lab, so ``c_r`` drifting linearly with ``r`` is the roll.

    Independent of :func:`tilt_from_beam_box` in the way that matters: it never
    looks at the aperture, so it measures the detector against the **rotation
    axis** rather than against the slits.

    Requires a full 360-degree scan; a 180-degree scan leaves the cosine term
    uncancelled and this raises rather than returning a biased number.

    ``min_air_fraction`` drops **truncated** rows. A centre of mass only means
    the axis position if the whole specimen is inside the field of view; where
    it is not, the clipped part is missing from one side and the centroid moves
    toward the other. Measured on Ce ``ht525_s2``: rows 600-1500 gave a
    rotation axis stable to **1.7 px over 900 rows**, while rows 300-500 gave
    1158, 1193 and 1154 with three to six times the angular scatter -- and
    those rows had an air fraction of 0.005-0.024 against 0.04-0.06 in the
    bulk. Including them moved the fitted angle by more than a degree.
    """
    ang = np.asarray(angles_deg, dtype=np.float64)
    span = float(np.abs(ang[-1] - ang[0]))
    if span < 350.0:
        raise ValueError(
            f"the scan spans {span:.1f} deg. This method averages the "
            "centre-of-mass oscillation away over a full turn; over "
            f"{span:.1f} deg the oscillation does not cancel and the answer "
            "would be biased by the specimen's own asymmetry."
        )

    d = np.asarray(dark, dtype=np.float64)
    w = np.asarray(whites, dtype=np.float64)
    white = w.mean(axis=0) if w.ndim == 3 else w
    denom = white - d
    if not np.any(denom > 0):
        raise ValueError("the white field is nowhere brighter than the dark")
    denom = np.where(denom <= 0, np.nan, denom)

    n_frames, ny, nx = data.shape
    del nx

    # Restrict to the illuminated COLUMNS as well as the rows. Outside the
    # beam `white - dark` is ~0, the transmission ratio is noise and the clip
    # floor turns it into -log(1e-6) = 13.8 -- an enormous fake absorber at
    # both ends of every row, which dominates the centre of mass and moves
    # around from frame to frame. Fixing only the rows left the residual
    # scatter at 48 px on the Ce scan and the answer wandering between -2.6
    # and -4.6 deg between runs.
    col_illum = np.nan_to_num(np.nanmean(np.where(denom > 0, white - d, np.nan),
                                         axis=0), nan=0.0)
    lit_cols = np.nonzero(col_illum >= 0.2 * float(col_illum.max()))[0]
    if lit_cols.size < 16:
        raise ValueError(
            f"only {lit_cols.size} illuminated columns; the beam does not "
            "cover enough of the detector to locate a rotation axis"
        )
    c0, c1 = int(lit_cols.min()), int(lit_cols.max()) + 1
    data = data[:, :, c0:c1]
    d = d[:, c0:c1]
    denom = denom[:, c0:c1]
    white = white[:, c0:c1]
    nx = c1 - c0
    x = np.arange(c0, c1, dtype=np.float64)
    com = np.full((ny, n_frames), np.nan)
    prof_sum = np.zeros((ny, nx), dtype=np.float64)
    for i in range(n_frames):
        with np.errstate(invalid="ignore", divide="ignore"):
            a = -np.log(np.clip((data[i] - d) / denom, 1e-6, None))
        a = np.where(np.isfinite(a), a, 0.0)
        # Only positive attenuation carries specimen; noise about zero in the
        # air region would drag the centroid toward the frame centre.
        a = np.clip(a, 0.0, None)
        prof_sum += a
        tot = a.sum(axis=1)
        good = tot > 0
        com[good, i] = (a[good] * x).sum(axis=1) / tot[good]

    # Air fraction per row, from the angle-averaged profile: how much of the
    # illuminated width is essentially empty. A row with no air is truncated.
    row_max = prof_sum.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        air_frac = np.where(
            row_max[:, 0] > 0,
            (prof_sum < 0.05 * np.where(row_max > 0, row_max, 1.0)).mean(axis=1),
            0.0,
        )
    not_truncated = air_frac >= min_air_fraction

    row_signal = np.nanmax(com, axis=1) - np.nanmin(com, axis=1)
    centres = np.nanmean(com, axis=1)
    n_seen = np.sum(np.isfinite(com), axis=1)

    # Restrict to rows that actually see the specimen. Without this the fit
    # includes every unilluminated row, whose centre of mass is noise about
    # the frame centre: on the Ce scan that pulled all 2031 rows into the fit
    # and returned -2.52 deg with 40 px of scatter.
    from .center import slices_with_signal
    try:
        lit = set(slices_with_signal(
            data, d, np.stack([white, white]), k=max(64, ny // 8)))
    except ValueError:
        lit = set(range(ny))
    in_lit = np.array([r in lit for r in range(ny)])
    usable = np.nonzero((n_seen > 0.9 * n_frames) & np.isfinite(centres)
                        & (row_signal > 0) & in_lit & not_truncated)[0]
    if usable.size < 20:
        raise ValueError(
            f"only {usable.size} detector rows carry a usable sinogram "
            "centre of mass; the specimen may be too weakly absorbing for "
            "this method"
        )

    r = usable.astype(np.float64)
    c = centres[usable]
    for _ in range(2):
        A = np.vstack([r, np.ones_like(r)]).T
        sol, *_ = np.linalg.lstsq(A, c, rcond=None)
        resid = c - A @ sol
        s = resid.std()
        if s == 0:
            break
        keep = np.abs(resid) <= 3.0 * s
        if keep.all() or keep.sum() < 20:
            break
        r, c = r[keep], c[keep]

    A = np.vstack([r, np.ones_like(r)]).T
    sol, *_ = np.linalg.lstsq(A, c, rcond=None)
    slope = float(sol[0])
    rms = float(np.sqrt(np.mean((c - A @ sol) ** 2)))
    angle = math.degrees(math.atan(slope))

    reasons: List[str] = []
    if rms > max_rms_px:
        reasons.append(
            f"the per-row axis positions scatter {rms:.2f} px about a straight "
            f"line (limit {max_rms_px}), so they are not tracing one axis"
        )
    if r.size < 50:
        reasons.append(f"only {int(r.size)} rows survived; the lever arm is short")

    span_rows = float(r.max() - r.min()) if r.size else 0.0
    unc = (math.degrees(math.atan(rms / span_rows)) if span_rows > 0
           else float("nan"))

    return TiltResult(
        angle_deg=angle, uncertainty_deg=unc, method="rotation-axis drift",
        trustworthy=not reasons,
        reason="; ".join(reasons) if reasons else
               "axis positions lie on a straight line",
        detail={"n_rows": int(r.size), "row_span_px": span_rows,
                "fit_rms_px": rms, "axis_at_row0_px": float(sol[1]),
                "scan_span_deg": span, "lit_columns": [c0, c1],
                "n_rows_truncated": int((in_lit & ~not_truncated).sum()),
                "min_air_fraction": float(min_air_fraction)},
    )


def compare_tilt_estimates(
    box: TiltResult, axis: TiltResult, *, tol_deg: float = 0.05
) -> Dict[str, Any]:
    """Do the two independent routes agree, and if not, which broke?

    They measure the detector against different references — the slits and the
    rotation axis — so a disagreement is informative rather than merely a
    failure: it says the slits are not square to the rotation axis, and the
    rotation-axis number is the one the reconstruction actually needs.
    """
    delta = box.angle_deg - axis.angle_deg
    agree = abs(delta) <= tol_deg

    # Validity FIRST. The interesting reading of a disagreement -- "the slits
    # are not square to the rotation axis" -- is only available when both
    # numbers are real measurements. An earlier version skipped this and
    # recommended the rotation-axis value on any disagreement; on the Ce data
    # that meant confidently recommending -2.5172 deg from a fit whose
    # residual scatter was 40 px against a 2 px limit. Exactly the failure
    # this module exists to prevent, committed by the function meant to
    # adjudicate it.
    if not box.trustworthy and not axis.trustworthy:
        return {
            "verdict": "NO MEASUREMENT", "difference_deg": delta,
            "recommended_deg": float("nan"),
            "note": ("neither route produced a trustworthy angle "
                     f"(box: {box.reason}; axis: {axis.reason}). There is "
                     "nothing to compare and no value to use."),
            "box": box, "axis": axis, "tol_deg": tol_deg,
        }
    if not (box.trustworthy and axis.trustworthy):
        good, bad = (box, axis) if box.trustworthy else (axis, box)
        return {
            "verdict": "UNCERTAIN", "difference_deg": delta,
            "recommended_deg": good.angle_deg,
            "note": (f"only the {good.method} route is trustworthy "
                     f"({good.angle_deg:+.4f} deg); the {bad.method} route "
                     f"reported {bad.angle_deg:+.4f} deg but flagged itself "
                     f"invalid ({bad.reason}). The gap therefore says nothing "
                     "about whether the slits are square to the rotation "
                     "axis - that comparison needs two valid measurements."),
            "box": box, "axis": axis, "tol_deg": tol_deg,
        }

    if agree:
        verdict = "AGREE"
        note = ("both references give the same angle, so the detector roll is "
                "established and the slits are square to the rotation axis")
        use = 0.5 * (box.angle_deg + axis.angle_deg)
    else:
        verdict = "DISAGREE"
        note = (f"the beam box says {box.angle_deg:+.4f} deg and the rotation "
                f"axis says {axis.angle_deg:+.4f} deg, a {delta:+.4f} deg gap. "
                "Both are valid measurements against different references, so "
                "the slits are not square to the rotation axis. Use the "
                "rotation-axis value - that is the one the reconstruction "
                "geometry depends on.")
        use = axis.angle_deg

    return {"verdict": verdict, "difference_deg": delta,
            "recommended_deg": use, "note": note,
            "box": box, "axis": axis, "tol_deg": tol_deg}


def tilt_from_slice_shifts(
    cube: np.ndarray,
    shift_values: Tuple[float, float, float],
    *,
    slices: Optional[Sequence[int]] = None,
    method: str = "variance",
    crop: float = 0.5,
    min_span_rows: int = 20,
) -> TiltResult:
    """Detector roll from the best rotation-axis shift at different heights.

    If the detector is rolled, the rotation axis is not vertical in detector
    coordinates, so the *best shift* for the lowest slice differs from the best
    shift for the highest, and

        tan(roll) = d(best shift) / d(row)

    Better than :func:`tilt_from_rotation_axis` where it counts. The centre of
    mass needs a clean attenuation baseline and gets dragged around by
    background and truncation; the best-shift criterion is a sharpness
    optimum, which is both what the reconstruction actually cares about and
    far less fussy on a weakly absorbing specimen.

    **Not independent of** :func:`tilt_from_rotation_axis` — both measure the
    detector against the rotation axis. The independent one is
    :func:`tilt_from_beam_box`, which measures it against the slits.

    Resolution is set by the sweep, not by the fit: with a step ``s`` over a
    row span ``R`` nothing finer than ``atan(s / R)`` is visible. When the
    measured angle falls below that, this reports an **upper bound** rather
    than a detection, because a fitted slope through quantised picks always
    returns something.
    """
    from .center import find_center

    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError(f"cube must be 4-D (shift, slice, y, x); got {cube.shape}")
    n_slices = cube.shape[1]
    step = abs(float(shift_values[2])) or 1.0

    if slices is None:
        k = min(6, n_slices)
        slices = sorted({int(round(i * (n_slices - 1) / max(k - 1, 1)))
                         for i in range(k)})
    slices = [int(s) for s in slices]

    lo_edge = float(min(shift_values[0], shift_values[1]))
    hi_edge = float(max(shift_values[0], shift_values[1]))
    rows: List[float] = []
    picks: List[float] = []
    dropped: List[int] = []
    for s in slices:
        r = find_center(cube, shift_values, slice_idx=s, method=method, crop=crop)
        b = float(r["best_shift"])
        # An edge pick is not an interior optimum; it usually means that slice
        # has no specimen in it. Same rule as find_center_consensus.
        if abs(b - lo_edge) < 1e-9 or abs(b - hi_edge) < 1e-9 or not r["well_determined"]:
            dropped.append(s)
            continue
        rows.append(float(s))
        picks.append(b)

    reasons: List[str] = []
    if len(rows) < 3:
        raise ValueError(
            f"only {len(rows)} slices gave a usable shift (dropped {dropped}); "
            "need at least 3 to fit a trend against row"
        )

    r_arr = np.array(rows)
    p_arr = np.array(picks)
    span = float(r_arr.max() - r_arr.min())
    if span < min_span_rows:
        reasons.append(
            f"the usable slices span only {span:.0f} rows; the lever arm is "
            "too short to see a small roll"
        )

    A = np.vstack([r_arr, np.ones_like(r_arr)]).T
    sol, *_ = np.linalg.lstsq(A, p_arr, rcond=None)
    slope = float(sol[0])
    rms = float(np.sqrt(np.mean((p_arr - A @ sol) ** 2)))
    angle = math.degrees(math.atan(slope))

    resolution = math.degrees(math.atan(step / span)) if span > 0 else float("inf")
    if abs(angle) < resolution:
        reasons.append(
            f"|{angle:+.4f}| deg is below this sweep's resolution of "
            f"{resolution:.4f} deg (step {step} px over {span:.0f} rows). "
            "Report an UPPER BOUND, not a detection - a slope fitted through "
            "quantised picks always returns some number."
        )
    if dropped:
        reasons.append(f"slices {dropped} gave no interior optimum and were dropped")

    return TiltResult(
        angle_deg=angle,
        uncertainty_deg=(math.degrees(math.atan(rms / span)) if span > 0
                         else float("nan")),
        method="per-slice best shift",
        trustworthy=not reasons,
        reason="; ".join(reasons) if reasons else
               f"shift varies linearly with row over {span:.0f} rows",
        detail={"rows": rows, "shifts": picks, "slope_px_per_row": slope,
                "fit_rms_px": rms, "row_span": span,
                "resolution_deg": resolution, "dropped_slices": dropped,
                "sweep_step_px": step},
    )
