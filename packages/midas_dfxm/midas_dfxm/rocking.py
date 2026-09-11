"""Reduce a measured DFXM rocking scan to per-pixel maps, with checks that fail loudly.

A rocking scan is a stack of detector frames taken while an angle is swept through the
Bragg condition. Per pixel, the rocking-curve centre is a local lattice **tilt** (a theta or
chi rock at fixed 2theta) or a **d-spacing** (2theta moving, usually theta-2theta), and its
width is the local spread convolved with the instrument.

Three steps silently corrupt that answer when they are done wrong. Each has a check here
that can come back the other way.

:func:`check_frame_order`
    Frames paired with the wrong angles still give a smooth-looking map. A tutorial once
    sorted NaMnO2 frames with a filename pattern that matched a constant field, so the sort
    did nothing; its tilt map correlated with the correct one at r = +0.69 and -0.07 on two
    scans, with no error raised. The check scores how alike adjacent frames are against
    random orderings of the same frames, and -- when a scan has repeats -- how closely two
    repeats of the same point agree against random pairings. It rejects a scrambled
    pairing; it cannot see a reversed sweep, a wrong step or offset, or another smooth order.

:func:`reduce_rocking`
    The pedestal. On 6-ID-C and ID03 frames the detector pedestal carries ~90-98 % of the
    counts, so a first moment taken on raw frames is pulled toward the window centre and the
    tilt amplitude comes out many times too small, still looking smooth. The default
    subtracts a per-pixel baseline measured on the frames *outside* that pixel's own peak,
    then takes the moment inside the peak.

The curve shape
    A per-pixel centre is one number for a whole rocking curve. It is a tilt only when the
    curve is one peak. On a 6-ID-C scan of a Ba122 crystal every pixel's curve was 30-40
    mdeg wide and made of sharp 1-2 mdeg features; the peak-window centre followed whichever
    feature was tallest and jumped by ~19 mdeg along a line where two features were equally
    tall, a boundary that was not in the curves. ``reduce_rocking`` therefore reports, per
    pixel, whether the curve is single-peaked, and ``window="fixed"`` reduces the whole curve
    with a centre that cannot jump (the median of the curve), a width (10 % to 90 % span)
    and the disagreement between two centre definitions, which is the size of the
    "what does a centre mean here" systematic.

The error bar
    A model-free split-half: repeat parity when the scan has repeats, angle parity otherwise
    (also carries sampling error, so it is conservative). Repeat parity includes everything
    that changes between frames taken at the same position -- flux, beam or sample motion --
    and on the Ba122 scan above that was several times the photon noise, with the intensity
    pattern moving a few pixels between frames. ``repeat_excess`` maps where the frames at one
    position disagree beyond photon noise. Passing ``gain=`` adds the Poisson centroid error
    of :func:`midas_dfxm.centroid_uncertainty`, computed on the RECORDED counts, restricted to
    the peak window. Neither sees anything the two halves share -- angle errors per point,
    flux changes between points, drift -- nor a background that rises and falls with the
    rocking curve. :func:`baseline_sensitivity` measures the separate, systematic part: how
    far defensible baseline choices move the map.

Nothing here reads files. :mod:`midas_dfxm.io_6idc` builds a :class:`RockingScan` from the
APS 6-ID-C layouts, and :meth:`RockingScan.from_arrays` builds one from anything else.
Frames stay in detector (row, column) order: no flips are applied here.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "RockingScan",
    "RockingMaps",
    "FrameOrderCheck",
    "classify_motors",
    "check_frame_order",
    "reduce_rocking",
    "baseline_sensitivity",
    "example_rocking_scan",
]

# Motor names across 6-ID-C generations (lower-cased, spaces removed).
THETA_NAMES = ("theta", "th")
TWO_THETA_NAMES = ("two-theta", "tth", "2theta", "twotheta", "two_theta")
TILT_NAMES = THETA_NAMES + ("chi", "phi", "mu", "omega")
_TRANSLATION = re.compile(r"^(sam|sym|sample|sm)[a-z_]*[xyz]$|^[xyz]pos$")
FIXED_TOL_DEG = 1e-4


def _canon(name) -> str:
    return str(name).strip().lower().replace(" ", "")


def _first_key(motors, names):
    for k in motors:
        if _canon(k) in names:
            return k
    return None


def _finite(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    return v[np.isfinite(v)]


def _is_angle(name) -> bool:
    return _canon(name) in TILT_NAMES + TWO_THETA_NAMES


# --------------------------------------------------------------------------- scan model
def classify_motors(motors: dict, *, fixed_tol_deg: float = FIXED_TOL_DEG) -> dict:
    """Decide what a scan measures from which angles MOVE in its motor table.

    Returns a dict with ``scan_type`` (``"tilt"``, ``"strain"`` or ``"tilt2d"``), ``axes``
    (the motor column(s) that define the rocking coordinate), ``two_theta_key``, ``moving``
    (every moving column with its range and median step) and ``notes``.

    Rules: 2theta moving means a d-spacing (strain) scan, whether or not theta moves with it;
    otherwise one moving tilt motor is a tilt scan and two are a mesh. Never decide this from
    a filename: the NaMnO2 archive's filenames encode two coupled angles that make every scan
    look like theta-2theta, while its motor log shows 2theta fixed.
    """
    moving = {}
    for k, v in motors.items():
        f = _finite(v)
        if f.size > 1 and float(np.ptp(f)) > fixed_tol_deg:
            d = np.abs(np.diff(f))
            d = d[d > 0.1 * fixed_tol_deg]
            moving[k] = {"min": float(f.min()), "max": float(f.max()),
                         "step": float(np.median(d)) if d.size else 0.0}
    notes = []
    translating = [k for k in moving if _TRANSLATION.match(_canon(k))]
    if translating:
        notes.append(f"WARNING: sample translation {translating} moves during the scan; a "
                     "rocking curve assumes one sample position")
    tth = _first_key(motors, TWO_THETA_NAMES)
    th = _first_key(motors, THETA_NAMES)
    tilts = [k for k in motors if _canon(k) in TILT_NAMES and k in moving]
    base = {"two_theta_key": tth, "moving": moving, "notes": notes}
    if tth is not None and tth in moving:
        if th is not None and th in moving:
            a = np.asarray(motors[th], float)
            b = np.asarray(motors[tth], float)
            ok = np.isfinite(a) & np.isfinite(b)
            slope = float(np.polyfit(a[ok], b[ok], 1)[0])
            notes.append(f"{tth} and {th} both move: d({tth})/d({th}) = {slope:.3f}")
            if abs(slope - 2.0) > 0.05:
                notes.append("WARNING: 2theta does not track 2*theta, so the peak position "
                             "mixes a tilt with a d-spacing change")
        extra = [k for k in tilts if k != th]
        if extra:
            notes.append(f"WARNING: tilt motor {extra} also moves during a strain scan")
        return {"scan_type": "strain", "axes": (tth,), **base}
    if len(tilts) == 1:
        return {"scan_type": "tilt", "axes": (tilts[0],), **base}
    if len(tilts) == 2:
        return {"scan_type": "tilt2d", "axes": tuple(tilts), **base}
    if not tilts:
        angular = [k for k in motors if _is_angle(k)]
        raise ValueError(
            f"no rocking angle moves in this motor table (tolerance {fixed_tol_deg} deg). "
            f"Angular columns: {angular or 'none'}; moving columns: {sorted(moving) or 'none'}")
    raise ValueError(f"{len(tilts)} tilt motors move ({tilts}); a rocking scan sweeps one "
                     "angle, or two on a mesh")


@dataclass
class RockingScan:
    """One rocking scan: detector frames paired with the motor readings of each point.

    Attributes
    ----------
    frames : (M, H, W) float32
        One frame per scan point, in acquisition order, repeats already averaged.
    motors : dict of (M,) float arrays
        Every numeric column of the motor table, one value per point.
    scan_type : ``"tilt"``, ``"strain"`` or ``"tilt2d"``
        From :func:`classify_motors`.
    axes : tuple of str
        Motor column(s) defining the rocking coordinate.
    two_theta_deg : float or None
        Median 2theta of the motor log.
    halves : (A, B) or None
        Even- and odd-repeat averages, each ``(M, H, W)``, when every point had >= 2 kept
        repeats. They give a photon-noise-only split-half error bar.
    n_repeats : int
        Frames averaged per point. Recorded counts per point are
        ``n_repeats * (frames + dark_level)``.
    dark_level : float or (H, W) array
        Dark already subtracted from ``frames`` (added back for the error model only).
    source, notes, meta
        Provenance: where it came from and what the reader did to it.
    """

    frames: np.ndarray
    motors: dict
    scan_type: str
    axes: tuple
    two_theta_deg: Optional[float] = None
    halves: Optional[tuple] = None
    n_repeats: int = 1
    dark_level: object = 0.0
    source: str = ""
    notes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_arrays(cls, frames, motors: dict, *, halves=None, n_repeats: int = 1,
                    dark_level=0.0, source: str = "arrays", notes=None, meta=None,
                    fixed_tol_deg: float = FIXED_TOL_DEG) -> "RockingScan":
        """Build a scan from a ``(M, H, W)`` stack and a ``{motor: (M,)}`` table."""
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 3:
            raise ValueError(f"frames must be (M, H, W); got shape {frames.shape}")
        M = frames.shape[0]
        motors = {str(k): np.asarray(v, dtype=float).reshape(-1) for k, v in motors.items()}
        wrong = [k for k, v in motors.items() if v.size != M]
        if wrong:
            raise ValueError(f"motor columns {wrong} do not have one value per frame ({M}). "
                             "Refusing to guess the frame/angle pairing.")
        if halves is not None:
            halves = tuple(np.asarray(h, dtype=np.float32) for h in halves)
            if len(halves) != 2 or any(h.shape != frames.shape for h in halves):
                raise ValueError("halves must be two arrays shaped like frames")
        info = classify_motors(motors, fixed_tol_deg=fixed_tol_deg)
        tth = None
        if info["two_theta_key"] is not None:
            f = _finite(motors[info["two_theta_key"]])
            tth = float(np.median(f)) if f.size else None
        return cls(frames=frames, motors=motors, scan_type=info["scan_type"],
                   axes=tuple(info["axes"]), two_theta_deg=tth, halves=halves,
                   n_repeats=int(n_repeats), dark_level=dark_level, source=source,
                   notes=list(notes or []) + info["notes"],
                   meta={**(meta or {}), "moving": info["moving"]})

    @property
    def coordinate(self) -> np.ndarray:
        """Rocking coordinate per point, degrees: the tilt motor, or 2theta/2 for strain.

        ``(M,)`` for a 1-D scan, ``(M, 2)`` for a mesh.
        """
        if self.scan_type == "strain":
            return 0.5 * self.motors[self.axes[0]]
        if self.scan_type == "tilt":
            return self.motors[self.axes[0]]
        return np.stack([self.motors[a] for a in self.axes], -1)

    def summary(self) -> str:
        M, H, W = self.frames.shape
        coord = f"{self.axes[0]} / 2" if self.scan_type == "strain" else ", ".join(self.axes)
        lines = [f"source      {self.source}",
                 f"frames      {M} points x {H} x {W} px, {self.n_repeats} repeat(s) "
                 "averaged per point",
                 f"scan type   {self.scan_type} (rocking coordinate: {coord})"]
        if self.two_theta_deg is not None:
            lines.append(f"2theta      {self.two_theta_deg:.4f} deg (median of the motor log)")
        for k, m in self.meta.get("moving", {}).items():
            if _is_angle(k):
                lines.append(f"moves       {k}: {m['min']:.4f} .. {m['max']:.4f} deg, "
                             f"step {1000 * m['step']:.3f} mdeg")
        lines.append(f"split-half  {'repeat parity' if self.halves is not None else 'angle parity'}")
        lines += [f"note        {n}" for n in self.notes]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"RockingScan({self.scan_type}, axes={self.axes}, frames={self.frames.shape}, "
                f"n_repeats={self.n_repeats}, source={self.source!r})")


def _crop(a, roi):
    if roi is None:
        return a
    r0, r1, c0, c1 = roi
    return a[..., r0:r1, c0:c1]


def _order_1d(scan: RockingScan) -> np.ndarray:
    x = scan.coordinate
    return np.argsort(x, kind="stable") if x.ndim == 1 else np.arange(len(x))


def _value_scale(scan: RockingScan):
    """Factor from the rocking coordinate (deg) to the reported unit, and the unit.

    Tilt: mdeg. Strain: ``-cot(theta_B)`` in microstrain per degree of 2theta/2.
    """
    if scan.scan_type == "strain":
        if scan.two_theta_deg is None:
            raise ValueError("strain scan without a 2theta reading")
        cot = 1.0 / math.tan(math.radians(scan.two_theta_deg / 2.0))
        return -cot * math.pi / 180.0 * 1e6, "microstrain"
    return 1000.0, "mdeg"


# --------------------------------------------------------------------------- frame order
@dataclass
class FrameOrderCheck:
    """Result of :func:`check_frame_order`. ``consistent`` is the verdict.

    ``repeat_statistic`` (when the scan has repeats) is the mean squared difference between
    the two repeat halves of the same point, divided by its median over random pairings:
    well below 1 when repeats of one point agree.
    """

    statistic: float
    null_median: float
    null_p99: float
    p_value: float
    consistent: bool
    n_pixels: int
    n_perm: int
    message: str
    repeat_statistic: Optional[float] = None
    repeat_p_value: Optional[float] = None

    def __str__(self) -> str:
        return self.message


def _rows_unit(X):
    X = X - X.mean(1, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norm, out=np.zeros_like(X), where=norm > 0)


def _top_pixels(score, quantile=0.95):
    return np.flatnonzero(score >= np.quantile(score, quantile))


def check_frame_order(scan: RockingScan, *, roi=None, lit=None, n_perm: int = 200,
                      alpha: float = 0.01, max_pixels: int = 200_000,
                      seed: int = 0) -> FrameOrderCheck:
    """Is each frame paired with its own angle? Test it against random orderings.

    **Adjacent points.** Neighbouring points of a rocking scan see nearly the same
    diffracting region, so their baseline-subtracted frames correlate; a shuffled stack loses
    that. The statistic is the mean Pearson correlation between frames adjacent in the
    rocking coordinate, over the pixels that vary most; the null is the same statistic over
    ``n_perm`` random orderings (its false-positive rate measured 1.00-1.11 % per scan at the
    default ``alpha``, 100,000 random orders each). On the NaMnO2 practice scans (S006, S011, ROI (800, 1600, 900, 1700)) the
    statistic is 0.239 and 0.270 against a null 99th percentile of about 0.05.

    **Repeats of one point** (only when ``scan.halves`` exists). Reading point-major frames
    as repeat-major blends two half-speed sweeps: the averaged stack is still smooth, so the
    adjacent-point test passes it. The two repeat halves of one point, however, then come
    from far-apart angles. The statistic is the mean squared difference of half A and half B
    at the same point; the null pairs A with B at random points. Its pixels are chosen from
    each half separately, so the choice cannot depend on the pairing being tested: chosen on
    the averaged stack instead, pure noise passed 300 of 300 times.

    ``consistent`` requires every available test at ``p <= alpha``; the smallest p this can
    report is ``1 / (n_perm + 1)``.

    **What CONSISTENT does not certify.** It rules out a scrambled pairing. A reversed sweep
    (tilts change sign), a doubled step (tilts double), a constant offset, a cyclic shift or
    an interleaved order all keep neighbouring frames neighbouring and pass with the same
    score. So does repeat-major data read as point-major: each point's repeats then come from
    neighbouring angles, and at 2 repeats this passed 40 of 40 to 100 of 100 synthetic scans.
    Only the acquisition record tells point-major from repeat-major. A brighter first repeat
    can also let point-major data read as repeat-major pass, when the number of points shares
    a factor with the number of repeats: both halves of each misread point then hold
    first-repeat frames, and 51 points x 3 repeats at a weak peak passed 40 of 40. Removing each
    frame's level from the repeat statistic does not cure it; it lets strong-signal misreads
    through instead. So never switch ``order`` until the check passes. The motor table's
    direction, step and offset have to be right on their own. And
    ``consistent=False`` does not prove the order wrong: a scan with no contrast, or with a
    peak narrower than about one step, cannot confirm any order. For a mesh, frames are
    taken in table order.
    """
    F = _crop(scan.frames, roi)
    M = F.shape[0]
    if M < 5:
        return FrameOrderCheck(float("nan"), float("nan"), float("nan"), 1.0, False, 0, 0,
                               f"only {M} frames: frame order cannot be tested")
    order = _order_1d(scan)
    X = F[order].reshape(M, -1).astype(np.float64)
    X -= np.median(X, axis=0, keepdims=True)
    mask = None
    if lit is None:
        keep = _top_pixels(X.max(0))
    else:
        mask = np.asarray(lit, bool).reshape(-1)
        if mask.size != X.shape[1]:
            raise ValueError("lit mask does not match the (cropped) frame size")
        keep = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if keep.size > max_pixels:
        keep = rng.choice(keep, max_pixels, replace=False)
    Xu = _rows_unit(X[:, keep])
    C = Xu @ Xu.T
    stat = float(np.mean(np.diagonal(C, 1)))
    perms = [rng.permutation(M) for _ in range(n_perm)]
    null = np.array([np.mean(C[p[:-1], p[1:]]) for p in perms])
    pval = float((1.0 + np.sum(null >= stat)) / (1.0 + n_perm))
    ok = pval <= alpha
    med, p99 = float(np.median(null)), float(np.quantile(null, 0.99))
    parts = [f"adjacent-frame correlation {stat:.3f} vs random orders median {med:.3f}, "
             f"p99 {p99:.3f} (p = {pval:.3g})"]
    rstat = rp = None
    if scan.halves is not None:
        A, B = (_crop(h, roi)[order].reshape(M, -1).astype(np.float64) for h in scan.halves)
        if mask is None:
            score = ((A - np.median(A, axis=0, keepdims=True)).max(0)
                     + (B - np.median(B, axis=0, keepdims=True)).max(0))
            rkeep = _top_pixels(score)
        else:
            rkeep = np.flatnonzero(mask)
        if rkeep.size > max_pixels:
            rkeep = rng.choice(rkeep, max_pixels, replace=False)
        A, B = A[:, rkeep], B[:, rkeep]
        D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
        same = float(np.mean(np.diagonal(D)))
        rnull = np.array([np.mean(D[np.arange(M), p]) for p in perms])
        rp = float((1.0 + np.sum(rnull <= same)) / (1.0 + n_perm))
        rmed = float(np.median(rnull))
        rstat = same / rmed if rmed > 0 else float("nan")
        ok = ok and rp <= alpha
        parts.append(f"repeat halves of one point differ by {rstat:.3g}x the random-pairing "
                     f"median (p = {rp:.3g})")
    verdict = "CONSISTENT" if ok else "NOT CONFIRMED"
    msg = (f"frame order {verdict}: " + "; ".join(parts)
           + f" [{keep.size} px, {n_perm} permutations, smallest possible p "
             f"{1.0 / (1.0 + n_perm):.3g}]")
    if ok:
        msg += (". This rules out a scrambled pairing, not a reversed sweep or a wrong "
                "step or offset in the motor table.")
    else:
        msg += (". Check the frame/angle pairing (file order, motor table, point- vs "
                "repeat-major) before trusting any map.")
    return FrameOrderCheck(stat, med, p99, pval, bool(ok), int(keep.size), n_perm, msg,
                           rstat, rp)


# --------------------------------------------------------------------------- reduction
@dataclass
class RockingMaps:
    """Per-pixel maps from :func:`reduce_rocking`. All images are ``(H, W)`` of the ROI.

    ``value`` is the tilt (mdeg) or strain (microstrain) relative to ``reference_deg``; for a
    mesh it is ``(H, W, 2)``, one tilt per axis. ``sigma`` is the split-half error bar in the
    same unit (a 5x5-pixel local estimate), ``sigma_global`` the robust value over lit pixels.
    ``snr`` is the integrated intensity over its own noise, measured off-peak; ``lit`` is
    derived from it by default. ``settings`` records every choice, so the map can be redone.
    """

    scan_type: str
    axes: tuple
    unit: str
    value: np.ndarray
    centre_deg: np.ndarray
    reference_deg: object
    intensity: np.ndarray
    snr: np.ndarray
    baseline: np.ndarray
    lit: np.ndarray
    truncated: np.ndarray
    baseline_ok: np.ndarray
    fwhm_mdeg: Optional[np.ndarray]
    points_above_half: Optional[np.ndarray]
    sigma: Optional[np.ndarray]
    sigma_global: Optional[float]
    sigma_poisson: Optional[np.ndarray]
    pedestal_share: float
    split: Optional[str]
    settings: dict
    notes: list
    # shape of each pixel's curve (1-D scans)
    peak_share: Optional[np.ndarray] = None
    n_features: Optional[np.ndarray] = None
    single_peaked: Optional[np.ndarray] = None
    # window="fixed" only: the second centre definition and the whole-curve width
    centre_mass_deg: Optional[np.ndarray] = None
    centre_shift: Optional[np.ndarray] = None
    span_mdeg: Optional[np.ndarray] = None
    # repeats only: where frames at one position disagree beyond photon noise
    repeat_excess: Optional[np.ndarray] = None
    repeat_excess_spread: Optional[tuple] = None
    repeat_shift_px: Optional[np.ndarray] = None
    repeat_shift_rms: Optional[tuple] = None

    def summary(self) -> str:
        lit = self.lit
        n = int(lit.sum())
        good = lit & ~self.truncated
        fixed = self.settings.get("window") == "fixed"
        edge = ("signal still above noise at the ends of the signal window" if fixed
                else "peak window at a scan edge")
        lines = [f"{self.scan_type} map: {n} lit px ({lit.mean():.1%} of the ROI); "
                 f"{int((lit & self.truncated).sum())} of them have their {edge} "
                 "(centre may be biased; left out below)"]
        v = self.value if self.value.ndim == 2 else None
        if v is not None and good.any():
            q = np.nanpercentile(v[good], [2, 25, 50, 75, 98])
            q = np.where(np.abs(q) < 1e-6, 0.0, q)          # the median is 0 by construction
            what = "median of each pixel's curve" if fixed else "first moment in the peak window"
            lines.append(f"value       median {q[2]:.3g} {self.unit}, IQR [{q[1]:.3g}, {q[3]:.3g}], "
                         f"p2-p98 [{q[0]:.3g}, {q[4]:.3g}] (relative to the reference; {what})")
            lines.append(f"reference   {float(self.reference_deg):.5f} deg (median centre, lit pixels)")
        if self.single_peaked is not None and lit.any():
            sp = float(self.single_peaked[lit].mean())
            lines.append(f"shape       {sp:.1%} of lit pixels are single-peaked (>= 45 % of the "
                         "signal inside the half-max run, one feature above half max); for the "
                         f"other {1 - sp:.1%} the centre summarises a distribution of tilts, "
                         "not one tilt")
        if fixed and self.centre_shift is not None and good.any():
            d = np.abs(self.centre_shift[good])
            d = d[np.isfinite(d)]
            if d.size:
                lines.append(f"centre def. centre of mass differs from the median by "
                             f"{np.median(d):.3g} {self.unit} (median), {np.percentile(d, 90):.3g} "
                             "(p90): the size of the 'which centre' systematic")
        if self.fwhm_mdeg is not None and good.any():
            fw = np.nanmedian(self.fwhm_mdeg[good])
            na = np.nanmedian(self.points_above_half[good])
            if self.span_mdeg is not None:
                sp = np.nanmedian(self.span_mdeg[good])
                lines.append(f"width       10-90 % span of the whole curve, median {sp:.3g} mdeg -- "
                             "use this one: the curve may not be single-peaked (see shape). The "
                             f"tallest local feature alone is {fw:.3g} mdeg wide (contiguous half-max, "
                             f"median {na:.0f} points above its own half max) and is NOT the curve's "
                             "width unless a pixel is single-peaked.")
            else:
                lines.append(f"width       median {fw:.3g} mdeg (contiguous half-max, instrument "
                             f"included); median {na:.0f} points above half max")
        if self.sigma_global is not None:
            lines.append(f"error bar   {self.sigma_global:.3g} {self.unit} per pixel "
                         f"({self.split} split-half; one robust value over lit pixels, while "
                         "the sigma map is a local 5x5 estimate)")
        if self.repeat_excess_spread is not None:
            lo, hi, absx = self.repeat_excess_spread
            lines.append(f"repeats     frames at one position: variance spread p10-p90 = "
                         f"{lo:.2f}-{hi:.2f} x the lit-pixel median (photon noise alone gives a "
                         "spread near 1; a wide spread is flux or motion between frames)"
                         + (f"; {absx:.2f} x photon noise at the given gain" if absx is not None
                            else ""))
        if self.repeat_shift_rms is not None:
            ey, ex, fy, fx, n = self.repeat_shift_rms
            lines.append(f"repeats     between the two repeat halves the intensity envelope shifts "
                         f"{ey:.2f} px (rows) / {ex:.2f} px (cols) rms and the fine detail "
                         f"{fy:.2f} / {fx:.2f} px, signal-weighted rms over {n} frames (photon noise "
                         "alone: hundredths of a pixel on a full frame)")
        if self.sigma_poisson is not None and good.any():
            lines.append(f"Poisson     median {np.nanmedian(self.sigma_poisson[good]):.3g} "
                         f"{self.unit} (gain {self.settings['gain']})")
        share = self.pedestal_share
        lines.append(f"pedestal    {share:.3f} of the recorded counts in lit pixels was baseline"
                     + (f" -> a raw first moment would be diluted ~{1 / (1 - share):.0f}x"
                        if share < 1 else ""))
        lines.append("settings    " + ", ".join(f"{k}={self.settings[k]}" for k in
                     ("baseline", "baseline_percentile", "peak_halfwidth", "window", "lit")))
        lines += [f"note        {x}" for x in self.notes]
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Write every map and the settings to a compressed ``.npz``."""
        arrays = {k: getattr(self, k) for k in ("value", "centre_deg", "intensity", "snr",
                                                "baseline", "lit", "truncated", "baseline_ok")}
        for k in ("fwhm_mdeg", "points_above_half", "sigma", "sigma_poisson", "peak_share",
                  "n_features", "single_peaked", "centre_mass_deg", "centre_shift", "span_mdeg",
                  "repeat_excess", "repeat_shift_px"):
            if getattr(self, k) is not None:
                arrays[k] = getattr(self, k)
        meta = {"scan_type": self.scan_type, "axes": list(self.axes), "unit": self.unit,
                "reference_deg": np.asarray(self.reference_deg).tolist(),
                "sigma_global": self.sigma_global, "pedestal_share": self.pedestal_share,
                "split": self.split, "settings": self.settings, "notes": self.notes,
                "repeat_excess_spread": self.repeat_excess_spread,
                "repeat_shift_rms": self.repeat_shift_rms}
        np.savez_compressed(path, **arrays, meta=json.dumps(meta, default=str))


def _smooth3(s):
    out = s.copy()
    if s.shape[0] >= 3:
        out[1:-1] = (s[:-2] + s[1:-1] + s[2:]) / 3.0
        out[0] = 0.5 * (s[0] + s[1])
        out[-1] = 0.5 * (s[-2] + s[-1])
    return out


def _nanmedian0(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=0)


def _moment(s, x, W):
    sw = np.where(W, s, 0.0)
    den = sw.sum(0)
    num = (sw * x[:, None]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = num / den
    return np.where(den > 0, c, np.nan), den


def _peak_window(s, h_scale):
    """Argmax of the smoothed curve, contiguous points above half max, window half-width."""
    M = s.shape[0]
    sm = _smooth3(s)
    k = sm.argmax(0)
    cols = np.arange(s.shape[1])
    half = 0.5 * sm[k, cols]
    idx = np.arange(M)[:, None]
    below = sm < half[None]
    left = np.where(below & (idx < k[None]), idx, -1).max(0)
    right = np.where(below & (idx > k[None]), idx, M).min(0)
    n_above = np.maximum(right - left - 1, 1)
    h = np.ceil(h_scale * n_above).astype(int)
    return k, n_above, h


def _fwhm(s, x, k):
    """Contiguous half-max width around ``k``, interpolated (deg); also whether it hits an end."""
    M, N = s.shape
    cols = np.arange(N)
    lo, hi = np.clip(k - 1, 0, M - 1), np.clip(k + 1, 0, M - 1)
    peak = np.maximum(np.maximum(s[lo, cols], s[k, cols]), s[hi, cols])
    half = 0.5 * peak
    idx = np.arange(M)[:, None]
    below = s < half[None]
    left = np.where(below & (idx < k[None]), idx, -1).max(0)
    right = np.where(below & (idx > k[None]), idx, M).min(0)

    def cross(e, step):
        e = np.clip(e, 0, M - 1)
        e2 = np.clip(e + step, 0, M - 1)
        y1, y2 = s[e, cols], s[e2, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(np.abs(y2 - y1) > 0, (half - y1) / (y2 - y1), 0.0)
        return x[e] + np.clip(t, 0, 1) * (x[e2] - x[e])

    xl = np.where(left >= 0, cross(left, +1), x[0])
    xr = np.where(right < M, cross(right, -1), x[-1])
    return xr - xl, right - left - 1, (left < 0) | (right >= M)


def _baseline(I, inside, method, pct, min_pts):
    N = I.shape[1]
    if method == "outside_peak":
        n_out = (~inside).sum(0)
        b = _nanmedian0(np.where(inside, np.nan, I))
        ok = n_out >= min_pts
        return np.where(ok, b, np.percentile(I, pct, axis=0)), ok
    if method == "percentile":
        return np.percentile(I, pct, axis=0), np.ones(N, bool)
    if method == "none":
        return np.zeros(N), np.ones(N, bool)
    raise ValueError(f"unknown baseline {method!r}: use 'outside_peak', 'percentile' or 'none'")


def _window_sigma(s, b, x, W, com, gain, read_var):
    """:func:`midas_dfxm.centroid_uncertainty` restricted to the peak window.

    ``var(c) = sum_W (x - c)^2 (gain * recorded + read_var) / S^2`` with ``recorded = s + b``
    (the background returns for the variance), ``S = sum_W s``.
    """
    lever = (x[:, None] - com[None]) ** 2
    S = np.where(W, s, 0.0).sum(0)
    var = np.where(W, lever * (gain * (s + b) + read_var), 0.0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.maximum(var, 0.0)) / np.where(S > 0, S, np.nan)


def _snr(I, b, S, W, off):
    """Integrated intensity over its own noise; the per-frame noise is the MAD off-peak."""
    dev = _nanmedian0(np.where(off, np.abs(I - b), np.nan))
    n_off = off.sum(0)
    sig = 1.4826 * dev * np.sqrt(np.maximum(W.sum(0), 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((n_off >= 3) & (sig > 0), S / sig, np.nan)


def _contiguous_run(sm, k, frac):
    """First/last index of the run around ``k`` where ``sm >= frac * sm[k]`` (per column)."""
    M = sm.shape[0]
    cols = np.arange(sm.shape[1])
    idx = np.arange(M)[:, None]
    below = sm < (frac * sm[k, cols])[None]
    left = np.where(below & (idx < k[None]), idx, -1).max(0) + 1
    right = np.where(below & (idx > k[None]), idx, M).min(0) - 1
    return left, right


def _shape_stats(s):
    """Per pixel: share of the signal inside the half-max run, and the number of features.

    ``share`` = clipped signal inside the contiguous run above half maximum, divided by the
    clipped signal inside the contiguous run above 10 % of the maximum (both around the
    argmax of the 3-point-smoothed curve). One Gaussian gives ~0.78, one Lorentzian ~0.63, a
    30 mdeg box with sharp horns 0.1-0.3. ``n_features`` counts separate runs above half
    maximum anywhere in the scan, separated by dips below 35 % of the maximum (hysteresis, so
    a noisy flank does not split a peak). Single-peaked means share >= 0.45 and one feature.
    """
    M, N = s.shape
    sm = _smooth3(s)
    k = sm.argmax(0)
    cols = np.arange(N)
    idx = np.arange(M)[:, None]
    l5, r5 = _contiguous_run(sm, k, 0.5)
    l1, r1 = _contiguous_run(sm, k, 0.1)
    c = np.clip(s, 0.0, None)
    in5 = (idx >= l5[None]) & (idx <= r5[None])
    in1 = (idx >= l1[None]) & (idx <= r1[None])
    num = np.where(in5, c, 0.0).sum(0)
    den = np.where(in1, c, 0.0).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(den > 0, num / den, np.nan)
    mx = sm[k, cols]
    high = sm >= 0.5 * mx[None]
    deep = sm < 0.35 * mx[None]
    n_feat = np.zeros(N, int)
    armed = np.ones(N, bool)                  # a new feature may start
    for m in range(M):
        start = high[m] & armed
        n_feat += start
        armed = np.where(high[m], False, armed | deep[m])
    n_feat = np.where(mx > 0, n_feat, 0)
    return share, n_feat


def _line_baseline(I, x, outside):
    """Straight line per pixel through the frames flagged ``outside`` (a 1-D mask over M)."""
    xo = x[outside]
    xm = float(xo.mean())
    A = np.stack([np.ones_like(xo), xo - xm], 1)
    coef = np.linalg.pinv(A) @ I[outside]                  # (2, N)
    b = coef[0][None] + coef[1][None] * (x - xm)[:, None]
    res = I[outside] - b[outside]
    rms = np.sqrt((res ** 2).sum(0) / max(outside.sum() - 2, 1))
    return b, rms


def _quantiles(s, x, W, qs):
    """Positions where the cumulative clipped signal inside ``W`` reaches each ``q`` (deg)."""
    M, N = s.shape
    c = np.where(W, np.clip(s, 0.0, None), 0.0)
    cum = np.cumsum(c, 0)
    tot = cum[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.concatenate([np.zeros((1, N)), cum / np.where(tot > 0, tot, np.nan)[None]], 0)
    dx = np.diff(x)
    xe = np.concatenate([[x[0] - 0.5 * dx[0]], x[:-1] + 0.5 * dx, [x[-1] + 0.5 * dx[-1]]])
    out = {}
    cols = np.arange(N)
    for q in qs:
        k = np.clip(np.argmax(F >= q, 0), 1, M)
        f0, f1 = F[k - 1, cols], F[k, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(f1 > f0, (q - f0) / (f1 - f0), 0.0)
        out[q] = np.where(tot > 0, xe[k - 1] + t * (xe[k] - xe[k - 1]), np.nan)
    return out


def _box_sum(z, r):
    """Sum of ``z`` over a (2r+1)^2 box around each pixel (edges: smaller boxes)."""
    k = 2 * r + 1
    c = np.pad(z, ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def _repeat_excess(F, halves, r=3):
    """Where the frames at one position disagree beyond photon noise.

    Per pixel, over a (2r+1)^2 box: regress the box-summed squared difference of the two
    repeat halves on the box-summed recorded level, across the frames of the scan. If frames
    at one position differ by photon noise only, ``var(D) = a + S/g'`` with one ``g'`` for the
    whole detector, so the slope is one number everywhere; flux or motion between frames
    raises it where the image has structure. Box sums make the slope usable on a narrow peak,
    where only a few frames carry signal. Accumulated frame by frame, so memory stays flat.
    Returns the slope map (counts^-1, times the parity factor 1/nA + 1/nB).
    """
    M = F.shape[0]
    sS = sY = sSS = sSY = None
    for m in range(M):
        S = _box_sum(F[m].astype(np.float64), r)
        Y = _box_sum((halves[0][m].astype(np.float64) - halves[1][m]) ** 2, r)
        if sS is None:
            sS, sY, sSS, sSY = S.copy(), Y.copy(), S * S, S * Y
        else:
            sS += S; sY += Y; sSS += S * S; sSY += S * Y
    den = sSS - sS * sS / M
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (sSY - sS * sY / M) / den, np.nan)


def _repeat_shift(F, halves, lit, x, b):
    """Per frame, how the second repeat half differs from the first, as apparent shifts.

    Two scales, both gain-free, both over lit pixels, half 1 relative to half 0:

    * **envelope**: the intensity-weighted centroid (rows, cols) of the baseline-subtracted
      signal in half 1 minus half 0. What moves when the illumination or the whole intensity
      pattern moves.
    * **fine detail**: ``D = h0 - h1`` regressed on the frame itself (a flux change), its
      derivative along the scan (an angle offset) and the row and column gradients of the
      3x3-smoothed frame. The gradient coefficients are the shift of the pixel-scale structure.

    On 6-ID-C Ba122 frames the envelope moved by a few pixels between frames at one position
    while the fine detail stayed within a fraction of a pixel: the illumination moved over a
    still image. Photon noise alone gives hundredths of a pixel on both. Returns ``(M, 6)``:
    flux fraction, angle offset (deg), fine dy, fine dx, envelope dy, envelope dx (px).
    """
    M = F.shape[0]
    out = np.full((M, 6), np.nan)
    sel = lit
    if sel.sum() < 100:
        return out
    rr, cc = np.nonzero(sel)
    for m in range(M):
        Fm = F[m].astype(np.float64)
        sm = _box_sum(Fm, 1) / 9.0
        gy, gx = np.gradient(sm)
        lo, hi = max(m - 1, 0), min(m + 1, M - 1)
        dth = (F[hi].astype(np.float64) - F[lo]) / (x[hi] - x[lo]) if hi > lo else np.zeros_like(Fm)
        h0 = halves[0][m].astype(np.float64)[sel]
        h1 = halves[1][m].astype(np.float64)[sel]
        A = np.stack([Fm[sel], dth[sel], gy[sel], gx[sel]], 1)
        c, *_ = np.linalg.lstsq(A, h0 - h1, rcond=None)
        out[m, :4] = c
        s0 = np.clip(h0 - b[sel], 0.0, None)
        s1 = np.clip(h1 - b[sel], 0.0, None)
        if s0.sum() > 0 and s1.sum() > 0:
            out[m, 4] = (s1 * rr).sum() / s1.sum() - (s0 * rr).sum() / s0.sum()
            out[m, 5] = (s1 * cc).sum() / s1.sum() - (s0 * cc).sum() / s0.sum()
    return out


def _centre(s, x, W, mode):
    if mode == "median":
        return _quantiles(s, x, W, (0.5,))[0.5]
    return _moment(s, x, W)[0]


def _reduce_1d_chunk(I, halves, x, p, n_rep, dark):
    M, N = I.shape
    idx = np.arange(M)[:, None]
    fixed = p["window"] == "fixed"
    if fixed:
        lo, hi = p["signal_index"]
        Wf = np.zeros(M, bool)
        Wf[lo:hi + 1] = True
        inside = np.repeat(Wf[:, None], N, 1)
        if p["baseline"] == "ends" and (~Wf).sum() >= p["min_baseline_points"]:
            b, rms = _line_baseline(I, x, ~Wf)
            ok = np.ones(N, bool)
        else:
            b, ok = _baseline(I, inside, "percentile" if p["baseline"] == "ends" else p["baseline"],
                              p["baseline_percentile"], p["min_baseline_points"])
            ok = ok & ((~Wf).sum() >= p["min_baseline_points"])
            rms = 1.4826 * _nanmedian0(np.where(inside, np.nan, np.abs(I - b))) if (~Wf).any() \
                else np.full(N, np.nan)
    else:
        b = np.percentile(I, p["baseline_percentile"], axis=0)
        ok = np.ones(N, bool)
        inside = np.ones((M, N), bool)
        for _ in range(2):                   # window from the curve, baseline from outside it
            k, _n, h = _peak_window(I - b, p["peak_halfwidth"])
            inside = np.abs(idx - k[None]) <= h[None]
            if p["baseline"] != "outside_peak":
                break
            b, ok = _baseline(I, inside, "outside_peak", p["baseline_percentile"],
                              p["min_baseline_points"])
        if p["baseline"] != "outside_peak":
            b, ok = _baseline(I, inside, p["baseline"], p["baseline_percentile"],
                              p["min_baseline_points"])
    s = I - b
    k, _n, h = _peak_window(s, p["peak_halfwidth"])
    peak = np.abs(idx - k[None]) <= h[None]
    fwhm, n_above, edge_hit = _fwhm(s, x, k)
    if fixed:
        W = inside
        # signal still above noise just outside the window: the window cuts the curve
        guard = np.zeros(N, bool)
        for a0, a1 in ((max(lo - 2, 0), lo), (hi + 1, min(hi + 3, M))):
            if a1 > a0:
                guard |= s[a0:a1].mean(0) > 3.0 * rms / math.sqrt(a1 - a0)
        truncated = guard
        mode = "median"
    elif p["window"] == "full":
        W = np.ones((M, N), bool)
        truncated = edge_hit                  # the whole scan is the window: flag a cut-off peak
        mode = "moment"
    else:
        com, _ = _moment(s, x, peak)
        centre = np.rint(np.interp(np.where(np.isfinite(com), com, x[k]), x,
                                   np.arange(M))).astype(int)
        W = np.abs(idx - centre[None]) <= h[None]
        truncated = (centre - h < 0) | (centre + h > M - 1)
        mode = "moment"
    com, S = _moment(s, x, W)
    cen = _quantiles(s, x, W, (0.1, 0.5, 0.9)) if fixed else None
    share, n_feat = _shape_stats(s)
    b_px = b[(lo + hi) // 2] if b.ndim == 2 else b          # a line baseline: its value mid-window
    out = {"com": cen[0.5] if fixed else com, "intensity": S, "baseline": b_px, "ok": ok,
           "truncated": truncated, "fwhm": fwhm, "n_above": n_above, "total": I.sum(0),
           "snr": _snr(I, b, S, W, ~W if fixed else ~peak), "share": share, "n_feat": n_feat}
    if fixed:
        out["com_mass"] = com
        out["span"] = cen[0.9] - cen[0.1]
    if p["gain"] is not None:
        out["sig_p"] = _window_sigma(n_rep * s, n_rep * (b + dark), x, W, com, p["gain"],
                                     n_rep * p["read_var"])
    if p["split_half"]:
        if halves is not None:
            parts = [(halves[0], x, W, inside), (halves[1], x, W, inside)]
        else:
            parts = [(I[0::2], x[0::2], W[0::2], inside[0::2]),
                     (I[1::2], x[1::2], W[1::2], inside[1::2])]
        cs = []
        for J, xj, Wj, insj in parts:
            if fixed and p["baseline"] == "ends" and (~Wj[:, 0]).sum() >= 3:
                bj, _ = _line_baseline(J, xj, ~Wj[:, 0])
            else:
                bj, _ = _baseline(J, insj, "percentile" if p["baseline"] == "ends" else p["baseline"],
                                  p["baseline_percentile"],
                                  max(2, p["min_baseline_points"] // (1 if halves is not None else 2)))
            cs.append(_centre(J - bj, xj, Wj, mode))
        out["diff"] = cs[0] - cs[1]
    return out


def _reduce_mesh_chunk(I, halves, X, p):
    b = np.percentile(I, p["baseline_percentile"], axis=0)
    s = I - b
    W = s >= p["mesh_threshold"] * s.max(0)[None]
    coms = np.stack([_moment(s, X[:, a], W)[0] for a in range(X.shape[1])], -1)
    S = np.where(W, s, 0.0).sum(0)
    k = s.argmax(0)
    truncated = np.zeros(I.shape[1], bool)
    for a in range(X.shape[1]):
        xa = X[k, a]
        truncated |= np.isclose(xa, X[:, a].min()) | np.isclose(xa, X[:, a].max())
    out = {"com": coms, "intensity": S, "baseline": b, "ok": np.ones(I.shape[1], bool),
           "truncated": truncated, "total": I.sum(0), "snr": _snr(I, b, S, W, ~W)}
    if p["split_half"] and halves is not None:
        d = []
        for J in halves:
            sj = J - np.percentile(J, p["baseline_percentile"], axis=0)
            d.append(np.stack([_moment(sj, X[:, a], W)[0] for a in range(X.shape[1])], -1))
        out["diff"] = d[0] - d[1]
    return out


def _otsu(v, nbins=256):
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) == 0:
        return float(v.min()) if v.size else 0.0
    hist, edges = np.histogram(v, bins=nbins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    with np.errstate(invalid="ignore", divide="ignore"):
        between = w0 * w1 * (m0 / w0 - (m0[-1] - m0) / w1) ** 2
    i = int(np.nanargmax(np.nan_to_num(between[:-1], nan=-1.0)))
    return float(edges[i + 1])


def _box_rms_half(d, r=2):
    """sqrt(local mean of d^2)/2 over a (2r+1)^2 box, ignoring NaN: the split-half sigma."""
    m = np.isfinite(d)
    k = 2 * r + 1

    def box(z):
        c = np.pad(z, ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]

    num = box(np.where(m, d * d, 0.0))
    den = box(m.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, np.sqrt(num / den) / 2.0, np.nan)


def reduce_rocking(scan: RockingScan, *, roi=None, baseline: str = "outside_peak",
                   baseline_percentile: float = 25.0, peak_halfwidth: float = 2.0,
                   min_baseline_points: int = 6, window: str = "peak", lit="auto",
                   lit_snr: float = 10.0, reference: Optional[float] = None,
                   split_half: bool = True, gain: Optional[float] = None,
                   read_var: float = 0.0, mesh_threshold: float = 0.1,
                   chunk_rows: int = 128, signal_range=None) -> RockingMaps:
    """Per-pixel centre, width, intensity and error bar of a rocking scan.

    Two ways to reduce a 1-D scan:

    ``window="peak"`` (the default) assumes one peak per pixel and works inside a window
    around that pixel's own maximum. ``window="fixed"`` assumes nothing about the shape: one
    signal window for every pixel (``signal_range=(lo, hi)`` in the rocking coordinate, deg;
    by default the frames where the ROI-summed curve is above 1 % of its maximum, padded by
    3 frames), a straight-line baseline through the frames outside it (``baseline="ends"``,
    set automatically), the centre as the **median** of the pixel's clipped curve (the angle
    that splits its intensity in half, which cannot jump when a different feature becomes
    the tallest), the 10 % to 90 % span as a width, and ``centre_shift`` = centre of mass
    minus median, the size of the "which centre" systematic. A pixel is ``truncated`` there
    when its signal is still above noise just outside the window. Use it when the ``shape``
    line of the summary says many pixels are not single-peaked. Either way the centre is one
    number for a whole curve; where the curve is a distribution of tilts, the map shows how
    that distribution's weight is placed, not one tilt.

    Per pixel, with ``window="peak"``:

    1. **Baseline** (``baseline="outside_peak"``, the default): locate the pixel's peak on a
       3-point-smoothed curve, open a window of ``peak_halfwidth`` x (points above half
       max) either side, and take the median of the frames OUTSIDE it; repeated twice so the
       window and the baseline agree. Pixels with fewer than ``min_baseline_points`` frames
       outside fall back to the ``baseline_percentile`` percentile and are flagged in
       ``baseline_ok``. ``"percentile"`` uses that percentile of all frames; ``"none"`` is a
       control only -- it reproduces the pedestal dilution.
    2. **Centre**: first moment of the baseline-subtracted curve inside the window, re-centred
       once on the moment (``window="full"`` uses every frame).
    3. **Width**: argmax-local contiguous half-max crossings, interpolated (not a second
       moment, which diverges on Lorentzian tails). Instrument resolution is included.
    4. **Error bar**: split-half difference of the centre -- repeat parity if ``scan.halves``
       exists, else even vs odd frames -- giving ``sigma = SD(A - B) / 2``. With ``gain``,
       also the Poisson/read-noise centroid error on recorded counts. It belongs to these
       settings (a wider window admits more noisy frames), and it cannot see anything both
       halves share: angle errors per point, flux changes between points, drift, or a
       background that rises and falls with the rocking curve (a halo around the grain),
       which pulls tilts toward zero by roughly one to two times its height as a fraction
       of the peak.
    5. **Lit**: ``lit="auto"`` keeps pixels whose integrated intensity is at least
       ``lit_snr`` times its own noise, the noise measured as the MAD of the frames outside
       the peak (no gain needed). Because the window sits on each pixel's own maximum, a
       pixel with no signal still scores above zero: on planted scans ``lit_snr=5`` let in
       1.2-1.5 % of pure-noise pixels and the default 10 let in 0.06-0.2 %, while keeping
       >= 99.8 % of the planted signal. ``"otsu"`` thresholds log intensity instead (only
       sensible with background in the ROI); a number is a minimum integrated intensity; a
       boolean ``(H, W)`` array is used as given.

    Units: a ``tilt`` value is ``1000 * (centre - reference)`` mdeg. A ``strain`` value is
    ``-cot(theta_B) * (centre - reference)`` in microstrain, with the centre taken on
    2theta/2 and ``theta_B = two_theta_deg / 2`` -- a **relative** d-spacing change (the
    refraction offset and any 2theta zero error cancel; one reflection gives Delta d / d, not
    an elastic strain tensor). ``reference`` defaults to the median centre over lit pixels
    whose peak is inside the scan (over all lit pixels if there are none, with a note).

    ``truncated`` flags pixels whose peak window runs off the end of the scan (for
    ``window="full"``: whose half-max region does) -- a peak cut off by the range, but also a
    very wide or double-peaked pixel. Their centres may be biased; the summary excludes them.

    With 3 or fewer points above half maximum (the median over good pixels) a note says the
    widths are quantized by the step and what the error bar does and does not include.

    A mesh (``tilt2d``) uses a simpler estimator: percentile baseline, moments over the
    frames above ``mesh_threshold`` of the pixel's maximum, split-half only with repeats.
    """
    if window not in ("peak", "full", "fixed"):
        raise ValueError("window must be 'peak', 'full' or 'fixed'")
    if window == "fixed" and baseline == "outside_peak":
        baseline = "ends"
    if baseline == "ends" and window != "fixed":
        raise ValueError("baseline='ends' goes with window='fixed'")
    p = dict(baseline=baseline, baseline_percentile=float(baseline_percentile),
             peak_halfwidth=float(peak_halfwidth), min_baseline_points=int(min_baseline_points),
             window=window, split_half=bool(split_half), gain=gain, read_var=float(read_var),
             mesh_threshold=float(mesh_threshold), signal_index=None)
    if baseline != "ends":
        _baseline(np.zeros((3, 1)), np.zeros((3, 1), bool), baseline, 25.0, 1)   # validate name
    scale, unit = _value_scale(scan)
    F = _crop(scan.frames, roi)
    M, H, Wd = F.shape
    mesh = scan.scan_type == "tilt2d"
    order = _order_1d(scan)
    F = F[order]
    halves = None if scan.halves is None else tuple(_crop(h, roi)[order] for h in scan.halves)
    coord = np.asarray(scan.coordinate, float)[order]
    dark = scan.dark_level
    dark = float(dark) if np.ndim(dark) == 0 else _crop(np.asarray(dark, float), roi)
    range_note = None
    if window == "fixed" and not mesh:
        if signal_range is None:
            # the ROI-summed curve above its own floor (its 10th percentile over frames): the
            # per-pixel noise averages out in the sum, so 1 % of the peak is far above it
            field = F.astype(np.float64).sum(axis=(1, 2))
            floor = float(np.percentile(field, 10.0))
            low = field[field <= np.median(field)]
            noise = 1.4826 * float(np.median(np.abs(low - np.median(low)))) if low.size > 3 else 0.0
            thr = max(0.01 * (field.max() - floor), 5.0 * noise)     # whichever is larger
            on = np.flatnonzero(field - floor > thr)
            lo, hi = max(int(on[0]) - 3, 0), min(int(on[-1]) + 3, M - 1)
            range_note = "auto"
        else:
            a, c = sorted(float(v) for v in signal_range)
            lo, hi = int(np.searchsorted(coord, a)), int(np.searchsorted(coord, c, side="right") - 1)
            lo, hi = max(lo, 0), min(max(hi, lo), M - 1)
            range_note = "given"
        p["signal_index"] = (lo, hi)
    parts = []
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        I = F[:, r0:r1].reshape(M, -1).astype(np.float64)
        hv = None if halves is None else tuple(h[:, r0:r1].reshape(M, -1).astype(np.float64)
                                               for h in halves)
        if mesh:
            parts.append(_reduce_mesh_chunk(I, hv, coord, p))
        else:
            dk = dark if np.ndim(dark) == 0 else dark[r0:r1].reshape(-1)
            parts.append(_reduce_1d_chunk(I, hv, coord, p, scan.n_repeats, dk))

    def join(key):
        a = np.concatenate([c[key] for c in parts], axis=0)
        return a.reshape((H, Wd) + a.shape[1:])

    com = join("com")
    intensity = join("intensity")
    snr = join("snr")
    b = join("baseline")
    truncated = join("truncated")
    ok = join("ok")
    total = join("total")

    if isinstance(lit, str):
        if lit == "auto":
            lit_mask = np.nan_to_num(snr, nan=0.0) >= lit_snr
            lit_desc = f"auto (SNR >= {lit_snr:g}, noise measured off-peak)"
        elif lit == "otsu":
            pos = intensity > 0
            thr = _otsu(np.log10(intensity[pos])) if pos.sum() > 1 else np.inf
            lit_mask = pos & (np.log10(np.where(pos, intensity, 1.0)) > thr)
            lit_desc = "otsu (log intensity)"
        else:
            raise ValueError("lit must be 'auto', 'otsu', a number or a boolean mask")
    elif np.ndim(lit) == 0:
        lit_mask = intensity > float(lit)
        lit_desc = f"intensity > {float(lit)}"
    else:
        lit_mask = np.asarray(lit, bool)
        if lit_mask.shape != (H, Wd):
            raise ValueError(f"lit mask shape {lit_mask.shape} != ROI {(H, Wd)}")
        lit_desc = "user mask"
    finite = np.isfinite(com) if com.ndim == 2 else np.all(np.isfinite(com), -1)
    lit_mask = lit_mask & finite
    good = lit_mask & ~truncated

    notes = list(scan.notes)
    ref_pixels = good
    if not good.any():
        notes.append("WARNING: no lit pixel with its peak inside the scan window")
        if lit_mask.any():
            ref_pixels = lit_mask
            notes.append("reference taken over all lit pixels, truncated ones included")
    if mesh:
        ref = ((np.nanmedian(com[ref_pixels], axis=0) if ref_pixels.any() else np.zeros(2))
               if reference is None else np.asarray(reference, float))
    else:
        ref = ((float(np.nanmedian(com[ref_pixels])) if ref_pixels.any() else float("nan"))
               if reference is None else float(reference))
    value = (com - ref) * scale

    sigma = sigma_global = sigma_p = None
    split = None
    if split_half and "diff" in parts[0]:
        d = join("diff") * abs(scale)
        split = "repeat parity" if halves is not None else "angle parity"
        if mesh:
            sigma = np.stack([_box_rms_half(np.where(lit_mask, d[..., a], np.nan))
                              for a in range(d.shape[-1])], -1)
            dd = d[good]
            sigma_global = (float(np.median([1.4826 * np.nanmedian(np.abs(dd[:, a] - np.nanmedian(dd[:, a])))
                                             for a in range(d.shape[-1])]) / 2.0)
                            if good.any() else None)
        else:
            sigma = _box_rms_half(np.where(lit_mask, d, np.nan))
            dd = d[good & np.isfinite(d)]
            sigma_global = (float(1.4826 * np.median(np.abs(dd - np.median(dd))) / 2.0)
                            if dd.size else None)
        if split == "angle parity":
            notes.append("error bar from even vs odd frames: includes sampling error, "
                         "conservative below ~3 points per FWHM")
    if gain is not None and not mesh:
        sigma_p = join("sig_p") * abs(scale)

    lit_counts = total[lit_mask].sum()
    share = float(M * b[lit_mask].sum() / lit_counts) if lit_counts > 0 else float("nan")

    fwhm = None if mesh else join("fwhm") * 1000.0
    n_above = None if mesh else join("n_above")
    if n_above is not None and good.any():
        n_med = float(np.nanmedian(n_above[good]))
        if n_med <= 3:
            notes.append(
                f"coarse sampling: median {n_med:.0f} points above half max, so widths are "
                "quantized by the step (stripes in the width map follow the angle grid) and "
                + ("the repeat-parity error bar does not include the step"
                   if halves is not None else
                   "the even/odd error bar includes the step and may overstate the error"))
    extra = {}
    if not mesh:
        extra["peak_share"] = join("share")
        extra["n_features"] = join("n_feat")
        extra["single_peaked"] = (np.nan_to_num(extra["peak_share"], nan=0.0) >= 0.45) & \
            (extra["n_features"] == 1)
        if lit_mask.any():
            not_single = 1.0 - float(extra["single_peaked"][lit_mask].mean())
            if window == "peak" and not_single > 0.2:
                notes.append(
                    f"WARNING: {not_single:.0%} of lit pixels are not single-peaked; the "
                    "peak-window centre follows whichever feature is tallest and can jump where "
                    "that changes. Look at raw curves and use window='fixed'.")
        if window == "fixed":
            extra["centre_mass_deg"] = join("com_mass")
            extra["centre_shift"] = (extra["centre_mass_deg"] - com) * scale
            extra["span_mdeg"] = join("span") * 1000.0
            notes.append(f"signal window frames {p['signal_index'][0]}-{p['signal_index'][1]} "
                         f"({coord[p['signal_index'][0]]:.4f} to {coord[p['signal_index'][1]]:.4f} deg, "
                         f"{range_note}); baseline a line through the {M - 1 - p['signal_index'][1] + p['signal_index'][0]} frames outside it")
            if not ok.all():
                notes.append(f"{int((~ok).sum())} px: too few frames outside the signal window "
                             "for a line baseline, percentile baseline used")
        if halves is not None and lit_mask.sum() >= 50:
            ptc = _repeat_excess(F, halves)
            sel = lit_mask & np.isfinite(ptc)
            if sel.sum() >= 50:
                v_ = np.sort(ptc[sel])
                ref = float(v_[int(0.05 * v_.size):max(int(0.95 * v_.size), 1)].mean())  # trimmed mean
                if ref > 0:
                    rel = ptc / ref
                    extra["repeat_excess"] = np.where(lit_mask, rel, np.nan)
                    lo_, hi_ = np.percentile(rel[sel], [10, 90])
                    absx = None
                    if gain is not None and scan.n_repeats >= 2:
                        nA = (scan.n_repeats + 1) // 2
                        nB = scan.n_repeats - nA
                        absx = float(ref * gain / (1.0 / nA + 1.0 / max(nB, 1)))
                    extra["repeat_excess_spread"] = (float(lo_), float(hi_), absx)
                    if hi_ / max(lo_, 1e-9) > 3.0:
                        notes.append(
                            "frames at one position differ by more than photon noise in part of "
                            "the field (repeat_excess map): flux, beam or sample motion between "
                            "frames. Averaging blurs whatever moves; the repeat-parity error bar "
                            "includes it, a photon-only error bar would not.")
        if halves is not None and lit_mask.sum() >= 100:
            shifts = _repeat_shift(F, halves, lit_mask, coord, b)
            fld = F.astype(np.float64).sum(axis=(1, 2))
            flo = float(np.percentile(fld, 10.0))
            on = fld - flo > 0.1 * (fld.max() - flo)
            on = on if on.sum() >= 3 else np.ones(M, bool)
            w = np.clip(fld - flo, 0.0, None) * on            # frames weighted by their signal
            w = w / w.sum() if w.sum() > 0 else on / on.sum()
            rms = tuple(float(np.sqrt(np.nansum(w * np.nan_to_num(shifts[:, k]) ** 2))) for k in (4, 5, 2, 3))
            extra["repeat_shift_px"] = shifts[:, 2:]
            extra["repeat_shift_rms"] = rms + (int(on.sum()),)
            if max(rms[:2]) > 0.5 or max(rms[2:]) > 0.3:
                notes.append(
                    f"between the two repeat halves of one position the intensity envelope shifts by "
                    f"{rms[0]:.2f} px (rows) / {rms[1]:.2f} px (cols) rms and the fine detail by "
                    f"{rms[2]:.2f} / {rms[3]:.2f} px (repeat_shift_px): frames at one position are not "
                    "copies of each other. Averaging blurs by that much; the repeat-parity error bar "
                    "includes it.")
    import midas_dfxm as _dx
    settings = dict(p, lit=lit_desc, lit_snr=lit_snr, roi=roi, load_roi=scan.meta.get("roi"),
                    reference=reference, chunk_rows=chunk_rows, n_repeats=scan.n_repeats,
                    source=scan.source, midas_dfxm=getattr(_dx, "__version__", "?"),
                    signal_range=(None if p["signal_index"] is None else
                                  (float(coord[p["signal_index"][0]]), float(coord[p["signal_index"][1]]))))
    return RockingMaps(scan_type=scan.scan_type, axes=scan.axes, unit=unit, value=value,
                       centre_deg=com, reference_deg=ref, intensity=intensity, snr=snr,
                       baseline=b, lit=lit_mask, truncated=truncated, baseline_ok=ok,
                       fwhm_mdeg=fwhm, points_above_half=n_above, sigma=sigma,
                       sigma_global=sigma_global, sigma_poisson=sigma_p,
                       pedestal_share=share, split=split, settings=settings, notes=notes,
                       **extra)


def baseline_sensitivity(scan: RockingScan, *, roi=None, variants=None, block: int = 8,
                         **kwargs) -> list:
    """How far defensible baseline choices move the map -- measured where noise averages out.

    Reduces the scan with the default settings (``kwargs`` override them) and with each
    variant, over the pixels lit and untruncated in the DEFAULT reduction (selecting on a
    variant's own flags would favour that variant). Each map is referenced to its own median
    there, so a uniform offset does not count.

    For each variant it returns:

    ``block_rms``, ``block_slope``
        The RMS difference and the regression slope between means of ``block`` x ``block``
        pixel blocks lying entirely inside the selection. **This is the systematic to quote**
        beside the error bar; a slope far from 1 means the choice rescales every tilt.
    ``pixel_spread``
        1.4826 x MAD of the per-pixel difference. It mixes estimator noise with the
        systematic, so it bounds the systematic per pixel from above; it does not estimate it.
    ``n_pixels``, ``n_blocks``, ``variant``, ``unit``

    Why blocks: on the NaMnO2 practice scans the per-pixel RMS difference between variants
    came out 2-11x the split-half error bar, driven by SNR 5-10 pixels and a few outliers,
    while 8x8 block means moved by at most 0.7 mdeg with slopes 0.99-1.01. A small block
    difference says the variants agree with each other; it does not show that either is
    free of a bias they share.
    """
    scale, unit = _value_scale(scan)
    first = (lambda v: v if v.ndim == 2 else v[..., 0])
    base = reduce_rocking(scan, roi=roi, split_half=False, **kwargs)
    if variants is None:
        if kwargs.get("window") == "fixed":
            lo, hi = base.settings["signal_range"]
            x = np.sort(np.asarray(scan.coordinate, float).reshape(len(scan.coordinate), -1)[:, 0])
            step = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
            variants = [dict(signal_range=(lo - 3 * step, hi + 3 * step)),
                        dict(signal_range=(lo + 3 * step, hi - 3 * step)),
                        dict(baseline="percentile", baseline_percentile=25.0),
                        dict(baseline="percentile", baseline_percentile=50.0)]
        else:
            variants = [dict(baseline="outside_peak", peak_halfwidth=1.5),
                        dict(baseline="outside_peak", peak_halfwidth=3.0),
                        dict(baseline="percentile", baseline_percentile=25.0),
                        dict(baseline="percentile", baseline_percentile=50.0)]
    sel = base.lit & ~base.truncated
    bc = first(base.centre_deg) * scale
    rows = []
    for var in variants:
        m = reduce_rocking(scan, roi=roi, split_half=False, **{**kwargs, **var})
        mc = first(m.centre_deg) * scale
        both = sel & np.isfinite(bc) & np.isfinite(mc)
        row = dict(variant=var, n_pixels=int(both.sum()), n_blocks=0, block_rms=float("nan"),
                   block_slope=float("nan"), pixel_spread=float("nan"), unit=unit)
        if both.sum() >= 3:
            a = np.where(both, bc - np.median(bc[both]), 0.0)
            c = np.where(both, mc - np.median(mc[both]), 0.0)
            d = (c - a)[both]
            row["pixel_spread"] = float(1.4826 * np.median(np.abs(d - np.median(d))))
            H, W = both.shape
            h, w = H // block, W // block
            if h and w:
                shp = (h, block, w, block)
                full = both[:h * block, :w * block].reshape(shp).all(axis=(1, 3))
                if full.sum() >= 3:
                    ba = a[:h * block, :w * block].reshape(shp).mean(axis=(1, 3))[full]
                    bb = c[:h * block, :w * block].reshape(shp).mean(axis=(1, 3))[full]
                    ba = ba - np.median(ba)
                    bb = bb - np.median(bb)
                    den = float(np.dot(ba, ba))
                    row.update(n_blocks=int(full.sum()),
                               block_rms=float(np.sqrt(np.mean((bb - ba) ** 2))),
                               block_slope=float(np.dot(ba, bb) / den) if den > 0 else float("nan"))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- synthetic scan
def _erf(z):
    """Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7); keeps scipy out of the dependencies."""
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return np.sign(z) * y


def _shift2d(img, dy, dx):
    """Bilinear shift of a 2-D image by (dy, dx) pixels, edges repeated."""
    H, W = img.shape
    r = np.clip(np.arange(H) - dy, 0, H - 1)
    c = np.clip(np.arange(W) - dx, 0, W - 1)
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, H - 1); c1 = np.minimum(c0 + 1, W - 1)
    fr = (r - r0)[:, None]; fc = (c - c0)[None]
    return ((1 - fr) * (1 - fc) * img[r0][:, c0] + (1 - fr) * fc * img[r0][:, c1]
            + fr * (1 - fc) * img[r1][:, c0] + fr * fc * img[r1][:, c1])


def example_rocking_scan(kind: str = "broad", *, shape=(64, 64), n_points: int = 101,
                         step_mdeg: float = 1.0, n_repeats: int = 4, pedestal: float = 100.0,
                         amplitude: float = 80.0, gain: float = 1.0, motion_px: float = 0.0,
                         seed: int = 0, two_theta: bool = False) -> RockingScan:
    """A synthetic theta rock with a known answer, for the tutorial and the tests.

    ``two_theta=True`` logs 2theta moving with theta (a theta-2theta scan), so the scan
    classifies as ``strain`` and the maps come out in microstrain; the planted structure is
    the same, in the rocking coordinate 2theta/2.

    ``kind="single"``: one 5 mdeg peak per pixel whose centre follows a smooth 24 mdeg ramp
    across the field. Every reduction should recover the ramp.

    ``kind="broad"``: what a Ba122 crystal gave at 6-ID-C. Every pixel's curve is a 30 mdeg
    box with a sharp horn at each end and NO tilt anywhere; only the horn heights change,
    smoothly, across a line down the middle of the field. A peak-window centre follows the
    taller horn and steps by ~26 mdeg across that line; the median of the curve does not.

    ``kind="step"``: the same box, plus a real 20 mdeg tilt step across the line (left half
    -12, right half +8). Both centres must show it.

    Frames are Poisson at ``gain`` electrons per count over a ``pedestal``, ``n_repeats`` per
    point, averaged; the repeat halves are kept for the split-half error bar. ``motion_px``
    plants a rigid random shift of the image per frame (rms, in pixels), to show what frames
    that are not copies of each other do. ``scan.meta["truth"]`` holds the planted tilt map
    (mdeg) and the planted horn ratio.
    """
    if kind not in ("single", "broad", "step"):
        raise ValueError("kind must be 'single', 'broad' or 'step'")
    rng = np.random.default_rng(seed)
    H, W = shape
    x0 = 7.300
    x = x0 + 1e-3 * step_mdeg * np.arange(n_points)                     # deg
    rr, cc = np.mgrid[0:H, 0:W]
    u = (cc - (W - 1) / 2) / (W / 2)                                      # -1 .. 1 across
    v = (rr - (H - 1) / 2) / (H / 2)
    xm = x[:, None, None]
    if kind == "single":
        tilt = 12.0 * u + 4.0 * v                                          # mdeg, smooth
        cen = x0 + 1e-3 * (38.0 + tilt)
        sig = 1e-3 * 5.0 / 2.3548
        signal = np.exp(-0.5 * ((xm - cen[None]) / sig) ** 2)
        ratio = np.ones((H, W))
    else:
        tilt = np.zeros((H, W)) if kind == "broad" else np.where(u < 0, -12.0, 8.0)
        e1 = x0 + 1e-3 * (37.5 + tilt)
        e2 = x0 + 1e-3 * (67.5 + tilt)
        t = u + 0.3 * v
        hl = 2.5 * (0.5 - 0.5 * np.tanh(3 * t))                            # left horn height
        hr = 2.5 * (0.5 + 0.5 * np.tanh(3 * t))                            # right horn height
        ratio = hl / hr
        box = 0.5 * (_erf((xm - e1[None]) / (1e-3 * math.sqrt(2))) - _erf((xm - e2[None]) / (1e-3 * math.sqrt(2))))
        horns = (hl[None] * np.exp(-0.5 * ((xm - e1[None] - 1e-3) / 1e-3) ** 2)
                 + hr[None] * np.exp(-0.5 * ((xm - e2[None] + 1e-3) / 1e-3) ** 2))
        signal = box + horns
    beam = np.exp(-0.5 * (u / 0.7) ** 2 - 0.5 * (v / 0.7) ** 2)          # illumination falls off
    clean = pedestal + amplitude * beam[None] * signal
    frames = np.zeros((n_points, H, W)); A = np.zeros_like(frames); B = np.zeros_like(frames)
    nA = 0; nB = 0
    for r in range(n_repeats):
        img = clean
        if motion_px > 0:
            img = np.stack([_shift2d(clean[m] - pedestal, *rng.normal(0, motion_px, 2)) + pedestal
                            for m in range(n_points)])
        f = rng.poisson(np.clip(img, 0, None) * gain) / gain
        frames += f
        if r % 2 == 0:
            A += f; nA += 1
        else:
            B += f; nB += 1
    frames /= n_repeats
    halves = (A / nA, B / nB) if nB else None
    tth = 2.0 * x if two_theta else np.full(n_points, 2 * 8.32)
    motors = {"th": x, "tth": tth, "Num": np.arange(n_points)}
    return RockingScan.from_arrays(frames.astype(np.float32), motors, halves=halves,
                                   n_repeats=n_repeats, source=f"example_rocking_scan({kind!r})",
                                   meta={"truth": {"tilt_mdeg": tilt, "horn_ratio": ratio,
                                                   "kind": kind, "gain": gain,
                                                   "motion_px": motion_px}})
