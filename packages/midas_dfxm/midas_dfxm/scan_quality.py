"""Pre-reduction quality control for a measured DFXM rocking scan.

A raw APS 6-ID-C rocking scan is a table of scan POINTS (motor positions along the rocking
curve), each acquired as several REPEAT exposures at the same nominal position (the repeat
count varies scan to scan -- 2, 10, 20 have all been seen). Nothing upstream of this module
checks, before reduction, whether the acquisition itself is usable: a table-length mismatch,
saturated pixels, a dead or duplicated frame, or a photon-starved exposure time all currently
surface only as a mysterious downstream failure or a quietly wrong map. Concretely: scan S224
(``DFXM_S224``, 0.05 s exposure) wasted a full jitter-estimator run before that estimator's own
internal flux gate finally caught that its signal was unusable (see
``NOTE_S224_frames.md``: raw per-pixel z-score maxes out around 3 where a healthy scan reaches
15-20, and the gate fails to recover a known +3% synthetic flux perturbation on 3 of 4 probed
points). This module is meant to catch exactly that, in seconds, before anything downstream
runs. Three further real-data bugs, found running this module on 6-ID-C scan S168 (61 points x 2
repeats; see ``NOTE_S168_frames.md``), are fixed the same way -- by looking at the raw frames, not
by tuning a threshold until the symptom disappeared:

1. **Static hot pixels were misclassified as per-frame cosmic-ray spikes.** S168's every frame
   tripped :func:`_check_spike` at hundreds of the SAME pixel locations -- a signature of fixed
   detector defects, not per-frame events (a real cosmic ray lands at a different, ~random
   location each frame; the chance of one hitting the exact same pixel more than once in a scan of
   hundreds of frames is negligible). :func:`_static_hot_pixel_mask` finds pixels that trip the
   spike criterion at the SAME location far more often than chance would allow, GIVEN the scan's
   own observed candidate rate (not a fixed fraction of frames -- a flat "majority of frames" bar
   was tried first and was not strict enough on real data, see that function's docstring), and
   excludes them from the per-frame check from then on, reporting them once as a scan-level
   static-defect map instead of a per-frame flag each.
2. **The Poisson-noise floors assumed gain=1 (counts per photon), which is wrong for this
   detector** (a real photon-transfer measurement on this beamtime gives var ~ 1.1-1.3*mean +
   19-29, i.e. gain != 1 and a real read-noise floor). :func:`estimate_gain` measures both
   directly from the scan's own repeat-to-repeat variance-vs-mean relationship (a standard
   photon-transfer-curve fit) and :func:`assess_scan_quality` uses it by default instead of
   silently assuming 1.0.
3. **A whole-point brightness anomaly that both repeats agree on was invisible**: S168 point 32
   has its entire frame elevated (lit-region total +55% over its neighbours, see
   ``NOTE_S168_frames.md``'s independently-measured +35%), but both repeats agree with each other,
   so there is no WITHIN-point disagreement for :func:`_check_count_anomaly` to catch. Two prior
   attempts to build a point-vs-neighbours check on the lit-region signal itself were retracted
   because a real rocking curve legitimately rises and falls fast near its peak and a narrow local
   fit aliases that shape as an anomaly (see :func:`_check_pedestal_drift`'s docstring for exactly
   how they failed). The fix instead uses the OFF-SAMPLE background -- pixels never part of the
   lit/diffracting region at any point in the scan -- which cannot track the rocking curve's shape
   at all, so a local robust comparison across neighbouring points is curve-shape-safe by
   construction.

**Policy: flag with reasons, never silently exclude.** Every check here produces a reasoned,
inspectable :class:`QualityCheck` (what was measured, against what threshold, pass or flag)
attached to the specific frame or point it concerns. Nothing in :func:`assess_scan_quality` drops
a frame, reweights an average, or otherwise changes the data -- it only reports. Acting on a
report (building a repeat-filtered scan for downstream reduction) is the separate, explicitly
opt-in :func:`apply_quality_filter`.

**A critical fact this module is built around: individual repeat exposures are not retained by**
:class:`midas_dfxm.rocking.RockingScan`. Its own docstring says so directly -- ``frames`` is
"one frame per scan point, ... repeats already averaged" -- and reading the loader
(:mod:`midas_dfxm.io_6idc`, the ``indexed`` layout) confirms it: repeats are accumulated into a
running sum and only the average (plus, optionally, a 2-way even/odd "repeat parity" split used
for the reduction's error bar) ever reaches a ``RockingScan``. Frame-level checks (saturation,
per-repeat count anomalies, cosmic-ray-like spikes, a dead or duplicated frame) are about
individual exposures, so this module reads the raw per-repeat frames itself. It never touches
:mod:`midas_dfxm.rocking` or :mod:`midas_dfxm.io_6idc` internals to do it (both are being edited
elsewhere concurrently and are read-only dependencies here): :class:`RawRepeatScan` is a new,
standalone, ``(n_points, n_repeats, H, W)`` container, with its own small loader for the 2025-era
"indexed" 6-ID-C layout (``data_NNNNN.tif`` + a per-point motor CSV -- the layout every real
example in ``datasetJ_dfxm_dec2025`` uses). It generalises the idea behind
:func:`midas_dfxm.io_6idc.load_6idc_scan`'s ``drop_first_repeat``: that option compares repeat 0
against the median of the OTHER repeats, for the WHOLE SCAN, and only ever drops repeat 0. Here,
every repeat of every point is compared against its OWN sibling repeats at that SAME point, with
a robust (MAD-based) statistic, for any of the checks that need it -- and nothing is ever
dropped automatically.

Three tiers, each built from the one below it:

:class:`FrameQualityResult` (one repeat exposure)
    Saturation; an anomalous total/lit-region count relative to the point's own other repeats;
    a localised spike (cosmic ray / detector glitch) distinct from a smooth PSF peak; a dead,
    frozen or duplicated frame.

:class:`PointQualityResult` (one scan position, all its repeats)
    How many repeats survive frame-level flagging; a point-level flux-health check (the point's
    own lit-region SNR -- only fires when NOTHING is lit; it does NOT catch S224, see below); a
    whole-frame brightness anomaly from the off-sample pedestal against neighbouring points
    (:func:`_check_pedestal_drift`, the S168 point-32 catch). Carries the point's actual motor
    coordinate(s).

:class:`ScanQualityReport` (the whole scan)
    Frame-count vs motor-table row-count consistency (also catches a restarted-scan CSV with a
    duplicated header block silently doubling the row count); scan-axis step regularity; the
    static-hot-pixel map; whether the scan brackets a rocking-curve peak
    (:func:`_check_curve_bracketing` -- what S224's problem turned out to be); and a rollup
    verdict built from POINT-level failures (a hot first repeat at every point is excludable,
    not "not usable").

**What real data changed (datasetJ S168/S995/S996/S1050/S224; LAB_NOTEBOOK 12d has the numbers).**
The first version called every point of every real scan unusable. Static hot pixels (0.1-0.9 %
of a frame) must be masked before the spike test; most remaining "spikes" are stable sample
texture, excluded when their MEDIAN sibling-repeat z exceeds ``spike_zscore/4``, and a frame is
flagged on impact (``max_spike_fraction``), not on any single hit; the hot first repeat is real
and systematic; ``pedestal_drift`` needs a 4-point window and a 0.5 % effect size; and S224 is
an unbracketed peak, not something the point-level SNR or the (unstable) auto-gain can see.

If handed a plain :class:`midas_dfxm.rocking.RockingScan` (no raw repeats available),
:func:`assess_scan_quality` still runs in a documented DEGRADED mode: true frame-level checks are
skipped (there is nothing to compare -- said explicitly in the report), but a coarser two-way
comparison is still possible when ``scan.halves`` exists (the even/odd repeat-parity split), and
the point-level flux check and every scan-level check still run in full, since they only need the
per-point average.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .io_6idc import find_motor_tables, read_motor_table
from .rocking import RockingScan, classify_motors

__all__ = [
    "RawRepeatScan",
    "QualityCheck",
    "FrameQualityResult",
    "PointQualityResult",
    "ScanQualityReport",
    "load_6idc_repeat_frames",
    "estimate_gain",
    "assess_scan_quality",
    "apply_quality_filter",
]

_INDEXED = re.compile(r"^data_(\d+)\.tiff?$", re.IGNORECASE)
FIXED_TOL_DEG = 1e-4

# 1.4826*MAD is only an UNBIASED estimate of sigma asymptotically (large n); at the few-to-tens
# of repeats this module actually sees (2-20), it systematically UNDERESTIMATES sigma (e.g.
# ratio-to-true-sigma ~0.84 at n=2, ~0.89 at n=8, ~0.96 at n=20). :func:`estimate_gain` needs an
# absolute (not just relative/threshold) variance estimate, so this bias would otherwise show up
# directly as a systematically LOW gain estimate. The table below is from a 60000-trial Monte
# Carlo calibration against a standard normal (see the development notes for the exact script);
# values are 1/ratio, i.e. the multiplicative correction applied to 1.4826*MAD before squaring it
# into a variance. Not used anywhere else in this module: every OTHER MAD-based comparison here is
# a relative z-score against a threshold that was itself calibrated using the same (uncorrected)
# convention, so correcting it there would just require re-deriving those thresholds for no
# benefit.
_MAD_UNBIAS_N = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 100])
_MAD_UNBIAS_C = np.array([1.1922, 1.4911, 1.3587, 1.2193, 1.1903, 1.1424, 1.1263, 1.0996, 1.0949,
                          1.0761, 1.0549, 1.0431, 1.0278, 1.0157, 1.0082])


def _mad_unbias_factor(n):
    """Small-sample correction for 1.4826*MAD as an estimator of sigma, at ``n`` samples (see
    ``_MAD_UNBIAS_N``/``_MAD_UNBIAS_C`` above). Linearly interpolated between tabulated points;
    clamped to the table's own range (correction -> 1.0 well before n=100)."""
    n = max(int(n), 2)
    if n >= _MAD_UNBIAS_N[-1]:
        return 1.0
    return float(np.interp(n, _MAD_UNBIAS_N, _MAD_UNBIAS_C))


def _poisson_min_recurrence(mu, significance, max_k=200):
    """Smallest ``k >= 1`` such that ``P(X >= k) < significance`` for ``X ~ Poisson(mu)`` --
    used by :func:`_static_hot_pixel_mask` to turn "a pixel that trips the per-frame spike test
    at the SAME location ``k`` or more times" into a statement with a stated false-positive rate,
    given the scan's OWN observed candidate rate, rather than an arbitrary fixed fraction of
    frames (see that function's docstring for why a flat "majority of frames" cutoff turned out
    to systematically under-catch real static defects on 6-ID-C data: this detector's spike
    candidates are not simply "always hot" or "never hot" -- real S168 data shows a population
    that fires in anywhere from 1 to ~15 of 122 frames, ALL of it statistically incompatible with
    a cosmic ray landing on the same pixel by chance, well before reaching a 50%-of-frames rate)."""
    mu = max(float(mu), 0.0)
    if mu <= 0.0:
        return 1
    pmf = math.exp(-mu)
    cdf = pmf
    for k in range(1, max_k + 1):
        if 1.0 - cdf < significance:
            return k
        pmf *= mu / k
        cdf += pmf
    return max_k


# --------------------------------------------------------------------------- raw repeat scan
@dataclass
class RawRepeatScan:
    """Individual repeat exposures of a rocking scan, NOT averaged: ``(n_points, R, H, W)``.

    This is what :class:`midas_dfxm.rocking.RockingScan` does not keep (see module docstring).
    Build one with :meth:`from_arrays`, or read real 6-ID-C frames with
    :func:`load_6idc_repeat_frames`.

    Attributes
    ----------
    frames : (n_points, n_repeats, H, W) float32
        Repeat exposures in acquisition order, dark-subtracted if a dark was given, otherwise
        raw. Not averaged.
    motors : dict of (n_points,) float arrays
        One value per scan POINT (not per frame), exactly as in :class:`RockingScan`.
    scan_type, axes, two_theta_deg
        From :func:`midas_dfxm.rocking.classify_motors`, same meaning as on ``RockingScan``.
    dark_level, source, notes, meta
        Provenance, same meaning as on ``RockingScan``.
    """

    frames: np.ndarray
    motors: dict
    scan_type: str
    axes: tuple
    two_theta_deg: Optional[float] = None
    dark_level: object = 0.0
    source: str = ""
    notes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_arrays(cls, frames, motors: dict, *, dark_level=0.0, source: str = "arrays",
                    notes=None, meta=None, fixed_tol_deg: float = FIXED_TOL_DEG) -> "RawRepeatScan":
        """Build a scan from a ``(n_points, R, H, W)`` stack and a ``{motor: (n_points,)}`` table."""
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4:
            raise ValueError(f"frames must be (n_points, n_repeats, H, W); got shape {frames.shape}")
        n_points = frames.shape[0]
        motors = {str(k): np.asarray(v, dtype=float).reshape(-1) for k, v in motors.items()}
        wrong = [k for k, v in motors.items() if v.size != n_points]
        if wrong:
            raise ValueError(f"motor columns {wrong} do not have one value per POINT ({n_points}, "
                             f"not per frame -- frames has {frames.shape[1]} repeats/point). "
                             "Refusing to guess the point/angle pairing.")
        info = classify_motors(motors, fixed_tol_deg=fixed_tol_deg)
        tth = None
        if info["two_theta_key"] is not None:
            f = np.asarray(motors[info["two_theta_key"]], float)
            f = f[np.isfinite(f)]
            tth = float(np.median(f)) if f.size else None
        return cls(frames=frames, motors=motors, scan_type=info["scan_type"],
                   axes=tuple(info["axes"]), two_theta_deg=tth, dark_level=dark_level,
                   source=source, notes=list(notes or []) + info["notes"],
                   meta={**(meta or {}), "moving": info["moving"]})

    @property
    def n_points(self) -> int:
        return self.frames.shape[0]

    @property
    def n_repeats(self) -> int:
        return self.frames.shape[1]

    @property
    def coordinate(self) -> np.ndarray:
        """Rocking coordinate per point, degrees -- see :attr:`RockingScan.coordinate`."""
        if self.scan_type == "strain":
            return 0.5 * self.motors[self.axes[0]]
        if self.scan_type == "tilt":
            return self.motors[self.axes[0]]
        return np.stack([self.motors[a] for a in self.axes], -1)

    def point_label(self, p: int) -> str:
        """``"th=15.7232"`` or ``"th=15.7232, chi=0.1000"`` -- the point's own motor reading(s),
        for reporting a flagged point as "re-measure <label>"."""
        return ", ".join(f"{a}={self.motors[a][p]:.4f}" for a in self.axes) or f"point {p}"

    def to_rocking_scan(self, repeat_mask: Optional[np.ndarray] = None) -> RockingScan:
        """Average repeats into a :class:`RockingScan`, honouring an optional ``(n_points, R)``
        boolean keep-mask (default: keep everything). Surviving repeats at each point are split
        even/odd for a photon-noise split-half error bar, the same convention
        :mod:`midas_dfxm.io_6idc` uses. A point with EVERY repeat excluded raises -- the caller
        (:func:`apply_quality_filter`) decides what to do about that before this is called."""
        n_points, R = self.frames.shape[:2]
        if repeat_mask is None:
            repeat_mask = np.ones((n_points, R), dtype=bool)
        repeat_mask = np.asarray(repeat_mask, dtype=bool)
        if repeat_mask.shape != (n_points, R):
            raise ValueError(f"repeat_mask must be {(n_points, R)}; got {repeat_mask.shape}")
        counts = repeat_mask.sum(1)
        if (counts == 0).any():
            raise ValueError("repeat_mask excludes every repeat at point(s) "
                             f"{np.flatnonzero(counts == 0).tolist()}; nothing left to average there")
        H, W = self.frames.shape[2:]
        avg = np.empty((n_points, H, W), dtype=np.float32)
        even = np.zeros((n_points, H, W), dtype=np.float64)
        odd = np.zeros((n_points, H, W), dtype=np.float64)
        n_even = np.zeros(n_points, int)
        n_odd = np.zeros(n_points, int)
        for p in range(n_points):
            idx = np.flatnonzero(repeat_mask[p])
            avg[p] = self.frames[p, idx].mean(0)
            for j, r in enumerate(idx):
                if j % 2 == 0:
                    even[p] += self.frames[p, r]
                    n_even[p] += 1
                else:
                    odd[p] += self.frames[p, r]
                    n_odd[p] += 1
        halves = None
        if (n_even >= 1).all() and (n_odd >= 1).all():
            halves = ((even / n_even[:, None, None]).astype(np.float32),
                      (odd / n_odd[:, None, None]).astype(np.float32))
        return RockingScan.from_arrays(
            avg, self.motors, halves=halves, n_repeats=int(np.median(counts)),
            dark_level=self.dark_level, source=self.source, notes=list(self.notes),
            meta=dict(self.meta))


def _tiff_reader():
    try:
        import tifffile
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("reading TIFF frames needs `tifffile` (pip install tifffile)") from e
    return tifffile.imread


def _crop(a, roi):
    if roi is None:
        return a
    r0, r1, c0, c1 = roi
    return a[..., r0:r1, c0:c1]


def load_6idc_repeat_frames(scan_dir: str, motor_table: Optional[str] = None, *, roi=None,
                            dark=None, frames_per_point: Optional[int] = None) -> RawRepeatScan:
    """Read one 6-ID-C ``indexed`` (2025-, ``data_NNNNN.tif``) scan WITHOUT averaging repeats.

    This reads exactly the layout :func:`midas_dfxm.io_6idc.load_6idc_scan` calls ``"indexed"``
    -- point-major ``data_NNNNN.tif`` frames plus a per-point motor CSV, ``R = n_frames /
    n_points`` -- with the same refusals for an incomplete transfer (a zero-byte frame, a gap in
    the frame index, a frame count that is not a whole number of repeats). It does not read the
    older ``named`` (2021-2023) layout: that layout stores one frame per POINT already, with no
    repeats to compare frame-by-frame, so the frame-level checks in this module do not apply to
    it -- load it with :func:`midas_dfxm.io_6idc.load_6idc_scan` and reduce it directly, or wrap
    its ``(M, H, W)`` frames as ``RawRepeatScan.from_arrays(frames[:, None], scan.motors)`` if a
    scan/point-level-only quality report is still wanted (repeat count 1 everywhere).

    Parameters
    ----------
    scan_dir : folder holding ``data_NNNNN.tif`` frames.
    motor_table : the scan's motor CSV; auto-found with :func:`midas_dfxm.io_6idc.find_motor_tables`
        if omitted (and the load stops unless exactly one candidate is found).
    roi : ``(row0, row1, col0, col1)`` crop applied as frames are read. Strongly recommended: a
        full Zyla frame is 2560 x 2160 px, and every repeat is kept here (unlike
        ``load_6idc_scan``, which only ever keeps one averaged frame per point), so an uncropped
        20-repeat scan needs ~20x the memory of the equivalent reduced scan.
    dark : ``None``, a number, a 2-D array, or a single dark-frame TIFF path, subtracted from
        every frame. (Unlike ``load_6idc_scan``, a folder of dark frames to average is not
        supported here -- pass a single already-averaged dark array or file.)
    frames_per_point : override the repeat count instead of inferring it from
        ``n_frames // n_points``.

    Returns
    -------
    RawRepeatScan
    """
    reader = _tiff_reader()
    scan_dir = os.path.abspath(scan_dir)
    if not os.path.isdir(scan_dir):
        raise FileNotFoundError(scan_dir)
    names = [f for f in os.listdir(scan_dir) if _INDEXED.match(f)]
    if not names:
        raise FileNotFoundError(
            f"no data_NNNNN.tif frames in {scan_dir}. load_6idc_repeat_frames only reads the "
            "indexed (2025-) 6-ID-C layout, which has repeats to compare frame-by-frame; this "
            "looks like the older named layout (2021-2023), which has one frame per point "
            "already -- see this function's docstring for how to still get a scan/point-level "
            "quality report there.")
    notes = []
    if motor_table is None:
        found = find_motor_tables(scan_dir)
        if len(found) != 1:
            raise ValueError(
                f"found {len(found)} motor tables for {os.path.basename(scan_dir)}: {found}. "
                + ("Pass motor_table= explicitly -- and pick the one from the SAME beamtime as "
                   "the frames." if found else "Pass motor_table= explicitly."))
        motor_table = found[0]
        notes.append(f"motor table found automatically: {motor_table}")
    table = read_motor_table(motor_table)
    index = {int(_INDEXED.match(f).group(1)): os.path.join(scan_dir, f) for f in names}
    idx = sorted(index)
    first = idx[0]
    missing = sorted(set(range(first, first + len(idx))) - set(idx))
    if missing or idx[-1] != first + len(idx) - 1:
        gaps = sorted(set(range(first, idx[-1] + 1)) - set(idx))
        raise ValueError(f"{scan_dir}: frame indices are not contiguous ({len(gaps)} missing, "
                         f"e.g. {gaps[:10]}). Files were lost in transfer; re-transfer the scan.")
    paths = [index[i] for i in idx]
    empty = [p for p in paths if os.path.getsize(p) == 0]
    if empty:
        raise ValueError(f"{len(empty)} zero-byte frame file(s), e.g. {empty[:3]}. The scan is "
                         "incomplete; re-transfer it rather than reading around the gap.")
    n_points = len(next(iter(table.values())))
    n = len(paths)
    if frames_per_point is None:
        if n % n_points:
            raise ValueError(
                f"{n} frames for {n_points} motor-table points is not a whole number of "
                "repeats: frames are missing, or the table belongs to another scan. "
                "n_frames // n_points would read the wrong frames without an error.")
        R = n // n_points
    else:
        R = int(frames_per_point)
        if R * n_points != n:
            raise ValueError(f"frames_per_point={R} x {n_points} points != {n} frames")
    dark_arr = 0.0
    if dark is not None:
        if isinstance(dark, (int, float, np.floating, np.integer)):
            dark_arr = float(dark)
        elif isinstance(dark, str):
            dark_arr = _crop(np.asarray(reader(dark), dtype=np.float64), roi).astype(np.float32)
        else:
            dark_arr = _crop(np.asarray(dark, dtype=np.float64), roi).astype(np.float32)
    shape = _crop(np.asarray(reader(paths[0])), roi).shape
    frames = np.empty((n_points, R) + shape, dtype=np.float32)
    for pt in range(n_points):
        for r in range(R):
            a = _crop(np.asarray(reader(paths[pt * R + r]), dtype=np.float32), roi)
            frames[pt, r] = a - dark_arr
    notes.append(f"{n} frames = {n_points} points x {R} repeats, read point-major; individual "
                 "repeats kept (not averaged) for frame-level quality checks")
    meta = {"scan_dir": scan_dir, "motor_table": os.path.abspath(motor_table), "roi": roi,
            "repeats_per_point": R}
    dark_level = dark_arr if np.ndim(dark_arr) else float(dark_arr)
    return RawRepeatScan.from_arrays(frames, table, dark_level=dark_level,
                                     source=f"{scan_dir} + {os.path.basename(motor_table)}",
                                     notes=notes, meta=meta)


# --------------------------------------------------------------------------- check result types
@dataclass
class QualityCheck:
    """One reasoned, inspectable check. ``flagged=False`` is a pass.

    ``value`` and ``threshold`` are on the same scale (e.g. both a z-score, or both a pixel
    count) so a caller or a human can see by how much a check passed or failed, not just the
    boolean. ``message`` restates that in words.
    """

    name: str
    flagged: bool
    value: float
    threshold: float
    message: str

    def __bool__(self) -> bool:
        return not self.flagged

    def __str__(self) -> str:
        return self.message


@dataclass
class FrameQualityResult:
    """Every check run on ONE repeat exposure, at a given (point, repeat)."""

    point: int
    repeat: int
    checks: list

    @property
    def flagged(self) -> bool:
        return any(c.flagged for c in self.checks)

    @property
    def flagged_checks(self) -> list:
        return [c for c in self.checks if c.flagged]

    def __repr__(self) -> str:
        state = "FLAGGED" if self.flagged else "ok"
        return f"FrameQualityResult(point={self.point}, repeat={self.repeat}, {state})"


@dataclass
class PointQualityResult:
    """Every check run on ONE scan point: its repeats' :class:`FrameQualityResult` list plus
    point-level checks. ``coordinate`` carries the point's actual motor reading(s) so a flagged
    point can be reported as "re-measure th=X.XXX[, chi=Y.YYY]" (``coordinate_label``)."""

    point: int
    coordinate: dict
    coordinate_label: str
    frames: list
    n_repeats: int
    n_surviving: int
    checks: list
    granularity: str        # "repeats" (true individual exposures), "halves" (2-way split
                             # only, degraded RockingScan input) or "average-only" (no
                             # repeat-level information at all)
    off_sample_mean: float = float("nan")   # this point's mean level over pixels never part of
                             # the lit region at ANY point in the scan (see _off_sample_mask /
                             # _check_pedestal_drift); NaN if no off-sample pixels were found.
    off_sample_sigma_floor: float = 0.0     # analytic Poisson(+read-noise) floor on the scatter
                             # of off_sample_mean itself (shot noise of averaging n_off pixels
                             # over this point's k repeats) -- floors _check_pedestal_drift's
                             # empirical local sigma the same way _robust_sigma floors every
                             # other robust comparison in this module.
    lit_fraction: float = float("nan")      # fraction of this point's pixels in its lit (signal)
                             # region -- feeds the scan-level curve_bracketing check.

    @property
    def flagged(self) -> bool:
        return any(c.flagged for c in self.checks) or any(f.flagged for f in self.frames)

    @property
    def point_level_flagged(self) -> bool:
        """True when a POINT-level check failed (too few surviving repeats, photon-starved, whole-
        frame brightness anomaly): the point needs re-measuring. Distinct from :attr:`flagged`,
        which is also True when only individual repeats are flagged (those can simply be excluded
        by :func:`apply_quality_filter`; e.g. the documented hot first repeat)."""
        return any(c.flagged for c in self.checks)

    @property
    def flagged_checks(self) -> list:
        return [c for c in self.checks if c.flagged]

    def __repr__(self) -> str:
        state = "FLAGGED" if self.flagged else "ok"
        return f"PointQualityResult(point={self.point}, {self.coordinate_label}, {state})"


@dataclass
class ScanQualityReport:
    """The whole-scan quality report: every :class:`PointQualityResult`, scan-level checks, and
    a rollup ``verdict`` built from them (never an independent judgement -- see
    :meth:`_rollup`)."""

    points: list
    scan_checks: list
    verdict: str
    granularity: str
    settings: dict
    notes: list

    @property
    def flagged_points(self) -> list:
        return [p for p in self.points if p.flagged]

    def summary(self) -> str:
        n = len(self.points)
        nf = len(self.flagged_points)
        lines = [f"scan quality: {self.verdict}",
                 f"granularity  {self.granularity}",
                 f"points       {n} total, {nf} flagged"]
        for c in self.scan_checks:
            mark = "FLAG" if c.flagged else "ok  "
            lines.append(f"  [{mark}] {c.name}: {c.message}")
        for p in self.flagged_points[:50]:
            lines.append(f"  point {p.point} ({p.coordinate_label}):")
            for c in p.flagged_checks:
                lines.append(f"    [FLAG] {c.name}: {c.message}")
            for fr in p.frames:
                for c in fr.flagged_checks:
                    lines.append(f"    [FLAG] repeat {fr.repeat} {c.name}: {c.message}")
        if nf > 50:
            lines.append(f"  ... and {nf - 50} more flagged point(s)")
        lines += [f"note        {n_}" for n_ in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------- shared helpers
def _robust_sigma(values, gain, poisson_ref=None, extra_var=0.0):
    """MAD-based scatter, floored by a Poisson (+ read-noise) estimate so a coincidentally tiny
    empirical MAD (or one computed from very few points) cannot manufacture an arbitrarily large
    z-score. ``extra_var`` is an additive noise VARIANCE (e.g. read-noise variance, already scaled
    by however many independent pixels it was summed over by the caller) added under the
    square root alongside the Poisson term ``ref * gain``."""
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    ref = med if poisson_ref is None else poisson_ref
    poisson = math.sqrt(max(ref, 1.0) * gain + max(extra_var, 0.0))
    return med, max(1.4826 * mad, poisson)


def _point_lit_reference(frames_r, gain, baseline_percentile, lit_sigma_mult, read_noise_var=0.0):
    """Per-point pedestal, lit mask and mean frame, from the point's OWN repeats only.

    ``frames_r`` is ``(k, H, W)`` -- either the true repeats (``RawRepeatScan``) or a coarser
    2-way split (``RockingScan.halves``) or a single average (k=1, no repeat information at
    all). The lit mask and pedestal are reused by both the count-anomaly check (per-repeat
    totals inside it) and the point-level flux/photon-budget check, so both are working from the
    same region. ``read_noise_var`` is the per-pixel additive read-noise variance (see
    :func:`estimate_gain`), included in the Poisson floor used to build the lit mask.
    """
    mean_frame = frames_r.mean(0).astype(np.float64)
    pedestal = float(np.percentile(mean_frame, baseline_percentile))
    excess = mean_frame - pedestal
    k = frames_r.shape[0]
    if k >= 2:
        med = np.median(frames_r, axis=0).astype(np.float64)
        mad = np.median(np.abs(frames_r - med[None]), axis=0).astype(np.float64)
        noise = 1.4826 * mad
    else:
        noise = np.zeros_like(mean_frame)
    poisson_floor = np.sqrt(np.maximum(pedestal, 0.0) * gain + max(read_noise_var, 0.0))
    noise = np.maximum(noise, poisson_floor)
    lit = excess > lit_sigma_mult * np.maximum(noise, 1.0)
    return pedestal, lit, mean_frame


def _connected_component_sizes(mask):
    """4-connected component sizes of a (typically sparse) boolean mask, visiting only True
    pixels -- fast when few pixels are flagged, which is exactly the regime a spike candidate
    mask should be in on real data. Returns a list of (size, list-of-(row,col)) pairs."""
    coords = set(zip(*np.nonzero(mask)))
    comps = []
    while coords:
        seed = next(iter(coords))
        stack = [seed]
        coords.discard(seed)
        cells = [seed]
        while stack:
            i, j = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (i + di, j + dj)
                if nb in coords:
                    coords.discard(nb)
                    stack.append(nb)
                    cells.append(nb)
        comps.append(cells)
    return comps


# --------------------------------------------------------------------------- frame-level checks
def _check_saturation(frame, saturation_value, min_count):
    if saturation_value is None:
        return QualityCheck("saturation", False, 0.0, float("nan"),
                            "saturation check disabled (no saturation_value given)")
    n_sat = int(np.count_nonzero(frame >= saturation_value))
    frac = n_sat / frame.size
    flagged = n_sat >= min_count
    msg = (f"{n_sat} px ({frac:.2%}) at or above the saturation value {saturation_value:g}"
          + (f" (>= {min_count} triggers a flag)" if flagged else ""))
    return QualityCheck("saturation", flagged, float(n_sat), float(min_count), msg)


def _check_count_anomaly(frame, sibling_totals_incl_self, lit, gain, z_threshold, min_repeats,
                         read_noise_var=0.0):
    k = len(sibling_totals_incl_self)
    if k < min_repeats:
        return QualityCheck("count_anomaly", False, float("nan"), z_threshold,
                            f"skipped: only {k} repeat(s) at this point (need >= {min_repeats} "
                            "for a robust per-point comparison)")
    total = float(np.where(lit, frame, 0.0).sum())
    n_lit = int(np.count_nonzero(lit))
    med, sigma = _robust_sigma(sibling_totals_incl_self, gain,
                               extra_var=n_lit * max(read_noise_var, 0.0))
    z = (total - med) / sigma if sigma > 0 else 0.0
    flagged = abs(z) > z_threshold
    direction = "high" if z > 0 else "low"
    msg = (f"lit-region total {total:.0f} counts is {abs(z):.1f} sigma {direction} vs this "
          f"point's other repeats (median {med:.0f}, robust sigma {sigma:.1f}; threshold "
          f"{z_threshold:.1f} sigma)")
    return QualityCheck("count_anomaly", flagged, float(z), float(z_threshold), msg)


def _median8(neigh):
    """Median of an ``(8, H, W)`` stack along axis 0: the mean of the 4th and 5th order statistics.
    Exactly equal to ``np.median(neigh, axis=0)`` but ~2x faster (partial sort, not full sort)."""
    p = np.partition(neigh, [3, 4], axis=0)
    return 0.5 * (p[3] + p[4])


def _spike_candidate_mask(frame, gain, spike_zscore, min_spike_excess, read_noise_var=0.0):
    """The per-pixel "looks like a spike relative to its immediate 8-neighbourhood" test, shared
    by :func:`_check_spike` (one frame) and :func:`_static_hot_pixel_mask` (every frame in the
    scan, to find pixels that trip this same test over and over at a FIXED location)."""
    H, W = frame.shape
    pad = np.pad(frame.astype(np.float32), 1, mode="edge")
    neigh = np.stack([pad[i:i + H, j:j + W] for i in range(3) for j in range(3)
                      if not (i == 1 and j == 1)], axis=0)          # (8, H, W)
    local_med = _median8(neigh)
    local_mad = _median8(np.abs(neigh - local_med[None]))
    local_med = local_med.astype(np.float64)
    poisson_var = np.maximum(local_med, 0.0) * gain + max(read_noise_var, 0.0)
    local_sigma = np.maximum(1.4826 * local_mad.astype(np.float64), np.sqrt(poisson_var))
    local_sigma = np.maximum(local_sigma, 1.0)
    residual = frame.astype(np.float64) - local_med
    z = residual / local_sigma
    candidates = (z > spike_zscore) & (residual > min_spike_excess)
    return candidates, residual, z


def _local_z_at(frame, ys, xs, gain, read_noise_var=0.0):
    """Local-8-neighbourhood z-value (the same statistic :func:`_spike_candidate_mask` uses), but
    only at a small list of pixel locations -- for cheaply re-measuring a handful of candidate
    locations in SIBLING repeat frames without a full-frame convolution (see
    :func:`_check_spike`'s ``sibling_frames`` docstring)."""
    H, W = frame.shape
    out = np.zeros(len(ys), dtype=np.float64)
    frame64 = frame.astype(np.float64)
    for i, (y, x) in enumerate(zip(ys, xs)):
        y0, y1 = max(y - 1, 0), min(y + 2, H)
        x0, x1 = max(x - 1, 0), min(x + 2, W)
        patch = frame64[y0:y1, x0:x1]
        keep = np.ones(patch.shape, dtype=bool)
        keep[y - y0, x - x0] = False
        neigh = patch[keep]
        med = float(np.median(neigh))
        mad = float(np.median(np.abs(neigh - med)))
        poisson_var = max(med, 0.0) * gain + max(read_noise_var, 0.0)
        sigma = max(1.4826 * mad, math.sqrt(poisson_var), 1.0)
        out[i] = (float(frame64[y, x]) - med) / sigma
    return out


def _check_spike(frame, gain, spike_zscore, max_spike_pixels, min_spike_excess,
                 read_noise_var=0.0, hot_mask=None, sibling_frames=None,
                 stable_zscore=None, max_spike_fraction=1e-5):
    """``sibling_frames``: this point's OTHER repeat exposures (same motor position), if any.

    A genuine cosmic ray or detector glitch is a TRANSIENT, single-exposure event: it should NOT
    reproduce at the same pixel location in a DIFFERENT repeat taken at the same scan point (same
    angle, same sample, same illumination -- nothing has moved). Real, fine-scale sample texture
    (speckle) does the opposite: it is part of the diffraction image itself, so it reproduces
    across repeats of the SAME point. Measured directly on real 6-ID-C data (S996): candidate
    pixels that survive static-hot-pixel exclusion are overwhelmingly in the locally brightest
    ~10% of pixels at BOTH a strong and a weak scan point, and checking specific candidate
    locations across a point's 10 repeats shows the "spike" value stable and elevated in every
    single repeat there, not a one-off -- i.e. these are real image texture, not per-frame noise,
    and the "isolated small cluster = cosmic ray, broad cluster = real PSF" size-only rule this
    check otherwise uses does not hold for DFXM speckle at pixel scale. Stable texture is
    only elevated, not necessarily z > ``spike_zscore``, in every repeat (measured S996 point 77:
    stable candidates have z = 3-7 in the other repeats; the one repeat that clears the strict
    threshold is typically the documented hot first repeat), so the sibling test is deliberately
    LENIENT: a candidate is excluded when its MEDIAN local z across the sibling repeats exceeds
    ``stable_zscore`` (default ``spike_zscore / 4``). A candidate that is NOT elevated in the
    siblings (median z near 0; measured -1 to 0.7 for genuine one-offs) is kept, since that IS what
    a
    genuine cosmic ray looks like. (An earlier attempt calibrated a scan-wide count threshold from
    the background candidate rate instead -- rejected: it also silently absorbed a genuinely WRONG
    detector gain's excess false-positive rate as if it were normal background, defeating the
    purpose of the check; this per-candidate, per-point, across-repeats test does not have that
    failure mode, since a systematic gain error inflates noise the same way in every repeat and so
    does not, on its own, make one repeat look elevated relative to its siblings.)
    """
    candidates, residual, z = _spike_candidate_mask(frame, gain, spike_zscore, min_spike_excess,
                                                     read_noise_var=read_noise_var)
    n_before_mask = int(candidates.sum())
    hot_note = ""
    if hot_mask is not None and hot_mask.any():
        candidates = candidates & ~hot_mask
        n_excluded = n_before_mask - int(candidates.sum())
        if n_excluded:
            hot_note = (f" ({n_excluded} candidate px excluded as known static hot pixels -- "
                       "see the scan-level static_hot_pixels check)")
    stable_note = ""
    if sibling_frames is not None and len(sibling_frames) > 0 and candidates.any():
        bar = (spike_zscore / 4.0) if stable_zscore is None else float(stable_zscore)
        ys, xs = np.nonzero(candidates)
        z_sib = np.stack([_local_z_at(sib, ys, xs, gain, read_noise_var=read_noise_var)
                          for sib in sibling_frames], axis=0)
        stable = np.median(z_sib, axis=0) > bar
        n_stable = int(stable.sum())
        if n_stable:
            candidates = candidates.copy()
            candidates[ys[stable], xs[stable]] = False
            stable_note = (f" ({n_stable} candidate px excluded: their MEDIAN local z across this "
                          f"point's {len(sibling_frames)} other repeat(s) exceeds {bar:g}, i.e. "
                          "the pixel is elevated in the other exposures too -- stable structure "
                          "such as sample texture, not a one-off event)")
    n_candidates = int(candidates.sum())
    if n_candidates == 0:
        return QualityCheck("spike", False, 0.0, float(max_spike_pixels),
                            "no pixel exceeds the local-neighbourhood spike threshold "
                            f"(z > {spike_zscore:g}, excess > {min_spike_excess:g} counts)"
                            + hot_note + stable_note)
    comps = _connected_component_sizes(candidates)
    spikes = [c for c in comps if len(c) <= max_spike_pixels]
    largest = max((len(c) for c in comps), default=0)
    if not spikes:
        return QualityCheck(
            "spike", False, float(largest), float(max_spike_pixels),
            f"{n_candidates} anomalous px found, but the largest connected group is {largest} "
            f"px (> {max_spike_pixels}): reads as a broad feature (e.g. a real PSF peak), not "
            "an isolated spike" + hot_note + stable_note)
    n_spike_px = sum(len(c) for c in spikes)
    allowance = int(max_spike_fraction * frame.size)
    rows_cols = spikes[0]
    r0, c0 = rows_cols[0]
    if n_spike_px <= allowance:
        return QualityCheck(
            "spike", False, float(n_spike_px), float(max_spike_pixels),
            f"{n_spike_px} isolated transient px (e.g. at row {r0}, col {c0}) not reproduced in "
            f"sibling repeats, but within the {allowance} px allowance for a frame of "
            f"{frame.size} px (max_spike_fraction={max_spike_fraction:g}): reported, not flagged"
            + hot_note + stable_note)
    return QualityCheck(
        "spike", True, float(n_spike_px), float(max_spike_pixels),
        f"{len(spikes)} isolated spike(s), {n_spike_px} px total (largest cluster "
        f"{max(len(c) for c in spikes)} px, e.g. at row {r0}, col {c0}), each wildly "
        f"inconsistent with its immediate 8-neighbourhood (z > {spike_zscore:g}), too small "
        f"to be a real PSF feature (<= {max_spike_pixels} px), and NOT reproduced in any sibling "
        "repeat at this point: looks like a real cosmic ray or detector glitch" + hot_note
        + stable_note)


def _static_hot_pixel_mask(get_group, n_points, gain, spike_zscore, min_spike_excess,
                           significance, min_hits, read_noise_var=0.0, max_frames=None, seed=0):
    """Find pixels that trip :func:`_spike_candidate_mask`'s per-frame test at the SAME location
    over and over across the scan: the signature of a fixed detector defect (a permanently hot,
    stuck, or intermittently flickering pixel), not a per-frame cosmic ray.

    The discriminating test is exactly the one a real cosmic ray fails: it lands at a different,
    effectively random pixel each frame, so the chance of it hitting the exact same location twice
    in a scan of hundreds of frames should be tiny -- UNLESS the scan's own candidate rate is high
    enough that a coincidence becomes plausible, which is exactly why this uses the scan's OWN
    observed rate rather than a fixed number or fraction.

    **Two earlier versions of this function were tried on real S168 data and were both WRONG --
    both left the scan reading "not usable, 61 of 61 points flagged", unchanged from before any
    fix, and the reason each failed is instructive:**

    1. A flat "hot in more than half the scan's frames" FRACTION threshold. Measured on S168 (122
       frames, gain auto-estimated at 2.77), the candidate rate was ~775 px/frame -- not the
       handful per frame a "cosmic ray" mental model suggests -- and the per-pixel hit-count
       distribution was NOT simply "always hot" vs "hit once by chance": a population of ~1900
       pixels fired in anywhere from 1 to ~15 of the 122 frames (a clean gap then separated them
       from a second population hit in >=61, i.e. a true majority). A 50%-of-frames bar left all
       ~1900 of them in the ordinary per-frame check, and their combined residual rate was still
       high enough that essentially every frame kept at least one small isolated "spike".
    2. A single-pass STATISTICAL threshold using the scan's GLOBAL candidate rate (``mu = total
       candidate-events / total pixels``) with :func:`_poisson_min_recurrence`: better (S168 gave
       ``mu ~ 0.05``, threshold 4 hits, comfortably inside the 15-to-61 gap above without being
       tuned to it), but the scan STILL read "not usable" afterwards. The reason: that one global
       ``mu`` is itself inflated ~3.6x by the ~2000 already-obviously-defective pixels' own
       contribution, which understated how rare even a SECOND hit at one location should be for
       everything else, and let a genuinely non-random population of pixels recurring 2-15 times
       (each still statistically incompatible with chance -- see below) stay in the per-frame
       check. Their combined residual rate (measured directly: ~5.7 candidate px/frame after
       excluding only the >=4-hit pixels) was still enough to flag nearly every frame.

    **The fix**: ``mu`` must come from the BACKGROUND population only. This iterates like a
    sigma-clip -- exclude whatever the current threshold calls a defect, recompute ``mu`` from
    what is left, and repeat until the threshold stops moving. On S168 this converges immediately
    to a threshold of 2 hits: excluding every pixel that fires even TWICE at the exact same
    location leaves a background rate low enough (``mu ~ 0.0002``) that a THIRD hit would already
    need probability ~2e-8 by chance -- i.e. 2 hits is the smallest threshold that is
    self-consistent, not merely "small enough to feel safe". ``min_hits`` (default 2) is a floor
    under the statistical threshold: it is also, not coincidentally, the smallest number for which
    "the same pixel, twice" is itself meaningful (a single hit is definitionally indistinguishable
    from one real, isolated event, and this function must not treat every pixel a real cosmic ray
    has ever touched as a permanent defect).

    A residual single-pixel candidate rate can still remain after this (S168, measured directly:
    2236 static pixels excluded, ~2.8 candidate px/frame left over at genuinely non-recurring
    locations) -- this is NOT a defect to exclude: it is exactly the real, rare, per-frame
    cosmic-ray-like noise :func:`_check_spike` exists to catch, at a rate that is unavoidably
    higher than intuition suggests once a frame has ~1.8 million independent pixels to land on.
    Whether that residual rate is enough to push a report's overall VERDICT to "not usable" is a
    separate, real property of a low repeat count combined with a very large frame, NOT something
    this function's job (excluding known-bad LOCATIONS) can or should silence: on real S168 (only
    2 repeats/point), the fix takes the verdict from "not usable, 61 of 61 points flagged" (every
    point misattributed to fixed defects) to "not usable, 58 of 61 points flagged" -- correctly
    attributed now to a genuine, if unfortunately still-common-at-this-frame-size, residual event
    rate colliding with a 2-repeat design, which :func:`apply_quality_filter` (excluding just the
    hit repeat, keeping the other) remains the right tool for, not a change to this function.

    ``max_frames`` (default: use every frame) subsamples frame INDICES (not pixels) when a scan
    has more frames than this, trading a little precision on ``mu`` for bounded runtime on a very
    large real scan; ``mu`` and the hit-count threshold are computed relative to however many
    frames were actually sampled, not the full scan.

    Returns
    -------
    mask : (H, W) bool -- True where a static hot pixel was found (empty array if the scan has no
        frames at all).
    frac : (H, W) float -- hit fraction per pixel, over the frames actually sampled (informational
        only -- NOT the gating criterion; see above).
    hit_count : (H, W) int
    n_frames_sampled : int
    threshold_k : int -- the hit-count threshold actually applied (``mask = hit_count >=
        threshold_k``).
    """
    total_frames = 0
    per_point_k = []
    for p in range(n_points):
        k = int(get_group(p).shape[0])
        per_point_k.append(k)
        total_frames += k
    if total_frames == 0:
        return (np.zeros((0, 0), dtype=bool), np.zeros((0, 0)), np.zeros((0, 0), dtype=np.int64),
                0, min_hits)

    chosen = None
    if max_frames is not None and total_frames > max_frames:
        rng = np.random.default_rng(seed)
        chosen = frozenset(rng.choice(total_frames, size=int(max_frames), replace=False).tolist())

    hit_count = None
    n_frames_sampled = 0
    idx = 0
    for p in range(n_points):
        k = per_point_k[p]
        if k == 0:
            continue
        grp = get_group(p)
        for r in range(k):
            use = chosen is None or idx in chosen
            if use:
                cand, _, _ = _spike_candidate_mask(grp[r], gain, spike_zscore, min_spike_excess,
                                                   read_noise_var=read_noise_var)
                if hit_count is None:
                    hit_count = np.zeros(cand.shape, dtype=np.int64)
                hit_count += cand
                n_frames_sampled += 1
            idx += 1
    if hit_count is None or n_frames_sampled == 0:
        return (np.zeros((0, 0), dtype=bool), np.zeros((0, 0)), np.zeros((0, 0), dtype=np.int64),
                0, min_hits)
    n_total_px = hit_count.size
    # mu (the chance-coincidence rate) must come from the BACKGROUND population only, not from
    # the whole pixel population: on real S168 data, ~2000 pixels fire in 60-100% of frames
    # (obviously real defects), and folding their huge contribution into a single global mean
    # inflates mu ~3.6x, which in turn OVER-STATES how often a pixel could plausibly hit twice by
    # pure chance and lets real, if less extreme, recurring defects (pixels firing in, say,
    # 5-12% of frames -- nowhere near "always", but still never explained by chance, see below)
    # slip back into the ordinary per-frame check. This iterates like a simple sigma-clip: exclude
    # whatever the current threshold calls a defect from the "background" pool, recompute mu from
    # what is left, and repeat until the threshold stops moving (in practice this converges in 1-2
    # steps on real data).
    threshold_k = max(int(min_hits), 1)
    for _ in range(20):
        background_events = float(hit_count[hit_count < threshold_k].sum())
        mu = background_events / n_total_px if n_total_px else 0.0
        new_k = max(int(min_hits), _poisson_min_recurrence(mu, significance))
        if new_k == threshold_k:
            break
        threshold_k = new_k
    frac = hit_count / n_frames_sampled
    mask = hit_count >= threshold_k
    return mask, frac, hit_count, n_frames_sampled, threshold_k


def _check_static_hot_pixels(mask, n_total_px, n_frames_sampled, threshold_k, significance,
                             max_hot_pixel_fraction):
    """Scan-level report of :func:`_static_hot_pixel_mask`'s result -- once, not per frame."""
    if n_total_px == 0 or mask.size == 0:
        return QualityCheck("static_hot_pixels", False, 0.0, float(max_hot_pixel_fraction),
                            "skipped: no frames available to check for static hot pixels")
    n_hot = int(mask.sum())
    px_frac = n_hot / n_total_px
    flagged = px_frac > max_hot_pixel_fraction
    if n_hot == 0:
        msg = (f"no static hot pixels found (none trip the spike criterion at least "
              f"{threshold_k} time(s) at the same location across {n_frames_sampled} sampled "
              f"frame(s) -- the threshold at which even chance coincidence would be less likely "
              f"than {significance:g}, given this scan's own candidate rate) -- the per-frame "
              "spike check runs on every pixel")
    else:
        ys, xs = np.nonzero(mask)
        example = f"e.g. row {int(ys[0])}, col {int(xs[0])}"
        msg = (f"{n_hot} static hot pixel(s) ({px_frac:.3%} of {n_total_px} px) trip the "
              f"per-frame spike criterion at least {threshold_k} time(s) at the SAME location "
              f"({example}) across {n_frames_sampled} sampled frame(s) -- a coincidence rate "
              f"below {significance:g} given this scan's own candidate rate (see "
              "_static_hot_pixel_mask's docstring), i.e. these read as fixed detector defects, "
              "not per-frame cosmic rays, and are excluded from the per-frame spike check for "
              "the rest of this report rather than flagged ~once per frame each")
        if flagged:
            msg += (f". WARNING: {px_frac:.2%} of all pixels read as a static hot pixel, above "
                   f"the {max_hot_pixel_fraction:.2%} sanity ceiling for a real hot-pixel "
                   "population -- check the estimated gain and spike thresholds before trusting "
                   "this as a real defect map rather than a symptom of a bad noise model")
    return QualityCheck("static_hot_pixels", flagged, float(px_frac),
                        float(max_hot_pixel_fraction), msg)


def _check_dead_or_duplicate(frame, repeat_idx, all_frames_at_point, lit,
                             dead_variance_ratio, duplicate_atol, duplicate_rtol,
                             flat_variance_floor):
    k = all_frames_at_point.shape[0]
    var_r = float(np.var(frame[lit])) if lit.any() else float(np.var(frame))
    # absolute flatness, regardless of siblings: a genuinely frozen/blank frame
    if var_r < flat_variance_floor:
        return QualityCheck("dead_frame", True, var_r, flat_variance_floor,
                            f"variance {var_r:.3g} is below the flatness floor "
                            f"{flat_variance_floor:.3g}: frame looks blank or frozen "
                            "(e.g. a shutter fault)")
    # exact/near-exact duplicate of another repeat at the same point
    for other in range(k):
        if other == repeat_idx:
            continue
        if np.allclose(frame, all_frames_at_point[other], rtol=duplicate_rtol,
                       atol=duplicate_atol):
            return QualityCheck("dead_frame", True, 0.0, duplicate_atol,
                                f"identical (within {duplicate_atol:g} + {duplicate_rtol:g} "
                                f"rel.) to repeat {other} at this point: looks like a "
                                "duplicated file, not two independent exposures")
    if k < 2:
        return QualityCheck("dead_frame", False, var_r, float("nan"),
                            "skipped: only one repeat at this point, nothing to compare against")
    others_var = [float(np.var(all_frames_at_point[o][lit])) if lit.any()
                 else float(np.var(all_frames_at_point[o])) for o in range(k) if o != repeat_idx]
    ref = float(np.median(others_var)) if len(others_var) >= 2 else others_var[0]
    if ref <= 0:
        return QualityCheck("dead_frame", False, var_r, float("nan"),
                            "skipped: this point's other repeats have no measurable structure "
                            "either (see the point-level flux check)")
    ratio = var_r / ref
    flagged = ratio < dead_variance_ratio
    msg = (f"structure (variance) {var_r:.3g} is {ratio:.2f}x this point's sibling-repeat "
          f"median ({ref:.3g}); threshold {dead_variance_ratio:.2f}x"
          + (" -- looks dead/frozen relative to its siblings" if flagged else ""))
    return QualityCheck("dead_frame", flagged, float(ratio), float(dead_variance_ratio), msg)


def _assess_frame(point, repeat, frame, all_frames_at_point, sibling_totals, lit, *, gain,
                  saturation_value, saturation_min_count, count_z_threshold,
                  min_repeats_for_robust_check, spike_zscore, max_spike_pixels,
                  min_spike_excess, dead_variance_ratio, duplicate_atol, duplicate_rtol,
                  flat_variance_floor, read_noise_var=0.0, hot_mask=None,
                  max_spike_fraction=1e-5) -> FrameQualityResult:
    siblings = [all_frames_at_point[i] for i in range(all_frames_at_point.shape[0])
               if i != repeat] if all_frames_at_point.shape[0] > 1 else None
    checks = [
        _check_saturation(frame, saturation_value, saturation_min_count),
        _check_count_anomaly(frame, sibling_totals, lit, gain, count_z_threshold,
                             min_repeats_for_robust_check, read_noise_var=read_noise_var),
        _check_spike(frame, gain, spike_zscore, max_spike_pixels, min_spike_excess,
                    read_noise_var=read_noise_var, hot_mask=hot_mask, sibling_frames=siblings,
                    max_spike_fraction=max_spike_fraction),
        _check_dead_or_duplicate(frame, repeat, all_frames_at_point, lit, dead_variance_ratio,
                                 duplicate_atol, duplicate_rtol, flat_variance_floor),
    ]
    return FrameQualityResult(point=point, repeat=repeat, checks=checks)


# --------------------------------------------------------------------------- point-level checks
def _check_survival(n_surviving, n_repeats, min_survive_fraction):
    frac = n_surviving / n_repeats if n_repeats else 0.0
    flagged = frac < min_survive_fraction
    msg = (f"{n_surviving} of {n_repeats} repeats survive frame-level checks ({frac:.0%})"
          + (f"; below the {min_survive_fraction:.0%} floor" if flagged else ""))
    return QualityCheck("repeat_survival", flagged, float(frac), float(min_survive_fraction), msg)


def _check_point_flux(pedestal, lit, mean_frame, gain, min_snr, read_noise_var=0.0):
    if not lit.any():
        return QualityCheck("flux_health", True, 0.0, float(min_snr),
                            "no pixel reaches the lit threshold above this point's own pedestal "
                            "and repeat-to-repeat noise: no detectable signal at all (the S224 "
                            "photon-starvation symptom -- see NOTE_S224_frames.md)")
    n_lit = int(lit.sum())
    total_lit_signal = float(np.clip(mean_frame - pedestal, 0.0, None)[lit].sum())
    noise_var_total = max(total_lit_signal, 0.0) * gain + n_lit * max(read_noise_var, 0.0)
    snr = math.sqrt(max(noise_var_total, 0.0))
    flagged = snr < min_snr
    msg = (f"lit-region signal {total_lit_signal:.0f} counts above pedestal "
          f"({n_lit} px), SNR {snr:.1f} (needs >= {min_snr:.1f}) at gain={gain:g}"
          + (f", read_noise_var={read_noise_var:.1f}" if read_noise_var > 0 else "")
          + (": photon-starved -- this point's own signal is not clearly distinguishable from "
             "shot noise" if flagged else ""))
    return QualityCheck("flux_health", flagged, float(snr), float(min_snr), msg)


def _assess_point(point, frames_r, motors_row, coordinate_label, granularity, *, gain,
                  baseline_percentile, lit_sigma_mult, min_snr, min_survive_fraction,
                  frame_kwargs, read_noise_var=0.0, off_sample_mask=None) -> PointQualityResult:
    k = frames_r.shape[0]
    pedestal, lit, mean_frame = _point_lit_reference(frames_r, gain, baseline_percentile,
                                                     lit_sigma_mult, read_noise_var=read_noise_var)
    sibling_totals = [float(np.where(lit, frames_r[r], 0.0).sum()) for r in range(k)]
    frame_results = [
        _assess_frame(point, r, frames_r[r], frames_r, sibling_totals, lit, gain=gain,
                     read_noise_var=read_noise_var, **frame_kwargs)
        for r in range(k)
    ] if granularity != "average-only" else []
    n_surviving = sum(1 for f in frame_results if not f.flagged) if frame_results else k
    checks = []
    if frame_results:
        checks.append(_check_survival(n_surviving, k, min_survive_fraction))
    checks.append(_check_point_flux(pedestal, lit, mean_frame, gain, min_snr,
                                    read_noise_var=read_noise_var))
    off_sample_mean = float("nan")
    off_sample_sigma_floor = 0.0
    if off_sample_mask is not None and off_sample_mask.any():
        off_sample_mean = float(mean_frame[off_sample_mask].mean())
        n_off = int(off_sample_mask.sum())
        off_sample_sigma_floor = math.sqrt(
            (max(off_sample_mean, 0.0) * gain + max(read_noise_var, 0.0))
            / max(n_off * max(k, 1), 1))
    return PointQualityResult(point=point, coordinate=motors_row, coordinate_label=coordinate_label,
                              frames=frame_results, n_repeats=k, n_surviving=n_surviving,
                              checks=checks, granularity=granularity,
                              off_sample_mean=off_sample_mean,
                              off_sample_sigma_floor=off_sample_sigma_floor,
                              lit_fraction=_mean_signal_fraction(frames_r, mean_frame, pedestal, gain, read_noise_var))


# --------------------------------------------------------------------------- scan-level checks
def _check_frame_table_consistency(n_points, n_repeats, motors):
    expected = n_points * n_repeats
    dup = None
    if n_points >= 4 and n_points % 2 == 0:
        half = n_points // 2
        for k, v in motors.items():
            if np.ptp(v) == 0:
                continue
            if np.allclose(v[:half], v[half:], atol=1e-9, rtol=0):
                dup = k
                break
    flagged = dup is not None
    msg = (f"{n_points} motor-table points x {n_repeats} repeat(s) = {expected} frames")
    if flagged:
        msg += (f". WARNING: column {dup!r} repeats EXACTLY across the first and second half "
               "of the table -- this is the signature of a restarted-scan CSV with a "
               "duplicated header block silently doubling the row count (the known bug class "
               "behind the NaMnO2 archive's duplicated motor log, see io_6idc.py)")
    return QualityCheck("frame_table_consistency", flagged, float(n_points), float(expected), msg)


def _check_step_regularity(coordinate, step_outlier_frac):
    coord = np.asarray(coordinate, dtype=float)
    if coord.ndim == 1:
        if coord.size < 3:
            return QualityCheck("step_regularity", False, float("nan"), step_outlier_frac,
                                "skipped: fewer than 3 points")
        steps = np.diff(coord)
        med = float(np.median(np.abs(steps)))
        if med == 0:
            return QualityCheck("step_regularity", True, 0.0, step_outlier_frac,
                                "the rocking coordinate does not change between points at all")
        dev = np.abs(np.abs(steps) - med) / med
        bad = np.flatnonzero(dev > step_outlier_frac)
        flagged = bad.size > 0
        msg = f"median step {1000 * med:.3f} mdeg"
        if flagged:
            msg += (f"; {bad.size} step(s) deviate by more than {step_outlier_frac:.0%}, e.g. "
                   f"between points {int(bad[0])} and {int(bad[0]) + 1} "
                   f"({1000 * abs(steps[bad[0]]):.3f} mdeg)")
        return QualityCheck("step_regularity", flagged, float(dev.max()) if dev.size else 0.0,
                            float(step_outlier_frac), msg)
    # mesh: check each axis over its own sorted unique values
    bad_axes = []
    worst = 0.0
    for a in range(coord.shape[-1]):
        vals = np.sort(np.unique(np.round(coord[:, a], 6)))
        if vals.size < 3:
            continue
        steps = np.diff(vals)
        med = float(np.median(steps))
        if med <= 0:
            continue
        dev = np.abs(steps - med) / med
        worst = max(worst, float(dev.max()) if dev.size else 0.0)
        if (dev > step_outlier_frac).any():
            bad_axes.append(a)
    flagged = len(bad_axes) > 0
    msg = (f"irregular grid spacing on axis index {bad_axes}" if flagged
          else "grid spacing regular on every axis")
    return QualityCheck("step_regularity", flagged, worst, float(step_outlier_frac), msg)


def _mean_signal_fraction(frames_r, mean_frame, pedestal, gain, read_noise_var=0.0):
    """Fraction of pixels whose REPEAT-AVERAGED signal is significant (> 5 sigma of the mean's own
    noise) AND spatially contiguous (>= 6 of the 3x3 block), i.e. the size of the real diffracting
    region at this point.

    Two measured pitfalls shaped this. (1) The lit mask :func:`_point_lit_reference` builds uses the
    SINGLE-frame noise against the averaged frame, which demands ~60 counts/px and finds nothing at
    all on a 20-repeat scan whose average is clearly significant (S1050); the noise of a mean is
    sigma/sqrt(k). (2) Isolated significant pixels are static hot pixels, not signal (0.1-0.9 % of
    the frame), so contiguity is required."""
    from scipy.ndimage import uniform_filter
    k = frames_r.shape[0]
    floor = math.sqrt(max(pedestal, 0.0) * gain + max(read_noise_var, 0.0))
    if k >= 2:
        med = np.median(frames_r, axis=0).astype(np.float64)
        noise = np.maximum(1.4826 * np.median(np.abs(frames_r - med[None]), axis=0), floor)
    else:
        noise = np.full(mean_frame.shape, floor)
    sig = (mean_frame - pedestal) > 5.0 * np.maximum(noise / math.sqrt(k), 1.0)
    dense = uniform_filter(sig.astype(np.float32), size=3, mode="constant") * 9.0 >= 5.5
    return float((dense & sig).mean())


def _check_curve_bracketing(points, coordinate, max_edge_over_peak, min_peak_fraction):
    """Scan-level: does the scan actually contain a rocking curve -- signal that rises from the
    scan edges to a peak? -- and is there any contiguous signal region at all?

    S224 (found while validating, and NOT what its first documented reading said): its
    significant contiguous region is ~21 % of the frame at EVERY point (0.198-0.213), i.e. the scan
    edges are as bright as its centre and no peak is bracketed (its own notes: 89 % of the maximum
    already at the first point). The healthy scans rise from a few percent at the edges to a peak
    (S996 0.5 -> 10.7 %, S995 0.5 -> 12.5 %, S1050 3-4 -> 17 %; edge/peak 0.05, 0.04, 0.25).
    This is the survey rule "does each scan's window bracket its own peak" (Notebook 5l) made
    automatic; a moment/width reduction of an unbracketed curve is biased. It is deliberately NOT
    built from the point-level SNR (huge on any multi-megapixel frame) or the auto-gain (unstable
    on such scans). The default 0.6 sits between the measured populations (0.98 vs <= 0.25).
    Skipped for mesh (multi-axis) coordinates and scans shorter than 6 points."""
    lf = np.array([pt.lit_fraction for pt in points], dtype=float)
    coord = np.asarray(coordinate)
    if lf.size < 6 or not np.isfinite(lf).all() or coord.ndim != 1:
        return QualityCheck("curve_bracketing", False, float("nan"), float(max_edge_over_peak),
                            "skipped: needs a 1-D scan of at least 6 points with signal fractions")
    peak = float(lf.max())
    if peak < min_peak_fraction:
        return QualityCheck(
            "curve_bracketing", True, peak, float(min_peak_fraction),
            f"the largest contiguous significant-signal region at ANY point is {peak:.2e} of the "
            f"frame (needs >= {min_peak_fraction:.0e}): no usable signal region anywhere in the scan")
    ratio = float(max(lf[:2].mean(), lf[-2:].mean()) / peak)
    flagged = ratio > max_edge_over_peak
    msg = (f"signal region is {peak:.1%} of the frame at its peak and {ratio:.0%} of that at the "
          f"brighter scan edge (threshold {max_edge_over_peak:.0%})"
          + ("; the peak is NOT bracketed -- the scan edges are as bright as its centre, so widths "
             "and centroids from this scan are biased (Notebook 5l)" if flagged else ""))
    return QualityCheck("curve_bracketing", flagged, ratio, float(max_edge_over_peak), msg)


def _off_sample_mask(get_group, n_points, gain, baseline_percentile, lit_sigma_mult,
                     read_noise_var=0.0):
    """Pixels that are never part of the lit (diffracting) region at ANY point in the scan --
    genuine off-sample background. Pass 1 of 2 for :func:`_check_pedestal_drift`: only each
    point's boolean lit mask is kept (via :func:`_point_lit_reference`, discarding its mean frame
    immediately) so this does not multiply the report's peak memory use on a large real scan; the
    per-point mean frame needed to actually MEASURE the off-sample level is recomputed in the
    main per-point loop (:func:`_assess_point`), which already needs it for the other point-level
    checks.
    """
    ever_lit = None
    for p in range(n_points):
        grp = get_group(p)
        if grp.shape[0] == 0:
            continue
        _, lit, _ = _point_lit_reference(grp, gain, baseline_percentile, lit_sigma_mult,
                                         read_noise_var=read_noise_var)
        ever_lit = lit.copy() if ever_lit is None else (ever_lit | lit)
    if ever_lit is None:
        return np.zeros((0, 0), dtype=bool)
    return ~ever_lit


def _check_pedestal_drift(off_sample_means, off_sample_sigma_floors, window, z_threshold,
                          min_relative_change=0.005):
    """Point-vs-neighbours check for a whole-POINT brightness anomaly that both of a point's own
    repeats agree on (so :func:`_check_count_anomaly` -- a WITHIN-point comparison -- cannot see
    it at all), built from each point's OFF-SAMPLE background level (:func:`_off_sample_mask`),
    never from the lit/signal region.

    This exists because of S168 point 32 (see the module docstring and ``NOTE_S168_frames.md``):
    its entire frame reads elevated, both repeats agree, and the point-level flux-health check
    passes fine because the point is too BRIGHT, not starved. Two earlier attempts to catch this
    by comparing the LIT-region signal to its neighbours were tried and retracted:

    (a) A leave-one-out quadratic fit through each pixel's own 4 neighbouring points
        (m-2, m-1, m+1, m+2) on the SUMMED lit-region curve, flagging a large ratio of actual to
        predicted. A real rocking curve curves fast near a sharp peak; a quadratic through only 4
        points spanning the peak does not track that curvature well, and the residual reads as a
        false anomaly -- it flagged points 21, 22, 23 and 31, all near the true peak, not just the
        real defect at 32.
    (b) The same leave-one-out-quadratic idea applied per pixel instead of to the summed curve.
        Measured on the real data, it gave z = 3.95 at the true defect (point 32) -- BELOW its own
        5-sigma threshold, i.e. it would have missed the real problem -- while false-flagging
        unrelated points (55, 57, 58) that show no other evidence of a defect.

    Both failed for the same underlying reason: the lit/signal region's own intensity legitimately
    rises and falls fast near a real peak, so ANY local fit built from it risks aliasing the
    curve's own shape as an anomaly (approach (a)), or getting swamped by the curve's genuine
    point-to-point change relative to the size of the real defect (approach (b), which needed a
    per-pixel S/N that a single pixel does not have).

    The off-sample background sidesteps this completely: since it is (by construction, via
    :func:`_off_sample_mask`) never illuminated by the diffracted beam at ANY point in the scan, it
    has NO rocking-curve shape to alias against -- comparing a point's off-sample level to a
    robust LOCAL baseline built from its neighbouring points (median/MAD, excluding the point
    itself, in a window that shrinks near the ends of the scan rather than borrowing points from
    across the whole range) is safe regardless of how sharp the real peak is. A genuine whole-frame
    brightness event (point 32's problem) still shows up here, just with a much smaller relative
    excursion than in the lit region (true signal +55% vs off-sample background +3.5% at point 32,
    measured directly from S168 -- see the module docstring), because most of the off-sample level
    is detector background rather than beam-related stray light.

    Parameters
    ----------
    off_sample_means : sequence of float, one per point (NaN where unavailable).
    off_sample_sigma_floors : sequence of float, one per point -- an analytic Poisson(+read-noise)
        floor on the scatter of that point's OWN ``off_sample_means`` value (shot noise of
        averaging a finite number of off-sample pixels over a finite number of repeats), flooring
        the empirical local MAD-based sigma the same way :func:`_robust_sigma` floors every other
        robust comparison in this module -- without it, a scan with few points and/or a small
        off-sample pixel count can give a coincidentally tiny empirical MAD from its handful of
        neighbours and manufacture a spurious large z-score (caught on the synthetic clean-scan
        regression test during development: a 14-point, single-realization synthetic scan
        occasionally gave z=5 at its last point from sampling noise alone, before this floor was
        added).
    window : how many points on EACH side to consider as candidate neighbours. Near the ends of
        the scan this is naturally one-sided rather than symmetric (e.g. point 0 only has
        neighbours to its right); if that still leaves fewer than 4 usable neighbours, the window
        is WIDENED (not narrowed) until either 4 are found or the whole scan has been tried, and
        the check is skipped for that point rather than run on too little data.
    min_relative_change : effect-size floor (default 0.5 %): a point is flagged only if it ALSO
        differs from its local baseline by more than this fraction of the baseline level. The
        statistical floor above is a shot-noise number and is far tighter than real detector
        jitter on a smooth scan: S996's off-sample level moves ~0.01 counts point to point on a
        level of 107.7 with a 0.0017-count shot floor, so point 11 (0.014 % below its neighbours)
        reads z = -9 and is physically nothing. S168's real defect is +4 % (point 32) and +2 %
        (point 33).
    z_threshold : robust-sigma threshold (default from :func:`assess_scan_quality`: 4.0, chosen
        with margin above the largest clean-scan excursion observed on real data during
        development, 2.65 sigma on S996, while still well below the 4.8 sigma measured at S168's
        real defect).

    Returns
    -------
    list of QualityCheck, index-aligned with ``off_sample_means``.
    """
    vals = np.asarray(off_sample_means, dtype=np.float64)
    floors = np.asarray(off_sample_sigma_floors, dtype=np.float64)
    n = len(vals)
    checks = []
    for p in range(n):
        if not np.isfinite(vals[p]):
            checks.append(QualityCheck(
                "pedestal_drift", False, float("nan"), float(z_threshold),
                "skipped: no off-sample (never-lit-anywhere-in-the-scan) pixels available"))
            continue
        idx = []
        w = window
        while w <= max(n - 1, window):
            lo, hi = max(0, p - w), min(n, p + w + 1)
            idx = [i for i in range(lo, hi) if i != p and np.isfinite(vals[i])]
            if len(idx) >= 4 or (lo == 0 and hi == n):
                break
            w += 1
        if len(idx) < 4:
            checks.append(QualityCheck(
                "pedestal_drift", False, float("nan"), float(z_threshold),
                "skipped: fewer than 4 usable neighbouring points to build a local baseline"))
            continue
        loc = vals[idx]
        med = float(np.median(loc))
        mad = float(np.median(np.abs(loc - med)))
        floor = float(floors[p]) if np.isfinite(floors[p]) else 0.0
        sigma = max(1.4826 * mad, floor, 1e-9)
        z = (vals[p] - med) / sigma
        rel = abs(vals[p] - med) / max(abs(med), 1e-12)
        flagged = abs(z) > z_threshold and rel > min_relative_change
        direction = "high" if z > 0 else "low"
        msg = (f"off-sample background level {vals[p]:.3f} is {abs(z):.1f} sigma {direction} vs "
              f"a robust local baseline from its {len(idx)} nearest usable neighbouring points "
              f"(median {med:.3f}, sigma {sigma:.3f} = max(empirical {1.4826 * mad:.3f}, "
              f"Poisson floor {floor:.3f}); threshold {z_threshold:.1f} sigma, and a "
              f"change of {100 * rel:.3f} % vs the {100 * min_relative_change:.1f} % floor)")
        if flagged:
            msg += (" -- a whole-frame brightness anomaly (the off-sample background moved along "
                   "with, presumably, the real signal), not a rocking-curve feature (this check "
                   "never looks at the lit/signal region -- see this function's docstring)")
        checks.append(QualityCheck("pedestal_drift", flagged, float(z), float(z_threshold), msg))
    return checks


def _rollup(points, scan_checks, not_usable_fraction):
    """Verdict from POINT-level failures, never from flagged individual repeats alone: a scan whose
    only issue is one hot repeat per point (S996: repeat 0 reads 4.6 % high at ~half the points,
    the documented hot first repeat) is fully usable -- exclude that repeat -- and must not read
    "not usable" because every point "contains a flagged frame"."""
    n = len(points)
    n_bad = sum(1 for p in points if p.point_level_flagged)
    n_fr = sum(1 for p in points if p.flagged and not p.point_level_flagged)
    critical = any(c.flagged for c in scan_checks)
    if n == 0:
        return "not usable: no points"
    if critical or (n_bad / n) > not_usable_fraction:
        return f"not usable ({n_bad} of {n} points failed point-level checks{'; scan-level check failed' if critical else ''})"
    parts = []
    if n_bad:
        parts.append(f"{n_bad} of {n} points flagged")
    if n_fr:
        parts.append(f"{n_fr} of {n} points have flagged repeats only (excludable, see apply_quality_filter)")
    return "usable" + (", " + "; ".join(parts) if parts else "")


# --------------------------------------------------------------------------- gain / repeat access
def _repeat_groups(scan):
    """Dispatch a RawRepeatScan or RockingScan to ``(n_points, granularity, get_group,
    n_repeats_reported)`` -- the same dispatch :func:`assess_scan_quality` needs, shared with
    :func:`estimate_gain` so both work from any scan type the same way."""
    if isinstance(scan, RawRepeatScan):
        n_points, R = scan.frames.shape[:2]
        return n_points, "repeats", (lambda p: scan.frames[p]), R
    if isinstance(scan, RockingScan):
        n_points = scan.frames.shape[0]
        if scan.halves is not None:
            get_group = lambda p: np.stack([scan.halves[0][p], scan.halves[1][p]], 0)  # noqa: E731
            return n_points, "halves", get_group, scan.n_repeats
        get_group = lambda p: scan.frames[p][None]                                    # noqa: E731
        return n_points, "average-only", get_group, scan.n_repeats
    raise TypeError(f"needs a RawRepeatScan or a RockingScan; got {type(scan)!r}")


def estimate_gain(scan, *, min_repeats=2, n_pixels=4000, percentile_range=(1.0, 99.0),
                  n_bins=30, min_pixels_per_bin=20, min_r_squared=0.5, seed=0):
    """Estimate detector gain and additive read-noise variance from the scan's OWN
    repeat-to-repeat variance-vs-mean relationship (a standard photon-transfer-curve, PTC, fit),
    instead of assuming gain=1.0 (see the module docstring's item 2 -- a real measurement on the
    6-ID-C beamtime this module was built against gives var ~ 1.1-1.3*mean + 19-29, not gain=1).

    Convention
    ----------
    This module's ``gain`` is COUNTS PER PHOTON (see :func:`assess_scan_quality`), so for
    Poisson-distributed photon counts, ``var(counts) = gain * mean(counts) + read_noise_var`` --
    the SLOPE of a var-vs-mean regression is ``gain`` directly (the opposite of the more common
    electrons-per-count PTC convention, where gain is the reciprocal of the slope).

    Method
    ------
    For every scan point with at least ``min_repeats`` repeat exposures, a FIXED random subset of
    ``n_pixels`` pixel locations (same locations at every point, chosen once from the first
    qualifying point's frame shape) gives that point's own (median, robust-variance) pair per
    sampled pixel: the per-pixel MEDIAN and MAD-based robust variance across that point's repeats
    (the same robust statistic :func:`_point_lit_reference` already uses elsewhere in this
    module), not a plain sample mean/variance -- this down-weights the single repeat
    ``io_6idc.py`` documents as reading hot (+2.5-4.3% at higher repeat counts): a plain sample
    variance is pulled around by that one repeat, while a median of >= 3 values is not. Pairs are
    pooled across every qualifying point, trimmed to ``percentile_range`` of the pooled MEAN
    (drops empty corners and a few extreme pixels without needing to know in advance which pixels
    are defective), binned into ``n_bins`` quantile-spaced bins (median per bin -- another robust
    step), and fit with an ordinary least-squares line.

    Uniform random pixel sampling is used DELIBERATELY, not biased toward the brightest pixels: an
    earlier attempt at biasing the sample toward high-mean pixels (via a whole-scan-averaged
    brightness image) mostly re-selected the same static hot-pixel defects
    :func:`_static_hot_pixel_mask` hunts down elsewhere in this module -- pixels elevated in
    nearly every frame regardless of the rocking curve, which do not follow clean photon-transfer
    statistics and visibly corrupted the fit (tried and discarded during development on real S996
    data: the fitted slope rose from ~1.1-1.3, close to an independent PTC measurement on this
    beamtime's detector, to ~1.7 and swung around with the percentile-range cut instead of
    settling). Uniform sampling,
    pooled over every qualifying point -- which already spans a real range of illumination as the
    rocking curve itself rises and falls -- gave a stable fit and is used instead.

    A note on the intercept
    ------------------------
    This module's ``pedestal``/mean-frame values are RAW ADU counts (no bias/dark subtraction
    unless the caller passed ``dark=`` to :func:`load_6idc_repeat_frames`). If the detector has a
    non-negligible fixed bias baked into those raw counts, the fitted intercept equals
    ``read_noise_var - gain * bias`` and can come out NEGATIVE even though the true additive
    read-noise variance is positive -- the bias shifts the intercept along the mean axis, it does
    not change the slope (gain) estimate. This is the boring, mundane explanation for a negative
    fitted intercept (confirmed on real S996 data during development: a raw-ADU pedestal of
    ~100-115 counts, with the low-mean var-vs-mean plateau matching ``gain*(mean-bias)+read_noise``
    far better than ``gain*mean+read_noise``) -- not evidence of non-Poisson detector behaviour.
    ``read_noise_var`` is therefore clipped to ``>= 0.0`` before being returned for use as a noise
    floor; the raw (possibly negative) fitted intercept is kept in ``info["intercept_raw"]`` for
    inspection.

    Parameters
    ----------
    scan : RawRepeatScan or RockingScan with ``halves`` -- needs at least 2 repeats somewhere to
        measure anything; a plain average-only RockingScan (no repeat-parity split) cannot be
        used and always falls back (see ``info["fallback"]``).
    min_repeats : points with fewer repeats than this are skipped (default 2: even 2 repeats give
        one valid variance sample per pixel, just a noisier one -- see the docstring of
        :func:`assess_scan_quality` for how this trades off against ``min_repeats_for_robust_check``
        elsewhere in the module, which is deliberately higher).
    n_pixels : how many pixel locations to sample (default 4000; capped to the frame's pixel
        count for a small image).
    percentile_range, n_bins, min_pixels_per_bin : PTC binning controls, see Method above.
    min_r_squared : the fitted line must explain at least this fraction of the binned variances'
        own spread (default 0.5) or the fit is rejected as noise-fit-to-noise and gain falls back
        to 1.0 -- catches a photon-starved scan (no real signal dynamic range anywhere to resolve
        a slope from) passing every earlier check yet still returning a physically meaningless
        slope; see Method above and the module docstring's item 2 for the real S224 case this was
        found on (a naive fit gave gain=0.28, with no physical meaning). 0.5 is not an arbitrary
        round number: measured directly on real data, S224 (genuinely photon-starved) gives R^2 ~
        0.13-0.15, while S996 (a real, previously-analyzed, signal-bearing scan) gives R^2 ~ 0.77 --
        a first attempt at this gate used 0.8 and INCORRECTLY rejected S996's legitimate fit too;
        0.5 sits with wide margin in the gap between the two real measurements rather than being
        tuned to either one.
    seed : RNG seed for the pixel subsample (deterministic given the same scan shape).

    Returns
    -------
    gain : float -- falls back to 1.0 if there is not enough data to fit, the fit's slope is not
        positive, or the fit's R^2 is below ``min_r_squared`` (``info["fallback"]`` says which,
        ``info["notes"]`` says why).
    read_noise_var : float -- clipped to >= 0.0 (see "A note on the intercept" above); 0.0 on
        fallback.
    info : dict with ``n_points_used``, ``n_bins_used``, ``slope_raw``, ``intercept_raw``,
        ``fallback`` (bool), ``notes`` (list of str).
    """
    n_points, granularity, get_group, _ = _repeat_groups(scan)
    notes = []
    fallback_gain, fallback_rn = 1.0, 0.0

    def _fallback(reason, n_points_used=0, n_bins_used=0, slope_raw=None, intercept_raw=None):
        notes.append(f"{reason}; falling back to gain={fallback_gain:g}, "
                     f"read_noise_var={fallback_rn:g}")
        return fallback_gain, fallback_rn, dict(n_points_used=n_points_used,
                                                n_bins_used=n_bins_used, slope_raw=slope_raw,
                                                intercept_raw=intercept_raw, fallback=True,
                                                notes=notes)

    if granularity != "repeats":
        return _fallback(
            f"gain estimation needs true individual repeat exposures (granularity='repeats'); "
            f"got {granularity!r} -- an averaged/half-summed frame does not carry single-exposure "
            "photon-counting statistics, so its repeat-to-repeat variance would not measure the "
            "detector's actual per-exposure gain")

    rng = np.random.default_rng(seed)
    means_all, vars_all = [], []
    n_points_used = 0
    min_k_used = None
    rows = cols = None
    for p in range(n_points):
        grp = get_group(p)
        k = grp.shape[0]
        if k < min_repeats:
            continue
        if rows is None:
            H, W = grp.shape[1:]
            n_sub = min(n_pixels, H * W)
            idx = rng.choice(H * W, size=n_sub, replace=False)
            rows, cols = np.unravel_index(idx, (H, W))
        sub = grp[:, rows, cols].astype(np.float64)          # (k, n_sub)
        med = np.median(sub, axis=0)
        mad = np.median(np.abs(sub - med[None]), axis=0)
        var = (_mad_unbias_factor(k) * 1.4826 * mad) ** 2
        means_all.append(med)
        vars_all.append(var)
        n_points_used += 1
        min_k_used = k if min_k_used is None else min(min_k_used, k)

    if n_points_used < 2 or rows is None:
        return _fallback(f"gain estimation skipped: only {n_points_used} point(s) with >= "
                         f"{min_repeats} repeats (need >= 2)", n_points_used=n_points_used)

    means = np.concatenate(means_all)
    varss = np.concatenate(vars_all)
    sel = means > 0
    means, varss = means[sel], varss[sel]
    if means.size == 0:
        return _fallback("gain estimation skipped: no positive-mean pixel samples",
                         n_points_used=n_points_used)
    lo, hi = np.percentile(means, percentile_range)
    sel = (means >= lo) & (means <= hi)
    means, varss = means[sel], varss[sel]
    if means.size < min_pixels_per_bin * 3:
        return _fallback("gain estimation skipped: too few pixel samples survive trimming",
                         n_points_used=n_points_used)

    edges = np.quantile(means, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    bin_idx = np.digitize(means, edges) - 1
    bm, bv = [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() >= min_pixels_per_bin:
            bm.append(float(np.median(means[m])))
            bv.append(float(np.median(varss[m])))
    if len(bm) < 3:
        return _fallback("gain estimation skipped: too few populated bins for a regression",
                         n_points_used=n_points_used, n_bins_used=len(bm))

    bm_arr, bv_arr = np.asarray(bm), np.asarray(bv)
    slope, intercept = np.polyfit(bm_arr, bv_arr, 1)
    if not np.isfinite(slope) or slope <= 0:
        return _fallback(f"gain estimation gave a non-positive or non-finite slope ({slope!r})",
                         n_points_used=n_points_used, n_bins_used=len(bm), slope_raw=float(slope)
                         if np.isfinite(slope) else None, intercept_raw=float(intercept)
                         if np.isfinite(intercept) else None)

    # A slope needs real SIGNAL dynamic range to be resolvable from noise, not just "enough
    # points/bins": a photon-starved scan (e.g. S224, see NOTE_S224_frames.md) can easily pass
    # every check above -- plenty of points, plenty of populated bins -- while every pixel at
    # every point sits at essentially the SAME pedestal level, so the pooled (mean, variance)
    # pairs are pure scatter with no real trend to fit. Measured on real S224 data during
    # development: this fell all the way through to a fitted slope of 0.28 (vs. this beamtime's
    # detector reading ~1.1-1.3 or ~2.8 on other scans, see the module docstring) -- a number with
    # no physical meaning, not "gain is different here", that then made every downstream Poisson
    # floor too tight and inflated the static-hot-pixel count further. R^2 (how well the fitted
    # line actually explains the binned variances, vs. just using their own mean) catches this
    # directly: a real photon-transfer relationship fits the quantile-binned medians very well
    # (R^2 close to 1); noise fit to noise does not.
    ss_res = float(np.sum((bv_arr - (slope * bm_arr + intercept)) ** 2))
    ss_tot = float(np.sum((bv_arr - np.mean(bv_arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r_squared < min_r_squared:
        return _fallback(
            f"gain estimation gave a poor fit (R^2={r_squared:.2f} < {min_r_squared:g}): the "
            "pooled (mean, variance) pairs do not show a clear linear trend, most likely because "
            "this scan has too little real signal dynamic range to resolve a slope from noise "
            "(e.g. a photon-starved scan) rather than because the true gain is unusual",
            n_points_used=n_points_used, n_bins_used=len(bm), slope_raw=float(slope),
            intercept_raw=float(intercept))

    read_noise_var = max(intercept, 0.0)
    if intercept < 0:
        notes.append(f"fitted intercept ({intercept:.1f}) is negative -- consistent with a "
                     "raw-ADU bias offset not subtracted from the frames (see docstring "
                     "'A note on the intercept'); clipped to 0.0 for use as a noise floor")
    if min_k_used is not None and min_k_used <= 2:
        notes.append(
            f"every qualifying point had only {min_k_used} repeat(s): with just 2 samples, real "
            "repeat-to-repeat differences (e.g. a genuine few-percent flux drift between repeats, "
            "as documented for some 6-ID-C scans -- see NOTE_S168_frames.md) are indistinguishable "
            "from shot noise and inflate this estimate; treat a 2-repeat gain estimate as an "
            "order-of-magnitude check that gain != 1, not a precise photon-transfer measurement -- "
            "prefer a same-beamtime scan with more repeats, or an explicit gain= override, when "
            "precision matters")
    return float(slope), float(read_noise_var), dict(
        n_points_used=n_points_used, n_bins_used=len(bm), slope_raw=float(slope),
        intercept_raw=float(intercept), fallback=False, notes=notes)


# --------------------------------------------------------------------------- main API
def assess_scan_quality(
    scan, *, gain: Optional[float] = None, read_noise_var: Optional[float] = None,
    saturation_value: Optional[float] = 65535.0,
    saturation_min_count: int = 1, count_z_threshold: float = 6.0,
    min_repeats_for_robust_check: int = 4, spike_zscore: float = 8.0,
    max_spike_pixels: int = 4, min_spike_excess: float = 10.0,
    dead_variance_ratio: float = 0.1, duplicate_atol: float = 1e-3,
    duplicate_rtol: float = 1e-5, flat_variance_floor: float = 1e-6,
    baseline_percentile: float = 25.0, lit_sigma_mult: float = 5.0, min_snr: float = 10.0,
    min_survive_fraction: float = 0.5, step_outlier_frac: float = 0.3,
    not_usable_fraction: float = 0.5, max_spike_fraction: float = 1e-5,
    max_edge_over_peak: float = 0.6, min_peak_signal_fraction: float = 1e-3,
    min_repeats_for_gain: int = 2, gain_n_pixels: int = 4000,
    gain_percentile_range=(1.0, 99.0), gain_n_bins: int = 30,
    hot_pixel_significance: float = 1e-6, hot_pixel_min_hits: int = 2,
    hot_pixel_max_frames: Optional[int] = 400, max_hot_pixel_fraction: float = 0.02,
    pedestal_drift_window: int = 4, pedestal_drift_zscore: float = 4.0,
    pedestal_drift_min_relative: float = 0.005,
) -> ScanQualityReport:
    """Assess a DFXM rocking scan's usability BEFORE any reduction. Never modifies ``scan``.

    Parameters
    ----------
    scan : :class:`RawRepeatScan` or :class:`midas_dfxm.rocking.RockingScan`
        A ``RawRepeatScan`` (e.g. from :func:`load_6idc_repeat_frames`) gets the full three-tier
        report, with true frame-level checks on every individual repeat exposure. A plain
        ``RockingScan`` (repeats already averaged -- see the module docstring) gets a DEGRADED
        report: frame-level checks run on the 2-way ``scan.halves`` split if present (coarser
        than individual repeats), or are skipped entirely if not; the point-level flux check and
        every scan-level check still run in full, since they only need the per-point average.
        ``report.granularity`` records which mode ran.
    gain : detector gain, counts per photon (see :func:`estimate_gain` for the convention).
        Default ``None``: AUTO-ESTIMATE from the scan's own repeat-to-repeat variance-vs-mean
        relationship via :func:`estimate_gain`, rather than silently assuming 1.0 -- a real
        photon-transfer measurement on the 6-ID-C beamtime this module targets gives gain
        ~1.1-1.3, not 1 (see the module docstring). Pass an explicit float to override the
        auto-estimate with a known value (in which case ``read_noise_var`` also defaults to 0.0
        unless given explicitly -- a manual ``gain=`` override does not get a free read-noise
        estimate). The estimate actually used (auto or given) is recorded in
        ``report.settings['gain']``, and if auto-estimated, the fit diagnostics are in
        ``report.settings['gain_info']``.
    read_noise_var : additive per-pixel read-noise variance (same units as ``gain``'s Poisson
        term). Default ``None``: auto-estimated alongside ``gain`` when ``gain`` is also ``None``;
        0.0 if ``gain`` is given explicitly and this is not.
    saturation_value : ADU value at or above which a pixel is saturated (default 65535.0, a
        16-bit ceiling -- e.g. 6-ID-C's Andor/Zyla). ``None`` disables the check.
    count_z_threshold : a repeat's lit-region total must differ from its point's OTHER repeats by
        more than this many robust-sigma to be flagged (default 6.0: with typically 4-20
        repeats/point and a MAD-based sigma that is itself noisy at small sample sizes, 6 sigma
        keeps the false-positive rate low while still catching the kind of effect
        ``drop_first_repeat`` looks for -- io_6idc.py's own indexed-layout note measures a hot
        first repeat at +2.5-4.3%, which for a well-exposed point is many robust-sigma).
    min_repeats_for_robust_check : count-anomaly and dead-frame checks need enough repeats for a
        median/MAD to mean anything; below this (default 4) they are skipped with a note rather
        than run on too little data (2 repeats gives a degenerate, symmetric comparison that
        cannot tell which of the two is the outlier).
    spike_zscore, max_spike_pixels, min_spike_excess : a candidate spike pixel must be more than
        ``spike_zscore`` local (8-neighbour) robust-sigma above its neighbourhood AND more than
        ``min_spike_excess`` counts above it (default 8.0 sigma, 10 counts: strict, because a
        real DFXM peak also has locally elevated pixels and the check must not fire on it -- see
        the "distinct from a smooth PSF peak" requirement). It is then only called a spike if its
        connected cluster is <= ``max_spike_pixels`` (default 4): a real PSF feature covers many
        more contiguous pixels than a single cosmic-ray hit.
    dead_variance_ratio : a repeat's pixel variance below this fraction (default 0.1, i.e. 10x
        lower structure) of its sibling repeats' median variance is flagged as dead/frozen.
    duplicate_atol, duplicate_rtol : two repeats within this tolerance of each other (default
        ``np.allclose`` defaults tightened to 1e-3 absolute / 1e-5 relative) are flagged as a
        duplicated frame, independent of the variance-ratio check and regardless of repeat count.
    flat_variance_floor : a repeat with variance below this (default 1e-6) is flagged as dead
        regardless of its siblings -- an outright blank/frozen frame.
    baseline_percentile, lit_sigma_mult : how a point's own pedestal and lit mask are estimated
        from its own repeats (default: 25th percentile pedestal, lit = more than 5 sigma above
        it, sigma from repeat-to-repeat scatter floored by a Poisson estimate).
    min_snr : the point-level photon-budget floor (default 10.0, i.e. >= 100 net lit-region
        counts at gain=1): an ABSOLUTE per-point detectability threshold, not a comparison to
        neighbouring points, because a uniformly photon-starved scan (every point equally
        starved, e.g. S224 at 0.05 s exposure -- see NOTE_S224_frames.md, whose own flux gate
        could not recover a known +3% perturbation on 3 of 4 probed points) would pass any
        purely relative check. SNR = 10 is a common, defensible generic detection floor (1%
        relative shot-noise error); tune per detector/reflection.
    min_survive_fraction : a point is flagged if fewer than this fraction of its repeats survive
        frame-level checks (default 0.5, per the design brief: "flag if too few remain, e.g.
        more than half").
    step_outlier_frac : a scan-axis step deviating from the median step by more than this
        fraction (default 0.3) is flagged.
    not_usable_fraction : the scan rollup reads "not usable" once more than this fraction
        (default 0.5) of points are flagged, or any scan-level check fails.
    min_repeats_for_gain, gain_n_pixels, gain_percentile_range, gain_n_bins : passed through to
        :func:`estimate_gain` (as ``min_repeats``, ``n_pixels``, ``percentile_range``, ``n_bins``)
        when ``gain`` is auto-estimated; ignored if ``gain`` is given explicitly.
    hot_pixel_significance, hot_pixel_min_hits, hot_pixel_max_frames, max_hot_pixel_fraction :
        passed through to :func:`_static_hot_pixel_mask` / :func:`_check_static_hot_pixels` -- a
        pixel tripping the per-frame spike criterion at the SAME location often enough that pure
        chance (a cosmic ray coincidentally landing there more than once, given the BACKGROUND
        rate left over once already-obvious defects are set aside -- an iteratively self-consistent
        estimate, not the scan's raw overall rate; see that function's docstring for why the naive
        version of this undercorrected on real data) would explain it with probability below
        ``hot_pixel_significance`` (default 1e-6; see :func:`_poisson_min_recurrence`) is treated
        as a fixed detector defect and excluded from the per-frame spike check for the rest of the
        report, floored at ``hot_pixel_min_hits`` (default 2 -- the same location trips twice)
        absolute hits regardless of how low that computed threshold would otherwise be.
        ``hot_pixel_max_frames`` (default 400; ``None`` uses every frame) bounds runtime on a very
        large scan by subsampling frame indices for this pass. The
        resulting static-defect population is itself flagged as suspicious if it exceeds
        ``max_hot_pixel_fraction`` (default 2%) of all pixels, which is more consistent with a
        wrong gain/threshold than with a real hot-pixel map.
    max_edge_over_peak, min_peak_signal_fraction : scan-level peak-bracketing check (defaults 0.6
        and 1e-3; see :func:`_check_curve_bracketing`).
    max_spike_fraction : fraction of a frame's pixels that may be isolated transient spikes before
        the frame is flagged (default 1e-5; see :func:`_check_spike`).
    pedestal_drift_window, pedestal_drift_zscore : passed through to
        :func:`_check_pedestal_drift` -- the point-vs-neighbours whole-frame-brightness check
        built from each point's off-sample (never-lit-anywhere-in-the-scan) background level
        (default window 4 points each side -- 8 lets slow pedestal drift inflate the local sigma and
        drops S168's real defect from z=11.7 to 3.7 --, threshold 4.0 robust-sigma and 0.5 % effect-size floor; see that function's
        docstring for why this is NOT built from the lit/signal region, and for the two retracted
        attempts that were).

    Returns
    -------
    ScanQualityReport
    """
    n_points, granularity, get_group, n_repeats_reported = _repeat_groups(scan)

    gain_info = None
    if gain is None:
        gain, auto_read_noise_var, gain_info = estimate_gain(
            scan, min_repeats=min_repeats_for_gain, n_pixels=gain_n_pixels,
            percentile_range=gain_percentile_range, n_bins=gain_n_bins)
        if read_noise_var is None:
            read_noise_var = auto_read_noise_var
    elif read_noise_var is None:
        read_noise_var = 0.0

    hot_mask, hot_frac, hot_hits, n_frames_sampled, hot_threshold_k = _static_hot_pixel_mask(
        get_group, n_points, gain, spike_zscore, min_spike_excess, hot_pixel_significance,
        hot_pixel_min_hits, read_noise_var=read_noise_var, max_frames=hot_pixel_max_frames)
    n_total_px = int(hot_mask.size)

    off_sample = None
    if pedestal_drift_window is not None:
        off_sample = _off_sample_mask(get_group, n_points, gain, baseline_percentile,
                                      lit_sigma_mult, read_noise_var=read_noise_var)

    frame_kwargs = dict(saturation_value=saturation_value,
                        saturation_min_count=saturation_min_count,
                        count_z_threshold=count_z_threshold,
                        min_repeats_for_robust_check=min_repeats_for_robust_check,
                        spike_zscore=spike_zscore, max_spike_pixels=max_spike_pixels,
                        min_spike_excess=min_spike_excess,
                        dead_variance_ratio=dead_variance_ratio,
                        duplicate_atol=duplicate_atol, duplicate_rtol=duplicate_rtol,
                        flat_variance_floor=flat_variance_floor, hot_mask=hot_mask,
                        max_spike_fraction=max_spike_fraction)

    points = []
    for p in range(n_points):
        motors_row = {k: float(v[p]) for k, v in scan.motors.items()}
        if scan.axes:
            label = ", ".join(f"{a}={scan.motors[a][p]:.4f}" for a in scan.axes)
        else:
            label = f"point {p}"
        points.append(_assess_point(
            p, get_group(p), motors_row, label, granularity, gain=gain,
            baseline_percentile=baseline_percentile, lit_sigma_mult=lit_sigma_mult,
            min_snr=min_snr, min_survive_fraction=min_survive_fraction,
            frame_kwargs=frame_kwargs, read_noise_var=read_noise_var, off_sample_mask=off_sample))

    if pedestal_drift_window is not None:
        drift_checks = _check_pedestal_drift(
            [pt.off_sample_mean for pt in points],
            [pt.off_sample_sigma_floor for pt in points],
            pedestal_drift_window, pedestal_drift_zscore, pedestal_drift_min_relative)
        for pt, c in zip(points, drift_checks):
            pt.checks.append(c)

    scan_checks = [
        _check_frame_table_consistency(n_points, n_repeats_reported, scan.motors),
        _check_step_regularity(scan.coordinate, step_outlier_frac),
        _check_static_hot_pixels(hot_mask, n_total_px, n_frames_sampled, hot_threshold_k,
                                 hot_pixel_significance, max_hot_pixel_fraction),
        _check_curve_bracketing(points, scan.coordinate, max_edge_over_peak,
                                min_peak_signal_fraction),
    ]
    verdict = _rollup(points, scan_checks, not_usable_fraction)
    notes = list(scan.notes)
    if granularity == "halves":
        notes.append("DEGRADED granularity: input was a RockingScan whose individual repeats "
                     "are not retained (see module docstring). Frame-level checks ran on the "
                     "2-way even/odd repeat-parity split only, not on every individual "
                     "exposure -- pass a RawRepeatScan (e.g. from load_6idc_repeat_frames) for "
                     "the full per-exposure report.")
    elif granularity == "average-only":
        notes.append("DEGRADED granularity: input was a RockingScan with no repeat-parity "
                     "split at all (scan.halves is None, e.g. n_repeats <= 1). Frame-level "
                     "checks did not run; only the point-level flux check and scan-level "
                     "checks are meaningful here.")
    if gain_info is not None:
        notes.extend(f"gain estimation: {n}" for n in gain_info["notes"])
        notes.append(f"gain auto-estimated from the scan's own data: gain={gain:.4g} counts/"
                     f"photon, read_noise_var={read_noise_var:.4g} (see report.settings"
                     "['gain_info'] for the fit diagnostics; pass gain= explicitly to override)")
    if granularity == "repeats":
        by_rep = {}
        n_pts_fl = 0
        for pt in points:
            fl = [f.repeat for f in pt.frames if f.flagged]
            n_pts_fl += bool(fl)
            for r in fl:
                by_rep[r] = by_rep.get(r, 0) + 1
        tot = sum(by_rep.values())
        if tot and n_pts_fl >= 0.2 * len(points):
            r_top, n_top = max(by_rep.items(), key=lambda kv: kv[1])
            if n_top >= 0.8 * tot:
                notes.append(f"SYSTEMATIC: {n_top} of {tot} flagged repeats are repeat index {r_top}, "
                             f"at {n_pts_fl} of {len(points)} points -- a per-acquisition effect, not "
                             "random bad frames. For repeat index 0 this is the documented hot first "
                             "repeat (io_6idc drop_first_repeat); exclude it or use "
                             "apply_quality_filter.")
    settings = dict(gain=gain, read_noise_var=read_noise_var, gain_info=gain_info,
                    saturation_value=saturation_value,
                    count_z_threshold=count_z_threshold, spike_zscore=spike_zscore,
                    max_spike_pixels=max_spike_pixels, dead_variance_ratio=dead_variance_ratio,
                    min_snr=min_snr, min_survive_fraction=min_survive_fraction,
                    step_outlier_frac=step_outlier_frac, not_usable_fraction=not_usable_fraction,
                    hot_pixel_significance=hot_pixel_significance,
                    n_static_hot_pixels=int(hot_mask.sum()), hot_pixel_threshold_k=hot_threshold_k,
                    pedestal_drift_window=pedestal_drift_window,
                    pedestal_drift_zscore=pedestal_drift_zscore)
    return ScanQualityReport(points=points, scan_checks=scan_checks, verdict=verdict,
                             granularity=granularity, settings=settings, notes=notes)


def apply_quality_filter(scan: RawRepeatScan, report: ScanQualityReport, *,
                         drop_flagged_points: bool = False) -> RockingScan:
    """Build a repeat-filtered :class:`RockingScan` from ``scan`` and its :func:`assess_scan_quality`
    report. NOT called automatically -- :func:`assess_scan_quality` only reports; this is the
    separate, explicit opt-in step that acts on the report. ``scan`` and ``report`` are never
    modified.

    Excludes exactly the repeats flagged in ``report`` (by any frame-level check) when averaging
    each point, generalising :func:`midas_dfxm.io_6idc.load_6idc_scan`'s ``drop_first_repeat``
    (which only ever compares repeat 0 to the others, for the whole scan) to any repeat of any
    point. A point with EVERY repeat flagged falls back to averaging all of them anyway (nothing
    else to average), noted in the result rather than silently producing a NaN point.

    Parameters
    ----------
    scan : the RawRepeatScan the report was computed from.
    report : the ScanQualityReport from :func:`assess_scan_quality` (``report.granularity`` must
        be ``"repeats"`` -- a degraded report has no individual repeats to exclude).
    drop_flagged_points : if True, ALSO remove whole points flagged at the point level (not just
        individual repeats) from the returned scan entirely -- a more consequential action
        (it shrinks the rocking curve itself), so it defaults to off; the report already told the
        caller which points to re-measure.

    Returns
    -------
    RockingScan
    """
    if report.granularity != "repeats":
        raise ValueError(f"apply_quality_filter needs a report with granularity='repeats' "
                         f"(individual repeat exposures); got {report.granularity!r}. A "
                         "degraded report has no repeat-level information to filter on.")
    n_points, R = scan.frames.shape[:2]
    mask = np.ones((n_points, R), dtype=bool)
    n_excluded = 0
    for p in report.points:
        for f in p.frames:
            if f.flagged:
                mask[f.point, f.repeat] = False
                n_excluded += 1
    fallback_points = [p for p in range(n_points) if not mask[p].any()]
    for p in fallback_points:
        mask[p] = True
    notes = list(scan.notes)
    notes.append(f"quality filter applied: {n_excluded} of {n_points * R} repeat(s) excluded "
                f"across {n_points} points (report verdict: {report.verdict})")
    if fallback_points:
        notes.append(f"{len(fallback_points)} point(s) had every repeat flagged; kept all "
                     f"repeats there rather than average zero frames (points: "
                     f"{fallback_points[:10]}{', ...' if len(fallback_points) > 10 else ''}) "
                     "-- re-measure these points rather than trust their average")
    rs = scan.to_rocking_scan(repeat_mask=mask)
    rs.notes.extend(notes[len(scan.notes):])
    if drop_flagged_points:
        keep = np.array([p.point for p in report.points if not p.flagged], dtype=int)
        if keep.size == 0:
            raise ValueError("drop_flagged_points=True would remove every point; refusing")
        dropped = n_points - keep.size
        motors = {k: v[keep] for k, v in rs.motors.items()}
        rs = RockingScan.from_arrays(
            rs.frames[keep], motors,
            halves=None if rs.halves is None else tuple(h[keep] for h in rs.halves),
            n_repeats=rs.n_repeats, dark_level=rs.dark_level, source=rs.source,
            notes=rs.notes + [f"drop_flagged_points: removed {dropped} of {n_points} "
                              "point(s) flagged at the point level"],
            meta=rs.meta)
    return rs


if __name__ == "__main__":                                                   # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Assess a 6-ID-C DFXM rocking scan's quality before reduction.")
    parser.add_argument("scan_dir", help="folder of data_NNNNN.tif frames (indexed layout)")
    parser.add_argument("--motor-table", default=None)
    parser.add_argument("--roi", nargs=4, type=int, default=None,
                        metavar=("ROW0", "ROW1", "COL0", "COL1"))
    args = parser.parse_args()
    roi = tuple(args.roi) if args.roi else None
    raw = load_6idc_repeat_frames(args.scan_dir, args.motor_table, roi=roi)
    print(f"loaded {raw.source}: {raw.n_points} points x {raw.n_repeats} repeats")
    report = assess_scan_quality(raw)
    print(report.summary())
