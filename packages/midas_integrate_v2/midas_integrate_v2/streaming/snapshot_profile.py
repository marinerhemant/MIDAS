"""Per-frame diagnostics for still-frame (no rotation) time series.

For each frame (or sum of frames):

* :func:`ring_profile` -- mean intensity per scattering-angle bin. The MEAN is
  deliberate: at a few ms per frame most pixels hold 0 or 1 count, and a median
  rails at exactly those values.
* :func:`fit_matrix_scale` -- one isotropic scale ``s`` of a known reference
  phase ("matrix") from the centroids of its rings: ``d_obs = s * d_ref``. The
  scale is the per-frame expansion (a relative thermometer when temperature is
  the only driver; thermal and uniform elastic dilatation are indistinguishable
  from peak positions alone).
* :func:`sharp_peaks` -- sharp lines above a robust background (a halo is not a
  line), e.g. to list lines that no known phase explains.
* :func:`band_excess` -- mean counts in one angular band minus another (e.g. a
  diffuse-halo band against a quiet band): a melt / amorphous indicator.
* :func:`windows_from_trace` -- a fixed, stated rule to pick "before" and
  "after" windows from a halo-type trace, so windows are chosen from traces only,
  before any spot is looked at.

No material knowledge lives here: reference d-spacings are inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = ["ring_profile", "MatrixFit", "fit_matrix_scale", "sharp_peaks", "band_excess",
           "Windows", "windows_from_trace"]


def ring_profile(img: np.ndarray, ok: np.ndarray, tth_deg: np.ndarray, step_deg: float,
                 tth0: Optional[float] = None, min_pixels: int = 30
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(bin centres, mean counts per pixel) over valid pixels."""
    t0 = float(np.nanmin(tth_deg[ok])) if tth0 is None else tth0
    b = np.floor((tth_deg - t0) / step_deg).astype(np.int64)
    valid = ok & np.isfinite(tth_deg) & (b >= 0)
    nb = int(b[valid].max()) + 1
    num = np.bincount(b[valid], weights=img[valid].astype(np.float64), minlength=nb)
    cnt = np.bincount(b[valid], minlength=nb)
    prof = np.where(cnt >= min_pixels, num / np.maximum(cnt, 1), np.nan)
    return t0 + (np.arange(nb) + 0.5) * step_deg, prof


def _tth(d, lam):
    return np.degrees(2 * np.arcsin(np.clip(lam / (2 * np.asarray(d, float)), -1, 1)))


@dataclass
class MatrixFit:
    scale_coarse: float
    scale: float                       # nan when fewer than min_rings rings were usable
    per_ring_scale: dict = field(default_factory=dict)   # index into d_ref -> scale
    n_rings: int = 0
    n_lines: int = 0                   # reference lines whose predicted ring lies inside the profile
    scale_spread: float = float("nan")  # 1.4826 x MAD of per-ring scales (relative)

    @property
    def completeness(self) -> float:
        return self.n_rings / self.n_lines if self.n_lines else 0.0

    def consistent(self, spread_tol: float = 0.003, min_rings: int = 3) -> bool:
        """A real phase gives the same scale from every ring; a pattern that merely
        catches neighbouring peaks of other phases does not (per-ring scales scatter)."""
        return bool(self.n_rings >= min_rings and np.isfinite(self.scale)
                    and np.isfinite(self.scale_spread) and self.scale_spread <= spread_tol)


def _robust_background(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray, deg: int = 2,
                       n_iter: int = 4, clip: float = 2.5) -> Optional[np.ndarray]:
    """Polynomial background through (x, y) with positive outliers (rings) clipped
    iteratively; evaluated at ``x_eval``. None if too few points."""
    keep = np.isfinite(y)
    if keep.sum() < deg + 4:
        return None
    for _ in range(n_iter):
        c = np.polyfit(x[keep], y[keep], deg)
        r = y - np.polyval(c, x)
        s = 1.4826 * float(np.median(np.abs(r[keep]))) + 1e-12
        new = np.isfinite(y) & (r < clip * s)
        if new.sum() < deg + 4 or np.array_equal(new, keep):
            break
        keep = new
    return np.polyval(c, x_eval)


def _smooth(y: np.ndarray, n: int) -> np.ndarray:
    """Boxcar of n bins (edges shrink the kernel). On a coarse detector binned finely the
    profile is a comb of filled and empty bins; peak finding needs it smoothed."""
    if n <= 1:
        return y
    k = np.ones(n)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")


def _half_max_run(sig: np.ndarray, i: int) -> tuple:
    """Index range [j, k] of the contiguous run above half maximum around index i."""
    hm = sig[i] / 2
    j, k = i, i
    while j > 0 and sig[j - 1] > hm:
        j -= 1
    while k < len(sig) - 1 and sig[k + 1] > hm:
        k += 1
    return j, k


def _fwhm(sig: np.ndarray, i: int, step: float) -> float:
    """Full width at half maximum of the contiguous run around index i."""
    j, k = _half_max_run(sig, i)
    return (k - j + 1) * step


def fit_matrix_scale(tc: np.ndarray, prof: np.ndarray, d_ref: Sequence[float], lam: float,
                     scale_grid: np.ndarray = np.arange(0.98, 1.05, 0.0002),
                     window_deg: float = 0.06, min_peak: Optional[float] = None,
                     min_peak_sigma: float = 8.0, min_rings: int = 3,
                     bg_mult: float = 3.0, max_fwhm: Optional[float] = None,
                     ring_tol: float = 0.004, ring_tol_deg: float = 0.0) -> MatrixFit:
    """Fit one isotropic scale of the reference d-lines to a ring profile.

    Coarse: the grid scale maximising the summed profile at the predicted ring
    positions. Fine: per ring, the background-subtracted centroid inside
    +/- ``window_deg`` -> per-ring scale; the median over usable rings is returned.

    Background: a robust quadratic through the flanks ``window_deg`` to
    ``bg_mult * window_deg`` either side (rings in the flanks clipped), then a line
    through the window edges for what the quadratic leaves. A straight line alone is
    not enough: the top of a broad halo then passes as a ring. What separates a halo
    from a ring in the end is the width test (ii).

    A ring is usable if (i) its peak above background exceeds ``min_peak_sigma``
    times the profile noise (robust scatter of bin-to-bin differences) and
    ``min_peak`` if given -- relative on purpose: an absolute floor rejects every
    ring of an attenuated or short-exposure series although the rings are tens of
    sigma significant; and (ii) it is sharp: FWHM <= ``max_fwhm`` (default
    ``window_deg``). Finally rings whose scale differs from the median of the others
    by more than ``ring_tol`` (relative) are dropped one at a time, worst first: with
    three rings a MAD spread ignores one arbitrary ring, so a window that merely
    caught some other line must not count. (Rings of one phase agree within a few
    0.1 %; a ring off by more has usually caught an overlapping line of another phase.)
    Per ring the tolerance is the larger of ``ring_tol`` and ``ring_tol_deg`` of 2theta
    expressed in d at that ring (pass one pixel's 2theta: on a coarse detector a
    one-pixel centroid error is more than 0.4 % in d at low angle).
    """
    d_ref = np.asarray(d_ref, float)
    ok = np.isfinite(prof)
    p = np.where(ok, prof, 0.0)
    fin = prof[ok]
    noise = 1.4826 * float(np.median(np.abs(np.diff(fin)))) / np.sqrt(2) if fin.size > 2 else 0.0
    thresh = max(min_peak or 0.0, min_peak_sigma * noise)
    wmax = window_deg if max_fwhm is None else max_fwhm
    step = float(np.median(np.diff(tc))) if len(tc) > 1 else window_deg
    nsm = max(1, int(round(wmax / step / 4)))
    score = [np.interp(_tth(s * d_ref, lam), tc, p).sum() for s in scale_grid]
    s0 = float(scale_grid[int(np.argmax(score))])
    per = {}
    n_lines = 0
    for i, t in enumerate(_tth(s0 * d_ref, lam)):
        m = (tc > t - window_deg) & (tc < t + window_deg) & ok
        if m.sum() < 8:
            continue
        n_lines += 1
        fl = (np.abs(tc - t) >= window_deg) & (np.abs(tc - t) <= bg_mult * window_deg)
        x, y = tc[m], prof[m]
        bg = _robust_background(tc[fl], prof[fl], x)
        if bg is None:
            continue
        sgn = y - bg
        # the quadratic leaves a plateau under a halo top: remove it with a line through
        # the window edges (the width test below then separates a halo from a ring)
        e = np.r_[0:3, len(x) - 3:len(x)]
        sgn = sgn - np.polyval(np.polyfit(x[e], sgn[e], 1), x)
        sm = _smooth(sgn, nsm)
        k = int(np.argmax(sm))
        if sm[k] <= thresh or _fwhm(sm, k, step) > wmax:
            continue
        # centroid over the whole window (after the edge-line correction the residual is ~0
        # at the edges): on a spotty ring it averages every spot, not only the strongest
        wgt = np.clip(sgn, 0, None)
        tpk = float((x * wgt).sum() / wgt.sum())
        d_obs = lam / (2 * np.sin(np.radians(tpk / 2)))
        per[i] = d_obs / d_ref[i]
    th = np.radians(_tth(s0 * d_ref, lam) / 2)
    tol = np.maximum(ring_tol, np.radians(ring_tol_deg) / 2 / np.tan(th))
    while len(per) >= 2:
        v = np.array(list(per.values()))
        dev = {k: abs(sc / np.median(np.delete(v, j)) - 1) / tol[k] for j, (k, sc) in enumerate(per.items())}
        worst = max(dev, key=dev.get)
        if dev[worst] <= 1.0:
            break
        del per[worst]
    v = np.array(list(per.values()))
    s = float(np.median(v)) if len(per) >= min_rings else float("nan")
    spread = float(1.4826 * np.median(np.abs(v / np.median(v) - 1))) if v.size >= 2 else float("nan")
    return MatrixFit(s0, s, per, len(per), n_lines, spread)


def sharp_peaks(tc: np.ndarray, prof: np.ndarray, *, bg_half: float = 0.4, min_sigma: float = 8.0,
                max_fwhm: float = 0.08) -> list:
    """Sharp lines of a profile: local maxima more than ``min_sigma`` x profile noise
    above a robust quadratic background over +/- max(``bg_half``, 4 ``max_fwhm``) (so a
    diffuse halo is not a line), with FWHM <= ``max_fwhm``; lines closer than
    ``max_fwhm`` / 2 are one (the strongest). Conservative: a line on a halo top can be
    missed. Returns dicts (tth, height, snr, fwhm)."""
    ok = np.isfinite(prof)
    fin = prof[ok]
    if fin.size < 10:
        return []
    bg_half = max(bg_half, 4 * max_fwhm)
    noise = 1.4826 * float(np.median(np.abs(np.diff(fin)))) / np.sqrt(2) + 1e-12
    step = float(np.median(np.diff(tc)))
    nsm = max(1, int(round(max_fwhm / step / 4)))
    idx = np.arange(len(prof))
    filled = np.interp(idx, idx[ok], prof[ok])
    sp = _smooth(filled, nsm)
    # candidates: local maxima of the profile minus a wide moving average, so a ring on
    # the steep flank of a halo (not a maximum of the profile itself) is still found
    hp = sp - _smooth(filled, max(3, int(round(4 * max_fwhm / step))))
    y = np.where(ok & (hp > 0), hp, -np.inf)
    h = max(2, nsm)
    cand = [i for i in range(h, len(y) - h) if ok[i] and y[i] >= np.max(y[i - h:i + h + 1])]
    out = []
    for i in cand:
        w = (np.abs(tc - tc[i]) <= bg_half) & ok
        bg = _robust_background(tc[w], sp[w], tc[w])
        if bg is None:
            continue
        sig = sp[w] - bg
        j = int(np.nonzero(np.nonzero(w)[0] == i)[0][0])
        # local noise: bins at the detector corners hold few pixels and are noisier
        raw = prof[w]
        # (second differences: a halo's slope is not noise)
        nloc = max(noise, 1.4826 * float(np.median(np.abs(np.diff(raw, 2)))) / np.sqrt(6)) / np.sqrt(nsm)
        if sig[j] < min_sigma * nloc:
            continue
        fw = _fwhm(sig, j, step)
        if fw > max_fwhm:
            continue
        lo, hi = max(j - 3, 0), min(j + 4, len(sig))
        pos = np.clip(sig[lo:hi], 0, None)
        tpk = float((tc[w][lo:hi] * pos).sum() / pos.sum()) if pos.sum() > 0 else float(tc[i])
        out.append(dict(tth=tpk, height=float(sig[j]), snr=float(sig[j] / nloc), fwhm=float(fw)))
    kept = []                                              # one entry per line: strongest wins
    for p in sorted(out, key=lambda q: -q["height"]):
        if all(abs(p["tth"] - q["tth"]) > max_fwhm / 2 for q in kept):
            kept.append(p)
    return sorted(kept, key=lambda q: q["tth"])


def band_excess(tc: np.ndarray, prof: np.ndarray, band: tuple, base: tuple) -> float:
    """Mean profile in ``band`` minus mean in ``base`` (angles in the units of tc)."""
    mb = (tc > band[0]) & (tc < band[1])
    m0 = (tc > base[0]) & (tc < base[1])
    return float(np.nanmean(prof[mb]) - np.nanmean(prof[m0]))


@dataclass
class Windows:
    before: tuple            # inclusive frame range
    after: tuple
    onset: int
    baseline: float
    rule: str


def windows_from_trace(first_frame: np.ndarray, trace: np.ndarray, n_frames: int,
                       baseline_until: int = 200, rise: Optional[float] = None, guard: int = 50,
                       after_len: int = 600, min_before: int = 200,
                       rise_sigma: float = 6.0) -> Optional[Windows]:
    """Before/after windows from a trace (e.g. per-window band excess).

    baseline = median of the trace for windows starting before ``baseline_until``;
    onset = first window whose trace exceeds baseline + rise, where rise is ``rise``
    if given, else ``rise_sigma`` times the robust scatter of the baseline windows
    (relative, so attenuated or short-exposure series are treated like bright ones);
    before = [0, onset - guard]; after = the last ``after_len`` frames.
    Returns None when there is no onset or the before window is shorter than
    ``min_before`` frames. The rule is stated in the result so it can be reported.
    """
    first_frame = np.asarray(first_frame)
    trace = np.asarray(trace, float)
    bw = trace[first_frame < baseline_until]
    base = float(np.nanmedian(bw))
    if rise is None:
        mad = 1.4826 * float(np.nanmedian(np.abs(bw - base))) if bw.size else 0.0
        rise = max(rise_sigma * mad, 1e-12)
    on = np.flatnonzero(trace > base + rise)
    if len(on) == 0:
        return None
    onset = int(first_frame[on[0]])
    before = (0, onset - guard)
    if before[1] - before[0] < min_before:
        return None
    rule = (f"baseline=median(trace[first<{baseline_until}]); onset=first trace>baseline+{rise:.4g}; "
            f"before=[0,onset-{guard}]; after=last {after_len} frames")
    return Windows(before, (n_frames - after_len, n_frames - 1), onset, base, rule)
