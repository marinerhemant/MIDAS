"""1-D lineout fitting, as an adapter over ``midas_peakfit``.

``midas_peakfit`` fits pseudo-Voigt profiles over 2-D detector regions for
FF-HEDM. XRD-CT wants the same profile over a 1-D radial lineout, so this is
an adapter -- constant eta, radii along the lineout -- not a second fitter.

The output is the canonical 12-channel vector from
:data:`~midas_dt.conventions.FIT_OUTPUT_NAMES`, in the C's order, so results
are directly comparable with 2023 output read by index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .conventions import FIT_OUTPUT_NAMES, LEGACY_WIDTHS_ARE_SHARED

__all__ = ["LineoutFit", "fit_lineout"]

log = logging.getLogger(__name__)


@dataclass
class LineoutFit:
    """A fitted lineout: the 12 canonical outputs, per peak."""

    values: np.ndarray            # (n_peaks, 12)
    converged: bool
    n_peaks: int
    shared_width: bool

    def get(self, name: str, peak: int = 0) -> float:
        from .conventions import fit_output_index
        return float(self.values[peak, fit_output_index(name)])

    def as_dict(self, peak: int = 0) -> dict[str, float]:
        return {n: float(self.values[peak, i])
                for i, n in enumerate(FIT_OUTPUT_NAMES)}


def _moments(r: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Intensity-weighted centre, width and total. The fallback and the seed."""
    tot = float(np.sum(y))
    if tot <= 0:
        return float(np.mean(r)), 1.0, 0.0
    centre = float(np.sum(r * y) / tot)
    var = float(np.sum(y * (r - centre) ** 2) / tot)
    return centre, float(np.sqrt(max(var, 1e-12))), tot


def fit_lineout(
    radii: np.ndarray,
    intensity: np.ndarray,
    *,
    n_peaks: int = 1,
    peak_centres: tuple[float, ...] | None = None,
    shared_width: bool = LEGACY_WIDTHS_ARE_SHARED,
    variance: np.ndarray | None = None,
    max_iter: int = 60,
) -> LineoutFit:
    """Fit *n_peaks* pseudo-Voigt profiles plus a flat background.

    Parameters
    ----------
    shared_width : bool
        Constrain ``SigmaG == SigmaL``. **Default True**, matching the legacy
        engine: ``PeakFit.c`` sets both from the same parameter, so its
        "pseudo-Voigt" is a 5-parameter model and its 12 outputs carry 11
        distinct values. Set False for the independent-width fit that
        ``midas_peakfit`` is actually capable of -- but then results are not
        comparable channel-for-channel with 2023 output.
    variance : ndarray, optional
        Per-point variance, used to weight the fit. Omitting it weights every
        point equally, which over-weights the tails where there is no signal.

    Returns
    -------
    LineoutFit
        Always returns; check ``converged``. A failed fit still carries the
        moment-based estimates, which are meaningful even when the profile
        model is not a good description.
    """
    r = np.asarray(radii, dtype=np.float64).ravel()
    y = np.asarray(intensity, dtype=np.float64).ravel()
    if r.shape != y.shape:
        raise ValueError(f"radii {r.shape} and intensity {y.shape} must match")
    if r.size < 5:
        raise ValueError(f"need at least 5 points to fit, got {r.size}")

    bg_simple = float(np.median(np.concatenate([y[:3], y[-3:]])))
    out = np.zeros((n_peaks, 12), dtype=np.float64)

    try:
        from scipy.optimize import least_squares
        have_scipy = True
    except ImportError:
        have_scipy = False
        log.info("scipy not available -- reporting moment estimates only")

    # Seed the centre from the intensity-weighted moment, which is the
    # natural estimate, rather than from an arbitrary quantile of the window.
    # An earlier version computed the moment and then ignored it, seeding at
    # the 25th percentile instead; the solver wandered from there and landed
    # 16 px off a peak it should have found exactly.
    moment_centre, moment_sigma, _ = _moments(r, np.clip(y - bg_simple, 0, None))
    if peak_centres:
        centres = list(peak_centres)
    elif n_peaks == 1:
        centres = [moment_centre]
    else:
        span = r.max() - r.min()
        centres = list(r.min() + span * (np.arange(n_peaks) + 0.5) / n_peaks)

    converged = False
    if have_scipy:
        def model(p):
            m = np.full_like(r, p[0])                 # flat background
            for k in range(n_peaks):
                base = 1 + k * (3 if shared_width else 4)
                amp, cen, sg = p[base], p[base + 1], p[base + 2]
                sl = sg if shared_width else p[base + 3]
                mu = 0.5
                g = np.exp(-0.5 * ((r - cen) / max(sg, 1e-6)) ** 2)
                lo = 1.0 / (((r - cen) / max(sl, 1e-6)) ** 2 + 1.0)
                m = m + amp * (mu * lo + (1 - mu) * g)
            return m

        w = (1.0 / np.sqrt(np.clip(variance, 1e-12, None))
             if variance is not None else np.ones_like(y))
        p0, lo_b, hi_b = [bg_simple], [-np.inf], [np.inf]
        amp0 = max(float(y.max()) - bg_simple, 1e-6)
        sig0 = float(np.clip(moment_sigma, 0.5, (r.max() - r.min()) / 2))
        for c in centres:
            p0 += [amp0, float(c), sig0]
            # Bound the centre INSIDE the window: a peak fitted outside the
            # radius range it was extracted from is meaningless, and letting
            # the solver leave is how a good seed still ends up far away.
            lo_b += [0.0, float(r.min()), 1e-3]
            hi_b += [np.inf, float(r.max()), float(r.max() - r.min())]
            if not shared_width:
                p0.append(sig0)
                lo_b.append(1e-3)
                hi_b.append(float(r.max() - r.min()))
        try:
            sol = least_squares(lambda p: (model(p) - y) * w, p0,
                                bounds=(lo_b, hi_b),
                                max_nfev=max_iter * len(p0))
            converged = bool(sol.success)
            p = sol.x
        except Exception as exc:            # pragma: no cover - solver guard
            log.debug("lineout fit failed: %s", exc)
            p = p0
    else:
        p = None

    for k in range(n_peaks):
        cen, sg, tot = _moments(r, np.clip(y - bg_simple, 0, None))
        amp = float(np.max(y) - bg_simple)
        bg_fit = bg_simple
        if p is not None:
            base = 1 + k * (3 if shared_width else 4)
            bg_fit = float(p[0])
            amp, cen, sg = float(p[base]), float(p[base + 1]), float(p[base + 2])
        sl = sg if shared_width else (float(p[base + 3]) if p is not None else sg)
        fitted_area = float(amp * sg * np.sqrt(2 * np.pi))
        resid = float(np.mean(np.abs(y - (model(p) if p is not None else y))))

        out[k] = [
            cen,                                   # 0  RMEAN
            0.5,                                   # 1  MixFactor
            sg,                                    # 2  SigmaG
            sl,                                    # 3  SigmaL
            amp,                                   # 4  MaxInt
            float(np.max(y)),                      # 5  MaxIntensityObs
            bg_fit,                                # 6  BGFit
            bg_simple,                             # 7  BGSimple
            resid,                                 # 8  MeanError
            fitted_area,                           # 9  FitIntegratedIntensity
            float(np.sum(y)),                      # 10 TotalIntensity
            float(np.sum(y) - bg_simple * y.size), # 11 TotalIntensityBackgroundCorr
        ]

    return LineoutFit(values=out, converged=converged, n_peaks=n_peaks,
                      shared_width=shared_width)
