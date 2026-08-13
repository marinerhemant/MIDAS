"""Finding the rotation axis, from the diffraction signal itself.

Absorption CT finds the axis from the transmission sinogram. XRD-CT often has
no usable transmission channel, so the axis has to come from the diffraction
data — which is noisier and, at a single (Q, eta) bin, can be almost empty.

Two independent methods, because they fail differently:

``com``
    Centre-of-mass consistency. For a parallel beam the sinogram's centre of
    mass traces ``c + A.cos(omega) + B.sin(omega)``; fitting that gives the
    offset directly, in one pass, with no reconstruction. Cheap and robust to
    noise, but it assumes the sample is fully inside the field of view — a
    truncated sample drags the centre of mass inward.

``sweep``
    Reconstruct at a range of shifts and score each. Slower by the number of
    candidates, but it makes no assumption about truncation.

When they disagree, the axis is not well determined by either and the answer
is to look at the images, not to average the two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .sinogram import SinogramStack

__all__ = ["CentreResult", "centre_of_mass_shift", "find_centre"]

log = logging.getLogger(__name__)


@dataclass
class CentreResult:
    """Estimated rotation-axis offset, in detector pixels."""

    shift: float
    method: str
    well_determined: bool
    detail: dict

    def describe(self) -> str:
        flag = "" if self.well_determined else "  [POORLY DETERMINED]"
        return f"axis shift {self.shift:+.3f} px ({self.method}){flag}"


def centre_of_mass_shift(
    stack: SinogramStack, *, bins: np.ndarray | None = None,
    min_signal: float = 0.0,
) -> CentreResult:
    """Axis offset from the sinogram's centre of mass.

    For a parallel beam the centre of mass of each projection follows

        com(omega) = c + A.cos(omega) + B.sin(omega)

    so a linear least-squares fit of that model gives ``c``, and the offset is
    ``c`` relative to the detector centre. One pass, no reconstruction.

    Parameters
    ----------
    bins : array of int, optional
        Which sinogram bins to use. Default: the brightest few, since a single
        (Q, eta) bin can be almost empty and its centre of mass then measures
        noise. Bins are summed before the fit, which is valid because centre
        of mass is linear in intensity.
    min_signal : float
        Projections whose total falls at or below this are dropped — their
        centre of mass is undefined.

    Notes
    -----
    Assumes the sample lies wholly within the field of view. A truncated
    sample biases the centre of mass toward the detector centre and therefore
    biases this estimate toward zero. :func:`find_centre` cross-checks against
    the sweep for exactly that reason.
    """
    inten = stack.intensity
    if bins is None:
        totals = inten.reshape(stack.n_bins, -1).sum(axis=1)
        k = max(1, min(8, stack.n_bins))
        bins = np.argsort(totals)[-k:]
    sino = inten[np.asarray(bins)].sum(axis=0)          # (n_omega, n_trans)

    x = np.arange(sino.shape[1], dtype=np.float64)
    centre = (sino.shape[1] - 1) / 2.0
    weight = sino.sum(axis=1)
    good = weight > min_signal
    if good.sum() < 4:
        return CentreResult(0.0, "com", False,
                            {"reason": "too few projections with signal",
                             "n_good": int(good.sum())})

    com = (sino[good] @ x) / weight[good]
    om = np.deg2rad(np.asarray(stack.omega_deg, dtype=np.float64)[good])

    # com = c + A cos + B sin  -- linear in (c, A, B)
    design = np.stack([np.ones_like(om), np.cos(om), np.sin(om)], axis=1)
    coef, residuals, rank, _ = np.linalg.lstsq(design, com, rcond=None)
    c, A, B = coef
    fit = design @ coef
    rms = float(np.sqrt(np.mean((com - fit) ** 2)))
    spread = float(np.std(com))

    # If the residual is comparable to the signal the model did not describe
    # the data, and c is not meaningful.
    well = bool(rank == 3 and spread > 0 and rms < 0.5 * spread)
    shift = float(c - centre)
    if not well:
        log.warning(
            "centre-of-mass fit is poor (rms %.3f vs spread %.3f): the sample "
            "may be truncated, or the chosen bins may carry no signal", rms, spread
        )
    return CentreResult(shift, "com", well,
                        {"c": float(c), "A": float(A), "B": float(B),
                         "rms": rms, "com_spread": spread,
                         "n_projections": int(good.sum())})


def find_centre(
    stack: SinogramStack,
    *,
    method: str = "com",
    half_width: float = 5.0,
    step: float = 0.5,
    cross_check: bool = True,
    n_cpus: int = 4,
    **recon_kw,
) -> CentreResult:
    """Estimate the rotation-axis offset.

    ``method='com'`` uses :func:`centre_of_mass_shift`; ``method='sweep'``
    reconstructs across ``+/- half_width`` and picks the sharpest.

    With ``cross_check`` (default) the ``com`` result is confirmed by a short
    sweep around it. Agreement is the evidence that neither failure mode
    (truncation for com, noise for sweep) is active; disagreement downgrades
    the result to poorly-determined rather than silently picking one.
    """
    if method == "com":
        res = centre_of_mass_shift(stack)
        if not cross_check or not res.well_determined:
            return res
        sweep = _sweep(stack, centre=res.shift, half_width=max(2.0, step * 4),
                       step=step, n_cpus=n_cpus, **recon_kw)
        agree = abs(sweep.shift - res.shift) <= max(1.0, 2 * step)
        if not agree:
            log.warning(
                "centre-of-mass (%+.3f) and sweep (%+.3f) disagree by more than "
                "%.1f px. Neither is trustworthy on its own here -- inspect the "
                "reconstructions rather than averaging them.",
                res.shift, sweep.shift, max(1.0, 2 * step),
            )
        return CentreResult(
            res.shift if agree else sweep.shift, "com+sweep", agree,
            {"com": res.detail, "sweep": sweep.detail,
             "com_shift": res.shift, "sweep_shift": sweep.shift},
        )
    if method == "sweep":
        return _sweep(stack, centre=0.0, half_width=half_width, step=step,
                      n_cpus=n_cpus, **recon_kw)
    raise ValueError(f"method must be 'com' or 'sweep'; got {method!r}")


def _sweep(stack: SinogramStack, *, centre: float, half_width: float,
           step: float, n_cpus: int, **recon_kw) -> CentreResult:
    """Reconstruct across a shift range and score each by image variance."""
    from midas_tomo import run_tomo_from_sinos
    from midas_tomo.center import find_center as _score

    import tempfile
    from pathlib import Path

    lo, hi = centre - half_width, centre + half_width
    n = int(round((hi - lo) / step)) + 1
    if n % 2:                     # the engine reconstructs shifts in pairs
        hi += step
        n += 1

    # One bright bin is enough to score sharpness, and keeps the sweep cheap.
    totals = stack.intensity.reshape(stack.n_bins, -1).sum(axis=1)
    best_bin = int(np.argmax(totals))
    sino = stack.intensity[best_bin][np.newaxis]

    with tempfile.TemporaryDirectory(prefix="midas_dt_centre_") as tmp:
        cube = run_tomo_from_sinos(
            sino, Path(tmp), stack.omega_deg, shifts=[lo, hi, step],
            filter_nr=recon_kw.get("filter_nr", 2), do_log=False,
            extra_pad=recon_kw.get("extra_pad", True), n_cpus=n_cpus,
            do_cleanup=True,
        )
    res = _score(cube, (lo, hi, step))
    log.info("shift sweep over %d candidates -> %+.3f px", n, res["best_shift"])
    return CentreResult(float(res["best_shift"]), "sweep",
                        bool(res["well_determined"]),
                        {"scores": res["scores"].tolist(),
                         "shifts": res["shifts"].tolist(),
                         "bin": best_bin})
