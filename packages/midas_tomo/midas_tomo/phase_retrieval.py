"""Paganin single-material phase retrieval.

Why it is needed here
---------------------
Both 1-ID datasets in this campaign were taken at a propagation distance of
100 mm (``tomo_metastr: D~100.000000mm``), so their contrast is largely
*phase*, not absorption. An FBP of propagation-contrast projections is
edge-enhancement dominated: the interior of the specimen has no contrast to
threshold, and the reconstruction thresholds into a hollow shell or, as
measured on bt_1id_jun25b NMC811 s5, into nothing usable at all — no threshold
plateau (volume swinging 372 134 to 3 721 um^3) and a mask filling the whole
field of view.

Paganin's filter converts that edge signal back into something proportional to
projected thickness, on the assumption that the specimen is a **single
material** with a fixed ``delta/beta``.

The method, and the one free parameter
--------------------------------------
For a transmission image ``I / I_0``::

    I_filtered = F^-1 { F[I / I_0] / (1 + pi * lambda * z * (delta/beta) * f^2) }

and the FBP input is ``-ln(I_filtered)`` as usual. Written this way the null is
exact **by construction**: at ``delta_beta = 0`` the denominator is 1, the
filter is the identity, and the reconstruction is bit-identical to the
unfiltered one. That is asserted in the tests rather than assumed.

``delta_beta`` is the only free parameter and it is a strong smoother — it
multiplies ``f^2``, so raising it low-passes the projections. Choosing it by
eye is choosing how big the specimen looks. :func:`sweep_delta_beta` exists so
it can be chosen against a stated criterion (threshold stationarity of the
resulting mask) and the sensitivity reported, rather than tuned until a
picture looks right.

What this does not do
---------------------
It does not make a weakly absorbing specimen strongly absorbing. Paganin
recovers *phase* contrast; if the projections have neither phase nor
absorption signal above the noise, there is nothing to retrieve and the honest
outcome is that the dataset cannot give a mask. That outcome is permitted and
must be reported, not tuned away.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["PLANCK_KEV_UM", "delta_beta_from_materials", "paganin_filter",
           "sweep_delta_beta", "wavelength_um"]

log = logging.getLogger(__name__)

#: h*c in keV.um, so wavelength_um = PLANCK_KEV_UM / energy_keV
PLANCK_KEV_UM = 1.23984193e-3


def wavelength_um(energy_kev: float) -> float:
    """X-ray wavelength in micrometres."""
    if not (energy_kev > 0):
        raise ValueError(f"energy_kev must be > 0; got {energy_kev}")
    return PLANCK_KEV_UM / float(energy_kev)


def paganin_filter(
    projections: np.ndarray,
    *,
    pixel_size_um: float,
    distance_mm: float,
    energy_kev: float,
    delta_beta: float,
    pad_frac: float = 0.25,
) -> np.ndarray:
    """Apply the Paganin single-material filter to flat-corrected projections.

    ``projections`` is ``(n, ny, nx)`` or ``(ny, nx)`` **transmission**
    (``I / I_0``), not raw counts and not ``-log``. The return has the same
    shape and is still transmission, so it drops straight into a pipeline whose
    next step takes the logarithm — the engine's ``doLog``, for instance.

    ``delta_beta = 0`` returns the input unchanged (exactly).

    ``pad_frac`` edge-pads before the FFT. Without it the filter wraps the
    opposite side of the frame into every edge, which puts a bright rim on the
    reconstruction that looks like a sample boundary.
    """
    p = np.asarray(projections, dtype=np.float64)
    squeeze = p.ndim == 2
    if squeeze:
        p = p[None]
    if p.ndim != 3:
        raise ValueError(
            f"projections must be 2-D or 3-D (n, ny, nx); got {p.shape}"
        )
    if float(delta_beta) < 0:
        raise ValueError(f"delta_beta must be >= 0; got {delta_beta}")
    if not (pixel_size_um > 0):
        raise ValueError(f"pixel_size_um must be > 0; got {pixel_size_um}")
    if distance_mm < 0:
        raise ValueError(f"distance_mm must be >= 0; got {distance_mm}")
    if float(delta_beta) == 0.0 or float(distance_mm) == 0.0:
        # Exactly the identity. Return the input rather than round-tripping it
        # through an FFT, so the null is bit-exact and not merely close.
        return np.asarray(projections)

    lam = wavelength_um(energy_kev)
    z = float(distance_mm) * 1000.0                    # mm -> um
    _, ny, nx = p.shape
    py, px_ = int(pad_frac * ny), int(pad_frac * nx)

    fy = np.fft.fftfreq(ny + 2 * py, d=float(pixel_size_um))
    fx = np.fft.fftfreq(nx + 2 * px_, d=float(pixel_size_um))
    f2 = fy[:, None] ** 2 + fx[None, :] ** 2
    denom = 1.0 + math.pi * lam * z * float(delta_beta) * f2

    out = np.empty_like(p)
    for i in range(p.shape[0]):
        padded = np.pad(p[i], ((py, py), (px_, px_)), mode="edge")
        filt = np.fft.ifft2(np.fft.fft2(padded) / denom).real
        out[i] = filt[py:py + ny, px_:px_ + nx]

    # A filtered transmission below zero is unphysical and would make the
    # subsequent log a NaN; it means delta_beta is far too large for this data.
    n_bad = int((out <= 0).sum())
    if n_bad:
        log.warning(
            "%d filtered pixels are <= 0 (delta_beta=%g is very strong for "
            "this data); clipping so the log stays finite", n_bad, delta_beta,
        )
        out = np.clip(out, 1e-9, None)
    return out[0] if squeeze else out


def delta_beta_from_materials(
    mu_per_um: float, electron_density_per_um3: float, energy_kev: float
) -> float:
    """``delta/beta`` from a measured ``mu`` and an electron density.

    ``delta = r_e lambda^2 n_e / (2 pi)`` and ``beta = mu lambda / (4 pi)``, so
    ``delta/beta = 2 r_e lambda n_e / mu``. Provided so the parameter can be
    *estimated* from the specimen rather than dialled; it is still an estimate,
    because the single-material assumption is exactly that.
    """
    if not (mu_per_um > 0):
        raise ValueError(f"mu_per_um must be > 0; got {mu_per_um}")
    r_e_um = 2.8179403262e-9      # classical electron radius, um
    lam = wavelength_um(energy_kev)
    return float(2.0 * r_e_um * lam * electron_density_per_um3 / mu_per_um)


def sweep_delta_beta(
    projections: np.ndarray,
    values: Sequence[float],
    *,
    pixel_size_um: float,
    distance_mm: float,
    energy_kev: float,
    score,
    pad_frac: float = 0.25,
) -> Dict[str, Any]:
    """Filter at several ``delta/beta`` and score each, so the choice is stated.

    ``score`` is called as ``score(filtered)`` and returns a number that is
    better when larger. The point is not to automate the choice but to make it
    reportable: ``delta_beta`` multiplies ``f^2``, so it directly controls how
    big the specimen comes out, and a value picked by eye is an unstated
    parameter in every downstream volume.
    """
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError("need at least two delta_beta values to sweep")
    scores = []
    for v in vals:
        filt = paganin_filter(
            projections, pixel_size_um=pixel_size_um, distance_mm=distance_mm,
            energy_kev=energy_kev, delta_beta=v, pad_frac=pad_frac,
        )
        scores.append(float(score(filt)))
    best = int(np.argmax(scores))
    return {"delta_beta": vals, "scores": scores,
            "best_delta_beta": vals[best], "best_score": scores[best],
            "monotonic": bool(best in (0, len(vals) - 1))}
