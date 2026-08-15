"""A v2-native E-step: the calibrant cake built through ``midas_integrate_v2``.

`midas_calibrate_v2` currently computes its E-step through ``midas_calibrate``
v1, which calls v1's ``build_map`` / ``build_csr`` / ``integrate``
(``estep.py:23,29,184,195``). This module is the v2 replacement for the one
function in that chain that touches the v1 integrator, ``integrate_cake``.

Everything downstream is untouched: ``extract_fitted_points`` operates on the
returned cake, so it is reused verbatim rather than reimplemented. Copying it
would be the easiest way to introduce a difference that has nothing to do with
the backend under test.

**Registered as an experiment, not a default.** See
``dev/paper/PREREGISTER_v2_estep.md``: this lands behind an explicit choice,
with v1 remaining the default, until a comparison on both an archived and a
planted frame confirms agreement to 50 ppm in Lsd and 0.05 px in beam centre.

Two things deliberately reused rather than rewritten, because each is a place a
hand-rolled version would silently differ:

* ``spec_from_v1_params`` for the params conversion. The legacy ``p0..p3`` are
  **not** positional in v2 -- the mapping is a permutation (``p0 -> a2``,
  ``p1 -> a4``, ``p2 -> iso_R2``, ``p3 -> phi4``) taken from
  ``midas_distortion``'s canonical table. Writing it out here is how every
  radius gets quietly distorted.
* ``_calibration_to_integration_params`` and ``CakeProfile`` from v1, so the
  R/eta grid and the returned structure are identical by construction and the
  only variable is the binning backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

__all__ = ["integrate_cake_v2", "run_estep_v2", "V2_UNAVAILABLE"]

V2_UNAVAILABLE = None
try:
    from midas_integrate_v2 import (
        HardBinGeometry,
        SubpixelBinGeometry,
        integrate_hard,
        integrate_subpixel,
        spec_from_v1_params,
    )
except ImportError as _exc:                       # pragma: no cover
    V2_UNAVAILABLE = str(_exc)


def integrate_cake_v2(
    params,
    image: np.ndarray,
    rt,
    *,
    dark: Optional[np.ndarray] = None,
    cake_mode: str = "auto",
):
    """``midas_calibrate.estep.integrate_cake``, built on ``midas_integrate_v2``.

    Signature and return type match the v1 function exactly so it can be
    substituted without touching any caller.

    ``cake_mode`` maps onto v2's binning classes:

    ==================  ===========================================
    v1 mode             v2 equivalent
    ==================  ===========================================
    ``"floor"``         :class:`HardBinGeometry`  (one bin per pixel)
    ``"bilinear"``      :class:`SubpixelBinGeometry`
    ``"auto"``          subpixel, matching v1's accurate default
    ==================  ===========================================

    v1's ``"auto"`` also degrades to floor binning when RAM looks tight, which
    v2 does not need: it never materialises the 4-corner CSR expansion that
    made that necessary. The mapping above therefore reproduces v1's *intent*
    (accurate binning) rather than its memory-pressure fallback -- a difference
    worth knowing when a comparison run disagrees on a large detector.
    """
    if V2_UNAVAILABLE is not None:               # pragma: no cover
        raise ImportError(
            f"the v2 E-step needs midas-integrate-v2: {V2_UNAVAILABLE}")

    from midas_calibrate.estep import (
        CakeProfile,
        _calibration_to_integration_params,
    )

    if dark is not None:
        image = image - dark

    # Identical R range to v1, from the same expression, so the grids match.
    px = 0.5 * (params.pxY + params.pxZ) if params.pxZ > 0 else params.pxY
    half_px = 0.5 * params.Width / px
    R_min = max(0.0, float(rt.r_ideal_px.min()) - half_px - 1.0)
    R_max = float(rt.r_ideal_px.max()) + half_px + 1.0

    ip = _calibration_to_integration_params(
        params, R_min=R_min, R_max=R_max,
        R_bin_size=params.RBinSize, eta_bin_size=params.EtaBinSize,
    )
    spec = spec_from_v1_params(ip, dtype=torch.float64, requires_grad=False)

    img_t = torch.as_tensor(np.asarray(image), dtype=torch.float64).contiguous()
    if cake_mode == "floor":
        geom = HardBinGeometry.from_spec(spec)
        cake_t = integrate_hard(img_t, geom)
    else:
        geom = SubpixelBinGeometry.from_spec(spec)
        cake_t = integrate_subpixel(img_t, geom)

    cake = np.asarray(cake_t.detach().cpu().numpy(), dtype=np.float64)
    # v1 returns [n_R, n_eta]; transpose only if v2 handed back the transpose.
    if cake.shape == (ip.n_eta_bins, ip.n_r_bins) and ip.n_r_bins != ip.n_eta_bins:
        cake = cake.T

    R_edges = np.linspace(ip.RMin, ip.RMin + ip.RBinSize * ip.n_r_bins,
                          ip.n_r_bins + 1)
    eta_edges = np.linspace(ip.EtaMin, ip.EtaMax, ip.n_eta_bins + 1)
    return CakeProfile(
        R_centers=0.5 * (R_edges[:-1] + R_edges[1:]),
        eta_centers=0.5 * (eta_edges[:-1] + eta_edges[1:]),
        intensity=cake,
    )


def run_estep_v2(params, image, rt, *, dark=None, cake_mode="auto"):
    """``run_estep`` with the v2 cake; point extraction is v1's, unchanged."""
    from midas_calibrate.estep import extract_fitted_points

    cake = integrate_cake_v2(params, image, rt, dark=dark, cake_mode=cake_mode)
    fits = extract_fitted_points(cake, params, rt)
    return cake, fits
