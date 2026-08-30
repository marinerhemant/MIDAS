"""Per-pixel intensity correction modules: polarization + solid-angle.

Mirrors the multiplicative factors v1 applies inside ``MapperCore.c``
(``corrected /= sa`` and ``corrected /= polFactor``), but as torch
``nn.Module``s so the corrections themselves are differentiable through
the geometry and through their own configurable parameters
(polarization fraction / plane angle).

Conventions match v1:

- Polarization factor:  ``1 - PF · sin²(2θ) · cos²(η - plane)``;
  ``corrected = raw / polFactor``.
- Solid-angle factor:   **exact tilt-aware form**
  ``Ω_pix / Ω_ref = Lsd² · (n̂·r) / |r|³``, where ``r`` is the lab-frame
  vector from sample to pixel and ``n̂`` is the lab-frame detector
  normal (``TRs · (1, 0, 0)``). For a perpendicular detector this
  reduces to ``cos³(2θ)``; for a tilted detector it captures the
  per-pixel incidence angle exactly. Mirrors v1's
  :func:`midas_integrate.geometry.solid_angle_factor`. **No
  approximation** — was previously the flat-detector ``cos³(2θ)`` form
  in v0.7-v0.8.1 (silently wrong on tilted detectors); fixed in v0.8.2.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


_DEG2RAD = math.pi / 180.0


#: Azimuth of the polarization plane, in MIDAS η, for a HORIZONTALLY polarized
#: beam — i.e. every storage ring in normal operation.
#:
#: **This is 90, not 0, and the difference is not cosmetic.** It was 0 until
#: 2026-08-29, which placed the correction on the vertical axis: the right
#: functional form applied a quarter turn away, which *adds* the azimuthal
#: modulation it is supposed to remove.
#:
#: The reason 0 looked right is a convention transcription error. MIDAS η is
#: ``atan2(-y, z)`` (``midas_integrate.geometry.calc_eta_angle``, and the same
#: expression at ``_mapper_numba.py:97`` and ``:665`` where the η bins the
#: correction consumes are actually built), so **η = 0 is VERTICAL** and
#: η = ±90 is horizontal. pyFAI measures its azimuth ``chi`` from the
#: horizontal detector X axis instead. The old docstring here read
#: "0 = horizontal at η = 0 (pyFAI convention)" — it took pyFAI's *number*
#: without taking pyFAI's *axis*.
#:
#: Established three independent ways, all measured:
#:
#: 1. **Source.** η = atan2(-y, z) ⇒ η = 0 is vertical. Exact, from the code.
#: 2. **pyFAI cross-check.** With ``factor=+1`` (linear horizontal) and
#:    ``axis_offset=0``, pyFAI's polarization array has its node (P = 0.0025)
#:    at ``chi = 0`` and its maximum (P = 1.0000) at ``chi = ±90``; the chi ≈ 0
#:    pixel lies at Δcol = +200, Δrow = -1 from the beam centre, i.e. purely
#:    horizontal. Both packages put the node ON the polarization axis, so the
#:    only difference is where each measures azimuth from.
#: 3. **Real data.** On 1-ID CeO2 in the corrected (R, η) frame, plane = 90
#:    flattens a powder ring's cos(2η) modulation (2.813 % → 0.744 %, 0.26×, at
#:    2θ = 12.53°) while plane = 0 worsens it (1.84×), scaling as sin²2θ. The
#:    obvious confound — a residual detector tilt also makes a cos(2η)
#:    signature — is refuted by a null model: rendering without polarization and
#:    integrating with a deliberate 0.05° tilt error gives ≤ 0.15 %, i.e.
#:    20-200× too small to explain a 2-3 % modulation.
#:
#: Set this to 0.0 to reproduce output generated before 2026-08-29, or to the
#: measured azimuth for a beamline whose polarization is not horizontal.
POL_PLANE_HORIZONTAL_ETA_DEG = 90.0


def two_theta_from_R(R_px: torch.Tensor, *, Lsd: torch.Tensor,
                      px: torch.Tensor) -> torch.Tensor:
    """``2θ = atan(R · px / Lsd)``.  Differentiable."""
    return torch.atan(R_px * px / Lsd)


def polarization_factor(
    R_px: torch.Tensor,
    eta_deg: torch.Tensor,
    *,
    Lsd: torch.Tensor,
    px: torch.Tensor,
    pol_fraction: torch.Tensor,
    pol_plane_eta_deg: torch.Tensor,
    model: str = "mixture",
) -> torch.Tensor:
    """Per-pixel polarization correction factor.

    The integration kernel divides recorded counts by this factor.

    ``model="mixture"`` (**default since 2026-08-29**) is the standard
    partially-polarized result (Kåhn 1982; the form pyFAI uses)::

        P = ½ · (1 + cos²2θ - PF · cos(2(η - plane)) · sin²2θ)

    ``model="midas"`` is v1's older form, kept for reproducing historical
    output::

        P = 1 - PF · sin²(2θ) · cos²(η - plane)

    **The two are algebraically IDENTICAL at PF = 1** and diverge below it,
    because the ``"midas"`` form *scales* the fully-polarized correction
    instead of *mixing* the two orthogonal polarization states — which is not
    what a partially polarized beam does. MEASURED difference:

    ===========  ==========================================
    PF           max relative difference (2θ ≤ 60°)
    ===========  ==========================================
    1.00         0.000 %   (identical)
    0.99         1.48 %    <- the shipped PolarizationFraction default
    0.95         6.98 %
    0.50         42.9 %
    ===========  ==========================================

    The physical tell: a genuinely unpolarized beam has no preferred azimuth,
    so its correction cannot depend on η. ``"mixture"`` at PF = 0 gives
    ``½(1 + cos²2θ)`` and satisfies that; ``"midas"`` at PF = 0.5 still varies
    with η by 0.375 at 2θ = 60°.

    .. warning::

       Switching this default **changes integrated intensities**. At the
       shipped ``PolarizationFraction = 0.99`` the shift is ~1.5 % at
       2θ = 60° and smaller below; peak *positions* are unaffected. Anything
       quantitative produced before 2026-08-29 (PDF, Rietveld, texture) used
       the ``"midas"`` form — pass ``model="midas"`` to reproduce it exactly.

    .. warning::

       ``pol_plane_eta_deg`` **also changed on 2026-08-29**, from 0 to 90, and
       that one matters far more than the model switch. MIDAS η is measured
       from the VERTICAL, so the old default put the correction a quarter turn
       away from the beam's actual polarization — the right shape on the wrong
       axis, which *adds* the azimuthal modulation instead of removing it.
       See :data:`POL_PLANE_HORIZONTAL_ETA_DEG`. Measured per-pixel difference
       between the two planes at PF = 0.99:

       ========  ==============================
       2θ        max relative difference in P
       ========  ==============================
       5°          0.8 %
       10°         3.1 %
       20°        13.1 %
       30°        32.9 %
       45°        98.5 %
       60°       292.6 %
       ========  ==============================

       Who is actually affected, measured rather than assumed:

       * **A 1-D pattern over a full ring, or any η range symmetric under a
         quarter turn, does not move at all** — exactly 0.0000 % at every 2θ
         above. A 90° shift only relabels which azimuth gets which factor, so
         the mean over such a range is identical. A 90° arc moves by ≤ 0.074 %
         (2θ = 60°), still negligible.
       * **Anything resolved in η carries the full error above**: caked
         patterns, texture, per-η peak areas, azimuthal-uniformity residuals,
         and partial rings on a clipped or tiled detector.

       Note also that ``PolarizationCorrection`` defaults to **0** (off), so
       the old default only ever bit users who deliberately enabled the
       correction — who then got an η modulation *worse* than leaving it off.
       Pass ``pol_plane_eta_deg=0.0`` to reproduce earlier output.
    """
    two_theta = two_theta_from_R(R_px, Lsd=Lsd, px=px)
    eta_offset_rad = (eta_deg - pol_plane_eta_deg) * _DEG2RAD
    if model == "midas":
        s2t = torch.sin(two_theta)
        ce = torch.cos(eta_offset_rad)
        return 1.0 - pol_fraction * s2t * s2t * ce * ce
    if model == "mixture":
        c2t = torch.cos(two_theta)
        s2t = torch.sin(two_theta)
        return 0.5 * (1.0 + c2t * c2t
                      - pol_fraction * torch.cos(2.0 * eta_offset_rad) * s2t * s2t)
    raise ValueError(
        f"model must be 'midas' or 'mixture', got {model!r}"
    )


def solid_angle_factor_flat(
    R_px: torch.Tensor,
    *,
    Lsd: torch.Tensor,
    px: torch.Tensor,
) -> torch.Tensor:
    """Flat-detector solid-angle factor ``cos³(2θ)``.

    **Approximate** — only correct for a perpendicular (zero-tilt)
    detector. Use :func:`solid_angle_factor_tilted` for any detector
    with a real tilt. Kept here for explicit opt-in / regression
    comparison only; the production :class:`SolidAngleCorrection`
    module uses the exact tilt-aware form.
    """
    two_theta = two_theta_from_R(R_px, Lsd=Lsd, px=px)
    c = torch.cos(two_theta)
    return c * c * c


def solid_angle_factor_tilted(
    Y_px: torch.Tensor,
    Z_px: torch.Tensor,
    *,
    Ycen: torch.Tensor,
    Zcen: torch.Tensor,
    TRs: torch.Tensor,                     # (3, 3) tilt matrix
    Lsd: torch.Tensor,
    pxY: torch.Tensor,
    pxZ: torch.Tensor,
) -> torch.Tensor:
    """**Exact** tilt-aware solid-angle factor for any detector pose.

    Returns ``Ω_pix / Ω_ref`` where:

        Ω_pix = A_pix · |n̂·r̂| / r²    (true solid angle of the pixel)
        Ω_ref = A_pix / Lsd²            (on-axis reference)

    so

        Ω_pix / Ω_ref = Lsd² · (n̂ · r) / |r|³

    with ``r`` = lab-frame vector sample → pixel and
    ``n̂`` = lab-frame detector normal = ``TRs · (1, 0, 0)``.

    Reduces to ``cos³(2θ)`` for a perpendicular detector
    (``TRs = I``); captures the local incidence angle exactly for
    any tilt.

    Bit-identical to :func:`midas_integrate.geometry.solid_angle_factor`
    (v1's reference implementation) at fp64.
    """
    Yc = (-Y_px + Ycen) * pxY
    Zc = ( Z_px - Zcen) * pxZ
    abcpr_x = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    abcpr_y = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    abcpr_z = TRs[2, 1] * Yc + TRs[2, 2] * Zc
    XYZ_x = Lsd + abcpr_x
    XYZ_y = abcpr_y
    XYZ_z = abcpr_z
    nx = TRs[0, 0]; ny = TRs[1, 0]; nz = TRs[2, 0]
    n_dot_r = nx * XYZ_x + ny * XYZ_y + nz * XYZ_z
    r_mag2  = XYZ_x * XYZ_x + XYZ_y * XYZ_y + XYZ_z * XYZ_z
    r3 = r_mag2 * torch.sqrt(r_mag2)
    return Lsd * Lsd * n_dot_r / r3


class PolarizationCorrection(nn.Module):
    """``nn.Module`` wrapper around :func:`polarization_factor`.

    Parameters
    ----------
    pol_fraction :
        Polarization fraction (0 = unpolarised, 1 = fully horizontally
        polarised). Default 0.99 matches v1's ``PolarizationFraction``.
    pol_plane_eta_deg :
        Azimuth of the polarization plane, in MIDAS η (deg). Defaults to
        :data:`POL_PLANE_HORIZONTAL_ETA_DEG` = **90**, which is horizontal —
        MIDAS measures η from the VERTICAL (``atan2(-y, z)``), so 90 is the
        horizontal polarization every storage ring delivers. Pass 0.0 to
        reproduce output from before 2026-08-29; see the constant's docstring
        for why 0 was wrong and how that was established.
    refinable :
        Whether the two parameters are :class:`nn.Parameter`. Defaults
        False — typical use freezes them at the calibrant-known values.
    """
    def __init__(self, *, pol_fraction: float = 0.99,
                 pol_plane_eta_deg: float = POL_PLANE_HORIZONTAL_ETA_DEG,
                 refinable: bool = False,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        pf = torch.tensor(pol_fraction, dtype=dtype)
        pe = torch.tensor(pol_plane_eta_deg, dtype=dtype)
        if refinable:
            self.pol_fraction = nn.Parameter(pf)
            self.pol_plane_eta_deg = nn.Parameter(pe)
        else:
            self.register_buffer("pol_fraction", pf)
            self.register_buffer("pol_plane_eta_deg", pe)

    def forward(
        self,
        R_px: torch.Tensor,
        eta_deg: torch.Tensor,
        *,
        Lsd: torch.Tensor,
        px: torch.Tensor,
    ) -> torch.Tensor:
        return polarization_factor(
            R_px, eta_deg,
            Lsd=Lsd, px=px,
            pol_fraction=self.pol_fraction,
            pol_plane_eta_deg=self.pol_plane_eta_deg,
        )


class SolidAngleCorrection(nn.Module):
    """Exact tilt-aware solid-angle factor.

    Bit-identical to v1's :func:`midas_integrate.geometry.solid_angle_factor`
    (which the MIDAS calibration paper uses). For a perpendicular
    detector this reduces to ``cos³(2θ)``; for any tilt the per-pixel
    incidence angle is exact.

    Forward signature uses positional ``(Y_px, Z_px)`` and named
    geometry kwargs (different from v0.8.1's
    ``forward(R_px, Lsd=, px=)``) — the exact form needs the full
    pixel coordinates and the tilt matrix, not just the radial R.
    """
    def forward(
        self,
        Y_px: torch.Tensor,
        Z_px: torch.Tensor,
        *,
        Ycen: torch.Tensor,
        Zcen: torch.Tensor,
        TRs: torch.Tensor,
        Lsd: torch.Tensor,
        pxY: torch.Tensor,
        pxZ: torch.Tensor,
    ) -> torch.Tensor:
        return solid_angle_factor_tilted(
            Y_px, Z_px,
            Ycen=Ycen, Zcen=Zcen, TRs=TRs, Lsd=Lsd,
            pxY=pxY, pxZ=pxZ,
        )


__all__ = [
    "two_theta_from_R",
    "polarization_factor",
    "solid_angle_factor_flat",
    "solid_angle_factor_tilted",
    "PolarizationCorrection",
    "SolidAngleCorrection",
]
