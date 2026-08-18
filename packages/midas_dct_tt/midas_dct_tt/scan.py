"""Scan builders: the TT psi sweep and DCT's Bragg-flash omega selection.

Phase 1 of ``implementation_plan.md``.

**TT** is the easy one: the alignment holds ``G`` on the rotation axis, so every
``psi`` diffracts and the scan is just a list of angles
(:func:`psi_scan`). Nothing is selected, because nothing switches off.

**DCT** is the interesting one. The sample turns about the vertical lab axis and
a reflection is in the Bragg condition only at *discrete* omega -- the "flash".
:func:`bragg_flashes` solves for them in closed form rather than by scanning and
thresholding, which matters because the whole technique's sparsity (10-60 spots
per grain over 360 deg) comes from this selection.

The solve
---------
With ``G(omega) = R_z(s*omega) G_sample`` (``s`` the stage sign, see
:data:`~midas_dct_tt.conventions.DCT_OMEGA_SIGN_AERO`) and ``k_in = k xhat``, the
elastic Bragg condition ``2 k.G + |G|^2 = 0`` reads

    Gx cos(a) - Gy sin(a) = -|G|^2 / (2k),      a = s * omega

i.e. ``R cos(a - phi) = C`` with ``R = hypot(Gx, Gy)``,
``phi = atan2(-Gy, Gx)``, ``C = -|G|^2/(2k)``. Hence

* ``|C| <= R``  -> exactly **two** flashes per 360 deg, ``a = phi +/- acos(C/R)``;
* ``|C| > R``   -> **none**: the reflection never reaches Bragg on this axis.

That second case is real physics, not a solver failure. A reflection whose ``G``
lies too close to the rotation axis keeps too much of its length out of the plane
the rotation sweeps and can never satisfy the condition -- so a vertical-axis DCT
scan is *blind* to part of reciprocal space. It is one of the reasons a grain
yields only tens of spots. Friedel partners (``-G``) give the other two flashes
and are the standard way to recover coverage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .conventions import DCT_OMEGA_SIGN_AERO, dct_sample_rotation
from .geometry import bragg_angle_deg, diffracted_beam_direction, incident_wavevector

__all__ = ["BraggFlash", "bragg_flashes", "dct_omega_scan", "psi_scan"]


def psi_scan(n: int = 180, *, span=(0.0, 360.0), endpoint: bool = False, dtype=torch.float64,
             device=None) -> torch.Tensor:
    """Tomographic rotation angles for a TT scan, in degrees ``(n,)``.

    ``endpoint=False`` (the default) omits the closing angle, since 0 and 360 deg
    are the same projection and including both double-counts one view in a
    reconstruction.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    lo, hi = float(span[0]), float(span[1])
    if endpoint:
        return torch.linspace(lo, hi, n, dtype=dtype, device=device)
    step = (hi - lo) / n
    return lo + step * torch.arange(n, dtype=dtype, device=device)


def dct_omega_scan(n_frames: int = 720, *, span=(0.0, 360.0), dtype=torch.float64,
                   device=None) -> torch.Tensor:
    """Frame *centres* of a DCT rotation scan, in degrees ``(n_frames,)``.

    DCT integrates each frame over a finite omega step, so the natural
    parameterisation is bin centres: frame ``i`` covers
    ``[lo + i*d, lo + (i+1)*d)`` and is labelled by its centre. A flash at omega
    belongs to frame ``floor((omega - lo)/d)``.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    lo, hi = float(span[0]), float(span[1])
    step = (hi - lo) / n_frames
    return lo + step * (torch.arange(n_frames, dtype=dtype, device=device) + 0.5)


@dataclass
class BraggFlash:
    """One omega at which a reflection satisfies the Bragg condition.

    Attributes
    ----------
    omega_deg : float
        Stage angle of the flash, in ``[0, 360)``.
    G_lab : (3,) tensor
        The scattering vector at that angle, in the lab frame.
    G_sample : (3,) tensor
        The same vector in the sample frame, i.e. the input ``g_sample``. Kept
        because Friedel partnership is an antiparallel relation *there* and not
        in the lab -- see :func:`midas_dct_tt.pairing.friedel_pairs`.
    k_out : (3,) tensor
        Unit diffracted-beam direction ``k_in + G`` -- the topograph's projection
        direction, and where the spot lands.
    theta_deg : float
        Bragg angle of the reflection.
    """

    omega_deg: float
    G_lab: torch.Tensor
    k_out: torch.Tensor
    theta_deg: float
    G_sample: torch.Tensor = None


def bragg_flashes(
    g_sample: torch.Tensor,
    wavelength_A,
    *,
    omega_sign: float = DCT_OMEGA_SIGN_AERO,
) -> list:
    """Closed-form omega values where ``g_sample`` flashes into Bragg.

    Parameters
    ----------
    g_sample : (3,) tensor
        Reciprocal vector in the **sample frame**, ``2*pi/d`` convention.
    wavelength_A : float or tensor
        Incident wavelength, Angstrom.
    omega_sign : float
        Stage handedness; defaults to the 1-ID aero (clockwise) convention. A
        wrong sign here mirrors every reconstructed grain shape -- see
        :func:`~midas_dct_tt.conventions.dct_sample_rotation`.

    Returns
    -------
    list[BraggFlash]
        Two entries (the generic case), or **empty** when the reflection can
        never reach Bragg on this rotation axis. Empty is a physical answer, not
        an error: check the length rather than assuming two.

    Raises
    ------
    ValueError
        If the reflection is inaccessible at this wavelength (``d < lambda/2``),
        or if ``G`` lies exactly along the rotation axis (its projection into the
        rotation plane vanishes, so omega has no effect and the question is
        ill-posed).
    """
    g = g_sample if isinstance(g_sample, torch.Tensor) else torch.as_tensor(g_sample)
    if not torch.is_floating_point(g):
        g = g.to(torch.float64)

    k_in = incident_wavevector(wavelength_A, device=g.device, dtype=g.dtype)
    k_mag = torch.linalg.vector_norm(k_in)
    g_mag = torch.linalg.vector_norm(g)
    theta_deg = float(bragg_angle_deg(g_mag, k_mag))     # raises if d < lambda/2

    gx, gy = float(g[0]), float(g[1])
    R = math.hypot(gx, gy)
    C = -float(g_mag) ** 2 / (2.0 * float(k_mag))

    if R < 1e-12:
        raise ValueError(
            "G lies along the rotation axis: omega does not move it, so 'the omega "
            "at which it flashes' is undefined. This is the topotomography "
            "configuration, not a DCT one -- use tt_alignment()."
        )
    ratio = C / R
    if abs(ratio) > 1.0:
        return []                                        # blind reflection

    phi = math.atan2(-gy, gx)
    delta = math.acos(max(-1.0, min(1.0, ratio)))

    flashes = []
    for a in (phi + delta, phi - delta):
        omega = math.degrees(a) / omega_sign             # a = s * omega, s = +-1
        omega = omega % 360.0
        R_om = dct_sample_rotation(omega, omega_sign=omega_sign,
                                   device=g.device, dtype=g.dtype)
        G_lab = R_om @ g
        flashes.append(
            BraggFlash(
                omega_deg=omega,
                G_lab=G_lab,
                k_out=diffracted_beam_direction(k_in, G_lab),
                theta_deg=theta_deg,
                G_sample=g,
            )
        )
    flashes.sort(key=lambda f: f.omega_deg)
    return flashes
