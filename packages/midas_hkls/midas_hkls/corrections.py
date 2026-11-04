"""Per-reflection intensity corrections for single-crystal rotation data.

What turns an integrated spot intensity into |F|²: polarization at the actual
azimuth, the rotation-method Lorentz factor, and absorption. Everything here is
a pure function of geometry (numpy, no torch), so it can be shared by any
spot-wise code: single-crystal reduction, DFXM, NF/FF forward models.

Conventions (MIDAS lab frame)
-----------------------------
* x along the beam, z up, y completes the frame. Angles in degrees, lengths in
  mm / cm⁻¹ as named on each argument.
* η is the MIDAS azimuth, ``η = atan2(-y, z)`` of the diffracted ray's
  transverse part, so **η = 0 is VERTICAL** and η = ±90° is horizontal. See
  ``midas_integrate_v2.corrections.intensity.POL_PLANE_HORIZONTAL_ETA_DEG``
  for how that was established; this module uses the same convention.
* ``s0`` is the unit incident direction, ``s`` the unit diffracted direction.
  :func:`scattered_direction` builds ``s`` from (2θ, η).
* Sample rotation by +ω is right-handed about ``axis``
  (:func:`rotate_about_axis`). The sign matters for absorption (the DAC axis
  moves with ω) and not for the Lorentz factor. Feed the ω sign that makes the
  rotation right-handed for YOUR detector/stage; this module does not guess it.

What each factor means
----------------------
``I_obs  ∝  |F|² · L · P · T``   (T = transmission, i.e. 1/A)

so ``|F|²_obs ∝ I_obs / (L · P · T)``. :func:`rotation_lorentz` returns ``L``,
:func:`polarization_factor` returns ``P``, :func:`diamond_anvil_transmission`
and :func:`slab_self_absorption` return transmissions ``T`` in [0, 1].

Not here
--------
Extinction (see :mod:`midas_hkls.extinction`; it is a scalar bound, not a
correction), multiple scattering, diamond Bragg spikes, thermal diffuse
scattering, detector efficiency/parallax, and beam footprint/illumination
volume. Each is either a scale that a per-frame scale factor absorbs or
something this module does not model; do not read a missing term as zero.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "BEAM_DIRECTION",
    "VERTICAL_AXIS",
    "scattered_direction",
    "rotate_about_axis",
    "polarization_factor",
    "rotation_lorentz",
    "diamond_mu_per_cm",
    "diamond_anvil_transmission",
    "slab_self_absorption",
]

BEAM_DIRECTION = np.array([1.0, 0.0, 0.0])
VERTICAL_AXIS = np.array([0.0, 0.0, 1.0])

#: density of diamond [g/cm³] (room temperature)
DIAMOND_DENSITY_G_CM3 = 3.515


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / n


def scattered_direction(two_theta_deg, eta_deg):
    """Unit diffracted direction ``s`` for the beam along +x.

    ``s = (cos 2θ, -sin 2θ sin η, sin 2θ cos η)``, which inverts
    ``η = atan2(-y, z)``. Shape ``(..., 3)``.
    """
    tt = np.radians(np.asarray(two_theta_deg, dtype=float))
    eta = np.radians(np.asarray(eta_deg, dtype=float))
    tt, eta = np.broadcast_arrays(tt, eta)
    return np.stack([np.cos(tt), -np.sin(tt) * np.sin(eta), np.sin(tt) * np.cos(eta)], axis=-1)


def rotate_about_axis(v, axis, omega_deg):
    """Rodrigues rotation of ``v`` by +ω about the unit ``axis`` (right-handed)."""
    v = np.asarray(v, dtype=float)
    k = _unit(axis)
    w = np.radians(np.asarray(omega_deg, dtype=float))[..., None]
    return (v * np.cos(w)
            + np.cross(k, v) * np.sin(w)
            + k * np.sum(k * v, axis=-1, keepdims=True) * (1.0 - np.cos(w)))


def polarization_factor(two_theta_deg, eta_deg, *, pol_fraction=0.99,
                        pol_plane_eta_deg=90.0):
    """Polarization factor ``P`` for a partially linearly polarized beam.

    ``P = ½ (1 + cos²2θ − f cos(2(η − η_pol)) sin²2θ)``

    ``f`` is the linear polarization fraction (0 = unpolarized, 1 = fully
    polarized), ``η_pol`` the azimuth of the polarization plane in MIDAS η
    (90° = horizontal, every storage ring). It is the incoherent mixture of the
    two transverse polarization states, equal to ``midas_integrate_v2``'s
    ``model="mixture"`` (tested), and the ring average over η is
    ``½(1 + cos²2θ)`` for any ``f``.

    The observed intensity is ``∝ P``; divide by it to correct.

    ``two_theta_deg`` / ``eta_deg`` may be torch tensors: the same formula is then evaluated in torch and is differentiable (``midas_2d`` uses this so its
    forward model stays differentiable end to end); the result is a torch tensor of the 2θ dtype/device.
    """
    if type(two_theta_deg).__module__.startswith("torch") or type(eta_deg).__module__.startswith("torch"):
        import torch
        tt_t = torch.as_tensor(two_theta_deg)
        if not torch.is_floating_point(tt_t):
            tt_t = tt_t.to(torch.get_default_dtype())
        tt = torch.deg2rad(tt_t)
        d = torch.deg2rad(torch.as_tensor(eta_deg, dtype=tt.dtype, device=tt.device) - pol_plane_eta_deg)
        return 0.5 * (1.0 + torch.cos(tt) ** 2 - pol_fraction * torch.cos(2.0 * d) * torch.sin(tt) ** 2)
    tt = np.radians(np.asarray(two_theta_deg, dtype=float))
    d = np.radians(np.asarray(eta_deg, dtype=float) - pol_plane_eta_deg)
    s2 = np.sin(tt) ** 2
    return 0.5 * (1.0 + np.cos(tt) ** 2 - pol_fraction * np.cos(2.0 * d) * s2)


def rotation_lorentz(s, *, s0=BEAM_DIRECTION, axis=VERTICAL_AXIS, min_abs=1e-3):
    """Rotation-method Lorentz factor ``L = 1 / |ê · (s0 × s)|``.

    Derivation: the Ewald condition ``f(ω) = |k0 + q(ω)| − k`` has
    ``|df/dω| = k |ê · (s0 × s)|`` at the crossing, and the ω-integrated
    intensity of a reflection is proportional to ``1/|df/dω|``. For a vertical
    axis and a beam along x this is ``1 / (sin 2θ |sin η|)`` (tested against it
    and against a finite-difference of the Ewald function).

    Reflections within ``min_abs`` of the rotation axis (``|ê·(s0×s)|`` below
    it) return **NaN**, not a large number: L diverges there, the
    integrated intensity is dominated by the unmodelled beam divergence and
    mosaic, and a silent 1e3-weighted reflection would dominate any fit. Count
    the NaNs you drop.

    ``s`` has shape ``(..., 3)``; returns shape ``(...)``.
    """
    s = _unit(s)
    d = np.abs(np.sum(_unit(axis) * np.cross(_unit(s0), s), axis=-1))
    with np.errstate(divide="ignore"):
        L = 1.0 / d
    return np.where(d < min_abs, np.nan, L)


def diamond_mu_per_cm(wavelength_A):
    """Linear absorption coefficient of diamond [cm⁻¹] at ``wavelength_A``.

    NIST XCOM carbon (``midas_hkls.absorption``) at 3.515 g/cm³.
    """
    from .absorption import linear_absorption_coefficient
    return float(linear_absorption_coefficient(
        "C", wavelength_A, density_g_cm3=DIAMOND_DENSITY_G_CM3))


def diamond_anvil_transmission(s, c, *, mu_per_cm, t_up_mm, t_down_mm=None,
                               s0=BEAM_DIRECTION, half_opening_deg=None):
    """Transmission through the two diamond anvils for one reflection.

    ``c`` is the DAC axis (normal to the culet faces) in the lab frame AT THE
    ω OF THE OBSERVATION; get it from :func:`rotate_about_axis`. The incident
    ray crosses the upstream anvil (thickness ``t_up_mm``) along its direction
    ``s0``, path ``t_up/|s0·c|``. The diffracted ray ``s`` leaves through the
    downstream anvil when it travels the same way along ``c`` as the beam, and
    back through the upstream anvil otherwise (back-reflection).

    ``half_opening_deg``: a ray making a larger angle with ``c`` than this is
    blocked (seat / gasket / cell body) and the transmission is **0**, which is
    how a narrow-opening cell loses reflections. ``None`` disables the cut.

    This is a flat-slab model: it ignores the anvil's conical shape and the
    seat, and treats the sample as a point at the culet. ``t_*_mm`` are
    effective thicknesses along ``c`` and are the quantities to misspecify when
    probing sensitivity.

    Returns T in [0, 1], shape of ``s`` without the last axis.
    """
    s = _unit(s)
    c = _unit(c)
    s0 = _unit(s0)
    t_down = t_up_mm if t_down_mm is None else t_down_mm
    ci = np.sum(s0 * c, axis=-1)
    co = np.sum(s * c, axis=-1)
    ai, ao = np.abs(ci), np.abs(co)
    same_side = np.sign(co) == np.sign(ci)
    t_out = np.where(same_side, t_down, t_up_mm)
    mu_mm = mu_per_cm / 10.0
    with np.errstate(divide="ignore", invalid="ignore"):
        path = t_up_mm / ai + t_out / ao
        T = np.exp(-mu_mm * path)
    ok = (ai > 0) & (ao > 0)
    if half_opening_deg is not None:
        cut = np.cos(np.radians(half_opening_deg))
        ok &= (ai >= cut) & (ao >= cut)
    return np.where(ok, T, 0.0)


def slab_self_absorption(cos_in, cos_out, *, mu_per_cm, thickness_um, laue):
    """Mean transmission over depth of a plane-parallel crystal plate.

    ``cos_in``/``cos_out``: |cosine| between the incident / diffracted ray and
    the plate normal. ``laue=True`` when the diffracted ray leaves through the
    face opposite the entrance, ``False`` when it leaves through the entrance
    face. Depth is uniformly illuminated. Returns 1 for ``mu*thickness = 0``.

    Laue:   ``(e^{-a} − e^{-b}) / (b − a)``,  a = μh/cos_in, b = μh/cos_out.
    Bragg:  ``(1 − e^{-(a+b)}) / (a + b)``.
    """
    ci = np.abs(np.asarray(cos_in, dtype=float))
    co = np.abs(np.asarray(cos_out, dtype=float))
    mu_h = mu_per_cm * thickness_um * 1e-4          # um -> cm
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = mu_h / ci, mu_h / co
        if laue:
            close = np.abs(b - a) < 1e-9
            A = np.where(close, np.exp(-a), (np.exp(-a) - np.exp(-b)) / np.where(close, 1.0, b - a))
        else:
            tot = a + b
            A = np.where(tot < 1e-12, 1.0, -np.expm1(-tot) / np.where(tot < 1e-12, 1.0, tot))
    return A
