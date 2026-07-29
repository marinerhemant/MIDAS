"""Coherent and partially-coherent DFXM image formation.

Conventional DFXM is modelled as *incoherent* imaging: every voxel diffracts according to
how well it satisfies the local Bragg condition, and those intensities add. With the
coherent flux of an upgraded source, the diffracted amplitudes instead **interfere**, and
the phase of the exit wave -- ``phi = g . u`` for a displacement field ``u`` -- becomes
part of the measurement. That changes the image formation qualitatively:

    incoherent   I = R  (*) |h|^2          transfer function = autocorrelation of the pupil
    coherent     I = | sqrt(R) e^{i phi} (*) h |^2      transfer function = the pupil itself

where ``R`` is the per-voxel diffracted intensity the current forward already predicts and
``h`` is the complex amplitude point-spread function of the objective. The consequence
that matters for instrument design: in the coherent case the pupil's aberration phase
enters the image **linearly**, rather than through an intensity autocorrelation.

Partial coherence is represented at first order by the coherent fraction ``mu``, mixing the
two limits. That is a deliberately simple model of a partially coherent source (a full
treatment needs a coherent-mode decomposition of the mutual coherence function); it is
adequate for asking how results trend between the limits, and is labelled as such.

Everything is torch-differentiable, so coherent forward models can be inverted or used for
experiment design with the same machinery as the incoherent ones.
"""
from __future__ import annotations

import math

import torch

_ANGSTROM_PER_UM = 1e4

from .aberration import ABERRATIONS, _pupil_grid, coeffs_to_tensor, zernike_terms


def coherent_psf(coeffs, *, defocus=0.0, grid_size=64, extent=1.4, apodization=1.5,
                 device=None, dtype=torch.float64):
    """Complex amplitude PSF ``h`` of the objective (normalized so ``sum |h|^2 = 1``).

    This is the coherent counterpart of :func:`midas_dfxm.aberration.aberrated_psf`, which
    returns the intensity PSF ``|h|^2``. Gradients flow to the Zernike coefficients.
    """
    c = coeffs_to_tensor(coeffs, device=device, dtype=dtype)
    uu, vv, r2, aperture = _pupil_grid(grid_size, extent, device=c.device, dtype=c.dtype)
    apod = torch.exp(-apodization * r2) * aperture
    Z = zernike_terms(uu, vv)
    defocus = torch.as_tensor(defocus, device=c.device, dtype=c.dtype)
    phase = defocus * Z["defocus"]
    for i, name in enumerate(ABERRATIONS):
        phase = phase + c[i] * Z[name]
    cplx = torch.complex128 if c.dtype == torch.float64 else torch.complex64
    pupil = apod.to(cplx) * torch.exp(1j * phase.to(cplx))
    h = torch.fft.fftshift(torch.fft.fft2(pupil))
    return h / torch.sqrt((h.real ** 2 + h.imag ** 2).sum())


def _complex_dtype(t):
    """Matching complex dtype for a real tensor (MPS has no float64/complex128)."""
    return torch.complex128 if t.dtype in (torch.float64, torch.complex128) else torch.complex64


def _fftconv(a, k):
    """Centered same-size FFT convolution (complex or real ``a``, kernel ``k``)."""
    K = torch.fft.fft2(torch.fft.ifftshift(k))
    return torch.fft.ifft2(torch.fft.fft2(a) * K)


def coherent_image(amplitude, h):
    """Coherent image ``|amplitude (*) h|^2`` -- amplitudes interfere."""
    return _fftconv(amplitude, h).abs() ** 2


def incoherent_image(intensity, h):
    """Incoherent image ``intensity (*) |h|^2`` -- intensities add (conventional DFXM)."""
    cplx = _complex_dtype(intensity)
    hi = (h.real ** 2 + h.imag ** 2).to(intensity.dtype)
    return _fftconv(intensity.to(cplx), hi.to(cplx)).real


def partially_coherent_image(amplitude, h, coherent_fraction):
    """First-order partial coherence: mix the coherent and incoherent limits by ``mu``.

    ``coherent_fraction=1`` is fully coherent, ``0`` fully incoherent. A first-order
    stand-in for a coherent-mode decomposition, useful for trending between the limits.

    Both limits are normalized to unit total flux **before** mixing. Partial coherence
    redistributes photons; it does not create or destroy them. This matters in practice:
    the two limits use different PSF normalizations (amplitude convolution vs intensity
    convolution), so their raw sums can differ by orders of magnitude -- measured 1.5 vs
    7.6e3 on a typical field -- and a raw linear mix would simply track whichever is
    larger, leaving ``mu`` with almost no effect until it reached 1.
    """
    mu = float(coherent_fraction)
    I_c = coherent_image(amplitude, h)
    I_i = incoherent_image((amplitude.abs() ** 2), h)
    I_c = I_c / I_c.sum().clamp_min(1e-30)
    I_i = I_i / I_i.sum().clamp_min(1e-30)
    return mu * I_c + (1.0 - mu) * I_i


def dislocation_phase(x, y, g_dot_b, *, core=(0.0, 0.0)):
    """Exit-wave phase of a dislocation: the ``2 pi (g.b)`` winding around the core.

    ``phi = g . u`` winds by ``2 pi (g.b)`` on a circuit around the line.

    **``g.b`` must be an integer.** For a perfect dislocation ``b`` is a lattice
    translation and ``g`` a reciprocal-lattice vector, so their product is necessarily an
    integer -- which is exactly what makes ``exp(i phi)`` single-valued and the position of
    the branch cut unobservable. A non-integer ``g.b`` is unphysical and produces an image
    that depends on where the (arbitrary) cut is placed; this is gated by a test.

    Note ``g.b = 0`` is the invisible case (no phase modulation at all); non-zero integers
    are visible and carry a phase vortex of that charge.
    """
    gb = float(g_dot_b)
    if abs(gb - round(gb)) > 1e-9:
        raise ValueError(
            f"g.b = {gb} is not an integer. For a perfect dislocation g.b is always an "
            "integer (b is a lattice translation, g a reciprocal-lattice vector); a "
            "non-integer value makes exp(i g.u) multivalued and the image depends on the "
            "arbitrary branch-cut position."
        )
    dx, dy = x - core[0], y - core[1]
    theta = torch.atan2(dy, dx)
    return gb * theta


def dislocation_exit_wave(positions, dislocations, reflection, *, lattice_params,
                          orientation=None, res_width=0.02, shape=None):
    """Complex exit wave of a real dislocation field, for coherent imaging.

    Connects the coherent path to the actual elasticity instead of an idealized phase:

    * the **phase** is ``phi = g . u`` with ``u`` the anisotropic Stroh displacement
      (:meth:`StrohDislocation.displacement`);
    * the **amplitude** is ``sqrt(R)`` with ``R`` the Lorentzian Bragg response to the
      local reciprocal-space shift ``dQ = (F^-T - I) Q0`` -- the same contrast mechanism
      the incoherent forward uses.

    Use dislocations built with ``core_model="compact"`` so that ``u`` and ``beta`` are the
    same field outside the core; with the default Lorentzian core they disagree near the
    line. ``res_width`` is the reciprocal-space acceptance (same units as ``|Q|``).

    Note the winding of ``phi`` around each core is ``2 pi (g.b)`` with ``g`` a
    reciprocal-lattice vector and ``b`` a lattice translation, so it is automatically an
    integer multiple of ``2 pi`` -- the exit wave is single-valued by construction.
    """
    from .field_inverse import reference_Q
    from .dislocation import dislocation_deformation_field

    if not isinstance(dislocations, (list, tuple)):
        dislocations = [dislocations]
    dtype = positions.dtype
    eye = torch.eye(3, device=positions.device, dtype=dtype)
    orient = eye if orientation is None else torch.as_tensor(orientation, dtype=dtype)
    Q0 = reference_Q(reflection, orient, torch.as_tensor(lattice_params, dtype=dtype))

    # amplitude: Bragg response to the local reciprocal-space shift
    F = dislocation_deformation_field(positions, list(dislocations),
                                      lattice_params=lattice_params).F
    dQ = torch.einsum("nij,j->ni", torch.linalg.inv(F).transpose(-1, -2) - eye, Q0)
    R = res_width ** 2 / (res_width ** 2 + (dQ ** 2).sum(-1))

    # phase: g . u, summed over dislocations (displacements superpose).
    # UNITS: displacements are micrometres, Q is per-Angstrom -- convert or the phase is
    # 1e4 too small. Caught by the winding gate: g.u around a core must be 2 pi x integer.
    u = torch.zeros_like(positions)
    for d in dislocations:
        u = u + d.displacement(positions)
    phi = (u * _ANGSTROM_PER_UM) @ Q0

    if shape is not None:
        R, phi = R.reshape(shape), phi.reshape(shape)
    return exit_amplitude(R, phi)


def exit_amplitude(intensity, phase):
    """Complex exit wave ``sqrt(I) e^{i phi}`` from a diffracted intensity and a phase."""
    cplx = _complex_dtype(intensity)
    amp = torch.sqrt(intensity.clamp_min(0)).to(cplx)
    return amp * torch.exp(1j * phase.to(cplx))
