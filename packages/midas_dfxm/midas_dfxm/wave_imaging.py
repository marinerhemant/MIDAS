"""Wave image formation: a diffracted exit wave imaged through the DFXM objective.

Stage B of the physical-optics forward. Given the complex exit wave leaving the
sample (from the Takagi-Taupin solver, :mod:`midas_dfxm.takagi_taupin`, or any
field-to-exit-wave map), this forms the detector image through the magnifying
objective as a Fourier-optics imaging system, using the coherent amplitude PSF
(with pupil aberration + defocus) and a partial-coherence blend:

    coherent    I = | psi_exit (*) h |^2                 (self-consistent phases)
    incoherent  I = |psi_exit|^2 (*) |h|^2               (source washes phase out)
    partial     I = mix(coherent, incoherent; coh_frac)

``h`` is the objective's complex amplitude PSF (``optics.amplitude_psf``) sampled
at the detector grid. In the incoherent + delta-PSF limit this reduces to the
squared exit-wave magnitude -- the quantity the geometrical forward renders --
which is the reduction gate the composed dynamical forward is checked against.

Everything is torch-differentiable in the exit wave and the Zernike coefficients.
"""
from __future__ import annotations

from typing import Optional

import torch

from .coherence import coherent_image, incoherent_image, partially_coherent_image
from .optics import ObjectiveOptics
from .takagi_taupin import solve_tt_laue, pink_deviation_offsets


def _fit_center(h: torch.Tensor, shape) -> torch.Tensor:
    """Center-pad or center-crop the (complex) PSF ``h`` to ``shape``; renormalize |h|^2=1."""
    H, W = shape
    ph, pw = h.shape[-2:]
    if (ph, pw) == (H, W):
        return h
    if ph > H or pw > W:                                          # crop tails
        r0, c0 = max(0, (ph - H) // 2), max(0, (pw - W) // 2)
        h = h[r0:r0 + min(ph, H), c0:c0 + min(pw, W)]
        ph, pw = h.shape[-2:]
    if (ph, pw) != (H, W):                                        # pad
        big = torch.zeros(H, W, dtype=h.dtype, device=h.device)
        r0, c0 = (H - ph) // 2, (W - pw) // 2
        big[r0:r0 + ph, c0:c0 + pw] = h
        h = big
    return h / torch.sqrt((h.abs() ** 2).sum())


def dfxm_image_wave(
    exit_wave: torch.Tensor,
    optics: ObjectiveOptics,
    *,
    coeffs=None,
    defocus: float = 0.0,
    coherent_fraction: float = 1.0,
    psf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Detector intensity ``(H, W)`` from a complex ``exit_wave`` ``(H, W)``.

    ``coherent_fraction`` -- 1 fully coherent, 0 fully incoherent, in between a
    partial-coherence blend. ``coeffs``/``defocus`` set the objective aberration
    and the through-focal knob (pass a defocus series for phase diversity). ``psf``
    overrides the amplitude PSF (else built from ``optics.amplitude_psf``).
    """
    h = psf if psf is not None else optics.amplitude_psf(coeffs, defocus=defocus,
                                                         dtype=torch.float64)
    h = _fit_center(h, exit_wave.shape[-2:])                      # PSF must match image size
    if coherent_fraction >= 1.0:
        return coherent_image(exit_wave, h)
    if coherent_fraction <= 0.0:
        return incoherent_image(exit_wave.abs() ** 2, h)
    return partially_coherent_image(exit_wave, h, coherent_fraction)


def dfxm_image_dynamical(
    chi0, chih, chihbar, *,
    wavelength_A: float, theta_B_deg: float, thickness_um: float,
    optics: ObjectiveOptics, hu: Optional[torch.Tensor] = None, dx_um: float = 1.0,
    y: float = 0.0, C: float = 1.0, n_depth: int = 300, ny: Optional[int] = None,
    coherent_fraction: float = 0.0, coeffs=None, defocus: float = 0.0,
    checkpoint: bool = False,
) -> torch.Tensor:
    """End-to-end dynamical DFXM image: crystal -> TT exit wave -> objective -> detector.

    Composes the full physical-optics forward. ``hu`` is the deformation phase
    ``h . u(z, x)`` over the scattering plane, shape ``(n_depth, nx)`` (``None`` for
    a perfect crystal). The Takagi-Taupin solver gives the diffracted exit wave
    ``Dh(x)``; it is extended along the out-of-plane ``y`` (default ``nx`` rows,
    invariant -- valid for a line defect / y-uniform field) and imaged by the
    objective (:func:`dfxm_image_wave`). Fully differentiable from ``hu`` (and the
    susceptibilities, aberration, deviation) to the detector image.

    Reduction anchor: a perfect crystal gives a spatially uniform image whose value
    is the dynamical Laue rocking-curve intensity; in the thin (kinematical) +
    incoherent + delta-PSF limit the imaged ``|Dh|^2`` is the kinematical diffracted
    intensity the geometrical forward renders.
    """
    nx = 1 if hu is None else hu.shape[-1]
    _, Dh = solve_tt_laue(chi0, chih, chihbar, wavelength_A=wavelength_A,
                          theta_B_deg=theta_B_deg, thickness_um=thickness_um, y=y, C=C,
                          hu=hu, dx_um=dx_um, n_depth=n_depth, checkpoint=checkpoint)
    if nx == 1:                                                   # perfect crystal: fill a plane
        ny = ny or min(optics.detector_shape)
        nx = ny
        exit2d = Dh.reshape(1, 1).expand(ny, nx).clone()
    else:
        ny = ny or nx
        exit2d = Dh.reshape(1, nx).expand(ny, nx).clone()        # y-invariant extension
    return dfxm_image_wave(exit2d, optics, coeffs=coeffs, defocus=defocus,
                           coherent_fraction=coherent_fraction)


def dfxm_image_dynamical_pink(
    chi0, chih, chihbar, *, wavelength_A: float, theta_B_deg: float, thickness_um: float,
    optics: ObjectiveOptics, bandwidth: float, hu=None, dx_um: float = 1.0, y0: float = 0.0,
    C: float = 1.0, n_depth: int = 300, ny=None, n_lambda: int = 15, shape: str = "gaussian",
    coherent_fraction: float = 0.0, coeffs=None, defocus: float = 0.0,
):
    """Pink-beam dynamical DFXM image: incoherent sum of the forward over the spectrum.

    A fractional energy bandwidth maps to deviation offsets (:func:`pink_deviation_offsets`);
    each wavelength forms its own dynamical image (:func:`dfxm_image_dynamical`) and they add
    in intensity (different wavelengths do not interfere). ``bandwidth = 0`` reduces to the
    monochromatic image. Pink illumination lets weakly-diffracting / broadened-peak (deformed)
    regions contribute, its practical advantage. Differentiable in ``hu`` and the aberration.
    """
    ys, w = pink_deviation_offsets(bandwidth, theta_B_deg, chih, C=C, n=n_lambda, shape=shape)
    total = None
    for yo, wi in zip(ys.tolist(), w.tolist()):
        img = dfxm_image_dynamical(chi0, chih, chihbar, wavelength_A=wavelength_A,
                                   theta_B_deg=theta_B_deg, thickness_um=thickness_um,
                                   optics=optics, hu=hu, dx_um=dx_um, y=y0 + yo, C=C,
                                   n_depth=n_depth, ny=ny, coherent_fraction=coherent_fraction,
                                   coeffs=coeffs, defocus=defocus)
        total = wi * img if total is None else total + wi * img
    return total


def dfxm_image_dynamical_chromatic_pink(
    crystal, hkl, *, theta_B_deg: float, thickness_um: float, E0_keV: float,
    optics: ObjectiveOptics, n_lenses: int, radius_um: float, object_distance_m: float,
    spacing_m: float = 1.6e-3, bandwidth: float = 0.0, n_lambda: int = 15,
    shape: str = "gaussian", spectrum=None, hu=None, dx_um: float = 1.0, y0: float = 0.0,
    C: float = 1.0, n_depth: int = 300, ny=None, coherent_fraction: float = 0.0, coeffs=None,
    disperse_chi: bool = True, chromatic_psf: bool = True, y_cut: float = 60.0,
    absorption: bool = True,
):
    """Full chromatic pink-beam dynamical DFXM image, incoherent over the spectrum ``S(lambda)``.

    Each spectral sample forms its own dynamical image and they add in intensity, with **both** the
    per-energy dynamical diffraction and the objective's chromatic aberration included:

    - ``disperse_chi=True`` re-solves the Takagi--Taupin exit wave at each wavelength's own
      ``chi(lambda)`` (hence its extinction length and absorption); ``False`` freezes ``chi`` at the
      centre wavelength. (The two agree to ``~1e-3`` -- crystal-side dispersion is negligible at DFXM
      bandwidths -- so ``False`` is a safe, faster default in practice.)
    - ``chromatic_psf=True`` gives each energy its own objective defocus from the CRL chromaticity
      (:func:`beamline.chromatic_defocus_coeffs`); ``False`` uses the in-focus PSF (the older
      deviation-blur model). The chromatic pedestal is realized *in the diffracted image only to the
      extent the sample diffracts a broad energy band*: at a single setting a near-perfect crystal
      passes just a narrow near-Bragg sub-band (small defocus, near in-focus), while a strongly
      deformed grain -- or the rocking-integrated image -- diffracts a wide band and sees the full
      chromatic blur. (The band-selecting deviation is applied here, so this coupling is automatic.)

    ``bandwidth=0`` reduces to the monochromatic :func:`dfxm_image_dynamical`. Differentiable in
    ``hu``, the aberration, the CRL design, and (through ``spectrum``) the spectral weights.
    """
    import math
    from .takagi_taupin import susceptibility_fourier
    from .beamline import chromatic_defocus_coeffs
    from .pink import spectrum_grid, H_C_KEV_A

    if spectrum is not None:
        energies, lambdas, weights = spectrum()
    else:
        energies, lambdas, weights = spectrum_grid(E0_keV, bandwidth, n_lambda, shape)
    lam0 = H_C_KEV_A / E0_keV
    s2 = math.sin(math.radians(theta_B_deg)) ** 2
    c0_0, ch_0, chb_0 = susceptibility_fourier(crystal, hkl, wavelength_A=lam0, absorption=absorption)
    if chromatic_psf:
        defoc = chromatic_defocus_coeffs(energies, E0_keV=E0_keV, n_lenses=n_lenses,
                                         radius_um=radius_um, object_distance_m=object_distance_m,
                                         spacing_m=spacing_m)
    else:
        defoc = torch.zeros(len(lambdas), dtype=torch.float64)
    total = None
    for lam, w, cf in zip(lambdas.tolist(), weights, defoc):
        if disperse_chi:
            c0, ch, chb = susceptibility_fourier(crystal, hkl, wavelength_A=float(lam),
                                                 absorption=absorption)
            lam_eval = float(lam)
        else:
            c0, ch, chb, lam_eval = c0_0, ch_0, chb_0, lam0
        y = y0 + 2.0 * s2 * ((lam - lam0) / lam0) / (C * abs(complex(ch)))
        if abs(float(y)) > y_cut:                                 # far off-Bragg: does not diffract
            continue
        img = dfxm_image_dynamical(c0, ch, chb, wavelength_A=lam_eval, theta_B_deg=theta_B_deg,
                                   thickness_um=thickness_um, optics=optics, hu=hu, dx_um=dx_um,
                                   y=y, C=C, n_depth=n_depth, ny=ny,
                                   coherent_fraction=coherent_fraction, coeffs=coeffs, defocus=cf)
        total = w * img if total is None else total + w * img
    if total is None:
        n = ny or min(optics.detector_shape)
        return torch.zeros(n, n, dtype=torch.float64)
    return total
