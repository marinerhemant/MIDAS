"""Detector realism: point spread, photon statistics, read noise.

The forward model up to here is noiseless and treats a pixel as a point sample of
a mass-conserving splat. That is the right core -- it keeps the operator an exact
Radon transform, which is what made the external validation against
``skimage.transform.radon`` meaningful at 4e-12 -- but it is not what a detector
records, and conclusions drawn from noiseless data have already misled this
package once: every C2 number was noiseless, and adding the pre-registered 0.1%
noise moved the sym-closed leakage from 1.8e-4 deg to 4.3e-3 +/- 2.1e-3 deg, i.e.
**24x the effect being measured**, and flipped the verdict.

So this module exists to make "with noise" the default way results are reported,
per ``dev/paper/PREREGISTER.md`` item 2.

Relationship to :mod:`midas_dfxm.detector`
------------------------------------------
``midas_dfxm.detector`` already provides ``apply_psf``, ``gaussian_psf_kernel``,
``add_detector_noise``, ``quantize_16bit`` and a full ``detector_model`` chain, and
**that module should be preferred for any single 2-D DFXM image.** This one exists
for two measured reasons, not by oversight:

1. **Batching.** TT works on ``(n_psi, nu, nv)`` stacks;
   ``midas_dfxm.detector.apply_psf`` raises ``NotImplementedError`` on a 5-D input,
   so it cannot process a scan.
2. **Edge physics.** It reflect-pads, which at a detector edge *creates* flux --
   measured **1.034** of the input for a spot 4 px from the edge. Zero padding
   gives **0.952**, i.e. the flux that blurs past the edge is lost, which is what
   a detector does. For TT that matters: the spot moves during a ``psi`` scan, so
   an edge artefact becomes ``psi``-modulated and imitates real signal.

In the regime where both apply (interior spot, 2-D) they agree to ~1e-3, the
residual being kernel radius ``3 sigma`` there against ``4 sigma`` here;
``tests/test_detector.py`` pins that agreement so the two cannot drift apart.
For 16-bit quantisation and the full camera chain, call
``midas_dfxm.detector.quantize_16bit`` / ``detector_model`` -- not reimplemented here.

What is and is not differentiable
---------------------------------
:func:`apply_psf` is differentiable and belongs in the *forward model* -- a real
detector's blur is part of the operator, and a reconstruction that ignores it
solves the wrong problem. The noise functions are **data generation only**:
``torch.poisson`` has no useful gradient and none is wanted, since noise is not a
parameter to fit. Keeping the two separated in the API prevents the common error
of blurring the data but not the model.

Ordering matters and is not interchangeable
-------------------------------------------
Physically: photons arrive and are counted, the scintillator/optic then spreads
each event, and the result is read out. So Poisson statistics apply to the
*unblurred* photon field and read noise is added *after* the PSF.
:func:`simulate_detector` does it in that order.

Blurring already-Poisson data produces **spatially correlated** noise; sampling
Poisson *after* the blur produces **white** noise. Getting it backwards misstates
the correlation structure, which is what a least-squares fit assumes away when it
treats residuals as independent.

.. warning::
   **This models only ONE Poisson stage, and an indirect detector has two.** A
   scintillator + optics + camera is a Poisson *cluster* process: N x-rays, each
   producing K ~ Poisson(m-bar) detected photo-electrons, each landing via the
   PSF. That gives ``rho(1) = C(1) / (1/m-bar + C(0))`` with
   ``C(d) = sum_x PSF(x) PSF(x+d)``, so the ~0.9 correlation quoted here is the
   **m-bar -> infinity limit**, i.e. an upper bound, not a value:

   ========  ==========  ===================
   m-bar     lag-1 rho   variance vs 1-stage
   ========  ==========  ===================
   5         0.163       6.66x
   30        0.466       1.98x
   100       0.693       1.32x
   infinity  0.896       1.00x
   ========  ==========  ===================

   At 71.7 keV with LuAG:Ce (~25 ph/keV), finite-NA collection (~1-5%) and ~0.7
   camera QE, **m-bar ~ 10-70** -- so the real detector sits near rho ~ 0.5 with
   roughly **twice** the pixel variance this model gives. Any covariance or
   error-bar reasoning built on ~0.9 is optimistic by about 2x.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "add_photon_noise",
    "effective_dof",
    "noise_power_spectrum",
    "add_two_stage_noise",
    "add_read_noise",
    "apply_psf",
    "gaussian_kernel_1d",
    "simulate_detector",
]


def gaussian_kernel_1d(sigma_px: float, *, truncate: float = 4.0, dtype=None,
                       device=None) -> torch.Tensor:
    """Normalised 1-D Gaussian kernel, radius ``ceil(truncate * sigma)``.

    Normalised to sum exactly 1 so that convolution conserves total counts --
    which is what makes a blurred topograph still integrate to the same path
    length.
    """
    if sigma_px <= 0:
        raise ValueError("sigma_px must be > 0")
    r = int(math.ceil(truncate * float(sigma_px)))
    x = torch.arange(-r, r + 1, dtype=dtype or torch.float64, device=device)
    k = torch.exp(-0.5 * (x / float(sigma_px)) ** 2)
    return k / k.sum()


def apply_psf(image: torch.Tensor, sigma_px: float, *, truncate: float = 4.0
              ) -> torch.Tensor:
    """Separable Gaussian point spread. ``(..., nu, nv) -> (..., nu, nv)``.

    Differentiable in ``image``. Edges are handled by zero padding, i.e. flux that
    blurs off the detector is lost, which is what a real detector does -- and is
    consistent with :func:`~midas_dct_tt.project.offdetector_fraction`, so the two
    loss channels do not double count.

    A symmetric PSF leaves the centroid of an on-detector spot unchanged **in the
    interior** (verified to 3.6e-15, flux to 1.0000000000); it reduces peak height
    and broadens the profile. That is why blur degrades the *intensity* channel
    more than the *position* channel.

    .. warning::
       **Near an edge that fails.** Zero padding truncates the kernel, so flux is
       lost asymmetrically and the centroid moves. At ``sigma = 2`` px (kernel
       radius 8):

       ==================  ==========  ==================
       distance from edge  flux kept   centroid shift px
       ==================  ==========  ==================
       12                  0.9999959   5.4e-05
       8                   0.9987506   1.2e-02
       6                   0.9898836   7.7e-02
       4                   0.9520241   **0.287**
       2                   0.8697036   0.622
       ==================  ==========  ==================

       At ``pixel_um = 1`` a spot 4 px from the edge acquires a 0.287 um
       systematic -- 43% of the 0.66 um centroid separation the B2 result rests
       on. Worse, a spot moves during a ``psi`` scan, so the bias is
       ``psi``-modulated, which is exactly the signature the anisotropic-acceptance
       intensity channel looks for. Keep spots away from the edge, or crop.
    """
    if sigma_px == 0:
        return image
    k = gaussian_kernel_1d(sigma_px, truncate=truncate, dtype=image.dtype,
                           device=image.device)
    r = (k.numel() - 1) // 2
    lead = image.shape[:-2]
    x = image.reshape(-1, 1, *image.shape[-2:])
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    return x.reshape(*lead, *image.shape[-2:])


def add_photon_noise(image: torch.Tensor, photons_per_unit: float, *,
                     generator: torch.Generator = None) -> torch.Tensor:
    """Poisson photon statistics. ``photons_per_unit`` converts image units to counts.

    Returns the image back in its original units, so the only visible effect is
    the noise: the relative noise level is ``1/sqrt(photons)``, which is the
    number to quote. Negative values (impossible physically, but reachable if a
    caller has already added read noise) are clamped to zero before sampling.

    **Data generation only** -- not differentiable.
    """
    if photons_per_unit <= 0:
        raise ValueError("photons_per_unit must be > 0")
    counts = (image.detach() * float(photons_per_unit)).clamp_min(0.0)
    return torch.poisson(counts, generator=generator) / float(photons_per_unit)


def add_read_noise(image: torch.Tensor, sigma: float, *,
                   generator: torch.Generator = None) -> torch.Tensor:
    """Additive Gaussian read noise of standard deviation ``sigma`` (image units).

    **Data generation only** -- not differentiable.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return image
    n = torch.randn(image.shape, generator=generator, dtype=image.dtype,
                    device=image.device)
    return image.detach() + n * float(sigma)


def add_two_stage_noise(image: torch.Tensor, xrays_per_unit: float,
                        photoelectrons_per_xray: float, psf_px: float, *,
                        generator: torch.Generator = None) -> torch.Tensor:
    """Indirect-detection noise: an X-ray Poisson stage, then a photo-electron one.

    A scintillator + optics + camera is a Poisson **cluster** process, not a single
    Poisson: ``N`` X-rays arrive, each liberates ``K ~ Poisson(m_bar)`` *detected*
    photo-electrons, and each of those lands independently via the PSF. Sampling
    that faithfully needs both stages::

        N   ~ Poisson(lambda)                # X-ray arrivals, unblurred
        S   ~ Poisson( PSF (x) (m_bar * N) ) # spread, then detection/landing

    which reproduces the analytic moments exactly::

        Var(S)  = m_bar*lambda + m_bar^2*lambda*C(0)
        rho(1)  = C(1) / (1/m_bar + C(0)),   C(d) = sum_x PSF(x) PSF(x+d)

    Why the single-stage model is optimistic. It is the ``m_bar -> infinity`` limit,
    where the landing term vanishes and only the correlated cluster term survives:

    ========  ==========  ===================
    m_bar     lag-1 rho   variance vs 1-stage
    ========  ==========  ===================
    5         0.163       6.66x
    30        0.466       1.98x
    100       0.693       1.32x
    infinity  0.896       1.00x
    ========  ==========  ===================

    At 71.7 keV with LuAG:Ce (~25 ph/keV), finite-NA collection (~1-5%) and ~0.7
    camera QE, **m_bar ~ 10-70** -- so a real detector sits near ``rho ~ 0.5`` with
    roughly **twice** the pixel variance the single-stage model reports. Any
    error-bar or covariance argument built on the single-stage figure is optimistic
    by about 2x.

    ``xrays_per_unit`` converts image units to X-ray counts; the result is returned
    in the original units. **Data generation only** -- not differentiable.
    """
    if xrays_per_unit <= 0 or photoelectrons_per_xray <= 0:
        raise ValueError("xrays_per_unit and photoelectrons_per_xray must be > 0")
    lam = (image.detach() * float(xrays_per_unit)).clamp_min(0.0)
    n_xray = torch.poisson(lam, generator=generator)
    mu = n_xray * float(photoelectrons_per_xray)
    if psf_px:
        mu = apply_psf(mu, psf_px)
    s = torch.poisson(mu.clamp_min(0.0), generator=generator)
    return s / (float(xrays_per_unit) * float(photoelectrons_per_xray))


def simulate_detector(image: torch.Tensor, *, psf_px: float = 0.0,
                      photons_per_unit: float = None, read_noise: float = 0.0,
                      photoelectrons_per_xray: float = None,
                      generator: torch.Generator = None) -> torch.Tensor:
    """Photon noise -> PSF -> read noise, in that physical order.

    Pass ``photons_per_unit=None`` to skip the Poisson stage (e.g. when quoting a
    labelled noiseless sanity check). Note that only ``psf_px`` belongs in the
    *model* as well as the data: if you simulate with a PSF you must reconstruct
    with one, or the fit is solving a different operator than the one that made
    the data.

    Set ``photoelectrons_per_xray`` to use the physically correct **two-stage**
    chain (:func:`add_two_stage_noise`) instead of the single-Poisson
    approximation. Prefer it for anything that will carry an error bar: the
    single-stage form is the infinite-light-yield limit and understates the
    variance of a real indirect detector by about 2x.
    """
    if photoelectrons_per_xray is not None:
        if photons_per_unit is None:
            raise ValueError(
                "photoelectrons_per_xray needs photons_per_unit, which sets the "
                "X-ray arrival rate for the first Poisson stage"
            )
        out = add_two_stage_noise(image, photons_per_unit, photoelectrons_per_xray,
                                  psf_px, generator=generator)
        return add_read_noise(out, read_noise, generator=generator) if read_noise else out

    out = image
    if photons_per_unit is not None:
        out = add_photon_noise(out, photons_per_unit, generator=generator)
    if psf_px:
        out = apply_psf(out, psf_px)
    if read_noise:
        out = add_read_noise(out, read_noise, generator=generator)
    return out


def noise_power_spectrum(shape, psf_px: float, *,
                         photoelectrons_per_xray: float = None,
                         dtype=torch.float64, device=None) -> torch.Tensor:
    """Power spectrum of the detector noise, ``(nu, nv)``, up to an overall scale.

    For the two-stage chain the noise is white *landing* plus PSF-correlated
    *cluster*, so in Fourier space::

        S(k) = 1/m_bar + |PSF(k)|^2

    The white term floors the spectrum, which is what makes whitening
    well-conditioned: with a pure convolution (``m_bar -> infinity``) the inverse
    filter would blow up at high frequency.
    """
    nu, nv = shape
    mb = float("inf") if photoelectrons_per_xray is None else float(photoelectrons_per_xray)
    if psf_px <= 0:
        return torch.ones(nu, nv, dtype=dtype, device=device)
    k = gaussian_kernel_1d(psf_px, dtype=dtype, device=device)
    r = (k.numel() - 1) // 2

    def _p2(n):
        pad = torch.zeros(n, dtype=dtype, device=device)
        pad[:r + 1] = k[r:]
        pad[n - r:] = k[:r]
        return torch.fft.fft(pad).abs() ** 2

    floor = 0.0 if mb == float("inf") else 1.0 / mb
    return floor + _p2(nu).view(-1, 1) * _p2(nv).view(1, -1)


def effective_dof(shape, psf_px: float, *, photoelectrons_per_xray: float = None,
                  mode: str = "chi2") -> float:
    """Fraction of pixels that are statistically **independent**, in ``(0, 1]``.

    A detector PSF correlates neighbouring pixels, so an unweighted sum of squares
    -- which is what :func:`~midas_dct_tt.inverse.profiled_intensity_residual` and
    :func:`~midas_dct_tt.field_inverse.fit_deformation_field` use -- overcounts the
    independent information. That does **not** bias the fitted values (ordinary
    least squares stays unbiased under correlated noise); it makes the *error bars
    and chi-squared* optimistic.

    There are **two** different answers and they are not interchangeable -- a
    distinction easy to get wrong, and worth 3x here:

    ``mode="chi2"``
        ``(sum S)^2 / (N sum S^2)`` -- for a goodness-of-fit sum or a variance
        estimate, where every frequency contributes.
    ``mode="mean"``
        ``mean(S) / S(0)`` -- for a **global or low-frequency parameter** (a
        uniform ``F``, an intensity scale, a rigid rotation), which is dominated
        by the ``k = 0`` mode where the correlated noise piles up. This is the one
        that applies to most fitted quantities, and it is much the harsher.

    Measured at ``m_bar = 30``:

    ==========  ==============  ==============  ====================
    PSF sigma   N_eff/N (chi2)  N_eff/N (mean)  error bars optimistic
    ==========  ==============  ==============  ====================
    0.8 px      0.3507          0.1535          2.6x
    1.5 px      0.2231          0.0665          **3.9x**
    ==========  ==============  ==============  ====================

    Validated against direct simulation: 400 realisations at ``sigma = 1.5``,
    ``m_bar = 30`` give an empirical ``N_eff/N`` of **0.073** against the predicted
    0.0665, the residual being finite-size and edge effects.

    **Multiply a fitted parameter's standard error by ``1/sqrt(effective_dof)``**
    before quoting it.
    """
    S = noise_power_spectrum(shape, psf_px,
                             photoelectrons_per_xray=photoelectrons_per_xray)
    if mode == "chi2":
        return float(S.sum() ** 2 / (S.numel() * (S * S).sum()))
    if mode == "mean":
        return float(S.mean() / S.reshape(-1)[0])
    raise ValueError(f"mode must be 'chi2' or 'mean', got {mode!r}")
