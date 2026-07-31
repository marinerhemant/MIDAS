"""Vibration / jitter PSF for the 6-ID-C polymer-optics DFXM microscope.

The instrument paper (Qiao, Shi, Kenesei, Last, Assoufid & Islam, Rev. Sci. Instrum.
91, 113703, 2020) reports that relative positional instabilities between the objective
and the sample, in the 1--100 Hz band, degrade the long-exposure resolution by a factor
of ~2 and that the degradation *saturates* for exposures beyond ~1 s. Short exposures
freeze the jitter; summing many short exposures after image registration removes it.
This module is the forward model of that effect: it turns a relative-jitter amplitude,
a frequency band and an exposure time into (i) the effective 2-D blur PSF that convolves
the image and (ii) the resolution-vs-exposure curve -- so we can predict what acquisition
scheme recovers the diffraction+detector limit, which is the quantity a differentiable
digital twin is for.

Physics. Model the relative sample<->objective displacement per axis as a band-limited
stationary process, ``d(t) = sum_k a_k sin(2*pi*f_k*t + phi_k)`` with the ``f_k`` spanning
``[f_lo, f_hi]`` at equal power, so ``Var_t d = sigma^2`` (each mode contributes
``a_k^2/2``; ``a_k = sigma*sqrt(2/M)``). What blurs the image during an exposure of length
``tau`` is the *spread of d(t) over that window*: the recorded image is
``I(x) = (1/tau) \\int_0^tau I_true(x - d(t)) dt = I_true (*) p_tau``, where ``p_tau`` is the
occupation density of ``d(t)`` on ``[0, tau]``. As ``tau -> inf`` (many periods of the
slowest mode) ``p_tau`` fills the full stationary distribution (rms ``sigma``); as
``tau -> 0`` the jitter is frozen (``p_tau -> delta``, only a rigid offset). The blur width
therefore rises from ~0 to ``sigma`` between ``tau ~ 1/f_hi`` and ``tau ~ 1/f_lo`` and
saturates beyond -- reproducing the paper's ``>1 s`` saturation. ``sigma`` here is the
*relative* jitter; it is calibrated to the observed degradation (the LDV ambient rms
<100 nm/axis is a per-component lower bound, not the relative motion).

Everything is torch, float64, differentiable in ``sigma`` and device-portable.
"""
from __future__ import annotations

import math

import torch


def _jitter_series(sigma_um, freqs_hz, exposure_s, *, n_t=4096, n_real=64,
                   dtype=torch.float64, device=None):
    """Relative-displacement samples ``d(t)`` on ``[0, exposure]`` for one axis.

    Returns ``(n_real, n_t)``: ``n_real`` phase realisations of the band-limited process,
    each sampled at ``n_t`` times. Deterministic (fixed golden-ratio phase lattice) so the
    forward model is reproducible. Differentiable in ``sigma_um``.
    """
    sigma = torch.as_tensor(sigma_um, dtype=dtype, device=device)
    f = torch.as_tensor(freqs_hz, dtype=dtype, device=device)
    M = f.numel()
    a = sigma * math.sqrt(2.0 / M)                                   # equal-power amplitudes
    t = torch.linspace(0.0, float(exposure_s), n_t, dtype=dtype, device=device)   # (n_t,)
    # deterministic quasi-random phases (golden ratio), distinct per mode and realisation
    k = torch.arange(n_real, dtype=dtype, device=device).unsqueeze(1)             # (n_real,1)
    m = torch.arange(M, dtype=dtype, device=device).unsqueeze(0)                  # (1,M)
    phi = 2.0 * math.pi * torch.frac(0.61803398875 * (m + 1) * (k + 1))           # (n_real,M)
    # d(t) = sum_k a sin(2pi f_k t + phi) ; shape (n_real, n_t)
    arg = 2.0 * math.pi * f.view(1, M, 1) * t.view(1, 1, n_t) + phi.unsqueeze(-1)
    return (a * torch.sin(arg)).sum(dim=1)


def effective_blur_rms_um(sigma_um, freqs_hz, exposure_s, **kw) -> torch.Tensor:
    """RMS spread of the relative displacement *within* one exposure (per axis), in um.

    This is the blur that convolves the image. -> ``sigma`` for long exposures,
    -> 0 for exposures short compared with ``1/f_hi``. Differentiable in ``sigma``.
    """
    d = _jitter_series(sigma_um, freqs_hz, exposure_s, **kw)         # (n_real, n_t)
    var_t = d.var(dim=1, unbiased=False)                            # spread over the window
    return var_t.mean().sqrt()


def jitter_psf(sigma_um, pixel_um, size, exposure_s, freqs_hz, **kw) -> torch.Tensor:
    """Effective 2-D jitter PSF on a ``size x size`` pixel grid (normalised to sum 1).

    Occupation histogram of the 2-D relative displacement ``(d_x(t), d_y(t))`` over the
    exposure, binned to the detector pixel. Isotropic jitter (same ``sigma`` per axis,
    independent phase lattices). Convolve an optical PSF with this to get the recorded PSF.
    """
    dx = _jitter_series(sigma_um, freqs_hz, exposure_s, **kw).reshape(-1)
    dy = _jitter_series(sigma_um, freqs_hz, exposure_s,
                        **{**kw, "n_real": kw.get("n_real", 64) + 1}).reshape(-1)
    n = min(dx.numel(), dy.numel())
    dx, dy = dx[:n], dy[:n]
    half = size / 2.0
    ix = torch.clamp((dx / pixel_um + half).long(), 0, size - 1)
    iy = torch.clamp((dy / pixel_um + half).long(), 0, size - 1)
    psf = torch.zeros(size, size, dtype=dx.dtype, device=dx.device)
    psf.index_put_((iy, ix), torch.ones_like(dx), accumulate=True)
    return psf / psf.sum()


def resolution_vs_exposure(sigma_um, base_res_um, freqs_hz, exposures_s, **kw) -> torch.Tensor:
    """Predicted resolution (um) vs exposure: ``sqrt(base^2 + blur(tau)^2)`` per exposure.

    ``base_res_um`` is the vibration-free resolution (diffraction (+) detector, in
    quadrature). Reproduces the paper's factor-of-~2 long-exposure degradation and its
    saturation beyond ~1 s. Differentiable in ``sigma`` and ``base_res``.
    """
    base = torch.as_tensor(base_res_um, dtype=torch.float64, device=kw.get("device"))
    out = []
    for tau in exposures_s:
        blur = effective_blur_rms_um(sigma_um, freqs_hz, tau, **kw)
        out.append(torch.sqrt(base ** 2 + blur ** 2))
    return torch.stack(out)


def frames_to_recover(sigma_um, base_res_um, freqs_hz, target_res_um, short_exposure_s,
                      total_dose_s, **kw) -> dict:
    """Design a short-exposure + register + sum acquisition that recovers ``target_res``.

    A single short exposure of length ``short_exposure_s`` has resolution
    ``sqrt(base^2 + blur(short)^2)``; registration + sum of ``N`` such frames keeps that
    single-frame resolution while reaching the photon statistics of the long exposure.
    Returns the single-short-frame resolution, whether it meets ``target_res``, and the
    number of frames ``N = total_dose / short_exposure`` needed for equal dose.
    """
    short_res = float(torch.sqrt(
        torch.as_tensor(base_res_um, dtype=torch.float64) ** 2
        + effective_blur_rms_um(sigma_um, freqs_hz, short_exposure_s, **kw) ** 2))
    return {
        "short_frame_res_um": short_res,
        "meets_target": short_res <= target_res_um,
        "n_frames": int(math.ceil(total_dose_s / short_exposure_s)),
        "short_exposure_s": short_exposure_s,
    }
