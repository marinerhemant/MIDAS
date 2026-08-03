"""Coherent diffraction & differentiable phase retrieval (BCDI thrust).

Two faces of coherence:

* :func:`coherent_speckle` -- finite-size *speckle* straight from atomic
  coordinates: ``|A(Q)|^2`` on a 2-D/3-D Q grid (reuses
  :func:`midas_2d.debye.coherent_intensity`).  This is the coordinate-level
  coherent forward, differentiable w.r.t. positions.

* :func:`bcdi_forward` + :func:`phase_retrieval` -- the object-level BCDI model
  used by the Hruszkewycz/Cherukara thrust: a complex object
  ``psi(r) = rho(r) exp(i Q0 . u(r))`` (amplitude = electron density / support,
  phase = lattice displacement projected on the Bragg vector) produces speckle
  ``|FFT(psi)|^2``.  :func:`phase_retrieval` inverts it by autograd -- the
  differentiable alternative to iterative ER/HIO, and the natural place to drop
  in a learned (Cherukara-style) prior.

All complex-valued and torch-differentiable; CPU/CUDA/MPS portable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "coherent_speckle",
    "bcdi_forward",
    "phase_retrieval",
]


def coherent_speckle(coords, elements, q_grid):
    """``|A(Q)|^2`` over an arbitrary grid of Q-vectors (shape (..., 3))."""
    from .debye import coherent_intensity
    return coherent_intensity(coords, elements, q_grid)


def bcdi_forward(psi, *, shift=True):
    """Far-field coherent intensity ``|FFT(psi)|^2`` of a complex object.

    Parameters
    ----------
    psi : complex tensor
        The complex object (1-, 2-, or 3-D).
    shift : bool
        Apply ``fftshift`` so the zero frequency is centred.

    Returns
    -------
    real tensor, same shape as ``psi``.
    """
    import torch
    F = torch.fft.fftn(psi)
    if shift:
        F = torch.fft.fftshift(F)
    return F.real * F.real + F.imag * F.imag


def phase_retrieval(measured_intensity, support, *, init=None, steps=600,
                    lr=0.05, beta_tv=0.0, loss="amplitude"):
    """Differentiable (autograd) phase retrieval of a complex object.

    Fits ``|FFT(psi)|^2`` to ``measured`` over a complex object constrained to
    live inside ``support`` (a 0/1 mask, applied every forward so the constraint
    is exact and differentiable).

    Parameters
    ----------
    measured_intensity : real tensor
        Measured speckle ``|FFT(psi_true)|^2`` (fftshifted), same shape as the
        object.
    support : real tensor (0/1)
        Object-domain support mask.
    init : complex tensor, optional
        Starting object. The default is a random phase on the magnitude implied
        by Parseval, ``sqrt(sum(I) / n_voxels / n_support)``, which is the
        correct scale for a roughly uniform object.
    steps, lr : optimisation controls.
    beta_tv : float
        Optional total-variation weight on |psi| (smoothness prior).
    loss : {"amplitude", "intensity", "poisson"}
        Which residual to minimise. **The default changed to "amplitude"**;
        it matters far more than any other knob here:

        ``"amplitude"``
            ``|| |FFT(psi)| - sqrt(measured) ||^2``. Coherent diffraction spans
            many decades, so an intensity-domain L2 residual is dominated by the
            few brightest voxels -- on a typical BCDI pattern (dynamic range
            ~1e13) the brightest 0.01% of voxels carry ~90% of the loss weight,
            the fringes contribute almost no gradient, and the fit stalls far
            from the solution. Working in amplitude equalises the decades.
        ``"intensity"``
            ``|| |FFT(psi)|^2 - measured ||^2``. The historical behaviour; kept
            for reproducibility, but see above before choosing it.
        ``"poisson"``
            Poisson negative log-likelihood (via
            :func:`midas_2d.instrument.poisson_nll`), the statistically correct
            choice when ``measured`` is photon counts. The overall scale is
            profiled out, so ``measured`` need not match ``|FFT|^2`` in units.

    Returns
    -------
    dict
        ``psi`` (recovered complex object), ``history`` (loss list), ``loss``
        (the mode used).

    Notes
    -----
    The intensity determines the object only up to the conjugate twin: psi(r)
    and ``conj(psi(-r))`` produce identical ``|FFT|^2``. Score both when
    comparing against a known truth. For a displacement-encoded phase
    (``psi = s exp(-i G.u)``) the twin also flips the sign of ``u``, i.e. swaps
    tension for compression -- resolve it with known facets or a second
    reflection, not with the optimiser.

    This is a plain autograd descent, useful as a differentiable building block
    and as a place to drop in a learned prior. It is not a substitute for
    ER/HIO with shrinkwrap on hard data.
    """
    import torch

    modes = ("amplitude", "intensity", "poisson")
    if loss not in modes:
        raise ValueError(f"loss must be one of {modes}, got {loss!r}")

    measured = torch.as_tensor(measured_intensity)
    support = torch.as_tensor(support, dtype=measured.dtype, device=measured.device)
    eps = torch.finfo(measured.dtype).tiny

    if init is None:
        # Parseval for an unnormalised FFT: sum|A|^2 = n_voxels * sum|psi|^2, so a
        # roughly uniform object inside the support has |psi| = the value below.
        n_supp = torch.clamp(support.sum(), min=1.0)
        mag = torch.sqrt(measured.sum() / measured.numel() / n_supp) * support
        phase = 2 * torch.pi * torch.rand(measured.shape, device=measured.device,
                                          dtype=measured.dtype)
        init = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    re = (init.real * support).clone().detach().requires_grad_(True)
    im = (init.imag * support).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([re, im], lr=lr)

    obs_amp = torch.sqrt(torch.clamp(measured, min=0.0))
    norm_int = measured.pow(2).sum() + eps
    norm_amp = obs_amp.pow(2).sum() + eps

    history: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        psi = torch.complex(re * support, im * support)   # hard support constraint
        pred = bcdi_forward(psi)
        if loss == "intensity":
            obj = ((pred - measured) ** 2).sum() / norm_int
        elif loss == "amplitude":
            # clamp before sqrt: d(sqrt)/dx is infinite at 0 and pred hits 0 at
            # every fringe minimum.
            obj = ((torch.sqrt(torch.clamp(pred, min=eps)) - obs_amp) ** 2).sum() / norm_amp
        else:
            from .instrument import poisson_nll
            scale = (measured.sum() / torch.clamp(pred.sum(), min=eps)).detach()
            obj = poisson_nll(pred * scale, measured) / measured.numel()
        if beta_tv > 0:
            amp = psi.abs()
            tv = sum((torch.diff(amp, dim=d).abs().sum()
                      for d in range(amp.dim())))
            obj = obj + beta_tv * tv
        obj.backward()
        opt.step()
        history.append(float(obj.detach()))

    psi = torch.complex(re * support, im * support).detach()
    return {"psi": psi, "history": history, "loss": loss}
