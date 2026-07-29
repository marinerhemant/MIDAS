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
                    lr=0.05, beta_tv=0.0):
    """Differentiable (autograd) phase retrieval of a complex object.

    Minimises ``|| |FFT(psi)|^2 - measured ||^2`` over a complex object
    constrained to live inside ``support`` (a 0/1 mask, applied every forward
    so the constraint is exact and differentiable).

    Parameters
    ----------
    measured_intensity : real tensor
        Measured speckle ``|FFT(psi_true)|^2`` (fftshifted), same shape as the
        object.
    support : real tensor (0/1)
        Object-domain support mask.
    init : complex tensor, optional
        Starting object; defaults to a random object inside the support seeded
        from the measured magnitude.
    steps, lr : optimisation controls.
    beta_tv : float
        Optional total-variation weight on |psi| (smoothness prior).

    Returns
    -------
    dict: ``psi`` (recovered complex object), ``history`` (loss list).
    """
    import torch

    measured = torch.as_tensor(measured_intensity)
    support = torch.as_tensor(support, dtype=measured.dtype, device=measured.device)

    if init is None:
        # Random phase on a magnitude guess derived from the measured speckle.
        mag = torch.sqrt(torch.clamp(measured.mean()) + 0.0) * support
        phase = 2 * torch.pi * torch.rand(measured.shape, device=measured.device,
                                          dtype=measured.dtype)
        init = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    re = (init.real * support).clone().detach().requires_grad_(True)
    im = (init.imag * support).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([re, im], lr=lr)

    history: list[float] = []
    obs_norm = measured.pow(2).sum() + 1e-12
    for _ in range(steps):
        opt.zero_grad()
        psi = torch.complex(re * support, im * support)   # hard support constraint
        pred = bcdi_forward(psi)
        loss = ((pred - measured) ** 2).sum() / obs_norm
        if beta_tv > 0:
            amp = psi.abs()
            tv = sum((torch.diff(amp, dim=d).abs().sum()
                      for d in range(amp.dim())))
            loss = loss + beta_tv * tv
        loss.backward()
        opt.step()
        history.append(float(loss))

    psi = torch.complex(re * support, im * support).detach()
    return {"psi": psi, "history": history}
