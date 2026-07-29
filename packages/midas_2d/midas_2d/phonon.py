"""Coherent-phonon dynamics: the oscillatory ultrafast signal.

A femtosecond pump launches a *coherent* lattice motion -- most visibly an
out-of-plane acoustic "breathing" mode of a platelet -- that shows up as a
damped oscillation of Bragg intensity and peak position vs. pump-probe delay.
Here that is modelled as a transient out-of-plane strain wave

    eps(t) = A * exp(-t / tau) * cos(2 pi f t + phi)

applied to the coordinates (the c-axis dilates by ``1 + eps``), forwarded
through the coherent engine, and -- crucially -- *fit* back to recover the
phonon frequency ``f``, damping ``tau`` and amplitude ``A`` from the measured
time series.  Everything differentiable.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "strain_wave",
    "apply_out_of_plane_strain",
    "bragg_timeseries",
    "fit_coherent_phonon",
]


def strain_wave(t, amp, freq, tau, phi=0.0):
    """Damped cosine strain ``A exp(-t/tau) cos(2 pi f t + phi)``."""
    import torch
    t = torch.as_tensor(t)
    amp = torch.as_tensor(amp, dtype=t.dtype, device=t.device)
    freq = torch.as_tensor(freq, dtype=t.dtype, device=t.device)
    tau = torch.as_tensor(tau, dtype=t.dtype, device=t.device)
    phi = torch.as_tensor(phi, dtype=t.dtype, device=t.device)
    return amp * torch.exp(-t / tau) * torch.cos(2 * math.pi * freq * t + phi)


def apply_out_of_plane_strain(coords, eps):
    """Dilate the z-coordinate by ``(1 + eps)`` (uniform out-of-plane strain)."""
    import torch
    coords = torch.as_tensor(coords)
    scale = torch.ones(3, dtype=coords.dtype, device=coords.device)
    scale = torch.stack([scale[0], scale[1], 1.0 + torch.as_tensor(eps, dtype=coords.dtype,
                                                                   device=coords.device)])
    return coords * scale


def bragg_timeseries(coords0, elements, q_vec, t, amp, freq, tau, phi=0.0):
    """Coherent Bragg intensity at ``q_vec`` for each delay in ``t``, under a
    transient out-of-plane strain wave.

    Returns a tensor shape ``t.shape`` (one intensity per delay).
    """
    import torch
    from .debye import coherent_intensity

    t = torch.as_tensor(t)
    eps_t = strain_wave(t, amp, freq, tau, phi)              # (T,)
    out = []
    for e in eps_t:
        c = apply_out_of_plane_strain(coords0, e)
        out.append(coherent_intensity(c, elements, q_vec.reshape(1, 3))[0])
    return torch.stack(out)


def fit_coherent_phonon(observed, coords0, elements, q_vec, t, *,
                        init=None, steps=1500, lr=0.02):
    """Recover ``(amp, freq, tau)`` from a measured Bragg-intensity time series.

    Frequency and tau are kept positive via softplus.  Returns a dict with the
    recovered floats and the loss history.
    """
    import torch
    from .inverse import cosine_loss, fit

    observed = torch.as_tensor(observed)
    dt, dev = observed.dtype, observed.device

    if init is None:
        init = {"amp": 0.01, "freq": 1.0, "tau": 1.0}
    raw_amp = torch.tensor(init["amp"], dtype=dt, device=dev, requires_grad=True)
    raw_freq = torch.tensor(math.log(math.expm1(init["freq"])), dtype=dt, device=dev,
                            requires_grad=True)
    raw_tau = torch.tensor(math.log(math.expm1(init["tau"])), dtype=dt, device=dev,
                           requires_grad=True)
    sp = torch.nn.functional.softplus

    def loss_fn():
        pred = bragg_timeseries(coords0, elements, q_vec, t,
                                raw_amp, sp(raw_freq), sp(raw_tau))
        # scale-invariant on shape + a small absolute term to pin amplitude
        return cosine_loss(pred, observed) + 0.05 * ((pred - observed) ** 2).mean()

    fit([raw_amp, raw_freq, raw_tau], loss_fn, steps=steps, lr=lr)
    return {
        "amp": float(raw_amp.detach().abs()),
        "freq": float(sp(raw_freq).detach()),
        "tau": float(sp(raw_tau).detach()),
    }
