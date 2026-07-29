"""Turn diffraction signals into transport / coupling *coefficients*.

The forwards in this package already predict how a Bragg reflection's amplitude
and position evolve.  This module adds the physical models whose parameters are
the numbers an ultrafast-science room actually wants, and fits them through the
differentiable diffraction observables:

* **Two-temperature model** (:func:`two_temperature_model`) -- a pump deposits
  energy into the electrons, which transfer it to the lattice at a rate set by
  the electron-phonon coupling ``g``.  The lattice temperature drives the
  Debye-Waller amplitude drop, so :func:`fit_electron_phonon_coupling` recovers
  ``g`` from the Bragg-intensity time series.

* **1-D heat diffusion** (:func:`heat_diffusion_1d`) -- a surface-deposited
  temperature spreads through the film with diffusivity ``kappa``; the depth
  temperature profile drives a depth-dependent strain (and hence the asymmetric,
  shifting Bragg peak), so :func:`fit_thermal_diffusivity` recovers ``kappa``
  from the depth-resolved diffraction time series.

Everything is a fixed-grid, differentiable solver (RK4 in time / explicit FTCS
in space), so gradients flow from the diffraction loss to the coefficients.
Units are reduced unless physical constants are supplied.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "two_temperature_model",
    "lattice_T_to_intensity_ratio",
    "fit_electron_phonon_coupling",
    "heat_diffusion_1d",
    "fit_thermal_diffusivity",
]


# ============================================================ two-temperature

def _pump_source(t, amp, t0, width):
    import torch
    return amp * torch.exp(-0.5 * ((t - t0) / width) ** 2)


def two_temperature_model(t, *, g, C_e=1.0, C_l=3.0, pump_amp=1.0, pump_t0=0.3,
                          pump_width=0.08, Te0=0.0, Tl0=0.0):
    """Solve the coupled two-temperature ODEs on the uniform grid ``t``.

        C_e dTe/dt = -g (Te - Tl) + S(t)
        C_l dTl/dt =  g (Te - Tl)

    ``S(t)`` is a Gaussian pump.  RK4; differentiable in ``g`` (and the heat
    capacities / pump params).  Returns ``(Te, Tl)`` each shape ``t.shape``.
    """
    import torch
    t = torch.as_tensor(t)
    g = torch.as_tensor(g, dtype=t.dtype, device=t.device)
    C_e = torch.as_tensor(C_e, dtype=t.dtype, device=t.device)
    C_l = torch.as_tensor(C_l, dtype=t.dtype, device=t.device)
    dt = (t[1] - t[0])

    def deriv(tc, Te, Tl):
        S = _pump_source(tc, pump_amp, pump_t0, pump_width)
        dTe = (-g * (Te - Tl) + S) / C_e
        dTl = (g * (Te - Tl)) / C_l
        return dTe, dTl

    Te_list = [torch.as_tensor(Te0, dtype=t.dtype, device=t.device)]
    Tl_list = [torch.as_tensor(Tl0, dtype=t.dtype, device=t.device)]
    for n in range(t.numel() - 1):
        tc, Te, Tl = t[n], Te_list[-1], Tl_list[-1]
        k1e, k1l = deriv(tc, Te, Tl)
        k2e, k2l = deriv(tc + dt / 2, Te + dt / 2 * k1e, Tl + dt / 2 * k1l)
        k3e, k3l = deriv(tc + dt / 2, Te + dt / 2 * k2e, Tl + dt / 2 * k2l)
        k4e, k4l = deriv(tc + dt, Te + dt * k3e, Tl + dt * k3l)
        Te_list.append(Te + dt / 6 * (k1e + 2 * k2e + 2 * k3e + k4e))
        Tl_list.append(Tl + dt / 6 * (k1l + 2 * k2l + 2 * k3l + k4l))
    return torch.stack(Te_list), torch.stack(Tl_list)


def lattice_T_to_intensity_ratio(Tl, *, q_perp, k_spring, kB=1.0):
    """Bragg-intensity ratio ``I(t)/I(0) = exp(-q_perp^2 dU^2)`` from the lattice
    temperature, with the excess MSD ``dU^2 = kB Tl / k_spring`` (equipartition)."""
    import torch
    Tl = torch.as_tensor(Tl)
    dU2 = kB * Tl / float(k_spring)
    return torch.exp(-(float(q_perp) ** 2) * dU2)


def fit_electron_phonon_coupling(obs_ratio, t, *, q_perp, k_spring, C_e=1.0,
                                 C_l=3.0, pump_t0=0.3, pump_width=0.08,
                                 init_g=0.5, init_amp=1.0, steps=1500, lr=0.03,
                                 kB=1.0):
    """Recover the electron-phonon coupling ``g`` (and pump amplitude) from a
    measured Bragg-intensity ratio time series.

    Returns dict with ``g``, ``pump_amp`` (floats) and the loss history.
    """
    import torch
    from .inverse import fit, relative_l2_loss

    obs_ratio = torch.as_tensor(obs_ratio)
    dt_t = obs_ratio.dtype
    raw_g = torch.tensor(math.log(math.expm1(init_g)), dtype=dt_t, requires_grad=True)
    raw_amp = torch.tensor(math.log(math.expm1(init_amp)), dtype=dt_t, requires_grad=True)
    sp = torch.nn.functional.softplus

    def loss_fn():
        Te, Tl = two_temperature_model(t, g=sp(raw_g), C_e=C_e, C_l=C_l,
                                       pump_amp=sp(raw_amp), pump_t0=pump_t0,
                                       pump_width=pump_width)
        pred = lattice_T_to_intensity_ratio(Tl, q_perp=q_perp, k_spring=k_spring, kB=kB)
        return relative_l2_loss(pred, obs_ratio)

    out = fit([raw_g, raw_amp], loss_fn, steps=steps, lr=lr)
    return {"g": float(sp(raw_g).detach()), "pump_amp": float(sp(raw_amp).detach()),
            "loss": out["loss"]}


# ============================================================== heat diffusion

def heat_diffusion_1d(T0, *, kappa, dz, dt, n_steps):
    """Evolve ``dT/dt = kappa d^2T/dz^2`` with insulated (Neumann) boundaries.

    Explicit FTCS; differentiable in ``kappa`` and the initial field ``T0``.
    Requires ``kappa*dt/dz^2 <= 0.5`` for stability.

    Parameters
    ----------
    T0 : tensor (Nz,)
        Initial depth temperature profile (e.g. surface-deposited).
    Returns
    -------
    T : tensor (n_steps+1, Nz)
        Temperature field over time.
    """
    import torch
    T0 = torch.as_tensor(T0)
    kappa = torch.as_tensor(kappa, dtype=T0.dtype, device=T0.device)
    r = kappa * dt / (dz * dz)
    frames = [T0]
    T = T0
    for _ in range(n_steps):
        lap = torch.zeros_like(T)
        lap[1:-1] = T[2:] - 2 * T[1:-1] + T[:-2]
        # Conservative zero-flux (Neumann) ends: single-sided flux so the
        # total heat is exactly conserved (sum of lap telescopes to 0).
        lap[0] = T[1] - T[0]
        lap[-1] = T[-2] - T[-1]
        T = T + r * lap
        frames.append(T)
    return torch.stack(frames)


def fit_thermal_diffusivity(obs_strain_zt, z, T0, *, alpha, dt, n_steps,
                            init_kappa=0.5, steps=400, lr=0.05):
    """Recover the thermal diffusivity ``kappa`` from a depth-resolved strain
    time series ``eps(z, t)`` (strain = ``alpha * T``).

    ``obs_strain_zt`` : tensor (n_steps+1, Nz).
    Returns dict with ``kappa`` (float) and loss.
    """
    import torch
    from .inverse import fit, relative_l2_loss

    z = torch.as_tensor(z)
    dz = float(z[1] - z[0])
    obs = torch.as_tensor(obs_strain_zt)
    raw = torch.tensor(math.log(math.expm1(init_kappa)), dtype=obs.dtype,
                       requires_grad=True)
    sp = torch.nn.functional.softplus

    def loss_fn():
        T = heat_diffusion_1d(T0, kappa=sp(raw), dz=dz, dt=dt, n_steps=n_steps)
        pred = float(alpha) * T
        return relative_l2_loss(pred, obs)

    out = fit([raw], loss_fn, steps=steps, lr=lr)
    return {"kappa": float(sp(raw).detach()), "loss": out["loss"]}
