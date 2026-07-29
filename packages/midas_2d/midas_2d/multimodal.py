"""Multi-modal fusion: X-ray structure + optical electronic signal.

The single most group-specific frontier capability (Schaller does ultrafast
*optical*; the X-ray gives structure).  After a pump, the lattice strain has two
contributions that are *degenerate in the X-ray data alone*:

* an **electronic** part -- carriers deform the lattice through the deformation
  potential ``Xi``; it tracks the carrier density ``n(t)`` (fast, recombines),
* a **thermal** part -- ``alpha * T(t)``; energy that has flowed to the lattice
  (slow, accumulates).

A transient-absorption (optical) trace ``O(t) ~ n(t)`` sees ONLY the electronic
population, so fusing it with the X-ray strain breaks the degeneracy and pins
the **deformation potential Xi** and the thermal coefficient separately.

All differentiable; :func:`fit_multimodal` does the joint inversion and
:func:`xray_only_degeneracy` demonstrates why one probe is not enough.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "carrier_density",
    "lattice_temperature_from_carriers",
    "strain_two_channel",
    "optical_signal",
    "fit_multimodal",
    "xray_only_degeneracy",
]


def carrier_density(t, amp, tau_rise, tau_decay):
    """Excited carrier population ``n(t) = amp (1 - e^{-t/tr}) e^{-t/td}``."""
    import torch
    t = torch.as_tensor(t)
    amp = torch.as_tensor(amp, dtype=t.dtype, device=t.device)
    tr = torch.as_tensor(tau_rise, dtype=t.dtype, device=t.device)
    td = torch.as_tensor(tau_decay, dtype=t.dtype, device=t.device)
    return amp * (1.0 - torch.exp(-t / tr)) * torch.exp(-t / td)


def lattice_temperature_from_carriers(t, n, tau_ep):
    """Lattice temperature from energy flowing out of the carriers:
    ``dT/dt = n / tau_ep`` -> ``T(t) = cumtrapz(n) / tau_ep`` (monotonic rise)."""
    import torch
    t = torch.as_tensor(t)
    n = torch.as_tensor(n, dtype=t.dtype, device=t.device)
    tau_ep = torch.as_tensor(tau_ep, dtype=t.dtype, device=t.device)  # keep grad
    dt = torch.diff(t, prepend=t[:1])
    return torch.cumsum(n * dt, dim=0) / tau_ep


def strain_two_channel(n, T, Xi, alpha):
    """Lattice strain ``eps(t) = Xi * n(t) + alpha * T(t)`` (electronic +
    thermal)."""
    import torch
    n = torch.as_tensor(n)
    T = torch.as_tensor(T, dtype=n.dtype, device=n.device)
    return torch.as_tensor(Xi, dtype=n.dtype, device=n.device) * n + \
        torch.as_tensor(alpha, dtype=n.dtype, device=n.device) * T


def optical_signal(n, c_opt=1.0):
    """Transient-absorption signal ``O(t) = c_opt * n(t)`` (electronic only)."""
    import torch
    return float(c_opt) * torch.as_tensor(n)


def _forward(t, p):
    """Build (optical, strain) from a parameter dict of tensors."""
    n = carrier_density(t, p["amp"], p["tau_rise"], p["tau_decay"])
    T = lattice_temperature_from_carriers(t, n, p["tau_ep"])
    eps = strain_two_channel(n, T, p["Xi"], p["alpha"])
    O = optical_signal(n, 1.0)
    return O, eps


def fit_multimodal(O_obs, S_obs, t, *, use_optical=True, init=None, steps=2000,
                   lr=0.03):
    """Joint inversion of an optical trace ``O_obs`` and an X-ray strain trace
    ``S_obs`` for the deformation potential ``Xi`` and thermal coefficient
    ``alpha`` (plus carrier time constants).

    Set ``use_optical=False`` to fit the X-ray alone (for the degeneracy demo).
    Returns dict of recovered floats.
    """
    import torch
    from .inverse import fit, relative_l2_loss

    t = torch.as_tensor(t)
    O_obs = torch.as_tensor(O_obs)
    S_obs = torch.as_tensor(S_obs)
    if init is None:
        init = dict(amp=0.5, tau_rise=0.1, tau_decay=1.5, tau_ep=1.0, Xi=0.5, alpha=0.5)

    sp = torch.nn.functional.softplus
    raw = {k: torch.tensor(math.log(math.expm1(v)), dtype=t.dtype, requires_grad=True)
           for k, v in init.items()}

    def params():
        return {k: sp(v) for k, v in raw.items()}

    def loss_fn():
        O, eps = _forward(t, params())
        # normalise each channel so neither dominates by scale
        xray = relative_l2_loss(eps, S_obs)
        if use_optical:
            return xray + relative_l2_loss(O, O_obs)
        return xray

    fit(list(raw.values()), loss_fn, steps=steps, lr=lr)
    return {k: float(sp(v).detach()) for k, v in raw.items()}


def xray_only_degeneracy(t, S_obs, Xi_grid, *, fixed_taus, init_alpha=0.5,
                         steps=400, lr=0.05):
    """For each candidate ``Xi`` on a grid, fit ``alpha`` (X-ray strain only) and
    report the residual -- a flat residual valley shows the electronic/thermal
    degeneracy that the optical channel removes.

    Returns ``(Xi_grid, residual)``.
    """
    import torch
    from .inverse import fit, relative_l2_loss

    t = torch.as_tensor(t)
    S_obs = torch.as_tensor(S_obs)
    sp = torch.nn.functional.softplus
    res = []
    for Xi in Xi_grid:
        raw_alpha = torch.tensor(math.log(math.expm1(init_alpha)), dtype=t.dtype,
                                 requires_grad=True)

        def loss_fn():
            n = carrier_density(t, fixed_taus["amp"], fixed_taus["tau_rise"],
                                fixed_taus["tau_decay"])
            T = lattice_temperature_from_carriers(t, n, fixed_taus["tau_ep"])
            eps = strain_two_channel(n, T, float(Xi), sp(raw_alpha))
            return relative_l2_loss(eps, S_obs)

        out = fit([raw_alpha], loss_fn, steps=steps, lr=lr)
        res.append(out["loss"])
    import torch as _t
    return _t.as_tensor(Xi_grid, dtype=t.dtype), _t.tensor(res, dtype=t.dtype)
