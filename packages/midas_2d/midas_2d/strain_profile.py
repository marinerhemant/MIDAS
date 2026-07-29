"""Depth-resolved strain + a unified temperature model.

A laser-excited film does not strain uniformly: the lattice spacing varies with
depth ``z`` (a transient strain gradient / picosecond-acoustic pulse), which
makes the Bragg peak *asymmetric* and distorts the thickness fringes -- exactly
the "d-spacing vs depth" signature seen in time-resolved XRD.

Because the engine scatters from explicit atomic coordinates, a depth profile is
just a per-atom out-of-plane displacement ``u_z(z)`` applied before the coherent
sum -- and, being differentiable, a measured asymmetric peak can be inverted
back to ``u_z(z)`` (a depth-resolved strain reconstruction).

Two faces:

* **Kinematic** -- :func:`apply_depth_displacement`, :func:`depth_resolved_amplitude`,
  profile builders (:func:`linear_strain`, :func:`exponential_strain`,
  :func:`acoustic_pulse`), and :func:`recover_depth_strain` (the inverse).

* **Unified thermal** -- a single depth temperature field ``T(z)`` drives BOTH
  the shift (thermal expansion ``eps = alpha * dT``, integrated to a
  displacement) AND the amplitude (local Debye-Waller ``u_perp^2 = kB T / k``):
  :func:`temperature_to_displacement`, :func:`temperature_to_msd`,
  :func:`thermal_rod`.

Convention: platelet normal = +z; ``z`` in Angstrom; ``Q`` in 1/A.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .debye import atomic_form_factors

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "interp1d",
    "displacement_from_control",
    "apply_depth_displacement",
    "depth_resolved_amplitude",
    "depth_resolved_intensity",
    "linear_strain",
    "exponential_strain",
    "acoustic_pulse",
    "strain_to_displacement",
    "temperature_to_displacement",
    "temperature_to_msd",
    "thermal_rod",
    "recover_depth_strain",
]


# ----------------------------------------------------------------- helpers

def interp1d(x, xp, fp):
    """Differentiable piecewise-linear interpolation of ``fp(xp)`` at ``x``.

    ``xp`` must be sorted ascending.  Values outside ``[xp[0], xp[-1]]`` are
    clamped to the endpoints.
    """
    import torch
    x = torch.as_tensor(x)
    xp = torch.as_tensor(xp, dtype=x.dtype, device=x.device)
    fp = torch.as_tensor(fp, dtype=x.dtype, device=x.device)
    idx = torch.searchsorted(xp, x.clamp(min=float(xp[0]), max=float(xp[-1])), right=True)
    idx = idx.clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    t = (x - x0) / (x1 - x0).clamp(min=1e-12)
    return f0 + t * (f1 - f0)


def displacement_from_control(z, z_ctrl, u_ctrl):
    """Per-atom out-of-plane displacement by interpolating control values."""
    return interp1d(z, z_ctrl, u_ctrl)


def apply_depth_displacement(coords, u_z):
    """Return coords with the z-coordinate shifted by per-atom ``u_z``."""
    import torch
    coords = torch.as_tensor(coords)
    u_z = torch.as_tensor(u_z, dtype=coords.dtype, device=coords.device)
    shifted = coords.clone()
    shifted = torch.stack([coords[:, 0], coords[:, 1], coords[:, 2] + u_z], dim=1)
    return shifted


# ------------------------------------------------------ depth-resolved forward

def depth_resolved_amplitude(coords, elements, q_vec, *, u_z=None, msd_perp=None):
    """Coherent amplitude with a per-atom depth displacement and (optional)
    per-atom out-of-plane Debye-Waller.

    ``u_z`` : (M,) displacement (A) added to each atom's z.
    ``msd_perp`` : (M,) out-of-plane mean-square displacement (A^2) -> per-atom
                   factor ``exp(-1/2 q_z^2 msd_perp)``.
    """
    import torch
    coords = torch.as_tensor(coords)
    q_vec = torch.as_tensor(q_vec, dtype=coords.dtype, device=coords.device)
    if u_z is not None:
        coords = apply_depth_displacement(coords, u_z)

    qmag = torch.linalg.vector_norm(q_vec, dim=-1)
    f = atomic_form_factors(elements, qmag)                  # (..., M)
    if msd_perp is not None:
        msd_perp = torch.as_tensor(msd_perp, dtype=coords.dtype, device=coords.device)
        qz = q_vec[..., 2:3]                                 # (..., 1)
        f = f * torch.exp(-0.5 * qz ** 2 * msd_perp)         # broadcast (..., M)
    phase = q_vec @ coords.T
    re = (f * torch.cos(phase)).sum(dim=-1)
    im = (f * torch.sin(phase)).sum(dim=-1)
    return torch.complex(re, im)


def depth_resolved_intensity(coords, elements, q_vec, *, u_z=None, msd_perp=None):
    A = depth_resolved_amplitude(coords, elements, q_vec, u_z=u_z, msd_perp=msd_perp)
    return A.real * A.real + A.imag * A.imag


# ----------------------------------------------------------- profile builders

def strain_to_displacement(z, eps_of_z):
    """Integrate a local strain ``eps(z)`` to a displacement ``u(z)=∫_0^z eps``.

    ``z`` sorted ascending; ``eps_of_z`` sampled at the same points.
    """
    import torch
    z = torch.as_tensor(z)
    eps = torch.as_tensor(eps_of_z, dtype=z.dtype, device=z.device)
    dz = torch.diff(z, prepend=z[:1])
    return torch.cumsum(eps * dz, dim=0)


def linear_strain(z, eps_surface, eps_substrate):
    """Linear strain gradient from substrate (z=0) to surface (z=max)."""
    import torch
    z = torch.as_tensor(z)
    zmin, zmax = z.min(), z.max()
    frac = (z - zmin) / (zmax - zmin).clamp(min=1e-12)
    return eps_substrate + frac * (eps_surface - eps_substrate)


def exponential_strain(z, eps0, depth):
    """Surface-localised strain ``eps0 exp(-(zmax - z)/depth)`` (hot surface)."""
    import torch
    z = torch.as_tensor(z)
    return eps0 * torch.exp(-(z.max() - z) / depth)


def acoustic_pulse(z, *, amp, center, width, wavevector):
    """A Thomsen-style propagating strain pulse: a Gaussian-enveloped
    oscillation centred at depth ``center`` (use to model picosecond acoustics
    at a given delay)."""
    import torch
    z = torch.as_tensor(z)
    env = torch.exp(-0.5 * ((z - center) / width) ** 2)
    return amp * env * torch.cos(wavevector * (z - center))


# ------------------------------------------------------------- unified thermal

def temperature_to_displacement(z, dT, alpha):
    """Displacement from a depth temperature rise: ``eps = alpha * dT`` then
    integrate.  ``dT`` sampled at ``z``; ``alpha`` linear-expansion coeff."""
    import torch
    eps = float(alpha) * torch.as_tensor(dT)
    return strain_to_displacement(z, eps)


def temperature_to_msd(dT, *, k, kB=1.0, T0=1.0):
    """Local out-of-plane MSD from temperature: ``u^2 = kB (T0 + dT) / k``."""
    import torch
    dT = torch.as_tensor(dT)
    return kB * (T0 + dT) / float(k)


def thermal_rod(coords, elements, q_vec, z_atom, dT_atom, *, alpha, k, kB=1.0, T0=1.0):
    """One temperature field -> BOTH peak shift (expansion) and amplitude (DWF).

    ``z_atom`` / ``dT_atom`` are per-atom depth and temperature rise.
    """
    import torch
    z_atom = torch.as_tensor(z_atom)
    # expansion displacement (integrate eps = alpha dT over depth, per atom)
    order = torch.argsort(z_atom)
    inv = torch.argsort(order)
    u_sorted = temperature_to_displacement(z_atom[order], dT_atom[order], alpha)
    u_z = u_sorted[inv]
    msd = temperature_to_msd(dT_atom, k=k, kB=kB, T0=T0)
    return depth_resolved_intensity(coords, elements, q_vec, u_z=u_z, msd_perp=msd)


# --------------------------------------------------------------- inverse

def recover_depth_strain(observed, coords, elements, rod_qvec, z_ctrl, *,
                         n_ctrl=6, steps=800, lr=0.02, msd_perp=None,
                         smooth_weight=0.0):
    """Recover the depth displacement profile ``u(z)`` from a measured rod /
    Bragg-peak intensity curve.

    ``rod_qvec`` : (L, 3) Q-vectors along the scanned rod.
    ``z_ctrl``  : (n_ctrl,) control depths (sorted) at which u is parameterised.
    ``smooth_weight`` : penalise the second difference of the control points
        (a smoothness prior on the strain field) -- stabilises the recovered
        *derivative* (strain), which is otherwise noisy near the boundaries.

    Returns dict with ``z_ctrl``, ``u_ctrl`` (recovered), ``u_atom``.
    """
    import torch
    from .inverse import cosine_loss, fit

    coords = torch.as_tensor(coords)
    z_atom = coords[:, 2]
    z_ctrl = torch.as_tensor(z_ctrl, dtype=coords.dtype, device=coords.device)
    u_ctrl = torch.zeros(z_ctrl.numel(), dtype=coords.dtype, device=coords.device,
                         requires_grad=True)

    def loss_fn():
        u_atom = displacement_from_control(z_atom, z_ctrl, u_ctrl)
        pred = depth_resolved_intensity(coords, elements, rod_qvec,
                                        u_z=u_atom, msd_perp=msd_perp)
        loss = cosine_loss(pred, observed)
        if smooth_weight > 0:
            d2 = u_ctrl[2:] - 2 * u_ctrl[1:-1] + u_ctrl[:-2]
            loss = loss + smooth_weight * (d2 ** 2).sum()
        return loss

    fit([u_ctrl], loss_fn, steps=steps, lr=lr)
    u_atom = displacement_from_control(z_atom, z_ctrl, u_ctrl).detach()
    return {"z_ctrl": z_ctrl.detach(), "u_ctrl": u_ctrl.detach(), "u_atom": u_atom}
