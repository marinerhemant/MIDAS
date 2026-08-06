"""Differentiable molecular dynamics -> learn the potential from diffraction.

The grand-unification piece: a velocity-Verlet integrator whose every step is
autograd-differentiable, so gradients flow from a measured diffraction *movie*,
through the trajectory, back to the **interatomic potential parameters**.

Demonstration physics: launch a coherent out-of-plane displacement (a pump),
integrate the lattice under a (an)harmonic potential, and watch the Bragg
intensity oscillate at twice the phonon frequency ``omega = sqrt(k/m)``.  Fitting
the spring constant ``k`` to that oscillation -- by differentiating through the
MD -- recovers the potential stiffness from the diffraction time series.

This is the seam where MD, ultrafast measurement, and ML inversion meet:
diffraction becomes a differentiable loss on a simulation.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "harmonic_force",
    "velocity_verlet",
    "bragg_from_trajectory",
    "coherent_mode_kick",
    "recover_potential_from_movie",
]


def coherent_mode_kick(coords_eq, amp):
    """Initial displacement for a coherent out-of-plane STANDING-WAVE mode
    ``u_z(z) = amp * sin(pi (z - zmin)/(zmax - zmin))``.

    This is non-uniform on purpose: a *uniform* kick moves every atom in phase
    (a rigid translation), to which ``|A(Q)|^2`` is invariant -- so it produces
    no Bragg-intensity oscillation.  A standing wave modulates the structure
    factor and gives the measurable coherent-phonon signal.
    """
    import torch
    coords_eq = torch.as_tensor(coords_eq)
    z = coords_eq[:, 2]
    zmin, zmax = z.min(), z.max()
    mode = torch.sin(math.pi * (z - zmin) / (zmax - zmin).clamp(min=1e-9))
    r0 = torch.stack([coords_eq[:, 0], coords_eq[:, 1],
                      coords_eq[:, 2] + float(amp) * mode], dim=1)
    return r0


def harmonic_force(r, r_eq, k_vec, *, anharmonic=0.0):
    """Restoring force ``F = -k (r - r_eq) - anharmonic (r - r_eq)^3`` (per axis).

    ``k_vec`` is (3,) -- anisotropic in/out of plane.  The optional cubic-ish
    quartic term adds anharmonicity (so the model can stiffen/soften with
    amplitude).
    """
    import torch
    d = r - r_eq
    F = -k_vec * d
    if anharmonic:
        F = F - float(anharmonic) * d ** 3
    return F


def velocity_verlet(r0, v0, force_fn, *, dt, n_steps, mass=1.0):
    """Differentiable velocity-Verlet NVE integration.

    Parameters
    ----------
    r0, v0 : tensor (M, 3)
        Initial positions and velocities.
    force_fn : callable(r) -> (M, 3)
        Differentiable force (gradients to its captured parameters flow through).
    dt, n_steps, mass : floats.

    Returns
    -------
    traj : tensor (n_steps+1, M, 3)
    """
    import torch
    r = torch.as_tensor(r0)
    v = torch.as_tensor(v0, dtype=r.dtype, device=r.device)
    m = float(mass)
    a = force_fn(r) / m
    frames = [r]
    for _ in range(n_steps):
        r = r + v * dt + 0.5 * a * dt * dt
        a_new = force_fn(r) / m
        v = v + 0.5 * (a + a_new) * dt
        a = a_new
        frames.append(r)
    return torch.stack(frames)


def bragg_from_trajectory(traj, elements, q_vec):
    """Coherent Bragg intensity at ``q_vec`` for each MD frame.  Returns (T,)."""
    import torch
    from .debye import coherent_intensity
    q_vec = torch.as_tensor(q_vec, dtype=traj.dtype, device=traj.device).reshape(1, 3)
    return torch.stack([coherent_intensity(fr, elements, q_vec)[0] for fr in traj])


def recover_potential_from_movie(obs_intensity, coords_eq, elements, q_vec, *,
                                 amp0, dt, n_steps, mass=1.0, init_k=5.0,
                                 steps=400, lr=0.05):
    """Recover the out-of-plane spring constant ``k_perp`` by differentiating an
    MD trajectory to match a measured Bragg-intensity movie.

    The pump is modelled as an initial uniform out-of-plane displacement of
    amplitude ``amp0`` (a coherent mode); the lattice then rings at
    ``omega = sqrt(k_perp/m)`` and the Bragg intensity at twice that.

    Returns dict with ``k_perp`` (float), ``omega`` and the loss.
    """
    import torch
    from .inverse import cosine_loss, fit

    coords_eq = torch.as_tensor(coords_eq)
    obs = torch.as_tensor(obs_intensity)

    # Coarse k from the observed oscillation frequency (the cosine-loss landscape
    # is multimodal in frequency, so a coarse estimate is needed to land in the
    # right basin -- then gradient refinement polishes).  NOTE: the Bragg
    # intensity oscillates at 2*omega (|A|^2 is an even function of the
    # displacement, so its period is halved), hence the factor of 1/2.
    k0 = float(init_k)
    osc = obs - obs.mean()
    sgn = torch.sign(osc)
    cross = torch.where(sgn[1:] != sgn[:-1])[0]
    if cross.numel() >= 2:
        half = float(cross.diff().float().mean())
        period_I = 2.0 * half * float(dt)         # period of the intensity (= pi/omega)
        if period_I > 0:
            omega_I = 2.0 * math.pi / period_I    # = 2*omega_phonon
            omega_phonon = omega_I / 2.0
            k0 = float(mass) * omega_phonon * omega_phonon

    raw = torch.tensor(math.log(math.expm1(max(k0, 1e-3))), dtype=coords_eq.dtype,
                       requires_grad=True)
    sp = torch.nn.functional.softplus

    # initial coherent out-of-plane STANDING-WAVE kick (non-uniform -> visible
    # to |A(Q)|^2), zero velocity.
    r0 = coherent_mode_kick(coords_eq, amp0)
    v0 = torch.zeros_like(r0)

    def loss_fn():
        k_vec = torch.stack([torch.tensor(0.0, dtype=coords_eq.dtype),
                             torch.tensor(0.0, dtype=coords_eq.dtype), sp(raw)])
        force_fn = lambda r: harmonic_force(r, coords_eq, k_vec)
        traj = velocity_verlet(r0, v0, force_fn, dt=dt, n_steps=n_steps, mass=mass)
        pred = bragg_from_trajectory(traj, elements, q_vec)
        return cosine_loss(pred, obs)

    out = fit([raw], loss_fn, steps=steps, lr=lr)
    k = float(sp(raw).detach())
    return {"k_perp": k, "omega": math.sqrt(k / mass), "loss": out["loss"]}
