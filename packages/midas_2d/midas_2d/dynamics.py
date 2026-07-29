"""Diffraction-as-a-loss-on-dynamics: invert to the (an)harmonic potential.

The novel closure.  We already turn coordinates into diffraction and recover
mean-square displacements; this module goes one layer deeper -- to the
*restoring force* that produced those displacements:

* :func:`stiffness_from_msd` -- classical equipartition ``<u^2> = kBT / k`` gives
  the per-direction effective spring constant from an MSD.  Anisotropic in/out
  of plane.  A transient *drop* in ``k_perp`` is lattice softening.

* :func:`thermal_ensemble` -- generate a thermal cloud of frames whose anisotropic
  variance is set by the stiffnesses, via the reparameterisation trick
  (``u = sqrt(kBT/k) * eps`` with fixed ``eps``).  This is **differentiable in
  the stiffnesses**, so gradients flow  diffraction -> coordinates -> potential.

Putting them together (see :func:`recover_stiffness`) lets you fit measured
diffraction directly for ``k_par(t)``, ``k_perp(t)`` -- transient softening read
straight off the patterns, with the whole chain differentiable.

Units are reduced (``kBT`` in energy units, ``u^2`` in A^2 -> ``k`` in
energy/A^2); pass a physical ``kBT`` for absolute spring constants.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "stiffness_from_msd",
    "msd_from_stiffness",
    "thermal_ensemble",
    "recover_stiffness",
]


def stiffness_from_msd(u, kBT=1.0):
    """Effective spring constant ``k = kBT / <u^2>`` (equipartition)."""
    import torch
    u = torch.as_tensor(u)
    return float(kBT) / torch.clamp(u, min=1e-12)


def msd_from_stiffness(k, kBT=1.0):
    """Mean-square displacement ``<u^2> = kBT / k``."""
    import torch
    k = torch.as_tensor(k)
    return float(kBT) / torch.clamp(k, min=1e-12)


def thermal_ensemble(coords0, k_par, k_perp, *, n_frames=48, kBT=1.0,
                     generator=None, eps=None):
    """Differentiable thermal cloud of frames from anisotropic stiffness.

    ``frames = coords0 + sqrt(kBT/k) * eps`` with ``eps`` a *fixed* standard
    normal (reparameterisation trick), so the result is differentiable in
    ``k_par`` / ``k_perp``.

    Parameters
    ----------
    coords0 : tensor (M, 3)
        Equilibrium positions (A).
    k_par, k_perp : scalar tensors
        In-plane (x, y) and out-of-plane (z) spring constants.
    n_frames : int
    eps : tensor (n_frames, M, 3), optional
        Pre-drawn noise (pass to keep it fixed across an optimisation).

    Returns
    -------
    frames : tensor (n_frames, M, 3)
    """
    import torch
    coords0 = torch.as_tensor(coords0)
    M = coords0.shape[0]
    dt, dev = coords0.dtype, coords0.device
    k_par = torch.as_tensor(k_par, dtype=dt, device=dev)
    k_perp = torch.as_tensor(k_perp, dtype=dt, device=dev)
    if eps is None:
        if generator is None:
            eps = torch.randn(n_frames, M, 3, dtype=dt, device=dev)
        else:
            eps = torch.randn(n_frames, M, 3, dtype=dt, device=dev, generator=generator)
    sigma = torch.stack([
        torch.sqrt(kBT / torch.clamp(k_par, min=1e-12)),
        torch.sqrt(kBT / torch.clamp(k_par, min=1e-12)),
        torch.sqrt(kBT / torch.clamp(k_perp, min=1e-12)),
    ])                                                       # (3,)
    return coords0[None] + sigma * eps                       # broadcast (F,M,3)


def recover_stiffness(observed_ratio, coords0, elements, q_vecs, *, kBT=1.0,
                      n_frames=48, steps=600, lr=0.05, seed=0, init_k=1.0):
    """Fit ``(k_par, k_perp)`` so the thermal-ensemble diffraction matches a
    measured intensity *ratio* (relative to the cold/reference lattice).

    Returns a dict with ``k_par``, ``k_perp``, ``u_par``, ``u_perp`` (floats).

    The ratio ``I(k)/I(reference)`` cancels the absolute scale; we model the
    reference as the zero-displacement pattern.  ``init_k`` seeds both spring
    constants (use a value near the expected baseline for fast convergence).

    Note: very stiff directions (tiny MSD) are weakly identifiable -- the
    lattice barely moves, so diffraction cannot constrain them.  The robustly
    recovered quantity is the MSD ``u = kBT/k``.
    """
    import torch
    from .debye import coherent_intensity, ensemble_intensity
    from .inverse import fit

    coords0 = torch.as_tensor(coords0)
    gen = torch.Generator(device=coords0.device).manual_seed(int(seed))
    eps = torch.randn(n_frames, *coords0.shape, dtype=coords0.dtype,
                      device=coords0.device, generator=gen)

    I_ref = coherent_intensity(coords0, elements, q_vecs)    # cold reference

    raw0 = float(torch.log(torch.expm1(torch.tensor(max(init_k, 1e-3)))))
    raw = torch.full((2,), raw0, dtype=coords0.dtype, device=coords0.device,
                     requires_grad=True)                     # log-stiffness

    def loss_fn():
        k = torch.nn.functional.softplus(raw) + 1e-3
        frames = thermal_ensemble(coords0, k[0], k[1], kBT=kBT,
                                  n_frames=n_frames, eps=eps)
        I = ensemble_intensity(frames, elements, q_vecs, coherent=True)
        pred_ratio = I / I_ref
        return ((pred_ratio - observed_ratio) ** 2).sum() / observed_ratio.pow(2).sum()

    fit([raw], loss_fn, steps=steps, lr=lr)
    k = (torch.nn.functional.softplus(raw) + 1e-3).detach()
    return {
        "k_par": float(k[0]), "k_perp": float(k[1]),
        "u_par": float(kBT / k[0]), "u_perp": float(kBT / k[1]),
    }
