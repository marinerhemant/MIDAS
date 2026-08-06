"""Transient, anisotropic disorder for few-layer materials.

Optical excitation disorders a CdSe nanoplatelet
*anisotropically* -- the out-of-plane mean-square displacement grows differently
from the in-plane one, and it evolves with pump-probe delay.  This module gives
that signature two equivalent faces:

1. **Analytic Debye-Waller** (:class:`AnisotropicMSD`, :class:`TransientMSD`):
   a differentiable temperature factor ``T(Q) = exp(-1/2 Q . U . Q)`` with a
   diagonal Cartesian MSD tensor ``U = diag(u_par, u_par, u_perp)`` (Angstrom^2),
   optionally a function of time.  ``u_par``/``u_perp`` are fittable.

2. **MD-derived** (:func:`msd_tensor_from_frames`): read the *same* tensor
   straight off an MD trajectory as the covariance of atomic displacements.
   This closes the loop with :mod:`midas_2d.debye` -- the ensemble spread of
   coordinates *is* the disorder, and this function recovers the effective ``U``
   that an analytic DWF would need to reproduce it.

Convention: ``Q`` in 1/A, ``U`` in A^2, platelet normal = +z (the c axis).
``T`` multiplies the structure-factor *amplitude*; intensity carries ``T^2``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "dwf_amplitude",
    "AnisotropicMSD",
    "TransientMSD",
    "msd_tensor_from_frames",
]


def dwf_amplitude(q_vec, u_par, u_perp):
    """Anisotropic Debye-Waller *amplitude* factor ``exp(-1/2 Q.U.Q)``.

    Parameters
    ----------
    q_vec : tensor (..., 3)
        Scattering vectors (1/A).
    u_par, u_perp : float or tensor
        In-plane (x, y) and out-of-plane (z) mean-square displacements (A^2).

    Returns
    -------
    tensor (...)
        The amplitude multiplier; square it for the intensity multiplier.
    """
    import torch
    q_vec = torch.as_tensor(q_vec)
    u_par = torch.as_tensor(u_par, dtype=q_vec.dtype, device=q_vec.device)
    u_perp = torch.as_tensor(u_perp, dtype=q_vec.dtype, device=q_vec.device)
    qx, qy, qz = q_vec[..., 0], q_vec[..., 1], q_vec[..., 2]
    quad = u_par * (qx * qx + qy * qy) + u_perp * (qz * qz)
    return torch.exp(-0.5 * quad)


class AnisotropicMSD:
    """Fittable static anisotropic MSD tensor (positivity via softplus).

    ``u_par``/``u_perp`` are stored as unconstrained ``raw_*`` parameters; the
    physical values are ``softplus(raw)`` so optimisation stays unconstrained.
    """
    def __init__(self, u_par=1e-3, u_perp=1e-3, *, dtype=None, device=None,
                 requires_grad=True):
        import torch
        dtype = dtype or torch.float64
        self.raw_par = _inv_softplus(torch.as_tensor(u_par, dtype=dtype, device=device))
        self.raw_perp = _inv_softplus(torch.as_tensor(u_perp, dtype=dtype, device=device))
        if requires_grad:
            self.raw_par = self.raw_par.clone().detach().requires_grad_(True)
            self.raw_perp = self.raw_perp.clone().detach().requires_grad_(True)

    @property
    def u_par(self):
        import torch
        return torch.nn.functional.softplus(self.raw_par)

    @property
    def u_perp(self):
        import torch
        return torch.nn.functional.softplus(self.raw_perp)

    def parameters(self):
        return [self.raw_par, self.raw_perp]

    def amplitude(self, q_vec):
        return dwf_amplitude(q_vec, self.u_par, self.u_perp)


class TransientMSD:
    """Per-delay anisotropic MSD: ``u_par(t)``, ``u_perp(t)`` over T pump-probe
    delays.  Each delay has its own fittable (softplus-positive) pair, so a full
    time series of patterns is inverted jointly with shared structure.
    """
    def __init__(self, n_delays, u_par0=1e-3, u_perp0=1e-3, *, dtype=None,
                 device=None, requires_grad=True):
        import torch
        dtype = dtype or torch.float64
        rp = _inv_softplus(torch.full((n_delays,), float(u_par0), dtype=dtype, device=device))
        rq = _inv_softplus(torch.full((n_delays,), float(u_perp0), dtype=dtype, device=device))
        if requires_grad:
            rp = rp.clone().detach().requires_grad_(True)
            rq = rq.clone().detach().requires_grad_(True)
        self.raw_par = rp
        self.raw_perp = rq
        self.n_delays = int(n_delays)

    @property
    def u_par(self):
        import torch
        return torch.nn.functional.softplus(self.raw_par)

    @property
    def u_perp(self):
        import torch
        return torch.nn.functional.softplus(self.raw_perp)

    def parameters(self):
        return [self.raw_par, self.raw_perp]

    def amplitude(self, q_vec, delay_index):
        return dwf_amplitude(q_vec, self.u_par[delay_index], self.u_perp[delay_index])

    def anisotropy(self):
        """``u_perp / u_par`` per delay -- the out-of-plane/in-plane ratio that
        is the experimental disordering signature."""
        return self.u_perp / self.u_par


def msd_tensor_from_frames(frames, *, reference=None):
    """Effective MSD tensor ``U`` (A^2) from an MD trajectory.

    ``U = < (r - <r>) (r - <r>)^T >`` averaged over atoms and frames.  The
    diagonal entries are ``(u_xx, u_yy, u_zz)``; for a platelet,
    ``u_zz`` is ``u_perp`` and ``(u_xx + u_yy)/2`` is ``u_par``.

    Parameters
    ----------
    frames : tensor (F, M, 3)
    reference : tensor (M, 3), optional
        Mean positions; defaults to the per-atom time average.

    Returns
    -------
    U : tensor (3, 3)
    """
    import torch
    frames = torch.as_tensor(frames)
    if frames.dim() == 2:
        frames = frames.unsqueeze(0)
    ref = reference if reference is not None else frames.mean(dim=0)  # (M,3)
    disp = (frames - ref[None]).reshape(-1, 3)                        # (F*M, 3)
    U = (disp.T @ disp) / disp.shape[0]                               # (3,3)
    return U


def _inv_softplus(y):
    """Inverse of softplus so that softplus(raw) == y for the init value."""
    import torch
    y = torch.as_tensor(y)
    # raw = log(exp(y) - 1); stable for small/large y.
    return torch.where(y > 20, y, torch.log(torch.expm1(y.clamp(min=1e-12))))
