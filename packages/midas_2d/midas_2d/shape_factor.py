"""Finite-size interference (shape) functions for few-layer / 2D crystals.

A finite crystal of N1 x N2 x N3 unit cells scatters with a *lattice-sum*
amplitude on top of the unit-cell structure factor:

    A(q) = F(q) . S(q),     S(q) = prod_i  sum_{n=0}^{N_i - 1} exp(i n phi_i)

where ``phi_i = q . a_i = 2 pi x_i`` and ``x_i`` is the (real-valued) Miller
index along axis ``i``.  The measured interference factor is the Laue function

    |S(q)|^2 = prod_i  sin^2(N_i pi x_i) / sin^2(pi x_i)

which equals ``N_i^2`` at integer ``x_i`` (the Bragg condition) and produces
``N_i - 1`` minima and ``N_i - 2`` subsidiary maxima ("Laue oscillations" /
thickness fringes) between successive Bragg peaks.  For a colloidal
nanoplatelet the in-plane counts are large (near-delta in-plane) while the
out-of-plane count ``N3`` is the few-monolayer thickness that sets the visible
fringes along the rod.

The semi-infinite limit (``N -> inf`` along an axis) is the crystal-truncation
rod ``|S_z|^2 -> 1 / (4 sin^2(pi x_z))`` connecting the bulk Bragg peaks.

All functions are torch-differentiable (including w.r.t. a real-valued ``N``,
so the layer count is a fittable parameter) and run on CPU / CUDA / MPS.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover
    import torch
    Tensor = torch.Tensor

__all__ = [
    "laue_interference_1d",
    "interference_factor",
    "truncation_rod",
    "nanoplatelet_rod",
]

_PI = math.pi


def laue_interference_1d(x, N):
    """1-D Laue interference function ``sin^2(N pi x) / sin^2(pi x)``.

    Parameters
    ----------
    x : tensor
        Real-valued Miller index (continuous reciprocal coordinate) along one
        axis.  Any shape.
    N : float or tensor
        Number of unit cells along that axis.  May be a differentiable scalar
        tensor so the layer count can be fit.

    Returns
    -------
    tensor
        Same shape as ``x``.  Equals ``N^2`` at integer ``x`` (Bragg peaks).

    Notes
    -----
    The ratio is ``0/0`` at integer ``x``.  We use the standard "double-where"
    guard so both the forward value *and* the gradients are finite:

      * value at integer ``x`` -> ``N^2`` (exact limit),
      * d/dN at integer ``x`` -> ``2N`` (correct: d(N^2)/dN),
      * d/dx at integer ``x`` -> ``0`` (correct: the Laue function is a smooth
        local maximum there).

    This avoids the NaN-gradient trap of dividing by a vanishing denominator.
    """
    import torch

    x = torch.as_tensor(x)
    N = torch.as_tensor(N, dtype=x.dtype, device=x.device)

    s = torch.sin(_PI * x)
    num = torch.sin(_PI * N * x)
    s2 = s * s

    # Near an integer x, sin^2(pi x) ~ 0.  Replace the denominator by 1 in that
    # region BEFORE dividing so the masked branch is finite, then overwrite the
    # result with the analytic limit N^2.
    near_int = s2 < 1e-12
    s2_safe = torch.where(near_int, torch.ones_like(s2), s2)
    ratio = (num * num) / s2_safe
    Nsq = (N * N).expand_as(ratio)
    return torch.where(near_int, Nsq, ratio)


def interference_factor(hkl_cont, N):
    """3-D shape factor ``|S(q)|^2 = prod_i Laue(x_i, N_i)``.

    Parameters
    ----------
    hkl_cont : tensor, shape (..., 3)
        Continuous (real-valued) Miller indices.
    N : sequence/tensor of length 3
        Unit-cell counts ``(N1, N2, N3)`` along the three crystal axes.  Entries
        may be differentiable.  Use a large value (e.g. 1e4) for "effectively
        infinite" in-plane extent.

    Returns
    -------
    tensor, shape (...,)
        The product of the three 1-D Laue functions.
    """
    import torch

    hkl_cont = torch.as_tensor(hkl_cont)
    N = torch.as_tensor(N, dtype=hkl_cont.dtype, device=hkl_cont.device)
    if N.shape[-1] != 3:
        raise ValueError("N must have 3 components (N1, N2, N3)")

    out = torch.ones(hkl_cont.shape[:-1], dtype=hkl_cont.dtype, device=hkl_cont.device)
    for i in range(3):
        out = out * laue_interference_1d(hkl_cont[..., i], N[..., i])
    return out


def truncation_rod(x):
    """Crystal-truncation-rod intensity ``1 / (4 sin^2(pi x))``.

    This is the ``N -> inf`` semi-infinite limit of :func:`laue_interference_1d`
    (an abruptly terminated crystal).  It diverges at integer ``x`` (the Bragg
    peaks) by construction; the denominator is clamped to keep it finite for
    plotting/fitting.
    """
    import torch
    x = torch.as_tensor(x)
    s = torch.sin(_PI * x)
    s2 = torch.clamp(s * s, min=1e-12)
    return 1.0 / (4.0 * s2)


def nanoplatelet_rod(l_cont, n_layers, *, hk=(0.0, 0.0), n_in_plane=1.0e4):
    """Convenience: interference along a single rod through (h, k) for a platelet.

    Parameters
    ----------
    l_cont : tensor
        Continuous ``l`` index sampled along the rod (the scan coordinate).
    n_layers : float or tensor
        Out-of-plane unit-cell count ``N3`` (the few-monolayer thickness).
    hk : tuple of float
        The fixed in-plane indices ``(h, k)`` defining which rod to scan.
    n_in_plane : float
        Large in-plane count (near-delta in-plane); the default makes the
        in-plane Laue factors essentially constant along the rod.

    Returns
    -------
    tensor
        ``|S|^2`` along the rod, same shape as ``l_cont``.
    """
    import torch
    l_cont = torch.as_tensor(l_cont)
    h = torch.full_like(l_cont, float(hk[0]))
    k = torch.full_like(l_cont, float(hk[1]))
    hkl = torch.stack([h, k, l_cont], dim=-1)
    # Build N as a length-3 tensor, preserving autograd if n_layers is a tensor.
    n3 = torch.as_tensor(n_layers, dtype=l_cont.dtype, device=l_cont.device)
    n_ip = torch.as_tensor(n_in_plane, dtype=l_cont.dtype, device=l_cont.device)
    N = torch.stack([n_ip, n_ip, n3])
    return interference_factor(hkl, N)
