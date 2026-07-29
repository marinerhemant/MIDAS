"""Rocking curves and reciprocal-space maps for finite crystals.

A finite crystal sweeps a *width* through the Bragg condition set by its size:
the rocking-curve FWHM along the rod is ``~0.886 / N`` reciprocal-lattice units
(the sinc^2 / Fejer-kernel central-peak width), so the curve width is a direct
thickness gauge.  This module builds those curves (from either the analytic
shape-factor forward or the MD-coupled coordinate forward) and assembles 2-D
reciprocal-space maps.

All outputs are torch tensors; the forward callables are differentiable.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "analytic_rod_model",
    "md_rod_model",
    "rocking_curve",
    "fwhm",
    "thickness_from_fwhm",
    "reciprocal_space_map",
    "thickness_loss_scan",
]

# Central-peak FWHM of |sin(N pi x)/sin(pi x)|^2, in units of x (recip-lattice).
_SINC2_FWHM_CONST = 0.8859


def analytic_rod_model(crystal_t, hk, N, *, wavelength_A=1.0, apply_lp=False
                       ) -> "Callable":
    """Return ``f(l) -> I(l)`` using the analytic |F|^2 . |S|^2 forward."""
    import torch
    from .forward import rod_intensity

    def model(l):
        l = torch.as_tensor(l)
        hkl = torch.stack([
            torch.full_like(l, float(hk[0])),
            torch.full_like(l, float(hk[1])),
            l,
        ], dim=-1)
        return rod_intensity(crystal_t, hkl, N, wavelength_A=wavelength_A,
                             apply_lp=apply_lp)
    return model


def md_rod_model(coords, elements, *, a, hk, dwf=None) -> "Callable":
    """Return ``f(l) -> I(l)`` from explicit atomic coordinates (cubic cell of
    constant ``a``), optionally multiplied by an anisotropic DWF amplitude.

    ``dwf`` is an object with an ``.amplitude(q_vec)`` method (e.g.
    :class:`midas_2d.disorder.AnisotropicMSD`) or ``None``.
    """
    import torch
    from .debye import coherent_amplitude

    def model(l):
        l = torch.as_tensor(l)
        hkl = torch.stack([
            torch.full_like(l, float(hk[0])),
            torch.full_like(l, float(hk[1])),
            l,
        ], dim=-1)
        q_vec = (2.0 * math.pi / a) * hkl
        A = coherent_amplitude(coords, elements, q_vec)
        if dwf is not None:
            A = A * dwf.amplitude(q_vec).to(A.dtype)
        return A.real * A.real + A.imag * A.imag
    return model


def rocking_curve(model, l0, *, half_width=0.5, n=401):
    """Sample ``model(l)`` across a Bragg point ``l0``.

    Returns ``(delta_l, intensity)`` where ``delta_l`` is the rod-coordinate
    offset from ``l0`` (the rocking variable, proportional to the sample tilt).
    """
    import torch
    dl = torch.linspace(-half_width, half_width, n, dtype=torch.float64)
    I = model(l0 + dl)
    return dl, I


def fwhm(x, y):
    """Full width at half maximum of a single-peaked curve ``y(x)``.

    Linear interpolation of the half-max crossings on each side of the peak.
    """
    import torch
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    ipk = int(torch.argmax(y))
    half = y[ipk] / 2.0

    def _cross(idx_range):
        prev = ipk
        for i in idx_range:
            if y[i] < half:
                # interpolate between i and prev
                x0, x1 = x[prev], x[i]
                y0, y1 = y[prev], y[i]
                t = (half - y0) / (y1 - y0)
                return x0 + t * (x1 - x0)
            prev = i
        return x[idx_range[-1]]

    left = _cross(range(ipk, -1, -1))
    right = _cross(range(ipk, len(x)))
    return float((right - left).abs())


def thickness_from_fwhm(fwhm_l):
    """Estimate the out-of-plane cell count ``N`` from a rocking FWHM (in
    rod-coordinate units): ``N ~ 0.886 / FWHM``."""
    return _SINC2_FWHM_CONST / float(fwhm_l)


def thickness_loss_scan(crystal_t, rods, obs_curves, dl, n3_values, *,
                        n_inplane=1e4):
    """Cosine-loss vs out-of-plane cell count ``N3``, summed over one or more
    rods.  A single rod is multimodal in ``N3`` (a basin per integer); summing
    several rods sharpens to the true thickness.

    Parameters
    ----------
    crystal_t : CrystalTensor
    rods : list[(h, k)]
        In-plane indices of each rod scanned.
    obs_curves : list[tensor]
        Measured rocking curve (vs ``dl``) for each rod, parallel to ``rods``.
    dl : tensor
        Rod-coordinate offsets (the same grid used for ``obs_curves``).
    n3_values : 1-D tensor
        Candidate thicknesses to evaluate.

    Returns
    -------
    dict: ``total`` (loss vs n3, summed) and ``per_rod`` (list of loss curves).
    """
    import torch
    from .inverse import cosine_loss

    per_rod = []
    for (hk, obs) in zip(rods, obs_curves):
        losses = []
        for n3 in n3_values:
            N = torch.tensor([n_inplane, n_inplane, float(n3)], dtype=torch.float64)
            pred = analytic_rod_model(crystal_t, hk, N)(1.0 + dl)
            losses.append(float(cosine_loss(pred, obs)))
        per_rod.append(torch.tensor(losses, dtype=torch.float64))
    total = torch.stack(per_rod).sum(dim=0)
    return {"total": total, "per_rod": per_rod}


def reciprocal_space_map(coords, elements, *, a, h0, qx_range, qz_range,
                         n_qx=120, n_qz=200):
    """2-D coherent reciprocal-space map ``|A(Q)|^2`` in the (qx, qz) plane.

    Scans the in-plane index around ``h0`` (``qx``) and the out-of-plane index
    (``qz``) at fixed ``k = h0``.  Indices are in reciprocal-lattice units;
    converted to physical Q via ``2 pi / a`` for the cubic cell.

    Returns ``(H, L, I)`` grids of shape (n_qz, n_qx).
    """
    import torch
    from .debye import coherent_intensity

    h = torch.linspace(h0 + qx_range[0], h0 + qx_range[1], n_qx, dtype=torch.float64)
    l = torch.linspace(qz_range[0], qz_range[1], n_qz, dtype=torch.float64)
    H, L = torch.meshgrid(h, l, indexing="xy")                # (n_qz, n_qx)
    K = torch.full_like(H, float(h0))
    hkl = torch.stack([H, K, L], dim=-1)                      # (n_qz, n_qx, 3)
    q_vec = (2.0 * math.pi / a) * hkl
    I = coherent_intensity(coords, elements, q_vec)
    return H, L, I
