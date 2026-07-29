"""Amortised inference for midas_2d.

The generic surrogate (``ParameterMLP``, ``train_surrogate``) now lives in the
shared ``midas_invert`` leaf; only the diffraction-specific dataset generator
stays here.
"""
from __future__ import annotations

from midas_invert.surrogate import ParameterMLP, train_surrogate

__all__ = ["make_dataset", "ParameterMLP", "train_surrogate"]


def make_dataset(crystal_t, *, n=600, n_points=64, n3_range=(2.0, 8.0),
                 uperp_range=(0.0, 0.10), half_width=0.5, seed=0, dtype=None):
    """Generate (rocking-curve, [N3, u_perp]) pairs via the forward model.

    Each curve is the (1 1 l) rocking curve, peak-normalised so only shape
    carries information.  Returns ``(X, Y)`` with ``X`` (n, n_points), ``Y`` (n, 2).
    """
    import torch
    from .disorder import dwf_amplitude
    from .rocking import analytic_rod_model, rocking_curve

    dtype = dtype or torch.float64
    g = torch.Generator().manual_seed(int(seed))
    n3 = torch.rand(n, generator=g, dtype=dtype) * (n3_range[1] - n3_range[0]) + n3_range[0]
    up = torch.rand(n, generator=g, dtype=dtype) * (uperp_range[1] - uperp_range[0]) + uperp_range[0]

    X = torch.empty(n, n_points, dtype=dtype)
    for i in range(n):
        N = torch.stack([torch.tensor(1e4, dtype=dtype), torch.tensor(1e4, dtype=dtype), n3[i]])
        model = analytic_rod_model(crystal_t, (1.0, 1.0), N)
        dl, I = rocking_curve(model, 1.0, half_width=half_width, n=n_points)
        q = (2 * torch.pi / 6.077) * torch.stack(
            [torch.ones_like(dl), torch.ones_like(dl), 1.0 + dl], dim=-1)
        I = I * dwf_amplitude(q, 0.0, up[i]) ** 2
        X[i] = I / I.max()
    Y = torch.stack([n3, up], dim=1)
    return X, Y
