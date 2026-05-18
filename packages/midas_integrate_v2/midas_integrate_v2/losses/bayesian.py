"""Bayesian-prior regularisation: Gaussian prior on any spec parameter.

Pure regulariser — designed to compose with any data loss. Add the
return value to your data loss before ``.backward()``.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import torch
import torch.nn as nn


class GaussianPriorLoss(nn.Module):
    """``loss = Σ_i 0.5 · ((θ_i - μ_i) / σ_i)²``.

    Parameters
    ----------
    priors :
        Mapping from spec attribute name (e.g. ``"Lsd"``) to a
        ``(mean, sigma)`` tuple in physical units. ``mean`` and
        ``sigma`` may be Python floats or torch tensors.

    Forward signature:
        ``loss(spec) -> scalar``
    """

    def __init__(self, priors: Mapping[str, Tuple[float, float]]):
        super().__init__()
        self._names = list(priors.keys())
        for name, (mu, sig) in priors.items():
            self.register_buffer(f"_mu_{name}",
                                  torch.as_tensor(float(mu), dtype=torch.float64))
            self.register_buffer(f"_sig_{name}",
                                  torch.as_tensor(float(sig), dtype=torch.float64))

    def forward(self, spec) -> torch.Tensor:
        terms = []
        for name in self._names:
            theta = getattr(spec, name)
            mu = getattr(self, f"_mu_{name}")
            sig = getattr(self, f"_sig_{name}")
            z = (theta - mu) / sig
            terms.append(0.5 * z * z)
        if not terms:
            return torch.tensor(0.0, dtype=torch.float64)
        return torch.stack(terms).sum()


__all__ = ["GaussianPriorLoss"]
