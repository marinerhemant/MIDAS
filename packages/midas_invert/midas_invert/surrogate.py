"""Amortised-inference surrogate: a small MLP that reads parameters off a signal.

Domain-agnostic.  The differentiable forward of whichever experiment is used as
a data generator (pattern -> parameters pairs); this trains a network to invert
in one pass.  Target standardisation is handled internally for stable training.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["ParameterMLP", "train_surrogate"]


class ParameterMLP:
    """Tiny MLP ``signal -> parameters`` with output standardisation."""
    def __init__(self, n_in, n_out, hidden=128, dtype=None):
        import torch
        import torch.nn as nn
        dtype = dtype or torch.float64
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        ).to(dtype)
        self.y_mean = None
        self.y_std = None

    def parameters(self):
        return list(self.net.parameters())

    def __call__(self, X):
        out = self.net(X)
        if self.y_mean is not None:
            out = out * self.y_std + self.y_mean
        return out


def train_surrogate(X, Y, *, epochs=300, lr=1e-3, val_frac=0.2, hidden=128, seed=0):
    """Train a :class:`ParameterMLP` and report held-out per-output MAE.

    Returns ``(model, info)`` with ``info`` holding ``val_mae``, ``history``,
    ``val_pred``, ``val_true``.
    """
    import torch

    torch.manual_seed(int(seed))
    n = X.shape[0]
    perm = torch.randperm(n)
    n_val = int(val_frac * n)
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xva, Yva = X[ti], Y[ti], X[vi], Y[vi]

    model = ParameterMLP(X.shape[1], Y.shape[1], hidden=hidden, dtype=X.dtype)
    model.y_mean = Ytr.mean(0, keepdim=True)
    model.y_std = Ytr.std(0, keepdim=True) + 1e-8

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(Xtr) - Ytr) ** 2 / model.y_std ** 2).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    with torch.no_grad():
        pred_va = model(Xva)
        val_mae = (pred_va - Yva).abs().mean(0)
    return model, {"val_mae": val_mae, "history": history,
                   "val_pred": pred_va.detach(), "val_true": Yva}
