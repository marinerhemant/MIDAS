"""Gradient-fitting loop and scale-aware losses (domain-agnostic).

Shared by midas_2d, and intended for HEDM / Laue inversions (pf-/grain-ODF,
spectrum recovery, ...).  Nothing here knows about diffraction; the loss
closure decides what is fit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = ["fit", "relative_l2_loss", "cosine_loss"]


def relative_l2_loss(pred, obs, *, eps=1e-12):
    """Scale-robust loss ``||pred - obs||^2 / (||obs||^2 + eps)``."""
    import torch
    pred = torch.as_tensor(pred)
    obs = torch.as_tensor(obs)
    return ((pred - obs) ** 2).sum() / (obs.pow(2).sum() + eps)


def cosine_loss(pred, obs, *, eps=1e-12):
    """Scale-*invariant* shape loss ``1 - <pred,obs>^2 / (|pred|^2 |obs|^2)``.

    Smooth and free of the argmax kink that peak-normalisation introduces -- the
    right choice when only the *shape* of a curve matters and absolute scale is
    a nuisance (rocking curves, fringe profiles, spectra).
    """
    import torch
    pred = torch.as_tensor(pred).reshape(-1)
    obs = torch.as_tensor(obs).reshape(-1)
    num = (pred @ obs) ** 2
    den = (pred @ pred) * (obs @ obs) + eps
    return 1.0 - num / den


def fit(params, loss_fn, *, steps=400, lr=0.05, optimizer="adam",
        callback=None, log_every=0):
    """Gradient-fit a list of leaf tensors to minimise ``loss_fn()``.

    Parameters
    ----------
    params : sequence of tensors with requires_grad=True
    loss_fn : callable () -> scalar tensor
    steps, lr : optimisation controls
    optimizer : {"adam", "lbfgs"}
    callback : callable(step, loss_float), optional
    log_every : int; if > 0, record loss every ``log_every`` steps.

    Returns
    -------
    dict with ``loss`` (final float) and ``history`` (list of floats).
    """
    import torch

    params = [p for p in params if p is not None]
    history: list[float] = []

    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=steps,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            if log_every:
                history.append(float(loss.detach()))
            return loss

        opt.step(closure)
        with torch.no_grad():
            final = loss_fn()
        return {"loss": float(final.detach()), "history": history}

    opt = torch.optim.Adam(params, lr=lr)
    last = float("nan")
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if log_every and (step % log_every == 0 or step == steps - 1):
            history.append(last)
        if callback is not None:
            callback(step, last)
    return {"loss": last, "history": history}
