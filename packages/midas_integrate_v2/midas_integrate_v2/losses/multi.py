"""Multi-image loss aggregators.

Two patterns covered:

- :class:`MultiImageLoss` — list of ``(image, spec, [extra args…])``
  tuples; useful for multi-distance calibration where each frame has
  its own ``Lsd`` (or other unshared parameter) but the loss function is
  the same. Each item is integrated independently via
  :func:`integrate_with_corrections` and the resulting per-item losses
  are reduced (mean, sum, weighted).

- :class:`BatchedSpecLoss` — single shared spec, many images. Uses
  :func:`integrate_soft_batch` under the hood so the geometry is
  evaluated only once per refinement step regardless of how many
  images.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..binning import SoftBinGeometry, integrate_soft_batch
from ..corrections import integrate_with_corrections
from ..spec import IntegrationSpec


_REDUCTIONS = ("mean", "sum", "none")


def _reduce(values: torch.Tensor, reduction: str,
            weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is not None:
        if weights.shape != values.shape:
            raise ValueError(
                f"weights shape {tuple(weights.shape)} != "
                f"values shape {tuple(values.shape)}"
            )
        wsum = weights.sum() + 1e-30
        if reduction == "mean":
            return (weights * values).sum() / wsum
        if reduction == "sum":
            return (weights * values).sum()
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"unknown reduction {reduction!r} (use one of {_REDUCTIONS})")


class MultiImageLoss(nn.Module):
    """Aggregate a per-image loss over a list of independent items.

    Parameters
    ----------
    loss_fn :
        Callable ``loss_fn(int2d, spec, *extra) -> scalar``. Examples:
        any of :class:`EtaUniformityLoss`, :class:`PeakPositionLoss`,
        or a lambda that wraps :class:`ProfileMSELoss` with a target.
    reduction :
        How to combine per-item scalars: ``"mean"`` (default), ``"sum"``,
        or ``"none"`` (returns a 1-D tensor of per-item losses).
    weights :
        Optional weights for each item; same length as the call list.
        Combined with the reduction (``mean`` becomes weighted mean).
    integrate_kwargs :
        Extra kwargs forwarded to :func:`integrate_with_corrections`
        (e.g. ``polarization``, ``solid_angle``, ``apply_trans_opt``).
        These apply to every item in the call.

    Forward signature:
        ``loss(items, **shared_extras) -> scalar``

    Where ``items`` is a list of tuples ``(image, spec)`` or
    ``(image, spec, *per_item_args)``. ``per_item_args`` are forwarded
    after ``int2d`` and ``spec`` to ``loss_fn`` for that item;
    ``shared_extras`` apply to every item.
    """

    def __init__(
        self,
        loss_fn: Callable,
        *,
        reduction: str = "mean",
        weights: Optional[Sequence[float]] = None,
        integrate_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {_REDUCTIONS}, got {reduction!r}"
            )
        self.loss_fn = loss_fn
        self.reduction = reduction
        self._weights = (None if weights is None
                          else torch.as_tensor(list(weights),
                                                 dtype=torch.float64))
        self.integrate_kwargs = dict(integrate_kwargs or {})

    def forward(self, items, **shared_extras) -> torch.Tensor:
        if not items:
            raise ValueError("items list is empty")
        if self._weights is not None and len(self._weights) != len(items):
            raise ValueError(
                f"weights length {len(self._weights)} != items length {len(items)}"
            )
        per_item = []
        for it in items:
            if len(it) < 2:
                raise ValueError(
                    f"each item must be (image, spec, ...), got {it}"
                )
            image, spec = it[0], it[1]
            extras = it[2:]
            int2d = integrate_with_corrections(
                image, spec, **self.integrate_kwargs,
            )
            L = self.loss_fn(int2d, spec, *extras, **shared_extras)
            per_item.append(L)
        per_item_t = torch.stack(per_item)
        return _reduce(per_item_t, self.reduction, self._weights)


class BatchedSpecLoss(nn.Module):
    """Single shared spec, many images — uses ``integrate_soft_batch``.

    Faster than :class:`MultiImageLoss` for the common multi-frame case
    because the per-pixel R/η is computed once and reused across all
    images, rather than once per image.

    Parameters
    ----------
    loss_fn :
        Callable ``loss_fn(int2d, spec, target_2d_or_None) -> scalar``,
        applied to each image's 2D integrated array. Receives the spec
        for context (e.g. for binning info) but typically only uses
        ``int2d`` and the matching target.
    reduction, weights :
        Same as :class:`MultiImageLoss`.
    apply_trans_opt :
        Forwarded to :func:`integrate_soft_batch`.

    Forward signature:
        ``loss(images_3d, spec, geom, targets_3d=None) -> scalar``

    ``images_3d``: ``(N, NrPixelsZ, NrPixelsY)``.
    ``geom``: a :class:`SoftBinGeometry` matching ``spec`` (build once
    per spec change).
    ``targets_3d``: optional ``(N, n_eta, n_r)`` per-image targets;
    forwarded to ``loss_fn`` as the third arg if non-None.
    """

    def __init__(
        self,
        loss_fn: Callable,
        *,
        reduction: str = "mean",
        weights: Optional[Sequence[float]] = None,
        apply_trans_opt: bool = True,
    ):
        super().__init__()
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {_REDUCTIONS}, got {reduction!r}"
            )
        self.loss_fn = loss_fn
        self.reduction = reduction
        self._weights = (None if weights is None
                          else torch.as_tensor(list(weights),
                                                 dtype=torch.float64))
        self.apply_trans_opt = bool(apply_trans_opt)

    def forward(
        self,
        images_3d: torch.Tensor,
        spec: IntegrationSpec,
        geom: SoftBinGeometry,
        targets_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if images_3d.ndim != 3:
            raise ValueError(
                f"images_3d must be (N, NrPixelsZ, NrPixelsY), got "
                f"{tuple(images_3d.shape)}"
            )
        n = images_3d.shape[0]
        if self._weights is not None and len(self._weights) != n:
            raise ValueError(
                f"weights length {len(self._weights)} != n_images {n}"
            )
        int2d_b = integrate_soft_batch(images_3d, geom,
                                         apply_trans_opt=self.apply_trans_opt)
        per_item = []
        for i in range(n):
            tgt = None if targets_3d is None else targets_3d[i]
            args = (int2d_b[i], spec) if tgt is None else (int2d_b[i], spec, tgt)
            per_item.append(self.loss_fn(*args))
        per_item_t = torch.stack(per_item)
        return _reduce(per_item_t, self.reduction, self._weights)


__all__ = ["MultiImageLoss", "BatchedSpecLoss"]
