"""ImTransOpt — apply v1's image transform opcodes to a torch tensor.

Mirrors :func:`midas_integrate.detector_mapper._apply_trans_opt_forward`
exactly (same op codes, same composition order). v1's hot path bakes
the transform into the precomputed map; v2's soft path computes
geometry on the un-transformed pixel grid and integrates by reading
``image[z, y]``, so the user must feed a forward-transformed image —
or let v2 apply it for them.

Op codes:
  0 = no-op
  1 = flip Y (cols, ``[:, ::-1]``)
  2 = flip Z (rows, ``[::-1, :]``)
  3 = transpose (only if NrPixelsY == NrPixelsZ)
"""
from __future__ import annotations

from typing import Sequence

import torch


def apply_trans_opt_forward(
    image: torch.Tensor,
    trans_opt: Sequence[int],
    *,
    NrPixelsY: int,
    NrPixelsZ: int,
) -> torch.Tensor:
    """Forward-apply v1's ImTransOpt sequence to a 2-D torch tensor.

    ``image`` has shape ``(NrPixelsZ, NrPixelsY)`` and is returned with
    the same shape (transpose only legal when square). Composition order
    matches v1: ops are applied in the order they appear in
    ``trans_opt``.

    Returns a contiguous tensor (so the underlying view doesn't surprise
    downstream ``.flatten()`` calls).
    """
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2-D (Z, Y), got shape {tuple(image.shape)}"
        )
    out = image
    for opt in trans_opt:
        if opt == 0:
            continue
        if opt == 1:
            out = torch.flip(out, dims=(1,))           # flip Y
        elif opt == 2:
            out = torch.flip(out, dims=(0,))           # flip Z
        elif opt == 3:
            if NrPixelsY != NrPixelsZ:
                raise ValueError(
                    "TransOpt=3 (transpose) requires NrPixelsY == NrPixelsZ "
                    f"(got {NrPixelsY} x {NrPixelsZ})"
                )
            out = out.T
        else:
            raise ValueError(f"unknown ImTransOpt op code {opt}")
    return out.contiguous()


def needs_trans_opt(trans_opt: Sequence[int]) -> bool:
    """True when ``trans_opt`` contains any non-trivial op."""
    return any(op != 0 for op in trans_opt)


__all__ = ["apply_trans_opt_forward", "needs_trans_opt"]
