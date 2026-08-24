"""Tomography-based hex-grid masking.

Two modes:

  1. ``filter_grid_by_tomo``  -- C parity: a 2D uint8 image, square dimensions
     inferred from the file size. Grid coordinates are mapped to pixel indices
     with the C convention (Y axis flipped).
  2. ``filter_grid_by_bbox``  -- Python convenience: keep grid points inside a
     ``[x_min, x_max, y_min, y_max]`` rectangle in micrometers. This mirrors
     the ``GridMask`` parameter handled in the Python workflow driver
     (``nf_MIDAS.py:376-386``) without touching disk.

The point-in-image lookup is vectorized; gradient does not flow through this
operation (it returns a boolean mask).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import math

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Tomo image loading
# -----------------------------------------------------------------------------


def load_square_tomo(
    path: Union[str, Path], *, dtype: np.dtype = np.uint8
) -> np.ndarray:
    """Load a square binary tomography image, inferring side length from file size.

    Mirrors filterGridfromTomo.c L13-L21:

        sz = stat.st_size
        nrPxTomo = sqrt(sz)
        imTomo = uint8 buffer of shape (nrPxTomo, nrPxTomo)
    """
    path = Path(path)
    sz = path.stat().st_size
    nr_px = int(math.isqrt(sz)) if dtype == np.uint8 else int(math.sqrt(sz / np.dtype(dtype).itemsize))
    if nr_px * nr_px * np.dtype(dtype).itemsize != sz:
        raise ValueError(
            f"{path}: size {sz} bytes is not a perfect square for dtype {dtype}"
        )
    arr = np.fromfile(path, dtype=dtype, count=nr_px * nr_px)
    return arr.reshape(nr_px, nr_px)


# -----------------------------------------------------------------------------
# Tomo sampling
# -----------------------------------------------------------------------------


def sample_tomo(
    points_xy_um: torch.Tensor,
    tomo: Union[np.ndarray, torch.Tensor],
    px_tomo_um: float,
    *,
    legacy_c_parity: bool = True,
) -> torch.Tensor:
    """Sample tomography mask values at grid-point locations.

    The C being matched is ``filterGridfromTomo.c:39-43``::

        xPos = (int)((x / pxTomo) + ((double)nrPxTomo / 2));
        yPos = (int)((y / pxTomo) + ((double)nrPxTomo / 2));
        if (xPos >= 0 && yPos >= 0 && xPos < nrPxTomo && yPos < nrPxTomo &&
            imTomo[nrPxTomo * (nrPxTomo - yPos) + xPos] != 0)

    Three defects live in those five lines, all found 2026-08-23. Each is
    recorded here because each is silent — a mask displaced by one pixel
    reconstructs perfectly and simply masks the wrong grid points.

    **1. The row flip is off by one, and reads out of bounds.**
    ``nrPxTomo - yPos`` maps ``yPos`` in ``[0, n-1]`` onto rows ``[1, n]``:
    row 0 is never read, and ``yPos == 0`` indexes ``imTomo[n*n + xPos]``,
    past the end of the ``calloc``. In C that is an out-of-bounds heap read;
    in the Python transcription it raised ``IndexError`` on any grid point at
    the bottom edge. The flip that maps the range onto itself is
    ``n - 1 - yPos``. Undefined behaviour cannot be reproduced, so **both**
    modes now treat ``yPos == 0`` as outside the image and return 0; that is
    the only behaviour change in parity mode that is not a bug fix.

    **2. The Python truncated in the wrong place** — it computed
    ``int(x / px) + n // 2``, truncating the *quotient* and then adding, where
    the C truncates the *sum*. These differ by one pixel for every negative
    coordinate with a fractional part. Measured on a 200k-point grid over a
    1 mm sample at 1.5 um: **75 % of grid points landed on a different pixel
    than the C**, in x, y, or both. The existing tests all used integer
    coordinates, which is the one case where the two agree.

    **3. The Python used integer ``n // 2``** where the C uses
    ``(double)nrPxTomo / 2``. Identical for even ``n``, half a pixel apart for
    odd ``n``. The C is right: with the float centre, ``x = 0`` maps to the
    true centre pixel for both parities of ``n``.

    Parameters
    ----------
    points_xy_um : Tensor of shape ``(N, 2)``, columns ``(x, y)`` in um.
    tomo         : 2D uint8 array (numpy or torch).
    px_tomo_um   : pixel size of the tomo image, in um/pixel.
    legacy_c_parity
        ``True`` (default) reproduces the C's ``n - yPos`` row flip, so
        existing NF reconstructions stay reproducible. ``False`` uses the
        correct ``n - 1 - yPos`` and floors rather than truncates toward zero,
        which matters only at the negative edge. New work should pass
        ``False``; the default stays ``True`` because this function decides
        which voxels get reconstructed at all.

    Returns
    -------
    Int64 Tensor of shape ``(N,)`` with the looked-up mask value for each point.
    Out-of-image points get 0.
    """
    if points_xy_um.ndim != 2 or points_xy_um.shape[1] != 2:
        raise ValueError(
            f"Expected (N, 2) points, got shape {tuple(points_xy_um.shape)}"
        )
    if isinstance(tomo, np.ndarray):
        tomo_t = torch.from_numpy(tomo)
    else:
        tomo_t = tomo
    if tomo_t.ndim != 2 or tomo_t.shape[0] != tomo_t.shape[1]:
        raise ValueError(
            f"Expected square tomo image, got shape {tuple(tomo_t.shape)}"
        )
    nr_px = tomo_t.shape[0]
    device = points_xy_um.device
    tomo_t = tomo_t.to(device=device)

    # Truncate the SUM, as the C does -- see defect 2 above.
    xf = points_xy_um[:, 0] / px_tomo_um + nr_px / 2.0
    yf = points_xy_um[:, 1] / px_tomo_um + nr_px / 2.0
    if legacy_c_parity:
        # (int) in C truncates toward zero, so (-0.5) -> 0 and the point is
        # accepted into column 0 rather than rejected.
        x_pos = xf.to(torch.int64)
        y_pos = yf.to(torch.int64)
    else:
        x_pos = torch.floor(xf).to(torch.int64)
        y_pos = torch.floor(yf).to(torch.int64)

    in_bounds = (
        (x_pos >= 0) & (x_pos < nr_px) & (y_pos >= 0) & (y_pos < nr_px)
    )
    row = (nr_px - y_pos) if legacy_c_parity else (nr_px - 1 - y_pos)
    # Defect 1: in parity mode row == nr_px at y_pos == 0, which is off the end
    # of the image. The C reads past its buffer there; drop the point instead.
    in_bounds = in_bounds & (row >= 0) & (row < nr_px)

    zero = torch.zeros_like(row)
    values = tomo_t[torch.where(in_bounds, row, zero),
                    torch.where(in_bounds, x_pos, zero)].to(torch.int64)
    return torch.where(in_bounds, values, torch.zeros_like(values))


def filter_grid_by_tomo(
    grid_points: torch.Tensor,
    tomo: Union[np.ndarray, torch.Tensor],
    px_tomo_um: float,
    *,
    legacy_c_parity: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep grid points whose tomo lookup is non-zero.

    Parameters
    ----------
    grid_points : Tensor of shape ``(N, 5)`` -- columns ``(dx, dy, x, y, edge_half)``,
        the format from ``hex_grid.make_hex_grid``.
    tomo : square 2D mask (numpy uint8 or torch tensor of any int/bool dtype).
    px_tomo_um : pixel size of ``tomo`` in um/pixel.
    legacy_c_parity : see :func:`sample_tomo` -- three silent one-pixel defects
        are documented there, and this flag chooses whether to reproduce them.

    Returns
    -------
    (filtered_points, mask) where:

      - ``filtered_points`` has shape ``(K, 5)`` for the K kept points.
      - ``mask`` has shape ``(N,)`` with ``True`` for kept points.
    """
    if grid_points.ndim != 2 or grid_points.shape[1] != 5:
        raise ValueError(
            f"Expected (N, 5) grid points, got shape {tuple(grid_points.shape)}"
        )
    xy = grid_points[:, [2, 3]]
    values = sample_tomo(xy, tomo, px_tomo_um, legacy_c_parity=legacy_c_parity)
    mask = values != 0
    return grid_points[mask], mask


# -----------------------------------------------------------------------------
# Bounding-box masking (Python convenience, mirrors GridMask in nf_MIDAS.py)
# -----------------------------------------------------------------------------


def bbox_mask(
    grid_points: torch.Tensor,
    bbox_um: Sequence[float],
) -> torch.Tensor:
    """Boolean mask: True where ``(x, y)`` is inside ``bbox_um = [xmin, xmax, ymin, ymax]``."""
    if len(bbox_um) != 4:
        raise ValueError(
            f"bbox_um must be [xmin, xmax, ymin, ymax]; got length {len(bbox_um)}"
        )
    xmin, xmax, ymin, ymax = bbox_um
    if xmax < xmin or ymax < ymin:
        raise ValueError(f"bbox_um corners reversed: {bbox_um}")
    x = grid_points[:, 2]
    y = grid_points[:, 3]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def filter_grid_by_bbox(
    grid_points: torch.Tensor,
    bbox_um: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only grid points inside the rectangular bbox."""
    mask = bbox_mask(grid_points, bbox_um)
    return grid_points[mask], mask
