"""Temporal and spatial median filters.

Both are differentiable in the subgradient sense: ``torch.median`` is piecewise
linear in the input; the gradient flows through whichever element happened to be
selected as the median. That matches the behavior of the C ``quick_select``.

Border handling for the spatial median matches ``ProcessImagesCombined.c`` L957-L990:
pixels within ``radius`` of any edge pass through unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def temporal_median(stack: torch.Tensor) -> torch.Tensor:
    """Per-pixel median across the first (frame) axis.

    Parameters
    ----------
    stack : Tensor of shape ``[N, Z, Y]``

    Returns
    -------
    Tensor of shape ``[Z, Y]`` with dtype matching ``stack``.
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected [N, Z, Y], got shape {tuple(stack.shape)}")
    # torch.median.values is the actual median (selected element); torch.median has a
    # well-defined backward that routes the gradient to the selected index.
    return stack.median(dim=0).values


def streaming_temporal_median(
    source,
    *,
    n_frames: Optional[int] = None,
    row_block: int = 0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Temporal median of a layer without holding the layer in memory.

    Reads a band of rows across the selected frames, medians it, and moves on.
    Peak memory is ``n_frames x row_block x Y`` instead of ``N x Z x Y`` -- at
    60 frames and 460 rows that is ~0.3 GB against the 141 GB a whole 20-ID
    layer needs at fp32, which is the difference between this data entering the
    pipeline and not.

    Parameters
    ----------
    source : :class:`~midas_nf_preprocess.process_images.io.FrameSource`
    n_frames : how many EVENLY SPACED frames to median. ``None``/0 = all of
        them, which is the default and matches :func:`temporal_median` exactly.

        Subsampling is opt-in and PROVISIONAL. Measured once, on
        ``NF_Au_cube_0802_000708`` rows 2070-2530: 60 frames against all 1440
        moved 0.043 % of pixels by at most 4 counts and left the detected blobs
        (>= 4 px at the production threshold) IDENTICAL on five frames spread
        across the sweep. That is one band of one distance of one scan, and a
        median biased high would suppress weak spots silently, so the default
        stays at "all frames".
    row_block : rows per pass. 0 = whole frame at once.

    Returns
    -------
    Tensor ``[Z, Y]``.

    Notes
    -----
    Uses ``torch.median`` per block, not ``np.median``, so the result is
    element-identical to :func:`temporal_median` on the same frames. The two
    disagree for an even frame count -- torch selects the lower of the two
    middle elements, numpy averages them -- and a median that shifts by half a
    count when the code path changes would be a silent difference between
    reductions.
    """
    total = int(source.n_frames)
    if n_frames and 0 < int(n_frames) < total:
        idx = torch.linspace(0, total - 1, int(n_frames)).round().to(torch.long)
        idx = sorted(set(int(i) for i in idx))
    else:
        idx = list(range(total))

    nz, ny = int(source.nz), int(source.ny)
    blk = int(row_block) if row_block and int(row_block) > 0 else nz
    out = torch.empty((nz, ny), dtype=dtype or torch.float32, device=device or "cpu")
    for r0 in range(0, nz, blk):
        r1 = min(r0 + blk, nz)
        band = torch.from_numpy(source.read_rows(idx, r0, r1))
        out[r0:r1] = band.to(device=out.device, dtype=out.dtype).median(dim=0).values
    return out


def _unfold_blocks(img: torch.Tensor, k: int) -> torch.Tensor:
    """Extract k x k neighborhoods around every pixel as the last dim.

    Returns a tensor of shape ``[Z - 2r, Y - 2r, k*k]`` where r = (k-1)/2. No padding;
    the caller is responsible for masking the border to match the C semantics.
    """
    z, y = img.shape
    # F.unfold expects [B, C, H, W]; we use [1, 1, Z, Y] and extract sliding windows.
    blocks = F.unfold(img.unsqueeze(0).unsqueeze(0), kernel_size=k, padding=0)
    # blocks: [1, k*k, (Z-2r)*(Y-2r)]
    r = (k - 1) // 2
    return blocks.squeeze(0).T.reshape(z - 2 * r, y - 2 * r, k * k)


def spatial_median(img: torch.Tensor, radius: int) -> torch.Tensor:
    """Replace each interior pixel with the median of its (2r+1) x (2r+1) neighborhood.

    Border behavior matches the C: pixels within ``radius`` of any edge pass through
    unchanged. ``radius=0`` returns ``img`` unchanged.

    Parameters
    ----------
    img : Tensor of shape ``[Z, Y]``
    radius : int, 0 <= radius. Common values: 0 (identity), 1 (3x3), 2 (5x5).

    Returns
    -------
    Tensor of shape ``[Z, Y]``.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected [Z, Y], got shape {tuple(img.shape)}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if radius == 0:
        return img

    k = 2 * radius + 1
    z, y = img.shape
    if z < k or y < k:
        # Window doesn't fit anywhere: pass through unchanged, like the C border path.
        return img

    blocks = _unfold_blocks(img, k)  # [Z-2r, Y-2r, k*k]
    interior_med = blocks.median(dim=-1).values  # [Z-2r, Y-2r]

    # Stitch into a full-size output: edges = original, interior = median.
    out = img.clone()
    out[radius : z - radius, radius : y - radius] = interior_med
    return out
