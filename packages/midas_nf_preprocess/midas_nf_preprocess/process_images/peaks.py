"""Peak detection: clean replacement for the C scanline flood-fill.

The C code (``ProcessImagesCombined.c`` L431-L562) does:

  1. LoG-convolve ``img`` -> ``L``.
  2. Find zero-crossings of ``L`` where ``img != 0`` -> ``edges``.
  3. Union-find over connected ``edges`` -> per-edge peak IDs.
  4. Scanline flood-fill the *interior* of each closed edge boundary with that
     peak's ID, with several state machines, a 900-element BFS queue, and a
     hardcoded "if no peaks were found, set pixel 2045 to 1" workaround.

We replace step (4) with a clean observation:

  The standard LoG kernel is negative-center (Mexican hat with the conventional
  ``-1/(pi*sigma^4)`` leading factor; matches the C kernel at L252). Convolving
  with a positive blob gives a *negative* response at the blob center and a
  positive response just outside; the zero-crossing IS the boundary. So
  ``peak_mask = ((L < 0) | zero_crossings(L)) & (img > 0)``
  is exactly the (interior + boundary) labeled region the C produces -- without
  any flood fill.

We then label-propagate connected components of ``peak_mask``. The propagation
is a min over the pixel itself + 8 neighbors, repeated until convergence. Each
iteration is a single ``stacked.min(dim=0)`` over 9 shifted views; converges in
~D iterations where D is the diameter of the largest blob (small for NF peaks).
Vectorized, all backends, no Python-level inner loop over pixels.

Soft surrogate (autograd path):

  ``spot_prob = sigmoid(-L / T_L) * sigmoid(img / T_img)``

is a smooth proxy for ``peak_mask.float()`` that you can backprop through. The
``-L`` matches the negative-center convention: large negative L (blob interior)
becomes a high spot probability. The hard ``labels`` output is detached.

Temperatures (per-tensor scaling):

  - ``T_img`` and ``T_L`` are picked separately so each sigmoid sees a
    well-conditioned argument (raw pixel intensities and LoG responses live on
    different scales).
  - With ``soft_temperature="auto"`` (default), each is set to the median of
    the absolute non-zero values of its tensor (a robust scale estimator).
    Computed under ``no_grad`` so the temperature does not participate in
    autograd.
  - Pass a float to override; the same value is used for both sigmoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F

# Optional fast connected-component labeller. scipy is already a hard dependency
# of the MIDAS stack, but keep this soft so peaks.py stays importable without it
# -- the torch propagation below is the fallback and gives identical structure.
try:  # pragma: no cover - trivial import guard
    import numpy as _np
    from scipy.ndimage import label as _SCIPY_LABEL

    _EIGHT_CONNECTED = _np.ones((3, 3), dtype=bool)
except Exception:  # pragma: no cover
    _SCIPY_LABEL = None
    _EIGHT_CONNECTED = None


# -----------------------------------------------------------------------------
# Connected components: 8-neighbor label propagation, all backends.
# -----------------------------------------------------------------------------


def _stack_3x3_min(values: torch.Tensor, sentinel: int) -> torch.Tensor:
    """Element-wise min over each pixel + its 8 neighbors, sentinel at borders.

    ``values`` has shape ``[H, W]``. Output has shape ``[H, W]``.
    """
    H, W = values.shape
    padded = torch.full(
        (H + 2, W + 2), sentinel, dtype=values.dtype, device=values.device
    )
    padded[1:-1, 1:-1] = values
    # Stack 9 shifted views along a new leading axis, then min over it.
    views = [
        padded[i : i + H, j : j + W] for i in range(3) for j in range(3)
    ]
    stacked = torch.stack(views, dim=0)
    return stacked.min(dim=0).values


def label_components(
    mask: torch.Tensor,
    *,
    max_iters: int = 1024,
    return_n: bool = False,
):
    """Label 8-connected components of a boolean mask via parallel propagation.

    Parameters
    ----------
    mask : Tensor of shape ``[H, W]``, dtype bool (or any tensor where ``!= 0``
        defines the foreground).
    max_iters : safety cap on iterations. Convergence is reached in roughly
        ``ceil(diameter)`` iterations where diameter is the longest geodesic
        path in the largest connected component.
    return_n : if True, also return the number of components found.

    Returns
    -------
    labels : Int64 Tensor of shape ``[H, W]``. 0 = background, 1..K = component IDs
        in arbitrary but deterministic order (smallest flat-index seed first).
    (optionally) n_components : int
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected [H, W], got shape {tuple(mask.shape)}")
    H, W = mask.shape
    device = mask.device
    fg = mask.to(torch.bool) if mask.dtype != torch.bool else mask

    # ---- fast path: scipy's single-pass C labeller --------------------------
    #
    # The iterative propagation below needs ~diameter passes to converge, and
    # each pass runs a 3x3 min-filter over the WHOLE image plus an
    # ``(new != old).any()`` check. On CUDA that ``.any()`` is a device->host
    # SYNC every iteration, so a frame becomes hundreds of tiny kernels each
    # followed by a stall: the GPU sits at ~0% while one CPU core spins. On a
    # 2048^2 frame with thousands of blobs that dominated the whole image-
    # reduction stage.
    #
    # Labels are INTEGERS -- nothing differentiable flows through them (the
    # caller is already inside ``torch.no_grad()``), so there is no autograd
    # reason to keep this in torch. scipy.ndimage.label is a single-pass
    # union-find in C.
    #
    # Connectivity must stay 8-way to match ``_stack_3x3_min`` (scipy defaults
    # to 4-way, hence the explicit 3x3 structure), and scipy already numbers
    # components contiguously 1..K, which is what the renumbering below does.
    if _SCIPY_LABEL is not None:
        fg_np = fg.detach().cpu().numpy()
        lab_np, n_comp = _SCIPY_LABEL(fg_np, structure=_EIGHT_CONNECTED)
        labels = torch.from_numpy(lab_np.astype("int64")).to(device)
        if return_n:
            return labels, int(n_comp)
        return labels

    # Initial labels: each foreground pixel gets its (1-indexed) flat index.
    # Background gets the sentinel (max int) so it never wins min comparisons.
    sentinel = torch.iinfo(torch.int64).max
    flat = torch.arange(1, H * W + 1, device=device, dtype=torch.int64).view(H, W)
    labels = torch.where(fg, flat, torch.full_like(flat, sentinel))

    # Iterate until stable.
    for _ in range(max_iters):
        new_labels = _stack_3x3_min(labels, sentinel)
        new_labels = torch.where(
            fg, new_labels, torch.full_like(new_labels, sentinel)
        )
        if not (new_labels != labels).any():
            labels = new_labels
            break
        labels = new_labels
    else:
        # Loop completed without break -- did not converge.
        # Return what we have; downstream code can warn if it cares.
        pass

    # Replace sentinel with 0 (background).
    labels = torch.where(fg, labels, torch.zeros_like(labels))

    # Renumber to contiguous IDs 1..K.
    unique_vals = torch.unique(labels)
    unique_vals = unique_vals[unique_vals != 0]
    n_components = int(unique_vals.numel())
    if n_components == 0:
        return (labels, 0) if return_n else labels

    max_label = int(labels.max().item())
    remap = torch.zeros(max_label + 1, dtype=torch.int64, device=device)
    remap[unique_vals] = torch.arange(1, n_components + 1, device=device, dtype=torch.int64)
    labels = remap[labels]

    return (labels, n_components) if return_n else labels


# -----------------------------------------------------------------------------
# Zero-crossings of the LoG response.
# -----------------------------------------------------------------------------


def zero_crossings(log_response: torch.Tensor) -> torch.Tensor:
    """Pixels where ``log_response`` changes sign across at least one 8-neighbor.

    Convention (matches C L289-L300):
      - sign-change condition: ``sign(self) != sign(neighbor)``, with
        ``sign(0)`` treated as positive (i.e. 0 falls into the same class as positives).
      - border pixels (no full 8-neighborhood) cannot be edges.

    Returns a bool Tensor of shape ``[H, W]``.
    """
    if log_response.ndim != 2:
        raise ValueError(f"Expected [H, W], got shape {tuple(log_response.shape)}")
    H, W = log_response.shape
    pos = log_response >= 0  # match C: (val >= 0 && neighbor < 0) || (val < 0 && neighbor >= 0)

    # Pad with a constant matching the center: doesn't matter for the border
    # (we'll mask it), but we need a consistent shape.
    padded = F.pad(pos.unsqueeze(0).unsqueeze(0).to(torch.uint8), (1, 1, 1, 1), value=1).squeeze()
    center = padded[1:-1, 1:-1]

    edges = torch.zeros_like(center, dtype=torch.bool)
    # 8 neighbor shifts
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = padded[1 + di : 1 + di + H, 1 + dj : 1 + dj + W]
            edges = edges | (center != shifted)

    # Border ineligible (matches C's range NrPixelsY+1 .. nTotalPixels - NrPixelsY-1)
    edges[0, :] = False
    edges[-1, :] = False
    edges[:, 0] = False
    edges[:, -1] = False
    return edges


# -----------------------------------------------------------------------------
# Peak finding: clean LoG-sign + CC + soft surrogate.
# -----------------------------------------------------------------------------


@dataclass
class PeakFindOutputs:
    """Bundle of outputs from a single peak-finding pass.

    All tensors are on the same device, with shape ``[Z, Y]``.

    Attributes
    ----------
    log_response : Tensor (autograd). LoG convolution output.
    spot_prob    : Tensor (autograd). Soft surrogate, smooth proxy for spot mask.
    labels       : Int64 Tensor (detached). Connected-component IDs, 0 = no spot.
    n_components : int. Number of distinct labels in ``labels`` (excluding 0).
    temperature_img : float. Sigmoid temperature applied to ``img``.
    temperature_log : float. Sigmoid temperature applied to ``-log_response``.
    """

    log_response: torch.Tensor
    spot_prob: torch.Tensor
    labels: torch.Tensor
    n_components: int
    temperature_img: float = 1.0
    temperature_log: float = 1.0


def auto_temperature(
    x: torch.Tensor,
    *,
    floor: float = 1e-6,
    quantile: float = 0.95,
    saturation_factor: float = 4.0,
) -> torch.Tensor:
    """Robust scale estimate for picking a sigmoid temperature.

    Returns ``quantile(|x|nonzero, q) / saturation_factor``. With the defaults
    ``q=0.95`` and ``saturation_factor=4``, ``sigmoid(x / T)`` evaluates to:

      - ~0.5 at ``x = 0``
      - ~0.88 at ``x`` equal to half the 95th percentile
      - ~0.98 at ``x`` equal to the 95th percentile

    so the sigmoid transitions smoothly across the bulk of the signal range
    rather than saturating at the dimmest non-zero pixels (a median of nonzero
    `|x|` is too small for sparse images like median-subtracted HEDM frames).

    Computed under ``no_grad`` on a detached copy, so the temperature itself
    does not participate in autograd.

    Falls back to ``1.0`` if ``x`` has no non-zero entries.
    """
    with torch.no_grad():
        absx = x.detach().abs()
        nonzero = absx[absx > 0]
        if nonzero.numel() == 0:
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        q_val = torch.quantile(nonzero, quantile)
        return torch.clamp(q_val / saturation_factor, min=floor)


def _resolve_temperature(
    user: Union[float, str, torch.Tensor],
    img: torch.Tensor,
    log_response: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick (T_img, T_log) from a user spec.

    - "auto": ``auto_temperature(img), auto_temperature(log_response)``
    - float / 0-d Tensor: same value for both, detached and broadcast.
    """
    if isinstance(user, str):
        if user.lower() != "auto":
            raise ValueError(f"Unknown temperature mode '{user}', use 'auto' or a float.")
        return auto_temperature(img), auto_temperature(log_response)
    if isinstance(user, torch.Tensor):
        t = user.detach().to(device=img.device, dtype=img.dtype)
    else:
        t = torch.tensor(float(user), device=img.device, dtype=img.dtype)
    if t <= 0:
        raise ValueError(f"Temperature must be positive, got {float(t)}")
    return t, t


def find_peaks(
    img: torch.Tensor,
    log_kernels: list[torch.Tensor],
    *,
    soft_temperature: Union[float, str, torch.Tensor] = "auto",
    cc_max_iters: int = 1024,
) -> PeakFindOutputs:
    """Multi-scale peak detection on a single 2D image.

    Parameters
    ----------
    img : Tensor of shape ``[Z, Y]``. Median-subtracted, spatially-smoothed image.
        Should be non-negative for the soft path to behave sensibly.
    log_kernels : list of square odd-size kernels (each ``[2r+1, 2r+1]``). The
        function applies each in order and combines via element-wise OR over the
        peak masks; labels from the *first* kernel take precedence on overlaps,
        matching the C ``Image4 then Image5`` precedence at L1004-L1012.
    soft_temperature : ``"auto"`` (default), or a positive float / 0-d Tensor.
        ``"auto"`` picks separate temperatures for the ``img`` and ``log_response``
        sigmoids equal to the median of each tensor's absolute non-zero values.
        Computed under ``no_grad``, so the temperature does not enter autograd.
    cc_max_iters : safety cap for label propagation.

    Returns
    -------
    PeakFindOutputs.
    """
    from .log_filter import apply_log

    if img.ndim != 2:
        raise ValueError(f"Expected img [Z, Y], got shape {tuple(img.shape)}")
    if not log_kernels:
        raise ValueError("Need at least one LoG kernel.")

    # --- Differentiable path: LoG response and soft surrogate ---
    # Use the FIRST kernel for the autograd-tracked log_response. (We could
    # combine across scales, but keeping it tied to one scale matches the
    # primary kernel that the C path uses for its labels.)
    log_response = apply_log(img, log_kernels[0])
    T_img, T_log = _resolve_temperature(soft_temperature, img, log_response)
    img_pos = torch.sigmoid(img / T_img)
    # Negative-center kernel: spot interior has L < 0, so -L > 0 there.
    log_neg = torch.sigmoid(-log_response / T_log)
    spot_prob = log_neg * img_pos  # smooth proxy for (L<0) & (img>0)

    # --- Detached path: hard mask, CC, multi-scale combine ---
    with torch.no_grad():
        img_d = img.detach()
        img_pos_hard = img_d > 0

        labels = torch.zeros(img.shape, dtype=torch.int64, device=img.device)
        n_total = 0
        for ki, kernel in enumerate(log_kernels):
            L = apply_log(img_d, kernel)
            inside = (L < 0) & img_pos_hard
            boundary = zero_crossings(L) & img_pos_hard
            peak_mask = inside | boundary
            if not peak_mask.any():
                continue
            scale_labels, n = label_components(
                peak_mask, max_iters=cc_max_iters, return_n=True
            )
            if n == 0:
                continue
            # Only fill where labels is currently 0 (first scale wins).
            empty = labels == 0
            scale_labels_offset = torch.where(
                scale_labels > 0, scale_labels + n_total, scale_labels
            )
            labels = torch.where(empty & (scale_labels > 0), scale_labels_offset, labels)
            n_total += n

    return PeakFindOutputs(
        log_response=log_response,
        spot_prob=spot_prob,
        labels=labels,
        n_components=n_total,
        temperature_img=float(T_img.detach().item()),
        temperature_log=float(T_log.detach().item()),
    )
