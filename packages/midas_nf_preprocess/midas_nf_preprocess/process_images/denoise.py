"""Non-local-means denoising in pure PyTorch, for CPU or GPU.

Why this exists
---------------
``_nlm_denoise_residual`` used to be scikit-image only, i.e. CPU only, and that
one fact shaped the whole frame loop: ``process_layer`` moves the entire stack
back to the host whenever ``NLMDenoise 1`` is set, because "NLM already forces
a host round-trip per frame".  A torch implementation removes the round trip,
so the stack can stay on the device.

Measured on a 4600x5320 20-ID frame (RTX A6000, ``h=4``, patch 5, distance 6):

===========================  ========  ==============
implementation                   time  vs 1 CPU thread
===========================  ========  ==============
skimage ``fast_mode=True``    26.9 s          1x
this module, CPU              (same order as skimage)
this module, CUDA              0.74 s         42x
===========================  ========  ==============

The algorithm
-------------
Identical in form to skimage's ``fast_mode`` (Darbon et al.).  For each offset
``t`` in the search window::

    d2   = boxsum_over_patch( (I(x) - I(x+t))**2 - 2*var )
    dist = max(d2, 0) / (h**2 * s**2)
    w    = exp(-dist),   with w = 0 where dist >= DISTANCE_CUTOFF

and ``out(x) = sum_t w_t(x) I(x+t) / sum_t w_t(x)``, the ``t = 0`` term
carrying weight 1.

Because ``d2`` is symmetric under ``t -> -t``, only the half-window
``{dx >= 1} U {dx = 0, dy >= 1}`` is evaluated and each weight is applied to
both endpoints of the pair.  That is skimage's own trick and it halves the
work (measured 1.03 s -> 0.74 s).

Agreement with scikit-image -- CLOSE, but NOT bit-equivalent
------------------------------------------------------------
This implements the published fast-NLM algorithm; scikit-image's variant
differs in some detail that has NOT been pinned down (its source ships only as
a compiled extension).  A patch-window shift of (-1, -1) improves agreement
markedly (corr 0.988 -> 0.995, max|diff| halved) but no shift makes it exact,
so at least one further difference remains.

Measured agreement:

* real 20-ID frames, ``h = 4``, thresholds 4/8/16, three omega positions,
  excluding the direct-beam stripe: **identical blob counts (>= 4 px) in all
  nine comparisons**, pixel IoU 0.978-0.990, correlation 0.99997.
* dense synthetic spots: correlation 0.988-0.998, blob counts within **+-2**
  out of ~110, pixel IoU 0.73-0.99 -- worst where contrast is highest.

The divergence concentrates in SATURATED, DENSELY-LIT regions.  Across a whole
20-ID frame including the direct-beam stripe the counts diverge badly (269 vs
340 blobs at threshold 8) and every extra blob lies in rows 4527-4548 -- the
beam.  There the patch distances are enormous, essentially all weights fall
past ``DISTANCE_CUTOFF``, and the two implementations resolve the resulting
0/0 differently.  Diffraction spots are unaffected, and spot matching excludes
that region anyway via ``r_min_px``.  Do not use this module to denoise the
direct beam itself.

That is why ``NLMBackend`` defaults to ``skimage``: switching silently would
perturb existing reductions.  Opt in when the speed matters, and keep any
comparison that has to be exact on a single backend.

A parity test on a small crop is NOT sufficient to catch this: an early check on
a 700-row tile containing 8 blobs showed exact agreement while the full frame
disagreed by 26 %.  Test on data with enough blobs to have power.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F

__all__ = ["nl_means_torch", "DISTANCE_CUTOFF"]

#: Weights below ``exp(-DISTANCE_CUTOFF)`` are dropped, as in scikit-image.
DISTANCE_CUTOFF = 5.0


def _boxsum(x: torch.Tensor, s: int, kw: torch.Tensor,
            kh: torch.Tensor) -> torch.Tensor:
    """Sum over an ``s x s`` window, same-size output, separable."""
    k = s // 2
    x = F.pad(x, (k, k, k, k), mode="constant", value=0.0)
    return F.conv2d(F.conv2d(x, kw), kh)


def nl_means_torch(
    image: torch.Tensor,
    *,
    patch_size: int = 5,
    patch_distance: int = 6,
    h: float = 1.0,
    sigma: float = 0.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Fast non-local means, in the form of scikit-image's ``fast_mode``.

    Close to scikit-image but NOT bit-equivalent -- see the module
    docstring for the measured agreement and where it breaks down.

    Parameters
    ----------
    image
        2-D tensor (or anything ``torch.as_tensor`` accepts).
    patch_size, patch_distance
        Patch side ``s`` and search radius ``d``, as in scikit-image.
    h
        Filter strength **in the units of the image**.  On photon-starved NF
        residuals this must be given absolutely: ``sigma_MAD`` is exactly 0
        when the residual is mostly exact zeros, so an ``h = factor * sigma``
        rule collapses (see ``NLMHAbsolute``).
    sigma
        Noise standard deviation; ``2*sigma**2`` is subtracted from the patch
        distance, exactly as scikit-image does.
    device, dtype
        Where to run and in what precision.  Defaults to the input's own
        device, and to ``float32``.  ``float16`` works and halves memory with
        no measurable speed change (the kernel is bandwidth-bound), but has not
        been parity-tested -- prefer ``float32`` for production reductions.

    Returns
    -------
    torch.Tensor
        Denoised image, same shape as the input, on ``device``.

    Notes
    -----
    Peak memory is ~4 frames' worth (1.1 GB for 4600x5320 float32).  The whole
    routine is differentiable, so it can sit inside an autograd graph.
    """
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {tuple(image.shape)}")
    dev = torch.device(device) if device is not None else image.device
    dt = dtype or torch.float32
    a = image.to(dev, dt)

    H, W = a.shape
    s, d = int(patch_size), int(patch_distance)
    if s % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {s}")
    var = float(sigma) ** 2
    h2s2 = float(h) ** 2 * float(s) ** 2
    if h2s2 <= 0:
        raise ValueError("h must be > 0; on photon-starved data set it "
                         "absolutely (NLMHAbsolute), not as a multiple of "
                         "sigma_MAD, which can be exactly 0")

    pad = d + s // 2
    p = F.pad(a[None, None], (pad, pad, pad, pad), mode="reflect")
    kw = torch.ones((1, 1, 1, s), dtype=p.dtype, device=dev)
    kh = torch.ones((1, 1, s, 1), dtype=p.dtype, device=dev)

    out = p.clone()                       # t = 0 term, weight 1
    wsum = torch.ones_like(p)
    zero = torch.zeros((), dtype=p.dtype, device=dev)

    # half of the search window; the mirror offset is handled by symmetry
    offsets = [(dy, dx) for dx in range(1, d + 1) for dy in range(-d, d + 1)]
    offsets += [(dy, 0) for dy in range(1, d + 1)]

    for dy, dx in offsets:
        shifted = torch.roll(p, shifts=(-dy, -dx), dims=(2, 3))     # p[x + t]
        diff2 = (p - shifted) ** 2 - 2.0 * var
        dist = _boxsum(diff2, s, kw, kh).clamp_(min=0.0) / h2s2
        w = torch.where(dist >= DISTANCE_CUTOFF, zero, torch.exp(-dist))
        out = out + w * shifted                                     # x   <- x+t
        wsum = wsum + w
        out = out + torch.roll(w * p, shifts=(dy, dx), dims=(2, 3))  # x+t <- x
        wsum = wsum + torch.roll(w, shifts=(dy, dx), dims=(2, 3))

    return (out / wsum)[0, 0, pad:pad + H, pad:pad + W]
