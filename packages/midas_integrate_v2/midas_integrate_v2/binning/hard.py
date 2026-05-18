"""Native-torch hard-binning geometry.

Companion to :class:`SoftBinGeometry`: each pixel is assigned to exactly
ONE (R bin, η bin) based on its centre. No interpolation, no polygon
area; just a single ``index_add`` per pixel per integrate.

When to use which:

- :class:`SoftBinGeometry` + :func:`integrate_soft` — differentiable in
  geometry, ~4× more contributions per pixel (linear interp). Use this
  for refinement.
- :class:`HardBinGeometry` + :func:`integrate_hard` — fastest pure-torch
  forward path, NOT differentiable in geometry (the bin assignment is
  ``floor((R - RMin) / RBinSize)``, whose gradient is zero almost
  everywhere). Use this for production batch integration when geometry
  is fixed.
- v1 ``build_map`` + ``integrate(mode='floor')`` — the same math as
  hard-bin but routed through v1's numba mapper. Use when you also want
  the v1-format Map.bin written to disk for downstream tooling.

The forward of hard-bin matches v1's ``floor`` integration to ULP
precision on a uniformly-illuminated image, modulo the in-range mask
behaviour at the very-low-R edge (v1 keeps any pixel with floor(R) in
range; hard-bin ditto).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ..forward import eval_pixel_REta
from ..spec import IntegrationSpec
from .mask import normalise_mask
from .trans_opt import apply_trans_opt_forward, needs_trans_opt


@dataclass
class HardBinGeometry:
    """Precomputed hard-binning geometry.

    Each pixel maps to exactly one flat bin index ``flat_bin[i] =
    eta_bin * n_r + r_bin``. Pixels whose ``(R, η)`` lands outside the
    binned ranges are masked out via the ``valid`` boolean tensor.

    Fields:
        flat_bin   : (n_pix,) long — flat (eta, r) bin index per pixel
        valid      : (n_pix,) bool — pixels that landed inside a bin
        n_r, n_eta : ints
        n_pixels_y, n_pixels_z : detector dimensions
        trans_opt  : list of ImTransOpt op codes (forwarded to integrate)

    Forward call:
        flat_out = zeros(n_eta * n_r)
        flat_out.index_add_(0, flat_bin[valid], image_flat[valid])
    """
    flat_bin: torch.Tensor
    valid: torch.Tensor
    n_r: int
    n_eta: int
    n_pixels_y: int
    n_pixels_z: int
    trans_opt: list = None        # type: ignore[assignment]

    @classmethod
    def from_spec(cls, spec: IntegrationSpec, *,
                   mask: Optional[np.ndarray] = None) -> "HardBinGeometry":
        """Build the hard-bin geometry.

        Parameters
        ----------
        mask :
            Optional 2D ``(NrPixelsZ, NrPixelsY)`` mask. Non-zero entries
            mark pixels to exclude from integration (beam stop, dead
            pixels, module gaps). v1 convention: 1.0 = masked.
        """
        spec.validate()
        with torch.no_grad():
            R, Eta = eval_pixel_REta(spec)
        R_flat = R.reshape(-1)
        Eta_flat = Eta.reshape(-1)
        rb = ((R_flat - spec.RMin) / spec.RBinSize).floor().long()
        eb = ((Eta_flat - spec.EtaMin) / spec.EtaBinSize).floor().long()
        n_r, n_eta = spec.n_r_bins, spec.n_eta_bins
        valid = (rb >= 0) & (rb < n_r) & (eb >= 0) & (eb < n_eta)
        # Apply mask: treat masked pixels as "not valid" so they never
        # contribute. Per-bin counts therefore exclude masked pixels and
        # the normalised output is correct.
        mask_np = normalise_mask(mask, NrPixelsY=spec.NrPixelsY,
                                  NrPixelsZ=spec.NrPixelsZ)
        if mask_np is not None:
            mask_t = torch.from_numpy(mask_np.reshape(-1))
            valid = valid & ~mask_t
        # Clamp out-of-range to 0 to keep the flat index well-defined;
        # the `valid` mask is the source of truth.
        rb_c = rb.clamp(0, n_r - 1)
        eb_c = eb.clamp(0, n_eta - 1)
        flat_bin = eb_c * n_r + rb_c
        return cls(
            flat_bin=flat_bin, valid=valid,
            n_r=n_r, n_eta=n_eta,
            n_pixels_y=spec.NrPixelsY, n_pixels_z=spec.NrPixelsZ,
            trans_opt=list(spec.TransOpt),
        )

    @property
    def device(self) -> torch.device:
        return self.flat_bin.device

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum().item())


def integrate_hard(
    image: torch.Tensor,
    geom: HardBinGeometry,
    *,
    apply_trans_opt: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Hard-bin integrate. Returns ``(n_eta, n_r)``.

    Each in-range pixel contributes its full intensity to its single
    nearest bin. When ``normalize=True``, divides the per-bin sum by the
    per-bin pixel count (so the output is mean-per-bin, matching v1's
    ``Normalize=1`` convention).
    """
    if image.shape != (geom.n_pixels_z, geom.n_pixels_y):
        raise ValueError(
            f"image shape {tuple(image.shape)} does not match "
            f"geometry ({geom.n_pixels_z}, {geom.n_pixels_y})"
        )
    if apply_trans_opt and geom.trans_opt and needs_trans_opt(geom.trans_opt):
        image = apply_trans_opt_forward(
            image, geom.trans_opt,
            NrPixelsY=geom.n_pixels_y, NrPixelsZ=geom.n_pixels_z,
        )
    img_flat = image.reshape(-1).to(torch.float64)
    n_bins = geom.n_eta * geom.n_r
    valid = geom.valid

    sums = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    sums = sums.index_add(0, geom.flat_bin[valid], img_flat[valid])

    if not normalize:
        return sums.reshape(geom.n_eta, geom.n_r)

    counts = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    ones = torch.ones_like(img_flat[valid])
    counts = counts.index_add(0, geom.flat_bin[valid], ones)
    out = sums / counts.clamp(min=1.0)
    return out.reshape(geom.n_eta, geom.n_r)


def integrate_hard_batch(
    images: torch.Tensor,
    geom: HardBinGeometry,
    *,
    apply_trans_opt: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Hard-bin integrate a batch of images at fixed geometry.

    ``images`` shape ``(N, NrPixelsZ, NrPixelsY)`` → ``(N, n_eta, n_r)``.
    """
    if images.ndim != 3 or images.shape[1:] != (geom.n_pixels_z,
                                                  geom.n_pixels_y):
        raise ValueError(
            f"images shape {tuple(images.shape)} does not match "
            f"(N, {geom.n_pixels_z}, {geom.n_pixels_y})"
        )
    if apply_trans_opt and geom.trans_opt and needs_trans_opt(geom.trans_opt):
        images = torch.stack([
            apply_trans_opt_forward(
                images[i], geom.trans_opt,
                NrPixelsY=geom.n_pixels_y, NrPixelsZ=geom.n_pixels_z,
            )
            for i in range(images.shape[0])
        ])
    n_imgs = images.shape[0]
    n_bins = geom.n_eta * geom.n_r
    img_flat = images.reshape(n_imgs, -1).to(torch.float64)  # (N, n_pix)
    valid = geom.valid
    valid_idx = geom.flat_bin[valid]                          # (n_valid,)
    valid_imgs = img_flat[:, valid]                            # (N, n_valid)

    sums = torch.zeros(n_imgs, n_bins, dtype=img_flat.dtype,
                        device=img_flat.device)
    sums = sums.index_add(1, valid_idx, valid_imgs)

    if not normalize:
        return sums.reshape(n_imgs, geom.n_eta, geom.n_r)

    counts = torch.zeros(n_bins, dtype=img_flat.dtype, device=img_flat.device)
    ones = torch.ones(int(valid.sum()), dtype=img_flat.dtype,
                       device=img_flat.device)
    counts = counts.index_add(0, valid_idx, ones)
    out = sums / counts.clamp(min=1.0).unsqueeze(0)
    return out.reshape(n_imgs, geom.n_eta, geom.n_r)


__all__ = ["HardBinGeometry", "integrate_hard", "integrate_hard_batch"]
