"""Native-torch precomputed soft-bin geometry.

Companion to :class:`midas_integrate_v2.kernels.IntegrationGeometry` (which
bridges to v1's numba CSR kernel). This one is pure torch end-to-end:

- :class:`SoftBinGeometry` holds the per-pixel ``(R, η)`` tensors plus
  the linear-interpolation soft-bin indices and weights, all precomputed
  from an :class:`IntegrationSpec` once via
  :meth:`SoftBinGeometry.from_spec`.
- :func:`integrate_soft` applies a precomputed geometry to any image,
  returning the 2D ``(n_eta, n_r)`` integrated array.
- Gradient flows from the integrated output back through the geometry's
  R / Eta tensors to the originating spec parameters (when
  ``requires_grad=True`` was set on those parameters).

When to use this instead of :func:`build_geometry`:

- You're refining ``spec`` every step but integrating many images per
  step — amortise the ``eval_pixel_REta`` cost.
- You want to avoid the numba JIT path entirely (e.g. you're running
  in a Python session that has already imported torch and don't want to
  trip the OpenMP thread-state interaction).
- You want gradient flow without recomputing the per-pixel forward each
  ``integrate`` call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..diff.soft_bin import soft_bin_indices_weights
from ..forward import eval_pixel_REta
from ..spec import IntegrationSpec
from .trans_opt import apply_trans_opt_forward, needs_trans_opt


@dataclass
class SoftBinGeometry:
    """Precomputed pure-torch soft-binning geometry.

    Fields:
        R, Eta         : (n_pix,) flat per-pixel R/Eta tensors
        rb0, rb1, rw0, rw1 : R soft-bin indices/weights (long, long, float, float)
        eb0, eb1, ew0, ew1 : Eta soft-bin indices/weights
        n_r, n_eta     : ints
        n_pixels_y, n_pixels_z : detector dimensions
        trans_opt      : list of ImTransOpt op codes, copied from the
                         originating spec. ``integrate_soft`` will
                         forward-apply these to incoming raw images.

    All tensors carry gradient when the originating spec had
    ``requires_grad=True`` on geometry parameters. ``rb0`` / ``rb1`` /
    ``eb0`` / ``eb1`` are integer indices and never carry gradient.
    """
    R: torch.Tensor
    Eta: torch.Tensor
    rb0: torch.Tensor
    rb1: torch.Tensor
    rw0: torch.Tensor
    rw1: torch.Tensor
    eb0: torch.Tensor
    eb1: torch.Tensor
    ew0: torch.Tensor
    ew1: torch.Tensor
    n_r: int
    n_eta: int
    n_pixels_y: int
    n_pixels_z: int
    trans_opt: list = None        # type: ignore[assignment]

    @classmethod
    def from_spec(cls, spec: IntegrationSpec) -> "SoftBinGeometry":
        """Precompute the soft-bin geometry from an :class:`IntegrationSpec`.

        Pure torch — no numba, no v1 dependency. Differentiable in every
        refinable spec parameter.
        """
        spec.validate()
        R, Eta = eval_pixel_REta(spec)
        R_flat = R.reshape(-1)
        Eta_flat = Eta.reshape(-1)
        rb0, rb1, rw0, rw1 = soft_bin_indices_weights(
            R_flat, R_min=spec.RMin, R_bin_size=spec.RBinSize,
            n_r=spec.n_r_bins,
        )
        eb0, eb1, ew0, ew1 = soft_bin_indices_weights(
            Eta_flat, R_min=spec.EtaMin, R_bin_size=spec.EtaBinSize,
            n_r=spec.n_eta_bins,
        )
        return cls(
            R=R_flat, Eta=Eta_flat,
            rb0=rb0, rb1=rb1, rw0=rw0, rw1=rw1,
            eb0=eb0, eb1=eb1, ew0=ew0, ew1=ew1,
            n_r=spec.n_r_bins, n_eta=spec.n_eta_bins,
            n_pixels_y=spec.NrPixelsY, n_pixels_z=spec.NrPixelsZ,
            trans_opt=list(spec.TransOpt),
        )

    @property
    def device(self) -> torch.device:
        return self.R.device

    @property
    def dtype(self) -> torch.dtype:
        return self.R.dtype


def integrate_soft(
    image: torch.Tensor,
    geom: SoftBinGeometry,
    *,
    apply_trans_opt: bool = True,
) -> torch.Tensor:
    """Integrate ``image`` against a precomputed :class:`SoftBinGeometry`.

    Parameters
    ----------
    image :
        Raw detector image, shape ``(NrPixelsZ, NrPixelsY)``. By default
        ``apply_trans_opt=True`` and the geometry's ``trans_opt`` is
        forward-applied to ``image`` before binning. Set
        ``apply_trans_opt=False`` if you've already pre-transformed the
        image (or know the geometry was built without TransOpt).

    Returns the 2D ``(n_eta, n_r)`` array. Differentiable both in
    ``image`` and (via ``geom.R``/``geom.Eta``) in the underlying spec
    parameters.
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
    img_flat = image.to(dtype=geom.dtype).reshape(-1)
    flat = torch.zeros(geom.n_eta * geom.n_r,
                        dtype=geom.dtype, device=geom.device)
    for ei, ew in ((geom.eb0, geom.ew0), (geom.eb1, geom.ew1)):
        for ri, rw in ((geom.rb0, geom.rw0), (geom.rb1, geom.rw1)):
            idx = ei * geom.n_r + ri
            flat = flat.index_add(0, idx, img_flat * ew * rw)
    return flat.reshape(geom.n_eta, geom.n_r)


def integrate_soft_batch(
    images: torch.Tensor,
    geom: SoftBinGeometry,
    *,
    apply_trans_opt: bool = True,
) -> torch.Tensor:
    """Integrate a batch of images sharing a single geometry.

    ``images`` must have shape ``(n_images, NrPixelsZ, NrPixelsY)``.
    Returns ``(n_images, n_eta, n_r)``. ``apply_trans_opt`` behaves as
    in :func:`integrate_soft`.
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
    out = torch.zeros(n_imgs, geom.n_eta, geom.n_r,
                       dtype=geom.dtype, device=geom.device)
    img_flat = images.to(dtype=geom.dtype).reshape(n_imgs, -1)  # (N, n_pix)
    for ei, ew in ((geom.eb0, geom.ew0), (geom.eb1, geom.ew1)):
        for ri, rw in ((geom.rb0, geom.rw0), (geom.rb1, geom.rw1)):
            idx = ei * geom.n_r + ri                             # (n_pix,)
            # index_add along the bin axis for each image; we vectorise by
            # flattening (n_imgs, n_bins) and using a batched scatter via
            # index_add over the trailing axis.
            contribs = img_flat * (ew * rw).unsqueeze(0)         # (N, n_pix)
            flat_out = out.reshape(n_imgs, -1)
            flat_out = flat_out.index_add(1, idx, contribs)
            out = flat_out.reshape(n_imgs, geom.n_eta, geom.n_r)
    return out


__all__ = ["SoftBinGeometry", "integrate_soft", "integrate_soft_batch"]
