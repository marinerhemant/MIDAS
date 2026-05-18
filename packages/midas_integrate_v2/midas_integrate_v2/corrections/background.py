"""Background subtraction (empty-cell / dark-baseline) as a torch nn.Module.

Differentiable subtraction of a reference frame ("empty cell" in PDF
parlance, or a baseline dark frame for general powder work) from the
sample frame before integration::

    I_corrected = clip_to_zero(I_sample - alpha · I_empty - offset)

Both ``alpha`` and ``offset`` may be made refinable so a downstream loss
(e.g., low-Q oscillation amplitude) can drive the auto-fit. ``fit_scale``
provides a quick L-BFGS optimisation against the integrated profile's
high-Q residual amplitude — the signature of an under- or over-subtracted
empty cell in PDF data.

This corresponds to Item 1 of the Differentiable Integrate Scope
Expansion plan (foundation for PDF capillary background removal).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class EmptySubtraction(nn.Module):
    """Subtract a scaled empty-cell frame before integration.

    Parameters
    ----------
    empty_image :
        Reference frame, same shape as the sample image. Stored as a
        buffer so it moves with the module across devices/dtypes.
    scale :
        Initial value of the multiplicative scale (``alpha``).
    offset :
        Initial value of the additive baseline offset (counts).
    refinable_scale :
        If True, ``scale`` becomes an ``nn.Parameter`` and gradient
        flows back through it.
    refinable_offset :
        Same for ``offset``.
    clip_negative :
        If True (default), the post-subtraction image is clipped at
        zero. Useful for downstream Poisson variance models which
        require non-negative counts. Disable for diagnostic round-trips.
    dtype :
        Working dtype (default ``torch.float64``).
    """

    def __init__(
        self,
        empty_image: torch.Tensor,
        *,
        scale: float = 1.0,
        offset: float = 0.0,
        refinable_scale: bool = False,
        refinable_offset: bool = False,
        clip_negative: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if empty_image.ndim != 2:
            raise ValueError(
                f"empty_image must be 2-D, got shape {tuple(empty_image.shape)}"
            )
        empty = empty_image.to(dtype=dtype).detach().clone()
        self.register_buffer("empty_image", empty)
        self.clip_negative = bool(clip_negative)

        scale_t = torch.as_tensor(float(scale), dtype=dtype)
        offset_t = torch.as_tensor(float(offset), dtype=dtype)
        if refinable_scale:
            self.scale = nn.Parameter(scale_t)
        else:
            self.register_buffer("scale", scale_t)
        if refinable_offset:
            self.offset = nn.Parameter(offset_t)
        else:
            self.register_buffer("offset", offset_t)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape != self.empty_image.shape:
            raise ValueError(
                f"image shape {tuple(image.shape)} does not match empty "
                f"shape {tuple(self.empty_image.shape)}"
            )
        out = image.to(self.empty_image.dtype) \
            - self.scale * self.empty_image - self.offset
        if self.clip_negative:
            out = torch.clamp(out, min=0.0)
        return out

    @torch.no_grad()
    def fit_scale(
        self,
        image: torch.Tensor,
        q_range: Tuple[float, float],
        spec,
        *,
        n_iter: int = 50,
        lr: float = 0.05,
    ) -> torch.Tensor:
        """Auto-fit ``scale`` by minimising high-Q oscillation amplitude.

        The signature of an under- (or over-) subtracted empty cell is
        a residual oscillation at high Q where the sample structure
        function should be flat (S(Q) → 1). We integrate the
        post-subtraction image to a 1D profile, take a high-Q window,
        and minimise the variance of the de-trended profile via a few
        L-BFGS steps. ``self.scale`` is updated in place.

        ``spec`` is a :class:`midas_integrate_v2.IntegrationSpec`. We
        use the polygon kernel (most accurate) for the auto-fit.
        """
        from ..binning import PolygonBinGeometry, integrate_polygon
        # Local import: avoid circular dep at module load.

        was_param = isinstance(self.scale, nn.Parameter)
        if not was_param:
            scale_value = float(self.scale.detach())
            del self.scale
            self.scale = nn.Parameter(
                torch.as_tensor(scale_value, dtype=self.empty_image.dtype)
            )

        geom = PolygonBinGeometry.from_spec(spec)
        q_lo, q_hi = q_range

        def _high_q_residual_var() -> torch.Tensor:
            sub = self.forward(image)
            int2d = integrate_polygon(sub, geom)
            prof = int2d.mean(dim=0)             # eta-averaged
            R_axis = (
                spec.RMin + (torch.arange(prof.shape[0], device=prof.device,
                                          dtype=prof.dtype) + 0.5)
                * spec.RBinSize
            )
            # R → Q proxy via small-angle approximation; exact mapping
            # not required — we just need a monotonic high-Q window.
            two_theta = torch.atan(R_axis * spec.pxY / spec.Lsd)
            q_proxy = (4 * torch.pi / spec.Wavelength) * torch.sin(0.5 * two_theta)
            mask = (q_proxy >= q_lo) & (q_proxy <= q_hi)
            window = prof[mask]
            if window.numel() < 4:
                return torch.tensor(0.0, dtype=prof.dtype, device=prof.device)
            return window.var(unbiased=False)

        with torch.enable_grad():
            opt = torch.optim.LBFGS(
                [self.scale], lr=lr, max_iter=n_iter, line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                loss = _high_q_residual_var()
                loss.backward()
                return loss

            opt.step(closure)

        fitted = self.scale.detach().clone()
        if not was_param:
            del self.scale
            self.register_buffer("scale", fitted)
        return fitted


__all__ = ["EmptySubtraction"]
