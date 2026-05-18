"""Conv NN-augmented residual model — ΔR(y, z) on a coarse grid.

Replaces or augments v1's thin-plate-spline Stage-3 correction with a small
conv network operating on the (y, z) detector grid.  The network output is
sampled at fitted-point pixel coordinates via bilinear interpolation, so
gradients flow cleanly to both the conv weights and the geometry parameters
(via the sampling coordinate, which itself depends on geometry through any
panel transforms).

Architecture (conv per the user's request):

    Input:  [1, 1, H_grid, W_grid]   coordinate-encoded ΔR map.
    Trunk:  3× Conv2d → ReLU, kernel 3, channels 8 → 16 → 8.
    Head:   Conv2d 8 → 1, kernel 1.
    Output: [H_grid, W_grid]          ΔR field in pixels (or μm — caller's
                                       choice via output scaling).

Sampling:
    Bilinear interpolation at each (y_pix, z_pix) using torch.nn.functional
    grid_sample.  Coordinates are normalised to [-1, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NNResidualConfig:
    grid_H: int = 64       # ΔR grid height
    grid_W: int = 64
    channels: tuple = (8, 16, 8)
    kernel: int = 3
    output_scale: float = 1.0     # px or μm; the network output is multiplied
                                   # by this and added to R_corr
    detector_H_px: int = 0         # used to normalise coords; set by caller
    detector_W_px: int = 0


class ResidualConvNet(nn.Module):
    """Small conv network producing a coarse ΔR map.

    The map is upsampled to detector resolution implicitly via bilinear
    sampling at the fitted-point coordinates.
    """

    def __init__(self, config: NNResidualConfig):
        super().__init__()
        self.config = config
        # Learnable coarse ΔR field.  Initialised to small Gaussian noise
        # rather than zeros — a zero seed combined with zero-init conv
        # weights and ReLU produces identically-zero output and
        # identically-zero gradients, leaving the network stuck.  The
        # noise scale (1e-2) is small enough that the initial ΔR is
        # negligible relative to typical residuals (~0.05 px) but large
        # enough that gradients flow.
        self.seed = nn.Parameter(
            torch.randn(1, 1, config.grid_H, config.grid_W,
                        dtype=torch.float64) * 1e-2
        )
        layers = []
        c_in = 1
        for c_out in config.channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=config.kernel,
                                     padding=config.kernel // 2,
                                     dtype=torch.float64))
            # LeakyReLU instead of ReLU so gradients survive the small-
            # init regime (a dead ReLU at zero is a fixed point of Adam).
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            c_in = c_out
        layers.append(nn.Conv2d(c_in, 1, kernel_size=1, dtype=torch.float64))
        self.trunk = nn.Sequential(*layers)
        self._init_weights_small()

    def _init_weights_small(self) -> None:
        """Kaiming-normal init scaled to keep the initial output small.

        We want the network to start near zero (so the analytical basis
        fits first) without freezing gradient flow.  Kaiming-normal
        with a 0.5× factor keeps activations well-scaled while leaving
        the output ≪ typical residual.  Biases get a tiny positive
        offset so LeakyReLU pre-activations are not all-zero at init.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu",
                                          a=0.1)
                with torch.no_grad():
                    m.weight.mul_(0.5)
                    if m.bias is not None:
                        nn.init.uniform_(m.bias, 0.0, 1e-3)

    def field(self) -> torch.Tensor:
        """Return the [grid_H, grid_W] ΔR field."""
        out = self.trunk(self.seed)            # [1, 1, grid_H, grid_W]
        return out[0, 0] * float(self.config.output_scale)

    def sample(self, Y_pix: torch.Tensor, Z_pix: torch.Tensor) -> torch.Tensor:
        """Bilinear-sample the field at the supplied pixel coordinates.

        Returns ΔR per fitted point.  Coordinates are normalised to [-1, 1]
        using the detector pixel extent (config.detector_*_px).
        """
        H = max(self.config.detector_H_px, 1)
        W = max(self.config.detector_W_px, 1)
        # grid_sample expects (N, 1, Hg, Wg) input and (N, Hp, Wp, 2) grid.
        field = self.field().unsqueeze(0).unsqueeze(0)   # [1, 1, Hg, Wg]
        # Build a [1, 1, Npts, 2] sampling grid (treat as a 1D line for one batch).
        # grid_sample uses (x, y) where x corresponds to W and y to H, and the
        # axis convention flips (Z_pix is the "horizontal" direction for our
        # detector layout, Y_pix the "vertical").
        grid_x = (Z_pix.double() / (W - 1)) * 2.0 - 1.0
        grid_y = (Y_pix.double() / (H - 1)) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, 1, -1, 2)
        sampled = F.grid_sample(field, grid, mode="bilinear",
                                 padding_mode="border", align_corners=True)
        return sampled.reshape(-1)


__all__ = ["NNResidualConfig", "ResidualConvNet"]
