"""Differentiable per-pixel gain --- track detector response drift.

A static flat-field correction (``binning/mask.py``'s ``flat_field``)
requires the user to acquire and maintain a separate flat-field image
that captures per-pixel sensitivity. Manufacturers ship one with the
detector; beamlines re-acquire periodically. Both go stale: the per-pixel
gain drifts over hours and days from sensor temperature swings, bias
voltage relaxation (CdTe), local heating from sample environments,
beam-induced effects, and module-level ASIC drift.

For soft-matter SAXS in particular, sub-percent gain drift contaminates
the absolute-intensity quantities (form-factor amplitudes, Porod
constants, kinetic rates) that the experiments are trying to measure.
The community works around this by acquiring periodic glassy-carbon
secondary-standard frames as an absolute-intensity reference; this
module turns that ritual into a closed-loop algorithm.

The differentiable gain treats the per-pixel multiplicative correction
as a **learnable parameter**. Each pixel ``i`` has a raw parameter
``r_i``; its applied gain is ``g_i = 1 + s · r_i`` (additive
parametrisation around unity, so the prior ``r_i = 0`` is the obvious
"no correction" default). At integrate time the image is multiplied
element-wise by ``g``.

Training: minimise (a) a calibration data loss against a reference such
as glassy-carbon (``ProfileMSELoss`` against the reference profile)
plus (b) a gain-unity prior pulling each ``g_i`` toward 1.0, plus
optionally (c) a smoothness prior (gain drift is spatially correlated
because it's driven by smooth temperature fields and sensor-bias
gradients), plus optionally (d) a module-block prior reflecting the
fact that Pilatus modules drift block-coherently.

After convergence the gain map is the per-pixel drift signature;
applying it forward to any subsequent frame is the drift correction.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class LearnableGain(nn.Module):
    """Per-pixel learnable multiplicative gain centred on unity.

    The gain is parametrised as ``g_i = 1 + scale · r_i`` where ``r_i``
    is the raw learnable parameter. With ``r_i = 0`` (default
    initialisation) every pixel gets ``g_i = 1`` and the input image
    passes through unchanged. The ``scale`` factor sets the natural
    range over which the optimiser explores; a value of 0.1 means a
    raw parameter of order 1 corresponds to a ``\\pm 10\\%`` gain change,
    which is a generous bound for typical detector drift.

    Parameters
    ----------
    NrPixelsZ, NrPixelsY :
        Detector dimensions.
    scale :
        Multiplier on the raw parameter. Smaller values constrain the
        gain to a tighter range but slow the optimiser; larger values
        let the gain explore more aggressive corrections. Default
        0.1 (i.e. each unit of raw parameter is a 10\\% gain change).
    static_gain :
        Optional fixed per-pixel multiplicative correction (shape
        ``(NrPixelsZ, NrPixelsY)``) applied AFTER the learnable gain.
        Use this to combine a known flat-field with the learnable drift
        component: ``image_corrected = image · g_learnable · g_static``.
    dtype :
        torch dtype of the learnable parameter and returned weights.
    """

    def __init__(
        self,
        NrPixelsZ: int,
        NrPixelsY: int,
        *,
        scale: float = 0.1,
        static_gain: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = float(scale)
        self.NrPixelsZ = int(NrPixelsZ)
        self.NrPixelsY = int(NrPixelsY)
        self.raw = nn.Parameter(
            torch.zeros((NrPixelsZ, NrPixelsY), dtype=dtype)
        )
        if static_gain is not None:
            if static_gain.shape != (NrPixelsZ, NrPixelsY):
                raise ValueError(
                    f"static_gain shape {tuple(static_gain.shape)} does "
                    f"not match ({NrPixelsZ}, {NrPixelsY})"
                )
            self.register_buffer(
                "static_gain", static_gain.to(dtype=dtype)
            )
        else:
            self.static_gain = None

    def forward(self) -> torch.Tensor:
        """Return the per-pixel applied gain ``g_i`` of shape (NZ, NY)."""
        g = 1.0 + self.scale * self.raw
        if self.static_gain is not None:
            g = g * self.static_gain
        return g

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Multiply ``image`` by the learned gain, element-wise.

        ``image`` must have shape ``(NrPixelsZ, NrPixelsY)``.
        """
        if image.shape != (self.NrPixelsZ, self.NrPixelsY):
            raise ValueError(
                f"image shape {tuple(image.shape)} does not match "
                f"({self.NrPixelsZ}, {self.NrPixelsY})"
            )
        return image * self.forward()

    def extract_gain_map(self) -> np.ndarray:
        """Return the current learned gain as a NumPy array (NZ, NY)."""
        with torch.no_grad():
            return self.forward().detach().cpu().numpy()

    def n_drifted_pixels(self, threshold: float = 0.01) -> int:
        """Pixels whose gain has drifted more than ``threshold`` from 1.0."""
        with torch.no_grad():
            return int((self.forward() - 1.0).abs().gt(threshold).sum())


def gain_unity_prior(
    gain: LearnableGain,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    """Quadratic prior pulling each gain toward 1.0.

    ``loss_prior = weight · mean((g - 1)²)``

    With ``weight`` large the gain barely moves; with small ``weight``
    the data loss dominates and pixels are free to drift. Tune so the
    cost of moving a *quiet* pixel away from unity exceeds the
    data-loss reduction it would produce. A reasonable starting point
    when training against a glassy-carbon reference is to size
    ``weight`` so that ``weight · (typical_drift)²`` matches the
    typical per-pixel data-loss residual at unit gain.
    """
    g = gain()
    return weight * ((g - 1.0) ** 2).mean()


def gain_smoothness_prior(
    gain: LearnableGain,
    *,
    weight: float = 1.0,
) -> torch.Tensor:
    """Total-variation prior on the gain map.

    Encodes the physical observation that gain drift is spatially
    smooth: it is driven by temperature gradients across the sensor,
    bias-voltage relaxation that varies smoothly with position, and
    panel-level ASIC drift. Sharp pixel-to-pixel gain jumps are
    unphysical (other than at module edges, addressed by
    :func:`gain_module_block_prior`).
    """
    g = gain()
    dy = (g[:, 1:] - g[:, :-1]).abs().mean()
    dz = (g[1:, :] - g[:-1, :]).abs().mean()
    return weight * (dy + dz)


def gain_module_block_prior(
    gain: LearnableGain,
    *,
    module_shape: Sequence[int],
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalise within-module gain variance.

    Pilatus and similar hybrid pixel detectors are tiled from
    independent ASIC modules; ASIC-level drift moves all pixels in
    a module together. This prior says: gain may differ between
    modules but should be approximately uniform within each.

    ``module_shape = (mod_Z, mod_Y)`` is the per-module pixel count
    (e.g. ``(195, 487)`` for a Pilatus3 module on the Z and Y axes).
    The detector dimensions must be exact multiples of ``module_shape``.

    The penalty is the mean within-module variance of the gain.
    """
    if len(module_shape) != 2:
        raise ValueError(
            f"module_shape must be a 2-tuple, got {module_shape}"
        )
    mZ, mY = int(module_shape[0]), int(module_shape[1])
    g = gain()
    NZ, NY = g.shape
    if NZ % mZ != 0 or NY % mY != 0:
        raise ValueError(
            f"detector ({NZ}, {NY}) is not an exact multiple of "
            f"module_shape ({mZ}, {mY})"
        )
    nZ_blocks, nY_blocks = NZ // mZ, NY // mY
    g_blocked = g.reshape(nZ_blocks, mZ, nY_blocks, mY)
    g_blocked = g_blocked.permute(0, 2, 1, 3).contiguous()
    g_flat = g_blocked.reshape(nZ_blocks * nY_blocks, mZ * mY)
    per_module_var = g_flat.var(dim=1)
    return weight * per_module_var.mean()


__all__ = [
    "LearnableGain",
    "gain_unity_prior",
    "gain_smoothness_prior",
    "gain_module_block_prior",
]
