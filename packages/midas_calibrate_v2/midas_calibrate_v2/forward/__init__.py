"""Differentiable forward-model primitives.

Composable, autograd-traced building blocks:

- :mod:`geometry` — pixel→(R,η) projection.
- :mod:`distortion` — analytical harmonic basis (p₀…p₁₄ extensible).
- :mod:`panels` — 5-DOF per-panel rigid body for multi-tile detectors.
- :mod:`parallax` — always-on differentiable parallax (no graph break).
- :mod:`bragg` — 2θ ↔ d-spacing ↔ wavelength.
- :mod:`peak_shape` — area-normalised pseudo-Voigt (TCH).
- :mod:`cake` — differentiable (R, η) cake integration.
- :mod:`nn_residual` — small conv NN ΔR(y, z) augmenter.
"""
from . import (geometry, distortion, panels, parallax, bragg, peak_shape,
                residual_corr, sanity)

__all__ = ["geometry", "distortion", "panels", "parallax", "bragg",
           "peak_shape", "residual_corr", "sanity"]
