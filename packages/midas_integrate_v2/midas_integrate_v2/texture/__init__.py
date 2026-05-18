"""Texture / pole-figure helpers built on the cake (η, R) array.

The cake captures a single sample orientation; for a full pole figure
the user must rotate the sample (χ, φ rotations) and re-integrate. This
module emits per-rotation pole-figure intensity slices and lets MTEX /
POPLA pick up the result.
"""
from .pole_figure import (
    cake_to_pole_figure,
    write_popla_pol,
)

__all__ = ["cake_to_pole_figure", "write_popla_pol"]
