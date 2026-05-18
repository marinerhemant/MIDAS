"""Pixel-index → physical (Yc, Zc) centroid mapping for different pixel lattices.

The forward model in :mod:`forward.geometry` and the binning geometries in
``midas_integrate_v2`` previously assumed a regular Cartesian pixel grid:

    Yc = (-Y_pix + BC_y) * pxY
    Zc = (Z_pix  - BC_z) * pxZ

This module factors that mapping out so non-Cartesian lattices (PIXIRAD-style
hexagonal-honeycomb arrays) can be plugged in without changing the
downstream tilt / projection / distortion code.

Supported lattices
------------------

``"cartesian"`` (default)
    The historical formula above.  Bit-identical to the pre-refactor code
    path.

``"hex_offset_y"``
    Regular hexagonal lattice with the apothem axis aligned with detector
    Y and odd-Z rows shifted by ``+apothem`` along Y.  Pitch along Y is
    ``2·apothem``; pitch along Z is ``apothem·√3``.  This is the PIXIRAD-1
    / PIXIRAD-8 geometry (apothem = 30 μm).

All outputs are torch tensors carrying autograd through the lattice scalars
(``apothem`` is refinable; ``pxY``/``pxZ`` enter only in the cartesian
branch).
"""
from __future__ import annotations

import math
from typing import Optional

import torch


SQRT3 = math.sqrt(3.0)
_DEG2RAD = 0.017453292519943295

# Canonical lattice names.  Add new ones here and dispatch in
# :func:`lattice_to_phys`.
LATTICES = ("cartesian", "hex_offset_y")


def _row_parity(Z_pix: torch.Tensor) -> torch.Tensor:
    """Return 0.0 / 1.0 row parity tolerant of float Z_pix inputs.

    Subpixel samplers pass ``Z + dz`` with dz∈(-0.5, +0.5); the parity is
    defined by the integer pixel the sample sits in, so we floor first.
    """
    return (torch.floor(Z_pix).to(torch.long) % 2).to(Z_pix.dtype)


def lattice_to_phys(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    *,
    lattice: str,
    BC_y: torch.Tensor,
    BC_z: torch.Tensor,
    pxY: Optional[torch.Tensor] = None,
    pxZ: Optional[torch.Tensor] = None,
    apothem: Optional[torch.Tensor] = None,
    orientation_deg: Optional[torch.Tensor] = None,
):
    """Map integer / float pixel indices to physical (Yc, Zc) in μm.

    Parameters
    ----------
    Y_pix, Z_pix : pixel-index tensors (any broadcastable shape).
    lattice : one of :data:`LATTICES`.
    BC_y, BC_z : beam-centre in pixel-index coordinates.
    pxY, pxZ : pixel pitch (μm).  Required for ``cartesian``.
    apothem : hex apothem (μm).  Required for ``hex_offset_y``.
    orientation_deg : optional small in-plane rotation of the lattice axes
        relative to the detector (Y, Z) frame.  Applied after the centroid
        mapping.

    Returns
    -------
    (Yc, Zc) : tensors of physical coordinates (μm).
    """
    if lattice == "cartesian":
        if pxY is None or pxZ is None:
            raise ValueError("cartesian lattice requires pxY and pxZ")
        Yc = (-Y_pix + BC_y) * pxY
        Zc = (Z_pix - BC_z) * pxZ
        if orientation_deg is not None and bool((orientation_deg != 0).any()):
            Yc, Zc = _apply_orientation(Yc, Zc, orientation_deg)
        return Yc, Zc

    if lattice == "hex_offset_y":
        if apothem is None:
            raise ValueError("hex_offset_y lattice requires apothem")
        a = apothem
        # Row-parity offset: odd Z rows are shifted by +a in the Y direction.
        # Multiply by 0.5 because the column pitch in Y is 2a — i.e. the
        # half-column shift expressed in column-index units is 0.5.
        parity = _row_parity(Z_pix)
        Yc = (BC_y - Y_pix - 0.5 * parity) * (2.0 * a)
        Zc = (Z_pix - BC_z) * (a * SQRT3)
        if orientation_deg is not None and bool((orientation_deg != 0).any()):
            Yc, Zc = _apply_orientation(Yc, Zc, orientation_deg)
        return Yc, Zc

    raise ValueError(
        f"Unknown lattice {lattice!r}; supported: {LATTICES}"
    )


def _apply_orientation(Yc: torch.Tensor, Zc: torch.Tensor,
                       orientation_deg: torch.Tensor):
    c = torch.cos(orientation_deg * _DEG2RAD)
    s = torch.sin(orientation_deg * _DEG2RAD)
    return c * Yc - s * Zc, s * Yc + c * Zc


def hex_pixel_pitch(apothem: torch.Tensor):
    """Return the (pxY_equiv, pxZ_equiv) pitch in μm for a hex lattice.

    Used wherever downstream code wants to plug "the pixel size" into a
    Cartesian formula (e.g. ``px_mean`` in the radial conversion).  The
    convention matches :func:`lattice_to_phys`: the Y axis is the apothem
    axis with pitch ``2a``, Z has pitch ``a√3``.
    """
    return 2.0 * apothem, apothem * SQRT3


def hex_pixel_area(apothem: torch.Tensor):
    """Geometric area of a single hex pixel cell, in μm²."""
    return 2.0 * SQRT3 * apothem * apothem


def hex_cell_vertices(apothem: torch.Tensor) -> torch.Tensor:
    """Return the 6 vertex offsets of a hex pixel cell, in μm, ordered
    counter-clockwise starting at the +Y vertex.

    The cell is the regular hexagon with apothem ``a`` whose flat edges
    are perpendicular to the Y axis (matching :data:`LATTICES` =
    ``hex_offset_y``).  Returned shape ``(6, 2)``.
    """
    a = float(apothem) if isinstance(apothem, torch.Tensor) else float(apothem)
    s = a * 2.0 / SQRT3                      # circumradius
    half_s = s * 0.5
    # +Y vertex, rotating clockwise around (0, 0) in (Y, Z) units
    return torch.tensor([
        ( a,        half_s),
        ( 0.0,      s),
        (-a,        half_s),
        (-a,       -half_s),
        ( 0.0,     -s),
        ( a,       -half_s),
    ], dtype=torch.float64)


__all__ = [
    "LATTICES",
    "SQRT3",
    "lattice_to_phys",
    "hex_pixel_pitch",
    "hex_pixel_area",
    "hex_cell_vertices",
]
