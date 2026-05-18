"""pyFAI ↔ midas-integrate-v2 coordinate conversion helpers.

The single most common bug when moving between pyFAI and MIDAS is the
**0.5 px BC ↔ PONI shift**:

- pyFAI uses **pixel-corner** indexing — pixel ``(0, 0)`` is the
  *corner* of the first pixel; the centre of pixel ``(i, j)`` is at
  position ``(i + 0.5, j + 0.5)``.
- MIDAS uses **pixel-centre** indexing — pixel ``(0, 0)`` IS the
  centre of the first pixel.

So the same physical beam-impact point is recorded as different
numbers in the two packages:

    poni = (BC + 0.5) * pixel_size

Skipping the ``+ 0.5`` gives a calibration that's off by half a pixel
in the beam centre — small but systematic, big enough to shift Bragg
peaks at high R and to break apparent vs. true d-spacings at the
per-mille level.

These helpers do the conversion correctly so you don't have to
remember the sign each time.
"""
from __future__ import annotations

from typing import Tuple


def bc_to_poni(
    BC_y_px: float, BC_z_px: float,
    pxY_um: float, pxZ_um: float,
) -> Tuple[float, float]:
    """Convert MIDAS BC (pixel-centre, px) to pyFAI PONI (pixel-corner, m).

    Returns ``(poni1_m, poni2_m)`` in metres.
    """
    poni1_m = (BC_y_px + 0.5) * pxY_um * 1e-6
    poni2_m = (BC_z_px + 0.5) * pxZ_um * 1e-6
    return poni1_m, poni2_m


def poni_to_bc(
    poni1_m: float, poni2_m: float,
    pxY_um: float, pxZ_um: float,
) -> Tuple[float, float]:
    """Convert pyFAI PONI (pixel-corner, m) to MIDAS BC (pixel-centre, px).

    Returns ``(BC_y_px, BC_z_px)`` in pixels.
    """
    BC_y_px = poni1_m / (pxY_um * 1e-6) - 0.5
    BC_z_px = poni2_m / (pxZ_um * 1e-6) - 0.5
    return BC_y_px, BC_z_px


def make_pyfai_integrator(spec, *, ImportError_msg: bool = True):
    """Build a pyFAI ``AzimuthalIntegrator`` with the correct BC ↔ PONI
    conversion applied automatically.

    Returns the pyFAI integrator object; raises ``ImportError`` if
    pyFAI isn't installed.

    Use this when comparing v2 results to a pyFAI baseline — guarantees
    you don't drop the 0.5 px shift.
    """
    try:
        import pyFAI
    except ImportError as e:
        raise ImportError(
            "pyFAI not installed. pip install pyFAI"
        ) from e
    poni1_m, poni2_m = bc_to_poni(
        float(spec.BC_y), float(spec.BC_z),
        spec.pxY, spec.pxZ,
    )
    return pyFAI.AzimuthalIntegrator(
        dist=float(spec.Lsd) * 1e-6,
        poni1=poni1_m, poni2=poni2_m,
        pixel1=spec.pxY * 1e-6, pixel2=spec.pxZ * 1e-6,
        wavelength=float(spec.Wavelength) * 1e-10,
    )


__all__ = ["bc_to_poni", "poni_to_bc", "make_pyfai_integrator"]
