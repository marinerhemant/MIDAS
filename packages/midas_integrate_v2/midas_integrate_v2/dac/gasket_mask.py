"""DAC gasket mask + per-ring η-coverage diagnostic.

For diamond-anvil-cell powder data the gasket geometry imposes an
η-window around the cell axis where signal can reach the detector.
Conventional 2-fold-symmetric DAC: open wedges around η = 0° and 180°
(or 90° and 270°, depending on the cell orientation).

API
---

- :func:`build_gasket_mask` returns a boolean detector mask: True =
  signal allowed, False = gasket-blocked. Compatible with
  :class:`~midas_integrate_v2.LearnableMask` and any pixel-wise weight.
- :func:`eta_coverage_per_ring` returns the fraction of η visible per
  ring; used by the CLI warning (Item 22) and by
  :func:`build_provenance` (Item 6).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def _wrap_eta_deg(eta: np.ndarray) -> np.ndarray:
    """Wrap into ``[-180, 180)``."""
    return ((eta + 180.0) % 360.0) - 180.0


def build_gasket_mask(
    NrPixelsY: int,
    NrPixelsZ: int,
    *,
    BC: Tuple[float, float],
    eta_open_deg: Tuple[float, float],
    symmetry: str = "two_fold",
    soft_edge_px: float = 0.0,
) -> np.ndarray:
    """Boolean detector mask for a DAC gasket geometry.

    Parameters
    ----------
    NrPixelsY, NrPixelsZ :
        Detector size in pixels (Y = horizontal, Z = vertical).
    BC :
        Beam centre as ``(BC_y, BC_z)`` in pixel coordinates.
    eta_open_deg :
        ``(eta_min, eta_max)`` of the *primary* open wedge in degrees,
        relative to detector +Y axis. The companion wedge (and any
        symmetric copies) are added automatically based on ``symmetry``.
    symmetry :
        - ``"single"``: only the primary wedge is open.
        - ``"two_fold"`` (default): primary + 180°-rotated copy.
        - ``"four_fold"``: primary + 3 90°-rotated copies (cubic
          gaskets / rare).
    soft_edge_px :
        If > 0, the mask is a smooth ramp from 1 to 0 over this many
        pixels at each wedge boundary (currently emitted as bool with
        the soft edge bands set to True for now — placeholder for a
        future float-mask path; users that want a soft mask should
        consume :class:`midas_integrate_v2.LearnableMask` directly).

    Returns
    -------
    mask : np.ndarray of shape ``(NrPixelsZ, NrPixelsY)``, dtype bool.
        True where signal is allowed; False where gasket blocks.
    """
    if symmetry not in ("single", "two_fold", "four_fold"):
        raise ValueError(f"symmetry must be 'single' / 'two_fold' / "
                          f"'four_fold', got {symmetry!r}")
    eta_min, eta_max = eta_open_deg
    BC_y, BC_z = BC
    yy = np.arange(NrPixelsY, dtype=np.float64)
    zz = np.arange(NrPixelsZ, dtype=np.float64)
    Z, Y = np.meshgrid(zz, yy, indexing="ij")
    dy = Y - BC_y
    dz = Z - BC_z
    eta_pix = np.degrees(np.arctan2(dz, dy))
    eta_pix = _wrap_eta_deg(eta_pix)

    def _in_wedge(eta_lo: float, eta_hi: float) -> np.ndarray:
        raw_width = float(eta_hi) - float(eta_lo)
        if raw_width >= 360.0 - 1e-9:
            return np.ones_like(eta_pix, dtype=bool)
        # Wraparound-safe: rotate eta so wedge starts at 0
        rotated = _wrap_eta_deg(eta_pix - eta_lo)
        width = raw_width % 360.0
        # rotated is in [-180, 180); shift to [0, 360)
        rotated_pos = (rotated + 360.0) % 360.0
        return rotated_pos <= width

    mask = _in_wedge(eta_min, eta_max)
    if symmetry in ("two_fold", "four_fold"):
        mask |= _in_wedge(eta_min + 180.0, eta_max + 180.0)
    if symmetry == "four_fold":
        mask |= _in_wedge(eta_min + 90.0, eta_max + 90.0)
        mask |= _in_wedge(eta_min + 270.0, eta_max + 270.0)
    # soft_edge_px: kept as bool for now — the LearnableMask gives a
    # full differentiable path. Note included in mask metadata.
    return mask.astype(bool)


def eta_coverage_per_ring(
    spec,
    mask: np.ndarray,
    ring_R_px: torch.Tensor | np.ndarray,
    *,
    capture_radius_px: float = 3.0,
) -> torch.Tensor:
    """Fraction of η in [0, 360) where ``mask`` is True per ring.

    Parameters
    ----------
    spec :
        :class:`~midas_integrate_v2.IntegrationSpec`. Used for BC and
        the eta-binning (n_eta_bins).
    mask :
        Detector mask shape ``(NrPixelsZ, NrPixelsY)`` (the output of
        :func:`build_gasket_mask`, or any user-supplied detector mask).
    ring_R_px :
        ``(n_rings,)`` tensor / array of ring radii in pixels.
    capture_radius_px :
        Pixels closer than this to a ring centre are counted toward
        the ring's η-histogram.

    Returns
    -------
    coverage : torch.Tensor, shape ``(n_rings,)``, dtype fp64.
        Each entry in ``[0, 1]``.
    """
    rings = np.asarray(ring_R_px, dtype=np.float64).reshape(-1)
    NrZ, NrY = mask.shape
    yy = np.arange(NrY, dtype=np.float64)
    zz = np.arange(NrZ, dtype=np.float64)
    Z, Y = np.meshgrid(zz, yy, indexing="ij")
    BC_y = float(spec.BC_y) if hasattr(spec.BC_y, "__float__") else float(spec.BC_y.detach())
    BC_z = float(spec.BC_z) if hasattr(spec.BC_z, "__float__") else float(spec.BC_z.detach())
    dy = Y - BC_y
    dz = Z - BC_z
    R = np.sqrt(dy * dy + dz * dz)
    eta = _wrap_eta_deg(np.degrees(np.arctan2(dz, dy)))

    coverage = np.zeros(rings.shape[0], dtype=np.float64)
    n_eta_bins = max(int(getattr(spec, "n_eta_bins", 0) or 0), 36)
    for k, R_ring in enumerate(rings):
        ring_pixels = np.abs(R - R_ring) <= capture_radius_px
        if not ring_pixels.any():
            coverage[k] = 0.0
            continue
        # Histogram η along the ring
        hist_open, _ = np.histogram(
            eta[ring_pixels & mask],
            bins=n_eta_bins, range=(-180.0, 180.0),
        )
        hist_total, _ = np.histogram(
            eta[ring_pixels],
            bins=n_eta_bins, range=(-180.0, 180.0),
        )
        # Fraction of η-bins where any open ring-pixel exists vs total
        open_bins = (hist_open > 0).sum()
        total_bins = (hist_total > 0).sum()
        coverage[k] = (open_bins / total_bins) if total_bins > 0 else 0.0
    return torch.as_tensor(coverage, dtype=torch.float64)


__all__ = ["build_gasket_mask", "eta_coverage_per_ring"]
