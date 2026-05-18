"""Empirical residual correction map (port of v1 C ``dg_residual_corr_lookup``).

After the harmonic distortion model is fitted, residual systematic
discrepancies between observed and predicted ring positions can be absorbed
into a smooth 2D ``ΔR(Y_pix, Z_pix)`` correction map.  v1 (C) does this via
a thin-plate-spline RBF fitted in AutoCalibrate and applied at runtime by
``dg_residual_corr_lookup``.  This module is the differentiable v2 port.

Two pieces:
  - :func:`residual_corr_lookup` — autograd-clean bilinear interpolation of
    the map at (Y, Z) pixel coordinates, returning ΔR in pixels.  Uses
    ``torch.nn.functional.grid_sample`` so the lookup is differentiable in
    BOTH the query coordinates and the map values.
  - :func:`build_residual_corr_map` — one-shot RBF fit + dense-grid
    evaluation (scipy, not autograd; the map is built once from data).

The map is stored as a 2D tensor of shape ``[NrPixelsZ, NrPixelsY]`` with
the same row-major layout as v1 C (``map[z * Ny + y]``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================ lookup

def residual_corr_lookup(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    corr_map: torch.Tensor,            # [Nz, Ny], ΔR in pixels
) -> torch.Tensor:
    """Bilinear ΔR lookup at the given pixel coordinates.

    Returns
    -------
    torch.Tensor
        Per-pixel ΔR in pixels, broadcastable to ``Y_pix`` / ``Z_pix``.
        Gradient flows through both ``Y_pix``/``Z_pix`` and ``corr_map``.

    Notes
    -----
    Out-of-bounds pixels are clamped to the edge (``padding_mode='border'``),
    matching the half-pixel clamp v1 C uses.
    """
    if corr_map is None:
        return torch.zeros_like(Y_pix)
    Nz, Ny = corr_map.shape
    if Ny < 2 or Nz < 2:
        raise ValueError(f"corr_map too small: shape={corr_map.shape}")
    # grid_sample input: (B, C, H, W) where H=Nz (rows), W=Ny (cols).
    # grid shape: (B, H_out, W_out, 2) with values in [-1, 1]; last dim is
    # (x_W, y_H) — x indexes the W=Ny axis, y indexes the H=Nz axis.
    orig_shape = Y_pix.shape
    y_flat = Y_pix.reshape(-1).to(dtype=corr_map.dtype, device=corr_map.device)
    z_flat = Z_pix.reshape(-1).to(dtype=corr_map.dtype, device=corr_map.device)
    # align_corners=True maps pixel index 0 -> -1 and Ny-1 -> +1, matching
    # v1's integer-pixel indexing semantics.
    x_norm = 2.0 * y_flat.clamp(0.0, Ny - 1.0) / (Ny - 1) - 1.0
    z_norm = 2.0 * z_flat.clamp(0.0, Nz - 1.0) / (Nz - 1) - 1.0
    grid = torch.stack([x_norm, z_norm], dim=-1).view(1, -1, 1, 2)
    inp = corr_map.unsqueeze(0).unsqueeze(0)   # [1, 1, Nz, Ny]
    out = F.grid_sample(inp, grid, mode="bilinear", align_corners=True,
                         padding_mode="border")
    return out.view(orig_shape)


# ============================================================ builder

def build_residual_corr_map(
    Y_pix: torch.Tensor,                # [N], non-outlier fit positions
    Z_pix: torch.Tensor,                # [N]
    delta_R_um: torch.Tensor,           # [N], RadFit - IdealR in micrometres
    *,
    NrPixelsY: int,
    NrPixelsZ: int,
    pxY: float,
    smoothing: Optional[float] = None,
    grid_size: int = 200,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Fit a thin-plate-spline RBF to per-fit radial residuals and evaluate
    on the full detector grid.

    Mirrors :func:`AutoCalibrateZarr.generate_residual_correction_map` so the
    resulting map can be substituted for v1's binary file.  The sign convention
    matches v1: the returned map is **negated** (correction subtracted from
    ``Rt`` brings observed closer to ideal, but the lookup *adds* it; storing
    ``-ΔR/px`` makes addition the right operation).

    Parameters
    ----------
    Y_pix, Z_pix : 1-D tensors of post-outlier-filter fit positions.
    delta_R_um : per-fit radial residual ``RadFit - IdealR`` in **micrometres**
        (matches the ``DeltaR`` column of v1's corr.csv).
    NrPixelsY, NrPixelsZ : detector dimensions.
    pxY : pixel pitch in micrometres (for the µm → px conversion).
    smoothing : RBF smoothing.  ``None`` → ``max(1, N * 0.001)`` (matches v1).
    grid_size : RBF eval grid size before upsampling to full detector.
    """
    try:
        from scipy.interpolate import RBFInterpolator
        from scipy.ndimage import zoom as _zoom
    except ImportError as e:
        raise ImportError(
            "build_residual_corr_map requires scipy.interpolate.RBFInterpolator"
        ) from e

    y_np = Y_pix.detach().cpu().numpy().astype(np.float64)
    z_np = Z_pix.detach().cpu().numpy().astype(np.float64)
    dR_np = delta_R_um.detach().cpu().numpy().astype(np.float64)
    if y_np.ndim != 1 or y_np.size != z_np.size or y_np.size != dR_np.size:
        raise ValueError("Y_pix, Z_pix, delta_R_um must be 1-D arrays of equal length")
    if y_np.size < 50:
        raise ValueError(f"need >=50 fit points to build residual corr map; got {y_np.size}")

    # Negate ΔR (px) so lookup-add brings observed toward ideal (v1 convention).
    dR_px = -dR_np / float(pxY)

    # Normalize coords for RBF stability.
    y_mean = float(np.mean(y_np)); y_std = max(float(np.std(y_np)), 1.0)
    z_mean = float(np.mean(z_np)); z_std = max(float(np.std(z_np)), 1.0)
    coords = np.column_stack([(y_np - y_mean) / y_std, (z_np - z_mean) / z_std])

    if smoothing is None:
        smoothing = max(1.0, len(dR_px) * 0.001)
    rbf = RBFInterpolator(coords, dR_px, kernel="thin_plate_spline",
                           smoothing=float(smoothing))

    # Coarse grid, then upsample to full detector.
    yg = np.linspace(0.0, float(NrPixelsY), grid_size)
    zg = np.linspace(0.0, float(NrPixelsZ), grid_size)
    YG, ZG = np.meshgrid(yg, zg)
    g = np.column_stack([(YG.ravel() - y_mean) / y_std,
                         (ZG.ravel() - z_mean) / z_std])
    coarse = rbf(g).reshape(grid_size, grid_size)
    full = _zoom(coarse, (NrPixelsZ / grid_size, NrPixelsY / grid_size), order=1)
    full = full[:NrPixelsZ, :NrPixelsY]
    return torch.as_tensor(full, dtype=dtype)


def save_residual_corr_bin(corr_map: torch.Tensor, path) -> None:
    """Write a residual map to v1-compatible raw binary.

    Layout matches v1 C ``DGResidualCorr.map``: ``NrPixelsY * NrPixelsZ``
    little-endian ``float64`` values in ``map[z * Ny + y]`` order.  The same
    file is consumed by :func:`midas_integrate.residual_corr.load_residual_correction_map`
    and by ``CalibrantIntegratorOMP`` via the ``ResidualCorrMapFN`` config key.
    """
    from pathlib import Path
    arr = corr_map.detach().cpu().numpy()
    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    Path(path).write_bytes(arr.tobytes())


def load_residual_corr_bin(path, NrPixelsY: int, NrPixelsZ: int,
                            dtype: torch.dtype = torch.float64,
                            device: str = "cpu") -> torch.Tensor:
    """Load a v1-format residual map binary into a torch tensor.

    Inverse of :func:`save_residual_corr_bin`.
    """
    from pathlib import Path
    raw = Path(path).read_bytes()
    expected = NrPixelsY * NrPixelsZ * 8
    if len(raw) != expected:
        raise ValueError(
            f"residual map size mismatch: got {len(raw)} bytes, "
            f"expected {expected} ({NrPixelsY}x{NrPixelsZ} float64)"
        )
    arr = np.frombuffer(raw, dtype=np.float64).reshape(NrPixelsZ, NrPixelsY)
    return torch.as_tensor(arr.copy(), dtype=dtype, device=device)


__all__ = ["residual_corr_lookup", "build_residual_corr_map",
           "save_residual_corr_bin", "load_residual_corr_bin"]
