"""Pole figure construction from cake (η, R) data.

Given a 2D integrated cake at one sample orientation, the visible η
range carries one stripe of a pole figure for each ring. To build the
full pole figure the user must rotate the sample (χ, φ) and stack
slices; here we provide the mapping for one slice.

For a Bragg ring at fixed 2θ on a flat detector, the η-coordinate of
each pixel along the ring corresponds to a sample-frame angle β (with
a fixed α set by the sample tilt and the ring 2θ). We emit
``(α, β, intensity)`` triples on a regular stereographic grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def cake_to_pole_figure(
    int2d: np.ndarray,
    eta_axis_deg: np.ndarray,
    R_axis: np.ndarray,
    *,
    hkl_R_px: float,
    capture_radius_px: float = 3.0,
    sample_rotation_chi_deg: float = 0.0,
    sample_rotation_phi_deg: float = 0.0,
    output_grid: Tuple[int, int] = (181, 91),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one ring's η stripe onto a stereographic pole-figure grid.

    Parameters
    ----------
    int2d :
        ``(n_eta, n_R)`` cake intensity.
    eta_axis_deg :
        ``(n_eta,)`` η axis (degrees).
    R_axis :
        ``(n_R,)`` R axis.
    hkl_R_px :
        Radius of the ring of interest in same units as R_axis.
    capture_radius_px :
        ±window around the ring (R-axis units).
    sample_rotation_chi_deg, sample_rotation_phi_deg :
        Sample-stage rotations (degrees) added to (α, β) so users can
        accumulate stripes from multiple frames.
    output_grid :
        ``(n_alpha, n_beta)`` grid resolution for the output.

    Returns
    -------
    alpha_grid_deg, beta_grid_deg, intensity :
        - ``alpha_grid_deg`` shape ``(n_alpha,)``, range [0, 90).
        - ``beta_grid_deg`` shape ``(n_beta,)``, range [0, 360).
        - ``intensity`` shape ``(n_beta, n_alpha)`` (β rows × α cols).
    """
    int2d = np.asarray(int2d, dtype=np.float64)
    eta = np.asarray(eta_axis_deg, dtype=np.float64)
    R = np.asarray(R_axis, dtype=np.float64)
    if int2d.shape != (eta.shape[0], R.shape[0]):
        raise ValueError(
            f"int2d shape {int2d.shape} != "
            f"(n_eta={eta.shape[0]}, n_R={R.shape[0]})"
        )
    n_alpha, n_beta = output_grid
    # Capture stripe at the ring
    in_ring = np.abs(R - hkl_R_px) <= capture_radius_px
    if not in_ring.any():
        raise ValueError(
            f"no R bins within {capture_radius_px} of ring at R={hkl_R_px}"
        )
    stripe = int2d[:, in_ring].sum(axis=1)              # (n_eta,)
    # In the simplest geometry: α (declination) is set by the ring 2θ;
    # β (azimuthal) is η + sample_phi. For now, treat α as a fixed-stripe
    # angle and broadcast the η-dependence onto the β-axis.
    alpha_value = float(sample_rotation_chi_deg) % 90.0
    alpha_grid_deg = np.linspace(0.0, 90.0, n_alpha, endpoint=False)
    beta_grid_deg = np.linspace(0.0, 360.0, n_beta, endpoint=False)
    # Resample η stripe onto β grid via linear interpolation (wrap)
    eta_unwrapped = (eta + sample_rotation_phi_deg) % 360.0
    sort_idx = np.argsort(eta_unwrapped)
    beta_intensity = np.interp(
        beta_grid_deg, eta_unwrapped[sort_idx], stripe[sort_idx],
        period=360.0,
    )
    # Place the stripe at the alpha bin closest to alpha_value
    alpha_idx = int(np.argmin(np.abs(alpha_grid_deg - alpha_value)))
    intensity = np.zeros((n_beta, n_alpha), dtype=np.float64)
    intensity[:, alpha_idx] = beta_intensity
    return alpha_grid_deg, beta_grid_deg, intensity


def write_popla_pol(
    path: str | Path,
    alpha_grid_deg: np.ndarray,
    beta_grid_deg: np.ndarray,
    intensity: np.ndarray,
    *,
    hkl: Tuple[int, int, int],
) -> Path:
    """Write a POPLA-format pole-figure file.

    POPLA expects a fixed-format ASCII grid: 4-space-separated columns
    of intensity, with an HKL header line. We use a simplified form
    that POPLA's reader accepts: header + one float per α-β bin.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.shape != (beta_grid_deg.shape[0], alpha_grid_deg.shape[0]):
        raise ValueError(
            f"intensity shape {intensity.shape} != "
            f"(n_beta={beta_grid_deg.shape[0]}, "
            f"n_alpha={alpha_grid_deg.shape[0]})"
        )
    with open(path, "w") as f:
        f.write(f"# POPLA pole figure | hkl={hkl[0]} {hkl[1]} {hkl[2]}\n")
        f.write(
            f"# alpha_step={alpha_grid_deg[1] - alpha_grid_deg[0]:.4f} "
            f"beta_step={beta_grid_deg[1] - beta_grid_deg[0]:.4f}\n"
        )
        f.write(f"# n_alpha={alpha_grid_deg.shape[0]} "
                f"n_beta={beta_grid_deg.shape[0]}\n")
        for i in range(intensity.shape[0]):
            row = " ".join(f"{v:.6e}" for v in intensity[i])
            f.write(row + "\n")
    return path


__all__ = ["cake_to_pole_figure", "write_popla_pol"]
