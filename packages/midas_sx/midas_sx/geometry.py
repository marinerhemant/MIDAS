"""Scan and detector geometry for single-crystal rotation data.

The detector model is ``midas_defect.geometry.Geometry`` (the MIDAS-canonical pixel -> lab map:
``Y = -(col - bcy) * px``, ``Z = (row - bcz) * px``, beam along +X, sample rotation
``q_lab = R_z(+omega) q_sample``). Nothing about it is re-implemented here; this module adds only
what a frame series needs on top: the omega of a frame, and spot directions.

omega bookkeeping
-----------------
``Scan.omega_start`` is the ω at the START EDGE of frame 0 and ``omega_step`` the (signed) width of a
frame. A spot's ``frame`` coordinate from ``find_blobs_3d`` is a fractional ARRAY INDEX, so index k is
frame k and its centre is ``omega_start + (k + 0.5) * omega_step``. Half a frame is 0.125 deg at a
0.25 deg step: small, but it is a systematic and a header that records the centre instead of the
edge shifts every hkl residual by it, so say which one yours is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from midas_defect.geometry import Geometry, pixel_to_qlab

__all__ = ["Scan", "make_geometry", "spot_directions", "Geometry"]


@dataclass(frozen=True)
class Scan:
    omega_start: float          # deg, start edge of frame 0
    omega_step: float           # deg per frame, signed
    n_frames: int

    def omega_of(self, frame):
        """ω [deg] at the centre of (fractional) frame index ``frame``."""
        return self.omega_start + (np.asarray(frame, dtype=float) + 0.5) * self.omega_step

    @property
    def omega_end(self) -> float:
        return self.omega_start + self.omega_step * self.n_frames


def make_geometry(*, wavelength_A: float, dist_mm: float, pixel_mm: float,
                  centre_rc: Tuple[float, float], shape: Tuple[int, int], scan: Scan,
                  label: str = "") -> Geometry:
    """A ``midas_defect`` Geometry with no tilts, distortion or wedge.

    ``centre_rc`` is the beam position as (row, col) in pixel coordinates with pixel centres at integers.
    """
    return Geometry(
        lsd_um=dist_mm * 1000.0, bcy_px=float(centre_rc[1]), bcz_px=float(centre_rc[0]),
        px_um=pixel_mm * 1000.0, wavelength_A=wavelength_A,
        n_pix_y=int(shape[1]), n_pix_z=int(shape[0]),
        omega_first_deg=scan.omega_start + 0.5 * scan.omega_step,   # centre of frame 0
        omega_step_deg=scan.omega_step, n_frames=scan.n_frames, label=label)


def spot_directions(row, col, geom: Geometry):
    """Unit diffracted directions and angles for detector positions.

    Returns ``(s, two_theta_deg, eta_deg)`` with ``s`` of shape (N, 3) in the lab frame (beam = +X) and
    η the MIDAS azimuth ``atan2(-s_y, s_z)`` (0 = vertical).
    """
    import torch
    q = pixel_to_qlab(np.atleast_1d(np.asarray(row, float)), np.atleast_1d(np.asarray(col, float)),
                      geom, device="cpu", dtype=torch.float64).detach().cpu().numpy()   # CPU: the default device may be MPS, which has no float64
    k0 = 2.0 * np.pi / geom.wavelength_A
    s = q / k0
    s[:, 0] += 1.0                                  # q = k0 (s - s0), s0 = +X
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    two_theta = np.degrees(np.arccos(np.clip(s[:, 0], -1.0, 1.0)))
    eta = np.degrees(np.arctan2(-s[:, 1], s[:, 2]))
    return s, two_theta, eta
