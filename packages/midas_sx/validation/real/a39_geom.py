"""Shared geometry helpers for A39 (Pilatus, swung detector), importable by the step scripts. The swing of the detector (2theta = alpha) happens about the omega axis; in the file the
swing direction is along the array's columns or rows (unknown a priori: a hypothesis), the beam hits D tan(alpha) off the perpendicular foot (Beam_xy) in either sense, and the plane
is tilted by +-alpha about the lab axis the swing direction maps to after the dihedral transform (cols' -> Z: tz, rows' -> Y: ty)."""
import numpy as np
from dataclasses import replace
from midas_sx.conventions import transform_rc
from midas_sx.geometry import Scan, make_geometry


def swing_geometry(H, conv, axis, tz_sign, off_sign, scale=1.0, dcentre=(0.0, 0.0)):
    """H: dict of header values (shape, wavelength_A, distance_mm, pixel_mm, angle_start, angle_step, n_frames, extra{beam_xy_px, detector_2theta_deg}).
    axis: 'cols' or 'rows' (swing direction in the FILE array). Returns (Geometry, Scan) in the transformed array."""
    nr, nc = H["shape"]
    alpha = H["extra"]["detector_2theta_deg"]
    bx, by = H["extra"]["beam_xy_px"]                            # (column, row) of the perpendicular foot in the file
    D = H["distance_mm"] * scale
    a = np.radians(alpha) if (tz_sign or off_sign) else 0.0
    shift = D * np.tan(a) / H["pixel_mm"]
    foot = np.array([by, bx], float)                             # (row, col)
    beam = foot + off_sign * shift * (np.array([0.0, 1.0]) if axis == "cols" else np.array([1.0, 0.0]))
    r0, c0 = (float(v[0]) for v in transform_rc(np.array([beam[0]]), np.array([beam[1]]), nr, nc, conv.dihedral))
    # which axis of the transformed array does the swing direction become?
    p0 = np.array([nr / 2.0, nc / 2.0])
    step = np.array([0.0, 1.0]) if axis == "cols" else np.array([1.0, 0.0])
    q0 = np.array([float(v[0]) for v in transform_rc(np.array([p0[0]]), np.array([p0[1]]), nr, nc, conv.dihedral)])
    q1 = np.array([float(v[0]) for v in transform_rc(np.array([p0[0] + step[0]]), np.array([p0[1] + step[1]]), nr, nc, conv.dihedral)])
    to_cols = abs(q1[1] - q0[1]) > abs(q1[0] - q0[0])
    shape = (nr, nc) if conv.dihedral < 4 else (nc, nr)
    scan = Scan(conv.omega_sign * H["angle_start"], conv.omega_sign * H["angle_step"], H["n_frames"])
    g = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=D / np.cos(a), pixel_mm=H["pixel_mm"], centre_rc=(r0 + dcentre[0], c0 + dcentre[1]), shape=shape, scan=scan)
    kw = dict(tz_deg=float(tz_sign * alpha)) if to_cols else dict(ty_deg=float(tz_sign * alpha))
    return replace(g, **kw), scan
