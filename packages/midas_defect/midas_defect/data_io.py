"""I/O helpers for midas_defect.

Owns:
  * `load_voxel_npz(path)` — parse the sparse voxel NPZ produced by
    `demk_volume.py` (or any drop-in equivalent) into a `VoxelCloud`
    in the *sample frame*.
  * `VoxelCloud` — dataclass packaging the q-vectors, intensity, and the
    geometry / lattice metadata that came with the file.

The NPZ schema produced by `demk_volume.py`:

    indices              (N, 3) int32     [k_frame, z_bin, y_bin]
    values               (N,)  float32    pooled residual intensity
    pooled_shape         (2,)  int32      (n_z_bins, n_y_bins)
    det_bin              scalar int32     detector pixel binning factor
    threshold            scalar float32   threshold used during dump
    omega_first          scalar float32   ω of frame index 0 (deg)
    omega_step           scalar float32   ω step between frames (deg)
    bcy_px, bcz_px       scalar float32   beam-centre pixels
    px_um                scalar float32   pixel pitch (microns)
    lsd_um               scalar float32   sample-detector distance (microns)
    lambda_a             scalar float32   wavelength (Å)
    layer_start_filenr   scalar int32     starting file number of the layer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import math
import numpy as np

from .geometry import Geometry


__all__ = ["VoxelCloud", "load_voxel_npz", "save_voxel_npz_from_cloud"]


@dataclass
class VoxelCloud:
    """A sparse 3-D voxel cloud in sample-frame reciprocal space."""
    qx: np.ndarray              # (N,) float64
    qy: np.ndarray
    qz: np.ndarray
    intensity: np.ndarray       # (N,) float64
    omega_deg: np.ndarray       # (N,) float64 — ω at which the voxel was observed
    det_y_px: np.ndarray        # (N,) float64 — detector pixel coords (centre of bin)
    det_z_px: np.ndarray
    geometry: Geometry          # the geometry that fed the conversion
    threshold: float            # threshold the producer used when dumping
    layer_start_filenr: int

    def __len__(self) -> int:
        return len(self.qx)

    def n_voxels(self) -> int:
        return len(self.qx)

    @property
    def q_mag(self) -> np.ndarray:
        return np.sqrt(self.qx * self.qx + self.qy * self.qy + self.qz * self.qz)


def load_voxel_npz(path: Union[str, Path]) -> VoxelCloud:
    """Load a sparse voxel NPZ (as produced by `demk_volume.py`).

    Returns
    -------
    VoxelCloud with q-vectors in the SAMPLE frame (rotated by R_z(-ω), the
    MIDAS vertical-axis convention).

    .. note::
       This loader uses an **approximate** pixel→lab map: it applies neither the
       detector tilt nor the radial lens distortion (the NPZ schema does not
       carry those parameters). For the validated, distortion-aware conversion
       that reproduces the published numbers, use the example driver's
       ``voxels_to_qsample`` (which reuses ``apply_tilt_distortion``) or
       ``geometry.pixel_to_qlab`` with a fully-populated `Geometry`.
    """
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    indices  = d["indices"]
    values   = d["values"]
    det_bin  = int(d["det_bin"])
    bcy_px   = float(d["bcy_px"])
    bcz_px   = float(d["bcz_px"])
    px_um    = float(d["px_um"])
    lsd_um   = float(d["lsd_um"])
    lambda_a = float(d["lambda_a"])
    om0      = float(d["omega_first"])
    om_step  = float(d["omega_step"])
    threshold = float(d["threshold"])
    layer_start_filenr = int(d["layer_start_filenr"])
    pooled_shape = tuple(int(x) for x in d["pooled_shape"])

    # rehydrate detector-pixel coords at the bin centres
    k_frame = indices[:, 0].astype(np.float64)
    z_bin   = indices[:, 1].astype(np.float64)
    y_bin   = indices[:, 2].astype(np.float64)
    z_px = (z_bin + 0.5) * det_bin
    y_px = (y_bin + 0.5) * det_bin
    omega_deg = om0 + k_frame * om_step

    # lab-frame q (vectorized numpy; the same math as geometry.pixel_to_qlab
    # but without going through torch — we want a numpy-only loader so that
    # downstream code can opt into torch on its own terms).
    y_um = -(y_px - bcy_px) * px_um
    z_um =  (bcz_px - z_px) * px_um
    x_um = np.full_like(y_um, lsd_um)
    norm = np.sqrt(x_um * x_um + y_um * y_um + z_um * z_um)
    k_f_x = x_um / norm
    k_f_y = y_um / norm
    k_f_z = z_um / norm
    k0 = 2.0 * math.pi / lambda_a
    qlx = k0 * (k_f_x - 1.0)
    qly = k0 * k_f_y
    qlz = k0 * k_f_z

    # rotate lab -> sample by R_z(-ω) (MIDAS convention: vertical ω axis,
    # z invariant, x/y mixed — matches geometry.qlab_to_qsample / _spot_to_gv)
    om_rad = np.deg2rad(omega_deg)
    c = np.cos(-om_rad)
    s = np.sin(-om_rad)
    qsx = c * qlx - s * qly
    qsy = s * qlx + c * qly
    qsz = qlz

    # Geometry — only includes scalar fields the producer stored. n_pix_*
    # come from pooled_shape × det_bin (with the trailing rows the producer
    # discarded during max-pool).
    n_pix_z = pooled_shape[0] * det_bin
    n_pix_y = pooled_shape[1] * det_bin
    # NPZ does not carry the original n_frames; estimate from observed range
    n_frames_est = int(round(360.0 / abs(om_step))) if om_step else 1
    geom = Geometry(
        lsd_um=lsd_um,
        bcy_px=bcy_px,
        bcz_px=bcz_px,
        px_um=px_um,
        wavelength_A=lambda_a,
        n_pix_y=n_pix_y,
        n_pix_z=n_pix_z,
        omega_first_deg=om0,
        omega_step_deg=om_step,
        n_frames=n_frames_est,
        label=str(path),
    )

    return VoxelCloud(
        qx=qsx.astype(np.float64),
        qy=qsy.astype(np.float64),
        qz=qsz.astype(np.float64),
        intensity=values.astype(np.float64),
        omega_deg=omega_deg.astype(np.float64),
        det_y_px=y_px.astype(np.float64),
        det_z_px=z_px.astype(np.float64),
        geometry=geom,
        threshold=threshold,
        layer_start_filenr=layer_start_filenr,
    )


def save_voxel_npz_from_cloud(
    cloud: VoxelCloud,
    new_intensity: np.ndarray,
    path: Union[str, Path],
    *,
    det_bin: int,
    drop_zero: bool = True,
    new_threshold: Optional[float] = None,
) -> None:
    """Write a voxel NPZ in the same schema as `demk_volume.py`.

    Used to ship a Bragg-residual cloud (or any reweighted cloud) into the
    rods CLI without recomputing the geometry conversion.

    Parameters
    ----------
    cloud
        The original `VoxelCloud` (we reuse its detector/ω coordinates).
    new_intensity
        Per-voxel intensity to write (same length as `cloud.qx`).
    det_bin
        Detector pixel binning factor. Must match the original NPZ schema.
    drop_zero
        If True, drop voxels whose `new_intensity <= 0` so the output stays
        sparse.
    new_threshold
        Stored as the "threshold" field. Defaults to the original cloud's.
    """
    if len(new_intensity) != len(cloud):
        raise ValueError(
            f"intensity length {len(new_intensity)} != cloud size {len(cloud)}"
        )

    g = cloud.geometry
    # rehydrate indices from det_y_px / det_z_px / omega_deg
    k_frame = np.rint((cloud.omega_deg - g.omega_first_deg) / g.omega_step_deg
                      ).astype(np.int32)
    z_bin = np.rint(cloud.det_z_px / det_bin - 0.5).astype(np.int32)
    y_bin = np.rint(cloud.det_y_px / det_bin - 0.5).astype(np.int32)
    indices = np.stack([k_frame, z_bin, y_bin], axis=1)
    values = new_intensity.astype(np.float32)

    if drop_zero:
        keep = values > 0
        indices = indices[keep]
        values = values[keep]

    pooled_z = g.n_pix_z // det_bin
    pooled_y = g.n_pix_y // det_bin

    np.savez_compressed(
        path,
        indices=indices,
        values=values,
        pooled_shape=np.array([pooled_z, pooled_y], dtype=np.int32),
        det_bin=np.int32(det_bin),
        threshold=np.float32(new_threshold if new_threshold is not None
                              else cloud.threshold),
        omega_first=np.float32(g.omega_first_deg),
        omega_step=np.float32(g.omega_step_deg),
        bcy_px=np.float32(g.bcy_px),
        bcz_px=np.float32(g.bcz_px),
        px_um=np.float32(g.px_um),
        lsd_um=np.float32(g.lsd_um),
        lambda_a=np.float32(g.wavelength_A),
        layer_start_filenr=np.int32(cloud.layer_start_filenr),
    )
