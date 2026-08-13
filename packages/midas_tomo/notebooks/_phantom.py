"""
Synthetic-phantom data generator for the MIDAS TOMO tutorial notebooks.

Produces a tiny but realistic tomography dataset — a sample sphere mounted on
a tapered support pillar — projected over a configurable angle range. The
emitted data deliberately includes a few "dead pixel" detector columns so
notebooks can demonstrate Vo stripe-removal.

Public API
----------
``make_phantom(nz, ny, nx) -> np.ndarray``
    Build the 3D attenuation map.

``forward_project(volume, angles_deg) -> np.ndarray``
    Parallel-beam line integrals at the given angles.

``make_acquisition(volume, angles_deg, *, add_rings=True, n_darks=10,
                   n_whites=10, photon_count=2000, dark_level=100,
                   white_level=20000, rng=None) -> dict``
    Assemble a synthetic acquisition (dark, whites_before, whites_after,
    projections in detector counts). Includes Poisson + Gaussian noise and
    optional dead/flicker columns for ring artefacts.

``write_synthetic_hdf5(path, acq, angles_deg, *, shift=0.0, rot=0.0,
                       crop_xl=4, crop_xr=4, crop_zl=4, crop_zr=4)``
    Write a /exchange-style HDF5 file that ``process_hdf.py`` accepts.

``write_synthetic_tiff_stack(folder, acq, *, stem='phantom',
                             start_dark=1, start_white_before=100,
                             start_data=200, start_white_after=None,
                             zero_pad=6)``
    Write per-frame TIFFs mirroring the APS naming convention used by
    ``prepare_data_*.py`` scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


def make_phantom(nz: int = 64, ny: int = 64, nx: int = 64) -> np.ndarray:
    """3-D attenuation phantom: tapered support pillar + sample sphere.

    Geometry (in voxel units, centred on the volume):
      - Tapered pillar along z, radius 14 → 6 from z=0 to z=nz//2.
      - Sample sphere at z=int(0.78*nz), radius 0.10*nz, denser than pillar.
      - Two interior voids (lighter spots) inside the sphere.
      - Two harder inclusions inside the pillar (a tantalum-like flake and a
        small grain) — both denser than the pillar.

    The attenuation map is in arbitrary "linear-mu" units; the acquisition
    helper rescales it into a transmission range that fits a uint16 detector.
    """
    z, y, x = np.indices((nz, ny, nx))
    cy, cx = ny / 2.0, nx / 2.0
    vol = np.zeros((nz, ny, nx), dtype=np.float32)

    # Tapered pillar
    pillar_top = nz // 2
    pillar_r_top = max(6, int(0.10 * nx))
    pillar_r_bot = max(10, int(0.22 * nx))
    for zi in range(0, pillar_top + 1):
        # linear taper from bottom to top
        frac = zi / max(1, pillar_top)
        r = pillar_r_bot + (pillar_r_top - pillar_r_bot) * frac
        mask = (y[zi] - cy) ** 2 + (x[zi] - cx) ** 2 < r * r
        vol[zi][mask] = 0.30

    # Sample sphere
    sphere_z = int(0.78 * nz)
    sphere_r = max(4, int(0.10 * nz))
    mask = ((z - sphere_z) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) < sphere_r ** 2
    vol[mask] = 0.55

    # Interior voids in the sphere
    for dz, dy, dx, r in [(2, -2, 1, 2), (-2, 1, -2, 1.5)]:
        m = ((z - (sphere_z + dz)) ** 2 + (y - (cy + dy)) ** 2 +
             (x - (cx + dx)) ** 2) < r ** 2
        vol[m] = 0.10

    # Inclusions in the pillar
    incl_z = pillar_top - max(3, nz // 10)
    for dy, dx, r, mu in [(-3, 2, 2.0, 0.85), (4, -4, 1.5, 0.85)]:
        m = ((z - incl_z) ** 2 + (y - (cy + dy)) ** 2 +
             (x - (cx + dx)) ** 2) < r ** 2
        vol[m] = mu

    return vol


def forward_project(volume: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Simulate parallel-beam projections of *volume* at each angle.

    Returns an array of shape ``(n_angles, nz, nx)`` of line-integral values.

    Rotation is performed in the x-y plane (axes ``(2, 1)``) and the beam is
    integrated along the y-axis. Bilinear interpolation via
    :func:`scipy.ndimage.rotate`.
    """
    from scipy.ndimage import rotate
    nz, ny, nx = volume.shape
    out = np.zeros((len(angles_deg), nz, nx), dtype=np.float32)
    for i, theta in enumerate(angles_deg):
        rot = rotate(volume, -float(theta), axes=(2, 1), reshape=False,
                     order=1, mode='constant', cval=0.0)
        out[i] = rot.sum(axis=1)
    return out


@dataclass
class Acquisition:
    """Bundle of detector frames ready for HDF5 / TIFF / raw export."""
    dark: np.ndarray                  # (nz, nx) float32
    whites_before: np.ndarray         # (n_whites, nz, nx) float32
    whites_after: np.ndarray          # (n_whites, nz, nx) float32
    projections: np.ndarray           # (n_angles, nz, nx) uint16
    angles_deg: np.ndarray            # (n_angles,) float32
    ring_columns: np.ndarray          # x-indices of injected dead/flicker cols

    @property
    def whites(self) -> np.ndarray:
        return np.concatenate([self.whites_before, self.whites_after], axis=0)


def make_acquisition(volume: np.ndarray, angles_deg: np.ndarray, *,
                     add_rings: bool = True, n_darks: int = 10,
                     n_whites: int = 10, photon_count: int = 2000,
                     dark_level: float = 100.0, white_level: float = 20000.0,
                     rng: Optional[np.random.Generator] = None) -> Acquisition:
    """Turn a phantom volume into a noisy uint16 acquisition.

    Steps:
      1. Project the volume at every angle → line-integral sinogram.
      2. Convert to transmission via Beer–Lambert: T = exp(-mu * proj).
      3. Scale to a target white level, add a small spatial gradient.
      4. Inject dead/flicker detector columns (ring sources).
      5. Apply Poisson + read noise; clip to uint16.

    The dark and white frames are returned as ``float32`` averages because
    that is what MIDAS reads from disk; the projections are ``uint16`` to
    match real detectors.
    """
    if rng is None:
        rng = np.random.default_rng(seed=20260513)

    n_angles = len(angles_deg)
    projs = forward_project(volume, angles_deg)
    nz, nx = projs.shape[1], projs.shape[2]

    # A gentle horizontal gradient in the flat field
    grad = np.linspace(0.97, 1.03, nx, dtype=np.float32)[None, :]
    flat = np.broadcast_to(grad, (nz, nx))

    # Beer–Lambert and scale to detector counts
    transmission = np.exp(-projs)  # (n_angles, nz, nx)
    counts = transmission * white_level * flat  # broadcast over angle axis

    # Optional dead/flicker columns → rings in the reconstruction
    ring_cols = np.array([], dtype=int)
    if add_rings:
        ring_cols = rng.choice(np.arange(nx // 8, nx - nx // 8),
                               size=max(3, nx // 32), replace=False)
        # 60 % "dead" (constant low), 40 % "flicker" (high variance)
        n_dead = int(0.6 * len(ring_cols))
        dead = ring_cols[:n_dead]
        flick = ring_cols[n_dead:]
        for c in dead:
            counts[:, :, c] *= 0.6           # systematic 40 % undercount
        for c in flick:
            counts[:, :, c] *= rng.normal(
                loc=1.0, scale=0.15, size=(n_angles, nz)
            )

    # Poisson + read noise on projections
    proj_counts = rng.poisson(np.clip(counts, 0, None)).astype(np.float32)
    proj_counts += rng.normal(0.0, 8.0, proj_counts.shape).astype(np.float32)
    proj_counts = np.clip(proj_counts + dark_level, 0, 65535)
    projections_u16 = proj_counts.astype(np.uint16)

    # Dark frames: noisy around dark_level
    darks_stack = (dark_level
                   + rng.normal(0.0, 3.0, (n_darks, nz, nx))).astype(np.float32)
    dark_mean = darks_stack.mean(axis=0).astype(np.float32)

    # White frames: scaled flat with Poisson + read noise
    white_clean = (flat * white_level + dark_level).astype(np.float32)
    whites_b = np.stack(
        [rng.poisson(white_clean).astype(np.float32) for _ in range(n_whites)],
        axis=0,
    )
    whites_a = np.stack(
        [rng.poisson(white_clean).astype(np.float32) for _ in range(n_whites)],
        axis=0,
    )

    return Acquisition(
        dark=dark_mean,
        whites_before=whites_b,
        whites_after=whites_a,
        projections=projections_u16,
        angles_deg=np.asarray(angles_deg, dtype=np.float32),
        ring_columns=ring_cols,
    )


def write_synthetic_hdf5(path: str, acq: Acquisition, *, shift: float = 0.0,
                         rot: float = 0.0, crop_xl: int = 4, crop_xr: int = 4,
                         crop_zl: int = 4, crop_zr: int = 4) -> None:
    """Write *acq* to an APS-style HDF5 file that ``process_hdf.py`` accepts.

    Layout::

        /exchange/dark              (n_darks_avg, nz, nx)  uint16
        /exchange/bright            (n_whites_total, nz, nx)  uint16
        /exchange/data              (n_angles, nz, nx)  uint16
        /measurement/process/scan_parameters/start  scalar
        /measurement/process/scan_parameters/step   scalar
        /analysis/process/analysis_parameters/{CropXL,CropXR,CropZL,CropZR,shift}
    """
    import h5py
    angles = acq.angles_deg
    step = float(angles[1] - angles[0]) if len(angles) > 1 else 0.1
    start = float(angles[0])
    with h5py.File(path, 'w') as f:
        # exchange data — single dark image to keep file small; bright stack
        dark_u16 = acq.dark.astype(np.uint16)
        # store dark as 3D with N=1 to mirror real data exchange layout
        f.create_dataset('exchange/dark', data=dark_u16[None, :, :])
        whites_all = np.concatenate(
            [acq.whites_before, acq.whites_after], axis=0
        ).astype(np.uint16)
        f.create_dataset('exchange/bright', data=whites_all)
        f.create_dataset('exchange/data', data=acq.projections)
        # scan / analysis parameters
        for name, val in [
            ('measurement/process/scan_parameters/start', np.array([start])),
            ('measurement/process/scan_parameters/step', np.array([step])),
            ('analysis/process/analysis_parameters/CropXL', np.array([crop_xl])),
            ('analysis/process/analysis_parameters/CropXR', np.array([crop_xr])),
            ('analysis/process/analysis_parameters/CropZL', np.array([crop_zl])),
            ('analysis/process/analysis_parameters/CropZR', np.array([crop_zr])),
            ('analysis/process/analysis_parameters/shift', np.array([shift])),
        ]:
            f.create_dataset(name, data=val)
        if rot != 0.0:
            f.create_dataset(
                'analysis/process/analysis_parameters/RotationAngle',
                data=np.array([rot]))


def write_synthetic_tiff_stack(folder: str, acq: Acquisition, *,
                               stem: str = 'phantom', start_dark: int = 1,
                               start_white_before: int = 100,
                               start_data: int = 200,
                               start_white_after: Optional[int] = None,
                               zero_pad: int = 6) -> dict:
    """Write the acquisition out as per-frame TIFFs.

    Mirrors the APS folder layout: each frame is a uint16 TIFF named
    ``<stem>_<index>.tif`` where the index zero-pads to *zero_pad* digits.

    Returns a dict describing the index ranges so a notebook can show the
    user what to read::

        {'dark': (1, 10), 'white_before': (100, 109),
         'data': (200, 1559), 'white_after': (1560, 1569), 'pattern': '...'}
    """
    import tifffile
    os.makedirs(folder, exist_ok=True)
    n_darks = acq.whites_before.shape[0]   # match white count for simplicity
    n_whites_b = acq.whites_before.shape[0]
    n_whites_a = acq.whites_after.shape[0]
    n_data = acq.projections.shape[0]
    if start_white_after is None:
        start_white_after = start_data + n_data

    info = {
        'pattern': f'{folder}/{stem}_<index:0{zero_pad}d>.tif',
        'dark':         (start_dark,          start_dark + n_darks - 1),
        'white_before': (start_white_before,  start_white_before + n_whites_b - 1),
        'data':         (start_data,          start_data + n_data - 1),
        'white_after':  (start_white_after,   start_white_after + n_whites_a - 1),
    }

    def _path(i):
        return os.path.join(folder, f'{stem}_{i:0{zero_pad}d}.tif')

    # Darks: we only emit a dark *average* in acq, so synthesise N frames
    # around it that average back to that value.
    rng = np.random.default_rng(seed=42)
    for k in range(n_darks):
        frame = (acq.dark + rng.normal(0, 2.0, acq.dark.shape)).astype(np.uint16)
        tifffile.imwrite(_path(start_dark + k), frame)

    for k, w in enumerate(acq.whites_before.astype(np.uint16)):
        tifffile.imwrite(_path(start_white_before + k), w)

    for k, p in enumerate(acq.projections):
        tifffile.imwrite(_path(start_data + k), p)

    for k, w in enumerate(acq.whites_after.astype(np.uint16)):
        tifffile.imwrite(_path(start_white_after + k), w)

    return info


def make_sinograms_only(volume: np.ndarray, angles_deg: np.ndarray, *,
                        add_rings: bool = True,
                        rng: Optional[np.random.Generator] = None
                        ) -> np.ndarray:
    """Return float32 sinograms shaped (n_slices, n_thetas, det_xdim).

    Skips the dark/white step entirely — this matches the
    ``areSinos=1`` / ``run_tomo_from_sinos`` input.
    """
    if rng is None:
        rng = np.random.default_rng(seed=20260513)
    projs = forward_project(volume, angles_deg)
    # Move axes to (slice, theta, x)
    sinos = np.transpose(projs, (1, 0, 2)).astype(np.float32)
    if add_rings:
        nx = sinos.shape[-1]
        cols = rng.choice(np.arange(nx // 8, nx - nx // 8),
                          size=max(3, nx // 32), replace=False)
        for c in cols:
            sinos[:, :, c] += 0.05
    # Small additive noise
    sinos += rng.normal(0.0, 0.01, sinos.shape).astype(np.float32)
    return sinos
