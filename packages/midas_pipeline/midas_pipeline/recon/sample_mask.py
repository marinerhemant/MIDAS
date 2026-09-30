"""A sample mask on the pf voxel grid, from a tomogram (or any array), for the reconstruct stage.

Why: every tomographic reconstruction of a pf sinogram is > 0 outside the sample (FBP streaks, MLEM
noise), and the grain map was ``argmax`` with -1 only where every grain is <= 0, so vacuum got labels.
On 20-ID-E Fe9Cr (2026-09-28) the sample filled ~6,800 of 14,641 voxels, and a half-split taken over the
whole grid measured vacuum (FBP 0.33 all-grid vs 0.56 on the sample). A mask fixes both the map and any
score. It also decides which edge voxels are material: vacuum voxels next to a grain score high
completeness, which neither completeness nor the omega-shuffle null can tell apart from material.

The tomogram route reuses what exists: ``midas_transforms.geometry.tomo`` builds a ``SampleShape``
(explicit pixel size, rotation-axis position and in-plane handedness; nothing is defaulted, see
manuals/tomo/COORDINATES.md), ``SampleShape.contains`` answers per point, and
``midas_transforms.geometry.registration.centroid_containment_check`` (+ ``meta_null``) checks the
registration against the grain centroids. This module only puts the answer on the pf grid.

Grid convention: the pf voxel of flat index v = i * n + j sits at ``Output/voxel_grid.csv`` (x_um, y_um),
and a mask is an (n, n) bool array indexed [i, j], the same array layout as the reconstruct stage's maps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = ["voxel_xy_um", "mask_from_sample_shape", "load_sample_mask", "write_sample_mask", "registration_report"]


def voxel_xy_um(voxel_grid_csv: Union[str, Path]) -> np.ndarray:
    """(n*n, 2) voxel centres in um, in flat-index order, from ``Output/voxel_grid.csv``."""
    t = np.loadtxt(voxel_grid_csv, skiprows=1, ndmin=2)
    idx = t[:, 0].astype(int)
    if not np.array_equal(idx, np.arange(idx.size)):
        raise ValueError(f"{voxel_grid_csv}: voxel_idx is not 0..N-1 in order")
    n = int(round(np.sqrt(idx.size)))
    if n * n != idx.size:
        raise ValueError(f"{voxel_grid_csv}: {idx.size} voxels is not a square grid")
    return t[:, 1:3]


def mask_from_sample_shape(shape, voxel_grid_csv: Union[str, Path], *, z_um: float = 0.0,
                           translation_um: Sequence[float] = (0.0, 0.0, 0.0), threshold: float = 0.5) -> np.ndarray:
    """(n, n) bool: is each pf voxel centre inside ``shape`` (a midas_transforms SampleShape) at height z_um."""
    xy = voxel_xy_um(voxel_grid_csv)
    pts = np.c_[xy, np.full(len(xy), float(z_um))]
    inside = np.asarray(shape.contains(pts, threshold=threshold, translation_um=translation_um), bool)
    n = int(round(np.sqrt(len(xy))))
    return inside.reshape(n, n)


def load_sample_mask(path: Union[str, Path], n: int) -> np.ndarray:
    """Read a mask (.npy or .tif; nonzero = sample) and check it is (n, n)."""
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        m = tifffile.imread(str(p))
    else:
        m = np.load(p)
    m = np.asarray(m) != 0
    if m.shape != (n, n):
        raise ValueError(f"sample mask {p} is {m.shape}; the reconstruction grid is ({n}, {n})")
    if not m.any():
        raise ValueError(f"sample mask {p} is empty")
    return m


def write_sample_mask(mask: np.ndarray, path: Union[str, Path]) -> Path:
    p = Path(path)
    np.save(p, np.asarray(mask, bool))
    return p if p.suffix == ".npy" else p.with_suffix(p.suffix + ".npy")


def registration_report(shape, centroids_um: np.ndarray, **kw) -> dict:
    """V2 containment of grain centroids (held-out fraction is the result) and its mirrored meta-null,
    from midas_transforms.geometry.registration. Returned as a plain dict for logging / JSON."""
    from midas_transforms.geometry.registration import centroid_containment_check, meta_null
    v2 = centroid_containment_check(shape, np.asarray(centroids_um, float), **kw)
    mn = meta_null(centroid_containment_check, shape, np.asarray(centroids_um, float), **kw)
    as_dict = lambda r: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(r).items()} \
        if hasattr(r, "__dict__") else {"result": str(r)}
    return {"centroid_containment": as_dict(v2), "meta_null_mirrored": as_dict(mn)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m midas_pipeline.recon.sample_mask",
                                 description="pf-grid sample mask from a midas_tomo NXtomoproc reconstruction")
    ap.add_argument("nxtomoproc"); ap.add_argument("voxel_grid_csv"); ap.add_argument("out_npy")
    ap.add_argument("--pixel-size-um", type=float, required=True)
    ap.add_argument("--rot-axis-ix", type=float, required=True); ap.add_argument("--rot-axis-iy", type=float, required=True)
    ap.add_argument("--in-plane", required=True, help="one of midas_stress.frames.TOMO_IN_PLANE; no default")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--slice-range", type=int, nargs=2); ap.add_argument("--slice-pitch-um", type=float)
    ap.add_argument("--z-um", type=float, default=0.0)
    ap.add_argument("--translation-um", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--shift-index", type=int)
    ap.add_argument("--grain-centroids", help="optional CSV of grain centroids x y z (um) for the V2 check")
    a = ap.parse_args(argv)
    from midas_transforms.geometry.tomo import from_nxtomoproc
    shape = from_nxtomoproc(a.nxtomoproc, pixel_size_um=a.pixel_size_um, rot_axis_ix=a.rot_axis_ix,
                            rot_axis_iy=a.rot_axis_iy, in_plane=a.in_plane, threshold=a.threshold,
                            shift_index=a.shift_index, slice_range=tuple(a.slice_range) if a.slice_range else None,
                            slice_pitch_um=a.slice_pitch_um)
    m = mask_from_sample_shape(shape, a.voxel_grid_csv, z_um=a.z_um, translation_um=a.translation_um)
    out = write_sample_mask(m, a.out_npy)
    rep = {"mask": str(out), "voxels_inside": int(m.sum()), "grid": list(m.shape), "in_plane": a.in_plane,
           "translation_um": list(a.translation_um), "registration": "NOT verified"}
    if a.grain_centroids:
        rep["registration"] = registration_report(shape, np.loadtxt(a.grain_centroids, delimiter=",", ndmin=2)[:, :3])
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
