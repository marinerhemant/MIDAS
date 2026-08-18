"""Independent NumPy oracle for the projection integral.

Phase 1 of ``implementation_plan.md``, design principle 5: validation against an
independent implementation is a deliverable, not an afterthought. Same pattern as
``midas_dfxm.validate``.

This is deliberately written the *other* way round from
:mod:`midas_dct_tt.project`: explicit Python loops over voxels, scalar
arithmetic, no broadcasting, no torch. It is slow and that is fine -- its only
job is to be wrong in different ways than the fast path, so that agreement to
floating-point means the geometry is right rather than that one bug was copied
into two places.

What it does *not* check: the physics upstream of the integral (acceptance,
structure factor, deformation). Those have their own analytic-limit tests. An
oracle only tells you the two implementations agree.
"""
from __future__ import annotations

import numpy as np

__all__ = ["numpy_project_rays_to_plane"]


def numpy_project_rays_to_plane(
    positions_lab,
    values,
    directions,
    *,
    normal,
    distance_um,
    voxel_volume_um3,
    pixel_um=1.0,
    detector_shape=(256, 256),
    center_px=None,
):
    """Reference implementation of :func:`midas_dct_tt.project.project_rays_to_plane`.

    Pure NumPy, explicit loops. Arguments and return value match the torch
    version (accepts anything ``np.asarray`` handles, including detached
    tensors); returns an ``(n_u, n_v)`` float64 array.
    """
    pos = np.asarray(positions_lab, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)

    if dirs.ndim == 1:
        dirs = np.tile(dirs, (pos.shape[0], 1))

    nu, nv = detector_shape
    if center_px is None:
        cu, cv = (nu - 1) / 2.0, (nv - 1) / 2.0
    else:
        cu, cv = center_px

    # Detector basis, matching midas_dfxm.optics.detector_basis exactly.
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(n, up))) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    u_vec = np.cross(up, n)
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(n, u_vec)

    img = np.zeros((nu, nv), dtype=np.float64)
    scale = voxel_volume_um3 / (pixel_um ** 2)

    for i in range(pos.shape[0]):
        r = pos[i]
        d = dirs[i]
        d = d / np.linalg.norm(d)
        denom = float(np.dot(d, n))
        if denom <= 0.0:
            continue                                  # ray cannot reach the plane
        t = (float(distance_um) - float(np.dot(r, n))) / denom
        hit = r + t * d
        u_px = float(np.dot(hit, u_vec)) / pixel_um + cu
        v_px = float(np.dot(hit, v_vec)) / pixel_um + cv
        if not (np.isfinite(u_px) and np.isfinite(v_px)):
            continue

        u0 = int(np.floor(u_px))
        v0 = int(np.floor(v_px))
        du = u_px - u0
        dv = v_px - v0
        for iu, wu in ((u0, 1.0 - du), (u0 + 1, du)):
            for iv, wv in ((v0, 1.0 - dv), (v0 + 1, dv)):
                if 0 <= iu < nu and 0 <= iv < nv:
                    img[iu, iv] += wu * wv * val[i] * scale
    return img
