"""FBP must return a reconstruction registered to the scan axis.

A point phantom at a known (x, y), forward-projected through the pf
convention ``s = x sin(w) + y cos(w)`` with scan column i at
``s = -25 + i``, must come back at the same (x, y). Before the crop was
computed about the (N-1)/2 centre, every feature landed 1.00 um low in
both axes -- a constant offset that silently mis-registered every pf
shape reconstruction against its voxel map.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.recon.fbp import fbp_recon_per_grain


def _project(mask, ax, omegas, n_scans):
    sino = np.zeros((len(omegas), n_scans))
    px, py = np.nonzero(mask)
    X, Y = ax[px], ax[py]
    w = np.radians(omegas)
    s = np.sin(w)[:, None] * X[None, :] + np.cos(w)[:, None] * Y[None, :]
    col = np.rint(s - ax[0]).astype(int)
    ok = (col >= 0) & (col < n_scans)
    for r in range(len(omegas)):
        c = col[r][ok[r]]
        if c.size:
            np.add.at(sino[r], c, 1.0)
    return sino


# 51 and 31 pass under the naive round() form too; 33/49/53/65 are the
# values where it silently reverts to the -1 um bug, so they must stay.
@pytest.mark.parametrize("n_scans", [31, 33, 49, 51, 53, 65])
def test_fbp_is_registered_to_the_scan_axis(tmp_path, n_scans):
    half = (n_scans - 1) // 2
    ax = np.arange(n_scans, dtype=float) - half
    omegas = np.linspace(-180.0, 180.0, 120, endpoint=False)
    targets = [(0, 0), (6, 0), (0, 6), (-8, 4)]

    sinos = np.zeros((len(targets), len(omegas), n_scans))
    for i, (tx, ty) in enumerate(targets):
        m = np.zeros((n_scans, n_scans), bool)
        ix, iy = int(tx + half), int(ty + half)
        m[ix - 1:ix + 2, iy - 1:iy + 2] = True
        sinos[i] = _project(m, ax, omegas, n_scans)

    rec = fbp_recon_per_grain(
        sinos, np.tile(omegas, (len(targets), 1)),
        np.full(len(targets), len(omegas), dtype=np.int32),
        n_scans, tmp_path, num_cpus=1, use_gpu=False,
    )

    for i, (tx, ty) in enumerate(targets):
        r = rec[i]
        k = np.unravel_index(np.argmax(r), r.shape)
        win = np.zeros_like(r)
        sl = (slice(max(k[0] - 3, 0), k[0] + 4), slice(max(k[1] - 3, 0), k[1] + 4))
        win[sl] = np.clip(r[sl], 0, None)
        tot = win.sum()
        cx = (win * ax[:, None]).sum() / tot
        cy = (win * ax[None, :]).sum() / tot
        assert abs(cx - tx) < 0.35, f"x off by {cx - tx:+.2f} at target {(tx, ty)}"
        assert abs(cy - ty) < 0.35, f"y off by {cy - ty:+.2f} at target {(tx, ty)}"
