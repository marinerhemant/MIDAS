"""The E-step's pixel inversion must use the same distortion as the forward model.

``extract_fitted_points`` turns each (R, eta) centroid back into a detector pixel
with ``invert_REta_to_pixel_batch``. The cake it reads was built WITH p0..p14,
and the M-step judges the points with ``predict_R_at_pixel``, which applies them.
The inversion was called without them, so with nonzero p every point landed off
its true pixel (+0.09 px mean at p2 = p4 = 0.002, +0.18 px at 0.004, growing with
radius) and the M-step then fitted that inconsistency into Lsd and the p terms.

Found by a /verify refuter on a real CCD frame set (2026-10-12); on those
real frames it moved the patched free-distortion Lsd by 4-10 um. At p = 0 the
bug is invisible, which is why nothing caught it.

The check is a round trip: render rings THROUGH a distorted geometry, extract at
that exact geometry, and require the M-step's own forward model to put every
point on its ring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from test_clipped_bin_rejection import _params, _ring_image

from midas_calibrate.estep import run_estep
from midas_calibrate.geometry_torch import predict_R_at_pixel
from midas_calibrate.param_vector import pack
from midas_calibrate.rings import build_ring_table


def _ring_residuals(p):
    rt = build_ring_table(p)
    _cake, fits = run_estep(p, _ring_image(p), rt)
    Y = torch.tensor([f.Y_pix for f in fits])
    Z = torch.tensor([f.Z_pix for f in fits])
    R = predict_R_at_pixel(Y, Z, pack(p), px=p.pxY, rho_d=p.RhoD).numpy()
    ring = np.array([f.ring_idx for f in fits])
    return ring, R - rt.r_ideal_px[ring]


@pytest.mark.parametrize("pval", [0.0, 0.002, 0.004])
def test_points_lie_on_their_rings_under_the_forward_model(pval):
    p = _params(N=512, Lsd=400_000.0, ty=0.30, tz=-0.45, p2=pval, p4=pval)
    ring, d = _ring_residuals(p)
    assert d.size > 100
    for r in np.unique(ring):
        assert abs(d[ring == r].mean()) < 0.03, (
            f"ring {r + 1}: mean {d[ring == r].mean():+.3f} px off its ring at p = {pval}")
