"""Synthetic phantom and analytic sinogram, for tests that need real data.

Deliberately dependency-free: no skimage, no scipy. The projector is a plain
nearest-neighbour line integral, which is enough to produce a sinogram whose
reconstruction has recognisable structure. It is not accurate enough to serve
as ground truth for a quantitative accuracy claim, and no test should use it
that way.
"""

from __future__ import annotations

import numpy as np

__all__ = ["shepp_logan_like", "forward_project", "make_sino_dataset"]


def shepp_logan_like(n: int = 64) -> np.ndarray:
    """A few overlapping ellipses. Not the canonical Shepp-Logan constants.

    Enough structure (nested contrast, off-centre features) to make a
    mis-centred or artefact-ridden reconstruction visibly worse, which is all
    the tests need.
    """
    # Centre on (n-1)/2, the same origin forward_project uses. Centring on
    # n/2 instead leaves the phantom half a pixel off-centre, which makes it
    # subtly left-right asymmetric and shows up as the reconstruction
    # correlating better with fliplr(phantom) than with the phantom itself.
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    y = (y - c) / c
    x = (x - c) / c
    img = np.zeros((n, n))

    def ellipse(cx, cy, rx, ry, val):
        m = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
        img[m] += val

    ellipse(0.0, 0.0, 0.72, 0.92, 1.0)       # skull
    ellipse(0.0, 0.0, 0.66, 0.86, -0.8)      # interior
    ellipse(0.22, 0.0, 0.14, 0.22, 0.5)      # right inclusion
    ellipse(-0.22, 0.0, 0.14, 0.22, 0.5)     # left inclusion
    ellipse(0.0, -0.38, 0.20, 0.10, 0.35)    # lower bar
    return np.clip(img, 0.0, None)


def forward_project(img: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Radon transform matching gridrec's angle convention.

    Returns ``(n_angles, n_det)``. Each pixel is splatted into the two
    neighbouring detector bins with linear weights; nearest-neighbour
    splatting produced a visibly aliased sinogram that only reached ~0.86
    correlation after reconstruction.

    **Convention.** The detector coordinate is ``t = x sin(theta) + y
    cos(theta)``, not the ``x cos + y sin`` you might write first. The two
    differ by a transpose of the image, which is exactly what an earlier
    version of this file got wrong: reconstructions correlated ~0.0 with the
    phantom as-is and +0.86 with its transpose. The engine was right; the
    projector was not.
    """
    n = img.shape[0]
    angles = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    yc, xc = yy - c, xx - c
    flat = img.ravel()

    sino = np.zeros((len(angles), n), dtype=np.float64)
    for i, th in enumerate(angles):
        t = (xc * np.sin(th) + yc * np.cos(th) + c).ravel()
        lo = np.floor(t).astype(np.int64)
        frac = t - lo
        for offset, w in ((0, 1.0 - frac), (1, frac)):
            idx = lo + offset
            keep = (idx >= 0) & (idx < n)
            sino[i] += np.bincount(
                idx[keep], weights=(flat * w)[keep], minlength=n
            )[:n]
    return sino


def make_sino_dataset(n: int = 64, n_angles: int = 90):
    """``(phantom, sinogram, angles)`` covering 0-180 degrees."""
    phantom = shepp_logan_like(n)
    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    return phantom, forward_project(phantom, angles), angles
