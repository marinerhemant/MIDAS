"""The residual map is added to R_px from pixel_to_REta, which is in MEAN-pixel units: the um -> px conversion must use the
mean pitch, not pxY, for non-square pixels."""
import numpy as np
import torch

from midas_calibrate_v2.forward.residual_corr import build_residual_corr_map


def _fits(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = torch.tensor(rng.uniform(5, 195, n), dtype=torch.float64)
    z = torch.tensor(rng.uniform(5, 195, n), dtype=torch.float64)
    return y, z


def test_px_mean_sets_the_um_to_pixel_scale():
    y, z = _fits()
    dR_um = torch.full_like(y, 30.0)                      # a constant 30 um radial residual
    kw = dict(NrPixelsY=200, NrPixelsZ=200, pxY=172.0, smoothing=1e-3)
    m_mean = build_residual_corr_map(y, z, dR_um, px_mean=186.0, **kw)
    m_y = build_residual_corr_map(y, z, dR_um, **kw)       # no px_mean: square-pixel behaviour (divides by pxY)
    centre = (slice(60, 140), slice(60, 140))
    np.testing.assert_allclose(float(m_mean[centre].mean()), -30.0 / 186.0, rtol=2e-2)
    np.testing.assert_allclose(float(m_y[centre].mean()), -30.0 / 172.0, rtol=2e-2)
    assert abs(float(m_mean[centre].mean()) / float(m_y[centre].mean()) - 172.0 / 186.0) < 5e-3
