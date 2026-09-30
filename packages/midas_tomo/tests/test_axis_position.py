"""Where the rotation axis lands in the output slice, as documented at AutoCentering
(c_src/tomo_utils.c, TomoConfig.auto_centering, run_tomo shifts).

A point sitting ON the rotation axis projects to the same detector column at every
angle, so it reconstructs to the axis pixel whatever the angles. Placing it at detector
column xdim/2 - shift and reading where it lands pins the convention:

    auto_centering=True  (default): (iy, ix) = (N/2 - 1, N/2 - 1 - round(shift))
    auto_centering=False          : (iy, ix) = (N/2 - 1, N/2 - 1)

Neither is N/2. Registering a slice to another frame (midas_stress.frames.tomo_grid_to_midas
rot_axis_ix / rot_axis_iy) needs these numbers, and a one-to-several pixel error there is a
silent translation of the sample.
"""
import numpy as np
import pytest

from midas_tomo import backend_c
from midas_tomo.api import run_tomo_from_sinos

pytestmark = pytest.mark.skipif(not backend_c.available(), reason="C engine not built")

THETAS = np.arange(-180.0, 180.01, 1.0)


def _axis_pixel(tmp_path, xdim, shift, auto_centering):
    cols = np.arange(xdim)
    axis_col = xdim / 2 - shift
    sino = np.tile(np.exp(-0.5 * ((cols - axis_col) / 1.0) ** 2).astype(np.float32), (len(THETAS), 1))
    im = run_tomo_from_sinos(sino, tmp_path, THETAS, shifts=shift, do_log=False, n_cpus=2,
                             auto_centering=auto_centering)[0, 0]
    w = np.clip(im - 0.5 * im.max(), 0, None)
    yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    return im.shape[0], (w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()


@pytest.mark.parametrize("xdim,shift", [(256, 0.0), (256, -6.0), (256, 3.3), (250, -4.0)])
def test_axis_with_auto_centering(tmp_path, xdim, shift):
    n, iy, ix = _axis_pixel(tmp_path, xdim, shift, True)
    assert iy == pytest.approx(n / 2 - 1, abs=0.2)
    assert ix == pytest.approx(n / 2 - 1 - round(shift), abs=0.2)


@pytest.mark.parametrize("xdim,shift", [(256, 0.0), (256, -6.0), (256, 3.3)])
def test_axis_without_auto_centering(tmp_path, xdim, shift):
    n, iy, ix = _axis_pixel(tmp_path, xdim, shift, False)
    assert iy == pytest.approx(n / 2 - 1, abs=0.2)
    assert ix == pytest.approx(n / 2 - 1, abs=0.2)
