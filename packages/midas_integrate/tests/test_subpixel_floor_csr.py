"""SubPixelLevel > 1 in floor mode: a sub-pixel entry carries y = raw_y + offset with |offset| < 0.5 (e.g. 2.75 or 3.25 for pixel 3, K = 2).
The floor CSR must put it in its PARENT pixel (nearest integer), not truncate it into the neighbour (2.75 -> 2). The truncating cast moved about half of all sub-pixel
entries to the neighbouring pixel, so a pixel's column of the CSR no longer summed to the pixel's area."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from midas_integrate.detector_mapper import build_map
from midas_integrate.kernels import _build_floor_csr, build_csr
from midas_integrate.params import IntegrationParams


def test_floor_csr_puts_subpixel_entries_in_their_parent_pixel():
    y = np.array([3.25, 2.75, 3.0, 3.0, 3.0])
    z = np.array([5.0, 5.0, 5.25, 4.75, 5.0])
    csr = _build_floor_csr(bin_index=np.zeros(5, dtype=np.int64), pixel_y=y, pixel_z=z, frac=np.ones(5), n_bins=1, n_pixels_y=8, n_pixels_z=8,
                           device="cpu", dtype=torch.float64)
    cols = csr.to_dense().numpy().ravel().nonzero()[0]
    assert cols.tolist() == [5 * 8 + 3], cols                     # every entry belongs to pixel (z=5, y=3)
    assert csr.to_dense().numpy().ravel()[5 * 8 + 3] == 5.0


@pytest.mark.parametrize("K", [1, 2, 3])
def test_every_interior_pixel_column_sums_to_its_area(K):
    ny = nz = 40
    p = IntegrationParams(NrPixelsY=ny, NrPixelsZ=nz, pxY=172.0, pxZ=172.0, Lsd=300_000.0, BC_y=20.3, BC_z=19.7, RhoD=1.0e6,
                          RMin=3.0, RMax=26.0, RBinSize=1.0, EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0, SubPixelLevel=K)
    m = build_map(p, verbose=False, auto_load=False)
    geom = build_csr(m.as_pixel_map(), n_r=p.n_r_bins, n_eta=p.n_eta_bins, n_pixels_y=ny, n_pixels_z=nz, device="cpu", dtype=torch.float64,
                     bc_y=p.BC_y, bc_z=p.BC_z, build_modes=("floor",))
    col = geom.csr_floor.to_dense().numpy().sum(axis=0).reshape(nz, ny)
    Y, Z = np.meshgrid(np.arange(ny), np.arange(nz))
    r = np.hypot(Y - p.BC_y, Z - p.BC_z)
    inside = (r > 6.0) & (r < 22.0)                                # whole pixel inside the radial range, away from the beam centre
    assert inside.sum() > 500
    assert np.abs(col[inside] - 1.0).max() < 1e-5, (K, float(np.abs(col[inside] - 1.0).max()))
