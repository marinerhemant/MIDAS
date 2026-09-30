"""pf-grid sample mask from a midas_transforms SampleShape (recon/sample_mask.py)."""
import numpy as np
import pytest
from midas_pipeline.recon.sample_mask import load_sample_mask, mask_from_sample_shape, voxel_xy_um


def _grid(tmp_path, n=21, step=10.0):
    pos = (np.arange(n) - n // 2) * step
    rows = [(i * n + j, pos[i], pos[j], 0.0, -1) for i in range(n) for j in range(n)]
    p = tmp_path / "voxel_grid.csv"
    np.savetxt(p, rows, header="voxel_idx x_um y_um z_um grain_id", comments="", fmt=["%d", "%.4f", "%.4f", "%.4f", "%d"])
    return p


def test_voxel_xy_reads_the_grid(tmp_path):
    xy = voxel_xy_um(_grid(tmp_path, n=5))
    assert xy.shape == (25, 2) and xy[1].tolist() == [-20.0, -10.0]


def test_a_tomogram_disc_becomes_a_disc_on_the_pf_grid(tmp_path):
    tomo = pytest.importorskip("midas_transforms.geometry.tomo")
    yy, xx = np.mgrid[0:41, 0:41]
    vol = ((xx - 20) ** 2 + (yy - 20) ** 2 <= 10 ** 2).astype(float)[None]        # radius 10 px = 100 um
    shape = tomo.from_array(vol, pixel_size_um=10.0, rot_axis_ix=20, rot_axis_iy=20, in_plane="xy", threshold=0.5)
    m = mask_from_sample_shape(shape, _grid(tmp_path), z_um=0.0)
    assert m.shape == (21, 21)
    assert 280 <= m.sum() <= 330                          # ~ pi * 10^2 = 314 voxels of 10 um
    assert m[10, 10] and not m[0, 0] and np.array_equal(m, m[::-1, ::-1])


def test_load_sample_mask_checks_shape(tmp_path):
    np.save(tmp_path / "m.npy", np.ones((4, 4)))
    assert load_sample_mask(tmp_path / "m.npy", 4).all()
    with pytest.raises(ValueError, match="grid is"):
        load_sample_mask(tmp_path / "m.npy", 5)
