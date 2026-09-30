"""WriteGreyResidual: grey levels of every lit pixel, aligned with SpotsInfo.bin."""
from __future__ import annotations

import numpy as np
import torch

from midas_nf_preprocess.process_images import ProcessImagesPipeline, ProcessParams
from midas_nf_preprocess.process_images.spots_io import SpotsBitMask


def _stack(n=3, size=20):
    z = torch.arange(size, dtype=torch.float64).view(-1, 1)
    y = torch.arange(size, dtype=torch.float64).view(1, -1)
    s = torch.zeros(n, size, size, dtype=torch.float64)
    for j in range(n):
        cz, cy = 5 + j * 4, 6 + j * 3
        s[j] = 100 + 800 * torch.exp(-((z - cz) ** 2 + (y - cy) ** 2) / (2 * 2.0 ** 2))
    return s


def _params(grey):
    return ProcessParams(nr_pixels_y=20, nr_pixels_z=20, nr_files_per_distance=3, n_distances=1,
                         log_mask_radius=3, sigma=1.5, mean_filt_radius=0, write_grey_residual=grey)


def _words(bm, path):
    bm.write(path)
    return np.fromfile(path, dtype="<u4")


def test_off_by_default_leaves_the_bitmask_unchanged(tmp_path):
    a = ProcessImagesPipeline(_params(0), device="cpu", dtype=torch.float64)
    b = ProcessImagesPipeline(_params(1), device="cpu", dtype=torch.float64)
    assert a.grey is None and b.grey is not None
    ma, mb = a.process_layer(1, stack=_stack()), b.process_layer(1, stack=_stack())
    assert np.array_equal(_words(ma, tmp_path / "a.bin"), _words(mb, tmp_path / "b.bin"))


def test_every_grey_pixel_is_a_set_bit_at_the_same_flipped_coordinate(tmp_path):
    pipe = ProcessImagesPipeline(_params(1), device="cpu", dtype=torch.float64)
    bm: SpotsBitMask = pipe.process_layer(1, stack=_stack())
    assert pipe.grey.n_pixels == bm.count_bits() > 0
    g = np.load(pipe.grey.write(tmp_path / "SpotsGrey.npz"))
    bits = np.unpackbits(_words(bm, tmp_path / "SpotsInfo.bin").view(np.uint8), bitorder="little")
    k = ((g["layer"].astype(np.int64) * 3 + g["frame"]) * 20 + g["y"]) * 20 + g["z"]
    assert bits[k].all()
    assert np.all(g["value"] > 0)                       # lit pixels carry the filtered (clamped > 0) level
    assert set(g["blanket_layer"].tolist()) == {0}
