"""Item 4 — EDFFrameSource."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

fabio = pytest.importorskip("fabio")
from midas_integrate_v2.streaming import EDFFrameSource


def test_edf_single_frame(tmp_path: Path):
    img = np.arange(64, dtype=np.uint16).reshape(8, 8)
    out_path = tmp_path / "frame.edf"
    edf = fabio.edfimage.EdfImage(data=img)
    edf.write(str(out_path))

    src = EDFFrameSource(out_path)
    assert src.n_frames == 1
    assert src.frame_shape == (8, 8)
    fid, arr = next(iter(src))
    np.testing.assert_array_equal(arr.astype(np.uint16), img)


def test_edf_multi_file_glob(tmp_path: Path):
    paths = []
    for k in range(3):
        img = (np.arange(64, dtype=np.uint16) + 10 * k).reshape(8, 8)
        p = tmp_path / f"shot_{k:03d}.edf"
        fabio.edfimage.EdfImage(data=img).write(str(p))
        paths.append(p)
    src = EDFFrameSource(str(tmp_path / "shot_*.edf"))
    assert src.n_frames == 3
    seen = list(src)
    assert seen[0][0] == "shot_000"
    np.testing.assert_array_equal(seen[2][1].astype(np.uint16),
                                    (np.arange(64, dtype=np.uint16) + 20).reshape(8, 8))
