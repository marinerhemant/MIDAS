"""Item 3 — GEBinaryFrameSource."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_integrate_v2.streaming import GEBinaryFrameSource


def _make_synthetic_ge(path: Path, *, side: int, n_frames: int,
                        with_header: bool = True, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 60_000, size=(n_frames, side, side),
                           dtype=np.uint16)
    with open(path, "wb") as f:
        if with_header:
            f.write(b"\x00" * 8192)
        f.write(frames.tobytes())
    return frames


def test_ge_iter_with_header_2048(tmp_path: Path):
    side = 256  # smaller than 2048 to keep the test light
    frames = _make_synthetic_ge(tmp_path / "data.ge3", side=side, n_frames=4)
    src = GEBinaryFrameSource(tmp_path / "data.ge3", side=side)
    assert src.n_frames == 4
    assert src.frame_shape == (side, side)
    seen = list(src)
    assert len(seen) == 4
    fid, img = seen[0]
    assert fid == "ge_00000"
    np.testing.assert_array_equal(img.astype(np.uint16), frames[0])


def test_ge_random_access_no_header(tmp_path: Path):
    side = 128
    frames = _make_synthetic_ge(tmp_path / "data_noheader.ge2",
                                  side=side, n_frames=3, with_header=False)
    src = GEBinaryFrameSource(tmp_path / "data_noheader.ge2", side=side)
    fid, img = src.get(2)
    np.testing.assert_array_equal(img.astype(np.uint16), frames[2])
    assert fid == "ge_00002"


def test_ge_skip_frame(tmp_path: Path):
    side = 64
    frames = _make_synthetic_ge(tmp_path / "data.ge3", side=side, n_frames=5)
    src = GEBinaryFrameSource(tmp_path / "data.ge3", side=side, skip_frame=2)
    assert src.n_frames == 3
    fid, img = next(iter(src))
    np.testing.assert_array_equal(img.astype(np.uint16), frames[2])
