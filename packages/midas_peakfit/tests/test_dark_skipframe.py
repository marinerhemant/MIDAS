"""SkipFrame must never turn an existing dark into zeros.

Regression: ``ZarrParams.finalize`` did ``nDarks = max(0, nDarks - skipFrame)``,
so a 1-frame ``exchange/dark`` with ``SkipFrame 1`` gave ``nDarks == 0`` and
``load_corrections`` then substituted an all-zero dark with no warning. On a
20-ID-D Varex zarr (dark mean ~1856) this left the raw pedestal in every
frame and every ring band became one giant blob.

Contract tested here:
  * dark with MORE than skipFrame frames -> leading skipFrame frames dropped
    (C parity, unchanged);
  * dark with <= skipFrame frames -> used whole, with a UserWarning;
  * no dark at all -> zeros, no warning.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import zarr

from midas_peakfit.zarr_io import load_corrections, parse_zarr_params

NZ, NY = 16, 16


def _write_zarr(path, dark, skip_frame):
    data = np.full((4, NZ, NY), 2000, dtype=np.uint16)
    with zarr.ZipStore(str(path), mode="w") as store:
        root = zarr.open_group(store=store, mode="w")
        root.create_dataset("exchange/data", data=data, chunks=(1, NZ, NY))
        if dark is not None:
            root.create_dataset("exchange/dark", data=dark, chunks=(1, NZ, NY))
        ap = root.require_group("analysis/process/analysis_parameters")
        ap.create_dataset("SkipFrame", data=np.array([skip_frame], dtype=np.int32))
        ap.create_dataset("PixelSize", data=np.array([150.0]))
        ap.create_dataset("Width", data=np.array([1000.0]))
        sp = root.require_group("measurement/process/scan_parameters")
        sp.create_dataset("start", data=np.array([0.0]))
        sp.create_dataset("step", data=np.array([0.25]))
    return path


def _load(path):
    p = parse_zarr_params(str(path))  # parse + finalize
    load_corrections(str(path), p)
    return p


def test_single_frame_dark_with_skipframe_is_kept_and_warns(tmp_path):
    rng = np.random.default_rng(0)
    dark = (1850 + rng.integers(0, 20, size=(1, NZ, NY))).astype(np.uint16)
    path = _write_zarr(tmp_path / "one_dark.MIDAS.zip", dark, skip_frame=1)
    with pytest.warns(UserWarning, match="dark is used whole"):
        p = _load(path)
    assert p.skipFrame == 1
    assert p.nDarks == 1
    np.testing.assert_array_equal(p.dark, dark[0].astype(np.float64))
    assert p.dark.mean() > 1800  # not the silent all-zero dark


def test_multi_frame_dark_drops_exactly_skipframe(tmp_path):
    # Frame 0 is a wild throwaway; frames 1..9 are distinct and known.
    dark = np.empty((10, NZ, NY), dtype=np.uint16)
    dark[0] = 60000
    for k in range(1, 10):
        dark[k] = 1800 + k
    path = _write_zarr(tmp_path / "ten_dark.MIDAS.zip", dark, skip_frame=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = _load(path)
    assert p.nDarks == 9
    np.testing.assert_allclose(p.dark, dark[1:].astype(np.float64).mean(axis=0))


def test_no_dark_gives_zeros_without_warning(tmp_path):
    path = _write_zarr(tmp_path / "no_dark.MIDAS.zip", None, skip_frame=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = _load(path)
    assert p.nDarks == 0
    assert p.dark.shape == (NZ, NY)
    assert (p.dark == 0).all()
