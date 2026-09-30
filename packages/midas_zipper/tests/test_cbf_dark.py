"""A CBF sweep with a `Dark` (TIFF) must zip, dark-subtracted.

Regression: `process_multifile_scan` imported `tifffile` only when the DATA
was TIFF, but reads every non-GE dark with it, so a CBF scan carrying a
`Dark` key died with `UnboundLocalError: ... 'tifffile'` (found on the
DESY P21.2 Varex Inconel FF scan, 2026-09-29).
"""

from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

from midas_zipper import ff_zip
import midas_zipper._read_cbf as rc


def test_cbf_sweep_with_tiff_dark(tmp_path, monkeypatch):
    n = 8
    frames = [np.full((n, n), 100 + 10 * i, dtype=np.int32) for i in range(3)]
    for i in range(3):
        (tmp_path / f"scan_{i + 1:05d}.cbf").write_bytes(b"")   # existence only
    dark = np.full((n, n), 95, dtype=np.int32)
    tifffile.imwrite(tmp_path / "dark.tif", dark)

    hdr = {"X-Binary-Size-Fastest-Dimension": n,
           "X-Binary-Size-Second-Dimension": n,
           "X-Binary-Element-Type": "signed 32-bit integer"}
    monkeypatch.setattr(rc, "read_cbf_metadata", lambda fn: hdr)
    idx = {str(tmp_path / f"scan_{i + 1:05d}.cbf"): i for i in range(3)}
    monkeypatch.setattr(rc, "read_cbf",
                        lambda fn, check_md5=False: ({}, frames[idx[str(fn)]]))

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="w")
    z_groups = {"exc": root.require_group("exchange")}
    config = {"dataFN": str(tmp_path / "scan_00001.cbf"),
              "darkFN": str(tmp_path / "dark.tif"),
              "numFilesPerScan": 3, "SkipFrame": 0}
    ff_zip.process_multifile_scan("cbf", config, z_groups)

    data = root["exchange/data"][:]
    assert data.shape == (3, n, n)
    np.testing.assert_array_equal(data[:, 0, 0], [5, 15, 25])
    assert not root["exchange/dark"][:].any()   # data already dark-subtracted


@pytest.mark.parametrize("dt", [np.int32, np.uint16, np.uint32, np.int16, np.float32, np.float64])
def test_datatype_string_covers_reader_dtypes(dt):
    """CBF frames are int32; 'unknown' made midas_peakfit raise KeyError."""
    pytest.importorskip("midas_peakfit", reason="cross-checks datatype names against midas_peakfit's own table")
    from midas_peakfit.zarr_io import _bytes_per_px, canonical_pixel_type
    name = ff_zip.datatype_string(dt)
    assert _bytes_per_px(canonical_pixel_type(name)) == np.dtype(dt).itemsize


def test_datatype_string_refuses_unknown():
    with pytest.raises(ValueError):
        ff_zip.datatype_string(np.complex64)
