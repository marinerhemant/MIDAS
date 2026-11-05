"""Lossless archive encoding: the round trip is bit-exact, whatever the data."""

from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from midas_zipper import archive


def _random_words(rng, dtype, shape):
    info = np.iinfo(dtype)
    x = rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)
    x.flat[:4] = [info.min, info.max, 0, info.max]   # extremes always present
    return x


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.uint32, np.int32])
def test_zigzag_roundtrip_is_exact_for_any_words(dtype):
    rng = np.random.default_rng(0)
    a = _random_words(rng, dtype, (3, 17, 19))
    ref = _random_words(rng, dtype, (17, 19))
    z = archive.zigzag_encode(a, ref)
    assert z.dtype == np.dtype(f"u{np.dtype(dtype).itemsize}")
    back = archive.zigzag_decode(z, ref, dtype)
    assert back.dtype == np.dtype(dtype)
    assert np.array_equal(back, a)


def test_zigzag_small_residuals_give_small_words():
    ref = np.full((4, 4), 1000, dtype=np.uint16)
    a = (ref + np.array([-3, -1, 0, 2], dtype=np.int16)[None, :]).astype(np.uint16)
    assert archive.zigzag_encode(a, ref)[0].tolist() == [5, 1, 0, 4]   # -3, -1, 0, +2


def test_common_shift_finds_the_bit_grid():
    rng = np.random.default_rng(1)
    twelve_in_sixteen = (rng.integers(0, 4096, (4, 8, 8)) << 4).astype(np.uint16)
    assert archive.common_shift(twelve_in_sixteen) == 4
    assert archive.common_shift(twelve_in_sixteen + np.uint16(1)) == 0
    assert archive.common_shift(np.zeros((2, 2, 2), np.uint16)) == 0


def _make_zip(path, data, dark):
    store = zarr.ZipStore(str(path), mode="w")
    root = zarr.group(store=store, overwrite=True)
    from numcodecs import Blosc
    comp = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    exc = root.create_group("exchange")
    N, Z, Y = data.shape
    exc.create_dataset("data", data=data, chunks=(1, Z, Y), compressor=comp)
    exc.create_dataset("dark", data=dark, chunks=(1, Z, Y), compressor=comp)
    exc.create_dataset("bright", data=dark, chunks=(1, Z, Y), compressor=comp)
    root.create_group("measurement/process/scan_parameters").create_dataset(
        "datatype", data=np.bytes_(b"uint16"))
    store.close()


def _pedestal_scan(rng, dtype=np.uint16, n=24, shape=(48, 40), grid=0):
    ped = rng.integers(200, 900, shape)
    dark = np.stack([ped + rng.integers(-2, 3, shape) for _ in range(4)])
    frames = ped[None] + rng.poisson(3, (n, *shape)) - 1
    frames[:, 5, 7] += 4000                              # a bright pixel
    return (frames << grid).astype(dtype), (dark << grid).astype(dtype)


def _size(path):
    return path.stat().st_size


def test_convert_and_restore_are_bit_exact_and_smaller(tmp_path):
    rng = np.random.default_rng(2)
    data, dark = _pedestal_scan(rng)
    src, arc, back = tmp_path / "a.zip", tmp_path / "a.arc.zip", tmp_path / "a.back.zip"
    _make_zip(src, data, dark)
    name, sizes = archive.convert_zip(src, arc, mode="dark", log=lambda *a: None)
    assert name == "dark"
    assert _size(arc) < _size(src)
    z = zarr.open(zarr.ZipStore(str(arc), mode="r"), mode="r")
    assert "data" not in z["exchange"]                   # a naive reader must fail, not misread
    assert z["exchange/data_resid"].attrs["encoding"] == archive.ENCODING
    assert np.array_equal(archive.read_frames(arc), data)
    assert np.array_equal(archive.read_frames(arc, 3, 9), data[3:9])
    archive.restore_zip(arc, back)
    zb = zarr.open(zarr.ZipStore(str(back), mode="r"), mode="r")
    assert np.array_equal(zb["exchange/data"][()], data)
    assert np.array_equal(zb["exchange/dark"][()], dark)  # untouched
    assert bytes(zb["measurement/process/scan_parameters/datatype"][()]) == b"uint16"


def test_twelve_bits_in_sixteen_needs_the_dark_on_the_bit_grid(tmp_path):
    rng = np.random.default_rng(3)
    data, dark = _pedestal_scan(rng, grid=4)
    dark = dark + np.uint16(rng.integers(0, 16, dark.shape))   # a dark that is off the grid
    src, arc = tmp_path / "t.zip", tmp_path / "t.arc.zip"
    _make_zip(src, data, dark)
    name, sizes = archive.convert_zip(src, arc, mode="dark", log=lambda *a: None)
    assert name == "dark_q"
    assert sizes["dark_q"] < sizes["dark"]
    assert np.array_equal(archive.read_frames(arc), data)


def test_incompressible_frames_are_left_alone(tmp_path):
    rng = np.random.default_rng(4)
    data = rng.integers(0, 65536, (8, 32, 32)).astype(np.uint16)
    src = tmp_path / "n.zip"
    _make_zip(src, data, np.zeros((2, 32, 32), np.uint16))
    before = src.read_bytes()
    name, _ = archive.convert_zip(src, None, mode="auto", log=lambda *a: None)
    assert name == "raw"
    assert src.read_bytes() == before


def test_median_background_without_a_dark(tmp_path):
    rng = np.random.default_rng(5)
    data, _ = _pedestal_scan(rng)
    src, arc = tmp_path / "m.zip", tmp_path / "m.arc.zip"
    _make_zip(src, data, np.zeros((1, *data.shape[1:]), np.uint16))   # placeholder dark
    name, _ = archive.convert_zip(src, arc, mode="median", log=lambda *a: None)
    assert name.startswith("median")
    assert np.array_equal(archive.read_frames(arc), data)


def test_uint32_gap_sentinels_survive(tmp_path):
    rng = np.random.default_rng(6)
    data, dark = _pedestal_scan(rng, dtype=np.uint32)
    data[:, :3, :] = np.uint32(4294967295)                # Eiger-style gap pixels
    src, arc = tmp_path / "g.zip", tmp_path / "g.arc.zip"
    _make_zip(src, data, dark)
    name, _ = archive.convert_zip(src, arc, mode="auto", log=lambda *a: None)
    assert name != "raw"
    assert np.array_equal(archive.read_frames(arc), data)


def test_flag_keeps_raw_frames_where_the_default_clips(tmp_path):
    """With a dark file the default path zeroes pixels below the dark; -archiveLossless must not."""
    tifffile = pytest.importorskip("tifffile")
    from midas_zipper import ff_zip
    rng = np.random.default_rng(7)
    n, nf = 16, 6
    dark = rng.integers(90, 100, (n, n)).astype(np.uint16)
    frames = (dark[None] + rng.integers(-4, 6, (nf, n, n))).astype(np.uint16)   # some below the dark
    assert (frames < dark[None]).any()
    for i in range(nf):
        tifffile.imwrite(tmp_path / f"scan_{i + 1:05d}.tif", frames[i])
    tifffile.imwrite(tmp_path / "dark.tif", dark)

    def zip_scan(out, **extra):
        store = zarr.ZipStore(str(out), mode="w")
        root = zarr.group(store=store, overwrite=True)
        z_groups = {"exc": root.create_group("exchange")}
        config = {"dataFN": str(tmp_path / "scan_00001.tif"), "darkFN": str(tmp_path / "dark.tif"),
                  "numFilesPerScan": nf, "SkipFrame": 0, "PixelValue": 2, "numPxY": n, "numPxZ": n, **extra}
        ff_zip.process_multifile_scan("tiff", config, z_groups)
        store.close()

    default, arc = tmp_path / "default.zip", tmp_path / "arc.zip"
    zip_scan(default)
    clipped = zarr.open(zarr.ZipStore(str(default), mode="r"), mode="r")["exchange/data"][()]
    assert not np.array_equal(clipped, frames)                    # the default is lossy

    zip_scan(arc, archiveLossless="dark")
    archive.convert_zip(arc, None, mode="dark", log=lambda *a: None)
    assert np.array_equal(archive.read_frames(arc), frames)


def test_raw_verdict_with_dst_copies_the_source(tmp_path):
    rng = np.random.default_rng(8)
    data = rng.integers(0, 65536, (8, 32, 32)).astype(np.uint16)
    src, dst = tmp_path / "r.zip", tmp_path / "r.out.zip"
    _make_zip(src, data, np.zeros((2, 32, 32), np.uint16))
    name, _ = archive.convert_zip(src, dst, mode="auto", log=lambda *a: None)
    assert name == "raw"
    assert dst.read_bytes() == src.read_bytes()


def test_failed_conversion_leaves_no_partial_file(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    data, dark = _pedestal_scan(rng)
    src, dst = tmp_path / "f.zip", tmp_path / "f.arc.zip"
    _make_zip(src, data, dark)
    before = src.read_bytes()
    monkeypatch.setattr(archive, "zigzag_decode", lambda z, ref, dt: np.zeros_like(z).view(dt))
    for target in (dst, None):
        with pytest.raises(RuntimeError, match="round trip failed"):
            archive.convert_zip(src, target, mode="dark", log=lambda *a: None)
        assert not dst.exists() and not list(tmp_path.glob("*.archive.tmp"))
        assert src.read_bytes() == before


def test_scan_median_is_the_median_and_respects_a_tiny_budget():
    rng = np.random.default_rng(10)
    frames = rng.integers(0, 500, (40, 24, 16)).astype(np.uint16)
    z = zarr.array(frames, chunks=(1, 24, 16))
    med, n = archive.scan_median(z, nsub=64, budget=1e12)
    idx = archive._evenly(40, 64)
    assert n == len(idx) == 40
    assert np.array_equal(med, np.rint(np.median(frames[idx], axis=0)).astype(np.uint16))
    _, n_small = archive.scan_median(z, nsub=64, budget=1)      # budget below one frame: the 8-frame floor
    assert n_small == 8
