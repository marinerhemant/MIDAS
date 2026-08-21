"""I/O round-trip tests for csv + binary modules."""

from pathlib import Path

import numpy as np

from midas_transforms.io import binary as bio
from midas_transforms.io import csv as csv_io
from midas_transforms.params import (
    ParamsTest, ZarrParams, read_paramstest, write_paramstest,
)


def test_paramstest_roundtrip(tmp_path: Path, tiny_paramstest):
    f = tmp_path / "paramstest.txt"
    write_paramstest(tiny_paramstest, f)
    p2 = read_paramstest(f)
    assert p2.RingNumbers == tiny_paramstest.RingNumbers
    assert p2.RingRadii == tiny_paramstest.RingRadii
    assert p2.Wavelength == tiny_paramstest.Wavelength
    assert p2.Lsd == tiny_paramstest.Lsd
    assert p2.SpaceGroup == tiny_paramstest.SpaceGroup


def test_inputall_csv_roundtrip(tmp_path: Path, tiny_inputall):
    f = tmp_path / "InputAll.csv"
    csv_io.write_inputall_csv(f, tiny_inputall)
    a2 = csv_io.read_inputall_csv(f)
    np.testing.assert_allclose(a2, tiny_inputall, rtol=0, atol=1e-5)


def test_inputall_extra_csv_roundtrip(tmp_path: Path, tiny_inputall_extra):
    f = tmp_path / "InputAllExtraInfoFittingAll.csv"
    csv_io.write_inputall_extra_csv(f, tiny_inputall_extra)
    a2 = csv_io.read_inputall_extra_csv(f)
    np.testing.assert_allclose(a2, tiny_inputall_extra, rtol=0, atol=1e-5)


def test_spots_bin_roundtrip(tmp_path: Path):
    arr = np.random.default_rng(0).random((37, 9)).astype(np.float64)
    f = tmp_path / "Spots.bin"
    bio.write_spots_bin(f, arr)
    a2 = bio.read_spots_bin(f)
    np.testing.assert_array_equal(a2, arr)


def test_extrainfo_bin_roundtrip(tmp_path: Path):
    arr = np.random.default_rng(1).random((37, 16)).astype(np.float64)
    f = tmp_path / "ExtraInfo.bin"
    bio.write_extrainfo_bin(f, arr)
    a2 = bio.read_extrainfo_bin(f)
    np.testing.assert_array_equal(a2, arr)


def test_data_ndata_bin_roundtrip(tmp_path: Path):
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    ndata = np.array([[3, 0], [2, 3], [3, 5]], dtype=np.int32)
    df = tmp_path / "Data.bin"
    nf = tmp_path / "nData.bin"
    bio.write_data_ndata_bin(df, nf, data, ndata)
    np.testing.assert_array_equal(bio.read_data_bin(df), data)
    np.testing.assert_array_equal(bio.read_ndata_bin(nf), ndata)


def test_zarr_params_required_keys_missing(tmp_path: Path):
    """ZarrParams must raise KeyError listing required keys when none are present."""
    import zarr
    z = tmp_path / "empty.zip"
    with zarr.ZipStore(str(z), mode="w") as store:
        root = zarr.group(store=store)
        root.attrs["meta"] = "empty"

    from midas_transforms.params import read_zarr_params, REQUIRED_FITSETUP_KEYS
    import pytest
    with pytest.raises(KeyError) as exc:
        read_zarr_params(z)
    msg = str(exc.value)
    # Should mention each required key
    for k in REQUIRED_FITSETUP_KEYS:
        assert k in msg, f"missing required-key {k!r} in error message: {msg}"


def test_margstrain_roundtrips_to_paramstest(tmp_path: Path, tiny_paramstest):
    """MargStrain must reach the C refiner, which reads paramstest.

    The strain search box used to be a compiled-in +-0.01. It is now the
    `MargStrain` key, so it has to survive write -> read like MargABC does.
    """
    tiny_paramstest.MargStrain = 0.025
    f = tmp_path / "paramstest.txt"
    write_paramstest(tiny_paramstest, f)
    assert "MargStrain" in f.read_text()
    assert read_paramstest(f).MargStrain == 0.025


def test_margstrain_defaults_to_zero_meaning_c_default(tiny_paramstest):
    """0 is the 'unset' sentinel: the C side falls back to 0.01 (10000 ue)."""
    assert tiny_paramstest.MargStrain == 0.0
