"""Issue #12: the frame-merge keys must travel from the archive to the merge.

``read_zarr_params`` carried ``OverlapLength`` / ``UsePixelOverlap`` /
``UseMaximaPositions`` as dataclass fields but never read them, so the FF
merge always ran with (2.0, 0, 0) whatever the params file said -- including
``LocalMaximaOnly 1``, whose zipper-forced OverlapLength 3 / UseMaximaPositions
1 were silently ignored. Archives without the keys must read exactly as before.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from midas_transforms.params import ZarrParams, read_zarr_params

AP = "analysis/process/analysis_parameters"

_REQUIRED = {
    "Lsd": np.array([1_000_000.0]),
    "Wavelength": np.array([0.172979]),
    "PixelSize": np.array([200.0]),
    "YCen": np.array([1024.0]),
    "ZCen": np.array([1024.0]),
    "tx": np.array([0.0]),
    "ty": np.array([0.1]),
    "tz": np.array([-0.2]),
    "LatticeParameter": np.array([4.08, 4.08, 4.08, 90.0, 90.0, 90.0]),
    "RingThresh": np.array([[1.0, 80.0], [2.0, 80.0]]),
}


def _make_zarr(path: Path, extra: dict | None = None) -> Path:
    with zarr.ZipStore(str(path), mode="w") as store:
        root = zarr.group(store=store)
        ap = root.require_group(AP)
        for k, v in {**_REQUIRED, **(extra or {})}.items():
            ap.create_dataset(k, data=v)
    return path


def _fields(zp: ZarrParams) -> dict:
    d = dataclasses.asdict(zp)
    d.pop("dist_coeffs_v2", None)   # array; compared separately
    return d


def test_non_default_merge_keys_are_read(tmp_path):
    z = _make_zarr(tmp_path / "a.zip", {
        "OverlapLength": np.array([3.5], dtype=np.double),
        "UsePixelOverlap": np.array([1], dtype=np.int32),
        "UseMaximaPositions": np.array([1], dtype=np.int32),
    })
    zp = read_zarr_params(z)
    assert zp.OverlapLength == 3.5
    assert zp.UsePixelOverlap == 1
    assert zp.UseMaximaPositions == 1


def test_absent_merge_keys_keep_old_defaults_bit_identical(tmp_path):
    zp = read_zarr_params(_make_zarr(tmp_path / "old.zip"))
    assert (zp.OverlapLength, zp.UsePixelOverlap, zp.UseMaximaPositions) == (2.0, 0, 0)
    # Every field of an old archive equals what the dataclass default gives
    # for these keys, and adding the keys changes nothing else.
    zp_new = read_zarr_params(_make_zarr(tmp_path / "new.zip", {
        "OverlapLength": np.array([3.0]),
        "UsePixelOverlap": np.array([1], dtype=np.int32),
        "UseMaximaPositions": np.array([1], dtype=np.int32),
    }))
    a, b = _fields(zp), _fields(zp_new)
    for k in ("OverlapLength", "UsePixelOverlap", "UseMaximaPositions"):
        a.pop(k), b.pop(k)
    assert a == b
    np.testing.assert_array_equal(np.asarray(zp.dist_coeffs_v2),
                                  np.asarray(zp_new.dist_coeffs_v2))


def test_defaults_agree_with_registry():
    reg = pytest.importorskip("midas_params.registry")
    by_name = {s.name: s for s in reg.PARAMS}
    zp = ZarrParams()
    assert float(by_name["OverlapLength"].default) == zp.OverlapLength
    assert int(by_name["UsePixelOverlap"].default) == zp.UsePixelOverlap
    assert int(by_name["UseMaximaPositions"].default) == zp.UseMaximaPositions


def _zip_from_params_text(tmp_path: Path, body: str) -> Path:
    ff_zip = pytest.importorskip("midas_zipper.ff_zip")
    pf = tmp_path / "ps.txt"
    pf.write_text(body)
    config = ff_zip.parse_parameter_file(str(pf))
    out = tmp_path / "z.zip"
    with zarr.ZipStore(str(out), mode="w") as store:
        root = zarr.group(store=store)
        groups = ff_zip.create_zarr_structure(root)
        ff_zip.write_analysis_parameters(groups, config)
    return out


_BASE_TEXT = "\n".join([
    "Lsd 1000000", "Wavelength 0.172979", "px 200", "BC 1024 1024",
    "tx 0", "ty 0.1", "tz -0.2", "LatticeConstant 4.08 4.08 4.08 90 90 90",
    "RingThresh 1 80", "RingThresh 2 80", "OmegaStep 0.25", "OmegaStart 0",
]) + "\n"


def test_zipper_writes_merge_keys_and_reader_sees_them(tmp_path):
    z = _zip_from_params_text(
        tmp_path,
        _BASE_TEXT + "OverlapLength 2.5\nUsePixelOverlap 1\nUseMaximaPositions 1\n",
    )
    with zarr.ZipStore(str(z), mode="r") as store:
        ap = zarr.group(store=store)[AP]
        assert ap["OverlapLength"].dtype == np.float64
        assert ap["UsePixelOverlap"].dtype == np.int32
        assert ap["UseMaximaPositions"].dtype == np.int32
    zp = read_zarr_params(z)
    assert (zp.OverlapLength, zp.UsePixelOverlap, zp.UseMaximaPositions) == (2.5, 1, 1)


def test_local_maxima_only_forcing_reaches_the_merge(tmp_path):
    z = _zip_from_params_text(tmp_path, _BASE_TEXT + "LocalMaximaOnly 1\n")
    zp = read_zarr_params(z)
    assert zp.OverlapLength == 3.0
    assert zp.UseMaximaPositions == 1


def test_local_maxima_only_overrides_explicit_merge_keys(tmp_path):
    """The zipper's forcing rewrites keys already written from the params file
    (a duplicate zip entry in a ZipStore); the forced value is what is read."""
    z = _zip_from_params_text(
        tmp_path,
        _BASE_TEXT + "LocalMaximaOnly 1\nOverlapLength 2.5\nUseMaximaPositions 0\n",
    )
    zp = read_zarr_params(z)
    assert zp.OverlapLength == 3.0
    assert zp.UseMaximaPositions == 1


def test_pipeline_passes_archive_overlap_length_to_merge(tmp_path, monkeypatch):
    """Pipeline.from_zarr(...).run() is the FF path in midas_pipeline."""
    import midas_transforms.pipeline as P

    z = _make_zarr(tmp_path / "a.zip", {"OverlapLength": np.array([4.25])})
    seen = {}

    class _Stop(Exception):
        pass

    def _fake_merge(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(P, "merge_overlapping_peaks", _fake_merge)
    dummy = tmp_path / "AllPeaks_PS.bin"
    dummy.write_bytes(b"")

    with pytest.raises(_Stop):
        P.Pipeline.from_zarr(z, allpeaks_ps_bin=dummy, result_folder=tmp_path,
                             device="cpu", dtype="float64").run()
    assert seen["overlap_length"] == 4.25

    # An explicit override still wins over the archive.
    seen.clear()
    with pytest.raises(_Stop):
        P.Pipeline.from_zarr(z, allpeaks_ps_bin=dummy, result_folder=tmp_path,
                             device="cpu", dtype="float64",
                             overlap_length=1.5).run()
    assert seen["overlap_length"] == 1.5
