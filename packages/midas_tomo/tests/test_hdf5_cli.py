"""HDF5 /exchange reading, the NXtomo-flavoured writer, and CLI wiring."""

from __future__ import annotations

import numpy as np
import pytest

from midas_tomo.hdf5 import crop_slice

h5py = pytest.importorskip("h5py")

from midas_tomo.hdf5 import read_exchange, write_recon_hdf5  # noqa: E402


# ------------------------------------------------------------- crop_slice
def test_crop_slice_zero_right_keeps_the_tail():
    # The whole point: arr[left:-0] is empty, which is how the legacy driver
    # silently reconstructed nothing when CropXR was 0.
    a = np.arange(10)
    assert list(a[crop_slice(2, 0, 10)]) == list(range(2, 10))


def test_crop_slice_normal():
    a = np.arange(10)
    assert list(a[crop_slice(2, 3, 10)]) == [2, 3, 4, 5, 6]


def test_crop_slice_rejects_over_crop():
    with pytest.raises(ValueError, match="leaves nothing"):
        crop_slice(6, 6, 10)


def test_crop_slice_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        crop_slice(-1, 0, 10)


# ---------------------------------------------------------------- fixture
def _write_exchange(path, *, n_frames=8, nz=6, nx=10,
                    cropxl=1, cropxr=0, cropzl=1, cropzr=1,
                    shift=1.5, rotation=None):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("exchange/data",
                          data=rng.integers(0, 4000, (n_frames, nz, nx), dtype=np.uint16))
        hf.create_dataset("exchange/dark",
                          data=rng.normal(100, 5, (nz, nx)).astype(np.float32))
        hf.create_dataset("exchange/bright",
                          data=rng.normal(4000, 50, (2, nz, nx)).astype(np.float32))
        p = "analysis/process/analysis_parameters"
        hf.create_dataset(f"{p}/CropXL", data=np.array([cropxl]))
        hf.create_dataset(f"{p}/CropXR", data=np.array([cropxr]))
        hf.create_dataset(f"{p}/CropZL", data=np.array([cropzl]))
        hf.create_dataset(f"{p}/CropZR", data=np.array([cropzr]))
        hf.create_dataset(f"{p}/shift", data=np.array([shift]))
        if rotation is not None:
            hf.create_dataset(f"{p}/RotationAngle", data=np.array([rotation]))
        s = "measurement/process/scan_parameters"
        hf.create_dataset(f"{s}/start", data=np.array([0.0]))
        hf.create_dataset(f"{s}/step", data=np.array([180.0 / n_frames]))
    return path


# ---------------------------------------------------------------- reading
def test_read_exchange_shapes_and_angles(tmp_path):
    fn = _write_exchange(tmp_path / "scan.h5", n_frames=8, nz=6, nx=10,
                         cropxl=1, cropxr=2, cropzl=1, cropzr=1)
    scan = read_exchange(fn)
    assert scan.data.shape == (8, 4, 7)     # nz 6-1-1=4, nx 10-1-2=7
    assert scan.dark.shape == (4, 7)
    assert scan.whites.shape == (2, 4, 7)
    assert scan.det_xdim == 7 and scan.det_ydim == 4
    assert scan.n_frames == 8
    assert scan.shift == pytest.approx(1.5)
    assert scan.angles[0] == pytest.approx(0.0)
    assert scan.angles[-1] == pytest.approx(180.0 / 8 * 7)


def test_read_exchange_zero_right_crop_is_not_empty(tmp_path):
    # Regression for the legacy arr[left:-0] bug.
    fn = _write_exchange(tmp_path / "scan.h5", nz=6, nx=10, cropxr=0, cropzr=0,
                         cropxl=1, cropzl=1)
    scan = read_exchange(fn)
    assert scan.data.size > 0
    assert scan.dark.shape == (5, 9)


def test_read_exchange_slab(tmp_path):
    fn = _write_exchange(tmp_path / "scan.h5", nz=10, cropzl=1, cropzr=1)
    scan = read_exchange(fn, slab=(2, 6))
    assert scan.data.shape[1] == 4


def test_read_exchange_slab_must_be_even(tmp_path):
    # The engine reconstructs slices in pairs.
    fn = _write_exchange(tmp_path / "scan.h5", nz=10)
    with pytest.raises(ValueError, match="even number of slices"):
        read_exchange(fn, slab=(2, 5))


def test_read_exchange_missing_param_names_the_key(tmp_path):
    fn = tmp_path / "bad.h5"
    with h5py.File(fn, "w") as hf:
        hf.create_dataset("exchange/dark", data=np.zeros((4, 4), np.float32))
    with pytest.raises(KeyError, match="CropXL"):
        read_exchange(fn)


def test_rotation_is_skipped_when_zero(tmp_path):
    # RotationAngle == 0 must not pull in SciPy.
    fn = _write_exchange(tmp_path / "scan.h5", rotation=0.0)
    scan = read_exchange(fn)
    assert scan.rotation_angle == 0.0


def test_rotation_applied_when_requested(tmp_path):
    pytest.importorskip("scipy")
    fn = _write_exchange(tmp_path / "scan.h5", nz=8, nx=8, rotation=90.0,
                         cropxl=0, cropxr=0, cropzl=0, cropzr=0)
    straight = read_exchange(fn, apply_rotation=False)
    rolled = read_exchange(fn, apply_rotation=True)
    assert rolled.dark.shape == straight.dark.shape
    assert not np.allclose(rolled.dark, straight.dark)


# ---------------------------------------------------------------- writing
def test_write_recon_hdf5_round_trip(tmp_path):
    recon = np.arange(2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 3, 8, 8)
    angles = np.linspace(0, 180, 12)
    out = write_recon_hdf5(tmp_path / "r.h5", recon, angles=angles,
                           shifts=np.array([-1.0, 1.0]),
                           metadata={"source": "test.h5", "filter": 2})
    with h5py.File(out, "r") as hf:
        np.testing.assert_array_equal(hf["entry/reconstruction/data"][...], recon)
        np.testing.assert_allclose(hf["entry/reconstruction/rotation_angle"][...], angles)
        assert hf["entry"].attrs["definition"] == "NXtomoproc"
        assert hf["entry"].attrs["source"] == "test.h5"
        assert hf["entry/reconstruction/data"].attrs["axes"] == "shift:slice:y:x"


def test_write_recon_hdf5_rejects_wrong_rank(tmp_path):
    with pytest.raises(ValueError, match="recon must be 4-D"):
        write_recon_hdf5(tmp_path / "r.h5", np.zeros((3, 8, 8), np.float32))


# -------------------------------------------------------------------- CLI
def test_cli_reports_missing_backend(capsys, monkeypatch):
    from midas_tomo import cli

    monkeypatch.setattr(cli.backend_c, "available", lambda **kw: False)
    monkeypatch.setattr(cli.backend_c, "why_unavailable", lambda **kw: "no binary here")
    rc = cli.main(["-dataFN", "nope.h5", "-nCPUs", "2"])
    assert rc == 2
    assert "no binary here" in capsys.readouterr().err


def test_cli_refuses_unsupported_deterministic(capsys, monkeypatch, tmp_path):
    from midas_tomo import cli

    monkeypatch.setattr(cli.backend_c, "available", lambda **kw: True)
    monkeypatch.setattr(cli.backend_c, "supports_deterministic", lambda **kw: False)
    rc = cli.main(["-dataFN", str(tmp_path / "x.h5"), "-nCPUs", "2", "--deterministic"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ignored silently" in err


def test_cli_ring_removal_defaults_to_off():
    # Behaviour change from the legacy driver, which always wrote
    # `ringRemovalCoefficient 1.0`. Pinning it so it is not re-introduced.
    from midas_tomo.cli import _build_parser

    args = _build_parser().parse_args(["-dataFN", "x.h5", "-nCPUs", "4"])
    assert args.ringRemoval == 0.0


def test_cli_cleanup_flags_are_mutually_exclusive():
    from midas_tomo.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["-dataFN", "x.h5", "-nCPUs", "4", "--tuneCleanup", "--cleanup", "3", "31", "11"]
        )


# ------------------------------------------------------- streaming stager
def test_stage_matches_in_memory_staging(tmp_path):
    """Streaming to disk must produce byte-identical input to the in-memory path.

    This is what lets the C be built without HDF5: if the two staging routes
    disagreed, "read it in Python instead" would be a different pipeline
    rather than the same one.
    """
    from midas_tomo.hdf5 import stage_exchange_to_binary

    fn = _write_exchange(tmp_path / "scan.h5", n_frames=10, nz=6, nx=10,
                         cropxl=1, cropxr=2, cropzl=1, cropzr=1)

    scan = read_exchange(fn)
    in_memory = tmp_path / "mem.bin"
    with in_memory.open("wb") as f:
        scan.dark.astype(np.float32).tofile(f)
        scan.whites.astype(np.float32).tofile(f)
        scan.data.astype(np.uint16).tofile(f)

    streamed = tmp_path / "stream.bin"
    meta = stage_exchange_to_binary(fn, streamed, frames_per_chunk=3)

    assert streamed.read_bytes() == in_memory.read_bytes()
    assert meta.n_frames == 10
    assert meta.data.size == 0, "projections should be on disk, not in RAM"
    np.testing.assert_array_equal(meta.dark, scan.dark)
    np.testing.assert_allclose(meta.angles, scan.angles)


def test_stage_chunk_size_does_not_change_output(tmp_path):
    from midas_tomo.hdf5 import stage_exchange_to_binary

    fn = _write_exchange(tmp_path / "scan.h5", n_frames=9, nz=6, nx=8)
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    stage_exchange_to_binary(fn, a, frames_per_chunk=2)
    stage_exchange_to_binary(fn, b, frames_per_chunk=100)
    assert a.read_bytes() == b.read_bytes()
