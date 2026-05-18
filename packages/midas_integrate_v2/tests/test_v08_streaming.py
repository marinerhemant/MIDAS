"""v0.8: streaming + outlier rejection + normalisation + HDF5 + CLI batch."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    spec_from_v1_params,
    NumpyArraySource, TIFFGlobSource,
    HDF5FrameSource, ZarrFrameSource,
    FrameNormalizer, reject_cosmic_rays, integrate_stream,
    write_h5,
)


def _spec(NY=24, NZ=24):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    return spec_from_v1_params(p, requires_grad=False)


def _stack(N=5, NY=24, NZ=24, *, peak_R_px=6.0):
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - NY / 2.0 - 0.37)
    Zc = (zz - NZ / 2.0 + 0.41)
    R = np.sqrt(Yc * Yc + Zc * Zc)
    base = np.exp(-(R - peak_R_px) ** 2 / 2.0)
    return np.stack([base * (i + 1) for i in range(N)], axis=0)


# ── NumpyArraySource ──

def test_numpy_source_iterates():
    stack = _stack(N=4)
    src = NumpyArraySource(stack)
    assert src.n_frames == 4
    assert src.frame_shape == (24, 24)
    seen = list(src)
    assert len(seen) == 4
    assert all(isinstance(fid, str) for fid, _ in seen)


def test_numpy_source_random_access():
    stack = _stack(N=4)
    src = NumpyArraySource(stack)
    fid, img = src.get(2)
    np.testing.assert_array_equal(img, stack[2])


def test_numpy_source_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="3-D"):
        NumpyArraySource(np.zeros((24, 24)))


# ── TIFFGlobSource ──

def test_tiff_glob_source(tmp_path):
    pytest.importorskip("tifffile")
    import tifffile
    stack = _stack(N=3, NY=16, NZ=16)
    for i in range(3):
        tifffile.imwrite(tmp_path / f"frame_{i:03d}.tif", stack[i])
    src = TIFFGlobSource(str(tmp_path / "frame_*.tif"))
    assert src.n_frames == 3
    assert src.frame_shape == (16, 16)
    fids = [fid for fid, _ in src]
    assert fids == ["frame_000", "frame_001", "frame_002"]


def test_tiff_glob_source_no_files_raises(tmp_path):
    pytest.importorskip("tifffile")
    with pytest.raises(FileNotFoundError):
        TIFFGlobSource(str(tmp_path / "noexist*.tif"))


# ── HDF5FrameSource ──

def test_hdf5_frame_source(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    stack = _stack(N=5, NY=16, NZ=16)
    p = tmp_path / "frames.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("frames", data=stack)
        f.create_dataset("ids",
                          data=np.array([f"f{i}" for i in range(5)],
                                          dtype="S"))
    src = HDF5FrameSource(p, dataset="frames", ids_dataset="ids")
    assert src.n_frames == 5
    fids = [fid for fid, _ in src]
    assert fids == ["f0", "f1", "f2", "f3", "f4"]


def test_hdf5_frame_source_missing_dataset_raises(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    p = tmp_path / "frames.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("other", data=np.zeros((3, 8, 8)))
    with pytest.raises(KeyError):
        HDF5FrameSource(p, dataset="frames")


# ── FrameNormalizer ──

def test_normalizer_default_formula():
    img = np.full((4, 4), 100.0)
    norm = FrameNormalizer(
        monitor={"f0": 10.0},
        exposure_s={"f0": 2.0},
        transmission={"f0": 0.5},
    )
    out = norm("f0", img)
    # 100 / (10 * 2 * 0.5) = 10
    np.testing.assert_array_almost_equal(out, 10.0)


def test_normalizer_dark_subtraction():
    img = np.full((4, 4), 100.0)
    dark = np.full((4, 4), 5.0)
    norm = FrameNormalizer(dark=dark)
    out = norm("f0", img)
    # No factors set ⇒ defaults of 1.0; (100 - 5) / 1 = 95
    np.testing.assert_array_almost_equal(out, 95.0)


def test_normalizer_clip_negatives():
    img = np.zeros((4, 4))
    dark = np.full((4, 4), 10.0)
    norm = FrameNormalizer(dark=dark, clip_negatives=True)
    out = norm("f0", img)
    np.testing.assert_array_almost_equal(out, 0.0)
    norm2 = FrameNormalizer(dark=dark, clip_negatives=False)
    out2 = norm2("f0", img)
    np.testing.assert_array_almost_equal(out2, -10.0)


def test_normalizer_custom_formula():
    img = np.full((4, 4), 100.0)
    # Custom: I / (m + e + t) instead of I / (m * e * t)
    formula = lambda I, m, e, t: I / (m + e + t)
    norm = FrameNormalizer(
        monitor={"f0": 1.0}, exposure_s={"f0": 2.0},
        transmission={"f0": 1.0},
        formula=formula,
    )
    out = norm("f0", img)
    np.testing.assert_array_almost_equal(out, 25.0)


# ── reject_cosmic_rays ──

def test_outlier_rejection_recovers_planted_cosmics():
    rng = np.random.default_rng(0)
    base = rng.normal(100, 5, size=(8, 16, 16))
    # Plant 3 cosmic-ray-like outliers
    base[2, 5, 5] = 5000
    base[6, 10, 10] = -3000
    base[4, 8, 12] = 7000
    cleaned, mask = reject_cosmic_rays(base, n_sigma=5.0,
                                          mode="replace_with_median")
    assert mask[2, 5, 5]
    assert mask[6, 10, 10]
    assert mask[4, 8, 12]
    # Cleaned values should be near the median (~100)
    assert abs(cleaned[2, 5, 5] - 100) < 20
    # Most "good" pixels not flagged
    assert mask.sum() < 30


def test_outlier_rejection_flag_only_doesnt_modify():
    rng = np.random.default_rng(1)
    base = rng.normal(100, 5, size=(8, 12, 12))
    base[3, 4, 4] = 5000
    cleaned, mask = reject_cosmic_rays(base, mode="flag_only")
    np.testing.assert_array_equal(cleaned, base)
    assert mask[3, 4, 4]


def test_outlier_rejection_replace_with_nan():
    rng = np.random.default_rng(2)
    base = rng.normal(100, 5, size=(8, 8, 8))
    base[2, 2, 2] = 5000
    cleaned, _ = reject_cosmic_rays(base, mode="replace_with_nan")
    assert np.isnan(cleaned[2, 2, 2])


def test_outlier_rejection_too_few_frames_raises():
    base = np.zeros((2, 8, 8))
    with pytest.raises(ValueError, match="N>=3"):
        reject_cosmic_rays(base)


def test_outlier_rejection_unknown_mode_raises():
    base = np.zeros((4, 8, 8))
    with pytest.raises(ValueError, match="unknown mode"):
        reject_cosmic_rays(base, mode="something")


# ── integrate_stream ──

def test_integrate_stream_against_numpy_source():
    s = _spec()
    stack = _stack(N=4, NY=s.NrPixelsY, NZ=s.NrPixelsZ)
    src = NumpyArraySource(stack)
    out = integrate_stream(s, src, mode="hard")
    assert out["n_processed"] == 4
    assert out["profiles"].shape == (4, s.n_r_bins)
    # Profiles should be increasing in i (we scaled the stack by (i+1))
    sums = out["profiles"].sum(axis=1)
    assert (np.diff(sums) > 0).all()


def test_integrate_stream_with_writer_callback():
    s = _spec()
    stack = _stack(N=3, NY=s.NrPixelsY, NZ=s.NrPixelsZ)
    src = NumpyArraySource(stack)
    seen = []
    def writer(fid, r_axis, prof):
        seen.append((fid, prof.shape))
    out = integrate_stream(s, src, mode="hard", writer=writer)
    assert out["n_processed"] == 3
    assert len(seen) == 3
    assert "profiles" not in out


def test_integrate_stream_normaliser_applies():
    s = _spec()
    stack = _stack(N=2, NY=s.NrPixelsY, NZ=s.NrPixelsZ) * 10
    src = NumpyArraySource(stack, ids=["a", "b"])
    norm = FrameNormalizer(exposure_s={"a": 1.0, "b": 2.0})
    out = integrate_stream(s, src, mode="hard", normaliser=norm)
    # Frame "b" was scaled by 1.5 (relative to original) before
    # normalisation, then divided by 2 → ratio b/a should be ~ (2 * 1)/(1 * 2)
    # = 1.  With unit normalisation but doubled raw intensity, b stays
    # at 2 * raw_a / 2 = raw_a. Sums should be roughly equal.
    sums = out["profiles"].sum(axis=1)
    ratio = sums[1] / max(1e-30, sums[0])
    # raw ratio is 2 (b is twice a); normalisation by exposure 1 vs 2
    # cancels → ratio ≈ 1
    assert abs(ratio - 1.0) < 0.05


def test_integrate_stream_polygon_mode():
    s = _spec()
    stack = _stack(N=2, NY=s.NrPixelsY, NZ=s.NrPixelsZ)
    src = NumpyArraySource(stack)
    out = integrate_stream(s, src, mode="polygon")
    assert out["profiles"].shape == (2, s.n_r_bins)


def test_integrate_stream_shape_mismatch_raises():
    s = _spec()
    stack = np.zeros((3, s.NrPixelsZ + 1, s.NrPixelsY))
    src = NumpyArraySource(stack)
    with pytest.raises(ValueError, match="shape"):
        integrate_stream(s, src, mode="hard")


def test_integrate_stream_unknown_mode_raises():
    s = _spec()
    stack = _stack(N=2, NY=s.NrPixelsY, NZ=s.NrPixelsZ)
    src = NumpyArraySource(stack)
    with pytest.raises(ValueError, match="unknown mode"):
        integrate_stream(s, src, mode="foo")


# ── HDF5 output ──

def test_write_h5_round_trips(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    profiles = np.random.rand(5, 100)
    sigmas   = np.sqrt(profiles)
    r_axis   = np.linspace(1, 100, 100)
    ids      = [f"f{i}" for i in range(5)]
    p = write_h5(tmp_path / "p.h5",
                  profiles=profiles, r_axis=r_axis,
                  sigmas=sigmas, frame_ids=ids)
    assert p.exists()
    with h5py.File(p, "r") as f:
        np.testing.assert_array_equal(f["profiles"][:], profiles)
        np.testing.assert_array_equal(f["sigmas"][:], sigmas)
        np.testing.assert_array_equal(f["r_axis_px"][:], r_axis)
        loaded_ids = [s.decode() for s in f["frame_ids"][:]]
        assert loaded_ids == ids
        # NeXus-strict layout (Item 33): file root is NXroot, the
        # signal group at /entry/data carries the NXdata class.
        assert f.attrs.get("NX_class") == "NXroot"
        assert f["entry"].attrs.get("NX_class") == "NXentry"
        assert f["entry/data"].attrs.get("NX_class") == "NXdata"
        assert f["entry/data"].attrs.get("signal") == "profiles"


def test_write_h5_with_provenance(tmp_path):
    from midas_integrate_v2 import build_provenance
    s = _spec()
    md = build_provenance(s, integrate_mode="polygon")
    profiles = np.random.rand(3, 50)
    p = write_h5(tmp_path / "p.h5",
                  profiles=profiles,
                  r_axis=np.linspace(1, 50, 50),
                  metadata=md)
    pytest.importorskip("h5py")
    import h5py, json
    with h5py.File(p, "r") as f:
        meta_json = f.attrs.get("metadata_json")
        assert meta_json is not None
        parsed = json.loads(meta_json)
        assert parsed["package"] == "midas_integrate_v2"
        assert parsed["integrate_mode"] == "polygon"


def test_write_h5_extra_datasets(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    profiles = np.random.rand(3, 50)
    extra = {"monitor": np.array([1.0, 1.1, 1.2]),
             "timestamps": np.array([100.0, 200.0, 300.0])}
    p = write_h5(tmp_path / "p.h5",
                  profiles=profiles,
                  r_axis=np.linspace(1, 50, 50),
                  extra_datasets=extra)
    with h5py.File(p, "r") as f:
        np.testing.assert_array_equal(f["monitor"][:], extra["monitor"])
        np.testing.assert_array_equal(f["timestamps"][:], extra["timestamps"])


# ── CLI batch ──

def test_cli_batch_help_runs():
    res = subprocess.run(
        [sys.executable, "-c",
          "from midas_integrate_v2.cli import batch_main; batch_main(['--help'])"],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    assert res.returncode == 0
    assert "midas-integrate-v2-batch" in res.stdout


def test_cli_batch_tiff_glob_to_csv(tmp_path):
    pytest.importorskip("tifffile")
    import tifffile
    s = _spec()
    stack = _stack(N=3, NY=s.NrPixelsY, NZ=s.NrPixelsZ)
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for i in range(3):
        tifffile.imwrite(frame_dir / f"img_{i:03d}.tif", stack[i])

    # Write minimal paramstest
    from midas_integrate_v2 import v1_params_from_spec
    p1 = v1_params_from_spec(s)
    paramstest = tmp_path / "p.txt"
    paramstest.write_text(
        f"NrPixelsY {p1.NrPixelsY}\nNrPixelsZ {p1.NrPixelsZ}\n"
        f"px {p1.pxY}\nLsd {p1.Lsd}\nBC {p1.BC_y} {p1.BC_z}\n"
        f"tx 0\nty 0\ntz 0\nRhoD {p1.RhoD}\n"
        f"Wavelength 1.0\nParallax 0\n"
        f"RMin {p1.RMin}\nRMax {p1.RMax}\nRBinSize {p1.RBinSize}\n"
        f"EtaMin {p1.EtaMin}\nEtaMax {p1.EtaMax}\n"
        f"EtaBinSize {p1.EtaBinSize}\n"
    )

    out_dir = tmp_path / "profiles"
    from midas_integrate_v2.cli import batch_main
    rc = batch_main([
        str(paramstest),
        "--image-glob", str(frame_dir / "img_*.tif"),
        "--mode", "hard",
        "--out-dir", str(out_dir),
        "--out-format", "csv",
    ])
    assert rc == 0
    csvs = sorted(out_dir.glob("*.csv"))
    assert len(csvs) == 3


def test_cli_batch_hdf5_to_h5(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    s = _spec()
    stack = _stack(N=4, NY=s.NrPixelsY, NZ=s.NrPixelsZ)
    h5_in = tmp_path / "in.h5"
    with h5py.File(h5_in, "w") as f:
        f.create_dataset("frames", data=stack)
    from midas_integrate_v2 import v1_params_from_spec
    p1 = v1_params_from_spec(s)
    paramstest = tmp_path / "p.txt"
    paramstest.write_text(
        f"NrPixelsY {p1.NrPixelsY}\nNrPixelsZ {p1.NrPixelsZ}\n"
        f"px {p1.pxY}\nLsd {p1.Lsd}\nBC {p1.BC_y} {p1.BC_z}\n"
        f"tx 0\nty 0\ntz 0\nRhoD {p1.RhoD}\n"
        f"Wavelength 1.0\nParallax 0\n"
        f"RMin {p1.RMin}\nRMax {p1.RMax}\nRBinSize {p1.RBinSize}\n"
        f"EtaMin {p1.EtaMin}\nEtaMax {p1.EtaMax}\n"
        f"EtaBinSize {p1.EtaBinSize}\n"
    )
    out_dir = tmp_path / "out"
    from midas_integrate_v2.cli import batch_main
    rc = batch_main([
        str(paramstest),
        "--hdf5", str(h5_in),
        "--mode", "hard",
        "--out-dir", str(out_dir),
        "--out-format", "h5",
    ])
    assert rc == 0
    out_h5 = out_dir / "profiles.h5"
    assert out_h5.exists()
    with h5py.File(out_h5, "r") as f:
        assert f["profiles"].shape == (4, s.n_r_bins)
