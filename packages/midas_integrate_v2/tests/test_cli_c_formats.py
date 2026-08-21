"""CLI wiring for the two C-era output formats: --out-format zarr / h5-stacked.

The unit-level layout checks live in test_zarr_gsas.py / test_h5_stacked.py;
these drive the whole batch CLI so the geometry, the area weights and the
frame ordering are exercised end to end.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import spec_from_v1_params, v1_params_from_spec
from midas_integrate_v2.cli import batch_main


def _spec(NY=24, NZ=24):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    return spec_from_v1_params(p, requires_grad=False)


def _stack(N=6, NY=24, NZ=24, *, peak_R_px=6.0):
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - NY / 2.0 - 0.37)
    Zc = (zz - NZ / 2.0 + 0.41)
    R = np.sqrt(Yc * Yc + Zc * Zc)
    base = np.exp(-(R - peak_R_px) ** 2 / 2.0)
    return np.stack([base * (i + 1) for i in range(N)], axis=0)


def _paramstest(tmp_path, s):
    p1 = v1_params_from_spec(s)
    path = tmp_path / "p.txt"
    path.write_text(
        f"NrPixelsY {p1.NrPixelsY}\nNrPixelsZ {p1.NrPixelsZ}\n"
        f"px {p1.pxY}\nLsd {p1.Lsd}\nBC {p1.BC_y} {p1.BC_z}\n"
        f"tx 0\nty 0\ntz 0\nRhoD {p1.RhoD}\n"
        f"Wavelength 1.0\nParallax 0\n"
        f"RMin {p1.RMin}\nRMax {p1.RMax}\nRBinSize {p1.RBinSize}\n"
        f"EtaMin {p1.EtaMin}\nEtaMax {p1.EtaMax}\n"
        f"EtaBinSize {p1.EtaBinSize}\n"
    )
    return path


def _h5_input(tmp_path, s, n=6):
    import h5py
    stack = _stack(N=n, NY=s.NrPixelsY, NZ=s.NrPixelsZ)
    path = tmp_path / "in.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=stack)
    return path, stack


def test_cli_zarr_output(tmp_path):
    zarr = pytest.importorskip("zarr")
    if int(zarr.__version__.split(".")[0]) >= 3:
        pytest.skip("MIDAS zarr layout is zarr-format 2")
    pytest.importorskip("h5py")
    s = _spec()
    h5_in, _ = _h5_input(tmp_path, s)
    out_dir = tmp_path / "out"
    rc = batch_main([
        str(_paramstest(tmp_path, s)),
        "--hdf5", str(h5_in), "--mode", "hard",
        "--out-dir", str(out_dir), "--out-format", "zarr",
        "--omega-sum-frames", "2",
        "--omega-start", "0", "--omega-step", "0.25",
    ])
    assert rc == 0
    out = out_dir / "integrated.zarr.zip"
    assert out.exists()

    z = zarr.open(str(out), mode="r")
    assert set(z.keys()) >= {"REtaMap", "OmegaSumFrame", "Omegas",
                             "SumFrames", "InstrumentParameters"}
    assert z["REtaMap"].shape == (5, s.n_r_bins, s.n_eta_bins)
    # 6 frames in chunks of 2
    assert len(list(z["OmegaSumFrame"].keys())) == 3
    assert np.allclose(np.asarray(z["Omegas"]),
                       [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    # The CLI must populate the area row from the real geometry, not zeros.
    assert np.asarray(z["REtaMap"])[3].sum() > 0
    # Instrument parameters come from the paramstest, not the defaults.
    assert np.asarray(z["InstrumentParameters"]["Lam"])[0] == pytest.approx(1.0)
    assert np.asarray(z["InstrumentParameters"]["Distance"])[0] == \
        pytest.approx(float(s.Lsd))


def test_cli_h5_stacked_output(tmp_path):
    h5py = pytest.importorskip("h5py")
    s = _spec()
    h5_in, _ = _h5_input(tmp_path, s)
    out_dir = tmp_path / "out"
    rc = batch_main([
        str(_paramstest(tmp_path, s)),
        "--hdf5", str(h5_in), "--mode", "hard",
        "--out-dir", str(out_dir), "--out-format", "h5-stacked",
        "--omega-sum-frames", "3",
    ])
    assert rc == 0
    out = out_dir / "integrated_stacked.h5"
    assert out.exists()

    with h5py.File(out, "r") as f:
        assert f["OmegaSumFrame"].shape == (2, s.n_r_bins, s.n_eta_bins)
        assert f["lineouts"].shape == (6, s.n_r_bins)
        assert f["lineouts_simple_mean"].shape == (6, s.n_r_bins)
        # The two lineouts are the C's area-weighted and unweighted means;
        # if they were identical one of them would be wrong.
        assert not np.allclose(f["lineouts"][:], f["lineouts_simple_mean"][:])
        assert f["frame_names"].shape == (6,)
        assert set(f["geometry_maps"].keys()) == {
            "R_map", "TTh_map", "Eta_map", "Area_map", "Q_map"}
        assert f["geometry_maps"]["Area_map"][()].sum() > 0
        assert "metadata" in f.attrs


def test_cli_zarr_and_stacked_agree_on_the_summed_cake(tmp_path):
    """Both formats carry the same 2-D data; a transpose slip in one would
    show up here even though each format's own test passes."""
    zarr = pytest.importorskip("zarr")
    if int(zarr.__version__.split(".")[0]) >= 3:
        pytest.skip("MIDAS zarr layout is zarr-format 2")
    h5py = pytest.importorskip("h5py")
    s = _spec()
    h5_in, _ = _h5_input(tmp_path, s)
    params = _paramstest(tmp_path, s)

    common = ["--hdf5", str(h5_in), "--mode", "hard",
              "--omega-sum-frames", "-1"]
    d1, d2 = tmp_path / "a", tmp_path / "b"
    assert batch_main([str(params), *common, "--out-dir", str(d1),
                       "--out-format", "zarr"]) == 0
    assert batch_main([str(params), *common, "--out-dir", str(d2),
                       "--out-format", "h5-stacked"]) == 0

    z = zarr.open(str(d1 / "integrated.zarr.zip"), mode="r")
    zsum = np.asarray(z["OmegaSumFrame"]["LastFrameNumber_5"])
    with h5py.File(d2 / "integrated_stacked.h5", "r") as f:
        hsum = f["OmegaSumFrame"][0]
    assert zsum.shape == hsum.shape == (s.n_r_bins, s.n_eta_bins)
    assert np.allclose(zsum, hsum)
    # ...and both equal /SumFrames, which accumulates independently.
    assert np.allclose(zsum, np.asarray(z["SumFrames"]))
