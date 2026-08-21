"""The MIDAS .zarr.zip writer must reproduce the C integrator's layout.

Reference: ``IntegratorZarrOMP.c:2074-2168`` for the datasets and attributes,
and ``IntegratorZarrOMP.c:1755-1762`` for the REtaMap row definitions.
"""
import math

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
if int(zarr.__version__.split(".")[0]) >= 3:
    pytest.skip("MIDAS zarr layout is zarr-format 2; zarr<3 required",
                allow_module_level=True)

import torch

from midas_integrate_v2.io.zarr_gsas import (
    DEFAULT_INSTRUMENT_PARAMS,
    INSTRUMENT_PARAM_NAMES,
    GSASZarrWriter,
    instrument_params_from_spec,
    reta_map,
    write_gsas_zarr_zip,
)
from midas_integrate_v2.spec import IntegrationSpec


@pytest.fixture
def spec():
    """A small R-mode spec: 8 R bins x 6 eta bins."""
    return IntegrationSpec(
        Lsd=torch.tensor(1_000_000.0),        # um
        BC_y=torch.tensor(1024.0), BC_z=torch.tensor(1024.0),
        tx=torch.tensor(0.0), ty=torch.tensor(0.0), tz=torch.tensor(0.0),
        Wavelength=torch.tensor(0.413263),
        NrPixelsY=2048, NrPixelsZ=2048,
        pxY=200.0, pxZ=200.0,
        RMin=10.0, RMax=90.0, RBinSize=10.0,        # -> 8 bins
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,  # -> 6 bins
    )


def _cake(spec, seed):
    rng = np.random.default_rng(seed)
    return rng.random((spec.n_eta_bins, spec.n_r_bins)) * 100.0


def _open(path):
    return zarr.open(str(path), mode="r")


# ---------------------------------------------------------------- REtaMap


def test_reta_map_rows_match_the_c(spec):
    area = np.arange(spec.n_eta_bins * spec.n_r_bins, dtype=np.float64).reshape(
        spec.n_eta_bins, spec.n_r_bins)
    m = reta_map(spec, area)
    n_r, n_eta = spec.n_r_bins, spec.n_eta_bins
    assert m.shape == (5, n_r, n_eta)

    # Row 0: RMean = (RBinsLow + RBinsHigh)/2, uniform in R-mode.
    r_expected = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    assert np.allclose(m[0, :, 0], r_expected)
    # Constant along eta.
    assert np.allclose(m[0], r_expected[:, None])

    # Row 1: atand(RMean * px / Lsd)
    tth = np.degrees(np.arctan(r_expected * spec.pxY / float(spec.Lsd)))
    assert np.allclose(m[1, :, 0], tth)

    # Row 2: EtaMean = (EtaBinsLow + EtaBinsHigh)/2
    eta_expected = spec.EtaMin + spec.EtaBinSize * (np.arange(n_eta) + 0.5)
    assert np.allclose(m[2, 0, :], eta_expected)
    assert np.allclose(m[2], eta_expected[None, :])

    # Row 3: totArea, transposed from v2's (n_eta, n_r)
    assert np.allclose(m[3], area.T)

    # Row 4: Q = (4 pi / lam) sin(twotheta/2)
    lam = float(spec.Wavelength)
    q = (4.0 * math.pi / lam) * np.sin(np.arctan(
        r_expected * spec.pxY / float(spec.Lsd)) / 2.0)
    assert np.allclose(m[4, :, 0], q)


def test_reta_map_warns_without_area(spec):
    with pytest.warns(UserWarning, match="BinArea"):
        m = reta_map(spec, None)
    assert np.all(m[3] == 0.0)


def test_reta_map_accepts_either_area_orientation(spec):
    area = np.random.default_rng(0).random(
        (spec.n_eta_bins, spec.n_r_bins))
    assert np.allclose(reta_map(spec, area)[3], reta_map(spec, area.T)[3])


def test_reta_map_rejects_wrong_area_shape(spec):
    with pytest.raises(ValueError, match="neither"):
        reta_map(spec, np.zeros((3, 3)))


def test_q_mode_r_edges_follow_the_c_formula():
    s = IntegrationSpec(
        Lsd=torch.tensor(1_000_000.0),
        BC_y=torch.tensor(1024.0), BC_z=torch.tensor(1024.0),
        Wavelength=torch.tensor(0.5),
        NrPixelsY=2048, NrPixelsZ=2048, pxY=200.0, pxZ=200.0,
        RMin=10.0, RMax=90.0, RBinSize=10.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
        QMin=1.0, QMax=5.0, QBinSize=1.0,
    )
    assert s.q_mode_active
    with pytest.warns(UserWarning):
        m = reta_map(s, None)
    lam, lsd, px = 0.5, 1_000_000.0, 200.0
    i = np.arange(s.n_r_bins, dtype=float)
    lo = (lsd / px) * np.tan(2 * np.arcsin((1.0 + 1.0 * i) * lam / (4 * math.pi)))
    hi = (lsd / px) * np.tan(2 * np.arcsin((1.0 + 1.0 * (i + 1)) * lam / (4 * math.pi)))
    assert np.allclose(m[0, :, 0], 0.5 * (lo + hi))


# ------------------------------------------------------- instrument params


def test_instrument_params_from_spec(spec):
    p = instrument_params_from_spec(spec)
    assert p["Lam"] == pytest.approx(float(spec.Wavelength))
    assert p["Distance"] == pytest.approx(float(spec.Lsd))
    assert p["Polariz"] == pytest.approx(spec.PolarizationFraction)
    # The GSAS-II profile coefficients keep their documented defaults.
    for k in ("SH_L", "U", "V", "W", "X", "Y", "Z"):
        assert p[k] == DEFAULT_INSTRUMENT_PARAMS[k]


def test_instrument_params_overrides_and_validation(spec):
    p = instrument_params_from_spec(spec, {"U": 2.5})
    assert p["U"] == 2.5
    with pytest.raises(ValueError, match="unknown instrument parameter"):
        instrument_params_from_spec(spec, {"NotAParam": 1.0})


# ------------------------------------------------------------- full layout


def test_full_c_layout_round_trip(tmp_path, spec):
    cakes = [_cake(spec, i) for i in range(6)]
    area = np.ones((spec.n_eta_bins, spec.n_r_bins))
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(
        out, cakes, spec=spec, bin_area=area,
        omegas=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
        omega_sum_frames=2, individual_save=True, sum_images=True,
        temperatures=[300.0] * 6, pressures=[1.0] * 6,
        currents=[2.0] * 6, currents_i0=[4.0] * 6,
    )
    z = _open(out)

    # Every group the C wrote must be present.
    assert set(z.keys()) >= {"REtaMap", "IntegrationResult", "OmegaSumFrame",
                             "Omegas", "SumFrames", "InstrumentParameters"}

    # /REtaMap + its attrs
    assert z["REtaMap"].shape == (5, spec.n_r_bins, spec.n_eta_bins)
    a = dict(z["REtaMap"].attrs)
    assert a["nRBins"] == spec.n_r_bins
    assert a["nEtaBins"] == spec.n_eta_bins
    assert a["Header"] == "Radius,2Theta,Eta,BinArea,Q"
    assert a["Units"] == "Pixels,Degrees,Degrees,Pixels,InvAngstrom"

    # /IntegrationResult/FrameNr_<i> — transposed to (n_r, n_eta)
    assert sorted(z["IntegrationResult"].keys()) == sorted(
        f"FrameNr_{i}" for i in range(6))
    for i, c in enumerate(cakes):
        ds = z["IntegrationResult"][f"FrameNr_{i}"]
        assert ds.shape == (spec.n_r_bins, spec.n_eta_bins)
        assert np.allclose(np.asarray(ds), c.T)
        assert dict(ds.attrs)["Header"] == "Radius,Eta"
        assert dict(ds.attrs)["Units"] == "Pixels,Degrees"
    assert dict(z["IntegrationResult"]["FrameNr_3"].attrs)["omega"] == 0.75

    # /OmegaSumFrame — 6 frames in chunks of 2 -> last frames 1, 3, 5
    osf = z["OmegaSumFrame"]
    assert sorted(osf.keys()) == sorted(
        ["LastFrameNumber_1", "LastFrameNumber_3", "LastFrameNumber_5"])
    ds = osf["LastFrameNumber_3"]
    assert np.allclose(np.asarray(ds), (cakes[2] + cakes[3]).T)
    at = dict(ds.attrs)
    assert at["LastFrameNumber"] == 3
    assert at["Number Of Frames Summed"] == 2
    assert at["FirstOme"] == pytest.approx(0.5)
    assert at["LastOme"] == pytest.approx(0.75)
    assert at["Temperature"] == pytest.approx(300.0)
    assert at["Pressure"] == pytest.approx(1.0)
    assert at["I"] == pytest.approx(2.0)
    assert at["I0"] == pytest.approx(4.0)

    # /Omegas
    assert np.allclose(np.asarray(z["Omegas"]),
                       [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    assert dict(z["Omegas"].attrs)["Units"] == "Degrees"

    # /SumFrames
    assert np.allclose(np.asarray(z["SumFrames"]),
                       np.sum(cakes, axis=0).T)
    sa = dict(z["SumFrames"].attrs)
    assert sa["nFrames"] == 6
    assert sa["Header"] == "Radius,Eta"

    # /InstrumentParameters — each a 1-element array, as the C wrote it
    ip = z["InstrumentParameters"]
    assert sorted(ip.keys()) == sorted(INSTRUMENT_PARAM_NAMES)
    for k in INSTRUMENT_PARAM_NAMES:
        assert np.asarray(ip[k]).shape == (1,)
    assert np.asarray(ip["Lam"])[0] == pytest.approx(float(spec.Wavelength))
    assert np.asarray(ip["Distance"])[0] == pytest.approx(float(spec.Lsd))


def test_cake_transposed_to_c_convention(tmp_path, spec):
    """v2 returns (n_eta, n_r); the file must hold (n_r, n_eta)."""
    cake = _cake(spec, 7)
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, [cake], spec=spec, bin_area=np.ones_like(cake),
                        individual_save=True)
    ds = _open(out)["IntegrationResult"]["FrameNr_0"]
    assert ds.shape == (spec.n_r_bins, spec.n_eta_bins)
    assert np.allclose(np.asarray(ds), cake.T)


def test_trailing_partial_chunk_is_written(tmp_path, spec):
    """5 frames in chunks of 2 leaves a partial chunk that must not be lost."""
    cakes = [_cake(spec, i) for i in range(5)]
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, cakes, spec=spec, bin_area=np.ones(
        (spec.n_eta_bins, spec.n_r_bins)), omega_sum_frames=2)
    osf = _open(out)["OmegaSumFrame"]
    assert sorted(osf.keys()) == sorted(
        ["LastFrameNumber_1", "LastFrameNumber_3", "LastFrameNumber_4"])
    tail = osf["LastFrameNumber_4"]
    assert dict(tail.attrs)["Number Of Frames Summed"] == 1
    assert np.allclose(np.asarray(tail), cakes[4].T)


def test_omega_sum_frames_minus_one_sums_everything(tmp_path, spec):
    cakes = [_cake(spec, i) for i in range(4)]
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, cakes, spec=spec, bin_area=np.ones(
        (spec.n_eta_bins, spec.n_r_bins)), omega_sum_frames=-1)
    osf = _open(out)["OmegaSumFrame"]
    assert list(osf.keys()) == ["LastFrameNumber_3"]
    assert np.allclose(np.asarray(osf["LastFrameNumber_3"]),
                       np.sum(cakes, axis=0).T)
    assert dict(osf["LastFrameNumber_3"].attrs)["Number Of Frames Summed"] == 4


def test_omega_sum_frames_zero_omits_the_group(tmp_path, spec):
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, [_cake(spec, 0)], spec=spec,
                        bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                        omega_sum_frames=0)
    assert "OmegaSumFrame" not in _open(out)


def test_individual_save_off_by_default(tmp_path, spec):
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, [_cake(spec, 0)], spec=spec,
                        bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)))
    assert "IntegrationResult" not in _open(out)


def test_omega_defaults_to_frame_index(tmp_path, spec):
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, [_cake(spec, i) for i in range(3)], spec=spec,
                        bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)))
    assert np.allclose(np.asarray(_open(out)["Omegas"]), [0.0, 1.0, 2.0])


def test_nan_masked_bins_do_not_poison_the_sum(tmp_path, spec):
    """The C writes NAN into masked bins; one masked frame must not wipe out
    SumFrames / OmegaSumFrame everywhere."""
    a = _cake(spec, 1)
    b = _cake(spec, 2)
    b[0, 0] = np.nan
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, [a, b], spec=spec, bin_area=np.ones_like(a),
                        omega_sum_frames=-1)
    summed = np.asarray(_open(out)["SumFrames"])
    assert np.isfinite(summed).all()
    assert summed[0, 0] == pytest.approx(a.T[0, 0])


def test_rejects_wrong_cake_shape(tmp_path, spec):
    out = tmp_path / "t.zarr.zip"
    with pytest.raises(ValueError, match="neither"):
        write_gsas_zarr_zip(out, [np.zeros((3, 3))], spec=spec,
                            bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)))


def test_torch_tensor_cakes_accepted(tmp_path, spec):
    cake = torch.rand(spec.n_eta_bins, spec.n_r_bins, dtype=torch.float64)
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, [cake], spec=spec, individual_save=True,
                        bin_area=torch.ones(spec.n_eta_bins, spec.n_r_bins))
    ds = _open(out)["IntegrationResult"]["FrameNr_0"]
    assert np.allclose(np.asarray(ds), cake.numpy().T)


def test_single_zattrs_member_per_node(tmp_path, spec):
    """A ZipStore appends; a second attrs write would leave two .zattrs
    members for one node and a reader could pick up the stale one."""
    import zipfile
    out = tmp_path / "t.zarr.zip"
    write_gsas_zarr_zip(out, [_cake(spec, i) for i in range(4)], spec=spec,
                        bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                        omega_sum_frames=2, individual_save=True)
    names = zipfile.ZipFile(out).namelist()
    assert len(names) == len(set(names)), "duplicate members in the zip"


def test_writer_is_usable_incrementally(tmp_path, spec):
    """The streaming path: frames added one at a time, nothing held."""
    cakes = [_cake(spec, i) for i in range(3)]
    out = tmp_path / "t.zarr.zip"
    with GSASZarrWriter(out, spec=spec,
                        bin_area=np.ones((spec.n_eta_bins, spec.n_r_bins)),
                        omega_sum_frames=3) as w:
        for i, c in enumerate(cakes):
            w.add_frame(c, omega=i * 0.1)
    z = _open(out)
    assert np.allclose(np.asarray(z["OmegaSumFrame"]["LastFrameNumber_2"]),
                       np.sum(cakes, axis=0).T)
