"""FIX-3: a v2 calibration result must export the FF paramstest that
midas-peakfit + midas-transforms consume — geometry + p0..p14 (translated
from v2 names) AND the Stage-4 spline written as residual_corr.bin with a
ResidualCorrectionMap line.

Synthetic end-to-end proof of the wiring chain:
  v2 result (unpacked + spline) → write_ff_paramstest → paramstest.txt + .bin
  → (zipper-parseable, carries ResidualCorrectionMap) → .bin reproduces ΔR.
The bin → consumer half is pinned by the transforms (reshape (Z,Y)) and peakfit
(reshape (Z,Y).T) round-trip tests; together they cover calibration → spots.
"""
import numpy as np
import pytest
import torch

from types import SimpleNamespace

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.to_v1 import (
    write_ff_paramstest, ff_paramstest_from_auto_result, _V2_TO_V1_DISTORTION,
)


def test_ff_paramstest_from_auto_result_reconciled_v1_and_v2(tmp_path):
    """The exporter writes the FULL distortion in BOTH namings — the v2
    harmonics AND the equivalent v1 p0..p14 (same physical model, mapped via
    midas_distortion) — so the output works with v1-only (C) and v2 readers
    alike. Plus geometry, residual map, and template thresholds verbatim."""
    import numpy as np
    from midas_distortion import P_COEF_NAMES, v2_coeffs_from_named, v2_to_v1_coeffs

    tmpl = tmp_path / "ps.txt"
    tmpl.write_text("RingThresh 1 80\nMinNrSpots 4\np0 9\np3 9\nLsd 1\ntx 7\n")
    dist = {nm: 0.0 for nm in P_COEF_NAMES}
    dist.update({"iso_R2": -1.11e-3, "iso_R4": 8e-4, "a2": 4.6e-4, "phi2": 33.0,
                 "a4": -5.2e-4, "phi4": -6.87})
    res = SimpleNamespace(
        Lsd=958874.0, BC_y=1390.0, BC_z=1422.0, tx=0.0, ty=-0.173, tz=0.048,
        distortion=dist, NrPixelsY=2880, NrPixelsZ=2880,
        residual_corr_bin_path="/calib/residual_corr.bin",
        residual_corr_map=object())  # not None ⇒ calibration KEPT the map

    out = ff_paramstest_from_auto_result(res, tmpl, tmp_path / "v2.txt",
                                         raw_folder="/ni/raw")
    txt = out.read_text()
    lines = [l for l in txt.splitlines() if l.strip() and not l.startswith("#")]
    kv = {l.split()[0]: l.split()[1:] for l in lines}
    keys = [l.split()[0] for l in lines]

    # v2 names present with the right values.
    assert "iso_R2 -0.00111" in txt and "phi4 -6.87" in txt and "a2 0.00046" in txt
    # AND the full v1 p0..p14, mapped consistently from the v2 harmonics
    # (template's stale p0/p3 replaced, not duplicated).
    assert {f"p{i}" for i in range(15)} <= set(keys), "reconciled output needs p0..p14"
    expect_v1 = v2_to_v1_coeffs(v2_coeffs_from_named(dist))
    got_v1 = np.array([float(kv[f"p{i}"][0]) for i in range(15)])
    np.testing.assert_allclose(got_v1, expect_v1, rtol=1e-6, atol=1e-12)
    # e.g. phi4 (v2) == p3 (v1) == -6.87
    assert float(kv["p3"][0]) == pytest.approx(-6.87)
    # geometry (all tilts), detector, raw folder, residual map, thresholds.
    assert "tx 0" in txt and "ty -0.173" in txt and "tz 0.048" in txt
    assert keys.count("tx") == 1  # template tx replaced, not duplicated
    assert "NrPixelsY 2880" in txt and "RawFolder /ni/raw/" in txt
    assert "ResidualCorrectionMap /calib/residual_corr.bin" in txt
    assert "MinNrSpots 4" in txt and "RingThresh 1 80" in txt


def test_ff_export_drops_discarded_residual_map(tmp_path):
    """When calibration discarded the residual map (residual_corr_map is None),
    the exporter must NOT emit a ResidualCorrectionMap line — applying a map
    calibration rejected would degrade the reconstruction."""
    tmpl = tmp_path / "ps.txt"
    tmpl.write_text("RingThresh 1 80\nLsd 1\n")
    res = SimpleNamespace(
        Lsd=1e6, BC_y=1024.0, BC_z=1024.0, tx=0.0, ty=0.0, tz=0.0,
        distortion={"iso_R2": 1e-3}, NrPixelsY=2048, NrPixelsZ=2048,
        residual_corr_bin_path="/calib/residual_corr.bin",
        residual_corr_map=None)            # discarded
    out = ff_paramstest_from_auto_result(res, tmpl, tmp_path / "v2.txt",
                                         raw_folder="/ni/raw")
    txt = out.read_text()
    assert "ResidualCorrectionMap" not in txt
    assert "discarded by calibration" in txt


class _FakeResult:
    """Duck-typed FourStageResult: an unpacked dict + a Stage-4 spline."""
    def __init__(self, unpacked, spline_fn):
        self.unpacked = unpacked
        self.stage3_spline_fn = spline_fn


def _template(NY, NZ, px):
    p = V1Params()
    p.NrPixelsY, p.NrPixelsZ = NY, NZ
    p.pxY, p.pxZ = px, 0.0
    p.Lsd = 1.0e6
    p.Wavelength = 0.18
    p.RhoD = NY * px
    p.MaxRingRad = NY * px
    return p


def test_write_ff_paramstest_full_chain(tmp_path):
    NY, NZ, px = 5, 7, 150.0

    # Distinct nonzero distortion values, keyed by v2 name, so the v2→v1
    # p-index mapping is verifiable per-coefficient.
    v2_dist = {
        "iso_R2": 1e-3, "iso_R4": 2e-3, "iso_R6": 3e-3,
        "a1": 4e-3, "phi1": 11.0, "a2": 5e-3, "phi2": 22.0,
        "a3": 6e-3, "phi3": 33.0, "a4": 7e-3, "phi4": 44.0,
        "a5": 8e-3, "phi5": 55.0, "a6": 9e-3, "phi6": 66.0,
    }
    geom = {"Lsd": 1.0023e6, "BC_y": 1024.5, "BC_z": 980.25,
            "ty": 0.12, "tz": -0.34, "Wavelength": 0.1729}
    unpacked = {k: torch.tensor(float(v), dtype=torch.float64)
                for k, v in {**v2_dist, **geom}.items()}

    def spline_predict(ys, zs):  # ΔR µm, asymmetric in Y vs Z
        return 30.0 * np.sin(2 * np.pi * zs / NZ) + 12.0 * np.cos(2 * np.pi * ys / NY)

    result = _FakeResult(unpacked, spline_predict)
    ptest = write_ff_paramstest(result, _template(NY, NZ, px), tmp_path)

    # 1) paramstest + residual_corr.bin both written
    assert ptest.exists()
    binp = tmp_path / "residual_corr.bin"
    assert binp.exists()

    # 2) geometry + p0..p14 round-trip with the v2→v1 mapping applied
    rp = V1Params.from_file(ptest)
    assert abs(rp.Lsd - geom["Lsd"]) < 1e-3
    assert abs(rp.BC_y - geom["BC_y"]) < 1e-6
    assert abs(rp.BC_z - geom["BC_z"]) < 1e-6
    assert abs(rp.ty - geom["ty"]) < 1e-9
    assert abs(rp.tz - geom["tz"]) < 1e-9
    for v2name, v1name in _V2_TO_V1_DISTORTION.items():
        got = getattr(rp, v1name)
        assert abs(got - v2_dist[v2name]) < 1e-9, (
            f"{v2name}→{v1name}: paramstest has {got}, expected {v2_dist[v2name]}"
        )

    # 3) the param file is zipper-parseable and carries ResidualCorrectionMap
    #    pointing at the bin (this is what lands in the zarr analysis_params).
    try:
        from midas_zipper.ff_zip import parse_parameter_file, write_analysis_parameters  # noqa: F401
    except Exception:
        cfg = None
    else:
        cfg = parse_parameter_file(str(ptest))
        assert cfg.get("ResidualCorrectionMap") == str(binp), (
            "ResidualCorrectionMap not carried by the param file → zipper path"
        )

    # 4) the bin reproduces the spline ΔR (z-major (NrPixelsZ, NrPixelsY)).
    md = np.fromfile(binp, dtype=np.float64)
    assert md.size == NY * NZ
    grid = md.reshape(NZ, NY)  # grid[z, y] = ΔR(Y=y, Z=z)/px
    for z in range(NZ):
        for y in range(NY):
            truth = spline_predict(np.array([y]), np.array([z]))[0] / px
            assert abs(grid[z, y] - truth) < 1e-12


def test_write_ff_paramstest_no_spline(tmp_path):
    """No Stage-4 spline → paramstest written, no bin, no ResidualCorrectionMap."""
    NY, NZ, px = 4, 4, 200.0
    unpacked = {"Lsd": torch.tensor(1e6), "a2": torch.tensor(1e-3)}
    result = _FakeResult(unpacked, None)
    ptest = write_ff_paramstest(result, _template(NY, NZ, px), tmp_path)
    assert ptest.exists()
    assert not (tmp_path / "residual_corr.bin").exists()
    assert "ResidualCorrectionMap" not in ptest.read_text()
