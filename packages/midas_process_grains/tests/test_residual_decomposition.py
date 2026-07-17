"""Tests for the signed per-spot residual decomposition diagnostics."""

import math

import numpy as np
import pytest

from midas_process_grains.compute.residual_decomposition import (
    SPOT_RESIDUAL_COLS,
    build_spot_residual_row,
    decompose_residuals,
    summarize_residuals,
)


def _fb_row(y_obs, z_obs, ome_obs, y_exp, z_exp, ome_exp, internal_angle=0.1):
    """Build a synthetic 22-double FitBest row with the columns we consume."""
    row = np.zeros(22)
    row[1], row[2], row[3] = y_obs, z_obs, ome_obs
    row[7], row[8], row[9] = y_exp, z_exp, ome_exp
    row[19] = internal_angle
    return row


def test_build_row_pure_radial():
    # Spot on +Z axis (eta = 0 in MIDAS convention atan2(-y, z)); observed
    # 10 um further out than expected -> dRad = +10, dTan = 0.
    row = build_spot_residual_row(0, 5, 1, _fb_row(0.0, 1010.0, 30.0, 0.0, 1000.0, 30.0))
    d = dict(zip(SPOT_RESIDUAL_COLS, row))
    assert d["drad_um"] == pytest.approx(10.0)
    assert d["dtan_um"] == pytest.approx(0.0, abs=1e-9)
    assert d["dy_um"] == pytest.approx(0.0)
    assert d["dz_um"] == pytest.approx(10.0)
    assert d["eta_deg"] == pytest.approx(0.0)
    assert d["dome_deg"] == pytest.approx(0.0)
    assert d["r_exp_um"] == pytest.approx(1000.0)


def test_build_row_pure_tangential():
    # Spot on +Z axis, displaced along +Y only: tangential magnitude = dy,
    # radial ~ 0 (to first order in dy/r).
    row = build_spot_residual_row(0, 6, 2, _fb_row(10.0, 1000.0, 0.0, 0.0, 1000.0, 0.0))
    d = dict(zip(SPOT_RESIDUAL_COLS, row))
    assert abs(d["drad_um"]) < 0.1
    assert abs(d["dtan_um"]) == pytest.approx(10.0, rel=1e-3)


def test_build_row_omega_wrap():
    # Omega residual wraps across the -180/180 seam.
    row = build_spot_residual_row(0, 7, 1, _fb_row(0, 1000, -179.9, 0, 1000, 179.9))
    d = dict(zip(SPOT_RESIDUAL_COLS, row))
    assert d["dome_deg"] == pytest.approx(0.2, abs=1e-9)


def test_build_row_rejects_padding():
    assert build_spot_residual_row(0, 0, 0, np.zeros(22)) is None


def test_decompose_shapes_and_values():
    rng = np.random.default_rng(7)
    rows = []
    # grain 0: +5 um radial bias on ring 1; grain 1: -5 um on ring 2.
    for gi, (bias, ring) in enumerate([(5.0, 1), (-5.0, 2)]):
        for _ in range(50):
            eta = rng.uniform(-math.pi, math.pi)
            # MIDAS eta convention: y = -r sin(eta), z = r cos(eta)
            r_exp = 1000.0
            y_exp, z_exp = -r_exp * math.sin(eta), r_exp * math.cos(eta)
            r_obs = r_exp + bias
            y_obs, z_obs = -r_obs * math.sin(eta), r_obs * math.cos(eta)
            rows.append(build_spot_residual_row(
                gi, 100 * gi, ring,
                _fb_row(y_obs, z_obs, 10.0, y_exp, z_exp, 10.0, 0.3),
            ))
    tbl = np.asarray(rows)
    diag = decompose_residuals(tbl, n_grains=3)

    assert diag["grain_med_drad_um"].shape == (3,)
    assert diag["grain_med_drad_um"][0] == pytest.approx(5.0, abs=1e-6)
    assert diag["grain_med_drad_um"][1] == pytest.approx(-5.0, abs=1e-6)
    assert np.isnan(diag["grain_med_drad_um"][2])  # no rows for grain 2
    assert diag["grain_n_spots"].tolist() == [50, 50, 0]
    assert diag["grain_med_internal_angle_deg"][0] == pytest.approx(0.3)

    # per-ring ppm: 5 um on 1005 um radius ~ +4975 ppm
    i1 = list(diag["ring_nr"]).index(1)
    assert diag["ring_drad_ppm"][i1] == pytest.approx(1e6 * 5.0 / 1000.0, rel=2e-2)

    # eta profile arrays are 12 bins of 30 deg
    assert diag["eta_bin_lo_deg"].shape == (12,)
    assert int(diag["eta_n_spots"].sum()) == 100

    # scalars present and finite
    assert np.isfinite(diag["overall_med_drad_um"])
    assert diag["overall_med_dome_deg"] == pytest.approx(0.0)

    # summary renders without error and flags nothing at ~0 ppm median bias
    text = summarize_residuals(diag)
    assert "per-ring dR/R" in text


def test_decompose_empty():
    diag = decompose_residuals(np.zeros((0, len(SPOT_RESIDUAL_COLS))), n_grains=2)
    assert diag["grain_med_dy_um"].shape == (2,)
    assert np.isnan(diag["overall_med_dtan_um"])
    assert summarize_residuals(diag)  # does not raise


def test_h5_roundtrip(tmp_path):
    """residuals group + spot_table survive the diagnostics h5 writer."""
    h5py = pytest.importorskip("h5py")
    from midas_process_grains.io.consolidated import write_diagnostics_h5

    rows = [build_spot_residual_row(
        0, 1, 1, _fb_row(0.0, 1010.0, 30.0, 0.0, 1000.0, 30.0))]
    tbl = np.asarray(rows)
    diag = decompose_residuals(tbl, n_grains=1)

    class _Res:
        n_grains = 1
        diagnostics = {"residuals": diag, "residuals_spot_table": tbl}
        sg_nr = 225
        lattice_reference = np.array([3.6, 3.6, 3.6, 90, 90, 90])
        mode = "spot_aware"

    p = tmp_path / "d.h5"
    write_diagnostics_h5(p, _Res())
    with h5py.File(p, "r") as f:
        assert "residuals" in f
        st = f["residuals/spot_table"]
        assert st.shape == (1, len(SPOT_RESIDUAL_COLS))
        assert st.attrs["columns"].startswith("grain_idx,spot_id")
        assert f["residuals/overall_med_drad_um"][()] == pytest.approx(10.0)
        assert f["residuals/grain_med_drad_um"][0] == pytest.approx(10.0)
