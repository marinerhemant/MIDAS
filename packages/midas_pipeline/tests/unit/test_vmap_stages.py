"""Integration tests for the V-map pipeline stages (P8 of the V-map plan).

Covers:
- ``calc_radius_v`` no-op when ``vmap.run=False`` and when required
  inputs (hkls.csv, crystal_cif, wavelength) are missing.
- ``calc_radius_v`` end-to-end on a synthetic layer: writes
  ``Radius_V.csv`` and ``I_theory_per_ring.csv`` with the right schema
  and row counts.
- ``refine_vmap`` no-op when ``vmap.run=False`` and when upstream
  ``Radius_V.csv`` is missing.
- ``refine_vmap`` end-to-end on a single-grain synthetic layout:
  recovers per-voxel V via joint LBFGS, writes H5 (or npz fallback) +
  per-ring K CSV + loss history.
- Stage registration: ``calc_radius_v`` and ``refine_vmap`` appear in
  ``all_stage_names()`` and the ``_LAYER_RESULT_FIELD_BY_STAGE`` map.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pytest


# Shared test fixtures: a CIF that ships with midas_hkls (CeO2, cubic Fm-3m).
CEO2_CIF = Path(
    "/Users/hsharma/opt/MIDAS/packages/midas_hkls/tests/data/ceo2.cif",
)


# -------------------------------------------------------------- helpers


def _write_hkls_csv(path: Path):
    """Minimal MIDAS hkls.csv with 3 rings for the shipped CeO2 CIF
    (cubic, Fm-3m, a=5.4112 Å) at Cu-Kα λ=1.5418 Å."""
    path.write_text(
        "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n"
        "1 1 1 3.124 1 0 0 0 14.275 28.550 100.0\n"
        "2 0 0 2.706 2 0 0 0 16.535 33.070 120.0\n"
        "2 2 0 1.913 3 0 0 0 23.745 47.490 170.0\n"
    )


def _write_input_extra(path: Path, rows):
    """Write a 15-column InputAllExtraInfo CSV. ``rows`` is list of dicts
    with keys spot_id, ring_nr, omega, eta, integ_I."""
    with path.open("w") as f:
        f.write("% header\n")
        n = 15
        for r in rows:
            row = [0.0] * n
            row[2]  = float(r["omega"])
            row[4]  = float(r["spot_id"])
            row[5]  = float(r["ring_nr"])
            row[6]  = float(r["eta"])
            row[14] = float(r["integ_I"])
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def _make_pipeline_config(tmp_path: Path, **vmap_kwargs):
    from midas_pipeline import PipelineConfig, ScanGeometry, VMapConfig

    cfg = PipelineConfig(
        result_dir=str(tmp_path),
        params_file=str(tmp_path / "paramstest.txt"),
        scan=ScanGeometry.pf_uniform(n_scans=4, scan_step_um=5.0, beam_size_um=5.0),
    )
    for k, v in vmap_kwargs.items():
        setattr(cfg.vmap, k, v)
    return cfg


def _make_ctx(tmp_path: Path, cfg) -> "object":
    from midas_pipeline.stages._base import StageContext

    layer_dir = tmp_path
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    return StageContext(
        config=cfg, layer_nr=1, layer_dir=layer_dir, log_dir=log_dir,
    )


# -------------------------------------------------------------- calc_radius_v


def test_calc_radius_v_noop_when_vmap_run_false(tmp_path):
    from midas_pipeline.stages import calc_radius_v
    cfg = _make_pipeline_config(tmp_path)  # vmap.run defaults False
    ctx = _make_ctx(tmp_path, cfg)
    res = calc_radius_v.run(ctx)
    assert res.skipped


def test_calc_radius_v_noop_when_no_hkls(tmp_path):
    from midas_pipeline.stages import calc_radius_v
    cfg = _make_pipeline_config(tmp_path, run=True, crystal_cif=str(CEO2_CIF),
                                wavelength_A=1.5418)
    ctx = _make_ctx(tmp_path, cfg)
    res = calc_radius_v.run(ctx)
    assert res.skipped


def test_calc_radius_v_noop_when_no_crystal_cif(tmp_path):
    _write_hkls_csv(tmp_path / "hkls.csv")
    _write_input_extra(
        tmp_path / "InputAllExtraInfoFittingAll0.csv",
        [{"spot_id": 1, "ring_nr": 1, "omega": 10.0, "eta": 30.0, "integ_I": 100.0}],
    )
    from midas_pipeline.stages import calc_radius_v
    cfg = _make_pipeline_config(tmp_path, run=True, wavelength_A=1.5418)
    ctx = _make_ctx(tmp_path, cfg)
    res = calc_radius_v.run(ctx)
    assert res.skipped


def test_calc_radius_v_end_to_end_writes_outputs(tmp_path):
    """Build a tiny synthetic layer; calc_radius_v writes Radius_V + I_theory."""
    _write_hkls_csv(tmp_path / "hkls.csv")
    _write_input_extra(
        tmp_path / "InputAllExtraInfoFittingAll0.csv",
        [
            {"spot_id": 1, "ring_nr": 1, "omega": 10.0, "eta": 30.0, "integ_I": 250.0},
            {"spot_id": 2, "ring_nr": 2, "omega": 20.0, "eta": 60.0, "integ_I": 180.0},
            {"spot_id": 3, "ring_nr": 3, "omega": 30.0, "eta": -45.0, "integ_I": 90.0},
        ],
    )
    from midas_pipeline.stages import calc_radius_v
    cfg = _make_pipeline_config(
        tmp_path, run=True, crystal_cif=str(CEO2_CIF), wavelength_A=1.5418,
    )
    ctx = _make_ctx(tmp_path, cfg)
    res = calc_radius_v.run(ctx)

    assert not res.skipped
    assert res.n_spots == 3
    assert res.n_rings == 3

    radius_csv = Path(res.radius_csv)
    theory_csv = Path(res.theory_csv)
    assert radius_csv.exists() and theory_csv.exists()

    # Radius_V columns: spot_id, scan_nr, ring_nr, ring_idx, intensity, V_rel, ω, η
    arr = np.loadtxt(radius_csv, comments="#", skiprows=1)
    assert arr.shape == (3, 8)
    assert (arr[:, 0] == [1, 2, 3]).all()             # spot_id
    assert (arr[:, 2] == [1, 2, 3]).all()             # ring_nr
    # V_rel = intensity / I_theory(ring) — must be positive for all 3 spots
    assert (arr[:, 5] > 0).all()

    theory = np.loadtxt(theory_csv, comments="#", skiprows=1)
    assert theory.shape == (3, 3)
    # I_theory is positive for at least one ring (CeO2 (1,1,1) is allowed).
    assert (theory[:, 2] > 0).any()


# -------------------------------------------------------------- refine_vmap


def _setup_refine_inputs(tmp_path: Path, n_voxels: int = 4):
    """Build the Output/ files that refine_vmap consumes (synthetic).

    Layout: a single grain whose voxels are a linear strip along +x with
    a paramstest BeamSize.  Each scan_nr maps to one voxel.  Generates
    spot intensities consistent with V_true and K_true so the recovery
    is exact (modulo LBFGS convergence).
    """
    out_dir = tmp_path / "Output"
    out_dir.mkdir(exist_ok=True)
    # paramstest with BeamSize so refine_vmap can derive voxel_size.
    (tmp_path / "paramstest.txt").write_text("BeamSize 5.0\nWavelength 1.5418\n")
    # positions.csv: scan_nr -> y. We use one y-position per voxel along
    # the x-axis (so the projection equals v_x for ω=π/2).
    np.savetxt(tmp_path / "positions.csv", np.arange(n_voxels) * 5.0)
    # voxel_grid: each voxel is at (i*5, 0, 0) in grain 0.
    with (out_dir / "voxel_grid.csv").open("w") as f:
        f.write("voxel_idx x_um y_um z_um grain_id\n")
        for i in range(n_voxels):
            f.write(f"{i} {i*5.0:.4f} {i*5.0:.4f} 0.0 0\n")
    # I_theory_per_ring: single ring
    np.savetxt(
        out_dir / "I_theory_per_ring.csv",
        np.array([[1, 3.30, 11.0]]),
        header="ring_number two_theta_deg I_theory",
        fmt=["%d", "%.6f", "%.6e"], comments="",
    )
    # Radius_V: per-spot. We want predicted intensities matching
    # K_true * I_theory * V_true[v] (each spot illuminates exactly one voxel).
    K_true = 4.2
    V_true = np.array([0.5, 1.0, 1.5, 0.7])[:n_voxels]
    rows = []
    for s, v_idx in enumerate(range(n_voxels)):
        I_obs = K_true * 11.0 * V_true[v_idx]
        rows.append([s, v_idx, 1, 0, I_obs, 0.0, 90.0, 30.0])  # ω=90° → s_proj = v_y = scan_pos
    np.savetxt(
        out_dir / "Radius_V.csv",
        np.array(rows),
        header="spot_id scan_nr ring_number ring_idx intensity V_rel omega_deg eta_deg",
        fmt=["%d", "%d", "%d", "%d", "%.6e", "%.6e", "%.6f", "%.6f"],
        comments="",
    )
    return V_true, K_true


def test_refine_vmap_noop_when_vmap_run_false(tmp_path):
    from midas_pipeline.stages import refine_vmap
    cfg = _make_pipeline_config(tmp_path)
    ctx = _make_ctx(tmp_path, cfg)
    res = refine_vmap.run(ctx)
    assert res.skipped


def test_refine_vmap_noop_when_no_radius_csv(tmp_path):
    from midas_pipeline.stages import refine_vmap
    cfg = _make_pipeline_config(tmp_path, run=True)
    ctx = _make_ctx(tmp_path, cfg)
    res = refine_vmap.run(ctx)
    assert res.skipped


def test_refine_vmap_end_to_end_recovers_V(tmp_path):
    V_true, K_true = _setup_refine_inputs(tmp_path)
    from midas_pipeline.stages import refine_vmap
    cfg = _make_pipeline_config(
        tmp_path, run=True, refine_V=True, refine_K=False,
        max_iter=80, tolerance=1e-10,
    )
    ctx = _make_ctx(tmp_path, cfg)
    res = refine_vmap.run(ctx)

    assert not res.skipped
    assert res.n_voxels == 4
    assert res.n_rings == 1
    assert res.n_iterations > 0
    # Loss should drop well below initial
    loss_path = Path(res.loss_history_csv)
    assert loss_path.exists()
    loss = np.loadtxt(loss_path, comments="#", skiprows=1)
    if loss.ndim == 1:
        loss = loss.reshape(1, -1)
    if loss.shape[0] >= 2:
        assert loss[-1, 1] <= loss[0, 1]

    # Recover V (h5 ideally; fall back to npz)
    v_map_path = Path(res.v_map_h5)
    assert v_map_path.exists()
    if v_map_path.suffix == ".h5":
        import h5py
        with h5py.File(v_map_path, "r") as f:
            V_est = f["voxels/V"][:]
            K_est = f["rings/K"][:]
    else:
        npz = np.load(v_map_path)
        V_est = npz["V"]
        K_est = npz["K"]
    # The V/K split is intrinsically ambiguous (V·c, K/c gives the same
    # predictions); compare the identifiable product V·K per voxel.
    np.testing.assert_allclose(V_est * K_est[0], V_true * K_true, rtol=1e-4)
    # Sanity: V positive everywhere (softplus parameterization).
    assert (V_est > 0).all()


# -------------------------------------------------------------- registration


def test_stages_are_registered_in_pipeline():
    from midas_pipeline import all_stage_names
    from midas_pipeline.pipeline import _LAYER_RESULT_FIELD_BY_STAGE
    names = all_stage_names()
    assert "calc_radius_v" in names
    assert "refine_vmap" in names
    assert _LAYER_RESULT_FIELD_BY_STAGE["calc_radius_v"] == "calc_radius_v"
    assert _LAYER_RESULT_FIELD_BY_STAGE["refine_vmap"] == "refine_vmap"


def test_pipeline_config_has_vmap_and_soft_attribution():
    from midas_pipeline import (
        PipelineConfig, ScanGeometry, SoftAttributionConfig, VMapConfig,
    )
    cfg = PipelineConfig(
        result_dir="/tmp/x", params_file="/tmp/x/p.txt",
        scan=ScanGeometry.pf_uniform(n_scans=2, scan_step_um=5.0, beam_size_um=5.0),
    )
    assert isinstance(cfg.vmap, VMapConfig)
    assert isinstance(cfg.soft_attribution, SoftAttributionConfig)
    assert cfg.vmap.run is False
    assert cfg.soft_attribution.enable is False
