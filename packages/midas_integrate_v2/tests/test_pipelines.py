"""Items 30 + 24 — Energy-sweep + drift trajectory pipelines."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import spec_from_v1_params
from midas_integrate_v2.pipelines import (
    DriftTrajectory, fit_drift_trajectory, run_energy_sweep,
)


def _spec(NY=32, NZ=32):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0, BC_z=NZ / 2.0, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=0.5,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
        Wavelength=0.18,
    )
    return spec_from_v1_params(p, requires_grad=False)


def test_run_energy_sweep_basic():
    spec = _spec()
    energies_eV = [60_000.0, 65_000.0, 70_000.0]
    frames = [
        np.full((spec.NrPixelsZ, spec.NrPixelsY), 50.0)
        for _ in energies_eV
    ]
    res = run_energy_sweep(energies_eV, frames, spec)
    assert len(res.profiles) == 3
    assert all(np.isfinite(p).all() for p in res.profiles)
    # Wavelength scaling should change Q axis between energies
    Q_axes = res.Q_axes
    assert Q_axes[0][-1] != Q_axes[2][-1]


def test_run_energy_sweep_writes_dat(tmp_path):
    spec = _spec()
    energies_eV = [60_000.0, 65_000.0]
    frames = [
        np.full((spec.NrPixelsZ, spec.NrPixelsY), 100.0)
        for _ in energies_eV
    ]
    run_energy_sweep(energies_eV, frames, spec, out_dir=tmp_path)
    files = sorted(tmp_path.glob("E_*.dat"))
    assert len(files) == 2


def test_drift_constant_falls_back_to_mean():
    base_spec = _spec()
    anchors = {
        0:  {"Lsd": 1_000_000.0, "BC_y": 16.0, "BC_z": 16.0},
        50: {"Lsd": 1_000_010.0, "BC_y": 16.05, "BC_z": 15.95},
    }
    sample_idx = list(range(0, 51, 10))
    drift = fit_drift_trajectory(
        anchors, sample_idx, base_spec, parametrization="constant",
    )
    assert isinstance(drift, DriftTrajectory)
    np.testing.assert_allclose(drift.Lsd_t, drift.Lsd_t.mean(),
                                rtol=1e-12)


def test_drift_linear_passes_through_anchors():
    base_spec = _spec()
    anchors = {
        0:    {"Lsd": 1_000_000.0, "BC_y": 16.0, "BC_z": 16.0},
        100:  {"Lsd": 1_000_500.0, "BC_y": 16.5, "BC_z": 15.5},
    }
    sample_idx = list(range(0, 101, 10))
    drift = fit_drift_trajectory(
        anchors, sample_idx, base_spec, parametrization="linear",
    )
    # At anchor frame 0, Lsd_t should match anchor exactly
    idx0 = np.where(drift.frame_indices == 0)[0][0]
    assert drift.Lsd_t[idx0] == pytest.approx(1_000_000.0, rel=1e-6)
    idx_end = np.where(drift.frame_indices == 100)[0][0]
    assert drift.Lsd_t[idx_end] == pytest.approx(1_000_500.0, rel=1e-6)


def test_drift_spline_smooths_anchors():
    base_spec = _spec()
    anchors = {
        0:    {"Lsd": 1_000_000.0, "BC_y": 16.0, "BC_z": 16.0},
        25:   {"Lsd": 1_000_200.0, "BC_y": 16.1, "BC_z": 16.0},
        50:   {"Lsd": 1_000_300.0, "BC_y": 16.3, "BC_z": 15.9},
        100:  {"Lsd": 1_000_500.0, "BC_y": 16.5, "BC_z": 15.5},
    }
    sample_idx = list(range(0, 101, 5))
    drift = fit_drift_trajectory(
        anchors, sample_idx, base_spec,
        parametrization="spline", n_knots=5,
    )
    # Spline must be smooth (no big jumps between consecutive samples)
    diffs = np.abs(np.diff(drift.Lsd_t))
    assert diffs.max() < 200.0
