"""Unit tests for the diagnostic gates.

These tests construct minimal synthetic fixtures (no calibrant image
required) so the gates can be exercised in CI.  Real-data integration
is covered by ``run_robust_smoke.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import pytest


# -------------------------------------------------- strain_cap_check


def test_strain_cap_ok():
    from midas_calibrate_v2.pipelines.diagnostics import strain_cap_check

    history = [SimpleNamespace(mean_strain_uE=7.5)]
    res = strain_cap_check(history, threshold_uE=100.0, warn_uE=50.0)
    assert res.severity == "ok"
    assert "7.50" in res.message or "7.5" in res.message


def test_strain_cap_warn():
    from midas_calibrate_v2.pipelines.diagnostics import strain_cap_check

    history = [SimpleNamespace(mean_strain_uE=72.0)]
    res = strain_cap_check(history, threshold_uE=100.0, warn_uE=50.0)
    assert res.severity == "warn"


def test_strain_cap_fail():
    from midas_calibrate_v2.pipelines.diagnostics import strain_cap_check

    # B6 basin escapes were ≥ 800 μϵ; 100 μϵ cap rejects them.
    history = [SimpleNamespace(mean_strain_uE=820.0)]
    res = strain_cap_check(history, threshold_uE=100.0)
    assert res.severity == "fail"
    assert "basin escape" in res.message.lower()


def test_strain_cap_nan():
    from midas_calibrate_v2.pipelines.diagnostics import strain_cap_check

    history = [SimpleNamespace(mean_strain_uE=float("nan"))]
    res = strain_cap_check(history)
    assert res.severity == "fail"
    assert "diverged" in res.message.lower() or "nan" in res.message.lower()


def test_strain_cap_empty_history():
    from midas_calibrate_v2.pipelines.diagnostics import strain_cap_check

    res = strain_cap_check([])
    assert res.severity == "warn"


# -------------------------------------------------- basin_check


@dataclass
class _FakeV1:
    Lsd: float = 1_000_000.0
    BC_y: float = 1024.0
    BC_z: float = 1024.0


def test_basin_check_ok():
    from midas_calibrate_v2.pipelines.diagnostics import basin_check

    seed = _FakeV1()
    unp = {"Lsd": torch.tensor(1_000_500.0),
           "BC_y": torch.tensor(1024.5),
           "BC_z": torch.tensor(1024.0)}
    res = basin_check(seed, unp)
    assert res.severity == "ok"
    assert abs(res.metrics["delta_Lsd_pct"] - 0.05) < 1e-6
    assert abs(res.metrics["delta_BC_px"] - 0.5) < 1e-6


def test_basin_check_warn():
    from midas_calibrate_v2.pipelines.diagnostics import basin_check

    # 2% Lsd drift — warn but not fail
    seed = _FakeV1()
    unp = {"Lsd": torch.tensor(1_020_000.0),
           "BC_y": torch.tensor(1024.0),
           "BC_z": torch.tensor(1024.0)}
    res = basin_check(seed, unp)
    assert res.severity == "warn"


def test_basin_check_fail_lsd():
    from midas_calibrate_v2.pipelines.diagnostics import basin_check

    # 10% Lsd drift — basin escape
    seed = _FakeV1()
    unp = {"Lsd": torch.tensor(1_100_000.0),
           "BC_y": torch.tensor(1024.0),
           "BC_z": torch.tensor(1024.0)}
    res = basin_check(seed, unp)
    assert res.severity == "fail"


def test_basin_check_fail_bc():
    from midas_calibrate_v2.pipelines.diagnostics import basin_check

    seed = _FakeV1()
    unp = {"Lsd": torch.tensor(1_000_000.0),
           "BC_y": torch.tensor(1100.0),
           "BC_z": torch.tensor(1024.0)}
    res = basin_check(seed, unp)
    assert res.severity == "fail"


# -------------------------------------------------- cross_validation_gate


def _make_fake_fits(med_train_uE: float, med_test_uE: float,
                    n_per_ring: int = 50, seed: int = 0):
    """Build a fake FittedDataset-like object with controlled
    train/test residual medians."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n_rings = 12
    n_train_rings = 8
    rows = []
    for r in range(n_rings):
        med = med_train_uE if r < n_train_rings else med_test_uE
        # half-Gaussian magnitude for an absolute-residual-like dist
        rows.append((r, np.abs(rng.normal(0, med * 1.4826, n_per_ring))))
    Y = np.concatenate([np.full(p[1].shape, p[0] * 100.0)
                        for p in rows])
    Z = np.concatenate([np.full(p[1].shape, p[0] * 50.0)
                        for p in rows])
    ring_idx = np.concatenate([np.full(p[1].shape, p[0], dtype=np.int64)
                               for p in rows])
    abs_r_uE = np.concatenate([p[1] for p in rows])

    # cross_validation_gate calls pseudo_strain_residual; we monkey-patch
    # by using a SimpleNamespace fake with the expected attributes and
    # patch pseudo_strain_residual to return the synthetic residuals.
    return SimpleNamespace(
        Y_pix=torch.tensor(Y),
        Z_pix=torch.tensor(Z),
        ring_two_theta_deg=torch.tensor(np.zeros_like(Y)),
        ring_idx=torch.tensor(ring_idx),
        rho_d=torch.tensor(1000.0),
        panel_idx=None,
    ), abs_r_uE


def test_cross_validation_gate_ok(monkeypatch):
    """Test/train residuals match — gate passes."""
    from midas_calibrate_v2.pipelines import diagnostics

    fake_fits, abs_r_uE = _make_fake_fits(med_train_uE=10.0,
                                            med_test_uE=10.0)
    abs_r_strain = abs_r_uE * 1e-6     # convert μϵ → strain

    def fake_residual(*a, **k):
        # Return signed residual whose abs() reproduces abs_r_strain.
        return torch.tensor(abs_r_strain)

    monkeypatch.setattr(diagnostics, "pseudo_strain_residual", fake_residual)

    res = diagnostics.cross_validation_gate(fake_fits, {})
    assert res.severity == "ok"
    assert abs(res.metrics["ratio"] - 1.0) < 0.3


def test_cross_validation_gate_fail(monkeypatch):
    """Held-out median 2× train median — gate fails."""
    from midas_calibrate_v2.pipelines import diagnostics

    fake_fits, abs_r_uE = _make_fake_fits(med_train_uE=10.0,
                                            med_test_uE=25.0)
    abs_r_strain = abs_r_uE * 1e-6

    def fake_residual(*a, **k):
        return torch.tensor(abs_r_strain)

    monkeypatch.setattr(diagnostics, "pseudo_strain_residual", fake_residual)

    res = diagnostics.cross_validation_gate(
        fake_fits, {}, fail_med_ratio=1.5, warn_med_ratio=1.2,
    )
    assert res.severity == "fail"
    assert res.metrics["ratio"] > 1.5


# -------------------------------------------------- run_all_gates / summarise


def test_run_all_gates_ok():
    from midas_calibrate_v2.pipelines.diagnostics import (
        run_all_gates, summarise, worst_severity,
    )

    seed = _FakeV1()
    unp = {"Lsd": torch.tensor(1_000_000.0),
           "BC_y": torch.tensor(1024.0),
           "BC_z": torch.tensor(1024.0)}
    history = [SimpleNamespace(mean_strain_uE=8.0)]

    diags = run_all_gates(v1_init=seed, unpacked=unp, history=history,
                          fits=None)
    assert worst_severity(diags) == "ok"
    text = summarise(diags)
    assert "✓" in text or "ok" in text.lower()


def test_run_all_gates_fail():
    from midas_calibrate_v2.pipelines.diagnostics import (
        run_all_gates, worst_severity,
    )

    seed = _FakeV1()
    # Lsd walked 10 % — basin_check fails.
    unp = {"Lsd": torch.tensor(1_100_000.0),
           "BC_y": torch.tensor(1024.0),
           "BC_z": torch.tensor(1024.0)}
    # Strain blew up — strain_cap fails.
    history = [SimpleNamespace(mean_strain_uE=2000.0)]

    diags = run_all_gates(v1_init=seed, unpacked=unp, history=history,
                          fits=None)
    assert worst_severity(diags) == "fail"


# -------------------------------------------------- ResidualConvNet init smoke


def test_residual_conv_net_nonzero_init():
    """Regression test for the dead-ReLU bug: the network output should
    not be identically zero at init, otherwise gradients vanish."""
    from midas_calibrate_v2.forward.nn_residual import (
        ResidualConvNet, NNResidualConfig,
    )

    cfg = NNResidualConfig(grid_H=32, grid_W=32,
                            detector_H_px=2048, detector_W_px=2048)
    m = ResidualConvNet(cfg)
    out = m.field()
    assert torch.is_tensor(out)
    # Non-zero spread: max(abs) > 1e-6 confirms gradient flow is alive.
    assert out.abs().max().item() > 1e-6, \
        "ResidualConvNet collapsed to zero output at init — dead-ReLU regression"
