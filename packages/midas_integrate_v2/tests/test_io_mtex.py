"""Item 12 — MTEX cake + pole-figure exporters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_integrate_v2.io import write_mtex_epf, write_mtex_xpc


def test_xpc_writes_header_and_grid(tmp_path: Path):
    n_eta, n_r = 4, 3
    eta = np.array([-90.0, 0.0, 90.0, 180.0])
    R = np.array([5.0, 10.0, 15.0])
    int2d = np.arange(n_eta * n_r, dtype=np.float64).reshape(n_eta, n_r)
    out = tmp_path / "cake.xpc"
    write_mtex_xpc(out, int2d, eta, R, R_units="px",
                    hkl_rings=[(1, 1, 1), (2, 0, 0), (2, 2, 0)])
    text = out.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # 4 header lines + n_eta * n_r data
    data_lines = [ln for ln in lines if not ln.startswith("#")]
    assert len(data_lines) == n_eta * n_r
    # Last data line should be the (3,2) cell
    last = data_lines[-1].split()
    assert float(last[0]) == pytest.approx(180.0)
    assert float(last[1]) == pytest.approx(15.0)
    assert float(last[2]) == pytest.approx(int2d[3, 2])
    assert "rings_hkl" in text


def test_epf_writes_header_and_grid(tmp_path: Path):
    n_a, n_b = 5, 7
    alpha = np.linspace(0.0, 90.0, n_a)
    beta = np.linspace(0.0, 360.0, n_b)
    pole = np.random.default_rng(0).random((n_a, n_b))
    out = tmp_path / "fig.epf"
    write_mtex_epf(out, pole, alpha, beta, hkl=(1, 1, 1))
    text = out.read_text()
    assert "hkl=1 1 1" in text
    data = [ln for ln in text.splitlines() if not ln.startswith("#") and ln.strip()]
    assert len(data) == n_a * n_b


def test_xpc_shape_mismatch_raises(tmp_path: Path):
    eta = np.array([0.0, 90.0])
    R = np.array([1.0, 2.0, 3.0])
    int2d = np.zeros((3, 2))  # wrong shape
    with pytest.raises(ValueError):
        write_mtex_xpc(tmp_path / "bad.xpc", int2d, eta, R)
