"""Item 25 — Pole-figure exporter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_integrate_v2.texture import cake_to_pole_figure, write_popla_pol


def _synthetic_cake():
    n_eta, n_r = 360, 64
    eta = np.linspace(-180.0, 180.0, n_eta, endpoint=False)
    R = np.linspace(0.0, 64.0, n_r)
    int2d = np.zeros((n_eta, n_r))
    # Plant ring at R=20 with η-dependent intensity
    ring_idx = (R - 20.0).argmin().__index__() if False else int(np.argmin(np.abs(R - 20.0)))
    int2d[:, ring_idx] = 100.0 + 50.0 * np.cos(np.deg2rad(eta) * 2.0)
    return int2d, eta, R


def test_cake_to_pole_basic_shapes():
    int2d, eta, R = _synthetic_cake()
    a, b, intensity = cake_to_pole_figure(
        int2d, eta, R, hkl_R_px=20.0,
        capture_radius_px=1.0,
        output_grid=(91, 181),
    )
    assert a.shape == (91,)
    assert b.shape == (181,)
    assert intensity.shape == (181, 91)
    # The non-zero stripe should sit at α=0 (chi=0 default)
    nz = (intensity != 0).any(axis=0)
    assert nz[0]
    assert intensity.sum() > 0


def test_cake_to_pole_no_ring_raises():
    int2d, eta, R = _synthetic_cake()
    with pytest.raises(ValueError):
        cake_to_pole_figure(int2d, eta, R, hkl_R_px=10000.0,
                              capture_radius_px=0.5)


def test_popla_writer(tmp_path: Path):
    int2d, eta, R = _synthetic_cake()
    a, b, intensity = cake_to_pole_figure(
        int2d, eta, R, hkl_R_px=20.0,
        capture_radius_px=1.0,
        output_grid=(91, 181),
    )
    out = tmp_path / "ring111.pol"
    write_popla_pol(out, a, b, intensity, hkl=(1, 1, 1))
    text = out.read_text()
    assert "hkl=1 1 1" in text
    data_rows = [ln for ln in text.splitlines() if not ln.startswith("#") and ln.strip()]
    assert len(data_rows) == 181
