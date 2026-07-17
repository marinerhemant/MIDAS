"""Regression: _build_model must use the paramstest forward-model geometry
(beam centre, detector size, omega, tilts) instead of the old hardcoded
placeholders (BC=1024, n_pixels=2048, omega=-180/0.25/1440).

The placeholder detector (2048²) on a real 2880² Varex + off-by-400px BC
clipped ~80% of theoretical spots off the too-small detector and collapsed
the per-voxel completeness denominator (62 → 7); see the P2-10 diagnosis.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from midas_fit_grain.config import FitConfig

_PARAMSTEST = """\
Distance 800456.4;
px 150.0;
Wavelength 0.2254;
RingNumbers 3;
RingRadii 189927.75;
LatticeParameter 3.592 3.592 3.592 90 90 90;
YBCFit 1435.686;
ZBCFit 1334.789;
txFit -0.2737;
tyFit -0.4244;
tzFit -0.2355;
OmegaStart -180;
OmegaStep 0.25;
NrPixelsY 2880;
NrPixelsZ 2880;
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "paramstest.txt"
    p.write_text(text)
    return p


def test_fitconfig_parses_forward_geometry(tmp_path: Path):
    cfg = FitConfig.from_param_file(str(_write(tmp_path, _PARAMSTEST)))
    assert cfg.y_BC == pytest.approx(1435.686)      # from YBCFit, not 1024
    assert cfg.z_BC == pytest.approx(1334.789)
    assert cfg.tx == pytest.approx(-0.2737)
    assert cfg.ty == pytest.approx(-0.4244)
    assert cfg.tz == pytest.approx(-0.2355)
    assert cfg.omega_start == pytest.approx(-180.0)
    assert cfg.omega_step == pytest.approx(0.25)
    assert cfg.n_frames == 1440                       # derived 360/0.25
    assert cfg.n_pixels_y == 2880 and cfg.n_pixels_z == 2880


def test_ycen_zcen_fallback_when_no_fit(tmp_path: Path):
    """Master-param files carry YCen/ZCen (no *Fit) — must still populate BC."""
    txt = ("Distance 1e6;\npx 150;\nRingNumbers 3;\nRingRadii 500;\n"
           "YCen 1440;\nZCen 1441;\n")
    cfg = FitConfig.from_param_file(str(_write(tmp_path, txt)))
    assert cfg.y_BC == pytest.approx(1440.0)
    assert cfg.z_BC == pytest.approx(1441.0)


def test_build_model_uses_parsed_geometry(tmp_path: Path):
    import numpy as np
    from midas_fit_grain.driver import _build_model
    cfg = FitConfig.from_param_file(str(_write(tmp_path, _PARAMSTEST)))
    hkls_int = np.array([[3, 1, 1]], dtype=np.int64)
    thetas_deg = np.array([6.0], dtype=np.float64)
    ring_nr = np.array([3], dtype=np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # NO placeholder warning
        model, _ = _build_model(cfg, device="cpu", dtype="float64",
                                hkls_int=hkls_int, thetas_deg=thetas_deg,
                                ring_nr=ring_nr)
    assert int(model.n_pixels_y) == 2880
    assert int(model.n_pixels_z) == 2880
    assert float(model.y_BC if not isinstance(model.y_BC, (list, tuple))
                 else model.y_BC[0]) == pytest.approx(1435.686)


def test_build_model_warns_on_missing_geometry(tmp_path: Path):
    from midas_fit_grain.driver import _build_model
    import numpy as np
    # paramstest WITHOUT the forward-geometry keys → placeholder fallback.
    txt = ("Distance 1e6;\npx 150;\nWavelength 0.22;\nRingNumbers 3;\n"
           "RingRadii 500;\nLatticeParameter 3.6 3.6 3.6 90 90 90;\n")
    cfg = FitConfig.from_param_file(str(_write(tmp_path, txt)))
    with pytest.warns(UserWarning, match="missing forward-model geometry"):
        _build_model(cfg, device="cpu", dtype="float64",
                     hkls_int=np.array([[3, 1, 1]], dtype=np.int64),
                     thetas_deg=np.array([6.0]), ring_nr=np.array([3]))
