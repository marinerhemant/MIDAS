"""The tolerance-mode sinogram window follows the omega step.

It was a fixed 1 deg in omega and eta. On ESRF ma5608 (OmegaStep 0.124 deg) a
grain's own spots scatter only |d_omega| p90 0.26 / |d_eta| p95 0.10 deg about
their row, and a 1 deg window let in same-ring spots from elsewhere: a
0.25-0.35 deg window cut the sinogram cells lying outside the grain's
projected support from 0.49 to 0.33. The default is now 2 x |OmegaStep|; a
configured value wins; with no OmegaStep the old 1 deg is kept.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from midas_pipeline.config import ReconConfig
from midas_pipeline.stages.find_grains_stage import _read_omega_step, _sino_tolerances


def _layer(tmp_path, text):
    (tmp_path / "paramstest.txt").write_text(text)
    return tmp_path


def test_default_is_twice_the_omega_step(tmp_path):
    layer = _layer(tmp_path, "SpaceGroup 167;\nOmegaStep -0.124309;\n")
    assert _read_omega_step(layer) == pytest.approx(0.124309)
    ome, eta, how = _sino_tolerances(ReconConfig(), layer)
    assert ome == pytest.approx(0.248618) and eta == pytest.approx(0.248618)
    assert how == "2xOmegaStep/2xOmegaStep"


def test_configured_values_win(tmp_path):
    layer = _layer(tmp_path, "OmegaStep 0.25\n")
    cfg = SimpleNamespace(sino_tol_ome_deg=0.35, sino_tol_eta_deg=-1.0)
    ome, eta, how = _sino_tolerances(cfg, layer)
    assert (ome, eta) == (0.35, pytest.approx(0.5))
    assert how == "configured/2xOmegaStep"


def test_no_omega_step_keeps_the_old_one_degree(tmp_path):
    layer = _layer(tmp_path, "SpaceGroup 225\n")
    assert _sino_tolerances(ReconConfig(), layer)[:2] == (1.0, 1.0)
    assert _sino_tolerances(ReconConfig(), tmp_path / "missing")[:2] == (1.0, 1.0)


def test_cli_exposes_the_window():
    from midas_pipeline.cli import _build_parser
    ns = _build_parser().parse_args(["run", "--params", "p.txt", "--result", "r",
                                     "--sino-tol-ome", "0.3", "--sino-tol-eta", "0.15"])
    assert (ns.sino_tol_ome, ns.sino_tol_eta) == (0.3, 0.15)
