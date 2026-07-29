import json
from pathlib import Path

import numpy as np
import pytest

from midas_defect.reports import (
    cover_figure,
    length_scale_hierarchy,
    load_master_inventory_csv,
    matrix_twin_summary_figure,
    schmid_mechanism_figure,
    write_master_inventory_csv,
)
from midas_defect.types import AnalysisResult, BootUnit


def _result(name: str, median: float, units: str = "m^-2", per_grain=None):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    boot = rng.normal(median, 0.05 * abs(median) + 1e-12, size=64)
    return AnalysisResult(
        name=name,
        units=units,
        boot_unit=BootUnit.GRAIN,
        n_boot=64,
        population_median=float(np.median(boot)),
        population_ci=(float(np.percentile(boot, 16)), float(np.percentile(boot, 84))),
        bootstrap_samples=boot,
        per_grain=per_grain,
        metadata={"source": name, "_test": 1},
    )


# -- inventory --------------------------------------------------------------

def test_inventory_roundtrip_preserves_results(tmp_path: Path):
    rng = np.random.default_rng(0)
    results = [
        _result("rho_matrix", 7.74e12, per_grain=rng.normal(7.74e12, 1e12, size=50)),
        _result("rho_twin", 3.95e12, per_grain=rng.normal(3.95e12, 8e11, size=50)),
        _result("q_U_matrix", 1.96, units="dimensionless"),
    ]
    out_path = tmp_path / "inventory.csv"
    write_master_inventory_csv(results, out_path)
    assert out_path.exists()
    loaded = load_master_inventory_csv(out_path)
    assert len(loaded) == 3
    for r_in, r_out in zip(results, loaded):
        assert r_out.name == r_in.name
        assert r_out.units == r_in.units
        assert r_out.boot_unit is r_in.boot_unit
        assert r_out.n_boot == r_in.n_boot
        np.testing.assert_allclose(r_out.bootstrap_samples, r_in.bootstrap_samples)
        if r_in.per_grain is not None:
            np.testing.assert_allclose(r_out.per_grain, r_in.per_grain)
        else:
            assert r_out.per_grain is None
        assert r_out.metadata["source"] == r_in.name
        assert r_out.metadata["_test"] == 1


def test_inventory_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c" / "inventory.csv"
    results = [_result("x", 1.0)]
    write_master_inventory_csv(results, nested)
    assert nested.exists()


def test_inventory_handles_empty_results(tmp_path: Path):
    out_path = tmp_path / "empty.csv"
    write_master_inventory_csv([], out_path)
    loaded = load_master_inventory_csv(out_path)
    assert loaded == []


# -- figures ----------------------------------------------------------------

def _is_png(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_matrix_twin_summary_figure_produces_png(tmp_path: Path):
    rng = np.random.default_rng(0)
    results = [
        _result("rho_matrix", 7.74e12, per_grain=rng.normal(7.74e12, 1e12, size=20)),
        _result("rho_twin", 3.95e12, per_grain=rng.normal(3.95e12, 8e11, size=20)),
        _result("q_U_matrix", 1.96),
        _result("q_U_twin", 2.99),
    ]
    out_path = tmp_path / "summary.png"
    matrix_twin_summary_figure(results, out_path)
    assert out_path.exists() and _is_png(out_path)


def test_matrix_twin_summary_handles_missing_pairs_without_crashing(tmp_path: Path):
    # rho has both; q_U has only matrix -> figure still renders.
    results = [
        _result("rho_matrix", 7.0e12),
        _result("rho_twin", 3.0e12),
        _result("q_U_matrix", 1.96),
    ]
    out_path = tmp_path / "summary_partial.png"
    matrix_twin_summary_figure(results, out_path, metrics=["rho", "q_U"])
    assert out_path.exists() and _is_png(out_path)


def test_schmid_mechanism_figure_produces_png(tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 200
    schmid = rng.uniform(0.20, 0.50, size=n)
    dEps = (schmid - 0.30) * 0.02 + rng.normal(scale=0.001, size=n)
    tier_edges = np.percentile(schmid, [0, 33, 67, 100])
    out_path = tmp_path / "schmid.png"
    schmid_mechanism_figure(schmid, dEps, tier_edges, out_path)
    assert out_path.exists() and _is_png(out_path)


def test_length_scale_hierarchy_produces_png(tmp_path: Path):
    out_path = tmp_path / "hierarchy.png"
    length_scale_hierarchy(
        {
            "lamella 9R": (31.0, 25.0, 40.0),
            "coherent D": (200.0, 150.0, 280.0),
            "correlation length": (5e4, 3e4, 7e4),
        },
        out_path,
    )
    assert out_path.exists() and _is_png(out_path)


def test_cover_figure_produces_png(tmp_path: Path):
    rng = np.random.default_rng(0)
    results = [
        _result("rho_matrix", 7.74e12, per_grain=rng.lognormal(np.log(7.74e12), 0.5, size=50)),
        _result("rho_twin", 3.95e12, per_grain=rng.lognormal(np.log(3.95e12), 0.5, size=50)),
        _result("alpha_SF", 1.08e-4),
        _result("Friedel_asym", 0.174),
    ]
    out_path = tmp_path / "cover.png"
    cover_figure(results, out_path, headline_metrics=["rho_matrix", "rho_twin", "alpha_SF"])
    assert out_path.exists() and _is_png(out_path)
