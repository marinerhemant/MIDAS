import json

import numpy as np
import pytest

from midas_defect.types import AnalysisResult, BootUnit, CrystalPhase


def _sample_result(n_boot: int = 16, with_per_grain: bool = True) -> AnalysisResult:
    rng = np.random.default_rng(0)
    boot = rng.normal(5.0, 1.0, size=n_boot)
    per_grain = rng.normal(5.0, 1.0, size=20) if with_per_grain else None
    if per_grain is not None:
        per_grain[3] = np.nan  # NaN-tolerance check
    return AnalysisResult(
        name="rho_total",
        units="m^-2",
        boot_unit=BootUnit.GRAIN,
        n_boot=n_boot,
        population_median=float(np.median(boot)),
        population_ci=(float(np.percentile(boot, 16)), float(np.percentile(boot, 84))),
        bootstrap_samples=boot,
        per_grain=per_grain,
        metadata={
            "q_U_grid": np.linspace(-0.5, 3.5, 5),
            "rng_seed": np.int64(123),
            "tier": "T3",
        },
    )


def test_construct_then_roundtrip_via_dict():
    r = _sample_result()
    d = r.to_dict()

    json.dumps(d)  # must be JSON-serializable

    r2 = AnalysisResult.from_dict(d)
    assert r2.name == r.name
    assert r2.units == r.units
    assert r2.boot_unit is BootUnit.GRAIN
    assert r2.n_boot == r.n_boot
    assert r2.population_median == pytest.approx(r.population_median)
    assert r2.population_ci == r.population_ci
    np.testing.assert_allclose(r2.bootstrap_samples, r.bootstrap_samples)
    np.testing.assert_allclose(r2.per_grain, r.per_grain, equal_nan=True)
    np.testing.assert_allclose(r2.metadata["q_U_grid"], r.metadata["q_U_grid"])
    assert r2.metadata["rng_seed"] == 123


def test_population_ci_ordering_enforced():
    with pytest.raises(ValueError, match="population_ci must be ordered"):
        AnalysisResult(
            name="x",
            units="",
            boot_unit=BootUnit.GRAIN,
            n_boot=1,
            population_median=0.0,
            population_ci=(1.0, 0.0),
            bootstrap_samples=np.array([0.0]),
        )


def test_bootstrap_samples_length_must_match_n_boot():
    with pytest.raises(ValueError, match="does not match n_boot"):
        AnalysisResult(
            name="x",
            units="",
            boot_unit=BootUnit.GRAIN,
            n_boot=5,
            population_median=0.0,
            population_ci=(0.0, 0.0),
            bootstrap_samples=np.zeros(3),
        )


def test_per_grain_nan_tolerated_and_survives_roundtrip():
    r = _sample_result()
    assert np.isnan(r.per_grain[3])
    r2 = AnalysisResult.from_dict(r.to_dict())
    assert np.isnan(r2.per_grain[3])
    finite_mask = np.isfinite(r2.per_grain)
    assert finite_mask.sum() == r.per_grain.size - 1


def test_optional_arrays_default_to_none():
    r = _sample_result(with_per_grain=False)
    assert r.per_grain is None
    assert r.per_pair is None
    assert r.per_reflection is None
    d = r.to_dict()
    assert d["per_grain"] is None
    assert d["per_pair"] is None
    assert d["per_reflection"] is None


def test_crystal_phase_enum_values():
    assert CrystalPhase.FCC.value == "FCC"
    assert CrystalPhase.BCC.value == "BCC"
    assert CrystalPhase.HCP.value == "HCP"


def test_bootstrap_samples_must_be_1d():
    with pytest.raises(ValueError, match="must be 1-D"):
        AnalysisResult(
            name="x",
            units="",
            boot_unit=BootUnit.GRAIN,
            n_boot=4,
            population_median=0.0,
            population_ci=(0.0, 0.0),
            bootstrap_samples=np.zeros((2, 2)),
        )
