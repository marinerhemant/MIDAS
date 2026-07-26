"""Calibrant resolution: named lookups, custom dicts, and crystal-system
validation.

Locks in the fix for PR #56's review findings: registered vs. custom
calibrant dicts must resolve to the same lattice, and a custom dict that
omits a symmetry-required parameter (or supplies a conflicting one) must
raise instead of silently building the wrong metric tensor.
"""
from __future__ import annotations

import pytest

from midas_calibrate_v2.seed.calibrant import CALIBRANTS, resolve_calibrant
import midas_calibrate_v2.pipelines.auto as auto_module
import midas_calibrate_v2.seed.auto_seed as auto_seed_module


def test_named_calibrant_resolves_full_spec():
    cal = resolve_calibrant("CeO2")
    assert cal["name"] == "CeO2"
    assert cal["sg"] == 225
    assert cal["a"] == cal["b"] == cal["c"] == 5.4116
    assert cal["alpha"] == cal["beta"] == cal["gamma"] == 90.0


def test_named_calibrant_lookup_is_case_insensitive():
    assert resolve_calibrant("laB6")["name"] == "LaB6"
    assert resolve_calibrant("AL2O3")["name"] == "Al2O3"


def test_custom_cubic_dict_matches_named():
    named = resolve_calibrant("CeO2")
    custom = resolve_calibrant({"a": 5.4116, "sg": 225})
    assert custom["name"] == "<custom>"
    for key in ("a", "b", "c", "alpha", "beta", "gamma", "sg"):
        assert custom[key] == named[key]


def test_custom_trigonal_dict_without_gamma_matches_registered_al2o3():
    # This is the literal case the review used to demonstrate the 10.3%
    # d-spacing bug: omitting gamma must force 120, not default 90.
    named = CALIBRANTS["Al2O3"]
    custom = resolve_calibrant({"a": 4.7589, "c": 12.9920, "sg": 167})
    assert custom["gamma"] == 120.0
    for key in ("a", "b", "c", "alpha", "beta", "gamma"):
        assert custom[key] == named[key]


def test_hexagonal_dict_missing_c_raises():
    with pytest.raises(ValueError, match="hexagonal"):
        resolve_calibrant({"a": 2.95, "sg": 194})


def test_trigonal_dict_with_conflicting_gamma_raises():
    with pytest.raises(ValueError, match="gamma"):
        resolve_calibrant({"a": 4.7589, "c": 12.9920, "sg": 167, "gamma": 90.0})


def test_missing_required_keys_raises_value_error():
    with pytest.raises(ValueError):
        resolve_calibrant({"sg": 225})
    with pytest.raises(ValueError):
        resolve_calibrant({"a": 5.4116})


def test_unknown_named_calibrant_raises():
    with pytest.raises(KeyError):
        resolve_calibrant("Unobtainium")


def test_invalid_calibrant_type_raises():
    with pytest.raises(TypeError):
        resolve_calibrant(1.0)


def test_registry_is_shared_between_pipelines_and_seed_modules():
    assert auto_module.CALIBRANTS is CALIBRANTS
    assert auto_seed_module._CALIBRANTS is CALIBRANTS
