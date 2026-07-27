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


def test_non_integer_space_group_raises():
    with pytest.raises(ValueError, match="integer"):
        resolve_calibrant({"a": 5.4116, "sg": 225.9})
    # an integer-valued float is fine (225.0 == 225)
    assert resolve_calibrant({"a": 5.4116, "sg": 225.0})["sg"] == 225


@pytest.mark.parametrize("bad", [
    {"a": 0.0, "sg": 225},
    {"a": -5.4116, "sg": 225},
    {"a": 4.0, "b": -5.0, "c": 6.0, "sg": 16},
])
def test_nonpositive_lattice_length_raises(bad):
    with pytest.raises(ValueError, match="positive"):
        resolve_calibrant(bad)


def test_out_of_range_lattice_angle_raises():
    # sg 14 is monoclinic: beta is free, alpha/gamma forced to 90.
    with pytest.raises(ValueError, match=r"\(0, 180\)"):
        resolve_calibrant({"a": 4.0, "b": 5.0, "c": 6.0, "sg": 14, "beta": 200.0})
