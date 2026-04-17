"""Registry sanity checks — catch schema drift before validator runs."""

from __future__ import annotations

import pytest

from midas_params.registry import PARAMS, by_name, for_path, required_for, wizard_visible_for
from midas_params.schema import ParamSpec, ParamType, Path
from midas_params.validators import VALIDATORS
from midas_params.crossfield import RULES, RULE_SPECS


def test_names_are_unique():
    """Canonical names must not collide."""
    names = [p.name for p in PARAMS]
    assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


def test_no_alias_shadows_canonical():
    """An alias should never be a canonical name of another spec."""
    canonical = {p.name for p in PARAMS}
    for p in PARAMS:
        for a in p.aliases:
            assert a not in canonical or a == p.name, \
                f"Alias {a!r} on {p.name!r} shadows canonical name."


def test_all_validator_refs_resolve():
    """Every string in ParamSpec.validators must exist in VALIDATORS."""
    for p in PARAMS:
        for v in p.validators:
            assert v in VALIDATORS, f"{p.name} references unknown validator {v!r}"


def test_all_crossfield_rules_resolve():
    """Every RULE_SPECS.check must exist in RULES."""
    for rule in RULE_SPECS:
        assert rule.check in RULES, f"Cross-field rule {rule.name} references unknown check {rule.check!r}"


def test_required_is_subset_of_applies_to():
    """Cannot require a parameter for a path it doesn't apply to."""
    for p in PARAMS:
        assert p.required_for <= p.applies_to, \
            f"{p.name}: required_for={p.required_for} not ⊆ applies_to={p.applies_to}"


def test_by_name_resolves_aliases():
    by = by_name()
    # LatticeParameter is an alias of LatticeConstant
    assert by["LatticeParameter"].name == "LatticeConstant"
    # Distance is an alias of Lsd
    assert by["Distance"].name == "Lsd"
    # SGNr is an alias of SpaceGroup
    assert by["SGNr"].name == "SpaceGroup"


def test_path_counts_are_reasonable():
    """Sanity check: FF should have ~100+ applicable params, NF similar."""
    assert len(for_path(Path.FF)) >= 80, "FF path suspiciously small"
    assert len(for_path(Path.NF)) >= 80, "NF path suspiciously small"
    assert len(required_for(Path.FF)) >= 15, "FF required keys suspiciously few"
    assert len(required_for(Path.NF)) >= 15, "NF required keys suspiciously few"


def test_hidden_specs_excluded_from_wizard():
    """hidden_in_wizard=True entries should not appear in wizard_visible_for."""
    all_ff = for_path(Path.FF)
    visible = wizard_visible_for(Path.FF)
    hidden = [p for p in all_ff if p.hidden_in_wizard]
    assert not any(p in visible for p in hidden), \
        "hidden_in_wizard entry leaked into wizard view"


def test_zarr_renames_unique():
    """Two different text-file keys should not map to the same Zarr key,
    unless one is an alias of the other."""
    renames = {}
    for p in PARAMS:
        if p.zarr_rename is None:
            continue
        # Skip composite rename like "YCen+ZCen" — expected to appear once (BC)
        prior = renames.get(p.zarr_rename)
        if prior and prior.name != p.name:
            pytest.fail(
                f"Zarr rename collision: {p.zarr_rename!r} claimed by both "
                f"{prior.name!r} and {p.name!r}"
            )
        renames[p.zarr_rename] = p


def test_defaults_have_correct_type():
    """If default is set, it should match the declared type (loosely)."""
    type_checks = {
        ParamType.INT: int,
        ParamType.BOOL: (int, bool),  # bool parsed from 0/1 int
        ParamType.FLOAT: (int, float),
        ParamType.STR: str,
        ParamType.PATH: str,
    }
    for p in PARAMS:
        if p.default is None:
            continue
        expected = type_checks.get(p.type)
        if expected is None:
            continue  # skip list types
        assert isinstance(p.default, expected), \
            f"{p.name}: default {p.default!r} is not {expected}"
