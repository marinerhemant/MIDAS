"""Registry sanity checks — catch schema drift before validator runs."""

from __future__ import annotations

import pytest

from midas_params.registry import PARAMS, by_name, for_path, required_for, wizard_visible_for
from midas_params.schema import ParamSpec, ParamType, Path, Stage
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


def test_distortion_coefficients_not_applied_to_nf():
    """p0..p14 and tolP0..tolP14 are an FF/PF/RI distortion model. NF uses a
    direct pinhole+tilts inversion with no distortion polynomial — NF users
    should get an 'unknown key' warning if they copy a FF file's p-coefficients
    into an NF config."""
    by = by_name()
    for i in range(15):
        p = by[f"p{i}"]
        assert Path.NF not in p.applies_to, \
            f"p{i} should not apply to NF (NF has no distortion polynomial)"
        tol = by[f"tolP{i}"]
        assert Path.NF not in tol.applies_to, \
            f"tolP{i} should not apply to NF"


def test_distortion_file_not_applied_to_nf():
    by = by_name()
    assert Path.NF not in by["DistortionFile"].applies_to


def test_ri_has_integration_keys():
    """RI should have all the integration-specific keys IntegratorZarrOMP reads."""
    ri_names = {p.name for p in for_path(Path.RI)}
    required_ri = [
        "RMin", "RMax", "RBinSize", "EtaMin", "EtaMax", "EtaBinSize",
        "OmegaStart", "OmegaStep", "OmegaSumFrames",
        "PeakLocation", "FitROIPadding", "SNIPIterations", "AutoDetectPeaks",
        "MultiplePeaks", "DoSmoothing", "FitROIAuto",
        "SolidAngleCorrection", "PolarizationCorrection", "PolarizationFraction",
        "SumImages", "SaveIndividualFrames", "Normalize",
        "DistortionFile", "GradientCorrection",
        "QBinSize", "QMin", "QMax",
    ]
    missing = [k for k in required_ri if k not in ri_names]
    assert not missing, f"RI path missing expected keys: {missing}"
    # Also require the path to be reasonably populated (regression against shrink).
    assert len(ri_names) >= 50, \
        f"RI path has only {len(ri_names)} keys; expected >= 50"


def test_eta_bin_size_is_dual_stage():
    """EtaBinSize is consumed by both indexing (LUT) and integration (caked η)."""
    spec = by_name()["EtaBinSize"]
    assert Stage.INDEXING in spec.stages
    assert Stage.INTEGRATION in spec.stages


def test_doublet_separation_is_calibration_only():
    """DoubletSeparation is used by the calibration executables, not peak fit."""
    spec = by_name()["DoubletSeparation"]
    assert spec.category == "Calibration"
    assert Stage.CALIBRATION in spec.stages
    assert Stage.PEAK_SEARCH not in spec.stages


def test_px_and_lsd_aliases():
    by = by_name()
    assert by["PixelSize"].name == "px"
    assert by["DetDist"].name == "Lsd"
    assert by["Distance"].name == "Lsd"  # pre-existing alias still works


def test_omega_first_file_applies_to_pf_and_ri():
    """OmegaFirstFile: PF canonical (per-scan start) + RI (IntegratorZarrOMP alias)."""
    spec = by_name()["OmegaFirstFile"]
    assert Path.PF in spec.applies_to
    assert Path.RI in spec.applies_to


def test_pf_equals_ff_plus_beamsize():
    """PF uses the same MIDAS_ParamParser as FF, so its applicable set must
    be a superset of FF's. In practice the only PF-exclusive key is BeamSize
    (nScans is shared with FF, defaulting to 1).

    Failing this test means someone accidentally narrowed a key's scope to
    a PF-or-FF subset — which hides keys that the shared parser accepts
    from whichever path was excluded."""
    ff = {p.name for p in for_path(Path.FF)}
    pf = {p.name for p in for_path(Path.PF)}
    ff_only = ff - pf
    pf_only = pf - ff
    assert not ff_only, \
        f"FF has keys PF doesn't (shared parser should accept them): {sorted(ff_only)}"
    assert pf_only == {"BeamSize"}, \
        f"PF's only FF-exclusive key should be BeamSize; got: {sorted(pf_only)}"


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
