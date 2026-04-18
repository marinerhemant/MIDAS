"""Wizard behavior tests.

Covers:
  - OmegaEnd derivation (from OmegaStart + OmegaStep × nFrames)
  - `back` navigation returning to the previous prompt
  - Registry-level correctness (MinConfidence/MinFracAccept NF-only,
    OmegaEnd no longer required)
"""

from __future__ import annotations

import builtins

import pytest

from midas_params import Path
from midas_params.registry import by_name, required_for
from midas_params.wizard import (
    WizardState,
    _derive_seeds,
    _prompt_for,
)


# ─── Path-scoping regressions ────────────────────────────────────────────────


def test_omega_end_not_required_for_ff():
    required = {s.name for s in required_for(Path.FF)}
    assert "OmegaEnd" not in required


def test_omega_end_not_required_for_pf():
    required = {s.name for s in required_for(Path.PF)}
    assert "OmegaEnd" not in required


def test_min_confidence_nf_only():
    b = by_name()
    spec = b["MinConfidence"]
    assert spec.applies_to == frozenset({Path.NF})


def test_min_frac_accept_nf_only():
    b = by_name()
    spec = b["MinFracAccept"]
    assert spec.applies_to == frozenset({Path.NF})


def test_completeness_not_in_nf_path():
    """Completeness is FF/PF concept; NF uses MinConfidence."""
    b = by_name()
    assert Path.NF not in b["Completeness"].applies_to


# ─── OmegaEnd derivation ─────────────────────────────────────────────────────


def test_derive_omega_end_positive_step():
    state = WizardState(
        values={},
        seed={"OmegaStart": 0.0, "OmegaStep": 0.25, "StartNr": 1, "EndNr": 721},
        source={},
        path=Path.FF,
    )
    _derive_seeds(state)
    assert "OmegaEnd" in state.seed
    # 0 + 0.25 * (721 - 1 + 1) = 0 + 0.25 * 721 = 180.25
    assert state.seed["OmegaEnd"] == pytest.approx(180.25)
    assert "derived" in state.source["OmegaEnd"]


def test_derive_omega_end_negative_step():
    state = WizardState(
        values={},
        seed={"OmegaStart": 180.0, "OmegaStep": -0.25, "StartNr": 1, "EndNr": 1440},
        source={},
        path=Path.FF,
    )
    _derive_seeds(state)
    # 180 + (-0.25) * 1440 = 180 - 360 = -180
    assert state.seed["OmegaEnd"] == pytest.approx(-180)


def test_derive_omega_end_respects_existing_seed():
    """If the user already supplied OmegaEnd, derivation must not overwrite."""
    state = WizardState(
        values={},
        seed={"OmegaStart": 0.0, "OmegaStep": 0.25,
              "StartNr": 1, "EndNr": 721, "OmegaEnd": 99.0},
        source={"OmegaEnd": "param-file:user.txt"},
        path=Path.FF,
    )
    _derive_seeds(state)
    assert state.seed["OmegaEnd"] == 99.0
    # Source should still be the original
    assert "derived" not in state.source["OmegaEnd"]


def test_derive_omega_end_missing_inputs():
    """If any input is missing, no derivation happens."""
    state = WizardState(
        values={},
        seed={"OmegaStart": 0.0, "OmegaStep": 0.25},  # no StartNr/EndNr
        source={},
        path=Path.FF,
    )
    _derive_seeds(state)
    assert "OmegaEnd" not in state.seed


# ─── 'back' navigation ───────────────────────────────────────────────────────


def _with_inputs(responses: list[str]):
    """Context-style helper: monkeypatch input() to return from a list."""
    it = iter(responses)

    def fake_input(prompt=""):
        return next(it)

    return fake_input


def test_back_returns_back_sentinel(monkeypatch):
    """Typing 'back' at a prompt returns 'back' from _prompt_for."""
    monkeypatch.setattr(builtins, "input", _with_inputs(["back"]))
    state = WizardState(values={}, seed={}, source={}, path=Path.FF)
    spec = by_name()["StartNr"]
    result = _prompt_for(spec, state)
    assert result == "back"
    # Nothing committed
    assert "StartNr" not in state.values


def test_skip_returns_none_and_drops_key(monkeypatch):
    """Typing 'skip' on an optional key removes any prior value."""
    monkeypatch.setattr(builtins, "input", _with_inputs(["skip"]))
    # Use an optional key — Padding is optional with default
    spec = by_name()["Padding"]
    state = WizardState(values={"Padding": 4}, seed={}, source={}, path=Path.FF)
    # Padding is optional for FF (not in required_for)
    assert Path.FF not in spec.required_for
    result = _prompt_for(spec, state)
    assert result is None
    assert "Padding" not in state.values


def test_skip_aliases_delete_drop(monkeypatch):
    """'del', 'delete', 'drop' are aliases for 'skip'."""
    spec = by_name()["Padding"]
    for cmd in ("del", "delete", "drop", "!del"):
        monkeypatch.setattr(builtins, "input", _with_inputs([cmd]))
        state = WizardState(values={"Padding": 6}, seed={}, source={}, path=Path.FF)
        result = _prompt_for(spec, state)
        assert result is None
        assert "Padding" not in state.values


def test_skip_works_on_required_with_warning(monkeypatch, capsys):
    """Typing 'skip' on a required key removes it and emits a warning."""
    monkeypatch.setattr(builtins, "input", _with_inputs(["skip"]))
    spec = by_name()["Wavelength"]  # required for FF
    assert Path.FF in spec.required_for
    state = WizardState(values={"Wavelength": 0.22}, seed={}, source={},
                         path=Path.FF)
    result = _prompt_for(spec, state)
    assert result is None
    assert "Wavelength" not in state.values
    out = capsys.readouterr().out
    assert "deleted" in out.lower()
    assert "required" in out.lower()
    assert "validator will flag" in out.lower()


def test_skip_multi_entry(monkeypatch):
    """'skip' in the multi-entry loop also drops the whole key."""
    monkeypatch.setattr(builtins, "input", _with_inputs(["skip"]))
    spec = by_name()["RingThresh"]
    assert spec.multi_entry
    state = WizardState(values={"RingThresh": [[1, 100]]}, seed={}, source={},
                         path=Path.FF)
    result = _prompt_for(spec, state)
    assert result is None
    assert "RingThresh" not in state.values


def test_plain_enter_accepts_seed(monkeypatch):
    monkeypatch.setattr(builtins, "input", _with_inputs([""]))
    spec = by_name()["SpaceGroup"]
    state = WizardState(
        values={}, seed={"SpaceGroup": 225}, source={"SpaceGroup": "seeded"},
        path=Path.FF,
    )
    result = _prompt_for(spec, state)
    assert result is None
    assert state.values["SpaceGroup"] == 225


def test_typed_value_overrides_seed(monkeypatch):
    monkeypatch.setattr(builtins, "input", _with_inputs(["229"]))
    spec = by_name()["SpaceGroup"]
    state = WizardState(
        values={}, seed={"SpaceGroup": 225}, source={"SpaceGroup": "seeded"},
        path=Path.FF,
    )
    result = _prompt_for(spec, state)
    assert result is None
    assert state.values["SpaceGroup"] == 229


def test_revisit_shows_previously_entered_as_default(monkeypatch):
    """After typing a value, backing up and revisiting shows that value as default."""
    state = WizardState(values={"SpaceGroup": 229}, seed={},
                         source={}, path=Path.FF)
    # Simulate user hitting Enter on revisit — previously-entered value should stick
    monkeypatch.setattr(builtins, "input", _with_inputs([""]))
    spec = by_name()["SpaceGroup"]
    result = _prompt_for(spec, state)
    assert result is None
    assert state.values["SpaceGroup"] == 229  # unchanged


def test_required_key_rejects_empty_input(monkeypatch, capsys):
    """Enter on a required key with no seed/default/typical loops with error,
    then user types back to escape."""
    # Wavelength is single-entry, required for FF, no default/typical
    monkeypatch.setattr(builtins, "input", _with_inputs(["", "back"]))
    spec = by_name()["Wavelength"]
    assert Path.FF in spec.required_for
    assert spec.default is None and spec.typical is None
    state = WizardState(values={}, seed={}, source={}, path=Path.FF)
    result = _prompt_for(spec, state)
    # First Enter rejected (required, no seed); second response 'back' escapes
    assert result == "back"
    captured = capsys.readouterr()
    assert "required" in captured.out.lower()


def test_multi_entry_back_before_first_entry(monkeypatch):
    """On a multi-entry required key, typing 'back' before adding any entry
    returns to the previous prompt."""
    # RingThresh is multi-entry + required for FF
    monkeypatch.setattr(builtins, "input", _with_inputs(["back"]))
    spec = by_name()["RingThresh"]
    assert spec.multi_entry
    assert Path.FF in spec.required_for
    state = WizardState(values={}, seed={}, source={}, path=Path.FF)
    result = _prompt_for(spec, state)
    assert result == "back"


# ─── Feedback / confirmation line after each prompt ──────────────────────────


def test_confirmation_line_typed_value(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", _with_inputs(["229"]))
    spec = by_name()["SpaceGroup"]
    state = WizardState(values={}, seed={}, source={}, path=Path.FF)
    _prompt_for(spec, state)
    out = capsys.readouterr().out
    assert "SpaceGroup: 229" in out
    assert "(you entered)" in out


def test_confirmation_line_seed_accepted(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", _with_inputs([""]))
    spec = by_name()["Wavelength"]
    state = WizardState(
        values={}, seed={"Wavelength": 0.22291},
        source={"Wavelength": "param-file:ps.txt"}, path=Path.FF,
    )
    _prompt_for(spec, state)
    out = capsys.readouterr().out
    assert "Wavelength: 0.22291" in out
    assert "(seed)" in out


def test_confirmation_line_default_accepted(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", _with_inputs([""]))
    spec = by_name()["Padding"]
    state = WizardState(values={}, seed={}, source={}, path=Path.FF)
    _prompt_for(spec, state)
    out = capsys.readouterr().out
    assert "Padding: 6" in out
    assert "(default)" in out


def test_confirmation_line_typical_accepted(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", _with_inputs([""]))
    spec = by_name()["MargABC"]
    state = WizardState(values={}, seed={}, source={}, path=Path.FF)
    _prompt_for(spec, state)
    out = capsys.readouterr().out
    assert "MargABC: 4" in out
    assert "(typical)" in out


def test_confirmation_line_skip(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", _with_inputs(["skip"]))
    spec = by_name()["HeadSize"]  # optional
    state = WizardState(values={"HeadSize": 8192}, seed={}, source={}, path=Path.FF)
    _prompt_for(spec, state)
    out = capsys.readouterr().out
    assert "deleted" in out.lower()
    assert "HeadSize" not in state.values


def test_confirmation_multi_entry(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input",
                         _with_inputs(["1 100", "2 150", ""]))
    spec = by_name()["RingThresh"]
    state = WizardState(values={}, seed={}, source={}, path=Path.FF)
    _prompt_for(spec, state)
    out = capsys.readouterr().out
    assert "RingThresh: 2 entries" in out
    assert "you entered" in out


# ─── NrPixels ↔ NrPixelsY/Z derivation ───────────────────────────────────────


def test_derive_nrpixels_from_y_and_z():
    state = WizardState(
        values={}, seed={"NrPixelsY": 2048, "NrPixelsZ": 2048},
        source={}, path=Path.FF,
    )
    _derive_seeds(state)
    assert state.seed["NrPixels"] == 2048
    assert "derived" in state.source["NrPixels"]


def test_derive_nrpixels_from_asymmetric_y_z():
    """Non-square detector: NrPixels = max(Y, Z)."""
    state = WizardState(
        values={}, seed={"NrPixelsY": 1024, "NrPixelsZ": 512},
        source={}, path=Path.FF,
    )
    _derive_seeds(state)
    assert state.seed["NrPixels"] == 1024


def test_derive_y_z_from_nrpixels():
    state = WizardState(
        values={}, seed={"NrPixels": 2048}, source={}, path=Path.FF,
    )
    _derive_seeds(state)
    assert state.seed["NrPixelsY"] == 2048
    assert state.seed["NrPixelsZ"] == 2048
    assert "derived from NrPixels" in state.source["NrPixelsY"]


def test_derive_nrpixels_respects_existing():
    """If user supplied all three, don't overwrite."""
    state = WizardState(
        values={},
        seed={"NrPixels": 999, "NrPixelsY": 1000, "NrPixelsZ": 1001},
        source={"NrPixels": "user"},
        path=Path.FF,
    )
    _derive_seeds(state)
    assert state.seed["NrPixels"] == 999
    assert "user" in state.source["NrPixels"]


# ─── Validator: nrpixels_either_or cross-field rule ──────────────────────────


def test_validator_accepts_nrpixels_alone(tmp_path):
    from midas_params.validator import validate
    fn = tmp_path / "p.txt"
    fn.write_text("NrPixels 2048\n")
    r = validate(str(fn), Path.FF)
    np_errors = [i for i in r.errors if i.rule == "nrpixels_either_or"]
    assert not np_errors


def test_validator_accepts_y_and_z_without_nrpixels(tmp_path):
    from midas_params.validator import validate
    fn = tmp_path / "p.txt"
    fn.write_text("NrPixelsY 2048\nNrPixelsZ 1024\n")
    r = validate(str(fn), Path.FF)
    np_errors = [i for i in r.errors if i.rule == "nrpixels_either_or"]
    assert not np_errors


def test_validator_flags_missing_both(tmp_path):
    from midas_params.validator import validate
    fn = tmp_path / "p.txt"
    fn.write_text("# no detector size\n")
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "nrpixels_either_or" for i in r.errors)


def test_validator_flags_only_y_set(tmp_path):
    """Having only one of Y/Z isn't enough — need both or NrPixels."""
    from midas_params.validator import validate
    fn = tmp_path / "p.txt"
    fn.write_text("NrPixelsY 2048\n")  # no Z, no NrPixels
    r = validate(str(fn), Path.FF)
    assert any(i.rule == "nrpixels_either_or" for i in r.errors)


def test_validator_warns_on_conflict(tmp_path):
    """If user sets all three and they disagree, warn."""
    from midas_params.validator import validate
    fn = tmp_path / "p.txt"
    fn.write_text("NrPixels 1000\nNrPixelsY 2048\nNrPixelsZ 2048\n")
    r = validate(str(fn), Path.FF)
    warns = [i for i in r.warnings if i.rule == "nrpixels_either_or"]
    assert warns
    assert "disagrees" in warns[0].message
