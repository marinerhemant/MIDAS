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
