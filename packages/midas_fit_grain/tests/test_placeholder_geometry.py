"""Missing forward-model geometry must stop the run, not warn and continue.

This was a warning. The run still exited 0, and on a 2880x2880 Varex detector
the 2048x2048 placeholder put refined grain positions 34 um away from two
independent C refiners where the correct geometry puts them 3.4 um away. The
wrong answer was indistinguishable from a real python-refiner defect for
several hours of analysis.

A run that stops is recoverable. A run that completes with a detector that
does not exist is not.
"""

import os

import pytest

from midas_fit_grain import driver


class _Cfg:
    """Minimal stand-in carrying only what _build_model reads for this check."""

    def __init__(self, **kw):
        # a complete, valid geometry
        self.Lsd = 958874.75
        self.y_BC = 1390.28
        self.z_BC = 1422.39
        self.px = 150.0
        self.omega_start = 180.0
        self.omega_step = -0.2
        self.n_frames = 1800
        self.n_pixels_y = 2880
        self.n_pixels_z = 2880
        self.MinEta = 6.0
        self.Wavelength = 0.172979
        for k, v in kw.items():
            setattr(self, k, v)


def _missing_keys(cfg):
    """Re-derive what _build_model considers missing, without building it."""
    placeholders = ("y_BC", "z_BC", "omega_start", "omega_step",
                    "n_frames", "n_pixels_y", "n_pixels_z")
    return [k for k in placeholders if not getattr(cfg, k, 0)]


def test_a_complete_geometry_is_not_flagged():
    assert _missing_keys(_Cfg()) == []


@pytest.mark.parametrize("key", ["n_pixels_y", "n_pixels_z", "omega_step",
                                 "omega_start", "y_BC", "z_BC", "n_frames"])
def test_each_key_is_detected_when_absent(key):
    """Every one of these silently changes predicted spot positions."""
    assert _missing_keys(_Cfg(**{key: 0})) == [key]


def test_env_override_is_spelled_as_documented(monkeypatch):
    """The escape hatch must exist and be the exact name the error names."""
    src = (driver.__file__.replace(".pyc", ".py"))
    with open(src) as fh:
        text = fh.read()
    assert "MIDAS_ALLOW_PLACEHOLDER_GEOMETRY" in text
    # the message must tell the user how to proceed, and where the keys come from
    assert "midas-transforms" in text
    assert "ff_MIDAS" in text, "must name the chain that omits these keys"


def test_refusal_is_the_default_not_the_warning(monkeypatch):
    """Guard against a future edit downgrading this back to a warning."""
    src = driver.__file__.replace(".pyc", ".py")
    with open(src) as fh:
        text = fh.read()
    i = text.index("_missing = [k for k in _PLACEHOLDER")
    block = text[i:i + 2000]
    assert "raise ValueError" in block, "missing geometry must raise by default"
    raise_pos = block.index("raise ValueError")
    warn_pos = block.index("warnings.warn")
    assert raise_pos < warn_pos, (
        "the raise must come first: the warning is only for the explicit "
        "opt-out path"
    )


def test_env_values_accepted(monkeypatch):
    """Accept the usual truthy spellings, reject everything else."""
    for good in ("1", "true", "TRUE", "yes", "on", " 1 "):
        monkeypatch.setenv("MIDAS_ALLOW_PLACEHOLDER_GEOMETRY", good)
        assert os.environ["MIDAS_ALLOW_PLACEHOLDER_GEOMETRY"].strip().lower() \
            in ("1", "true", "yes", "on")
    for bad in ("0", "false", "no", ""):
        monkeypatch.setenv("MIDAS_ALLOW_PLACEHOLDER_GEOMETRY", bad)
        assert os.environ["MIDAS_ALLOW_PLACEHOLDER_GEOMETRY"].strip().lower() \
            not in ("1", "true", "yes", "on")
