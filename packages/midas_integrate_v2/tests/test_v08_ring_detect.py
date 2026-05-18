"""v0.8: ring auto-detect + material suggestion."""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from midas_integrate_v2 import (
    detect_rings, suggest_material,
    DetectedRing, MaterialMatch, CALIBRANTS,
)


def _synth_profile(d_spacings_A, *, Lsd_um=657_437.0, px_um=172.0,
                    lam_A=0.172979, RMin=10.0, RMax=1000.0,
                    RBinSize=2.0, peak_width_px=2.0, noise_level=0.01):
    """Build a 1-D profile with planted Gaussian peaks at the predicted
    R for each input d-spacing. Useful for ring-detect tests."""
    n_r = int((RMax - RMin) / RBinSize)
    r_axis = RMin + RBinSize * (np.arange(n_r) + 0.5)
    profile = np.zeros(n_r)
    for d in d_spacings_A:
        two_theta = 2.0 * np.arcsin(lam_A / (2.0 * d))
        R0 = (Lsd_um / px_um) * np.tan(two_theta)
        if R0 < RMin or R0 > RMax:
            continue
        profile += np.exp(-(r_axis - R0) ** 2 / (2 * peak_width_px ** 2))
    rng = np.random.default_rng(42)
    profile += rng.normal(0, noise_level, size=n_r)
    return r_axis, profile


# ── detect_rings ──

def test_detect_rings_finds_planted_peaks():
    """Plant the first 4 CeO₂ rings; detector should find all 4."""
    d_planted = CALIBRANTS["ceo2"][:4]
    r_axis, profile = _synth_profile(d_planted)
    rings = detect_rings(
        r_axis, profile,
        Lsd_um=657_437.0, px_um=172.0, wavelength_A=0.172979,
        min_relative_height=0.5,
    )
    assert len(rings) >= 4
    # Each detected ring's d should be within tolerance of one planted d
    for ring in rings:
        closest_d_diff = min(abs(ring.d_spacing_A - d) for d in d_planted)
        assert closest_d_diff < 0.05


def test_detect_rings_returns_dataclass_fields():
    r_axis, profile = _synth_profile([3.124])
    rings = detect_rings(
        r_axis, profile,
        Lsd_um=657_437.0, px_um=172.0, wavelength_A=0.172979,
        min_relative_height=0.1,
    )
    assert len(rings) >= 1
    r = rings[0]
    assert isinstance(r, DetectedRing)
    assert r.R_px > 0
    assert r.intensity > 0
    assert r.two_theta_deg > 0
    assert r.d_spacing_A > 0


def test_detect_rings_max_rings_limit():
    """When more peaks exist than max_rings, keep the strongest."""
    d_planted = CALIBRANTS["ceo2"][:8]
    r_axis, profile = _synth_profile(d_planted)
    rings = detect_rings(
        r_axis, profile,
        Lsd_um=657_437.0, px_um=172.0, wavelength_A=0.172979,
        min_relative_height=0.1, max_rings=4,
    )
    assert len(rings) <= 4


def test_detect_rings_shape_mismatch_raises():
    with pytest.raises(ValueError, match="profile shape"):
        detect_rings(np.arange(100), np.zeros(50),
                      Lsd_um=1e6, px_um=200, wavelength_A=1.0)


# ── suggest_material ──

def test_suggest_material_picks_correct_calibrant():
    """Plant CeO₂ rings; the suggester should rank CeO₂ first."""
    d_planted = CALIBRANTS["ceo2"][:5]
    r_axis, profile = _synth_profile(d_planted)
    rings = detect_rings(
        r_axis, profile,
        Lsd_um=657_437.0, px_um=172.0, wavelength_A=0.172979,
        min_relative_height=0.5,
    )
    matches = suggest_material(rings)
    assert len(matches) >= 1
    assert matches[0].name == "ceo2"
    assert matches[0].n_matched >= 4


def test_suggest_material_lab6():
    d_planted = CALIBRANTS["lab6"][:5]
    r_axis, profile = _synth_profile(d_planted)
    rings = detect_rings(
        r_axis, profile,
        Lsd_um=657_437.0, px_um=172.0, wavelength_A=0.172979,
        min_relative_height=0.5,
    )
    matches = suggest_material(rings)
    assert matches[0].name == "lab6"


def test_suggest_material_custom_calibrant():
    """User-supplied custom calibrant via the ``custom`` arg."""
    d_planted = [4.0, 2.5, 1.8]
    r_axis, profile = _synth_profile(d_planted)
    rings = detect_rings(
        r_axis, profile,
        Lsd_um=657_437.0, px_um=172.0, wavelength_A=0.172979,
        min_relative_height=0.5,
    )
    matches = suggest_material(
        rings, custom={"my_material": [4.0, 2.5, 1.8]},
    )
    # custom should be ranked first because all 3 detected rings match it
    assert matches[0].name == "my_material"


def test_suggest_material_empty_input_returns_empty():
    matches = suggest_material([])
    assert matches == []


def test_suggest_material_subset_of_candidates():
    d_planted = CALIBRANTS["ceo2"][:4]
    r_axis, profile = _synth_profile(d_planted)
    rings = detect_rings(
        r_axis, profile,
        Lsd_um=657_437.0, px_um=172.0, wavelength_A=0.172979,
        min_relative_height=0.5,
    )
    matches = suggest_material(rings, candidates=["lab6", "si"])
    assert {m.name for m in matches} == {"lab6", "si"}
    # CeO₂ wasn't a candidate so neither lab6 nor si should match well
    assert all(m.n_matched < 4 for m in matches)


def test_built_in_calibrants_present():
    """Sanity check on the builtin calibrant table."""
    assert "ceo2" in CALIBRANTS
    assert "lab6" in CALIBRANTS
    assert "si" in CALIBRANTS
    for name, dl in CALIBRANTS.items():
        assert isinstance(dl, list)
        assert len(dl) >= 5
        assert all(d > 0 for d in dl)
