"""SubpixelBinGeometry warns when the radial bin width sits on a sampling-aliasing resonance of the K x K sub-sample grid.

Basis (adversarial verification on noiseless synthetic scenes, 2026-10-14, established for the narrowed claim): the sector-to-sector scatter of fitted ring positions spikes at
exact radial bin widths w = m/K px (K=4: 0.25 / 0.5 / 0.75 / 1.0 px) and at sqrt(2)-related widths (K*w = 1.414); spikes are narrower than 0.01 px.
Pins: the helper's resonance set, the warning (class, filterable, once per build), that it changes nothing numerically, and that off-resonance,
K = 1 and K > 8 stay silent.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import warnings

import pytest
import torch

from midas_integrate_v2 import SubpixelBinGeometry
from midas_integrate_v2.binning.subpixel import SubpixelResonanceWarning, resonant_bin_width

from test_spec_pixel_corrections import _spec


@pytest.mark.parametrize("K,w", [(4, 0.25), (4, 0.5), (4, 0.75), (4, 1.0), (2, 0.5), (8, 0.25), (4, 0.3536), (4, 0.249)])
def test_resonant_widths_are_flagged(K, w):
    assert resonant_bin_width(K, w) is not None


@pytest.mark.parametrize("K,w", [(4, 0.2), (4, 0.3), (4, 0.26), (4, 0.225), (8, 0.2), (8, 0.3), (1, 0.5), (1, 1.0), (16, 0.25), (4, 0.0)])
def test_off_resonance_k1_and_large_k_are_silent(K, w):
    assert resonant_bin_width(K, w) is None


def _build(K, w, **kw):
    s = _spec()
    s.RBinSize = w
    s.RMax = s.RMin + w * 20
    return SubpixelBinGeometry.from_spec(s, K=K, **kw)


def test_warning_is_raised_once_per_build_and_names_the_direction():
    with pytest.warns(SubpixelResonanceWarning, match=r"K=4 with RBinSize=0\.25") as rec:
        _build(4, 0.25)
    assert len([r for r in rec if issubclass(r.category, SubpixelResonanceWarning)]) == 1
    with pytest.warns(SubpixelResonanceWarning, match="diagonal"):
        _build(4, 0.3536)


def test_no_warning_off_resonance():
    with warnings.catch_warnings():
        warnings.simplefilter("error", SubpixelResonanceWarning)
        _build(4, 0.2)
        _build(4, 0.3)
        _build(1, 0.5)
        _build(16, 0.25)


def test_warning_changes_nothing_numerically_and_is_filterable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SubpixelResonanceWarning)
        a = _build(4, 0.25)
    with pytest.warns(SubpixelResonanceWarning):
        b = _build(4, 0.25)
    assert torch.equal(a.flat_bin, b.flat_bin) and torch.equal(a.valid, b.valid)


def test_q_mode_is_not_checked():
    s = _spec()
    s.RBinSize = 0.25
    s.Wavelength = torch.tensor(0.18, dtype=torch.float64)
    s.QMin, s.QMax, s.QBinSize = 0.5, 2.0, 0.05
    assert s.q_mode_active
    with warnings.catch_warnings():
        warnings.simplefilter("error", SubpixelResonanceWarning)
        SubpixelBinGeometry.from_spec(s, K=4)
