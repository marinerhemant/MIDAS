"""apply_omega_offset: a per-detector omega offset reaches every frame's omega (#75 follow-up)."""
from __future__ import annotations

import numpy as np
import pytest

from midas_peakfit.params import ZarrParams
from midas_peakfit.zarr_io import apply_omega_offset, frame_omega


def _params():
    p = ZarrParams()
    p.omegaStart, p.omegaStep = -30.0, 0.25
    return p


def test_offset_shifts_every_frame_by_minus_the_offset():
    p = _params()
    before = np.array([frame_omega(p, i) for i in (0, 7, 239)])
    apply_omega_offset(p, 0.65)
    after = np.array([frame_omega(p, i) for i in (0, 7, 239)])
    np.testing.assert_allclose(after - before, -0.65)


def test_zero_offset_changes_nothing():
    p = _params()
    apply_omega_offset(p, 0.0)
    assert (p.omegaStart, p.omegaStep) == (-30.0, 0.25)


def test_omega_center_override_is_shifted_too():
    p = _params()
    p.omegaCenter = np.array([-29.9, -29.6, -29.3])
    apply_omega_offset(p, 0.5)
    np.testing.assert_allclose([frame_omega(p, i) for i in range(3)], [-30.4, -30.1, -29.8])
