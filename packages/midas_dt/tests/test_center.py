"""Rotation-axis estimation from the diffraction sinogram."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.center import centre_of_mass_shift
from midas_dt.channels import Channel
from midas_dt.sinogram import assemble


def _stack_with_shift(shift_px=0.0, n_trans=64, n_omega=90, n_bins=2, seed=0):
    """Sinogram of an off-centre blob, with the rotation axis displaced.

    The sample sits well inside the field of view, so the centre-of-mass
    method's truncation assumption holds.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(n_trans, dtype=np.float64)
    centre = (n_trans - 1) / 2.0 + shift_px
    omega = np.linspace(0.0, 180.0, n_omega, endpoint=False)
    inten = np.zeros((n_trans, n_omega, 1, n_bins))
    for b in range(n_bins):
        for i, th in enumerate(np.deg2rad(omega)):
            pos = centre + 6.0 * np.cos(th)          # off-axis feature
            prof = 100.0 * (b + 1) * np.exp(-((x - pos) / 4.0) ** 2)
            inten[:, i, 0, b] = prof + rng.normal(0, 0.4, n_trans)
    return assemble(inten, np.abs(inten), omega, Channel(105, 125), snake=False)


@pytest.mark.parametrize("true_shift", [0.0, 2.0, -3.5])
def test_com_recovers_a_known_shift(true_shift):
    res = centre_of_mass_shift(_stack_with_shift(true_shift))
    assert res.well_determined, res.detail
    assert res.shift == pytest.approx(true_shift, abs=0.4), (
        f"recovered {res.shift:+.3f}, expected {true_shift:+.3f}"
    )


def test_com_fit_recovers_the_cosine_amplitude():
    """The fitted A should match the feature's off-axis radius (6 px)."""
    res = centre_of_mass_shift(_stack_with_shift(0.0))
    assert abs(res.detail["A"]) == pytest.approx(6.0, abs=1.0)


def test_com_flags_a_signal_free_sinogram():
    """Pure noise must be reported as poorly determined, not given a number."""
    rng = np.random.default_rng(1)
    inten = np.abs(rng.normal(0, 1, (32, 60, 1, 2)))
    st = assemble(inten, inten.copy(), np.linspace(0, 180, 60),
                  Channel(105, 125), snake=False)
    res = centre_of_mass_shift(st)
    assert not res.well_determined


def test_com_drops_empty_projections():
    st = _stack_with_shift(1.0)
    st.intensity[:, :10, :] = 0.0        # blank the first ten rotations
    res = centre_of_mass_shift(st, min_signal=1.0)
    assert res.detail["n_projections"] == st.n_omega - 10
    assert res.shift == pytest.approx(1.0, abs=0.6)


def test_com_reports_too_few_usable_projections():
    st = _stack_with_shift(0.0)
    st.intensity[:] = 0.0
    res = centre_of_mass_shift(st, min_signal=1.0)
    assert not res.well_determined
    assert "too few projections" in res.detail["reason"]


def test_describe_marks_a_poor_result():
    st = _stack_with_shift(0.0)
    st.intensity[:] = 0.0
    res = centre_of_mass_shift(st, min_signal=1.0)
    assert "POORLY DETERMINED" in res.describe()
