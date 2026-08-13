"""Variance propagation and sinogram assembly."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.channels import Channel
from midas_dt.reduce import poisson_variance
from midas_dt.sinogram import SinogramStack, assemble


# ------------------------------------------------------------- variance
def test_poisson_variance_equals_counts():
    counts = np.array([0.0, 1.0, 100.0, 10_000.0])
    np.testing.assert_allclose(poisson_variance(counts), counts)


def test_negative_counts_clip_for_variance_only():
    """Over-subtraction gives negative intensity, which is legitimate; a
    negative VARIANCE is not, so only the variance is clipped."""
    var = poisson_variance(np.array([-5.0, 10.0]))
    assert var[0] == 0.0
    assert var[1] == 10.0


def test_dark_subtraction_adds_its_own_noise():
    """Subtracting a dark removes its signal but ADDS its noise."""
    counts = np.array([100.0])
    dark = np.array([9.0])
    assert poisson_variance(counts, dark=dark)[0] == pytest.approx(109.0)


def test_read_noise_is_a_floor_in_quadrature():
    assert poisson_variance(np.array([0.0]), read_noise=3.0)[0] == pytest.approx(9.0)


def test_gain_scales_variance():
    # counts/gain: a gain of 2 means each count is 2 detector units
    assert poisson_variance(np.array([100.0]), gain=2.0)[0] == pytest.approx(50.0)


# ------------------------------------------------------------ assembly
def _fake(n_trans=4, n_frames=6, n_eta=2, n_r=3, seed=0):
    rng = np.random.default_rng(seed)
    inten = rng.uniform(1, 100, (n_trans, n_frames, n_eta, n_r))
    return inten, inten.copy()          # Poisson: variance == counts


def test_assemble_transposes_to_tomo_layout():
    inten, var = _fake()
    st = assemble(inten, var, np.linspace(0, 180, 6), Channel(105, 125), snake=False)
    # (translation, frame, eta, r) -> (eta*r, frame, translation)
    assert st.intensity.shape == (2 * 3, 6, 4)
    assert st.n_bins == 6 and st.n_omega == 6 and st.n_translations == 4


def test_assemble_preserves_values_at_the_right_index():
    inten, var = _fake()
    st = assemble(inten, var, np.linspace(0, 180, 6), Channel(105, 125), snake=False)
    for e in range(2):
        for r in range(3):
            b = st.bin_index(e, r)
            np.testing.assert_allclose(st.intensity[b, :, :], inten[:, :, e, r].T)


def test_snake_correction_applies_to_variance_too():
    """If only the intensity were un-snaked, sigma would sit on the wrong
    pixel -- and no test of the intensity alone would notice."""
    inten, var = _fake()
    var = var * 3.0                       # make them distinguishable
    st = assemble(inten, var, np.linspace(0, 180, 6), Channel(105, 125), snake=True)
    np.testing.assert_allclose(st.variance, st.intensity * 3.0)


def test_snake_flag_is_recorded():
    inten, var = _fake()
    st = assemble(inten, var, np.linspace(0, 180, 6), Channel(105, 125), snake=True)
    assert st.limits.snake_corrected is True
    assert any("Self-absorption" in w for w in st.limits.warnings())


def test_sigma_is_sqrt_variance():
    inten, var = _fake()
    st = assemble(inten, var, np.linspace(0, 180, 6), Channel(105, 125), snake=False)
    np.testing.assert_allclose(st.sigma, np.sqrt(st.variance))


def test_assemble_rejects_shape_mismatch():
    inten, var = _fake()
    with pytest.raises(ValueError, match="must match"):
        assemble(inten, var[:, :-1], np.linspace(0, 180, 6), Channel(105, 125), snake=False)


def test_assemble_rejects_wrong_omega_length():
    inten, var = _fake()
    with pytest.raises(ValueError, match="omega has"):
        assemble(inten, var, np.linspace(0, 180, 5), Channel(105, 125), snake=False)


def test_assemble_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expected \(translation, frame"):
        assemble(np.zeros((3, 4)), np.zeros((3, 4)), np.zeros(4),
                 Channel(105, 125), snake=False)


def test_bin_index_is_bounds_checked():
    inten, var = _fake()
    st = assemble(inten, var, np.linspace(0, 180, 6), Channel(105, 125), snake=False)
    with pytest.raises(IndexError):
        st.bin_index(5, 0)
