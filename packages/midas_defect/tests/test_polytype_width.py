"""The half-max width estimator must not saturate.

Regression tests for a bug found 2026-09-09: the polytype width estimators
reported the extent of the samples above half max, which is biased low by up to
one sample spacing and quantised to that spacing.  On a real calibrant it
returned exactly 1.00 px at six different ring radii -- one saturation level
reported six times, read as six agreeing measurements.

Every test here fails on that estimator.
"""
import math

import numpy as np
import pytest

from midas_defect.polytype._width import fwhm_half_max
from midas_defect.polytype.satellite_doublet import _wfwhm
from midas_defect.polytype.lamella_thickness import _fwhm_radial


def _gauss_samples(fwhm, n=20000, seed=0, bg=0.10):
    """Samples from a Gaussian of the given FWHM in a wide flat background."""
    rng = np.random.default_rng(seed)
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return np.r_[rng.normal(0.0, sigma, n),
                 rng.uniform(-3.0, 3.0, int(bg * n))]


class TestFwhmHalfMax:
    """Tolerances below are set from measurement, not taste.

    The bug only bites when the sampling is COARSE relative to the peak, and
    only when the peak is OFFSET from the sample grid -- a symmetric Gaussian
    centred exactly on a grid point puts its half-max crossings on grid points
    too, and both estimators then return the exact answer.  Every accuracy test
    here therefore uses a coarse grid and an off-grid centre.

    On the grid used below (spacing 0.25, centre offset 0.125) the old estimator
    returned 0.75x / 0.88x / 0.92x of the true FWHM at 1.0 / 2.0 / 3.0, against
    1.04x / 1.01x / 1.00x for this one.
    """

    GRID = np.linspace(-5.0, 5.0, 41)      # spacing 0.25
    CTR = 0.125                            # half a spacing off-grid

    def test_recovers_gaussian_width(self):
        for true_fwhm in (1.0, 2.0, 3.0):
            sigma = true_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            y = np.exp(-0.5 * ((self.GRID - self.CTR) / sigma) ** 2)
            got = fwhm_half_max(self.GRID, y)
            assert got == pytest.approx(true_fwhm, rel=0.08), (
                f"true {true_fwhm}, got {got}")

    def test_recovers_lorentzian_width(self):
        true_fwhm = 2.0
        y = 1.0 / (1.0 + 4.0 * ((self.GRID - self.CTR) / true_fwhm) ** 2)
        assert fwhm_half_max(self.GRID, y) == pytest.approx(true_fwhm, rel=0.06)

    def test_no_bias_when_the_peak_wanders_across_the_grid(self):
        """Sweeping the centre through one full sample spacing must not modulate
        the answer.  The old estimator steps by a whole spacing as the peak
        crosses each sample; this one must stay flat."""
        true_fwhm = 2.0
        sigma = true_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        got = np.array([
            fwhm_half_max(self.GRID, np.exp(-0.5 * ((self.GRID - c) / sigma) ** 2))
            for c in np.linspace(0.0, 0.25, 12)
        ])
        assert np.all(np.isfinite(got))
        assert np.ptp(got) < 0.10 * true_fwhm, f"modulation {np.ptp(got):.4f}"
        assert np.mean(got) == pytest.approx(true_fwhm, rel=0.05)

    def test_is_not_quantised(self):
        """Sweeping the true width must not collapse onto a few discrete outputs.

        This is the rail test.  The old estimator, sampled this way, emits a
        handful of values (multiples of the sample spacing); the interpolating
        one must emit essentially as many distinct values as there are inputs.
        """
        x = np.linspace(-5.0, 5.0, 101)          # spacing 0.1
        widths = np.linspace(0.42, 0.58, 33)     # all within +-1 spacing
        out = []
        for w in widths:
            sigma = w / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            out.append(fwhm_half_max(x, np.exp(-0.5 * (x / sigma) ** 2)))
        out = np.array(out)
        assert np.all(np.isfinite(out))
        assert len(np.unique(np.round(out, 6))) >= 0.8 * len(widths)
        # and monotone in the true width, which a quantised estimator is not
        assert np.all(np.diff(out) > -1e-9)

    def test_subsample_spacing_widths_are_nan_not_invented(self):
        """A peak narrower than the sampling is unresolved -- say so, don't guess."""
        x = np.linspace(-5.0, 5.0, 101)          # spacing 0.1
        sigma = 0.01 / 2.3548                    # FWHM 0.01, a tenth of a spacing
        y = np.exp(-0.5 * (x / sigma) ** 2)
        assert math.isnan(fwhm_half_max(x, y))

    def test_peak_at_edge_is_nan(self):
        x = np.linspace(0.0, 10.0, 101)
        y = np.exp(-x / 2.0)                     # maximum at the first sample
        assert math.isnan(fwhm_half_max(x, y))

    def test_scales_with_the_abscissa(self):
        x = np.linspace(-5.0, 5.0, 401)
        y = np.exp(-0.5 * ((x - 0.013) / 0.5) ** 2)
        base = fwhm_half_max(x, y)
        for k in (0.1, 10.0, 1000.0):
            assert fwhm_half_max(k * x, y) == pytest.approx(k * base, rel=1e-9)

    def test_baseline_is_honoured(self):
        x = np.linspace(-5.0, 5.0, 401)
        sigma = 1.0 / 2.3548
        y = np.exp(-0.5 * (x / sigma) ** 2)
        assert fwhm_half_max(x, y + 7.0, baseline=7.0) == pytest.approx(
            fwhm_half_max(x, y), rel=1e-9)

    def test_degenerate_inputs(self):
        assert math.isnan(fwhm_half_max([0.0, 1.0], [1.0, 1.0]))
        assert math.isnan(fwhm_half_max(np.linspace(0, 1, 10), np.zeros(10)))
        assert math.isnan(fwhm_half_max(np.linspace(0, 1, 10), np.full(10, np.nan)))


class TestCallersUseIt:
    def test_wfwhm_is_unbiased(self):
        """satellite_doublet._wfwhm on its own 40-bin grid.

        Old vs new, measured: 0.78x/1.10x at FWHM 0.30, 0.93x/1.06x at 0.50,
        0.93x/1.04x at 1.00.  rel=0.12 passes the new one and fails the old at
        the narrow end, which is where a 40-bin histogram actually hurts.
        """
        for true_fwhm in (0.30, 0.50, 1.00):
            x = _gauss_samples(true_fwhm)
            got = _wfwhm(x, np.ones_like(x))
            assert got == pytest.approx(true_fwhm, rel=0.12), (
                f"true {true_fwhm}, got {got}")

    def test_wfwhm_does_not_saturate_across_widths(self):
        """Distinct true widths must give distinct answers."""
        got = [_wfwhm(_gauss_samples(w), np.ones(22000)) for w in (0.3, 0.4, 0.5)]
        assert all(np.isfinite(got))
        assert len(set(np.round(got, 4))) == 3
        assert got[0] < got[1] < got[2]

    def test_fwhm_radial_is_unbiased(self):
        """lamella_thickness._fwhm_radial, sampled on its own 80-bin grid."""
        rng = np.random.default_rng(1)
        axis = np.array([0.0, 0.0, 1.0])
        q_target = 2.0
        for true_fwhm in (0.05, 0.10):
            sigma = true_fwhm / 2.3548
            proj = rng.normal(q_target, sigma, 40000)
            qs = np.c_[np.zeros_like(proj), np.zeros_like(proj), proj]
            got = _fwhm_radial(qs, np.ones_like(proj), axis, q_target)
            # old read 0.91x at both widths; new reads 1.02x / 0.99x
            assert got == pytest.approx(true_fwhm, rel=0.06), (
                f"true {true_fwhm}, got {got}")
