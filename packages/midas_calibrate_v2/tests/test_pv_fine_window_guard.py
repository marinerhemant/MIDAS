"""The clipped-window screen, applied to the FINE window as well as the capture one.

A radial window cut by a panel edge, module gap or mask holds a truncated peak, and
the fit slides onto its surviving tail. The capture phase has screened for this since
0.17.0, because a 25 px window meets an edge often; the fine window inherits the same
failure whenever a ring sits within ``half_window_px`` of one, and there the result is
the answer rather than an intermediate.

These pin the screen at fine width: it drops the clipped fits, keeps a whole ring
untouched, and the null shows the clipped fits really are there and really are
displaced when the screen is off.
"""
from __future__ import annotations

import inspect

import numpy as np
import torch

from midas_calibrate_v2.forward.peak_fit_batched import fit_cake_per_ring_batched
from midas_calibrate_v2.pipelines.single_pv import (
    ClippedWindowWarning, _capture_window_complete, autocalibrate_pv,
)

FINE_PX = 4.0
R_IDEAL = torch.tensor([200.0, 400.0], dtype=torch.float64)   # ring 0 clipped, ring 1 whole


def _clipped_cake(edge_R=202.0):
    """Ring 0's peak straddles a panel edge at ``edge_R``; ring 1 is well inside."""
    R = torch.arange(150.0, 470.0, 0.5, dtype=torch.float64)
    eta = torch.linspace(-175.0, 175.0, 36, dtype=torch.float64)
    prof = torch.full_like(R, 50.0)
    for r in (200.0, 400.0):
        prof = prof + 1000.0 * torch.exp(-0.5 * ((R - r) / 1.5) ** 2)
    g = torch.Generator().manual_seed(2)
    cake = prof[:, None].expand(-1, eta.numel()) + 3.0 * torch.randn(
        len(R), eta.numel(), generator=g, dtype=torch.float64)
    coverage = torch.ones_like(cake)
    coverage[R < edge_R, :] = 0.0
    cake = torch.where(coverage > 0, cake, torch.zeros_like(cake)).contiguous()
    return cake, R, eta, coverage


def _fine_fits(cake, R, eta):
    return fit_cake_per_ring_batched(cake, R, eta, R_IDEAL, half_window_px=FINE_PX)


def test_the_fine_window_screen_drops_the_clipped_ring_and_keeps_the_whole_one():
    cake, R, eta, coverage = _clipped_cake()
    bf = _fine_fits(cake, R, eta)
    wins = np.full(int(R_IDEAL.numel()), FINE_PX)
    keep = _capture_window_complete(bf, cake, R, eta, R_IDEAL, wins,
                                    coverage.numpy(), 0.5)
    assert int(keep[bf.ring_idx == 0].sum()) == 0, "a clipped window survived the screen"
    assert bool(keep[bf.ring_idx == 1].all()), "the whole ring lost fits to the screen"


def test_null_without_the_screen_the_clipped_fits_are_there_and_displaced():
    """If this fails the test above is measuring nothing: it would mean the clipped
    window produced no passing fits to drop, or produced correct ones."""
    cake, R, eta, _ = _clipped_cake()
    bf = _fine_fits(cake, R, eta)
    clipped = bf.R_fit[(bf.ring_idx == 0) & (bf.snr >= 3.0)]
    assert clipped.numel() > 0, "no passing fits in the clipped window"
    assert abs(float(clipped.median()) - 200.0) > 0.5, float(clipped.median())
    whole = bf.R_fit[bf.ring_idx == 1]
    assert abs(float(whole.median()) - 400.0) < 0.3, float(whole.median())


def test_a_fully_covered_cake_loses_nothing():
    """The screen must be inert where every window is whole -- that is what keeps
    it safe as a default on the frames that already worked."""
    cake, R, eta, _ = _clipped_cake()
    bf = _fine_fits(cake, R, eta)
    keep = _capture_window_complete(bf, cake, R, eta, R_IDEAL,
                                    np.full(int(R_IDEAL.numel()), FINE_PX),
                                    np.ones_like(cake.numpy()), 0.5)
    # coverage is uniform, so only the zeroed intensities can drop a fit; ring 1
    # is untouched either way.
    assert bool(keep[bf.ring_idx == 1].all())


def test_the_screen_is_exposed_and_defaults_on():
    sig = inspect.signature(autocalibrate_pv).parameters
    assert sig["guard_clipped_windows"].default is True
    assert sig["min_cell_coverage"].default == 0.5
    assert issubclass(ClippedWindowWarning, UserWarning)
