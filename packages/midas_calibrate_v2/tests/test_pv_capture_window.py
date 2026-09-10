"""The capture phase of ``autocalibrate_pv``.

The fine window (``half_window_px``, 4 px) only sees rings the geometry already
puts within a few px. On a large-tilt, off-panel CeO2 frame the tilt-blind seed
put 24 of 30 rings further than that (median 10.4 px), and the fine-window loop
settled 1.9 deg from the rings in tz; a 25 px window alone reached them. These
pin the pieces that make a wide window safe as a default: it is never wide enough
to hold a neighbouring ring, it really does catch a ring the fine window cannot,
and each fit is labelled with the ring it belongs to.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_calibrate_v2.forward.peak_fit_batched import fit_cake_per_ring_batched
from midas_calibrate_v2.pipelines.single_pv import (
    _CAPTURE_GAP_FRAC, _capture_windows_px, _fit_capture_windows,
)


def test_windows_capped_by_the_gap_and_floored_at_the_fine_window():
    w = _capture_windows_px(np.array([100.0, 300.0, 312.0, 700.0]),
                            capture_window_px=25.0, fine_window_px=4.0)
    assert w[0] == 25.0 and w[3] == 25.0
    assert w[1] == pytest.approx(_CAPTURE_GAP_FRAC * 12.0) and w[2] == w[1]
    assert np.all(_capture_windows_px(np.array([100.0, 105.0]), capture_window_px=25.0,
                                      fine_window_px=4.0) == 4.0)
    assert _capture_windows_px(np.array([250.0]), capture_window_px=25.0,
                               fine_window_px=4.0)[0] == 25.0


def test_a_window_holding_its_ring_cannot_hold_the_neighbour():
    for gap in np.linspace(9.0, 200.0, 60):
        w = _capture_windows_px(np.array([500.0, 500.0 + gap]), capture_window_px=25.0,
                                fine_window_px=4.0)[0]
        assert w < gap / 2.0, (gap, w)


def _cake(r_true, *, R_lo=150.0, R_hi=470.0, dR=0.5, n_eta=36, sigma=1.2, amp=1000.0, seed=0):
    R = torch.arange(R_lo, R_hi, dR, dtype=torch.float64)
    eta = torch.linspace(-175.0, 175.0, n_eta, dtype=torch.float64)
    prof = torch.full_like(R, 50.0)
    for r in r_true:
        prof = prof + amp * torch.exp(-0.5 * ((R - r) / sigma) ** 2)
    g = torch.Generator().manual_seed(seed)
    cake = prof[:, None].expand(-1, n_eta) + 3.0 * torch.randn(len(R), n_eta, generator=g,
                                                               dtype=torch.float64)
    return cake.contiguous(), R, eta


R_IDEAL = torch.tensor([200.0, 400.0, 412.0], dtype=torch.float64)   # ring 0 isolated; 1, 2 a pair
R_TRUE = [210.0, 400.0, 412.0]                                          # ring 0 sits 10 px out


def test_a_wide_window_needs_the_peak_start():
    """A wide window alone is not enough. Started at the predicted ring, the LM
    does not walk to a narrow peak 10 px away; it rails at the window bound.
    Started at the brightest bin, it lands on the peak."""
    cake, R, eta = _cake(R_TRUE)
    ring0 = R_IDEAL[:1]
    centre = fit_cake_per_ring_batched(cake, R, eta, ring0, half_window_px=25.0)
    assert abs(float(centre.R_fit.median()) - 210.0) > 5.0, float(centre.R_fit.median())
    peak = fit_cake_per_ring_batched(cake, R, eta, ring0, half_window_px=25.0, init_center="peak")
    assert abs(float(peak.R_fit.median()) - 210.0) < 0.3, float(peak.R_fit.median())


def test_capture_fits_keep_their_ring_labels():
    cake, R, eta = _cake(R_TRUE)
    cap = _fit_capture_windows(cake, R, eta, R_IDEAL, capture_window_px=25.0, fine_window_px=4.0)
    for k, r in enumerate(R_TRUE):
        got = cap.R_fit[cap.ring_idx == k]
        assert got.numel() == eta.numel(), (k, got.numel())
        assert abs(float(got.median()) - r) < 0.3, (k, float(got.median()), r)


def test_a_clipped_capture_window_is_dropped():
    """A ring just beyond the panel edge: its peak is off the detector, its tail is
    on it. Unguarded, the wide window fits that tail and the fits pass the SNR cut
    (measured on an off-panel Eiger frame: +23 px, and BC_y pulled onto its bound).
    With the coverage map the clipped windows are dropped, and a whole ring is kept."""
    R = torch.arange(150.0, 470.0, 0.5, dtype=torch.float64)
    eta = torch.linspace(-175.0, 175.0, 36, dtype=torch.float64)
    prof = torch.full_like(R, 50.0)
    for r in (200.0, 400.0):
        prof = prof + 1000.0 * torch.exp(-0.5 * ((R - r) / 3.0) ** 2)
    g = torch.Generator().manual_seed(1)
    cake = prof[:, None].expand(-1, eta.numel()) + 3.0 * torch.randn(len(R), eta.numel(), generator=g, dtype=torch.float64)
    coverage = torch.ones_like(cake)
    off_panel = R < 208.0                  # ring 0's peak lies off the detector
    coverage[off_panel, :] = 0.0
    cake = torch.where(coverage > 0, cake, torch.zeros_like(cake)).contiguous()
    ideal = torch.tensor([200.0, 400.0], dtype=torch.float64)
    unguarded = _fit_capture_windows(cake, R, eta, ideal, capture_window_px=25.0, fine_window_px=4.0)
    edge = unguarded.R_fit[(unguarded.ring_idx == 0) & (unguarded.snr >= 3.0)]
    # the null: without the guard the clipped ring does produce passing fits, off its true centre
    assert edge.numel() > 0 and abs(float(edge.median()) - 200.0) > 5.0, (edge.numel(), float(edge.median()) if edge.numel() else None)
    guarded = _fit_capture_windows(cake, R, eta, ideal, capture_window_px=25.0, fine_window_px=4.0,
                                   coverage=coverage.numpy())
    assert int((guarded.ring_idx == 0).sum()) == 0
    whole = guarded.R_fit[guarded.ring_idx == 1]
    assert whole.numel() == eta.numel() and abs(float(whole.median()) - 400.0) < 0.3
