"""Model-free detector-space peak tracking: windowed_centroid + track_peak_over_raster.

Built on synthetic frame stacks with known ground truth, so each property can fail:

- windowed_centroid recovers a known block's centroid and refuses to report one below the
  SNR floor
- a peak drifting smoothly across a raster is tracked end to end, with per-step
  re-centering (not a single static window) required to follow drift larger than one window
- a position with no real signal stops the walk there (STATUS_NO_SIGNAL) without propagating
- a decoy peak that would pull the centroid too far/too bright is REJECTED, not silently
  accepted as a continuation, but its own (real) values are still recorded rather than dropped
- a rejected step does not enqueue neighbors, same as a no-signal step
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from midas_defect.peak_tracker import (
    windowed_centroid, track_peak_over_raster, snap_to_local_max,
    STATUS_NO_SIGNAL, STATUS_REJECTED, STATUS_TRACKED,
)

RNG = np.random.default_rng(20260913)
NF, NR, NC = 9, 40, 60
BG = 100.0


def _blank():
    return RNG.normal(BG, 1.5, size=(NF, NR, NC))


def _plant(frames, row, col, frame, *, amp=2000.0, half=1):
    frames = frames.copy()
    f0, f1 = frame - half, frame + half + 1
    r0, r1 = row - half, row + half + 1
    c0, c1 = col - half, col + half + 1
    frames[f0:f1, r0:r1, c0:c1] += amp
    return frames


# ---------------------------------------------------------------------------
# windowed_centroid
# ---------------------------------------------------------------------------

def test_windowed_centroid_recovers_planted_block():
    frames = _plant(_blank(), row=20, col=30, frame=4, amp=3000.0, half=1)
    r = windowed_centroid(frames, 20, 30, 4, win_rc=10, win_f=3, min_snr=5.0)
    assert r is not None
    assert r["centroid_row"] == pytest.approx(20.0, abs=0.3)
    assert r["centroid_col"] == pytest.approx(30.0, abs=0.3)
    assert r["centroid_frame"] == pytest.approx(4.0, abs=0.3)
    assert r["intensity"] > 0
    assert r["peak_max"] > r["background"]


def test_windowed_centroid_refuses_below_snr_floor():
    frames = _blank()  # no planted signal anywhere
    r = windowed_centroid(frames, 20, 30, 4, win_rc=10, win_f=3, min_snr=5.0)
    assert r is None


def test_windowed_centroid_empty_window_returns_none():
    frames = _blank()
    r = windowed_centroid(frames, -50, -50, 4, win_rc=10, win_f=3)
    assert r is None


# ---------------------------------------------------------------------------
# snap_to_local_max
# ---------------------------------------------------------------------------

def test_snap_to_local_max_finds_true_peak_from_an_imprecise_click():
    proj = np.zeros((60, 60))
    proj[30, 40] = 500.0
    row, col = snap_to_local_max(proj, row=33, col=37, radius=8)
    assert (row, col) == (30, 40)


def test_snap_to_local_max_clips_to_array_bounds():
    proj = np.zeros((20, 20))
    proj[0, 0] = 10.0
    row, col = snap_to_local_max(proj, row=0, col=0, radius=5)
    assert 0 <= row < 20 and 0 <= col < 20


# ---------------------------------------------------------------------------
# track_peak_over_raster -- normal walk + re-centering
# ---------------------------------------------------------------------------

def _drifting_loader(n_rows, n_cols, row0, col0, frame0, drow, dcol):
    """Position p's blob sits at (row0 + r*drow, col0 + c*dcol, frame0), r,c = divmod(p, n_cols)."""
    def loader(p):
        r, c = divmod(p, n_cols)
        row = int(row0 + r * drow)
        col = int(col0 + c * dcol)
        return _plant(_blank(), row=row, col=col, frame=frame0, amp=3000.0, half=1)
    return loader


def test_tracks_smoothly_drifting_peak_across_a_raster():
    n_rows, n_cols = 3, 3
    loader = _drifting_loader(n_rows, n_cols, row0=15, col0=15, frame0=4, drow=2, dcol=2)
    grids = track_peak_over_raster(loader, n_rows, n_cols, seed_pos=4,
                                   seed_row=19, seed_col=19, seed_frame=4,
                                   win_rc=8, win_f=3, log=lambda *a, **k: None)
    assert grids["tracked"].all()
    assert (grids["status"] == STATUS_TRACKED).all()
    for r in range(n_rows):
        for c in range(n_cols):
            assert grids["centroid_row"][r, c] == pytest.approx(15 + 2 * r, abs=0.3)
            assert grids["centroid_col"][r, c] == pytest.approx(15 + 2 * c, abs=0.3)


def test_recentering_tracks_drift_beyond_a_single_static_window():
    """Total drift (4 steps x 4 px = 16 px) exceeds win_rc=5, but each hop (4 px) does not --
    only works because each step re-centers the search window on the PREVIOUS step's own
    centroid rather than reusing the original seed position."""
    n_rows, n_cols = 1, 5
    loader = _drifting_loader(n_rows, n_cols, row0=20, col0=10, frame0=4, drow=0, dcol=4)
    grids = track_peak_over_raster(loader, n_rows, n_cols, seed_pos=0,
                                   seed_row=20, seed_col=10, seed_frame=4,
                                   win_rc=5, win_f=3, max_step_px=6.0,
                                   log=lambda *a, **k: None)
    assert grids["tracked"].all()
    for c in range(n_cols):
        assert grids["centroid_col"][0, c] == pytest.approx(10 + 4 * c, abs=0.3)
    # a single fixed window from the seed could never have reached the far end directly
    assert abs((10 + 4 * (n_cols - 1)) - 10) > 5  # > win_rc


def test_no_signal_position_stops_the_walk_without_propagating():
    n_rows, n_cols = 1, 4
    real_loader = _drifting_loader(n_rows, n_cols, row0=20, col0=10, frame0=4, drow=0, dcol=1)

    def loader(p):
        if p == 2:
            return _blank()  # position 2 has no real peak at all
        return real_loader(p)

    grids = track_peak_over_raster(loader, n_rows, n_cols, seed_pos=0,
                                   seed_row=20, seed_col=10, seed_frame=4,
                                   win_rc=6, win_f=3, log=lambda *a, **k: None)
    assert grids["status"][0, 0] == STATUS_TRACKED
    assert grids["status"][0, 1] == STATUS_TRACKED
    assert grids["status"][0, 2] == STATUS_NO_SIGNAL
    assert np.isnan(grids["centroid_col"][0, 2])
    # position 3 was never reachable except through 2, which did not propagate
    assert grids["status"][0, 3] == STATUS_NO_SIGNAL
    assert not grids["tracked"][0, 2:].any()


# ---------------------------------------------------------------------------
# track_peak_over_raster -- hijack / continuity guard
# ---------------------------------------------------------------------------

def test_decoy_peak_is_rejected_not_silently_accepted():
    n_rows, n_cols = 1, 3

    def loader(p):
        if p == 0:
            return _plant(_blank(), row=20, col=10, frame=4, amp=3000.0, half=1)
        if p == 1:
            return _plant(_blank(), row=20, col=11, frame=4, amp=3000.0, half=1)  # true, 1px hop
        # p == 2: no true continuation -- only a bright decoy far off within the search window
        return _plant(_blank(), row=20, col=16, frame=4, amp=6000.0, half=1)

    grids = track_peak_over_raster(loader, n_rows, n_cols, seed_pos=0,
                                   seed_row=20, seed_col=10, seed_frame=4,
                                   win_rc=8, win_f=3, max_step_px=3.0,
                                   max_intensity_ratio=10.0,  # isolate the step_px trigger
                                   log=lambda *a, **k: None)
    assert grids["status"][0, 0] == STATUS_TRACKED
    assert grids["status"][0, 1] == STATUS_TRACKED
    assert grids["status"][0, 2] == STATUS_REJECTED
    # the rejected step's own (real) centroid is still recorded, not dropped to NaN
    assert grids["centroid_col"][0, 2] == pytest.approx(16.0, abs=0.3)
    assert not grids["tracked"][0, 2]


def test_rejected_step_does_not_propagate_to_neighbors():
    n_rows, n_cols = 1, 4

    def loader(p):
        if p <= 1:
            return _plant(_blank(), row=20, col=10 + p, frame=4, amp=3000.0, half=1)
        if p == 2:
            return _plant(_blank(), row=20, col=16, frame=4, amp=6000.0, half=1)  # decoy
        return _plant(_blank(), row=20, col=17, frame=4, amp=3000.0, half=1)  # p == 3

    grids = track_peak_over_raster(loader, n_rows, n_cols, seed_pos=0,
                                   seed_row=20, seed_col=10, seed_frame=4,
                                   win_rc=8, win_f=3, max_step_px=3.0,
                                   max_intensity_ratio=10.0, log=lambda *a, **k: None)
    assert grids["status"][0, 2] == STATUS_REJECTED
    # position 3 is only reachable through 2 -- a rejected step must not have enqueued it
    assert grids["status"][0, 3] == STATUS_NO_SIGNAL


# ---------------------------------------------------------------------------
# plot_peak_maps
# ---------------------------------------------------------------------------

def test_plot_peak_maps_runs_on_a_mixed_status_result():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from midas_defect.peak_tracker import plot_peak_maps

    n_rows, n_cols = 1, 3

    def loader(p):
        if p == 0:
            return _plant(_blank(), row=20, col=10, frame=4, amp=3000.0, half=1)
        if p == 1:
            return _plant(_blank(), row=20, col=11, frame=4, amp=3000.0, half=1)
        return _plant(_blank(), row=20, col=16, frame=4, amp=6000.0, half=1)  # decoy -> rejected

    grids = track_peak_over_raster(loader, n_rows, n_cols, seed_pos=0,
                                   seed_row=20, seed_col=10, seed_frame=4,
                                   win_rc=8, win_f=3, max_step_px=3.0,
                                   max_intensity_ratio=10.0, log=lambda *a, **k: None)
    fig, axes = plot_peak_maps(grids, show=False)
    assert axes.size >= 1
    plt.close(fig)


def test_plot_peak_maps_falls_back_without_a_status_array():
    """An .npz saved by a pre-status version of track_peak_over_raster only has `tracked` --
    plotting it must degrade to the old two-state rendering, not raise."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from midas_defect.peak_tracker import plot_peak_maps

    n = (2, 2)
    old_style = {f: np.full(n, 1.0) for f in
                ("centroid_row", "centroid_col", "centroid_frame", "intensity",
                 "sigma_row", "sigma_col", "sigma_frame", "peak_max", "background")}
    old_style["tracked"] = np.array([[True, False], [True, True]])
    fig, axes = plot_peak_maps(old_style, show=False)
    assert axes.size >= 1
    plt.close(fig)
