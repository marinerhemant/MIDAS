"""Saturated regions: flagged, never left behind as unflagged fragments.

End-to-end through the real per-frame chain (midas_peakfit threshold -> regions
-> seed / saturated_region -> fit / build_saturated_rows) and the real merge.

The failure being guarded (AlON 1-ID, 2026-09-22): a bright reflection spread
over several omega frames saturates in its core frames. Those regions used to
be dropped, so the merge stitched the unsaturated edge frames into small spots
with NO flag, and grain matching accepted them -- 4.5 % of large-grain spots,
~30x too faint, and alone a -0.4 slope of log(I/LP) vs log(L).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("midas_peakfit")
torch = pytest.importorskip("torch")
from scipy.special import erf  # noqa: E402

from midas_peakfit.connected import filter_regions_by_size, find_regions  # noqa: E402
from midas_peakfit.fit import fit_regions  # noqa: E402
from midas_peakfit.lm import LMConfig  # noqa: E402
from midas_peakfit.postfit import build_saturated_rows  # noqa: E402
from midas_peakfit.preprocess import apply_threshold  # noqa: E402
from midas_peakfit.seeds import saturated_region, seed_region  # noqa: E402

from midas_transforms.io.csv import saturated_mask  # noqa: E402
from midas_transforms.merge.core import (  # noqa: E402
    SATURATED_RETURN_CODE, _merge_frames, merge_saturated_only,
)

NP = 128
YC = ZC = 64.0
SAT_SPOT = (64.0 + 30.0, 64.0 + 22.0)      # bright, saturates in its core frames
FAR_SPOT = (64.0 - 35.0, 64.0 - 10.0)      # ordinary, never saturates
SIG = 1.5
INT_SAT = 14000.0


def _omega_weights(n_span: int, nf: int) -> np.ndarray:
    so = n_span / 4.0
    edges = np.arange(nf + 1) - nf / 2.0
    w = np.diff(0.5 * (1.0 + erf(edges / (so * math.sqrt(2.0)))))
    return w / w.sum()


def _gauss(y0, z0):
    yy, zz = np.mgrid[0:NP, 0:NP].astype(float)
    g = np.exp(-0.5 * ((yy - y0) ** 2 + (zz - z0) ** 2) / SIG ** 2)
    return g / g.sum()


def _peakfit(sat_total=3.0e6, far_total=2.0e4, n_span=10, nf=24, thresh=10.0):
    """Per-frame (fitted rows, saturated rows) exactly as midas-peakfit writes them."""
    g_sat, g_far = _gauss(*SAT_SPOT), _gauss(*FAR_SPOT)
    w = _omega_weights(n_span, nf)
    good = np.full((NP, NP), thresh)
    frames, sat_frames = [], []
    for k in range(nf):
        img = apply_threshold(sat_total * w[k] * g_sat + far_total * w[k] * g_far, good)
        regs = filter_regions_by_size(find_regions(img, good), 1, 10000)
        srs, sats = [], []
        for r in regs:
            s = seed_region(r, img, None, Ycen=YC, Zcen=ZC, int_sat=INT_SAT,
                            max_n_peaks=50, panels=[])
            if s is None:
                sats.append(saturated_region(r, img, None, Ycen=YC, Zcen=ZC, panels=[]))
            else:
                srs.append(s)
        outs, _ = fit_regions(srs, omega=0.25 * k, Ycen=YC, Zcen=ZC, do_peak_fit=1,
                              local_maxima_only=0, device=torch.device("cpu"),
                              dtype=torch.float64, lm_config=LMConfig())
        rows = np.concatenate([o.rows for o in outs]) if outs else np.zeros((0, 29))
        frames.append(rows)
        sat_frames.append(build_saturated_rows(
            sats, omega=0.25 * k, Ycen=YC, Zcen=ZC, spot_id_start=rows.shape[0] + 1))
    return frames, sat_frames


@pytest.fixture(scope="module")
def peaks():
    frames, sat_frames = _peakfit()
    assert sum(s.shape[0] for s in sat_frames) >= 2, "fixture must saturate >= 2 core frames"
    return frames, sat_frames


def _near(res, yz, r=6.0):
    return np.hypot(res[:, 3] - yz[0], res[:, 4] - yz[1]) < r


def test_legacy_merge_leaves_unflagged_fragments(peaks):
    """The bug, pinned: without the saturated record, the reflection survives
    only as edge fragments that carry no flag and a fraction of its intensity."""
    frames, _ = peaks
    legacy, _ = _merge_frames(frames, overlap_length=2.0)
    frag = legacy[_near(legacy, SAT_SPOT)]
    assert frag.shape[0] >= 1
    assert not saturated_mask(frag[:, 17]).any()
    assert frag[:, 1].max() < 0.2 * 3.0e6


def test_flag_mode_is_legacy_plus_flags(peaks):
    """Default (IncludeSaturatedSpots 0): every row, position and SpotID is the
    legacy one -- indexing input unchanged -- and every fragment of the
    saturated reflection is flagged; the unrelated spot is untouched."""
    frames, sat_frames = peaks
    legacy, legacy_map = _merge_frames(frames, overlap_length=2.0)
    flagged, fmap = _merge_frames(frames, overlap_length=2.0, sat_frames=sat_frames)
    assert flagged.shape == legacy.shape
    assert fmap == legacy_map
    np.testing.assert_array_equal(flagged[:, :17], legacy[:, :17])
    near = _near(flagged, SAT_SPOT)
    assert near.any()
    assert saturated_mask(flagged[near, 17]).all(), "an unflagged fragment survived"
    np.testing.assert_array_equal(flagged[~near], legacy[~near])
    far = _near(flagged, FAR_SPOT)
    assert far.sum() == 1 and flagged[far, 17][0] != SATURATED_RETURN_CODE


def test_include_mode_gives_one_flagged_spot(peaks):
    """IncludeSaturatedSpots 1: the saturated core and its edge fragments become
    ONE flagged merged spot at the reflection; the unrelated spot is unchanged."""
    frames, sat_frames = peaks
    legacy, _ = _merge_frames(frames, overlap_length=2.0)
    inc, imap = _merge_frames(frames, overlap_length=2.0, sat_frames=sat_frames,
                              include_saturated=True)
    near = _near(inc, SAT_SPOT)
    assert near.sum() == 1, f"{near.sum()} merged spots at the saturated reflection"
    row = inc[near][0]
    assert row[17] == SATURATED_RETURN_CODE
    assert np.hypot(row[3] - SAT_SPOT[0], row[4] - SAT_SPOT[1]) < 0.5
    # it carries the fragments' intensity and more (the clipped core is a lower bound)
    assert row[1] > legacy[_near(legacy, SAT_SPOT), 1].sum()
    far_l = legacy[_near(legacy, FAR_SPOT)]
    far_i = inc[_near(inc, FAR_SPOT)]
    np.testing.assert_array_equal(far_i[:, 1:], far_l[:, 1:])
    # merge map: SpotIDs 1..N, and every constituent frame of the reflection listed once
    assert sorted({sid for sid, _, _ in imap}) == list(range(1, inc.shape[0] + 1))
    np.testing.assert_array_equal(inc[:, 0], np.arange(1, inc.shape[0] + 1))


def test_saturated_only_merge_is_one_reflection(peaks):
    _, sat_frames = peaks
    s = merge_saturated_only(sat_frames, overlap_length=2.0)
    assert s.shape[0] == 1 and s[0, 17] == SATURATED_RETURN_CODE


def test_no_saturation_is_bit_identical_to_legacy():
    """No saturated rows (empty sibling, or none at all): output is the legacy
    merge exactly, in either mode."""
    frames, sat_frames = _peakfit(sat_total=1.0e5)
    assert sum(s.shape[0] for s in sat_frames) == 0
    legacy, lmap = _merge_frames(frames, overlap_length=2.0)
    for inc in (False, True):
        out, omap = _merge_frames(frames, overlap_length=2.0, sat_frames=sat_frames,
                                  include_saturated=inc)
        np.testing.assert_array_equal(out, legacy)
        assert omap == lmap
