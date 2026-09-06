"""Front end: frames -> mask, background, 3-D spot list.

Built on synthetic data with known ground truth, so each test can fail.
The properties pinned here are the ones whose violation was expensive in the
analysis this module came from:

- a reflection spanning several omega frames is ONE object, not one per frame
- the polar background keeps single-crystal signal and removes powder
- every selecting step reports what it discarded
- a zero-width blob does not emit a huge finite aspect ratio
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from midas_defect.ingest import (
    build_mask, polar_median_background, subtract_background,
    count_signed_blobs, choose_sectors, find_blobs_3d,
    detect_powder_rings, flag_powder, SPOT_COLUMNS,
)

RNG = np.random.default_rng(20260901)
NR, NC, NF = 128, 140, 9
CY, CX = 64.0, 70.0


def _geom():
    yy, xx = np.mgrid[0:NR, 0:NC]
    r = np.hypot(yy - CY, xx - CX)
    tth = r * 0.02                                   # deg, linear is fine here
    azi = np.degrees(np.arctan2(yy - CY, xx - CX))
    return tth, azi, r


def _blank(bg=300.0):
    return RNG.normal(bg, 6.0, size=(NF, NR, NC))


def _add_spot(frames, frame_c, row, col, amp, *, span=1.2, sigma=1.6):
    yy, xx = np.mgrid[0:NR, 0:NC]
    g = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma ** 2))
    for k in range(frames.shape[0]):
        frames[k] += amp * g * np.exp(-((k - frame_c) ** 2) / (2 * span ** 2))
    return frames


def _add_ring(frames, r_target, amp, sigma=3.0):
    _, _, r = _geom()
    ring = amp * np.exp(-((r - r_target) ** 2) / (2 * sigma ** 2))
    frames += ring[None, :, :]
    return frames


# ---------------------------------------------------------------------- mask

def test_mask_separates_gap_from_low_count_and_reports_both():
    frames = _blank()
    frames[:, 10:14, :] = -1.0                      # module gap
    frames[:, 60, 20:26] = 2.0                      # dead pixels
    m = build_mask(frames, low_count_threshold=20.0, grow=0)
    assert m.negative.sum() == 4 * NC
    assert m.low_count.sum() == 6
    assert m.counts["total"] == 4 * NC + 6
    assert m.counts["grown"] == 0
    assert "gap/defect" in str(m)


def test_mask_grow_only_adds():
    frames = _blank()
    frames[:, 10:14, :] = -1.0
    m0 = build_mask(frames, grow=0)
    m2 = build_mask(frames, grow=2)
    assert m2.mask.sum() > m0.mask.sum()
    assert np.all(m2.mask[m0.mask])                 # grow never unmasks
    assert m2.counts["grown"] > 0


def test_mask_uses_the_median_not_a_single_frame():
    """A pixel low in ONE frame is statistics; only persistent lows are bad."""
    frames = _blank()
    frames[3, 50, 50] = 0.0                         # one bad sample
    m = build_mask(frames, grow=0)
    assert not m.mask[50, 50]


def test_mask_rejects_bad_shapes():
    with pytest.raises(ValueError):
        build_mask(np.zeros((NR, NC)))


# ---------------------------------------------------------- background model

def test_polar_background_removes_a_ring_and_keeps_a_spot():
    tth, azi, r = _geom()
    frames = _blank(bg=400.0)
    frames = _add_ring(frames, 30.0, 900.0, sigma=3.0)
    frames = _add_spot(frames, 4.0, 95.0, 100.0, 4000.0)
    mask = np.zeros((NR, NC), bool)

    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.02)
    on_ring = np.abs(r - 30.0) < 2.0
    # the powder ring is gone: >90 % of a 900-count ring removed
    assert abs(np.median(sub[4][on_ring])) < 0.1 * 900.0
    # the crystal spot survives, big
    assert sub[4, 95, 100] > 1500.0


def test_a_ring_narrower_than_the_smoothing_window_SURVIVES():
    """Pin the documented limit: it is a real failure mode, not a bug.

    smooth_bins x tth_bin sets the narrowest ring the model can follow. A
    sharper ring is smoothed out of the background and carried through into
    the spot list. Anyone changing the defaults must see this fail.
    """
    tth, azi, r = _geom()
    frames = _blank(bg=400.0)
    frames = _add_ring(frames, 30.0, 900.0, sigma=1.0)     # very sharp
    mask = np.zeros((NR, NC), bool)
    on_ring = np.abs(r - 30.0) < 1.5

    wide = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    narrow = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.005)
    assert np.median(wide[4][on_ring]) > 200.0      # survives the wide window
    assert np.median(narrow[4][on_ring]) < np.median(wide[4][on_ring])


def test_one_sector_assumes_uniformity_and_azimuthal_structure_defeats_it():
    """n_sectors=1 is an assumption, not a default. Show it failing."""
    tth, azi, r = _geom()
    frames = _blank(bg=400.0)
    frames = _add_ring(frames, 30.0, 900.0)
    # azimuth-dependent absorption, as a DAC through anvils and gasket
    frames = frames * (1.0 + 0.45 * np.cos(np.radians(azi)))[None, :, :]
    mask = np.zeros((NR, NC), bool)

    s1 = subtract_background(frames, tth, azi, mask, n_sectors=1, tth_bin=0.05)
    s8 = subtract_background(frames, tth, azi, mask, n_sectors=16, tth_bin=0.05)
    on_ring = np.abs(r - 30.0) < 2.0
    assert np.std(s8[4][on_ring]) < np.std(s1[4][on_ring])


def test_count_signed_blobs_sees_a_planted_negative():
    stack = RNG.normal(0.0, 5.0, size=(NF, NR, NC))
    stack[3:6, 40:46, 40:46] += 900.0
    pos, neg = count_signed_blobs(stack, np.zeros((NR, NC), bool),
                                  threshold=200.0, min_vol=10)
    assert pos >= 1 and neg == 0
    stack[3:6, 80:86, 80:86] -= 900.0
    pos, neg = count_signed_blobs(stack, np.zeros((NR, NC), bool),
                                  threshold=200.0, min_vol=10)
    assert neg >= 1


def test_choose_sectors_returns_an_auditable_table():
    tth, azi, _ = _geom()
    frames = _blank(bg=400.0)
    frames = _add_ring(frames, 30.0, 900.0)
    frames = _add_spot(frames, 4.0, 95.0, 100.0, 4000.0)
    mask = np.zeros((NR, NC), bool)
    ch = choose_sectors(frames, tth, azi, mask, candidates=(1, 8),
                        threshold=200.0, tth_bin=0.05)
    assert set(ch.table["n_sectors"]) == {1, 8}
    assert {"positive", "negative", "neg_over_pos"} <= set(ch.table.columns)
    assert ch.n_sectors in (1, 8)
    assert ch.stack.shape == frames.shape


# ------------------------------------------------------------- 3-D spot find

def test_a_reflection_spanning_frames_is_one_object_not_one_per_frame():
    """The reason this module is 3-D. Per-frame labelling would give ~5."""
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_spot(frames, 4.0, 64.0, 70.0, 6000.0, span=1.5)
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)

    df, counts = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                               split_ratio=3.0, gap_bridge=0, return_counts=True)
    assert len(df) == 1, f"expected one object, got {len(df)}"
    assert df.iloc[0]["n_frames"] >= 3          # it really does span frames
    assert df.iloc[0]["frame"] == pytest.approx(4.0, abs=0.6)
    assert df.iloc[0]["row"] == pytest.approx(64.0, abs=0.8)
    assert df.iloc[0]["col"] == pytest.approx(70.0, abs=0.8)
    assert list(df.columns) == list(SPOT_COLUMNS)


def test_two_separated_peaks_split_and_the_count_is_reported():
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_spot(frames, 4.0, 60.0, 60.0, 6000.0, sigma=1.4)
    frames = _add_spot(frames, 4.0, 60.0, 68.0, 5000.0, sigma=1.4)
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df, counts = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                               split_ratio=1.5, gap_bridge=0, return_counts=True)
    assert len(df) == 2
    assert counts["sub_peaks"] == 2
    assert set(counts) >= {"labelled", "rejected_small", "kept", "blobs_split"}


def test_counts_account_for_every_rejected_blob():
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_spot(frames, 4.0, 64.0, 70.0, 6000.0)
    frames[4, 20, 20] += 5000.0                     # a single hot voxel
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df, c = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                          gap_bridge=0, return_counts=True)
    assert c["labelled"] == c["kept"] + c["rejected_small"]
    assert c["rejected_small"] >= 1                  # the hot voxel was dropped


def test_gap_bridge_reconnects_a_split_streak_and_invents_nothing():
    tth, azi, _ = _geom()
    frames = _blank()
    for col in range(55, 86):
        frames = _add_spot(frames, 4.0, 64.0, float(col), 2600.0, sigma=1.0)
    mask = np.zeros((NR, NC), bool)
    mask[:, 69:72] = True                            # a module gap across it
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.02)

    # count 3-D COMPONENTS, with splitting off, so this measures connectivity
    _, raw = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                           split_ratio=1e9, gap_bridge=0, return_counts=True)
    _, bridged = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                               split_ratio=1e9, gap_bridge=21, return_counts=True)
    assert raw["kept"] == 2                          # the gap cut it in two
    assert bridged["kept"] == 1                      # and bridging rejoined it

    # and on a blank detector it creates nothing
    blank = subtract_background(_blank(), tth, azi, mask, n_sectors=8, tth_bin=0.02)
    assert len(find_blobs_3d(blank, mask, threshold=200.0, min_vol=10,
                             gap_bridge=21)) == 0


def test_zero_width_blob_reports_infinite_aspect_not_a_huge_number():
    """A finite 3e5 reads like a measurement and survives a filter. inf does not."""
    stack = np.zeros((3, NR, NC), np.float32)
    stack[1, 40, 30:40] = 5000.0                     # one row: width is exactly 0
    df = find_blobs_3d(stack, np.zeros((NR, NC), bool), threshold=200.0,
                       min_vol=5, split_ratio=0.0, gap_bridge=0)
    assert len(df) == 1
    assert df.iloc[0]["width_px"] == pytest.approx(0.0, abs=1e-12)
    assert np.isinf(df.iloc[0]["aspect"])


def test_find_blobs_rejects_mismatched_mask():
    with pytest.raises(ValueError):
        find_blobs_3d(np.zeros((3, 10, 10)), np.zeros((11, 10), bool))


# --------------------------------------------------------- powder separation

def test_rings_are_found_from_the_data_with_no_table():
    tth, azi, r = _geom()
    frames = _blank(bg=200.0)
    frames = _add_ring(frames, 30.0, 1200.0, sigma=2.0)
    frames = _add_ring(frames, 48.0, 900.0, sigma=2.0)
    img = frames.sum(0)
    rings = detect_powder_rings(img, tth, np.zeros((NR, NC), bool),
                                tth_bin=0.02, tth_range=(0.2, 1.6))
    found = np.sort(rings.centre_deg)
    for expect in (30.0 * 0.02, 48.0 * 0.02):
        assert np.min(np.abs(found - expect)) < 0.05, f"missed ring at {expect}"


def _add_spots(frames, r_target, amp, azimuths_deg, sigma=3.0):
    """A single-crystal multiplicity family: one |G|, a handful of azimuths."""
    yy, xx = np.mgrid[0:NR, 0:NC]
    for a in azimuths_deg:
        sy = CY + r_target * np.sin(np.radians(a))
        sx = CX + r_target * np.cos(np.radians(a))
        frames += (amp * np.exp(-(((yy - sy) ** 2 + (xx - sx) ** 2)
                                  / (2 * sigma ** 2))))[None, :, :]
    return frames


def test_a_spot_family_at_one_radius_is_not_a_ring():
    r"""Regression: bright reflections sharing one \|G\| forged a powder ring.

    A crystal puts (0,1,5), (1,0,5), (0,-1,5), (-1,0,5) at exactly one radius.
    On a max-projection they lift the azimuthal-median profile enough to be
    found as a peak, and the spurious ring then let `flag_powder` discard the
    crystal's own brightest reflections. Continuity is what separates them: the
    real ring occupies nearly every sector, the family occupies four.
    """
    tth, azi, _ = _geom()
    mask = np.zeros((NR, NC), bool)
    frames = _add_spots(_blank(bg=200.0), 30.0, 4000.0, [0, 90, 180, 270])
    img = frames.sum(0)

    permissive = detect_powder_rings(img, tth, mask,
                                     tth_bin=0.02, tth_range=(0.2, 1.6))
    assert np.min(np.abs(permissive.centre_deg - 0.6)) < 0.05, \
        "precondition: without the continuity test the family IS taken for a ring"
    assert np.all(np.isnan(permissive.occupancy))

    strict = detect_powder_rings(img, tth, mask, azimuth_deg=azi,
                                 tth_bin=0.02, tth_range=(0.2, 1.6))
    assert len(strict) == 0 or np.min(np.abs(strict.centre_deg - 0.6)) > 0.05, \
        "four spots at one radius must not be a ring"


def test_continuity_keeps_a_real_ring():
    tth, azi, _ = _geom()
    mask = np.zeros((NR, NC), bool)
    img = _add_ring(_blank(bg=200.0), 30.0, 1200.0, sigma=2.0).sum(0)
    rings = detect_powder_rings(img, tth, mask, azimuth_deg=azi,
                                tth_bin=0.02, tth_range=(0.2, 1.6))
    assert np.min(np.abs(rings.centre_deg - 0.6)) < 0.05, \
        "a continuous ring must survive the continuity test"
    i = int(np.argmin(np.abs(rings.centre_deg - 0.6)))
    assert rings.occupancy[i] > 0.9


def test_continuity_excludes_gap_sectors_from_the_denominator():
    """Sectors lost to module gaps must not count as ring-absent."""
    tth, azi, _ = _geom()
    img = _add_ring(_blank(bg=200.0), 30.0, 1200.0, sigma=2.0).sum(0)
    mask = np.zeros((NR, NC), bool)
    mask[:, : NC // 3] = True                      # blank a third of the detector
    rings = detect_powder_rings(img, tth, mask, azimuth_deg=azi,
                                tth_bin=0.02, tth_range=(0.2, 1.6))
    assert np.min(np.abs(rings.centre_deg - 0.6)) < 0.05, \
        "a ring must survive a masked detector region"


def test_powder_flag_needs_BOTH_halves():
    """Either half alone mislabels; the conjunction is the discriminant."""
    rings = type("R", (), {"centre_deg": np.array([10.0]),
                           "width_deg": np.array([0.1])})()

    # 12 spots on the ring at spread azimuths -> powder
    n = 12
    tth = np.full(n, 10.0); eta = np.linspace(-170, 170, n); rad = np.full(n, 300.0)
    assert flag_powder(tth, eta, rad, rings).all()

    # same 12 azimuths but OFF the ring -> not powder (ring half fails)
    assert not flag_powder(np.full(n, 12.0), eta, rad, rings).any()

    # on the ring but only 3 of them -> not powder (companion half fails)
    assert not flag_powder(tth[:3], eta[:3], rad[:3], rings).any()


def test_powder_flag_spares_a_crystal_row_on_a_ring():
    """A good index puts several reflections at one |G|; that is not powder."""
    rings = type("R", (), {"centre_deg": np.array([10.0]),
                           "width_deg": np.array([0.1])})()
    tth = np.full(4, 10.0); eta = np.array([-20., -7., 6., 19.])
    rad = np.full(4, 300.0)
    assert not flag_powder(tth, eta, rad, rings, min_companions=8).any()
