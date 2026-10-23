"""Front end: frames -> mask, background, 3-D spot list.

Built on synthetic data with known ground truth, so each test can fail.
The properties pinned here are the ones whose violation was expensive in the
analysis this module came from:

- a reflection spanning several omega frames is ONE object, not one per frame
- the polar background keeps single-crystal signal and removes powder
- every selecting step reports what it discarded
- a zero-width blob does not emit a huge finite aspect ratio
- skewness/kurtosis tell a single peak from an overlapping pair, in-plane and
  in omega, and match closed-form values on a synthetic discrete uniform
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
from scipy import ndimage

from midas_defect.ingest import (
    build_mask, polar_median_background, subtract_background,
    count_signed_blobs, choose_sectors, find_blobs_3d,
    detect_powder_rings, flag_powder, SPOT_COLUMNS,
)
from midas_defect.ingest import _bridge_gaps, _grouped_median

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


def _add_aniso_spot(frames, frame_c, row, col, amp, sig_major, sig_minor,
                    angle_deg, *, span=1.5):
    """An elongated, tilted 2-D Gaussian spot -- a single, symmetric peak."""
    yy, xx = np.mgrid[0:NR, 0:NC]
    th = np.radians(angle_deg)
    dy, dx = yy - row, xx - col
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    g = np.exp(-(u ** 2 / (2 * sig_major ** 2) + v ** 2 / (2 * sig_minor ** 2)))
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


def test_return_labels_reconstructs_the_table_not_a_separate_computation():
    """Every label's own voxel count and mean position must match its row --
    this is a QC hook for the watershed split, so it must actually agree with
    what the table reports, not just look plausible.
    """
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_spot(frames, 4.0, 60.0, 60.0, 6000.0, sigma=1.4)
    frames = _add_spot(frames, 4.0, 60.0, 68.0, 5000.0, sigma=1.4)
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df, labels = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                               split_ratio=1.5, gap_bridge=0, return_labels=True)
    assert len(df) == 2
    assert labels.shape == sub.shape
    assert labels.dtype.kind == "i"
    for i, row in df.iterrows():
        voxels = labels == (i + 1)
        assert int(voxels.sum()) == int(row["volume_vox"])
        _, r, c = np.nonzero(voxels)
        assert abs(r.mean() - row["row"]) < 1.0
        assert abs(c.mean() - row["col"]) < 1.0
    assert int((labels > 0).sum()) == int(df["volume_vox"].sum())


def test_return_labels_combines_with_return_counts_and_defaults_are_unaffected():
    stack = np.zeros((3, NR, NC), np.float32)
    stack[1, 40, 30:40] = 5000.0
    mask = np.zeros((NR, NC), bool)
    df, counts, labels = find_blobs_3d(stack, mask, threshold=200.0, min_vol=5,
                                       split_ratio=0.0, gap_bridge=0,
                                       return_counts=True, return_labels=True)
    assert len(df) == 1
    assert counts["sub_peaks"] == 1
    assert int((labels == 1).sum()) == int(df.iloc[0]["volume_vox"])

    # requesting labels must not change the table found without them
    plain = find_blobs_3d(stack, mask, threshold=200.0, min_vol=5,
                          split_ratio=0.0, gap_bridge=0)
    assert plain.equals(df)


# --------------------------------------------------- shape: skew and kurtosis

def test_single_anisotropic_peak_has_near_zero_skew_and_kurtosis():
    """A single, symmetric (if elongated) peak is neither skewed nor bimodal.

    These are the reference values every "is this contaminated?" threshold
    below must sit clearly outside of. Truncation at ``threshold`` (only
    pixels above it enter the moments) keeps this from being exactly zero.
    """
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_aniso_spot(frames, 4.0, 64.0, 70.0, 20000.0,
                             sig_major=5.0, sig_minor=1.5, angle_deg=30.0)
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                       split_ratio=1e9, gap_bridge=0)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["aspect"] > 1.5, "the elongation itself should still be visible"
    for col in ("skew_length", "skew_width"):
        assert abs(r[col]) < 0.2, f"{col}={r[col]!r} should be near zero"
    for col in ("kurt_length", "kurt_width"):
        assert r[col] > -0.5, f"{col}={r[col]!r} should not look bimodal"


def test_equal_weight_overlap_is_platykurtic_not_skewed():
    """Two same-brightness overlapping domains: symmetric, so kurtosis (not
    skewness) is what flags it -- the signature a real, roughly-equal-weight
    multi-domain contamination should leave on an otherwise ordinary spot.
    """
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_spot(frames, 4.0, 60.0, 60.0, 6000.0, sigma=2.2)
    frames = _add_spot(frames, 4.0, 60.0, 68.0, 6000.0, sigma=2.2)
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                       split_ratio=1e9, gap_bridge=0)
    assert len(df) == 1, "the two lobes must merge into one connected blob"
    r = df.iloc[0]
    assert abs(r["skew_length"]) < 0.2
    assert r["kurt_length"] < -1.0, "a symmetric two-lobe blend is platykurtic"


def test_unequal_weight_overlap_is_skewed():
    """Two different-brightness overlapping domains -- the more realistic
    case, since two real domains rarely contribute equal intensity -- pulls
    skewness away from zero, distinguishing it from both the single-peak and
    the equal-weight cases above.
    """
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_spot(frames, 4.0, 60.0, 60.0, 6000.0, sigma=2.2)
    frames = _add_spot(frames, 4.0, 60.0, 68.0, 3000.0, sigma=2.2)
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                       split_ratio=1e9, gap_bridge=0)
    assert len(df) == 1
    assert abs(df.iloc[0]["skew_length"]) > 0.5


def test_uniform_bar_kurtosis_matches_the_closed_form_discrete_uniform():
    """A flat-top streak's kurtosis has a known analytic answer: this closes
    the gap that length_px/pos_angle_deg were never checked against a
    predicted value (only the degenerate width=0 case was, above).
    """
    n = 60
    stack = np.zeros((3, NR, NC), np.float32)
    stack[1, 40, 20:20 + n] = 5000.0
    mask = np.zeros((NR, NC), bool)
    df = find_blobs_3d(stack, mask, threshold=200.0, min_vol=5,
                       split_ratio=0.0, gap_bridge=0)
    assert len(df) == 1
    r = df.iloc[0]
    expected = -6.0 * (n ** 2 + 1) / (5.0 * (n ** 2 - 1))   # discrete uniform
    assert r["kurt_length"] == pytest.approx(expected, abs=0.02)
    assert np.isnan(r["kurt_width"]), "width is exactly 0, so kurtosis is undefined"


def test_bimodal_omega_profile_is_caught_even_when_in_plane_looks_normal():
    """Two domains coincident in (row, col) but offset in omega look like one
    ordinary spot in-plane -- the omega-axis moments are what catches this,
    which is exactly why omega was kept as its own axis rather than dropped.
    """
    tth, azi, _ = _geom()
    mask = np.zeros((NR, NC), bool)

    # equal weight: symmetric in omega -> kurtosis catches it, skew doesn't.
    frames = _blank()
    frames = _add_spot(frames, 2.0, 64.0, 70.0, 6000.0, span=1.2)
    frames = _add_spot(frames, 6.0, 64.0, 70.0, 6000.0, span=1.2)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                       split_ratio=1e9, gap_bridge=0)
    assert len(df) == 1
    r = df.iloc[0]
    assert abs(r["skew_omega"]) < 0.2
    assert r["kurt_omega"] < -0.8, "well below a single peak's ~-0.4 baseline"

    # unequal weight: asymmetric in omega -> skew picks it up instead.
    frames = _blank()
    frames = _add_spot(frames, 2.0, 64.0, 70.0, 6000.0, span=1.2)
    frames = _add_spot(frames, 6.0, 64.0, 70.0, 2500.0, span=1.2)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                       split_ratio=1e9, gap_bridge=0)
    assert len(df) == 1
    assert abs(df.iloc[0]["skew_omega"]) > 0.3


# ------------------------------------------------- full-blob vs segmented shape

def test_full_columns_equal_segmented_ones_when_a_blob_never_splits():
    """A blob that the watershed never divides has `sub_id`'s own voxels == `blob_id`'s own
    voxels -- so every `full_*` column must equal its unprefixed counterpart EXACTLY (same
    `_region_shape_stats` call, same input), not just approximately."""
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_spot(frames, 4.0, 64.0, 70.0, 6000.0, span=1.5)
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                       split_ratio=1e9, gap_bridge=0)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["full_n_sub_peaks"] == 1
    for col in ("volume_vox", "n_frames", "peak_counts", "length_px", "width_px",
               "pos_angle_deg", "aspect", "skew_length", "skew_width",
               "kurt_length", "kurt_width", "omega_width_frames", "skew_omega",
               "kurt_omega", "extent_frame", "extent_row", "extent_col"):
        assert r[f"full_{col}"] == pytest.approx(r[col], nan_ok=True), col


def test_full_columns_describe_the_whole_blob_when_it_splits():
    """Two overlapping (wide enough to touch -- unlike the separated-peaks test above, which
    gives two independent `blob_id`s) lobes merge into ONE connected-component blob but watershed
    -split into several `sub_id`s -- `full_*` must describe the WHOLE blob (every lobe together),
    not any one sub-peak alone: bigger volume/extent than any sub-peak's own, and the SAME
    full_* numbers on every row (they share one blob_id)."""
    tth, azi, _ = _geom()
    frames = _blank()
    frames = _add_spot(frames, 4.0, 60.0, 60.0, 6000.0, sigma=2.2)
    frames = _add_spot(frames, 4.0, 60.0, 68.0, 6000.0, sigma=2.2)
    mask = np.zeros((NR, NC), bool)
    sub = subtract_background(frames, tth, azi, mask, n_sectors=8, tth_bin=0.05)
    df = find_blobs_3d(sub, mask, threshold=200.0, min_vol=10,
                       split_ratio=1.5, gap_bridge=0)
    assert len(df) >= 2, "the blob must actually have split for this test to mean anything"
    assert (df["blob_id"] == df["blob_id"].iloc[0]).all(), "every sub-peak shares one blob_id"
    assert (df["full_n_sub_peaks"] == len(df)).all()
    # full_* is identical across every row of the same blob -- one shared computation
    for col in ("full_volume_vox", "full_extent_row", "full_extent_col", "full_length_px"):
        assert df[col].nunique() == 1, f"{col} must be the same for every row of one blob_id"
    # the whole blob spans (and outweighs) any one sub-peak alone
    assert df["full_volume_vox"].iloc[0] > df["volume_vox"].max()
    assert df["full_extent_col"].iloc[0] >= df["extent_col"].max()
    assert df["full_volume_vox"].iloc[0] == pytest.approx(df["volume_vox"].sum())


def test_extent_is_the_plain_bounding_box_not_the_covariance_width():
    """`extent_row`/`extent_col` are `max - min + 1` in raw array indices -- a different
    number from the intensity-weighted, rotated `width_px`, by construction on a diagonal bar
    (its axis-aligned bounding box is much larger than its own true, rotated width)."""
    stack = np.zeros((3, NR, NC), np.float32)
    for i in range(10):
        stack[1, 40 + i, 30 + i] = 5000.0        # a 10-voxel diagonal line, 1 voxel thick
    df = find_blobs_3d(stack, np.zeros((NR, NC), bool), threshold=200.0,
                       min_vol=5, split_ratio=0.0, gap_bridge=0)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["extent_row"] == 10 and r["extent_col"] == 10
    assert r["extent_frame"] == 1
    # the bar's true width along its own minor axis is ~0 (one voxel thick) -- nothing like
    # its 10-voxel axis-aligned bounding box.
    assert r["width_px"] < 1.0
    assert r["full_extent_row"] == 10 and r["full_extent_col"] == 10


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


# ---------------------------------------------------------------------------
# equivalence: the 2026-10-22 performance rewrites must not move a single bit
# ---------------------------------------------------------------------------
# Each reference function below is a FROZEN COPY of the pre-rewrite
# implementation (the version at this commit's parent), kept only as ground
# truth -- never call it expecting it to be fast. The public functions above
# must match it bit-for-bit over randomized trials, not just "look similar."


def _reference_bridge_gaps(binary: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    """Pre-rewrite ``_bridge_gaps``: unconditional box ``binary_closing``,
    no separable fast path."""
    if size <= 1:
        return binary
    st = np.ones((1, size, size), bool)
    closed = ndimage.binary_closing(binary, structure=st)
    return binary | (mask[None, :, :] & closed)


def test_bridge_gaps_separable_fast_path_matches_box_closing():
    """``_bridge_gaps``'s odd-size path (maximum/minimum_filter1d) must equal the box
    ``binary_closing`` it replaces, bit-for-bit, not just visually similar."""
    rng = np.random.default_rng(20261022)
    for size in (1, 3, 5, 7, 9, 11, 21, 25):
        for _ in range(8):
            nf = int(rng.integers(2, 6))
            ny = int(rng.integers(15, 50))
            nx = int(rng.integers(15, 50))
            binary = rng.random((nf, ny, nx)) < 0.05
            mask = rng.random((ny, nx)) < 0.1
            fast = _bridge_gaps(binary, mask, size)
            ref = _reference_bridge_gaps(binary, mask, size)
            assert np.array_equal(fast, ref), f"size={size} mismatch"


def test_bridge_gaps_even_size_is_unchanged():
    """Even sizes are deliberately NOT routed through the separable path (its filter-origin
    convention does not line up with binary_closing's for a box structuring element) -- confirm
    the fallback still matches the pre-rewrite box definition exactly."""
    rng = np.random.default_rng(20261023)
    for size in (2, 4, 6, 20):
        binary = rng.random((4, 30, 30)) < 0.05
        mask = rng.random((30, 30)) < 0.1
        fast = _bridge_gaps(binary, mask, size)
        ref = _reference_bridge_gaps(binary, mask, size)
        assert np.array_equal(fast, ref)


def _reference_polar_median_background(frame, tth_deg, azimuth_deg, mask, *,
                                       n_sectors=8, tth_bin=0.02,
                                       min_pixels_per_cell=8, smooth_bins=5):
    """Pre-rewrite ``polar_median_background``: recomputes its own (ti, ai, good, cell) bin
    geometry inline every call, rather than accepting it precomputed."""
    frame = np.asarray(frame, dtype=np.float64)
    lo, hi = float(np.nanmin(tth_deg)), float(np.nanmax(tth_deg))
    n_tth = int((hi - lo) / tth_bin) + 1
    ti = np.clip(((tth_deg - lo) / tth_bin).astype(np.int32), 0, n_tth - 1)
    ai = np.clip(((azimuth_deg + 180.0) / (360.0 / n_sectors)).astype(np.int32),
                 0, n_sectors - 1)
    good = ~np.asarray(mask, bool)
    cell = (ti * n_sectors + ai)[good]
    value = frame[good]
    n_cells = n_tth * n_sectors
    med, have = _grouped_median(cell, value, n_cells, min_pixels_per_cell)
    med = med.reshape(n_tth, n_sectors)
    have = have.reshape(n_tth, n_sectors)
    for j in range(n_sectors):
        nz = np.flatnonzero(have[:, j])
        if nz.size:
            med[:, j] = np.interp(np.arange(n_tth), nz, med[nz, j])
    if smooth_bins > 1:
        med = ndimage.uniform_filter1d(med, int(smooth_bins), axis=0)
    return med[ti, ai]


def _reference_subtract_background(frames, tth_deg, azimuth_deg, mask, *,
                                   n_sectors=8, tth_bin=0.02):
    """Pre-rewrite ``subtract_background``: calls the per-frame-recomputing reference
    background above fresh for every frame, instead of hoisting the frame-invariant bin
    geometry out of the loop once."""
    frames = np.asarray(frames)
    mask = np.asarray(mask, bool)
    out = np.empty(frames.shape, dtype=np.float32)
    for k in range(frames.shape[0]):
        a = frames[k].astype(np.float64, copy=True)
        a[mask] = np.median(a[~mask]) if (~mask).any() else 0.0
        s = a - _reference_polar_median_background(a, tth_deg, azimuth_deg, mask,
                                                   n_sectors=n_sectors, tth_bin=tth_bin)
        s[mask] = 0.0
        out[k] = s.astype(np.float32)
    return out


def test_polar_median_background_matches_pre_hoist_reference():
    tth, azi, _ = _geom()
    frame = _blank()[0]
    mask = build_mask(_blank()).mask
    for n_sectors in (1, 8, 24, 96):
        fast = polar_median_background(frame, tth, azi, mask, n_sectors=n_sectors)
        ref = _reference_polar_median_background(frame, tth, azi, mask, n_sectors=n_sectors)
        assert np.array_equal(fast, ref)


def test_subtract_background_hoisted_bins_match_per_frame_recompute():
    """The whole point of hoisting (ti, ai, good, cell) out of subtract_background's per-frame
    loop is that it changes nothing about the result -- that geometry never depended on the
    frame. Confirm it over real multi-frame data, not just the single-frame function above."""
    tth, azi, _ = _geom()
    frames = _add_spot(_blank(), 4, 60, 65, 900.0)
    mask = build_mask(frames).mask
    for n_sectors in (1, 8, 24, 96):
        fast = subtract_background(frames, tth, azi, mask, n_sectors=n_sectors)
        ref = _reference_subtract_background(frames, tth, azi, mask, n_sectors=n_sectors)
        assert np.array_equal(fast, ref)


def test_detect_powder_rings_grouped_median_matches_per_bin_loop_reference():
    """``detect_powder_rings``'s radial profile used a Python ``for i in range(n_bin):
    np.median(chunk)`` loop; it now reuses ``_grouped_median``. Confirm the profile -- and so
    everything downstream of it (peaks, widths, occupancy) -- is unchanged."""
    tth, azi, _ = _geom()
    mask = np.zeros((NR, NC), bool)
    img = _add_ring(_blank(bg=200.0), 30.0, 1200.0, sigma=2.0).sum(0)
    good = ~mask
    lo, hi = 0.2, 1.6
    tth_bin = 0.02
    sel = good & (tth >= lo) & (tth < hi)
    n_bin = int((hi - lo) / tth_bin) + 1
    idx = ((tth[sel] - lo) / tth_bin).astype(np.int32)
    val = img[sel]

    # reference: the original sort + searchsorted + per-bin np.median loop
    order = np.argsort(idx, kind="stable")
    idx_s, val_s = idx[order], val[order]
    edges = np.searchsorted(idx_s, np.arange(n_bin + 1))
    ref_prof = np.full(n_bin, np.nan)
    for i in range(n_bin):
        chunk = val_s[edges[i]:edges[i + 1]]
        if chunk.size >= 8:
            ref_prof[i] = np.median(chunk)

    med, have = _grouped_median(idx, val, n_bin, min_pixels_per_cell=8)
    fast_prof = np.where(have, med, np.nan)

    assert np.array_equal(fast_prof, ref_prof, equal_nan=True)

    # and the public function's own result, end to end, is unaffected (same data/params as
    # test_continuity_keeps_a_real_ring, which already pins this exact ring being found)
    rings = detect_powder_rings(img, tth, mask, azimuth_deg=azi,
                                tth_bin=tth_bin, tth_range=(lo, hi))
    assert np.min(np.abs(rings.centre_deg - 0.6)) < 0.05


# ---------------------------------------------------------------------------
# equivalence: `max_workers` (threads) must never change the result
# ---------------------------------------------------------------------------
# These are NOT performance tests -- they only need enough frames/candidates for a given
# max_workers to actually split the work across more than one thread, so the claim "disjoint
# output slices, so any max_workers gives the serial result" is checked rather than assumed.


def test_bridge_gaps_max_workers_is_bit_identical():
    rng = np.random.default_rng(20261024)
    nf, ny, nx = 17, 24, 30     # deliberately not a multiple of any worker count below
    binary = rng.random((nf, ny, nx)) < 0.05
    mask = rng.random((ny, nx)) < 0.1
    for size in (3, 21):
        ref = _bridge_gaps(binary, mask, size, max_workers=1)
        for max_workers in (2, 3, 5, 8, 100):   # 100 > nf exercises the min(max_workers, nf) clamp
            out = _bridge_gaps(binary, mask, size, max_workers=max_workers)
            assert np.array_equal(out, ref), f"size={size} max_workers={max_workers}"


def test_subtract_background_max_workers_is_bit_identical():
    tth, azi, _ = _geom()
    frames = _add_spot(_blank(), 4, 60, 65, 900.0)
    mask = build_mask(frames).mask
    for n_sectors in (1, 24):
        ref = subtract_background(frames, tth, azi, mask, n_sectors=n_sectors, max_workers=1)
        for max_workers in (2, 3, NF, 2 * NF):
            out = subtract_background(frames, tth, azi, mask, n_sectors=n_sectors,
                                      max_workers=max_workers)
            assert np.array_equal(out, ref), f"n_sectors={n_sectors} max_workers={max_workers}"


def test_choose_sectors_max_workers_is_bit_identical():
    tth, azi, _ = _geom()
    frames = _add_spot(_blank(), 4, 60, 65, 900.0)
    frames = _add_ring(frames, 20.0, 500.0, sigma=1.5)
    mask = build_mask(frames).mask
    ref = choose_sectors(frames, tth, azi, mask, max_workers=1)
    for max_workers in (2, 3, 5, 10):
        out = choose_sectors(frames, tth, azi, mask, max_workers=max_workers)
        assert out.n_sectors == ref.n_sectors
        assert out.table.equals(ref.table)
        assert np.array_equal(out.stack, ref.stack)


def test_find_blobs_3d_max_workers_is_bit_identical():
    tth, azi, _ = _geom()
    frames = _add_spot(_blank(), 4, 60, 65, 900.0)
    mask = build_mask(frames).mask
    sub = subtract_background(frames, tth, azi, mask)
    ref, ref_counts = find_blobs_3d(sub, mask, gap_bridge=5, return_counts=True, max_workers=1)
    for max_workers in (2, 4, 8):
        out, counts = find_blobs_3d(sub, mask, gap_bridge=5, return_counts=True,
                                    max_workers=max_workers)
        assert counts == ref_counts
        assert out.equals(ref)
