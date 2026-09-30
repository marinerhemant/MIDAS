"""Synthetic pixel-overlap merge tests.

These run without external goldens and demonstrate the algorithmic
behaviour: pixel-overlap matches by shared pixel-list count (asymmetric
forward/reverse construction), while centroid-distance matches by
Euclidean (Y, Z) distance with a strict mutual-best gate. The two can
disagree on dense or near-coincident peak data.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_transforms.io import zarr_io
from midas_transforms.merge.core import (
    N_PEAK_COLS, _merge_frames, _pixel_overlap_match,
)


def _peak(spot_id, ii, omega, ycen, zcen, radius, eta):
    """Build a single 29-col peak row matching the AllPeaks_PS.bin schema."""
    p = np.zeros(N_PEAK_COLS, dtype=np.float64)
    p[0] = spot_id     # SpotID
    p[1] = ii          # IntegratedIntensity
    p[2] = omega       # Omega
    p[3] = ycen        # YCen
    p[4] = zcen        # ZCen
    p[5] = ii          # IMax (use II for simplicity)
    p[6] = radius      # Radius
    p[7] = eta         # Eta
    p[8] = 1.0         # SigmaR
    p[9] = 0.1         # SigmaEta
    p[10] = 10         # NrPx
    p[11] = 10         # NrPxTot
    p[26] = ii         # RawSumIntensity
    return p


def test_pixel_overlap_match_basic_label_map():
    """Two cur peaks, two new peaks; pixel-list overlaps determine matches."""
    cur_pixels = [
        np.array([[5, 5], [5, 6], [6, 5]], dtype=np.int16),     # cur 0
        np.array([[20, 20], [20, 21], [21, 20]], dtype=np.int16),  # cur 1
    ]
    new_pixels = [
        np.array([[5, 5], [5, 6], [7, 7]], dtype=np.int16),     # 2 px overlap with cur 0
        np.array([[20, 21], [21, 20]], dtype=np.int16),          # 2 px overlap with cur 1
    ]
    best, has = _pixel_overlap_match(cur_pixels, new_pixels, nr_pixels=64)
    assert has[0] and has[1]
    assert best[0] == 0
    assert best[1] == 1


def test_pixel_overlap_match_asymmetric_winner():
    """Two new peaks both prefer the same cur; cur picks the one with
    the highest overlap (matches C reverse-pass logic)."""
    cur_pixels = [
        np.array([[10, 10], [10, 11], [11, 10], [11, 11]], dtype=np.int16),
    ]
    new_pixels = [
        # 2 pixels overlap
        np.array([[10, 10], [10, 11]], dtype=np.int16),
        # 4 pixels overlap (all of cur's)
        np.array([[10, 10], [10, 11], [11, 10], [11, 11]], dtype=np.int16),
    ]
    best, has = _pixel_overlap_match(cur_pixels, new_pixels, nr_pixels=64)
    assert has[0]
    assert best[0] == 1   # the new peak with 4-pixel overlap wins


def test_pixel_overlap_match_no_overlap():
    """Disjoint pixel sets → no matches."""
    cur_pixels = [np.array([[5, 5]], dtype=np.int16)]
    new_pixels = [np.array([[100, 100]], dtype=np.int16)]
    best, has = _pixel_overlap_match(cur_pixels, new_pixels, nr_pixels=256)
    assert not has[0]
    assert best[0] == -1


def test_pixel_overlap_match_out_of_bounds_pixels_dropped():
    """Pixels outside [0, nr_pixels) are silently ignored."""
    cur_pixels = [
        np.array([[-1, 5], [5, 5], [5, -1], [5, 1000]], dtype=np.int16),
    ]
    new_pixels = [np.array([[5, 5]], dtype=np.int16)]
    best, has = _pixel_overlap_match(cur_pixels, new_pixels, nr_pixels=64)
    # The (5, 5) pixel is in bounds for both → should match.
    assert has[0]
    assert best[0] == 0


def test_pixel_overlap_diverges_from_centroid_when_peaks_are_close():
    """Two near-coincident peaks in adjacent frames: centroid distance
    can be tied, but pixel-overlap unambiguously picks the one with
    most shared pixels."""
    # Frame 0: one peak at (50, 50) with some pixel footprint.
    # Frame 1: two peaks, one at (50.5, 50.5) [close] and one at (51, 51)
    # [also close]. Both within margin=2 distance.
    f0 = np.stack([_peak(1, 1000.0, 0.0, 50.0, 50.0, 100.0, 30.0)], axis=0)
    f1 = np.stack([
        _peak(2, 800.0, 1.0, 50.5, 50.5, 100.5, 30.5),   # closer in centroid space
        _peak(3, 1200.0, 1.0, 51.0, 51.0, 101.0, 31.0),   # farther in centroid space
    ], axis=0)

    pix_f0 = [
        np.array([[50, 50], [50, 51], [51, 50], [51, 51]], dtype=np.int16),
    ]
    pix_f1 = [
        # Peak 2: only 1-pixel overlap with cur (at 50, 51)
        np.array([[50, 51], [50, 52], [51, 52]], dtype=np.int16),
        # Peak 3: 3-pixel overlap with cur (at 50,51 / 51,50 / 51,51)
        np.array([[50, 51], [51, 50], [51, 51]], dtype=np.int16),
    ]

    out_centroid, _ = _merge_frames(
        [f0, f1], overlap_length=2.0,
    )
    out_pixel, mm_pixel = _merge_frames(
        [f0, f1], overlap_length=2.0,
        pixel_frames=[pix_f0, pix_f1], nr_pixels=128,
    )

    # Centroid mode picks peak 2 (closer in (Y, Z) space).
    # Pixel-overlap picks peak 3 (more shared pixels).
    # Both produce 2 final clusters (one merged + one orphan), but the
    # cluster contents differ.
    assert out_centroid.shape[0] == 2
    assert out_pixel.shape[0] == 2

    # Pixel-overlap mode: peak 1 + peak 3 should be one cluster (merged).
    px_constituents = {}
    for sid, fn, pid in mm_pixel:
        px_constituents.setdefault(sid, []).append((fn, pid))
    multi = [c for c in px_constituents.values() if len(c) > 1]
    assert len(multi) == 1
    # The merged pair should include peak_id=3 (the high-overlap one).
    merged_peak_ids = {pid for (_, pid) in multi[0]}
    assert 3 in merged_peak_ids


def test_allpeaks_px_reader_round_trip(tmp_path):
    """Build an AllPeaks_PX.bin in the C format and verify our reader
    reconstructs it identically."""
    out = tmp_path / "AllPeaks_PX.bin"

    n_frames = 3
    nr_pixels = 256
    frames_pix = [
        [
            np.array([[10, 20], [11, 21], [12, 22]], dtype=np.int16),  # 3 px
            np.array([[100, 100]], dtype=np.int16),                    # 1 px
        ],
        [
            np.empty((0, 2), dtype=np.int16),                          # 0 px (peak with no pixels)
        ],
        [
            np.array([[5, 5], [6, 6]], dtype=np.int16),
            np.array([[50, 50], [51, 51], [52, 52], [53, 53]], dtype=np.int16),
            np.array([[200, 200]], dtype=np.int16),
        ],
    ]

    # Write the file in the C format.
    n_peaks = np.array([len(f) for f in frames_pix], dtype=np.int32)
    header_size = 8 + n_frames * 4 + n_frames * 8
    offsets = np.zeros(n_frames, dtype=np.int64)
    pos = header_size
    for f in range(n_frames):
        offsets[f] = pos
        for pix in frames_pix[f]:
            pos += 4 + pix.shape[0] * 4    # int32 nPx + (int16, int16) * nPx

    with open(out, "wb") as fp:
        fp.write(np.int32(n_frames).tobytes())
        fp.write(np.int32(nr_pixels).tobytes())
        fp.write(n_peaks.tobytes())
        fp.write(offsets.tobytes())
        for f in range(n_frames):
            for pix in frames_pix[f]:
                fp.write(np.int32(pix.shape[0]).tobytes())
                fp.write(pix.astype(np.int16).tobytes())

    nr_read, frames_read = zarr_io.read_allpeaks_px_frames(out)
    assert nr_read == nr_pixels
    assert len(frames_read) == n_frames
    for f in range(n_frames):
        assert len(frames_read[f]) == len(frames_pix[f])
        for k in range(len(frames_pix[f])):
            np.testing.assert_array_equal(frames_read[f][k], frames_pix[f][k])


def _mode_fixture():
    """Frame 0: one peak. Frame 1: peak 2 is centroid-nearest (0.7 px),
    peak 3 shares more pixels. Centroid pairs 1-2; overlap-first pairs 1-3."""
    f0 = np.stack([_peak(1, 1000.0, 0.0, 50.0, 50.0, 100.0, 30.0)], axis=0)
    f1 = np.stack([
        _peak(2, 800.0, 1.0, 50.5, 50.5, 100.5, 30.5),
        _peak(3, 1200.0, 1.0, 51.0, 51.0, 101.0, 31.0),
    ], axis=0)
    pix = [
        [np.array([[50, 50], [50, 51], [51, 50], [51, 51]], dtype=np.int16)],
        [np.array([[50, 51], [50, 52], [51, 52]], dtype=np.int16),
         np.array([[50, 51], [51, 50], [51, 51]], dtype=np.int16)],
    ]
    return [f0, f1], pix


def _merged_ids(res):
    g = {}
    for sid, fn, pid in res.merge_map:
        g.setdefault(sid, set()).add(pid)
    return [s for s in g.values() if len(s) > 1]


@pytest.mark.parametrize("mode, expect", [(0, {1, 2}), (1, {1, 2}), (False, {1, 2}),
                                          (True, {1, 2}), (2, {1, 3})])
def test_use_pixel_overlap_modes(tmp_path, mode, expect):
    """Issue #13: UsePixelOverlap 1 is the C output (centroid matching);
    pixel-overlap-first matching is opt-in as UsePixelOverlap 2."""
    from midas_transforms.merge import merge_overlapping_peaks

    frames, pix = _mode_fixture()
    res = merge_overlapping_peaks(
        frames=frames, pixel_frames=pix, nr_pixels=128, overlap_length=2.0,
        use_pixel_overlap=mode, result_folder=tmp_path, write=False,
        device="cpu", dtype="float64",
    )
    assert _merged_ids(res) == [expect]


def test_use_pixel_overlap_1_needs_no_px_file(tmp_path):
    """Mode 1 matches by centroid, so a missing AllPeaks_PX.bin is not an error;
    mode 2 still requires it."""
    from midas_transforms.merge import merge_overlapping_peaks

    frames, _ = _mode_fixture()
    kw = dict(frames=frames, overlap_length=2.0, result_folder=tmp_path,
              write=False, device="cpu", dtype="float64")
    merge_overlapping_peaks(use_pixel_overlap=1, **kw)
    with pytest.raises(FileNotFoundError):
        merge_overlapping_peaks(use_pixel_overlap=2, **kw)


def test_use_pixel_overlap_rejects_unknown_mode(tmp_path):
    from midas_transforms.merge import merge_overlapping_peaks

    frames, _ = _mode_fixture()
    with pytest.raises(ValueError, match="UsePixelOverlap"):
        merge_overlapping_peaks(frames=frames, overlap_length=2.0,
                                use_pixel_overlap=3, result_folder=tmp_path,
                                write=False, device="cpu", dtype="float64")
