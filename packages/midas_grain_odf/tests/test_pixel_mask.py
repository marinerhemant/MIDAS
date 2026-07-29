"""Sanity tests for the per-pixel Voronoi mask."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from midas_grain_odf.spot_extract import (
    SpotPatchSpec, compute_per_spot_pixel_mask,
)


def test_no_competitor_returns_all_ones():
    spec = SpotPatchSpec(
        n_spots=2, patch_F=5, patch_P=11,
        sigma_yz=1.0, sigma_f=0.6,
        anchor_y=torch.tensor([100.0, 200.0]),
        anchor_z=torch.tensor([300.0, 400.0]),
        anchor_f=torch.tensor([500.0, 600.0]),
    )
    mask = compute_per_spot_pixel_mask(spec)
    assert mask.shape == (2, 5, 11, 11)
    assert torch.allclose(mask, torch.ones_like(mask))


def test_two_close_spots_partition_voronoi():
    """Two spots 10 px apart in y; mask boundary should be midway."""
    spec = SpotPatchSpec(
        n_spots=2, patch_F=1, patch_P=21,
        sigma_yz=1.0, sigma_f=0.6,
        anchor_y=torch.tensor([100.0, 110.0]),       # 10 px apart in y
        anchor_z=torch.tensor([300.0, 300.0]),
        anchor_f=torch.tensor([500.0, 500.0]),
    )
    other_y = torch.tensor([100.0, 110.0])
    other_z = torch.tensor([300.0, 300.0])
    other_f = torch.tensor([500.0, 500.0])
    other_id = torch.tensor([0, 1], dtype=torch.long)

    mask = compute_per_spot_pixel_mask(
        spec, other_y, other_z, other_f, other_id,
    )
    # spot 0's patch is centered at (100, 300, 500). Pixel y_local
    # in (yi - 10) corresponds to y_global = 100 + (yi - 10).
    # spot 1 is at y_global=110. Midpoint = 105 = yi=15.
    # Pixels with yi <= 15 are closer to spot 0, > 15 closer to spot 1.
    spot0_mask = mask[0, 0, 10, :]  # F=0, P=10 (own anchor row)
    # Cells with yi=0..15 should be 1; yi=16..20 should be 0.
    # But we use the patch's local y axis. yi index runs 0..20; the
    # anchor sits at center yi=10. For spot 0 the global y of cell yi
    # is 100 + (yi - 10).
    # Distance to spot 0 anchor: |yi - 10|.
    # Distance to spot 1 anchor: |yi - 10 - 10| = |yi - 20|.
    # spot 0 wins when |yi-10| <= |yi-20|, i.e. yi <= 15.
    # NB: mask shape is (S, F, P, P) with axes (s, f, y, z); the row at
    # the anchor's z (z_local = 10) is what we read above.
    # Wait — anchor is at (P-1)/2 = 10 in BOTH y and z. The y axis is
    # mask[s, 0, y_local, z_local]. We want y axis: mask[0, 0, :, 10].
    spot0_y_row = mask[0, 0, :, 10]
    expected = torch.tensor([1.0] * 16 + [0.0] * 5)
    assert torch.allclose(spot0_y_row, expected), \
        f"spot 0 y-row: {spot0_y_row.tolist()} vs expected {expected.tolist()}"


def test_self_excluded_from_competitor_set():
    """A spot listed both in spec and in `other_*` with id=self must
    not mask itself out (own_d2 always 0 when sub_id == s)."""
    spec = SpotPatchSpec(
        n_spots=1, patch_F=1, patch_P=5,
        sigma_yz=1.0, sigma_f=0.6,
        anchor_y=torch.tensor([0.0]),
        anchor_z=torch.tensor([0.0]),
        anchor_f=torch.tensor([0.0]),
    )
    mask = compute_per_spot_pixel_mask(
        spec,
        other_anchor_y=torch.tensor([0.0]),
        other_anchor_z=torch.tensor([0.0]),
        other_anchor_f=torch.tensor([0.0]),
        other_spot_id=torch.tensor([0], dtype=torch.long),
    )
    assert torch.allclose(mask, torch.ones_like(mask))


if __name__ == "__main__":
    test_no_competitor_returns_all_ones()
    test_two_close_spots_partition_voronoi()
    test_self_excluded_from_competitor_set()
    print("OK")
