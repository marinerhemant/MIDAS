"""8-connected component labeling, region filtering, region-pixel extraction.

The C tool uses iterative DFS in row-major scan order. We use SciPy's
``ndimage.label`` with the 8-connectivity structure ``ones((3,3))``. The two
implementations agree on the *partition* of pixels into regions; they may
differ in the *labels* assigned, but downstream code is label-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import ndimage

from midas_peakfit.params import MAX_OVERLAPS_PER_IMAGE


# 8-connectivity structure for ndimage.label
_STRUCT_8 = np.ones((3, 3), dtype=np.int32)


@dataclass
class Region:
    """A connected component of bright pixels.

    All coordinates are in the *transposed* image frame (i.e. matching
    ``imgCorrBC``: row=Y, col=Z).
    """

    id: int
    pixel_rows: np.ndarray  # int32, shape (n,) — Y indices
    pixel_cols: np.ndarray  # int32, shape (n,) — Z indices
    intensities: np.ndarray  # float64, shape (n,)
    raw_sum: float = 0.0
    threshold: float = 0.0  # goodCoords value at first pixel (for fitting bound)
    n_pixels: int = field(init=False)

    def __post_init__(self):
        self.n_pixels = self.pixel_rows.size


def find_regions(
    img_corr: np.ndarray, good_coords: np.ndarray
) -> List[Region]:
    """Label 8-connected non-zero blobs in ``img_corr`` and return regions.

    Truncates at ``MAX_OVERLAPS_PER_IMAGE`` regions (matching the C cap;
    excess regions are discarded).
    """
    bool_image = img_corr > 0
    labels, n_regions = ndimage.label(bool_image, structure=_STRUCT_8)
    if n_regions == 0:
        return []

    if n_regions > MAX_OVERLAPS_PER_IMAGE:
        n_regions = MAX_OVERLAPS_PER_IMAGE

    # ndimage.find_objects returns slice tuples per label
    slices = ndimage.find_objects(labels, max_label=n_regions)

    regions: List[Region] = []
    for label_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        # Extract pixel coords belonging to this label within the bbox
        sub_labels = labels[sl]
        sub_img = img_corr[sl]
        sub_thresh = good_coords[sl]
        rows_local, cols_local = np.where(sub_labels == label_id)
        if rows_local.size == 0:
            continue
        rows = (rows_local + sl[0].start).astype(np.int32)
        cols = (cols_local + sl[1].start).astype(np.int32)
        ints = sub_img[rows_local, cols_local]
        thr = float(sub_thresh[rows_local[0], cols_local[0]])

        regions.append(
            Region(
                id=label_id,
                pixel_rows=rows,
                pixel_cols=cols,
                intensities=ints.astype(np.float64),
                raw_sum=float(ints.sum()),
                threshold=thr,
            )
        )
    return regions


def filter_regions_by_size(
    regions: List[Region], min_n_px: int, max_n_px: int
) -> List[Region]:
    """Drop regions where ``nPx <= minNrPx`` or ``nPx >= maxNrPx``.

    Note: both bounds are STRICT, matching C ``processImageFrame`` line 1465-1466:
        if (nrPixelsThisRegion <= minNrPx || nrPixelsThisRegion >= maxNrPx)
            continue;

    So the default ``minNrPx = 1`` already drops single-pixel regions, and
    raising it to 2 drops the 2-pixel ones. That step is expensive and, on the
    evidence, unjustified. Measured on one bt_1id_jun25b s1 scan (8069 peaks):

    * 2-pixel regions are **25.8 %** of all peaks -- a quarter of the spot list.
    * They are not noise: median ``IMax - BG`` is 701 counts over a background
      of 192. The *5*-pixel regions are the weakest (407), so region size is not
      a proxy for signal.
    * They are not positioned worse. The fit is formally underdetermined
      (``nPeaks * 8 + 1`` free parameters, so even a 5-px region interpolates
      exactly, median ``FitRMSE`` 3e-11), but the seeder box-bounds the centre
      to R +/- 1 px and Eta +/- dEta, so no fit can travel further than
      sqrt(2) px from a genuine local maximum. Measured travel from the seed is
      a median of 0.42 px for 2-px regions versus 0.50 px for >12-px regions,
      and 2-px regions rail against the bound *less* often (2.3 % vs 19.2 %).

    An underdetermined least-squares fit is not the same thing as an
    uninformative one when the parameter is boxed. Raise ``minNrPx`` because
    small regions are demonstrably spurious in *your* data, not on the
    identifiability argument.
    """
    return [r for r in regions if min_n_px < r.n_pixels < max_n_px]


__all__ = ["Region", "find_regions", "filter_regions_by_size"]
