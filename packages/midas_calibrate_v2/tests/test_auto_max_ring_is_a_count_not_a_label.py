"""auto_detect_max_ring returns a ROW COUNT; max_ring_number is a ring_nr LABEL.

`build_ring_table` merges radius-identical reflections, so for CeO2 the
`ring_nr` labels have gaps (the (333)/(511) and (442)/(600) pairs each keep one
label). The pipelines stored the count straight into `spec.max_ring_number`,
and `filter_ring_table` then compared it with the labels: a cap of 13 rows kept
12 rings, silently dropping the outermost distinct ring (the highest-strain one,
~19 uE on the Varex calibrant). Found by two independent refuters on
2026-10-07; the paper's 15-row headline had double-counted the same two pairs.
"""
from __future__ import annotations

import numpy as np

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.rings import build_ring_table

from midas_calibrate_v2.pipelines._common import filter_ring_table, ring_number_cap


def _ceo2_table(dedup=True):
    v1 = V1Params()
    v1.Lsd = 895_925.0
    v1.Wavelength = 0.196793
    v1.pxY = v1.pxZ = 150.0
    v1.SpaceGroup = 225
    v1.LatticeConstant = (5.4116, 5.4116, 5.4116, 90.0, 90.0, 90.0)
    v1.MaxRingRad = 1500.0                       # px, as the Varex runs use
    kw = {} if dedup else {"dedup_d_rel_tol": 0.0}
    return build_ring_table(v1, **kw)


def test_the_premise_labels_have_gaps_after_dedup():
    rt = _ceo2_table()
    labels = np.asarray(rt.ring_nr)
    assert np.any(np.diff(labels) > 1), (
        "expected the CeO2 table to carry merged rings (gapped ring_nr); "
        "if it no longer does, this test no longer exercises the bug")


def test_the_cap_keeps_exactly_the_requested_rows():
    rt = _ceo2_table()
    for n_rows in range(1, len(rt.ring_nr)):
        cap = ring_number_cap(rt, n_rows)
        kept = filter_ring_table(rt, max_ring_number=cap)
        assert len(kept.ring_nr) == n_rows, (n_rows, cap, list(rt.ring_nr))


def test_the_old_passthrough_dropped_a_ring():
    """The regression itself: using the count as a label loses rows."""
    rt = _ceo2_table()
    lost = [n for n in range(1, len(rt.ring_nr))
            if len(filter_ring_table(rt, max_ring_number=n).ring_nr) < n]
    assert lost, "count-as-label no longer loses rows; premise gone"


def test_no_cap_when_everything_is_kept():
    rt = _ceo2_table()
    assert ring_number_cap(rt, 0) == 0
    assert ring_number_cap(rt, len(rt.ring_nr)) == 0
    assert ring_number_cap(rt, len(rt.ring_nr) + 5) == 0


def test_without_dedup_labels_are_contiguous_and_cap_is_identity():
    rt = _ceo2_table(dedup=False)
    assert list(rt.ring_nr) == list(range(1, len(rt.ring_nr) + 1))
    for n_rows in (1, 5, len(rt.ring_nr) - 1):
        assert ring_number_cap(rt, n_rows) == n_rows
