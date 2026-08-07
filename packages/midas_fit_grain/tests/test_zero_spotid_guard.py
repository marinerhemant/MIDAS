"""A SpotID of 0 must never reach ProcessKey.bin.

SpotIDs are 1-based rows of ExtraInfo.bin. C ProcessGrains computes
``rowSpotID = SpotID - 1`` and dereferences ``InputMatrix[rowSpotID][0]``
(ProcessGrains.c:966), so a 0 becomes ``InputMatrix[-1]`` and the process dies
with SIGSEGV and no diagnostic.

Measured on the datasetA Ni layer: refining ``IndexerOMP``'s seeds with
midas_fit_grain produced 329 zero entries across 204 of 57021 seeds, all in
the trailing slots of their grain's match list. C ProcessGrains segfaulted
deterministically at the same point on every run, while the same refiner's
output from the pipeline's own seeds (which happened to contain no zeros) ran
to completion -- so the crash looked like a format incompatibility and was not.

The refinement input itself is clean (SpotID 1..624820, no zeros, no
duplicates), so the zero is introduced inside the refiner: an unfilled
observation slot counted in ``n_spots`` and then matched.

Two guards, because either alone leaves a hole:
  * ``calc_angle_errors`` never matches a spot without a SpotID, so n_matched,
    completeness, meanRadius, FitBest.bin and ProcessKey.bin stay consistent;
  * ``write_process_key_row`` refuses to write one, so any future path that
    reintroduces it fails loudly here instead of in a consumer's SIGSEGV.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_fit_grain import c_port
from midas_fit_grain.io_binary import write_process_key_row


# ------------------------------------------------------- the writer guard ---

def test_writer_refuses_a_zero_spot_id(tmp_path):
    p = tmp_path / "ProcessKey.bin"
    with pytest.raises(ValueError, match="non-positive SpotID"):
        write_process_key_row(p, 0, np.array([12, 0, 34], dtype=np.int32))


def test_writer_refuses_a_negative_spot_id(tmp_path):
    p = tmp_path / "ProcessKey.bin"
    with pytest.raises(ValueError, match="non-positive SpotID"):
        write_process_key_row(p, 3, np.array([-1], dtype=np.int32))


def test_writer_reports_where(tmp_path):
    p = tmp_path / "ProcessKey.bin"
    with pytest.raises(ValueError) as ei:
        write_process_key_row(p, 7, np.array([5, 6, 0, 0], dtype=np.int32))
    msg = str(ei.value)
    assert "row 7" in msg
    assert "[2, 3]" in msg, "must name the offending slots"


def test_writer_accepts_a_clean_row(tmp_path):
    p = tmp_path / "ProcessKey.bin"
    write_process_key_row(p, 0, np.array([1, 2, 3], dtype=np.int32))
    got = np.fromfile(p, dtype=np.int32)[:3]
    assert got.tolist() == [1, 2, 3]


def test_writer_accepts_an_empty_row(tmp_path):
    """A grain with no matched spots is legitimate, not an error."""
    p = tmp_path / "ProcessKey.bin"
    write_process_key_row(p, 0, np.array([], dtype=np.int32))


# ------------------------------------------------------ the matcher guard ---

def _one_spot(spot_id, y=1000.0, z=0.0, ome=0.0, ring=1):
    """One observed-spot row in the (S, 10) spotsYZO layout."""
    r = np.zeros(10, dtype=np.float64)
    r[0], r[1], r[2] = y, z, ome        # YLab, ZLab, Omega
    r[3] = spot_id                      # SpotID
    r[4] = ome                          # OmegaIni
    r[5], r[6] = y, z                   # YOrig, ZOrig
    r[7] = ring                         # RingNumber
    return r


def _call(spots_yzo):
    return c_port.calc_angle_errors(
        pos=np.zeros(3),
        orient_mat=np.eye(3),
        lat_c=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]),
        spots_yzo=spots_yzo,
        hkls_int=np.array([[1, 1, 1], [2, 0, 0]], dtype=np.int64),
        ring_nr_per_hkl=np.array([1, 2], dtype=np.int64),
        lsd=1_000_000.0,
        wavelength=0.2066,
        omega_ranges=np.array([[-180.0, 180.0]]),
        box_sizes=np.array([[-1e6, 1e6, -1e6, 1e6]]),
        min_eta=6.0,
        wedge_deg=0.0,
        chi_deg=0.0,
        weight_mask=1.0,
        weight_fit_rmse=0.0,
    )


def test_a_spot_without_an_id_is_never_matched():
    """Whatever else happens, no zero may appear among the matched SpotIDs."""
    spots = np.stack([_one_spot(0.0), _one_spot(0.0), _one_spot(17.0)])
    spots_comp, _, _, n_matched = _call(spots)
    ids = spots_comp[:n_matched, 0]
    assert not (ids < 1).any(), f"unmatched-slot IDs leaked: {ids.tolist()}"


def test_the_guard_is_present_and_precedes_the_match():
    """The skip has to come before matching, or n_matched still counts it."""
    import inspect
    src = inspect.getsource(c_port.calc_angle_errors)
    body = src.split("for sp in range(S):")[-1]
    guard = body.index("spots_yzo[sp, 3] < 1.0")
    same_ring = body.index("same_ring")
    assert guard < same_ring, "the SpotID guard must precede the ring match"


def test_padding_only_input_matches_nothing():
    spots_comp, _, err, n_matched = _call(np.stack([_one_spot(0.0)] * 4))
    assert n_matched == 0
    assert spots_comp.shape[0] == 0
    assert err == (0.0, 0.0, 0.0)
