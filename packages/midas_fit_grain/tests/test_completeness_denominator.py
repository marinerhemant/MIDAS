"""Completeness is matched / EXPECTED, not matched / observed.

``driver.py`` computed ``n_matched / obs.n_spots``. ``obs.n_spots`` is the
number of spots the INDEXER assigned to the seed -- which is the numerator of
the real definition, not its denominator. Dividing a grain's observed spots by
its own observed spots is ~1 by construction, so the column carried no
information and the ``Completeness`` gate could never fire on a python-refined
run.

C reads it straight from the indexer record (FitUnified.c:1716):

    NrExpected = tmpArr[14]; NrObserved = tmpArr[15];
    completeness = NrObserved / NrExpected;

Measured on the datasetA Ni layer, 57021 seeds shared with the C refiners:

                             ==1.0    <0.9   median
      C refiners             38.0%   23.9%   0.9741
      python, before         94.8%    0.0%   1.0000
      python, after          36.7%   24.1%   0.9741   (p95 err 0.0086)

Seed row 3716 (SpotID 192842) is the clean single case: NrExpected 112,
NrObserved 56, and BOTH refiners match the same 56 spots. C reports
56/112 = 0.500000; python reported 56/56 = 1.000000. Same numerator, and the
whole gap is the denominator.

The scanning driver already had this right -- scan_driver.py:226 is
``n_matched / max(n_expected, 1)``. Only the FF driver diverged.
"""

from __future__ import annotations

import inspect

from midas_fit_grain import driver


def test_the_denominator_is_the_expected_count():
    src = inspect.getsource(driver.refine_block_from_disk)
    assert "float(n_matched) / float(n_exp)" in src, (
        "completeness must divide by the predicted-reflection count"
    )


def test_the_expected_count_is_read_from_the_indexer_record():
    """IndexBest col 13 = n_expected, col 14 = n_observed (15-col legacy)."""
    src = inspect.getsource(driver.refine_block_from_disk)
    assert "seed_n_expected.append(int(rec[13]))" in src
    assert "n_observed = int(rec[14])" in src, (
        "col 14 stays the observed count — the two must not be swapped"
    )


def test_the_consolidated_adapter_supplies_it_too():
    """The c-omp path adapts a 16-col record; its n_expected is at col 14."""
    src = inspect.getsource(driver)
    assert "index_best[v, 13] = rec[14]" in src, (
        "without this the c-omp backend falls back to matched/observed"
    )


def test_a_missing_expected_count_falls_back_and_is_reported():
    """Silently reverting to the old formula is how this hid for a session."""
    src = inspect.getsource(driver.refine_block_from_disk)
    assert "if n_exp > 0:" in src
    assert "n_no_expected += 1" in src
    assert "n_no_expected" in inspect.getsource(driver.refine_block_from_disk)


def test_the_seed_row_3716_arithmetic():
    """The measured case, as arithmetic: same numerator, different divisor."""
    n_matched, n_expected, n_observed = 56, 112, 56
    assert n_matched / n_expected == 0.5              # what C reports
    assert n_matched / n_observed == 1.0              # what python reported
    assert n_matched / n_expected != n_matched / n_observed


def test_scan_driver_still_uses_the_expected_count():
    """The FF fix must not drift away from the definition PF already used."""
    from midas_fit_grain import scan_driver
    src = inspect.getsource(scan_driver)
    assert "float(n_matched) / max(int(n_expected), 1)" in src
