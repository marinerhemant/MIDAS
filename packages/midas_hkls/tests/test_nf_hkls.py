"""Parity tests for ``emit_nf_hkls_csv`` against the C ``GetHKLListNF``
reference output bundled with NF_HEDM/Example/sim.

We do not require byte-for-byte equality with the C output because the
C uses an unstable ``qsort`` for d-spacing sorting, so the *intra-ring*
row order is implementation-defined. Instead we check that:

  - the row count matches,
  - the sorted set of (h, k, l) tuples matches,
  - per-row d-spacing, RingNr, g-vector, theta, 2θ, and radius columns
    match within float tolerance once both sides are sorted by
    (RingNr, h, k, l).
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from midas_hkls import Lattice, SpaceGroup, emit_nf_hkls_csv


# Bundled C reference: NF_HEDM/Example/sim/hkls.csv was produced by
# `GetHKLListNF NF_HEDM/Example/sim/test_ps_au.txt`. The param file has two
# Lsd lines (one per detector distance); the C param parser overwrites
# ``cfg->Lsd`` on each call, so the LAST ``Lsd`` (= 10290.724494) is what
# the binary actually uses for the d_min cutoff and Radius column.
_REF_HKLS_CSV = Path(__file__).resolve().parents[3] / (
    "NF_HEDM/Example/sim/hkls.csv"
)


def _au_inputs():
    return {
        "space_group": SpaceGroup.from_number(225),
        "lattice": Lattice(4.08, 4.08, 4.08, 90.0, 90.0, 90.0),
        "wavelength_A": 0.172979,
        "lsd_um": 10290.724494,    # LAST Lsd in the param file, see note above
        "max_ring_rad_um": 2800.0,
    }


def _read_ref(path: Path):
    """Read the C reference, return ``(N, 11) ndarray``."""
    with open(path) as f:
        header = f.readline()
        assert header.strip() == "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius"
        rows = []
        for line in f:
            tokens = line.split()
            if len(tokens) != 11:
                continue
            rows.append([float(x) for x in tokens])
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.skipif(
    not _REF_HKLS_CSV.exists(),
    reason="NF_HEDM/Example/sim/hkls.csv not found (run the C pipeline once first)",
)
def test_au_row_count_matches_c_reference():
    ref = _read_ref(_REF_HKLS_CSV)
    rows = emit_nf_hkls_csv(**_au_inputs())
    assert len(rows) == ref.shape[0], (
        f"row count mismatch: got {len(rows)}, ref {ref.shape[0]}"
    )


@pytest.mark.skipif(
    not _REF_HKLS_CSV.exists(),
    reason="NF_HEDM/Example/sim/hkls.csv not found",
)
def test_au_hkl_set_matches_c_reference():
    ref = _read_ref(_REF_HKLS_CSV)
    rows = emit_nf_hkls_csv(**_au_inputs())
    py_arr = np.asarray(rows, dtype=np.float64)

    ref_hkl = {tuple(int(x) for x in r[:3]) for r in ref}
    py_hkl = {tuple(int(x) for x in r[:3]) for r in py_arr}
    assert ref_hkl == py_hkl, f"missing in py: {ref_hkl - py_hkl}, extra: {py_hkl - ref_hkl}"


@pytest.mark.skipif(
    not _REF_HKLS_CSV.exists(),
    reason="NF_HEDM/Example/sim/hkls.csv not found",
)
def test_au_per_row_columns_match_c_reference():
    ref = _read_ref(_REF_HKLS_CSV)
    rows = emit_nf_hkls_csv(**_au_inputs())
    py_arr = np.asarray(rows, dtype=np.float64)

    # Sort both by (RingNr, h, k, l) so we can compare element-wise.
    def _key(arr):
        return np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0], arr[:, 4]))
    ref_sorted = ref[_key(ref)]
    py_sorted = py_arr[_key(py_arr)]

    # h, k, l, RingNr must be exactly equal (integers stored as floats).
    np.testing.assert_array_equal(ref_sorted[:, [0, 1, 2, 4]], py_sorted[:, [0, 1, 2, 4]])

    # D-spacing, g-vector, Theta, 2Theta, Radius: tight numerical tolerance.
    np.testing.assert_allclose(ref_sorted[:, 3], py_sorted[:, 3], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ref_sorted[:, 5:8], py_sorted[:, 5:8], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ref_sorted[:, 8], py_sorted[:, 8], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ref_sorted[:, 9], py_sorted[:, 9], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ref_sorted[:, 10], py_sorted[:, 10], rtol=1e-12, atol=1e-12)


def test_writes_correct_header_and_first_row_format():
    """Smoke test: header line is exact + a row is parseable."""
    buf = StringIO()
    rows = emit_nf_hkls_csv(**_au_inputs(), fp=buf)
    text = buf.getvalue()
    first_line, second_line = text.splitlines()[:2]
    assert first_line == "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius"
    tokens = second_line.split()
    assert len(tokens) == 11
    # First three should be integer-valued.
    for tok in tokens[:3]:
        assert "." not in tok, f"hkl column should be integer, got {tok!r}"
    # RingNr must be 1 for the smallest-2θ row.
    assert int(float(tokens[4])) >= 1


def test_d_spacing_sorted_descending():
    rows = emit_nf_hkls_csv(**_au_inputs())
    ds = [r[3] for r in rows]
    assert all(ds[i] >= ds[i + 1] for i in range(len(ds) - 1)), \
        "rows must be sorted by descending d-spacing"


def test_ring_numbering_monotonic_and_starts_at_one():
    rows = emit_nf_hkls_csv(**_au_inputs())
    ring_nrs = [int(r[4]) for r in rows]
    assert ring_nrs[0] == 1
    # RingNr is non-decreasing and increments by at most 1 between rows.
    for prev, curr in zip(ring_nrs, ring_nrs[1:]):
        assert curr in (prev, prev + 1)


# --- the Miller box must not truncate long-axis cells (found on Ruby/corundum, 2026-10-18) ----------------
def _corundum_inputs():
    # HS-brief ruby cell, SG 167 (R-3c, hexagonal axes), 60 keV, NF geometry of the P21.2 Ruby scan
    return {
        "space_group": SpaceGroup.from_number(167),
        "lattice": Lattice(4.7607, 4.7607, 12.9947, 90.0, 90.0, 120.0),
        "wavelength_A": 12.398419843320026 / 60.0,
        "lsd_um": 6525.0,
        "max_ring_rad_um": 1600.0,
    }


def test_default_box_does_not_truncate_long_c_axis():
    """With the old fixed +-10 box (0 0 12), (1 0 -14), (2 -1 12), (3 -1 -11) (all strong, all inside the ring cutoff)
    were silently dropped. The default (automatic) box must return exactly the set a very large explicit box returns."""
    auto = emit_nf_hkls_csv(**_corundum_inputs())
    big = emit_nf_hkls_csv(**_corundum_inputs(), h_max=30, k_max=30, l_max=30)
    old = emit_nf_hkls_csv(**_corundum_inputs(), h_max=10, k_max=10, l_max=10)
    hkls = lambda rows: {tuple(int(x) for x in r[:3]) for r in rows}
    assert hkls(auto) == hkls(big)
    assert len(auto) == len(big) == 442
    for t in ((0, 0, 12), (0, 0, -12), (1, 0, -14), (2, -1, 12), (3, -1, -11)):
        assert t in hkls(auto), t
        assert t not in hkls(old), t          # documents the old truncation
    assert len(old) == 410 and len(auto) - len(old) == 32


def test_default_box_equals_old_box_when_old_box_was_enough():
    """cubic Au (the C-parity case): d_min is large enough that a/d_min < 10, so the automatic box is the old box."""
    auto = emit_nf_hkls_csv(**_au_inputs())
    old = emit_nf_hkls_csv(**_au_inputs(), h_max=10, k_max=10, l_max=10)
    assert [r[:5] for r in auto] == [r[:5] for r in old]
