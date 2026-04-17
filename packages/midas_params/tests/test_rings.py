"""Ring enumeration tests — crystal geometry sanity checks."""

from __future__ import annotations

import math

import pytest

from midas_params.rings import (
    RingInfo,
    enumerate_rings,
    format_ring_table,
    recommend_rings,
    _d_cubic,
    _d_hexagonal,
    _extinct_fcc,
    _extinct_bcc,
)


# ─── d-spacing formulas ──────────────────────────────────────────────────────


def test_d_cubic_known_values():
    # Au (a=4.08 Å)
    assert _d_cubic(1, 1, 1, 4.08) == pytest.approx(2.356, abs=1e-3)
    assert _d_cubic(2, 0, 0, 4.08) == pytest.approx(2.040, abs=1e-3)


def test_d_hexagonal():
    # Zn (a=2.665 Å, c=4.947 Å); d(002) = c/2 = 2.4735
    assert _d_hexagonal(0, 0, 2, 2.665, 4.947) == pytest.approx(2.4735, abs=1e-4)


# ─── Extinction rules ────────────────────────────────────────────────────────


def test_extinct_fcc_allows_all_same_parity():
    assert not _extinct_fcc(1, 1, 1)    # all odd → allowed
    assert not _extinct_fcc(2, 0, 0)    # all even → allowed
    assert not _extinct_fcc(2, 2, 0)    # all even → allowed
    assert _extinct_fcc(1, 1, 0)        # mixed → forbidden
    assert _extinct_fcc(2, 1, 0)        # mixed → forbidden


def test_extinct_bcc_wants_even_sum():
    assert not _extinct_bcc(1, 1, 0)    # sum=2, allowed
    assert not _extinct_bcc(2, 0, 0)    # sum=2, allowed
    assert not _extinct_bcc(2, 1, 1)    # sum=4, allowed
    assert _extinct_bcc(1, 0, 0)        # sum=1, forbidden
    assert _extinct_bcc(2, 1, 0)        # sum=3, forbidden


# ─── enumerate_rings ─────────────────────────────────────────────────────────


def test_fcc_au_first_five_rings():
    """Au FCC (a=4.08, SG 225) at standard HEDM wavelength/Lsd: first 5 rings
    must be 111, 200, 220, 311, 222."""
    rings = enumerate_rings(
        wavelength=0.22291, lsd_um=1_000_000,
        lattice=[4.08, 4.08, 4.08, 90, 90, 90],
        space_group=225, rho_d_um=204800, max_rings=5,
    )
    hkls = [r.hkl for r in rings]
    assert hkls == [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2)]


def test_bcc_extinction_applied():
    """BCC Fe (a=2.866, SG 229): first ring should be 110 (not 100, which is extinct)."""
    rings = enumerate_rings(
        wavelength=1.5418, lsd_um=100_000,
        lattice=[2.866, 2.866, 2.866, 90, 90, 90],
        space_group=229, rho_d_um=100_000, max_rings=3,
    )
    assert rings[0].hkl == (1, 1, 0)


def test_rings_sorted_by_two_theta():
    rings = enumerate_rings(
        wavelength=0.22291, lsd_um=1_000_000,
        lattice=[4.08, 4.08, 4.08, 90, 90, 90],
        space_group=225, rho_d_um=204800, max_rings=10,
    )
    # 2θ should be strictly increasing (d-spacings strictly decreasing)
    for a, b in zip(rings, rings[1:]):
        assert a.two_theta < b.two_theta


def test_on_detector_flag():
    """Rings beyond RhoD are flagged on_detector=False."""
    rings = enumerate_rings(
        wavelength=0.22291, lsd_um=1_000_000,
        lattice=[4.08, 4.08, 4.08, 90, 90, 90],
        space_group=225,
        rho_d_um=150_000,   # restrictive — only low-2θ rings fit
        max_rings=10,
    )
    # First rings fit, later ones don't
    assert rings[0].on_detector
    assert not rings[-1].on_detector


def test_orthorhombic_distinct_rings():
    """Orthorhombic must not collapse (0,1,0) and (0,0,1) into the same hkl display."""
    rings = enumerate_rings(
        wavelength=1.5418, lsd_um=100_000,
        lattice=[4.76, 10.22, 5.99, 90, 90, 90],
        space_group=62, rho_d_um=100_000, max_rings=10,
    )
    hkls = [r.hkl for r in rings]
    # a=4.76, b=10.22, c=5.99 → (0,1,0) has largest d
    assert hkls[0] == (0, 1, 0)
    assert hkls[1] == (0, 0, 1)
    # Each hkl should be unique (the old bug collapsed all to (1,0,0))
    distinct_d = {r.d_spacing for r in rings}
    assert len(distinct_d) == len(rings)


def test_monoclinic_produces_rings():
    rings = enumerate_rings(
        wavelength=1.5418, lsd_um=100_000,
        lattice=[5.68, 15.2, 6.51, 90, 114.5, 90],
        space_group=15, rho_d_um=100_000, max_rings=5,
    )
    assert len(rings) == 5
    for r in rings:
        assert r.d_spacing > 0
        assert 0 < r.two_theta < 180


def test_triclinic_produces_rings():
    """General triclinic case should not raise; metric-tensor formula handles it."""
    rings = enumerate_rings(
        wavelength=1.5418, lsd_um=100_000,
        lattice=[7.0, 8.0, 9.0, 85, 95, 100],
        space_group=2, rho_d_um=100_000, max_rings=5,
    )
    assert rings
    for r in rings:
        assert r.d_spacing > 0


def test_recommend_rings_returns_first_onscreen():
    rings = enumerate_rings(
        wavelength=0.22291, lsd_um=1_000_000,
        lattice=[4.08, 4.08, 4.08, 90, 90, 90],
        space_group=225, rho_d_um=204800, max_rings=10,
    )
    rec = recommend_rings(rings, max_recommend=3)
    assert rec == [1, 2, 3]


def test_format_ring_table_plain():
    rings = enumerate_rings(
        wavelength=0.22291, lsd_um=1_000_000,
        lattice=[4.08, 4.08, 4.08, 90, 90, 90],
        space_group=225, rho_d_um=204800, max_rings=3,
    )
    out = format_ring_table(rings, use_color=False)
    assert "Ring" in out
    assert "(h k l)" in out
    assert "1" in out  # ring number
    assert "\033[" not in out  # no ANSI when use_color=False


def test_hexagonal_lattice_works():
    """Mg hexagonal (a=3.21, c=5.21): enumerate without crashing."""
    rings = enumerate_rings(
        wavelength=1.5418, lsd_um=100_000,
        lattice=[3.21, 3.21, 5.21, 90, 90, 120],
        space_group=194, rho_d_um=100_000, max_rings=5,
    )
    assert rings  # at least some rings found
    for r in rings:
        assert r.d_spacing > 0
        assert r.two_theta > 0
