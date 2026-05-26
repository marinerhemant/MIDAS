"""Tests for the extended twin-relations dispatcher.

Covers:
* default_cubic_twin_relations — Σ3 / Σ9 / Σ27a / Σ27b operators
* default_hcp_twin_relations — Ti-Al c/a = 1.587 verifies expected misori
  (85° for tension {10-12}, etc.)
* default_trigonal_twin_relations — 60° around c
* default_twin_relations_for — dispatcher coverage by space group
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from midas_process_grains.compute.twins import (
    TwinRelation,
    default_fcc_twin_relations,
    default_cubic_twin_relations,
    default_hcp_twin_relations,
    default_tetragonal_twin_relations,
    default_trigonal_twin_relations,
    default_twin_relations_for,
    hcp_hkil_to_cartesian_normal,
    tetragonal_hkl_to_cartesian_normal,
)


# ---------------------------------------------------------------------------
# Cubic Σ3 / Σ9 / Σ27
# ---------------------------------------------------------------------------


def test_cubic_default_includes_sigma3_sigma9_sigma27ab():
    tw = default_cubic_twin_relations()
    names = [t.name for t in tw]
    assert sum("Sigma3" in n for n in names) == 4   # four <111> Σ3 variants
    assert sum(n == "FCC_Sigma9"   for n in names) == 1
    assert sum(n == "FCC_Sigma27a" for n in names) == 1
    assert sum(n == "FCC_Sigma27b" for n in names) == 1


def test_cubic_default_subset_via_include_kwarg():
    tw = default_cubic_twin_relations(include=("Sigma3",))
    assert all("Sigma3" in t.name for t in tw)
    assert len(tw) == 4


def test_cubic_bcc_lattice_label():
    tw = default_cubic_twin_relations(lattice="BCC")
    assert all(t.name.startswith("BCC_") for t in tw)


def test_sigma9_operator_matches_38_94_around_110():
    """A 38.94° rotation around <110> is the Σ9 CSL boundary."""
    tw = [t for t in default_cubic_twin_relations() if t.name == "FCC_Sigma9"][0]
    # Quat → axis/angle
    q = tw.quaternion
    angle = 2.0 * math.degrees(math.acos(min(max(q[0], -1.0), 1.0)))
    assert abs(angle - 38.94) < 0.05
    # Axis should be on <110> direction (normalised)
    axis = q[1:] / np.linalg.norm(q[1:])
    expected_axis = np.array([1, 1, 0]) / np.sqrt(2.0)
    assert np.allclose(axis, expected_axis, atol=1e-6)


# ---------------------------------------------------------------------------
# HCP K1 cartesian normal helper
# ---------------------------------------------------------------------------


def test_hcp_hkil_normal_rejects_inconsistent_i_index():
    with pytest.raises(ValueError):
        hcp_hkil_to_cartesian_normal(1, 0, 0, 2, c_over_a=1.587)


def test_hcp_hkil_normal_1012_for_ti():
    """{10-12} K1 normal for Ti c/a = 1.587 — verify cartesian direction."""
    n = hcp_hkil_to_cartesian_normal(1, 0, -1, 2, c_over_a=1.587)
    # Manually: (h, (h + 2k)/√3, l/(c/a)) = (1, 1/√3, 2/1.587)
    expected = np.array([1.0, 1.0 / math.sqrt(3), 2.0 / 1.587])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(n, expected, atol=1e-9)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# HCP twin disorientation under SG 194 symmetry
# ---------------------------------------------------------------------------


def _quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def _hcp_tension_1012_expected_misori_deg(c_over_a: float) -> float:
    """Closed-form: tension {10-12} misori = 2·arctan(c_a / √3)."""
    return 2.0 * math.degrees(math.atan2(c_over_a, math.sqrt(3.0)))


def test_hcp_tension_1012_misori_85deg_for_ti():
    """Applying the 180°-around-K1 operator to identity, then computing
    the symmetry-aware disorientation under SG 194, must give ~85° for
    Ti c/a = 1.587 (the textbook value for the tension twin).

    With variant-level labelling, 6 K1 variants per system → 6
    operators returned for tension_1012. ALL 6 must produce 85°
    disorientation (they're hex-6-fold equivalents).
    """
    from midas_stress.orientation import misorientation_quat_batch

    tw = default_hcp_twin_relations(c_over_a=1.587, systems=("tension_1012",))
    assert len(tw) == 6   # 6 K1 variants per system
    T = tw[0].quaternion

    # Grain A = identity; grain B = A · T (B is the twin of A)
    qA = np.array([1.0, 0.0, 0.0, 0.0])
    qB = _quat_mul(qA, T)
    qB /= np.linalg.norm(qB)

    qa = torch.from_numpy(np.ascontiguousarray(qA[None, :]))
    qb = torch.from_numpy(np.ascontiguousarray(qB[None, :]))
    miso_rad = misorientation_quat_batch(qa, qb, 194).numpy()[0]
    miso_deg = math.degrees(miso_rad)

    expected = _hcp_tension_1012_expected_misori_deg(1.587)
    assert abs(expected - 85.0) < 0.5, (
        f"Closed-form HCP {{10-12}} misori for c/a=1.587 should be ~85°, got {expected:.2f}°"
    )
    assert abs(miso_deg - expected) < 1.0, (
        f"SG-194 disorientation of the {{10-12}} twin should match the closed-form "
        f"~{expected:.2f}°, got {miso_deg:.2f}°"
    )


def test_hcp_all_systems_emit_six_variants_each():
    """5 systems × 6 K1 variants = 30 operators returned (variant-level
    labelling — each named with both the system AND the K1 quad)."""
    systems = (
        "tension_1012", "compression_1011", "compression_2112",
        "tension_1121", "compression_1122",
    )
    tw = default_hcp_twin_relations(c_over_a=1.587, systems=systems)
    assert len(tw) == 30   # 5 systems × 6 K1 variants
    sys_counts = {sys_name: sum(1 for op in tw if sys_name in op.name)
                  for sys_name in systems}
    for sys_name, count in sys_counts.items():
        assert count == 6, f"{sys_name} got {count} variants, expected 6"


def test_hcp_rejects_unknown_system():
    with pytest.raises(ValueError, match="Unknown HCP twin system"):
        default_hcp_twin_relations(c_over_a=1.587, systems=("foobar",))


# ---------------------------------------------------------------------------
# Trigonal
# ---------------------------------------------------------------------------


def test_trigonal_returns_one_60deg_around_c():
    tw = default_trigonal_twin_relations()
    assert len(tw) == 1
    assert tw[0].angle_deg == 60.0
    assert tw[0].axis == (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_cubic_fcc_sg225():
    tw = default_twin_relations_for(225)
    assert len(tw) == 7   # 4 Σ3 + 1 Σ9 + 1 Σ27a + 1 Σ27b
    assert all(t.name.startswith("FCC_") for t in tw)


def test_dispatcher_cubic_bcc_sg229():
    tw = default_twin_relations_for(229)
    assert all(t.name.startswith("BCC_") for t in tw)


def test_dispatcher_hcp_sg194_requires_c_over_a():
    with pytest.raises(ValueError, match="c_over_a"):
        default_twin_relations_for(194)


def test_dispatcher_hcp_sg194_with_c_over_a():
    tw = default_twin_relations_for(194, c_over_a=1.587)
    assert len(tw) == 30   # 5 systems × 6 K1 variants
    assert all("HCP_" in t.name for t in tw)


def test_dispatcher_trigonal_sg167():
    tw = default_twin_relations_for(167)
    assert len(tw) == 1
    assert tw[0].name == "Trigonal_c60"


def test_dispatcher_unsupported_returns_empty():
    """Orthorhombic / monoclinic / triclinic — no defaults yet."""
    for sg in (14, 71):
        tw = default_twin_relations_for(sg)
        assert tw == [], f"SG {sg} should produce empty twin set"


def test_dispatcher_hcp_system_subset():
    tw = default_twin_relations_for(
        194, c_over_a=1.587, hcp_systems=("tension_1012",),
    )
    assert len(tw) == 6   # 1 system × 6 K1 variants
    assert all("tension_1012" in t.name for t in tw)


# ---------------------------------------------------------------------------
# Tetragonal K1 cartesian normal helper
# ---------------------------------------------------------------------------


def test_tetragonal_hkl_normal_101_for_fept():
    """{101} K1 normal for FePt c/a = 0.967 — verify cartesian direction."""
    n = tetragonal_hkl_to_cartesian_normal(1, 0, 1, c_over_a=0.967)
    # n = (h, k, l/(c/a)) = (1, 0, 1/0.967)
    expected = np.array([1.0, 0.0, 1.0 / 0.967])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(n, expected, atol=1e-9)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-9


def test_tetragonal_hkl_normal_in_cubic_limit():
    """When c/a == 1, the {101} normal collapses to the cubic <101> = (1, 0, 1)/√2."""
    n = tetragonal_hkl_to_cartesian_normal(1, 0, 1, c_over_a=1.0)
    np.testing.assert_allclose(n, np.array([1, 0, 1]) / math.sqrt(2.0), atol=1e-12)


def test_tetragonal_hkl_rejects_zero_normal():
    with pytest.raises(ValueError, match="Degenerate"):
        tetragonal_hkl_to_cartesian_normal(0, 0, 0, c_over_a=1.0)


# ---------------------------------------------------------------------------
# Tetragonal twin builder
# ---------------------------------------------------------------------------


def test_tetragonal_default_emits_four_systems():
    tw = default_tetragonal_twin_relations(c_over_a=0.967)
    assert len(tw) == 4
    sys_names = [t.name for t in tw]
    assert any("twin_101" in s for s in sys_names)
    assert any("twin_011" in s for s in sys_names)
    assert any("twin_112" in s for s in sys_names)
    assert any("twin_103" in s for s in sys_names)


def test_tetragonal_rejects_unknown_system():
    with pytest.raises(ValueError, match="Unknown tetragonal twin system"):
        default_tetragonal_twin_relations(c_over_a=1.0, systems=("foo",))


def test_tetragonal_supports_110_twin():
    """The {110} twin is the cubic-limit 90°-about-c twin — exposed
    as an optional system."""
    tw = default_tetragonal_twin_relations(c_over_a=1.0, systems=("twin_110",))
    assert len(tw) == 1
    # K1 normal for {110} when c/a doesn't enter (l = 0): n = (1, 1, 0)/√2
    n = np.array(tw[0].axis)
    expected = np.array([1, 1, 0]) / math.sqrt(2.0)
    np.testing.assert_allclose(n, expected, atol=1e-9)


def test_tetragonal_101_disorientation_under_4mmm():
    """Apply the {101} twin operator to identity, compute disorientation
    under tetragonal symmetry (P4/mmm = SG 123 = 8 sym ops). The result
    must be non-trivial (≠ 0°) AND ≤ 180°."""
    from midas_stress.orientation import misorientation_quat_batch

    tw = default_tetragonal_twin_relations(c_over_a=0.967, systems=("twin_101",))
    T = tw[0].quaternion

    qA = np.array([1.0, 0.0, 0.0, 0.0])
    qB = _quat_mul(qA, T)
    qB /= np.linalg.norm(qB)

    qa = torch.from_numpy(np.ascontiguousarray(qA[None, :]))
    qb = torch.from_numpy(np.ascontiguousarray(qB[None, :]))
    # SG 123 = P4/mmm tetragonal
    miso_rad = misorientation_quat_batch(qa, qb, 123).numpy()[0]
    miso_deg = math.degrees(miso_rad)
    assert 0.5 < miso_deg <= 180.0, (
        f"{{101}} twin disorientation for c/a=0.967 should be non-trivial; "
        f"got {miso_deg:.2f}°"
    )


# ---------------------------------------------------------------------------
# Dispatcher: tetragonal
# ---------------------------------------------------------------------------


def test_dispatcher_tetragonal_sg123_requires_c_over_a():
    with pytest.raises(ValueError, match="c_over_a"):
        default_twin_relations_for(123)


def test_dispatcher_tetragonal_sg123_with_c_over_a():
    tw = default_twin_relations_for(123, c_over_a=0.967)
    assert len(tw) == 4
    assert all("Tetragonal_" in t.name for t in tw)


def test_dispatcher_tetragonal_system_subset():
    tw = default_twin_relations_for(
        139, c_over_a=0.967, tetragonal_systems=("twin_101",),
    )
    assert len(tw) == 1
    assert "twin_101" in tw[0].name


def test_dispatcher_tetragonal_covers_sg_range_75_to_142():
    """Spot-check a few tetragonal space groups across the 75-142 range."""
    for sg in (75, 99, 123, 136, 139, 142):
        tw = default_twin_relations_for(sg, c_over_a=1.0)
        assert len(tw) == 4, f"SG {sg} should dispatch to tetragonal"
