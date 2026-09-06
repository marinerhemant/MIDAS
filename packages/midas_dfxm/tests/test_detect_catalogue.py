"""Regression tests for four defects found in detect.py / dislocation.py on 2026-09-04.

All four were found by adversarial verification of claim c06d5e00a8fa, and all four
were present in the released 0.7.0. Each test fails on the pre-fix code.
"""
from __future__ import annotations

import pytest
import torch

from midas_dfxm import cubic_stiffness
from midas_dfxm.detect import _candidate_dislocations, _normalize
from midas_dfxm.dislocation import (dislocation_deformation_field, fcc_slip_systems,
                                    stroh_dislocation)

DT = torch.float64
CU = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)


def _catalogue(C6=CU, **kw):
    kw.setdefault("characters", ("edge", "screw"))
    kw.setdefault("signs", (1, -1))
    return list(_candidate_dislocations(
        C6, fcc_slip_systems(), positions_core=(0.0, 0.0, 0.0),
        core_radius_um=0.4, burgers_length_A=2.556, crystal=None, **kw))


def test_screws_are_not_duplicated_across_slip_planes():
    """A screw's line is b, so its field is plane-independent; fcc <110> lies in two
    {111}, so enumerating the plane emitted every screw twice with identical fields."""
    cat = _catalogue()
    screws = [lab for lab, _ in cat if lab[2] == "screw"]
    edges = [lab for lab, _ in cat if lab[2] == "edge"]
    assert len(screws) == 12, f"expected 12 distinct screws (6 b x 2 signs), got {len(screws)}"
    assert len(edges) == 24, f"expected 24 edges (12 systems x 2 signs), got {len(edges)}"
    assert len(cat) == 36
    # and each (b, sign) appears exactly once
    keys = [(lab[1], lab[3]) for lab in screws]
    assert len(set(keys)) == len(keys)


def test_screw_slip_normal_is_reported_as_undetermined():
    """Returning one of the two containing planes as if determined is a false claim."""
    for lab, _ in _catalogue():
        if lab[2] == "screw":
            assert lab[0] is None
        else:
            assert lab[0] is not None


def test_screw_field_really_is_plane_independent():
    """The premise of the deduplication, checked directly rather than assumed."""
    systems = fcc_slip_systems()
    b = systems[0][1]
    planes = [n for n, bb in systems if tuple(bb) == tuple(b)]
    assert len(planes) == 2, "fcc <110> should lie in exactly two {111}"
    pts = torch.randn(64, 3, dtype=DT)
    fields = []
    for n in planes:
        d = stroh_dislocation(CU, burgers=b, slip_normal=n, character="screw",
                              burgers_length_A=2.556, core_radius_um=0.4)
        fields.append(dislocation_deformation_field(pts, d).F)
    diff = (fields[0] - fields[1]).abs().max() / fields[0].abs().max()
    assert float(diff) < 1e-12, f"screw field depends on slip plane: rel diff {float(diff):.2e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_nan_when_a_sampled_point_lies_on_the_dislocation_line(dtype):
    """The core guard was a literal 1e-300, which underflows to 0.0 in float32, so the
    on-line voxel evaluated 0/0. Eight of the 48 fcc candidates returned NaN."""
    C6 = cubic_stiffness(168.4, 121.4, 75.4, dtype=dtype)
    xs = torch.linspace(-5.0, 5.0, 21, dtype=dtype)          # includes exactly 0
    gx, gy, gz = torch.meshgrid(xs, xs, torch.zeros(1, dtype=dtype), indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    for lab, disl in _catalogue(C6):
        F = dislocation_deformation_field(pts, disl).F
        assert not torch.isnan(F).any(), f"NaN deformation for {lab} at {dtype}"


def test_normalize_zeroes_an_invisible_row_instead_of_amplifying_roundoff():
    """A g.b = 0 row has mean-removed norm ~1 ulp; dividing by it produced unit-norm
    noise that then took full weight in match_residual."""
    stack = torch.stack([
        torch.full((64,), 3.7, dtype=DT),        # constant -> invisible
        torch.randn(64, dtype=DT),               # real contrast
    ])
    out = _normalize(stack)
    assert float(out[0].norm()) == 0.0
    assert abs(float(out[1].norm()) - 1.0) < 1e-12


def test_edge_line_is_built_in_cartesian_not_miller_space():
    """`cross(n, b)` on raw indices mixes a reciprocal normal with a direct direction
    and is only valid for cubic. Checked on a tetragonal cell, where they differ."""
    from midas_hkls.crystal import Crystal, Lattice, SpaceGroup

    from midas_dfxm.dislocation import _cartesian_geometry
    # The cell must have a != b. Measured 2026-09-04: the Miller cross, converted back
    # as a direct vector, coincides with the Cartesian one to ~1e-16 for cubic,
    # tetragonal AND hexagonal (all a = b), and differs only when a != b --
    # orthorhombic 9.8e-2, monoclinic 1.0e-2, triclinic 5.8e-2 in unit-vector distance.
    # So the old code was wrong for orthorhombic and lower, not merely "non-cubic".
    crystal = Crystal(lattice=Lattice(3.0, 4.0, 5.0, 90.0, 90.0, 90.0),
                      space_group=SpaceGroup.from_number(1))
    # b must lie in the plane: h*u + k*v + l*w = 0.
    n, b = (1, 1, 1), (1, -1, 0)
    b_cart, n_cart, _ = _cartesian_geometry(b, n, b, crystal=crystal,
                                            dtype=DT, device=torch.device("cpu"))
    cart_cross = torch.linalg.cross(n_cart, b_cart)
    miller_cross = torch.as_tensor([
        n[1] * b[2] - n[2] * b[1],
        n[2] * b[0] - n[0] * b[2],
        n[0] * b[1] - n[1] * b[0]], dtype=DT)
    cart_hat = cart_cross / cart_cross.norm()
    mill_hat = miller_cross / miller_cross.norm()
    # they must differ for a non-cubic cell -- if not, this test proves nothing
    assert float((cart_hat - mill_hat).abs().max()) > 1e-6, "test cell is degenerate"
    # and the catalogue must use the Cartesian one
    d_char = stroh_dislocation(CU, burgers=b, slip_normal=n, character="edge",
                               burgers_length_A=2.556, core_radius_um=0.4,
                               crystal=crystal)
    d_line = stroh_dislocation(CU, burgers=b, slip_normal=n, line=miller_cross,
                               burgers_length_A=2.556, core_radius_um=0.4,
                               crystal=crystal)
    pts = torch.randn(32, 3, dtype=DT)
    Fa = dislocation_deformation_field(pts, d_char).F
    Fb = dislocation_deformation_field(pts, d_line).F
    assert float((Fa - Fb).abs().max()) > 1e-9, (
        "character= and Miller line= give the same field; the test cell is not "
        "discriminating and this regression is not actually locked down")
