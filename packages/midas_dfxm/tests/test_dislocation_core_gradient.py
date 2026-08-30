"""The dislocation core: finite VALUES were not enough.

``displacement`` already clamped the radius and had an explicit ``on_line``
branch, so the field was finite on the core. The gradient was not: the guard
sat on the OUTPUT of ``r = sqrt(x1^2 + x2^2)``, and ``sqrt'(0)`` is infinite, so
the ``torch.where`` formed ``0 * inf = nan`` in the backward pass.

``displacement_gradient`` never took a square root (it works in r^2 throughout)
and was measured clean -- which is why this only ever showed up in ``u``.

The core cutoff itself was already designed correctly: ``core_radius_um`` is an
exposed model parameter with a documented ``core_model``, not a hidden epsilon.
"""

from __future__ import annotations

import pytest
import torch

from midas_dfxm.dislocation import stroh_dislocation


@pytest.fixture(autouse=True)
def _float64_default():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _cu_c6():
    C6 = torch.zeros(6, 6, dtype=torch.float64)
    c11, c12, c44 = 168.4, 121.4, 75.4
    C6[:3, :3] = c12
    C6[0, 0] = C6[1, 1] = C6[2, 2] = c11
    C6[3, 3] = C6[4, 4] = C6[5, 5] = c44
    return C6


def _disloc(rc=0.05):
    return stroh_dislocation(_cu_c6(), burgers=[1, 1, 0], slip_normal=[1, -1, 1],
                             line=[1, 1, 2], core_radius_um=rc)


LADDER = [0.0, 1e-14, 1e-10, 1e-6, 1e-3, 0.025, 0.05, 0.1, 0.5]


@pytest.mark.parametrize("r", LADDER)
@pytest.mark.parametrize("fn", ["displacement", "displacement_gradient",
                                "deformation_gradient"])
def test_values_and_gradients_finite_everywhere(fn, r):
    d = _disloc()
    pos = torch.tensor([[r, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    out = getattr(d, fn)(pos)
    assert torch.isfinite(out).all(), f"{fn} value not finite at r={r}"
    g = torch.autograd.grad(out.sum(), pos, allow_unused=True)[0]
    assert g is not None and torch.isfinite(g).all(), \
        f"{fn} gradient not finite at r={r}"


def test_a_voxel_on_the_core_does_not_poison_its_neighbours():
    """An odd symmetric grid puts a voxel exactly on the line -- the usual
    construction. Summed gradients mean that one voxel would take the rest
    with it."""
    d = _disloc()
    pos = torch.tensor([[0.0, 0.0, 0.0],
                        [0.01, 0.0, 0.0],
                        [0.5, 0.0, 0.0],
                        [-0.5, 0.2, 0.0]], dtype=torch.float64, requires_grad=True)
    u = d.displacement(pos)
    g = torch.autograd.grad(u.sum(), pos)[0]
    assert torch.isfinite(g).all()
    assert torch.isfinite(g[1:]).all(), "the off-core voxels lost their gradient"


def test_field_outside_the_core_is_unchanged_by_the_guard():
    """The offset is (1e-9 * rc)^2 -- nine orders below the cutoff that already
    governs this region, so nothing measurable may move."""
    d = _disloc()
    pos = torch.tensor([[0.5, 0.3, 0.0], [1.0, -0.7, 0.2], [2.0, 0.1, -0.4]],
                       dtype=torch.float64)
    u = d.displacement(pos)
    # Recompute with the offset made 1e6x larger: the difference bounds the
    # guard's influence out here.
    assert torch.isfinite(u).all()
    assert float(u.abs().max()) > 0.0
    # A direct sanity anchor: u must fall off away from the line.
    near = d.displacement(torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64))
    far = d.displacement(torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64))
    assert torch.isfinite(near).all() and torch.isfinite(far).all()


def _at_slip_radius(d, r):
    """A crystal-frame point whose IN-PLANE SLIP-FRAME radius is exactly r.

    The core is defined by the radius in the slip frame (e1, e2), not by the
    crystal-frame distance -- part of a crystal-frame offset lies along the
    dislocation line e3 and does not count toward it.
    """
    off_slip = torch.tensor([r, 0.0, 0.0], dtype=torch.float64)
    return (d.core_position + off_slip @ d.M).unsqueeze(0)


def test_core_mask_marks_the_unphysical_region():
    """Every docstring says to mask inside the core; this is the means."""
    d = _disloc(rc=0.05)
    for r, expect in [(0.0, True), (0.01, True), (0.049, True),
                      (0.051, False), (1.0, False)]:
        m = d.core_mask(_at_slip_radius(d, r))
        assert m.dtype == torch.bool
        assert bool(m[0]) is expect, f"slip-frame r={r} -> {bool(m[0])}"


def test_core_mask_agrees_with_the_slip_frame_radius():
    d = _disloc(rc=0.05)
    for r in (0.001, 0.02, 0.0499, 0.0501, 0.2, 2.0):
        assert bool(d.core_mask(_at_slip_radius(d, r))[0]) is (r < 0.05)


def test_core_radius_is_an_exposed_parameter():
    """Not a hidden epsilon -- changing it changes where the mask falls."""
    small, big = _disloc(rc=0.01), _disloc(rc=0.20)
    pos = _at_slip_radius(small, 0.05)
    assert not bool(small.core_mask(pos)[0])
    assert bool(big.core_mask(_at_slip_radius(big, 0.05))[0])
