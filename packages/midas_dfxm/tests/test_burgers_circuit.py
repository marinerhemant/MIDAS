"""Convention-free gate on the Stroh field: the Burgers circuit must close on b.

The defining property of a dislocation is that the closed-loop integral of the distortion
around its line equals the Burgers vector:

    contour integral of  beta . dl  =  b

This is independent of every sign, frame and normalization convention, which makes it the
right regression gate. A missing factor of 2 in the Stroh prefactor (summing one root per
conjugate pair needs ``2 Im{...}``) made this integral return ``b/2`` and went unnoticed
because most downstream results are ratio-based or use the same forward on both sides.
"""
import math

import pytest
import torch

from midas_dfxm import cubic_stiffness, stroh_dislocation

torch.set_default_dtype(torch.float64)


def _circuit(d, radius=5.0, n=20000):
    """Closed-loop integral of beta.dl in the plane perpendicular to the line."""
    th = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    loop = torch.stack([radius * torch.cos(th), radius * torch.sin(th),
                        torch.zeros_like(th)], -1)
    dl = torch.stack([-radius * torch.sin(th), radius * torch.cos(th),
                      torch.zeros_like(th)], -1) * (2 * math.pi / n)
    pts = loop @ d.M + d.core_position          # slip-frame loop -> crystal frame
    return torch.einsum("nij,nj->i", d.displacement_gradient(pts), dl @ d.M)


@pytest.mark.unit
@pytest.mark.parametrize("character", ["edge", "screw"])
@pytest.mark.parametrize("b_len_A", [2.556, 3.615])
def test_burgers_circuit_closes_on_b(character, b_len_A):
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=torch.float64)
    d = stroh_dislocation(C, burgers=(1, -1, 0), slip_normal=(1, 1, 1),
                          character=character, burgers_length_A=b_len_A,
                          core_radius_um=1e-4)
    got = float(_circuit(d).norm())
    expect = b_len_A * 1e-4                      # Angstrom -> micrometre
    assert abs(got / expect - 1.0) < 1e-5, f"circuit gave {got/expect:.4f} |b|"


@pytest.mark.unit
def test_circuit_is_radius_independent():
    """A closed circuit must give b regardless of how far out it is drawn."""
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=torch.float64)
    d = stroh_dislocation(C, burgers=(1, -1, 0), slip_normal=(1, 1, 1),
                          character="edge", burgers_length_A=2.556, core_radius_um=1e-4)
    vals = [float(_circuit(d, radius=r).norm()) for r in (1.0, 5.0, 20.0)]
    assert max(vals) / min(vals) - 1.0 < 1e-6


@pytest.mark.unit
def test_displacement_gradient_matches_displacement():
    """grad(u) must reproduce beta -- the two must never drift apart again.

    Compared far from the core, where the Lorentzian cutoff on beta (absent from u)
    is negligible.
    """
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=torch.float64)
    d = stroh_dislocation(C, burgers=(1, -1, 0), slip_normal=(1, 1, 1),
                          character="edge", burgers_length_A=2.556, core_radius_um=0.05)
    torch.manual_seed(0)
    pts = torch.randn(400, 3) * 20.0
    pts = pts[pts[:, :2].norm(dim=1) > 20.0][:40]
    h = 1e-4
    num = torch.zeros(len(pts), 3, 3)
    for j in range(3):
        e = torch.zeros(3); e[j] = h
        num[:, :, j] = (d.displacement(pts + e) - d.displacement(pts - e)) / (2 * h)
    ana = d.displacement_gradient(pts)
    assert float((num - ana).abs().max() / ana.abs().max()) < 5e-3
