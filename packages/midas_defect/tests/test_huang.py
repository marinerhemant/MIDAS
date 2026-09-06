"""Gates on the near-Bragg (Huang) diffuse forward.

Two of these are negative results, and they carry more weight than the positive
ones:

* `test_asymmetry_does_not_distinguish_interstitial_from_vacancy` -- the
  q/-q asymmetry is real and large, but at first order in the displacement it
  depends on the SQUARE of the defect strength, so it is identical for the two
  loop characters. The implementation plan for this module assumed the opposite.
  Encoding the truth stops the wrong claim being made from these numbers.
* `test_open_lines_contribute_nothing_here_either` -- same scope limit as the
  small-angle side.
"""
import math

import pytest
import torch

ddd = pytest.importorskip("midas_ddd")
from midas_ddd import (  # noqa: E402
    combine,
    find_loops,
    isotropic_stiffness,
    prismatic_loop,
    relaxation_volumes_um3,
    straight_line,
)
from midas_defect.huang import (  # noqa: E402
    huang_amplitude,
    huang_intensity,
    huang_vs_small_angle_ratio,
    qG_asymmetry,
)

LAM, MU = 100.0, 75.0
# Cu 111 at 3.07 1/A. The kernel works in inverse MICROMETERS, so 1e4 x that.
G_CU_111 = torch.tensor([3.07e4, 0.0, 0.0], dtype=torch.float64)


def _loop(scale=1.0, n_seg=48):
    return prismatic_loop(radius_um=0.005, burgers=(0.0, 0.0, 1.0),
                          n_segments=n_seg, burgers_scale_b=scale)


def _unit(v):
    v = torch.as_tensor(v, dtype=torch.float64)
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# The Huang law
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_huang_intensity_follows_the_inverse_q_squared_law():
    """Near a Bragg peak the strain-field diffuse intensity goes as 1/q^2.

    u~ carries 1/q for the dilatation part, so |G.u~|^2 ~ (G/q)^2 -- the
    signature that says this is Huang scattering and not something else.
    """
    C6 = isotropic_stiffness(LAM, MU)
    net = _loop()
    d = _unit(torch.tensor([[0.3, 0.0, 1.0]]))
    I = [float(huang_intensity(net, G_CU_111, d * qm, C6)[0]) for qm in (50.0, 100.0)]
    assert I[0] / I[1] == pytest.approx(4.0, rel=0.05)


@pytest.mark.unit
def test_G_over_q_squared_is_an_upper_bound_attained_at_the_best_direction():
    """(G/q)^2 is the NAIVE estimate, and it is a bound, not the typical value.

    Over random q directions the maximum reaches the bound almost exactly, while
    the median sits ~18x below it. Advising a collaborator with the bound would
    overstate the near-Bragg advantage by more than an order of magnitude.

    The exact bound is ``((|G| + |q|)/|q|)^2``, not ``(G/q)^2``: the amplitude
    contracts against ``G + q``, whose magnitude exceeds ``|G|`` when q points
    along G. At q = 0.1 1/A that is a 6 % difference, which is why the looser
    form fails a 5 % tolerance.
    """
    C6 = isotropic_stiffness(LAM, MU)
    net = _loop()
    torch.manual_seed(0)
    d = _unit(torch.randn(400, 3, dtype=torch.float64))
    Gmag = float(torch.linalg.norm(G_CU_111))
    for qm in (10.0, 100.0, 1000.0):
        r = huang_vs_small_angle_ratio(net, G_CU_111, d * qm, C6)
        bound = ((Gmag + qm) / qm) ** 2
        assert float(r.max()) == pytest.approx(bound, rel=0.01)
        assert float(r.median()) < 0.2 * bound


@pytest.mark.unit
def test_the_advantage_can_vanish_in_special_directions():
    """Where G.u~ nearly vanishes but q.u~ does not, small-angle briefly wins.

    Pinned because "near-Bragg is always 1e4x stronger" is the kind of claim
    that survives into a proposal unchallenged.
    """
    C6 = isotropic_stiffness(LAM, MU)
    torch.manual_seed(0)
    d = _unit(torch.randn(400, 3, dtype=torch.float64))
    r = huang_vs_small_angle_ratio(_loop(), G_CU_111, d * 100.0, C6)
    assert float(r.min()) < 1.0


@pytest.mark.unit
def test_ratio_scales_as_q_to_the_minus_two():
    """The (G/q)^2 scaling itself, along a fixed direction."""
    C6 = isotropic_stiffness(LAM, MU)
    net = _loop()
    d = _unit(torch.tensor([[0.3, 0.0, 1.0]]))
    r1 = float(huang_vs_small_angle_ratio(net, G_CU_111, d * 100.0, C6)[0])
    r2 = float(huang_vs_small_angle_ratio(net, G_CU_111, d * 300.0, C6)[0])
    assert r1 / r2 == pytest.approx(9.0, rel=0.3)


@pytest.mark.unit
def test_median_advantage_across_a_realistic_saxs_range():
    """The number to actually quote: median over directions, not the bound."""
    C6 = isotropic_stiffness(LAM, MU)
    net = _loop()
    torch.manual_seed(0)
    d = _unit(torch.randn(400, 3, dtype=torch.float64))
    med_lo = float(huang_vs_small_angle_ratio(net, G_CU_111, d * 1000.0, C6).median())
    med_hi = float(huang_vs_small_angle_ratio(net, G_CU_111, d * 10.0, C6).median())
    assert 10.0 < med_lo < 1e3          # q = 0.1 1/A
    assert 1e4 < med_hi < 1e7           # q = 1e-3 1/A


@pytest.mark.unit
def test_intensity_scales_with_the_square_of_the_relaxation_volume():
    C6 = isotropic_stiffness(LAM, MU)
    d = _unit(torch.tensor([[0.3, 0.0, 1.0]])) * 100.0
    small = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=48)
    big = prismatic_loop(radius_um=0.010, burgers=(0, 0, 1.0), n_segments=48)
    dv_s = float(relaxation_volumes_um3(small, find_loops(small))[0])
    dv_b = float(relaxation_volumes_um3(big, find_loops(big))[0])
    I_s = float(huang_intensity(small, G_CU_111, d, C6)[0])
    I_b = float(huang_intensity(big, G_CU_111, d, C6)[0])
    assert I_b / I_s == pytest.approx((dv_b / dv_s) ** 2, rel=0.02)


# ---------------------------------------------------------------------------
# Asymmetry -- present, but NOT a defect-type probe here
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_asymmetry_exists_and_grows_with_q_over_G():
    """Using (G + q) rather than G is what creates it. It is O(q/G)."""
    C6 = isotropic_stiffness(LAM, MU)
    net = _loop()
    d = _unit(torch.tensor([[0.3, 0.0, 1.0]]))
    a_small = float(qG_asymmetry(net, G_CU_111, d * 100.0, C6)[0])
    a_large = float(qG_asymmetry(net, G_CU_111, d * 1000.0, C6)[0])
    assert abs(a_small - 1.0) > 1e-3                       # genuinely present
    assert abs(a_large - 1.0) > abs(a_small - 1.0)         # grows with q/G


@pytest.mark.unit
def test_asymmetry_does_not_distinguish_interstitial_from_vacancy():
    """A NEGATIVE result, deliberately pinned.

    The implementation plan assumed the two loop characters would give opposite
    asymmetry. They do not. At first order the intensity is
    ``|(G+q).u~(q)|^2``; flipping the loop character flips the sign of ``u~``,
    which the modulus squared discards. The asymmetry here measures the geometry
    of the strain field, not the sign of the relaxation volume.

    Recovering the character needs a term LINEAR in the defect strength --
    interference between the long-range field (odd in q) and the localized core
    / Laue scattering from the extra or missing atoms (even in q). That core
    term is not modelled. Do not read defect character off this forward model.
    """
    C6 = isotropic_stiffness(LAM, MU)
    d = _unit(torch.tensor([[0.3, 0.0, 1.0]]))
    for qm in (100.0, 300.0, 1000.0):
        a_i = float(qG_asymmetry(_loop(+1.0), G_CU_111, d * qm, C6)[0])
        a_v = float(qG_asymmetry(_loop(-1.0), G_CU_111, d * qm, C6)[0])
        assert a_i == pytest.approx(a_v, rel=1e-12), (
            f"at q={qm} the asymmetry differs between characters; if this ever "
            "becomes true the core term has been added and the module docstring "
            "needs rewriting")
        assert abs(a_i - 1.0) > 1e-3, "the asymmetry must be non-trivial to matter"


@pytest.mark.unit
def test_flipping_character_leaves_the_intensity_unchanged():
    """The same statement at the level of a single point."""
    C6 = isotropic_stiffness(LAM, MU)
    q = _unit(torch.tensor([[0.3, 0.0, 1.0]])) * 200.0
    assert float(huang_intensity(_loop(+1.0), G_CU_111, q, C6)[0]) == pytest.approx(
        float(huang_intensity(_loop(-1.0), G_CU_111, q, C6)[0]), rel=1e-12)


# ---------------------------------------------------------------------------
# Bookkeeping and scope
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_open_lines_contribute_nothing_here_either():
    C6 = isotropic_stiffness(LAM, MU)
    line = straight_line(length_um=0.5, n_segments=16)
    res = huang_amplitude(line, G_CU_111, _unit(torch.tensor([[0.3, 0, 1.0]])) * 100.0, C6)
    assert res.n_loops == 0
    assert float(res.intensity.max()) == 0.0
    assert res.n_segments_ignored == 16


@pytest.mark.unit
def test_mixed_population_equals_the_loop_contribution():
    C6 = isotropic_stiffness(LAM, MU)
    q = _unit(torch.tensor([[0.3, 0.0, 1.0]])) * 200.0
    loop = _loop()
    both = combine([loop, straight_line(length_um=0.4, n_segments=12)])
    assert float(huang_intensity(both, G_CU_111, q, C6)[0]) == pytest.approx(
        float(huang_intensity(loop, G_CU_111, q, C6)[0]), rel=1e-12)


@pytest.mark.unit
def test_structure_factor_scales_the_intensity_quadratically():
    C6 = isotropic_stiffness(LAM, MU)
    q = _unit(torch.tensor([[0.3, 0.0, 1.0]])) * 200.0
    a = float(huang_intensity(_loop(), G_CU_111, q, C6, structure_factor=1.0)[0])
    b = float(huang_intensity(_loop(), G_CU_111, q, C6, structure_factor=3.0)[0])
    assert b / a == pytest.approx(9.0, rel=1e-10)


@pytest.mark.unit
def test_warns_when_q_is_no_longer_near_the_reflection():
    C6 = isotropic_stiffness(LAM, MU)
    q = _unit(torch.tensor([[1.0, 0.0, 0.0]])) * (0.5 * float(torch.linalg.norm(G_CU_111)))
    res = huang_amplitude(_loop(), G_CU_111, q, C6)
    assert any("no longer 'near'" in w for w in res.warnings)


@pytest.mark.unit
def test_refuses_the_bragg_peak_itself():
    C6 = isotropic_stiffness(LAM, MU)
    with pytest.raises(ValueError, match="singular|limit"):
        huang_intensity(_loop(), G_CU_111, torch.zeros(1, 3, dtype=torch.float64), C6)


@pytest.mark.autograd
def test_huang_is_differentiable_in_the_loop_geometry():
    C6 = isotropic_stiffness(LAM, MU)
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=16)
    net.nodes_um = net.nodes_um.clone().requires_grad_(True)
    q = _unit(torch.tensor([[0.3, 0.0, 1.0], [1.0, 0.2, 0.0]])) * 200.0
    huang_intensity(net, G_CU_111, q, C6).sum().backward()
    assert net.nodes_um.grad is not None
    assert torch.isfinite(net.nodes_um.grad).all()
    assert float(net.nodes_um.grad.abs().max()) > 0
