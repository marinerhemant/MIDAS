"""Gates on the Fourier kernel.

The primary anchor is the closed-form small-q limit for a prismatic loop under
isotropic elasticity,

    q . u~(q -> 0) = i dV [ kappa + (1 - kappa) (n.qhat)^2 ],
    dV = b . A,  kappa = lambda / (lambda + 2 mu),

which is independent of this implementation: it is derived from the tensor
expression, not from the code under test. `test_closed_form_matches_the_full_
tensor_expression` re-derives it from scratch with numpy so the gate itself
cannot rot.

Two negative controls carry the most weight:

* `test_transverse_gauge_would_destroy_the_signal` -- the shortcut that avoids
  choosing a cut surface returns exactly zero along the loop normal. It is the
  reason the kernel does the surface integral rather than the cheap line
  integral, and without this test that decision looks arbitrary.
* `test_open_line_contributes_nothing` -- a line encloses no area, so it has no
  cut surface and `u_tilde` (the displacement field of CLOSED loops) must ignore
  it. That is a statement about `u_tilde`, not about scattering: the small-angle
  amplitude of open lines comes from the line-integral form, gated in
  `test_line_term.py`.
"""
import math

import numpy as np
import pytest
import torch

from midas_ddd import (
    combine,
    find_loops,
    polygon_area_exact_um2,
    prismatic_loop,
    relaxation_volumes_um3,
    straight_line,
)
from midas_ddd.fourier import (
    acoustic_tensor,
    loop_vertices,
    isotropic_stiffness,
    loop_cut_surface,
    prismatic_loop_small_q_limit,
    q_dot_u_tilde,
    surface_form_factor,
    u_tilde,
)
from midas_ddd.elasticity import _voigt_to_tensor

LAM, MU = 100.0, 75.0
KAPPA = LAM / (LAM + 2 * MU)
B_CU_A = 2.556
B_CU_UM = B_CU_A * 1e-4


def _unit(v):
    v = torch.as_tensor(v, dtype=torch.float64)
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)


DIRS = _unit(torch.tensor([
    [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0],
    [1.0, 0, 1.0], [1.0, 1.0, 0], [1.0, 1.0, 1.0], [2.0, -1.0, 0.5],
], dtype=torch.float64))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_quadrature_weights_sum_to_exactly_one():
    """The q -> 0 limit inherits this sum exactly. The published 10-digit
    coefficients sum to 0.9999999996, which showed up as a constant 4e-10
    error against the gate -- q-independent, hence a normalisation bug rather
    than a physics one.

    THE BOUND IS PLATFORM TOLERANCE, NOT A DEFECT BEING ADMITTED. It was
    ``abs=1e-16``, which is 0.45 ULP at 1.0 -- i.e. it demanded that a
    seven-term float sum be BIT-exact. That is a platform assertion wearing a
    tolerance, and it failed on CPython 3.11 (1.0000000000000002) while
    passing on 3.12.

    Tightening it further was tried first and did not work: switching to
    :func:`math.fsum`, which is correctly rounded and order-independent, still
    returned 1.0000000000000002 on the 3.11 runner. Since fsum depends only on
    the input values, ``_TRI_BARY`` itself must differ there -- and it is
    computed at import by :func:`_normalised_tri_rule`, not stored as
    literals. On this machine the builtin sum, fsum and exact ``Fraction``
    arithmetic all give exactly 1.0 from the same source, so WHY the runner's
    values differ is not explained. The bound is set so the answer does not
    matter.

    Eight ULP is 1.78e-15. The normalisation bug this test exists to catch --
    published 10-digit coefficients summing to 0.9999999996, a q-independent
    4e-10 error against the closed-form gate -- is 1.8 million ULP, so the
    bound still detects it with a margin of about 225000x. Nothing that would
    have been caught before is missed now.
    """
    import math
    from midas_ddd.fourier import _TRI_BARY
    tol = 8 * math.ulp(1.0)                      # 1.78e-15
    assert math.fsum(r[3] for r in _TRI_BARY) == pytest.approx(1.0, abs=tol)
    for l0, l1, l2, _ in _TRI_BARY:
        assert math.fsum((l0, l1, l2)) == pytest.approx(1.0, abs=tol)


@pytest.mark.unit
def test_acoustic_tensor_is_symmetric_and_scales_as_q_squared():
    C4 = _voigt_to_tensor(isotropic_stiffness(LAM, MU))
    q = torch.tensor([[0.3, -0.7, 1.1]], dtype=torch.float64)
    K = acoustic_tensor(C4, q)[0]
    assert torch.allclose(K, K.T, atol=1e-12)
    K2 = acoustic_tensor(C4, 2.0 * q)[0]
    assert torch.allclose(K2, 4.0 * K, rtol=1e-12)


@pytest.mark.unit
def test_surface_form_factor_at_zero_q_is_the_exact_area_vector():
    """This is what makes the dV gate exact rather than quadrature-limited."""
    R, n = 0.005, 32
    net = prismatic_loop(radius_um=R, burgers=(0, 0, 1), n_segments=n)
    v0, v1, v2 = loop_cut_surface(net, find_loops(net)[0])
    q0 = torch.zeros(1, 3, dtype=torch.float64)
    I = surface_form_factor(v0, v1, v2, q0)[0]
    assert float(I.imag.abs().max()) < 1e-18
    assert float(I.real[2]) == pytest.approx(polygon_area_exact_um2(R, n), rel=1e-14)
    assert float(I.real[0]) == pytest.approx(0.0, abs=1e-18)


@pytest.mark.unit
def test_cut_surface_area_vectors_sum_to_the_loop_area():
    R, n = 0.01, 24
    net = prismatic_loop(radius_um=R, burgers=(1, 1, 1), n_segments=n)
    v0, v1, v2 = loop_cut_surface(net, find_loops(net)[0])
    A = (0.5 * torch.linalg.cross(v1 - v0, v2 - v0)).sum(dim=0)
    assert float(torch.linalg.norm(A)) == pytest.approx(
        polygon_area_exact_um2(R, n), rel=1e-12)


# ---------------------------------------------------------------------------
# The closed-form gate
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_closed_form_matches_the_full_tensor_expression():
    """Re-derive the gate independently, in numpy, from the tensor expression.

    Keeps the gate honest: if it only ever agreed with the torch kernel, the two
    could be wrong together.
    """
    rng = np.random.default_rng(0)
    d = np.eye(3)
    worst = 0.0
    for lam, mu in [(100.0, 75.0), (60.0, 40.0), (200.0, 30.0), (10.0, 90.0)]:
        C = (lam * np.einsum('ij,kl->ijkl', d, d)
             + mu * (np.einsum('ik,jl->ijkl', d, d) + np.einsum('il,jk->ijkl', d, d)))
        kappa = lam / (lam + 2 * mu)
        n = np.array([1.0, 2.0, -1.0]); n /= np.linalg.norm(n)
        b, A = n * B_CU_UM, n * 7.85e-5
        dV = float(b @ A)
        for _ in range(100):
            qh = rng.normal(size=3); qh /= np.linalg.norm(qh)
            q = 1e-6 * qh
            K = np.einsum('ijkl,j,l->ik', C, q, q)
            u = 1j * np.einsum('ki,ijmn,j,m,n->k', np.linalg.inv(K), C, q, b, A)
            pred = dV * (kappa + (1 - kappa) * float(n @ qh) ** 2)
            worst = max(worst, abs(complex(q @ u).imag - pred) / abs(dV))
    assert worst < 1e-12, f"closed form deviates from the tensor expression by {worst:.2e}"


@pytest.mark.unit
def test_kernel_reproduces_the_small_q_limit_in_every_direction():
    """The primary correctness anchor."""
    R, nseg = 0.005, 64
    normal = (0.0, 0.0, 1.0)
    net = prismatic_loop(radius_um=R, burgers=normal, n_segments=nseg)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    C6 = isotropic_stiffness(LAM, MU)

    q = DIRS * 1e-2                     # deep in the small-q limit
    got = q_dot_u_tilde(net, q, C6)
    pred = prismatic_loop_small_q_limit(dV, normal, DIRS, LAM, MU)
    assert torch.allclose(got, pred, rtol=1e-8, atol=0.0), (
        f"max rel err {float((got - pred).abs().max() / pred.abs().max()):.3e}")


@pytest.mark.unit
def test_along_the_loop_normal_the_limit_is_exact_at_every_q():
    """For q along the normal, q.r = 0 everywhere on the (planar) cut surface,
    so exp(-i q.r) = 1 and there is no finite-q correction at all. Machine
    precision at q = 10 1/um is therefore the expected result, not luck."""
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=64)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    C6 = isotropic_stiffness(LAM, MU)
    nhat = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    for qm in (1e-2, 1e0, 1e1):
        got = q_dot_u_tilde(net, nhat * qm, C6)[0]
        pred = prismatic_loop_small_q_limit(dV, (0, 0, 1), nhat, LAM, MU)[0]
        assert abs(got - pred) / abs(pred) < 1e-13, f"q={qm}"


@pytest.mark.unit
def test_finite_q_departure_scales_as_q_squared():
    """Off the normal there IS a finite-q correction, and it must be O(q^2).
    A different power would mean the cut surface or the phase is wrong."""
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=64)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    C6 = isotropic_stiffness(LAM, MU)
    d = _unit(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))
    pred = prismatic_loop_small_q_limit(dV, (0, 0, 1), d, LAM, MU)[0]
    errs = []
    for qm in (1e-1, 1e0):
        got = q_dot_u_tilde(net, d * qm, C6)[0]
        errs.append(abs(got - pred) / abs(pred))
    assert errs[1] / errs[0] == pytest.approx(100.0, rel=0.05)


@pytest.mark.unit
def test_in_plane_to_normal_contrast_ratio_is_kappa():
    """The loop-versus-void discriminator, as a number.

    A void is isotropic; a loop scatters kappa*dV in its own plane and dV along
    its normal. If this ratio ever comes back as 1, the kernel has lost the
    anisotropy and loops have become indistinguishable from voids.
    """
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=64)
    C6 = isotropic_stiffness(LAM, MU)
    q = torch.tensor([[1.0, 0, 0], [0, 0, 1.0]], dtype=torch.float64) * 1e-2
    got = q_dot_u_tilde(net, q, C6)
    assert float(got[0].imag / got[1].imag) == pytest.approx(KAPPA, rel=1e-6)
    assert KAPPA < 0.99, "kappa ~ 1 would make this test vacuous"


@pytest.mark.unit
@pytest.mark.parametrize("lam,mu", [(60.0, 40.0), (200.0, 30.0), (10.0, 90.0)])
def test_gate_holds_for_other_moduli(lam, mu):
    net = prismatic_loop(radius_um=0.004, burgers=(1, 1, 0), n_segments=48)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    got = q_dot_u_tilde(net, DIRS * 1e-2, isotropic_stiffness(lam, mu))
    pred = prismatic_loop_small_q_limit(dV, (1, 1, 0), DIRS, lam, mu)
    assert torch.allclose(got, pred, rtol=1e-7, atol=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("R", [0.002, 0.005, 0.02])
def test_amplitude_scales_with_relaxation_volume(R):
    """|q.u~| must track dV = b.A, i.e. go as R^2."""
    C6 = isotropic_stiffness(LAM, MU)
    n = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    net = prismatic_loop(radius_um=R, burgers=(0, 0, 1), n_segments=64)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    got = q_dot_u_tilde(net, n * 1e-2, C6)[0]
    assert float(got.imag) == pytest.approx(dV, rel=1e-12)


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_transverse_gauge_would_destroy_the_signal():
    """Why the kernel integrates a cut surface instead of taking the cheap
    Stokes line integral.

    Stokes fixes only the part of I perpendicular to q. Dropping the parallel
    part -- the tempting way to avoid choosing a cut surface -- zeroes the
    amplitude along the loop normal, which is where a loop scatters MOST
    strongly. This test pins that down so nobody 'simplifies' the kernel into
    silence.
    """
    d = np.eye(3)
    C = (LAM * np.einsum('ij,kl->ijkl', d, d)
         + MU * (np.einsum('ik,jl->ijkl', d, d) + np.einsum('il,jk->ijkl', d, d)))
    nhat = np.array([0.0, 0.0, 1.0])
    b, A = nhat * B_CU_UM, nhat * 7.85e-5
    q = 1e-6 * nhat
    K = np.einsum('ijkl,j,l->ik', C, q, q)

    def qu(I):
        u = 1j * np.einsum('ki,ijmn,j,m,n->k', np.linalg.inv(K), C, q, b, I)
        return abs(complex(q @ u))

    full = qu(A)
    transverse = qu(A - nhat * (nhat @ A))
    assert full == pytest.approx(abs(float(b @ A)), rel=1e-10)
    assert transverse < 1e-12 * full, (
        "the transverse projection is supposed to lose the entire signal here")


@pytest.mark.unit
def test_open_line_contributes_nothing():
    """A line encloses no area, so `u_tilde` -- which needs a cut surface -- ignores
    it and says how much line it ignored. Small-angle scattering from lines is the
    line-integral form's job (`test_line_term.py`), not this kernel's."""
    C6 = isotropic_stiffness(LAM, MU)
    line = straight_line(length_um=0.5, n_segments=32)
    res = u_tilde(line, DIRS * 1e-2, C6)
    assert res.n_loops == 0
    assert float(res.u_tilde.abs().max()) == 0.0
    assert res.n_segments_ignored == 32
    assert any("no cut surface" in w or "open segment" in w for w in res.warnings)


@pytest.mark.unit
def test_mixed_population_gives_exactly_the_loop_contribution():
    """A loop plus a line in one box must equal the loop alone -- and must SAY
    how much line length it dropped, rather than quietly returning a number
    that looks like it accounted for everything."""
    C6 = isotropic_stiffness(LAM, MU)
    loop = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=48)
    both = combine([loop, straight_line(length_um=0.5, n_segments=32)])
    q = DIRS * 1e-2
    assert torch.allclose(q_dot_u_tilde(both, q, C6), q_dot_u_tilde(loop, q, C6),
                          rtol=1e-12)
    res = u_tilde(both, q, C6)
    assert res.n_loops == 1 and res.n_segments_ignored == 32
    assert res.line_length_ignored_um == pytest.approx(0.5, rel=1e-9)


@pytest.mark.unit
def test_vacancy_loop_flips_the_sign():
    """Interstitial and vacancy loops must differ by a sign. This is the
    interstitial/vacancy typing signal; losing it collapses the two."""
    C6 = isotropic_stiffness(LAM, MU)
    kw = dict(radius_um=0.005, burgers=(0, 0, 1), n_segments=48)
    q = DIRS * 1e-2
    a = q_dot_u_tilde(prismatic_loop(**kw, burgers_scale_b=+1.0), q, C6)
    b = q_dot_u_tilde(prismatic_loop(**kw, burgers_scale_b=-1.0), q, C6)
    assert torch.allclose(a, -b, rtol=1e-12)
    assert float(a.abs().max()) > 0


@pytest.mark.unit
def test_two_loops_superpose_linearly():
    """Linear elasticity: amplitudes add. Guards against any accidental
    per-loop normalisation."""
    C6 = isotropic_stiffness(LAM, MU)
    a = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=32,
                       center_um=(0.1, 0, 0), cell_size_um=2.0)
    b = prismatic_loop(radius_um=0.003, burgers=(1, 1, 0), n_segments=32,
                       center_um=(-0.1, 0, 0), cell_size_um=2.0)
    q = DIRS * 1e-2
    both = combine([a, b])
    assert torch.allclose(q_dot_u_tilde(both, q, C6),
                          q_dot_u_tilde(a, q, C6) + q_dot_u_tilde(b, q, C6),
                          rtol=1e-10)


@pytest.mark.unit
def test_translating_a_loop_only_changes_the_phase():
    """|q.u~| is translation invariant; the phase carries the position. If the
    magnitude moved, the cut surface would be anchored wrongly."""
    C6 = isotropic_stiffness(LAM, MU)
    q = DIRS * 0.5
    a = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=48,
                       center_um=(0, 0, 0), cell_size_um=20.0)
    b = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=48,
                       center_um=(3.0, -1.0, 2.0), cell_size_um=20.0)
    assert torch.allclose(q_dot_u_tilde(a, q, C6).abs(),
                          q_dot_u_tilde(b, q, C6).abs(), rtol=1e-10)


@pytest.mark.unit
def test_u_tilde_refuses_q_equals_zero():
    C6 = isotropic_stiffness(LAM, MU)
    net = prismatic_loop(radius_um=0.005, n_segments=12)
    with pytest.raises(ValueError, match="singular|limit"):
        u_tilde(net, torch.zeros(1, 3, dtype=torch.float64), C6)


@pytest.mark.unit
def test_quadrature_warning_fires_when_q_outruns_the_triangulation():
    C6 = isotropic_stiffness(LAM, MU)
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=6)
    res = u_tilde(net, _unit(torch.tensor([[1.0, 1.0, 0.0]])) * 5e3, C6)
    assert res.quadrature_qh > 0.5
    assert any("quadrature" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_gradient_flows_to_node_positions():
    C6 = isotropic_stiffness(LAM, MU)
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=16)
    net.nodes_um = net.nodes_um.clone().requires_grad_(True)
    amp = q_dot_u_tilde(net, DIRS * 1e-1, C6).abs().sum()
    amp.backward()
    assert net.nodes_um.grad is not None
    assert torch.isfinite(net.nodes_um.grad).all()
    assert float(net.nodes_um.grad.abs().max()) > 0


@pytest.mark.autograd
def test_gradient_wrt_loop_radius_matches_finite_difference():
    """dV goes as R^2, so d|q.u~|/dR should be ~ 2 dV / R along the normal."""
    C6 = isotropic_stiffness(LAM, MU)
    nhat = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)

    def amp(R):
        net = prismatic_loop(radius_um=float(R), burgers=(0, 0, 1), n_segments=64)
        return float(q_dot_u_tilde(net, nhat * 1e-2, C6)[0].imag)

    R0, h = 0.005, 1e-7
    fd = (amp(R0 + h) - amp(R0 - h)) / (2 * h)
    analytic = 2.0 * amp(R0) / R0            # exact for an R^2 law
    assert fd == pytest.approx(analytic, rel=1e-5)


@pytest.mark.autograd
def test_gradient_flows_to_the_elastic_constants():
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=16)
    lam = torch.tensor(LAM, dtype=torch.float64, requires_grad=True)
    C6 = isotropic_stiffness(lam, MU)
    q_dot_u_tilde(net, DIRS * 1e-1, C6).abs().sum().backward()
    assert lam.grad is not None and torch.isfinite(lam.grad)
    # kappa depends on lambda, so the in-plane amplitude must actually move.
    assert float(lam.grad.abs()) > 0


# ---------------------------------------------------------------------------
# Radial subdivision of the cut surface
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_plain_fan_is_badly_wrong_at_high_qR():
    """Why `loop_cut_surface` subdivides radially.

    A fan from the centroid has an edge of length R running out to each vertex,
    so the quadrature parameter |q|h is set by the loop RADIUS however finely the
    polygon is discretised -- subdividing the circumference does nothing for it.
    At qR = 10, in the middle of a real SAXS detector's range, a plain fan is
    ~20 % wrong. This test exists so nobody 'simplifies' the rings away.
    """
    C6 = isotropic_stiffness(LAM, MU)
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=64)
    d = _unit(torch.tensor([[1.0, 0.0, 0.3]], dtype=torch.float64))
    q = d * 2000.0                                   # qR = 10
    ref = q_dot_u_tilde(net, q, C6, n_rings=128)[0]
    fan = q_dot_u_tilde(net, q, C6, n_rings=1)[0]
    assert abs(fan - ref) / abs(ref) > 0.1, "the fan is supposed to fail here"


@pytest.mark.unit
@pytest.mark.parametrize("qR", [2.0, 5.0, 10.0, 20.0])
def test_auto_rings_converge_the_cut_surface_quadrature(qR):
    """Automatic radial subdivision must track a heavily refined reference."""
    C6 = isotropic_stiffness(LAM, MU)
    R = 0.005
    net = prismatic_loop(radius_um=R, burgers=(0, 0, 1), n_segments=64)
    d = _unit(torch.tensor([[1.0, 0.0, 0.3]], dtype=torch.float64))
    q = d * (qR / R)
    ref = q_dot_u_tilde(net, q, C6, n_rings=128)[0]
    auto = q_dot_u_tilde(net, q, C6)[0]
    assert abs(auto - ref) / abs(ref) < 1e-6


@pytest.mark.unit
def test_subdivision_preserves_the_cut_surface_area():
    """Rings change the triangulation, never the surface it triangulates."""
    net = prismatic_loop(radius_um=0.005, burgers=(1, 1, 1), n_segments=24)
    lp = find_loops(net)[0]
    areas = []
    for nr in (1, 3, 8, 17):
        v0, v1, v2 = loop_cut_surface(net, lp, n_rings=nr)
        areas.append((0.5 * torch.linalg.cross(v1 - v0, v2 - v0)).sum(dim=0))
    for A in areas[1:]:
        assert torch.allclose(A, areas[0], rtol=1e-12, atol=1e-20)
    assert float(torch.linalg.norm(areas[0])) == pytest.approx(
        polygon_area_exact_um2(0.005, 24), rel=1e-12)


@pytest.mark.unit
def test_polygon_shape_error_is_separate_from_quadrature_error():
    """A 64-gon is not a circle, and that difference is a MODELLING choice.

    Pinned so the two are never confused: with the quadrature converged to 1e-8,
    a 64-gon and a 256-gon still differ by ~1 % at qR = 10. That is the polygon,
    not the integrator -- raising `n_rings` will never remove it, and only
    `n_segments` will.
    """
    C6 = isotropic_stiffness(LAM, MU)
    d = _unit(torch.tensor([[1.0, 0.0, 0.3]], dtype=torch.float64))
    q = d * 2000.0                                   # qR = 10
    coarse = q_dot_u_tilde(
        prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=64), q, C6)[0]
    fine = q_dot_u_tilde(
        prismatic_loop(radius_um=0.005, burgers=(0, 0, 1), n_segments=256), q, C6)[0]
    shape_diff = abs(fine - coarse) / abs(coarse)
    assert 1e-3 < shape_diff < 1e-1


# ---------------------------------------------------------------------------
# Scope of the closed forms, and what the anisotropy does NOT buy you
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_general_law_handles_a_shear_loop_where_the_prismatic_form_is_zero():
    """A pure shear loop (b perpendicular to A) has dV = 0 but scatters.

    `prismatic_loop_small_q_limit` would return exactly 0 here -- not imprecise,
    qualitatively wrong -- which is why it now refuses dV = 0 and why
    `loop_small_q_limit` exists.
    """
    from midas_ddd.fourier import loop_small_q_limit
    R = 0.005
    net = prismatic_loop(radius_um=R, burgers=(0, 0, 1.0), n_segments=64)
    net.burgers_b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).expand(
        net.n_segments, 3).clone()
    A = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * polygon_area_exact_um2(R, 64)
    b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) * B_CU_UM
    assert float(b @ A) == pytest.approx(0.0, abs=1e-30)

    d = _unit(torch.tensor([[1.0, 0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64))
    got = q_dot_u_tilde(net, d * 1e-2, isotropic_stiffness(LAM, MU))
    gen = loop_small_q_limit(b, A, d, LAM, MU)
    assert torch.allclose(got, gen, rtol=5e-3)
    assert float(got.abs().max()) > 0, "a shear loop must scatter"

    with pytest.raises(ValueError, match="shear loop"):
        prismatic_loop_small_q_limit(0.0, (0, 0, 1), d, LAM, MU)


@pytest.mark.unit
def test_general_law_reduces_to_the_prismatic_form_when_b_is_parallel_to_A():
    from midas_ddd.fourier import loop_small_q_limit
    R, nseg = 0.005, 64
    A = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * polygon_area_exact_um2(R, nseg)
    b = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * B_CU_UM
    gen = loop_small_q_limit(b, A, DIRS, LAM, MU)
    pris = prismatic_loop_small_q_limit(float(b @ A), (0, 0, 1), DIRS, LAM, MU)
    assert torch.allclose(gen, pris, rtol=1e-12)


@pytest.mark.unit
def test_a_uniaxial_plate_precipitate_is_indistinguishable_from_a_loop():
    """The refutation of the loop-vs-void framing, kept as a test.

    At q -> 0 the amplitude depends only on (dV, n). Any defect with the same
    uniaxial dipole tensor -- a coherent plate precipitate, a lenticular void --
    gives an identical signature. This discriminates uniaxial dipoles from
    isotropic ones, NOT loops from voids.
    """
    from midas_ddd.fourier import loop_small_q_limit
    R, nseg = 0.005, 64
    A = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * polygon_area_exact_um2(R, nseg)
    b = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * B_CU_UM
    dV = float(b @ A)

    loop = loop_small_q_limit(b, A, DIRS, LAM, MU)
    # A platelet inclusion with the SAME relaxation volume and habit normal:
    # P = dV diag(lam, lam, lam+2mu), identical to the loop's.
    plate = prismatic_loop_small_q_limit(dV, (0, 0, 1), DIRS, LAM, MU)
    assert torch.allclose(loop, plate, rtol=1e-12)


@pytest.mark.unit
def test_randomly_oriented_loops_are_exactly_degenerate_with_a_void():
    """The practical kill on the discriminator.

    Averaging over isotropically distributed loop normals gives the isotropic
    void value (1+nu)/(3(1-nu)) exactly, so an unbiased loop population is
    indistinguishable from a void population at q -> 0.
    """
    nu = LAM / (2.0 * (LAM + MU))
    void_value = (1.0 + nu) / (3.0 * (1.0 - nu))
    g = torch.Generator().manual_seed(0)
    normals = _unit(torch.randn(4000, 3, generator=g, dtype=torch.float64))
    for qd in _unit(torch.tensor([[0.0, 0, 1.0], [1.0, 1.0, 0.3]], dtype=torch.float64)):
        vals = prismatic_loop_small_q_limit(
            1.0, (0, 0, 1), qd.unsqueeze(0), LAM, MU)  # placeholder shape
        # average kappa + (1-kappa)(n.qhat)^2 over the normal distribution
        kappa = LAM / (LAM + 2 * MU)
        avg = float((kappa + (1 - kappa) * (normals @ qd) ** 2).mean())
        assert avg == pytest.approx(void_value, abs=0.02)


# ---------------------------------------------------------------------------
# The Laue term: the TOTAL small-angle amplitude (Ehrhart/Trinkaus/Larson 1982)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_total_small_angle_amplitude_vanishes_along_the_loop_normal():
    """THE correction. `q_dot_u_tilde` (distortion only) is MAXIMAL along the
    normal at dV; the total, with the Laue term, is exactly ZERO there.

    Ehrhart, Trinkaus & Larson, Phys. Rev. B 25 (1982) 834, Eq. (8b):
    the amplitude carries a factor `q x A~(q)`, which vanishes for q || A.
    Shipping the distortion term alone as "the small-angle scattering of a loop"
    gets the answer maximally wrong in exactly the direction it claims is
    strongest.
    """
    from midas_ddd.fourier import small_angle_amplitude
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=96)
    C6 = isotropic_stiffness(LAM, MU)
    nrm = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])

    total = small_angle_amplitude(net, nrm * 1e-2, C6)[0]
    strain = q_dot_u_tilde(net, nrm * 1e-2, C6)[0]
    assert abs(complex(total)) < 1e-12 * dV, "the total must vanish along the normal"
    assert abs(complex(strain)) == pytest.approx(dV, rel=1e-6), \
        "...while the distortion term alone is at its maximum there"


@pytest.mark.unit
def test_total_small_angle_matches_the_closed_form_in_every_direction():
    """`i dV (1 - kappa) sin^2(theta)` -- zero on the normal, peak in the plane."""
    from midas_ddd.fourier import (prismatic_loop_small_q_limit_total,
                                   small_angle_amplitude)
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=96)
    C6 = isotropic_stiffness(LAM, MU)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    got = small_angle_amplitude(net, DIRS * 1e-2, C6)
    pred = prismatic_loop_small_q_limit_total(dV, (0, 0, 1), DIRS, LAM, MU)
    assert torch.allclose(got.abs(), pred.abs(), rtol=1e-6)


@pytest.mark.unit
def test_distortion_and_total_sum_to_dV():
    """The two laws are complementary: kappa + (1-kappa)cos^2 and (1-kappa)sin^2
    add to 1 identically. Pins the relation between the two entry points."""
    from midas_ddd.fourier import (prismatic_loop_small_q_limit_total,
                                   small_angle_amplitude)
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=96)
    C6 = isotropic_stiffness(LAM, MU)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    tot = small_angle_amplitude(net, DIRS * 1e-2, C6).abs()
    strain = q_dot_u_tilde(net, DIRS * 1e-2, C6).abs()
    assert torch.allclose(tot + strain, torch.full_like(tot, dV), rtol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("lam,mu", [(60.0, 40.0), (200.0, 30.0), (10.0, 90.0)])
def test_total_law_holds_for_other_moduli(lam, mu):
    from midas_ddd.fourier import (prismatic_loop_small_q_limit_total,
                                   small_angle_amplitude)
    net = prismatic_loop(radius_um=0.004, burgers=(1, 1, 0), n_segments=64)
    C6 = isotropic_stiffness(lam, mu)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    got = small_angle_amplitude(net, DIRS * 1e-2, C6).abs()
    pred = prismatic_loop_small_q_limit_total(dV, (1, 1, 0), DIRS, lam, mu).abs()
    assert torch.allclose(got, pred, rtol=1e-5)


@pytest.mark.unit
def test_variant_averaging_still_kills_the_null():
    """The null is a better signature than the old 2.5:1 contrast -- but it does
    not survive an unbiased population any better. <sin^2> = 2/3, isotropic."""
    kappa = LAM / (LAM + 2 * MU)
    g = torch.Generator().manual_seed(0)
    normals = _unit(torch.randn(4000, 3, generator=g, dtype=torch.float64))
    for qd in _unit(torch.tensor([[0.0, 0, 1.0], [1.0, 1.0, 0.3]], dtype=torch.float64)):
        avg = float(((1 - kappa) * (1 - (normals @ qd) ** 2)).mean())
        assert avg == pytest.approx((1 - kappa) * 2.0 / 3.0, abs=0.02)
