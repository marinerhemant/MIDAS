"""Gates on the real-space (Mura) distortion field.

The headline is
`test_elastic_and_total_differ_by_exactly_the_relaxation_volume`, which ties
this module to :mod:`midas_ddd.fourier`. The two kernels initially appeared to
disagree by a clean functional factor; they do not disagree at all. Mura returns
the ELASTIC distortion, the Fourier kernel works with the TOTAL, and the two
differ by the plastic eigendistortion on the cut surface, whose trace integrates
to exactly the relaxation volume. Pinning that relation is what makes the pair
trustworthy rather than merely individually plausible.

The other two gates matter because they test different components:
`test_long_segment_reproduces_the_infinite_screw` and `test_burgers_circuit`
exercise the rotational/deviatoric part, and neither of them would notice an
error in the dilatational part -- which is precisely where the cross-modality
relation bites.
"""
import math

import pytest
import torch

from midas_ddd import (
    find_loops,
    isotropic_stiffness,
    prismatic_loop,
    relaxation_volumes_um3,
    straight_line,
)
from midas_ddd.realspace import (
    SegmentDislocation,
    green_gradient_isotropic,
    lame_from_voigt,
    network_distortion,
    segment_dislocations,
)

B_CU_UM = 2.556e-4
LT = lambda v: torch.tensor(v, dtype=torch.float64)   # noqa: E731


def _segment(burgers, start, end, lam=100.0, mu=75.0, core=1e-7, n_quad=8):
    return SegmentDislocation(
        start_um=LT(start), end_um=LT(end), burgers_um=LT(burgers),
        lam=LT(lam), mu=LT(mu), core_radius_um=core, n_quad=n_quad)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_lame_from_voigt_is_exact_for_an_isotropic_stiffness():
    lam, mu = lame_from_voigt(isotropic_stiffness(100.0, 75.0))
    assert float(lam) == pytest.approx(100.0, rel=1e-12)
    assert float(mu) == pytest.approx(75.0, rel=1e-12)


@pytest.mark.unit
def test_zero_length_segment_gives_zero_field():
    sd = _segment([0, 0, B_CU_UM], [1.0, 0, 0], [1.0, 0, 0])
    B = sd.displacement_gradient(LT([[0.5, 0.5, 0.0]]))
    assert float(B.abs().max()) == 0.0


# ---------------------------------------------------------------------------
# The rotational part: infinite-line limit and Burgers circuit
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("r", [0.5, 1.0, 2.0])
def test_long_segment_reproduces_the_infinite_screw(r):
    """A 400 um screw segment must match b/(2 pi r) near its midpoint.

    This is also the test that exposed why fixed-order quadrature is not enough:
    the Mura integrand falls as 1/R^2, so eight Gauss nodes spread over a long
    segment miss the peak entirely and the answer came out ~1000x low and RISING
    with r. `_segment_distortion` subdivides into panels sized by the closest
    approach; without that this test fails by three orders of magnitude.
    """
    L = 400.0
    sd = _segment([0.0, 0.0, B_CU_UM], [0.0, 0.0, -L / 2], [0.0, 0.0, L / 2])
    B = sd.displacement_gradient(LT([[r, 0.0, 0.0]]))[0]
    assert float(B[2, 1]) == pytest.approx(B_CU_UM / (2 * math.pi * r), rel=2e-3)


@pytest.mark.unit
def test_burgers_circuit_around_a_segment_returns_b():
    """The convention-free check: the closed integral of beta.dl equals b."""
    L = 400.0
    sd = _segment([0.0, 0.0, B_CU_UM], [0.0, 0.0, -L / 2], [0.0, 0.0, L / 2])
    N, r = 2000, 1.0
    th = torch.linspace(0, 2 * math.pi, N + 1, dtype=torch.float64)[:-1]
    pts = torch.stack([r * torch.cos(th), r * torch.sin(th), torch.zeros_like(th)], -1)
    dl = torch.stack([-r * torch.sin(th), r * torch.cos(th),
                      torch.zeros_like(th)], -1) * (2 * math.pi / N)
    circ = torch.einsum("nij,nj->i", sd.displacement_gradient(pts), dl)
    assert float(circ[2]) == pytest.approx(B_CU_UM, rel=1e-3)
    assert float(circ[0].abs()) < 1e-12 * B_CU_UM
    assert float(circ[1].abs()) < 1e-12 * B_CU_UM


@pytest.mark.unit
def test_field_falls_off_with_distance():
    L = 400.0
    sd = _segment([0.0, 0.0, B_CU_UM], [0.0, 0.0, -L / 2], [0.0, 0.0, L / 2])
    mags = [float(sd.displacement_gradient(LT([[r, 0.0, 0.0]]))[0].abs().max())
            for r in (1.0, 2.0, 4.0)]
    assert mags[0] > mags[1] > mags[2]
    assert mags[0] / mags[1] == pytest.approx(2.0, rel=0.05)     # 1/r


# ---------------------------------------------------------------------------
# The dilatational part, and the tie to the Fourier kernel
# ---------------------------------------------------------------------------

def _trace_integral_over_ball(lam, mu, R_um=0.05, n=64, half_mult=5.0,
                              core_frac=0.2):
    """``int tr(beta^e) dV / dV`` over a ball centred on a prismatic loop."""
    net = prismatic_loop(radius_um=R_um, burgers=(0, 0, 1.0), n_segments=24,
                         cell_size_um=1.0)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    half = half_mult * R_um
    g = torch.linspace(-half, half, n, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], -1)
    beta = network_distortion(pts, net, isotropic_stiffness(lam, mu),
                              core_radius_um=core_frac * R_um)
    tr = torch.einsum("nii->n", beta)
    dv = (2 * half / (n - 1)) ** 3
    mask = torch.linalg.vector_norm(pts, dim=-1) <= half
    return float(tr[mask].sum() * dv) / dV


@pytest.mark.slow
@pytest.mark.parametrize("lam,mu", [(100.0, 75.0), (60.0, 40.0), (10.0, 90.0)])
def test_elastic_trace_integral_matches_the_closed_form(lam, mu):
    """``int tr(beta^e) dV = dV (2/3)(1-2nu)/(1-nu)`` over a ball.

    The only gate here that exercises the DILATATIONAL part of the field. The
    screw and Burgers-circuit tests are both blind to it.
    """
    nu = lam / (2.0 * (lam + mu))
    expected = (2.0 / 3.0) * (1.0 - 2.0 * nu) / (1.0 - nu)
    assert _trace_integral_over_ball(lam, mu) == pytest.approx(expected, abs=0.01)


@pytest.mark.slow
@pytest.mark.parametrize("lam,mu", [(100.0, 75.0), (60.0, 40.0), (10.0, 90.0)])
def test_elastic_and_total_traces_sum_to_one_TRACE_ONLY(lam, mu):
    """Eshelby's relaxation-volume partition, restated for this pair of kernels.

    ``dV = dV_inf + dV_I``: the plastic relaxation volume splits into the
    infinite-medium part and the image part. In units of dV,

        elastic (real space):  (2/3)(1-2nu)/(1-nu)
        total   (Fourier)   :  (1+nu)/(3(1-nu))     [= Eshelby factor 3(1-nu)/(1+nu), inverted]

    and they sum to 1. This is **not ours**: Lazar (2017), arXiv:1702.04981,
    Eqs. (77), (94), (95). Eshelby (1956) has the factor itself.

    READ THE NEXT TEST BEFORE TRUSTING THIS ONE. The sphere average of
    ``qhat_i qhat_j P_ij`` is ``tr(P)/3`` for ANY symmetric P, so this checks the
    TRACE of the dipole tensor and nothing else. It is blind to the anisotropy
    that `midas_ddd.fourier` is built around. It was briefly advertised in this
    repo as "the cross-modality gate"; it is not one, and
    `test_trace_only_check_is_blind_to_the_anisotropy` exists to keep it from
    being promoted back.
    """
    nu = lam / (2.0 * (lam + mu))
    elastic = _trace_integral_over_ball(lam, mu)
    total = (1.0 + nu) / (3.0 * (1.0 - nu))
    assert elastic + total == pytest.approx(1.0, abs=0.015)


@pytest.mark.unit
def test_trace_only_check_is_blind_to_the_anisotropy():
    """The negative control on the test above: a VOID passes it identically.

    Three dipole tensors of equal trace -- a true prismatic loop, an isotropic
    void, and a loop with its anisotropy sign-flipped -- all give the same sphere
    average, so all three pass the trace-only check. Distinguishing them is the
    entire point of the anisotropy claim, and that check cannot do it.
    """
    lam, mu = 100.0, 75.0
    kappa = lam / (lam + 2 * mu)
    n = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    I3 = torch.eye(3, dtype=torch.float64)
    nn = torch.outer(n, n)

    models = {
        "prismatic loop": kappa * I3 + (1 - kappa) * nn,
        "isotropic void": (kappa + (1 - kappa) / 3.0) * I3,
        "sign-flipped": (kappa + 2 * (1 - kappa) / 3.0) * I3 - (1 - kappa) * nn,
    }
    g = torch.Generator().manual_seed(0)
    d = torch.randn(200000, 3, generator=g, dtype=torch.float64)
    d = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True)

    avgs, in_plane = {}, {}
    for k, P in models.items():
        avgs[k] = float(torch.einsum("ki,kj,ij->k", d, d, P).mean())
        in_plane[k] = float(P[0, 0])

    # All three agree on the sphere average...
    for k in models:
        assert avgs[k] == pytest.approx(avgs["prismatic loop"], abs=2e-3)
    # ...while differing grossly in the direction-resolved response.
    assert abs(in_plane["prismatic loop"] - in_plane["isotropic void"]) > 0.15
    assert abs(in_plane["prismatic loop"] - in_plane["sign-flipped"]) > 0.35


@pytest.mark.slow
@pytest.mark.parametrize("radius_mult", [20, 50, 100])
def test_far_field_recovers_the_FULL_dipole_tensor(radius_mult):
    """THE real cross-kernel gate: direction-resolved, all six components of P.

    The Fourier kernel's whole anisotropy claim rests on the dipole tensor
    ``P_ij = C_ijkl b_k A_l``. This checks that the real-space Mura field's far
    field IS the elastic dipole field of that same P -- not merely that their
    traces agree. Replace the loop with a void, flip the sign of the anisotropy,
    or exaggerate it, and this fails; the trace-only check above passes all three.

    Convergence is 1/r as the higher multipoles die off: 5.8e-3 at 20R, 9.3e-4 at
    50R, 2.3e-4 at 100R.
    """
    lam, mu, R = 100.0, 75.0, 0.01
    net = prismatic_loop(radius_um=R, burgers=(0, 0, 1.0), n_segments=96,
                         cell_size_um=10.0)
    dV = float(relaxation_volumes_um3(net, find_loops(net))[0])
    b_um = net.burgers_um()[0]
    A = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) * dV / float(
        torch.linalg.norm(b_um))

    eye = torch.eye(3, dtype=torch.float64)
    C = (lam * torch.einsum("ij,kl->ijkl", eye, eye)
         + mu * (torch.einsum("ik,jl->ijkl", eye, eye)
                 + torch.einsum("il,jk->ijkl", eye, eye)))
    P = torch.einsum("ijkl,k,l->ij", C, b_um, A)
    # Sanity: a prismatic loop's dipole is dV*diag(lam, lam, lam+2mu).
    assert float(P[0, 0]) == pytest.approx(dV * lam, rel=1e-9)
    assert float(P[2, 2]) == pytest.approx(dV * (lam + 2 * mu), rel=1e-9)

    def beta_dipole(pts, h=1e-5):
        out = torch.zeros(pts.shape[0], 3, 3, dtype=torch.float64)
        lt, mt = torch.tensor(lam), torch.tensor(mu)
        for j in range(3):
            e = torch.zeros(3, dtype=torch.float64); e[j] = h
            dG = (green_gradient_isotropic(pts + e, lt, mt, 0.0)
                  - green_gradient_isotropic(pts - e, lt, mt, 0.0)) / (2 * h)
            out[:, :, j] = torch.einsum("nikl,kl->ni", dG, P)
        return out

    g = torch.Generator().manual_seed(0)
    d = torch.randn(40, 3, generator=g, dtype=torch.float64)
    d = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    pts = d * (radius_mult * R)

    mura = network_distortion(pts, net, isotropic_stiffness(lam, mu),
                              core_radius_um=0.05 * R)
    dip = beta_dipole(pts)
    rel = float((mura - dip).abs().max() / dip.abs().max())
    assert rel < 0.15 / radius_mult * 20, f"rel={rel:.3e} at {radius_mult}R"


@pytest.mark.unit
def test_symmetry_equivalent_variants_destroy_the_anisotropy_exactly():
    """A hard physical limit on the loop-vs-void discriminator.

    Real irradiation produces ALL symmetry-equivalent loop variants, not one
    aligned set. For any cubic family -- <111> Frank loops in FCC, <100> in
    BCC/Fe, <110> -- the variant average of ``n n`` is EXACTLY ``I/3``, so the
    population's dipole tensor is exactly isotropic and its angular contrast is
    exactly 1: indistinguishable from a void.

    The anisotropy is therefore only observable with a single crystal AND a
    variant-selected population (e.g. under applied stress). An unbiased
    population, or any powder, gives nothing. The demo figure uses aligned loops,
    which is the maximally favourable case.
    """
    import itertools

    lam, mu = 100.0, 75.0
    kappa = lam / (lam + 2 * mu)
    for family in ((1, 1, 1), (1, 0, 0), (1, 1, 0)):
        seen = set()
        for signs in itertools.product((-1, 1), repeat=3):
            for perm in itertools.permutations(range(3)):
                v = torch.tensor([family[perm[i]] * signs[i] for i in range(3)],
                                 dtype=torch.float64)
                if float(torch.linalg.norm(v)) == 0:
                    continue
                v = v / torch.linalg.norm(v)
                nz = torch.nonzero(v).flatten()
                if float(v[nz[0]]) < 0:
                    v = -v
                seen.add(tuple(round(float(x), 9) for x in v))
        V = torch.tensor(sorted(seen), dtype=torch.float64)
        M = torch.stack([torch.outer(v, v) for v in V]).mean(dim=0)
        assert torch.allclose(M, torch.eye(3, dtype=torch.float64) / 3.0,
                              atol=1e-12), f"family {family}"
        # ...hence exactly zero angular contrast for the averaged population.
        g = torch.Generator().manual_seed(0)
        d = torch.randn(20000, 3, generator=g, dtype=torch.float64)
        d = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        vals = kappa + (1 - kappa) * torch.einsum("ki,kj,ij->k", d, d, M)
        assert float(vals.max() / vals.min()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_open_lines_DO_contribute_in_real_space():
    """Unlike the Fourier kernel, an open network has a perfectly good field.

    The deformation population is invisible to small-angle scattering but images
    fine in DFXM. If this ever returns zero, the real-space path has inherited a
    scope limit that only applies to the scattering side.
    """
    net = straight_line(length_um=1.0, n_segments=8)
    assert find_loops(net) == []
    pts = LT([[0.3, 0.2, 0.0], [0.5, -0.1, 0.1]])
    beta = network_distortion(pts, net, isotropic_stiffness(100.0, 75.0))
    assert float(beta.abs().max()) > 0


@pytest.mark.unit
def test_segment_dislocations_covers_every_segment():
    net = prismatic_loop(radius_um=0.01, burgers=(0, 0, 1.0), n_segments=12)
    segs = segment_dislocations(net, isotropic_stiffness(100.0, 75.0))
    assert len(segs) == net.n_segments


@pytest.mark.unit
def test_network_distortion_is_the_sum_over_segments():
    """Linear elasticity: no per-segment normalisation may creep in."""
    net = prismatic_loop(radius_um=0.01, burgers=(0, 0, 1.0), n_segments=8)
    C6 = isotropic_stiffness(100.0, 75.0)
    pts = LT([[0.05, 0.02, 0.03]])
    total = network_distortion(pts, net, C6)
    by_hand = sum(sd.displacement_gradient(pts)
                  for sd in segment_dislocations(net, C6))
    assert torch.allclose(total, by_hand, rtol=1e-12)


@pytest.mark.unit
def test_chunking_does_not_change_the_answer():
    net = prismatic_loop(radius_um=0.01, burgers=(0, 0, 1.0), n_segments=8)
    C6 = isotropic_stiffness(100.0, 75.0)
    pts = LT([[0.05, 0.02, 0.03], [0.02, -0.04, 0.01], [0.0, 0.06, -0.02]])
    a = network_distortion(pts, net, C6, chunk=1)
    b = network_distortion(pts, net, C6, chunk=1024)
    assert torch.allclose(a, b, rtol=1e-12)


@pytest.mark.autograd
def test_distortion_is_differentiable_in_the_node_positions():
    net = prismatic_loop(radius_um=0.01, burgers=(0, 0, 1.0), n_segments=8)
    net.nodes_um = net.nodes_um.clone().requires_grad_(True)
    pts = LT([[0.05, 0.02, 0.03]])
    network_distortion(pts, net, isotropic_stiffness(100.0, 75.0)).abs().sum().backward()
    assert net.nodes_um.grad is not None
    assert torch.isfinite(net.nodes_um.grad).all()
    assert float(net.nodes_um.grad.abs().max()) > 0
