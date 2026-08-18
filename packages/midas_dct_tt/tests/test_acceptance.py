import math
import pytest
import torch

"""Phase 1 tests: the objective-free (na = 0) acceptance.

Includes the measured design trade-off between missing cone and longitudinal
acceptance -- the reason "just use a low-theta reflection" is not free advice.
"""
import math

import pytest
import torch

from midas_dct_tt import (
    attach_uniform_field,
    sphere_grain,
    tt_alignment,
    tt_resolution,
    tt_resolution_widths,
)

DT = torch.float64
HKL = (1, 1, 1)
LAMBDA_HEXM = 0.172979      # ~71.7 keV, 1-ID / HEXM
LAMBDA_ESRF = 0.7293        # ~17 keV


def _alignment(wavelength_A):
    g = attach_uniform_field(sphere_grain(3.0, spacing_um=1.5))
    return tt_alignment(g.field.reference_G(HKL), wavelength_A)


# ---------------------------------------------------------------------------
# na = 0
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_widths_reduce_to_the_divergence_and_bandwidth_forms():
    q, tt, div_v, div_h, eps = 3.0, 15.0, 0.53e-3, 0.31e-3, 1.4e-4
    w = tt_resolution_widths(q, two_theta_deg=tt, div_v=div_v, div_h=div_h,
                             energy_spread=eps)
    theta = math.radians(tt / 2.0)
    assert abs(w["sigma_rock"] - 0.5 * q * div_v) < 1e-18
    assert abs(w["sigma_roll"] - 0.5 * q * div_h / math.sin(theta)) / w["sigma_roll"] < 1e-6
    want_par = 0.5 * q * math.sqrt(4 * eps ** 2 + (1 / math.tan(theta)) ** 2 * div_v ** 2)
    assert abs(w["sigma_par"] - want_par) / want_par < 1e-6


@pytest.mark.unit
def test_rock_width_is_independent_of_theta():
    """sigma_rock = |Q| div_v / 2 carries no theta -- a useful invariant."""
    a = tt_resolution_widths(3.0, two_theta_deg=5.0)
    b = tt_resolution_widths(3.0, two_theta_deg=40.0)
    assert abs(a["sigma_rock"] - b["sigma_rock"]) < 1e-18


# ---------------------------------------------------------------------------
# the ResolutionFunction wrapper
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_acceptance_is_centred_on_the_aligned_G():
    """Weight exactly 1 at q_nom: a perfect crystal pays nothing."""
    al = _alignment(LAMBDA_HEXM)
    res = tt_resolution(al)
    assert abs(float(res.weight(al.G_lab)) - 1.0) < 1e-15


@pytest.mark.unit
def test_acceptance_falls_off_as_a_gaussian_along_q():
    al = _alignment(LAMBDA_HEXM)
    res = tt_resolution(al)
    q_hat = al.G_lab / torch.linalg.vector_norm(al.G_lab)
    for k in (0.5, 1.0, 2.0):
        w = float(res.weight(al.G_lab + k * float(res.sigma_par) * q_hat))
        assert abs(w - math.exp(-0.5 * k ** 2)) < 1e-12


@pytest.mark.unit
def test_geometric_mean_preserves_the_transverse_acceptance_area():
    """The documented approximation: isotropic sigma_perp with the same area."""
    al = _alignment(LAMBDA_HEXM)
    res = tt_resolution(al)
    w = tt_resolution_widths(float(torch.linalg.vector_norm(al.G_lab)),
                             two_theta_deg=2.0 * float(al.theta_deg))
    assert abs(float(res.sigma_perp) ** 2 - w["sigma_rock"] * w["sigma_roll"]) < 1e-15


@pytest.mark.unit
def test_anisotropy_is_reported_and_large():
    """rock << roll is a thin plate, not a disc; the caller is told the ratio."""
    al = _alignment(LAMBDA_HEXM)
    res = tt_resolution(al)
    assert res.anisotropy > 10.0


# ---------------------------------------------------------------------------
# the design trade-off (measured 2026-08-03)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_low_theta_shrinks_the_missing_cone_but_widens_the_strain_acceptance():
    """The central tension in reflection selection, as numbers.

    Going from ~17 keV to ~71.7 keV on fcc (111):

    * missing cone 10.0 deg -> 2.4 deg      (tomographic coverage improves 4x)
    * sigma_par/|Q|  1.5e-3 -> 6.4e-3       (strain contrast degrades 4x)

    Both scale as cot(theta), so they trade off directly and neither reflection
    choice dominates. This is why reflection selection deserves a
    Fisher-information treatment rather than a rule of thumb.
    """
    hexm, esrf = _alignment(LAMBDA_HEXM), _alignment(LAMBDA_ESRF)
    cone_hexm, cone_esrf = float(hexm.theta_deg), float(esrf.theta_deg)
    rel_hexm = float(tt_resolution(hexm).sigma_par) / float(hexm.G_lab.norm())
    rel_esrf = float(tt_resolution(esrf).sigma_par) / float(esrf.G_lab.norm())

    assert cone_hexm < cone_esrf                 # better tomography at high energy
    assert rel_hexm > rel_esrf                   # worse strain contrast, same move
    # Both ratios track cot(theta), so they cancel to ~1.
    ratio = (cone_esrf / cone_hexm) / (rel_hexm / rel_esrf)
    assert abs(ratio - 1.0) < 0.1


@pytest.mark.unit
def test_longitudinal_acceptance_is_divergence_limited_not_bandwidth_limited():
    """At HEXM energies the cot(theta) divergence term swamps the bandwidth term.

    sigma_par^2 ~ (|Q|/2)^2 [4 eps^2 + cot^2(theta) div_v^2]; at theta = 2.4 deg
    the second term is ~2000x the first. Consequence worth acting on: a narrower
    monochromator does **not** sharpen TT's strain sensitivity at high energy --
    collimating the beam does.
    """
    theta = math.radians(float(_alignment(LAMBDA_HEXM).theta_deg))
    eps, div_v = 1.4e-4, 0.53e-3
    bandwidth_term = 4 * eps ** 2
    divergence_term = (1 / math.tan(theta)) ** 2 * div_v ** 2
    assert divergence_term / bandwidth_term > 1000.0

    # And the model agrees: halving the bandwidth barely moves sigma_par.
    q, tt = 3.0, 2.0 * math.degrees(theta)
    base = tt_resolution_widths(q, two_theta_deg=tt, energy_spread=eps)["sigma_par"]
    tight = tt_resolution_widths(q, two_theta_deg=tt, energy_spread=eps / 2)["sigma_par"]
    assert abs(tight / base - 1.0) < 1e-3


# --- anisotropic (thin-plate) acceptance -----------------------------------
def _aniso_setup(hkl=(2, 0, 0)):
    from midas_dct_tt import attach_uniform_field, sphere_grain, tt_alignment
    from midas_dct_tt.acceptance import tt_resolution_aniso
    g = attach_uniform_field(sphere_grain(3.0, spacing_um=1.0))
    al = tt_alignment(g.field.reference_G(hkl), 0.172979)
    return g, al, tt_resolution_aniso(al)


def test_aniso_frame_is_orthonormal_and_physically_assigned():
    """roll must be OUT of the scattering plane, rock IN it -- that assignment is
    forced by which rotation breaks the Bragg condition, not chosen."""
    _, al, r = _aniso_setup()
    e_par, e_rock, e_roll = r.frame()
    M = torch.stack([e_par, e_rock, e_roll])
    assert torch.allclose(M @ M.T, torch.eye(3, dtype=M.dtype), atol=1e-12)
    assert abs(float(e_roll @ al.k_in)) < 1e-9          # roll is out of plane
    khat = al.k_in / torch.linalg.vector_norm(al.k_in)
    assert abs(float(torch.linalg.det(torch.stack([khat, e_par, e_rock])))) < 1e-12


def test_aniso_preserves_transverse_area_of_the_isotropic_form():
    from midas_dct_tt.acceptance import tt_resolution
    _, al, r = _aniso_setup()
    assert r.sigma_perp_equivalent == pytest.approx(
        float(tt_resolution(al).sigma_perp), rel=1e-12)


def test_aniso_reduces_to_isotropic_when_widths_are_equal():
    from midas_dfxm.resolution import aligned_resolution
    from midas_dct_tt.acceptance import AnisotropicTTResolution
    _, al, r = _aniso_setup()
    s = 3e-3
    a = AnisotropicTTResolution(q_nom=al.G_lab, k_in=al.k_in, sigma_par=r.sigma_par,
                                sigma_rock=s, sigma_roll=s)
    iso = aligned_resolution(al.G_lab, sigma_par=r.sigma_par, sigma_perp=s)
    torch.manual_seed(0)
    Q = al.G_lab + torch.randn(64, 3, dtype=al.G_lab.dtype) * 1e-3
    assert torch.allclose(a.weight(Q), iso.weight(Q), atol=1e-12)
    assert a.anisotropy == pytest.approx(1.0)


def test_acceptance_is_a_thin_plate_not_a_disc():
    """Measured 15-24x at 71.7 keV; the isotropic form is not a small correction."""
    for hkl in [(1, 1, 1), (2, 0, 0), (2, 2, 0)]:
        _, _, r = _aniso_setup(hkl)
        assert r.anisotropy > 10.0, f"{hkl}: anisotropy only {r.anisotropy}"


def test_backscatter_is_an_error_not_a_fallback():
    from midas_dct_tt.acceptance import AnisotropicTTResolution
    q = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    r = AnisotropicTTResolution(q_nom=q, k_in=q * 5.0)
    with pytest.raises(ValueError, match="scattering plane is undefined"):
        r.frame()


def test_aniso_weight_is_differentiable():
    _, al, r = _aniso_setup()
    Q = (al.G_lab + 1e-4).clone().requires_grad_(True)
    r.weight(Q).sum().backward()
    assert Q.grad is not None and torch.isfinite(Q.grad).all()
    assert float(Q.grad.abs().max()) > 0



def _moments(F, *, model, grain, als, hkls, res, psi, det, supersample=1):
    """Raw image moments per reflection and psi: ``(m0 (R,S), m1 (R,S,2))``.

    Inlined rather than imported from ``dev/paper/runs/_afit.py``: ``dev/`` is
    private by policy and is not in the distribution, so the import worked only
    on the one machine that has it and raised ModuleNotFoundError everywhere
    else -- including CI, where it failed the 0.1.0 release. Mass and first
    moment are linear in the image, bounded and division-free, which is why this
    is the quantity the test compares.
    """
    import torch as _torch
    from midas_dct_tt import local_Q, topograph_stack
    n = grain.n_voxels
    m0s, m1s = [], []
    for G0, al, hkl, r in zip([grain.field.reference_G(h) for h in hkls],
                              als, hkls, res):
        Q = local_Q(F, G0, model=model).expand(n, 3)
        stack = topograph_stack(grain, al, psi, detector=det, hkl=hkl,
                                resolution=r, supersample=supersample,
                                Q_sample=Q)
        nu, nv = stack.shape[-2:]
        iu = _torch.arange(nu, dtype=stack.dtype).view(1, nu, 1)
        iv = _torch.arange(nv, dtype=stack.dtype).view(1, 1, nv)
        m0s.append(stack.sum(dim=(1, 2)))
        m1s.append(_torch.stack([(stack * iu).sum(dim=(1, 2)),
                                 (stack * iv).sum(dim=(1, 2))], dim=-1))
    return _torch.stack(m0s), _torch.stack(m1s)


def test_anisotropy_is_what_makes_intensity_carry_psi_information():
    """The load-bearing test.

    An isotropic transverse acceptance is invariant under the TT rotation, so the
    intensity channel collapses to one scalar per reflection and a rotation-only
    fit to it is degenerate. The true plate breaks that: the strained transverse
    offset sweeps between the tight and loose axes, giving a 180-deg-period
    oscillation whose amplitude carries the transverse strain.
    """
    from midas_dct_tt import (PlaneDetector, psi_scan, tt_alignment,
                              tt_resolution)
    from midas_dct_tt.acceptance import tt_resolution_aniso

    g, _, _ = _aniso_setup()
    SET = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    als = [tt_alignment(g.field.reference_G(h), 0.172979) for h in SET]
    psi = psi_scan(48)
    det = PlaneDetector(pixel_um=1.0, shape=(128, 128), distance_um=5000.0)
    I3 = torch.eye(3, dtype=torch.float64)
    E = torch.tensor([[1.0, 0.4, -0.2], [0.4, -0.6, 0.3], [-0.2, 0.3, -0.4]],
                     dtype=torch.float64)
    E = 0.5 * (E + E.T); E = E - I3 * torch.trace(E) / 3
    E = E / torch.linalg.matrix_norm(E)

    def spread(res):
        o0, _ = _moments(I3 + 1e-3 * E, model="exact", supersample=3, grain=g,
                        als=als, hkls=SET, res=res, psi=psi, det=det)
        return float(((o0.max(dim=1).values - o0.min(dim=1).values)
                      / o0.mean(dim=1).abs()).max()), o0

    iso, _ = spread([tt_resolution(a) for a in als])
    ani, o0 = spread([tt_resolution_aniso(a) for a in als])
    assert iso < 1e-9, f"isotropic should be psi-invariant, got {iso:.2e}"
    assert ani > 0.1, f"anisotropic should modulate strongly, got {ani:.2e}"

    # The signal must live ENTIRELY in the 2-psi harmonic. Asserting a
    # correlation with cos(2 psi) specifically would be wrong: the PHASE depends
    # on the strain orientation, and that assertion fails for the same strain
    # rotated 60/62/65/90 deg about lab z (measured +0.300/+0.198/-0.003/+0.278).
    # Fit the exact 3-term form instead; the residual is the invariant.
    p = torch.deg2rad(psi.to(torch.float64))
    A = torch.stack([torch.ones_like(p), torch.cos(2 * p), torch.sin(2 * p)], dim=1)
    y = torch.log(o0[0])
    coef = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
    resid = float((y.unsqueeze(1) - A @ coef).norm())
    assert resid < 1e-9 * max(float(y.std()), 1e-30) * len(p), \
        f"log m0 is not exactly a + b cos2psi + c sin2psi (residual {resid:.2e})"
    assert float(coef[1:].norm()) > 0.01, "no 2-psi modulation at all"
    for k in (1, 3):
        B = torch.stack([torch.ones_like(p), torch.cos(k * p), torch.sin(k * p)], dim=1)
        c = torch.linalg.lstsq(B, y.unsqueeze(1)).solution
        assert float(c[1:].norm()) < 1e-9, f"unexpected harmonic {k}"


# --- orientation-sampling lower bound ---------------------------------------
def test_rock_bound_is_independent_of_the_reflection():
    """|Q| cancels: rock = div_v/2, a property of the BEAM not the reflection.

    So a higher-index reflection buys longitudinal resolution but nothing in
    orientation about the rocking axis -- a real planning consequence.
    """
    from midas_dct_tt import orientation_resolution_deg
    vals = []
    for hkl in [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2)]:
        _, al, _ = _aniso_setup(hkl)
        vals.append(orientation_resolution_deg(al)["rock"])
    assert max(vals) - min(vals) < 1e-4 * vals[0], f"rock bound varied: {vals}"
    # div_v, NOT div_v/2 -- the /2 came from the superseded na=0 widths
    assert vals[0] == pytest.approx(math.degrees(0.53e-3), rel=1e-3)


def test_roll_is_unbounded_not_merely_wide():
    """A rotation about k_in leaves rho invariant exactly, so a fixed TT setting
    places NO bound on orientation about the beam."""
    from midas_dct_tt import orientation_resolution_deg
    for th in (0.5, 2.0, 10.0):
        r = orientation_resolution_deg(theta_deg=th)
        assert r["roll"] == float("inf")
        assert r["ratio"] == float("inf")


def test_curvature_scale_is_finite_and_reflection_dependent():
    """The number sometimes mistaken for a roll width: a validity radius for the
    planar-slab picture, not an orientation bound."""
    from midas_dct_tt import orientation_resolution_deg
    vals = []
    for hkl in [(1, 1, 1), (4, 0, 0)]:
        _, al, _ = _aniso_setup(hkl)
        vals.append(orientation_resolution_deg(al)["curvature_scale"])
    assert all(1.0 < v < 20.0 for v in vals)
    assert vals[0] > vals[1] * 1.3, "curvature scale must vary with reflection"


def test_orientation_bound_rejects_ambiguous_anisotropic_input():
    from midas_dct_tt import orientation_resolution_deg
    with pytest.raises(ValueError, match="azimuth-dependent"):
        orientation_resolution_deg(theta_deg=2.0, div_h=0.31e-3, div_v=0.53e-3)


def test_orientation_bound_requires_a_theta():
    from midas_dct_tt import orientation_resolution_deg
    with pytest.raises(ValueError, match="alignment .*or an explicit theta_deg"):
        orientation_resolution_deg()


# --- objective-free acceptance (the physically correct one for a bare detector) --
def _of_setup(hkl=(2, 0, 0)):
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    g, al, r = _aniso_setup(hkl)
    return g, al, r, ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab)


def test_objective_free_is_centred_on_the_ewald_condition():
    """rho = |k_in+Q|^2 - |k_in|^2 must vanish at the aligned reflection."""
    _, al, _, a = _of_setup()
    assert abs(float(a.rho(al.G_lab.unsqueeze(0)))) < 1e-12
    assert float(a.weight(al.G_lab.unsqueeze(0))) == pytest.approx(1.0, abs=1e-12)


def test_objective_free_widths_match_an_independent_derivation():
    """Cross-check against a brute-force quadrature over the beam (verify agent).

    Reproduced here from a completely different route -- the exact rho and a
    bisection on weight = exp(-1/2) -- so a regression in either would break this.
    """
    _, al, r, a = _of_setup()
    e_par, e_rock, e_roll = r.frame()
    ref = {"rock": 1.83213e-03, "par": 3.80388e-02, "roll": 3.64608e-01}
    got = {"rock": a.effective_sigma(al.G_lab, e_rock),
           "par": a.effective_sigma(al.G_lab, e_par),
           "roll": a.effective_sigma(al.G_lab, e_roll)}
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-4), f"{k}: {got[k]:.7e} vs {v:.7e}"


def test_true_anisotropy_is_two_hundred_not_twenty():
    """The na=0 model understates the roll:rock ratio by ~10x."""
    _, al, r, a = _of_setup()
    e_par, e_rock, e_roll = r.frame()
    ratio = a.effective_sigma(al.G_lab, e_roll) / a.effective_sigma(al.G_lab, e_rock)
    assert ratio > 150, f"true anisotropy only {ratio:.0f}"
    assert r.anisotropy < 30, "the na=0 model should be the understating one"


def test_roll_is_unconstrained_to_first_order():
    """k_out . e_roll = 0 exactly, so rho picks up only +u^2 out of the plane.

    This is WHY roll is ~200x wider, and it is a statement about the geometry, not
    about the beam.
    """
    _, al, r, a = _of_setup()
    _, _, e_roll = r.frame()
    k_out = al.k_in + al.G_lab
    assert abs(float(k_out @ e_roll)) < 1e-9
    for u in (1e-4, 1e-3, 1e-2):
        d = float(a.rho((al.G_lab + u * e_roll).unsqueeze(0)))
        assert d == pytest.approx(u ** 2, rel=1e-6), "roll response is not quadratic"


def test_acceptance_gradient_is_along_k_out_a_slab_not_an_ellipsoid():
    """grad_Q rho = 2 k_out, so the acceptance is a slab perpendicular to k_out.

    An axis-aligned diagonal Gaussian in the (par, rock, roll) frame cannot
    represent that, which is the structural reason AnisotropicTTResolution is
    still wrong even with correct widths.
    """
    _, al, _, a = _of_setup()
    Q = al.G_lab.clone().requires_grad_(True)
    a.rho(Q.unsqueeze(0)).backward()
    k_out = (al.k_in + al.G_lab).to(Q.grad.dtype)
    g = Q.grad / torch.linalg.vector_norm(Q.grad)
    assert torch.allclose(g, k_out / torch.linalg.vector_norm(k_out), atol=1e-10)


def test_objective_free_weight_is_differentiable():
    _, al, _, a = _of_setup()
    Q = (al.G_lab + 1e-5).clone().requires_grad_(True)
    a.weight(Q.unsqueeze(0)).sum().backward()
    assert Q.grad is not None and torch.isfinite(Q.grad).all()


def test_objective_free_weight_carries_the_density_prefactor():
    """<delta(rho)> has a 1/sigma_rho(Q) factor, and sigma_rho VARIES with Q.

    Dropping it is not an overall constant -- it tilts the profile and made the
    par width 1.11% too wide. Tolerance is 1e-4, not the 2% that hid the defect.
    """
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, r = _aniso_setup()
    a = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab)
    e_par, e_rock, e_roll = r.frame()
    exact = {"rock": 1.832131e-03, "par": 3.803882e-02, "roll": 3.646085e-01}
    got = {"rock": a.effective_sigma(al.G_lab, e_rock),
           "par": a.effective_sigma(al.G_lab, e_par),
           "roll": a.effective_sigma(al.G_lab, e_roll)}
    for k, v in exact.items():
        assert got[k] == pytest.approx(v, rel=1e-4), f"{k}: {got[k]:.7e} vs {v:.7e}"


def test_roll_rotation_about_the_beam_is_exactly_unconstrained():
    """A lattice rotation about k_in preserves |Q| AND k_in.Q, so rho is invariant
    EXACTLY at any angle -- not merely to first order. The finite 'roll width' is
    Ewald-sphere curvature measured along a straight line, not an acceptance."""
    import math

    from midas_dfxm.conventions import rotation_matrix
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, _ = _aniso_setup()
    a = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab)
    khat = al.k_in / torch.linalg.vector_norm(al.k_in)
    for phi in (0.03, 6.0, 45.0, 90.0):
        Q = rotation_matrix(khat, torch.as_tensor(phi, dtype=torch.float64)) @ al.G_lab
        assert abs(float(a.rho(Q.unsqueeze(0)))) < 1e-9
        assert float(a.weight(Q.unsqueeze(0))) == pytest.approx(1.0, abs=1e-9)


def test_objective_free_requires_q_nom():
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, _ = _aniso_setup()
    a = ObjectiveFreeAcceptance(k_in=al.k_in)
    with pytest.raises(ValueError, match="q_nom is required"):
        a.weight(al.G_lab.unsqueeze(0))


# --- finite coherent domain -------------------------------------------------
def test_domain_broadening_adds_in_quadrature():
    """Closed form: sigma_rock_eff = hypot(sigma_rock_beam, 2*pi/L)."""
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, r = _aniso_setup()
    _, e_rock, _ = r.frame()
    base = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab)
    s0 = base.effective_sigma(al.G_lab, e_rock)
    for L in (3.0, 1.0, 0.5):
        a = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab, domain_size_um=L)
        got = a.effective_sigma(al.G_lab, e_rock)
        assert got == pytest.approx(math.hypot(s0, 2 * math.pi / (L * 1e4)), rel=2e-3)


def test_domain_broadening_matters_below_a_micron():
    """Ignorable at 3 um, 1.21x at 0.5 um -- and sub-micron is exactly the scale a
    per-voxel inverse claims to resolve."""
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, r = _aniso_setup()
    _, e_rock, _ = r.frame()
    s0 = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab).effective_sigma(al.G_lab, e_rock)
    w = {}
    for L in (3.0, 1.0, 0.5):
        a = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab, domain_size_um=L)
        w[L] = a.effective_sigma(al.G_lab, e_rock) / s0
    assert w[3.0] < 1.02
    assert 1.04 < w[1.0] < 1.09
    assert w[0.5] > 1.15
    assert w[3.0] < w[1.0] < w[0.5]


def test_domain_does_not_make_roll_finite():
    """A rotation about k_in maps the broadened reciprocal point onto itself, so
    rho stays exactly invariant however small the domain."""
    from midas_dfxm.conventions import rotation_matrix
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, _ = _aniso_setup()
    a = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab, domain_size_um=0.2)
    khat = al.k_in / torch.linalg.vector_norm(al.k_in)
    for phi in (5.0, 45.0, 90.0):
        Q = rotation_matrix(khat, torch.as_tensor(phi, dtype=torch.float64)) @ al.G_lab
        assert float(a.weight(Q.unsqueeze(0))) == pytest.approx(1.0, abs=1e-9)


def test_domain_size_must_be_positive():
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, _ = _aniso_setup()
    with pytest.raises(ValueError, match="domain_size_um must be > 0"):
        ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab,
                                domain_size_um=-1.0).sigma_domain()


# --- monochromator energy/divergence coupling -------------------------------
def test_mono_coupling_swings_the_narrow_axis_by_a_factor_of_three_seven():
    """After a vertically-diffracting mono, eps and div_v are locked by
    d(eps) = -cot(theta_M) d(alpha_v). At Si(111)/71.7 keV cot = 36.2, so the Qx
    term dominates and sigma_rock scales 2.73x or 0.73x by sense.

    Independently predicted by adversarial verification before implementation.
    """
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, r = _aniso_setup()
    _, e_rock, _ = r.frame()
    base = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab)
    s0 = base.effective_sigma(al.G_lab, e_rock)
    got = {}
    for sense in (+1.0, -1.0):
        a = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab,
                                    mono_bragg_deg=1.58, mono_sense=sense)
        got[sense] = a.effective_sigma(al.G_lab, e_rock) / s0
    assert got[+1.0] == pytest.approx(2.73, rel=0.02)
    assert got[-1.0] == pytest.approx(0.73, rel=0.02)
    assert got[+1.0] / got[-1.0] == pytest.approx(3.75, rel=0.05)


def test_no_mono_is_the_independent_default():
    from midas_dct_tt.acceptance import ObjectiveFreeAcceptance
    _, al, _ = _aniso_setup()
    a = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab)
    assert a.mono_bragg_deg is None
    assert a.sigma_domain() == 0.0
