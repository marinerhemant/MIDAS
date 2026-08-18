"""Gates on the radial orientation kernel, and the decisive operator gate.

:func:`test_monte_carlo_pole_figure_matches_the_closed_form` is the one that
matters. It plants a kernel, **samples actual orientations from it**, rotates the
{hkl} normals into the sample frame, counts them into spherical caps, and compares
that histogram against what :mod:`midas_dt.gsh` predicts in closed form. The two
paths share only the geometry -- one is a Monte-Carlo count of discrete
orientations, the other an analytic harmonic expansion -- so agreement is evidence
about the operator rather than algebra restating itself.

It exists because the conjugation in :func:`kernel_to_gsh` is **invisible to a
symmetry test**: get it wrong and the result is still symmetric, still smooth, and
still wrong. Only a comparison against sampled orientations catches it.

One subtlety is baked in. The Monte-Carlo side counts normals inside a *finite*
cap, so it is cap-**averaged**. Evaluating the model pointwise compares two
different quantities and biases the fitted slope low on sharp features -- measured
0.969 pointwise against 1.008 cap-averaged. That residual was independent of the
truncation order (flat from L=12 to L=20) but shrank when the kernel was
broadened, which is the signature of the *comparison* rather than of the operator.
So the model is smoothed over the same cap the data are.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_dt.gsh import SymGSH, cubic_rotations
from midas_dt.texture_kernel import (
    halfwidth_deg,
    kappa_for_halfwidth,
    kernel_profile,
    kernel_to_gsh,
    radial_coeffs,
    sample_kernel,
    sample_kernel_angles,
)

LATTICES = {
    "Ti_alpha": (2.9505, 2.9505, 4.6826, 90.0, 90.0, 120.0),   # hcp, P6_3/mmc
    "Ti_omega": (4.6250, 4.6250, 2.8130, 90.0, 90.0, 120.0),   # P6/mmm
    "CeO2": (5.41165, 5.41165, 5.41165, 90.0, 90.0, 90.0),     # Fm-3m
}


# ------------------------------------------------------------ radial profile
def test_kernel_profile_is_non_negative():
    """The property the whole positivity argument rests on."""
    w = np.linspace(0.0, np.pi, 501)
    for kappa in (0.5, 2.0, 10.0, 100.0):
        assert (kernel_profile(w, kappa) >= 0.0).all()


def test_kernel_profile_peaks_at_zero_and_decays():
    w = np.linspace(0.0, np.pi, 201)
    p = kernel_profile(w, 20.0)
    assert p[0] == pytest.approx(1.0)
    assert np.all(np.diff(p) <= 1e-15)


@pytest.mark.parametrize("target", (5.0, 16.0, 40.0, 90.0))
def test_halfwidth_round_trips(target):
    assert halfwidth_deg(kappa_for_halfwidth(target)) == pytest.approx(target)


def test_halfwidth_is_where_the_profile_is_half():
    kappa = kappa_for_halfwidth(16.0)
    assert kernel_profile(np.radians(16.0), kappa) == pytest.approx(0.5)


def test_kappa_for_halfwidth_rejects_impossible_widths():
    for bad in (0.0, -3.0, 180.0, 200.0):
        with pytest.raises(ValueError, match="half-width"):
            kappa_for_halfwidth(bad)


def test_sharper_kernels_need_larger_kappa():
    assert kappa_for_halfwidth(5.0) > kappa_for_halfwidth(40.0)


# ------------------------------------------------------------ radial coeffs
def test_ahat_zero_is_exactly_one():
    """The kernel is Haar-normalised, so it shares the uniform ODF's l=0 term.

    This is what makes a pole-figure *ratio* against uniform a clean comparison:
    every normalisation convention cancels and only the shape is under test.
    """
    for kappa in (kappa_for_halfwidth(w) for w in (8.0, 16.0, 40.0)):
        assert radial_coeffs(8, kappa)[0] == pytest.approx(1.0, abs=1e-6)


def test_radial_coeffs_decay_per_degree_of_freedom():
    """``Ahat_l`` itself RISES before it falls; ``Ahat_l / (2l+1)`` decays.

    Worth pinning, because "the coefficients decay with l" is the natural
    assumption and it is false: ``chi_l(0) = 2l+1``, so a concentrated kernel's
    raw coefficients grow roughly as ``2l+1`` before the profile's width cuts
    them off. For a 16-degree kernel ``Ahat`` peaks at l=4, not l=0.
    """
    a = radial_coeffs(12, kappa_for_halfwidth(16.0))
    assert a[4] > a[0]                                    # rises first
    per_dof = a / (2 * np.arange(len(a)) + 1)
    assert np.all(np.diff(per_dof[:9]) < 0)               # but this decays
    assert per_dof[0] == pytest.approx(1.0, abs=1e-6)


def test_a_broader_kernel_decays_faster_in_l():
    """A wide kernel is band-limited low; a sharp one carries high orders."""
    sharp = radial_coeffs(12, kappa_for_halfwidth(8.0))
    broad = radial_coeffs(12, kappa_for_halfwidth(40.0))
    assert sharp[8] > broad[8]


def _symmetrised_peak(hw_deg: float, L: int) -> float:
    """Peak of the cubic-symmetrised kernel from its expansion truncated at L."""
    from midas_dt.gsh import invariant_basis
    grp = cubic_rotations()
    a = radial_coeffs(40, kappa_for_halfwidth(hw_deg))
    return sum(a[l] * (2 * l + 1) * invariant_basis(l, grp).shape[1]
               for l in range(0, L + 1))


def test_sharp_kernels_lose_amplitude_under_truncation_and_wide_ones_do_not():
    """The measured direction, which is the opposite of the intuitive one.

    A 40-degree kernel truncated at the operator's L=6 keeps ~all of its peak; an
    8-degree kernel keeps under 10 % and needs L~22. The mechanism is symmetry,
    not bandwidth: cubic has M(2) = 0, so the l=2 term -- the largest coefficient
    for any kernel sharper than ~30 degrees -- is annihilated, and L=6 leaves only
    l = 0, 4, 6.
    """
    full_8 = _symmetrised_peak(8.0, 40)
    assert _symmetrised_peak(8.0, 6) / full_8 < 0.10
    assert _symmetrised_peak(8.0, 22) / full_8 > 0.85

    full_40 = _symmetrised_peak(40.0, 40)
    assert _symmetrised_peak(40.0, 6) / full_40 > 0.99   # already band-limited

    full_16 = _symmetrised_peak(16.0, 40)
    assert 0.4 < _symmetrised_peak(16.0, 6) / full_16 < 0.55


def test_cubic_symmetry_annihilates_l2():
    """Why the truncation loss above is so severe for a sharp kernel."""
    from midas_dt.gsh import invariant_basis
    grp = cubic_rotations()
    assert invariant_basis(2, grp).shape[1] == 0
    a = radial_coeffs(12, kappa_for_halfwidth(16.0))
    assert a[2] == max(a[:3])          # and l=2 is the biggest term being lost


# ----------------------------------------------------------------- sampling
def test_sampled_angles_follow_the_haar_weighted_density():
    """Forgetting the Haar factor gives a texture sharper than kappa asks for."""
    kappa = kappa_for_halfwidth(20.0)
    rng = np.random.default_rng(0)
    ang = sample_kernel_angles(kappa, 200_000, rng)
    w = np.linspace(0.0, np.pi, 2001)
    dens = kernel_profile(w, kappa) * (1.0 - np.cos(w))
    expect_mean = float(np.trapezoid(w * dens, w) / np.trapezoid(dens, w))
    assert ang.mean() == pytest.approx(expect_mean, rel=0.02)
    assert ang.min() >= 0.0 and ang.max() <= np.pi


def test_sharper_kernel_samples_smaller_angles():
    rng = np.random.default_rng(1)
    sharp = sample_kernel_angles(kappa_for_halfwidth(5.0), 20_000, rng)
    broad = sample_kernel_angles(kappa_for_halfwidth(40.0), 20_000, rng)
    assert sharp.mean() < broad.mean()


def test_sample_kernel_is_symmetrised_on_the_right():
    """Right multiplication, because a forward model uses n_sample = g h_crystal.

    Left multiplication would symmetrise the SAMPLE frame -- a different physical
    statement -- and the sampled set would still look symmetric, so only a test
    that checks *which* frame is invariant can catch it. Here: applying a group
    element to the crystal direction must leave the sampled normal distribution
    unchanged.
    """
    rng = np.random.default_rng(2)
    centre = Rotation.from_rotvec(np.array([[0.2, -0.4, 0.5]]))
    group = cubic_rotations()
    mats = sample_kernel(centre, kappa_for_halfwidth(20.0), 4000, rng,
                         group=group).as_matrix()
    h = np.array([0.0, 0.0, 1.0])
    base = np.sort((mats @ h)[:, 2])
    for S in group[:4]:
        moved = np.sort((mats @ S.apply(h))[:, 2])
        # the two histograms must agree to sampling noise
        assert np.abs(np.percentile(base, [10, 50, 90])
                      - np.percentile(moved, [10, 50, 90])).max() < 0.08


def test_sample_kernel_accepts_matrices_for_the_group():
    rng = np.random.default_rng(3)
    centre = Rotation.from_rotvec([[0.0, 0.0, 0.1]])
    drawn = sample_kernel(centre, kappa_for_halfwidth(20.0), 50, rng,
                          group=cubic_rotations().as_matrix())
    assert len(drawn) == 50


# --------------------------------------------------------- kernel -> GSH
def test_kernel_to_gsh_rejects_a_too_short_coefficient_vector():
    basis = SymGSH(6, group=cubic_rotations())
    centre = Rotation.from_rotvec([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="ahat has"):
        kernel_to_gsh(basis, centre, radial_coeffs(4, 20.0))


def test_kernel_to_gsh_l0_term_is_the_uniform_one():
    """``Ahat_0 == 1``, so the kernel's l=0 coefficient equals the uniform ODF's.

    In this coefficient convention the uniform ODF is ``coef[0] = 1`` (see
    ``test_gsh.test_uniform_pole_density_equals_the_family_multiplicity``), so the
    kernel must land on exactly 1 regardless of where it is centred -- a kernel
    and the uniform distribution carry the same total weight.
    """
    basis = SymGSH(6, group=cubic_rotations())
    for rotvec in ([[0.0, 0.0, 0.0]], [[0.4, -0.2, 0.7]]):
        T = kernel_to_gsh(basis, Rotation.from_rotvec(rotvec),
                          radial_coeffs(6, kappa_for_halfwidth(16.0)))
        assert abs(T[0, 0]) == pytest.approx(1.0, rel=1e-6)


def test_uniform_odf_pole_figure_is_flat_for_every_symmetry():
    """A prerequisite for the MC gate: no spurious structure from the basis."""
    pg = pytest.importorskip("midas_hkls.point_group")
    for sg, name, hkl in [(194, "Ti_alpha", (1, 0, 0)),
                          (191, "Ti_omega", (0, 0, 1)),
                          (225, "CeO2", (1, 1, 1))]:
        lat = LATTICES[name]
        group = pg.proper_rotations_from_space_group(sg, lat)
        basis = SymGSH(6, group=group, lattice=lat)
        fam = basis.families(hkl)
        c = np.zeros(basis.n_coef, dtype=complex)
        c[0] = 1.0
        ys = Rotation.random(20, rng=np.random.default_rng(3)).as_matrix()[:, :, 0]
        vals = np.array([float((basis.pole_row(fam, y[None, :]) @ c).real)
                         for y in ys])
        assert np.ptp(vals) / abs(vals.mean()) < 1e-9, f"{name} is not flat"


# ------------------------------------------------- the decisive operator gate
def _cap_dirs(y, n, cos_cap, rs):
    """``n`` directions uniform inside the spherical cap around ``y``."""
    z = rs.uniform(cos_cap, 1.0, n)
    ph = rs.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    loc = np.stack([s * np.cos(ph), s * np.sin(ph), z], axis=1)
    a = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(a, y), float(a @ y)
    if np.linalg.norm(v) < 1e-12:
        return loc if c > 0 else -loc
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return loc @ (np.eye(3) + vx + vx @ vx / (1.0 + c)).T


def _monte_carlo_gate(sg, lattice_name, hkl, *, l_max=12, hw_deg=16.0,
                      cap_deg=7.0, n_dir=40, n_sample=150_000):
    """Returns (contrast, correlation, slope) of MC pole figure vs closed form."""
    pg = pytest.importorskip("midas_hkls.point_group")
    lat = LATTICES[lattice_name]
    group = pg.proper_rotations_from_space_group(sg, lat)
    basis = SymGSH(l_max, group=group, lattice=lat)
    fam = basis.families(hkl)
    kappa = kappa_for_halfwidth(hw_deg)

    ax = np.array([0.31, -0.66, 0.68])
    centre = Rotation.from_rotvec(
        (np.radians(41.0) * ax / np.linalg.norm(ax))[None, :])
    T = kernel_to_gsh(basis, centre, radial_coeffs(l_max, kappa))[:, 0]

    rng = np.random.default_rng(2024)
    mats = sample_kernel(centre, kappa, n_sample, rng, group=group).as_matrix()
    ns = np.einsum("kab,mb->kma", mats, fam).reshape(-1, 3)

    ys = Rotation.random(n_dir, rng=np.random.default_rng(11)).as_matrix()[:, :, 0]
    cos_cap = np.cos(np.radians(cap_deg))
    c_unif = np.zeros_like(T)
    c_unif[0] = T[0]

    def model_at(y):
        row = basis.pole_row(fam, y[None, :])
        return float((row @ T).real / (row @ c_unif).real)

    rs = np.random.default_rng(5)
    mc, model = [], []
    for y in ys:
        hit = int((ns @ y >= cos_cap).sum())
        mc.append(hit / (len(ns) * (1.0 - cos_cap) / 2.0))
        # smooth the MODEL over the same cap the data are counted in
        model.append(float(np.mean([model_at(d)
                                    for d in _cap_dirs(y, 160, cos_cap, rs)])))
    mc, model = np.array(mc), np.array(model)
    return (float(np.ptp(mc)), float(np.corrcoef(mc, model)[0, 1]),
            float(np.polyfit(model, mc, 1)[0]))


@pytest.mark.parametrize("sg,name,hkl", [
    (194, "Ti_alpha", (0, 0, 1)),      # hcp basal
    (194, "Ti_alpha", (1, 0, 0)),      # hcp prism -- the hexagonal metric
    (191, "Ti_omega", (0, 0, 1)),
    (225, "CeO2", (1, 1, 1)),          # cubic control, same code path
])
def test_monte_carlo_pole_figure_matches_the_closed_form(sg, name, hkl):
    contrast, corr, slope = _monte_carlo_gate(sg, name, hkl)
    assert contrast > 1.0, f"vacuous test: pole-figure contrast only {contrast:.3f}"
    assert corr > 0.997, f"SHAPE MISMATCH corr {corr:.5f} -- a convention is wrong"
    assert 0.93 < slope < 1.07, f"AMPLITUDE MISMATCH slope {slope:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("sg,name,hkl", [(194, "Ti_alpha", (1, 0, 0)),
                                         (225, "CeO2", (1, 1, 1))])
def test_monte_carlo_gate_at_full_precision(sg, name, hkl):
    """The same gate with 8x the samples: corr must reach 0.999."""
    contrast, corr, slope = _monte_carlo_gate(sg, name, hkl, n_sample=1_200_000)
    assert corr > 0.999, f"corr {corr:.6f}"
    assert 0.97 < slope < 1.03, f"slope {slope:.3f}"
