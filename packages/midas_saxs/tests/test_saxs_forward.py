"""Gates on the SAXS forward model.

The headline is `test_radial_average_cannot_tell_them_apart_but_azimuth_can`.
Everything else supports it. A dislocation loop and a void are both "a small
thing that scatters at small angle"; what separates them is that the void is
isotropic and the loop is not, and that difference survives only on the 2-D
frame. If that test ever fails, the reason for building a detector simulator
rather than an I(q) model has evaporated.
"""
import math

import pytest
import torch

from midas_saxs import (
    SAXSGeometry,
    SpherePopulation,
    azimuthal_profile,
    loop_equivalent_sphere_radius_A,
    pixel_to_q,
    radial_average,
    simulate_frame,
    sphere_form_factor_squared,
    void_intensity,
)

ddd = pytest.importorskip("midas_ddd", reason="dislocation source needs midas-ddd")
from midas_ddd import combine, isotropic_stiffness, prismatic_loop, straight_line  # noqa: E402
from midas_saxs.strain_source import (  # noqa: E402
    electron_density_per_A3,
    loop_amplitude,
    loop_intensity,
)

LAM, MU = 100.0, 75.0
KAPPA = LAM / (LAM + 2 * MU)
B_CU_A = 2.556
# Cromer-Mann gives f(0) = 28.9859 for Cu, not exactly Z = 29 -- the analytic
# sum rule is approximate by ~0.05 %. Derive the density rather than assuming Z.
RHO_CU = electron_density_per_A3(["Cu"], [4], 3.615 ** 3)     # 2.4543 e/A^3


def _geom(n=128, beamstop=8):
    return SAXSGeometry(lsd_um=2.0e6, bcy_px=n // 2, bcz_px=n // 2, px_um=75.0,
                        wavelength_A=0.7293, n_pix_y=n, n_pix_z=n,
                        beamstop_radius_px=beamstop)


def _aligned_loops(n_loops=12, R_um=0.005, n_seg=24, seed=0):
    """A population of prismatic loops, all normals along +z."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_loops):
        c = (torch.rand(3, generator=g, dtype=torch.float64) * 2 - 1) * 0.4
        out.append(prismatic_loop(radius_um=R_um, burgers=(0, 0, 1.0),
                                  n_segments=n_seg, center_um=tuple(c.tolist()),
                                  cell_size_um=1.0))
    return combine(out, cell_size_um=1.0)


# ---------------------------------------------------------------------------
# Migration: the primitives that moved out of midas_pdf
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_midas_pdf_reexports_the_same_objects():
    """midas_pdf.saxs must forward to midas_saxs, not carry a second copy."""
    pdf_saxs = pytest.importorskip("midas_pdf.saxs")
    import midas_saxs

    for name in ("sphere_form_factor_squared", "ellipsoid_form_factor_squared",
                 "cylinder_form_factor_squared", "percus_yevick_S",
                 "SAXSModel", "lognormal_quadrature_nodes",
                 "guinier_fit", "porod_fit",
                 "core_shell_sphere_form_factor_squared"):
        assert getattr(pdf_saxs, name) is getattr(midas_saxs, name), (
            f"{name} in midas_pdf.saxs is not the midas_saxs object -- there must "
            "be exactly one definition")


@pytest.mark.unit
def test_deep_import_path_still_resolves():
    """`from midas_pdf.saxs.form_factors import ...` was a real historical path."""
    from midas_pdf.saxs.form_factors import sphere_form_factor_squared as pdf_sff
    import midas_saxs
    assert pdf_sff is midas_saxs.sphere_form_factor_squared


@pytest.mark.unit
def test_sphere_form_factor_carries_the_volume_squared():
    """`sphere_form_factor_squared` is V^2 |F_hat|^2, NOT the normalised |F_hat|^2.

    Worth pinning: `SpherePopulation.intensity` multiplies it by
    ``n * delta_rho^2`` and nothing else, which is only correct because the
    volume already lives in here. Multiplying by V^2 again -- the natural
    mistake if you assume P(0) = 1 -- would inflate I(0) by 2.7e11 for a 50 A
    sphere.
    """
    R = 50.0
    V = 4.0 / 3.0 * math.pi * R ** 3
    q = torch.tensor([1e-8], dtype=torch.float64)
    assert float(sphere_form_factor_squared(q, R)[0]) == pytest.approx(V ** 2, rel=1e-8)


@pytest.mark.unit
def test_sphere_population_forward_intensity_is_n_times_delta_rho_V_squared():
    """The consequence of the above, at the level a caller actually uses."""
    pop = SpherePopulation(radius_A=50.0, number_density_per_A3=3.0,
                           delta_rho_e_per_A3=-2.0)
    q = torch.tensor([1e-9], dtype=torch.float64)
    assert float(pop.intensity(q)[0]) == pytest.approx(pop.forward_intensity(), rel=1e-8)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pixel_to_q_matches_the_analytic_flat_detector_formula():
    """With zero tilt and no distortion, |q| must be 4 pi sin(theta) / lambda."""
    g = _geom(n=64)
    r_px = 20.0
    q = pixel_to_q(torch.tensor([g.bcz_px]), torch.tensor([g.bcy_px + r_px]), g)
    two_theta = math.atan2(r_px * g.px_um, g.lsd_um)
    expected = 4 * math.pi * math.sin(0.5 * two_theta) / g.wavelength_A
    assert float(torch.linalg.norm(q[0])) == pytest.approx(expected, rel=1e-12)
    assert float(torch.linalg.norm(q[0])) == pytest.approx(
        g.q_at_pixel_radius(r_px), rel=1e-12)


@pytest.mark.unit
def test_beam_centre_pixel_has_exactly_zero_q():
    g = _geom(n=64, beamstop=0)
    q = pixel_to_q(torch.tensor([g.bcz_px]), torch.tensor([g.bcy_px]), g)
    assert float(torch.linalg.norm(q[0])) == pytest.approx(0.0, abs=1e-18)


@pytest.mark.unit
def test_direct_beam_pixel_is_masked_even_with_no_beamstop():
    """`beamstop_mask` is a STRICT ``r > 0``, so the q = 0 pixel is excluded even
    at radius 0. That is deliberate: the kernel is singular there, and it means
    a user who forgets the beamstop gets a usable frame rather than a crash.
    The explicit q = 0 guard inside `simulate_frame` is defence behind it."""
    g = _geom(n=32, beamstop=0)
    mask = g.beamstop_mask()
    assert not bool(mask[g.bcz_px, g.bcy_px])
    assert int((~mask).sum()) == 1              # exactly the one pixel
    fr = simulate_frame(g, network=_aligned_loops(2), stiffness=isotropic_stiffness(LAM, MU),
                        electron_density_e_per_A3=RHO_CU)
    assert torch.isfinite(fr.intensity).all()


@pytest.mark.unit
def test_beamstop_masks_the_expected_pixels():
    g = _geom(n=64, beamstop=6)
    mask = g.beamstop_mask()
    assert not bool(mask[g.bcz_px, g.bcy_px])          # centre is masked
    assert bool(mask[0, 0])                            # a corner is not


@pytest.mark.unit
def test_electron_density_from_midas_hkls():
    """4 Cu atoms in a 3.615 A cube.

    f(0) from the Cromer-Mann coefficients is 28.9859, not exactly Z = 29 -- the
    analytic sum rule is approximate. Assert against the table, and separately
    that the table is within 0.1 % of Z, which is the check that would catch a
    genuinely wrong species.
    """
    from midas_hkls.form_factors import form_factor
    f0 = float(form_factor(0.0, "Cu"))
    assert f0 == pytest.approx(29.0, rel=1e-3)
    rho = electron_density_per_A3(["Cu"], [4], 3.615 ** 3)
    assert rho == pytest.approx(4 * f0 / 3.615 ** 3, rel=1e-12)


@pytest.mark.unit
def test_electron_density_rejects_bad_input():
    with pytest.raises(ValueError, match="elements but"):
        electron_density_per_A3(["Cu", "O"], [4], 47.0)
    with pytest.raises(ValueError, match="must be positive"):
        electron_density_per_A3(["Cu"], [4], 0.0)


# ---------------------------------------------------------------------------
# Loop versus void -- the point of the package
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_void_frame_is_azimuthally_flat():
    """The null the loop anisotropy is measured against."""
    g = _geom(n=96)
    voids = SpherePopulation(radius_A=50.0, number_density_per_A3=1e-9,
                             delta_rho_e_per_A3=-RHO_CU, label="voids")
    fr = simulate_frame(g, particles=[voids], sample_volume_A3=1.0)
    _, I, _ = azimuthal_profile(fr, q_centre_inv_A=0.008, q_width_inv_A=0.002)
    assert float(I.max() / I.min()) == pytest.approx(1.0, rel=2e-3)


@pytest.mark.unit
def test_loop_frame_has_an_azimuthal_NULL_along_the_loop_normal():
    """With the Laue term the amplitude is dV(1-kappa)sin^2(theta), so the
    intensity goes as sin^4 and has an exact NULL along the loop normal.

    The previous version of this test asserted a modulation depth of
    1 - kappa^2 with the MAXIMUM along the normal -- the distortion-only law,
    which is backwards (Ehrhart/Trinkaus/Larson 1982 Eq. 8b).
    """
    g = _geom(n=96)
    fr = simulate_frame(g, network=_aligned_loops(8), stiffness=isotropic_stiffness(LAM, MU),
                        electron_density_e_per_A3=RHO_CU, incoherent_loops=True)
    _, I, _ = azimuthal_profile(fr, q_centre_inv_A=0.006, q_width_inv_A=0.0015, n_bins=36)
    I = I / I.max()
    # A null, not a 2.5:1 contrast. The floor is set by the beam's small
    # along-x q component, not by kappa.
    assert float(I.min()) < 0.02, f"expected a near-null, got {float(I.min()):.3f}"


@pytest.mark.unit
def test_loop_modulation_has_two_fold_symmetry_about_the_normal():
    """MINIMA along +/-z (the loop normal), maxima along +/-y (in the loop plane).

    Note the swap relative to the distortion-only law: including the Laue term
    moves the extremum from the normal to the plane.
    """
    g = _geom(n=96)
    fr = simulate_frame(g, network=_aligned_loops(8), stiffness=isotropic_stiffness(LAM, MU),
                        electron_density_e_per_A3=RHO_CU, incoherent_loops=True)
    az, I, _ = azimuthal_profile(fr, q_centre_inv_A=0.006, q_width_inv_A=0.0015, n_bins=36)
    trough = float(az[int(torch.argmin(I))]) % 180.0
    peak = float(az[int(torch.argmax(I))]) % 180.0
    assert trough == pytest.approx(90.0, abs=10.0)     # null along z, the normal
    assert min(peak, 180.0 - peak) == pytest.approx(0.0, abs=10.0)   # peak along y


@pytest.mark.unit
def test_azimuth_discriminates_far_more_strongly_than_the_radial_average():
    """Both reductions carry information; the azimuthal one carries vastly more.

    An earlier version claimed the radial average was essentially blind
    (shape difference < 0.25). With the Laue term included that is no longer
    true: the loop's form factor is a Bessel `2 A J1(q_A R)/(q_A R)`, not a
    sphere's, so the radial profiles differ measurably (~0.4). The honest
    statement is comparative, not absolute -- the azimuthal contrast exceeds
    20x against a void's 1.00, while the radial shapes differ by a factor
    smaller than 2.
    """
    g = _geom(n=96)
    C6 = isotropic_stiffness(LAM, MU)
    net = _aligned_loops(8)

    loop_fr = simulate_frame(g, network=net, stiffness=C6,
                             electron_density_e_per_A3=RHO_CU, incoherent_loops=True)
    R_eq = loop_equivalent_sphere_radius_A(50.0, B_CU_A)
    voids = SpherePopulation(radius_A=R_eq, number_density_per_A3=1.0,
                             delta_rho_e_per_A3=-RHO_CU, label="voids")
    void_fr = simulate_frame(g, particles=[voids], sample_volume_A3=1.0)

    ql, Il, _ = radial_average(loop_fr, n_bins=40)
    qv, Iv, _ = radial_average(void_fr, n_bins=40)
    radial_diff = float((Il / Il[0] - Iv / Iv[0]).abs().max())

    _, Al, _ = azimuthal_profile(loop_fr, q_centre_inv_A=0.006, q_width_inv_A=0.0015)
    _, Av, _ = azimuthal_profile(void_fr, q_centre_inv_A=0.006, q_width_inv_A=0.0015)
    loop_contrast = float(Al.max() / Al.min().clamp(min=1e-300))
    void_contrast = float(Av.max() / Av.min())

    assert void_contrast < 1.01, "the void must be azimuthally flat"
    assert loop_contrast > 20.0, "the loops must show the near-null"
    # The comparative statement, which is what survives.
    assert loop_contrast / max(radial_diff, 1e-9) > 20.0
    assert radial_diff < 2.0, "radial shapes should still be broadly similar"


@pytest.mark.unit
def test_loop_is_far_weaker_than_a_void_of_the_same_radius():
    """dV = pi R^2 b versus 4 pi R^3 / 3: a factor 27 in volume, ~700 in I(0).

    The number the collaborator needs before committing beamtime.
    """
    R_A = 50.0
    dV = math.pi * R_A ** 2 * B_CU_A
    V_void = 4.0 / 3.0 * math.pi * R_A ** 3
    assert (V_void / dV) ** 2 == pytest.approx(700.0, rel=0.1)
    assert loop_equivalent_sphere_radius_A(R_A, B_CU_A) == pytest.approx(16.9, rel=0.02)


# ---------------------------------------------------------------------------
# Source terms
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_loop_forward_amplitude_vanishes_along_the_normal_and_peaks_in_plane():
    """A(q->0) = rho_e dV (1-kappa) sin^2(theta): ZERO on the normal.

    This test previously asserted `rho_e dV` along the normal -- the
    distortion-only value, which the Laue term cancels exactly.
    """
    from midas_ddd import find_loops, relaxation_volumes_um3
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=64)
    dV_um3 = float(relaxation_volumes_um3(net, find_loops(net))[0])
    C6 = isotropic_stiffness(LAM, MU)
    scale = RHO_CU * dV_um3 * 1e12

    along = torch.tensor([[0.0, 0.0, 1e-4]], dtype=torch.float64)
    inplane = torch.tensor([[1e-4, 0.0, 0.0]], dtype=torch.float64)
    a_n = float(loop_amplitude(net, along, C6, electron_density_e_per_A3=RHO_CU).abs()[0])
    a_p = float(loop_amplitude(net, inplane, C6, electron_density_e_per_A3=RHO_CU).abs()[0])
    assert a_n < 1e-9 * scale, "the total must vanish along the loop normal"
    assert a_p == pytest.approx((1.0 - KAPPA) * scale, rel=1e-5)


@pytest.mark.unit
def test_incoherent_and_coherent_agree_for_a_single_loop():
    """With one loop there is no interference, so the two must coincide."""
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=32)
    q = torch.tensor([[1e-3, 0.0, 2e-3], [0.0, 1e-3, 1e-3]], dtype=torch.float64)
    C6 = isotropic_stiffness(LAM, MU)
    coh = loop_intensity(net, q, C6, electron_density_e_per_A3=RHO_CU, incoherent=False)
    inc = loop_intensity(net, q, C6, electron_density_e_per_A3=RHO_CU, incoherent=True)
    assert torch.allclose(coh, inc, rtol=1e-10)


@pytest.mark.unit
def test_incoherent_sum_scales_linearly_with_loop_count():
    """Intensities add for a dilute random population: 2N loops give 2x."""
    C6 = isotropic_stiffness(LAM, MU)
    # Probe IN the loop plane -- along the normal the amplitude is now exactly
    # zero, so a ratio there would be 0/0.
    q = torch.tensor([[2e-3, 0.0, 0.0]], dtype=torch.float64)
    a = loop_intensity(_aligned_loops(4, seed=1), q, C6,
                       electron_density_e_per_A3=RHO_CU, incoherent=True)
    b = loop_intensity(_aligned_loops(8, seed=1), q, C6,
                       electron_density_e_per_A3=RHO_CU, incoherent=True)
    assert float(b[0] / a[0]) == pytest.approx(2.0, rel=1e-9)


@pytest.mark.unit
def test_an_edge_line_scatters_into_a_streak_perpendicular_to_itself():
    """An edge line along lab z puts its reciprocal-space sheet at q_z = 0: a streak
    along the detector row through the beam centre, and almost nothing off it.

    midas-saxs 0.1.x returned an identically zero frame here, on the grounds that
    open lines have no relaxation volume. That is true as q -> 0 and wrong at finite
    q, where the line's dilatation field scatters (Thomson, Levine & Long 1999,
    Eq. 6; gated in midas_ddd/tests/test_line_term.py).
    """
    g = _geom(n=64, beamstop=4)
    line = straight_line(length_um=2.0, burgers=(1.0, 0, 0), line=(0, 0, 1.0),
                         slip_normal=(0, 1.0, 0), n_segments=10)
    fr = simulate_frame(g, network=line, stiffness=isotropic_stiffness(LAM, MU),
                        electron_density_e_per_A3=RHO_CU)
    assert set(fr.components) == {"lines"}
    row = g.bcz_px
    on = fr.intensity[row, :][fr.mask[row, :]]
    off = fr.intensity[row + 10, :][fr.mask[row + 10, :]]
    assert float(on.mean()) > 100.0 * float(off.mean())
    assert any("terminate inside the medium" in w for w in fr.warnings)


@pytest.mark.unit
def test_include_lines_false_reproduces_the_loops_only_frame_and_says_what_it_left_out():
    g = _geom(n=48)
    fr = simulate_frame(g, network=straight_line(length_um=0.5, n_segments=16),
                        stiffness=isotropic_stiffness(LAM, MU),
                        electron_density_e_per_A3=RHO_CU, include_lines=False)
    assert float(fr.intensity.abs().max()) == 0.0
    assert any("left out" in w for w in fr.warnings)


@pytest.mark.unit
def test_an_isotropic_screw_line_leaves_the_frame_empty():
    """No dilatation, no small-angle signal -- with the edge as positive control."""
    g = _geom(n=48)
    C6 = isotropic_stiffness(LAM, MU)
    kw = dict(length_um=2.0, line=(0, 0, 1.0), slip_normal=(0, 1.0, 0), n_segments=10)
    screw = simulate_frame(g, network=straight_line(burgers=(0, 0, 1.0), **kw), stiffness=C6,
                           electron_density_e_per_A3=RHO_CU)
    edge = simulate_frame(g, network=straight_line(burgers=(1.0, 0, 0), **kw), stiffness=C6,
                          electron_density_e_per_A3=RHO_CU)
    assert float(edge.intensity.max()) > 0.0
    assert float(screw.intensity.max()) <= 1e-20 * float(edge.intensity.max())


@pytest.mark.unit
def test_loops_and_lines_are_separate_components_that_sum_to_the_total():
    g = _geom(n=48)
    C6 = isotropic_stiffness(LAM, MU)
    line = straight_line(length_um=0.5, burgers=(1.0, 0, 0), line=(0, 0, 1.0),
                         slip_normal=(0, 1.0, 0), n_segments=8, cell_size_um=1.0)
    net = combine([_aligned_loops(3), line], cell_size_um=1.0)
    fr = simulate_frame(g, network=net, stiffness=C6, electron_density_e_per_A3=RHO_CU)
    assert set(fr.components) == {"loops", "lines"}
    assert torch.allclose(fr.components["loops"] + fr.components["lines"], fr.intensity,
                          rtol=1e-12)
    coherent = simulate_frame(g, network=net, stiffness=C6, electron_density_e_per_A3=RHO_CU,
                              incoherent_loops=False)
    assert set(coherent.components) == {"dislocations"}


@pytest.mark.unit
def test_loops_only_frame_is_unchanged_by_the_line_machinery():
    """Regression: a loops-only network gives exactly the per-loop incoherent sum."""
    g = _geom(n=48)
    C6 = isotropic_stiffness(LAM, MU)
    net = _aligned_loops(4)
    fr = simulate_frame(g, network=net, stiffness=C6, electron_density_e_per_A3=RHO_CU)
    ref = loop_intensity(net, fr.q[fr.mask], C6, electron_density_e_per_A3=RHO_CU,
                         incoherent=True)
    assert set(fr.components) == {"loops"}
    assert torch.allclose(fr.intensity[fr.mask], ref, rtol=1e-12)


def _periodic_bowed_line(cell=1.0, n=40, relabel_seed=None):
    """A bowed edge line along x that closes only through the x faces of a periodic cell."""
    dt = torch.float64
    s = torch.arange(n, dtype=dt) / n
    pts = torch.stack([s * cell - 0.5 * cell, 0.10 * cell * torch.sin(2 * math.pi * s),
                       0.05 * cell * torch.sin(4 * math.pi * s + 0.7)], dim=1)
    pts = pts - cell * torch.round(pts / cell)
    segs = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)
    if relabel_seed is not None:
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(relabel_seed))
        moved = torch.empty_like(pts)
        moved[perm] = pts
        pts, segs = moved, perm[segs]
    return ddd.DislocationNetwork(
        nodes_um=pts, segments=segs,
        burgers_b=torch.tensor([0.0, 0.3, 1.0], dtype=dt).expand(n, 3).clone(),
        normals=torch.zeros(n, 3, dtype=dt), b_magnitude_A=B_CU_A,
        cell_min_um=torch.full((3,), -cell / 2, dtype=dt),
        cell_max_um=torch.full((3,), cell / 2, dtype=dt), pbc=(True, True, True),
        constraints=torch.zeros(n, dtype=torch.int64))


@pytest.mark.unit
def test_a_periodic_line_is_a_lattice_component_that_ignores_node_numbering():
    """A line closing through the periodic boundary scatters only on the cell's
    reciprocal lattice, so the frame shows it at a stated resolution. Renumbering its
    nodes changes nothing; the single-window version it replaces did change (/verify
    claim 8305670629a7)."""
    g = _geom(n=48)
    C6 = isotropic_stiffness(LAM, MU)
    fr = simulate_frame(g, network=_periodic_bowed_line(), stiffness=C6,
                        electron_density_e_per_A3=RHO_CU)
    assert set(fr.components) == {"periodic_lines"}
    again = simulate_frame(g, network=_periodic_bowed_line(relabel_seed=7), stiffness=C6,
                           electron_density_e_per_A3=RHO_CU)
    m = fr.mask
    peak = float(fr.intensity[m].max())
    assert peak > 0.0
    assert torch.allclose(fr.intensity[m], again.intensity[m], rtol=1e-10, atol=1e-12 * peak)
    assert not any("floor of the periodic-cell intensity" in w for w in fr.warnings)


@pytest.mark.unit
def test_a_pure_screw_periodic_network_is_reported_as_roundoff():
    """A screw has no dilatation in isotropic elasticity. The frame is float64 residue and
    must say so, instead of rendering it as a structured image (which is what happened to
    the notebook's first ExaDiS lines network, all <110> screws)."""
    g = _geom(n=48)
    C6 = isotropic_stiffness(LAM, MU)
    dt = torch.float64
    n, cell = 40, 1.0
    s = torch.arange(n, dtype=dt) / n
    pts = torch.stack([s * cell - 0.5 * cell, torch.zeros(n, dtype=dt), torch.zeros(n, dtype=dt)], 1)
    segs = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n], dim=1)

    def net(bvec):
        return ddd.DislocationNetwork(
            nodes_um=pts, segments=segs, burgers_b=torch.tensor(bvec, dtype=dt).expand(n, 3).clone(),
            normals=torch.zeros(n, 3, dtype=dt), b_magnitude_A=B_CU_A,
            cell_min_um=torch.full((3,), -cell / 2, dtype=dt), cell_max_um=torch.full((3,), cell / 2, dtype=dt),
            pbc=(True, True, True), constraints=torch.zeros(n, dtype=torch.int64))

    screw = simulate_frame(g, network=net([1.0, 0.0, 0.0]), stiffness=C6, electron_density_e_per_A3=RHO_CU)
    edge = simulate_frame(g, network=net([0.0, 1.0, 0.0]), stiffness=C6, electron_density_e_per_A3=RHO_CU)
    assert any("roundoff, not signal" in w for w in screw.warnings)
    assert not any("roundoff" in w for w in edge.warnings)
    assert float(edge.intensity.max()) > 1e20 * float(screw.intensity.max())


@pytest.mark.unit
def test_pixels_below_the_periodic_floor_are_flagged():
    fr = simulate_frame(_geom(n=48), network=_periodic_bowed_line(cell=0.2),
                        stiffness=isotropic_stiffness(LAM, MU), electron_density_e_per_A3=RHO_CU)
    assert any("floor of the periodic-cell intensity" in w for w in fr.warnings)


@pytest.mark.unit
def test_coherent_loops_in_a_periodic_cell_go_through_the_lattice_and_say_so():
    """Several loops summed coherently in a periodic cell have no total off the
    lattice that the network defines. One loop alone does, and stays exact."""
    g = _geom(n=48)
    C6 = isotropic_stiffness(LAM, MU)
    kw = dict(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=24, cell_size_um=1.0,
              pbc=(True, True, True))
    centres = ((0.0, 0.0, 0.0), (0.2, -0.1, 0.3), (-0.3, 0.25, -0.1))
    three = combine([prismatic_loop(center_um=c, **kw) for c in centres], cell_size_um=1.0)
    fr = simulate_frame(g, network=three, stiffness=C6, electron_density_e_per_A3=RHO_CU,
                        incoherent_loops=False)
    assert set(fr.components) == {"loops"}
    assert any("coherent sum in a periodic cell" in w for w in fr.warnings)
    one = prismatic_loop(center_um=(0.1, 0.0, 0.0), **kw)
    coh = simulate_frame(g, network=one, stiffness=C6, electron_density_e_per_A3=RHO_CU,
                         incoherent_loops=False)
    inc = simulate_frame(g, network=one, stiffness=C6, electron_density_e_per_A3=RHO_CU)
    assert torch.allclose(coh.intensity, inc.intensity, rtol=1e-12)
    assert not any("coherent sum in a periodic cell" in w for w in coh.warnings)


@pytest.mark.unit
def test_network_amplitudes_leaves_winding_lines_to_the_intensity_path():
    from midas_saxs.strain_source import network_amplitudes

    amps, info = network_amplitudes(
        _periodic_bowed_line(), torch.tensor([[0.004, 0.002, 0.0]], dtype=torch.float64),
        isotropic_stiffness(LAM, MU), electron_density_e_per_A3=RHO_CU)
    assert amps["lines"].shape[0] == 0 and info["n_winding_components"] == 1
    assert any("close only through the periodic boundary" in w for w in info["warnings"])


@pytest.mark.unit
def test_network_without_material_is_refused():
    g = _geom(n=32)
    with pytest.raises(ValueError, match="electron_density|stiffness"):
        simulate_frame(g, network=_aligned_loops(2))


@pytest.mark.unit
def test_components_are_reported_separately():
    """So you can see whether the loops are visible above the voids at all."""
    g = _geom(n=64)
    voids = SpherePopulation(radius_A=50.0, number_density_per_A3=1e-9,
                             delta_rho_e_per_A3=-RHO_CU, label="voids")
    fr = simulate_frame(g, network=_aligned_loops(4), stiffness=isotropic_stiffness(LAM, MU),
                        electron_density_e_per_A3=RHO_CU, particles=[voids],
                        sample_volume_A3=1.0)
    assert set(fr.components) == {"loops", "voids"}
    total = sum(fr.components.values())
    assert torch.allclose(total, fr.intensity, rtol=1e-12)


@pytest.mark.unit
def test_masked_pixels_are_zero_not_garbage():
    g = _geom(n=64, beamstop=10)
    fr = simulate_frame(g, particles=[SpherePopulation(
        radius_A=50.0, number_density_per_A3=1e-9, delta_rho_e_per_A3=-RHO_CU)],
        sample_volume_A3=1.0)
    assert float(fr.intensity[~fr.mask].abs().max()) == 0.0
    assert float(fr.intensity[fr.mask].min()) > 0.0


@pytest.mark.unit
def test_polydispersity_broadens_but_conserves_the_forward_intensity():
    q = torch.logspace(-3, -1, 40, dtype=torch.float64)
    mono = SpherePopulation(radius_A=50.0, number_density_per_A3=1.0,
                            delta_rho_e_per_A3=1.0, sigma_lognormal=0.0)
    poly = SpherePopulation(radius_A=50.0, number_density_per_A3=1.0,
                            delta_rho_e_per_A3=1.0, sigma_lognormal=0.3)
    Im, Ip = mono.intensity(q), poly.intensity(q)
    # Polydispersity washes out the form-factor minima, so the deep dips fill in.
    assert float(Ip.min()) > float(Im.min())


@pytest.mark.unit
def test_azimuthal_profile_refuses_an_empty_ring():
    g = _geom(n=48)
    fr = simulate_frame(g, particles=[SpherePopulation(
        radius_A=50.0, number_density_per_A3=1e-9, delta_rho_e_per_A3=-1.0)])
    with pytest.raises(ValueError, match="no unmasked pixels"):
        azimuthal_profile(fr, q_centre_inv_A=5.0, q_width_inv_A=0.001)


# ---------------------------------------------------------------------------
# Differentiability -- the deliverable
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_frame_is_differentiable_in_the_loop_geometry():
    """An ML inversion trains against this, so gradients must reach the loops."""
    C6 = isotropic_stiffness(LAM, MU)
    net = prismatic_loop(radius_um=0.005, burgers=(0, 0, 1.0), n_segments=16)
    net.nodes_um = net.nodes_um.clone().requires_grad_(True)
    q = torch.tensor([[0.0, 1e-3, 2e-3], [0.0, 2e-3, 1e-3]], dtype=torch.float64)
    loop_intensity(net, q, C6, electron_density_e_per_A3=RHO_CU).sum().backward()
    assert net.nodes_um.grad is not None
    assert torch.isfinite(net.nodes_um.grad).all()
    assert float(net.nodes_um.grad.abs().max()) > 0


@pytest.mark.autograd
def test_particle_intensity_is_differentiable_in_the_radius():
    q = torch.tensor([0.01, 0.02], dtype=torch.float64)
    R = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    sphere_form_factor_squared(q, R).sum().backward()
    assert R.grad is not None and float(R.grad.abs()) > 0


# ---------------------------------------------------------------------------
# Absolute units -- anchored to the textbook SAXS invariant
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_intensity_reproduces_the_textbook_saxs_invariant():
    """``int I(q) q^2 dq = 2 pi^2 drho^2 phi (1-phi)`` for a dilute two-phase system.

    A reference completely independent of this package: it fixes the electron-unit
    normalisation of `SpherePopulation.intensity` end to end -- the V^2 inside the
    form factor, the number density and the contrast all at once. A stray factor
    anywhere shows up here.

    The ~0.2 % shortfall is the Porod tail beyond the integration limit, not an
    error: with I -> 2 pi drho^2 S / q^4, the missing piece is
    2 pi drho^2 S / q_max = 0.19 % of the total at q_max = 10 1/A.
    """
    R_A, n, drho = 50.0, 1e-9, -2.4543
    pop = SpherePopulation(radius_A=R_A, number_density_per_A3=n,
                           delta_rho_e_per_A3=drho)
    q = torch.logspace(-5, 1.0, 200000, dtype=torch.float64)
    Q_num = float(torch.trapz(pop.intensity(q) * q ** 2, q))
    phi = pop.volume_fraction
    Q_exact = 2 * math.pi ** 2 * drho ** 2 * phi * (1 - phi)
    assert Q_num / Q_exact == pytest.approx(1.0, abs=0.005)


@pytest.mark.unit
def test_classical_electron_radius_is_right():
    """r_e = 2.8179403262e-15 m, expressed in angstroms."""
    from midas_saxs.detector import R_E_A
    assert R_E_A == pytest.approx(2.8179403262e-5, rel=1e-12)


@pytest.mark.unit
def test_absolute_units_scale_every_component_by_r_e_squared():
    """`absolute_units=True` turns electrons^2 into a cross-section, and must do
    it to the components as well as the total -- otherwise they stop summing."""
    from midas_saxs.detector import R_E_A
    g = _geom(n=48)
    voids = SpherePopulation(radius_A=50.0, number_density_per_A3=1e-9,
                             delta_rho_e_per_A3=-RHO_CU, label="voids")
    rel = simulate_frame(g, particles=[voids], sample_volume_A3=1.0)
    absu = simulate_frame(g, particles=[voids], sample_volume_A3=1.0,
                          absolute_units=True)
    ratio = absu.intensity[absu.mask] / rel.intensity[rel.mask]
    assert torch.allclose(ratio, torch.full_like(ratio, R_E_A ** 2), rtol=1e-12)
    total = sum(absu.components.values())
    assert torch.allclose(total, absu.intensity, rtol=1e-12)
