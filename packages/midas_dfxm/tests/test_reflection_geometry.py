"""Reflection-geometry DFXM adopted from Yildirim et al. (arXiv:2608.09841, 2026):
absorption depth-weighting, single-frame flank difference, and directional visibility.
"""
import math

import torch

from midas_dfxm import (
    GoniometerSetting,
    aligned_resolution,
    attenuation_1e_depth,
    cubic_stiffness,
    depth_weighted_intensity,
    directional_visibility,
    flank_difference_intensity,
    make_uniform_field,
    reference_q_nom,
    reflection_depth_weight,
    stroh_dislocation,
    surface_depth,
)
DT = torch.float64
CELL = (5.868, 5.868, 5.868, 90.0, 90.0, 90.0)     # MCT-like (a ~ 6.47 is the real MCT; value here only sets q)


def test_attenuation_1e_depth_reproduces_paper():
    """z_1e = Lambda sin(theta_B)/2. The paper's 1.2 um at MCT(111), 17 keV, 2theta=11.11 deg
    implies Lambda ~ 24.8 um; the formula must round-trip that to 1.2 um."""
    two_theta = 11.11
    lam = 2.0 * 1.2 / math.sin(math.radians(0.5 * two_theta))   # invert to get Lambda
    z1e = attenuation_1e_depth(two_theta, lam)
    assert abs(z1e - 1.2) < 1e-6


def test_reflection_depth_weight_profile():
    """weight(0)=1, weight(z_1e)=1/e, monotone decreasing into the sample."""
    two_theta, lam = 11.11, 24.8
    z1e = attenuation_1e_depth(two_theta, lam)
    z = torch.tensor([0.0, z1e, 2 * z1e, 5 * z1e], dtype=DT)
    w = reflection_depth_weight(z, two_theta_deg=two_theta, attenuation_length_um=lam)
    assert abs(float(w[0]) - 1.0) < 1e-12
    assert abs(float(w[1]) - math.exp(-1.0)) < 1e-9
    assert torch.all(w[1:] < w[:-1])                            # strictly decreasing


def test_surface_depth_zeroed_at_shallowest():
    pos = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 3.]], dtype=DT)
    # surface normal +z (outward): shallowest is the largest z
    d = surface_depth(pos, (0.0, 0.0, 1.0))
    assert float(d.min()) == 0.0
    assert torch.allclose(d, torch.tensor([3.0, 2.0, 0.0], dtype=DT))


def test_depth_weighting_attenuates_deep_voxels():
    """A uniform crystal imaged in reflection: deep voxels contribute exponentially less."""
    field = make_uniform_field(shape=(1, 1, 40), spacing_um=1.0, lattice_params=CELL, dtype=DT)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-3)
    lam, two_theta = 24.8, 11.11
    Iw = depth_weighted_intensity(field, hkl, center, res, surface_normal=(0.0, 0.0, 1.0),
                                  two_theta_deg=two_theta, attenuation_length_um=lam)
    depth = surface_depth(field.positions, (0.0, 0.0, 1.0))
    order = torch.argsort(depth)
    Iw_sorted = Iw[order]
    # weighted intensity decreases monotonically with depth (uniform base response)
    assert torch.all(Iw_sorted[1:] <= Iw_sorted[:-1] + 1e-12)
    z1e = attenuation_1e_depth(two_theta, lam)
    deepest, shallow = float(Iw_sorted[-1]), float(Iw_sorted[0])
    assert deepest < shallow * math.exp(-float(depth.max()) / z1e) + 1e-9


def test_flank_difference_antisymmetric_for_perfect_crystal():
    """A strain-free crystal is symmetric about the aligned setting, so the +/- flank
    weak-beam images are equal and their difference vanishes; a d-spacing offset breaks it."""
    perfect = make_uniform_field(shape=(6, 6, 1), spacing_um=1.0, lattice_params=CELL, dtype=DT)
    hkl = (1, 1, 1)
    center = GoniometerSetting()
    q_nom = reference_q_nom(perfect, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-3)
    diff = flank_difference_intensity(perfect, hkl, res, center=center, flank_deg=0.03, axis="chi")
    assert float(diff.abs().max()) < 1e-9                       # on-peak & symmetric -> cancels

    # a population sitting OFF the aligned rocking peak (as substrate vs layer do, separated
    # by (Delta d/d) tan theta) samples the two flanks unequally -> nonzero difference.
    off_peak = GoniometerSetting(chi=0.08)
    diff2 = flank_difference_intensity(perfect, hkl, res, center=off_peak, flank_deg=0.03, axis="chi")
    assert float(diff2.abs().max()) > 1e-6


def test_directional_visibility_flips_with_line_direction():
    """phi (rocking) senses d u_g/dx, chi senses d u_g/dy. A line along y varies in x
    (shows in phi); a line along x varies in y (shows in chi). The dominance must flip."""
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=DT)
    g = (torch.arange(21, dtype=DT) - 10) * 0.05
    GX, GY = torch.meshgrid(g, g, indexing="xy")
    # OFF the slip plane. At z = 0 an edge dislocation's u_g is a step function, so both
    # gradients vanish away from the core and the two std's are rounding noise (~1e-20
    # against a beta of ~1e-3). Asserting a dominance ordering there compares noise: it
    # held on macOS (s_x = 2.8e-20) and failed on Linux (s_x = exactly 0.0). z = 0.1 puts
    # the sample where the signal is real (s_x ~ 1e-4).
    pos = torch.stack([GX.reshape(-1), GY.reshape(-1),
                       torch.full((21 * 21,), 0.1, dtype=DT)], -1)

    # BOTH must be visible in g, i.e. g.b != 0, or there is nothing to compare.
    # The original second case used b = (0,1,0) against g = (1,0,0): g.b = 0, so that
    # dislocation is invisible by the classic criterion and its whole du_g field is
    # identically zero. The assertion only looked satisfied because the old code
    # returned 0.0 for that degenerate case, so `1.0 > 0.0` read as a flip.
    #
    # Both below carry b = (1,0,0) (g.b = 1); only the line direction differs, which is
    # the variable under test.
    d_line_y = stroh_dislocation(C, burgers=(1, 0, 0), slip_normal=(0, 0, 1),
                                 character="edge", line=(0, 1, 0), core_model="compact")
    d_line_x = stroh_dislocation(C, burgers=(1, 0, 0), slip_normal=(0, 1, 0),
                                 character="edge", line=(0, 0, 1), core_model="compact")
    g_dir = (1.0, 0.0, 0.0)
    vy = directional_visibility(d_line_y, pos, g_dir)
    vx = directional_visibility(d_line_x, pos, g_dir)
    assert vy["phi_frac"] + vy["chi_frac"] == 1.0 or abs(vy["phi_frac"] + vy["chi_frac"] - 1.0) < 1e-6
    # line || y is more phi-visible than line || x
    assert vy["phi_frac"] > vx["phi_frac"]
