"""Texture tools: family directions per crystal frame, weighted pole densities with a
matched null, and inverse-pole-figure coordinates (hexagonal).

Each test pins a case with a known answer: a uniform population must give ~1 MRD
everywhere, an a-axis fibre must peak exactly at the fibre axis, a weight of 2 must
equal a duplicated grain, and the frame conversion must agree with the lattice.
"""
import math

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from midas_plotting.ipf import (A_ALONG_X, BUSING_LEVY, family_directions, ipf_sector_coords,
                                to_busing_levy)
from midas_plotting.laue import (SURFACE_NORMAL_34IDE, _random_orientations, pole_density,
                                 pole_figure_density, texture_strength)

N = SURFACE_NORMAL_34IDE / np.linalg.norm(SURFACE_NORMAL_34IDE)


def _same_axes(A, B, tol=1e-9):
    """Row sets equal up to order and sign."""
    for a in A:
        if not np.any(np.all(np.abs(np.abs(B @ a) - 1.0) < tol, axis=0) if B.ndim == 1 else
                      np.abs(np.abs(B @ a) - 1.0) < tol):
            return False
    return len(A) == len(B)


@pytest.mark.parametrize("fam", ["0001", "<2-1-10>", "[10-10]"])
def test_family_directions_agree_between_frames(fam):
    a = family_directions(194, fam, A_ALONG_X)
    b = family_directions(194, fam, BUSING_LEVY)
    assert _same_axes(to_busing_levy(a, 194, A_ALONG_X), b)


def test_a_axis_family_contains_x_in_the_a_along_x_frame():
    a = family_directions(194, "<2-1-10>", A_ALONG_X)
    assert np.any(np.abs(np.abs(a @ np.array([1.0, 0, 0])) - 1.0) < 1e-12)
    m = family_directions(194, "<10-10>", BUSING_LEVY)
    assert np.any(np.abs(np.abs(m @ np.array([1.0, 0, 0])) - 1.0) < 1e-12)


def test_unknown_family_and_family_for_cubic_are_refused():
    with pytest.raises(ValueError):
        family_directions(194, "11-23")
    with pytest.raises(NotImplementedError):
        family_directions(225, "0001")


def _a_fibre(n, rng):
    """Orientations with the A_ALONG_X a-axis (crystal x) along N, random rotation about N."""
    x = np.array([1.0, 0, 0]); out = []
    for t in rng.uniform(0, 2 * np.pi, n):
        e1 = N; e2 = np.cross(N, x); e2 /= np.linalg.norm(e2); e3 = np.cross(e1, e2)
        c, s = math.cos(t), math.sin(t)
        # crystal x -> N ; crystal y, z -> rotated in the plane normal to N
        out.append(np.stack([e1, c * e2 + s * e3, -s * e2 + c * e3], 1))
    return np.array(out)


def test_uniform_population_is_about_one_mrd():
    rng = np.random.default_rng(0)
    grid, mrd = pole_density(_random_orientations(20000, rng), family_directions(194, "0001"))
    assert abs(mrd.mean() - 1.0) < 1e-12
    assert mrd.max() < 1.6 and mrd.min() > 0.5


def test_a_fibre_peaks_at_the_fibre_axis():
    rng = np.random.default_rng(1)
    om = _a_fibre(400, rng)
    grid, a = pole_density(om, family_directions(194, "<2-1-10>", A_ALONG_X))
    assert math.degrees(math.acos(min(1.0, abs(grid[np.argmax(a)] @ N)))) < 3.0
    _, c = pole_density(om, family_directions(194, "0001", A_ALONG_X))
    assert c[np.argmax(grid @ N)] < 0.05                     # c lies in the plane: nothing at N


def test_weight_two_equals_a_duplicated_grain():
    rng = np.random.default_rng(2)
    om = _random_orientations(30, rng); d = family_directions(194, "<10-10>", A_ALONG_X)
    w = np.ones(30); w[0] = 2.0
    _, m_w = pole_density(om, d, weights=w)
    _, m_dup = pole_density(np.concatenate([om, om[:1]]), d)
    assert np.allclose(m_w, m_dup, atol=1e-12)


def test_pole_figure_density_ratio_separates_fibre_from_random():
    rng = np.random.default_rng(3)
    d = family_directions(194, "<2-1-10>", A_ALONG_X)
    _, fib = pole_figure_density(_a_fibre(200, rng), d, n_null=40, seed=4, label="fibre")
    _, ran = pole_figure_density(_random_orientations(200, rng), d, n_null=40, seed=4, label="random")
    assert fib["ratio"] > 2.0
    assert ran["ratio"] < 1.25


def test_ipf_sector_coords_known_cases():
    th, ph = ipf_sector_coords(np.eye(3)[None], 194, (1.0, 0, 0), A_ALONG_X)   # a-axis along the sample axis
    assert th[0] == pytest.approx(90.0) and ph[0] == pytest.approx(30.0)      # the [2-1-10] corner
    th, ph = ipf_sector_coords(np.eye(3)[None], 194, (0, 0, 1.0), A_ALONG_X)  # c along the sample axis
    assert th[0] == pytest.approx(0.0, abs=1e-9)
    th, ph = ipf_sector_coords(np.eye(3)[None], 194, (1.0, 0, 0), BUSING_LEVY)  # BL: x is [10-10]
    assert ph[0] == pytest.approx(0.0)


def test_texture_strength_explicit_unit_weights_match_default():
    rng = np.random.default_rng(5)
    om = _random_orientations(80, rng)
    a = texture_strength(om, n_null=20, seed=6)
    b = texture_strength(om, n_null=20, seed=6, weights=np.ones(80))
    assert np.allclose(a, b)
