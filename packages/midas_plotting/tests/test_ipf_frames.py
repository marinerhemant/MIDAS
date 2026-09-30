"""Crystal-frame handling in IPF colouring (hexagonal/trigonal).

The colour ramps and triangle labels are written for the Busing-Levy frame
(a* along x) that MIDAS far-field builds its B matrix in. LaueMatching
orientation matrices map from an a-along-x frame, 30 deg away about c. These
tests pin the conversion against both lattice constructions directly, so the
rotation's sign is checked by geometry rather than restated.

The 1e-6 colour tolerance is round-off amplified by the sqrt gamma lift
(|diff| ~5e-9 at a corner), not a defect being admitted: a swapped label
moves a corner colour from green to blue, a difference of order 1.
"""
import math

import numpy as np
import pytest

from midas_plotting.ipf import (A_ALONG_X, BUSING_LEVY, direction_rgb, ipf_rgb_from_matrix,
                                sym_matrices, to_busing_levy)

A, C = 2.6649, 4.9468          # Zn (any hexagonal cell does)


def _a_along_x():
    """Columns a, b, c with a along x, b in the xy plane (LaueMatching, midas_hkls)."""
    g = math.radians(120.0)
    return np.array([[A, A * math.cos(g), 0.0], [0.0, A * math.sin(g), 0.0], [0.0, 0.0, C]])


def _busing_levy():
    """Columns a, b, c in the frame of FitUnified.c's B matrix (a* along x)."""
    al, be, ga = (math.radians(x) for x in (90.0, 90.0, 120.0))
    ca, cb, cg, sa, sb, sg = (math.cos(al), math.cos(be), math.cos(ga),
                              math.sin(al), math.sin(be), math.sin(ga))
    gpr = math.acos((ca * cb - cg) / (sa * sb))
    bpr = math.acos((cg * ca - cb) / (sg * sa))
    vol = A * A * C * sa * math.sin(bpr) * sg
    apr, bpr_, cpr = A * C * sa / vol, C * A * sb / vol, A * A * sg / vol
    B = np.array([[apr, bpr_ * math.cos(gpr), cpr * math.cos(bpr)],
                  [0.0, bpr_ * math.sin(gpr), -cpr * math.sin(bpr) * ca],
                  [0.0, 0.0, cpr * math.sin(bpr) * sa]])
    return np.linalg.inv(B).T                       # direct basis: A^T B = I


def _mb(uvtw):
    """Miller-Bravais direction [uvtw] -> three-index [UVW]."""
    u, v, _t, w = uvtw
    return np.array([2 * u + v, 2 * v + u, w], float)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("uvw", [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 3), (1, -1, 2), (0, 0, 1)])
@pytest.mark.parametrize("sg", [194, 166])
def test_a_along_x_conversion_matches_the_two_lattice_constructions(uvw, sg):
    va = _unit(_a_along_x() @ np.array(uvw, float))
    vbl = _unit(_busing_levy() @ np.array(uvw, float))
    got = _unit(to_busing_levy(va, sg, A_ALONG_X)[0])
    assert np.allclose(got, vbl, atol=1e-12)


@pytest.mark.parametrize("label, uvtw, corner", [
    ("[10-10]", (1, 0, -1, 0), (1.0, 0.0, 0.0)),
    ("[2-1-10]", (2, -1, -1, 0), (math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0)),
    ("[0001]", (0, 0, 0, 1), (0.0, 0.0, 1.0)),
])
def test_hexagonal_key_labels_hold_in_the_busing_levy_frame(label, uvtw, corner):
    """Each labelled direction, built from the FF lattice, gets its corner's colour."""
    v = _busing_levy() @ _mb(uvtw)
    assert np.allclose(direction_rgb(v, 194), direction_rgb(np.array(corner), 194), atol=1e-6), label


@pytest.mark.parametrize("label, uvtw, az", [("[10-10]", (1, 0, -1, 0), 0.0),
                                             ("[01-10]", (0, 1, -1, 0), 60.0)])
def test_trigonal_key_labels_hold_in_the_busing_levy_frame(label, uvtw, az):
    v = _busing_levy() @ _mb(uvtw)
    corner = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
    assert np.allclose(direction_rgb(v, 166), direction_rgb(corner, 166), atol=1e-6), label


def test_a_along_x_matrices_are_symmetry_invariant_for_trigonal():
    """-3m orbits are frame-dependent: with the right frame, g.S_a and g colour alike."""
    rng = np.random.default_rng(3)
    q = rng.normal(size=(40, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    g = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    R = np.array([[math.cos(math.radians(-30)), -math.sin(math.radians(-30)), 0],
                  [math.sin(math.radians(-30)), math.cos(math.radians(-30)), 0], [0, 0, 1]])
    s_a = np.einsum("ji,sjk,kl->sil", R, sym_matrices(166), R)      # R^T S_bl R
    axis = (0.0, -math.sqrt(0.5), math.sqrt(0.5))
    ref = ipf_rgb_from_matrix(g, 166, axis, frame=A_ALONG_X)
    worst_right = worst_wrong = 0.0
    for s in s_a:
        gs = g @ s
        worst_right = max(worst_right, np.abs(ipf_rgb_from_matrix(gs, 166, axis, frame=A_ALONG_X) - ref).max())
        worst_wrong = max(worst_wrong, np.abs(ipf_rgb_from_matrix(gs, 166, axis, frame=BUSING_LEVY)
                                              - ipf_rgb_from_matrix(g, 166, axis, frame=BUSING_LEVY)).max())
    assert worst_right < 1e-9
    assert worst_wrong > 0.05          # the frame matters: ignoring it is not invariant


def test_frame_is_identity_for_cubic_and_rejects_unknown_names():
    d = np.array([[0.3, 0.4, 0.5]])
    assert np.array_equal(to_busing_levy(d, 225, A_ALONG_X), d)
    with pytest.raises(ValueError):
        to_busing_levy(d, 194, "a_star_along_y")


def test_laue_orientation_map_ipf_is_the_crystal_direction_along_the_normal():
    """c-axis along the surface normal must be the [0001] (red) corner in IPF-N."""
    import matplotlib
    matplotlib.use("Agg")
    from midas_plotting.laue import SURFACE_NORMAL_34IDE, orientation_map
    from midas_plotting.solutions import LaueSolutions
    n = SURFACE_NORMAL_34IDE / np.linalg.norm(SURFACE_NORMAL_34IDE)
    x = np.array([1.0, 0.0, 0.0]); y = np.cross(n, x)
    om = np.stack([x, y, n], 1)[None]                # crystal z (c) -> lab n
    sol = LaueSolutions(image=np.array([1]), grain=np.array([0]), n_matches=np.array([20]),
                        orient_mat=om, pos=np.array([[0.0, 0.0]]))
    ax = orientation_map(sol, color="ipf", space_group=194)
    rgb = ax.get_images()[0].get_array()[0, 0]
    assert rgb[0] > 0.9 and rgb[1] < 0.2 and rgb[2] < 0.2
