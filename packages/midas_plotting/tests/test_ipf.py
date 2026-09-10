"""IPF colouring.

The property that matters: colour is a function of the crystal direction along
a sample axis, so it is invariant under symmetry-equivalent descriptions of the
same orientation. That is what makes one grain one colour, and it is what
Euler->RGB does not give.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_plotting.ipf import CUBIC, HEXAGONAL, ipf_rgb, laue_class, sym_matrices


def test_laue_class_known_families():
    assert laue_class(225) == CUBIC
    assert laue_class(229) == CUBIC
    assert laue_class(194) == HEXAGONAL
    assert laue_class(168) == HEXAGONAL


def test_laue_class_refuses_unimplemented_rather_than_guessing():
    """A silent fallback to cubic would recolour a map with no other symptom.

    SG 139 (tetragonal) and 69 (orthorhombic) were on this list until their
    triangles were implemented; triclinic and monoclinic still are not, and the
    contract is that they REFUSE rather than guess.

    SG 75-88 is the subtle one. It is tetragonal by CRYSTAL SYSTEM but its Laue
    class is 4/m, not 4/mmm: no in-plane mirror, so its triangle spans 90 deg to
    [010] rather than 45 deg to [110]. Colouring it with the 4/mmm ramp folds by
    a 2-fold the crystal does not have and yields a plausible, wrong map. It is
    deliberately excluded until `_rgb_tetragonal_4m` exists.
    """
    for sg in (1, 2, 5, 15):                      # triclinic + monoclinic
        with pytest.raises(NotImplementedError, match="not implemented"):
            laue_class(sg)
    for sg in (75, 81, 88):                       # Laue class 4/m, NOT 4/mmm
        with pytest.raises(NotImplementedError, match="not implemented"):
            laue_class(sg)


def test_tetragonal_range_is_the_laue_class_not_the_crystal_system():
    """4/mmm starts at 89. The boundary is the whole point of the split."""
    from midas_plotting.ipf import laue_class, sector_deg, TETRAGONAL
    with pytest.raises(NotImplementedError):
        laue_class(88)
    assert laue_class(89) == TETRAGONAL
    assert sector_deg(89) == 45.0 and sector_deg(142) == 45.0


def test_tetragonal_and_orthorhombic_triangles_are_distinct():
    """4/mmm and mmm must not be the same colouring, and neither may be cubic.

    The discriminator is [010]: under 4/mmm a 4-fold makes it equivalent to
    [100] (both GREEN), while mmm has no in-plane rotation so [010] is the
    third vertex (BLUE). A tetragonal map coloured with the mmm triangle -- or
    with the cubic one, which folds by a 3-fold the crystal does not have --
    would look plausible and be wrong.
    """
    import numpy as np
    from midas_plotting.ipf import direction_rgb
    dirs = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    tet = direction_rgb(dirs, space_group=139)
    ort = direction_rgb(dirs, space_group=69)
    assert tet[0, 0] > 0.9 and ort[0, 0] > 0.9        # [001] is red in both
    assert np.allclose(tet[1], tet[2], atol=1e-6), "4/mmm: [100] and [010] must match"
    assert not np.allclose(ort[1], ort[2], atol=0.5), "mmm: [100] and [010] must differ"
    assert ort[2, 2] > 0.9, "mmm: [010] is the blue vertex"


def test_sym_matrices_are_proper_rotations():
    for sg, n_expected in ((225, 24), (194, 12)):
        S = sym_matrices(sg)
        assert S.shape == (n_expected, 3, 3), (sg, S.shape)
        dets = np.linalg.det(S)
        np.testing.assert_allclose(dets, 1.0, atol=1e-9)
        for M in S:
            np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-9)


def test_rgb_in_unit_range():
    rng = np.random.default_rng(0)
    e = rng.uniform(0, np.pi, size=(200, 3))
    for sg in (225, 194):
        rgb = ipf_rgb(e, sg)
        assert rgb.shape == (200, 3)
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0
        assert np.isfinite(rgb).all()


def test_cube_on_axis_is_red():
    """The identity orientation puts [001] along Z -> the [001] triangle corner."""
    rgb = ipf_rgb(np.zeros((1, 3)), 225)[0]
    assert rgb[0] > 0.9
    assert rgb[1] < 0.2 and rgb[2] < 0.2


def test_symmetry_equivalent_orientations_get_the_same_colour():
    """The whole point: one grain, one colour.

    Rotating by a symmetry operator describes the SAME crystal, so the colour
    must not move. An Euler->RGB scheme fails this badly.
    """
    from midas_stress.orientation import (
        euler_to_orient_mat_batch, orient_mat_to_euler,
    )

    rng = np.random.default_rng(3)
    e0 = rng.uniform(0, np.pi, size=(12, 3))
    g0 = np.asarray(euler_to_orient_mat_batch(e0)).reshape(-1, 3, 3)
    S = sym_matrices(225)

    base = ipf_rgb(e0, 225)
    for op in S[1:6]:
        # symmetry acts on the RIGHT for crystal->lab matrices (MIDAS convention):
        # midas_stress.misorientation_om_batch calls g.S equivalent and S.g a
        # DIFFERENT orientation (61 deg). Asserting the wrong side is what let the
        # g-vs-g^T bug survive until 2026-09-03.
        g1 = np.einsum("nij,jk->nik", g0, op)
        e1 = np.array([np.asarray(orient_mat_to_euler(m.ravel().tolist())).ravel()
                       for m in g1])
        np.testing.assert_allclose(ipf_rgb(e1, 225), base, atol=1e-6)


def test_empty_input():
    assert ipf_rgb(np.zeros((0, 3)), 225).shape == (0, 3)


def test_zero_axis_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        ipf_rgb(np.zeros((1, 3)), 225, axis=(0, 0, 0))


def test_gamma_one_disables_the_perceptual_lift():
    e = np.array([[0.3, 0.4, 0.5]])
    lin = ipf_rgb(e, 225, gamma=1.0)
    sq = ipf_rgb(e, 225, gamma=0.5)
    np.testing.assert_allclose(sq, np.sqrt(lin), atol=1e-9)


def test_hexagonal_c_axis_along_z_is_red():
    """[0001] || Z is the first corner of the hexagonal triangle."""
    rgb = ipf_rgb(np.zeros((1, 3)), 194)[0]
    assert rgb[0] > 0.9


# --- trigonal (Laue -3m), added for NMC811 R-3m / SG 166 -------------------

def _random_oms(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


@pytest.mark.parametrize("sg", [166, 194, 225])
def test_ipf_colour_is_symmetry_invariant(sg):
    """colour(S.g) == colour(g) for every proper rotation S.

    This is the property that makes an IPF map mean anything: one grain, one
    colour, whichever symmetrically equivalent orientation the indexer
    happened to report. It is also what caught the first -3m triangle, which
    hand-folded the azimuth assuming a 2-fold axis at 0 deg and moved the
    colour by up to 0.99 under this test.
    """
    from midas_plotting.ipf import ipf_rgb_from_matrix, sym_matrices
    g = _random_oms(2000, seed=sg)
    base = ipf_rgb_from_matrix(g, sg)
    for s in sym_matrices(sg):
        got = ipf_rgb_from_matrix(np.einsum("nij,jk->nik", g, s), sg)
        np.testing.assert_allclose(got, base, atol=1e-9)


def test_trigonal_sector_is_twice_the_hexagonal_one():
    """-3m has 6 proper rotations to 6/mmm's 12, so a 60 deg sector, not 30.

    Measured from the operator set, so a change there cannot silently
    mis-scale the ramp.
    """
    from midas_plotting.ipf import _trigonal_sector_deg, sym_matrices
    assert _trigonal_sector_deg(sym_matrices(166)) == pytest.approx(60.0)
    assert _trigonal_sector_deg(sym_matrices(194)) == pytest.approx(30.0)


def test_trigonal_is_not_the_hexagonal_colouring():
    """R-3m must not be quietly coloured as 6/mmm -- that merges orientations."""
    from midas_plotting.ipf import ipf_rgb_from_matrix
    g = _random_oms(3000, seed=7)
    assert np.abs(ipf_rgb_from_matrix(g, 166)
                  - ipf_rgb_from_matrix(g, 194)).mean() > 0.05


def test_laue_class_still_refuses_what_it_cannot_draw():
    from midas_plotting.ipf import laue_class
    assert laue_class(166) == "trigonal"
    with pytest.raises(NotImplementedError):
        laue_class(2)          # triclinic: no triangle implemented
