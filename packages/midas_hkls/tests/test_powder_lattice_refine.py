"""Tests for refine_lattice_from_d_spacings — powder determination of the cell.

The point of this routine is that it is a DIRECT least squares on 1/d^2, which is
linear in the reciprocal metric tensor. So there is no starting guess, and the
answer cannot drift toward whatever cell a per-grain refinement was seeded with.
"""

import numpy as np
import pytest

from midas_hkls.lattice import Lattice, refine_lattice_from_d_spacings


HEX_HKLS = [(0, 0, 3), (1, 0, 1), (0, 1, 2), (1, 0, 4), (1, 1, 0),
            (1, 0, 5), (1, 1, 3), (0, 2, 1), (2, 0, 2), (1, 0, 7)]
CUBIC_HKLS = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2),
              (4, 0, 0), (3, 3, 1), (4, 2, 0)]


def _d(lat, hkls):
    return np.array([lat.d_spacing(*h) for h in hkls])


def test_recovers_a_planted_hexagonal_cell_exactly():
    truth = Lattice.for_system("hexagonal", a=2.85074, c=14.32299)
    fit = refine_lattice_from_d_spacings(HEX_HKLS, _d(truth, HEX_HKLS), "hexagonal")
    assert fit.lattice.a == pytest.approx(truth.a, rel=1e-10)
    assert fit.lattice.c == pytest.approx(truth.c, rel=1e-10)
    assert fit.rms_strain < 1e-12
    assert fit.n_reflections == len(HEX_HKLS)


def test_no_starting_guess_dependence():
    """The whole point: the API takes no initial cell, so it cannot drift."""
    import inspect
    sig = inspect.signature(refine_lattice_from_d_spacings)
    assert set(sig.parameters) == {"hkls", "d_obs", "system", "weights"}


def test_recovers_cell_under_noise_without_bias():
    truth = Lattice.for_system("hexagonal", a=2.85074, c=14.32299)
    d0 = _d(truth, HEX_HKLS)
    rng = np.random.default_rng(0)
    a_err, c_err = [], []
    for _ in range(200):
        d = d0 * (1 + rng.normal(0, 200e-6, d0.size))   # 200 ue scatter
        f = refine_lattice_from_d_spacings(HEX_HKLS, d, "hexagonal")
        a_err.append(f.lattice.a / truth.a - 1)
        c_err.append(f.lattice.c / truth.c - 1)
    # unbiased: the mean error is far below the per-reflection scatter
    assert abs(np.mean(a_err)) < 3e-5
    assert abs(np.mean(c_err)) < 3e-5


def test_uniform_dilatation_is_recovered_as_a_cell_change():
    """A wrong wavelength/distance scales every d; the cell absorbs it (rule 9)."""
    truth = Lattice.for_system("hexagonal", a=2.85074, c=14.32299)
    d = _d(truth, HEX_HKLS) * 1.001
    fit = refine_lattice_from_d_spacings(HEX_HKLS, d, "hexagonal")
    assert fit.lattice.a == pytest.approx(truth.a * 1.001, rel=1e-9)
    assert fit.lattice.c == pytest.approx(truth.c * 1.001, rel=1e-9)
    assert fit.rms_strain < 1e-12       # still a perfect fit -- hence degenerate


def test_cubic_and_tetragonal_and_orthorhombic():
    cub = Lattice.for_system("cubic", a=5.4116)
    f = refine_lattice_from_d_spacings(CUBIC_HKLS, _d(cub, CUBIC_HKLS), "cubic")
    assert f.lattice.a == pytest.approx(5.4116, rel=1e-10)

    tet = Lattice.for_system("tetragonal", a=3.9, c=4.6)
    hk = [(1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (2, 0, 0), (1, 1, 2)]
    f = refine_lattice_from_d_spacings(hk, _d(tet, hk), "tetragonal")
    assert f.lattice.a == pytest.approx(3.9, rel=1e-10)
    assert f.lattice.c == pytest.approx(4.6, rel=1e-10)

    ort = Lattice.for_system("orthorhombic", a=4.0, b=5.0, c=6.0)
    hk = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1),
          (2, 1, 1)]
    f = refine_lattice_from_d_spacings(hk, _d(ort, hk), "orthorhombic")
    assert (f.lattice.a, f.lattice.b, f.lattice.c) == pytest.approx(
        (4.0, 5.0, 6.0), rel=1e-9)


def test_residuals_expose_a_wrong_hkl_assignment():
    truth = Lattice.for_system("hexagonal", a=2.85074, c=14.32299)
    d = _d(truth, HEX_HKLS).copy()
    d[3] *= 1.02                      # one badly mis-assigned ring
    fit = refine_lattice_from_d_spacings(HEX_HKLS, d, "hexagonal")
    assert fit.rms_strain > 1e-3
    assert int(np.argmax(np.abs(fit.residual_strain))) == 3


def test_sigma_shrinks_with_more_reflections():
    truth = Lattice.for_system("hexagonal", a=2.85074, c=14.32299)
    rng = np.random.default_rng(3)
    d = _d(truth, HEX_HKLS) * (1 + rng.normal(0, 300e-6, len(HEX_HKLS)))
    few = refine_lattice_from_d_spacings(HEX_HKLS[:4], d[:4], "hexagonal")
    many = refine_lattice_from_d_spacings(HEX_HKLS, d, "hexagonal")
    assert many.sigma["a"] < few.sigma["a"]
    assert np.isfinite(many.sigma["c"])


def test_rejects_bad_input():
    truth = Lattice.for_system("hexagonal", a=2.85, c=14.3)
    with pytest.raises(ValueError, match="unknown crystal system"):
        refine_lattice_from_d_spacings(HEX_HKLS, _d(truth, HEX_HKLS), "dodecagonal")
    with pytest.raises(ValueError, match="same length"):
        refine_lattice_from_d_spacings(HEX_HKLS, [1.0, 2.0], "hexagonal")
    with pytest.raises(ValueError, match="at least 2 reflections"):
        refine_lattice_from_d_spacings([(0, 0, 3)], [4.7], "hexagonal")
    with pytest.raises(ValueError, match="positive"):
        refine_lattice_from_d_spacings([(0, 0, 3), (1, 0, 1)], [4.7, -1.0],
                                       "hexagonal")


def test_weights_are_honoured():
    truth = Lattice.for_system("hexagonal", a=2.85074, c=14.32299)
    d = _d(truth, HEX_HKLS).copy()
    d[0] *= 1.05                                   # one very bad reflection
    w = np.ones(len(HEX_HKLS)); w[0] = 1e-8        # ...down-weighted to nothing
    fit = refine_lattice_from_d_spacings(HEX_HKLS, d, "hexagonal", weights=w)
    assert fit.lattice.a == pytest.approx(truth.a, rel=1e-6)
    assert fit.lattice.c == pytest.approx(truth.c, rel=1e-6)
