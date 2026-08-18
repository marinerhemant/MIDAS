"""The 180-degree parity separator: rotation is odd, everything static is even."""
import numpy as np
import pytest

from midas_dct_tt.parity import (antipodal_pairs, even_power_fraction,
                                 parity_correlation)


def _scene(rng, n=48):
    """A per-voxel scalar field, projected once; used for both parities."""
    f = rng.normal(size=(n, n))
    for _ in range(3):                      # smooth it a little
        f = 0.25 * (np.roll(f, 1, 0) + np.roll(f, -1, 0)
                    + np.roll(f, 1, 1) + np.roll(f, -1, 1))
    return f


def _stack(base, psi, odd):
    """Build views: antipodes are mirrored in u, and negated iff `odd`."""
    ims = []
    for p in psi:
        im = base if (p % 360.0) < 180.0 else base[:, ::-1] * (-1.0 if odd else 1.0)
        ims.append(im)
    return np.stack(ims)


PSI = np.arange(0.0, 360.0, 15.0)


def test_pairs_found_once_each():
    pairs = antipodal_pairs(PSI)
    assert len(pairs) == len(PSI) // 2
    for i, j in pairs:
        assert abs((PSI[j] - PSI[i]) % 360.0 - 180.0) < 1e-9


def test_pure_rotation_is_odd():
    rng = np.random.default_rng(0)
    rho, per, pairs = parity_correlation(_stack(_scene(rng), PSI, odd=True), PSI)
    assert len(pairs) == len(PSI) // 2
    assert rho == pytest.approx(-1.0, abs=1e-6)
    assert np.all(per < 0)
    assert even_power_fraction(rho) == pytest.approx(0.0, abs=1e-6)


def test_static_scalar_is_even():
    rng = np.random.default_rng(1)
    rho, _, _ = parity_correlation(_stack(_scene(rng), PSI, odd=False), PSI)
    assert rho == pytest.approx(1.0, abs=1e-6)
    assert even_power_fraction(rho) == pytest.approx(1.0, abs=1e-6)


def test_mixture_recovers_the_even_power_fraction():
    """A known rotation/strain mix is recovered to within a few percent."""
    rng = np.random.default_rng(2)
    a, b = _scene(rng), _scene(rng)
    a /= a.std()
    b /= b.std()
    for freq in (0.1, 0.25, 0.5):
        amp = np.sqrt(freq / (1.0 - freq))          # even/odd amplitude ratio
        ims = []
        for p in PSI:
            if (p % 360.0) < 180.0:
                ims.append(a + amp * b)
            else:
                ims.append(-a[:, ::-1] + amp * b[:, ::-1])
            # ^ odd part flips sign, even part does not
        rho, _, _ = parity_correlation(np.stack(ims), PSI)
        assert even_power_fraction(rho) == pytest.approx(freq, abs=0.05)


def test_validity_mask_is_intersected():
    rng = np.random.default_rng(3)
    ims = _stack(_scene(rng), PSI, odd=True)
    valid = np.ones(ims.shape, bool)
    valid[:, :24, :] = False                        # blank ROWS: not mirrored
    rho, _, pairs = parity_correlation(ims, PSI, valid=valid)
    assert len(pairs) == len(PSI) // 2              # still enough pixels
    assert rho == pytest.approx(-1.0, abs=1e-6)

    # blanking COLUMNS instead is the interesting case: the mirror maps the
    # surviving half onto the blanked half, so the intersection empties and the
    # pair is correctly dropped rather than silently scored on nothing.
    vcol = np.ones(ims.shape, bool)
    vcol[:, :, :24] = False
    _, _, pcol = parity_correlation(ims, PSI, valid=vcol)
    assert pcol == []


def test_pairs_below_min_pixels_are_skipped():
    rng = np.random.default_rng(4)
    ims = _stack(_scene(rng), PSI, odd=True)
    valid = np.zeros(ims.shape, bool)
    valid[:, :2, :2] = True
    rho, per, pairs = parity_correlation(ims, PSI, valid=valid, min_pixels=100)
    assert pairs == [] and per.size == 0 and np.isnan(rho)


def test_shape_is_validated():
    with pytest.raises(ValueError):
        parity_correlation(np.zeros((4, 4)), [0, 180])
    with pytest.raises(ValueError):
        parity_correlation(np.zeros((2, 4, 4)), [0, 180], valid=np.ones((2, 4), bool))
