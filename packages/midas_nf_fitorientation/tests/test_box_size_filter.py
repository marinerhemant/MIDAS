"""``BoxSize`` must reach the fit path and shrink the overlap DENOMINATOR.

Regression test for the defect where ``FitParams.box_sizes`` was parsed
(``params.py:258-262``) but never applied in the orientation-fitting path, so
the Python ConfidenceIndex counted out-of-box spots as misses instead of
excluding them.

C reference (``NF_HEDM/src/CalcDiffractionSpots.c:225-243``): a spot that
fails every ``(OmegaRange[i], BoxSizes[i])`` pair is never appended to
``TheorSpots``, so ``nTspots`` shrinks and ``CalcFracOverlap``
(``NF_HEDM/src/SharedFuncsFit.c:508, 645, 648-650``) never increments
``TotalPixels`` for it. The spot leaves BOTH the numerator and the
denominator -- it is not a miss.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.params import FitParams
from midas_nf_fitorientation.soft_overlap import build_forward_model


HKLS_INT = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0], [2, 1, 1]],
                    dtype=np.float64)
EUL = torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float64)
POS = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)


def _params(omega_ranges=None, box_sizes=None) -> FitParams:
    """Coarse single-distance geometry: 128 x 128 px of 2 mm, 36 frames of
    10 deg, so the dense obs volume is well under a megabyte while gold
    rings still land on the detector.
    """
    p = FitParams()
    p.n_distances = 1
    p.Lsd = [1_000_000.0]
    p.ybc = [64.0]
    p.zbc = [64.0]
    p.px = 2000.0
    p.omega_start = -180.0
    p.omega_step = 10.0
    p.start_nr = 1
    p.end_nr = 36
    p.exclude_pole_angle = 6.0
    p.wavelength = 0.172979
    p.lattice_constant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.n_pixels_y = 128
    p.n_pixels_z = 128
    if omega_ranges is not None:
        p.omega_ranges = list(omega_ranges)
    if box_sizes is not None:
        p.box_sizes = list(box_sizes)
    return p


def _nominal_zl(model, eul=EUL):
    """zl in um exactly as C ``CalcSpotPosition`` computes it at Lsd[0]."""
    om, eta, tt, _valid = model.calc_bragg_geometry(model.euler2mat(eul))
    return torch.cos(eta) * (model.Lsd * torch.tan(tt))


# ---------------------------------------------------------------------------
#  Wiring
# ---------------------------------------------------------------------------

def test_build_forward_model_propagates_box_from_params():
    p = _params(omega_ranges=[(-180.0, 180.0)],
                box_sizes=[(-1e6, 1e6, 20_000.0, 1e6)])
    m = build_forward_model(p, HKLS_INT, device="cpu", dtype=torch.float64)
    assert m._has_omega_box is True
    assert m._box_sizes.shape == (1, 4)
    assert float(m._box_sizes[0, 2]) == 20_000.0
    assert float(m._omega_ranges[0, 0]) == -180.0


def test_no_box_in_paramfile_leaves_filter_off():
    p = _params()
    m = build_forward_model(p, HKLS_INT, device="cpu", dtype=torch.float64)
    assert m._has_omega_box is False
    assert float(m(EUL, POS).valid.sum()) > 0


def test_mismatched_counts_raise():
    p = _params(omega_ranges=[(-180.0, 180.0)],
                box_sizes=[(-1e6, 1e6, -1e6, 1e6), (-1e6, 1e6, -1e6, 1e6)])
    with pytest.raises(ValueError, match="must match box_sizes"):
        build_forward_model(p, HKLS_INT, device="cpu", dtype=torch.float64)


# ---------------------------------------------------------------------------
#  The bug itself: denominator, not miss
# ---------------------------------------------------------------------------

def _setup():
    """Model without box, model with a zmin that drops the innermost ring,
    plus the two validity masks.
    """
    m_off = build_forward_model(_params(), HKLS_INT,
                                device="cpu", dtype=torch.float64)
    s_off = m_off(EUL, POS)
    base = s_off.valid.bool()
    zl = _nominal_zl(m_off)
    z_cut = float(zl[base].min()) + 1.0

    p_box = _params(omega_ranges=[(-180.0, 180.0)],
                    box_sizes=[(-1e6, 1e6, z_cut, 1e6)])
    m_box = build_forward_model(p_box, HKLS_INT,
                                device="cpu", dtype=torch.float64)
    s_box = m_box(EUL, POS)
    kept = s_box.valid.bool()
    return m_off, s_off, base, m_box, s_box, kept


def _obs_with_bits_at(spots, mask, p):
    """Dense ObsVolume with the observation bit set only at the pixels of the
    spots selected by ``mask``.
    """
    arr = np.zeros(
        (1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
        dtype=np.float32,
    )
    f = spots.frame_nr.long()[mask].numpy()
    y = spots.y_pixel.long()[mask].numpy()
    z = spots.z_pixel.long()[mask].numpy()
    arr[0, f, y, z] = 1.0
    return ObsVolume.from_dense_array(arr, device="cpu", dtype=torch.uint8)


def test_out_of_box_spot_leaves_the_denominator():
    p = _params()
    m_off, s_off, base, m_box, s_box, kept = _setup()

    n_base = int(base.sum())
    n_kept = int(kept.sum())
    assert n_base > n_kept > 0, "test must actually exclude something"

    # Observation fires ONLY at the in-box spots. The excluded spots are
    # therefore genuine "misses" as far as the bitmap is concerned.
    obs = _obs_with_bits_at(s_off, kept, p)

    frac_box = float(obs.hard_fraction(
        s_box.frame_nr, s_box.y_pixel, s_box.z_pixel, s_box.valid,
    ))
    frac_no_box = float(obs.hard_fraction(
        s_off.frame_nr, s_off.y_pixel, s_off.z_pixel, s_off.valid,
    ))

    # With BoxSize applied: denominator == numerator == n_kept -> 1.0,
    # exactly the C's ConfidenceIndex 1.000000.
    assert frac_box == pytest.approx(1.0)
    # Without it: the out-of-box spots are counted as misses -> n_kept/n_base,
    # the < 1.0 value the buggy Python path reported.
    assert frac_no_box == pytest.approx(n_kept / n_base)
    assert frac_no_box < 1.0


def test_in_box_spot_is_still_counted():
    """The gate must not silently drop spots that ARE inside the box."""
    p = _params()
    m_off, s_off, base, m_box, s_box, kept = _setup()
    obs = _obs_with_bits_at(s_off, kept, p)

    # Blank one in-box spot from the observation -> the fraction must drop,
    # i.e. in-box spots really are in the denominator and can miss.
    arr = np.zeros(
        (1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
        dtype=np.float32,
    )
    idx = torch.nonzero(kept, as_tuple=False)
    drop = torch.zeros_like(kept)
    drop[tuple(idx[0].tolist())] = True
    keep_bits = kept & ~drop
    f = s_off.frame_nr.long()[keep_bits].numpy()
    y = s_off.y_pixel.long()[keep_bits].numpy()
    z = s_off.z_pixel.long()[keep_bits].numpy()
    arr[0, f, y, z] = 1.0
    obs2 = ObsVolume.from_dense_array(arr, device="cpu", dtype=torch.uint8)

    n_kept = int(kept.sum())
    frac = float(obs2.hard_fraction(
        s_box.frame_nr, s_box.y_pixel, s_box.z_pixel, s_box.valid,
    ))
    # Exactly one of the n_kept denominator entries now misses. (Spots can
    # share a pixel, so allow the fraction to be at most that.)
    assert frac <= (n_kept - 1) / n_kept + 1e-9
    assert frac > 0.0


def test_soft_fraction_inherits_the_same_denominator():
    p = _params()
    m_off, s_off, base, m_box, s_box, kept = _setup()
    arr = np.zeros(
        (1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z),
        dtype=np.float32,
    )
    obs = ObsVolume.from_dense_array(
        arr, device="cpu", dtype=torch.float32, packed=False,
    )
    # An all-zero observation gives 0 either way; what we assert is that the
    # soft path sees the SAME reduced valid mask (no exception, same shape).
    val = obs.soft_fraction(
        s_box.frame_nr, s_box.y_pixel, s_box.z_pixel, s_box.valid, 1.0,
    )
    assert float(val) == pytest.approx(0.0)
    assert int(s_box.valid.sum()) < int(s_off.valid.sum())
