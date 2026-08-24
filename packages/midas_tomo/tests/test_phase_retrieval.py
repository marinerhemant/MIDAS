"""Paganin phase retrieval, and the null that makes it safe to enable.

The load-bearing test is the null: ``delta_beta = 0`` must return the input
**bit-identically**, so switching the filter on cannot change a reconstruction
except through the parameter that was deliberately set.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_tomo.phase_retrieval import (
    delta_beta_from_materials,
    paganin_filter,
    sweep_delta_beta,
    wavelength_um,
)

GEOM = dict(pixel_size_um=0.708, distance_mm=100.0, energy_kev=51.93)


def _edge_enhanced(ny=64, nx=64, edge_px=1.0):
    """A disc whose boundary carries a bright/dark fringe pair.

    That over/undershoot at the edge is what propagation contrast looks like,
    and it is what the filter has to turn back into a filled object.
    """
    iy, ix = np.mgrid[0:ny, 0:nx].astype(np.float64)
    r = np.hypot(ix - (nx - 1) / 2, iy - (ny - 1) / 2)
    inside = 1.0 - 0.02 / (1.0 + np.exp((r - 18.0) / 0.8))
    fringe = 0.06 * np.exp(-((r - 18.0) / edge_px) ** 2) * np.sign(r - 18.0)
    return inside + fringe


# ------------------------------------------------------------------ the null

@pytest.mark.parametrize("shape", [(64, 64), (3, 32, 40)])
def test_delta_beta_zero_is_BIT_IDENTICAL(shape):
    """N1: enabling the filter with no strength must change nothing at all.

    Not 'close' -- identical, so a run with the filter available but off is
    provably the same run as before it existed.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 1.0, shape)
    out = paganin_filter(p, delta_beta=0.0, **GEOM)
    assert np.array_equal(out, p)


def test_zero_propagation_distance_is_also_the_identity():
    """Contact imaging has no phase contrast to retrieve."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 1.0, (16, 16))
    out = paganin_filter(p, delta_beta=800.0, pixel_size_um=0.708,
                         distance_mm=0.0, energy_kev=51.93)
    assert np.array_equal(out, p)


# ------------------------------------------------------------- it does work

def _propagate(transmission, delta_beta, *, pixel_size_um, distance_mm,
               energy_kev):
    """The forward problem: what the detector would see.

    Exactly the inverse of the filter -- multiply by the same denominator in
    Fourier space instead of dividing. Building the fixture this way makes the
    test a genuine round trip rather than a guess at what a fringe looks like.
    """
    lam = wavelength_um(energy_kev)
    z = distance_mm * 1000.0
    ny, nx = transmission.shape
    fy = np.fft.fftfreq(ny, d=pixel_size_um)
    fx = np.fft.fftfreq(nx, d=pixel_size_um)
    f2 = fy[:, None] ** 2 + fx[None, :] ** 2
    denom = 1.0 + math.pi * lam * z * delta_beta * f2
    return np.fft.ifft2(np.fft.fft2(transmission) * denom).real


def test_the_filter_INVERTS_propagation_round_trip():
    """The real test: synthesise what propagation would do to a known filled
    disc, then require the filter to bring the disc back."""
    ny = nx = 64
    iy, ix = np.mgrid[0:ny, 0:nx].astype(np.float64)
    r = np.hypot(ix - 31.5, iy - 31.5)
    truth = 1.0 - 0.05 / (1.0 + np.exp((r - 18.0) / 0.8))   # a filled disc

    db = 500.0
    seen = _propagate(truth, db, **GEOM)
    # propagation must actually have done something visible
    assert np.abs(seen - truth).max() > 1e-3

    back = paganin_filter(seen, delta_beta=db, pad_frac=0.0, **GEOM)
    assert np.abs(back - truth).max() < 1e-6, \
        f"round trip left {np.abs(back - truth).max():.3g}"


def test_the_filter_recovers_interior_contrast_from_a_pure_fringe():
    """The point of the method, stated on the quantity that matters: an image
    whose only signal is at the boundary must come back with a filled core."""
    ny = nx = 64
    iy, ix = np.mgrid[0:ny, 0:nx].astype(np.float64)
    r = np.hypot(ix - 31.5, iy - 31.5)
    truth = 1.0 - 0.05 / (1.0 + np.exp((r - 18.0) / 0.8))
    db = 500.0
    seen = _propagate(truth, db, **GEOM)

    core, air = r < 10.0, r > 26.0
    want = truth[air].mean() - truth[core].mean()
    before = seen[air].mean() - seen[core].mean()
    after = paganin_filter(seen, delta_beta=db, pad_frac=0.0, **GEOM)
    got = after[air].mean() - after[core].mean()

    # Note the direction: propagation slightly INFLATES this difference (edge
    # enhancement leaking into the windows), so the filter brings it back down
    # rather than up. The claim is recovery of the truth, not an increase.
    assert got == pytest.approx(want, rel=0.02)
    assert abs(got - want) < abs(before - want) / 10.0


def test_more_delta_beta_smooths_more():
    """It multiplies f^2, so it is a low-pass whose strength IS the parameter.
    Pinned because it is the reason the value cannot be picked by eye: it
    directly sets how big the specimen comes out."""
    img = _edge_enhanced()
    grads = []
    for db in (10.0, 300.0, 3000.0):
        out = paganin_filter(img, delta_beta=db, **GEOM)
        gy, gx = np.gradient(out)
        grads.append(float(np.mean(np.hypot(gy, gx))))
    assert grads[0] > grads[1] > grads[2]


def test_the_filter_preserves_the_mean():
    """The zero-frequency term is divided by exactly 1, so the DC level is
    untouched; a filter that changed it would rescale every transmission."""
    img = _edge_enhanced()
    # pad_frac=0 so the transform is over exactly this image: the zero
    # frequency is then divided by exactly 1. With padding the padded DC is
    # what is preserved, and the crop's mean legitimately differs.
    out = paganin_filter(img, delta_beta=1500.0, pad_frac=0.0, **GEOM)
    assert out.mean() == pytest.approx(img.mean(), rel=1e-9)


def test_padding_stops_the_opposite_edge_wrapping_in():
    """Without padding the FFT wraps, putting a rim on the reconstruction that
    reads as a sample boundary."""
    img = np.ones((64, 64))
    img[:, :32] = 0.5                      # a strong step across the frame
    padded = paganin_filter(img, delta_beta=3000.0, pad_frac=0.5, **GEOM)
    wrapped = paganin_filter(img, delta_beta=3000.0, pad_frac=0.0, **GEOM)
    # the right-hand edge is far from the step, so it should stay near 1.0
    assert abs(padded[:, -1].mean() - 1.0) < abs(wrapped[:, -1].mean() - 1.0)


# --------------------------------------------------------------- the refusals

def test_a_negative_delta_beta_is_refused():
    with pytest.raises(ValueError, match="delta_beta must be >= 0"):
        paganin_filter(np.ones((8, 8)), delta_beta=-1.0, **GEOM)


def test_a_bad_pixel_size_or_energy_is_refused():
    with pytest.raises(ValueError, match="pixel_size_um must be > 0"):
        paganin_filter(np.ones((8, 8)), delta_beta=1.0, pixel_size_um=0.0,
                       distance_mm=100.0, energy_kev=50.0)
    with pytest.raises(ValueError, match="energy_kev must be > 0"):
        wavelength_um(0.0)


def test_a_wrong_rank_input_is_refused():
    with pytest.raises(ValueError, match="must be 2-D or 3-D"):
        paganin_filter(np.ones((2, 2, 2, 2)), delta_beta=1.0, **GEOM)


# --------------------------------------------------------------- the physics

def test_the_wavelength_is_right():
    """12.398 keV.A, the number everyone checks against."""
    assert wavelength_um(12.398419) == pytest.approx(1.0e-4, rel=1e-5)  # 1 A
    assert wavelength_um(51.93) * 1e4 == pytest.approx(0.2388, abs=2e-4)


def test_delta_beta_can_be_estimated_from_the_specimen():
    """Provided so the parameter can be estimated rather than dialled."""
    db = delta_beta_from_materials(mu_per_um=6.53e-4,
                                   electron_density_per_um3=1.4e12,
                                   energy_kev=51.93)
    assert db > 0 and math.isfinite(db)
    # halving mu doubles delta/beta
    db2 = delta_beta_from_materials(mu_per_um=3.265e-4,
                                    electron_density_per_um3=1.4e12,
                                    energy_kev=51.93)
    assert db2 == pytest.approx(2 * db, rel=1e-9)


# ----------------------------------------------------------------- the sweep

def test_the_sweep_reports_every_score_not_just_the_winner():
    img = _edge_enhanced()
    r = sweep_delta_beta(img, [0.0, 100.0, 1000.0], score=lambda a: -a.std(),
                         **GEOM)
    assert len(r["scores"]) == 3
    assert r["best_delta_beta"] in (0.0, 100.0, 1000.0)


def test_the_sweep_flags_a_winner_at_the_end_of_the_range():
    """A best value at an endpoint means the range did not bracket an optimum,
    so the 'best' is just the largest value tried."""
    img = _edge_enhanced()
    r = sweep_delta_beta(img, [10.0, 100.0, 1000.0], score=lambda a: -a.std(),
                         **GEOM)
    assert r["monotonic"] is True


def test_the_sweep_needs_more_than_one_value():
    with pytest.raises(ValueError, match="at least two"):
        sweep_delta_beta(np.ones((8, 8)), [1.0], score=lambda a: 0.0, **GEOM)
