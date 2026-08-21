"""The seeder's threshold must not be defeated by one bright pixel.

`make_seed` thresholds at `max(SNR*sigma, percentile_cap)`. `sigma` is MAD-based
and so robust to a handful of outliers, but not to a frame that is broadly hot:
on a no-attenuator Pilatus reference (max 1.16e6 counts) MAD-sigma came out at
10636, putting 4*sigma at 42543 against a percentile cap of 504 — 84x too high,
rejecting every ring pixel. A short 0.3 s GE exposure failed the same way more
mildly (4*sigma = 488 vs cap 254) and then lost its fragmented rings to the arc
length cut.

The ladder relaxes only when the strict setting yields nothing, and records
which rung fired. These tests pin both halves: it must rescue a frame the strict
setting drops, and it must NOT change a frame the strict setting handles.
"""
import numpy as np
import pytest

from midas_calibrate_v2.seed.auto_seed import make_seed, Seed

PX = 200.0
LAMBDA = 12.398419 / 71.676          # Re K edge


def _rings(shape=(1024, 1024), bc=(512.0, 512.0), radii=(150, 240, 330, 410),
           amp=400.0, width=2.0, bg=100.0, seed=0):
    """A synthetic powder pattern: concentric Gaussian rings on a flat bg."""
    rng = np.random.default_rng(seed)
    nz, ny = shape
    Y, X = np.mgrid[0:nz, 0:ny]
    R = np.hypot(Y - bc[1], X - bc[0])
    img = np.full(shape, bg, dtype=np.float32)
    for r0 in radii:
        img += amp * np.exp(-0.5 * ((R - r0) / width) ** 2)
    img += rng.normal(0.0, 3.0, size=shape).astype(np.float32)
    return np.clip(img, 0, None)


def test_clean_frame_seeds_on_the_strict_rung():
    img = _rings()
    s = make_seed(img, wavelength_A=LAMBDA, px_um=PX, calibrant="CeO2",
                  use_diplib=False)
    assert isinstance(s, Seed)
    assert s.threshold_rung == 0, "a clean frame must not need any relaxation"
    assert "rung 0" in s.notes


def test_the_ladder_rescues_a_too_strict_threshold():
    """Force the strict rung to find nothing and check the ladder recovers.

    Driving `snr_threshold` up is a faithful stand-in for the real pathology —
    on the affected frames the sigma arm was 1.9x to 84x too high — and it tests
    the ladder itself rather than a sigma blow-up this synthetic generator
    cannot produce (its MAD stays at ~0.1 whatever saturation is added, because
    the real frames are broadly hot rather than locally saturated).
    """
    img = _rings()
    s = make_seed(img, wavelength_A=LAMBDA, px_um=PX, calibrant="CeO2",
                  use_diplib=False, snr_threshold=5000.0)
    assert s.n_measured >= 2
    assert s.threshold_rung >= 1, "the strict rung should have found nothing here"
    assert "NOT the strict setting" in s.notes


def test_a_relaxed_seed_agrees_with_the_strict_one():
    """Relaxing must recover the SAME geometry, not merely some geometry."""
    img = _rings()
    strict = make_seed(img, wavelength_A=LAMBDA, px_um=PX, calibrant="CeO2",
                       use_diplib=False)
    relaxed = make_seed(img, wavelength_A=LAMBDA, px_um=PX, calibrant="CeO2",
                        use_diplib=False, snr_threshold=5000.0)
    assert strict.threshold_rung == 0 and relaxed.threshold_rung >= 1
    assert abs(strict.BC_y - relaxed.BC_y) < 2.0
    assert abs(strict.BC_z - relaxed.BC_z) < 2.0
    assert abs(strict.Lsd_um - relaxed.Lsd_um) / strict.Lsd_um < 0.02


def test_a_frame_with_no_signal_still_fails():
    """Relaxing far enough will fit noise. It must not go that far.

    Measured motivation: an EIGER frame whose maximum was 12 counts produced
    3 spurious 'arcs' at a relaxed threshold. Refusing is the correct answer.
    """
    rng = np.random.default_rng(1)
    noise = rng.normal(5.0, 1.0, size=(1024, 1024)).astype(np.float32)
    with pytest.raises(RuntimeError, match="no arcs detected"):
        make_seed(np.clip(noise, 0, None), wavelength_A=LAMBDA, px_um=PX,
                  calibrant="CeO2", use_diplib=False)


def test_failure_message_says_the_ladder_was_tried():
    rng = np.random.default_rng(2)
    noise = rng.normal(5.0, 1.0, size=(512, 512)).astype(np.float32)
    with pytest.raises(RuntimeError) as e:
        make_seed(np.clip(noise, 0, None), wavelength_A=LAMBDA, px_um=PX,
                  calibrant="CeO2", use_diplib=False)
    msg = str(e.value)
    assert "relaxation" in msg and "too little signal" in msg


def test_seed_carries_the_rung_field():
    assert "threshold_rung" in Seed.__dataclass_fields__
    assert Seed.__dataclass_fields__["threshold_rung"].default == 0
