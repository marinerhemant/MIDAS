"""The seeder's relaxation ladder must be able to relax the THRESHOLD.

``make_seed`` thresholds the background-subtracted frame at
``max(snr_threshold·σ, percentile(diff, 1 - bright_fraction_cap))``, skeletonizes
what survives, and keeps connected components longer than an arc-length cut.
When that finds nothing it walks a relaxation ladder.

Every rung of that ladder used to relax only the **arc-length cut**. That is the
wrong knob for one of the two ways the threshold goes wrong:

  * σ too high (over-exposure, short noisy exposure) — the existing rungs handle
    it, by dropping the σ arm and shortening the cut;
  * **the rings genuinely cover more of the frame than ``bright_fraction_cap``**
    — then the percentile arm *itself* lands inside the peaks. Only the peak
    tops survive, and they skeletonize into a spray of short fragments. No
    arc-length relaxation can repair that, because the shape of the bright set
    is already destroyed.

The bright fraction is roughly (ring width × total circumference) / area, so it
grows as the detector shrinks: a 1 % cap encodes a large-detector assumption.
Measured on the 512×512 frame below, ring pixels are 29 % of the frame, the 1 %
cap puts the threshold at 3979 counts against a 4196-count maximum, and the
rings break into 1778 fragments of ≤ 11 px against a 51 px cut.

The consequence was not a wrong answer — ``calibrate`` fell back to the
chord-only seed and then failed both ``seed_provenance`` and ``strain_cap``,
loudly. It was a *usable frame the seeder could not seed*.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

A_CEO2 = 5.4116
WAVELENGTH = 0.172973


def _small_detector_frame(N=512, px=200.0, Lsd=400_000.0, sigma_px=1.6,
                          peak=4000.0, bg=25.0, seed=0):
    """A frame whose rings cover far more than 1 % of the detector.

    Rendered through ``midas_integrate``'s forward model rather than v2's own,
    so this is not an inverse crime against the package under test.
    """
    from midas_integrate.geometry import pixel_to_REta, build_tilt_matrix

    Yi, Zi = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float),
                         indexing="xy")
    R, _ = pixel_to_REta(Yi, Zi, Ycen=N / 2.0, Zcen=N / 2.0,
                         TRs=build_tilt_matrix(0.0, 0.30, -0.45), Lsd=Lsd,
                         RhoD=float(N) * px, px=px,
                         **{f"p{k}": 0.0 for k in range(15)}, parallax=False)
    img = np.full_like(R, bg)
    seen = set()
    for h in range(9):
        for k in range(9):
            for l in range(9):
                if h == k == l == 0 or not (h % 2 == k % 2 == l % 2):
                    continue
                s2 = h * h + k * k + l * l
                if s2 in seen:
                    continue
                ratio = WAVELENGTH / (2.0 * (A_CEO2 / math.sqrt(s2)))
                if ratio >= 1.0:
                    continue
                seen.add(s2)
                R0 = Lsd * math.tan(2.0 * math.asin(ratio)) / px
                if 25 < R0 < 0.72 * N:
                    img += peak * np.exp(-((R - R0) ** 2) / (2 * sigma_px ** 2))
    return np.random.default_rng(seed).poisson(np.clip(img, 0, None)).astype(float)


def test_the_frame_really_does_exceed_the_default_cap():
    """Guard the premise: if this stops being true the test below is vacuous."""
    from midas_calibrate_v2.seed.auto_seed import _background_subtract

    img = _small_detector_frame()
    diff, _bad, _sigma = _background_subtract(img, kernel_size=101, n_iters=3,
                                              use_diplib=False)
    bright_fraction = float((diff > 100.0).mean())
    assert bright_fraction > 0.05, (
        f"rings cover only {bright_fraction:.1%} of the frame; this frame no "
        f"longer exercises the over-covered regime")


@pytest.mark.slow
def test_seeds_a_frame_whose_rings_exceed_the_bright_fraction_cap():
    from midas_calibrate_v2.seed.auto_seed import make_seed

    img = _small_detector_frame()
    # use_diplib=False: diplib's median filter segfaults on macOS, which is why
    # the pipeline passes the same flag.
    seed = make_seed(img, wavelength_A=WAVELENGTH, px_um=200.0,
                     calibrant="CeO2", use_diplib=False)

    assert abs(seed.BC_y - 256.0) < 3.0
    assert abs(seed.BC_z - 256.0) < 3.0
    assert abs(seed.Lsd_um - 400_000.0) < 0.01 * 400_000.0
    assert seed.threshold_rung >= 4, (
        f"expected one of the wider-cap rungs, got rung {seed.threshold_rung} "
        f"({seed.notes})")


@pytest.mark.slow
def test_a_caller_who_already_widened_the_cap_gets_no_extra_rungs():
    """The wider rungs are conditional on being wider than the caller's cap.

    This is the guard that makes the change safe to add to a validated seeder:
    the rungs are appended AFTER the original four, so they are only reached
    once every existing rung has failed, and they are only added at all when
    they would actually loosen the caller's setting. A caller who has already
    passed a cap wider than both gets exactly the original ladder back.
    """
    from midas_calibrate_v2.seed.auto_seed import make_seed

    img = _small_detector_frame()
    seed = make_seed(img, wavelength_A=WAVELENGTH, px_um=200.0,
                     calibrant="CeO2", use_diplib=False,
                     bright_fraction_cap=0.20)
    assert seed.threshold_rung < 4, (
        f"rung {seed.threshold_rung} exists but the caller's cap (20 %) is "
        f"already wider than both added rungs, so none should have been added "
        f"({seed.notes})")
    # And it must still be a good seed on the widened cap.
    assert abs(seed.BC_y - 256.0) < 3.0
    assert abs(seed.Lsd_um - 400_000.0) < 0.01 * 400_000.0
