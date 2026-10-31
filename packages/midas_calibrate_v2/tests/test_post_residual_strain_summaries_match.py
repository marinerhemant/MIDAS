"""``post_residual_strain_uE`` and its median / trimmed siblings describe ONE set of residuals.

``AutoCalibrationResult`` documents the median and the 5 %-trimmed mean as "robust summaries of the
SAME residuals as ``post_residual_strain_uE``".  They were not: when the empirical residual map is kept,
the plain mean is taken AFTER the map (``single.py`` post-MAP block) while the median and trimmed mean
were read from ``history[-1]``, the last iteration BEFORE the map.  On a real GE frame that gave a mean of
28.3 ue next to a median of 606.5 ue and a trimmed mean of 1439.2 ue (the 5 %-trimmed mean of one set of absolute
residuals cannot exceed its plain mean; the median can, for a left-skewed set, so only the trimmed mean is tested); the
Varex frame gave a mean of 4.2 beside a median of 5.2.

The synthetic frame carries a radial distortion (p2 = 4e-4) that the fit with distortion off cannot absorb,
so the map is kept and changes the strain.  The invariant tested: the 5 %-trimmed mean (mean after dropping the worst 5 %) never exceeds the plain mean
of the same set.  Mixed sets can break it, one set cannot.  The synthetic frame is that of
``test_basin_check_not_vacuous`` with that distortion added.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest


def _frame(p2=0.0):
    from midas_integrate.geometry import pixel_to_REta, build_tilt_matrix
    N, px, LSD, WL, A = 512, 200.0, 400_000.0, 0.172973, 5.4116
    Yi, Zi = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float), indexing="xy")
    R, _ = pixel_to_REta(Yi, Zi, Ycen=N / 2.0 + 1.3, Zcen=N / 2.0 - 0.8,
                         TRs=build_tilt_matrix(0.0, 0.0, 0.0), Lsd=LSD, RhoD=float(N) * px, px=px,
                         **{f"p{k}": (p2 if k == 2 else 0.0) for k in range(15)}, parallax=False)
    img = np.full_like(R, 25.0)
    for s2 in (3, 4, 8, 11, 12, 16, 19, 20, 24, 27):
        ratio = WL / (2.0 * (A / math.sqrt(s2)))
        R0 = LSD * math.tan(2.0 * math.asin(ratio)) / px
        if 25 < R0 < 0.72 * N:
            img += 4000.0 * np.exp(-((R - R0) ** 2) / (2 * 1.6 ** 2))
    return np.random.default_rng(0).poisson(np.clip(img, 0, None)).astype(float), WL, px


@pytest.mark.slow
def test_median_and_trim_are_of_the_same_residuals_as_the_plain_mean():
    from midas_calibrate_v2 import calibrate
    # the frame carries a radial distortion the (switched-off) distortion fit cannot absorb, so the residual map is
    # kept and moves the strain; on a distortion-free frame the pre- and post-map sets nearly coincide and the
    # test would not bite (released 0.24.0: 24.3 plain / 22.4 trimmed there, 22.5 plain / 37.2 trimmed here)
    img, WL, px = _frame(p2=4e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = calibrate(img, wavelength=WL, pxY=px, pxZ=px, calibrant="CeO2", refine_tilts=True,
                        refine_distortion=False, build_residual_corr=True, verbose=False, n_iter=3)
    mean, med, trim = (res.post_residual_strain_uE, res.post_residual_strain_median_uE,
                       res.post_residual_strain_trim_uE)
    assert mean is not None and med is not None and trim is not None
    assert trim <= mean * (1 + 1e-9) + 1e-9, (
        f"trimmed mean {trim:.3f} exceeds the plain mean {mean:.3f}: the summaries are of different "
        f"residual sets (map kept: {res.residual_corr_map is not None}, in-loop {res.in_loop_strain_uE:.3f})")
    assert res.residual_corr_map is not None, "the residual map was not kept: the test does not exercise the post-map path"
    assert med <= mean * 2.0 + 1e-9, f"median {med:.3f} vs mean {mean:.3f}"
