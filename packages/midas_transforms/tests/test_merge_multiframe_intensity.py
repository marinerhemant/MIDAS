"""A spot's merged IntegratedIntensity must not depend on how many omega frames it spans.

End-to-end through the real per-frame chain (midas_peakfit threshold -> connected
regions -> seed -> pseudo-Voigt fit -> integrated intensity) and the real cross-frame
merge (``_merge_frames``). The same total intensity is spread over N = 1..20 frames
(a high-Lorentz-factor spot is exactly this: slow through the Ewald sphere, so the
same integrated intensity lands on more frames). If the merge dropped frames, capped
the omega window, or re-thresholded the tails, the merged intensity would fall with N.

Written for the AlON 1-ID investigation (2026-09-22) where log(I/LP) vs log L had a
slope of -0.55 for large grains. This test shows the multi-frame path is not the cause
for bright spots; there, the slope came from saturation-dropped regions (see
``midas_peakfit.seeds.find_regional_maxima``) leaving tiny fragments that were matched
-- now recorded and flagged, see test_merge_saturation.py.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pf = pytest.importorskip("midas_peakfit")
torch = pytest.importorskip("torch")
from scipy.special import erf  # noqa: E402

from midas_peakfit.connected import filter_regions_by_size, find_regions  # noqa: E402
from midas_peakfit.fit import fit_regions  # noqa: E402
from midas_peakfit.lm import LMConfig  # noqa: E402
from midas_peakfit.preprocess import apply_threshold  # noqa: E402
from midas_peakfit.seeds import seed_region  # noqa: E402

from midas_transforms.merge.core import _merge_frames  # noqa: E402

NP = 128
YC = ZC = 64.0
Y0, Z0 = 64.0 + 30.0, 64.0 + 22.0   # a spot at R ~ 37 px
SIG = 1.5                            # in-plane Gaussian sigma, px


def _omega_weights(n_span: int, pad: int = 6) -> np.ndarray:
    """Per-frame fraction of a Gaussian omega profile whose +-2 sigma covers n_span frames."""
    so = max(n_span / 4.0, 1e-3)
    nf = n_span + 2 * pad
    edges = np.arange(nf + 1) - nf / 2.0
    w = np.diff(0.5 * (1.0 + erf(edges / (so * math.sqrt(2.0)))))
    return w / w.sum()


def _merged_intensity(total: float, n_span: int, bg: float, thresh: float) -> np.ndarray:
    yy, zz = np.mgrid[0:NP, 0:NP].astype(float)
    shape = np.exp(-0.5 * ((yy - Y0) ** 2 + (zz - Z0) ** 2) / SIG ** 2)
    shape /= shape.sum()                      # exact discrete normalisation
    good = np.full((NP, NP), thresh)
    frames = []
    for k, wk in enumerate(_omega_weights(n_span)):
        img = apply_threshold(bg + total * wk * shape, good)
        regs = filter_regions_by_size(find_regions(img, good), 1, 10000)
        srs = [s for s in (seed_region(r, img, None, Ycen=YC, Zcen=ZC, int_sat=1e12,
                                       max_n_peaks=50, panels=[]) for r in regs)
               if s is not None]
        outs, _ = fit_regions(srs, omega=0.25 * k, Ycen=YC, Zcen=ZC, do_peak_fit=1,
                              local_maxima_only=0, device=torch.device("cpu"),
                              dtype=torch.float64, lm_config=LMConfig())
        frames.append(np.concatenate([o.rows for o in outs]) if outs
                      else np.zeros((0, 29)))
    merged, _ = _merge_frames(frames, overlap_length=2.0)
    return merged


SPANS = (1, 2, 4, 8, 12, 16, 20)


def _span_series(bg: float, thresh: float, total: float = 1.0e6) -> np.ndarray:
    ii = []
    for n in SPANS:
        merged = _merged_intensity(total, n, bg, thresh)
        # One spot, one chain: the merge must not fragment a stationary spot.
        assert merged.shape[0] == 1, f"N={n}: spot split into {merged.shape[0]} merged spots"
        ii.append(merged[0, 1])
    return np.asarray(ii)


def test_merged_intensity_independent_of_frame_count():
    """Dark-subtracted frames (the 1-ID GE case: RingThresh ~10 counts over ~0 pedestal).

    Bright spot, per-frame peak >> threshold even at N=20: the merged intensity must equal
    the truth within 3 % for every span, and so be flat in N.
    """
    total = 1.0e6
    ii = _span_series(bg=0.0, thresh=10.0, total=total)
    rel = ii / total - 1.0
    assert np.all(np.abs(rel) < 0.03), dict(zip(SPANS, np.round(rel, 4)))


def test_pedestal_biases_multiframe_intensity_up_not_down():
    """With a flat pedestal under the spot the merged intensity is NOT flat in N.

    ``model.integrated_intensity`` (a port of the C ``calculateIntegratedIntensity``) adds
    the fitted background to every pixel where the peak exceeds it, so each extra frame
    adds bg * footprint: +5 % at N=20 here. That is an upward bias, i.e. the opposite sign
    to a loss of intensity with frame count. Pin both facts: nothing is lost, and the
    pedestal term stays bounded. If this starts failing low, frames are being dropped.
    """
    ii = _span_series(bg=100.0, thresh=200.0)
    rel = ii / ii[0] - 1.0
    assert np.all(rel > -0.03), dict(zip(SPANS, np.round(rel, 4)))
    assert np.all(rel < 0.08), dict(zip(SPANS, np.round(rel, 4)))
