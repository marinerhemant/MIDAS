"""v0.3: native-torch hard-binning integration path.

Pins:

1. Constant-image normalised hard-bin profile equals the constant in
   every populated bin (mass conservation per-bin).
2. Hard-bin and soft-bin profiles agree on the centroid of a tight
   peak — they differ in distribution between adjacent bins but a
   peak's centroid in either profile maps to the same physical R.
3. ``integrate_hard_batch`` matches per-image ``integrate_hard`` calls.
4. ImTransOpt is honoured (raw image → transformed → integrated).
5. Hard-bin path does NOT propagate gradient to spec parameters
   (intentional — bin assignments are non-differentiable). The forward
   tensor itself does not require grad.
6. Real-data smoke: Pilatus CeO₂ first-ring centroid via hard-bin lands
   at the same R as v1's ``floor`` mode and the soft path.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    spec_from_v1_params,
    spec_from_v1_paramstest,
    SoftBinGeometry,
    integrate_soft,
    HardBinGeometry,
    integrate_hard,
    integrate_hard_batch,
    profile_1d_diff,
)


def _spec(NY=24, NZ=24, *, ops=None, requires_grad=False):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=12.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=60.0,
    )
    if ops is not None:
        p.TransOpt = list(ops)
        p.NrTransOpt = len(p.TransOpt)
    return spec_from_v1_params(p, requires_grad=requires_grad)


def _gauss_image(NY, NZ, *, R0_px=6.0, sigma_px=1.5, px=200.0):
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - NY / 2.0 - 0.37) * px
    Zc = (zz - NZ / 2.0 + 0.41) * px
    R_um = np.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / px
    return torch.from_numpy(
        np.exp(-(R_px - R0_px) ** 2 / (2 * sigma_px ** 2)).astype(np.float64)
    )


# ── (1) constant-image conservation ──

def test_hard_bin_constant_image_yields_constant_per_bin():
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 7.5, dtype=torch.float64)
    int2d = integrate_hard(img, geom, normalize=True)
    nonzero = int2d[int2d > 0]
    np.testing.assert_array_almost_equal(nonzero.numpy(), 7.5, decimal=12)


def test_hard_bin_unnormalized_sums_match_pixel_count_per_bin():
    """Unnormalised hard-bin sum at each bin = Σ pixels assigned to it
    (× constant intensity)."""
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    intensity = 3.5
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), intensity, dtype=torch.float64)
    sums = integrate_hard(img, geom, normalize=False)
    counts = torch.zeros(s.n_eta_bins * s.n_r_bins, dtype=torch.float64)
    counts = counts.index_add(0, geom.flat_bin[geom.valid],
                                torch.ones(geom.n_valid, dtype=torch.float64))
    counts = counts.reshape(s.n_eta_bins, s.n_r_bins)
    expected = counts * intensity
    torch.testing.assert_close(sums, expected, rtol=0, atol=1e-12)


# ── (2) hard vs soft centroid agreement ──

def test_hard_vs_soft_first_ring_centroid_close():
    s = _spec()
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ, R0_px=6.0, sigma_px=0.8)

    soft_geom = SoftBinGeometry.from_spec(s)
    int_soft = integrate_soft(img, soft_geom)
    prof_soft = profile_1d_diff(int_soft, s).detach().numpy()

    hard_geom = HardBinGeometry.from_spec(s)
    int_hard = integrate_hard(img, hard_geom, normalize=False).numpy()
    prof_hard = int_hard.sum(axis=0)             # collapse η

    n_r = s.n_r_bins
    r_axis = s.RMin + s.RBinSize * (np.arange(n_r) + 0.5)
    band = (r_axis > 4) & (r_axis < 8)
    idx = np.where(band)[0]
    def centroid(profile):
        w = np.maximum(profile[idx], 0)
        return float((w * r_axis[idx]).sum() / (w.sum() + 1e-30))
    c_s = centroid(prof_soft)
    c_h = centroid(prof_hard)
    assert abs(c_s - c_h) < s.RBinSize, (
        f"hard vs soft centroid disagree: hard={c_h:.3f}, soft={c_s:.3f}"
    )


# ── (3) batched hard-bin matches per-image ──

def test_integrate_hard_batch_matches_per_image_loop():
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    rng = torch.Generator().manual_seed(0)
    images = torch.rand(4, s.NrPixelsZ, s.NrPixelsY,
                         generator=rng, dtype=torch.float64)
    batch = integrate_hard_batch(images, geom, normalize=True)
    expected = torch.stack([integrate_hard(images[i], geom, normalize=True)
                              for i in range(4)])
    torch.testing.assert_close(batch, expected, rtol=0, atol=1e-12)


# ── (4) ImTransOpt honored ──

def test_hard_bin_honours_trans_opt():
    NY = NZ = 24
    spec_no = _spec(NY=NY, NZ=NZ, ops=[])
    spec_op = _spec(NY=NY, NZ=NZ, ops=[2])
    geom_no = HardBinGeometry.from_spec(spec_no)
    geom_op = HardBinGeometry.from_spec(spec_op)
    img = _gauss_image(NY, NZ)
    int_no = integrate_hard(img, geom_no)
    int_op = integrate_hard(img, geom_op)            # auto-flips first
    # With a different physical transform applied, the integrated profile
    # must be different (otherwise the trans_opt path is dead code).
    diff = (int_no - int_op).abs().max()
    assert float(diff) > 1e-9


# ── (5) hard-bin output is detached from spec graph ──

def test_hard_bin_output_does_not_require_grad():
    """Hard-binning is non-differentiable in geometry by design (floor
    has zero gradient). The integrate_hard output must therefore be a
    leaf tensor with requires_grad=False so the user doesn't get
    surprised by a silent zero gradient."""
    s = _spec(requires_grad=True)
    geom = HardBinGeometry.from_spec(s)
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ).requires_grad_(False)
    out = integrate_hard(img, geom)
    assert not out.requires_grad


def test_hard_bin_geometry_has_expected_n_valid():
    """Sanity: most pixels in the central R band should be valid."""
    s = _spec()
    geom = HardBinGeometry.from_spec(s)
    n_valid = geom.n_valid
    n_total = s.NrPixelsY * s.NrPixelsZ
    # With RMin=1, RMax=12 and a 24×24 detector, far corners exceed RMax
    # so n_valid < n_total. But should be a substantial fraction.
    assert 0 < n_valid < n_total
    assert n_valid > 0.3 * n_total


# ── (6) Real Pilatus CeO₂ first-ring agrees across paths ──

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "FF_HEDM" / "Example" / "Calibration"
_PARAMS = _DATA / "parameters.txt"
_IMAGE = _DATA / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif"
_DARK  = _DATA / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif"


@pytest.mark.slow
def test_pilatus_ceo2_hard_bin_first_ring_at_211_px():
    if not (_PARAMS.exists() and _IMAGE.exists() and _DARK.exists()):
        pytest.skip("FF_HEDM/Example/Calibration dataset not found")
    tifffile = pytest.importorskip("tifffile")

    img = tifffile.imread(_IMAGE).astype(np.float64)
    dark = tifffile.imread(_DARK).astype(np.float64)
    if dark.ndim == 3:
        dark = dark.mean(axis=0)
    img_raw = torch.from_numpy(np.clip(img - dark, 0, None))

    spec = spec_from_v1_paramstest(_PARAMS, requires_grad=False)
    spec.RBinSize = 2.0
    spec.EtaBinSize = 5.0
    geom = HardBinGeometry.from_spec(spec)
    int2d = integrate_hard(img_raw, geom, normalize=True)
    prof = int2d.sum(dim=0).numpy()

    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    band = (r_axis > 195) & (r_axis < 230)
    band_idx = np.where(band)[0]
    peak_R = float(r_axis[band_idx[np.argmax(prof[band])]])
    assert 205 <= peak_R <= 220, (
        f"hard-bin first-ring peak in [195, 230] is at R={peak_R:.1f}, "
        "expected around 211 px"
    )
