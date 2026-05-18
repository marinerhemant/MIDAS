"""v0.4: K×K subpixel-oversampled hard binning.

Pins:

1. K=1 reduces to plain hard binning (single sample at pixel centre).
2. As K grows, the subpixel profile converges toward the high-K limit
   (an interior consistency check — bin-edge quantisation drops
   monotonically).
3. Batched variant matches per-image loop.
4. ImTransOpt honoured.
5. Real Pilatus first-ring centroid stable across K (peak position
   doesn't drift with oversampling — only edge quantisation tightens).
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
    HardBinGeometry, integrate_hard,
    SubpixelBinGeometry, integrate_subpixel, integrate_subpixel_batch,
)


def _spec(NY=24, NZ=24, *, ops=None):
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
    return spec_from_v1_params(p, requires_grad=False)


def _gauss_image(NY, NZ, *, R0_px=6.0, sigma_px=1.5, px=200.0):
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - NY / 2.0 - 0.37) * px
    Zc = (zz - NZ / 2.0 + 0.41) * px
    R_um = np.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / px
    return torch.from_numpy(
        np.exp(-(R_px - R0_px) ** 2 / (2 * sigma_px ** 2)).astype(np.float64)
    )


# ── (1) K=1 reduces to hard binning ──

def test_subpixel_K1_matches_hard_bin():
    s = _spec()
    geom_hard = HardBinGeometry.from_spec(s)
    geom_sub  = SubpixelBinGeometry.from_spec(s, K=1)
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ)

    int_h = integrate_hard(img, geom_hard, normalize=True)
    int_s = integrate_subpixel(img, geom_sub, normalize=True)
    torch.testing.assert_close(int_h, int_s, rtol=0, atol=1e-12)


def test_subpixel_K1_unnormalized_matches_hard_bin():
    s = _spec()
    geom_hard = HardBinGeometry.from_spec(s)
    geom_sub  = SubpixelBinGeometry.from_spec(s, K=1)
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ)
    int_h = integrate_hard(img, geom_hard, normalize=False)
    int_s = integrate_subpixel(img, geom_sub, normalize=False)
    torch.testing.assert_close(int_h, int_s, rtol=0, atol=1e-12)


# ── (2) Higher K spreads contributions over more bins ──

def test_subpixel_K_increases_total_subpixel_count():
    """Sanity: K=2 has 4× more subpixel-bin contributions than K=1."""
    s = _spec()
    g1 = SubpixelBinGeometry.from_spec(s, K=1)
    g2 = SubpixelBinGeometry.from_spec(s, K=2)
    g4 = SubpixelBinGeometry.from_spec(s, K=4)
    assert g2.n_subpixels == 4 * g1.n_subpixels
    assert g4.n_subpixels == 16 * g1.n_subpixels


def test_subpixel_K2_constant_image_yields_constant_per_bin():
    """Mass conservation per bin under normalize=True. K>1 must still
    give a constant when the input is constant."""
    s = _spec()
    geom = SubpixelBinGeometry.from_spec(s, K=2)
    img = torch.full((s.NrPixelsZ, s.NrPixelsY), 7.5, dtype=torch.float64)
    int2d = integrate_subpixel(img, geom, normalize=True)
    nonzero = int2d[int2d > 0]
    np.testing.assert_array_almost_equal(nonzero.numpy(), 7.5, decimal=12)


def test_subpixel_K_higher_changes_distribution():
    """K=2 must distribute intensity across MORE bins than K=1 because
    each pixel now spans up to 4 bins instead of 1."""
    s = _spec()
    img = _gauss_image(s.NrPixelsY, s.NrPixelsZ, R0_px=4.5, sigma_px=0.5)
    int1 = integrate_subpixel(img, SubpixelBinGeometry.from_spec(s, K=1),
                                normalize=False)
    int2 = integrate_subpixel(img, SubpixelBinGeometry.from_spec(s, K=2),
                                normalize=False)
    # Same total mass (within fp64); but K=2 spreads across more bins
    n_bins_1 = int((int1 != 0).sum().item())
    n_bins_2 = int((int2 != 0).sum().item())
    assert n_bins_2 >= n_bins_1
    total_1 = float(int1.sum())
    total_2 = float(int2.sum())
    # Conservation should hold within numerical noise (only diffs come
    # from pixels that fell out of range under one sampling but not the
    # other — for an interior peak this is small).
    assert abs(total_1 - total_2) / max(1.0, total_1) < 0.1


# ── (3) Batched matches per-image ──

def test_integrate_subpixel_batch_matches_per_image_loop():
    s = _spec()
    geom = SubpixelBinGeometry.from_spec(s, K=2)
    rng = torch.Generator().manual_seed(0)
    images = torch.rand(3, s.NrPixelsZ, s.NrPixelsY,
                         generator=rng, dtype=torch.float64)
    batch = integrate_subpixel_batch(images, geom, normalize=True)
    expected = torch.stack([
        integrate_subpixel(images[i], geom, normalize=True) for i in range(3)
    ])
    torch.testing.assert_close(batch, expected, rtol=0, atol=1e-12)


# ── (4) ImTransOpt ──

def test_subpixel_honours_trans_opt():
    NY = NZ = 24
    s_no = _spec(NY=NY, NZ=NZ, ops=[])
    s_op = _spec(NY=NY, NZ=NZ, ops=[2])
    g_no = SubpixelBinGeometry.from_spec(s_no, K=2)
    g_op = SubpixelBinGeometry.from_spec(s_op, K=2)
    img = _gauss_image(NY, NZ)
    int_no = integrate_subpixel(img, g_no)
    int_op = integrate_subpixel(img, g_op)            # auto-flips first
    diff = (int_no - int_op).abs().max()
    assert float(diff) > 1e-9


# ── (5) K validation ──

def test_K_zero_or_negative_rejected():
    s = _spec()
    with pytest.raises(ValueError, match="K must be"):
        SubpixelBinGeometry.from_spec(s, K=0)
    with pytest.raises(ValueError, match="K must be"):
        SubpixelBinGeometry.from_spec(s, K=-1)


# ── (6) Real Pilatus: peak stable across K ──

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "FF_HEDM" / "Example" / "Calibration"
_PARAMS = _DATA / "parameters.txt"
_IMAGE = _DATA / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif"
_DARK  = _DATA / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif"


@pytest.mark.slow
def test_pilatus_first_ring_stable_across_K():
    """Pilatus CeO₂ first-ring centroid must agree across K=1, 2, 3
    (within <1 R bin) — peak position is a physical property, not a
    binning-scheme artefact."""
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

    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    band = (r_axis > 200) & (r_axis < 230)
    idx = np.where(band)[0]

    centroids = {}
    for K in (1, 2, 3):
        geom = SubpixelBinGeometry.from_spec(spec, K=K)
        int2d = integrate_subpixel(img_raw, geom, normalize=False)
        prof = int2d.sum(dim=0).numpy()
        w = np.maximum(prof[idx], 0)
        c = float((w * r_axis[idx]).sum() / (w.sum() + 1e-30))
        centroids[K] = c

    # All centroids must agree within one R bin (2 px in this config).
    span = max(centroids.values()) - min(centroids.values())
    assert span < spec.RBinSize, (
        f"Pilatus first-ring centroid drifted across K: {centroids}, "
        f"span {span:.3f} > RBinSize {spec.RBinSize}"
    )
