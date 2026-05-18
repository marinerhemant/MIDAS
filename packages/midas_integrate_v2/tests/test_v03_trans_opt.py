"""v0.3: native ImTransOpt support in the v2 integration path.

The v0.2 user had to apply ``ImTransOpt`` to the image themselves before
feeding the v2 path (v1 bakes it into the precomputed map; v2's
``eval_pixel_REta`` works on the un-transformed pixel grid). v0.3 makes
this automatic by storing the spec's ``TransOpt`` on
:class:`SoftBinGeometry` and forward-applying it inside
:func:`integrate_soft` / :func:`integrate_soft_batch` /
:func:`integrate_with_corrections`. ``apply_trans_opt=False`` opts out
when the user has already pre-transformed the image.

Five guarantees:

1. ``apply_trans_opt_forward`` matches v1's
   ``_apply_trans_opt_forward`` op-for-op on every supported op code.
2. ``integrate_soft(raw_image, geom)`` with TransOpt baked into the
   geometry is bit-identical to ``integrate_soft(pre_transformed,
   geom_no_op, apply_trans_opt=False)``.
3. ``integrate_with_corrections(raw_image, spec)`` honours
   ``spec.TransOpt`` automatically.
4. ``apply_trans_opt=False`` skips the transform (no double-application).
5. Real Pilatus CeO₂ first-ring centroid lands at the v1-predicted
   211 px when the v2 path is given a raw image (no manual flip).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_integrate.detector_mapper import _apply_trans_opt_forward as v1_trans
from midas_integrate.params import IntegrationParams

from midas_integrate_v2 import (
    spec_from_v1_params,
    spec_from_v1_paramstest,
    apply_trans_opt_forward,
    SoftBinGeometry,
    integrate_soft,
    integrate_soft_batch,
    integrate_with_corrections,
    profile_1d_diff,
)


# ── (1) op parity vs v1 ──

@pytest.mark.parametrize("ops,NY,NZ", [
    ([],     8, 6),
    ([0],    8, 6),
    ([1],    8, 6),
    ([2],    8, 6),
    ([2, 1], 8, 6),
    ([1, 2], 8, 6),
    ([3],    6, 6),
    ([2, 3], 6, 6),
])
def test_apply_trans_opt_forward_matches_v1(ops, NY, NZ):
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 1, size=(NZ, NY)).astype(np.float64)
    expected = v1_trans(arr, ops, NY, NZ)
    got = apply_trans_opt_forward(
        torch.from_numpy(arr), ops, NrPixelsY=NY, NrPixelsZ=NZ,
    ).numpy()
    np.testing.assert_array_equal(got, expected)


def test_apply_trans_opt_forward_rejects_non_2d():
    arr = torch.zeros(8, dtype=torch.float64)
    with pytest.raises(ValueError, match="must be 2-D"):
        apply_trans_opt_forward(arr, [2], NrPixelsY=8, NrPixelsZ=8)


def test_apply_trans_opt_forward_rejects_unknown_op():
    arr = torch.zeros(4, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown ImTransOpt"):
        apply_trans_opt_forward(arr, [99], NrPixelsY=4, NrPixelsZ=4)


def test_apply_trans_opt_transpose_requires_square():
    arr = torch.zeros(6, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="requires NrPixelsY == NrPixelsZ"):
        apply_trans_opt_forward(arr, [3], NrPixelsY=8, NrPixelsZ=6)


# ── (2) integrate_soft handles TransOpt internally ──

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


def _gauss_image(NY, NZ, *, R0_px=6.0, sigma_px=1.5,
                 BC_y=None, BC_z=None, px=200.0):
    BC_y = NY / 2.0 + 0.37 if BC_y is None else BC_y
    BC_z = NZ / 2.0 - 0.41 if BC_z is None else BC_z
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = -(yy - BC_y) * px
    Zc = (zz - BC_z) * px
    R_um = np.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / px
    return torch.from_numpy(
        np.exp(-(R_px - R0_px) ** 2 / (2 * sigma_px ** 2)).astype(np.float64)
    )


@pytest.mark.parametrize("ops", [[], [1], [2], [3], [2, 1]])
def test_integrate_soft_with_trans_opt_matches_pre_transformed(ops):
    """Two routes that must produce identical output:
    (a) raw image into integrate_soft with TransOpt-aware geometry, OR
    (b) pre-transformed image into a no-op-TransOpt geometry, with
        apply_trans_opt=False. Both should give the same 2D bin array."""
    NY = NZ = 24
    spec_with_ops = _spec(NY=NY, NZ=NZ, ops=ops)
    spec_no_ops   = _spec(NY=NY, NZ=NZ, ops=[])
    geom_with_ops = SoftBinGeometry.from_spec(spec_with_ops)
    geom_no_ops   = SoftBinGeometry.from_spec(spec_no_ops)

    img_raw = _gauss_image(NY, NZ)
    img_pre = apply_trans_opt_forward(
        img_raw, ops, NrPixelsY=NY, NrPixelsZ=NZ,
    )

    int_a = integrate_soft(img_raw, geom_with_ops)            # auto-applies
    int_b = integrate_soft(img_pre, geom_no_ops,
                            apply_trans_opt=False)           # no transform
    torch.testing.assert_close(int_a, int_b, rtol=0, atol=1e-15)


def test_integrate_soft_apply_trans_opt_false_does_not_transform():
    NY = NZ = 24
    spec = _spec(NY=NY, NZ=NZ, ops=[2])
    geom = SoftBinGeometry.from_spec(spec)
    img = _gauss_image(NY, NZ)

    auto = integrate_soft(img, geom, apply_trans_opt=True)
    skip = integrate_soft(img, geom, apply_trans_opt=False)
    # With TransOpt=2 these must differ (otherwise the flag is dead code)
    diff = (auto - skip).abs().max()
    assert float(diff) > 1e-9


def test_integrate_soft_no_trans_opt_is_identity_on_flag():
    """When TransOpt is empty / no-op, both branches give the same
    result regardless of the flag value."""
    NY = NZ = 24
    spec = _spec(NY=NY, NZ=NZ, ops=[])
    geom = SoftBinGeometry.from_spec(spec)
    img = _gauss_image(NY, NZ)

    on  = integrate_soft(img, geom, apply_trans_opt=True)
    off = integrate_soft(img, geom, apply_trans_opt=False)
    torch.testing.assert_close(on, off, rtol=0, atol=0)


def test_integrate_soft_batch_honours_trans_opt():
    NY = NZ = 24
    spec = _spec(NY=NY, NZ=NZ, ops=[2])
    geom = SoftBinGeometry.from_spec(spec)
    rng = torch.Generator().manual_seed(0)
    images = torch.rand(3, NZ, NY, generator=rng, dtype=torch.float64)
    batch = integrate_soft_batch(images, geom)
    expected = torch.stack([integrate_soft(images[i], geom)
                              for i in range(3)])
    # Batched vs sequential index_add can differ at sub-fp64 noise (1e-15)
    # because the accumulation order changes; not a math drift.
    torch.testing.assert_close(batch, expected, rtol=0, atol=1e-13)


# ── (3) integrate_with_corrections honours spec.TransOpt ──

def test_integrate_with_corrections_honours_spec_trans_opt():
    NY = NZ = 24
    spec = _spec(NY=NY, NZ=NZ, ops=[2], requires_grad=True)
    img = _gauss_image(NY, NZ)
    int_auto = integrate_with_corrections(img, spec)
    int_skip = integrate_with_corrections(img, spec, apply_trans_opt=False)
    diff = (int_auto - int_skip).abs().max()
    assert float(diff) > 1e-9


def test_integrate_with_corrections_no_op_trans_opt_no_change():
    NY = NZ = 24
    spec = _spec(NY=NY, NZ=NZ, ops=[], requires_grad=True)
    img = _gauss_image(NY, NZ)
    on  = integrate_with_corrections(img, spec, apply_trans_opt=True)
    off = integrate_with_corrections(img, spec, apply_trans_opt=False)
    torch.testing.assert_close(on, off, rtol=0, atol=0)


def test_integrate_with_corrections_grad_still_flows_under_trans_opt():
    NY = NZ = 24
    spec = _spec(NY=NY, NZ=NZ, ops=[2], requires_grad=True)
    img = _gauss_image(NY, NZ)
    int2d = integrate_with_corrections(img, spec)
    L = int2d.mean()
    L.backward()
    for f in ("Lsd", "BC_y", "BC_z", "ty", "tz"):
        g = getattr(spec, f).grad
        assert g is not None and torch.isfinite(g).all(), f"{f} grad bad"


# ── (4) Real Pilatus CeO₂ raw image works with no manual flip ──

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "FF_HEDM" / "Example" / "Calibration"
_PARAMS = _DATA / "parameters.txt"
_IMAGE = _DATA / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif"
_DARK  = _DATA / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif"


@pytest.mark.slow
def test_pilatus_raw_image_through_v2_lands_first_ring_at_211_px():
    """End-to-end: feed the v2 soft path a RAW Pilatus image (no manual
    flip!) and verify the first CeO₂ ring shows up where v1 puts it."""
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
    geom = SoftBinGeometry.from_spec(spec)
    int2d = integrate_soft(img_raw, geom)                      # raw input!
    prof = profile_1d_diff(int2d, spec).detach().numpy()

    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    band = (r_axis > 195) & (r_axis < 230)
    band_idx = np.where(band)[0]
    peak_R = float(r_axis[band_idx[np.argmax(prof[band])]])
    assert 205 <= peak_R <= 220, (
        f"v2 raw-image profile peak in [195, 230] is at R={peak_R:.1f}, "
        "expected first CeO₂ ring around 211 px"
    )
