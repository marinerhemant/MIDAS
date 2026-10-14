"""SolidAngleCorrection / PolarizationCorrection in the paramstest reach the hard, subpixel and
polygon integrators (and their variance versions, and Map.bin).

Before this, ``spec_from_v1_paramstest`` carried the flags and nothing in the binning path read them:
v1 applied ``PolarizationCorrection 1`` and ``midas-integrate-v2`` did not (flags on == flags off to
the last bit; 4-14 % in ring areas on a polarized, cos^3-weighted frame).

Pins:

1. Flags off -> ``pixel_divisor is None`` and the output is untouched.
2. The divisor equals an independent numpy evaluation (Kahn polarization + the tilt-aware solid angle
   Lsd^2 (n.r)/|r|^3), zero tilt and tilted.
3. hard / subpixel / polygon with the flags on == the same integrator on image / divisor with flags off
   (so the correction is I/(SA*POL) and the normaliser is untouched), single image and batch.
4. ``*_with_variance``: spec-derived correction scales the variance by 1/d^2, and an explicit
   ``correction=`` REPLACES it rather than stacking (the hand-built recipe in manuals/pdf stays right).
5. Map.bin written from a flags-on geometry reproduces the v2 result through v1's CSR kernel.
6. v1's own mapper (flags on) agrees with v2 polygon (flags on) to bin-centre-vs-pixel-centre accuracy.
7. The soft path applies them too, with a differentiable divisor (equal to integrate_with_corrections, gradients included);
   the CLI ``--mode soft`` applies them.
8. Non-square pixels: the divisor matches an independent lab-frame evaluation (R_px is in mean-pixel units).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2 import (
    spec_from_v1_params,
    HardBinGeometry, SubpixelBinGeometry, PolygonBinGeometry,
    integrate_hard, integrate_hard_batch, integrate_subpixel, integrate_subpixel_batch,
    integrate_polygon, integrate_hard_with_variance, integrate_polygon_with_variance,
    integrate_subpixel_with_variance,
)
from midas_integrate_v2.binning.pixel_correction import pixel_divisor_from_spec


def _params(*, sa=0, pol=0, pf=0.99, plane=90.0, tilt=(0.0, 0.0, 0.0), NY=48, NZ=40, pxY=200.0, pxZ=200.0):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=pxY, pxZ=pxZ, Lsd=20_000.0,           # short Lsd -> large 2theta, strong factors
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=26.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=30.0,
        SubPixelLevel=1,
    )
    p.tx, p.ty, p.tz = tilt
    p.SolidAngleCorrection = sa
    p.PolarizationCorrection = pol
    p.PolarizationFraction = pf
    p.PolarizationPlaneEtaDeg = plane
    return p


def _spec(**kw):
    return spec_from_v1_params(_params(**kw))


def _image(spec, seed=3):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(50.0, 500.0, (spec.NrPixelsZ, spec.NrPixelsY)))


# ---- independent reference (numpy, no package code) --------------------------------------
def _ref_divisor(p, sa, pol):
    """SA * POL per pixel from first principles. MIDAS pixel convention: pixel centre at the integer
    index, Yc = (BC_y - Y) px, Zc = (Z - BC_z) px; plane tilt R = Rx(tx) Ry(ty) Rz(tz) acting on
    (Yc, Zc); eta = atan2(-y, z). Tilt-aware solid angle = Lsd^2 (n.r)/|r|^3 with n the plane
    normal (first column of R) and r the lab vector to the pixel."""
    NY, NZ = p.NrPixelsY, p.NrPixelsZ
    Y, Z = np.meshgrid(np.arange(NY, dtype=float), np.arange(NZ, dtype=float))
    c = lambda a: math.cos(math.radians(a))
    s = lambda a: math.sin(math.radians(a))
    Rx = np.array([[1, 0, 0], [0, c(p.tx), -s(p.tx)], [0, s(p.tx), c(p.tx)]])
    Ry = np.array([[c(p.ty), 0, s(p.ty)], [0, 1, 0], [-s(p.ty), 0, c(p.ty)]])
    Rz = np.array([[c(p.tz), -s(p.tz), 0], [s(p.tz), c(p.tz), 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz
    Yc = (p.BC_y - Y) * p.pxY
    Zc = (Z - p.BC_z) * p.pxZ
    x = p.Lsd + R[0, 1] * Yc + R[0, 2] * Zc
    y = R[1, 1] * Yc + R[1, 2] * Zc
    z = R[2, 1] * Yc + R[2, 2] * Zc
    rn = np.sqrt(x * x + y * y + z * z)
    out = np.ones_like(x)
    if sa:
        n = R[:, 0]
        ndotr = n[0] * x + n[1] * y + n[2] * z
        out = out * (p.Lsd ** 2 * ndotr / rn ** 3)
    if pol:
        tth = np.arctan2(np.hypot(y, z), x)
        eta = np.degrees(np.arctan2(-y, z))
        c2, s2 = np.cos(tth) ** 2, np.sin(tth) ** 2
        out = out * 0.5 * (1 + c2 - p.PolarizationFraction * np.cos(2 * np.radians(eta - p.PolarizationPlaneEtaDeg)) * s2)
    return out.reshape(-1)


# ---- 1. flags off ------------------------------------------------------------------------
def test_flags_off_has_no_divisor():
    spec = _spec()
    assert pixel_divisor_from_spec(spec) is None
    for g in (HardBinGeometry.from_spec(spec), SubpixelBinGeometry.from_spec(spec, K=2),
              PolygonBinGeometry.from_spec(spec)):
        assert g.pixel_divisor is None


# ---- 2. divisor vs independent numpy -----------------------------------------------------
@pytest.mark.parametrize("sa,pol", [(1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("tilt", [(0.0, 0.0, 0.0), (0.4, -0.7, 0.3)])
def test_divisor_matches_independent_reference(sa, pol, tilt):
    p = _params(sa=sa, pol=pol, tilt=tilt, plane=37.5, pf=0.9)
    spec = spec_from_v1_params(p)
    got = pixel_divisor_from_spec(spec).numpy()
    ref = _ref_divisor(p, sa, pol)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=0)
    assert abs(got.mean() - 1.0) > 1e-3        # a non-trivial correction (the test can fail)


# ---- 3. integrators ----------------------------------------------------------------------
def _geoms(spec, K=2):
    return [
        (HardBinGeometry.from_spec(spec), integrate_hard),
        (SubpixelBinGeometry.from_spec(spec, K=K), integrate_subpixel),
        (PolygonBinGeometry.from_spec(spec), integrate_polygon),
    ]


@pytest.mark.parametrize("tilt", [(0.0, 0.0, 0.0), (0.4, -0.7, 0.3)])
def test_integrators_apply_I_over_divisor(tilt):
    on = _spec(sa=1, pol=1, tilt=tilt)
    off = _spec(sa=0, pol=0, tilt=tilt)
    img = _image(on)
    div = pixel_divisor_from_spec(on).reshape(img.shape)
    for (g_on, fn), (g_off, _) in zip(_geoms(on), _geoms(off)):
        assert g_on.pixel_divisor is not None and g_off.pixel_divisor is None
        want = fn(img / div, g_off)
        got = fn(img, g_on)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
        # and the correction really changed the answer
        assert (got - fn(img, g_off)).abs().max() > 1e-3 * img.mean()


def test_batch_variants_apply_divisor():
    on, off = _spec(sa=1, pol=1), _spec()
    imgs = torch.stack([_image(on, seed=s) for s in (1, 2, 3)])
    div = pixel_divisor_from_spec(on).reshape(imgs.shape[1:])
    for (g_on, _), (g_off, _), batch in zip(
            _geoms(on)[:2], _geoms(off)[:2], (integrate_hard_batch, integrate_subpixel_batch)):
        torch.testing.assert_close(batch(imgs, g_on), batch(imgs / div, g_off), rtol=1e-12, atol=1e-12)


# ---- 4. variance versions ----------------------------------------------------------------
@pytest.mark.parametrize("which", ["hard", "subpixel", "polygon"])
def test_variance_integrators_scale_mean_and_variance(which):
    on, off = _spec(sa=1, pol=1), _spec()
    fn = {"hard": integrate_hard_with_variance, "subpixel": integrate_subpixel_with_variance,
          "polygon": integrate_polygon_with_variance}[which]
    g_on = dict(zip(("hard", "subpixel", "polygon"), (a for a, _ in _geoms(on))))[which]
    g_off = dict(zip(("hard", "subpixel", "polygon"), (a for a, _ in _geoms(off))))[which]
    img = _image(on)
    div = pixel_divisor_from_spec(on).reshape(img.shape)
    m_on, s_on = fn(img, g_on)
    m_ref, s_ref = fn(img, g_off, correction=div)
    torch.testing.assert_close(m_on, m_ref, rtol=1e-12, atol=1e-12, equal_nan=True)
    torch.testing.assert_close(s_on, s_ref, rtol=1e-12, atol=1e-12, equal_nan=True)
    m_off, _ = fn(img, g_off)
    assert (torch.nan_to_num(m_on) - torch.nan_to_num(m_off)).abs().max() > 1e-3 * img.mean()


def test_explicit_correction_replaces_spec_correction_no_stacking():
    """The manuals/pdf recipe builds correction = sa*pf by hand; with the flags also set on the spec
    the result must equal the flags-off result with the same explicit correction."""
    on, off = _spec(sa=1, pol=1), _spec()
    img = _image(on)
    hand = torch.from_numpy(_ref_divisor(_params(sa=1, pol=1), 1, 1)).reshape(img.shape)
    m1, s1 = integrate_hard_with_variance(img, HardBinGeometry.from_spec(on), correction=hand)
    m2, s2 = integrate_hard_with_variance(img, HardBinGeometry.from_spec(off), correction=hand)
    torch.testing.assert_close(m1, m2, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(s1, s2, rtol=0, atol=0, equal_nan=True)


# ---- 5. Map.bin --------------------------------------------------------------------------
def test_map_bin_bakes_divisor_into_pixel_weight(tmp_path):
    from midas_integrate.bin_io import load_map, PixelMap
    from midas_integrate.kernels import build_csr as v1_build_csr, integrate as v1_integrate
    from midas_integrate_v2 import write_map_bin_from_geometry

    spec = _spec(sa=1, pol=1)
    img = _image(spec)
    for geom, fn in ((HardBinGeometry.from_spec(spec), integrate_hard),
                     (SubpixelBinGeometry.from_spec(spec, K=2), integrate_subpixel)):
        write_map_bin_from_geometry(geom, spec, tmp_path)
        pm = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
        g1 = v1_build_csr(pm, n_r=spec.n_r_bins, n_eta=spec.n_eta_bins,
                          n_pixels_y=spec.NrPixelsY, n_pixels_z=spec.NrPixelsZ,
                          device="cpu", dtype=torch.float64, bc_y=float(spec.BC_y), bc_z=float(spec.BC_z))
        via_v1 = v1_integrate(img, g1, mode="floor", normalize=True).T      # (n_eta, n_r)
        want = fn(img, geom)
        ok = want > 0
        torch.testing.assert_close(via_v1[ok], want[ok], rtol=1e-9, atol=1e-9)


# ---- 6. parity with v1's mapper ----------------------------------------------------------
def test_v2_polygon_correction_matches_v1_mapper_correction():
    """Compare the CORRECTION (flags-on / flags-off cake) rather than the cakes: v1 and v2 polygon
    already differ by up to ~13 % in the outermost bins with the flags off (edge clipping), which is
    unrelated. v1 evaluates the polarization factor at the bin centre, v2 at the pixel centre."""
    from midas_integrate.detector_mapper import build_map
    from midas_integrate.bin_io import PixelMap
    from midas_integrate.kernels import build_csr, integrate

    def both(flag):
        p = _params(sa=flag, pol=flag)
        spec = spec_from_v1_params(p)
        r1 = build_map(p, auto_load=False, verbose=False, use_numba=False)
        pm = PixelMap(pxList=r1.pxList, counts=r1.counts, offsets=r1.offsets, map_header=None, nmap_header=None)
        g1 = build_csr(pm, n_r=p.n_r_bins, n_eta=p.n_eta_bins, n_pixels_y=p.NrPixelsY,
                       n_pixels_z=p.NrPixelsZ, device="cpu", dtype=torch.float64, bc_y=p.BC_y, bc_z=p.BC_z)
        img = _image(spec)
        return (integrate(img, g1, mode="floor", normalize=True).T,
                integrate_polygon(img, PolygonBinGeometry.from_spec(spec)))

    a1, a2 = both(1)
    b1, b2 = both(0)
    ok = (a1 > 0) & (a2 > 0) & (b1 > 0) & (b2 > 0)
    assert int(ok.sum()) > 200
    r1, r2 = a1 / b1, a2 / b2
    assert float(r2[ok].max()) > 1.1                       # the correction is large here (the test can fail)
    rel = ((r1 - r2).abs() / r2)[ok]
    assert float(rel.median()) < 1e-3, float(rel.median())
    assert float(rel.max()) < 1.5e-2, float(rel.max())


# ---- 7. soft path: applied, and still differentiable --------------------------------------
from midas_integrate_v2 import SoftBinGeometry, integrate_soft, integrate_soft_batch
from midas_integrate_v2.corrections.integrated import integrate_with_corrections


def test_soft_applies_I_over_divisor_single_and_batch():
    on, off = _spec(sa=1, pol=1), _spec()
    img = _image(on)
    div = pixel_divisor_from_spec(on).reshape(img.shape)
    g_on, g_off = SoftBinGeometry.from_spec(on), SoftBinGeometry.from_spec(off)
    assert g_on.pixel_divisor is not None and g_off.pixel_divisor is None
    torch.testing.assert_close(integrate_soft(img, g_on), integrate_soft(img / div, g_off), rtol=1e-12, atol=1e-12)
    imgs = torch.stack([_image(on, seed=s) for s in (1, 2)])
    torch.testing.assert_close(integrate_soft_batch(imgs, g_on), integrate_soft_batch(imgs / div, g_off), rtol=1e-12, atol=1e-12)
    assert (integrate_soft(img, g_on) - integrate_soft(img, g_off)).abs().max() > 1e-3 * img.mean()


def test_soft_matches_integrate_with_corrections():
    """The two differentiable paths must agree when only the spec flags ask for the correction."""
    on = _spec(sa=1, pol=1, tilt=(0.0, 0.4, -0.3))
    img = _image(on)
    torch.testing.assert_close(integrate_soft(img, SoftBinGeometry.from_spec(on)),
                               integrate_with_corrections(img, on), rtol=1e-9, atol=1e-9)


def test_soft_divisor_carries_gradient_to_the_spec():
    """d(profile)/d(Lsd) through integrate_soft must equal integrate_with_corrections' and must differ from the one with a
    detached divisor, i.e. the correction's own geometry dependence is in the gradient."""
    def grad(path):
        spec = spec_from_v1_params(_params(sa=1, pol=1, tilt=(0.0, 0.4, -0.3)), requires_grad=True)
        img = _image(spec)
        if path == "soft":
            out = integrate_soft(img, SoftBinGeometry.from_spec(spec))
        elif path == "soft_detached":
            g = SoftBinGeometry.from_spec(spec); g.pixel_divisor = g.pixel_divisor.detach()
            out = integrate_soft(img, g)
        else:
            out = integrate_with_corrections(img, spec)
        w = torch.linspace(0.5, 1.5, out.numel(), dtype=out.dtype).reshape(out.shape)
        (out * w).sum().backward()
        return spec.Lsd.grad.clone()
    g_soft, g_iwc, g_det = grad("soft"), grad("iwc"), grad("soft_detached")
    assert torch.isfinite(g_soft) and g_soft != 0
    torch.testing.assert_close(g_soft, g_iwc, rtol=1e-6, atol=0)
    assert abs(float(g_soft - g_det)) > 1e-4 * abs(float(g_soft))


def test_cli_soft_mode_applies_the_flags():
    from midas_integrate_v2.cli import _build_geometry_and_integrate
    on, off = _spec(sa=1, pol=1), _spec()
    img = _image(on)
    div = pixel_divisor_from_spec(on).reshape(img.shape)
    p_on = _build_geometry_and_integrate(on, img.numpy(), mode="soft", K=2)
    p_flagsoff = _build_geometry_and_integrate(off, img.numpy(), mode="soft", K=2)
    assert np.abs(p_on - p_flagsoff).max() > 1e-3 * np.abs(p_on).max()
    p_ref = _build_geometry_and_integrate(off, (img / div).numpy(), mode="soft", K=2)
    np.testing.assert_allclose(p_on, p_ref, rtol=1e-10, atol=1e-10)


# ---- 8. non-square pixels (the square-pixel tests above cannot see a pxY / pxZ mix-up) ------------
@pytest.mark.parametrize("pxy,pxz", [(172.0, 200.0), (200.0, 150.0)])
@pytest.mark.parametrize("sa,pol", [(1, 0), (0, 1), (1, 1)])
def test_divisor_matches_independent_reference_non_square_pixels(pxy, pxz, sa, pol):
    p = _params(sa=sa, pol=pol, tilt=(0.0, 0.4, -0.3), plane=37.5, pf=0.9, pxY=pxy, pxZ=pxz)
    got = pixel_divisor_from_spec(spec_from_v1_params(p)).numpy()
    np.testing.assert_allclose(got, _ref_divisor(p, sa, pol), rtol=1e-9, atol=0)


def test_soft_matches_integrate_with_corrections_non_square_pixels():
    on = spec_from_v1_params(_params(sa=1, pol=1, tilt=(0.0, 0.4, -0.3), pxY=172.0, pxZ=200.0))
    img = _image(on)
    torch.testing.assert_close(integrate_soft(img, SoftBinGeometry.from_spec(on)),
                               integrate_with_corrections(img, on), rtol=1e-9, atol=1e-9)


def test_divisor_does_not_depend_on_whether_the_spec_was_validated_first():
    """A pixel-valued RhoD is resolved by spec.validate(); the divisor must use the resolved value either way."""
    a = spec_from_v1_params(_params(sa=1, pol=1))
    b = spec_from_v1_params(_params(sa=1, pol=1)); b.validate()
    torch.testing.assert_close(pixel_divisor_from_spec(a), pixel_divisor_from_spec(b), rtol=0, atol=0)
