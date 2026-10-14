"""Opt-in distance normalisation: ``spec.IntensityRefDistance_um`` = L_ref > 0 gives I * (Lsd / L_ref)^2 / (SA * POL).

Raw counts per pixel scale as 1/Lsd^2 (pixel solid angle = area cos^3 / Lsd^2). SolidAngleCorrection is normalised to 1 at normal
incidence, so it removes the angle dependence but not the 1/Lsd^2; GSAS-II multiplies by Lsd[m]^2 (checked against a real lineout set, 2026-10-13).

Pins:
1. Default off: no divisor, nothing changes; L_ref alone (no SA / POL flags) enables a constant divisor (L_ref / Lsd)^2.
2. hard / subpixel / polygon / soft honour it; the output is exactly (Lsd / L_ref)^2 times the flags-off output.
3. Physics: a flat-field scene whose counts follow the solid angle, K cos^3(2theta) / Lsd^2, gives a distance-independent profile at two
   distances once SA + L_ref are on (and does not without L_ref).
4. Soft path keeps the gradient to Lsd; integrate_with_corrections agrees with the soft integrator.
5. validate() rejects negative / non-finite L_ref; provenance header lists the corrections; to_v1 warns.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import warnings

import numpy as np
import pytest
import torch

from midas_integrate_v2 import (
    HardBinGeometry, SubpixelBinGeometry, PolygonBinGeometry, SoftBinGeometry,
    integrate_hard, integrate_subpixel, integrate_polygon, integrate_soft,
)
from midas_integrate_v2.binning.pixel_correction import pixel_divisor_from_spec
from midas_integrate_v2.corrections import integrate_with_corrections

from test_spec_pixel_corrections import _spec, _image   # same tiny-detector builders as the SA / POL tests

LREF = 1.0e6


def _with(spec, *, ref=0.0, lsd=None):
    if lsd is not None:
        spec.Lsd = torch.tensor(float(lsd), dtype=torch.float64)
    spec.IntensityRefDistance_um = ref
    return spec


def test_default_off_and_ref_alone_is_a_constant_divisor():
    s = _spec()
    assert s.IntensityRefDistance_um == 0.0
    assert HardBinGeometry.from_spec(s).pixel_divisor is None
    d = pixel_divisor_from_spec(_with(_spec(), ref=LREF))
    assert d is not None
    np.testing.assert_allclose(d.numpy(), (LREF / 20_000.0) ** 2, rtol=1e-14)


@pytest.mark.parametrize("sa,pol", [(0, 0), (1, 0), (1, 1)])
def test_divisor_is_ref_over_lsd_squared_times_the_flag_divisor(sa, pol):
    base = pixel_divisor_from_spec(_spec(sa=sa, pol=pol)) if (sa or pol) else torch.ones(48 * 40, dtype=torch.float64)
    s = _with(_spec(sa=sa, pol=pol), ref=LREF)
    np.testing.assert_allclose(pixel_divisor_from_spec(s).numpy(), base.numpy() * (LREF / 20_000.0) ** 2, rtol=1e-13)


@pytest.mark.parametrize("name,geom_cls,fn", [
    ("hard", HardBinGeometry, integrate_hard),
    ("subpixel", SubpixelBinGeometry, integrate_subpixel),
    ("polygon", PolygonBinGeometry, integrate_polygon),
])
def test_integrators_scale_by_lsd_over_ref_squared(name, geom_cls, fn):
    img = _image(_spec())
    off = fn(img, geom_cls.from_spec(_spec(sa=1, pol=1)))
    on = fn(img, geom_cls.from_spec(_with(_spec(sa=1, pol=1), ref=LREF)))
    torch.testing.assert_close(on, off * (20_000.0 / LREF) ** 2, rtol=1e-12, atol=0.0)


def test_soft_integrator_honours_it_and_keeps_the_gradient_to_lsd():
    s = _with(_spec(sa=1), ref=LREF)
    s.Lsd = torch.tensor(20_000.0, dtype=torch.float64, requires_grad=True)
    img = _image(s)
    out = integrate_soft(img, SoftBinGeometry.from_spec(s))
    ref_out = integrate_with_corrections(img, s)
    torch.testing.assert_close(out, ref_out, rtol=1e-10, atol=0.0)
    out.sum().backward()
    assert s.Lsd.grad is not None and abs(float(s.Lsd.grad)) > 0.0


def test_physics_distance_independent_profile_for_a_solid_angle_weighted_scene():
    """Counts per pixel = K cos^3(2theta) / Lsd^2 (an isotropic scatterer, zero tilt). SA divides out cos^3; L_ref removes 1/Lsd^2."""
    K = 7.0e12
    profiles = {}
    for with_ref in (False, True):
        for lsd in (15_000.0, 30_000.0):
            s = _with(_spec(sa=1), ref=LREF if with_ref else 0.0, lsd=lsd)
            NY, NZ = s.NrPixelsY, s.NrPixelsZ
            Z, Y = np.meshgrid(np.arange(NZ), np.arange(NY), indexing="ij")
            dy, dz = (Y - float(s.BC_y)) * s.pxY, (Z - float(s.BC_z)) * s.pxZ
            cos2th = lsd / np.sqrt(dy * dy + dz * dz + lsd * lsd)
            img = torch.from_numpy(K * cos2th ** 3 / lsd ** 2)
            i2d = integrate_hard(img, HardBinGeometry.from_spec(s)).numpy()          # (n_eta, n_r)
            full = (i2d > 0).all(axis=0)                                          # bins covered in every eta sector (no edge bins)
            profiles[(with_ref, lsd)] = np.where(full, i2d.mean(axis=0), np.nan)
    a, b = profiles[(True, 15_000.0)], profiles[(True, 30_000.0)]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    assert ok.sum() > 10
    # within the geometry / binning accuracy (hard binning of a smooth field): both distances give K / L_ref^2
    np.testing.assert_allclose(a[ok], K / LREF ** 2, rtol=3e-3)
    np.testing.assert_allclose(b[ok], K / LREF ** 2, rtol=3e-3)
    # negative control: without L_ref the two distances differ by (30/15)^2 = 4
    a0, b0 = profiles[(False, 15_000.0)], profiles[(False, 30_000.0)]
    ok0 = np.isfinite(a0) & np.isfinite(b0) & (a0 > 0) & (b0 > 0)
    assert np.median(a0[ok0] / b0[ok0]) == pytest.approx(4.0, rel=3e-3)


def test_validate_provenance_and_v1_bridge():
    from midas_integrate_v2.compat.to_v1 import v1_params_from_spec
    from midas_integrate_v2.io.writers import build_provenance
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="IntensityRefDistance_um"):
            _with(_spec(), ref=bad).validate()
    s = _with(_spec(sa=1), ref=LREF)
    md = build_provenance(s, integrate_mode="hard", integrate_K=1)
    assert md.spec_summary["corrections"]["IntensityRefDistance_um"] == LREF
    assert md.spec_summary["corrections"]["SolidAngleCorrection"] == 1
    assert "corrections" not in build_provenance(_spec(), integrate_mode="hard", integrate_K=1).spec_summary     # flags off: header unchanged
    with pytest.warns(UserWarning, match="no v1 equivalent"):
        v1_params_from_spec(s)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v1_params_from_spec(_spec())
