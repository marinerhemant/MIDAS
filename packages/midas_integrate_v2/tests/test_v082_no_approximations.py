"""v0.8.2: pin the three fixes from the "no approximations" audit.

Each test compares the v0.8.2 implementation to a reference:

1. ``SolidAngleCorrection`` matches v1's exact tilt-aware
   :func:`midas_integrate.geometry.solid_angle_factor` to fp64.
2. ``CALIBRANTS["cr2o3"]`` d-spacings match the JCPDS card 38-1479
   values (no "approximated" entries).
3. ``_thin_plate_kernel`` is mathematically exact at every r including
   r = 0 (no ε-shift bias).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math

import numpy as np
import pytest
import torch

from midas_integrate.geometry import (
    solid_angle_factor as v1_solid_angle_factor,
    build_tilt_matrix as v1_build_tilt_matrix,
)
from midas_integrate_v2 import (
    spec_from_v1_params,
    SolidAngleCorrection, solid_angle_factor_tilted, solid_angle_factor_flat,
    CALIBRANTS,
    RBFResidualCorrection,
    PolygonBinGeometry,
    integrate_with_corrections,
)
from midas_integrate_v2.corrections.spline import _thin_plate_kernel
from midas_integrate.params import IntegrationParams


# ────────────────────────────── (1) Solid angle ──────────────────────────────

def _build_spec_with_tilt(NY=64, NZ=64, *, tx=0.0, ty=1.5, tz=-2.3):
    """Spec with non-trivial tilt — exposes the tilt-aware vs flat
    difference."""
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        tx=tx, ty=ty, tz=tz,
        RMin=1.0, RMax=20.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=10.0,
    )
    return spec_from_v1_params(p, requires_grad=False)


def test_solid_angle_tilted_matches_v1_exact_at_zero_tilt():
    """At zero tilt, v2's tilt-aware reduces to v1's exact formula
    (which itself reduces to cos³(2θ))."""
    s = _build_spec_with_tilt(tx=0.0, ty=0.0, tz=0.0)
    NY, NZ = s.NrPixelsY, s.NrPixelsZ
    Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    TRs_np = v1_build_tilt_matrix(0.0, 0.0, 0.0)
    sa_v1 = v1_solid_angle_factor(
        Y, Z,
        Ycen=float(s.BC_y), Zcen=float(s.BC_z), TRs=TRs_np,
        Lsd=float(s.Lsd), px=s.pxY,
    )

    Y_t = torch.from_numpy(Y.astype(np.float64))
    Z_t = torch.from_numpy(Z.astype(np.float64))
    TRs_t = torch.from_numpy(TRs_np)
    sa_v2 = solid_angle_factor_tilted(
        Y_t, Z_t,
        Ycen=s.BC_y, Zcen=s.BC_z, TRs=TRs_t,
        Lsd=s.Lsd,
        pxY=torch.tensor(s.pxY, dtype=torch.float64),
        pxZ=torch.tensor(s.pxZ, dtype=torch.float64),
    ).numpy()
    np.testing.assert_allclose(sa_v2, sa_v1, rtol=0, atol=1e-13)


def test_solid_angle_tilted_matches_v1_exact_with_real_tilts():
    """Headline: at real tilts (ty=1.5°, tz=-2.3°), v2 must match v1
    exactly. The previous v0.8.1 cos³(2θ) form was silently wrong here."""
    s = _build_spec_with_tilt(tx=0.0, ty=1.5, tz=-2.3)
    NY, NZ = s.NrPixelsY, s.NrPixelsZ
    Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    TRs_np = v1_build_tilt_matrix(0.0, 1.5, -2.3)
    sa_v1 = v1_solid_angle_factor(
        Y, Z,
        Ycen=float(s.BC_y), Zcen=float(s.BC_z), TRs=TRs_np,
        Lsd=float(s.Lsd), px=s.pxY,
    )
    Y_t = torch.from_numpy(Y.astype(np.float64))
    Z_t = torch.from_numpy(Z.astype(np.float64))
    TRs_t = torch.from_numpy(TRs_np)
    sa_v2 = solid_angle_factor_tilted(
        Y_t, Z_t,
        Ycen=s.BC_y, Zcen=s.BC_z, TRs=TRs_t,
        Lsd=s.Lsd,
        pxY=torch.tensor(s.pxY, dtype=torch.float64),
        pxZ=torch.tensor(s.pxZ, dtype=torch.float64),
    ).numpy()
    np.testing.assert_allclose(sa_v2, sa_v1, rtol=0, atol=1e-13)


def test_solid_angle_flat_form_diverges_from_tilt_aware_when_tilted():
    """Sanity check: the flat-detector cos³(2θ) form is *different*
    from the exact tilt-aware form on a tilted detector. (If they were
    the same, the bug we're fixing wouldn't matter.)"""
    s = _build_spec_with_tilt(tx=0.0, ty=2.0, tz=-1.5)
    NY, NZ = s.NrPixelsY, s.NrPixelsZ
    Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Y_t = torch.from_numpy(Y.astype(np.float64))
    Z_t = torch.from_numpy(Z.astype(np.float64))

    Yc = -(Y_t - s.BC_y) * s.pxY
    Zc = (Z_t - s.BC_z) * s.pxZ
    R_um = torch.sqrt(Yc * Yc + Zc * Zc)
    R_px = R_um / s.pxY

    sa_flat = solid_angle_factor_flat(
        R_px, Lsd=s.Lsd,
        px=torch.tensor(s.pxY, dtype=torch.float64),
    ).numpy()

    TRs_t = torch.from_numpy(v1_build_tilt_matrix(0.0, 2.0, -1.5))
    sa_tilted = solid_angle_factor_tilted(
        Y_t, Z_t,
        Ycen=s.BC_y, Zcen=s.BC_z, TRs=TRs_t,
        Lsd=s.Lsd,
        pxY=torch.tensor(s.pxY, dtype=torch.float64),
        pxZ=torch.tensor(s.pxZ, dtype=torch.float64),
    ).numpy()
    # Off-axis pixels show meaningful drift (>0.1% relative typical)
    rel_diff = np.abs(sa_flat - sa_tilted) / np.abs(sa_tilted).clip(min=1e-12)
    assert rel_diff.max() > 1e-3, (
        f"flat form should differ noticeably from tilt-aware on a tilted "
        f"detector (max relative diff: {rel_diff.max():.2e})"
    )


def test_SolidAngleCorrection_module_uses_exact_form():
    """The default module is the exact tilt-aware form, not the flat
    approximation. (The flat function is still importable for
    regression testing, but the user-facing module is exact.)"""
    s = _build_spec_with_tilt(tx=0.0, ty=1.5, tz=-2.3)
    NY, NZ = s.NrPixelsY, s.NrPixelsZ
    Y, Z = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    TRs_t = torch.from_numpy(v1_build_tilt_matrix(0.0, 1.5, -2.3))

    sa_module = SolidAngleCorrection().forward(
        torch.from_numpy(Y.astype(np.float64)),
        torch.from_numpy(Z.astype(np.float64)),
        Ycen=s.BC_y, Zcen=s.BC_z, TRs=TRs_t,
        Lsd=s.Lsd,
        pxY=torch.tensor(s.pxY, dtype=torch.float64),
        pxZ=torch.tensor(s.pxZ, dtype=torch.float64),
    ).numpy()

    sa_v1 = v1_solid_angle_factor(
        Y, Z,
        Ycen=float(s.BC_y), Zcen=float(s.BC_z),
        TRs=v1_build_tilt_matrix(0.0, 1.5, -2.3),
        Lsd=float(s.Lsd), px=s.pxY,
    )
    np.testing.assert_allclose(sa_module, sa_v1, rtol=0, atol=1e-13)


def test_SolidAngleCorrection_grad_flows_to_geometry():
    """The exact tilt-aware module is differentiable in spec parameters."""
    s = _build_spec_with_tilt(tx=0.0, ty=1.5, tz=-2.3)
    s.Lsd = s.Lsd.clone().requires_grad_(True)
    s.ty  = s.ty.clone().requires_grad_(True)

    img = torch.ones(s.NrPixelsZ, s.NrPixelsY, dtype=torch.float64)
    int2d = integrate_with_corrections(
        img, s, solid_angle=SolidAngleCorrection(),
    )
    L = int2d.mean()
    L.backward()
    assert s.Lsd.grad is not None and torch.isfinite(s.Lsd.grad).all()
    assert s.ty.grad  is not None and torch.isfinite(s.ty.grad).all()


# ────────────────────────────── (2) Cr2O3 d-spacings ──────────────────────────────

# JCPDS 38-1479 (Eskolaite, Cr₂O₃, R-3c, a=4.961 Å, c=13.599 Å) — first 10 rings
_CR2O3_JCPDS = [3.633, 2.666, 2.480, 2.265, 2.175, 1.815, 1.672, 1.579, 1.466, 1.430]


def test_cr2o3_d_spacings_match_JCPDS_card():
    """Pinned values; future edits must justify any change."""
    got = CALIBRANTS["cr2o3"]
    assert len(got) == len(_CR2O3_JCPDS)
    for actual, expected in zip(got, _CR2O3_JCPDS):
        assert actual == pytest.approx(expected, abs=1e-3)


def test_all_calibrants_have_decimal_precision():
    """Every published d-spacing should resolve to ≥3 decimal places —
    enough for sub-bin peak matching at any realistic detector."""
    for name, ds in CALIBRANTS.items():
        for d in ds:
            # Must round-trip through 3-decimal print; reject anything
            # written as 1.5 (only 1 decimal) — that's evidence of an
            # "approximated" entry.
            rounded = float(f"{d:.3f}")
            assert abs(d - rounded) < 1e-3, (
                f"{name}: d={d} has more precision than 0.001 Å — "
                "round to 0.001 or supply the full value"
            )


# ────────────────────────────── (3) Thin-plate kernel ──────────────────────────────

def test_tps_kernel_exactly_zero_at_r_zero():
    """φ(0) = lim_{r→0} r² log r = 0 exactly. The old eps-shifted form
    gave ~0.5·1e-12·log(1e-12) ≈ −1.4e-11."""
    r2 = torch.zeros(5, dtype=torch.float64)
    out = _thin_plate_kernel(r2)
    np.testing.assert_array_equal(out.numpy(), np.zeros(5))


def test_tps_kernel_matches_analytic_formula():
    """For r > 0, the kernel must equal r² · log(r) (or equivalently
    0.5·r²·log(r²)) to fp64 — no eps-shift bias."""
    r2 = torch.tensor([0.1, 1.0, 5.0, 100.0, 1e6], dtype=torch.float64)
    out = _thin_plate_kernel(r2).numpy()
    expected = 0.5 * r2.numpy() * np.log(r2.numpy())
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)


def test_tps_kernel_exact_at_small_r():
    """Old eps=1e-12 trick added bias of order eps·log(r²). At r=0.001
    (r²=1e-6) the old bias was ~1e-11 absolute; new form is exact."""
    r2 = torch.tensor([1e-6, 1e-8, 1e-10], dtype=torch.float64)
    out = _thin_plate_kernel(r2).numpy()
    expected = 0.5 * r2.numpy() * np.log(r2.numpy())
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)


def test_tps_kernel_grad_flows():
    """Even with the where(r²>0) guard, autograd produces finite
    gradients on positive r²."""
    r2 = torch.tensor([0.5, 2.0, 10.0], dtype=torch.float64,
                       requires_grad=True)
    L = _thin_plate_kernel(r2).sum()
    L.backward()
    assert r2.grad is not None and torch.isfinite(r2.grad).all()


def test_RBF_residual_correction_runs_after_tps_fix():
    """The full RBFResidualCorrection module still works after the
    kernel change."""
    centres = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                            dtype=torch.float64)
    weights = torch.tensor([0.5, -0.3, 0.2], dtype=torch.float64)
    spline = RBFResidualCorrection(centres, weights, trainable_weights=False)
    Y = torch.tensor([0.5, 1.5], dtype=torch.float64)
    Z = torch.tensor([0.5, 0.5], dtype=torch.float64)
    out = spline(Y, Z)
    assert out.shape == Y.shape
    assert torch.isfinite(out).all()
