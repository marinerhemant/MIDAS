"""Tests for the residual correction map (port of v1 C ``dg_residual_corr_lookup``).

Cover:
  - Bilinear lookup correctness (matches a hand-computed reference).
  - Autograd through both query coords and map values (full differentiability).
  - Save/load round-trip in v1-compatible binary format.
  - Cross-package binary parity with :mod:`midas_integrate.residual_corr`
    (skipped if the legacy package isn't installed).
  - RBF builder produces a smooth map of the right shape and dtype.
  - ``check_rho_d_um`` accepts healthy values and refuses common
    unit-mistake values with a useful diagnostic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# ============================================================ lookup

def test_residual_corr_lookup_matches_manual_bilinear():
    """grid_sample lookup must agree with v1 C's manual bilinear formula
    to within fp64 tolerance at arbitrary fractional pixel positions."""
    from midas_calibrate_v2.forward.residual_corr import residual_corr_lookup

    # Build a deterministic map: ΔR = sin(y/100) * cos(z/80)  (smooth field).
    Nz, Ny = 256, 256
    y_idx = torch.arange(Ny, dtype=torch.float64)
    z_idx = torch.arange(Nz, dtype=torch.float64)
    Z, Y = torch.meshgrid(z_idx, y_idx, indexing="ij")
    corr_map = torch.sin(Y / 100.0) * torch.cos(Z / 80.0)

    # Query at a few fractional positions; compare to a NumPy manual bilinear
    # (the C-side lookup_python from midas_integrate).
    Y_q = torch.tensor([12.3, 47.9, 199.7, 0.0, Ny - 1.001], dtype=torch.float64)
    Z_q = torch.tensor([8.1, 100.5, 220.4, 0.0, Nz - 1.001], dtype=torch.float64)
    v2 = residual_corr_lookup(Y_q, Z_q, corr_map).numpy()

    cm = corr_map.numpy()
    ref = np.zeros_like(v2)
    for i, (y, z) in enumerate(zip(Y_q.numpy(), Z_q.numpy())):
        y = max(0.0, min(y, Ny - 1.001))
        z = max(0.0, min(z, Nz - 1.001))
        y0, z0 = int(y), int(z)
        fy, fz = y - y0, z - z0
        v00 = cm[z0, y0]; v10 = cm[z0, y0 + 1]
        v01 = cm[z0 + 1, y0]; v11 = cm[z0 + 1, y0 + 1]
        ref[i] = (v00 * (1 - fy) * (1 - fz) + v10 * fy * (1 - fz)
                  + v01 * (1 - fy) * fz + v11 * fy * fz)

    assert np.allclose(v2, ref, atol=1e-12), (
        f"residual_corr_lookup disagrees with manual bilinear:\n"
        f"  v2  = {v2}\n  ref = {ref}\n  diff = {v2 - ref}"
    )


def test_residual_corr_lookup_handles_out_of_bounds_via_border():
    """Pixels outside the map should clamp to the border, not wrap or NaN."""
    from midas_calibrate_v2.forward.residual_corr import residual_corr_lookup
    Nz, Ny = 16, 16
    corr_map = torch.full((Nz, Ny), 0.5, dtype=torch.float64)
    Y_q = torch.tensor([-5.0, 0.0, Ny + 100.0], dtype=torch.float64)
    Z_q = torch.tensor([Nz + 50.0, -3.0, Nz - 1.0], dtype=torch.float64)
    out = residual_corr_lookup(Y_q, Z_q, corr_map).numpy()
    # Uniform map → uniform output even at out-of-bounds (border replication).
    assert np.allclose(out, 0.5)


def test_residual_corr_lookup_zero_map_returns_zero():
    """A zero map produces zero correction everywhere — useful for the
    no-residual default path."""
    from midas_calibrate_v2.forward.residual_corr import residual_corr_lookup
    corr_map = torch.zeros((32, 32), dtype=torch.float64)
    Y_q = torch.tensor([1.2, 5.8, 30.1], dtype=torch.float64)
    Z_q = torch.tensor([10.0, 0.0, 31.0], dtype=torch.float64)
    out = residual_corr_lookup(Y_q, Z_q, corr_map)
    assert torch.allclose(out, torch.zeros_like(out))


# ============================================================ autograd

def test_residual_corr_lookup_grad_through_coords():
    """Gradient must flow through Y_pix / Z_pix (used when fitting BC, tilts)."""
    from midas_calibrate_v2.forward.residual_corr import residual_corr_lookup
    Nz, Ny = 32, 32
    corr_map = torch.linspace(0, 1, Nz * Ny, dtype=torch.float64).reshape(Nz, Ny)
    Y = torch.tensor([10.5], dtype=torch.float64, requires_grad=True)
    Z = torch.tensor([5.5], dtype=torch.float64, requires_grad=True)
    out = residual_corr_lookup(Y, Z, corr_map).sum()
    out.backward()
    assert Y.grad is not None and torch.isfinite(Y.grad).all()
    assert Z.grad is not None and torch.isfinite(Z.grad).all()
    assert Y.grad.item() != 0.0 or Z.grad.item() != 0.0


def test_residual_corr_lookup_grad_through_map():
    """Gradient must flow into the map values (so the map can itself be
    refined as a Parameter in a joint inverse problem)."""
    from midas_calibrate_v2.forward.residual_corr import residual_corr_lookup
    Nz, Ny = 16, 16
    corr_map = torch.zeros((Nz, Ny), dtype=torch.float64, requires_grad=True)
    Y = torch.tensor([4.3, 9.7], dtype=torch.float64)
    Z = torch.tensor([6.1, 12.2], dtype=torch.float64)
    out = residual_corr_lookup(Y, Z, corr_map).sum()
    out.backward()
    assert corr_map.grad is not None
    assert torch.isfinite(corr_map.grad).all()
    # The two queries should each touch four pixels; gradient should be
    # non-zero on at least 8 pixels.
    assert (corr_map.grad != 0).sum() >= 7   # allow 1 pixel slack from overlap


# ============================================================ I/O round-trip

def test_save_load_residual_corr_bin_roundtrip(tmp_path):
    """save_residual_corr_bin + load_residual_corr_bin must be bit-exact."""
    from midas_calibrate_v2.forward.residual_corr import (
        save_residual_corr_bin, load_residual_corr_bin,
    )
    Nz, Ny = 64, 48
    rng = np.random.default_rng(seed=42)
    cm = torch.as_tensor(rng.normal(size=(Nz, Ny)), dtype=torch.float64)
    p = tmp_path / "rc.bin"
    save_residual_corr_bin(cm, p)
    # Wire format matches v1 C: NY * NZ float64 in row-major (z, y) order.
    assert p.stat().st_size == Nz * Ny * 8
    loaded = load_residual_corr_bin(p, NrPixelsY=Ny, NrPixelsZ=Nz)
    assert loaded.dtype == torch.float64
    assert loaded.shape == cm.shape
    assert torch.equal(loaded, cm)


def test_residual_corr_bin_parity_with_midas_integrate(tmp_path):
    """The binary written by v2 must be loadable by the legacy
    midas_integrate package — same bytes, same shape, same values."""
    from midas_calibrate_v2.forward.residual_corr import save_residual_corr_bin
    try:
        from midas_integrate.residual_corr import load_residual_correction_map
    except ImportError:
        pytest.skip("midas_integrate not installed")

    Nz, Ny = 32, 24
    cm = torch.as_tensor(
        np.linspace(-1.0, 1.0, Nz * Ny).reshape(Nz, Ny),
        dtype=torch.float64,
    )
    p = tmp_path / "rc.bin"
    save_residual_corr_bin(cm, p)
    rc_v1 = load_residual_correction_map(str(p), NrPixelsY=Ny, NrPixelsZ=Nz)
    assert rc_v1.NrPixelsY == Ny
    assert rc_v1.NrPixelsZ == Nz
    assert np.array_equal(rc_v1.map, cm.numpy())


# ============================================================ builder

def test_build_residual_corr_map_shape_and_dtype():
    """RBF builder returns the requested shape, dtype, and a finite map."""
    from midas_calibrate_v2.forward.residual_corr import build_residual_corr_map
    pytest.importorskip("scipy.interpolate")

    # Synthetic ΔR field: ΔR(Y, Z) = 10 sin(2π Y / 2048) (µm).
    rng = np.random.default_rng(seed=7)
    n = 400
    Y = rng.uniform(100.0, 1900.0, size=n)
    Z = rng.uniform(100.0, 1900.0, size=n)
    dR = 10.0 * np.sin(2 * math.pi * Y / 2048.0)

    cm = build_residual_corr_map(
        torch.as_tensor(Y), torch.as_tensor(Z), torch.as_tensor(dR),
        NrPixelsY=2048, NrPixelsZ=2048, pxY=200.0,
    )
    assert cm.shape == (2048, 2048)
    assert cm.dtype == torch.float64
    assert torch.isfinite(cm).all()
    # Map units: pixels; with -ΔR/px convention, max |value| should be at
    # most ~ |max(ΔR)| / px = 10/200 = 0.05 px, give or take RBF smoothing
    # (synthetic + sparse data widens the bound — allow generous slack).
    assert cm.abs().max().item() < 1.0


def test_build_residual_corr_map_rejects_tiny_inputs():
    """Builder must refuse to fit when there are <50 points."""
    from midas_calibrate_v2.forward.residual_corr import build_residual_corr_map
    pytest.importorskip("scipy.interpolate")
    with pytest.raises(ValueError, match="need >=50"):
        build_residual_corr_map(
            torch.zeros(10), torch.zeros(10), torch.zeros(10),
            NrPixelsY=2048, NrPixelsZ=2048, pxY=200.0,
        )


# ============================================================ sanity check

def test_check_rho_d_um_accepts_healthy():
    """Healthy ranges (around detector diagonal) must pass silently."""
    from midas_calibrate_v2.forward.sanity import check_rho_d_um
    # 2048x2048 detector at 200 µm: diag ≈ 290 mm = 290000 µm.
    for healthy in (290_000, 200_000, 500_000, 1_400_000):
        # Should not raise.
        assert check_rho_d_um(
            float(healthy), 2048, 2048, 1024.0, 994.0, 200.0
        ) is None


def test_check_rho_d_um_catches_pixels_as_um():
    """The most common mistake: passing RhoD in pixels instead of µm.
    1024 px on a 200-µm detector is 200_000× too small."""
    from midas_calibrate_v2.forward.sanity import check_rho_d_um
    with pytest.raises(ValueError) as exc_info:
        check_rho_d_um(1024.0, 2048, 2048, 1024.0, 994.0, 200.0)
    msg = str(exc_info.value)
    assert "outside the sane range" in msg
    # The hint should suggest checking pixel units when value × pxY looks OK.
    assert "did you pass RhoD in pixels?" in msg


def test_check_rho_d_um_catches_grossly_too_large():
    """If RhoD is, say, 10× detector diagonal, refuse it."""
    from midas_calibrate_v2.forward.sanity import check_rho_d_um
    with pytest.raises(ValueError, match="outside the sane range"):
        check_rho_d_um(3_000_000.0, 2048, 2048, 1024.0, 994.0, 200.0)


def test_check_rho_d_um_rejects_non_positive():
    """RhoD ≤ 0 is always invalid."""
    from midas_calibrate_v2.forward.sanity import check_rho_d_um
    for bad in (0.0, -100.0):
        with pytest.raises(ValueError, match="must be positive"):
            check_rho_d_um(bad, 2048, 2048, 1024.0, 994.0, 200.0)


def test_check_rho_d_um_strict_false_returns_message():
    """Non-strict mode returns a diagnostic string instead of raising."""
    from midas_calibrate_v2.forward.sanity import check_rho_d_um
    msg = check_rho_d_um(
        1024.0, 2048, 2048, 1024.0, 994.0, 200.0, strict=False
    )
    assert msg is not None and "outside the sane range" in msg


def test_detector_max_corner_dist_um():
    """Manual reference: 2048×2048 detector at 200 µm, BC=(1024, 994).
    The farthest corner is (0, 2047) — dy=1024 (max of 1024, 1023),
    dz=1053 (max of 994, 1053).  Distance = sqrt(1024² + 1053²) × 200."""
    from midas_calibrate_v2.forward.sanity import detector_max_corner_dist_um
    d = detector_max_corner_dist_um(2048, 2048, 1024.0, 994.0, 200.0)
    expected = math.sqrt(1024.0 ** 2 + 1053.0 ** 2) * 200.0
    assert d == pytest.approx(expected, rel=1e-9)


# ============================================================ integration

def test_pixel_to_REta_with_residual_map_changes_R():
    """Wired-up forward call: pixel_to_REta with a non-trivial residual
    map must produce R that differs from the no-map call by the lookup."""
    from midas_calibrate_v2.forward.geometry import pixel_to_REta
    Y = torch.tensor([1024.0], dtype=torch.float64)
    Z = torch.tensor([700.0], dtype=torch.float64)
    common = dict(
        Lsd=torch.tensor(940710.0, dtype=torch.float64),
        BC_y=torch.tensor(1024.0, dtype=torch.float64),
        BC_z=torch.tensor(994.0, dtype=torch.float64),
        tx=torch.tensor(0.0, dtype=torch.float64),
        ty=torch.tensor(0.0, dtype=torch.float64),
        tz=torch.tensor(0.0, dtype=torch.float64),
        p_coeffs=torch.zeros(15, dtype=torch.float64),
        parallax=torch.tensor(0.0, dtype=torch.float64),
        pxY=torch.tensor(200.0, dtype=torch.float64),
        pxZ=torch.tensor(200.0, dtype=torch.float64),
        rho_d=torch.tensor(1024.0, dtype=torch.float64),
    )
    r0 = pixel_to_REta(Y, Z, **common, residual_corr_map=None).R_px
    cm = torch.full((2048, 2048), 0.25, dtype=torch.float64)
    r1 = pixel_to_REta(Y, Z, **common, residual_corr_map=cm).R_px
    assert torch.allclose(r1 - r0, torch.full_like(r0, 0.25), atol=1e-9)
