"""Tests for `midas_defect.geometry`."""

from __future__ import annotations

import dataclasses
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

CPU = torch.device("cpu")

from midas_defect.geometry import (
    Geometry,
    demk_default_geometry,
    pixel_to_qlab,
    qlab_to_qsample,
    qsample_to_qlab,
)


# ---------------------------------------------------------------------------
# 1. Synthetic correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_demk_default_fields_match_calibration():
    g = demk_default_geometry()
    assert g.lsd_um == pytest.approx(652665.6325, abs=1e-3)
    assert g.bcy_px == pytest.approx(698.4202, abs=1e-3)
    assert g.bcz_px == pytest.approx(813.6800, abs=1e-3)
    assert g.px_um == 172.0
    assert g.wavelength_A == pytest.approx(0.172979, abs=1e-6)
    assert (g.n_pix_y, g.n_pix_z) == (1475, 1679)
    assert g.n_frames == 1440
    # validated ω map: ω = 180 − 0.25·frame (start +180, CW step −0.25)
    assert g.omega_first_deg == 180.0
    assert g.omega_step_deg == -0.25
    assert g.tx_deg == pytest.approx(0.0957, abs=1e-4)
    # validated radial lens distortion present
    assert g.p_coeffs[:4] == pytest.approx((0.000230, 0.001234, 0.000211, 32.904494), abs=1e-6)
    assert g.rho_d_um == pytest.approx(219964.42, abs=1e-2)


@pytest.mark.unit
def test_paramstest_parser_roundtrip():
    """Synthesize a paramstest-like file, parse it, verify field-for-field."""
    body = """
        Lsd 652665.632540605729
        BC  698.420227947936 813.680034653341
        ty -0.196484201531
        tz  0.534276032234
        tx  0.0
        Wedge 0
        Wavelength 0.172973
        px 172.0
        NrPixelsY 1475
        NrPixelsZ 1679
        OmegaFirstFile -180
        OmegaStep 0.25
        NrFilesPerSweep 1440
        # comment line should be ignored
        UnknownKey 123
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(body)
        path = Path(fh.name)
    try:
        g = Geometry.from_paramstest(path)
        assert g.lsd_um == pytest.approx(652665.6325, abs=1e-3)
        assert g.bcy_px == pytest.approx(698.4202, abs=1e-3)
        assert g.bcz_px == pytest.approx(813.6800, abs=1e-3)
        assert g.px_um == 172.0
        assert g.wavelength_A == pytest.approx(0.172973, abs=1e-6)
        assert (g.n_pix_y, g.n_pix_z) == (1475, 1679)
        assert g.n_frames == 1440
        assert g.tx_deg == 0.0
        assert g.ty_deg == pytest.approx(-0.1965, abs=1e-4)
        assert g.tz_deg == pytest.approx(0.5343, abs=1e-4)
        assert g.wedge_deg == 0.0
    finally:
        path.unlink()


@pytest.mark.unit
def test_paramstest_parses_full_p0_p14_distortion():
    """Regression: the parser once read only p0..p3 and silently zeroed p4..p14.

    `midas_transforms.apply_tilt_distortion` consumes all 15 coefficients, so
    dropping the higher-order terms shifts predicted spot positions with no
    error raised. Values below are the 1-ID GE5 95 keV CeO2 calibration
    (pokharel_jul26), which populates the full set.
    """
    body = """
        Lsd 1666219.585298
        BC 1018.718310 1076.544304
        tx 0.000000
        ty -0.005156
        tz 0.950678
        p0 3.043933204e-05
        p1 -0.0002455996239
        p2 -0.0005507571608
        p3 -10.4095261
        p4 0.001322929365
        p5 -0.001041571336
        p6 -8.595090107
        p7 0.0005476891236
        p8 24.62538966
        p9 -4.579508501e-05
        p10 80.67522769
        p11 -4.154932876e-05
        p12 -90
        p13 -0.0001223074173
        p14 90
        Wavelength 0.130510
        px 200.000000
        NrPixelsY 2048
        NrPixelsZ 2048
        RhoD 297745.587613
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(body)
        path = Path(fh.name)
    try:
        g = Geometry.from_paramstest(path)
        assert len(g.p_coeffs) == 15
        # the terms that used to be dropped
        assert g.p_coeffs[4] == pytest.approx(0.001322929365, rel=1e-9)
        assert g.p_coeffs[6] == pytest.approx(-8.595090107, rel=1e-9)
        assert g.p_coeffs[8] == pytest.approx(24.62538966, rel=1e-9)
        assert g.p_coeffs[10] == pytest.approx(80.67522769, rel=1e-9)
        assert g.p_coeffs[12] == pytest.approx(-90.0, rel=1e-9)
        assert g.p_coeffs[14] == pytest.approx(90.0, rel=1e-9)
        assert g.rho_d_um == pytest.approx(297745.587613, rel=1e-9)
    finally:
        path.unlink()


@pytest.mark.unit
def test_paramstest_legacy_p0_p3_only_still_zero_pads():
    """Back-compat: a file with only p0..p3 must still give a 15-tuple."""
    body = """
        Lsd 652665.6325
        BC 698.42 813.68
        Wavelength 0.172979
        px 172.0
        NrPixelsY 1475
        NrPixelsZ 1679
        p0 0.000230
        p1 0.001234
        p2 0.000211
        p3 32.904494
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(body)
        path = Path(fh.name)
    try:
        g = Geometry.from_paramstest(path)
        assert len(g.p_coeffs) == 15
        assert g.p_coeffs[:4] == pytest.approx((0.000230, 0.001234, 0.000211, 32.904494), abs=1e-9)
        assert all(c == 0.0 for c in g.p_coeffs[4:])
    finally:
        path.unlink()


@pytest.mark.unit
def test_paramstest_missing_required_raises():
    """Missing 'Lsd' must surface as a clear ValueError."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("BC 100 200\nWavelength 0.17\npx 172\nNrPixelsY 1\nNrPixelsZ 1\n")
        path = Path(fh.name)
    try:
        with pytest.raises(ValueError, match="Lsd"):
            Geometry.from_paramstest(path)
    finally:
        path.unlink()


@pytest.mark.unit
def test_pixel_at_beam_centre_gives_zero_q():
    """A pixel exactly at the beam centre should map to q ~= 0 in lab frame."""
    g = demk_default_geometry()
    q = pixel_to_qlab(g.bcz_px, g.bcy_px, g, device=CPU)
    assert torch.allclose(q, torch.zeros(3, dtype=q.dtype), atol=1e-6)


@pytest.mark.unit
def test_pixel_magnitude_matches_two_theta_no_tilt():
    """For a no-tilt geometry, a pixel at (BCz, BCy + N) gives `|q| = 4π sin(θ)/λ`.

    With purely horizontal offset by N pixels:
      tan(2θ) = N * px / Lsd, no tilt.
    """
    # zero tilts AND distortion to recover the ideal pinhole magnitude
    g = dataclasses.replace(demk_default_geometry(), tx_deg=0.0, ty_deg=0.0,
                            tz_deg=0.0, p_coeffs=(0.0,) * 15)
    N = 200.0
    q = pixel_to_qlab(g.bcz_px, g.bcy_px + N, g, device=CPU)
    qmag = torch.linalg.vector_norm(q).item()
    two_theta = math.atan(N * g.px_um / g.lsd_um)
    qmag_expected = 4.0 * math.pi * math.sin(two_theta / 2.0) / g.wavelength_A
    assert qmag == pytest.approx(qmag_expected, rel=1e-6)


@pytest.mark.unit
def test_tilts_perturb_q_at_diagonal_pixel():
    """A nonzero tilt actually changes q at a diagonal pixel.

    Tilt rotations are about the beam-centre, so they leave invariant the
    pixel-offset components along the rotation axis. We deliberately use
    a diagonal offset (both row and col) AND a tz tilt (rotates X-Y plane)
    so the tilt has a meaningful effect.
    """
    g0 = dataclasses.replace(demk_default_geometry(), tx_deg=0.0, ty_deg=0.0, tz_deg=0.0)
    g1 = dataclasses.replace(g0, tz_deg=1.0)   # 1 deg about lab Z
    rows, cols = g0.bcz_px + 150.0, g0.bcy_px + 100.0
    q0 = pixel_to_qlab(rows, cols, g0, device=CPU)
    q1 = pixel_to_qlab(rows, cols, g1, device=CPU)
    assert not torch.allclose(q0, q1, atol=1e-6), "tz tilt did not perturb q"
    # |q| is approximately preserved by tilt-about-beam-centre
    assert torch.linalg.vector_norm(q0).item() == pytest.approx(
        torch.linalg.vector_norm(q1).item(), rel=1e-3
    )


@pytest.mark.unit
def test_tilt_about_axis_leaves_axis_aligned_offset_invariant():
    """Tilt about lab Y leaves a horizontal-only pixel-offset invariant.

    This is a positive check on the tilt-around-beam-centre convention:
    rotations about Y don't change vectors that are themselves along Y.
    """
    g0 = dataclasses.replace(demk_default_geometry(), tx_deg=0.0, ty_deg=0.0, tz_deg=0.0)
    g_ty = dataclasses.replace(g0, ty_deg=2.0)
    # purely horizontal offset
    q0 = pixel_to_qlab(g0.bcz_px, g0.bcy_px + 100.0, g0, device=CPU)
    q1 = pixel_to_qlab(g_ty.bcz_px, g_ty.bcy_px + 100.0, g_ty, device=CPU)
    assert torch.allclose(q0, q1, atol=1e-10), (
        "ty tilt unexpectedly changed q at a horizontal-only offset; "
        "expected invariance because rotation about Y fixes the Y axis."
    )


@pytest.mark.unit
def test_qlab_qsample_roundtrip():
    g = demk_default_geometry()
    rows = torch.tensor([100.0, 500.0, 1000.0], dtype=torch.float64)
    cols = torch.tensor([100.0, 500.0, 1000.0], dtype=torch.float64)
    qlab = pixel_to_qlab(rows, cols, g)
    for omega_deg in (-180.0, -45.0, 0.0, 30.0, 179.5):
        omega_rad = torch.deg2rad(torch.tensor(omega_deg, dtype=torch.float64))
        qs = qlab_to_qsample(qlab, omega_rad)
        ql = qsample_to_qlab(qs, omega_rad)
        assert torch.allclose(qlab, ql, atol=1e-12)


@pytest.mark.unit
def test_qsample_preserves_magnitude():
    """ω-rotation is rigid, so |q| stays constant."""
    g = demk_default_geometry()
    rows = torch.tensor([300.0, 700.0, 1200.0], dtype=torch.float64)
    cols = torch.tensor([400.0, 900.0, 1300.0], dtype=torch.float64)
    qlab = pixel_to_qlab(rows, cols, g)
    qmag_lab = torch.linalg.vector_norm(qlab, dim=-1)
    for omega_deg in (-180.0, -45.0, 0.0, 30.0, 179.5):
        omega_rad = torch.deg2rad(torch.tensor(omega_deg, dtype=torch.float64))
        qs = qlab_to_qsample(qlab, omega_rad)
        qmag_s = torch.linalg.vector_norm(qs, dim=-1)
        assert torch.allclose(qmag_lab, qmag_s, atol=1e-12)


@pytest.mark.unit
def test_as_hedm_geometry_roundtrip():
    """`as_hedm_geometry()` preserves scalar fields."""
    g = demk_default_geometry()
    H = g.as_hedm_geometry()
    assert H.Lsd == g.lsd_um
    assert H.y_BC == g.bcy_px
    assert H.z_BC == g.bcz_px
    assert H.px == g.px_um
    assert H.wavelength == g.wavelength_A
    assert (H.n_pixels_y, H.n_pixels_z) == (g.n_pix_y, g.n_pix_z)
    assert H.omega_start == g.omega_first_deg
    assert H.omega_step == g.omega_step_deg
    assert H.n_frames == g.n_frames


# ---------------------------------------------------------------------------
# 2. Autograd correctness
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_pixel_to_qlab_gradient_wrt_lsd_bc():
    """Refining Lsd / BCy / BCz to drive |q| to a target works under autograd."""
    g = demk_default_geometry()
    rows = torch.tensor([500.0], dtype=torch.float64)
    cols = torch.tensor([900.0], dtype=torch.float64)
    target_q = torch.tensor([0.0, 0.5, 0.1], dtype=torch.float64)

    # Wrap with diff parameters
    lsd = torch.tensor(g.lsd_um, dtype=torch.float64, requires_grad=True)
    bcy = torch.tensor(g.bcy_px, dtype=torch.float64, requires_grad=True)
    bcz = torch.tensor(g.bcz_px, dtype=torch.float64, requires_grad=True)

    def loss(lsd_, bcy_, bcz_):
        # rebuild geometry with the diff tensors (need to host them as Geometry fields)
        # We carry the diff tensors directly through the math, bypassing the
        # Geometry dataclass — this is fine because Geometry's only role is
        # field naming, the math is in `pixel_to_qlab`.
        y_um = -(cols - bcy_) * g.px_um
        z_um =  (bcz_ - rows) * g.px_um
        x_um = lsd_.expand_as(y_um)
        p_lab = torch.stack([x_um, y_um, z_um], dim=-1)
        norm = torch.linalg.vector_norm(p_lab, dim=-1, keepdim=True)
        k_f = p_lab / norm
        k_i = torch.zeros_like(k_f); k_i[..., 0] = 1.0
        k0 = 2.0 * math.pi / g.wavelength_A
        q = k0 * (k_f - k_i)
        return ((q.squeeze() - target_q) ** 2).sum()

    L = loss(lsd, bcy, bcz)
    g_lsd, g_bcy, g_bcz = torch.autograd.grad(L, (lsd, bcy, bcz))

    eps = 1e-2
    for grad_auto, p, name in [
        (g_lsd, lsd, "lsd"), (g_bcy, bcy, "bcy"), (g_bcz, bcz, "bcz"),
    ]:
        plus = p.detach().clone()
        minus = p.detach().clone()
        plus.add_(eps); minus.sub_(eps)
        Lp = loss(plus if name == "lsd" else lsd.detach(),
                  plus if name == "bcy" else bcy.detach(),
                  plus if name == "bcz" else bcz.detach())
        Lm = loss(minus if name == "lsd" else lsd.detach(),
                  minus if name == "bcy" else bcy.detach(),
                  minus if name == "bcz" else bcz.detach())
        g_fd = (Lp - Lm) / (2 * eps)
        assert grad_auto.item() == pytest.approx(g_fd.item(), rel=1e-3, abs=1e-6), (
            f"autograd disagrees with finite-difference for {name}: "
            f"auto={grad_auto.item():.6e}, fd={g_fd.item():.6e}"
        )


@pytest.mark.autograd
def test_qlab_qsample_gradient_wrt_omega():
    """Gradient of |q_sample - target|² w.r.t. ω is correct."""
    qlab = torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float64)
    target = torch.tensor([[0.1, 0.4, 0.6]], dtype=torch.float64)
    omega = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def loss(om):
        q = qlab_to_qsample(qlab, om)
        return ((q - target) ** 2).sum()

    L = loss(omega)
    g_auto = torch.autograd.grad(L, omega)[0]
    eps = 1e-6
    g_fd = (loss(omega + eps) - loss(omega - eps)) / (2 * eps)
    assert g_auto.item() == pytest.approx(g_fd.item(), rel=1e-4)


# ---------------------------------------------------------------------------
# 3. Device portability
# ---------------------------------------------------------------------------

@pytest.mark.device
def test_pixel_to_qlab_device_portable(_device_param, _dtype_param):
    if _device_param.type == "mps" and _dtype_param == torch.float64:
        pytest.skip("MPS does not support float64 in PyTorch")
    g = demk_default_geometry()
    rows = torch.tensor([100.0, 500.0, 1000.0], dtype=_dtype_param)
    cols = torch.tensor([100.0, 500.0, 1000.0], dtype=_dtype_param)
    q_cpu = pixel_to_qlab(rows, cols, g, device=torch.device("cpu"),
                          dtype=_dtype_param)
    q_dev = pixel_to_qlab(rows.to(_device_param), cols.to(_device_param),
                          g, device=_device_param, dtype=_dtype_param)
    tol = 1e-4 if _dtype_param == torch.float32 else 1e-10
    assert torch.allclose(q_cpu, q_dev.cpu(), atol=tol, rtol=tol)


@pytest.mark.device
def test_qlab_to_qsample_device_portable(_device_param, _dtype_param):
    if _device_param.type == "mps" and _dtype_param == torch.float64:
        pytest.skip("MPS does not support float64 in PyTorch")
    qlab = torch.tensor([[0.3, 0.4, 0.5], [-0.1, 0.2, -0.3]],
                        dtype=_dtype_param, device=_device_param)
    omega = torch.tensor(0.5, dtype=_dtype_param, device=_device_param)
    qs_dev = qlab_to_qsample(qlab, omega)
    qs_cpu = qlab_to_qsample(qlab.cpu(), omega.cpu())
    tol = 1e-5 if _dtype_param == torch.float32 else 1e-10
    assert torch.allclose(qs_cpu, qs_dev.cpu(), atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# 4. Real-data regression
# ---------------------------------------------------------------------------

@pytest.mark.real_data
def test_paramstest_loader_against_demk_calibration():
    """Read `parameters_final.txt` from the Demk calibration directory.

    Requires the file to be locally available, or we skip. We don't put the
    path itself in the assertion to keep the test machine-agnostic; instead
    we compare fields against the hard-coded `demk_default_geometry()`.
    """
    candidates = [
        Path(
            "/gdata/dm/1ID/2025/stubbins_sep25/analysis/ff/calib/"
            "parameters_final.txt"
        ),
        Path(__file__).parent / "fixtures" / "parameters_final.txt",
    ]

    def _can_read(p: Path) -> bool:
        # `.exists()` can raise PermissionError on /gdata when the parent
        # directory restricts traversal (sticky-bit case on copland).
        try:
            return p.exists() and os.access(p, os.R_OK)
        except (PermissionError, OSError):
            return False

    path = next((p for p in candidates if _can_read(p)), None)
    if path is None:
        pytest.skip(
            "parameters_final.txt not found locally; "
            "scp the file from copland to enable."
        )
    g_file = Geometry.from_paramstest(path)
    g_def  = demk_default_geometry()
    assert g_file.lsd_um == pytest.approx(g_def.lsd_um, rel=1e-9)
    assert g_file.bcy_px == pytest.approx(g_def.bcy_px, rel=1e-9)
    assert g_file.bcz_px == pytest.approx(g_def.bcz_px, rel=1e-9)
    assert g_file.px_um == g_def.px_um
    assert g_file.wavelength_A == pytest.approx(g_def.wavelength_A, abs=1e-5)
    assert g_file.n_pix_y == g_def.n_pix_y
    assert g_file.n_pix_z == g_def.n_pix_z
