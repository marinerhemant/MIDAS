"""Tests for the hex (PIXIRAD-style) pixel lattice support.

Phase A coverage:
- Cartesian fallback is bit-identical to the pre-refactor path.
- Hex centroid mapping matches hand-computed positions.
- Hex axial ratio is exactly √3 / 2 (canonical hex lattice).
- Autograd through Apothem flows.
- pixel_to_REta with lattice='hex_offset_y' produces a sensible
  pseudo-strain residual roundtrip on a synthetic ring.
"""
from __future__ import annotations

import math

import pytest
import torch


def test_lattice_cartesian_bit_identical():
    """The new ``lattice_to_phys`` call must produce *exactly* the same
    (Yc, Zc) as the historical formula for any cartesian inputs."""
    from midas_calibrate_v2.forward.lattice import lattice_to_phys

    Y = torch.linspace(0, 100, 17, dtype=torch.float64)
    Z = torch.linspace(0, 80, 13, dtype=torch.float64)
    YY, ZZ = torch.meshgrid(Y, Z, indexing="ij")
    BC_y = torch.tensor(48.5, dtype=torch.float64)
    BC_z = torch.tensor(33.25, dtype=torch.float64)
    pxY = torch.tensor(75.0, dtype=torch.float64)
    pxZ = torch.tensor(80.0, dtype=torch.float64)

    Yc, Zc = lattice_to_phys(
        YY, ZZ, lattice="cartesian",
        BC_y=BC_y, BC_z=BC_z, pxY=pxY, pxZ=pxZ,
    )
    Yc_ref = (-YY + BC_y) * pxY
    Zc_ref = (ZZ - BC_z) * pxZ
    assert torch.equal(Yc, Yc_ref)
    assert torch.equal(Zc, Zc_ref)


def test_hex_centroid_mapping_hand_computed():
    """Sanity-check the hex_offset_y mapping against hand-derived positions.

    PIXIRAD-1 geometry: apothem a = 30 μm, pitch_y = 2a = 60 μm,
    pitch_z = a√3 ≈ 51.96 μm. Odd-Z rows shift by +a in Y. We pick
    BC = (0, 0) so the BC-translation drops out and the mapping
    becomes pure pixel-index → physical.
    """
    from midas_calibrate_v2.forward.lattice import lattice_to_phys

    a = 30.0
    sqrt3 = math.sqrt(3.0)

    cases = [
        # (Y_pix, Z_pix) → (Yc_um, Zc_um); BC_y = BC_z = 0
        ((0,  0), (0.0,                0.0)),                       # even row, origin
        ((1,  0), (-1 * 2 * a,         0.0)),                       # even row, +1 col
        ((0,  1), (-0.5 * 2 * a,       1 * a * sqrt3)),             # odd row at col 0
        ((1,  1), (-1.5 * 2 * a,       1 * a * sqrt3)),             # odd row, +1 col
        ((5,  4), (-5 * 2 * a,         4 * a * sqrt3)),             # even row, far
        ((5,  5), (-5.5 * 2 * a,       5 * a * sqrt3)),             # odd row, far
    ]
    Y = torch.tensor([y for (y, _), _ in cases], dtype=torch.float64)
    Z = torch.tensor([z for (_, z), _ in cases], dtype=torch.float64)
    BC = torch.tensor(0.0, dtype=torch.float64)
    apot = torch.tensor(a, dtype=torch.float64)

    Yc, Zc = lattice_to_phys(
        Y, Z, lattice="hex_offset_y",
        BC_y=BC, BC_z=BC, apothem=apot,
    )
    for k, ((Yp, Zp), (Yref, Zref)) in enumerate(cases):
        assert math.isclose(float(Yc[k]), Yref, abs_tol=1e-9), \
            f"Yc mismatch at idx {k} (Y_pix={Yp}, Z_pix={Zp}): {float(Yc[k])} vs {Yref}"
        assert math.isclose(float(Zc[k]), Zref, abs_tol=1e-9), \
            f"Zc mismatch at idx {k} (Y_pix={Yp}, Z_pix={Zp}): {float(Zc[k])} vs {Zref}"


def test_hex_axial_ratio_is_sqrt3():
    """Pitch_z / pitch_y must equal √3/2 — the canonical hex lattice."""
    from midas_calibrate_v2.forward.lattice import hex_pixel_pitch

    a = torch.tensor(30.0, dtype=torch.float64)
    pY, pZ = hex_pixel_pitch(a)
    assert math.isclose(float(pZ / pY), math.sqrt(3.0) / 2.0, rel_tol=1e-12)


def test_hex_pixel_area():
    """Hex cell area = 2√3 · a² (rather than 4·a² for a square apothem 2a)."""
    from midas_calibrate_v2.forward.lattice import hex_pixel_area

    a = torch.tensor(30.0, dtype=torch.float64)
    area = float(hex_pixel_area(a))
    expected = 2.0 * math.sqrt(3.0) * 30.0 ** 2
    assert math.isclose(area, expected, rel_tol=1e-12)


def test_hex_lattice_row_parity_with_float_z():
    """Float Z (subpixel offsets) must still produce a consistent row
    parity — the offset is defined by the integer pixel the sample sits
    in, not by the fractional part."""
    from midas_calibrate_v2.forward.lattice import lattice_to_phys

    a = torch.tensor(30.0, dtype=torch.float64)
    BC = torch.tensor(0.0, dtype=torch.float64)

    # Sub-pixel offsets within row 1 (odd) should all share the same
    # row-parity-induced Y shift.
    Y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    Z = torch.tensor([1.0, 1.25, 1.49], dtype=torch.float64)
    Yc, _ = lattice_to_phys(
        Y, Z, lattice="hex_offset_y",
        BC_y=BC, BC_z=BC, apothem=a,
    )
    # All three should have parity = 1 (Z=1.x rounds-floor to 1, odd)
    expected_Yc = -0.5 * 2.0 * 30.0       # = -30
    for v in Yc.tolist():
        assert math.isclose(v, expected_Yc, abs_tol=1e-9)


def test_pixel_to_REta_cartesian_unchanged():
    """The refactor of forward/geometry.py must be bit-identical when
    lattice='cartesian' (the default)."""
    from midas_calibrate_v2.forward.geometry import pixel_to_REta

    torch.manual_seed(0)
    Y = torch.randn(40, dtype=torch.float64) * 200 + 1000
    Z = torch.randn(40, dtype=torch.float64) * 200 + 800
    Lsd = torch.tensor(895_900.0, dtype=torch.float64)
    BC_y = torch.tensor(1024.5, dtype=torch.float64)
    BC_z = torch.tensor(1024.5, dtype=torch.float64)
    tx = torch.tensor(0.05, dtype=torch.float64)
    ty = torch.tensor(0.12, dtype=torch.float64)
    tz = torch.tensor(0.07, dtype=torch.float64)
    p_coeffs = torch.zeros(15, dtype=torch.float64)
    parallax = torch.tensor(0.0, dtype=torch.float64)
    pxY = torch.tensor(75.0, dtype=torch.float64)
    pxZ = torch.tensor(75.0, dtype=torch.float64)

    # default lattice (cartesian) and explicit lattice="cartesian" must agree
    out1 = pixel_to_REta(
        Y, Z, Lsd=Lsd, BC_y=BC_y, BC_z=BC_z,
        tx=tx, ty=ty, tz=tz, p_coeffs=p_coeffs, parallax=parallax,
        pxY=pxY, pxZ=pxZ,
    )
    out2 = pixel_to_REta(
        Y, Z, Lsd=Lsd, BC_y=BC_y, BC_z=BC_z,
        tx=tx, ty=ty, tz=tz, p_coeffs=p_coeffs, parallax=parallax,
        pxY=pxY, pxZ=pxZ, lattice="cartesian",
    )
    assert torch.equal(out1.R_px, out2.R_px)
    assert torch.equal(out1.eta_deg, out2.eta_deg)
    assert torch.equal(out1.rad_um, out2.rad_um)


def test_pseudo_strain_hex_apothem_grad_flows():
    """∂residual / ∂Apothem must be non-trivial.

    Note: ∂R_px / ∂Apothem is exactly 0 in the forward (rad_um and
    px_mean both scale linearly with apothem so the ratio is
    apothem-invariant).  The geometric information about apothem enters
    only through R_pred = Lsd · tan(2θ) / px_mean, where px_mean is
    derived from apothem.  So the right gradient test is on the
    pseudo-strain residual ``r = 1 - R_obs / R_pred``, not on R_px.
    """
    from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

    apothem = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    Lsd = torch.tensor(500_000.0, dtype=torch.float64)
    BC_y = torch.tensor(256.0, dtype=torch.float64)
    BC_z = torch.tensor(238.0, dtype=torch.float64)
    rho_d = torch.tensor(2_000.0, dtype=torch.float64)
    z = torch.tensor(0.0, dtype=torch.float64)

    Y = torch.tensor([100.0, 200.0, 50.0], dtype=torch.float64)
    Z = torch.tensor([100.0, 50.0,  200.0], dtype=torch.float64)
    ring_two_theta = torch.tensor([8.0, 8.0, 8.0], dtype=torch.float64)

    sqrt3 = math.sqrt(3.0)
    p = {
        "Lsd": Lsd, "BC_y": BC_y, "BC_z": BC_z,
        "tx": z, "ty": z, "tz": z, "Parallax": z,
        "pxY": 2.0 * apothem, "pxZ": apothem * sqrt3,
        "Apothem": apothem, "LatticeOrientation": z,
        "iso_R2": z, "iso_R4": z, "iso_R6": z,
        "a1": z, "a2": z, "a3": z, "a4": z, "a5": z, "a6": z,
        "phi1": z, "phi2": z, "phi3": z, "phi4": z, "phi5": z, "phi6": z,
    }
    r = pseudo_strain_residual(
        Y, Z, ring_two_theta, p,
        rho_d=rho_d, lattice="hex_offset_y",
    )
    loss = (r * r).sum()
    loss.backward()
    g = apothem.grad
    assert g is not None
    assert torch.isfinite(g)
    assert abs(float(g)) > 1e-6, f"expected non-zero gradient through Apothem, got {float(g)}"


def test_pseudo_strain_hex_zero_residual_at_truth():
    """Synthetic ring at ground-truth geometry → residual ≈ 0 for hex
    lattice (sanity that the residual path is consistent with the
    forward path)."""
    import numpy as np

    from midas_calibrate_v2.forward.geometry import pixel_to_REta
    from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

    rng = np.random.default_rng(0)
    apothem = torch.tensor(30.0, dtype=torch.float64)
    Lsd = torch.tensor(500_000.0, dtype=torch.float64)
    BC_y = torch.tensor(256.0, dtype=torch.float64)
    BC_z = torch.tensor(238.0, dtype=torch.float64)
    rho_d = torch.tensor(2_000.0, dtype=torch.float64)

    # Forward-simulate a synthetic ring at 2θ = 8°
    two_theta_deg = torch.tensor(8.0, dtype=torch.float64)
    two_theta_rad = float(two_theta_deg) * math.pi / 180.0
    R_um = float(Lsd) * math.tan(two_theta_rad)

    # Pixel-pitch (mean) for the hex
    px_mean = (2.0 * float(apothem) + float(apothem) * math.sqrt(3.0)) * 0.5
    R_px = R_um / px_mean

    # Sample 200 points around the ring on the (Yc, Zc) physical plane
    eta = rng.uniform(0, 2 * math.pi, size=200)
    Yc = -R_um * np.cos(eta)             # matches forward sign convention
    Zc = +R_um * np.sin(eta)
    # Invert hex centroid mapping for each (Yc, Zc) → (Y_pix, Z_pix).
    # For BC=(BCy,BCz), pitch_y=2a, pitch_z=a√3, Z determines parity:
    # Y_pix = BC_y - Yc / (2a) - 0.5 * (Z_pix mod 2)
    # Z_pix = Zc / (a√3) + BC_z
    a = float(apothem)
    Z_pix = Zc / (a * math.sqrt(3.0)) + float(BC_z)
    parity = (np.floor(Z_pix).astype(np.int64) % 2).astype(np.float64)
    Y_pix = float(BC_y) - Yc / (2.0 * a) - 0.5 * parity

    Y_pix_t = torch.tensor(Y_pix, dtype=torch.float64)
    Z_pix_t = torch.tensor(Z_pix, dtype=torch.float64)
    ring_two_theta = torch.full((Y_pix_t.shape[0],), 8.0, dtype=torch.float64)

    # Parameter dict the same as unpack_spec would emit
    z = torch.tensor(0.0, dtype=torch.float64)
    p = {
        "Lsd": Lsd, "BC_y": BC_y, "BC_z": BC_z,
        "tx": z, "ty": z, "tz": z,
        "Parallax": z,
        "pxY": torch.tensor(2.0 * a), "pxZ": torch.tensor(a * math.sqrt(3.0)),
        "Apothem": apothem, "LatticeOrientation": z,
        "iso_R2": z, "iso_R4": z, "iso_R6": z,
        "a1": z, "a2": z, "a3": z, "a4": z, "a5": z, "a6": z,
        "phi1": z, "phi2": z, "phi3": z, "phi4": z, "phi5": z, "phi6": z,
    }

    r = pseudo_strain_residual(
        Y_pix_t, Z_pix_t, ring_two_theta, p,
        rho_d=rho_d, lattice="hex_offset_y",
    )
    # Residual is (1 - R_obs / R_pred); ground truth → ~0 (up to BC-inversion
    # numerical roundoff on the sample points).
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-9)


def test_pixel_to_REta_hex_orientation_zero_is_noop():
    """orientation_deg = 0 must produce the same output as orientation_deg = None."""
    from midas_calibrate_v2.forward.geometry import pixel_to_REta

    Y = torch.linspace(0, 256, 64, dtype=torch.float64)
    Z = torch.linspace(0, 256, 64, dtype=torch.float64)
    apothem = torch.tensor(30.0, dtype=torch.float64)
    Lsd = torch.tensor(500_000.0, dtype=torch.float64)
    BC_y = torch.tensor(128.0, dtype=torch.float64)
    BC_z = torch.tensor(128.0, dtype=torch.float64)
    z = torch.tensor(0.0, dtype=torch.float64)
    p_coeffs = torch.zeros(15, dtype=torch.float64)
    pxY = torch.tensor(0.0, dtype=torch.float64)
    pxZ = torch.tensor(0.0, dtype=torch.float64)

    base = pixel_to_REta(
        Y, Z, Lsd=Lsd, BC_y=BC_y, BC_z=BC_z, tx=z, ty=z, tz=z,
        p_coeffs=p_coeffs, parallax=z, pxY=pxY, pxZ=pxZ,
        lattice="hex_offset_y", apothem=apothem,
    )
    orient0 = pixel_to_REta(
        Y, Z, Lsd=Lsd, BC_y=BC_y, BC_z=BC_z, tx=z, ty=z, tz=z,
        p_coeffs=p_coeffs, parallax=z, pxY=pxY, pxZ=pxZ,
        lattice="hex_offset_y", apothem=apothem,
        orientation_deg=torch.tensor(0.0, dtype=torch.float64),
    )
    assert torch.equal(base.R_px, orient0.R_px)


def test_lattice_to_phys_unknown_raises():
    from midas_calibrate_v2.forward.lattice import lattice_to_phys

    Y = torch.tensor([0.0])
    Z = torch.tensor([0.0])
    BC = torch.tensor(0.0)
    with pytest.raises(ValueError, match="Unknown lattice"):
        lattice_to_phys(Y, Z, lattice="square", BC_y=BC, BC_z=BC,
                        pxY=torch.tensor(1.0), pxZ=torch.tensor(1.0))


def test_hex_synthetic_apothem_recovery():
    """End-to-end synthetic refinement: generate ring observations from
    a hex spec with apothem=30, perturb apothem to 30.6, refine, expect
    recovery to within 1e-2 μm.

    Uses gradient descent on apothem only (other params held fixed),
    which is the minimal demonstration that the autograd path is
    correctly wired through the new lattice plumbing.  Multi-distance
    data is not needed here because all other geometric scales are
    held fixed; in a real refinement at one distance, apothem and Lsd
    would be degenerate.
    """
    import numpy as np

    from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual

    a_true = 30.0
    Lsd = torch.tensor(800_000.0, dtype=torch.float64)
    BC_y = torch.tensor(256.0, dtype=torch.float64)
    BC_z = torch.tensor(238.0, dtype=torch.float64)
    rho_d = torch.tensor(2_000.0, dtype=torch.float64)
    z = torch.tensor(0.0, dtype=torch.float64)

    # Synthesise observations on 2 rings (2θ = 6° and 10°) at apothem 30.
    rings_deg = [6.0, 10.0]
    rng = np.random.default_rng(42)
    Y_list, Z_list, ring_list = [], [], []
    for theta_deg in rings_deg:
        theta = theta_deg * math.pi / 180.0
        R_um = float(Lsd) * math.tan(theta)
        eta = rng.uniform(0, 2 * math.pi, size=120)
        Yc = -R_um * np.cos(eta)
        Zc = +R_um * np.sin(eta)
        Z_pix = Zc / (a_true * math.sqrt(3.0)) + float(BC_z)
        parity = (np.floor(Z_pix).astype(np.int64) % 2).astype(np.float64)
        Y_pix = float(BC_y) - Yc / (2.0 * a_true) - 0.5 * parity
        Y_list.append(Y_pix)
        Z_list.append(Z_pix)
        ring_list.append(np.full_like(Y_pix, theta_deg))

    Y_pix_t = torch.tensor(np.concatenate(Y_list), dtype=torch.float64)
    Z_pix_t = torch.tensor(np.concatenate(Z_list), dtype=torch.float64)
    ring_two_theta = torch.tensor(np.concatenate(ring_list), dtype=torch.float64)

    # Perturb apothem and refine
    apothem = torch.tensor(30.6, dtype=torch.float64, requires_grad=True)
    sqrt3 = math.sqrt(3.0)
    optimizer = torch.optim.LBFGS([apothem], lr=0.5, max_iter=40,
                                   tolerance_grad=1e-12, tolerance_change=1e-14)

    def _closure():
        optimizer.zero_grad()
        p = {
            "Lsd": Lsd, "BC_y": BC_y, "BC_z": BC_z,
            "tx": z, "ty": z, "tz": z, "Parallax": z,
            "pxY": 2.0 * apothem, "pxZ": apothem * sqrt3,
            "Apothem": apothem, "LatticeOrientation": z,
            "iso_R2": z, "iso_R4": z, "iso_R6": z,
            "a1": z, "a2": z, "a3": z, "a4": z, "a5": z, "a6": z,
            "phi1": z, "phi2": z, "phi3": z, "phi4": z, "phi5": z, "phi6": z,
        }
        r = pseudo_strain_residual(
            Y_pix_t, Z_pix_t, ring_two_theta, p,
            rho_d=rho_d, lattice="hex_offset_y",
        )
        loss = 0.5 * (r * r).sum()
        loss.backward()
        return loss

    optimizer.step(_closure)
    recovered = float(apothem.detach())
    assert math.isclose(recovered, a_true, abs_tol=1e-2), \
        f"refinement failed: expected {a_true}, got {recovered}"


def test_hex_lattice_device_portability_cpu():
    """Verify lattice_to_phys and pixel_to_REta with hex run on CPU
    (smoke); MPS / CUDA paths are exercised by the dedicated device tests
    in tests/test_devices.py."""
    from midas_calibrate_v2.forward.geometry import pixel_to_REta
    from midas_calibrate_v2.forward.lattice import lattice_to_phys

    Y = torch.linspace(0, 100, 64, dtype=torch.float32)
    Z = torch.linspace(0, 100, 64, dtype=torch.float32)
    apothem = torch.tensor(30.0, dtype=torch.float32)
    BC = torch.tensor(50.0, dtype=torch.float32)

    Yc, Zc = lattice_to_phys(
        Y, Z, lattice="hex_offset_y",
        BC_y=BC, BC_z=BC, apothem=apothem,
    )
    assert Yc.dtype == torch.float32
    assert Zc.dtype == torch.float32

    z = torch.tensor(0.0, dtype=torch.float32)
    p_coeffs = torch.zeros(15, dtype=torch.float32)
    out = pixel_to_REta(
        Y, Z, Lsd=torch.tensor(500_000.0, dtype=torch.float32),
        BC_y=BC, BC_z=BC, tx=z, ty=z, tz=z,
        p_coeffs=p_coeffs, parallax=z,
        pxY=torch.tensor(0.0, dtype=torch.float32),
        pxZ=torch.tensor(0.0, dtype=torch.float32),
        lattice="hex_offset_y", apothem=apothem,
    )
    assert out.R_px.dtype == torch.float32
    assert torch.isfinite(out.R_px).all()
    assert torch.isfinite(out.eta_deg).all()
