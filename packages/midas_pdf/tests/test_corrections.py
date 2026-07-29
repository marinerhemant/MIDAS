import numpy as np
import torch

from midas_pdf.corrections import (
    apply_detector_efficiency,
    cos_scattering_angle,
    detector_efficiency,
    flat_plate_transmission,
    linear_attenuation_um,
)

WAVELENGTH_A = 0.1665  # ~74 keV


def test_linear_attenuation_element_and_compound():
    mu_si = linear_attenuation_um("Si", WAVELENGTH_A)
    assert mu_si > 0
    # CdTe compound (mass fractions ~ Cd 0.468, Te 0.532), density 5.85
    mu_cdte = linear_attenuation_um({"Cd": 0.468, "Te": 0.532}, WAVELENGTH_A,
                                    density_g_cm3=5.85)
    assert mu_cdte > mu_si        # high-Z detector absorbs much more


def test_detector_efficiency_increases_with_Q():
    q = torch.linspace(0.5, 25.0, 200, dtype=torch.float64)
    eta = detector_efficiency(q, wavelength_A=WAVELENGTH_A,
                              material={"Cd": 0.468, "Te": 0.532},
                              thickness_um=1000.0, density_g_cm3=5.85)
    assert torch.all((eta > 0) & (eta <= 1.0))
    assert float(eta[-1]) > float(eta[0])     # longer path at high angle


def test_apply_detector_efficiency_boosts_and_propagates_sigma():
    q = torch.linspace(0.5, 25.0, 100, dtype=torch.float64)
    I = torch.ones_like(q)
    sig = 0.1 * torch.ones_like(q)
    I2, sig2 = apply_detector_efficiency(
        I, q, wavelength_A=WAVELENGTH_A, material="Si",
        thickness_um=500.0, sigma=sig,
    )
    assert torch.all(I2 >= I)                 # divide by eta<=1 -> >= I
    assert torch.allclose(sig2 / I2, sig / I, atol=1e-9)  # relative error preserved


def test_flat_plate_transmission_limits():
    q = torch.linspace(0.0, 20.0, 100, dtype=torch.float64)
    A = flat_plate_transmission(q, wavelength_A=WAVELENGTH_A, mu_um=1e-3, thickness_um=100.0)
    assert torch.all((A > 0) & (A <= 1.0))
    # at Q=0 (psi=0) reduces to exp(-mu t)
    assert abs(float(A[0]) - np.exp(-1e-3 * 100.0)) < 1e-6


def test_detector_efficiency_differentiable():
    q = torch.linspace(1.0, 20.0, 40, dtype=torch.float64, requires_grad=True)
    eta = detector_efficiency(q, wavelength_A=WAVELENGTH_A, material="Si",
                              thickness_um=300.0)
    eta.sum().backward()
    assert q.grad is not None and torch.all(torch.isfinite(q.grad))


# ---------------------------------------------------------------------------
# Paalman-Pings tests
# ---------------------------------------------------------------------------

from midas_pdf.corrections import (
    paalman_pings_cell_only,
    paalman_pings_cylinder_in_cylinder,
)


def test_paalman_pings_zero_mu_returns_unity():
    """μ → 0 → all absorption factors → 1 (no absorption anywhere)."""
    q = torch.linspace(1.0, 20.0, 20, dtype=torch.float64)
    pp = paalman_pings_cylinder_in_cylinder(
        q, wavelength_A=WAVELENGTH_A,
        mu_sample_um=0.0, mu_container_um=0.0,
        R_sample_um=500.0, R_container_um=530.0, n_grid=32,
    )
    for key in ("A_s_sc", "A_c_sc", "A_c_c", "A_s_s"):
        assert torch.allclose(pp[key], torch.ones_like(pp[key]), atol=1e-8), key


def test_paalman_pings_bounded_between_zero_and_one():
    """All P-P factors must lie in (0, 1] for physical μ, R."""
    q = torch.linspace(1.0, 25.0, 40, dtype=torch.float64)
    pp = paalman_pings_cylinder_in_cylinder(
        q, wavelength_A=WAVELENGTH_A,
        mu_sample_um=1e-3, mu_container_um=1e-4,          # realistic
        R_sample_um=500.0, R_container_um=530.0, n_grid=32,
    )
    for key in ("A_s_sc", "A_c_sc", "A_c_c", "A_s_s"):
        A = pp[key]
        assert torch.all(A > 0) and torch.all(A <= 1.0 + 1e-9), key


def test_paalman_pings_sample_alone_below_container_shadowed():
    """A_s_s (sample without container) > A_s_sc (container casts shadow)."""
    q = torch.linspace(1.0, 25.0, 20, dtype=torch.float64)
    pp = paalman_pings_cylinder_in_cylinder(
        q, wavelength_A=WAVELENGTH_A,
        mu_sample_um=1e-3, mu_container_um=1e-4,
        R_sample_um=500.0, R_container_um=530.0, n_grid=32,
    )
    # Kapton wall adds a tiny extra attenuation; A_s_s should be >= A_s_sc.
    assert torch.all(pp["A_s_s"] >= pp["A_s_sc"] - 1e-6)


def test_paalman_pings_cell_only_matches_A_c_c():
    """The stand-alone empty-cell function should agree with A_c_c."""
    q = torch.linspace(1.0, 20.0, 25, dtype=torch.float64)
    pp = paalman_pings_cylinder_in_cylinder(
        q, wavelength_A=WAVELENGTH_A,
        mu_sample_um=1e-3, mu_container_um=1e-4,
        R_sample_um=500.0, R_container_um=530.0, n_grid=40,
    )
    A_cell = paalman_pings_cell_only(
        q, wavelength_A=WAVELENGTH_A, mu_container_um=1e-4,
        R_inner_um=500.0, R_outer_um=530.0, n_grid=40,
    )
    assert torch.allclose(pp["A_c_c"], A_cell, rtol=1e-5, atol=1e-8)


def test_paalman_pings_grid_convergence():
    """Increasing n_grid should not change the result by more than a few %."""
    q = torch.tensor([2.0, 10.0, 20.0], dtype=torch.float64)
    pp_coarse = paalman_pings_cylinder_in_cylinder(
        q, wavelength_A=WAVELENGTH_A,
        mu_sample_um=1e-3, mu_container_um=1e-4,
        R_sample_um=500.0, R_container_um=530.0, n_grid=24,
    )
    pp_fine = paalman_pings_cylinder_in_cylinder(
        q, wavelength_A=WAVELENGTH_A,
        mu_sample_um=1e-3, mu_container_um=1e-4,
        R_sample_um=500.0, R_container_um=530.0, n_grid=64,
    )
    for key in ("A_s_sc", "A_c_sc", "A_c_c", "A_s_s"):
        rel = torch.abs(pp_coarse[key] - pp_fine[key]) / pp_fine[key].clamp(min=1e-6)
        assert torch.all(rel < 0.02), f"{key}: max rel err {float(rel.max()):.3f}"
