"""Tests for `midas_defect.williamson_hall`."""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import pytest

from midas_defect.williamson_hall import (
    WHResult, williamson_hall, dislocation_density_per_grain,
    ModifiedWHResult, contrast_factors_for_fits, modified_williamson_hall,
)
from midas_defect.contrast_factor import cubic_stiffness, average_contrast_factor
from midas_defect.lattice import fcc_cu_crystal
from midas_defect.bragg_diffuse import predicted_reflection_points


def _bragg_cloud_with_strain(eps_inject, seed=0, n_per=14):
    """Synthetic on-lattice Bragg voxels with radial spread growing ∝ |q|."""
    cr = fcc_cu_crystal()
    U = np.eye(3)[None]
    P = predicted_reflection_points(U, cr, q_max_inv_A=8.0).numpy()
    rng = np.random.default_rng(seed)
    qb, I = [], []
    for p in P:
        pm = np.linalg.norm(p)
        for _ in range(n_per):
            rad = rng.normal(0.0, 0.001 + eps_inject * pm)
            qb.append(p + rad * p / pm); I.append(100.0)
    return np.array(qb), np.array(I), U, cr


def test_dislocation_density_monotonic_in_strain():
    """More injected radial spread ⇒ higher recovered dislocation density."""
    qa, Ia, U, cr = _bragg_cloud_with_strain(0.0006)
    qb, Ib, _, _ = _bragg_cloud_with_strain(0.0020)
    rho_lo = dislocation_density_per_grain(qa, Ia, U, cr).rho_median_per_m2
    rho_hi = dislocation_density_per_grain(qb, Ib, U, cr).rho_median_per_m2
    assert np.isfinite(rho_lo) and np.isfinite(rho_hi)
    assert rho_hi > rho_lo


def test_dislocation_density_uses_fcc_burgers_default():
    """Burgers defaults to a/√2 from the crystal cell when not supplied."""
    qa, Ia, U, cr = _bragg_cloud_with_strain(0.0010)
    res = dislocation_density_per_grain(qa, Ia, U, cr)
    assert res.burgers_A == pytest.approx(3.6356 / math.sqrt(2), abs=1e-3)


@dataclass
class _FakeFit:
    """Minimal AsterismFit-like stub with the fields williamson_hall consumes."""
    q_fit: np.ndarray
    sigma_eig: np.ndarray
    integrated_intensity: float
    hkl: tuple = (1, 0, 0)


def _synth_fits(
    n: int = 12,
    sigma_size: float = 0.012,   # 1/Å — corresponds to D ≈ 520 Å
    strain_eps: float = 0.004,   # dimensionless strain slope
    q_min: float = 1.0, q_max: float = 6.5,
    noise: float = 0.0005,
    rng: np.random.Generator | None = None,
) -> list[_FakeFit]:
    rng = rng or np.random.default_rng(0)
    q_mags = np.linspace(q_min, q_max, n)
    fits = []
    for q in q_mags:
        sigma_max = sigma_size + strain_eps * q + rng.normal(scale=noise)
        # split anisotropically so sigma_eig.max() returns sigma_max
        eigs = np.array([sigma_max, sigma_max * 0.6, sigma_max * 0.4])
        q_dir = np.array([q, 0.0, 0.0])  # along x; magnitude is all we need
        fits.append(_FakeFit(
            q_fit=q_dir,
            sigma_eig=eigs,
            integrated_intensity=100.0 + rng.uniform(-10, 10),
        ))
    return fits


@pytest.mark.unit
def test_williamson_hall_recovers_planted_slope_and_intercept():
    """A planted (σ_size=0.012, ε=0.004) should be recovered within 5 %."""
    fits = _synth_fits(n=20, sigma_size=0.012, strain_eps=0.004, noise=1e-5)
    res = williamson_hall(fits, burgers_A=4.29, sigma_kind="max")
    assert isinstance(res, WHResult)
    assert res.sigma_size == pytest.approx(0.012, rel=0.05)
    assert res.strain_eps == pytest.approx(0.004, rel=0.05)
    # domain size: 2π / σ_size = 2π/0.012 ≈ 524 Å
    assert res.domain_size_A == pytest.approx(2 * math.pi / 0.012, rel=0.05)
    # correlation should be very high (low noise, perfectly linear truth)
    assert res.correlation > 0.99
    assert res.n_hkls == 20


@pytest.mark.unit
def test_williamson_hall_dislocation_density_scales_with_eps_squared():
    """ρ = 2 ε² / b² — doubling ε quadruples ρ."""
    fits_low = _synth_fits(n=12, strain_eps=0.002, noise=1e-6)
    fits_hi  = _synth_fits(n=12, strain_eps=0.004, noise=1e-6)
    r_low = williamson_hall(fits_low, burgers_A=4.29)
    r_hi  = williamson_hall(fits_hi,  burgers_A=4.29)
    ratio = r_hi.dislocation_density_per_m2 / r_low.dislocation_density_per_m2
    assert ratio == pytest.approx(4.0, rel=0.05)


@pytest.mark.unit
def test_williamson_hall_burgers_vector_inversely_squared():
    """ρ ∝ 1/b² — doubling b reduces ρ by 4× for the same ε."""
    fits = _synth_fits(n=12, strain_eps=0.004, noise=1e-6)
    r_a = williamson_hall(fits, burgers_A=4.0)
    r_b = williamson_hall(fits, burgers_A=8.0)
    assert r_a.dislocation_density_per_m2 / r_b.dislocation_density_per_m2 == \
        pytest.approx(4.0, rel=1e-3)


@pytest.mark.unit
def test_williamson_hall_sigma_kind_mean_vs_max():
    """Anisotropic Σ → sigma_kind='mean' gives lower intercept than 'max'."""
    fits = _synth_fits(n=12, sigma_size=0.012, strain_eps=0.004, noise=1e-6)
    r_max = williamson_hall(fits, sigma_kind="max")
    r_mean = williamson_hall(fits, sigma_kind="mean")
    # mean eigenvalue is (1 + 0.6 + 0.4)/3 = 0.6667 of max → lower intercept+slope
    assert r_mean.sigma_size < r_max.sigma_size
    assert r_mean.strain_eps < r_max.strain_eps


@pytest.mark.unit
def test_williamson_hall_raises_with_too_few_fits():
    fits = _synth_fits(n=3)
    with pytest.raises(ValueError, match="at least 4"):
        williamson_hall(fits)


@pytest.mark.unit
def test_williamson_hall_unknown_sigma_kind():
    fits = _synth_fits(n=8)
    with pytest.raises(ValueError, match="unknown sigma_kind"):
        williamson_hall(fits, sigma_kind="median")


# ---------------------------------------------------------------------------
# Modified Williamson-Hall (contrast-factor corrected)
# ---------------------------------------------------------------------------

#: A spread of FCC reflections (distinct |q| and distinct H²) for mWH tests.
_MWH_HKLS = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1),
             (2, 2, 2), (4, 0, 0), (3, 3, 1), (4, 2, 0)]


def _mwh_fits_with_anisotropy(C6, a_A=3.615, eps_true=0.004, sigma_size=0.012):
    """Build fits whose breadth follows σ = σ_size + ε·(|q|·C̄^{1/2}).

    i.e. the *contrast-corrected* axis is the linear one — so plain WH (vs |q|)
    scatters and modified WH (vs |q|·C̄^{1/2}) collapses to a line.
    """
    fits = []
    for hkl in _MWH_HKLS:
        q = 2 * math.pi * math.sqrt(hkl[0] ** 2 + hkl[1] ** 2 + hkl[2] ** 2) / a_A
        cbar = float(average_contrast_factor(C6, hkl, family="fcc"))
        sigma = sigma_size + eps_true * (q * math.sqrt(cbar))
        eigs = np.array([sigma, sigma * 0.6, sigma * 0.4])
        fits.append(_FakeFit(q_fit=np.array([q, 0.0, 0.0]),
                             sigma_eig=eigs, integrated_intensity=100.0, hkl=hkl))
    return fits


@pytest.mark.unit
def test_modified_wh_collapses_anisotropic_scatter():
    """When broadening is truly ∝ |q|·C̄^{1/2}, modified WH correlates better
    than plain WH and recovers the planted strain."""
    C6 = cubic_stiffness(168.4, 121.4, 75.4)        # Cu
    fits = _mwh_fits_with_anisotropy(C6, eps_true=0.004, sigma_size=0.012)
    cbars = contrast_factors_for_fits(fits, C6, family="fcc")
    res = modified_williamson_hall(fits, cbars, burgers_A=2.556)
    assert isinstance(res, ModifiedWHResult)
    assert res.correlation > res.correlation_uncorrected
    assert res.correlation > 0.999
    assert res.strain_eps == pytest.approx(0.004, rel=0.05)
    assert res.sigma_size == pytest.approx(0.012, rel=0.05)


@pytest.mark.unit
def test_contrast_factors_for_fits_caches_by_hkl():
    """Equal-hkl fits get identical C̄ and the list aligns with fits."""
    C6 = cubic_stiffness(168.4, 121.4, 75.4)
    fits = _mwh_fits_with_anisotropy(C6)
    cbars = contrast_factors_for_fits(fits, C6, family="fcc")
    assert len(cbars) == len(fits)
    assert all(c > 0 for c in cbars)
    # (111) must have a smaller contrast factor than (200) for Cu
    by_hkl = {f.hkl: c for f, c in zip(fits, cbars)}
    assert by_hkl[(1, 1, 1)] < by_hkl[(2, 0, 0)]


@pytest.mark.unit
def test_modified_wh_raises_on_misaligned_contrast_factors():
    C6 = cubic_stiffness(168.4, 121.4, 75.4)
    fits = _mwh_fits_with_anisotropy(C6)
    with pytest.raises(ValueError, match="align"):
        modified_williamson_hall(fits, [0.2, 0.3])


@pytest.mark.unit
def test_modified_wh_raises_with_too_few_fits():
    C6 = cubic_stiffness(168.4, 121.4, 75.4)
    fits = _mwh_fits_with_anisotropy(C6)[:3]
    cbars = contrast_factors_for_fits(fits, C6, family="fcc")
    with pytest.raises(ValueError, match="at least 4"):
        modified_williamson_hall(fits, cbars)
