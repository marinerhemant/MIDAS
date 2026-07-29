"""Multiple scattering — first-principles estimators (Tier 2/3).

Two ingredients, both built on the differentiable cross-section in
:mod:`midas_pdf.cross_section`:

* :func:`slab_single_scattering_factor` — the **analytic, differentiable**
  attenuation-weighted single-scattering factor for a flat slab in symmetric
  transmission. Closed form, so it is exact and autograd-friendly.
* :func:`multiple_scattering_mc` — an **analog Monte-Carlo** photon-transport
  estimator for the same slab that separates single from multiple scattering and
  returns the multiple-scattering fraction β(Q). This is unambiguously correct
  (standard analog MC) and serves as the *reference* the differentiable
  double-scattering integral (Tier 3, next) will be validated against. The MC
  itself is not differentiable (it samples discrete scattering events).

The MC is parameterized by the dimensionless **optical depth** ``tau = mu * t``
and single-scattering **albedo** ``= Sigma_scatter / mu`` (both unit-safe);
:func:`slab_optical_params` estimates them from the composition, wavelength,
thickness, and density using the NIST mass-attenuation tables (already in
midas-hkls) and the scattering cross-section from this package.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from midas_hkls.absorption import mass_attenuation_coefficient

from .composition import Composition
from .corrections import cos_scattering_angle
from .cross_section import differential_cross_section, total_cross_section

__all__ = [
    "slab_single_scattering_factor",
    "slab_optical_params",
    "multiple_scattering_mc",
    "multiple_scattering_mc_cylinder",
    "cylinder_effective_tau",
    "CYLINDER_SLAB_FACTOR",
    "slab_double_scattering",
    "ms_background_on_grid",
]

_FOUR_PI = 4.0 * float(np.pi)
_R_E_A = 2.8179403262e-5            # classical electron radius, Å
_A_PER_UM = 1.0e4                   # Å per micron


def slab_single_scattering_factor(
    q: torch.Tensor | np.ndarray,
    *,
    thickness_um: float,
    mu_um: float,
    wavelength_A: float,
) -> torch.Tensor:
    """Attenuation-weighted single-scattering path factor A1(Q) for a flat slab
    in symmetric transmission (beam normal to the slab):

        A1 = ∫_0^t exp(-mu z) exp(-mu (t-z)/cos psi) dz
           = exp(-mu t / c) * (1 - exp(-mu t (1 - 1/c))) / (mu (1 - 1/c)),

    with ``c = cos psi`` and the ``c -> 1`` limit ``t exp(-mu t)``. Multiply by
    ``dσ/dΩ(Q)`` to get the single-scattering intensity. Differentiable in Q,
    thickness, mu, wavelength.
    """
    c = cos_scattering_angle(q, wavelength_A).clamp(min=1e-3)   # forward angles
    mt = mu_um * thickness_um
    beta = mu_um * (1.0 - 1.0 / c)                              # exponent rate
    bt = beta * thickness_um
    series = torch.where(
        bt.abs() < 1e-8,
        torch.as_tensor(thickness_um, dtype=torch.float64) * torch.ones_like(c),
        (1.0 - torch.exp(-bt)) / torch.where(beta.abs() < 1e-30,
                                             torch.ones_like(beta), beta),
    )
    return torch.exp(-mt / c) * series


def slab_optical_params(
    composition: Composition,
    *,
    wavelength_A: float,
    thickness_um: float,
    number_density_A3: float,
    packing_fraction: float = 1.0,
) -> Tuple[float, float, float]:
    """Estimate ``(mu_um, tau, albedo)`` for a slab.

    ``mu`` is the total linear attenuation (NIST mass-attenuation × mass density
    implied by the number density and mean atomic mass); the scattering linear
    coefficient comes from the total scattering cross-section in this package.
    ``tau = mu * thickness``; ``albedo = Sigma_scatter / mu``.
    """
    # scattering cross-section per atom (Thomson units -> Å^2)
    sigma_thom = float(total_cross_section(composition, wavelength_A=wavelength_A))
    sigma_scat_A2 = sigma_thom * _R_E_A * _R_E_A
    n_A3 = number_density_A3 * packing_fraction
    Sigma_scat_per_A = n_A3 * sigma_scat_A2                  # 1/Å
    Sigma_scat_um = Sigma_scat_per_A * _A_PER_UM             # 1/µm

    # total attenuation: composition-weighted mass-attenuation × mass density.
    from midas_hkls.absorption import atomic_mass
    fr = composition.fractions
    mbar = sum(f * atomic_mass(e) for e, f in zip(composition.elements, fr))  # g/mol
    # mass density from number density: rho = n[1/Å^3] * mbar[g/mol] / N_A * 1e24 Å^3/cm^3
    N_A = 6.02214076e23
    rho = n_A3 * mbar / N_A * 1.0e24                          # g/cm^3
    mu_over_rho = sum(
        f * float(mass_attenuation_coefficient(e, wavelength_A))
        for e, f in zip(composition.elements, fr)
    )                                                        # cm^2/g
    mu_per_cm = mu_over_rho * rho                             # 1/cm
    mu_um = mu_per_cm / _A_PER_UM                             # 1/µm

    # ensure scattering does not exceed total (numerical guard)
    Sigma_scat_um = min(Sigma_scat_um, mu_um)
    albedo = Sigma_scat_um / mu_um if mu_um > 0 else 0.0
    tau = mu_um * thickness_um
    return mu_um, tau, albedo


def _build_angle_sampler(
    composition: Composition, wavelength_A: float, n_grid: int = 720,
) -> Tuple[np.ndarray, np.ndarray]:
    """CDF over scattering angle psi from dσ/dΩ(psi) sin(psi); returns
    (psi_grid, cdf) for inverse-transform sampling."""
    psi = torch.linspace(1e-4, float(np.pi), n_grid, dtype=torch.float64)
    Q = (_FOUR_PI / wavelength_A) * torch.sin(0.5 * psi)
    dsig = differential_cross_section(Q, composition, wavelength_A=wavelength_A)
    w = (dsig * torch.sin(psi)).detach().cpu().numpy()
    w = np.clip(w, 0.0, None)
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    return psi.detach().cpu().numpy(), cdf


def _scatter_directions(dirs: np.ndarray, cos_s: np.ndarray, rng) -> np.ndarray:
    """Rotate each direction by scattering angle (cos_s) and a uniform azimuth."""
    n = dirs.shape[0]
    phi = rng.uniform(0.0, 2 * np.pi, n)
    sin_s = np.sqrt(np.clip(1.0 - cos_s**2, 0.0, 1.0))
    # build an orthonormal basis (u, v) perpendicular to each dir d
    d = dirs
    # pick a helper axis least aligned with d
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    near_z = np.abs(d[:, 2]) > 0.9
    helper[near_z] = np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, d)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(d, u)
    new = (cos_s[:, None] * d
           + (sin_s * np.cos(phi))[:, None] * u
           + (sin_s * np.sin(phi))[:, None] * v)
    new /= np.linalg.norm(new, axis=1, keepdims=True)
    return new


def multiple_scattering_mc(
    composition: Composition,
    *,
    wavelength_A: float,
    tau: float,
    albedo: float,
    n_photons: int = 200_000,
    n_psi: int = 90,
    max_events: int = 40,
    seed: int = 0,
) -> dict:
    """Analog Monte-Carlo multiple-scattering estimate for a flat slab.

    Photons enter normal to a slab of optical depth ``tau`` with single-
    scattering ``albedo``; scattering angles are sampled from the composition's
    differential cross-section. Forward-escaping photons are tallied by exit
    angle psi (-> Q) and by scatter order. Returns a dict with ``Q``, ``I_single``,
    ``I_multiple``, ``beta`` ( = I_multiple / (I_single + I_multiple) ), and the
    raw counts. **Not differentiable** (reference for the Tier-3 analytic model).
    """
    rng = np.random.default_rng(seed)
    psi_grid, cdf = _build_angle_sampler(composition, wavelength_A)

    z = np.zeros(n_photons)                       # depth in [0, 1] (thickness units)
    d = np.tile(np.array([0.0, 0.0, 1.0]), (n_photons, 1))   # along +z (beam)
    order = np.zeros(n_photons, dtype=np.int32)
    alive = np.ones(n_photons, dtype=bool)
    # accumulators for forward-escaping photons
    exit_cos = []      # cos(psi) of exit direction
    exit_order = []

    for _ in range(max_events):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        w = d[idx, 2]                              # z-cosine
        # optical free path ~ Exp(1); geometric (thickness units) = ell/tau along ray
        ell = rng.exponential(1.0, idx.size)
        # distance (thickness units) to z-boundary along the ray
        with np.errstate(divide="ignore", invalid="ignore"):
            dist_b = np.where(w > 0, (1.0 - z[idx]) / w,
                              np.where(w < 0, z[idx] / (-w), np.inf))
        path_geo = ell / tau                       # geometric distance to interaction
        escapes = path_geo >= dist_b
        # --- escaping photons ---
        esc = idx[escapes]
        if esc.size:
            we = d[esc, 2]
            fwd = we > 0                           # forward (transmission) exits
            scattered = order[esc] >= 1
            keep = fwd & scattered
            exit_cos.append(we[keep])
            exit_order.append(order[esc][keep])
            alive[esc] = False
        # --- interacting photons ---
        intr = idx[~escapes]
        if intr.size:
            z[intr] = z[intr] + path_geo[~escapes] * d[intr, 2]
            # scatter vs absorb
            u = rng.uniform(0.0, 1.0, intr.size)
            scatter = u < albedo
            absb = intr[~scatter]
            alive[absb] = False
            sc = intr[scatter]
            if sc.size:
                rr = rng.uniform(0.0, 1.0, sc.size)
                j = np.searchsorted(cdf, rr)
                j = np.clip(j, 0, psi_grid.size - 1)
                cos_s = np.cos(psi_grid[j])
                d[sc] = _scatter_directions(d[sc], cos_s, rng)
                order[sc] += 1

    exit_cos = np.concatenate(exit_cos) if exit_cos else np.array([])
    exit_order = np.concatenate(exit_order) if exit_order else np.array([])

    psi = np.arccos(np.clip(exit_cos, -1.0, 1.0))
    edges = np.linspace(0.0, np.pi / 2, n_psi + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    single = np.histogram(psi[exit_order == 1], bins=edges)[0].astype(float)
    double = np.histogram(psi[exit_order == 2], bins=edges)[0].astype(float)
    multi = np.histogram(psi[exit_order >= 2], bins=edges)[0].astype(float)
    total = single + multi
    beta = np.divide(multi, total, out=np.zeros_like(total), where=total > 0)
    Q = (_FOUR_PI / wavelength_A) * np.sin(0.5 * centers)
    return {
        "Q": torch.as_tensor(Q, dtype=torch.float64),
        "psi": torch.as_tensor(centers, dtype=torch.float64),
        "I_single": torch.as_tensor(single, dtype=torch.float64),
        "I_double": torch.as_tensor(double, dtype=torch.float64),   # exactly order 2
        "I_multiple": torch.as_tensor(multi, dtype=torch.float64),  # order >= 2
        "beta": torch.as_tensor(beta, dtype=torch.float64),
        "n_single": float(single.sum()),
        "n_multiple": float(multi.sum()),
    }


def slab_double_scattering(
    composition: Composition,
    *,
    wavelength_A: float,
    tau: float,
    albedo: float,
    q_max: float = 20.0,
    n_psi: int = 60,
    n_z: int = 48,
    n_theta: int = 96,
    n_phi: int = 96,
    geometric_series: bool = True,
) -> dict:
    """Differentiable double-scattering for a flat slab (the Tier-3 estimator).

    Computes the single- and double-scattering intensity into forward scattering
    angle ψ by closed-form-plus-quadrature integration (no sampling), so it is
    fully differentiable in composition / τ / albedo and validates against
    :func:`multiple_scattering_mc`.

    Geometry (thickness = 1, μ = ``tau``, scattering coefficient = ``albedo·τ``):

      I1(ψ) = (a τ) σ̂(ψ) A1(ψ)
      I2(ψ) = (a τ)^2 ∫dz1 e^{-τ z1} ∫dΩ1 σ̂(θ1) σ̂(χ2) e^{-τ(1-z1)/cosψ} J(z1,θ1,ψ)

    with ``a = albedo``, σ̂ the normalized differential cross-section, χ1 = θ1
    (incident along +z), cos χ2 = cosθ1 cosψ + sinθ1 sinψ cosφ1, and
    ``J = ∫_0^{s_max} e^{-τ(1 - cosθ1/cosψ) s} ds`` the analytic second-leg
    free-path integral (``s_max`` = distance from the first-scatter point to the
    slab boundary along Ω1). ``geometric_series=True`` adds the higher orders via
    ``I_mult = I1·ρ/(1-ρ)`` with ``ρ = I2/I1`` (Soper-style), to match the
    all-orders Monte-Carlo β; otherwise β reflects exactly-double scattering.

    Returns dict with ``Q``, ``psi``, ``I_single``, ``I_double``, ``beta``
    (selected estimator) and ``beta_double`` (exactly-double).
    """
    lam = wavelength_A
    a = torch.as_tensor(albedo, dtype=torch.float64)
    tau_t = torch.as_tensor(tau, dtype=torch.float64)
    eps = 1e-6

    sig_tot = total_cross_section(composition, wavelength_A=lam)

    def sigma_hat(Q):
        return differential_cross_section(Q, composition, wavelength_A=lam) / sig_tot

    # detector (forward transmission) angles, over the physical Q range only
    # (psi stays well below 90 deg for high-energy PDF, so cos psi is bounded
    # away from 0 -- no grazing-exit singularity).
    sin_half_max = min(q_max * lam / _FOUR_PI, 0.95)
    psi_max = 2.0 * float(np.arcsin(sin_half_max))
    psi = torch.linspace(eps, psi_max, n_psi, dtype=torch.float64)
    cos_psi = torch.cos(psi)
    Q_det = (_FOUR_PI / lam) * torch.sin(0.5 * psi)

    A1 = slab_single_scattering_factor(Q_det, thickness_um=1.0, mu_um=float(tau),
                                       wavelength_A=lam)
    I1 = a * tau_t * sigma_hat(Q_det) * A1                       # (n_psi,)

    # integration grids
    theta1 = torch.linspace(eps, float(np.pi) - eps, n_theta, dtype=torch.float64)
    cos_t1 = torch.cos(theta1)
    sin_t1 = torch.sin(theta1)
    phi1 = torch.linspace(0.0, 2 * float(np.pi), n_phi, dtype=torch.float64)
    z1 = torch.linspace(0.0, 1.0, n_z, dtype=torch.float64)

    Q1 = (_FOUR_PI / lam) * torch.sin(0.5 * theta1)
    sh1 = sigma_hat(Q1)                                         # (n_theta,)

    # χ2(ψ, θ1, φ1)
    cos_chi2 = (cos_t1[None, :, None] * cos_psi[:, None, None]
                + sin_t1[None, :, None] * torch.sin(psi)[:, None, None]
                * torch.cos(phi1)[None, None, :])
    cos_chi2 = cos_chi2.clamp(-1.0, 1.0)
    chi2 = torch.arccos(cos_chi2)
    Q2 = (_FOUR_PI / lam) * torch.sin(0.5 * chi2)
    sh2 = sigma_hat(Q2)                                        # (n_psi,n_theta,n_phi)
    sh2_phi = torch.trapz(sh2, phi1, dim=2)                    # ∫dφ1 σ̂(χ2): (n_psi,n_theta)

    # second-leg free-path integral J(ψ, θ1, z1)
    # s_max(z1, θ1): distance to slab boundary along Ω1
    cos_t1_safe = torch.where(cos_t1.abs() < 1e-4,
                              torch.full_like(cos_t1, 1e-4) * torch.sign(cos_t1 + 1e-12),
                              cos_t1)
    smax = torch.where(
        cos_t1[None, :] > 0,
        (1.0 - z1[:, None]) / cos_t1_safe[None, :],
        z1[:, None] / (-cos_t1_safe[None, :]),
    )                                                          # (n_z, n_theta)
    smax = smax.clamp(min=0.0, max=1.0e3)
    rate = tau_t * (1.0 - cos_t1[None, :] / cos_psi[:, None])  # a-rate (n_psi,n_theta)
    rate_b = rate[:, :, None]                                  # (n_psi,n_theta,1)
    smax_b = smax.transpose(0, 1)[None, :, :]                  # (1,n_theta,n_z)
    arg = rate_b * smax_b                                       # (n_psi,n_theta,n_z)
    J = torch.where(rate_b.abs() < 1e-8,
                    smax_b.expand_as(arg),
                    (1.0 - torch.exp(-arg)) / torch.where(rate_b.abs() < 1e-30,
                                                          torch.ones_like(rate_b), rate_b))
    # depth weighting exp(-τ z1) exp(-τ(1-z1)/cosψ)
    w_in = torch.exp(-tau_t * z1)                              # (n_z,)
    w_out = torch.exp(-tau_t * (1.0 - z1)[None, :] / cos_psi[:, None])  # (n_psi,n_z)
    depth = w_in[None, None, :] * w_out[:, None, :]            # (n_psi,1,n_z)
    Zint = torch.trapz(depth * J, z1, dim=2)                  # (n_psi,n_theta)

    integrand_theta = (sin_t1[None, :] * sh1[None, :]) * Zint * sh2_phi
    I2 = (a * a * tau_t * tau_t) * torch.trapz(integrand_theta, theta1, dim=1)  # (n_psi,)

    beta_double = I2 / (I1 + I2).clamp(min=1e-30)
    if geometric_series:
        rho = (I2 / I1.clamp(min=1e-30)).clamp(max=0.95)
        I_mult = I1 * rho / (1.0 - rho)
        beta = I_mult / (I1 + I_mult).clamp(min=1e-30)
    else:
        beta = beta_double
    return {
        "Q": Q_det, "psi": psi, "I_single": I1, "I_double": I2,
        "beta": beta, "beta_double": beta_double,
    }


# Calibrated slab-equivalent factor for a cylinder (beam through the diameter):
# tau_eff = CYLINDER_SLAB_FACTOR * mu * R reproduces the cylinder Monte-Carlo
# multiple-scattering fraction via the (geometry-insensitive) angular transport
# solver to ~1-3% rms over tau_radius in [0.3, 1.4]. The literal Cauchy mean
# chord of an infinite cylinder is 2R; the smaller MS-equivalent reflects that
# the beam enters along the diameter rather than isotropically. The exact factor
# drifts mildly with optical depth (1.3 -> 1.6); 1.45 is the best constant.
CYLINDER_SLAB_FACTOR = 1.45


def cylinder_effective_tau(mu_um: float, radius_um: float) -> float:
    """Slab-equivalent optical depth for a cylinder, ``CYLINDER_SLAB_FACTOR*mu*R``.

    Feed this as ``tau`` to :func:`midas_pdf.ms_transport.slab_transport_ms` to
    get a *differentiable* all-orders multiple-scattering estimate for a
    capillary, calibrated and validated against
    :func:`multiple_scattering_mc_cylinder`.
    """
    return CYLINDER_SLAB_FACTOR * mu_um * radius_um


def multiple_scattering_mc_cylinder(
    composition: Composition,
    *,
    wavelength_A: float,
    tau_radius: float,
    albedo: float,
    n_photons: int = 200_000,
    n_psi: int = 90,
    max_events: int = 60,
    seed: int = 0,
) -> dict:
    """Analog Monte-Carlo multiple scattering for an infinite cylinder (capillary).

    The beam enters along the cylinder diameter; the boundary is the circle
    ``x^2 + z^2 = R^2`` (axis along y, infinite). ``tau_radius = mu * R`` is the
    optical depth over one radius. Forward-escaping photons are tallied by exit
    angle (-> Q) and scatter order. Returns the same dict shape as
    :func:`multiple_scattering_mc`. Non-differentiable (reference for the
    transport-with-effective-tau approximation).
    """
    rng = np.random.default_rng(seed)
    psi_grid, cdf = _build_angle_sampler(composition, wavelength_A)
    R = 1.0
    mu = float(tau_radius)                                   # since R=1, tau_radius = mu*R

    # start on the entry face (x=0, z=-R), moving +z, spread along y is irrelevant
    pos = np.zeros((n_photons, 3))
    pos[:, 2] = -R
    d = np.tile(np.array([0.0, 0.0, 1.0]), (n_photons, 1))
    order = np.zeros(n_photons, dtype=np.int32)
    alive = np.ones(n_photons, dtype=bool)
    exit_cos, exit_order = [], []

    for _ in range(max_events):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break
        dd = d[idx]
        x, z = pos[idx, 0], pos[idx, 2]
        dx, dz = dd[:, 0], dd[:, 2]
        # distance to boundary x^2+z^2=R^2 along the ray (3-D ray, circular bound)
        A2 = dx * dx + dz * dz
        B = 2.0 * (x * dx + z * dz)
        C = x * x + z * z - R * R
        disc = np.clip(B * B - 4 * A2 * C, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_b = np.where(A2 > 1e-12, (-B + np.sqrt(disc)) / (2 * A2), np.inf)
        s_b = np.where(s_b > 0, s_b, np.inf)
        ell = rng.exponential(1.0, idx.size)
        s_free = ell / mu                                    # geometric free path
        escapes = s_free >= s_b
        esc = idx[escapes]
        if esc.size:
            we = d[esc, 2]
            keep = (we > 0) & (order[esc] >= 1)
            exit_cos.append(we[keep]); exit_order.append(order[esc][keep])
            alive[esc] = False
        intr = idx[~escapes]
        if intr.size:
            step = s_free[~escapes][:, None]
            pos[intr] = pos[intr] + step * d[intr]
            u = rng.uniform(0.0, 1.0, intr.size)
            scatter = u < albedo
            alive[intr[~scatter]] = False
            sc = intr[scatter]
            if sc.size:
                rr = rng.uniform(0.0, 1.0, sc.size)
                j = np.clip(np.searchsorted(cdf, rr), 0, psi_grid.size - 1)
                d[sc] = _scatter_directions(d[sc], np.cos(psi_grid[j]), rng)
                order[sc] += 1

    exit_cos = np.concatenate(exit_cos) if exit_cos else np.array([])
    exit_order = np.concatenate(exit_order) if exit_order else np.array([])
    psi = np.arccos(np.clip(exit_cos, -1.0, 1.0))
    edges = np.linspace(0.0, np.pi / 2, n_psi + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    single = np.histogram(psi[exit_order == 1], bins=edges)[0].astype(float)
    multi = np.histogram(psi[exit_order >= 2], bins=edges)[0].astype(float)
    total = single + multi
    beta = np.divide(multi, total, out=np.zeros_like(total), where=total > 0)
    Q = (_FOUR_PI / wavelength_A) * np.sin(0.5 * centers)
    return {
        "Q": torch.as_tensor(Q, dtype=torch.float64),
        "psi": torch.as_tensor(centers, dtype=torch.float64),
        "I_single": torch.as_tensor(single, dtype=torch.float64),
        "I_multiple": torch.as_tensor(multi, dtype=torch.float64),
        "beta": torch.as_tensor(beta, dtype=torch.float64),
        "n_single": float(single.sum()),
        "n_multiple": float(multi.sum()),
    }


def ms_background_on_grid(
    q_data: torch.Tensor | np.ndarray,
    intensity: torch.Tensor | np.ndarray,
    mc_result: dict,
) -> torch.Tensor:
    """Multiple-scattering background on the data grid, ready for ``background=``.

    Interpolates the MC fraction β(Q) onto ``q_data`` and returns
    ``b(Q) = β(Q) · I_meas(Q)`` — the estimated multiply-scattered part of the
    measured intensity (since ``I_multiple ≈ β·I_total ≈ β·I_meas``). Feed it as
    the ``background`` argument of :func:`midas_pdf.faber_ziman_S` /
    ``i_of_q_to_Gr``. This is the first-principles MS counterpart of the Tier-1
    lumped polynomial background (it isolates MS rather than lumping it with
    fluorescence/air).
    """
    q = torch.as_tensor(q_data, dtype=torch.float64)
    I = torch.as_tensor(intensity, dtype=torch.float64)
    qm = mc_result["Q"].detach().cpu().numpy()
    bm = mc_result["beta"].detach().cpu().numpy()
    order = np.argsort(qm)
    beta = np.interp(q.detach().cpu().numpy(), qm[order], bm[order],
                     left=float(bm[order][0]), right=float(bm[order][-1]))
    return torch.as_tensor(beta, dtype=torch.float64) * I
