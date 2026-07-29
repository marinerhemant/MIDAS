"""Q-dependent 1-D intensity corrections applied before normalization.

These act on the integrated I(Q) (the layer PDF reducers like PDFgetX3 work in),
and are all backed by the NIST mass-attenuation tables already shipped in
``midas_hkls`` (no new data) and are differentiable in Q / geometry.

* **Detector efficiency** -- a flat sensor of finite thickness absorbs (detects)
  a Q-dependent fraction of photons: more at high scattering angle, where the
  path through the sensor is longer. High-energy total scattering reaches large
  angles, so this is a real high-Q tilt that must be divided out.
* **Flat-plate sample self-absorption** -- analytic transmission factor for a
  slab. (Capillary/cylindrical geometry is already handled upstream by
  ``midas_integrate_v2.corrections.CylindricalAbsorption``.)

The scattering angle ``psi`` (= 2*theta_Bragg) follows from Q via
``cos psi = 1 - 2 (Q lambda / 4 pi)^2``.
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple, Union

import numpy as np
import torch

from midas_hkls.absorption import element_density, mass_attenuation_coefficient

__all__ = [
    "linear_attenuation_um",
    "cos_scattering_angle",
    "detector_efficiency",
    "apply_detector_efficiency",
    "flat_plate_transmission",
    "paalman_pings_cylinder_in_cylinder",
    "paalman_pings_cell_only",
]

_FOUR_PI = 4.0 * float(np.pi)


def cos_scattering_angle(q: torch.Tensor, wavelength_A: float | torch.Tensor) -> torch.Tensor:
    """cos(psi) for scattering angle psi (= 2*theta), from Q and wavelength."""
    q_t = torch.as_tensor(q, dtype=torch.float64)
    lam = torch.as_tensor(wavelength_A, dtype=torch.float64)
    return 1.0 - 2.0 * (q_t * lam / _FOUR_PI) ** 2


def linear_attenuation_um(
    material: Union[str, Mapping[str, float]],
    wavelength_A: float,
    *,
    density_g_cm3: Optional[float] = None,
) -> float:
    """Linear attenuation coefficient mu (units 1/micron) for an element or
    a mass-fraction compound ``{element: mass_fraction}``.

    Uses the NIST mass-attenuation coefficient(s) from midas-hkls
    (mu/rho, cm^2/g) times the density. For a single element the tabulated
    elemental density is used when ``density_g_cm3`` is None.
    """
    if isinstance(material, str):
        mu_over_rho = float(mass_attenuation_coefficient(material, wavelength_A))
        rho = element_density(material) if density_g_cm3 is None else density_g_cm3
    else:
        total = float(sum(material.values()))
        mu_over_rho = sum(
            (w / total) * float(mass_attenuation_coefficient(el, wavelength_A))
            for el, w in material.items()
        )
        if density_g_cm3 is None:
            raise ValueError("density_g_cm3 is required for a compound material")
        rho = density_g_cm3
    mu_per_cm = mu_over_rho * rho            # 1/cm
    return mu_per_cm / 1.0e4                 # 1/micron


def detector_efficiency(
    q: torch.Tensor | np.ndarray,
    *,
    wavelength_A: float,
    material: Union[str, Mapping[str, float]],
    thickness_um: float,
    density_g_cm3: Optional[float] = None,
) -> torch.Tensor:
    """Fraction of photons absorbed (detected) by a flat sensor, as a function
    of Q: ``eta(Q) = 1 - exp(-mu * t / cos psi)``, in [0, 1].

    Increases toward high Q. Divide the measured I(Q) by this to remove the
    high-Q efficiency tilt (see :func:`apply_detector_efficiency`).
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    mu = linear_attenuation_um(material, wavelength_A, density_g_cm3=density_g_cm3)
    cos_psi = cos_scattering_angle(q_t, wavelength_A).clamp(min=1e-3)
    path = thickness_um / cos_psi
    return 1.0 - torch.exp(-mu * path)


def apply_detector_efficiency(
    intensity: torch.Tensor,
    q: torch.Tensor,
    *,
    wavelength_A: float,
    material: Union[str, Mapping[str, float]],
    thickness_um: float,
    density_g_cm3: Optional[float] = None,
    sigma: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Divide I(Q) (and sigma) by the detector efficiency. Returns (I', sigma')."""
    eta = detector_efficiency(
        q, wavelength_A=wavelength_A, material=material,
        thickness_um=thickness_um, density_g_cm3=density_g_cm3,
    ).clamp(min=1e-6)
    I = torch.as_tensor(intensity, dtype=torch.float64) / eta
    sig = None if sigma is None else torch.as_tensor(sigma, dtype=torch.float64) / eta
    return I, sig


def flat_plate_transmission(
    q: torch.Tensor | np.ndarray,
    *,
    wavelength_A: float,
    mu_um: float,
    thickness_um: float,
) -> torch.Tensor:
    """Self-absorption transmission factor A(Q) for a flat plate in symmetric
    transmission (beam normal to the slab):

        A(Q) = exp(-mu t) * [1 - exp(-mu t (1/cos psi - 1))] / (mu t (1/cos psi - 1))

    -> exp(-mu t) at psi=0. Divide I(Q) by A(Q) to correct. ``mu_um`` is the
    linear attenuation (1/micron), e.g. from :func:`linear_attenuation_um`.
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    cos_psi = cos_scattering_angle(q_t, wavelength_A).clamp(min=1e-3)
    sec_minus_1 = (1.0 / cos_psi) - 1.0
    mt = mu_um * thickness_um
    x = mt * sec_minus_1
    # stable (1 - exp(-x))/x with the x->0 limit = 1
    series = torch.where(x.abs() < 1e-8, torch.ones_like(x), (1.0 - torch.exp(-x)) / x.clamp(min=1e-30))
    return torch.exp(torch.as_tensor(-mt, dtype=torch.float64)) * series


# ---------------------------------------------------------------------------
# Paalman-Pings absorption for cylinder-in-cylinder (capillary PDF geometry)
# ---------------------------------------------------------------------------

def _cylinder_paths_2d(x_p, y_p, cx, sy, R_s, R_c):
    """Path lengths for a source at (x_p, y_p) in the 2-D cross-section of a
    cylinder-in-cylinder (sample inside a hollow container).

    Beam is along +x. Scattered ray direction (cx, sy) is a unit vector.
    Returns (L_c_in, L_s_in, L_s_out, L_c_out): path segments through
    container and sample on the way IN and OUT.

    All inputs may be arrays; shapes broadcast.
    """
    # ---- IN paths (beam from -inf along +x to source) ---------------------
    # container entry at x = -sqrt(R_c² - y²)
    disc_c_in = np.maximum(R_c ** 2 - y_p ** 2, 0.0)
    x_c_entry = -np.sqrt(disc_c_in)
    # sample entry at x = -sqrt(R_s² - y²) if the source is above/below the sample
    inside_sample_col = np.abs(y_p) < R_s
    disc_s_in = np.maximum(R_s ** 2 - y_p ** 2, 0.0)
    x_s_entry = np.where(inside_sample_col, -np.sqrt(disc_s_in), x_p)
    # container inbound segment: from x_c_entry to x_s_entry (if beam hits sample)
    # or to x_p (if the source is in the annulus above/below sample)
    L_c_in = np.where(inside_sample_col,
                      x_s_entry - x_c_entry,
                      x_p - x_c_entry)
    # if source is in the sample column, sample segment is from x_s_entry to x_p
    L_s_in = np.where(inside_sample_col,
                      np.maximum(x_p - x_s_entry, 0.0),
                      0.0)
    # If source is IN the container annulus, but at |y|<R_s (i.e. in a sample column),
    # then the beam might traverse the sample first then reach source in the far
    # container wall.  Handle that: source at x_p > x_s_exit (=+sqrt(R_s²-y²)) means
    # source is in FAR container wall — must add sample segment IN.
    in_sample_region = (x_p ** 2 + y_p ** 2) < R_s ** 2 - 1e-12
    x_s_exit = np.where(inside_sample_col, np.sqrt(disc_s_in), x_p)
    source_beyond_sample = inside_sample_col & (~in_sample_region) & (x_p > x_s_exit - 1e-12)
    # for such sources: L_s_in = full sample thickness at that y; container in
    # = entry_wall + far-wall-up-to-source
    L_s_in = np.where(source_beyond_sample,
                      2.0 * np.sqrt(disc_s_in),
                      L_s_in)
    L_c_in = np.where(source_beyond_sample,
                      (x_s_entry - x_c_entry) + (x_p - x_s_exit),
                      L_c_in)

    # ---- OUT paths (ray from source along (cx, sy) exiting outside container)
    # (x_p + t cx)² + (y_p + t sy)² = R²  →  t² + 2 b t + (r² - R²) = 0
    b = x_p * cx + y_p * sy
    r2 = x_p ** 2 + y_p ** 2
    disc_s_out = b ** 2 - (r2 - R_s ** 2)
    disc_c_out = b ** 2 - (r2 - R_c ** 2)
    # For sources inside sample, t_s_out > 0
    t_s_out = np.where((disc_s_out > 0) & in_sample_region,
                       -b + np.sqrt(np.maximum(disc_s_out, 0.0)),
                       0.0)
    # Container exit: source inside container → positive root
    t_c_out = -b + np.sqrt(np.maximum(disc_c_out, 0.0))
    # If ray from container source RE-ENTERS the sample on its way out,
    # count the sample chord traversed.  This happens if the ray line intersects
    # r=R_s at two positive t values.
    has_two_sample_crossings = (~in_sample_region) & (disc_s_out > 0)
    if np.any(has_two_sample_crossings):
        t_s_near = -b - np.sqrt(np.maximum(disc_s_out, 0.0))
        t_s_far = -b + np.sqrt(np.maximum(disc_s_out, 0.0))
        # Only count segment where 0 <= t_s_near < t_s_far <= t_c_out
        seg = np.where(has_two_sample_crossings & (t_s_near > 0) & (t_s_far < t_c_out),
                       t_s_far - t_s_near, 0.0)
        L_s_out = t_s_out + seg
        L_c_out = t_c_out - t_s_out - seg
    else:
        L_s_out = t_s_out
        L_c_out = t_c_out - t_s_out

    # Clip negative to zero (edge-case numerical noise)
    return (np.maximum(L_c_in, 0.0),
            np.maximum(L_s_in, 0.0),
            np.maximum(L_s_out, 0.0),
            np.maximum(L_c_out, 0.0))


def paalman_pings_cylinder_in_cylinder(
    q: torch.Tensor | np.ndarray,
    *,
    wavelength_A: float,
    mu_sample_um: float,
    mu_container_um: float,
    R_sample_um: float,
    R_container_um: float,
    n_grid: int = 64,
) -> dict:
    """Paalman-Pings absorption factors for a cylindrical sample inside a
    hollow cylindrical container (capillary PDF geometry).

    Returns four Q-dependent factors as torch tensors:

        A_s_sc : scattering from sample,     attenuated by sample + container
        A_c_sc : scattering from container,  attenuated by sample + container
        A_c_c  : scattering from container,  attenuated by container ONLY
                 (i.e. the empty-cell measurement; sample not present)
        A_s_s  : scattering from sample,     attenuated by sample ONLY
                 (i.e. sample without container; hypothetical reference)

    The classic Paalman-Pings empty-cell subtraction is::

        I_sample_theory(Q) = [I_meas(Q) - (A_c_sc/A_c_c) · I_empty(Q)] / A_s_sc

    All μ values in 1/μm; radii in μm.  Numerical integration by a 2-D
    polar grid over the source volume (thin-lamella approximation along the
    cylinder axis, exact for cylindrical symmetry).

    ``n_grid`` controls both the radial and azimuthal discretisation
    (n_grid × n_grid); 64 is a good balance of accuracy vs speed
    (~50 ms for a full 3000-point Q grid).
    """
    q_np = np.asarray(q if not hasattr(q, "cpu") else q.cpu().numpy(), dtype=np.float64)
    tt = 2.0 * np.arcsin(np.clip(q_np * float(wavelength_A) / _FOUR_PI, -1.0, 1.0))
    cx = np.cos(tt); sy = np.sin(tt)                           # (nQ,)

    # --- sample source grid (polar, cell-centred) ----------------------
    r_s_edges = np.linspace(0.0, R_sample_um, n_grid + 1)
    r_s = 0.5 * (r_s_edges[:-1] + r_s_edges[1:])              # (n,)
    dr_s = np.diff(r_s_edges)                                  # (n,)
    phi = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dphi = 2.0 * np.pi / n_grid
    Rr_s, Pp = np.meshgrid(r_s, phi, indexing="ij")
    x_s = Rr_s * np.cos(Pp);  y_s = Rr_s * np.sin(Pp)
    dV_s = (Rr_s * dr_s[:, None] * dphi)                       # r·dr·dφ, shape (n, n)
    V_s = np.pi * R_sample_um ** 2

    # --- container source grid ----------------------------------------
    r_c_edges = np.linspace(R_sample_um, R_container_um, n_grid + 1)
    r_c = 0.5 * (r_c_edges[:-1] + r_c_edges[1:])
    dr_c = np.diff(r_c_edges)
    Rr_c, Pc = np.meshgrid(r_c, phi, indexing="ij")
    x_c = Rr_c * np.cos(Pc);  y_c = Rr_c * np.sin(Pc)
    dV_c = (Rr_c * dr_c[:, None] * dphi)
    V_c = np.pi * (R_container_um ** 2 - R_sample_um ** 2)

    A_s_sc = np.zeros(q_np.shape, dtype=np.float64)
    A_c_sc = np.zeros(q_np.shape, dtype=np.float64)
    A_c_c  = np.zeros(q_np.shape, dtype=np.float64)
    A_s_s  = np.zeros(q_np.shape, dtype=np.float64)

    for i in range(q_np.size):
        cx_i, sy_i = cx.flat[i], sy.flat[i]

        # -- sample source, attenuated by sample + container (measured with cell)
        L_c_in, L_s_in, L_s_out, L_c_out = _cylinder_paths_2d(
            x_s, y_s, cx_i, sy_i, R_sample_um, R_container_um)
        att = np.exp(-mu_sample_um * (L_s_in + L_s_out)
                     -mu_container_um * (L_c_in + L_c_out))
        A_s_sc.flat[i] = (att * dV_s).sum() / V_s

        # -- sample source, attenuated by sample ONLY (no container present)
        L_c_in0, L_s_in0, L_s_out0, L_c_out0 = _cylinder_paths_2d(
            x_s, y_s, cx_i, sy_i, R_sample_um, R_sample_um)  # R_c = R_s ⇒ no wall
        att_s = np.exp(-mu_sample_um * (L_s_in0 + L_s_out0))
        A_s_s.flat[i] = (att_s * dV_s).sum() / V_s

        # -- container source, attenuated by sample + container (with sample)
        Lc_in, Ls_in, Ls_out, Lc_out = _cylinder_paths_2d(
            x_c, y_c, cx_i, sy_i, R_sample_um, R_container_um)
        att_csc = np.exp(-mu_sample_um * (Ls_in + Ls_out)
                         -mu_container_um * (Lc_in + Lc_out))
        A_c_sc.flat[i] = (att_csc * dV_c).sum() / V_c

        # -- container source, attenuated by container ONLY (empty-cell measurement)
        # replace the "sample" region with vacuum ⇒ mu_sample = 0
        att_cc = np.exp(-mu_container_um * (Lc_in + Lc_out))
        A_c_c.flat[i] = (att_cc * dV_c).sum() / V_c

    return {
        "A_s_sc": torch.as_tensor(A_s_sc, dtype=torch.float64),
        "A_c_sc": torch.as_tensor(A_c_sc, dtype=torch.float64),
        "A_c_c":  torch.as_tensor(A_c_c,  dtype=torch.float64),
        "A_s_s":  torch.as_tensor(A_s_s,  dtype=torch.float64),
        "two_theta_rad": torch.as_tensor(tt, dtype=torch.float64),
    }


def paalman_pings_cell_only(
    q: torch.Tensor | np.ndarray,
    *,
    wavelength_A: float,
    mu_container_um: float,
    R_inner_um: float,
    R_outer_um: float,
    n_grid: int = 64,
) -> torch.Tensor:
    """Q-dependent absorption for empty cell alone (no sample inside).

    Returns ``A(Q)`` = mean attenuation over the container annulus.
    Useful as a stand-alone diagnostic; also returned as ``A_c_c`` by
    :func:`paalman_pings_cylinder_in_cylinder`.
    """
    q_np = np.asarray(q if not hasattr(q, "cpu") else q.cpu().numpy(),
                       dtype=np.float64)
    tt = 2.0 * np.arcsin(np.clip(q_np * float(wavelength_A) / _FOUR_PI, -1.0, 1.0))
    cx = np.cos(tt); sy = np.sin(tt)

    r_edges = np.linspace(R_inner_um, R_outer_um, n_grid + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    phi = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    dphi = 2.0 * np.pi / n_grid
    Rr, Pp = np.meshgrid(r, phi, indexing="ij")
    x = Rr * np.cos(Pp); y = Rr * np.sin(Pp)
    dV = Rr * dr[:, None] * dphi
    V = np.pi * (R_outer_um ** 2 - R_inner_um ** 2)

    out = np.zeros(q_np.shape, dtype=np.float64)
    for i in range(q_np.size):
        L_c_in, _, _, L_c_out = _cylinder_paths_2d(
            x, y, cx.flat[i], sy.flat[i], R_inner_um, R_outer_um)
        att = np.exp(-mu_container_um * (L_c_in + L_c_out))
        out.flat[i] = (att * dV).sum() / V
    return torch.as_tensor(out, dtype=torch.float64)
