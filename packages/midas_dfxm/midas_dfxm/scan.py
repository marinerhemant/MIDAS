"""Goniometer scan builders and Bragg-angle helpers.

Phase 1 of ``implementation_plan.md``.

A *mosaicity scan* rocks the two goniometer axes ``(chi, phi)`` about an aligned
setting, mapping local orientation. A *strain scan* varies the Bragg angle
``2*theta`` (equivalently incident energy) at fixed goniometer, mapping local
``d``-spacing. Weak-beam settings offset the goniometer from exact Bragg to image
individual dislocations.

Units: degrees for angles, Angstrom for wavelength; reciprocal vectors carry the
``2*pi`` convention (``|Q| = 2*pi/d``).
"""
from __future__ import annotations

import math

import torch

from .conventions import GoniometerSetting
from .field import DeformationField


def bragg_two_theta_deg(q_mag: float, wavelength_A: float) -> float:
    """Bragg ``2*theta`` (degrees) for ``|Q| = 2*pi/d`` at a given wavelength.

    ``sin(theta) = lambda |Q| / (4 pi)``.
    """
    s = wavelength_A * q_mag / (4.0 * math.pi)
    return math.degrees(2.0 * math.asin(s))


def reference_q_nom(
    field: DeformationField,
    hkl,
    goniometer_ref: GoniometerSetting,
) -> torch.Tensor:
    """Nominal accepted lab-frame vector: reference reflection at the aligned setting.

    ``q_nom = G_ref @ G0_sample``. This centres the resolution function
    (:func:`midas_dfxm.resolution.aligned_resolution`).
    """
    G0 = field.reference_G(hkl)
    G_ref = goniometer_ref.sample_rotation(device=G0.device, dtype=G0.dtype)
    return G_ref @ G0


def mosaicity_scan(
    center: GoniometerSetting,
    *,
    chi_range=(-0.2, 0.2),
    phi_range=(-0.2, 0.2),
    n_chi: int = 9,
    n_phi: int = 9,
) -> list[GoniometerSetting]:
    """Grid of goniometer settings rocking ``(chi, phi)`` about ``center`` (degrees)."""
    chis = torch.linspace(chi_range[0], chi_range[1], n_chi)
    phis = torch.linspace(phi_range[0], phi_range[1], n_phi)
    out = []
    for c in chis.tolist():
        for p in phis.tolist():
            out.append(
                GoniometerSetting(
                    mu=center.mu, omega=center.omega,
                    chi=center.chi + c, phi=center.phi + p,
                    frame=center.frame,
                )
            )
    return out


def rocking_scan(
    center: GoniometerSetting,
    *,
    axis: str = "chi",
    span=(-0.3, 0.3),
    n: int = 41,
) -> list[GoniometerSetting]:
    """1-D rocking scan about a single axis (``'chi'`` or ``'phi'``), degrees."""
    if axis not in ("chi", "phi"):
        raise ValueError("axis must be 'chi' or 'phi'")
    vals = torch.linspace(span[0], span[1], n).tolist()
    out = []
    for d in vals:
        kw = dict(mu=center.mu, omega=center.omega, chi=center.chi, phi=center.phi)
        kw[axis] = kw[axis] + d
        out.append(GoniometerSetting(**kw, frame=center.frame))
    return out


def scan_angles(settings: list[GoniometerSetting], axis: str = "chi") -> torch.Tensor:
    """Extract the swept angle (degrees) from a scan for plotting/fitting."""
    return torch.tensor([getattr(s, axis) for s in settings], dtype=torch.float64)
