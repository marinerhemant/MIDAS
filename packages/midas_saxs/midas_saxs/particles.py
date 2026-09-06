"""Density-contrast scatterers: voids, gas bubbles, precipitates.

The other half of an irradiated material's small-angle signal, and in practice
the loud half. A void of radius R displaces ``4 pi R^3 / 3`` of electron
density; a dislocation loop of the same radius displaces only its relaxation
volume ``pi R^2 b``. For R = 5 nm and b = 2.556 A that is a factor 27 in volume
and therefore **~700 in forward intensity**. Simulating dislocations alone and
comparing to a measured irradiated-material SAXS pattern will not work; this
module is why the comparison can be made honestly.

Built on the form factors that came into this package from ``midas_pdf.saxs``
(:mod:`midas_saxs.form_factors`, :mod:`midas_saxs.model`), so there is one
definition of "sphere form factor" in MIDAS and the polydispersity quadrature is
shared with the joint SAXS+PDF refinement.

Isotropy is the point
---------------------
These scatterers are isotropic: ``I`` depends on ``|q|`` only. Dislocation loops
are not -- they vary between ``kappa dV`` in-plane and ``dV`` along their normal.
That contrast is the loop-versus-void discriminator, and it is only visible on a
2-D detector, never in a radial average. :func:`void_intensity` deliberately
takes the full ``q`` vector and reduces it to ``|q|`` internally, so a caller can
feed both source terms the same pixel grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from .form_factors import sphere_form_factor_squared
from .model import lognormal_quadrature_nodes

__all__ = [
    "SpherePopulation",
    "void_intensity",
    "loop_equivalent_sphere_radius_A",
]


def loop_equivalent_sphere_radius_A(radius_A: float, burgers_A: float) -> float:
    """Radius of the sphere whose volume equals a loop's relaxation volume.

    ``dV = pi R^2 b``; the equivalent sphere has ``R_eq = (3 dV / 4 pi)^(1/3)``.
    A 5 nm loop in Cu comes out at 1.67 nm, i.e. it scatters at ``q -> 0`` like a
    void a third its size and about 700x weaker. Provided because that number is
    the single most useful sanity check when a simulated frame looks surprising.
    """
    dV = math.pi * radius_A ** 2 * burgers_A
    return (3.0 * dV / (4.0 * math.pi)) ** (1.0 / 3.0)


@dataclass
class SpherePopulation:
    """A polydisperse population of spherical density-contrast scatterers.

    Attributes
    ----------
    radius_A
        Median radius (angstroms).
    number_density_per_A3
        Scatterers per A^3. For reference, 1e22 m^-3 = 1e-8 um^-3 = 1e-32 A^-3.
    delta_rho_e_per_A3
        Electron-density contrast against the matrix. For a **void** this is
        ``-rho_matrix`` (all the density is missing); the sign is irrelevant to
        the intensity but is kept so amplitudes can be combined coherently
        later. For a precipitate it is ``rho_ppt - rho_matrix``.
    sigma_lognormal
        Relative standard deviation of ``ln(D)``. 0 means monodisperse.
    n_quadrature
        Nodes in the polydispersity quadrature.
    """

    radius_A: float
    number_density_per_A3: float
    delta_rho_e_per_A3: float
    sigma_lognormal: float = 0.0
    n_quadrature: int = 24
    label: str = ""

    @property
    def volume_A3(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius_A ** 3

    @property
    def volume_fraction(self) -> float:
        return self.number_density_per_A3 * self.volume_A3

    def forward_intensity(self) -> float:
        """``I(0)`` per A^3 of sample: ``n (delta_rho V)^2``, electrons² / A^3.

        The Guinier plateau. Comparing this between a void population and a loop
        population is the fastest way to see which one an experiment can detect.
        """
        return (self.number_density_per_A3
                * (self.delta_rho_e_per_A3 * self.volume_A3) ** 2)

    def intensity(self, q_inv_A: torch.Tensor) -> torch.Tensor:
        """``I(|q|)`` per A^3 of sample. Accepts ``(Q,)`` or ``(Q, 3)``."""
        q = torch.as_tensor(q_inv_A, dtype=torch.float64)
        if q.ndim >= 2 and q.shape[-1] == 3:
            q = torch.linalg.vector_norm(q, dim=-1)
        q = q.reshape(-1)

        contrast2 = (self.delta_rho_e_per_A3) ** 2
        if self.sigma_lognormal <= 0:
            P = sphere_form_factor_squared(q, self.radius_A)
            return self.number_density_per_A3 * contrast2 * P

        # Polydisperse: average |F|^2 over a lognormal in the DIAMETER, matching
        # the convention in midas_saxs.model.
        D_nodes, weights = lognormal_quadrature_nodes(
            2.0 * self.radius_A, self.sigma_lognormal, n_nodes=self.n_quadrature)
        total = torch.zeros_like(q)
        norm = 0.0
        for D, w in zip(D_nodes, weights):
            R = 0.5 * float(D)
            total = total + float(w) * sphere_form_factor_squared(q, R)
            norm += float(w)
        return self.number_density_per_A3 * contrast2 * total / max(norm, 1e-30)


def void_intensity(
    q_inv_A: torch.Tensor,
    *,
    radius_A: float,
    number_density_per_A3: float,
    matrix_electron_density_e_per_A3: float,
    sigma_lognormal: float = 0.0,
) -> torch.Tensor:
    """Convenience wrapper: a population of empty voids in a matrix.

    A void's contrast is the full matrix density with a negative sign -- there is
    simply nothing there.
    """
    pop = SpherePopulation(
        radius_A=radius_A,
        number_density_per_A3=number_density_per_A3,
        delta_rho_e_per_A3=-matrix_electron_density_e_per_A3,
        sigma_lognormal=sigma_lognormal,
        label="void",
    )
    return pop.intensity(q_inv_A)
