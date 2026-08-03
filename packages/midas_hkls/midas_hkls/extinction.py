"""Primary extinction, refraction, and the kinematical-validity boundary.

Scalar corrections that bound where kinematical diffraction stops being valid.
Shared here because none of it is technique-specific: DFXM, DCT, topotomography,
pink-beam and any other near-perfect-crystal work face the same question, and it
is X-ray physics of ``|F|``, the unit cell, the wavelength and ``theta`` -- which
is what this package is for.

**This is not dynamical diffraction theory.** It is a set of scalar corrections
*to* a kinematical calculation. It does not model Pendelloesung fringes, Borrmann
anomalous transmission, the Laue/Bragg asymmetry, or the redistribution of
intensity within a reflection. Takagi-Taupin is the honest treatment for a
near-perfect crystal and is deliberately out of scope. Treat ``t/Lambda >~ 0.5``
as "kinematical results are quantitatively wrong here", not as "apply the factor
and proceed".

What the three pieces are for
-----------------------------
* :func:`extinction_length_um` -- ``Lambda``, the length scale. A path much
  shorter than it is kinematical; much longer is dynamical.
* :func:`primary_extinction_factor` -- ``y = tanh(x)/x``, ``x = t/Lambda``
  (Darwin/Zachariasen, Laue case). Multiplies a kinematical intensity.
* :func:`refraction_shift_deg` -- the constant angular offset between the
  dynamical and kinematical Bragg conditions.

How these actually bite a field measurement (measured, midas-dfxm)
-------------------------------------------------------------------
A DFXM study decomposed the effect on a recovered deformation field, and the
result generalises to any centroid-based technique:

1. **Extinction is centroid-preserving.** It reshapes a rocking curve
   symmetrically about its centre, so even a spatially *varying* extinction
   strength leaves the recovered ``dQ`` -- and hence ``F`` -- unbiased. An
   attenuation is not a bias.
2. **Refraction is a gauge.** ``Delta_theta = |chi_0| / sin(2 theta)`` is constant
   for a fixed reflection and energy, so it shifts every voxel alike and is
   absorbed into the reference lattice. The *relative* field is unbiased.
3. **Only the spatially varying residual biases the field**, and it does so
   roughly 0.43:1 (a varying centroid shift of X is recovered as ~0.43 X of
   spurious strain). Bounding *that* needs Takagi-Taupin; bounding its
   consequence does not.

So for centroid/position observables the first two terms are benign, which is
worth knowing before anyone spends effort correcting them.

Units: micrometers for lengths, Angstrom for wavelength and cell dimensions,
degrees for angles. Torch is optional -- pass tensors and the torch path is used
(differentiable); pass floats and it stays in plain Python.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "CLASSICAL_ELECTRON_RADIUS_A",
    "extinction_length_um",
    "kinematical_path_limit_um",
    "primary_extinction_factor",
    "refraction_shift_deg",
    "susceptibility_chi0",
]

#: Classical electron radius in Angstrom (2.8179403262 fm).
CLASSICAL_ELECTRON_RADIUS_A = 2.8179403262e-5

_A_PER_UM = 1.0e4


def _is_torch(x) -> bool:
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def extinction_length_um(
    structure_factor_mag,
    *,
    unit_cell_volume_A3,
    wavelength_A,
    theta_deg,
    polarization_factor: float = 1.0,
):
    """Extinction (Pendelloesung) length ``Lambda``, in **micrometers**.

    ``Lambda = pi V cos(theta) / (r_e lambda |F| C)``

    ``|F|`` is the structure-factor modulus in electrons per unit cell -- the same
    quantity :func:`midas_hkls.structure_factors` returns (take ``abs``). ``V`` is
    the cell volume in Angstrom^3 (``Lattice.volume``), ``lambda`` in Angstrom.
    ``C`` is the polarization factor: 1 for sigma, ``cos(2 theta)`` for pi.

    Differentiable in every argument passed as a torch tensor.

    Raises
    ------
    ValueError
        If ``|F| = 0``. A forbidden reflection does not diffract, so it has no
        extinction length; returning ``inf`` would let a meaningless number flow
        downstream.
    """
    use_torch = any(_is_torch(v) for v in
                    (structure_factor_mag, unit_cell_volume_A3, wavelength_A,
                     theta_deg, polarization_factor))
    if use_torch:
        import torch
        F = structure_factor_mag if _is_torch(structure_factor_mag) else \
            torch.as_tensor(structure_factor_mag, dtype=torch.float64)
        if not torch.is_floating_point(F):
            F = F.to(torch.float64)
        kw = dict(dtype=F.dtype, device=F.device)
        V = torch.as_tensor(unit_cell_volume_A3, **kw)
        lam = torch.as_tensor(wavelength_A, **kw)
        th = torch.deg2rad(torch.as_tensor(theta_deg, **kw))
        C = torch.as_tensor(polarization_factor, **kw)
        denom = CLASSICAL_ELECTRON_RADIUS_A * lam * torch.abs(F) * C
        if bool((denom.detach() <= 0).any()):
            raise ValueError(
                "extinction length is undefined for |F| = 0: a forbidden "
                "reflection does not diffract, so it has no extinction length"
            )
        return (math.pi * V * torch.cos(th) / denom) / _A_PER_UM

    denom = (CLASSICAL_ELECTRON_RADIUS_A * float(wavelength_A)
             * abs(float(structure_factor_mag)) * float(polarization_factor))
    if denom <= 0.0:
        raise ValueError(
            "extinction length is undefined for |F| = 0: a forbidden reflection "
            "does not diffract, so it has no extinction length"
        )
    return (math.pi * float(unit_cell_volume_A3)
            * math.cos(math.radians(float(theta_deg))) / denom) / _A_PER_UM


def primary_extinction_factor(path_um, extinction_length_um):
    """Primary extinction factor ``y = tanh(x)/x`` with ``x = path / Lambda``.

    Multiplies a kinematical intensity. ``y -> 1`` as ``x -> 0`` (kinematical) and
    ``y -> 1/x`` for a thick perfect crystal (dynamical saturation: it cannot
    reflect more than the Darwin width allows).

    The ``x -> 0`` limit is evaluated by series, not as ``0/0``, so the value is
    exactly 1 and the gradient is finite at the origin.
    """
    if _is_torch(path_um) or _is_torch(extinction_length_um):
        import torch
        t = path_um if _is_torch(path_um) else torch.as_tensor(path_um, dtype=torch.float64)
        if not torch.is_floating_point(t):
            t = t.to(torch.float64)
        L = (extinction_length_um if _is_torch(extinction_length_um)
             else torch.as_tensor(extinction_length_um, dtype=t.dtype, device=t.device))
        L = L.to(dtype=t.dtype, device=t.device)
        x = t / L
        small = torch.abs(x) < 1e-6
        safe = torch.where(small, torch.ones_like(x), x)
        return torch.where(small, 1.0 - x ** 2 / 3.0, torch.tanh(safe) / safe)

    x = float(path_um) / float(extinction_length_um)
    if abs(x) < 1e-6:
        return 1.0 - x * x / 3.0
    return math.tanh(x) / x


def kinematical_path_limit_um(extinction_length_um, *, tolerance: float = 0.1):
    """Longest path for which extinction costs less than ``tolerance`` of intensity.

    Solves ``1 - y(x) = tolerance`` **exactly**, by bisection on the strictly
    decreasing ``y``, and returns ``x * Lambda``. For a 10% tolerance
    ``x = 0.5838``, i.e. a path of a bit over half an extinction length.

    The small-``x`` expansion ``y ~ 1 - x^2/3`` (giving ``x = sqrt(3 tol)``) is
    tempting but is 6% optimistic at ``tol = 0.1`` and 20% at ``tol = 0.2`` -- a
    validity boundary that is itself 20% wrong is not a validity boundary.
    """
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be in (0, 1)")
    target = 1.0 - tolerance
    lo, hi = 1e-12, 1.0e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.tanh(mid) / mid > target:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    if _is_torch(extinction_length_um):
        return extinction_length_um * x
    return float(extinction_length_um) * x


def susceptibility_chi0(crystal_t, wavelength_A, *, anomalous: bool = True):
    """Zeroth Fourier coefficient of the electric susceptibility, ``chi_0``.

    ``chi_0 = - r_e lambda^2 F(000) / (pi V)``, complex when ``anomalous=True``
    (``f'`` shifts the real part, ``f''`` the imaginary). Sets the refractive
    decrement and hence :func:`refraction_shift_deg`.

    ``crystal_t`` is a :class:`~midas_hkls.crystal_torch.CrystalTensor`
    (``Crystal.to_torch()``). Requires torch, since it goes through
    :func:`midas_hkls.structure_factors`.
    """
    import torch
    from .lattice_torch import cell_volume
    from .structure_factor import structure_factors

    F000 = structure_factors(crystal_t, [(0, 0, 0)], wavelength_A=wavelength_A,
                             anomalous=anomalous)[0]
    V = cell_volume(crystal_t.lattice_params)
    lam = (wavelength_A if _is_torch(wavelength_A)
           else torch.as_tensor(float(wavelength_A), dtype=V.dtype, device=V.device))
    return -CLASSICAL_ELECTRON_RADIUS_A * lam ** 2 * F000 / (math.pi * V)


def refraction_shift_deg(chi0, two_theta_deg):
    """Angular offset between the dynamical and kinematical Bragg conditions.

    ``Delta_theta = |chi_0| / sin(2 theta)``, in **degrees**. Typically 0.4-3.6
    mdeg for real DFXM/TT cases.

    It is **constant** for a fixed reflection and energy, so across a field it is
    a gauge: it shifts every voxel by the same amount and is absorbed into the
    reference lattice. The relative deformation field is unbiased by it. Only a
    *spatially varying* dynamical shift produces spurious strain (see the module
    docstring).
    """
    if _is_torch(chi0) or _is_torch(two_theta_deg):
        import torch
        c = chi0 if _is_torch(chi0) else torch.as_tensor(chi0)
        tt = (two_theta_deg if _is_torch(two_theta_deg)
              else torch.as_tensor(float(two_theta_deg), dtype=torch.float64))
        mag = torch.abs(c)
        return torch.rad2deg(mag / torch.sin(torch.deg2rad(tt)))
    return math.degrees(abs(chi0) / math.sin(math.radians(float(two_theta_deg))))
