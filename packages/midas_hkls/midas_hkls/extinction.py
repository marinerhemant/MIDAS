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
as "kinematical results are quantitatively wrong here" (for a perfect slab the 10 % loss is already at
``t/Lambda = 0.1785``), not as "apply the factor and proceed".

What the three pieces are for
-----------------------------
* :func:`extinction_length_um` -- ``Lambda``, the length scale. A path much
  shorter than it is kinematical; much longer is dynamical.
* :func:`laue_extinction_factor` -- the EXACT angle-integrated factor of a perfect
  crystal slab, symmetric Laue, ``(1/2A) int_0^{2A} J0``, ``A = pi t / Lambda``;
  :func:`laue_kinematical_path_limit_um` is its validity bound (10 % at
  ``0.1785 Lambda``). Multiplies a kinematical intensity.
* :func:`primary_extinction_factor` -- ``y = tanh(x)/x`` with ``x = t/Lambda``.
  **Not** the perfect-crystal Laue factor: at the same ``t/Lambda`` it loses ``pi^2``
  less intensity for small ``t/Lambda`` (see its docstring). Kept for models that
  supply their own ``x``.
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
    "laue_extinction_factor",
    "laue_kinematical_path_limit_um",
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

    .. warning::
       With ``Lambda`` from :func:`extinction_length_um` (the Pendelloesung period) this is **not** the angle-integrated
       reflectivity of a perfect crystal slab in the Laue case. That is :func:`laue_extinction_factor`, with
       ``A = pi path / Lambda``. At the same ``path / Lambda`` the tanh form loses ``pi^2`` less intensity for small
       ``path / Lambda`` (0.84 % against 8.0 % at 0.159, 3.2 % against 28.7 % at 0.318; PREREGISTER A21 and
       ``midas_sx/validation/extinction_convention_check.py``). Use this function only for a model whose own ``x`` you
       supply (a Darwin mosaic block, where ``x`` is quadratic in ``|F|``), never as the perfect-crystal Laue factor.

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
    """Longest path for which the **tanh** model (:func:`primary_extinction_factor`) costs less than ``tolerance``.

    For a perfect crystal slab in the Laue case use :func:`laue_kinematical_path_limit_um`: its 10 % bound is
    ``0.1785 Lambda``, a factor 3.3 below the 0.5838 Lambda given here.

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


_J0_FN = None


def _torch_j0():
    """``torch.special.bessel_j0`` is neither differentiable nor accurate (3.8e-7 near x = 5), so J0 goes through scipy with the
    analytic derivative -J1 (differentiable once, which is all the fits need)."""
    global _J0_FN
    if _J0_FN is not None:
        return _J0_FN
    import torch
    from scipy.special import j0, j1

    class _J0(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return torch.as_tensor(j0(x.detach().cpu().numpy()), dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return g * torch.as_tensor(-j1(x.detach().cpu().numpy()), dtype=x.dtype, device=x.device)

    _J0_FN = _J0.apply
    return _J0_FN


def _gl_nodes(n_per_panel: int = 16):
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(n_per_panel)
    return 0.5 * (x + 1.0), 0.5 * w                      # nodes / weights on [0, 1]


def laue_extinction_factor(path_um, extinction_length_um):
    """Exact angle-integrated primary-extinction factor of a perfect crystal slab, symmetric Laue, no absorption.

    ``R(A) = (1/2A) int_0^{2A} J0(u) du = int_0^1 J0(2 A s) ds`` with ``A = pi * path / Lambda`` and ``Lambda`` the
    Pendelloesung length of :func:`extinction_length_um`. It equals the integral over the deviation parameter ``y`` of the
    Takagi-Taupin intensity ``sin^2(A sqrt(1+y^2)) / (1+y^2)`` divided by its kinematic limit ``pi A`` (checked against that
    integral and against ``midas_dfxm.takagi_taupin`` in the tests and in ``midas_sx/validation``). Multiplies a kinematical
    integrated intensity. ``R -> 1 - A^2/3`` for small ``A`` and ``-> 1/(2A)`` (oscillating) for large ``A``.

    NOT the same function as :func:`primary_extinction_factor`: that one has ``tanh(x)/x`` with ``x = path/Lambda``, whose
    small-x loss is ``pi^2`` smaller than this one at the same ``path / Lambda``. Use this when ``Lambda`` is the Pendelloesung
    length and the crystal is a perfect block of thickness ``path``.

    Differentiable in both arguments for torch tensors (fixed Gauss-Legendre panels, 16 nodes per half period of the
    integrand). ``path_um = 0`` gives exactly 1 with a finite gradient. J0 comes from scipy in both paths
    (torch's own ``bessel_j0`` is not differentiable and is off by 4e-7 near x = 5), with -J1 as the backward.
    """
    import numpy as np
    n_per = 16
    s0, w0 = _gl_nodes(n_per)
    if _is_torch(path_um) or _is_torch(extinction_length_um):
        import torch
        t = path_um if _is_torch(path_um) else torch.as_tensor(path_um, dtype=torch.float64)
        if not torch.is_floating_point(t):
            t = t.to(torch.float64)
        L = (extinction_length_um if _is_torch(extinction_length_um)
             else torch.as_tensor(extinction_length_um, dtype=t.dtype, device=t.device))
        L = L.to(dtype=t.dtype, device=t.device)
        A = torch.broadcast_tensors(math.pi * t / L, t)[0]
        n_pan = max(1, int(math.ceil(float(torch.max(torch.abs(2.0 * A)).detach()) / math.pi))) if A.numel() else 1
        s = torch.as_tensor(np.concatenate([(k + s0) / n_pan for k in range(n_pan)]), dtype=A.dtype, device=A.device)
        w = torch.as_tensor(np.tile(w0 / n_pan, n_pan), dtype=A.dtype, device=A.device)
        flat = A.reshape(-1)
        chunk = max(1, (1 << 22) // len(s))                  # bound the (elements x nodes) intermediate to ~4M doubles
        out = [(_torch_j0()(2.0 * flat[i:i + chunk].unsqueeze(-1) * s) * w).sum(-1) for i in range(0, flat.numel(), chunk)]
        return (torch.cat(out) if out else flat.new_empty(0)).reshape(A.shape)
    from scipy.special import j0
    A = math.pi * np.asarray(path_um, float) / np.asarray(extinction_length_um, float)
    n_pan = max(1, int(math.ceil(float(np.max(np.abs(2.0 * A))) / math.pi)))
    s = np.concatenate([(k + s0) / n_pan for k in range(n_pan)])
    w = np.tile(w0 / n_pan, n_pan)
    out = (j0(2.0 * A[..., None] * s) * w).sum(-1)
    return float(out) if np.ndim(out) == 0 else out


def laue_kinematical_path_limit_um(extinction_length_um, *, tolerance: float = 0.1):
    """Longest path for which the exact Laue extinction (:func:`laue_extinction_factor`) costs less than ``tolerance``.

    Solves ``1 - R(pi t / Lambda) = tolerance`` on the first, monotonic branch of R by bisection. For 10 % this is
    ``t = 0.1785 Lambda`` (the tanh bound of :func:`kinematical_path_limit_um` is 0.5838 Lambda). ``R`` first reaches its
    smallest values only for ``A > 1``, so the first crossing is unique; tolerances above ~0.5 are refused.
    """
    if not 0.0 < tolerance <= 0.5:
        raise ValueError("tolerance must be in (0, 0.5]")
    lo, hi = 0.0, 1.0                                    # A in [0, 1]: R(1) = 0.7129 > 0.5, monotonic
    if 1.0 - float(laue_extinction_factor(hi / math.pi, 1.0)) < tolerance:
        hi = 2.0                                         # R(2) = 0.2562, still first crossing (R decreasing to 2.4)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - float(laue_extinction_factor(mid / math.pi, 1.0)) < tolerance:
            lo = mid
        else:
            hi = mid
    A = 0.5 * (lo + hi)
    t_over_L = A / math.pi
    if _is_torch(extinction_length_um):
        return extinction_length_um * t_over_L
    return float(extinction_length_um) * t_over_L


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
