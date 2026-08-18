"""The direct-beam channel, over the shared extinction physics in ``midas_hkls``.

Phase 1 of ``implementation_plan.md``; the mitigation for risk Section 9.1.

**The extinction physics now lives in :mod:`midas_hkls.extinction`** and is
re-exported here for convenience. None of it is technique-specific -- it is
X-ray physics of ``|F|``, the unit cell, the wavelength and ``theta`` -- so DFXM,
pink-beam and anything else touching near-perfect crystals share the same
implementation rather than three drifting copies:

* :func:`~midas_hkls.extinction.extinction_length_um` -- ``Lambda``
* :func:`~midas_hkls.extinction.primary_extinction_factor` -- ``y = tanh(x)/x``
* :func:`~midas_hkls.extinction.kinematical_path_limit_um` -- the validity bound
* :func:`~midas_hkls.extinction.refraction_shift_deg`,
  :func:`~midas_hkls.extinction.susceptibility_chi0` -- the constant dynamical
  offset, contributed from the DFXM analysis

What stays here is the one genuinely DCT-specific piece: the **direct-beam
extinction channel**, i.e. the depletion a grain causes in the transmitted beam
at exactly the omega where its spot flashes.

Kinematical validity, in one number
-----------------------------------
Extinction contrast is *the* classical contrast mechanism in X-ray topography,
and a near-perfect grain -- exactly what topotomography is used on -- diffracts
dynamically. Rather than leave that as an unstated assumption, ask for a number:
``t / Lambda``. Treat ``>~ 0.5`` as "kinematical results are quantitatively wrong
here". Takagi-Taupin is out of scope and is not promised.

Note for this package specifically: extinction is **centroid-preserving** (it
reshapes a rocking curve symmetrically), so it does not bias a position-channel
measurement -- which is what the Phase-3 deformation fit uses. Refraction is a
constant gauge. Only a *spatially varying* dynamical response produces spurious
strain. See the ``midas_hkls.extinction`` module docstring for the measured
decomposition.
"""
from __future__ import annotations

import torch
from midas_hkls.extinction import (
    CLASSICAL_ELECTRON_RADIUS_A,
    extinction_length_um,
    kinematical_path_limit_um,
    primary_extinction_factor,
    refraction_shift_deg,
)

import math

from .geometry import _float_tensor

__all__ = [
    "CLASSICAL_ELECTRON_RADIUS_A",
    "kinematical_validity",
    "coherent_path_um",
    "direct_beam_transmission",
    "extinction_weights",
    "extinction_length_um",
    "kinematical_path_limit_um",
    "primary_extinction_factor",
    "refraction_shift_deg",
]


def direct_beam_transmission(path_um, *, mu_per_um, diffracted_fraction=0.0):
    """Transmitted intensity fraction of the direct beam.

    ``exp(-mu * t) * (1 - diffracted_fraction)``: photo-absorption along the path
    through the sample, times the depletion caused by a grain diffracting power
    *out* of the transmitted beam. That second term is DCT's **extinction
    channel** -- the reason a grain appears as a dark patch in the direct beam at
    exactly the omega where its spot flashes, and an independent handle on the
    same grain's shape.

    ``diffracted_fraction`` is the integrated reflectivity along that ray
    (dimensionless, in ``[0, 1]``); it is a caller input rather than something
    computed here because it depends on the reflection, the beam's spectral and
    angular width, and the extinction regime -- for which
    :func:`~midas_hkls.extinction.primary_extinction_factor` is the knob.
    """
    t = _float_tensor(path_um)
    mu = _float_tensor(mu_per_um).to(dtype=t.dtype, device=t.device)
    dif = torch.as_tensor(diffracted_fraction, dtype=t.dtype, device=t.device)
    if bool((dif.detach() < 0).any()) or bool((dif.detach() > 1).any()):
        raise ValueError("diffracted_fraction must lie in [0, 1]")
    return torch.exp(-mu * t) * (1.0 - dif)


def coherent_path_um(Q_sample, shape, spacing_um, *, extinction_length_um,
                     max_path_um):
    """Distortion-limited coherent path per voxel, in micrometers.

    This is the quantity that turns extinction from a constant attenuation into a
    **contrast mechanism**, and the sign of the effect is the opposite of the
    naive guess:

    * a **perfect** region keeps the lattice within the Darwin width over a long
      path, so the coherent path is the full grain thickness, extinction is
      strong, and the region is **dim**;
    * a **distorted** region (a dislocation, a subgrain wall) throws the lattice
      out of the Darwin width after a short distance, so the coherent path is
      short, extinction is weak, and the region is **bright**.

    That inversion is why defects appear *bright* in an X-ray topograph -- the
    classical "direct image" -- and it is exactly the contrast a purely
    kinematical forward model cannot produce, because kinematically a distorted
    voxel simply falls out of the acceptance and gets *darker*.

    The criterion is coherence over the Darwin width. A perfect crystal reflects
    over a reciprocal-space range ``dq ~ 2*pi/Lambda``, so coherence survives while
    the local scattering vector stays inside that range::

        t_eff = min(max_path, dq_darwin / |grad Q|)

    with ``dq_darwin = 2*pi / (Lambda[um] * 1e4)`` in 1/Angstrom.

    ``|grad Q|`` is the full gradient magnitude on the voxel grid, not the
    directional derivative along each ray. That is the **worst-case** (shortest)
    coherent path and therefore the *upper* bound on extinction contrast; the
    ray-resolved form needs the beam direction per ``psi`` and is not implemented.
    Stated rather than hidden, because it means this over-estimates defect
    brightness for distortions perpendicular to the beam.

    This is primary extinction only. **It is not a dynamical calculation** --
    Takagi-Taupin is out of scope and is not promised. Use
    :func:`~midas_hkls.extinction.kinematical_path_limit_um` to check whether the
    kinematical baseline was even valid before reading contrast off this.
    """
    Q = Q_sample
    if Q.ndim != 2 or Q.shape[-1] != 3:
        raise ValueError(f"Q_sample must be (N, 3), got {tuple(Q.shape)}")
    if shape is None:
        raise ValueError(
            "coherent_path_um needs a regular grid to take a gradient; this grain "
            "has shape=None"
        )
    nx, ny, nz = shape
    if nx * ny * nz != Q.shape[0]:
        raise ValueError(f"grid {shape} does not match {Q.shape[0]} voxels")
    sp = spacing_um if isinstance(spacing_um, (tuple, list)) else (spacing_um,) * 3

    g = Q.reshape(nx, ny, nz, 3)
    grad2 = torch.zeros(nx, ny, nz, dtype=Q.dtype, device=Q.device)
    for axis, n, h in ((0, nx, sp[0]), (1, ny, sp[1]), (2, nz, sp[2])):
        if n < 2:
            continue
        d = torch.gradient(g, spacing=float(h), dim=axis)[0]
        grad2 = grad2 + (d ** 2).sum(dim=-1)
    grad = torch.sqrt(grad2).reshape(-1)

    lam = torch.as_tensor(extinction_length_um, dtype=Q.dtype, device=Q.device)
    dq_darwin = 2.0 * math.pi / (lam * 1.0e4)          # 1/Angstrom
    mx = torch.as_tensor(max_path_um, dtype=Q.dtype, device=Q.device)
    # where the lattice is perfect the gradient vanishes -> full path
    t = torch.where(grad > 0, dq_darwin / grad.clamp_min(1e-300),
                    torch.full_like(grad, float("inf")))
    return torch.minimum(t, mx.expand_as(t))


def extinction_weights(Q_sample, shape, spacing_um, *, extinction_length_um,
                       max_path_um):
    """Per-voxel primary-extinction factor ``y`` in ``(0, 1]``, ready to multiply
    a kinematical intensity.

    Composition of :func:`coherent_path_um` with
    :func:`~midas_hkls.extinction.primary_extinction_factor`. Pass the result as
    ``extinction=`` to :func:`~midas_dct_tt.forward.voxel_scattering` or
    :func:`~midas_dct_tt.forward.topograph_stack`.

    ``y -> 1`` (no extinction, full kinematical intensity) where the lattice is
    distorted; ``y -> Lambda/t`` where it is perfect and thick.
    """
    t = coherent_path_um(Q_sample, shape, spacing_um,
                         extinction_length_um=extinction_length_um,
                         max_path_um=max_path_um)
    return primary_extinction_factor(t, extinction_length_um)


def kinematical_validity(hkl, *, wavelength_A, thickness_um, crystal=None,
                         lattice_a_A: float = 3.6356):
    """Is a **kinematical** TT forward valid for this grain? Returns a report dict.

    Keys: ``extinction_length_um`` (``Lambda``), ``ratio`` (``t/Lambda``),
    ``intensity_dynamical``, ``intensity_kinematical``, ``relative_error``,
    ``regime`` and ``theta_deg``.

    Why this matters more for TT than for anything else in the package
    -----------------------------------------------------------------
    Extinction is *the* classical contrast mechanism in X-ray topography, and
    topotomography is used precisely on near-perfect, low-defect grains where
    dynamical diffraction is strongest. A kinematical forward mispredicts contrast
    exactly where the technique is most useful.

    The dynamical calculation is delegated to
    :mod:`midas_dfxm.takagi_taupin` -- a differentiable two-beam Takagi-Taupin
    solver validated to machine precision against the closed-form symmetric-Laue
    solution. Symmetric Laue *transmission* is the classic dynamical-topography
    geometry and is TT's geometry, so the port is a parameter choice, not a new
    derivation. Susceptibilities come from ``midas_hkls`` structure factors.

    Measured for **Cu** (a = 3.6356 A, the default reference crystal) at 71.7 keV
    ----------------------------------------------------------------------------
    ``Lambda`` = **34.96 um** (111), **37.22 um** (200), **45.84 um** (220),
    **51.97 um** (311). The material matters: across fcc metals the extinction
    length spans roughly a factor of seven (Al is several times longer, Au several
    times shorter), because it scales as ``1/|F|``. **Name the element whenever
    quoting these numbers**, and pass a matching ``crystal`` -- ``lattice_a_A``
    alone changes only the geometry. Sweeping thickness on (200), against the kinematical
    ``(pi t / Lambda)^2``:

    ======  ==========  ===========  ============  ===================
    t (um)  t/Lambda    I dynamical  I kinematical  verdict
    ======  ==========  ===========  ============  ===================
    1       0.027       0.00711      0.00713       kinematical, 0.3%
    3       0.081       0.06277      0.06413       kinematical, 2.1%
    10      0.269       0.55859      0.71255       marginal, **28%**
    30      0.806       0.32745      6.41          **20x wrong**
    100     2.687       0.69300      71.3          **103x wrong**
    ======  ==========  ===========  ============  ===================

    **Consequence, and it is not small: a kinematical TT forward is quantitatively
    wrong for ordinary HEDM grain sizes.** It holds to a few percent only below
    ~3-5 um. Beyond ``t/Lambda ~ 0.3`` it fails qualitatively, not just
    numerically -- kinematical intensity grows as ``t^2`` without bound while the
    true intensity saturates and then oscillates (Pendellosung).

    Everything else in this package is kinematical, and its synthetic grains are
    2-3 um (``t/Lambda`` ~ 0.03-0.08), i.e. **inside** the valid domain. Real TT
    grains at 50-200 um are **not**.
    """
    import math as _math

    from midas_dfxm.takagi_taupin import (bragg_angle_deg, extinction_length,
                                          laue_intensity_analytic,
                                          susceptibility_fourier)
    if crystal is None:
        from midas_dfxm.io import fcc_reference_crystal
        crystal = fcc_reference_crystal()
        if abs(lattice_a_A - 3.6356) > 1e-6:
            raise ValueError(
                f"lattice_a_A={lattice_a_A} was given without a matching crystal. "
                "The default reference crystal is Cu (a = 3.6356 A); changing only "
                "the lattice parameter alters the Bragg angle but NOT the structure "
                "factor, so the extinction length would be Cu's with another metal's "
                "geometry -- silently wrong by up to ~3x across fcc metals. Pass a "
                "crystal for the element you mean."
            )

    _, chih, chihbar = susceptibility_fourier(crystal, tuple(hkl),
                                              wavelength_A=wavelength_A,
                                              absorption=True)
    d = lattice_a_A / _math.sqrt(sum(int(i) ** 2 for i in hkl))
    theta = bragg_angle_deg(d, wavelength_A)
    lam_um = float(extinction_length(chih, chihbar, wavelength_A=wavelength_A,
                                     theta_B_deg=theta))
    ratio = float(thickness_um) / lam_um
    i_dyn = float(laue_intensity_analytic(float(thickness_um), 0.0, lam_um))
    i_kin = (_math.pi * float(thickness_um) / lam_um) ** 2
    rel = abs(i_kin - i_dyn) / max(i_dyn, 1e-30)
    regime = ("kinematical" if ratio < 0.1 else
              "marginal" if ratio < 0.3 else "dynamical")
    return {"extinction_length_um": lam_um, "ratio": ratio, "theta_deg": theta,
            "intensity_dynamical": i_dyn, "intensity_kinematical": i_kin,
            "relative_error": rel, "regime": regime}
