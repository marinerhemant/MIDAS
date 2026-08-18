"""TT/DCT reciprocal-space geometry: Bragg condition, diffracted beam, missing cone.

Phase 0 of ``implementation_plan.md``.

Frames and units are inherited unchanged from :mod:`midas_dfxm.conventions`, so
that a grain, a field, and a reflection mean the same thing in both packages:

* **Lab frame** ``(x, y, z)``: ``x`` along the incident beam, ``z`` vertical (up),
  ``y = z x x``. The incident wavevector is ``k_in = (2*pi/lambda) * xhat``.
* **Sample frame**: the grain reference frame; ``v_lab = R @ v_sample``.
* Reciprocal vectors carry the ``2*pi`` convention (``|G| = 2*pi/d``), matching
  :func:`midas_dfxm.field.reciprocal_basis`.
* Angles in **degrees**, positions in **micrometers**, wavelength/lattice in
  **Angstrom**.

Every angle returned here depends on ``|G| / (2 |k|)`` only, so the ``2*pi``
convention cancels out of the *results*. It does **not** cancel out of the
*inputs*: mixing a ``2*pi/d`` reciprocal vector with a ``1/lambda`` wavevector
inside one call silently rescales ``sin(theta)`` by ``2*pi``. Build ``k_in`` with
:func:`incident_wavevector` and ``G`` with ``midas_dfxm.field.reciprocal_basis``
and the two always agree.

The central geometric result
----------------------------
With the tomographic rotation axis ``a_hat`` parallel to ``G`` (the
topotomography alignment) and the elastic Bragg condition ``k_in . G =
-|G|^2/2``::

    cos angle(k_h, G_hat) = (k_in . G + |G|^2) / (|G| |k_in|)
                          = (|G|/2) / |k_in|
                          = sin(theta)

    =>  angle(k_h, a_hat) = 90 deg - theta        exactly, for every theta

Conventional parallel-beam tomography needs the projection direction
perpendicular to the rotation axis. TT is off by exactly ``theta``, so **TT is a
laminographic problem with a missing cone of half-angle theta**, not a standard
tomographic one. Low-``theta`` (high-energy, low-index) reflections therefore
give more complete coverage -- a reflection-selection criterion, quantified by
:func:`missing_cone_half_angle_deg`.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "LAB_X",
    "LAB_Y",
    "LAB_Z",
    "angle_between_deg",
    "angle_kh_to_axis_deg",
    "bragg_angle_deg",
    "bragg_condition_residual",
    "diffracted_beam_direction",
    "diffracted_wavevector",
    "incident_wavevector",
    "missing_cone_half_angle_deg",
    "rotation_axis_for_reflection",
    "wavevector_magnitude",
]

LAB_X = (1.0, 0.0, 0.0)
LAB_Y = (0.0, 1.0, 0.0)
LAB_Z = (0.0, 0.0, 1.0)

_TWO_PI = 2.0 * math.pi


def _as_tensor(x, *, ref: torch.Tensor) -> torch.Tensor:
    """Coerce ``x`` to a tensor on ``ref``'s device/dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def _float_tensor(x) -> torch.Tensor:
    """Coerce to a floating tensor, promoting Python scalars to **float64**.

    ``torch.as_tensor(0.17)`` gives float32, whose ~6e-8 relative error lands
    directly on ``sin(theta)`` and so on every angle downstream. Wavelengths and
    reciprocal magnitudes arrive here as bare Python floats constantly, so the
    promotion is done once, here, rather than trusted to the caller. Tensors keep
    their own dtype (that is how MPS float32 work stays float32).
    """
    if isinstance(x, torch.Tensor):
        return x if torch.is_floating_point(x) else x.to(torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


def _promote_pair(a, b) -> tuple:
    """Bring two magnitudes to a common dtype/device, Python scalars being *weak*.

    Mirrors torch's own promotion rule rather than inventing one: a Python float
    adopts the dtype of whatever tensor it is combined with, and only decides the
    dtype (float64) when there is no tensor at all. The alternative --
    ``promote_types(float32, float64) -> float64`` -- would silently push an MPS
    float32 tensor to a dtype MPS cannot hold, turning a wavelength argument into
    a device error.
    """
    tensors = [x for x in (a, b) if isinstance(x, torch.Tensor)]
    if tensors:
        dtype = tensors[0].dtype
        for t in tensors[1:]:
            dtype = torch.promote_types(dtype, t.dtype)
        if not torch.is_floating_point(torch.empty(0, dtype=dtype)):
            dtype = torch.float64
        device = tensors[0].device
    else:
        dtype, device = torch.float64, None
    return (
        torch.as_tensor(a, dtype=dtype, device=device),
        torch.as_tensor(b, dtype=dtype, device=device),
    )


def wavevector_magnitude(wavelength_A) -> torch.Tensor:
    """``|k| = 2*pi/lambda`` (1/Angstrom) for a wavelength in Angstrom.

    Differentiable in ``wavelength_A`` when it is a tensor.
    """
    return _TWO_PI / _float_tensor(wavelength_A)


def incident_wavevector(wavelength_A, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Incident wavevector ``k_in = (2*pi/lambda) xhat``, shape ``(3,)``.

    The beam direction is a *convention*, not a parameter: MIDAS/DFXM put the
    incident beam along lab ``x``. Everything downstream (alignment, projection,
    detector basis) assumes it.
    """
    ref = torch.zeros((), device=device, dtype=dtype)
    lam = _as_tensor(wavelength_A, ref=ref)
    k = _TWO_PI / lam
    xhat = torch.as_tensor(LAB_X, device=ref.device, dtype=ref.dtype)
    return k * xhat


def bragg_angle_deg(g_mag, k_mag):
    """Bragg ``theta`` (degrees) from ``sin(theta) = |G| / (2 |k|)``.

    Both magnitudes must use the *same* reciprocal convention (both with the
    ``2*pi`` factor, or both without); the ratio is convention-independent.

    Raises
    ------
    ValueError
        If ``|G| > 2|k|`` -- the reflection is inaccessible at this wavelength
        (its ``d``-spacing is below ``lambda/2``). Raised rather than clamped:
        silently clamping to 90 deg would put a physically absent reflection at
        backscatter and quietly corrupt a scan plan.
    """
    g, k = _promote_pair(g_mag, k_mag)
    s = g / (2.0 * k)
    if bool((torch.abs(s.detach()) > 1.0).any()):
        raise ValueError(
            f"|G|/(2|k|) = {float(torch.max(torch.abs(s.detach()))):.6g} > 1: reflection "
            "inaccessible at this wavelength (d < lambda/2). Check that |G| and |k| use "
            "the same 2*pi convention."
        )
    return torch.rad2deg(torch.asin(s))


def bragg_condition_residual(k_in: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Elastic Bragg residual ``2 k_in . G + |G|^2`` (units of ``|k|^2``).

    Zero exactly on the Ewald sphere: ``|k_in + G| = |k_in|``. Positive/negative
    off it. Differentiable in both arguments -- this is the quantity an alignment
    refinement drives to zero, and the one to assert on in tests instead of
    re-deriving the angle.
    """
    return 2.0 * torch.sum(k_in * G, dim=-1) + torch.sum(G * G, dim=-1)


def diffracted_wavevector(k_in: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """``k_h = k_in + G``. Exact; makes no assumption that Bragg is satisfied.

    ``|k_h| = |k_in|`` only when :func:`bragg_condition_residual` is zero.
    """
    return k_in + G


def diffracted_beam_direction(k_in: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Unit ``k_h`` for an arbitrary reciprocal vector ``G``.

    This is the TT/DCT generalisation of
    :func:`midas_dfxm.optics.diffracted_beam_direction`, which takes only
    ``2*theta`` and hardcodes the diffraction plane to lab ``x``-``z``. TT picks
    one grain's ``G`` and it points wherever crystallography puts it, so the
    2-theta-only form is not enough. For a ``G`` in the ``x``-``z`` plane the two
    agree exactly (locked by ``tests/test_dfxm_contract.py``).
    """
    kh = diffracted_wavevector(k_in, G)
    return kh / torch.linalg.vector_norm(kh, dim=-1, keepdim=True)


def angle_between_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (degrees) between two vectors, clamped against acos domain error."""
    a_hat = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True)
    b_hat = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    c = torch.sum(a_hat * b_hat, dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


def rotation_axis_for_reflection(
    theta_deg,
    *,
    azimuth_deg=90.0,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Lab direction the TT rotation axis must take for a Bragg angle ``theta``.

    In topotomography the rotation axis is parallel to ``G``, and ``G`` must sit
    on the Bragg cone about the incident beam::

        khat_in . a_hat = -sin(theta)

    which leaves one free parameter, the azimuth about the beam. Returns

        a_hat = -sin(theta) xhat + cos(theta) (cos(az) yhat + sin(az) zhat)

    ``azimuth_deg = 90`` (the default) puts the diffraction plane vertical with
    the beam deflected *upward*, i.e. ``k_h`` in the lab ``x``-``z`` plane --
    the same convention as :mod:`midas_dfxm.optics`.

    Note the sign: the axis tilts *upstream* by ``theta`` from the plane
    perpendicular to the beam. That tilt is precisely what makes TT
    laminographic (see :func:`missing_cone_half_angle_deg`).
    """
    ref = torch.zeros((), device=device, dtype=dtype)
    th = torch.deg2rad(_as_tensor(theta_deg, ref=ref))
    az = torch.deg2rad(_as_tensor(azimuth_deg, ref=ref))
    return torch.stack(
        [
            -torch.sin(th),
            torch.cos(th) * torch.cos(az),
            torch.cos(th) * torch.sin(az),
        ],
        dim=-1,
    )


def angle_kh_to_axis_deg(k_in: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Angle (degrees) between the diffracted beam and the TT rotation axis.

    Computed from the vectors, with no appeal to the ``90 - theta`` identity --
    so it is an *independent* path to that result and the right thing to test
    the identity against. Assumes the TT alignment ``a_hat = G_hat``; the caller
    is responsible for ``G`` actually satisfying Bragg (check with
    :func:`bragg_condition_residual`).
    """
    return angle_between_deg(diffracted_wavevector(k_in, G), G)


def missing_cone_half_angle_deg(theta_deg):
    """Half-angle of the TT missing cone: exactly the Bragg angle ``theta``.

    Parallel-beam tomography samples the full set of projection directions
    perpendicular to the rotation axis. TT's projection direction ``k_h`` sits at
    ``90 - theta`` from the axis instead, so rotating through 360 deg sweeps a
    *cone* of directions and leaves a symmetric cone of half-angle ``theta``
    about the axis unsampled -- the standard laminographic missing wedge/cone,
    with the axial (along-``G``) spatial frequencies the ones lost.

    Consequences worth designing around:

    * Low-``theta`` reflections (high energy, low index) reconstruct better.
      At 1-ID/HEXM energies ``theta ~ 2-8 deg``; at ESRF-typical energies it is
      several times larger.
    * Friedel-paired settings flip the cone and recover part of it.
    * A reconstructed sphere elongates along the axis by an amount set by this
      angle -- the Phase 2 falsifiable test.

    Trivial as arithmetic, deliberately named: the number that matters is the
    *consequence*, and every caller should reach it through this docstring
    rather than rediscovering it. Verified against the vector computation
    (:func:`angle_kh_to_axis_deg`) in the test suite.
    """
    return theta_deg
