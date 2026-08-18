"""TT and DCT scan conventions: rotation-axis-parallel-to-G alignment, DCT omega.

Phase 0 of ``implementation_plan.md``.

Frames follow :mod:`midas_dfxm.conventions` exactly (lab ``x`` along the beam,
``z`` up, ``v_lab = R @ v_sample``, angles in degrees) and all rotation matrices
are built by :func:`midas_dfxm.conventions.rotation_matrix`, which delegates to
:func:`midas_stress.orientation.axis_angle_to_orient_mat`. No rotation math is
re-derived here.

The two scan conventions
------------------------
**TT (topotomography).** One grain, one reflection. The goniometer is set so the
grain's ``G`` lies along the tomographic rotation axis ``a_hat``. Rotation about
an axis parallel to ``G`` leaves ``G`` invariant, so the Bragg condition holds
through the *entire* 360 deg sweep and ``k_h = k_in + G`` is fixed in the lab.
The grain volume sweeps through a stationary diffracted beam, giving hundreds of
topographs of one grain. :func:`tt_alignment` solves the alignment;
:meth:`TTAlignment.sample_rotation` generates the scan.

**DCT.** Box beam, sample rotated about the vertical lab axis, spots flashing
into the Bragg condition at crystallographically-fixed omega. The rotation is a
plain rotation about lab ``z`` -- with a sign that is a *hardware* property, not
a mathematical one (see :func:`dct_sample_rotation`).

Why alignment is a one-parameter family
---------------------------------------
Any additional rotation about ``a_hat`` also satisfies the alignment, because it
leaves ``G`` fixed. That degeneracy is not a defect of the solve -- it *is* the
TT scan variable ``psi``. :func:`tt_alignment` returns the minimal-angle
representative and ``psi`` parameterises the family from there.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from midas_dfxm.conventions import rotation_matrix

from .geometry import (
    LAB_Z,
    angle_kh_to_axis_deg,
    bragg_angle_deg,
    bragg_condition_residual,
    diffracted_beam_direction,
    diffracted_wavevector,
    incident_wavevector,
    rotation_axis_for_reflection,
)

__all__ = [
    "DCT_OMEGA_SIGN_AERO",
    "DCT_OMEGA_SIGN_CCW",
    "TTAlignment",
    "align_vector_to",
    "dct_sample_rotation",
    "tt_alignment",
]

# --- DCT omega sign ---------------------------------------------------------
# The 1-ID aero rotation stage turns CLOCKWISE when viewed along +z, i.e. the
# recorded omega runs opposite to the right-handed mathematical rotation about
# lab z. Every omega read from a 1-ID dataset must be negated before it enters a
# rotation matrix. A sign error here does not crash and does not degrade any fit
# -- it MIRRORS the reconstruction, undetectably, unless a known-chirality
# feature is present. Hence: explicit named constants, never a bare literal.
DCT_OMEGA_SIGN_AERO = -1.0   # 1-ID aero stage (clockwise) -- the default
DCT_OMEGA_SIGN_CCW = +1.0    # a stage that is genuinely right-handed about +z

_EPS = 1e-12


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)


def align_vector_to(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Minimal-angle rotation ``R`` with ``R @ source_hat == target_hat``.

    Both arguments are ``(3,)``; only their directions matter. Returns ``(3, 3)``.

    Degenerate cases are handled explicitly rather than left to produce a NaN:

    * already parallel -> identity;
    * antiparallel -> 180 deg about an arbitrary axis perpendicular to
      ``source``. The choice is arbitrary *and harmless here*, because any two
      such rotations differ by a rotation about ``target`` -- which is exactly
      the ``psi`` degeneracy the TT scan sweeps anyway.

    Differentiable in both arguments away from those two measure-zero
    configurations.
    """
    s = _unit(source)
    t = _unit(target.to(device=source.device, dtype=source.dtype))
    c = torch.sum(s * t, dim=-1).clamp(-1.0, 1.0)
    axis = torch.linalg.cross(s, t)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1)

    if bool(axis_norm.detach() < _EPS):
        if bool(c.detach() > 0):
            return torch.eye(3, device=s.device, dtype=s.dtype)
        # Antiparallel: build any unit vector perpendicular to s.
        probe = torch.as_tensor((0.0, 0.0, 1.0), device=s.device, dtype=s.dtype)
        if bool(torch.abs(torch.sum(s * probe)).detach() > 0.9):
            probe = torch.as_tensor((0.0, 1.0, 0.0), device=s.device, dtype=s.dtype)
        perp = _unit(torch.linalg.cross(s, probe))
        return rotation_matrix(perp, torch.as_tensor(180.0, device=s.device, dtype=s.dtype))

    angle_deg = torch.rad2deg(torch.acos(c))
    return rotation_matrix(axis / axis_norm, angle_deg)


@dataclass
class TTAlignment:
    """A solved topotomography alignment for one grain and one reflection.

    Attributes
    ----------
    rotation_axis : (3,) tensor
        ``a_hat``, the tomographic rotation axis in the lab frame. Parallel to
        the aligned ``G``.
    sample_to_lab : (3, 3) tensor
        ``R_align``: the sample->lab rotation at ``psi = 0``. The full setting at
        scan angle ``psi`` is :meth:`sample_rotation`.
    G_lab : (3,) tensor
        The aligned reciprocal vector in the lab frame (``= |G| a_hat``).
    k_in : (3,) tensor
        Incident wavevector, ``(2*pi/lambda) xhat``.
    theta_deg : scalar tensor
        Bragg angle of this reflection.

    All tensors share device/dtype. Differentiable in the sample-frame ``G`` and
    the wavelength that produced them.
    """

    rotation_axis: torch.Tensor
    sample_to_lab: torch.Tensor
    G_lab: torch.Tensor
    k_in: torch.Tensor
    theta_deg: torch.Tensor

    # -- the scan -----------------------------------------------------------
    def psi_rotation(self, psi_deg) -> torch.Tensor:
        """Rotation by ``psi`` (degrees) about the tomographic axis, ``(3, 3)``."""
        if not isinstance(psi_deg, torch.Tensor):
            psi_deg = torch.as_tensor(
                float(psi_deg), device=self.rotation_axis.device, dtype=self.rotation_axis.dtype
            )
        return rotation_matrix(self.rotation_axis, psi_deg)

    def sample_rotation(self, psi_deg) -> torch.Tensor:
        """Full sample->lab rotation at scan angle ``psi``: ``R_psi @ R_align``.

        ``G_lab`` is invariant under this whole family -- that invariance is the
        defining property of TT and is asserted in ``tests/test_conventions.py``.
        """
        return self.psi_rotation(psi_deg) @ self.sample_to_lab

    # -- derived geometry ---------------------------------------------------
    def diffracted_wavevector(self) -> torch.Tensor:
        """``k_h = k_in + G_lab``. Fixed in the lab for the whole scan."""
        return diffracted_wavevector(self.k_in, self.G_lab)

    def beam_direction(self) -> torch.Tensor:
        """Unit ``k_h``: the projection direction of every topograph in the scan."""
        return diffracted_beam_direction(self.k_in, self.G_lab)

    def bragg_residual(self) -> torch.Tensor:
        """``2 k_in . G + |G|^2``; zero when the alignment is exact."""
        return bragg_condition_residual(self.k_in, self.G_lab)

    def axis_beam_angle_deg(self) -> torch.Tensor:
        """Angle between ``k_h`` and the rotation axis, computed from vectors.

        Equals ``90 - theta`` exactly. Kept as an independent computation so the
        identity can be tested rather than assumed.
        """
        return angle_kh_to_axis_deg(self.k_in, self.G_lab)

    def missing_cone_deg(self) -> torch.Tensor:
        """Half-angle of the laminographic missing cone (``= theta``)."""
        return self.theta_deg

    def to(self, *, device=None, dtype=None) -> "TTAlignment":
        """Return a copy on the given device/dtype (all tensors moved together)."""
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return TTAlignment(
            rotation_axis=self.rotation_axis.to(**kw),
            sample_to_lab=self.sample_to_lab.to(**kw),
            G_lab=self.G_lab.to(**kw),
            k_in=self.k_in.to(**kw),
            theta_deg=self.theta_deg.to(**kw),
        )


def tt_alignment(
    g_sample: torch.Tensor,
    wavelength_A,
    *,
    azimuth_deg=90.0,
) -> TTAlignment:
    """Solve the TT goniometer alignment for one sample-frame reflection.

    Parameters
    ----------
    g_sample : (3,) tensor
        The grain's reciprocal vector in the **sample frame**, ``2*pi/d``
        convention -- e.g. ``midas_dfxm.field.DeformationField.reference_G(hkl)``.
    wavelength_A : float or tensor
        Incident wavelength in Angstrom.
    azimuth_deg : float, default 90
        Azimuth of the diffraction plane about the beam. 90 deg puts ``k_h`` in
        the lab ``x``-``z`` plane, deflected upward (the DFXM convention).

    Returns
    -------
    TTAlignment

    Raises
    ------
    ValueError
        If the reflection is inaccessible at this wavelength (``d < lambda/2``).

    Warning
    -------
    Align on the **actual** lattice, not the reference one. Passing an
    undeformed ``G0`` for a crystal that is rotated away from it leaves the scan
    off the Bragg condition by that rotation — measured at 0.5 deg of lattice
    rotation, the Bragg residual swings +/-5-9 sigma_perp across the psi sweep,
    and any quantity fitted from that scan acquires a spurious dependence on the
    grain's orientation (measured spread 3.7x, which collapses to 1.0x once the
    alignment uses the true lattice). If you have an orientation estimate, use it.

    Notes
    -----
    The solve is: (1) ``theta`` from ``|G|`` and ``|k|``; (2) the lab direction
    ``a_hat`` the axis must take, from :func:`rotation_axis_for_reflection`;
    (3) the minimal-angle rotation carrying ``G_hat`` onto ``a_hat``. Step 3 is
    where the one-parameter ``psi`` freedom is fixed to a representative.

    The returned alignment satisfies the Bragg condition *by construction*, not
    by refinement -- ``bragg_residual()`` is zero to floating-point, which makes
    it a real test rather than a tautology only because the residual is computed
    from ``G_lab`` through an independent path.
    """
    g = g_sample if isinstance(g_sample, torch.Tensor) else torch.as_tensor(g_sample)
    if not torch.is_floating_point(g):
        g = g.to(torch.float64)

    k_in = incident_wavevector(wavelength_A, device=g.device, dtype=g.dtype)
    g_mag = torch.linalg.vector_norm(g, dim=-1)
    k_mag = torch.linalg.vector_norm(k_in, dim=-1)
    theta_deg = bragg_angle_deg(g_mag, k_mag)

    axis = rotation_axis_for_reflection(
        theta_deg, azimuth_deg=azimuth_deg, device=g.device, dtype=g.dtype
    )
    R = align_vector_to(g, axis)
    return TTAlignment(
        rotation_axis=axis,
        sample_to_lab=R,
        G_lab=R @ g,
        k_in=k_in,
        theta_deg=theta_deg,
    )


def dct_sample_rotation(
    omega_deg,
    *,
    omega_sign: float = DCT_OMEGA_SIGN_AERO,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """DCT sample->lab rotation: ``omega_sign * omega`` about the vertical lab axis.

    DCT rotates the sample 360 deg about lab ``z`` while a box beam illuminates
    it; grains flash into the Bragg condition at crystallographically-fixed
    ``omega``.

    ``omega_sign`` defaults to :data:`DCT_OMEGA_SIGN_AERO` (``-1``) because the
    1-ID aero stage turns clockwise about ``+z``, so recorded ``omega`` values
    must be negated before use. **Pass** :data:`DCT_OMEGA_SIGN_CCW` **explicitly**
    for a right-handed stage. Getting this wrong mirrors the reconstruction and
    nothing in the fit residuals will tell you: a mirrored grain map fits the
    mirrored spot set exactly as well.
    """
    if not isinstance(omega_deg, torch.Tensor):
        omega_deg = torch.as_tensor(float(omega_deg), device=device, dtype=dtype)
    else:
        omega_deg = omega_deg.to(
            device=device or omega_deg.device, dtype=dtype or omega_deg.dtype
        )
    return rotation_matrix(LAB_Z, omega_sign * omega_deg)
