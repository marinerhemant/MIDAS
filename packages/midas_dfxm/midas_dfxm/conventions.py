"""DFXM frame & goniometer conventions, and the DFXM <-> MIDAS frame map.

Phase 0 of ``implementation_plan.md``.

Frames (right-handed, following Poulsen 2017 / DTU ``darling`` conventions)
--------------------------------------------------------------------------
* **Lab frame** ``(x, y, z)``: ``x`` along the incident beam, ``z`` vertical
  (up), ``y = z x x`` completes the right-handed set. The incident wavevector is
  ``k_in = (2*pi/lambda) * xhat``.
* **Sample frame**: the crystal/grain reference frame. A goniometer rotation
  ``G`` maps sample-frame vectors into the lab frame: ``v_lab = G @ v_sample``.
* **Imaging (objective) frame**: axis along the diffracted beam ``k_out`` at
  ``2*theta`` from ``x`` in the diffraction plane; handled in ``optics.py``.

Goniometer
----------
A DFXM diffractometer stacks a base tilt and rocking motors. We parameterise the
standard set as an ordered composition (outer-most first):

    G(mu, omega, chi, phi) = R_y(mu) @ R_z(omega) @ R_x(chi) @ R_y(phi)

with all angles in **degrees**. ``mu`` is the base tilt bringing the reflection
into the vertical diffraction condition; ``omega`` is the in-plane rotation;
``chi``/``phi`` are the two rocking axes that a *mosaicity scan* sweeps. The axis
assignment is a convention knob (some beamlines swap ``chi``/``phi`` axes); it is
centralised here and unit-tested for round-trip consistency, so downstream code
never re-derives it.

All builders are torch-differentiable and device/dtype-preserving; they delegate
the axis-angle rotation to :func:`midas_stress.orientation.axis_angle_to_orient_mat`
(Rodrigues, degrees) so we never re-port rotation math.

Units: degrees for all angles (per MIDAS convention), Angstrom for wavelength.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from midas_stress.orientation import axis_angle_to_orient_mat

# Principal lab axes as a convention anchor.
_XHAT = (1.0, 0.0, 0.0)
_YHAT = (0.0, 1.0, 0.0)
_ZHAT = (0.0, 0.0, 1.0)


def _as_tensor(x, *, ref: torch.Tensor) -> torch.Tensor:
    """Coerce ``x`` to a tensor matching ``ref``'s device/dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def rotation_matrix(axis, angle_deg) -> torch.Tensor:
    """Right-handed rotation of ``angle_deg`` (degrees) about a unit ``axis``.

    Thin, differentiable wrapper over
    :func:`midas_stress.orientation.axis_angle_to_orient_mat`. ``axis`` may be a
    3-tuple or tensor; ``angle_deg`` a float or tensor. Returns a ``(3, 3)``
    (or broadcast ``(..., 3, 3)``) rotation matrix.
    """
    if not isinstance(angle_deg, torch.Tensor):
        angle_deg = torch.as_tensor(float(angle_deg), dtype=torch.float64)
    axis_t = _as_tensor(axis, ref=angle_deg)
    axis_t = axis_t / torch.linalg.vector_norm(axis_t, dim=-1, keepdim=True)
    return axis_angle_to_orient_mat(axis_t, angle_deg)


def rot_x(angle_deg) -> torch.Tensor:
    """Rotation about lab ``x`` (incident-beam axis)."""
    return rotation_matrix(_XHAT, angle_deg)


def rot_y(angle_deg) -> torch.Tensor:
    """Rotation about lab ``y`` (horizontal, transverse)."""
    return rotation_matrix(_YHAT, angle_deg)


def rot_z(angle_deg) -> torch.Tensor:
    """Rotation about lab ``z`` (vertical)."""
    return rotation_matrix(_ZHAT, angle_deg)


@dataclass
class GoniometerSetting:
    """One DFXM goniometer setting (all angles in **degrees**).

    ``mu``    base tilt about lab ``y`` (brings reflection to the diffraction
              condition).
    ``omega`` in-plane rotation about lab ``z``.
    ``chi``   rocking about lab ``x``.
    ``phi``   rocking about lab ``y``.

    A *mosaicity scan* sweeps ``(chi, phi)``; a *strain scan* sweeps ``two_theta``
    (energy) at fixed goniometer. See :mod:`midas_dfxm.scan`.
    """

    mu: float = 0.0
    omega: float = 0.0
    chi: float = 0.0
    phi: float = 0.0

    def sample_rotation(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        """Return ``G``: the ``(3, 3)`` sample->lab rotation for this setting.

        ``v_lab = G @ v_sample``. Differentiable in the motor angles when they are
        passed as tensors (see :meth:`sample_rotation_grad`).
        """
        ref = torch.zeros((), device=device, dtype=dtype)
        mu = _as_tensor(self.mu, ref=ref)
        omega = _as_tensor(self.omega, ref=ref)
        chi = _as_tensor(self.chi, ref=ref)
        phi = _as_tensor(self.phi, ref=ref)
        return rot_y(mu) @ rot_z(omega) @ rot_x(chi) @ rot_y(phi)

    @staticmethod
    def compose(mu, omega, chi, phi) -> torch.Tensor:
        """Differentiable sample->lab rotation directly from tensor motor angles.

        Use this inside autograd graphs (e.g. refining motor angles): pass tensors
        with ``requires_grad=True``.
        """
        return rot_y(mu) @ rot_z(omega) @ rot_x(chi) @ rot_y(phi)
