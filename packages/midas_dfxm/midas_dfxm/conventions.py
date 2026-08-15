"""DFXM frame & goniometer conventions, and the lab-frame / scattering-plane map.

Phase 0 of ``implementation_plan.md``.

Frames
------
Which lab axis carries the beam, and which is up, is a **beamline convention**,
not physics. Two are in active use for the same experiment:

* **MIDAS / ESRF** (the package default): ``x`` along the incident beam, ``z``
  up, ``y`` outboard. Matches the MIDAS lab frame used by the rest of the
  packages, and Poulsen 2017 / DTU ``darling``.
* **APS (Park convention)**: ``Z`` along the beam, ``Y`` up, ``X`` outboard.

:class:`LabFrame` carries the choice, and any other right-handed (beam, up) pair
works too -- e.g. beam along ``x`` with ``y`` up. The third axis is *derived*
(``outboard = up x beam``), so a frame cannot be built inconsistently. Convert
between frames with :func:`frame_rotation` / :func:`convert_vector` /
:func:`convert_tensor` / :func:`convert_orientation`. For the MIDAS<->APS pair
these agree exactly with :mod:`midas_stress.frames` (unit-tested), which remains
the repo-wide authority for that map.

* **Sample frame**: the crystal/grain reference frame. A goniometer rotation
  ``G`` maps sample-frame vectors into the lab frame: ``v_lab = G @ v_sample``.
* **Imaging (objective) frame**: axis along the diffracted beam ``k_out`` at
  ``2*theta`` from the beam in the scattering plane; handled in ``optics.py``.

Scattering plane
----------------
:class:`ScatteringGeometry` says which plane ``k``, ``k'`` and ``Q`` live in:

* ``"vertical"`` -- the plane spanned by beam and up (ESRF ID06-HXM; the
  package default, and the geometry Poulsen 2017 Eqs. 58-63 are written for).
* ``"horizontal"`` -- the plane spanned by beam and outboard (APS 6-ID-C
  transmission geometry).
* any explicit unit ``deflection`` direction perpendicular to the beam.

This is **not** a relabelling: it changes which motor is the base tilt, and it
swaps which incident divergence limits ``sigma_rock`` versus ``sigma_roll``
(:func:`midas_dfxm.resolution.poulsen_resolution_widths`). Both follow from the
geometry here, so downstream code never re-derives them.

Goniometer
----------
A DFXM diffractometer stacks a base tilt and rocking motors. We parameterise the
standard set as an ordered composition (outer-most first), in terms of the
*named* frame axes rather than hard-coded ``x``/``y``/``z``:

    G(mu, omega, chi, phi) = R_ob(mu) @ R_up(omega) @ R_beam(chi) @ R_ob(phi)

which in the default MIDAS frame is the familiar
``R_y(mu) @ R_z(omega) @ R_x(chi) @ R_y(phi)``. All angles in **degrees**. In the
vertical-plane geometry ``mu`` is the base tilt bringing the reflection into the
diffraction condition and ``omega`` is the in-plane rotation; ``chi``/``phi`` are
the two rocking axes a *mosaicity scan* sweeps.

**Their roles are geometry-dependent.** In the horizontal-plane geometry the base
tilt is ``omega`` (about up), and ``mu`` is inert for a reflection whose ``Q``
lies along outboard -- it rotates about the very axis ``Q`` sits on. Ask the
beamline for its motor -> physical-axis map; it does not follow from the frame.
The axis assignment is a convention knob (some beamlines swap ``chi``/``phi``);
it is centralised here and unit-tested, so downstream code never re-derives it.

All builders are torch-differentiable and device/dtype-preserving; they delegate
the axis-angle rotation to :func:`midas_stress.orientation.axis_angle_to_orient_mat`
(Rodrigues, degrees) so we never re-port rotation math.

Units: degrees for all angles (per MIDAS convention), Angstrom for wavelength.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Lab frame: which axis is the beam, which is up
# ---------------------------------------------------------------------------

def _unit(v, what: str) -> tuple[float, float, float]:
    """Coerce to a plain 3-tuple unit vector, or raise."""
    t = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    if t.numel() != 3:
        raise ValueError(f"{what} must have 3 components, got {t.numel()}")
    n = float(torch.linalg.vector_norm(t))
    if n < 1e-12:
        raise ValueError(f"{what} must be a non-zero vector")
    return tuple(float(c) for c in (t / n))


def _cross3(a, b) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


@dataclass(frozen=True)
class LabFrame:
    """Which lab axis carries the beam and which is up. Everything else follows.

    ``outboard`` is **derived** as ``up x beam``, so ``(beam, outboard, up)`` is
    always right-handed and a frame cannot be built inconsistently. Give ``beam``
    and ``up`` as any orthogonal unit pair -- the named presets
    (:data:`MIDAS_FRAME`, :data:`APS_FRAME`, :data:`BEAM_X_UP_Y`) cover the
    conventions in use, but arbitrary directions are accepted.

    Examples
    --------
    >>> MIDAS_FRAME.outboard          # x beam, z up  -> y outboard
    (0.0, 1.0, 0.0)
    >>> APS_FRAME.outboard            # Z beam, Y up  -> X outboard
    (1.0, 0.0, 0.0)
    """

    beam: tuple[float, float, float] = _XHAT
    up: tuple[float, float, float] = _ZHAT
    name: str = ""

    def __post_init__(self):
        beam = _unit(self.beam, "beam")
        up = _unit(self.up, "up")
        dot = sum(a * b for a, b in zip(beam, up))
        if abs(dot) > 1e-9:
            raise ValueError(
                f"beam and up must be orthogonal (beam.up = {dot:.3g}). "
                "Give the two axes of the frame; outboard is derived as up x beam."
            )
        object.__setattr__(self, "beam", beam)
        object.__setattr__(self, "up", up)

    @property
    def outboard(self) -> tuple[float, float, float]:
        """Derived third axis: ``up x beam``, completing the right-handed set."""
        return _cross3(self.up, self.beam)

    def basis(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        """``(3, 3)`` matrix whose ROWS are ``(beam, outboard, up)`` in this frame's
        own components. Maps a vector's components to its physical
        ``(along-beam, outboard, up)`` components."""
        return torch.tensor([self.beam, self.outboard, self.up],
                            device=device, dtype=dtype)

    def axis(self, which: str, *, device=None, dtype=torch.float64) -> torch.Tensor:
        """Unit ``(3,)`` tensor for ``'beam'``, ``'outboard'`` (``'ob'``) or ``'up'``."""
        key = {"beam": self.beam, "outboard": self.outboard, "ob": self.outboard,
               "up": self.up}.get(which)
        if key is None:
            raise ValueError(f"unknown axis '{which}'; use 'beam', 'outboard' or 'up'")
        return torch.tensor(key, device=device, dtype=dtype)


#: MIDAS / ESRF convention: ``x`` beam, ``y`` outboard, ``z`` up. Package default.
MIDAS_FRAME = LabFrame(beam=_XHAT, up=_ZHAT, name="midas")
#: APS (Park) convention: ``X`` outboard, ``Y`` up, ``Z`` beam.
APS_FRAME = LabFrame(beam=_ZHAT, up=_YHAT, name="aps")
#: Beam along ``x`` with ``y`` up (outboard is then ``-z``).
BEAM_X_UP_Y = LabFrame(beam=_XHAT, up=_YHAT, name="beam_x_up_y")

#: Lookup for the named presets, for string-driven config.
NAMED_FRAMES = {"midas": MIDAS_FRAME, "esrf": MIDAS_FRAME,
                "aps": APS_FRAME, "park": APS_FRAME,
                "beam_x_up_y": BEAM_X_UP_Y}


def as_frame(frame) -> LabFrame:
    """Resolve a :class:`LabFrame`, a preset name, or ``None`` (-> MIDAS)."""
    if frame is None:
        return MIDAS_FRAME
    if isinstance(frame, LabFrame):
        return frame
    try:
        return NAMED_FRAMES[str(frame).lower()]
    except KeyError:
        raise ValueError(
            f"unknown frame '{frame}'; use a LabFrame or one of "
            f"{sorted(NAMED_FRAMES)}"
        ) from None


def frame_rotation(src, dst, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """``(3, 3)`` rotation taking vector COMPONENTS from ``src`` to ``dst``.

    ``v_dst = frame_rotation(src, dst) @ v_src``. Both arguments accept a
    :class:`LabFrame` or a preset name. For the MIDAS->APS pair this equals
    :data:`midas_stress.frames.R_MIDAS_TO_APS` exactly (unit-tested).
    """
    s, d = as_frame(src), as_frame(dst)
    return d.basis(device=device, dtype=dtype).T @ s.basis(device=device, dtype=dtype)


def convert_vector(v, src, dst) -> torch.Tensor:
    """Convert vector(s) ``(..., 3)`` from frame ``src`` to frame ``dst``."""
    v = torch.as_tensor(v)
    if not torch.is_floating_point(v):
        v = v.to(torch.float64)
    R = frame_rotation(src, dst, device=v.device, dtype=v.dtype)
    return v @ R.T


def convert_tensor(T, src, dst) -> torch.Tensor:
    """Convert rank-2 tensor(s) ``(..., 3, 3)`` between frames: ``R T R^T``.

    Use for strain, the deformation gradient ``F``, and the Nye tensor.
    """
    T = torch.as_tensor(T)
    if not torch.is_floating_point(T):
        T = T.to(torch.float64)
    R = frame_rotation(src, dst, device=T.device, dtype=T.dtype)
    return R @ T @ R.T


def convert_orientation(U, src, dst) -> torch.Tensor:
    """Convert orientation matri(ces) ``(..., 3, 3)`` between frames.

    An orientation maps crystal -> lab, so only the lab side rotates: ``R @ U``.
    (Contrast :func:`convert_tensor`, where both indices are lab indices.)
    """
    U = torch.as_tensor(U)
    if not torch.is_floating_point(U):
        U = U.to(torch.float64)
    R = frame_rotation(src, dst, device=U.device, dtype=U.dtype)
    return R @ U


# ---------------------------------------------------------------------------
# Scattering plane
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScatteringGeometry:
    """Which plane ``k``, ``k'`` and ``Q`` live in, in a given :class:`LabFrame`.

    The beam deflects from ``beam`` toward :attr:`deflection` by ``2*theta``;
    the scattering-plane normal ``beam x deflection`` is derived, so there is no
    sign to get wrong. ``plane`` is a shorthand:

    ===============  ==========================  ==============================
    ``plane``        deflection                  where
    ===============  ==========================  ==============================
    ``"vertical"``   ``+up``                     ESRF ID06-HXM (default)
    ``"horizontal"`` ``+outboard``               APS 6-ID-C transmission
    ``"custom"``     explicit ``deflection``     anything else
    ===============  ==========================  ==============================

    The default ``ScatteringGeometry()`` reproduces the previous hard-coded
    vertical ``x``-``z`` behaviour exactly.
    """

    frame: LabFrame = field(default=MIDAS_FRAME)
    plane: str = "vertical"
    deflection: tuple[float, float, float] | None = None

    def __post_init__(self):
        object.__setattr__(self, "frame", as_frame(self.frame))
        plane = str(self.plane).lower()
        if self.deflection is not None:
            d = _unit(self.deflection, "deflection")
            if abs(sum(a * b for a, b in zip(d, self.frame.beam))) > 1e-9:
                raise ValueError("deflection must be perpendicular to the beam")
            object.__setattr__(self, "deflection", d)
            object.__setattr__(self, "plane", "custom")
            return
        if plane == "vertical":
            object.__setattr__(self, "deflection", self.frame.up)
        elif plane == "horizontal":
            object.__setattr__(self, "deflection", self.frame.outboard)
        else:
            raise ValueError(
                f"plane must be 'vertical' or 'horizontal' (got '{self.plane}'), "
                "or pass an explicit deflection= direction"
            )
        object.__setattr__(self, "plane", plane)

    # -- directions ---------------------------------------------------------
    def beam_direction(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        return self.frame.axis("beam", device=device, dtype=dtype)

    def deflection_direction(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        return torch.tensor(self.deflection, device=device, dtype=dtype)

    def plane_normal(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        """Unit normal to the scattering plane, ``beam x deflection``.

        This is the **rock** axis: rotating about it moves ``Q`` within the
        scattering plane, i.e. changes ``theta``.
        """
        return torch.tensor(_cross3(self.frame.beam, self.deflection),
                            device=device, dtype=dtype)

    def k_out(self, two_theta_deg, *, device=None, dtype=torch.float64) -> torch.Tensor:
        """Unit diffracted-beam direction at ``2*theta``, in this geometry.

        ``k_out = cos(2 theta) * beam + sin(2 theta) * deflection``. Differentiable
        in ``two_theta_deg`` when it is a tensor.
        """
        ref = torch.zeros((), device=device, dtype=dtype)
        tt = torch.deg2rad(_as_tensor(two_theta_deg, ref=ref))
        b = self.beam_direction(device=ref.device, dtype=ref.dtype)
        d = self.deflection_direction(device=ref.device, dtype=ref.dtype)
        return torch.cos(tt) * b + torch.sin(tt) * d

    # -- resolution roles ---------------------------------------------------
    def divergences(self, div_v: float, div_h: float) -> dict:
        """Split incident divergence into in-plane / out-of-plane components.

        Poulsen 2017 Eqs. 58-63 are written for a **vertical** scattering plane,
        where the rocking width is limited by the *vertical* divergence and the
        rolling width by the *horizontal* one. That assignment is a property of
        the geometry, not of the physics: for a horizontal scattering plane the
        two swap. Resolving it here means
        :func:`midas_dfxm.resolution.poulsen_resolution_widths` never has to.

        Returns ``{'div_in_plane', 'div_out_of_plane'}``. For an oblique
        ``deflection`` the components combine in quadrature by projection, which
        reduces exactly to the two named cases.
        """
        up, ob = self.frame.up, self.frame.outboard
        d, n = self.deflection, tuple(float(c) for c in self.plane_normal())
        proj = lambda a, b: sum(x * y for x, y in zip(a, b))
        in_plane = math.sqrt((proj(d, up) * div_v) ** 2 + (proj(d, ob) * div_h) ** 2)
        out_plane = math.sqrt((proj(n, up) * div_v) ** 2 + (proj(n, ob) * div_h) ** 2)
        return {"div_in_plane": in_plane, "div_out_of_plane": out_plane}


#: The package default: vertical scattering plane in the MIDAS frame (ESRF ID06).
VERTICAL_MIDAS = ScatteringGeometry(frame=MIDAS_FRAME, plane="vertical")
#: APS 6-ID-C transmission geometry expressed in MIDAS frame labels.
HORIZONTAL_MIDAS = ScatteringGeometry(frame=MIDAS_FRAME, plane="horizontal")
#: APS 6-ID-C transmission geometry in APS (Park) frame labels.
HORIZONTAL_APS = ScatteringGeometry(frame=APS_FRAME, plane="horizontal")


def as_geometry(geometry) -> ScatteringGeometry:
    """Resolve a :class:`ScatteringGeometry`, a plane name, or ``None`` (-> default)."""
    if geometry is None:
        return VERTICAL_MIDAS
    if isinstance(geometry, ScatteringGeometry):
        return geometry
    return ScatteringGeometry(plane=str(geometry))


@dataclass
class GoniometerSetting:
    """One DFXM goniometer setting (all angles in **degrees**).

    ``mu``    base tilt about **outboard**.
    ``omega`` rotation about **up**.
    ``chi``   rocking about the **beam**.
    ``phi``   rocking about **outboard**.

    In the default MIDAS frame those are ``y``, ``z``, ``x``, ``y`` -- i.e.
    ``G = R_y(mu) @ R_z(omega) @ R_x(chi) @ R_y(phi)``, unchanged. Set ``frame``
    to compose about another convention's axes instead; the angle *values* are
    the same physical rotations either way, only their components change
    (``G_dst = R G_src R^T``, unit-tested).

    Which motor is the base tilt depends on the **scattering geometry**, not on
    the frame. In a vertical scattering plane it is ``mu``; in a horizontal one it
    is ``omega``, and ``mu`` is inert for a reflection whose ``Q`` lies along
    outboard. See :class:`ScatteringGeometry`.

    A *mosaicity scan* sweeps ``(chi, phi)``; a *strain scan* sweeps ``two_theta``
    (energy) at fixed goniometer. See :mod:`midas_dfxm.scan`.
    """

    mu: float = 0.0
    omega: float = 0.0
    chi: float = 0.0
    phi: float = 0.0
    frame: LabFrame = MIDAS_FRAME

    def __post_init__(self):
        self.frame = as_frame(self.frame)

    def sample_rotation(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        """Return ``G``: the ``(3, 3)`` sample->lab rotation for this setting.

        ``v_lab = G @ v_sample``, with components in :attr:`frame`. Differentiable
        in the motor angles when they are passed as tensors (see :meth:`compose`).
        """
        ref = torch.zeros((), device=device, dtype=dtype)
        return self.compose(
            _as_tensor(self.mu, ref=ref),
            _as_tensor(self.omega, ref=ref),
            _as_tensor(self.chi, ref=ref),
            _as_tensor(self.phi, ref=ref),
            frame=self.frame,
        )

    @staticmethod
    def compose(mu, omega, chi, phi, *, frame=None) -> torch.Tensor:
        """Differentiable sample->lab rotation directly from tensor motor angles.

        Use this inside autograd graphs (e.g. refining motor angles): pass tensors
        with ``requires_grad=True``.
        """
        f = as_frame(frame)
        return (rotation_matrix(f.outboard, mu)
                @ rotation_matrix(f.up, omega)
                @ rotation_matrix(f.beam, chi)
                @ rotation_matrix(f.outboard, phi))

    def in_frame(self, dst) -> "GoniometerSetting":
        """Same physical setting, expressed in another :class:`LabFrame`.

        The motor angles are unchanged -- only the axes they are taken about are
        relabelled -- so this is exactly the conjugation ``G_dst = R G_src R^T``.
        """
        return GoniometerSetting(mu=self.mu, omega=self.omega, chi=self.chi,
                                 phi=self.phi, frame=as_frame(dst))

    @classmethod
    def from_aps(cls, mu=0.0, omega=0.0, chi=0.0, phi=0.0) -> "GoniometerSetting":
        """Build a setting whose components are in the APS (Park) frame.

        Convenience for ingesting beamline geometry supplied in APS coordinates.
        Call :meth:`in_frame` to move it to the MIDAS frame the rest of the
        package works in.
        """
        return cls(mu=mu, omega=omega, chi=chi, phi=phi, frame=APS_FRAME)
