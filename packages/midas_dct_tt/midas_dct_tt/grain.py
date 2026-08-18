"""Grain container: differentiable voxel occupancy ``chi(r)`` + optional ``F(r)``.

Phase 0 of ``implementation_plan.md``.

A grain in DCT/TT is two fields on the same voxel grid:

* **occupancy** ``chi(r) in [0, 1]`` -- is this voxel inside the grain? This is
  what a shape reconstruction recovers, and what a topograph is a projection of.
* **deformation** ``F(r)`` -- the local deformation gradient, which sets both the
  local scattering vector ``Q = F^-T G0`` (so whether the voxel is inside the
  acceptance at all) and the direction it diffracts into (so where it lands on
  the detector). Reused wholesale from
  :class:`midas_dfxm.field.DeformationField`; never re-implemented here.

Occupancy is stored as unconstrained **logits**, with ``chi = sigmoid(logits)``.
That keeps ``chi`` in ``[0, 1]`` by construction under any optimiser, with no
projection step and no clamping to kill gradients at the boundary -- the
reconstruction in Phase 2 optimises ``logits`` directly.

Units: positions in micrometers (sample frame), lattice in Angstrom.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

__all__ = ["GrainShape", "logits_from_signed_distance"]


def logits_from_signed_distance(sdf_um: torch.Tensor, edge_width_um: float) -> torch.Tensor:
    """Occupancy logits from a signed distance field (negative = inside).

    ``chi = sigmoid(-sdf / edge_width)``: 0.5 exactly on the surface, saturating
    over a few ``edge_width`` either side. A *soft* boundary rather than a hard
    mask, for two reasons: the gradient w.r.t. shape parameters (radius, facet
    position) is non-zero, and partial-volume voxels at a real grain boundary are
    physically partial, not binary.

    Known bias (measured, not estimated)
    ------------------------------------
    On a **convex** surface the soft boundary systematically *over-counts*
    volume, because the outer shell it adds is larger than the inner shell it
    removes. The leading term is exactly

        dV = A * (2H) * w^2 * pi^2 / 6

    with ``A`` the surface area, ``2H`` the trace of the curvature (``2/R`` for a
    sphere) and ``w = edge_width_um``. Verified against a planted sphere
    (R = 6 um, 0.5 um voxels) to 0.03% at ``w = 0.5`` and ``w = 0.25`` um -- that
    is 6.9% and 1.7% of the sphere's volume respectively. Shapes with edges
    (boxes, facets) are worse per unit volume, since every convex edge
    contributes and small grains have a lot of edge for their size.

    The bias falls as ``w^2``, but **not indefinitely**: below ``w ~ 0.25 *
    spacing`` the grid no longer samples the transition, the quadrature fails,
    and the error changes sign (measured: -0.1% at ``w = 0.0625`` um on the same
    0.5 um grid, where the formula predicts +0.1%). There is a floor, and it sits
    near ``w = 0.25-0.5 * spacing``; the default in :mod:`midas_dct_tt.io` is
    ``0.5 * spacing``, the sampling-safe end of that window.

    This matters for *reported* volumes, not for plant-and-recover scoring: the
    forward model integrates ``chi`` itself, so ground truth and reconstruction
    share the parameterisation and the bias cancels. Do not quote a grain volume
    from a soft occupancy without either correcting by the formula above or
    stating ``w``.
    """
    if edge_width_um <= 0:
        raise ValueError("edge_width_um must be > 0 (a hard mask has no shape gradient)")
    return -sdf_um / edge_width_um


@dataclass
class GrainShape:
    """Voxelised grain: occupancy logits on a point cloud or regular grid.

    Attributes
    ----------
    positions : (N, 3) tensor
        Voxel centres in micrometers, **sample frame**, conventionally centred on
        the grain centroid.
    logits : (N,) tensor
        Occupancy logits; ``chi = sigmoid(logits)``.
    spacing_um : float or (3,) tuple
        Voxel pitch. Sets the voxel volume used by :meth:`volume_um3`.
    shape : tuple[int, int, int] | None
        ``(nx, ny, nz)`` if the voxels form a regular grid (enables reshaping to
        an image); ``None`` for an unstructured cloud.
    field : midas_dfxm.field.DeformationField | None
        Optional per-voxel ``F(r)`` on the *same* positions. Phase 1+ uses it;
        Phase 0 shape work leaves it ``None``.
    """

    positions: torch.Tensor
    logits: torch.Tensor
    spacing_um: float = 1.0
    shape: tuple = None
    field: object = None

    def __post_init__(self):
        if self.positions.shape[0] != self.logits.shape[0]:
            raise ValueError(
                f"positions ({self.positions.shape[0]}) and logits "
                f"({self.logits.shape[0]}) disagree on voxel count"
            )
        if self.positions.shape[-1] != 3:
            raise ValueError("positions must be (N, 3)")
        if self.field is not None and self.field.positions.shape[0] != self.positions.shape[0]:
            raise ValueError(
                "field is defined on a different voxel count than the occupancy; "
                "chi(r) and F(r) must share the grid"
            )

    # -- basic properties ---------------------------------------------------
    @property
    def n_voxels(self) -> int:
        return self.positions.shape[0]

    @property
    def occupancy(self) -> torch.Tensor:
        """``chi(r) = sigmoid(logits)``, shape ``(N,)``, differentiable in logits."""
        return torch.sigmoid(self.logits)

    @property
    def voxel_volume_um3(self) -> float:
        s = self.spacing_um
        if isinstance(s, (tuple, list)):
            return float(s[0]) * float(s[1]) * float(s[2])
        return float(s) ** 3

    def volume_um3(self) -> torch.Tensor:
        """Occupancy-weighted grain volume ``sum(chi) * v_voxel``, differentiable."""
        return self.occupancy.sum() * self.voxel_volume_um3

    def centroid_um(self) -> torch.Tensor:
        """Occupancy-weighted centroid ``(3,)`` in the sample frame."""
        w = self.occupancy
        return (w.unsqueeze(-1) * self.positions).sum(dim=0) / w.sum()

    # -- frames -------------------------------------------------------------
    def positions_lab(self, sample_to_lab: torch.Tensor) -> torch.Tensor:
        """Voxel centres in the lab frame: ``v_lab = R @ v_sample``, shape ``(N, 3)``.

        ``sample_to_lab`` is the ``(3, 3)`` rotation from
        :meth:`midas_dct_tt.conventions.TTAlignment.sample_rotation` (TT) or
        :func:`midas_dct_tt.conventions.dct_sample_rotation` (DCT). Applied as a
        right-multiply by ``R^T`` so the ``(N, 3)`` row layout is preserved.
        """
        R = sample_to_lab.to(device=self.positions.device, dtype=self.positions.dtype)
        return self.positions @ R.transpose(-1, -2)

    def occupancy_image(self) -> torch.Tensor:
        """Occupancy reshaped to ``(nx, ny, nz)``. Requires a regular grid."""
        if self.shape is None:
            raise ValueError("occupancy_image() needs a regular grid (shape is None)")
        return self.occupancy.reshape(self.shape)

    # -- plumbing -----------------------------------------------------------
    def with_field(self, field) -> "GrainShape":
        """Return a copy carrying a :class:`~midas_dfxm.field.DeformationField`."""
        return replace(self, field=field)

    def requires_grad_(self, flag: bool = True) -> "GrainShape":
        """Mark the occupancy logits as an optimisation variable (in place)."""
        self.logits.requires_grad_(flag)
        return self

    def to(self, *, device=None, dtype=None) -> "GrainShape":
        """Return a copy on the given device/dtype (field moved with it)."""
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return GrainShape(
            positions=self.positions.to(**kw),
            logits=self.logits.to(**kw),
            spacing_um=self.spacing_um,
            shape=self.shape,
            field=None if self.field is None else self.field.to(**kw),
        )
