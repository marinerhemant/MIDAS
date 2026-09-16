"""Synthetic-field generators and data loaders.

Phase 0 of ``implementation_plan.md``.

Until a collaborator's realistic field (orientation map + deformation-gradient
field) arrives, the whole forward/inverse chain is exercised on **synthetic**
fields planted here: a perfect crystal, a smooth orientation gradient (uniform
lattice curvature), a uniform elastic strain, and an isotropic screw dislocation
(clean analytic distortion for tests). The full anisotropic-elasticity (Stroh)
dislocation forward lands in Phase 4 (``dislocation.py``); the isotropic screw
here is only a Phase-0/1 test fixture.

``load_external_field`` parses a real LAMMPS atomic dump (MD output) via
:mod:`midas_dfxm.md_ingest`; see that module for the per-atom-to-voxel-field
ingestion pattern.

Units: positions in micrometers; lattice in Angstrom; Burgers vector in Angstrom.
"""
from __future__ import annotations

import torch

from midas_defect.lattice import fcc_cu_crystal
from midas_stress.orientation import axis_angle_to_orient_mat

from .field import DeformationField


def fcc_reference_crystal(a: float = 3.6356):
    """FCC reference crystal (reuses ``midas_defect.lattice.fcc_cu_crystal``)."""
    return fcc_cu_crystal(a=a)


def _grid_positions(shape, spacing_um, *, device, dtype) -> torch.Tensor:
    """Regular ``(nx, ny, nz)`` grid of voxel-centre positions (N, 3), microns."""
    nx, ny, nz = shape
    axes = []
    for n in (nx, ny, nz):
        c = (n - 1) / 2.0
        axes.append((torch.arange(n, device=device, dtype=dtype) - c) * spacing_um)
    gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)


def make_uniform_field(
    shape=(16, 16, 1),
    *,
    spacing_um: float = 1.0,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    orientation=None,
    device=None,
    dtype=torch.float64,
) -> DeformationField:
    """Perfect single crystal: ``F = I`` at every voxel.

    ``orientation`` is a ``(3, 3)`` reference orientation ``OM0`` (crystal->sample);
    defaults to identity. Returns a :class:`DeformationField` on a regular grid.
    """
    positions = _grid_positions(shape, spacing_um, device=device, dtype=dtype)
    n = positions.shape[0]
    F = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3).clone()
    if orientation is None:
        orientation = torch.eye(3, device=device, dtype=dtype)
    else:
        orientation = torch.as_tensor(orientation, device=device, dtype=dtype)
    latc = torch.as_tensor(lattice_params, device=device, dtype=dtype)
    return DeformationField(
        positions=positions,
        F=F,
        reference_orientation=orientation,
        lattice_params=latc,
        shape=tuple(shape),
    )


def ellipsoid_mask(
    positions: torch.Tensor,
    center_um,
    radius_um,
) -> torch.Tensor:
    """Boolean mask for voxels inside a sphere/ellipsoid region.

    ``center_um`` and ``radius_um`` are length-3 (or ``radius_um`` a scalar,
    broadcasting to a sphere). ``(N,)`` bool, ``True`` where
    ``sum(((r - center) / radius)**2) <= 1``. Shared by
    :func:`make_grain_in_matrix_field` and by callers that need the same region
    to place a defect (e.g. a DDD network) inside the planted grain.
    """
    center = torch.as_tensor(center_um, device=positions.device, dtype=positions.dtype)
    r = radius_um if isinstance(radius_um, (tuple, list)) else (radius_um,) * 3
    radius = torch.as_tensor(r, device=positions.device, dtype=positions.dtype)
    d = (positions - center) / radius
    return d.pow(2).sum(dim=-1) <= 1.0


def make_grain_in_matrix_field(
    shape=(48, 48, 24),
    *,
    spacing_um: float = 0.25,
    grain_shape: str = "ellipsoid",
    grain_center_um=(0.0, 0.0, 0.0),
    grain_radius_um=(6.0, 6.0, 6.0),
    grain_orientation=None,
    matrix_orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    device=None,
    dtype=torch.float64,
) -> DeformationField:
    """A discrete grain (sphere/ellipsoid) embedded in a matrix, each with its
    own crystal orientation, on a regular grid.

    ``reference_orientation`` of the returned field is ``matrix_orientation``
    (identity if not given): matrix voxels get ``F = I`` and grain voxels get
    the constant rotation ``matrix_orientation^T @ grain_orientation`` (so the
    grain's *physical* orientation is exactly ``grain_orientation``, and if
    ``grain_orientation`` is not given the field is uniform, i.e. no grain is
    visible). One ``F(r)``, one field -- the same contract every generator in
    this module uses; callers that need the region mask separately (e.g. to
    insert a DDD network only inside the grain) should call
    :func:`ellipsoid_mask` with the same ``grain_center_um``/``grain_radius_um``.

    ``grain_shape`` is currently the only allowed value ``"ellipsoid"`` (a
    sphere is an ellipsoid with equal radii); kept as an explicit parameter so
    a future non-ellipsoidal shape does not change the call signature.
    """
    if grain_shape != "ellipsoid":
        raise ValueError(f"grain_shape must be 'ellipsoid' (sphere = equal radii), got {grain_shape!r}")
    positions = _grid_positions(shape, spacing_um, device=device, dtype=dtype)
    n = positions.shape[0]
    if matrix_orientation is None:
        matrix_orientation = torch.eye(3, device=device, dtype=dtype)
    else:
        matrix_orientation = torch.as_tensor(matrix_orientation, device=device, dtype=dtype)
    if grain_orientation is None:
        grain_orientation = matrix_orientation
    else:
        grain_orientation = torch.as_tensor(grain_orientation, device=device, dtype=dtype)

    inside = ellipsoid_mask(positions, grain_center_um, grain_radius_um)

    eye = torch.eye(3, device=device, dtype=dtype)
    F = eye.expand(n, 3, 3).clone()
    R_grain = matrix_orientation.transpose(-1, -2) @ grain_orientation
    F[inside] = R_grain
    latc = torch.as_tensor(lattice_params, device=device, dtype=dtype)
    return DeformationField(
        positions=positions,
        F=F,
        reference_orientation=matrix_orientation,
        lattice_params=latc,
        shape=tuple(shape),
    )


def with_orientation_gradient(
    field: DeformationField,
    *,
    axis=(0.0, 0.0, 1.0),
    deg_per_um: float = 0.01,
    along=0,
) -> DeformationField:
    """Add a smooth lattice curvature: rotate about ``axis`` linearly along a
    spatial direction ``along`` (0=x, 1=y, 2=z) at ``deg_per_um``.

    Left-multiplies each voxel's ``F`` by a small position-dependent rotation, so
    the local rotation ``R`` from polar decomposition recovers the planted
    curvature (validated in tests). Differentiable in ``deg_per_um``.
    """
    pos = field.positions
    angle_deg = deg_per_um * pos[:, along]  # (N,)
    axis_t = torch.as_tensor(axis, device=pos.device, dtype=pos.dtype)
    Rg = axis_angle_to_orient_mat(axis_t, angle_deg)  # (N, 3, 3)
    F_new = Rg @ field.F
    return DeformationField(
        positions=pos,
        F=F_new,
        reference_orientation=field.reference_orientation,
        lattice_params=field.lattice_params,
        shape=field.shape,
    )


def with_uniform_strain(
    field: DeformationField,
    strain_tensor,
) -> DeformationField:
    """Superpose a uniform infinitesimal strain: ``F -> (I + eps) @ F``.

    ``strain_tensor`` is a ``(3, 3)`` symmetric tensor. Differentiable.
    """
    eps = torch.as_tensor(
        strain_tensor, device=field.F.device, dtype=field.F.dtype
    )
    eye = torch.eye(3, device=field.F.device, dtype=field.F.dtype)
    F_new = (eye + eps) @ field.F
    return DeformationField(
        positions=field.positions,
        F=F_new,
        reference_orientation=field.reference_orientation,
        lattice_params=field.lattice_params,
        shape=field.shape,
    )


def with_screw_dislocation(
    field: DeformationField,
    *,
    burgers_A: float = 2.556,
    line_axis=(0.0, 0.0, 1.0),
    core_position=(0.0, 0.0),
    core_radius_um: float = 0.5,
) -> DeformationField:
    """Superpose an **isotropic** screw dislocation (Phase-0/1 test fixture only).

    For a screw line along ``z`` through the core, ``u_z = (b/2pi) atan2(Y, X)``,
    giving a distortion with ``dF[2,0] = du_z/dX``, ``dF[2,1] = du_z/dY``. The core
    singularity is regularised by ``core_radius_um`` (``r^2 -> r^2 + r_c^2``). This
    is the isotropic limit; the anisotropic-elasticity (Stroh) version — needed for
    real defect typing — lands in Phase 4.

    Only ``line_axis = z`` is implemented in this fixture (asserted); general lines
    come with the Stroh forward.
    """
    line = torch.as_tensor(line_axis, device=field.F.device, dtype=field.F.dtype)
    assert torch.allclose(line, torch.tensor([0.0, 0.0, 1.0], device=line.device, dtype=line.dtype)), \
        "isotropic screw fixture supports line_axis=z only; use Phase-4 Stroh forward otherwise"
    pos = field.positions
    b_um = burgers_A * 1e-4  # Angstrom -> micrometer
    X = pos[:, 0] - core_position[0]
    Y = pos[:, 1] - core_position[1]
    r2 = X * X + Y * Y + core_radius_um ** 2
    duz_dx = -(b_um / (2.0 * torch.pi)) * Y / r2
    duz_dy = (b_um / (2.0 * torch.pi)) * X / r2
    dF = torch.zeros_like(field.F)
    dF[:, 2, 0] = duz_dx
    dF[:, 2, 1] = duz_dy
    return DeformationField(
        positions=pos,
        F=field.F + dF,
        reference_orientation=field.reference_orientation,
        lattice_params=field.lattice_params,
        shape=field.shape,
    )


def load_external_field(path: str, **kwargs) -> DeformationField:
    """Load an external collaborator's realistic field.

    Delegates to :func:`midas_dfxm.md_ingest.load_external_field`: the
    delivery format this was finalised against is a real LAMMPS atomic dump
    (MD output), which is also the format a collaborator's own MD run
    produces. See that module for the ingestion pattern (per-atom local
    deformation gradient -> voxel-binned :class:`DeformationField`) and its
    required ``crop_angstrom`` argument.
    """
    from .md_ingest import load_external_field as _load_md_field
    return _load_md_field(path, **kwargs)
