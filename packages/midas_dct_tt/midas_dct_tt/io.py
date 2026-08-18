"""Synthetic grain generators (loaders for real DCT/TT stacks land in Phase 2+).

Phase 0 of ``implementation_plan.md``.

Planted shapes for plant-and-recover validation. The three that matter for the
missing-cone study are here from the start:

* :func:`sphere_grain` -- isotropic, so any axial elongation in a reconstruction
  is an artefact of the missing cone and nothing else. The Phase 2 falsifiable
  test.
* :func:`plate_grain` -- worst case: a thin plate whose normal lies in the
  missing cone is nearly invisible to the scan.
* :func:`faceted_grain` -- a convex polyhedron, the realistic case, where facet
  orientation relative to the cone decides which facets survive.

All shapes are built from signed distance fields, so occupancy is soft at the
boundary and differentiable in the shape parameters (radius, facet distances) --
not just in the per-voxel logits.

Units: micrometers throughout; lattice in Angstrom.
"""
from __future__ import annotations

import torch
from midas_dfxm.io import make_uniform_field

from .grain import GrainShape, logits_from_signed_distance

__all__ = [
    "attach_uniform_field",
    "box_grain",
    "faceted_grain",
    "grain_from_sdf",
    "plate_grain",
    "regular_grid",
    "sphere_grain",
]


def regular_grid(shape, spacing_um: float, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Voxel-centre positions ``(N, 3)`` of a regular grid centred on the origin.

    Identical by construction to the grid
    :func:`midas_dfxm.io.make_uniform_field` builds, so a ``GrainShape`` and a
    ``DeformationField`` made at the same ``(shape, spacing_um)`` share voxels
    one-for-one. That equality is locked in ``tests/test_dfxm_contract.py``
    rather than assumed.
    """
    axes = []
    for n in shape:
        c = (n - 1) / 2.0
        axes.append((torch.arange(n, device=device, dtype=dtype) - c) * spacing_um)
    gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)


def _auto_shape(extent_um, spacing_um: float, padding_um: float) -> tuple:
    """Smallest odd-ish grid covering ``extent_um`` (half-widths) plus padding."""
    out = []
    for e in extent_um:
        n = int(2 * (float(e) + padding_um) / spacing_um) + 1
        out.append(max(n, 3))
    return tuple(out)


def grain_from_sdf(
    sdf_fn,
    *,
    shape,
    spacing_um: float = 1.0,
    edge_width_um: float = None,
    device=None,
    dtype=torch.float64,
) -> GrainShape:
    """Build a :class:`~midas_dct_tt.grain.GrainShape` from a signed-distance function.

    ``sdf_fn`` maps positions ``(N, 3)`` -> signed distance ``(N,)`` in
    micrometers, negative inside. ``edge_width_um`` defaults to half the voxel
    spacing: sharp enough that the boundary is one voxel wide, soft enough that
    the shape gradient survives.
    """
    if edge_width_um is None:
        edge_width_um = 0.5 * spacing_um
    positions = regular_grid(shape, spacing_um, device=device, dtype=dtype)
    sdf = sdf_fn(positions)
    return GrainShape(
        positions=positions,
        logits=logits_from_signed_distance(sdf, edge_width_um),
        spacing_um=spacing_um,
        shape=tuple(shape),
    )


def sphere_grain(
    radius_um=5.0,
    *,
    spacing_um: float = 1.0,
    padding_um: float = 2.0,
    edge_width_um: float = None,
    shape=None,
    device=None,
    dtype=torch.float64,
) -> GrainShape:
    """Sphere of radius ``radius_um``, centred at the origin.

    Differentiable in ``radius_um`` when it is passed as a tensor -- the planted
    parameter a shape recovery is scored against.
    """
    r = radius_um if isinstance(radius_um, torch.Tensor) else torch.as_tensor(
        float(radius_um), device=device, dtype=dtype
    )
    if shape is None:
        e = float(r.detach())
        shape = _auto_shape((e, e, e), spacing_um, padding_um)

    def sdf(p):
        return torch.linalg.vector_norm(p, dim=-1) - r.to(device=p.device, dtype=p.dtype)

    return grain_from_sdf(
        sdf, shape=shape, spacing_um=spacing_um, edge_width_um=edge_width_um,
        device=device, dtype=dtype,
    )


def box_grain(
    size_um=(8.0, 8.0, 8.0),
    *,
    spacing_um: float = 1.0,
    padding_um: float = 2.0,
    edge_width_um: float = None,
    shape=None,
    device=None,
    dtype=torch.float64,
) -> GrainShape:
    """Axis-aligned rectangular box of full side lengths ``size_um``, centred at 0."""
    half = torch.as_tensor([float(s) / 2.0 for s in size_um], device=device, dtype=dtype)
    if shape is None:
        shape = _auto_shape(half.tolist(), spacing_um, padding_um)

    def sdf(p):
        q = torch.abs(p) - half.to(device=p.device, dtype=p.dtype)
        outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
        return outside + inside

    return grain_from_sdf(
        sdf, shape=shape, spacing_um=spacing_um, edge_width_um=edge_width_um,
        device=device, dtype=dtype,
    )


def plate_grain(
    size_um=(10.0, 10.0, 1.5),
    *,
    normal=None,
    **kwargs,
) -> GrainShape:
    """Thin plate: a box with one short axis (default: thin along sample ``z``).

    The stress case for laminography. Orient the thin axis along the TT rotation
    axis and the plate sits squarely in the missing cone; orient it transverse
    and it reconstructs well. ``normal`` is accepted for symmetry with future
    arbitrary-orientation plates and must currently be ``None`` (axis-aligned).
    """
    if normal is not None:
        raise NotImplementedError(
            "non-axis-aligned plates: rotate the grain positions instead, or use "
            "faceted_grain with two opposed normals"
        )
    return box_grain(size_um, **kwargs)


def faceted_grain(
    normals,
    distances_um,
    *,
    spacing_um: float = 1.0,
    padding_um: float = 2.0,
    edge_width_um: float = None,
    shape=None,
    device=None,
    dtype=torch.float64,
) -> GrainShape:
    """Convex polyhedron: intersection of half-spaces ``n_i . r <= d_i``.

    ``normals`` is ``(M, 3)`` (normalised internally) and ``distances_um`` is
    ``(M,)``. The SDF is ``max_i (n_i . r - d_i)``, exact on the faces and a
    lower bound near edges/corners -- adequate for a soft occupancy boundary,
    and differentiable in ``distances_um`` (each facet's position is a free
    shape parameter for Phase 2 recovery).
    """
    n = torch.as_tensor(normals, device=device, dtype=dtype)
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    d = torch.as_tensor(distances_um, device=device, dtype=dtype)
    if n.shape[0] != d.shape[0]:
        raise ValueError(f"normals ({n.shape[0]}) and distances ({d.shape[0]}) disagree")
    if shape is None:
        e = float(torch.max(torch.abs(d)).detach())
        shape = _auto_shape((e, e, e), spacing_um, padding_um)

    def sdf(p):
        nn = n.to(device=p.device, dtype=p.dtype)
        dd = d.to(device=p.device, dtype=p.dtype)
        return torch.max(p @ nn.transpose(-1, -2) - dd, dim=-1).values

    return grain_from_sdf(
        sdf, shape=shape, spacing_um=spacing_um, edge_width_um=edge_width_um,
        device=device, dtype=dtype,
    )


def attach_uniform_field(
    grain: GrainShape,
    *,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    orientation=None,
) -> GrainShape:
    """Attach a perfect-crystal ``F = I`` field on the grain's own voxel grid.

    Thin wrapper over :func:`midas_dfxm.io.make_uniform_field` -- the reference
    lattice, reciprocal basis, and ``F`` container all come from ``midas_dfxm``.
    Requires a regular grid (``grain.shape is not None``) and verifies the two
    grids coincide before pairing them, because a silent half-voxel offset
    between ``chi(r)`` and ``F(r)`` would show up only as a small, plausible,
    entirely spurious strain.
    """
    if grain.shape is None:
        raise ValueError("attach_uniform_field needs a regular grid (grain.shape is None)")
    field = make_uniform_field(
        shape=grain.shape,
        spacing_um=grain.spacing_um if not isinstance(grain.spacing_um, (tuple, list)) else grain.spacing_um[0],
        lattice_params=lattice_params,
        orientation=orientation,
        device=grain.positions.device,
        dtype=grain.positions.dtype,
    )
    if not torch.allclose(field.positions, grain.positions, atol=1e-9):
        raise AssertionError(
            "midas_dfxm grid does not coincide with the grain grid -- the shared-voxel "
            "assumption in attach_uniform_field is broken (check spacing/shape conventions)"
        )
    return grain.with_field(field)
