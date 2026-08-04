"""Synthetic ground-truth generators and external-field adapters.

Item #3 of the post-Phase-5 roadmap.

Two jobs:

1. **Dense dislocation structures** beyond single defects — pile-ups and random
   ensembles at a target density — for the capability studies (#4) and to stress the
   inverse in the regime where manual g.b typing breaks down.
2. **External-field adapters** — build a :class:`DeformationField` directly from an
   externally-supplied deformation-gradient or strain array. This is the drop-in for
   a collaborator's realistic ``F(r)`` (and for any DDD / CP-FEM / MD field): the whole
   forward+inverse stack consumes it with zero rework, so the synthetic work is the
   validation harness, not a throwaway.

Everything is torch-differentiable and device-portable.
"""
from __future__ import annotations

import torch

from .dislocation import StrohDislocation, dislocation_deformation_field, fcc_slip_systems, stroh_dislocation
from .field import DeformationField


# --------------------------------------------------------------------------
# external-field adapters (the external / DDD / CP-FEM / MD drop-in)
# --------------------------------------------------------------------------
def field_from_deformation_gradient(
    F: torch.Tensor,
    positions: torch.Tensor,
    *,
    orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    shape=None,
) -> DeformationField:
    """Build a :class:`DeformationField` from a per-voxel deformation gradient ``F``.

    ``F`` is ``(N, 3, 3)``, ``positions`` ``(N, 3)`` (micrometers). The canonical
    adapter for an external field (collaborator, DDD, CP-FEM, MD). Differentiable.
    """
    device, dtype = F.device, F.dtype
    if orientation is None:
        orientation = torch.eye(3, device=device, dtype=dtype)
    else:
        orientation = torch.as_tensor(orientation, device=device, dtype=dtype)
    latc = torch.as_tensor(lattice_params, device=device, dtype=dtype)
    return DeformationField(positions=positions, F=F, reference_orientation=orientation,
                            lattice_params=latc, shape=shape)


def field_from_strain(
    strain: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotation: torch.Tensor | None = None,
    **kw,
) -> DeformationField:
    """Build a field from a per-voxel small-strain tensor (and optional rotation).

    ``strain`` is ``(N, 3, 3)`` symmetric (or ``(N, 6)`` Voigt
    ``[e11,e22,e33,e23,e13,e12]``); ``F = (I + eps)`` composed with an optional
    per-voxel rotation ``R`` as ``F = R @ (I + eps)``. Differentiable.
    """
    device, dtype = positions.device, positions.dtype
    strain = torch.as_tensor(strain, device=device, dtype=dtype)
    if strain.ndim == 2 and strain.shape[-1] == 6:
        e = strain
        eps = torch.zeros(e.shape[0], 3, 3, device=device, dtype=dtype)
        eps[:, 0, 0], eps[:, 1, 1], eps[:, 2, 2] = e[:, 0], e[:, 1], e[:, 2]
        eps[:, 1, 2] = eps[:, 2, 1] = e[:, 3]
        eps[:, 0, 2] = eps[:, 2, 0] = e[:, 4]
        eps[:, 0, 1] = eps[:, 1, 0] = e[:, 5]
    else:
        eps = strain
    eye = torch.eye(3, device=device, dtype=dtype)
    F = eye + eps
    if rotation is not None:
        F = torch.as_tensor(rotation, device=device, dtype=dtype) @ F
    return field_from_deformation_gradient(F, positions, **kw)


# --------------------------------------------------------------------------
# dense dislocation structures
# --------------------------------------------------------------------------
def dislocation_pileup(
    C6: torch.Tensor,
    *,
    burgers,
    slip_normal,
    n: int = 6,
    first_spacing_um: float = 1.0,
    growth: float = 1.4,
    along=(0.0, 1.0, 0.0),
    character: str = "edge",
    burgers_length_A: float = 2.556,
    core_radius_um: float = 0.3,
    crystal=None,
) -> list[StrohDislocation]:
    """A dislocation pile-up against a barrier: spacing grows geometrically from it.

    Same-sign dislocations at cumulative positions with spacing ``first_spacing_um *
    growth**k`` — the classic pile-up whose stress concentrates at the barrier.
    Returns a list of :class:`StrohDislocation`.
    """
    dtype, device = C6.dtype, C6.device
    axis = torch.as_tensor(along, dtype=dtype, device=device)
    axis = axis / torch.linalg.norm(axis)
    out, pos = [], 0.0
    for k in range(n):
        pos = pos + first_spacing_um * (growth ** k)
        out.append(stroh_dislocation(
            C6, burgers=burgers, slip_normal=slip_normal, character=character,
            burgers_length_A=burgers_length_A, core_position=tuple((pos * axis).tolist()),
            core_radius_um=core_radius_um, crystal=crystal))
    return out


def random_dislocation_ensemble(
    C6: torch.Tensor,
    *,
    bbox_um=((-10.0, 10.0), (-10.0, 10.0)),
    n: int = 20,
    seed: int = 0,
    slip_systems=None,
    characters=("edge", "screw"),
    burgers_length_A: float = 2.556,
    core_radius_um: float = 0.3,
    crystal=None,
) -> list[StrohDislocation]:
    """A random dislocation ensemble at a chosen count (a deformed-crystal proxy).

    Cores are drawn uniformly in the ``(x, y)`` bounding box; each dislocation gets a
    random slip system (from ``slip_systems`` or the FCC catalog), character, and sign.
    Deterministic given ``seed`` (a ``torch.Generator``; no global RNG state). The
    areal density is ``n / area`` — the knob the #4 capability study sweeps.
    """
    dtype, device = C6.dtype, C6.device
    systems = slip_systems if slip_systems is not None else fcc_slip_systems()
    g = torch.Generator().manual_seed(int(seed))
    (x0, x1), (y0, y1) = bbox_um
    xs = torch.rand(n, generator=g) * (x1 - x0) + x0
    ys = torch.rand(n, generator=g) * (y1 - y0) + y0
    sys_idx = torch.randint(len(systems), (n,), generator=g)
    chr_idx = torch.randint(len(characters), (n,), generator=g)
    signs = (torch.randint(2, (n,), generator=g) * 2 - 1)
    out = []
    for i in range(n):
        normal, b_dir = systems[int(sys_idx[i])]
        character = characters[int(chr_idx[i])]
        b = tuple(int(signs[i]) * v for v in b_dir)
        n_vec = torch.as_tensor(normal, dtype=dtype, device=device)
        b_vec = torch.as_tensor(b_dir, dtype=dtype, device=device)
        line = b_vec if character == "screw" else torch.linalg.cross(n_vec, b_vec)
        out.append(stroh_dislocation(
            C6, burgers=b, slip_normal=normal, line=line,
            burgers_length_A=burgers_length_A,
            core_position=(float(xs[i]), float(ys[i]), 0.0),
            core_radius_um=core_radius_um, crystal=crystal))
    return out


def ensemble_density_per_um2(dislocations, bbox_um) -> float:
    """Areal dislocation density (count / area) for a bounding box."""
    (x0, x1), (y0, y1) = bbox_um
    return len(dislocations) / ((x1 - x0) * (y1 - y0))
