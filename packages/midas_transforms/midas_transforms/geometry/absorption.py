"""Differentiable absorption ray-trace.

For each ray of origin :math:`\\mathbf{r}_0` and direction :math:`\\hat{\\mathbf{d}}`,
the path length through the binary sample mask is

.. math::

    L(\\hat{\\mathbf{d}}, \\mathbf{r}_0) = \\sum_k \\Delta t_k \\cdot m_k

where :math:`\\Delta t_k` is the length of the :math:`k`-th segment between
consecutive voxel-plane crossings and :math:`m_k \\in \\{0, 1\\}` is the mask
value at the segment midpoint.  Segment lengths are smooth functions of the
direction (parametric plane crossings), so the integral is differentiable
in :math:`\\hat{\\mathbf{d}}` *almost everywhere* — discontinuous only at the
measure-zero set of directions for which the ray exits one mask cell and
enters another simultaneously.  In practice this is fine for both
forward-prediction and refinement contexts.

The :math:`\\mu` factor in
``absorption_factor = exp(-\\mu (L_{\\rm in} + L_{\\rm out}))`` is smoothly
differentiable end-to-end.

This module is **torch-native** and **device-portable** (CPU / CUDA / MPS).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from .sample import SampleGrid


__all__ = ["path_length_in_sample", "absorption_factor"]


_EPS = 1e-30


def path_length_in_sample(
    grid_origin_um: "torch.Tensor",   # (3,) center of voxel (0,0,0)
    grid_shape: Tuple[int, int, int], # (nx, ny, nz)
    voxel_size_um: "torch.Tensor",     # 0-d scalar
    sample_mask_flat: "torch.Tensor",  # (nx*ny*nz,) bool, C-order with i fastest
    origin_um: "torch.Tensor",         # (N, 3) — ray starting points (lab µm)
    direction: "torch.Tensor",         # (N, 3) — UNIT vectors in lab frame
    *,
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":                   # (N,) — path length in µm
    """Path length along each ray inside the binary sample mask.

    Vectorized Siddon: enumerate all parametric plane crossings, sort,
    pick segment midpoints, gather mask values, integrate ``Σ Δt · m_k``.

    Notes
    -----
    * Voxel boundaries are at ``grid_origin + (i ∓ 0.5) * voxel_size`` for the
      i-th cell — voxel positions are *centers*, not corners.
    * Rays that miss the bounding box return 0.
    * Direction does NOT need to be of unit length; we trace the parametric
      ray ``r(t) = r₀ + t · direction`` for ``t ∈ [0, +∞)``.  If you want path
      length in physical units, pass a unit direction.
    """
    import torch

    dt = dtype or origin_um.dtype
    dev = device or origin_um.device

    N = origin_um.shape[0]
    nx, ny, nz = grid_shape

    # Bounding box of sample grid in lab space — voxel centers are at
    # `grid_origin + (i, j, k) * vs`, so corners extend half a voxel further:
    half = voxel_size_um * 0.5
    grid_min = grid_origin_um - half
    nxyz = torch.tensor([nx, ny, nz], dtype=dt, device=dev)
    grid_max = grid_origin_um + nxyz * voxel_size_um - half

    # Safe inverse direction: a near-zero component would mean the ray is parallel
    # to a face -> intersection at ±∞; the t-clamp below handles that gracefully.
    safe_dir = torch.where(
        direction.abs() < _EPS,
        torch.full_like(direction, _EPS),
        direction,
    )
    inv_dir = 1.0 / safe_dir                       # (N, 3)

    # Bounding-box intersection times per axis
    t1 = (grid_min.unsqueeze(0) - origin_um) * inv_dir
    t2 = (grid_max.unsqueeze(0) - origin_um) * inv_dir
    t_axis_lo = torch.minimum(t1, t2)              # (N, 3)
    t_axis_hi = torch.maximum(t1, t2)              # (N, 3)
    t_enter = t_axis_lo.max(dim=1).values           # (N,)
    t_exit  = t_axis_hi.min(dim=1).values           # (N,)
    t_enter = torch.clamp(t_enter, min=0.0)         # only forward ray (t > 0)
    misses = t_enter >= t_exit                      # (N,) bool

    # All plane-crossing times.  For axis a, planes are at corner positions
    # `grid_min[a] + i * vs` for i = 0..n[a].
    def _axis_t(a: int, n_planes: int) -> "torch.Tensor":
        idx = torch.arange(n_planes, dtype=dt, device=dev)
        pos = grid_min[a] + idx * voxel_size_um    # (n_planes,)
        return (pos.unsqueeze(0) - origin_um[:, a:a+1]) * inv_dir[:, a:a+1]   # (N, n_planes)

    t_x = _axis_t(0, nx + 1)
    t_y = _axis_t(1, ny + 1)
    t_z = _axis_t(2, nz + 1)

    # Concatenate plane crossings + the bbox entry/exit (so segments span [t_enter, t_exit])
    t_all = torch.cat([
        t_x, t_y, t_z,
        t_enter.unsqueeze(1), t_exit.unsqueeze(1),
    ], dim=1)                                      # (N, K)

    # Clip to [t_enter, t_exit].  Out-of-range crossings collapse to the endpoint
    # so their segment lengths become 0.
    t_clamped = torch.maximum(
        torch.minimum(t_all, t_exit.unsqueeze(1)),
        t_enter.unsqueeze(1),
    )

    # Sort ascending — gives the segment boundaries in increasing time.
    t_sorted, _ = torch.sort(t_clamped, dim=1)
    seg_lo = t_sorted[:, :-1]
    seg_hi = t_sorted[:, 1:]
    seg_len = seg_hi - seg_lo                       # (N, K-1) — many are zero
    seg_mid = 0.5 * (seg_lo + seg_hi)               # (N, K-1)

    # Position at segment midpoint -> voxel index (i, j, k)
    pos_mid = (
        origin_um.unsqueeze(1)
        + seg_mid.unsqueeze(-1) * direction.unsqueeze(1)
    )                                               # (N, K-1, 3)
    vox_xyz = ((pos_mid - grid_min.unsqueeze(0).unsqueeze(0)) / voxel_size_um).long()
    # Clamp to valid range (handles segments with seg_len = 0 at the bbox edges)
    vox_xyz[..., 0] = torch.clamp(vox_xyz[..., 0], 0, nx - 1)
    vox_xyz[..., 1] = torch.clamp(vox_xyz[..., 1], 0, ny - 1)
    vox_xyz[..., 2] = torch.clamp(vox_xyz[..., 2], 0, nz - 1)
    flat_idx = (
        vox_xyz[..., 0]
        + nx * vox_xyz[..., 1]
        + nx * ny * vox_xyz[..., 2]
    )                                               # (N, K-1)

    mask_vals = sample_mask_flat[flat_idx].to(seg_len.dtype)    # (N, K-1)
    path = (seg_len * mask_vals).sum(dim=1)         # (N,)
    return torch.where(misses, torch.zeros_like(path), path)


def absorption_factor(
    sample_grid: "SampleGrid",
    voxel_idx: "torch.Tensor",                 # (N,) int — origin voxel per ray
    incident_dirs: "torch.Tensor",              # (N, 3) — unit vec, beam-→sample
    diffracted_dirs: "torch.Tensor",            # (N, 3) — unit vec, sample-→detector
    mu_per_cm: "torch.Tensor",                  # 0-d or (N,) — μ in cm⁻¹
    *,
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":                            # (N,) — A ∈ (0, 1]
    """Absorption attenuation factor for incident + diffracted ray pairs.

    .. math::

        A = \\exp\\left(-\\mu \\cdot (L_{\\rm in} + L_{\\rm out})\\right)

    where :math:`L_{\\rm in}` is the path length from the sample entry point
    to the voxel, and :math:`L_{\\rm out}` is the path from the voxel to the
    sample exit along the diffracted direction.

    The ``sample_grid`` must have ``grid_origin_um`` and ``grid_shape`` set
    (i.e., built via :meth:`SampleGrid.from_regular_grid`).

    Differentiable in ``mu_per_cm`` and (almost everywhere) in
    ``incident_dirs`` / ``diffracted_dirs``.  Integer ``voxel_idx`` is not
    differentiable.
    """
    if sample_grid.grid_origin_um is None or sample_grid.grid_shape is None:
        raise ValueError(
            "absorption_factor requires a regular-grid SampleGrid; "
            "build with SampleGrid.from_regular_grid()"
        )
    import torch

    dt = dtype or incident_dirs.dtype
    dev = device or incident_dirs.device

    # Ray origins at the voxel centers (mid-voxel approximation; for sub-voxel
    # accuracy the user can supersample but the difference is sub-percent for
    # typical HEDM voxel sizes <= 5 µm).
    origins = sample_grid.voxel_positions[voxel_idx].to(dtype=dt, device=dev)

    # In-path: trace from the voxel back along -incident (toward beam source)
    L_in = path_length_in_sample(
        sample_grid.grid_origin_um.to(dtype=dt, device=dev),
        sample_grid.grid_shape,
        sample_grid.voxel_size_um.to(dtype=dt, device=dev),
        sample_grid.sample_mask.to(device=dev),
        origins,
        -incident_dirs,
    )
    # Out-path: trace from the voxel along the diffracted direction toward exit
    L_out = path_length_in_sample(
        sample_grid.grid_origin_um.to(dtype=dt, device=dev),
        sample_grid.grid_shape,
        sample_grid.voxel_size_um.to(dtype=dt, device=dev),
        sample_grid.sample_mask.to(device=dev),
        origins,
        diffracted_dirs,
    )
    # Convert µm -> cm, multiply by μ [cm⁻¹]
    total_cm = (L_in + L_out) * 1.0e-4
    return torch.exp(-mu_per_cm * total_cm)
