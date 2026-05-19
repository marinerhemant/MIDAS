"""SampleGrid — voxel grid + grain map + sample mask packaged as torch tensors.

All fields live on a single device (CPU / CUDA / MPS).  Continuous fields
(voxel positions, voxel size) are differentiable; integer / boolean fields
(grain map, sample mask) are not.

The grid is *agnostic* to PF vs FF: callers attach their own grain map and
sample mask.  Beam-path attribution lives in
:mod:`midas_transforms.geometry.beam` and is called by the forward model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = ["SampleGrid"]


@dataclass
class SampleGrid:
    """Voxel grid with grain assignment and sample mask.

    Parameters
    ----------
    voxel_positions : (N, 3) float — lab-frame coordinates (µm).  Differentiable.
    voxel_size_um   : 0-d float — cubic voxel edge length (µm).  Differentiable.
    sample_mask     : (N,) bool — True where the voxel is inside the sample.
    grain_map       : (N,) int  — grain id per voxel; ``-1`` = unassigned.

    Optional regular-grid topology (used by :mod:`midas_transforms.geometry.absorption`):

    grid_origin_um  : (3,) float — lab-frame position of voxel ``(0, 0, 0)``
                       (the voxel's *center*, not its corner).
    grid_shape      : (nx, ny, nz) int tuple — grid dimensions in C-order so that
                       ``flat_idx = i + nx * j + nx*ny * k``.

    If both are set, callers can treat the grid as topologically regular and
    use the ray-trace API.  If either is ``None``, only flat-list operations
    are available.
    """
    voxel_positions: "torch.Tensor"
    voxel_size_um:   "torch.Tensor"
    sample_mask:     "torch.Tensor"
    grain_map:       "torch.Tensor"
    grid_origin_um:  Optional["torch.Tensor"] = None
    grid_shape:      Optional[tuple] = None

    # -------------------------------------------------------------- properties

    @property
    def device(self) -> "torch.device":
        return self.voxel_positions.device

    @property
    def dtype(self) -> "torch.dtype":
        return self.voxel_positions.dtype

    @property
    def n_voxels(self) -> int:
        return int(self.voxel_positions.shape[0])

    # ----------------------------------------------------------------- queries

    def voxels_in_grain(self, g: int) -> "torch.Tensor":
        """(n_g,) int tensor of voxel indices in grain ``g`` AND in the sample."""
        import torch
        mask = (self.grain_map == int(g)) & self.sample_mask
        return mask.nonzero(as_tuple=False).squeeze(-1)

    def voxels_in_sample(self) -> "torch.Tensor":
        """(n_in,) int tensor of voxel indices where ``sample_mask`` is True."""
        return self.sample_mask.nonzero(as_tuple=False).squeeze(-1)

    def grain_ids(self) -> "torch.Tensor":
        """Unique grain ids (>= 0) present in ``grain_map`` and the sample."""
        import torch
        valid = self.sample_mask & (self.grain_map >= 0)
        return torch.unique(self.grain_map[valid])

    # ----------------------------------------------------------------- moves

    def to(
        self,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> "SampleGrid":
        """Move all tensors to ``device``/``dtype`` (only float fields take dtype)."""
        kw_float = {"device": device}
        if dtype is not None:
            kw_float["dtype"] = dtype
        new_origin = (
            self.grid_origin_um.to(**kw_float)
            if self.grid_origin_um is not None else None
        )
        return replace(
            self,
            voxel_positions=self.voxel_positions.to(**kw_float),
            voxel_size_um=self.voxel_size_um.to(**kw_float),
            sample_mask=self.sample_mask.to(device=device),
            grain_map=self.grain_map.to(device=device),
            grid_origin_um=new_origin,
        )

    # --------------------------------------------------------------- factories

    @classmethod
    def from_grain_centroids(
        cls,
        centroids_um,
        grain_ids=None,
        voxel_size_um: float = 1.0,
        *,
        sample_mask=None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> "SampleGrid":
        """Compact-grain FF factory: one "voxel" per grain at its centroid.

        Each grain is treated as a point at its centroid (typical FF mode
        when no tomographic mask is available).  ``voxel_size_um`` then
        acts as an effective grain extent for beam-profile fraction
        calculations — for compact-grain FF with ``scan_axis="none"``
        this is irrelevant, so leave at 1 µm.

        Parameters
        ----------
        centroids_um : (G, 3) array-like — grain centroid positions (µm).
        grain_ids    : (G,) array-like of int.  Defaults to 0..G-1.
        sample_mask  : (G,) array-like of bool.  Defaults to all-True.
        """
        import torch

        dt = dtype or torch.float64
        vp = torch.as_tensor(centroids_um, dtype=dt, device=device)
        if vp.ndim != 2 or vp.shape[1] != 3:
            raise ValueError(
                f"centroids_um must have shape (G, 3); got {tuple(vp.shape)}"
            )
        G = vp.shape[0]
        if grain_ids is None:
            gm = torch.arange(G, dtype=torch.int64, device=device)
        else:
            gm = torch.as_tensor(grain_ids, dtype=torch.int64, device=device)
            if gm.shape != (G,):
                raise ValueError(
                    f"grain_ids must have shape ({G},); got {tuple(gm.shape)}"
                )
        if sample_mask is None:
            sm = torch.ones(G, dtype=torch.bool, device=device)
        else:
            sm = torch.as_tensor(sample_mask, dtype=torch.bool, device=device)
            if sm.shape != (G,):
                raise ValueError(
                    f"sample_mask must have shape ({G},); got {tuple(sm.shape)}"
                )
        vs = torch.as_tensor(float(voxel_size_um), dtype=dt, device=device)
        return cls(voxel_positions=vp, voxel_size_um=vs, sample_mask=sm, grain_map=gm)

    @classmethod
    def from_regular_grid(
        cls,
        grid_origin_um,
        grid_shape: tuple,
        voxel_size_um: float,
        grain_map=None,
        sample_mask=None,
        *,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> "SampleGrid":
        """Build a :class:`SampleGrid` with an explicit regular topology.

        Voxel centers are placed at ``grid_origin + (i, j, k) * voxel_size`` for
        ``i ∈ [0, nx)``, ``j ∈ [0, ny)``, ``k ∈ [0, nz)`` in C-order.
        """
        import torch

        dt = dtype or torch.float64
        nx, ny, nz = (int(s) for s in grid_shape)
        n = nx * ny * nz
        # Build voxel centers (C-order: i fastest, then j, then k)
        ii, jj, kk = torch.meshgrid(
            torch.arange(nx, dtype=dt, device=device),
            torch.arange(ny, dtype=dt, device=device),
            torch.arange(nz, dtype=dt, device=device),
            indexing="ij",
        )
        # Reshape to (n, 3) in C-order so flat_idx = i + nx*j + nx*ny*k
        origin = torch.as_tensor(grid_origin_um, dtype=dt, device=device)
        # k slow → j → i fast: reshape with permutation to match
        vp = torch.stack([ii, jj, kk], dim=-1).reshape(-1, 3)
        # We built with ij-meshgrid; reorder so flat_idx = i + nx*j + nx*ny*k.
        # Stride-1 axis is k under ij; we want stride-1 to be i.  Permute.
        vp_3d = torch.stack([ii, jj, kk], dim=-1)        # (nx, ny, nz, 3)
        vp = vp_3d.permute(2, 1, 0, 3).reshape(-1, 3)    # k, j, i order → flat = i + nx*j + nx*ny*k
        vp = vp * voxel_size_um + origin

        if grain_map is None:
            gm = torch.full((n,), -1, dtype=torch.int64, device=device)
        else:
            gm = torch.as_tensor(grain_map, dtype=torch.int64, device=device).reshape(-1)
            if gm.numel() != n:
                raise ValueError(f"grain_map size {gm.numel()} != grid {n}")
        if sample_mask is None:
            sm = torch.ones(n, dtype=torch.bool, device=device)
        else:
            sm = torch.as_tensor(sample_mask, dtype=torch.bool, device=device).reshape(-1)
            if sm.numel() != n:
                raise ValueError(f"sample_mask size {sm.numel()} != grid {n}")
        return cls(
            voxel_positions=vp,
            voxel_size_um=torch.as_tensor(float(voxel_size_um), dtype=dt, device=device),
            sample_mask=sm,
            grain_map=gm,
            grid_origin_um=origin,
            grid_shape=(nx, ny, nz),
        )

    @classmethod
    def from_arrays(
        cls,
        voxel_positions,
        voxel_size_um: float,
        grain_map=None,
        sample_mask=None,
        *,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> "SampleGrid":
        """Build from numpy arrays / Python sequences.

        If ``grain_map`` is None, all voxels are unassigned (``-1``).
        If ``sample_mask`` is None, the entire grid is treated as sample
        (the Wenxi-pf default, per the V-map plan).
        """
        import torch

        dt = dtype or torch.float64
        vp = torch.as_tensor(voxel_positions, dtype=dt, device=device)
        if vp.ndim != 2 or vp.shape[1] != 3:
            raise ValueError(f"voxel_positions must have shape (N, 3); got {tuple(vp.shape)}")
        n = vp.shape[0]
        vs = torch.as_tensor(float(voxel_size_um), dtype=dt, device=device)
        if grain_map is None:
            gm = torch.full((n,), -1, dtype=torch.int64, device=device)
        else:
            gm = torch.as_tensor(grain_map, dtype=torch.int64, device=device)
            if gm.shape != (n,):
                raise ValueError(f"grain_map must have shape ({n},); got {tuple(gm.shape)}")
        if sample_mask is None:
            sm = torch.ones(n, dtype=torch.bool, device=device)
        else:
            sm = torch.as_tensor(sample_mask, dtype=torch.bool, device=device)
            if sm.shape != (n,):
                raise ValueError(f"sample_mask must have shape ({n},); got {tuple(sm.shape)}")
        return cls(voxel_positions=vp, voxel_size_um=vs, sample_mask=sm, grain_map=gm)
