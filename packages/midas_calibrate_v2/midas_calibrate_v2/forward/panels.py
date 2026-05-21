"""5-DOF per-panel rigid-body transforms for multi-tile detectors.

Per panel ``k`` we model:
  - δy_k, δz_k    in-plane translation (px)
  - δθ_k          in-plane rotation about the panel center (deg)
  - δLsd_k        per-panel sample-to-detector distance offset (μm)
  - δp2_k         per-panel isotropic radial distortion offset (added to p₂)

The panel grid is specified by ``(N_panels_y, N_panels_z)`` with module size
``(S_y, S_z)`` and gap arrays.  Panels are looked up per-pixel via the panel
index mask.

This module is differentiable in the per-panel parameters; the panel layout
itself is integer-valued and not refined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class PanelLayout:
    """Integer panel layout — fixed metadata, not refined."""

    n_panels_y: int
    n_panels_z: int
    panel_size_y: int   # px
    panel_size_z: int
    gaps_y: Tuple[int, ...] = ()  # gap widths between adjacent panels in y
    gaps_z: Tuple[int, ...] = ()
    panel_centers_y: Optional[torch.Tensor] = None  # [n_y, n_z] center y (px)
    panel_centers_z: Optional[torch.Tensor] = None
    panel_index_mask: Optional[torch.Tensor] = None  # [H, W] long; -1 in gaps

    def n_panels(self) -> int:
        return self.n_panels_y * self.n_panels_z

    @classmethod
    def regular(cls, n_y: int, n_z: int, sy: int, sz: int,
                gap_y=0, gap_z=0) -> "PanelLayout":
        """Construct a grid with optionally non-uniform gap widths.

        ``gap_y`` and ``gap_z`` may be either a single int (uniform gap) or a
        sequence of length ``n_y - 1`` / ``n_z - 1`` (non-uniform).  Pilatus
        6M, for example, has ``gap_y = (1, 7, 1, 7, 1)`` (chip-pair pattern).

        The panel centers and per-pixel index mask are pre-computed for fast
        per-pixel lookup at forward time.
        """
        if isinstance(gap_y, (int, float)):
            gaps_y = tuple([int(gap_y)] * max(n_y - 1, 0))
        else:
            gaps_y = tuple(int(g) for g in gap_y)
            if len(gaps_y) != max(n_y - 1, 0):
                raise ValueError(f"gap_y must have length {n_y - 1}, got {len(gaps_y)}")
        if isinstance(gap_z, (int, float)):
            gaps_z = tuple([int(gap_z)] * max(n_z - 1, 0))
        else:
            gaps_z = tuple(int(g) for g in gap_z)
            if len(gaps_z) != max(n_z - 1, 0):
                raise ValueError(f"gap_z must have length {n_z - 1}, got {len(gaps_z)}")

        # Cumulative panel start positions (in px) along each axis.
        y_starts = [0]
        for i in range(n_y - 1):
            y_starts.append(y_starts[-1] + sy + gaps_y[i])
        z_starts = [0]
        for j in range(n_z - 1):
            z_starts.append(z_starts[-1] + sz + gaps_z[j])

        cy = torch.tensor([y_starts[i] + 0.5 * sy for i in range(n_y)], dtype=torch.float64)
        cz = torch.tensor([z_starts[j] + 0.5 * sz for j in range(n_z)], dtype=torch.float64)
        cy_grid = cy[:, None].expand(n_y, n_z).contiguous()
        cz_grid = cz[None, :].expand(n_y, n_z).contiguous()

        H = (y_starts[-1] + sy) if n_y > 0 else 0
        W = (z_starts[-1] + sz) if n_z > 0 else 0
        mask = torch.full((H, W), -1, dtype=torch.long)
        for i in range(n_y):
            y0 = y_starts[i]
            y1 = y0 + sy
            for j in range(n_z):
                z0 = z_starts[j]
                z1 = z0 + sz
                mask[y0:y1, z0:z1] = i * n_z + j
        return cls(
            n_panels_y=n_y, n_panels_z=n_z,
            panel_size_y=sy, panel_size_z=sz,
            gaps_y=gaps_y, gaps_z=gaps_z,
            panel_centers_y=cy_grid,
            panel_centers_z=cz_grid,
            panel_index_mask=mask,
        )


def panel_idx_for_points(
    layout: PanelLayout,
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
) -> torch.Tensor:
    """Look up the integer panel index for each fitted (Y, Z) point.

    Uses ``layout.panel_index_mask`` (shape ``[NY, NZ]``, ``-1`` in gaps),
    which :meth:`PanelLayout.regular` builds in pixel space.  Points landing
    in a module gap, or outside the mask bounds, are returned as ``-1`` so the
    forward model leaves them untouched (see :func:`apply_panel_shifts`).

    Returns a long tensor of panel indices, same shape as ``Y_pix``.
    """
    if layout.panel_index_mask is None:
        raise ValueError("layout has no panel_index_mask; build via PanelLayout.regular")
    mask = layout.panel_index_mask
    H, W = mask.shape
    yi = torch.round(Y_pix).long()
    zi = torch.round(Z_pix).long()
    in_bounds = (yi >= 0) & (yi < H) & (zi >= 0) & (zi < W)
    yi_c = yi.clamp(0, H - 1)
    zi_c = zi.clamp(0, W - 1)
    idx = mask.to(Y_pix.device)[yi_c, zi_c]
    idx = torch.where(in_bounds, idx, torch.full_like(idx, -1))
    return idx


def apply_panel_shifts(
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
    panel_idx: torch.Tensor,         # [...] long; -1 for pixels in a gap
    layout: PanelLayout,
    delta_yz: torch.Tensor,          # [N, 2] (δy, δz) per panel
    delta_theta_deg: torch.Tensor,   # [N] in-plane rotation (deg)
    fix_panel_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply per-panel rigid-body shift (translation + in-plane rotation).

    Parameters
    ----------
    Y_pix, Z_pix : tensors of pixel coordinates.
    panel_idx : long tensor of panel indices (same shape as Y_pix), with -1 in
        gaps.  Pixels in gaps are returned untouched.
    layout : :class:`PanelLayout`.
    delta_yz, delta_theta_deg : refinable per-panel corrections.
    fix_panel_id : the reference panel whose offset is held at 0.

    Returns
    -------
    (Y_pix_new, Z_pix_new) — corrected pixel coordinates.
    """
    deg2rad = 0.017453292519943295
    n_panels = layout.n_panels()
    if delta_yz.shape != (n_panels, 2):
        raise ValueError(f"delta_yz must be [{n_panels}, 2], got {tuple(delta_yz.shape)}")
    if delta_theta_deg.shape != (n_panels,):
        raise ValueError(f"delta_theta must be [{n_panels}], got {tuple(delta_theta_deg.shape)}")

    # Gauge fix: ``fix_panel_id >= 0`` zeroes that panel's deltas in the
    # forward (the v1-compatible convention).  ``fix_panel_id < 0``
    # disables the mask entirely, so the caller is expected to apply the
    # softer Σ panel = 0 zero-sum penalty (Wright 2022 §3.2) — see
    # :func:`midas_calibrate_v2.compat.from_v1.add_panel_zero_sum_constraint`.
    if fix_panel_id is not None and fix_panel_id >= 0:
        mask_fix = torch.zeros(n_panels, dtype=delta_yz.dtype, device=delta_yz.device)
        mask_fix[fix_panel_id] = 1.0
        delta_yz = delta_yz * (1.0 - mask_fix.unsqueeze(-1))
        delta_theta_deg = delta_theta_deg * (1.0 - mask_fix)

    # Look up per-pixel corrections.  Use a sentinel index 0 for gap pixels;
    # we'll mask the result back out at the end.
    safe_idx = torch.where(panel_idx >= 0, panel_idx, torch.zeros_like(panel_idx))
    safe_idx = safe_idx.to(torch.long)

    cy_grid = layout.panel_centers_y.to(delta_yz.device, delta_yz.dtype).reshape(-1)
    cz_grid = layout.panel_centers_z.to(delta_yz.device, delta_yz.dtype).reshape(-1)

    cy = cy_grid[safe_idx]
    cz = cz_grid[safe_idx]
    dy = delta_yz[safe_idx, 0]
    dz = delta_yz[safe_idx, 1]
    dth = delta_theta_deg[safe_idx] * deg2rad

    cos_t = torch.cos(dth)
    sin_t = torch.sin(dth)

    Y_loc = Y_pix - cy
    Z_loc = Z_pix - cz
    Y_rot = cos_t * Y_loc - sin_t * Z_loc
    Z_rot = sin_t * Y_loc + cos_t * Z_loc
    Y_new = cy + Y_rot + dy
    Z_new = cz + Z_rot + dz

    # In-gap pixels: return original coordinates unchanged.
    in_gap = panel_idx < 0
    Y_out = torch.where(in_gap, Y_pix, Y_new)
    Z_out = torch.where(in_gap, Z_pix, Z_new)
    return Y_out, Z_out


def invert_panel_shifts(
    Y_pix_post: torch.Tensor,
    Z_pix_post: torch.Tensor,
    panel_idx: torch.Tensor,
    layout: PanelLayout,
    delta_yz: torch.Tensor,
    delta_theta_deg: torch.Tensor,
    fix_panel_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`apply_panel_shifts`.

    Given the *post-shift* pixel coords (the ones that, when fed through
    ``apply_panel_shifts`` then projection, give the observed (R, η)),
    recover the *pre-shift* coords used by callers that don't apply
    panels.  Used by ``_bake_fits_to_dataset`` so the (Y, Z) output of the
    panel-unaware ``invert_REta_to_pixel_batch`` is consistent with the
    panel-aware forward model in the M-step.
    """
    deg2rad = 0.017453292519943295
    n_panels = layout.n_panels()
    if fix_panel_id is not None and fix_panel_id >= 0:
        mask_fix = torch.zeros(n_panels, dtype=delta_yz.dtype, device=delta_yz.device)
        mask_fix[fix_panel_id] = 1.0
        delta_yz = delta_yz * (1.0 - mask_fix.unsqueeze(-1))
        delta_theta_deg = delta_theta_deg * (1.0 - mask_fix)

    safe_idx = torch.where(panel_idx >= 0, panel_idx,
                            torch.zeros_like(panel_idx)).to(torch.long)
    cy_grid = layout.panel_centers_y.to(delta_yz.device, delta_yz.dtype).reshape(-1)
    cz_grid = layout.panel_centers_z.to(delta_yz.device, delta_yz.dtype).reshape(-1)
    cy = cy_grid[safe_idx]
    cz = cz_grid[safe_idx]
    dy = delta_yz[safe_idx, 0]
    dz = delta_yz[safe_idx, 1]
    dth = delta_theta_deg[safe_idx] * deg2rad
    cos_t = torch.cos(dth)
    sin_t = torch.sin(dth)

    # Forward: Y_new - cy = cos·(Y - cy) - sin·(Z - cz) + dy
    #          Z_new - cz = sin·(Y - cy) + cos·(Z - cz) + dz
    # Invert with cos²+sin²=1 (orthogonal rot):
    Yn = Y_pix_post - cy - dy
    Zn = Z_pix_post - cz - dz
    Y_orig = cy + cos_t * Yn + sin_t * Zn
    Z_orig = cz - sin_t * Yn + cos_t * Zn

    in_gap = panel_idx < 0
    Y_out = torch.where(in_gap, Y_pix_post, Y_orig)
    Z_out = torch.where(in_gap, Z_pix_post, Z_orig)
    return Y_out, Z_out


def per_panel_lsd_offset(
    panel_idx: torch.Tensor,
    delta_lsd: torch.Tensor,         # [N] per-panel Lsd correction (μm)
    fix_panel_id: int = 0,
) -> torch.Tensor:
    """Look up per-panel Lsd offset for each pixel.  Reference panel is
    held at zero when ``fix_panel_id >= 0``; ``fix_panel_id < 0``
    disables the mask (use a Σ panel = 0 zero-sum penalty instead).
    """
    n_panels = delta_lsd.shape[0]
    if fix_panel_id is not None and fix_panel_id >= 0:
        mask_fix = torch.zeros(n_panels, dtype=delta_lsd.dtype, device=delta_lsd.device)
        mask_fix[fix_panel_id] = 1.0
        delta_lsd = delta_lsd * (1.0 - mask_fix)

    safe_idx = torch.where(panel_idx >= 0, panel_idx, torch.zeros_like(panel_idx)).to(torch.long)
    out = delta_lsd[safe_idx]
    out = torch.where(panel_idx < 0, torch.zeros_like(out), out)
    return out


def per_panel_p2_offset(
    panel_idx: torch.Tensor,
    delta_p2: torch.Tensor,
    fix_panel_id: int = 0,
) -> torch.Tensor:
    """Same pattern as Lsd offset, for the per-panel p₂ correction."""
    return per_panel_lsd_offset(panel_idx, delta_p2, fix_panel_id)


__all__ = ["PanelLayout", "panel_idx_for_points", "apply_panel_shifts",
           "invert_panel_shifts", "per_panel_lsd_offset", "per_panel_p2_offset"]
