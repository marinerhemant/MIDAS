"""Per-spot 3D patch construction.

Each measured spot is associated with a small (F, P, P) tensor centered on
an anchor location (anchor = grain-averaged predicted spot + Delta). The
forward path splats predicted spots (one per ODF sample) into the same
patch coordinate system.

We keep the splat differentiable by using a soft Gaussian kernel with
analytic weighting, as in fwd_sim/hedm_forward.predict_images, but on
per-spot patches rather than the full detector volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class SpotPatchSpec:
    """Geometry of the per-spot patch volumes.

    All patches share the same shape (F, P, P) for vectorization.

    Attributes
    ----------
    n_spots : int
        Number of spots S.
    patch_F : int
        Patch extent along the omega/frame axis.
    patch_P : int
        Patch extent along y and z (square).
    sigma_yz : float
        Gaussian kernel sigma in pixels for the (y, z) splat.
    sigma_f : float
        Gaussian kernel sigma in frames for the omega splat.
    anchor_y : Tensor (S,)
        Patch center along y, in detector pixel units.
    anchor_z : Tensor (S,)
        Patch center along z, in detector pixel units.
    anchor_f : Tensor (S,)
        Patch center along frame, in fractional frame index.
    """

    n_spots: int
    patch_F: int
    patch_P: int
    sigma_yz: float
    sigma_f: float
    anchor_y: torch.Tensor
    anchor_z: torch.Tensor
    anchor_f: torch.Tensor
    # Optional per-spot anisotropic kernel widths (in pixels for radial/eta,
    # frames for omega). When all four are provided the splatter switches
    # from isotropic (sigma_yz, sigma_f) to per-spot anisotropic kernels
    # oriented along the per-spot (radial, eta) axes derived from the spot
    # position relative to the beam center. Set on a SpotPatchSpec instance
    # like:
    #     spec.sigma_radial = torch.tensor(...)   # (S,) px
    #     spec.sigma_eta    = torch.tensor(...)   # (S,) px
    #     spec.sigma_f_per_spot = torch.tensor(...)  # (S,) frames
    #     spec.radial_y, spec.radial_z = ...       # (S,) unit vectors
    sigma_radial: Optional[torch.Tensor] = None
    sigma_eta: Optional[torch.Tensor] = None
    sigma_f_per_spot: Optional[torch.Tensor] = None
    radial_y: Optional[torch.Tensor] = None
    radial_z: Optional[torch.Tensor] = None


def splat_spots_to_patches_dense(
    spec: SpotPatchSpec,
    spot_y: torch.Tensor,
    spot_z: torch.Tensor,
    spot_f: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Dense Gaussian splatter (evaluates kernel at every patch cell).

    Memory: O(K * S * F * P^2). Suitable for small P; for real-data spreads
    (P ≳ 60) use ``splat_spots_to_patches_sparse`` instead.

    Returns
    -------
    patches : Tensor (S, F, P, P)
    """
    S = spec.n_spots
    F = spec.patch_F
    P = spec.patch_P
    device = spot_y.device
    dtype = spot_y.dtype

    local_y = spot_y - spec.anchor_y.to(dtype) + 0.5 * (P - 1)
    local_z = spot_z - spec.anchor_z.to(dtype) + 0.5 * (P - 1)
    local_f = spot_f - spec.anchor_f.to(dtype) + 0.5 * (F - 1)

    grid_f = torch.arange(F, device=device, dtype=dtype).reshape(1, 1, F, 1, 1)
    grid_y = torch.arange(P, device=device, dtype=dtype).reshape(1, 1, 1, P, 1)
    grid_z = torch.arange(P, device=device, dtype=dtype).reshape(1, 1, 1, 1, P)

    mu_f = local_f.reshape(*local_f.shape, 1, 1, 1)
    mu_y = local_y.reshape(*local_y.shape, 1, 1, 1)
    mu_z = local_z.reshape(*local_z.shape, 1, 1, 1)

    # Anisotropic per-spot kernel: oriented along (radial, eta) at each spot.
    # Used when the spec carries sigma_radial/sigma_eta/radial_y/radial_z.
    if (spec.sigma_radial is not None
            and spec.sigma_eta is not None
            and spec.radial_y is not None
            and spec.radial_z is not None):
        # Per-spot (S,) tensors broadcast to (1, S, 1, 1, 1) for kernel.
        s_rad = spec.sigma_radial.to(dtype).reshape(1, -1, 1, 1, 1)
        s_eta = spec.sigma_eta.to(dtype).reshape(1, -1, 1, 1, 1)
        ry = spec.radial_y.to(dtype).reshape(1, -1, 1, 1, 1)
        rz = spec.radial_z.to(dtype).reshape(1, -1, 1, 1, 1)
        s_f_use = (spec.sigma_f_per_spot.to(dtype).reshape(1, -1, 1, 1, 1)
                    if spec.sigma_f_per_spot is not None
                    else torch.tensor(spec.sigma_f, dtype=dtype, device=device))
        # Pixel offsets within patch.
        dy = grid_y - mu_y
        dz = grid_z - mu_z
        # Project onto per-spot (radial, eta) basis.
        # eta unit = (-rz, ry) (perpendicular to radial in detector plane).
        d_rad = dy * ry + dz * rz
        d_eta = -dy * rz + dz * ry
        arg = (
            ((grid_f - mu_f) / s_f_use) ** 2
            + (d_rad / s_rad) ** 2
            + (d_eta / s_eta) ** 2
        )
    else:
        arg = (
            ((grid_f - mu_f) / spec.sigma_f) ** 2
            + ((grid_y - mu_y) / spec.sigma_yz) ** 2
            + ((grid_z - mu_z) / spec.sigma_yz) ** 2
        )
    kernel = torch.exp(-0.5 * arg)

    w_eff = (weights.reshape(-1, 1) * valid).reshape(*valid.shape, 1, 1, 1)
    weighted = kernel * w_eff
    patches = weighted.sum(dim=0)
    return patches


def splat_spots_to_patches_sparse(
    spec: SpotPatchSpec,
    spot_y: torch.Tensor,
    spot_z: torch.Tensor,
    spot_f: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    radius_yz: int = 3,
    radius_f: int = 2,
    sigma_yz_row: Optional[torch.Tensor] = None,
    sigma_f_row: Optional[torch.Tensor] = None,
    sigma_radial_row: Optional[torch.Tensor] = None,
    sigma_eta_row: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Radius-bounded Gaussian splatter via scatter_add.

    For each predicted spot the kernel is evaluated only on the
    (2*radius_f + 1) * (2*radius_yz + 1)^2 cells around the rounded center,
    rather than the full F*P*P patch. Memory and compute both drop from
    O(K*S*F*P^2) to O(K*S*(2r+1)^3) — typically 10–1000x smaller for the
    P ≳ 60 patches needed for ±1° axis-angle spreads at production
    geometries.

    Gradient flow w.r.t. ``spot_y / spot_z / spot_f`` is preserved through
    the analytic Gaussian (the rounding is non-differentiable but is only
    used to *select* cells, not to compute kernel values).

    Truncation: at radius=3, sigma=1 the missing tails are < 0.3 % of the
    integral; at radius=2, sigma=0.6 they are < 0.05 %. Both are negligible
    compared to typical patch noise. Tighten if a deployment finds bias.
    """
    S = spec.n_spots
    F = spec.patch_F
    P = spec.patch_P
    K = spot_y.shape[0]
    device = spot_y.device
    dtype = spot_y.dtype

    local_y = spot_y - spec.anchor_y.to(dtype) + 0.5 * (P - 1)
    local_z = spot_z - spec.anchor_z.to(dtype) + 0.5 * (P - 1)
    local_f = spot_f - spec.anchor_f.to(dtype) + 0.5 * (F - 1)

    cy = local_y.detach().round().long()
    cz = local_z.detach().round().long()
    cf = local_f.detach().round().long()

    of = torch.arange(-radius_f, radius_f + 1, device=device, dtype=torch.long)
    oy = torch.arange(-radius_yz, radius_yz + 1, device=device, dtype=torch.long)
    oz = torch.arange(-radius_yz, radius_yz + 1, device=device, dtype=torch.long)
    OF, OY, OZ = torch.meshgrid(of, oy, oz, indexing="ij")
    OF = OF.flatten()
    OY = OY.flatten()
    OZ = OZ.flatten()
    L = OF.numel()

    cy_idx = cy.unsqueeze(-1) + OY.reshape(1, 1, -1)
    cz_idx = cz.unsqueeze(-1) + OZ.reshape(1, 1, -1)
    cf_idx = cf.unsqueeze(-1) + OF.reshape(1, 1, -1)

    in_bounds = (
        (cy_idx >= 0) & (cy_idx < P)
        & (cz_idx >= 0) & (cz_idx < P)
        & (cf_idx >= 0) & (cf_idx < F)
    )

    dy = cy_idx.to(dtype) - local_y.unsqueeze(-1)
    dz = cz_idx.to(dtype) - local_z.unsqueeze(-1)
    df = cf_idx.to(dtype) - local_f.unsqueeze(-1)

    # Optional per-row (e.g. per-voxel) widths broaden the kernel; dy/dz/df are
    # (K, S, L), so per-row sigma broadcasts as (K, 1, 1).
    s_f = (sigma_f_row.to(dtype).reshape(-1, 1, 1) if sigma_f_row is not None
           else spec.sigma_f)
    anisotropic = (
        sigma_radial_row is not None and sigma_eta_row is not None
        and spec.radial_y is not None and spec.radial_z is not None
    )
    if anisotropic:
        # Per-voxel anisotropic kernel oriented along the per-spot
        # (radial, eta) detector axes. Radial broadens by σ_ε (microstrain);
        # eta + frame broaden by σ_θ (orientation spread). The radial unit
        # vector is fixed per spot from the detector geometry.
        #
        # Shape handling: each row sigma can be (K,) for per-voxel-only
        # broadening (uniform across spots) or (K, S) for per-(voxel, spot)
        # broadening — needed by physical mWH where σ_ε is a dimensionless
        # microstrain whose pixel-width contribution scales as 2·Lsd·tan(θ_s)/px
        # per reflection.
        def _aniso_reshape(t):
            t = t.to(dtype)
            if t.dim() == 1:
                return t.reshape(-1, 1, 1)                            # (K, 1, 1)
            if t.dim() == 2:
                return t.unsqueeze(-1)                                # (K, S, 1)
            raise ValueError(f"sigma row must be 1-D or 2-D, got {t.dim()}-D")
        s_rad = _aniso_reshape(sigma_radial_row)
        s_eta = _aniso_reshape(sigma_eta_row)
        ry = spec.radial_y.to(dtype).reshape(1, -1, 1)               # (1, S, 1)
        rz = spec.radial_z.to(dtype).reshape(1, -1, 1)
        d_rad = dy * ry + dz * rz                                    # (K, S, L)
        d_eta = -dy * rz + dz * ry
        arg = (d_rad / s_rad) ** 2 + (d_eta / s_eta) ** 2 + (df / s_f) ** 2
    else:
        s_yz = (sigma_yz_row.to(dtype).reshape(-1, 1, 1) if sigma_yz_row is not None
                else spec.sigma_yz)
        arg = (
            (dy / s_yz) ** 2
            + (dz / s_yz) ** 2
            + (df / s_f) ** 2
        )
    kernel = torch.exp(-0.5 * arg)
    if anisotropic:
        # Conserve integrated intensity vs the instrumental width
        # (∝ σ_yz²·σ_f for the isotropic baseline).
        norm_row = (float(spec.sigma_yz) ** 2 * float(spec.sigma_f)) / (
            (s_rad * s_eta) * s_f)
        kernel = kernel * norm_row
    elif sigma_yz_row is not None or sigma_f_row is not None:
        # Conserve integrated intensity vs the instrumental width (∝ σ_yz²·σ_f):
        # broadening a voxel's splat must not create counts. spread=0 → factor 1.
        norm_row = (float(spec.sigma_yz) ** 2 * float(spec.sigma_f)) / (
            (s_yz ** 2) * s_f)
        kernel = kernel * norm_row

    w_eff = (weights.reshape(-1, 1) * valid).unsqueeze(-1)
    contributions = kernel * w_eff * in_bounds.to(dtype)

    cy_clamped = cy_idx.clamp(0, P - 1)
    cz_clamped = cz_idx.clamp(0, P - 1)
    cf_clamped = cf_idx.clamp(0, F - 1)
    flat_idx = cf_clamped * (P * P) + cy_clamped * P + cz_clamped

    contrib_flat = contributions.permute(1, 0, 2).reshape(S, K * L)
    idx_flat = flat_idx.permute(1, 0, 2).reshape(S, K * L)

    patches_flat = torch.zeros(S, F * P * P, dtype=dtype, device=device)
    patches_flat.scatter_add_(1, idx_flat, contrib_flat)
    return patches_flat.reshape(S, F, P, P)


# Default splatter: sparse for production performance, dense when the
# spec carries anisotropic per-spot kernel parameters (sparse splatter
# doesn't yet support per-spot anisotropic kernels — the radius-bounded
# round selects an isotropic neighborhood that may miss eta-elongated
# probability mass for elongated spots).
def splat_spots_to_patches(spec, *args, **kwargs):
    if (spec.sigma_radial is not None and spec.sigma_eta is not None
            and spec.radial_y is not None and spec.radial_z is not None):
        return splat_spots_to_patches_dense(spec, *args, **kwargs)
    return splat_spots_to_patches_sparse(spec, *args, **kwargs)


def compute_per_spot_pixel_mask(
    spec: SpotPatchSpec,
    other_anchor_y: Optional[torch.Tensor] = None,
    other_anchor_z: Optional[torch.Tensor] = None,
    other_anchor_f: Optional[torch.Tensor] = None,
    other_spot_id: Optional[torch.Tensor] = None,
    *,
    frame_to_pixel_scale: Optional[float] = None,
) -> torch.Tensor:
    """Build a (S, F, P, P) bool/float mask that retains only the pixels
    "belonging" to each spot under a Voronoi partition over all spot
    anchors that fall inside the patch.

    For each spot ``s``:
      - Look at every anchor in ``(other_anchor_y/z/f)`` whose
        ``other_spot_id != s`` and which is closer in (y, z, f) to any
        cell of spot s's patch than s's own anchor.
      - Pixels closer to a competing anchor are masked out (mask = 0).
      - Pixels closer to s's own anchor are kept (mask = 1).
      - If no competitor falls inside the patch's spatial bounding box
        plus a small margin, the mask is all-ones (no work needed).

    Frame distances are rescaled to (pixel-equivalent) by
    ``frame_to_pixel_scale`` so that the (y, z, f) distance metric is
    isotropic. Default = sigma_yz / sigma_f from the spec — frames-of-
    sigma get the same weight as pixels-of-sigma.

    Parameters
    ----------
    spec : SpotPatchSpec
        Patch geometry with anchor_y/z/f for the spots being fit.
        S = spec.n_spots.
    other_anchor_y/z/f : Tensor (N_other,)
        Anchors of *every* spot in the cluster (this grain's spots and
        all neighbors). May include the spot itself; ``other_spot_id``
        is used to identify and exclude self.
    other_spot_id : Tensor (N_other,) long
        Index into spec's spot list for entries that ARE this grain's
        spots; -1 (or any value not in [0, S)) for spots from other
        grains.

    Returns
    -------
    mask : Tensor (S, F, P, P) of dtype matching spec.anchor_y
        1.0 where pixel belongs to spot s, 0.0 where a different spot
        is closer.
    """
    S = spec.n_spots
    F = spec.patch_F
    P = spec.patch_P
    device = spec.anchor_y.device
    dtype = spec.anchor_y.dtype

    if other_anchor_y is None:
        return torch.ones(S, F, P, P, dtype=dtype, device=device)
    N_other = int(other_anchor_y.numel())
    if N_other == 0:
        return torch.ones(S, F, P, P, dtype=dtype, device=device)

    if frame_to_pixel_scale is None:
        sigf = max(float(spec.sigma_f), 1e-6)
        frame_to_pixel_scale = float(spec.sigma_yz) / sigf

    # Local pixel coordinates of every cell in the patch.
    f_idx = torch.arange(F, dtype=dtype, device=device)
    y_idx = torch.arange(P, dtype=dtype, device=device)
    z_idx = torch.arange(P, dtype=dtype, device=device)
    # Translate to global (y, z, f). For spot s, pixel (fi, yi, zi) sits at:
    #   y_global = anchor_y[s] + (yi - (P-1)/2)
    #   z_global = anchor_z[s] + (zi - (P-1)/2)
    #   f_global = anchor_f[s] + (fi - (F-1)/2)
    y_off = (y_idx - 0.5 * (P - 1))                # (P,)
    z_off = (z_idx - 0.5 * (P - 1))                # (P,)
    f_off = (f_idx - 0.5 * (F - 1))                # (F,)

    # Build competitor anchors once on device.
    oay = other_anchor_y.to(device).to(dtype)
    oaz = other_anchor_z.to(device).to(dtype)
    oaf = other_anchor_f.to(device).to(dtype)
    if other_spot_id is None:
        oid = torch.full((N_other,), -1, dtype=torch.long, device=device)
    else:
        oid = other_spot_id.to(device).to(torch.long)

    # Adaptive batching over spots AND competitors so that the
    # peak intermediate (S_b, F, P, P, k) fits within a memory budget.
    # Choose s_batch so a single competitor's distance tensor is <= 1 GB,
    # then chunk competitors so (s_batch * F * P^2 * k) <= 4 GB.
    _bps = torch.empty((), dtype=dtype).element_size()
    _per_cell_bytes = max(1, F * P * P * _bps)
    s_batch = max(1, min(S, (1024 ** 3) // _per_cell_bytes))
    _budget = 4 * 1024 ** 3
    chunk = max(1, min(16, _budget // (s_batch * _per_cell_bytes)))

    mask = torch.ones(S, F, P, P, dtype=dtype, device=device)
    s_idx_full = torch.arange(S, dtype=torch.long, device=device)

    for s0 in range(0, S, s_batch):
        s1 = min(s0 + s_batch, S)
        Sb = s1 - s0
        # Build (Sb, F, P, P) cell-coords for this batch only.
        ay_b = spec.anchor_y[s0:s1].view(Sb, 1, 1, 1)
        az_b = spec.anchor_z[s0:s1].view(Sb, 1, 1, 1)
        af_b = spec.anchor_f[s0:s1].view(Sb, 1, 1, 1)
        cell_y_b = ay_b + y_off.view(1, 1, P, 1)
        cell_z_b = az_b + z_off.view(1, 1, 1, P)
        cell_f_b = af_b + f_off.view(1, F, 1, 1)
        cell_y_b = cell_y_b.expand(Sb, F, P, P)
        cell_z_b = cell_z_b.expand(Sb, F, P, P)
        cell_f_b = cell_f_b.expand(Sb, F, P, P)

        # own distance squared
        own_dy = cell_y_b - ay_b
        own_dz = cell_z_b - az_b
        own_df = (cell_f_b - af_b) * frame_to_pixel_scale
        own_d2_b = own_dy ** 2 + own_dz ** 2 + own_df ** 2  # (Sb, F, P, P)

        s_idx_b = s_idx_full[s0:s1]                          # (Sb,)
        mask_b = torch.ones(Sb, F, P, P, dtype=dtype, device=device)

        for start in range(0, N_other, chunk):
            end = min(start + chunk, N_other)
            sub_y = oay[start:end].view(1, 1, 1, 1, -1)
            sub_z = oaz[start:end].view(1, 1, 1, 1, -1)
            sub_f = oaf[start:end].view(1, 1, 1, 1, -1)
            sub_id = oid[start:end].view(1, 1, 1, 1, -1)
            dy = cell_y_b.unsqueeze(-1) - sub_y
            dz = cell_z_b.unsqueeze(-1) - sub_z
            df = (cell_f_b.unsqueeze(-1) - sub_f) * frame_to_pixel_scale
            comp_d2 = dy ** 2 + dz ** 2 + df ** 2            # (Sb,F,P,P,k)
            is_self = (sub_id == s_idx_b.view(Sb, 1, 1, 1, 1))
            comp_d2 = torch.where(
                is_self, torch.full_like(comp_d2, float("inf")), comp_d2,
            )
            comp_min, _ = comp_d2.min(dim=-1)                 # (Sb,F,P,P)
            cell_keep = (own_d2_b <= comp_min).to(dtype)
            mask_b = mask_b * cell_keep
            del dy, dz, df, comp_d2, comp_min, cell_keep

        mask[s0:s1] = mask_b
        del cell_y_b, cell_z_b, cell_f_b, own_d2_b, mask_b

    return mask


def compute_adaptive_patch_size(
    model,
    R_avg_list: torch.Tensor,
    position_list: torch.Tensor,
    theta_max_deg: float,
    *,
    sigma_yz: float = 1.0,
    sigma_f: float = 0.6,
    margin_sigma: float = 3.0,
    px_per_deg: float = 50.0,
    frames_per_deg: Optional[float] = None,
    min_F: int = 5,
    min_P: int = 11,
    max_F: int = 51,
    max_P: int = 251,
) -> Tuple[int, int]:
    """Recommend a (patch_F, patch_P) sized to enclose every predicted spot's
    extent across the ODF support of theta_max_deg.

    Uses an analytical estimate from the spot Jacobian:

        patch_P = ceil(2 * px_per_deg * theta_max_deg
                       + 2 * margin_sigma * sigma_yz)
        patch_F = ceil(2 * frames_per_deg * theta_max_deg
                       + 2 * margin_sigma * sigma_f)

    The Jacobian factors default to typical FF-HEDM values:
      px_per_deg = 50  (pixels of (y, z) shift per degree of axis-angle
                        perturbation, conservative bound for ~mm-Lsd)
      frames_per_deg = 1 / omega_step (tracks the omega-rotation Jacobian)

    A previous implementation tried to measure the extent by forward-
    simulating sampled orientations within theta_max, but small-angle
    branch-swap pathologies near the omega boundary made the empirical
    range non-robust. The analytical estimate is conservative and
    deterministic.

    For a single grain pass R_avg_list of shape (1, 3, 3); for a
    multi-grain cluster pass the full (G, 3, 3) stack -- the analytical
    bound is grain-independent under this approximation, so the function
    returns a (patch_F, patch_P) appropriate for any grain in the cluster.

    Returns (patch_F, patch_P) -- both odd integers, clamped to
    [min_F, max_F] and [min_P, max_P].
    """
    if frames_per_deg is None:
        omega_step = float(getattr(model, "omega_step", 0.25))
        frames_per_deg = 1.0 / max(abs(omega_step), 1e-6)

    extent_yz = 2.0 * px_per_deg * theta_max_deg
    extent_f = 2.0 * frames_per_deg * theta_max_deg

    P = int(extent_yz + 2.0 * margin_sigma * sigma_yz + 0.5) + 1
    F = int(extent_f + 2.0 * margin_sigma * sigma_f + 0.5) + 1
    if P % 2 == 0:
        P += 1
    if F % 2 == 0:
        F += 1
    P = max(P, min_P); F = max(F, min_F)
    P = min(P, max_P); F = min(F, max_F)
    return F, P
