"""Per-spot Delta-table builder + shape-only image-MSE loss.

Implements the §6 strategy: pre-compute per-spot offset Delta from the
grain-averaged single-orientation prediction, splat ODF-weighted predicted
spots into Delta-anchored patches, image-MSE against measured patches.
The fixed-point Delta refresh (recompute Delta from ODF-weighted predicted
centroid, refit) is implemented in inversion.py.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from midas_grain_odf.spot_extract import SpotPatchSpec, splat_spots_to_patches


def compute_delta_table(
    pred_y: torch.Tensor,
    pred_z: torch.Tensor,
    pred_f: torch.Tensor,
    meas_y: torch.Tensor,
    meas_z: torch.Tensor,
    meas_f: torch.Tensor,
    valid: torch.Tensor,
    outlier_alpha: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-spot Delta = measured - predicted, with outlier filtering.

    All inputs are 1D over the spot dimension.

    Parameters
    ----------
    pred_y, pred_z, pred_f : Tensor (S,)
        Predicted spot centroids in detector pixels and fractional frames.
    meas_y, meas_z, meas_f : Tensor (S,)
        Measured spot centroids.
    valid : Tensor (S,)
        Validity mask from the forward model (1 = valid).
    outlier_alpha : float
        Spots whose ‖Delta_yz‖ exceeds alpha * median are flagged as outliers.

    Returns
    -------
    delta_y, delta_z, delta_f : Tensor (S,)
        Per-spot offsets (zero where invalid).
    keep : Tensor (S,)
        Bool mask of spots retained after validity + outlier filtering.
    """
    delta_y = meas_y - pred_y
    delta_z = meas_z - pred_z
    delta_f = meas_f - pred_f

    norm_yz = torch.sqrt(delta_y ** 2 + delta_z ** 2)
    valid_b = valid > 0.5

    if valid_b.any():
        med = norm_yz[valid_b].median()
        # Median-based outlier rejection. Use 1.0 px floor so that a perfect
        # synthetic dataset (Delta ~ 0) doesn't flag genuine residuals as
        # outliers.
        threshold = max(float(outlier_alpha) * float(med), 1.0)
        keep = valid_b & (norm_yz <= threshold)
    else:
        keep = torch.zeros_like(valid_b)

    # Zero out Delta where not kept.
    delta_y = torch.where(keep, delta_y, torch.zeros_like(delta_y))
    delta_z = torch.where(keep, delta_z, torch.zeros_like(delta_z))
    delta_f = torch.where(keep, delta_f, torch.zeros_like(delta_f))
    return delta_y, delta_z, delta_f, keep


def odf_weighted_predicted_centroid(
    spot_y: torch.Tensor,
    spot_z: torch.Tensor,
    spot_f: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the ODF-weighted centroid of predicted spots, per spot.

    Used to refresh Delta after one ODF fit.

    Parameters
    ----------
    spot_y, spot_z, spot_f : Tensor (K, S)
        Predicted detector coordinates per (orientation, spot).
    weights : Tensor (K,)
        ODF simplex weights.
    valid : Tensor (K, S)
        Validity per (orientation, spot).

    Returns
    -------
    cy, cz, cf : Tensor (S,)
        Per-spot weighted centroids (NaN-safe -- zero where total weight is 0).
    """
    w = (weights.reshape(-1, 1) * valid)        # (K, S)
    norm = w.sum(dim=0)                         # (S,)
    safe = norm.clamp(min=1e-12)
    cy = (w * spot_y).sum(dim=0) / safe
    cz = (w * spot_z).sum(dim=0) / safe
    cf = (w * spot_f).sum(dim=0) / safe
    # Where total weight is 0, fall back to single-orientation prediction
    # (use first valid orientation's value, else zero).
    fallback_y = spot_y[0]
    fallback_z = spot_z[0]
    fallback_f = spot_f[0]
    bad = norm < 1e-9
    cy = torch.where(bad, fallback_y, cy)
    cz = torch.where(bad, fallback_z, cz)
    cf = torch.where(bad, fallback_f, cf)
    return cy, cz, cf


def shape_mse_loss(
    spec: SpotPatchSpec,
    spot_y: torch.Tensor,
    spot_z: torch.Tensor,
    spot_f: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    measured_patches: torch.Tensor,
    delta_y: torch.Tensor,
    delta_z: torch.Tensor,
    delta_f: torch.Tensor,
    keep: torch.Tensor,
    loss_norm: str = "mean",
    ref_patch_F: int = 7,
    ref_patch_P: int = 31,
    pixel_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Image-MSE between Delta-shifted predicted patches and measured patches.

    The shift is applied by *moving the patch anchor* by Delta — equivalently,
    every predicted spot for spot s is offset by (delta_y[s], delta_z[s],
    delta_f[s]) before splatting. Measured patches are already cropped at
    (anchor + Delta) by the caller.

    Parameters
    ----------
    spec : SpotPatchSpec
        Patch geometry. The anchor fields define the *reference* (single-
        orientation predicted) center; this routine adds Delta on top.
    spot_y, spot_z, spot_f : Tensor (K, S)
        Predicted detector coordinates from the current ODF.
    weights : Tensor (K,)
    valid : Tensor (K, S)
    measured_patches : Tensor (S, F, P, P)
    delta_y, delta_z, delta_f : Tensor (S,)
        Per-spot Delta offsets (broadcasted across K).
    keep : Tensor (S,) bool
        Spots to include in the loss.

    Returns
    -------
    loss : Tensor scalar
    """
    # Apply Delta by shifting predicted-spot coords (per spot, broadcast over K).
    shifted_y = spot_y + delta_y.unsqueeze(0)
    shifted_z = spot_z + delta_z.unsqueeze(0)
    shifted_f = spot_f + delta_f.unsqueeze(0)

    # Build a Delta-shifted patch spec: anchors move by Delta so the splatter's
    # local coordinates land at the *measured* spot center. Per-spot anisotropic
    # kernel fields (sigma_radial/eta, sigma_f_per_spot, radial_y/z) are
    # detector-frame properties that do not move with Delta — carry them
    # through verbatim so the splatter still takes the anisotropic path.
    shifted_spec = SpotPatchSpec(
        n_spots=spec.n_spots,
        patch_F=spec.patch_F,
        patch_P=spec.patch_P,
        sigma_yz=spec.sigma_yz,
        sigma_f=spec.sigma_f,
        anchor_y=spec.anchor_y + delta_y,
        anchor_z=spec.anchor_z + delta_z,
        anchor_f=spec.anchor_f + delta_f,
        sigma_radial=spec.sigma_radial,
        sigma_eta=spec.sigma_eta,
        sigma_f_per_spot=spec.sigma_f_per_spot,
        radial_y=spec.radial_y,
        radial_z=spec.radial_z,
    )

    pred_patches = splat_spots_to_patches(
        shifted_spec, shifted_y, shifted_z, shifted_f, weights, valid,
    )

    # Branch out for c*-scaled and correlation losses BEFORE we compute
    # the raw squared residual. Those modes need the un-squared pred and
    # meas to compute the per-spot scaling / centering.
    if loss_norm in ("c_star_mse", "correlation"):
        # (S, K_cells)
        p_flat = pred_patches.reshape(pred_patches.shape[0], -1)
        m_flat = measured_patches.reshape(measured_patches.shape[0], -1)
        if pixel_mask is not None:
            mask_flat = pixel_mask.reshape(pixel_mask.shape[0], -1)
            p_flat = p_flat * mask_flat
            m_flat = m_flat * mask_flat
        pp = (p_flat * p_flat).sum(dim=1).clamp(min=1e-30)
        pm = (p_flat * m_flat).sum(dim=1)
        if loss_norm == "c_star_mse":
            # Per-spot OLS scalar: c* = ⟨p, m⟩ / ⟨p, p⟩.  Then minimize
            # Σ_s ||c*_s · p_s − m_s||² = Σ_s [⟨m_s, m_s⟩ − ⟨p_s, m_s⟩²/⟨p_s, p_s⟩].
            # Stop-grad on c* keeps the gradient on (p_flat) clean —
            # otherwise the closed-form c* introduces a circular gradient.
            c_star = (pm / pp).detach()
            scaled_diff = (c_star.unsqueeze(1) * p_flat - m_flat) ** 2
            per_spot = scaled_diff.sum(dim=1)
        else:  # correlation
            # Per-spot Pearson r²; minimize −Σ_s r²_s.
            # Use mean-centered versions of p and m. With OLS,
            # r² = ⟨p_c, m_c⟩² / (⟨p_c, p_c⟩ · ⟨m_c, m_c⟩).
            # Mean per-spot:
            n_cells = float(p_flat.shape[1])
            p_mean = p_flat.mean(dim=1, keepdim=True)
            m_mean = m_flat.mean(dim=1, keepdim=True)
            pc = p_flat - p_mean
            mc = m_flat - m_mean
            num = (pc * mc).sum(dim=1)
            den = (
                ((pc * pc).sum(dim=1).clamp(min=1e-30))
                * ((mc * mc).sum(dim=1).clamp(min=1e-30))
            )
            r2 = num * num / den
            per_spot = -r2  # minimize negative r²
        keep_f = keep.to(per_spot.dtype)
        return (per_spot * keep_f).sum()

    # Per-spot MSE; mask by keep, optionally per-pixel by pixel_mask.
    diff = (pred_patches - measured_patches) ** 2
    if pixel_mask is not None:
        # pixel_mask is (S, F, P, P); apply elementwise to the residual.
        diff = diff * pixel_mask
        # Per-spot normalizer becomes the mask sum (= count of "kept"
        # cells per spot), so the per-spot loss is the mean over the
        # spot's connected-component pixels.
        per_spot_keep_count = pixel_mask.flatten(1).sum(dim=1).clamp(min=1.0)
    else:
        per_spot_keep_count = None

    if loss_norm == "mean":
        if pixel_mask is not None:
            per_spot = diff.flatten(1).sum(dim=1) / per_spot_keep_count
        else:
            per_spot = diff.flatten(1).mean(dim=1)              # (S,)
    elif loss_norm == "ref":
        # Sum the squared residuals across cells, then normalize as if the
        # patch were a fixed (ref_patch_F x ref_patch_P x ref_patch_P) tile.
        # This keeps gradient magnitude scale-invariant when the patch grows
        # with adaptive sizing — the "active" signal cells contribute the
        # same amount regardless of how many background cells the patch has.
        cells_ref = float(ref_patch_F * ref_patch_P * ref_patch_P)
        per_spot = diff.flatten(1).sum(dim=1) / cells_ref   # (S,)
    else:
        raise ValueError(f"unknown loss_norm: {loss_norm!r}")
    keep_f = keep.to(per_spot.dtype)
    if loss_norm == "ref":
        # Sum across kept spots — do NOT divide by n_keep. The per-spot
        # residual is already normalized by ref_cells; an additional
        # /n_keep collapses the loss magnitude into the strong-Wolfe
        # stagnation regime where the line search accepts the initial
        # step (lr=1) without sufficient progress on small-gradient
        # parameters (logits in particular).
        return (per_spot * keep_f).sum()
    n_keep = keep_f.sum().clamp(min=1.0)
    return (per_spot * keep_f).sum() / n_keep
