"""Multi-image / multi-distance joint loss with shared/per-image params.

Each image contributes a pseudo-strain residual using its own per-image
parameters (Lsd, BC, tilts) plus the shared block (distortion, panels, pxY/Z).
The total loss is the simple sum.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .pseudo_strain import pseudo_strain_residual


def multi_image_loss(
    images_data: List[dict],
    *,
    shared_params: Dict[str, torch.Tensor],
    per_image_params: List[Dict[str, torch.Tensor]],
    panel_layout=None,
) -> torch.Tensor:
    """Joint loss across images.

    Parameters
    ----------
    images_data : list[dict]
        One per image.  Each dict carries:
          - ``Y_pix``, ``Z_pix``      [n_pts]
          - ``ring_two_theta_deg``    [n_pts]
          - ``rho_d``                 scalar
          - optional ``weights``, ``panel_idx``
    shared_params, per_image_params : dicts of tensors.

    Returns
    -------
    Scalar loss (sum of per-image ½‖r‖²).
    """
    if len(images_data) != len(per_image_params):
        raise ValueError("len(images_data) must match len(per_image_params)")
    total = torch.zeros((), dtype=torch.float64)
    for img, img_params in zip(images_data, per_image_params):
        merged = {**shared_params, **img_params}
        r = pseudo_strain_residual(
            img["Y_pix"], img["Z_pix"],
            img["ring_two_theta_deg"], merged,
            rho_d=img["rho_d"],
            weights=img.get("weights"),
            panel_layout=panel_layout,
            panel_idx=img.get("panel_idx"),
        )
        total = total + 0.5 * (r * r).sum()
    return total


__all__ = ["multi_image_loss"]
