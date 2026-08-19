"""Gauge constraint for the panel radial-expansion nullspace.

``fix_panel_id`` (or the softer ``Σ panel = 0``) removes the *translation*
nullspace of a per-panel (δy, δz) field: shift every module together and the
beam centre absorbs it. There is a second nullspace it does not touch.

Push module *k* outward by ``c · r_k / rbar`` and every ring radius grows by
``c · r/rbar``; change Lsd by ``δL`` and every ring radius grows by
``R · δL/L``. Same functional form, so the fit can move amplitude freely
between the two. Measured on a 48-panel Pilatus (2026-08-19), **11 %** of the
fitted panel field sat in that mode, and moving it into Lsd cancelled **73 %**
of its effect on 2θ — strongly, though not exactly, degenerate (a panel
expansion is a step function per module; Lsd is smooth in R).

Left alone, the expansion mode makes the panel numbers look like module
positions when they are partly a distance error in disguise. This adds
curvature along that direction only, so the data-determined content of the
panel field is untouched.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = ["expansion_mode", "panel_expansion_residual"]


def expansion_mode(panel_centers_y: torch.Tensor,
                   panel_centers_z: torch.Tensor,
                   bc_y: torch.Tensor | float,
                   bc_z: torch.Tensor | float) -> torch.Tensor:
    """Unit vector of the radial-expansion mode, shape ``[n_panels, 2]``.

    Module *k* displaced along its own outward radial direction, weighted by
    its distance from the beam centre — the shape an Lsd error takes when
    expressed as a panel field.
    """
    cy = panel_centers_y.reshape(-1)
    cz = panel_centers_z.reshape(-1)
    ry = cy - (bc_y if torch.is_tensor(bc_y) else torch.as_tensor(bc_y, dtype=cy.dtype, device=cy.device))
    rz = cz - (bc_z if torch.is_tensor(bc_z) else torch.as_tensor(bc_z, dtype=cz.dtype, device=cz.device))
    r = torch.sqrt(ry * ry + rz * rz).clamp(min=1e-9)
    rbar = r.mean()
    m = torch.stack([ry / r * r / rbar, rz / r * r / rbar], dim=1)   # = [ry, rz]/rbar
    n = torch.linalg.vector_norm(m).clamp(min=1e-30)
    return m / n


def panel_expansion_residual(unpacked: dict,
                             *,
                             panel_layout,
                             bc_y: torch.Tensor | float,
                             bc_z: torch.Tensor | float,
                             lambda_ex: float = 1e6) -> torch.Tensor:
    """One residual row penalising the expansion component of ``panel_delta_yz``.

    Returns an empty tensor when the spec carries no panel shifts, so callers
    can concatenate unconditionally.
    """
    dyz = unpacked.get("panel_delta_yz")
    if dyz is None or dyz.numel() == 0:
        ref = next(iter(unpacked.values()))
        return torch.zeros(0, dtype=ref.dtype, device=ref.device)
    if panel_layout is None or panel_layout.panel_centers_y is None:
        return torch.zeros(0, dtype=dyz.dtype, device=dyz.device)

    m = expansion_mode(
        panel_layout.panel_centers_y.to(dyz.device, dyz.dtype),
        panel_layout.panel_centers_z.to(dyz.device, dyz.dtype),
        bc_y, bc_z,
    )
    amp = (dyz * m).sum()            # projection onto the unit mode
    return (lambda_ex * amp).reshape(1)
