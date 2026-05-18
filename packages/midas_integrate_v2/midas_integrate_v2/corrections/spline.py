"""Stage-4 thin-plate spline residual correction as a torch nn.Module.

Calibrate-v2's Stage 4 fits an RBF (thin-plate) spline to the per-pixel
``ΔR(Y, Z)`` residual. v1 consumed it as a baked binary lookup. v2 keeps
it as a refinable layer:

- :class:`RBFResidualCorrection` — given trained centres + weights, evaluates
  the spline at any (Y, Z) and returns ``ΔR``. Can be jointly refined with
  geometry by registering its weights as ``nn.Parameter`` (default).
- :class:`IdentityResidualCorrection` — zero correction; useful as a default
  when no spline is configured.

Either is plugged into :func:`integrate_with_corrections` as the
``residual`` argument and applies ``R += ΔR(Y, Z)`` after the geometry
forward and per-ring offsets.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _thin_plate_kernel(r2: torch.Tensor) -> torch.Tensor:
    """Standard 2D thin-plate radial basis: ``φ(r) = r² log(r)`` for r > 0,
    ``φ(0) = 0`` (the limit, since ``r² log r → 0`` as r → 0).

    Implemented as ``0.5 · r² · log(r²)`` (equivalent for r > 0) with a
    ``torch.where`` short-circuit at ``r² == 0`` to avoid the
    ``log(0) = -inf`` × 0 NaN. **Mathematically exact** — no epsilon
    bias. Earlier versions used a ``log(r² + ε)`` shift with ε = 1e-12
    which biased non-coincident kernel values by O(ε log r) ≈ 1e-11
    (well below noise but not exact); this form is exact at every
    finite r including r = 0.
    """
    # log(r²) is undefined at r=0; guard with where so autograd still
    # works (gradient of the where=False branch is masked).
    safe_r2 = torch.where(r2 > 0, r2, torch.ones_like(r2))
    raw = 0.5 * r2 * torch.log(safe_r2)
    # At r=0 the limit is exactly 0 (r² log r² → 0).
    return torch.where(r2 > 0, raw, torch.zeros_like(r2))


class RBFResidualCorrection(nn.Module):
    """Differentiable thin-plate-spline residual correction.

    Parameters
    ----------
    centres :
        Tensor of shape ``(n_centres, 2)`` with (Y, Z) pixel coordinates
        of the spline knots.
    weights :
        Tensor of shape ``(n_centres,)`` of RBF coefficients. Refinable
        if ``trainable_weights=True`` (default).
    affine :
        Optional 3-vector ``(a0, aY, aZ)`` of the linear-trend term
        (``ΔR ~ a0 + aY·Y + aZ·Z``). If None, no linear trend.
    trainable_weights :
        Whether weights / affine are :class:`nn.Parameter`. Defaults True.
    """

    def __init__(
        self,
        centres: torch.Tensor,
        weights: torch.Tensor,
        *,
        affine: Optional[torch.Tensor] = None,
        trainable_weights: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if centres.ndim != 2 or centres.shape[1] != 2:
            raise ValueError(
                f"centres must be (n_centres, 2), got {tuple(centres.shape)}"
            )
        if weights.shape[0] != centres.shape[0]:
            raise ValueError(
                f"weights/centres mismatch: {weights.shape} vs {centres.shape}"
            )
        # Centres are knot positions — held as a buffer so they move with
        # .to(device) but aren't refined.
        self.register_buffer("centres", centres.to(dtype=dtype))
        if trainable_weights:
            self.weights = nn.Parameter(weights.to(dtype=dtype).clone())
        else:
            self.register_buffer("weights", weights.to(dtype=dtype))

        if affine is not None:
            if affine.shape != (3,):
                raise ValueError(f"affine must be shape (3,), got {affine.shape}")
            if trainable_weights:
                self.affine = nn.Parameter(affine.to(dtype=dtype).clone())
            else:
                self.register_buffer("affine", affine.to(dtype=dtype))
        else:
            self.affine = None

    def forward(self, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Evaluate ``ΔR(Y, Z)`` at the given pixel coordinates.

        ``Y`` and ``Z`` may be any broadcastable shape; the output has
        the same shape.
        """
        # Broadcast (n_pix, 2) - (n_centres, 2) → (n_pix, n_centres, 2).
        flat_Y = Y.reshape(-1)
        flat_Z = Z.reshape(-1)
        diff_y = flat_Y.unsqueeze(-1) - self.centres[:, 0].unsqueeze(0)
        diff_z = flat_Z.unsqueeze(-1) - self.centres[:, 1].unsqueeze(0)
        r2 = diff_y * diff_y + diff_z * diff_z
        K = _thin_plate_kernel(r2)             # (n_pix, n_centres)
        out = (K * self.weights.unsqueeze(0)).sum(dim=-1)
        if self.affine is not None:
            a0, aY, aZ = self.affine[0], self.affine[1], self.affine[2]
            out = out + a0 + aY * flat_Y + aZ * flat_Z
        return out.reshape(Y.shape)


class IdentityResidualCorrection(nn.Module):
    """Zero residual correction. Returns ``zeros_like(Y)``."""
    def forward(self, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(Y)


__all__ = ["RBFResidualCorrection", "IdentityResidualCorrection",
           "_thin_plate_kernel"]
