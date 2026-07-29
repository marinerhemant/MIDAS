"""Recovery metrics for the synthetic plant-and-recover validation.

Two main entry points:
  recovery_metrics(plant, fit) — per-voxel orientation/strain RMSE
  holdout_score(...) — convenience around fit_grain_peakshape's score

Plot helpers are deliberately lightweight (matplotlib only, no seaborn);
they save PNGs to a caller-supplied path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from midas_grain_odf.odf import matrix_to_axis_angle

from midas_pf_odf.simulate import SinglePhaseGrainPlant


@dataclass
class RecoveryReport:
    misorient_rad: torch.Tensor       # (G,) angular distance R_fit ↔ R_plant
    eps_residual: torch.Tensor        # (G, 6) ε_fit − ε_plant
    eps_rms_per_voxel: torch.Tensor   # (G,) Frobenius RMSE over Voigt comps
    eps_rms: float                    # bulk RMSE
    misorient_rms_deg: float
    eps_rms_per_component: torch.Tensor   # (6,)
    metadata: dict = field(default_factory=dict)


def _R_to_misorient_rad(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """Per-voxel rotation angle between two (G, 3, 3) stacks, in radians.

    Uses a numerically stable formula based on ``2 sin(θ/2) = ‖R_b − R_a‖_F``
    so that R_a == R_b returns exactly 0 (the trace + acos route biases by
    ~ sqrt(2 ε) near the identity).
    """
    diff = R_b - R_a
    s = 0.5 * diff.flatten(-2).norm(dim=-1)        # = sin(θ/2) for proper rotations
    s = s.clamp(max=1.0)
    return 2.0 * torch.asin(s)


def recovery_metrics(
    plant: SinglePhaseGrainPlant,
    R_fit: torch.Tensor,                 # (G, 3, 3)
    eps_fit: torch.Tensor,               # (G, 6)
) -> RecoveryReport:
    """Compare a fitted ``(R_V, ε_V)`` against the planted truth."""
    R_plant = plant.R_voxel.to(R_fit.dtype).to(R_fit.device)
    eps_plant = plant.eps_voxel.to(eps_fit.dtype).to(eps_fit.device)

    miso = _R_to_misorient_rad(R_plant, R_fit)              # (G,)
    eps_res = eps_fit - eps_plant                           # (G, 6)
    eps_rms_per_voxel = torch.sqrt((eps_res ** 2).mean(dim=-1))   # (G,)
    eps_rms = float(torch.sqrt((eps_res ** 2).mean()).item())
    eps_rms_per_component = torch.sqrt((eps_res ** 2).mean(dim=0))
    miso_rms_deg = float(
        torch.sqrt((miso ** 2).mean()).item() * 180.0 / math.pi
    )

    return RecoveryReport(
        misorient_rad=miso.detach(),
        eps_residual=eps_res.detach(),
        eps_rms_per_voxel=eps_rms_per_voxel.detach(),
        eps_rms=eps_rms,
        misorient_rms_deg=miso_rms_deg,
        eps_rms_per_component=eps_rms_per_component.detach(),
        metadata={"n_voxels": int(R_fit.shape[0])},
    )


def holdout_score(fit_result) -> Optional[float]:
    """Return the held-out 1 − SSE/SST score from a fit, if computed."""
    return getattr(fit_result, "holdout_score", None)


def plot_recovery(
    plant: SinglePhaseGrainPlant,
    R_fit: torch.Tensor,
    eps_fit: torch.Tensor,
    output_path: str,
    *,
    eps_voigt: int = 0,
    title: Optional[str] = None,
) -> None:
    """Multi-panel recovery diagnostic.

    Saves a PNG with:
      - planted ε_{eps_voigt}(x, y) map
      - recovered ε_{eps_voigt}(x, y)
      - per-voxel misorientation (deg)
      - per-voxel ε RMSE
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Gx, Gy = plant.grid_shape
    eps_p = plant.eps_voxel[:, eps_voigt].reshape(Gx, Gy).cpu().numpy()
    eps_r = eps_fit[:, eps_voigt].reshape(Gx, Gy).cpu().numpy()
    miso_deg = (_R_to_misorient_rad(plant.R_voxel.to(R_fit.dtype).to(R_fit.device),
                                     R_fit) * 180.0 / math.pi
                ).reshape(Gx, Gy).cpu().numpy()
    eps_rms_v = torch.sqrt(
        ((eps_fit - plant.eps_voxel.to(eps_fit.dtype).to(eps_fit.device)) ** 2).mean(dim=-1)
    ).reshape(Gx, Gy).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    im0 = axes[0, 0].imshow(eps_p, origin="lower")
    axes[0, 0].set_title(f"ε_V[{eps_voigt}] planted")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(eps_r, origin="lower")
    axes[0, 1].set_title(f"ε_V[{eps_voigt}] recovered")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(miso_deg, origin="lower")
    axes[1, 0].set_title("misorientation (deg)")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(eps_rms_v, origin="lower")
    axes[1, 1].set_title("per-voxel ε RMSE")
    plt.colorbar(im3, ax=axes[1, 1])

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
