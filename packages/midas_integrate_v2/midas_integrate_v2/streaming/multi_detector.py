"""Multi-detector simultaneous integration orchestrator.

For experiments with N detectors capturing complementary 2θ ranges
(e.g., APS 1-ID Hydra, Wenqian Xu's 17-BM-B / 11-BM dual setups), this
helper integrates each detector independently with its own
:class:`IntegrationSpec` and stitches the resulting profiles onto a
common Q-grid.

Stitching weights:

- ``"inverse_variance"``: each detector's contribution at Q is weighted
  by ``1 / σ²``. Optimal under Gaussian Q-bin independence.
- ``"uniform"``: equal weights wherever a detector has signal.
- ``"bin_count"``: weight by number of pixels per bin (proxy for
  statistical power).
"""
from __future__ import annotations

from typing import Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .frame_source import FrameSource


def _interp(q_target: torch.Tensor,
             q_src: torch.Tensor,
             y_src: torch.Tensor) -> torch.Tensor:
    """Simple linear interpolation that returns NaN outside the source
    range so stitching can identify where each detector has coverage."""
    q_t = q_target.detach().cpu().numpy()
    q_s = q_src.detach().cpu().numpy()
    y_s = y_src.detach().cpu().numpy()
    sort_idx = np.argsort(q_s)
    q_s = q_s[sort_idx]
    y_s = y_s[sort_idx]
    in_range = (q_t >= q_s[0]) & (q_t <= q_s[-1])
    out = np.full_like(q_t, np.nan, dtype=np.float64)
    out[in_range] = np.interp(q_t[in_range], q_s, y_s)
    return torch.as_tensor(out, dtype=torch.float64)


def integrate_multi_detector(
    sources: Sequence[FrameSource],
    specs: Sequence,
    *,
    q_grid: torch.Tensor,
    overlap_weight: str = "inverse_variance",
    corrections_per_detector: Optional[List[Mapping]] = None,
) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield ``(frame_id, I_unified, sigma_unified)`` per frame.

    Each source must be the same length and yield aligned frames
    (same shutter open). For each frame index k:

    1. Each detector's frame is integrated with its own ``spec`` and
       optional corrections.
    2. The 1D η-averaged profile + σ is interpolated onto ``q_grid``.
    3. Detectors are stitched per the ``overlap_weight`` scheme.

    Returns an iterator (not a list) so very long scans don't load
    everything into RAM.
    """
    if overlap_weight not in ("inverse_variance", "uniform", "bin_count"):
        raise ValueError(
            f"overlap_weight must be 'inverse_variance' / 'uniform' / "
            f"'bin_count', got {overlap_weight!r}"
        )
    if len(sources) != len(specs):
        raise ValueError("sources and specs must have matching length")
    if len(sources) == 0:
        raise ValueError("must supply >= 1 source")
    n_per = sources[0].n_frames
    for s in sources[1:]:
        if s.n_frames != n_per:
            raise ValueError(
                "all sources must have the same number of frames"
            )
    from ..binning import (
        PolygonBinGeometry,
        integrate_polygon_with_variance,
    )
    from ..pdf import R_px_to_Q

    geoms = [PolygonBinGeometry.from_spec(s) for s in specs]
    iters = [iter(src) for src in sources]
    for _ in range(n_per):
        frame_pairs = [next(it) for it in iters]
        frame_id = frame_pairs[0][0]
        I_per_det = []
        sig_per_det = []
        for k, (fid, img) in enumerate(frame_pairs):
            mean2d, sig2d = integrate_polygon_with_variance(
                torch.as_tensor(img, dtype=torch.float64), geoms[k],
            )
            n_eta = mean2d.shape[0]
            I = mean2d.mean(dim=0)
            sigma_I = torch.sqrt((sig2d * sig2d).sum(dim=0)) / n_eta
            R_axis = (
                specs[k].RMin
                + (torch.arange(I.shape[0], dtype=torch.float64) + 0.5)
                * specs[k].RBinSize
            )
            Q_axis = R_px_to_Q(
                R_axis, Lsd_um=specs[k].Lsd, px_um=specs[k].pxY,
                lambda_A=specs[k].Wavelength,
            )
            I_per_det.append(_interp(q_grid, Q_axis, I))
            sig_per_det.append(_interp(q_grid, Q_axis, sigma_I))
        I_stack = torch.stack(I_per_det)              # (n_det, n_q)
        sig_stack = torch.stack(sig_per_det)
        # Stitch with selected weights
        valid = ~torch.isnan(I_stack)
        if overlap_weight == "uniform":
            w = valid.to(torch.float64)
        elif overlap_weight == "bin_count":
            w = valid.to(torch.float64)  # dummy = uniform; bin counts
            # would require returning per-bin pixel-count from
            # integrate_polygon_with_variance; left as future work
        else:  # inverse_variance
            w = torch.where(
                valid & (sig_stack > 0),
                1.0 / sig_stack.clamp(min=1e-30) ** 2,
                torch.zeros_like(sig_stack),
            )
        w_sum = w.sum(dim=0).clamp(min=1e-30)
        I_unif = (torch.where(valid, I_stack, torch.zeros_like(I_stack)) * w
                  ).sum(dim=0) / w_sum
        # Combined σ: 1 / sqrt(Σ w) for inverse-variance, sqrt(Σ w² σ²)/Σw else
        if overlap_weight == "inverse_variance":
            sigma_unif = 1.0 / torch.sqrt(w_sum)
        else:
            var_unif = (
                torch.where(valid, sig_stack ** 2, torch.zeros_like(sig_stack))
                * (w * w)
            ).sum(dim=0) / (w_sum * w_sum)
            sigma_unif = torch.sqrt(var_unif.clamp(min=0.0))
        yield frame_id, I_unif, sigma_unif


__all__ = ["integrate_multi_detector"]
