"""Shared scaffolding for v2 pipelines.

Encapsulates the bridge to v1's E-step (we still rely on the proven C-backed
cake build via midas_integrate; the differentiable path is for the M-step
and beyond).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from midas_calibrate.estep import run_estep
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.refine import FittedPoint
from midas_calibrate.rings import RingTable, build_ring_table

from ..parameters.spec import CalibrationSpec


def filter_ring_table(rt: RingTable, *,
                       rings_to_exclude=(),
                       max_ring_number: int = 0) -> RingTable:
    """Apply v1-style ring-filtering to a RingTable.

    Returns a new ``RingTable`` with rows removed where:
      - ``rt.ring_nr`` is in ``rings_to_exclude``, OR
      - ``rt.ring_nr > max_ring_number`` (when ``max_ring_number > 0``).
    """
    import numpy as _np
    if not rings_to_exclude and max_ring_number <= 0:
        return rt
    keep = _np.ones(len(rt.ring_nr), dtype=bool)
    if rings_to_exclude:
        excl = set(int(r) for r in rings_to_exclude)
        keep &= ~_np.isin(rt.ring_nr, list(excl))
    if max_ring_number > 0:
        keep &= rt.ring_nr <= max_ring_number
    new = RingTable(
        ring_nr=rt.ring_nr[keep],
        h=rt.h[keep], k=rt.k[keep], l=rt.l[keep],
        d_spacing=rt.d_spacing[keep],
        two_theta_deg=rt.two_theta_deg[keep],
        multiplicity=rt.multiplicity[keep],
        r_ideal_px=rt.r_ideal_px[keep],
    )
    return new


@dataclass
class FittedDataset:
    """Bundle of E-step outputs in torch form."""

    Y_pix: torch.Tensor             # [n_pts]
    Z_pix: torch.Tensor
    ring_idx: torch.Tensor          # long
    snr: torch.Tensor
    ring_two_theta_deg: torch.Tensor   # [n_pts] expected 2θ at the ring
    rho_d: torch.Tensor                # px
    weights: Optional[torch.Tensor] = None
    panel_idx: Optional[torch.Tensor] = None
    rt: Optional[RingTable] = None
    # Per-fit ring d-spacing (Å); when populated AND Wavelength is in the
    # spec, pseudo_strain_residual recomputes 2θ inside via Bragg so the
    # autograd chain through λ stays unbroken.  Pinned-Wavelength callers
    # may leave this None (the ring_two_theta_deg constant is used).
    ring_d_spacing_A: Optional[torch.Tensor] = None


def run_estep_v1(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    dtype=torch.float64, device="cpu",
) -> FittedDataset:
    """Run v1's proven E-step and return a v2-friendly FittedDataset."""
    rt = build_ring_table(v1_params)
    cake, fits = run_estep(v1_params, image, rt, dark=dark)
    if not fits:
        raise RuntimeError("E-step produced no fitted points")

    Y = torch.tensor([p.Y_pix for p in fits], dtype=dtype, device=device)
    Z = torch.tensor([p.Z_pix for p in fits], dtype=dtype, device=device)
    rid = torch.tensor([p.ring_idx for p in fits], dtype=torch.long, device=device)
    snr = torch.tensor([p.snr for p in fits], dtype=dtype, device=device)

    rt_tt = torch.tensor(rt.two_theta_deg, dtype=dtype, device=device)
    rtt_per_pt = rt_tt[rid]

    px = 0.5 * (v1_params.pxY + v1_params.pxZ) if v1_params.pxZ > 0 else v1_params.pxY
    rho_d = v1_params.RhoD if v1_params.RhoD > 0 else v1_params.MaxRingRad

    # Default ring + SNR weights, matching v1.
    w = torch.ones_like(snr)
    if v1_params.WeightBySNR:
        med = snr.median().clamp(min=1e-6)
        w = w * (snr / med).clamp(min=0.1, max=10.0)

    return FittedDataset(
        Y_pix=Y, Z_pix=Z, ring_idx=rid, snr=snr,
        ring_two_theta_deg=rtt_per_pt,
        rho_d=torch.as_tensor(rho_d, dtype=dtype, device=device),
        weights=w, rt=rt,
    )


__all__ = ["FittedDataset", "run_estep_v1", "filter_ring_table"]
