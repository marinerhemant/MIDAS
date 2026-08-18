"""Shared scaffolding for v2 pipelines.

Encapsulates the bridge to v1's E-step (we still rely on the proven C-backed
cake build via midas_integrate; the differentiable path is for the M-step
and beyond).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from midas_calibrate.estep import run_estep
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.refine import FittedPoint
from midas_calibrate.rings import RingTable, build_ring_table, drop_blended_rings

from ..parameters.spec import CalibrationSpec


def filter_ring_table(rt: RingTable, *,
                       rings_to_exclude=(),
                       max_ring_number: int = 0) -> RingTable:
    """Apply v1-style ring-filtering to a RingTable.

    Returns a new ``RingTable`` with rows removed where:
      - ``rt.ring_nr`` is in ``rings_to_exclude``, OR
      - ``rt.ring_nr > max_ring_number`` (when ``max_ring_number > 0``).

    Uses ``RingTable.select`` so multi-phase columns (``phase_idx``,
    ``hkl_aliases``) survive the filter instead of being silently dropped.
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
    return rt.select(keep)


def ring_table_for(
    v1_params: V1Params,
    *,
    spec: Optional[CalibrationSpec] = None,
    verbose: bool = False,
) -> RingTable:
    """Build the ring table for an E-step, with every exclusion applied.

    One place decides which rings a calibration actually sees:

    * ``spec.rings_to_exclude`` / ``spec.max_ring_number`` (v1's
      ``RingsToExclude`` / ``MaxRingNumber``).  These used to be honoured only
      by the pseudo-Voigt pipelines -- the default centroid path built the
      table raw, so a user who set them and ran ``calibrate()`` had them
      silently ignored.
    * ``v1_params.MinRingSeparation`` -- drop rings that collide in radius.
      This is what a mixed-calibrant exposure needs; see
      :func:`midas_calibrate.rings.drop_blended_rings`.
    """
    rt = build_ring_table(v1_params)
    n0 = len(rt)
    if spec is not None:
        rt = filter_ring_table(
            rt,
            rings_to_exclude=getattr(spec, "rings_to_exclude", ()),
            max_ring_number=getattr(spec, "max_ring_number", 0),
        )
    min_sep = float(getattr(v1_params, "MinRingSeparation", 0.0) or 0.0)
    n_blend = 0
    if min_sep > 0:
        rt, n_blend = drop_blended_rings(
            rt, min_separation_px=min_sep,
            cross_phase_only=bool(getattr(
                v1_params, "BlendExcludeCrossPhaseOnly", False)),
        )
    if verbose and len(rt) != n0:
        extra = f", {n_blend} of them blended within {min_sep:g} px" if n_blend else ""
        print(f"[e-step] ring table: {n0} -> {len(rt)} rings{extra}", flush=True)
    if len(rt) == 0:
        raise RuntimeError(
            "ring filtering removed every ring — check RingsToExclude / "
            "MaxRingNumber / MinRingSeparation")
    return rt


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
    # Which calibrant produced each fitted point.  Enables the per-phase
    # residual breakdown, which is the whole reason to shoot two calibrants:
    # without it you cannot see the two disagreeing.
    phase_idx: Optional[torch.Tensor] = None       # long [n_pts]
    phase_names: Tuple[str, ...] = ()


def run_estep_v1(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    dtype=torch.float64, device="cpu",
    verbose: bool = False,
) -> FittedDataset:
    """Run v1's proven E-step and return a v2-friendly FittedDataset."""
    rt = ring_table_for(v1_params, spec=spec, verbose=verbose)
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
    # RhoD is µm (forward distortion uses ρ = R_um / RhoD); MaxRingRad
    # fallback is px and must be scaled to µm.
    rho_d = v1_params.RhoD if v1_params.RhoD > 0 else v1_params.MaxRingRad * px

    # Default ring + SNR weights, matching v1.
    w = torch.ones_like(snr)
    if v1_params.WeightBySNR:
        med = snr.median().clamp(min=1e-6)
        w = w * (snr / med).clamp(min=0.1, max=10.0)

    phase_idx = None
    if rt.phase_idx is not None:
        phase_idx = torch.as_tensor(rt.phase_idx, dtype=torch.long,
                                    device=device)[rid]

    return FittedDataset(
        Y_pix=Y, Z_pix=Z, ring_idx=rid, snr=snr,
        ring_two_theta_deg=rtt_per_pt,
        rho_d=torch.as_tensor(rho_d, dtype=dtype, device=device),
        weights=w, rt=rt,
        ring_d_spacing_A=torch.as_tensor(
            rt.d_spacing, dtype=dtype, device=device)[rid],
        phase_idx=phase_idx, phase_names=rt.phase_names,
    )


__all__ = ["FittedDataset", "run_estep_v1", "filter_ring_table", "ring_table_for"]
