"""Shared scaffolding for v2 pipelines.

Encapsulates the bridge to v1's E-step (we still rely on the proven C-backed
cake build via midas_integrate; the differentiable path is for the M-step
and beyond).
"""
from __future__ import annotations

import warnings
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
class RingQuality:
    """Per-ring measurables from one E-step, and the keep/drop decision."""
    ring_idx: int
    phase: str
    r_ideal_px: float
    n_eta: int                  # fits that survived the per-fit SNRMin cut
    snr_median: float           # median baseline-referenced SNR over those fits
    kept: bool
    reason: str = ""


def ring_quality(fits: "FittedDataset") -> List[RingQuality]:
    """Measure each ring's coverage and contrast from the extracted fits."""
    import numpy as _np
    rid = fits.ring_idx.detach().cpu().numpy()
    snr = (fits.snr_baseline.detach().cpu().numpy()
           if fits.snr_baseline is not None
           else fits.snr.detach().cpu().numpy())
    out: List[RingQuality] = []
    for r in sorted(set(int(v) for v in rid)):
        m = rid == r
        phase = "?"
        r_ideal = float("nan")
        if fits.rt is not None and r < len(fits.rt):
            phase = fits.rt.phase_of(r)
            r_ideal = float(fits.rt.r_ideal_px[r])
        out.append(RingQuality(ring_idx=r, phase=phase, r_ideal_px=r_ideal,
                               n_eta=int(m.sum()),
                               snr_median=float(_np.median(snr[m])) if m.any() else 0.0,
                               kept=True))
    return out


def filter_fits_by_ring_quality(
    fits: "FittedDataset",
    *,
    min_eta_bins: int = 0,
    min_ring_snr: float = 0.0,
    min_rings_kept: int = 4,
    verbose: bool = False,
) -> Tuple["FittedDataset", List[RingQuality]]:
    """Drop whole rings that this exposure does not actually measure.

    A ring table comes from crystallography and says nothing about whether a
    ring is measurable *here*.  Weak, vignetted or grainy rings still yield a
    centroid per η bin, and those centroids are noise the geometry then
    absorbs.  Measured on a real 1-ID CeO2+LaB6 frame, its main effect is
    STABILITY rather than a lower best-iterate: with it the alternating E<->M
    loop converged in two iterations (84.2, 84.4 ue) where the same fit without
    it wandered (91, 72, 139, 154 ue) and its apparent best was a lucky iterate
    the next one undid.  The larger single win on that frame was freezing the
    distortion (181 -> 72 ue), not this filter.

    Unlike ``auto_detect_max_ring``, which returns one radial cutoff, this
    removes rings INDIVIDUALLY: on an interleaved two-phase table a weak LaB6
    ring in the middle must not cap every ring outside it.

    ``min_eta_bins`` is an ABSOLUTE count and therefore scales with
    ``EtaBinSize``: on one real frame the best-covered ring carried 13 fits at
    5 deg bins and ~36 at 2 deg, so a threshold tuned at one binning is not
    portable to the other.  Pick it from the actual distribution
    (:func:`ring_quality` reports it) rather than copying a number.

    Returns ``(filtered_fits, per_ring_report)``.  With both thresholds at 0
    the dataset is returned unchanged.
    """
    import numpy as _np
    report = ring_quality(fits)
    if min_eta_bins <= 0 and min_ring_snr <= 0.0:
        return fits, report

    drop = set()
    for q in report:
        why = []
        if min_eta_bins > 0 and q.n_eta < min_eta_bins:
            why.append(f"only {q.n_eta} η bins < {min_eta_bins}")
        if min_ring_snr > 0.0 and q.snr_median < min_ring_snr:
            why.append(f"SNR {q.snr_median:.1f} < {min_ring_snr:g}")
        if why:
            q.kept = False
            q.reason = "; ".join(why)
            drop.add(q.ring_idx)

    if not drop:
        return fits, report
    # Never let the filter empty the dataset.  It runs on every E-step, so a
    # single bad iterate — where the geometry has wandered and no ring looks
    # sharp — would otherwise abort the whole calibration instead of just
    # scoring badly and being rejected by the best-iterate logic.  Keep the
    # best-ranked rings and say so.
    n_keep = len(report) - len(drop)
    if n_keep < min_rings_kept:
        ranked = sorted(report, key=lambda q: (q.n_eta, q.snr_median),
                        reverse=True)[:min_rings_kept]
        rescued = {q.ring_idx for q in ranked}
        for q in report:
            if q.ring_idx in rescued and not q.kept:
                q.kept = True
                q.reason += "  [kept: floor]"
        drop -= rescued
        warnings.warn(
            f"per-ring quality filter would have left {n_keep} of "
            f"{len(report)} rings (MinEtaBinsPerRing={min_eta_bins}, "
            f"MinRingSNR={min_ring_snr:g}); kept the {len(rescued)} best-ranked "
            "instead. Either the thresholds are too tight for this binning "
            "(they scale with EtaBinSize and Width) or the geometry has "
            "wandered — check the per-iteration strain.",
            RuntimeWarning, stacklevel=2)
    if not drop:
        return fits, report

    rid = fits.ring_idx.detach().cpu().numpy()
    keep = torch.as_tensor(~_np.isin(rid, list(drop)), dtype=torch.bool,
                           device=fits.ring_idx.device)

    def _sel(t):
        return None if t is None else t[keep]

    out = FittedDataset(
        Y_pix=fits.Y_pix[keep], Z_pix=fits.Z_pix[keep],
        ring_idx=fits.ring_idx[keep], snr=fits.snr[keep],
        ring_two_theta_deg=fits.ring_two_theta_deg[keep],
        rho_d=fits.rho_d, weights=_sel(fits.weights),
        panel_idx=_sel(fits.panel_idx), rt=fits.rt,
        ring_d_spacing_A=_sel(fits.ring_d_spacing_A),
        phase_idx=_sel(fits.phase_idx), phase_names=fits.phase_names,
        snr_baseline=_sel(fits.snr_baseline),
    )
    if verbose:
        by_phase: dict = {}
        for q in report:
            if not q.kept:
                by_phase.setdefault(q.phase, []).append(q)
        parts = ", ".join(f"{k} {len(v)}" for k, v in sorted(by_phase.items()))
        print(f"[e-step] ring quality: dropped {len(drop)} of {len(report)} rings "
              f"({parts}); {int(keep.sum())} of {keep.numel()} fits kept",
              flush=True)
        for q in report:
            if not q.kept:
                print(f"           drop ring {q.ring_idx:3d} ({q.phase}, "
                      f"R={q.r_ideal_px:7.1f} px): {q.reason}", flush=True)
    return out, report


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
    # Peak height over a linear baseline / scatter at the window ends.  This is
    # what the per-ring quality filter aggregates; ``snr`` above is peak/mean
    # and keeps its historical meaning for ``SNRMin``.
    snr_baseline: Optional[torch.Tensor] = None


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

    snr_b = torch.tensor([getattr(p, "snr_baseline", 0.0) for p in fits],
                         dtype=dtype, device=device)

    fd = FittedDataset(
        Y_pix=Y, Z_pix=Z, ring_idx=rid, snr=snr,
        ring_two_theta_deg=rtt_per_pt,
        rho_d=torch.as_tensor(rho_d, dtype=dtype, device=device),
        weights=w, rt=rt,
        ring_d_spacing_A=torch.as_tensor(
            rt.d_spacing, dtype=dtype, device=device)[rid],
        phase_idx=phase_idx, phase_names=rt.phase_names,
        snr_baseline=snr_b,
    )
    # Per-ring quality filter.  Applied AFTER extraction because it needs the
    # fits themselves, not the predicted radii — a ring can be perfectly well
    # separated and still be unmeasurable on this exposure.
    fd, _ = filter_fits_by_ring_quality(
        fd,
        min_eta_bins=int(getattr(v1_params, "MinEtaBinsPerRing", 0) or 0),
        min_ring_snr=float(getattr(v1_params, "MinRingSNR", 0.0) or 0.0),
        verbose=verbose,
    )
    return fd


__all__ = ["FittedDataset", "run_estep_v1", "filter_ring_table",
           "ring_table_for", "RingQuality", "ring_quality",
           "filter_fits_by_ring_quality"]
