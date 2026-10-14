"""Single-image v2 pipeline with batched pseudo-Voigt peak extraction.

Replaces the v1 cake-centroid E-step with a batched 1-D pV LM that fits the
peak center per (ring, η-bin), matching the C engine's per-spot peak-fit
fidelity.  This is the path that closes the strain-floor gap to v1 C.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from midas_calibrate.estep import integrate_cake
from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.rings import RingTable, build_ring_table
from midas_integrate.geometry import build_tilt_matrix, invert_REta_to_pixel_batch
from midas_peakfit import GenericLMConfig

from ..compat.from_v1 import spec_from_v1_params
from ..io.transforms import apply_im_trans
from ..forward.panels import PanelLayout, invert_panel_shifts
from ..forward.peak_fit_batched import fit_cake_per_ring_batched, BatchedFits
from ..forward.peak_fit_doublet import fit_doublet_pairs, DoubletFits
from ..forward.doublets import doublet_index_map
from ..inference.lm import lm_minimise
from ..loss.diagnostics import strain_summary
from ..loss.pseudo_strain import pseudo_strain_residual
from ..loss.robust_trim import stratified_trim, azimuth_deg, evaluate_full_strain
from ..parameters.spec import CalibrationSpec
from ..seed.auto_max_ring import auto_detect_max_ring
from ..seed.mask import apply_sentinel_mask
from ._common import (FittedDataset, _filter_by_snr, filter_ring_table,
                      ring_number_cap, ring_table_for)


@dataclass
class IterRecord:
    iteration: int
    n_fitted: int
    cost: float
    rc: int
    mean_strain_uE: float
    Lsd: float
    BC_y: float
    BC_z: float
    ty: float
    tz: float


@dataclass
class PVCalibrationResult:
    spec: CalibrationSpec
    unpacked: dict
    history: List[IterRecord]
    fits_final: Optional[FittedDataset] = None
    #: Iterations run in the capture window, before ``history`` (which holds the
    #: fine-window iterations only, as it always has).
    capture_history: List[IterRecord] = field(default_factory=list)
    #: One record per clipped-window screen that ran: ``phase``, ``iteration``,
    #: ``n_snr_pass`` (fits that cleared ``snr_min`` before the screen) and
    #: ``n_kept`` (of those, how many the screen left). Feeds
    #: ``pipelines.diagnostics.fit_retention_gate``.
    retention: List[dict] = field(default_factory=list)
    #: Why the fine-phase outer loop stopped: ``"settled"`` (strain stable AND the
    #: geometry moved less than ``param_tol_px`` over the last two iterations),
    #: ``"strain_floor"``, ``"strain_rising"``, ``"iteration_cap"`` or
    #: ``"strain_stable"`` (legacy rule, ``param_tol_px=None``).
    stop_reason: str = ""
    #: ``True`` only for ``"settled"``. NOT a statement that another seed would end
    #: here: on the Pilatus frame two seeds 100 um apart both settled (steps 0.045
    #: and 0.032 px) and ended 316 um apart. The legacy ``"strain_stable"`` stop says
    #: nothing about the parameters and reports ``False``.
    settled: bool = False
    #: Largest geometry step over the last (up to) two outer iterations, as a ring
    #: radius shift in pixels at the outermost fitted ring (see
    #: :func:`_geometry_step_px`). ``None`` with fewer than two fine iterations.
    final_step_px: Optional[float] = None


class GeometryNotSettledWarning(UserWarning):
    """The outer loop ended while the geometry was still moving by more than
    ``param_tol_px`` at the outermost ring. Any sigma quoted for this fit is
    conditional on the final iterate and says nothing about where the loop was
    heading. Seen on a Pilatus whose Lsd moved 49, 12, 76 um on the last three
    iterations and, from a seed shifted by 100 um, ended 248 um away (23 sigma).
    Settling is necessary, not sufficient: see ``PVCalibrationResult.settled``."""


class CaptureRangeWarning(UserWarning):
    """The final fine-window peak fits pile up at the window edge: the rings
    are further from the returned geometry than the window reaches."""


class ClippedWindowWarning(UserWarning):
    """Every fine-window fit sits in a radial window cut by a panel edge, gap
    or mask, so none could be dropped without leaving nothing to fit. The fits
    are truncated peaks and the geometry that follows is not trustworthy."""


class CaptureNotSettledWarning(UserWarning):
    """The capture phase ended because it ran out of iterations (``n_capture_max``),
    not because the rings stopped moving. The fine phase then starts from a geometry
    that is still far from the rings. Seen on a real Varex frame with a 14 degree
    tilt: with ten capture iterations and no distortion terms the blind start stopped
    2.0 degrees short in tz at 793 microstrain with nothing but a verbose line to say so;
    twenty iterations reached 0.05 degrees at 29 microstrain."""


def _capture_cap_message(shift_px: float, capture_shift_px: float, n_done: int,
                         n_capture_max: int):
    """Warning text when the capture phase ended on its iteration cap with the
    predicted rings still moving by at least ``capture_shift_px``; ``None`` otherwise."""
    if shift_px < capture_shift_px or n_done < n_capture_max:
        return None
    return (f"autocalibrate_pv: the capture phase used all n_capture_max={n_capture_max} "
            f"iterations and the last LM step still moved the predicted rings by "
            f"{shift_px:.2f} px (>= capture_shift_px={capture_shift_px:g}). The fine phase "
            f"starts from a geometry that has not reached the rings, and can settle on a "
            f"wrong one at a low strain. Raise n_capture_max, or start from a closer seed.")


class CoarseCakeBinWarning(UserWarning):
    """The radial cake bin of the parameter set is coarse. On the Varex CeO2 frame the kept
    strain was 49, 20.1, 7.42, 7.30 and 7.50 microstrain for RBinSize 2, 1, 0.5, 0.25 and
    0.125 px with everything else equal; a parameter file with ``RBinSize 1`` therefore gave
    a calibration nearly three times worse than the same data at 0.25 px, with no message."""


def _coarse_cake_bin_message(rbin_px: float, limit_px: float = 0.5):
    """Warning text when ``rbin_px`` exceeds ``limit_px`` (0.5 px: the first value at which the
    Varex kept strain was within 2 % of its 0.25 px value); ``None`` otherwise (0 means unset)."""
    if not (rbin_px and rbin_px > limit_px):
        return None
    return (f"autocalibrate_pv: RBinSize={rbin_px:g} px is coarse for the peak fits; on the Varex CeO2 "
            f"frame the kept strain was 20.1 uE at 1 px against 7.30 uE at 0.25 px. Set RBinSize to "
            f"0.25 (or smaller) in the parameter set before calibrating.")


class FitRetentionWarning(UserWarning):
    """The clipped-window screen removed most of the fits that had cleared the
    SNR cut. What is left is a thin, unrepresentative sample of the rings, and
    the geometry that follows is not trustworthy even though the run converges
    and reports a strain. Seen on a frame whose dark exceeded its image (99.8 %
    of the candidate fits dropped, 2 rings left) and on a multi-panel frame
    whose gap pixels had been clamped into valid data."""


def _screen_retention(bf, keep, snr_min) -> tuple:
    """``(n_snr_pass, n_kept)``: of the fits that cleared ``snr_min``, how many
    the clipped-window screen left. Counted over SNR-passing fits only, because
    the fitter returns a fit for every (ring, eta) cell, off-detector ones
    included, and those would swamp the ratio."""
    good = (bf.snr >= float(snr_min)) if snr_min is not None else torch.ones_like(keep)
    return int(good.sum()), int((good & keep).sum())


def _geometry_step_px(prev, cur, R_outer_px: float, px_um: float) -> float:
    """Largest first-order shift of the outermost fitted ring between two outer
    iterations, in pixels, from the change in each geometry parameter.

    * ``Lsd``: ``R * dLsd / Lsd`` (exact, ``R = Lsd * tan 2theta``).
    * beam centre: ``hypot(dBC_y, dBC_z)``.
    * tilts: ``R * tan(2theta) * d(angle)``, the small-tilt effect of a rotation of
      the detector plane on a ring of radius ``R`` (``tan 2theta = R * px / Lsd``).

    A convention, not a calibrated tolerance: it puts the three parameters on the
    one scale the data sees, a displacement of the rings.
    """
    import math
    R = float(R_outer_px)
    lsd = float(cur.Lsd)
    d_lsd = R * abs(float(cur.Lsd) - float(prev.Lsd)) / lsd
    d_bc = math.hypot(float(cur.BC_y) - float(prev.BC_y), float(cur.BC_z) - float(prev.BC_z))
    tan2t = R * float(px_um) / lsd
    d_t = max(abs(float(cur.ty) - float(prev.ty)), abs(float(cur.tz) - float(prev.tz)))
    return max(d_lsd, d_bc, R * tan2t * math.radians(d_t))


def _outer_loop_decision(history, R_outer_px: float, px_um: float, *,
                         param_tol_px: Optional[float],
                         strain_rtol: float = 0.01):
    """``(stop, reason, step_px)`` after the newest fine iteration.

    ``param_tol_px=None`` is the legacy rule: stop as soon as the strain changes by
    less than ``strain_rtol`` (or falls below 1 uE, or rises by 5 %). That rule
    watches the objective, not the parameters, and can stop while Lsd is still
    climbing by hundreds of um per iteration. With a tolerance, a stable strain
    stops the loop only when the geometry step over the last (up to) two
    iterations is also below it. A strain floor or a rising strain still stops it
    (the old behaviour: keep the best), but is not reported as settled.
    """
    if len(history) < 2:
        return False, "", None
    prev, cur = history[-2].mean_strain_uE, history[-1].mean_strain_uE
    steps = [_geometry_step_px(history[i - 1], history[i], R_outer_px, px_um)
             for i in range(max(1, len(history) - 2), len(history))]
    step = max(steps)
    floor = cur < 1.0
    rising = cur > prev * 1.05
    stable = abs(prev - cur) < strain_rtol * max(prev, 1.0)
    if param_tol_px is None:
        if floor or rising or stable:
            return True, "strain_stable", step
        return False, "", step
    if floor:
        return True, "strain_floor", step
    if rising:
        return True, "strain_rising", step
    if stable and step < param_tol_px:
        return True, "settled", step
    return False, "", step


def _note_retention(log, phase, iteration, n_pass, n_kept, min_retention,
                    stacklevel=3) -> None:
    if log is not None:
        log.append(dict(phase=phase, iteration=int(iteration),
                        n_snr_pass=int(n_pass), n_kept=int(n_kept)))
    if n_pass > 0 and (n_kept / n_pass) < float(min_retention):
        warnings.warn(FitRetentionWarning(
            f"the clipped-window screen kept {n_kept} of {n_pass} SNR-passing "
            f"fits ({100.0 * n_kept / n_pass:.1f} %) in the {phase} phase. The "
            f"remaining fits are too few to trust. Check for gap or dead-pixel "
            f"sentinels that were clamped instead of masked, a dark frame whose "
            f"exposure does not match the image, or a window wider than the "
            f"ring spacing."), stacklevel=stacklevel)


def _bake_fits_to_dataset(fits: BatchedFits, v1: V1Params, rt: RingTable,
                           dtype, device,
                           panel_layout: Optional[PanelLayout] = None,
                           panel_delta_yz: Optional[torch.Tensor] = None,
                           panel_delta_theta: Optional[torch.Tensor] = None,
                           fix_panel_id: int = 0) -> FittedDataset:
    """Convert (R_fit, eta_deg) → (Y_pix, Z_pix) via current geometry.

    The inversion runs in two stages: (a) the panel-unaware inverse
    projection from ``midas_integrate``, which gives post-panel-shift
    coords; (b) the inverse panel rigid-body, which gives the pre-shift
    coords the M-step's panel-aware forward model expects.  Without (b),
    the round-trip ``forward(inverse(R, η)) == R, η`` only holds at the
    reference panel and the strain residual on all other panels is
    dominated by the missing inverse.
    """
    px = 0.5 * (v1.pxY + v1.pxZ) if v1.pxZ > 0 else v1.pxY
    rho_d = v1.RhoD if v1.RhoD > 0 else v1.MaxRingRad * px   # µm; MaxRingRad is px
    TRs = build_tilt_matrix(v1.tx, v1.ty, v1.tz)

    R = fits.R_fit.detach().cpu().numpy()
    Eta = fits.eta_deg.detach().cpu().numpy()
    p_kw = {f"p{i}": float(getattr(v1, f"p{i}")) for i in range(15)}
    Y_pix, Z_pix = invert_REta_to_pixel_batch(
        R, Eta,
        Ycen=v1.BC_y, Zcen=v1.BC_z, TRs=TRs,
        Lsd=v1.Lsd, RhoD=rho_d, px=px, parallax=v1.Parallax,
        **p_kw,
    )

    rt_tt = torch.tensor(rt.two_theta_deg, dtype=dtype, device=device)
    panel_idx = None
    Y_t = torch.tensor(Y_pix, dtype=dtype, device=device)
    Z_t = torch.tensor(Z_pix, dtype=dtype, device=device)
    if panel_layout is not None and panel_layout.panel_index_mask is not None:
        mask_np = panel_layout.panel_index_mask.cpu().numpy()
        H, W = mask_np.shape
        Yi = np.clip(np.round(Y_pix).astype(int), 0, H - 1)
        Zi = np.clip(np.round(Z_pix).astype(int), 0, W - 1)
        panel_idx = torch.tensor(mask_np[Yi, Zi], dtype=torch.long, device=device)

        # Apply the inverse panel rigid-body so the M-step's panel-aware
        # forward model reproduces R_fit at these (Y, Z).  delta_yz and
        # delta_theta default to zero if not supplied — equivalent to
        # treating the inversion as panel-aware iff the panels are at
        # zero-shift already.
        if panel_delta_yz is None:
            panel_delta_yz = torch.zeros(panel_layout.n_panels(), 2,
                                          dtype=dtype, device=device)
        if panel_delta_theta is None:
            panel_delta_theta = torch.zeros(panel_layout.n_panels(),
                                              dtype=dtype, device=device)
        Y_t, Z_t = invert_panel_shifts(
            Y_t, Z_t, panel_idx, panel_layout,
            panel_delta_yz, panel_delta_theta,
            fix_panel_id=fix_panel_id,
        )
    rt_d = torch.as_tensor(rt.d_spacing, dtype=dtype, device=device)
    return FittedDataset(
        Y_pix=Y_t,
        Z_pix=Z_t,
        ring_idx=fits.ring_idx,
        snr=fits.snr,
        ring_two_theta_deg=rt_tt[fits.ring_idx],
        rho_d=torch.as_tensor(rho_d, dtype=dtype, device=device),
        weights=torch.ones(R.shape[0], dtype=dtype, device=device),
        panel_idx=panel_idx,
        rt=rt,
        ring_d_spacing_A=rt_d[fits.ring_idx],
    )


def _stratified_subsample(fits_ds: FittedDataset, max_per_ring: int,
                            seed: int = 0) -> FittedDataset:
    rng = np.random.default_rng(seed)
    rid = fits_ds.ring_idx.cpu().numpy()
    keep = []
    for ring in np.unique(rid):
        idx = np.where(rid == ring)[0]
        if len(idx) > max_per_ring:
            idx = rng.choice(idx, size=max_per_ring, replace=False)
        keep.append(idx)
    keep = torch.tensor(np.sort(np.concatenate(keep)), dtype=torch.long)
    return FittedDataset(
        Y_pix=fits_ds.Y_pix[keep],
        Z_pix=fits_ds.Z_pix[keep],
        ring_idx=fits_ds.ring_idx[keep],
        snr=fits_ds.snr[keep],
        ring_two_theta_deg=fits_ds.ring_two_theta_deg[keep],
        rho_d=fits_ds.rho_d,
        weights=(fits_ds.weights[keep] if fits_ds.weights is not None else None),
        panel_idx=(fits_ds.panel_idx[keep] if fits_ds.panel_idx is not None else None),
        rt=fits_ds.rt,
        ring_d_spacing_A=(fits_ds.ring_d_spacing_A[keep]
                           if fits_ds.ring_d_spacing_A is not None else None),
    )


#: A capture window is at most this fraction of the gap to the nearest other
#: ring. Below half the gap, a window that holds its own ring cannot also hold a
#: neighbour's centre; 0.45 leaves room for the peak width.
_CAPTURE_GAP_FRAC = 0.45

_BF_FIELDS = ("R_fit", "eta_deg", "ring_idx", "sigma", "gamma", "area", "snr", "rms", "rc")


def _capture_windows_px(r_ideal_px, *, capture_window_px: float,
                        fine_window_px: float) -> np.ndarray:
    """Per-ring capture half-window: ``capture_window_px``, capped at
    ``_CAPTURE_GAP_FRAC`` of the gap to the nearest other ring, never narrower
    than the fine window."""
    r = np.asarray(r_ideal_px, dtype=float)
    if r.size < 2:
        gap = np.full(r.shape, np.inf)
    else:
        d = np.abs(r[:, None] - r[None, :])
        np.fill_diagonal(d, np.inf)
        gap = d.min(axis=1)
    return np.clip(_CAPTURE_GAP_FRAC * gap, fine_window_px, capture_window_px)


def _capture_window_complete(bf: BatchedFits, cake_t, R_centers, eta_centers, rt_R_ideal,
                             wins, coverage, min_cell_coverage: float) -> torch.Tensor:
    """Per fit: is its radial window whole? The same two tests the centroid
    E-step applies to its windows (``midas_calibrate.estep.extract_fitted_points``):
    no cell reads <= 0, and every cell reaches ``min_cell_coverage`` of the
    best-covered cell in that window. The windows are rebuilt exactly as
    ``fit_cake_per_ring_batched`` builds them."""
    n_R = R_centers.numel()
    dR = float(R_centers[1] - R_centers[0])
    cov = torch.as_tensor(coverage, dtype=cake_t.dtype, device=cake_t.device)
    complete = torch.zeros((rt_R_ideal.numel(), eta_centers.numel()), dtype=torch.bool,
                           device=cake_t.device)
    for k in torch.unique(bf.ring_idx).tolist():
        n_win = max(7, int(round(2.0 * float(wins[k]) / dR)))
        c = int(torch.argmin((R_centers - rt_R_ideal[k]).abs()))
        lo = max(0, c - n_win // 2)
        hi = lo + n_win
        if hi > n_R:
            hi = n_R
            lo = hi - n_win
        I_blk, C_blk = cake_t[lo:hi, :], cov[lo:hi, :]
        ref = C_blk.max(dim=0).values
        ref = torch.where(ref > 0, ref, torch.ones_like(ref))
        complete[k] = ~(I_blk <= 0).any(dim=0) & (C_blk >= min_cell_coverage * ref).all(dim=0)
    j = torch.argmin((bf.eta_deg[:, None] - eta_centers[None, :]).abs(), dim=1)
    return complete[bf.ring_idx, j]


def _fit_capture_windows(cake_t, R_centers, eta_centers, rt_R_ideal, *,
                         capture_window_px: float, fine_window_px: float,
                         coverage=None, min_cell_coverage: float = 0.5,
                         verbose: bool = False, retention_log=None,
                         iteration: int = 0, min_retention: float = 0.5,
                         **fit_kw) -> BatchedFits:
    """Capture-phase peak fits. The batched fitter takes one window for every
    ring, so rings are grouped by (whole-px) window and fitted per group, with
    ``ring_idx`` mapped back to the full ring table.

    With ``coverage`` (the cake's per-cell coverage map), fits whose window is
    clipped by a detector edge, module gap or mask are dropped. A wide window
    meets an edge far more often than the fine one, and a clipped window fits
    the edge: measured on an off-panel Eiger frame, a ring 14 px beyond the
    panel edge gave 6 fits at +23 px, on the tail of its off-panel peak, which
    passed the SNR cut and pulled BC_y 20 px onto its bound."""
    wins = _capture_windows_px(rt_R_ideal.detach().cpu().numpy(),
                               capture_window_px=capture_window_px,
                               fine_window_px=fine_window_px)
    wins = np.maximum(np.floor(wins), fine_window_px)
    parts = []
    for w in np.unique(wins):
        idx = torch.as_tensor(np.flatnonzero(wins == w), dtype=torch.long,
                              device=rt_R_ideal.device)
        bf = fit_cake_per_ring_batched(cake_t, R_centers, eta_centers,
                                       rt_R_ideal[idx], half_window_px=float(w),
                                       init_center="peak", verbose=False, **fit_kw)
        if bf.R_fit.numel() == 0:
            continue
        parts.append(BatchedFits(**{f: (idx[bf.ring_idx] if f == "ring_idx"
                                        else getattr(bf, f)) for f in _BF_FIELDS}))
    if verbose:
        groups = ", ".join(f"±{w:g}px x{int((wins == w).sum())}" for w in np.unique(wins))
        print(f"  [pv capture] per-ring windows: {groups}", flush=True)
    if not parts:
        return fit_cake_per_ring_batched(cake_t, R_centers, eta_centers, rt_R_ideal,
                                         half_window_px=fine_window_px, **fit_kw)
    bf = BatchedFits(**{f: torch.cat([getattr(p, f) for p in parts]) for f in _BF_FIELDS})
    if coverage is not None:
        keep = _capture_window_complete(bf, cake_t, R_centers, eta_centers, rt_R_ideal,
                                        wins, coverage, min_cell_coverage)
        if verbose:
            print(f"  [pv capture] dropped {int((~keep).sum())} of {keep.numel()} fits "
                  f"whose window is clipped", flush=True)
        n_pass, n_kept = _screen_retention(bf, keep, fit_kw.get("snr_min"))
        _note_retention(retention_log, "capture", iteration, n_pass, n_kept,
                        min_retention)
        bf = BatchedFits(**{f: getattr(bf, f)[keep] for f in _BF_FIELDS})
    return bf


def _unpack_current(spec, dtype, device) -> dict:
    from ..parameters.pack import pack_spec, unpack_spec
    with torch.no_grad():
        x, info = pack_spec(spec, dtype=dtype, device=device)
        return unpack_spec(x, info, spec)


def _predicted_ring_shift_px(fits: FittedDataset, u_before: dict, u_after: dict, *,
                             panel_layout, spec, q: float = 0.95) -> float:
    """How far one LM step moved the predicted rings, in px: the change in each
    fitted point's radial offset from its ring, p95 over the points."""
    from ..forward.bragg import R_ideal_px
    from ..forward.distortion import build_p_coeffs
    from ..forward.geometry import pixel_to_REta
    if fits.Y_pix.numel() == 0:
        return 0.0
    dt, dev = fits.Y_pix.dtype, fits.Y_pix.device
    zero = torch.zeros((), dtype=dt, device=dev)

    def offset_px(u):
        out = pixel_to_REta(
            fits.Y_pix, fits.Z_pix,
            Lsd=u["Lsd"], BC_y=u["BC_y"], BC_z=u["BC_z"],
            tx=u.get("tx", zero), ty=u["ty"], tz=u["tz"],
            p_coeffs=build_p_coeffs(u, dtype=dt, device=dev),
            parallax=u.get("Parallax", zero),
            pxY=u["pxY"], pxZ=u.get("pxZ", u["pxY"]),
            rho_d=fits.rho_d, panel_layout=panel_layout, panel_idx=fits.panel_idx,
            delta_yz=u.get("panel_delta_yz"),
            fix_panel_id=getattr(spec, "fix_panel_id", 0),
            delta_theta=u.get("panel_delta_theta"),
            delta_lsd_panel=u.get("panel_delta_lsd"),
            delta_p2_panel=u.get("panel_delta_p2"),
        )
        px = 0.5 * (u["pxY"] + u.get("pxZ", u["pxY"]))
        return out.R_px - R_ideal_px(fits.ring_two_theta_deg, u["Lsd"], px)

    with torch.no_grad():
        d = (offset_px(u_after) - offset_px(u_before)).abs()
    return float(torch.quantile(d.detach().cpu(), q))


def _recentre_tilt_bounds(spec, unpacked) -> Optional[dict]:
    """Slide the ty/tz box onto the current fit, keeping its width.

    v1 hands the bounds over as ``init ± tolTilts`` -- centred on the SEED
    (``compat.from_v1.spec_from_v1_params``). That caps how far capture can travel
    however wide its window is: measured on a 14 deg Varex frame, a tz 0 seed with
    the default ``tolTilts`` of 3 rails at 3.0 and ends 11 deg away, while the same
    run with ``tolTilts`` widened by hand to 16 reaches the rings. Capture is the
    phase that is allowed to travel, so its box travels with it; the fine phase
    keeps whatever box it inherits.

    ``Parameter.__post_init__`` builds the logit transform only when it is None, so
    the transform is rebuilt here whenever the bounds move. Leaving it would keep
    mapping into the OLD box -- the same refresh ``four_stage`` does when it
    tightens its Stage-1 bounds.

    Returns the new centres when anything moved, else None.
    """
    from ..parameters.transforms import Logit
    moved = False
    centres: dict = {}
    for n in ("ty", "tz"):
        p = spec.parameters.get(n)
        if p is None or not p.refined or p.bounds is None or n not in unpacked:
            continue
        lo, hi = float(p.bounds[0]), float(p.bounds[1])
        half = 0.5 * (hi - lo)
        if not half > 0.0:
            continue
        cur = float(unpacked[n].detach().reshape(-1)[0])
        centres[n] = cur
        if (cur - half, cur + half) != (lo, hi):
            p.bounds = (cur - half, cur + half)
            p.transform = Logit(*p.bounds)
            moved = True
    return centres if moved else None


def _row_scale(attr, cache: dict, r_data, name: str) -> float:
    """Scale of a constraint row against the data rows: a number, or ``"auto"`` = the rms of the
    data residual at the first evaluation (frozen for the rest of that LM solve)."""
    if isinstance(attr, str):
        if attr != "auto":
            raise ValueError(f"{name} must be a number or 'auto', got {attr!r}")
        if name not in cache:
            cache[name] = max(float(r_data.detach().pow(2).mean().sqrt()), 1e-12)
        return cache[name]
    return float(attr)


def _make_residual_fn(ds, *, spec, panel_layout, v1_params):
    """Residual vector over fit set ``ds``: data rows plus every active
    constraint row (zero-sum, panel expansion, Gaussian priors).

    One builder for both the full and the post-trim fit set. They used to be
    two hand-copied closures inside ``autocalibrate_pv``, and the trimmed one
    silently lacked the expansion row, so ``spec.no_panel_expansion`` was dead
    under every ``trim_mode`` except "off". Which rows are active is read from
    ``spec`` here, so a caller cannot get the two sets out of step.
    """
    from ..loss.constraints import gaussian_prior_residual, zero_sum_residual
    from ..loss.panel_gauge import panel_expansion_residual
    from ..parameters.parameter import GaussianPrior

    # Sigma panel = 0 zero-sum gauge (or the F2 per-ring sum dr_k = 0 gauge):
    # curvature exactly along the gauge nullspace, data residual unaffected.
    zs_active = bool(getattr(spec, "zero_sum_panels", False)) or \
        bool(getattr(spec, "zero_sum_dr_k", False))
    zs_lambda = float(getattr(spec, "zero_sum_lambda",
                              getattr(spec, "zero_sum_dr_k_lambda", 1e6)))
    # Gauge rows are sqrt(lambda) * sum(delta); the data rows are strain (~1e-5). Unscaled, the
    # gauge curvature (1e6 per panel parameter) dwarfs the data curvature (~1e-5) in the
    # Marquardt diagonal, and the LM steps along the free panel directions vanish. The rows are
    # therefore scaled by the data noise: lambda_eff = lambda * s**2, with s a number
    # (``spec.zero_sum_row_scale``), the string "auto" (the default: rms of the data residual at the first
    # evaluation, frozen for the rest of that LM solve), or 1.0 (the unscaled legacy rows, which
    # froze the Pilatus panel block: kept strain 38.8 against 29.7 uE scaled; see tests/test_gauge_row_scale.py).
    zs_scale_attr = getattr(spec, "zero_sum_row_scale", "auto")
    ex_scale_attr = getattr(spec, "panel_expansion_row_scale", "auto")   # same convention
    zs_scale_cache: dict = {}
    # Per-block units of the zero-sum rows: the blocks are in px, deg, um and a pure number, so one
    # row weight is ~500 times stiffer on the dLsd block than on dy/dz in the LM's (bounded)
    # coordinates, where the data curvature is comparable across blocks. ``"bounds"`` (the default)
    # divides each block by the half-width of its parameter bounds (the natural scale of the bounded
    # coordinate); None keeps the raw units. On the Pilatus frame the raw-unit scaled rows left the dLsd
    # block at its seed (53 um rms); with bounds units it moves (571-654 um rms, fix-panel-28: 689).
    zs_units: dict = {}
    if getattr(spec, "zero_sum_block_units", "bounds") == "bounds":
        for _nm in ("panel_delta_yz", "panel_delta_theta", "panel_delta_lsd", "panel_delta_p2", "delta_r_k"):
            _q = spec.parameters.get(_nm) if isinstance(getattr(spec, "parameters", None), dict) else None
            _b = getattr(_q, "bounds", None) if _q is not None else None
            if _b is not None and float(_b[1]) > float(_b[0]):
                zs_units[_nm] = 0.5 * (float(_b[1]) - float(_b[0]))
    # Second panel nullspace: a uniform radial expansion of the modules is
    # degenerate with Lsd, and fix_panel_id / zero-sum do not touch it.
    ex_active = bool(getattr(spec, "no_panel_expansion", False)) \
        and panel_layout is not None
    ex_lambda = float(getattr(spec, "panel_expansion_lambda", 1e6))
    # Parameter.prior = GaussianPrior(mean, std) contributes (theta - mu)/sigma
    # rows so the LM cost matches -2 log posterior.
    has_prior = any(isinstance(q.prior, GaussianPrior)
                    for q in spec.parameters.values())

    def residual_fn(unpacked_now: dict) -> torch.Tensor:
        r = pseudo_strain_residual(
            ds.Y_pix, ds.Z_pix, ds.ring_two_theta_deg,
            unpacked_now,
            rho_d=ds.rho_d, weights=ds.weights,
            panel_layout=panel_layout, panel_idx=ds.panel_idx,
            fix_panel_id=getattr(spec, "fix_panel_id", 0),
            ring_idx=ds.ring_idx,
            ring_d_spacing_A=ds.ring_d_spacing_A,
        )
        if zs_active:
            zs_s = _row_scale(zs_scale_attr, zs_scale_cache, r, "zero_sum_row_scale")
            _up = ({k: (v / zs_units[k] if k in zs_units else v) for k, v in unpacked_now.items()}
                   if zs_units else unpacked_now)
            zs = zero_sum_residual(_up, lambda_zs=zs_lambda * zs_s * zs_s)
            if zs.numel() > 0:
                r = torch.cat([r, zs])
        if ex_active:
            ex_s = _row_scale(ex_scale_attr, zs_scale_cache, r, "panel_expansion_row_scale")
            ex = panel_expansion_residual(
                unpacked_now, panel_layout=panel_layout,
                bc_y=unpacked_now.get("BC_y", v1_params.BC_y),
                bc_z=unpacked_now.get("BC_z", v1_params.BC_z),
                lambda_ex=ex_lambda * ex_s)   # this row is lambda * amp (not sqrt(lambda) * amp)
            if ex.numel() > 0:
                r = torch.cat([r, ex])
        if has_prior:
            pr = gaussian_prior_residual(unpacked_now, spec)
            if pr.numel() > 0:
                r = torch.cat([r, pr])
        return r

    return residual_fn


def autocalibrate_pv(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    dark: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    spec: Optional[CalibrationSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter: int = 5,
    half_window_px: float = 4.0,
    capture_window_px: float = 25.0,      # capture phase: widest per-ring window,
                                            # capped below half the gap to the
                                            # next ring; 0 = start in the fine one
    capture_shift_px: float = 1.0,        # leave capture once an LM step moves
                                            # the predicted rings less than this
                                            # (p95 over the fitted points)
    n_capture_max: int = 10,              # capture iterations at most, run before
                                            # the n_iter fine ones
    recentre_capture_bounds: bool = True,  # during capture, slide the ty/tz box
                                            # onto each new fit (same width), so a
                                            # tilt-blind seed is not capped by its
                                            # own tolTilts. Fine phase unaffected
    guard_clipped_windows: bool = True,   # drop FINE-window fits whose radial
                                            # window is cut by a panel edge, gap
                                            # or mask. The capture phase screens
                                            # unconditionally (shipped in 0.17.0)
                                            # -- this knob cannot switch that off
    min_cell_coverage: float = 0.5,       # a cell in the window must reach this
                                            # fraction of the best-covered cell
                                            # in the same window to count as whole
    min_screen_retention: float = 0.5,    # warn (FitRetentionWarning) when the
                                            # clipped-window screen keeps less than
                                            # this fraction of the SNR-passing fits
    param_tol_px: Optional[float] = 0.05,  # fine-phase stop: strain stable AND the
                                            # geometry step (ring shift at the outer
                                            # ring, px) below this over the last two
                                            # iterations. None = legacy strain-only
    pv_max_iter: int = 50,
    snip_window: int = 0,           # 0 = none, recommend ~2× peak FWHM in bins
    doublet_separation_px: float = 0.0,  # 0 = none; v1 default 25 px
    snr_min: float = 3.0,
    max_per_ring: Optional[int] = None,
    lm_max_iter: int = 100,
    huber_delta: Optional[float] = None,
    trim_residual_pct: Optional[float] = None,
    trim_mode: str = "stratified",        # "stratified" | "global" | "multfactor" | "off"
    trim_n_eta_buckets: int = 8,
    trim_min_per_cell: int = 3,
    trim_use_mad: bool = True,
    trim_mad_k: float = 5.0,
    reuse_fits: bool = False,             # reuse cake+pV fits across outer iters
                                            # (matches v1's stable trim/LM loop;
                                            # avoids re-extract noise on Pilatus)
    drop_gap_fits: bool = True,           # drop fits whose centroid lands in
                                            # an inter-panel gap (panel_idx == -1).
                                            # These are cake-binning artefacts —
                                            # not real ring crossings.  v1 handles
                                            # this implicitly via GetPanelIndex.
    distribution_report: bool = True,     # show ASCII histogram + quantiles
    auto_max_ring: bool = True,           # auto-detect max ring (v1's 3 criteria)
    csv_screen_path: Optional[str] = None,    # write calibrant-screen CSV here
    csv_trace_path: Optional[str] = None,     # write per-iter CSV trace here
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
) -> PVCalibrationResult:
    """Alternating engine using batched pV peak fit instead of centroid.

    Runs in two phases. **Capture** fits each ring in a wide window
    (``capture_window_px``, capped so it cannot hold a neighbouring ring) and
    repeats until an LM step moves the predicted rings less than
    ``capture_shift_px``, or ``n_capture_max`` iterations. **Fine** then runs the
    usual ``n_iter`` iterations in ``half_window_px`` with the strain-change
    stop. ``history`` holds the fine iterations, ``capture_history`` the others.

    Why: the fine window only sees a ring the geometry already puts within a few
    px. From a seed further off it fits whatever lies inside the window, the LM
    agrees with those fits, and the loop settles on a wrong geometry instead of
    moving toward the rings. Measured on a large-tilt, off-panel CeO2 frame
    (150 um Varex): the tilt-blind seed put 24 of 30 rings more than 4 px away
    (median 10.4 px), the fine-window loop stopped 1.9 deg from the rings in tz,
    and a 25 px window alone reached them. ``capture_window_px=0`` restores the
    fine-window-only loop.

    Limits, measured on the same frame. v1 hands over bounds as ``init ±
    tolTilts``, centred on the SEED, so capture used to be capped by the seed it
    started from: from tz 0 with the default ``tolTilts`` of 3 the tilt railed at
    3.0 for half the capture phase and the run ended 11.2 deg out at 2030 µε.
    ``recentre_capture_bounds`` (default on) slides that box onto each new fit, and
    the same blind tz 0 start then walks 0.19 → 3.13 → 8.32 → 14.01 and lands
    0.051 deg from the reference at 28.4 µε. It needed **all 10** capture
    iterations to do it, so ``n_capture_max`` is the next thing that binds: a
    larger tilt, or a seed further out, runs out of iterations and says so only
    through ``CaptureRangeWarning``.

    Fits whose radial window is clipped by a detector edge, module gap or mask
    are dropped in **both** phases. Capture has screened since 0.17.0 and does so
    unconditionally; ``guard_clipped_windows`` governs the fine window only, so
    turning it off leaves the shipped capture behaviour intact. A clipped window
    holds a truncated peak, and the fit
    slides onto its tail: measured on an off-panel Eiger frame, a ring 14 px
    past the panel edge produced fits at +23 px that passed the SNR cut and
    pulled ``BC_y`` onto its bound. The wide capture window meets an edge far
    more often, which is why it was screened first, but the fine window inherits
    the same failure wherever a ring sits within ``half_window_px`` of one. If
    every fine fit is clipped the whole set is kept and a
    ``ClippedWindowWarning`` is raised, since returning nothing would be worse
    than returning something the caller has been told to distrust.
    """
    v1_params.validate()
    # RhoD to µm before anything reads it: the spec, the E-step and the bake
    # step all normalise the distortion polynomial by it. This pipeline used to
    # fall back to the pixel-valued MaxRingRad unconverted (see forward.sanity).
    from ..forward.sanity import resolve_v1_rho_d_um
    resolve_v1_rho_d_um(v1_params, verbose=verbose, label="autocalibrate_pv")
    if spec is None:
        spec = spec_from_v1_params(v1_params)
    # The image transform is part of the calibration description and rides on
    # the spec (see io/transforms.apply_im_trans for the three ways doing this
    # by hand fails silently). Guarded so the no-transform path -- every caller
    # that has ever worked -- is byte-identical to before.
    # Bad pixels: the RAW sentinels (-1/-2, the unsigned dtype max) are ALWAYS
    # masked, on top of any mask the caller gave. Done before the transform (the
    # mask is in the raw orientation and rides with the image) and before any
    # clamping. Without it a module gap is a block of valid pixels (1 after a
    # clamp, 2-3 once the dark's own sentinels are subtracted), the cake
    # counts it as covered, and the clipped-window screen cannot see it; capture
    # fits centred in a gap then pull the geometry away.
    image, dark, mask, _n_sent = apply_sentinel_mask(image, dark, mask)
    _cb = _coarse_cake_bin_message(float(getattr(v1_params, "RBinSize", 0.0) or 0.0))
    if _cb is not None:
        warnings.warn(CoarseCakeBinWarning(_cb), stacklevel=2)
    if _n_sent and verbose:
        print(f"  [pv] masked {_n_sent} pixels (sentinels + caller mask, "
              f"{100.0 * _n_sent / image.size:.2f} % of the frame)", flush=True)
    if spec.im_trans:
        image, dark, mask, spec.NrPixelsY, spec.NrPixelsZ = apply_im_trans(
            image, dark, mask, spec.im_trans)

    history: List[IterRecord] = []
    fits_final: Optional[FittedDataset] = None
    unpacked = None
    # Cached cake+pV fits when reuse_fits=True; populated on iter 0 and
    # reused on subsequent outer iters.  Holds the ring table, the
    # BatchedFits (R_fit, eta_deg, ring_idx, snr) and the rho_d.  Panel
    # rigid-body inversion is re-applied each iter (since panel deltas
    # may move during LM); R_fit/eta_deg in the cake do not.
    cached_bf = None
    cached_rt = None
    cached_max_ring_eff = None
    cached_full_fits_ds = None    # post-bake, post-SNR, post-subsample
    cached_n_raw = None
    cached_n_after_snr = None
    in_capture = capture_window_px > half_window_px and n_capture_max > 0
    capture_history: List[IterRecord] = []
    retention_log: List[dict] = []
    last_fine_edge_q90: Optional[float] = None
    stop_reason = ""
    final_step_px: Optional[float] = None
    px_mean = (0.5 * (v1_params.pxY + v1_params.pxZ) if v1_params.pxZ > 0
               else v1_params.pxY)

    for it in range(n_iter + (n_capture_max if in_capture else 0)):
        if not in_capture and len(history) >= n_iter:
            break
        if reuse_fits and cached_bf is not None and not in_capture:
            # Skip cake + peak fit + bake.  Reuse the FROZEN (Y_pix, Z_pix)
            # from iter 0 — bake-at-current-geom would un-stabilise the LM
            # the same way re-extract does.  Pixel positions are an
            # observation; they do not change between outer iters.
            bf = cached_bf
            rt = cached_rt
            max_ring_eff = cached_max_ring_eff
            fits_ds = cached_full_fits_ds
            n_raw = cached_n_raw
            n_after_snr = cached_n_after_snr
            n_used = int(fits_ds.Y_pix.numel())
            if verbose:
                print(f"  [pv iter {it}] reusing cached fits "
                      f"({bf.R_fit.numel()} cake fits → {n_used} pixel positions)",
                      flush=True)
            skip_extract = True
        else:
            skip_extract = False
            # ring_table_for (not build_ring_table) so MinRingSeparation and the
            # spec's ring exclusions reach this pipeline too — they used to apply
            # only to the centroid path, so a two-calibrant fit run through the
            # pseudo-Voigt pipeline silently kept its blended rings.
            rt = ring_table_for(v1_params, spec=spec)
            # Auto-detect max ring if not already capped — uses v1's 3-criteria
            # method (extent / separation / SNR).  Run once on iter 0 only.
            max_ring_eff = getattr(spec, "max_ring_number", 0)
            if auto_max_ring and max_ring_eff == 0 and it == 0:
                img_for_snr = (image - dark) if dark is not None else image
                mr = auto_detect_max_ring(
                    rt.r_ideal_px, v1_params.NrPixelsY, v1_params.NrPixelsZ,
                    v1_params.BC_y, v1_params.BC_z,
                    data=img_for_snr,
                )
                # mr counts rows; max_ring_number is a ring_nr label (see
                # ring_number_cap) -- the two differ once radius-degenerate
                # reflections have been merged.
                mr_nr = ring_number_cap(rt, mr)
                if mr_nr > 0:
                    spec.max_ring_number = mr_nr
                    max_ring_eff = mr_nr
                    if verbose:
                        print(f"  [pv iter {it}] auto-max-ring: keeping {mr} of "
                              f"{len(rt.ring_nr)} rings", flush=True)
            rt = filter_ring_table(
                rt,
                rings_to_exclude=getattr(spec, "rings_to_exclude", ()),
                max_ring_number=max_ring_eff,
            )
            if dark is not None:
                img_used = image - dark
            else:
                img_used = image
            cake_params = v1_params
            if in_capture:
                # The cake's R range is half a Width beyond the outermost rings;
                # widen it so their capture windows see data on both sides.
                cake_params = copy.copy(v1_params)
                cake_params.Width = max(float(v1_params.Width),
                                        2.0 * capture_window_px * px_mean)
            cake = integrate_cake(cake_params, img_used, rt, mask=mask)

            cake_t = torch.as_tensor(cake.intensity, dtype=dtype, device=device)
            R_centers = torch.as_tensor(cake.R_centers, dtype=dtype, device=device)
            eta_centers = torch.as_tensor(cake.eta_centers, dtype=dtype, device=device)
            rt_R_ideal = torch.as_tensor(rt.r_ideal_px, dtype=dtype, device=device)

            if in_capture:
                bf = _fit_capture_windows(
                    cake_t, R_centers, eta_centers, rt_R_ideal,
                    capture_window_px=capture_window_px,
                    fine_window_px=half_window_px,
                    coverage=getattr(cake, "coverage", None),
                    min_cell_coverage=min_cell_coverage, max_iter=pv_max_iter,
                    snr_min=snr_min, snip_window=snip_window,
                    dtype=dtype, device=device, verbose=verbose,
                    retention_log=retention_log, iteration=len(capture_history),
                    min_retention=min_screen_retention,
                )
            else:
                bf = fit_cake_per_ring_batched(
                    cake_t, R_centers, eta_centers, rt_R_ideal,
                    half_window_px=half_window_px, max_iter=pv_max_iter,
                    snr_min=snr_min, snip_window=snip_window,
                    dtype=dtype, device=device, verbose=verbose,
                )
                # Same clipped-window screen the capture phase applies, on the
                # one fixed window. The fine window is narrower, so it meets an
                # edge less often -- but when it does, the returned geometry is
                # the answer, not an intermediate one.
                cov_fine = getattr(cake, "coverage", None)
                if guard_clipped_windows and cov_fine is not None and bf.R_fit.numel() > 0:
                    wins_fine = np.full(int(rt_R_ideal.numel()), float(half_window_px))
                    keep = _capture_window_complete(
                        bf, cake_t, R_centers, eta_centers, rt_R_ideal,
                        wins_fine, cov_fine, min_cell_coverage)
                    n_drop = int((~keep).sum())
                    _p, _k = _screen_retention(bf, keep, snr_min)
                    _note_retention(retention_log, "fine", it, _p, _k,
                                    min_screen_retention, stacklevel=2)
                    if n_drop:
                        if bool(keep.any()):
                            bf = BatchedFits(**{f: getattr(bf, f)[keep]
                                                for f in _BF_FIELDS})
                            if verbose:
                                print(f"  [pv iter {it}] dropped {n_drop} of "
                                      f"{keep.numel()} fine-window fits whose "
                                      f"window is clipped", flush=True)
                        else:
                            warnings.warn(ClippedWindowWarning(
                                f"every fine-window fit ({keep.numel()}) sits in a "
                                f"radial window clipped by a panel edge, gap or "
                                f"mask at ±{half_window_px:g} px. Keeping them, "
                                f"because dropping them leaves nothing to fit — "
                                f"but they are cut peaks and the geometry that "
                                f"follows is not trustworthy. Check the mask and "
                                f"whether these rings reach the panel at all."),
                                stacklevel=2)
            # Optional doublet co-fitting: detect ring pairs within
            # ``doublet_separation_px`` and refit them with a 2-peak shared-bg
            # model.  Doublet results REPLACE the corresponding singleton
            # fits in the BatchedFits.
            if doublet_separation_px > 0:
                from ..forward.doublets import cluster_rings
                clusters = cluster_rings(
                    rt.r_ideal_px, min_separation_px=doublet_separation_px)
                partner, pairs = doublet_index_map(
                    rt.r_ideal_px, min_separation_px=doublet_separation_px,
                )
                pair_indices = [(g.i, g.j) for g in pairs]
                # Rings in a >=3-member blend cannot be co-fitted by the 2-peak
                # model.  Chaining them into overlapping pairs fits the interior
                # ring twice at two different centres, so drop their fits
                # instead and say so.
                n_ary = set(clusters.n_ary_ring_indices)
                if n_ary:
                    keep_n = torch.tensor(
                        [int(r) not in n_ary for r in bf.ring_idx.tolist()],
                        dtype=torch.bool)
                    n_dropped = int((~keep_n).sum())
                    if not bool(keep_n.any()):
                        raise RuntimeError(
                            f"doublet_separation_px={doublet_separation_px:g} "
                            f"groups every ring into blends of >=3 "
                            f"({len(clusters.n_ary)} cluster(s) covering rings "
                            f"{sorted(n_ary)}), so no fits are left. That "
                            f"separation is far larger than the ring spacing "
                            f"here — lower it, or exclude the blends up front "
                            f"with CalibrationParams.MinRingSeparation.")
                    bf = BatchedFits(**{
                        f: getattr(bf, f)[keep_n]
                        for f in ("R_fit", "eta_deg", "ring_idx", "sigma",
                                  "gamma", "area", "snr", "rms", "rc")})
                    if verbose:
                        print(f"  [pv iter {it}] dropped {n_dropped} fits from "
                              f"{len(clusters.n_ary)} blends of >=3 rings "
                              f"(rings {sorted(n_ary)}) — the 2-peak co-fitter "
                              f"cannot model them", flush=True)
                if pair_indices:
                    df = fit_doublet_pairs(
                        cake_t, R_centers, eta_centers, rt_R_ideal,
                        pair_indices=pair_indices,
                        half_window_px=half_window_px, max_iter=pv_max_iter,
                        snr_min=snr_min, snip_window=snip_window,
                        dtype=dtype, device=device, verbose=verbose,
                    )
                    if df.R_fit_lo.numel() > 0:
                        paired_rings = set()
                        for lo_i, hi_i in pair_indices:
                            paired_rings.add(lo_i); paired_rings.add(hi_i)
                        keep_mask = torch.tensor(
                            [int(r) not in paired_rings for r in bf.ring_idx.tolist()],
                            dtype=torch.bool,
                        )
                        bf = BatchedFits(
                            R_fit=torch.cat([bf.R_fit[keep_mask], df.R_fit_lo, df.R_fit_hi]),
                            eta_deg=torch.cat([bf.eta_deg[keep_mask], df.eta_deg, df.eta_deg]),
                            ring_idx=torch.cat([bf.ring_idx[keep_mask], df.ring_idx_lo, df.ring_idx_hi]),
                            sigma=torch.cat([bf.sigma[keep_mask], df.sigma_lo, df.sigma_hi]),
                            gamma=torch.cat([bf.gamma[keep_mask], df.gamma_lo, df.gamma_hi]),
                            area=torch.cat([bf.area[keep_mask], df.area_lo, df.area_hi]),
                            snr=torch.cat([bf.snr[keep_mask], df.snr, df.snr]),
                            rms=torch.cat([bf.rms[keep_mask], df.rms, df.rms]),
                            rc=torch.cat([bf.rc[keep_mask], df.rc, df.rc]),
                        )
                        if verbose:
                            print(f"  [pv iter {it}] doublet co-fit replaced "
                                  f"{int((~keep_mask).sum())} singletons with "
                                  f"{df.R_fit_lo.numel() * 2} doublet fits", flush=True)
            if verbose:
                offsets = (bf.R_fit - rt_R_ideal[bf.ring_idx]).detach().cpu().numpy()
                import numpy as _np
                if offsets.size:
                    print(f"  [pV-batched] offset stats |R_fit - R_ideal|: "
                          f"med={_np.median(_np.abs(offsets)):.3f} px  "
                          f"q90={_np.quantile(_np.abs(offsets), 0.9):.3f} px", flush=True)
                else:
                    # every fit was screened out; the retention warning carries the news
                    print("  [pV-batched] no fits survived the screen", flush=True)
            if not in_capture and bf.R_fit.numel() > 0:
                last_fine_edge_q90 = float(torch.quantile(
                    (bf.R_fit - rt_R_ideal[bf.ring_idx]).abs().detach().cpu(), 0.9))
            if reuse_fits and not in_capture:
                cached_bf = bf
                cached_rt = rt
                cached_max_ring_eff = max_ring_eff

        if not skip_extract:
            # Pass the *current* panel shifts from the spec so the inverse is
            # panel-aware on iter > 0 (when the M-step has moved them).
            # Only relevant in the non-reuse path; reuse_fits freezes Y/Z at
            # iter 0, by design.
            cur_dyz = None
            cur_dth = None
            if panel_layout is not None:
                if "panel_delta_yz" in spec.parameters:
                    init = spec.parameters["panel_delta_yz"].init
                    if isinstance(init, torch.Tensor):
                        cur_dyz = init.to(dtype=dtype, device=device)
                if "panel_delta_theta" in spec.parameters:
                    init = spec.parameters["panel_delta_theta"].init
                    if isinstance(init, torch.Tensor):
                        cur_dth = init.to(dtype=dtype, device=device)
            fix_id = getattr(spec, "fix_panel_id", 0)
            fits_ds = _bake_fits_to_dataset(bf, v1_params, rt, dtype, device,
                                              panel_layout=panel_layout,
                                              panel_delta_yz=cur_dyz,
                                              panel_delta_theta=cur_dth,
                                              fix_panel_id=fix_id)
            n_raw = int(fits_ds.Y_pix.numel())
            fits_ds = _filter_by_snr(fits_ds, snr_min=snr_min)
            n_after_snr = int(fits_ds.Y_pix.numel())
            # Drop fits that landed in inter-panel gaps (panel_idx == -1):
            # these are cake-binning artefacts, not real ring crossings.
            if drop_gap_fits and fits_ds.panel_idx is not None:
                gap_keep = fits_ds.panel_idx >= 0
                n_gap = int((~gap_keep).sum())
                if n_gap > 0:
                    fits_ds = FittedDataset(
                        Y_pix=fits_ds.Y_pix[gap_keep], Z_pix=fits_ds.Z_pix[gap_keep],
                        ring_idx=fits_ds.ring_idx[gap_keep], snr=fits_ds.snr[gap_keep],
                        ring_two_theta_deg=fits_ds.ring_two_theta_deg[gap_keep],
                        rho_d=fits_ds.rho_d,
                        weights=(fits_ds.weights[gap_keep] if fits_ds.weights is not None else None),
                        panel_idx=fits_ds.panel_idx[gap_keep],
                        rt=fits_ds.rt,
                        ring_d_spacing_A=(fits_ds.ring_d_spacing_A[gap_keep]
                                           if fits_ds.ring_d_spacing_A is not None else None),
                    )
                    if verbose:
                        print(f"  [pv iter {it}] dropped {n_gap} gap fits "
                              f"(panel_idx == -1)", flush=True)
            if max_per_ring is not None:
                fits_ds = _stratified_subsample(fits_ds, max_per_ring=max_per_ring)
            n_used = int(fits_ds.Y_pix.numel())
            if reuse_fits and not in_capture:
                cached_full_fits_ds = fits_ds
                cached_n_raw = n_raw
                cached_n_after_snr = n_after_snr

        if n_used == 0:
            if verbose:
                print(f"  [iter {it}] no fits passed SNR; aborting", flush=True)
            break

        residual_fn = _make_residual_fn(
            fits_ds, spec=spec, panel_layout=panel_layout, v1_params=v1_params)

        # Pre-LM trim of high-strain outliers — computed at the seeded
        # geometry.  Three modes:
        #
        # - "stratified" (default): per-(ring × η-bucket × panel) cells, MAD
        #   or percentile cutoff per cell.  Preserves spatial coverage; flags
        #   cells where rejections concentrate.
        # - "global": legacy single-percentile cutoff (matches
        #   trim_residual_pct semantics).  Has known spatial-bias risk.
        # - "off": no trim; rely on Huber loss for robustness.
        #
        # The full (un-trimmed) residuals are saved so reporting can use them.
        residual_fn_no_trim = residual_fn   # closes over the full FittedDataset
        full_fits_ds = fits_ds
        if trim_mode != "off":
            with torch.no_grad():
                from ..parameters.pack import pack_spec, unpack_spec
                x_full, info = pack_spec(spec, dtype=dtype, device=device)
                unpacked0 = unpack_spec(x_full, info, spec)
                r_pre_full = residual_fn(unpacked0).abs()
                # Trim only sees the data-residual rows; the zero-sum rows
                # (if active) have no per-fit panel/ring identity and would
                # confuse the cell-key construction.  Slice to N_data.
                n_data = int(fits_ds.Y_pix.numel())
                r_pre = r_pre_full[:n_data]
            n_before = int(fits_ds.Y_pix.numel())
            if trim_mode == "stratified":
                # Stratified trim — recommended default.  See loss.robust_trim
                # for the bias guards.
                keep_pct = trim_residual_pct if trim_residual_pct is not None else 90.0
                with torch.no_grad():
                    keep, trim_report = stratified_trim(
                        r_pre, fits_ds.ring_idx,
                        azimuth_deg(fits_ds.Y_pix, fits_ds.Z_pix,
                                    unpacked0["BC_y"], unpacked0["BC_z"]),
                        panel_idx=fits_ds.panel_idx,
                        keep_pct=keep_pct,
                        n_eta_buckets=trim_n_eta_buckets,
                        min_per_cell=trim_min_per_cell,
                        use_mad=trim_use_mad,
                        mad_k=trim_mad_k,
                    )
                if verbose:
                    print(f"  [pv iter {it}] stratified trim: "
                          f"{n_before} → {int(keep.sum())} "
                          f"({100.0 * (1 - int(keep.sum()) / max(n_before, 1)):.1f}% rejected)",
                          flush=True)
                    print("  " + trim_report.render().replace("\n", "\n  "),
                          flush=True)
            elif trim_mode == "global":
                if trim_residual_pct is None:
                    trim_residual_pct = 85.0
                cutoff = float(torch.quantile(r_pre, trim_residual_pct / 100.0))
                keep = r_pre <= cutoff
                if verbose:
                    print(f"  [pv iter {it}] global trim @ {trim_residual_pct}%: "
                          f"{n_before} → {int(keep.sum())} "
                          f"(cutoff |r|≤{cutoff:.4e})", flush=True)
            elif trim_mode == "multfactor":
                # v1-compatible iterative cut: |r| > factor × current_mean.
                # `trim_residual_pct` is reused as the multfactor (default 2.0)
                # to avoid adding yet another knob.  Matches v1 C's MultFactor.
                # WARNING: rejects 60–80% on this benchmark suite — has no
                # spatial guards.  Use ``stratified_multfactor`` for
                # spatially-balanced results.
                factor = trim_residual_pct if trim_residual_pct is not None else 2.0
                from ..loss.robust_trim import multfactor_trim
                with torch.no_grad():
                    keep, trim_report = multfactor_trim(r_pre, factor=factor)
                if verbose:
                    print(f"  [pv iter {it}] multfactor trim (factor={factor:.2f}): "
                          f"{n_before} → {int(keep.sum())} "
                          f"({100.0 * (1 - int(keep.sum()) / max(n_before, 1)):.1f}% rejected)",
                          flush=True)
            elif trim_mode == "stratified_multfactor":
                # Spatially-aware variant — preserves coverage across panels
                # / rings / η-buckets via per-cell |r| > factor × cell_mean
                # with a min_per_cell floor.  Recommended default for
                # multi-panel data (Pilatus) and any case where the global
                # multfactor's >60% rejection is unacceptable.
                factor = trim_residual_pct if trim_residual_pct is not None else 2.0
                from ..loss.robust_trim import stratified_multfactor_trim
                with torch.no_grad():
                    # Need η for bucketing — derive from current geometry.
                    from ..parameters.pack import pack_spec, unpack_spec
                    from ..forward.geometry import pixel_to_REta
                    x_full, info = pack_spec(spec, dtype=dtype, device=device)
                    unp_now = unpack_spec(x_full, info, spec)
                    p_eff = {k: v for k, v in unp_now.items()}
                    from ..forward.distortion import build_p_coeffs as _build_pc
                    p_coeffs_local = _build_pc(p_eff, dtype=dtype, device=device)
                    eta_per_fit = pixel_to_REta(
                        fits_ds.Y_pix, fits_ds.Z_pix,
                        Lsd=p_eff["Lsd"], BC_y=p_eff["BC_y"], BC_z=p_eff["BC_z"],
                        tx=p_eff.get("tx", torch.zeros((), dtype=dtype, device=device)),
                        ty=p_eff["ty"], tz=p_eff["tz"],
                        p_coeffs=p_coeffs_local,
                        parallax=p_eff.get("Parallax", torch.zeros((), dtype=dtype, device=device)),
                        pxY=p_eff["pxY"], pxZ=p_eff.get("pxZ", p_eff["pxY"]),
                        rho_d=fits_ds.rho_d,
                        panel_layout=panel_layout,
                        panel_idx=fits_ds.panel_idx,
                        delta_yz=p_eff.get("panel_delta_yz"),
                        fix_panel_id=getattr(spec, "fix_panel_id", 0),
                        delta_theta=p_eff.get("panel_delta_theta"),
                        delta_lsd_panel=p_eff.get("panel_delta_lsd"),
                        delta_p2_panel=p_eff.get("panel_delta_p2"),
                    ).eta_deg
                    keep, trim_report = stratified_multfactor_trim(
                        r_pre, fits_ds.ring_idx, eta_per_fit,
                        panel_idx=fits_ds.panel_idx,
                        factor=factor,
                        n_eta_buckets=trim_n_eta_buckets,
                        min_per_cell=trim_min_per_cell,
                    )
                if verbose:
                    print(f"  [pv iter {it}] stratified_multfactor trim "
                          f"(factor={factor:.2f}, min_per_cell={trim_min_per_cell}): "
                          f"{n_before} → {int(keep.sum())} "
                          f"({100.0 * (1 - int(keep.sum()) / max(n_before, 1)):.1f}% rejected)",
                          flush=True)
                    print("  " + trim_report.render().replace("\n", "\n  "),
                          flush=True)
            else:
                raise ValueError(f"unknown trim_mode {trim_mode!r}")

            fits_ds = FittedDataset(
                Y_pix=fits_ds.Y_pix[keep], Z_pix=fits_ds.Z_pix[keep],
                ring_idx=fits_ds.ring_idx[keep], snr=fits_ds.snr[keep],
                ring_two_theta_deg=fits_ds.ring_two_theta_deg[keep],
                rho_d=fits_ds.rho_d,
                weights=(fits_ds.weights[keep] if fits_ds.weights is not None else None),
                panel_idx=(fits_ds.panel_idx[keep] if fits_ds.panel_idx is not None else None),
                rt=fits_ds.rt,
                ring_d_spacing_A=(fits_ds.ring_d_spacing_A[keep]
                                   if fits_ds.ring_d_spacing_A is not None else None),
            )
            n_used = int(fits_ds.Y_pix.numel())

            residual_fn = _make_residual_fn(
                fits_ds, spec=spec, panel_layout=panel_layout, v1_params=v1_params)

        unpacked_pre = _unpack_current(spec, dtype, device) if in_capture else None
        unpacked, cost, rc = lm_minimise(
            spec, residual_fn,
            config=GenericLMConfig(max_iter=lm_max_iter, ftol_rel=1e-9,
                                    xtol_rel=1e-9, huber_delta=huber_delta),
            dtype=dtype, device=device,
        )

        for name, val in unpacked.items():
            if val.numel() == 1 and hasattr(v1_params, name):
                cur = getattr(v1_params, name)
                try:
                    setattr(v1_params, name, type(cur)(float(val.detach())))
                except Exception:
                    pass
            # Update the spec's init in place so the next iter warm-starts from
            # the converged values (this preserves any panel/auxiliary params
            # the caller added to the spec, which a from-scratch rebuild would
            # lose).
            if name in spec.parameters:
                if val.numel() == 1:
                    spec.parameters[name].init = float(val.detach())
                else:
                    spec.parameters[name].init = val.detach().cpu()

        with torch.no_grad():
            r_final = residual_fn(unpacked)
            mean_uE = float(r_final.abs().mean()) * 1e6

        # Honest evaluation on the *un-trimmed* full set — guards against
        # the trim hiding evidence (per spatial-bias concern).
        full_residual_fn = residual_fn_no_trim   # closure over original full_fits_ds

        def _full_resid(u):
            return pseudo_strain_residual(
                full_fits_ds.Y_pix, full_fits_ds.Z_pix,
                full_fits_ds.ring_two_theta_deg, u,
                rho_d=full_fits_ds.rho_d, weights=full_fits_ds.weights,
                panel_layout=panel_layout, panel_idx=full_fits_ds.panel_idx,
                fix_panel_id=getattr(spec, "fix_panel_id", 0),
                ring_idx=full_fits_ds.ring_idx,
                ring_d_spacing_A=full_fits_ds.ring_d_spacing_A,
            )
        full_mean_uE, full_med_uE, full_rms_uE = evaluate_full_strain(_full_resid, unpacked)
        full_mean_uE *= 1e6
        full_med_uE  *= 1e6
        full_rms_uE  *= 1e6
        if verbose:
            print(f"  [pv iter {it}] FULL-set strain (un-trimmed): "
                  f"mean={full_mean_uE:7.2f}  med={full_med_uE:7.2f}  "
                  f"rms={full_rms_uE:7.2f} μϵ", flush=True)
            if distribution_report:
                with torch.no_grad():
                    r_full_uE = _full_resid(unpacked).abs() * 1e6
                summary = strain_summary(
                    r_full_uE,
                    ring_idx=full_fits_ds.ring_idx,
                    eta_deg=full_fits_ds.ring_two_theta_deg,  # placeholder eta
                    panel_idx=full_fits_ds.panel_idx,
                )
                # Re-do with ACTUAL eta — we need the per-fit eta, not 2θ.
                # Compute eta on the fly from (Y, Z) and current geometry:
                from ..forward.geometry import pixel_to_REta
                from ..forward.distortion import build_p_coeffs as _build_pc
                p_eff = unpacked
                p_coeffs = _build_pc(p_eff, dtype=torch.float64)
                eta_per_fit = pixel_to_REta(
                    full_fits_ds.Y_pix, full_fits_ds.Z_pix,
                    Lsd=p_eff["Lsd"], BC_y=p_eff["BC_y"], BC_z=p_eff["BC_z"],
                    tx=p_eff.get("tx", torch.zeros((), dtype=torch.float64)),
                    ty=p_eff["ty"], tz=p_eff["tz"], p_coeffs=p_coeffs,
                    parallax=p_eff.get("Parallax", torch.zeros((), dtype=torch.float64)),
                    pxY=p_eff["pxY"], pxZ=p_eff.get("pxZ", p_eff["pxY"]),
                    rho_d=full_fits_ds.rho_d,
                    panel_layout=panel_layout, panel_idx=full_fits_ds.panel_idx,
                    delta_yz=p_eff.get("panel_delta_yz"),
                    delta_theta=p_eff.get("panel_delta_theta"),
                ).eta_deg.detach()
                summary = strain_summary(
                    r_full_uE,
                    ring_idx=full_fits_ds.ring_idx,
                    eta_deg=eta_per_fit,
                    panel_idx=full_fits_ds.panel_idx,
                )
                print(summary, flush=True)

        rec = IterRecord(
            iteration=(len(capture_history) if in_capture else len(history)),
            n_fitted=n_used, cost=cost, rc=rc,
            mean_strain_uE=mean_uE,
            Lsd=float(unpacked["Lsd"]),
            BC_y=float(unpacked["BC_y"]), BC_z=float(unpacked["BC_z"]),
            ty=float(unpacked["ty"]), tz=float(unpacked["tz"]),
        )
        if in_capture:
            shift_px = _predicted_ring_shift_px(
                fits_ds, unpacked_pre, unpacked, panel_layout=panel_layout, spec=spec)
            capture_history.append(rec)
            fits_final = fits_ds
            if verbose:
                print(f"  [pv capture {rec.iteration}] {n_raw}→{n_after_snr}→{n_used} fits  "
                      f"rc={rc} strain={mean_uE:8.2f} μϵ  "
                      f"Lsd={rec.Lsd:.2f} BC=({rec.BC_y:.3f},{rec.BC_z:.3f}) "
                      f"ty={rec.ty:.4f} tz={rec.tz:.4f}  rings moved {shift_px:.2f} px",
                      flush=True)
            if shift_px < capture_shift_px or len(capture_history) >= n_capture_max:
                _msg = _capture_cap_message(shift_px, capture_shift_px,
                                            len(capture_history), n_capture_max)
                if _msg is not None:
                    warnings.warn(CaptureNotSettledWarning(_msg), stacklevel=2)
                in_capture = False
                if verbose:
                    why = ("rings stopped moving" if shift_px < capture_shift_px
                           else f"n_capture_max={n_capture_max} reached")
                    print(f"  [pv capture] done after {len(capture_history)} "
                          f"iteration(s), {why}; fine window ±{half_window_px:g} px",
                          flush=True)
            # Only while capture continues: once it ends the fine phase keeps the
            # box it inherits, so the returned geometry is bounded by something the
            # caller set rather than by wherever capture happened to stop.
            if in_capture and recentre_capture_bounds:
                centres = _recentre_tilt_bounds(spec, unpacked)
                if centres is not None and verbose:
                    print(f"  [pv capture] tilt box re-centred on "
                          f"ty={centres.get('ty', float('nan')):.4f} "
                          f"tz={centres.get('tz', float('nan')):.4f}", flush=True)
            continue
        history.append(rec)
        fits_final = fits_ds

        if verbose:
            print(f"  [pv iter {it}] {n_raw}→{n_after_snr}→{n_used} fits  "
                  f"rc={rc} strain={mean_uE:8.2f} μϵ  "
                  f"Lsd={rec.Lsd:.2f} BC=({rec.BC_y:.3f},{rec.BC_z:.3f}) "
                  f"ty={rec.ty:.4f} tz={rec.tz:.4f}", flush=True)

        if len(history) >= 2:
            prev = history[-2].mean_strain_uE
            cur_ms = mean_uE
            # Strain-only stops are the legacy rule; with ``param_tol_px`` a
            # stable strain also needs a stable geometry. Strain increase signals
            # alternating-engine bias (the cake rebuild at the moved geometry
            # produces noisier fits than the previous iter), so keep the best.
            try:
                R_outer = float(torch.hypot(fits_ds.Y_pix - rec.BC_y,
                                            fits_ds.Z_pix - rec.BC_z).max())
            except Exception:
                R_outer = 0.0
            stop, why, step_px = _outer_loop_decision(
                history, R_outer, px_mean, param_tol_px=param_tol_px)
            final_step_px = step_px
            if stop:
                stop_reason = why
                if verbose:
                    print(f"  [pv iter {it}] terminating ({why}: strain "
                          f"{prev:.2f} → {cur_ms:.2f}, geometry step "
                          f"{step_px:.3f} px at the outer ring)", flush=True)
                break

    if not stop_reason and history:
        stop_reason = "iteration_cap"
    if (param_tol_px is not None and stop_reason in ("iteration_cap", "strain_rising",
                                                     "strain_floor")
            and final_step_px is not None and final_step_px >= param_tol_px):
        warnings.warn(GeometryNotSettledWarning(
            f"autocalibrate_pv: the outer loop ended ({stop_reason}) with the geometry "
            f"still moving by {final_step_px:.3f} px at the outermost ring (tolerance "
            f"{param_tol_px:g} px). The returned geometry is the last iterate, not a "
            f"fixed point; a sigma computed here is conditional on it and a different "
            f"seed can end elsewhere. Raise n_iter, or compare runs from shifted seeds."),
            stacklevel=2)

    if last_fine_edge_q90 is not None and last_fine_edge_q90 >= 0.95 * half_window_px:
        warnings.warn(CaptureRangeWarning(
            f"autocalibrate_pv: at least 10% of the final peak fits sit at the edge "
            f"of the ±{half_window_px:g} px window (q90 |R_fit - R_ideal| = "
            f"{last_fine_edge_q90:.2f} px). The rings are further from the returned "
            f"geometry than the window reaches, so the fit can agree with its own "
            f"fits and still be wrong. Raise capture_window_px or n_capture_max, or "
            f"start from a closer seed."), stacklevel=2)

    # Optional diagnostic CSVs.
    if csv_screen_path is not None and fits_final is not None and unpacked:
        from ..io.csvs import write_calibrant_screen_csv
        from ..forward.geometry import pixel_to_REta
        from ..forward.bragg import R_ideal_px
        from ..forward.distortion import build_p_coeffs as _build_pc
        with torch.no_grad():
            p_coeffs = _build_pc(unpacked, dtype=dtype)
            out = pixel_to_REta(
                fits_final.Y_pix, fits_final.Z_pix,
                Lsd=unpacked["Lsd"], BC_y=unpacked["BC_y"], BC_z=unpacked["BC_z"],
                tx=unpacked.get("tx", torch.zeros((), dtype=dtype)),
                ty=unpacked["ty"], tz=unpacked["tz"], p_coeffs=p_coeffs,
                parallax=unpacked.get("Parallax", torch.zeros((), dtype=dtype)),
                pxY=unpacked["pxY"], pxZ=unpacked.get("pxZ", unpacked["pxY"]),
                rho_d=fits_final.rho_d,
                panel_layout=panel_layout, panel_idx=fits_final.panel_idx,
                delta_yz=unpacked.get("panel_delta_yz"),
                fix_panel_id=getattr(spec, "fix_panel_id", 0),
                delta_theta=unpacked.get("panel_delta_theta"),
            )
            R_obs = out.R_px.cpu().numpy()
            px_mean = 0.5 * (float(unpacked["pxY"])
                              + float(unpacked.get("pxZ", unpacked["pxY"])))
            R_pred = R_ideal_px(fits_final.ring_two_theta_deg,
                                 unpacked["Lsd"],
                                 torch.tensor(px_mean, dtype=dtype)
                                 ).cpu().numpy()
            eta_per_fit = out.eta_deg.cpu().numpy()
        write_calibrant_screen_csv(
            csv_screen_path,
            ring_idx=fits_final.ring_idx.cpu().numpy(),
            eta_deg=eta_per_fit, R_obs_px=R_obs, R_pred_px=R_pred,
            panel_idx=(fits_final.panel_idx.cpu().numpy()
                        if fits_final.panel_idx is not None else None),
            snr=fits_final.snr.cpu().numpy(),
        )
        if verbose:
            print(f"  wrote calibrant-screen CSV: {csv_screen_path}", flush=True)
    if csv_trace_path is not None and history:
        from ..io.csvs import write_iteration_trace_csv
        write_iteration_trace_csv(csv_trace_path, history)
        if verbose:
            print(f"  wrote iteration-trace CSV: {csv_trace_path}", flush=True)

    return PVCalibrationResult(spec=spec, unpacked=unpacked or {},
                                history=history, fits_final=fits_final,
                                capture_history=capture_history,
                                retention=retention_log,
                                stop_reason=stop_reason,
                                settled=(stop_reason == "settled"),
                                final_step_px=final_step_px)


__all__ = ["autocalibrate_pv", "PVCalibrationResult", "IterRecord",
           "CaptureRangeWarning", "ClippedWindowWarning",
           "FitRetentionWarning", "GeometryNotSettledWarning",
           "CaptureNotSettledWarning", "CoarseCakeBinWarning"]
