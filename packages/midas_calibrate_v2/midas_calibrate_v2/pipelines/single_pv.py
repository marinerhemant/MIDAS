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
from ._common import FittedDataset, filter_ring_table, ring_table_for


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


class CaptureRangeWarning(UserWarning):
    """The final fine-window peak fits pile up at the window edge: the rings
    are further from the returned geometry than the window reaches."""


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


def _filter_by_snr(fits_ds: FittedDataset, snr_min: float = 3.0) -> FittedDataset:
    keep = fits_ds.snr >= snr_min
    keep &= torch.isfinite(fits_ds.Y_pix) & torch.isfinite(fits_ds.Z_pix)
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
                         verbose: bool = False, **fit_kw) -> BatchedFits:
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

    Limits, measured on the same frame. Capture moves the geometry only inside
    the seed-centred bounds: from tz 0 it reached tz 14 deg with ``tolTilts`` 16,
    needing 10 capture iterations (the default ``n_capture_max``), and with the
    default 3 deg it cannot. Capture fits whose window is clipped by a detector
    edge, module gap or mask are dropped; the fine-window fits are not screened
    that way.
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
    last_fine_edge_q90: Optional[float] = None
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
                if mr > 0:
                    spec.max_ring_number = mr
                    max_ring_eff = mr
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
                    coverage=getattr(cake, "coverage", None), max_iter=pv_max_iter,
                    snr_min=snr_min, snip_window=snip_window,
                    dtype=dtype, device=device, verbose=verbose,
                )
            else:
                bf = fit_cake_per_ring_batched(
                    cake_t, R_centers, eta_centers, rt_R_ideal,
                    half_window_px=half_window_px, max_iter=pv_max_iter,
                    snr_min=snr_min, snip_window=snip_window,
                    dtype=dtype, device=device, verbose=verbose,
                )
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
                print(f"  [pV-batched] offset stats |R_fit - R_ideal|: "
                      f"med={_np.median(_np.abs(offsets)):.3f} px  "
                      f"q90={_np.quantile(_np.abs(offsets), 0.9):.3f} px", flush=True)
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

        # If the spec carries a Σ panel = 0 zero-sum gauge (or the
        # F2 per-ring Σ δr_k = 0 gauge), append the constraint as
        # additional residual rows.  Adds curvature exactly along the
        # gauge nullspace; data residual is unaffected.
        _zs_active = bool(getattr(spec, "zero_sum_panels", False)) or \
                     bool(getattr(spec, "zero_sum_dr_k", False))
        _zs_lambda = float(getattr(spec, "zero_sum_lambda",
                                    getattr(spec, "zero_sum_dr_k_lambda", 1e6)))
        if _zs_active:
            from ..loss.constraints import zero_sum_residual
        # Second panel nullspace: a uniform radial expansion of the modules is
        # degenerate with Lsd, and fix_panel_id / zero-sum do not touch it.
        _ex_active = bool(getattr(spec, "no_panel_expansion", False)) \
            and panel_layout is not None
        _ex_lambda = float(getattr(spec, "panel_expansion_lambda", 1e6))
        if _ex_active:
            from ..loss.panel_gauge import panel_expansion_residual
        # Gaussian priors on individual parameters
        # (Parameter.prior = GaussianPrior(mean, std)) contribute
        # (θ - μ)/σ rows so LM cost matches -2 log posterior.  Active
        # whenever any parameter in the spec has a GaussianPrior.
        from ..parameters.parameter import GaussianPrior as _GP
        _has_gauss_prior = any(
            isinstance(p.prior, _GP)
            for p in spec.parameters.values()
        )
        if _has_gauss_prior:
            from ..loss.constraints import gaussian_prior_residual

        def residual_fn(unpacked_now: dict) -> torch.Tensor:
            r = pseudo_strain_residual(
                fits_ds.Y_pix, fits_ds.Z_pix, fits_ds.ring_two_theta_deg,
                unpacked_now,
                rho_d=fits_ds.rho_d, weights=fits_ds.weights,
                panel_layout=panel_layout, panel_idx=fits_ds.panel_idx,
                fix_panel_id=getattr(spec, "fix_panel_id", 0),
                ring_idx=fits_ds.ring_idx,
                ring_d_spacing_A=fits_ds.ring_d_spacing_A,
            )
            if _zs_active:
                zs = zero_sum_residual(unpacked_now, lambda_zs=_zs_lambda)
                if zs.numel() > 0:
                    r = torch.cat([r, zs])
            if _ex_active:
                ex = panel_expansion_residual(
                    unpacked_now, panel_layout=panel_layout,
                    bc_y=unpacked_now.get("BC_y", v1_params.BC_y),
                    bc_z=unpacked_now.get("BC_z", v1_params.BC_z),
                    lambda_ex=_ex_lambda)
                if ex.numel() > 0:
                    r = torch.cat([r, ex])
            if _has_gauss_prior:
                pr = gaussian_prior_residual(unpacked_now, spec)
                if pr.numel() > 0:
                    r = torch.cat([r, pr])
            return r

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

            def residual_fn(unpacked_now: dict) -> torch.Tensor:    # rebound
                r = pseudo_strain_residual(
                    fits_ds.Y_pix, fits_ds.Z_pix, fits_ds.ring_two_theta_deg,
                    unpacked_now,
                    rho_d=fits_ds.rho_d, weights=fits_ds.weights,
                    panel_layout=panel_layout, panel_idx=fits_ds.panel_idx,
                    fix_panel_id=getattr(spec, "fix_panel_id", 0),
                    ring_idx=fits_ds.ring_idx,
                    ring_d_spacing_A=fits_ds.ring_d_spacing_A,
                )
                if _zs_active:
                    from ..loss.constraints import zero_sum_residual as _zsr
                    zs = _zsr(unpacked_now, lambda_zs=_zs_lambda)
                    if zs.numel() > 0:
                        r = torch.cat([r, zs])
                if _has_gauss_prior:
                    from ..loss.constraints import (
                        gaussian_prior_residual as _gpr,
                    )
                    pr = _gpr(unpacked_now, spec)
                    if pr.numel() > 0:
                        r = torch.cat([r, pr])
                return r

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
                in_capture = False
                if verbose:
                    why = ("rings stopped moving" if shift_px < capture_shift_px
                           else f"n_capture_max={n_capture_max} reached")
                    print(f"  [pv capture] done after {len(capture_history)} "
                          f"iteration(s), {why}; fine window ±{half_window_px:g} px",
                          flush=True)
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
            # Break on convergence OR on strain increase (the latter signals
            # alternating-engine bias: the cake rebuild at the moved geometry
            # produces noisier fits than the previous iter, so keep the best).
            if (cur_ms < 1.0
                    or abs(prev - cur_ms) < 0.01 * max(prev, 1.0)
                    or cur_ms > prev * 1.05):
                if verbose:
                    print(f"  [pv iter {it}] terminating "
                          f"(converged or strain rising: {prev:.2f} → {cur_ms:.2f})",
                          flush=True)
                break

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
                                capture_history=capture_history)


__all__ = ["autocalibrate_pv", "PVCalibrationResult", "IterRecord", "CaptureRangeWarning"]
