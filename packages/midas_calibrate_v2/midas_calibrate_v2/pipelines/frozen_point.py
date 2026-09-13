"""Frozen-point, single joint-fit calibration.

Removes the alternating extract-then-refit structure entirely. Unlike
``autocalibrate_pv`` (cake extraction alternated with an LM refit, repeated
``n_iter`` times, so the extraction step re-runs at every new trial
geometry), this pipeline:

  1. picks a point cloud ONCE, directly on the raw image, via a genuine
     geometry-independent 2D local-maximum search (``forward.point_pick``);
  2. freezes those points forever;
  3. runs a SINGLE bounded LM fit of the geometry against them, using this
     package's existing exact-autograd forward model.

Because the point cloud never changes during the fit, there is no
re-extraction feedback loop for a bad early geometry estimate to
compound through -- which is what makes this pipeline noticeably more
tolerant of a rough initial tilt guess than the alternating pipelines,
particularly for large-tilt / off-detector-beam-centre geometries.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.rings import RingTable, build_ring_table

from ..compat.from_v1 import spec_from_v1_params
from ..forward.distortion import P_COEF_NAMES
from ..forward.point_pick import PickedPoints, pick_points
from ..inference.lm import GenericLMConfig, lm_minimise
from ..io.transforms import apply_im_trans
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.spec import CalibrationSpec
from ._common import FittedDataset, _filter_by_snr
from .single_pv import IterRecord, PVCalibrationResult

LOG = logging.getLogger(__name__)


def _dataset_from_picked(
    picked: PickedPoints, v1: V1Params, rt: RingTable, dtype, device,
) -> FittedDataset:
    # v1.RhoD is resolved to canonical µm (see forward.sanity.resolve_v1_rho_d_um)
    # by autocalibrate_frozen_point before this is ever called -- no fallback
    # needed here.
    rho_d = v1.RhoD
    Y = torch.tensor(picked.Y_pix, dtype=dtype, device=device)
    Z = torch.tensor(picked.Z_pix, dtype=dtype, device=device)
    rid = torch.tensor(picked.ring_idx, dtype=torch.long, device=device)
    snr = torch.tensor(picked.snr, dtype=dtype, device=device)
    rt_tt = torch.tensor(rt.two_theta_deg, dtype=dtype, device=device)
    rt_d = torch.as_tensor(rt.d_spacing, dtype=dtype, device=device)
    return FittedDataset(
        Y_pix=Y, Z_pix=Z, ring_idx=rid, snr=snr,
        ring_two_theta_deg=rt_tt[rid],
        rho_d=torch.as_tensor(rho_d, dtype=dtype, device=device),
        weights=None, rt=rt,
        ring_d_spacing_A=rt_d[rid],
    )


def _clone_spec(spec: CalibrationSpec) -> CalibrationSpec:
    """Copy ``spec`` so ``autocalibrate_frozen_point`` can freely mutate it
    (``tx.refined``, per-parameter ``.init``) without touching the caller's
    own object. ``dataclasses.replace(spec)`` alone is not enough: fields
    left unspecified are carried over BY REFERENCE, so the returned spec's
    ``.parameters`` dict -- and every ``Parameter`` inside it -- would still
    be the exact objects the caller passed in. Each ``Parameter`` is a flat
    dataclass of value types, so a shallow ``dataclasses.replace`` per entry
    is sufficient to break the aliasing.
    """
    cloned_params = {name: dataclasses.replace(p)
                      for name, p in spec.parameters.items()}
    return dataclasses.replace(spec, parameters=cloned_params)


def autocalibrate_frozen_point(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    spec: Optional[CalibrationSpec] = None,
    snr_min: float = 5.0,
    point_pick_kwargs: Optional[dict] = None,
    rings_to_exclude=(),
    max_ring_number: int = 0,
    lm_max_iter: int = 150,
    huber_delta: float = 3.0,
    lm_verbose: bool = False,
    verbose: bool = True,
    dtype=torch.float64, device="cpu",
) -> PVCalibrationResult:
    """One-shot frozen-point calibration.

    Returns a :class:`PVCalibrationResult` (same shape ``autocalibrate_pv``
    returns) whose ``history`` contains exactly one :class:`IterRecord` --
    kept for compatibility with existing reporting/plotting code, even
    though there is no outer loop here.
    """
    v1_params.validate()
    # RhoD to canonical µm before anything reads it (the spec, the point-pick
    # window placement, and the residual all normalise the distortion
    # polynomial by it) -- matches every other v2 pipeline's entry point.
    # This pipeline's own ad hoc fallback (``MaxRingRad * px``) happened to
    # already be dimensionally correct, but delegating to the single shared
    # implementation is what keeps it that way as the normalisation logic
    # evolves elsewhere.
    from ..forward.sanity import resolve_v1_rho_d_um
    resolve_v1_rho_d_um(v1_params, verbose=verbose,
                         label="autocalibrate_frozen_point")
    if spec is None:
        spec = spec_from_v1_params(v1_params)
    else:
        spec = _clone_spec(spec)

    # The image transform is part of the calibration description and rides
    # on the spec (see io/transforms.apply_im_trans for the three ways doing
    # this by hand fails silently) -- every other v2 pipeline applies it this
    # way. Guarded so the no-transform path -- every caller before this spec
    # field existed -- is byte-identical to before. This pipeline has no
    # separate dark/mask arguments of its own (masking is opt-in via
    # ``point_pick_kwargs["panel_mask"]``), so unlike the cake-based
    # pipelines only ``image`` rides through here -- a caller supplying both
    # a non-trivial ``im_trans`` AND a ``panel_mask`` must pass the mask
    # already in the post-transform frame.
    if spec.im_trans:
        image, _, _, spec.NrPixelsY, spec.NrPixelsZ = apply_im_trans(
            image, None, None, spec.im_trans)

    # tx (rotation about the beam) reaches the ring radii ONLY through the
    # azimuthal distortion harmonics: it shifts lab eta by exactly tx, and
    # D is evaluated at lab eta. With the harmonics free that makes
    # (tx, phi_k) -> (tx + d, phi_k + k*d) an exact gauge orbit -- refining
    # tx walks it and corrupts all six phases silently. With them frozen tx
    # is determined only by the frozen field, so a field fitted at the wrong
    # tx returns a confident wrong tx. Either way a single powder image
    # cannot refine it, so it is frozen here regardless of the caller's spec.
    if "tx" in spec.parameters:
        spec.parameters["tx"].refined = False

    # Window-size point_pick against the FULL (unfiltered) ring table, not
    # the caller's excluded subset: physical spots from an "excluded" ring
    # still exist in the image, and sizing a kept ring's window without
    # knowing about its real neighbour (because that neighbour was already
    # filtered out) lets the window balloon towards max_window_deg on that
    # side and pick up the neighbour's spots. rings_to_exclude/
    # max_ring_number are therefore applied AFTER picking, as a point
    # filter, so ring labels stay indexed into ``rt_full`` throughout (no
    # remapping needed).
    rt_full = build_ring_table(v1_params)
    keep_ring = np.ones(len(rt_full.ring_nr), dtype=bool)
    if rings_to_exclude:
        keep_ring &= ~np.isin(rt_full.ring_nr, list(rings_to_exclude))
    if max_ring_number > 0:
        keep_ring &= rt_full.ring_nr <= max_ring_number
    rt = rt_full

    pp_kwargs = dict(point_pick_kwargs or {})
    pp_kwargs.setdefault("snr_threshold", snr_min)
    pp_kwargs.setdefault("verbose", verbose)
    picked = pick_points(image, v1_params, rt, dtype=dtype, device=device,
                          **pp_kwargs)

    if not keep_ring.all():
        row_keep = keep_ring[picked.ring_idx]
        picked = PickedPoints(
            Y_pix=picked.Y_pix[row_keep], Z_pix=picked.Z_pix[row_keep],
            ring_idx=picked.ring_idx[row_keep], snr=picked.snr[row_keep],
            n_by_ring={i: (n if keep_ring[i] else 0)
                       for i, n in picked.n_by_ring.items()},
        )

    n_total = int(picked.Y_pix.shape[0])
    n_rings_hit = sum(1 for v in picked.n_by_ring.values() if v > 0)
    if verbose:
        print(f"  [autocalibrate_frozen_point] picked {n_total} points "
              f"across {n_rings_hit}/{len(picked.n_by_ring)} rings",
              flush=True)
        for i in sorted(picked.n_by_ring):
            n = picked.n_by_ring[i]
            if n > 0:
                print(f"    ring {i} (2θ={rt.two_theta_deg[i]:.3f}deg): "
                      f"{n} points", flush=True)

    if n_total == 0:
        raise RuntimeError(
            "point_pick found no points at the seed geometry -- check the "
            "seed geometry and snr_min"
        )

    fits_ds = _dataset_from_picked(picked, v1_params, rt, dtype, device)
    fits_ds = _filter_by_snr(fits_ds, snr_min=snr_min)
    n_used = int(fits_ds.Y_pix.numel())
    if n_used == 0:
        raise RuntimeError("all picked points were rejected by the SNR filter")

    def residual_fn(unpacked_now: dict) -> torch.Tensor:
        return pseudo_strain_residual(
            fits_ds.Y_pix, fits_ds.Z_pix, fits_ds.ring_two_theta_deg,
            unpacked_now,
            rho_d=fits_ds.rho_d, weights=fits_ds.weights,
            ring_idx=fits_ds.ring_idx,
            ring_d_spacing_A=fits_ds.ring_d_spacing_A,
        )

    unpacked, cost, rc = lm_minimise(
        spec, residual_fn,
        config=GenericLMConfig(max_iter=lm_max_iter, ftol_rel=1e-9,
                                xtol_rel=1e-9, huber_delta=huber_delta,
                                verbose=lm_verbose),
        dtype=dtype, device=device,
    )

    for name, val in unpacked.items():
        if val.numel() == 1 and hasattr(v1_params, name):
            cur = getattr(v1_params, name)
            try:
                setattr(v1_params, name, type(cur)(float(val.detach())))
            except Exception:
                LOG.debug("autocalibrate_frozen_point: could not write back "
                          "%s=%r onto v1_params (existing type %s)",
                          name, float(val.detach()), type(cur).__name__,
                          exc_info=True)
        if name in spec.parameters and val.numel() == 1:
            spec.parameters[name].init = float(val.detach())

    r_final = residual_fn(unpacked).detach()
    mean_uE = float(r_final.abs().mean()) * 1e6
    if verbose:
        print(f"  [autocalibrate_frozen_point] n_used={n_used} rc={rc} "
              f"strain={mean_uE:.2f} με  "
              f"Lsd={float(unpacked['Lsd']):.2f} "
              f"BC=({float(unpacked['BC_y']):.3f},{float(unpacked['BC_z']):.3f}) "
              f"ty={float(unpacked['ty']):.4f} tz={float(unpacked['tz']):.4f}",
              flush=True)

    rec = IterRecord(
        iteration=0, n_fitted=n_used, cost=cost, rc=rc,
        mean_strain_uE=mean_uE,
        Lsd=float(unpacked["Lsd"]),
        BC_y=float(unpacked["BC_y"]), BC_z=float(unpacked["BC_z"]),
        ty=float(unpacked["ty"]), tz=float(unpacked["tz"]),
    )
    return PVCalibrationResult(
        spec=spec, unpacked=unpacked, history=[rec], fits_final=fits_ds,
    )


def _bounded_geom_only_spec(
    v1: V1Params, *, bounds_tz_deg: float, bounds_ty_deg: float,
    bounds_bc_px: float, bounds_lsd_um: float,
) -> CalibrationSpec:
    """A geometry-only ``CalibrationSpec`` (all 15 distortion coeffs frozen)
    with bounds re-centered on ``v1``'s current Lsd/BC/ty/tz -- the "walk
    from wherever you currently are" bounds strategy that lets
    :func:`iterate_frozen_point_until_stable` cover far more ground per
    iteration than a single fixed-window fit.
    """
    spec = spec_from_v1_params(v1)
    spec.freeze(*P_COEF_NAMES)
    spec.parameters["BC_y"].bounds = (v1.BC_y - bounds_bc_px, v1.BC_y + bounds_bc_px)
    spec.parameters["BC_z"].bounds = (v1.BC_z - bounds_bc_px, v1.BC_z + bounds_bc_px)
    spec.parameters["Lsd"].bounds = (v1.Lsd - bounds_lsd_um, v1.Lsd + bounds_lsd_um)
    spec.parameters["ty"].bounds = (v1.ty - bounds_ty_deg, v1.ty + bounds_ty_deg)
    spec.parameters["tz"].bounds = (v1.tz - bounds_tz_deg, v1.tz + bounds_tz_deg)
    return spec


def _reseed(template: V1Params, fit: IterRecord) -> V1Params:
    reseeded = dataclasses.replace(
        template, Lsd=fit.Lsd, BC_y=fit.BC_y, BC_z=fit.BC_z,
        ty=fit.ty, tz=fit.tz,
    )
    # dataclasses.replace() carries every field not named above over BY
    # REFERENCE, so `reseeded.Refine`/`reseeded.extra` would otherwise be
    # the exact same dicts as `template`'s -- every "copy" in the iteration
    # chain sharing one mutable dict with the original v1_params. Nothing
    # in this loop writes to either today, so this is latent, not live; but
    # copying them (cheap: flat dicts of primitives) means it stays that
    # way even after a future change starts stashing per-iteration state.
    reseeded.Refine = dict(template.Refine)
    reseeded.extra = dict(template.extra)
    return reseeded


def _write_back_geometry(v1_params: V1Params, fit: IterRecord) -> None:
    """Set ``v1_params``'s geometry fields to match ``fit``.

    Called once, at the end of :func:`iterate_frozen_point_until_stable`,
    so the caller's mutated object ends up agreeing with the value the
    function actually returns -- see that function's docstring for why it
    otherwise wouldn't.
    """
    for name in ("Lsd", "BC_y", "BC_z", "ty", "tz"):
        cur = getattr(v1_params, name)
        try:
            setattr(v1_params, name, type(cur)(getattr(fit, name)))
        except Exception:
            LOG.debug("iterate_frozen_point_until_stable: could not write "
                      "back %s onto v1_params (existing type %s)",
                      name, type(cur).__name__, exc_info=True)


@dataclass
class IterateResult:
    """Outcome of :func:`iterate_frozen_point_until_stable`.

    ``converged=False`` means ``max_iter`` was reached without the trailing
    window ever satisfying the stability tolerances -- the caller's seed
    was (as far as this function can tell) outside this dataset's capture
    range, not a silently-wrong answer. Always check ``converged`` before
    trusting ``fit``.
    """

    converged: bool
    n_iter: int
    fit: IterRecord
    res: PVCalibrationResult
    history: List[IterRecord]


def iterate_frozen_point_until_stable(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    n_stable: int = 5,
    tol_tz_deg: float = 0.01,
    tol_ty_deg: float = 0.01,
    tol_bc_px: float = 0.05,
    tol_lsd_um: float = 5.0,
    max_iter: int = 60,
    bounds_tz_deg: float = 16.0,
    bounds_ty_deg: float = 10.0,
    bounds_bc_px: float = 100.0,
    bounds_lsd_um: float = 30_000.0,
    snr_min: float = 5.0,
    point_pick_kwargs: Optional[dict] = None,
    lm_max_iter: int = 150,
    verbose: bool = False,
) -> IterateResult:
    """Re-seed :func:`autocalibrate_frozen_point` from its own result until
    the geometry stops moving, to reach a good basin from even a genuinely
    blind starting guess (e.g. ``tz=0`` with no tilt information at all).

    Each iteration builds a fresh geometry-only spec (all distortion
    coefficients frozen) with Lsd/BC/ty/tz bounds **re-centered** on the
    *previous* iteration's converged geometry (``bounds_tz_deg`` etc.) --
    it is this re-centering, not a fixed window, that lets the estimate
    walk many degrees from a bad start across several iterations. Point
    picking is re-run from scratch each iteration (it depends on the
    geometry used to place its annular ring masks -- see
    ``forward.point_pick``), so this is not the same "one-shot, frozen
    point cloud" guarantee ``autocalibrate_frozen_point`` itself makes
    *within* one call; it is that one-shot mechanism used repeatedly as a
    well-behaved fixed-point iteration.

    **The stopping criterion is strict parameter stability, deliberately,
    with no strain gate.** On the real dataset this was validated against,
    a *loose* stability tolerance produces a false-positive: iterations
    partway through a genuine escape trajectory can sit in a "plateau"
    where tz creeps by only ~0.1-0.6 deg per iteration for 8+ iterations in
    a row -- stable enough to fool a loose criterion -- while still being
    ~10-12 deg from the true answer (strain 30x too high). The default
    tolerances here (0.01 deg tz/ty, 0.05 px BC, 5 um Lsd) have a
    two-order-of-magnitude margin against that plateau's per-step
    movement: they only fire once the fit has landed on a genuine,
    bit-reproducing fixed point. Do not loosen them without re-validating
    on your own data that no such plateau exists.

    Empirically (13-point sweep on the validating large-tilt, off-detector
    -beam-centre CeO2 dataset), this reliably reaches the correct basin --
    the identical fixed point every time, zero false-basin acceptances --
    from starting tz guesses across roughly a 35 deg range around the true
    tilt; well outside that range it correctly reports ``converged=False``
    rather than a wrong answer. It has been validated on one real dataset
    so far -- treat the capture range as dataset-specific until confirmed
    elsewhere.

    Like :func:`autocalibrate_frozen_point`, this mutates ``v1_params`` in
    place -- iteration 1 mutates it directly (it *is* the first ``seed``);
    later iterations mutate a reseeded copy instead (see :func:`_reseed`),
    so at the end the geometry this function actually returns is written
    back onto ``v1_params`` too, in addition to being returned. Without
    that final write-back, ``v1_params`` would be left at iteration 1's
    geometry -- typically far from the converged answer -- which is an
    easy trap for any caller who reads geometry back off the object they
    passed in rather than off ``IterateResult.fit``.

    Returns
    -------
    :class:`IterateResult`
    """
    seed = v1_params
    history: List[IterRecord] = []
    res: Optional[PVCalibrationResult] = None
    for i in range(1, max_iter + 1):
        spec = _bounded_geom_only_spec(
            seed, bounds_tz_deg=bounds_tz_deg, bounds_ty_deg=bounds_ty_deg,
            bounds_bc_px=bounds_bc_px, bounds_lsd_um=bounds_lsd_um,
        )
        res = autocalibrate_frozen_point(
            seed, image, spec=spec, snr_min=snr_min,
            point_pick_kwargs=point_pick_kwargs, lm_max_iter=lm_max_iter,
            verbose=False,
        )
        fit = res.history[0]
        history.append(fit)
        if verbose:
            print(f"  [iterate_frozen_point_until_stable] iter {i}: "
                  f"tz={fit.tz:.4f} ty={fit.ty:.4f} "
                  f"BC=({fit.BC_y:.2f},{fit.BC_z:.2f}) Lsd={fit.Lsd:.1f} "
                  f"strain={fit.mean_strain_uE:.2f}", flush=True)

        if len(history) >= n_stable:
            w = history[-n_stable:]
            tz_spread = max(f.tz for f in w) - min(f.tz for f in w)
            ty_spread = max(f.ty for f in w) - min(f.ty for f in w)
            bcy_spread = max(f.BC_y for f in w) - min(f.BC_y for f in w)
            bcz_spread = max(f.BC_z for f in w) - min(f.BC_z for f in w)
            lsd_spread = max(f.Lsd for f in w) - min(f.Lsd for f in w)
            if (tz_spread < tol_tz_deg and ty_spread < tol_ty_deg
                    and bcy_spread < tol_bc_px and bcz_spread < tol_bc_px
                    and lsd_spread < tol_lsd_um):
                _write_back_geometry(v1_params, fit)
                return IterateResult(converged=True, n_iter=i, fit=fit,
                                      res=res, history=history)
        seed = _reseed(seed, fit)

    _write_back_geometry(v1_params, history[-1])
    return IterateResult(converged=False, n_iter=max_iter, fit=history[-1],
                          res=res, history=history)


__all__ = ["autocalibrate_frozen_point",
           "iterate_frozen_point_until_stable", "IterateResult"]
