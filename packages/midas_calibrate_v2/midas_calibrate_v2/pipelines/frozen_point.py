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
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate.rings import RingTable, build_ring_table

from ..compat.from_v1 import spec_from_v1_params
from ..forward.distortion import P_COEF_NAMES, resolve_distortion_block
from ..forward.point_pick import PickedPoints, pick_points
from ..inference.lm import GenericLMConfig, lm_minimise
from ..io.transforms import apply_im_trans
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.pack import pack_spec, unpack_spec
from ..parameters.spec import CalibrationSpec
from ._common import FittedDataset, _filter_by_snr, ring_table_for
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


def _apply_distortion_refine(
    spec: CalibrationSpec,
    refine_distortion: Union[bool, str, Sequence[str]],
) -> None:
    """Freeze/thaw the 15 distortion coefficients on ``spec``.

    ``refine_distortion`` takes the same selector
    :func:`~midas_calibrate_v2.pipelines.auto.calibrate` does -- ``True``/
    ``False``, a :data:`~midas_calibrate_v2.forward.distortion.DISTORTION_BLOCKS`
    name (``"radial"``, ``"radial+2fold"``, ...), or an explicit sequence of
    v2 coefficient names -- resolved by the same
    :func:`~midas_calibrate_v2.forward.distortion.resolve_distortion_block`.
    One selector value therefore means the same thing regardless of which
    pipeline in this package it is handed to (a GUI can reuse it verbatim).
    """
    spec.freeze(*P_COEF_NAMES)
    thaw_names = resolve_distortion_block(refine_distortion)
    if thaw_names:
        spec.thaw(*thaw_names)


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


@dataclass
class PickCoverage:
    """How well the frozen point cloud actually covers the calibrant.

    ``converged`` (on :class:`IterateResult`) means the fitted PARAMETERS
    stopped moving -- it says nothing about whether the points that
    produced them were plentiful or evenly spread. This is that other
    check: how many rings contributed points, and how they are spread
    across 8 azimuthal (MIDAS-eta) octants -- exactly the kind of
    imbalance that can bias ``ty``/``tz``/``BC``, worse at large tilt (see
    ``forward.point_pick.pick_points``'s own docstring on azimuth-dependent
    SNR acceptance).
    """

    n_points: int
    n_rings_total: int
    n_rings_hit: int
    octant_by_ring: dict          # ring_idx -> length-8 list of counts
    mean_strain_uE: float


def _warn_if_thin_or_lopsided(
    coverage: PickCoverage, *, min_rings_hit: int = 3,
    min_octants_populated: int = 4,
) -> None:
    """Warn (not raise) when ``coverage`` looks too thin or too lopsided to
    trust without a closer look -- the thresholds are a floor for "clearly
    fine," not a validated pass/fail line; a caller with real trouble
    should inspect ``coverage`` itself, not just watch for this warning."""
    if coverage.n_rings_hit < min_rings_hit:
        warnings.warn(
            f"frozen-point fit used points from only {coverage.n_rings_hit} "
            f"ring(s) (of {coverage.n_rings_total} in the calibrant table) "
            "-- geometry fitted from this few rings is weakly constrained.",
            RuntimeWarning, stacklevel=3)
    octant_totals = np.zeros(8, dtype=np.int64)
    for counts in coverage.octant_by_ring.values():
        octant_totals += np.asarray(counts, dtype=np.int64)
    n_populated = int(np.count_nonzero(octant_totals))
    if n_populated < min_octants_populated:
        warnings.warn(
            f"frozen-point fit's accepted points cover only {n_populated}/8 "
            f"azimuthal octants (counts={octant_totals.tolist()}) -- an "
            "uneven azimuthal sample can bias ty/tz/BC; see "
            "forward.point_pick.pick_points's verbose-mode docstring.",
            RuntimeWarning, stacklevel=3)


@dataclass
class FrozenPointCalibrationResult(PVCalibrationResult):
    """:class:`~.single_pv.PVCalibrationResult`, plus the point-pick
    coverage diagnostics :func:`autocalibrate_frozen_point` computes along
    the way -- every field ``PVCalibrationResult`` has, so existing code
    written against that type still works unchanged."""

    coverage: Optional[PickCoverage] = None


def _huber_delta_auto(spec: CalibrationSpec, residual_fn, dtype, device,
                       *, k: float = 5.0) -> float:
    """Scale ``huber_delta`` to this fit's own residual units.

    ``_huberise`` (``midas_peakfit.lm_generic``) compares ``|r|`` to
    ``delta`` in whatever units the residual already is -- for
    ``pseudo_strain_residual`` that is dimensionless strain, O(1e-4), so a
    literal default like ``3.0`` never triggers and the fit is silently
    plain least squares. Evaluating the residual once at the spec's own
    initial (pre-optimisation) values gives a real per-dataset scale: a
    mispicked local maximum enters this pipeline's point cloud at full
    weight (there is no cake/profile step to average it down first), so
    outlier protection matters more here than in the cake pipelines.
    """
    x0, info = pack_spec(spec, dtype=dtype, device=device)
    unpacked0 = unpack_spec(x0, info, spec)
    r0 = residual_fn(unpacked0).detach()
    med = float(r0.abs().median())
    return k * med if med > 0 else 1.0


def autocalibrate_frozen_point(
    v1_params: V1Params,
    image: np.ndarray,
    *,
    spec: Optional[CalibrationSpec] = None,
    refine_distortion: Optional[Union[bool, str, Sequence[str]]] = None,
    snr_min: float = 5.0,
    point_pick_kwargs: Optional[dict] = None,
    rings_to_exclude=(),
    max_ring_number: int = 0,
    lm_max_iter: int = 150,
    huber_delta: Optional[float] = None,
    lm_verbose: bool = False,
    verbose: bool = True,
    dtype=torch.float64, device="cpu",
) -> FrozenPointCalibrationResult:
    """One-shot frozen-point calibration.

    Returns a :class:`FrozenPointCalibrationResult` -- a
    :class:`PVCalibrationResult` (same shape ``autocalibrate_pv`` returns,
    so existing reporting/plotting code keeps working unchanged) plus a
    ``coverage`` field (see :class:`PickCoverage`). ``history`` contains
    exactly one :class:`IterRecord`, kept for compatibility even though
    there is no outer loop here.

    Distortion refinement is controlled by ``v1_params.Refine["p0".."p14"]``
    when ``spec`` is not given (or by the caller's own ``spec`` otherwise) --
    the same mechanism ``autocalibrate_pv``, ``autocalibrate_four_stage``,
    ``autocalibrate_bayesian`` and ``autocalibrate_joint`` all use, and the
    one a GUI already builds via ``build_v1_params(..., refine=...)`` for
    every one of those. ``refine_distortion`` (default ``None``) is an
    *additional*, optional override with the same selector
    :func:`~midas_calibrate_v2.pipelines.auto.calibrate`'s own
    ``refine_distortion`` accepts -- see :func:`_apply_distortion_refine`.
    ``None`` leaves distortion refinement exactly as ``v1_params.Refine``/
    the caller's ``spec`` already says -- no behaviour change for existing
    callers. Passing anything else overrides whichever of those it would
    otherwise have been, the same way the forced ``tx`` freeze below always
    overrides the caller's spec.

    ``huber_delta`` (default ``None``) auto-scales to this fit's own
    residual units -- see :func:`_huber_delta_auto`; a fixed literal here
    (e.g. the ``3.0`` this used to default to) silently does nothing,
    since ``pseudo_strain_residual`` is O(1e-4) and never gets anywhere
    near it. Pass an explicit float to override.

    Tiled detectors (multi-panel Pilatus/Hydra) are fitted as a single
    panel: ``residual_fn`` does not pass ``panel_layout``/``panel_idx``
    through to :func:`~..loss.pseudo_strain.pseudo_strain_residual`. Fine
    for the single-panel Varex this pipeline was validated on; a caller
    with a genuinely tiled detector needs that plumbed through before this
    pipeline's fit is meaningful for it.
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

    if refine_distortion is not None:
        _apply_distortion_refine(spec, refine_distortion)

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
    # side and pick up the neighbour's spots. Exclusions are therefore
    # applied AFTER picking, as a point filter, so ring labels stay indexed
    # into ``rt_full`` throughout (no remapping needed).
    #
    # Two independent sources of exclusion feed that filter: this
    # function's own explicit rings_to_exclude/max_ring_number kwargs, AND
    # -- via ring_table_for, the same helper every other pipeline in this
    # package routes through (see single_pv.py) -- spec.rings_to_exclude/
    # spec.max_ring_number and v1_params.MinRingSeparation (blended-ring
    # dropping). Building rt_full straight from build_ring_table() alone
    # would silently ignore all three of those, which is exactly the bug
    # every other pipeline in this package was already fixed for.
    rt_full = build_ring_table(v1_params)
    rt_reduced = ring_table_for(v1_params, spec=spec, verbose=verbose)
    keep_ring = np.isin(rt_full.ring_nr, rt_reduced.ring_nr)
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
            octant_by_ring={i: c for i, c in picked.octant_by_ring.items()
                            if keep_ring[i]},
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

    huber_delta_eff = (
        _huber_delta_auto(spec, residual_fn, dtype, device)
        if huber_delta is None else huber_delta
    )
    if verbose:
        print(f"  [autocalibrate_frozen_point] huber_delta={huber_delta_eff:.3g}",
              flush=True)

    unpacked, cost, rc = lm_minimise(
        spec, residual_fn,
        config=GenericLMConfig(max_iter=lm_max_iter, ftol_rel=1e-9,
                                xtol_rel=1e-9, huber_delta=huber_delta_eff,
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
    coverage = PickCoverage(
        n_points=n_total, n_rings_total=len(picked.n_by_ring),
        n_rings_hit=n_rings_hit, octant_by_ring=picked.octant_by_ring,
        mean_strain_uE=mean_uE,
    )
    _warn_if_thin_or_lopsided(coverage)
    return FrozenPointCalibrationResult(
        spec=spec, unpacked=unpacked, history=[rec], fits_final=fits_ds,
        coverage=coverage,
    )


def _bounded_spec_for_iteration(
    v1: V1Params, *, bounds_tz_deg: float, bounds_ty_deg: float,
    bounds_bc_px: float, bounds_lsd_um: float,
    refine_distortion: Optional[Union[bool, str, Sequence[str]]],
) -> CalibrationSpec:
    """A ``CalibrationSpec`` for one iteration of
    :func:`iterate_frozen_point_until_stable`.

    Lsd/BC/ty/tz bounds are re-centered on ``v1``'s current geometry -- the
    "walk from wherever you currently are" bounds strategy that lets the
    iteration cover far more ground per step than a single fixed-window fit.

    Distortion refinement during THE LOOP is geometry-only, always, when
    ``refine_distortion`` is ``None`` -- ``v1.Refine`` is deliberately
    ignored here regardless of what it says (see
    :func:`iterate_frozen_point_until_stable`'s docstring for why: a
    default ``CalibrationParams()`` refines all 15 coefficients, and
    honouring that during an unstable search would refit distortion every
    iteration against a still-wrong geometry). ``v1.Refine`` IS honoured,
    but only via a single final refit at the converged geometry -- see
    that function. Passing ``refine_distortion`` explicitly is a
    deliberate, documented opt-in to thaw it during the loop itself
    instead; see :func:`_apply_distortion_refine`.
    """
    spec = spec_from_v1_params(v1)
    _apply_distortion_refine(spec, refine_distortion
                              if refine_distortion is not None else False)
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

    ``converged=True`` means only that the fitted PARAMETERS stopped
    moving -- it is not a fit-quality check and is not a substitute for
    looking at ``coverage`` (see :class:`PickCoverage`): a fit using points
    from very few rings, or from a lopsided azimuthal sample, can be
    perfectly stable and still weakly constrained or biased. Both
    :func:`autocalibrate_frozen_point` and this function already warn (via
    :func:`_warn_if_thin_or_lopsided`) when ``coverage`` looks thin or
    lopsided, but that warning uses a floor for "clearly fine," not a
    validated pass/fail line -- inspect ``coverage`` yourself for anything
    that matters.
    """

    converged: bool
    n_iter: int
    fit: IterRecord
    res: PVCalibrationResult
    history: List[IterRecord]
    coverage: Optional[PickCoverage] = None


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
    refine_distortion: Optional[Union[bool, str, Sequence[str]]] = None,
    snr_min: float = 5.0,
    point_pick_kwargs: Optional[dict] = None,
    lm_max_iter: int = 150,
    verbose: bool = False,
) -> IterateResult:
    """Re-seed :func:`autocalibrate_frozen_point` from its own result until
    the geometry stops moving, to reach a good basin from even a genuinely
    blind starting guess (e.g. ``tz=0`` with no tilt information at all).

    Each iteration builds a fresh spec with Lsd/BC/ty/tz bounds
    **re-centered** on the *previous* iteration's converged geometry
    (``bounds_tz_deg`` etc.) -- it is this re-centering, not a fixed window,
    that lets the estimate walk many degrees from a bad start across several
    iterations. Point picking is re-run from scratch each iteration (it
    depends on the geometry used to place its annular ring masks -- see
    ``forward.point_pick``), so this is not the same "one-shot, frozen
    point cloud" guarantee ``autocalibrate_frozen_point`` itself makes
    *within* one call; it is that one-shot mechanism used repeatedly as a
    well-behaved fixed-point iteration.

    **The loop itself is geometry-only, always, unless ``refine_distortion``
    is passed explicitly.** An earlier version of this function honoured
    ``v1_params.Refine`` during the loop the same way every sibling
    pipeline (``autocalibrate_pv``, ``autocalibrate_four_stage``,
    ``autocalibrate_bayesian``, ``autocalibrate_joint``) honours it for a
    single fit -- but ``CalibrationParams()``'s own default refines all 15
    coefficients, so a caller who never touched ``Refine`` at all got a
    silent, unrequested behaviour change: full 15-coefficient refits on
    every iteration of an unstable search, against a still-possibly-wrong
    geometry and freshly re-picked points each time -- exactly the kind of
    extraction-geometry feedback compounding this whole pipeline exists to
    avoid on the geometry side. Fixed by decoupling the two knobs:

    * ``refine_distortion=None`` (the default): the loop is geometry-only
      on every iteration, full stop, regardless of ``v1_params.Refine``.
      Once the geometry has converged, ``v1_params.Refine`` IS honoured --
      but via exactly one additional refit at the converged geometry
      (frozen point cloud, stable Lsd/BC/ty/tz), not during the search.
      That refit only runs if ``v1_params.Refine`` actually asks for some
      coefficient; if it does not (as with every caller in this codebase
      so far), nothing extra happens and the result is unchanged from the
      geometry-only-only behaviour this function has always had.
    * ``refine_distortion=<bool/block name/coefficient list>`` (an
      explicit, deliberate ask): thaws those coefficients during the loop
      itself, every iteration -- unchanged from before. This is a power-
      user opt-in into the less-validated combination described below; it
      is on you to check the result the way the validating notebook's
      "Distortion basis" section does.

    Either way the selector accepted is the same one
    :func:`_apply_distortion_refine` resolves (bool / block name / explicit
    coefficient list).

    **Refining distortion here (via ``refine_distortion``, during the
    loop) is less validated than the geometry-only mode.** The stopping
    criterion below only checks geometry (Lsd/BC/ty/tz) stability, never
    distortion's, and a fresh spec is built from ``v1_params`` each
    iteration, so a refined distortion coefficient does not carry over
    between iterations the way geometry does -- each iteration's LM
    instead re-fits distortion from ``v1_params``'s original value against
    that iteration's (still possibly wrong) geometry and its freshly
    re-picked points. If you pass ``refine_distortion`` here, check the
    final values against a separate refit at the converged geometry (as
    the validating notebook's "Distortion basis" section does) rather than
    trusting them as-is. The geometry-only mode (the default) remains the
    only mode actually validated end-to-end.

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
        spec = _bounded_spec_for_iteration(
            seed, bounds_tz_deg=bounds_tz_deg, bounds_ty_deg=bounds_ty_deg,
            bounds_bc_px=bounds_bc_px, bounds_lsd_um=bounds_lsd_um,
            refine_distortion=refine_distortion,
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
                if refine_distortion is None:
                    # v1_params.Refine is honoured here, once, at the
                    # converged geometry -- never during the unstable
                    # search above (see this function's docstring). Gated
                    # on there actually being something to thaw, so the
                    # common case (Refine already all-False, as every
                    # existing caller in this codebase does) costs nothing
                    # extra and returns byte-identical output to before
                    # this fix.
                    final_v1 = _reseed(seed, fit)
                    trial_spec = spec_from_v1_params(final_v1)
                    wants_distortion = any(
                        trial_spec.parameters[n].refined for n in P_COEF_NAMES
                        if n in trial_spec.parameters
                    )
                    if wants_distortion:
                        res = autocalibrate_frozen_point(
                            final_v1, image, snr_min=snr_min,
                            point_pick_kwargs=point_pick_kwargs,
                            lm_max_iter=lm_max_iter, verbose=False,
                        )
                        fit = res.history[0]
                        history.append(fit)
                _write_back_geometry(v1_params, fit)
                return IterateResult(converged=True, n_iter=i, fit=fit,
                                      res=res, history=history,
                                      coverage=res.coverage)
        seed = _reseed(seed, fit)

    _write_back_geometry(v1_params, history[-1])
    return IterateResult(converged=False, n_iter=max_iter, fit=history[-1],
                          res=res, history=history,
                          coverage=res.coverage if res is not None else None)


__all__ = ["autocalibrate_frozen_point", "iterate_frozen_point_until_stable",
           "IterateResult", "FrozenPointCalibrationResult", "PickCoverage"]
