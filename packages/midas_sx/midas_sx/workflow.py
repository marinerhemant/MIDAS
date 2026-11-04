"""One-call workflow: rotation frames -> corrected, merged F² (and a SHELX ``.hkl``) -> refined structure.

Nothing new is computed here. The functions only compose the steps in the order they were validated (``validation/run_c2.py`` for an untwinned
crystal, ``validation/run_pm2.py`` + ``refit_pm.py`` for a 90° twin about c), so that a user does not have to read the validation scripts to
know the order, the filters between the steps and which gates must not be skipped.

* :func:`reduce_scan`: spots -> orientation/geometry refinement -> profile model -> fixed-support integration with the corrections -> merge.
* :func:`reduce_twin90_scan`: the same for a crystal twinned by the 90° rotation about c (exact when a = b, split spots when a != b);
  the orientation is refined on the pair model, every predicted component or blend is integrated on a support widened by its separation.
* :func:`refine_structure` / :func:`refine_twin90_structure`: the weighted least-squares refinement on the result.

The orientation gate is not optional: a non-converged orientation raises :class:`OrientationNotConverged` unless ``allow_unconverged=True``
(``orient.index_and_refine``: the caller must not use it silently). A refinement that is not converged, or has GooF > 3, warns.

Status: every step is validated on SIMULATED data only (see ``validation/DESIGN.md``); nothing here has seen a real frame.
"""
from __future__ import annotations

import warnings
from dataclasses import replace, dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from midas_hkls import Crystal, Lattice, SpaceGroup, corrections as C

from .geometry import Geometry, Scan, make_geometry, spot_directions
from .integrate import DEFAULT_BACKGROUND, ProfileModel, build_profile
from .merge import merge_observations, r_int, scale_groups, scale_smooth
from .orient import OrientationResult, index_and_refine
from .pipeline import find_spots
from .predict import TWIN_90_C, enumerate_hkl, predicted_crossings, twin_targets
from .reduce import DacModel, PolarizationModel, reduce_spots
from .refine import RefineModel, RefinementResult, refine
from .twin_orient import RZ90, index_and_refine_twin

__all__ = ["ScanSetup", "Reduction", "TwinReduction", "MergedRuns", "merge_runs", "OrientationNotConverged", "reduce_scan", "reduce_twin90_scan",
           "refine_structure", "refine_twin90_structure", "predicted_targets"]


class OrientationNotConverged(RuntimeError):
    """The orientation/geometry refinement did not pass its gates (the result carries the reason)."""

    def __init__(self, result: OrientationResult):
        super().__init__(f"orientation refinement did not converge: {result.reason} (n_used {result.n_used} of {result.n_spots}, "
                         f"rms {result.rms_px:.2f} px / {result.rms_frame:.2f} frames)")
        self.result = result


@dataclass(frozen=True)
class ScanSetup:
    """What the header of a rotation scan must supply. ``omega_start`` is the START EDGE of frame 0 (see :class:`midas_sx.geometry.Scan`).

    Defaults that are silent: ``dac = DacModel()`` has ``mu_per_cm = 0`` (NO anvil absorption); pass a real model for a diamond-anvil cell. ``pol`` is 99 % horizontal."""
    wavelength_A: float
    dist_mm: float
    pixel_mm: float
    centre_rc: Tuple[float, float]                 # beam position (row, col), pixel centres at integers
    omega_start: float
    omega_step: float
    dac: DacModel = field(default_factory=DacModel)
    pol: PolarizationModel = field(default_factory=PolarizationModel)
    wedge_deg: float = 0.0                                   # tilt of the omega axis about the lab y axis (MIDAS Wedge)
    tilts_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # (tx, ty, tz) of the detector plane about the beam position, MIDAS convention; a swung detector (2theta != 0) needs one of them

    def scan(self, n_frames: int) -> Scan:
        return Scan(self.omega_start, self.omega_step, int(n_frames))

    def geometry(self, shape: Tuple[int, int], n_frames: int) -> Geometry:
        g = make_geometry(wavelength_A=self.wavelength_A, dist_mm=self.dist_mm, pixel_mm=self.pixel_mm, centre_rc=self.centre_rc,
                          shape=(int(shape[0]), int(shape[1])), scan=self.scan(n_frames))
        tx, ty, tz = (float(v) for v in self.tilts_deg)
        return replace(g, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=float(self.wedge_deg)) if (tx or ty or tz or self.wedge_deg) else g


@dataclass
class Reduction:
    obs: pd.DataFrame                  # every kept observation (hkl, omega, I, sigI, F2, sigF2, corrections ...)
    rejected: pd.DataFrame
    counts: Dict[str, int]             # one entry per rejection reason plus input and kept
    merged: pd.DataFrame               # unique reflections: h k l n F2 sigF2 sig_int sig_ext R_int_refl
    r_int: float
    U: np.ndarray
    geom: Geometry
    orientation: Optional[OrientationResult]
    profile: ProfileModel
    spots: pd.DataFrame
    mask: np.ndarray
    info: dict
    targets: Optional[pd.DataFrame] = None   # integrate_predicted=True only: every integrated predicted crossing (frame row col integrated blob_found blob_dist h_pred k_pred l_pred)


@dataclass
class TwinReduction:
    obs: pd.DataFrame                  # one row per measured target: kind (blend|c1|c2), f1, f2 component flags, p0 p1 p2 partner index, sep_px
    counts: Dict[str, Dict[str, int]]
    U1: np.ndarray
    U2: np.ndarray
    geom: Geometry
    orientation: OrientationResult
    pair_orientation_used: bool        # False: the pair model did not converge and the centroid model was used instead
    profile: ProfileModel
    targets: pd.DataFrame
    spots: pd.DataFrame
    mask: np.ndarray
    info: dict


def _corner_tth_max_deg(geom: Geometry) -> float:
    """Largest 2theta over the four detector corners (a swung or off-centre detector has no meaningful inscribed-edge value)."""
    nz, ny = float(geom.n_pix_z), float(geom.n_pix_y)
    _, tth, _ = spot_directions(np.array([0.0, 0.0, nz - 1.0, nz - 1.0]), np.array([0.0, ny - 1.0, 0.0, ny - 1.0]), geom)
    return float(np.max(tth))


def predicted_targets(spots: pd.DataFrame, lattice: Lattice, space_group: SpaceGroup, U: np.ndarray, geom: Geometry, scan: Scan, *,
                      d_min: Optional[float] = None, own_px: float = 2.5, own_frames: float = 2.5, margin_px: float = 3.0):
    """Every predicted Ewald crossing as an integration target, plus the neighbour list for the crowding guard.

    ``targets``: frame, row, col (the PREDICTED position), integrated (the sum of the found blob that owns the target, NaN when none),
    blob_found, blob_dist (max-norm distance to that blob in px, frames scaled by ``own_px / own_frames``), h_pred, k_pred, l_pred.
    A blob OWNS a target when it lies within ``own_px`` pixels and ``own_frames`` frames of it. ``neighbours``: the found blobs that own no
    target plus every target position, so that a target's own blob is not counted as a crowding neighbour (``reduce_spots`` excludes
    only IDENTICAL coordinates), while a strong blob that belongs to no prediction (ice, a second grain) still is. ``d_min`` defaults to the
    corner 2theta of the detector (off-detector crossings are dropped by ``predicted_crossings``).
    """
    from scipy.spatial import cKDTree
    if d_min is None:
        d_min = geom.wavelength_A / (2.0 * np.sin(np.radians(_corner_tth_max_deg(geom) / 2.0)))
    pc = predicted_crossings(U, lattice, enumerate_hkl(lattice, d_min, space_group), geom, scan, margin_px=margin_px).reset_index(drop=True)
    fs = own_px / own_frames
    pos_t = np.c_[pc.frame.to_numpy(float) * fs, pc.row.to_numpy(float), pc.col.to_numpy(float)]
    pos_b = np.c_[spots.frame.to_numpy(float) * fs, spots.row.to_numpy(float), spots.col.to_numpy(float)]
    dist, j = cKDTree(pos_b).query(pos_t, p=np.inf, distance_upper_bound=own_px) if len(spots) and len(pc) else (np.full(len(pc), np.inf), np.zeros(len(pc), int))
    found = np.isfinite(dist)
    integ = np.full(len(pc), np.nan)
    if "integrated" in spots and found.any():
        integ[found] = spots["integrated"].to_numpy(float)[j[found]]
    targets = pd.DataFrame(dict(frame=pc.frame.to_numpy(float), row=pc.row.to_numpy(float), col=pc.col.to_numpy(float), integrated=integ,
                                blob_found=found, blob_dist=np.where(found, dist, np.nan), h_pred=pc.h.to_numpy(int), k_pred=pc.k.to_numpy(int), l_pred=pc.l.to_numpy(int)))
    if len(spots) and len(pc):
        owned = np.isfinite(cKDTree(pos_t).query(pos_b, p=np.inf, distance_upper_bound=own_px)[0])
    else:
        owned = np.zeros(len(spots), bool)
    nb = pd.concat([spots.loc[~owned, ["frame", "row", "col"]], targets[["frame", "row", "col"]]], ignore_index=True)
    return targets, nb


def _profile(raw, spots, geom, mask, background=None) -> ProfileModel:
    s, tth, _ = spot_directions(spots.row.to_numpy(float), spots.col.to_numpy(float), geom)
    L = C.rotation_lorentz(s, min_abs=1e-3)
    ok = (tth > 2.0) & np.isfinite(L)
    return build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[ok], L[ok], mask=mask,
                         background=DEFAULT_BACKGROUND if background is None else background)[0]


def reduce_scan(raw: np.ndarray, crystal: Crystal, U0: np.ndarray, setup: ScanSetup, *, refine_orientation: bool = True,
                allow_unconverged: bool = False, find_kwargs: Optional[dict] = None, reduce_kwargs: Optional[dict] = None,
                background: Optional[str] = None, refine_tilts: bool = False, refine_wedge: bool = False,
                found: Optional[tuple] = None, margin: Optional[int] = None, robust_merge: bool = True, refine_wavelength: bool = False,
                integrate_predicted: bool = False, predicted_d_min: Optional[float] = None) -> Reduction:
    """Frames ``raw`` (n_frames, rows, cols) of one single crystal -> merged F².

    ``U0`` is the starting orientation (q_sample = U B h). With ``refine_orientation=True`` (default) U, the beam centre and the distance are
    refined from the measured spots first (``orient.index_and_refine``, validated from 0.3 deg / 0.5 px / 1 % off); with False ``U0`` and the
    header geometry are used as given. ``find_kwargs`` go to :func:`midas_sx.pipeline.find_spots`, ``reduce_kwargs`` to
    :func:`midas_sx.reduce.reduce_spots`. ``found`` = a precomputed ``(spots, mask, info)`` of :func:`midas_sx.pipeline.find_spots` for THIS stack and geometry (it skips the 20-minute spot finding of a large stack when the caller already ran it). ``refine_tilts`` / ``refine_wedge`` also refine the three detector tilts / the omega-axis wedge (a few tenths of a degree on a real
    goniometer). ``robust_merge`` (default True) merges the run with the error-model weights and the leave-one-out rejection of :func:`midas_sx.merge.merge_observations` (``robust=True``; validated on L-alanine, PREREGISTER A39-FLOOR step 1b); False gives the plain 1/sigma^2 mean. ``margin`` (int, default None) replaces the support margin chosen by the sweep (``margin=1`` is measured extents + 1 frame and pixel), for a data set whose sweep ends
    without a plateau. ``refine_wavelength`` (default False) also refines one joint wavelength / cell-scale eps in ``index_and_refine`` (see there: with the cell fixed it is equivalent to an isotropic scale (1 - eps) of the reference
    cell and is NOT an energy calibration); the refined wavelength is in ``Reduction.geom.wavelength_A`` and ``Reduction.orientation.lambda_cell_scale``. ``integrate_predicted`` (default False) integrates EVERY predicted Ewald crossing (:func:`predicted_targets`, box centred on the prediction, corrections at the calculated crossing) instead of only the found blobs: a reflection the blob finder missed, split or mis-assigned is measured, and weak reflections are not selected by the detection threshold; the observations carry ``blob_found`` / ``blob_dist`` / ``h_pred k_pred l_pred`` and ``Reduction.targets`` the target table. The integration box (half-width in frames ``ext_f0 + ext_f1 L + margin``) is wide enough for the 1-2 frame prediction offsets seen on real data, so the box is not re-centred. ``predicted_d_min`` pins the reflection pool. ``background`` is ``"mean"`` or ``"quadratic"`` (:func:`midas_sx.integrate.integrate_box`; default
    ``integrate.DEFAULT_BACKGROUND``). Raises :class:`OrientationNotConverged` (see the module docstring).
    """
    n_frames, nr, nc = raw.shape
    scan = setup.scan(n_frames)
    geom0 = setup.geometry((nr, nc), n_frames)
    spots, mask, info = found if found is not None else find_spots(raw, geom0, **(find_kwargs or {}))
    ores = None
    U, geom = np.array(U0, dtype=float), geom0          # a copy: Reduction.U must not alias the caller's array
    if refine_orientation:
        ores = index_and_refine(spots, crystal.lattice, U, geom0, scan, refine_tilts=refine_tilts, refine_wedge=refine_wedge, refine_wavelength=refine_wavelength)
        if not ores.converged and not allow_unconverged:
            raise OrientationNotConverged(ores)
        if not ores.converged:
            warnings.warn(f"reduce_scan: continuing with a NON-converged orientation (allow_unconverged=True): {ores.reason}", RuntimeWarning, stacklevel=2)
        U, geom = ores.U, ores.geom
    profile = _profile(raw, spots, geom, mask, background)
    if margin is not None:
        profile = profile.with_margin(margin)
    targets = None
    if integrate_predicted:
        targets, neigh = predicted_targets(spots, crystal.lattice, crystal.space_group, U, geom, scan, d_min=predicted_d_min)
        rk = dict(reduce_kwargs or {})
        rk.setdefault("neighbours", neigh)
        obs, rej, counts = reduce_spots(targets[["frame", "row", "col", "integrated"]], raw, geom, scan, crystal.lattice, U, profile, dac=setup.dac, pol=setup.pol, mask=mask, **rk)
        key = ["frame", "row", "col"]
        obs = obs.merge(targets[key + ["blob_found", "blob_dist", "h_pred", "k_pred", "l_pred"]], on=key, how="left", validate="many_to_one")
        info = dict(info, predicted=dict(n_targets=int(len(targets)), n_blob_found=int(targets.blob_found.sum()), n_kept=int(len(obs)),
                                         n_hkl_differs=int(((obs.h != obs.h_pred) | (obs.k != obs.k_pred) | (obs.l != obs.l_pred)).sum())))
    else:
        obs, rej, counts = reduce_spots(spots, raw, geom, scan, crystal.lattice, U, profile, dac=setup.dac, pol=setup.pol, mask=mask,
                                        **(reduce_kwargs or {}))
    merged = merge_observations(obs, crystal.space_group, robust=robust_merge)
    return Reduction(obs, rej, counts, merged, r_int(obs, crystal.space_group), U, geom, ores, profile, spots, mask, info, targets=targets)


@dataclass
class MergedRuns:
    obs: pd.DataFrame        # every observation of every run, F2 and sigF2 scaled, with the group column and (smooth) ``k_smooth``
    merged: pd.DataFrame     # one row per unique reflection (``merge_observations``)
    scales: pd.DataFrame     # per-group scale table of the scaling step (empty when scale=None)
    r_int: float
    scale: Optional[str]
    robust: bool


def merge_runs(runs, space_group: SpaceGroup, *, scale: Optional[str] = "smooth", robust: bool = True, group: str = "run", n_knots: int = 8, **scale_kwargs) -> MergedRuns:
    """Scale several runs (rotation scans) to each other and merge them: the validated multi-run entry point.

    ``runs``: a mapping ``{name: Reduction | observation DataFrame}``, or one DataFrame that already has the ``group`` column. ``scale``:
    ``"smooth"`` (default; :func:`midas_sx.merge.scale_smooth`, a piecewise-linear log scale of each run over the frame index, ``n_knots`` knots, fitted from symmetry equivalents only),
    ``"group"`` (one scale per run, :func:`~midas_sx.merge.scale_groups`) or ``None``. ``robust`` (default True): error-model weights and leave-one-out rejection in the merge.
    On L-alanine (3 runs of a Pilatus rotation scan) this took R_int from 15.7 % to 6.8 % and the model-free R_F2 against an independent reference from 14.2 % to 6.3 %
    (PREREGISTER A39-FLOOR, verified); the smooth scale absorbs anything smooth in the frame index that multiplies a whole run (beam or sample-volume drift, centring), so it
    needs equivalents spread over each run's whole range. With little redundancy (fewer than ``min_obs_per_knot`` shared observations per knot) it falls back to fewer knots, down to a plain run scale."""
    if isinstance(runs, pd.DataFrame):
        obs = runs.copy()
        if group not in obs.columns:
            raise ValueError(f"merge_runs: the DataFrame has no '{group}' column")
    else:
        parts = [(r if isinstance(r, pd.DataFrame) else r.obs).assign(**{group: name}) for name, r in runs.items()]      # a Reduction, or anything with an ``obs`` table
        if not parts:
            raise ValueError("merge_runs: no runs")
        obs = pd.concat(parts, ignore_index=True)
    if scale == "smooth":
        if "frame" not in obs.columns:
            raise ValueError("merge_runs: scale='smooth' needs a 'frame' column (use scale='group' otherwise)")
        obs, table = scale_smooth(obs, space_group, group=group, n_knots=n_knots, **scale_kwargs)
    elif scale == "group":
        obs, table = scale_groups(obs, space_group, group=group, **scale_kwargs)
    elif scale is None:
        table = pd.DataFrame()
    else:
        raise ValueError(f"merge_runs: scale must be 'smooth', 'group' or None, got {scale!r}")
    merged = merge_observations(obs, space_group, robust=robust)
    return MergedRuns(obs, merged, table, r_int(obs, space_group, merge_kwargs=dict(robust=True) if robust else None), scale, robust)


def reduce_twin90_scan(raw: np.ndarray, crystal: Crystal, U0: np.ndarray, setup: ScanSetup, *, d_min: Optional[float] = None, m0: float = 0.3,
                       allow_unconverged: bool = False, find_kwargs: Optional[dict] = None, background: Optional[str] = None) -> TwinReduction:
    """Frames of a crystal twinned by the 90° rotation about c (second component ``U2 = U1 Rz(+90°)``, partner index ``(k, -h, l)``).

    Chain of PREREGISTER A27: centroid orientation first, then the PAIR-model orientation (blends are one measurement of both components),
    then every predicted component or blend is integrated on a support widened by its separation. If the pair model does not converge the
    centroid orientation is used (with a warning) and ``pair_orientation_used`` is False. The twin law and the cell (a, b, c) are inputs, never fitted.
    ``d_min`` defaults to the inscribed detector edge (:func:`_tth_max_deg`), the validated definition; pass it explicitly to pin the reflection pool.
    ``background`` as in :func:`reduce_scan`. Feed the result to :func:`refine_twin90_structure`.
    """
    n_frames, nr, nc = raw.shape
    scan = setup.scan(n_frames)
    geom0 = setup.geometry((nr, nc), n_frames)
    spots, mask, info = find_spots(raw, geom0, **(find_kwargs or {}))
    lattice, sg = crystal.lattice, crystal.space_group
    if d_min is None:
        d_min = setup.wavelength_A / (2.0 * np.sin(np.radians(_tth_max_deg(geom0) / 2.0)))
    hkl = enumerate_hkl(lattice, d_min, sg)
    o0 = index_and_refine(spots, lattice, np.asarray(U0, float), geom0, scan)
    if not o0.converged and not allow_unconverged:
        raise OrientationNotConverged(o0)
    if not o0.converged:
        warnings.warn(f"reduce_twin90_scan: continuing with a NON-converged orientation (allow_unconverged=True): {o0.reason}", RuntimeWarning, stacklevel=2)
    o1 = index_and_refine_twin(spots, lattice, o0.U, o0.geom, scan, hkl, m0=m0)
    pair_ok = bool(o1.converged)
    if not pair_ok:
        warnings.warn(f"reduce_twin90_scan: the pair-model orientation did not converge ({o1.reason}); using the centroid orientation, which is biased for blended "
                      "pairs (PREREGISTER A27: 0.049 deg against 0.0055 deg at b/a 1.011). TwinReduction.pair_orientation_used is False.", RuntimeWarning, stacklevel=2)
        o1 = o0
    U1, geom = o1.U, o1.geom
    U2 = U1 @ RZ90
    profile = _profile(raw, spots, geom, mask, background)
    hw_f, hw_p = profile.half_widths(10.0)
    tg = twin_targets(U1, U2, lattice, geom, scan, hkl, twin_matrix=TWIN_90_C, d_res_px=2.0 * hw_p, d_res_frames=2.0 * hw_f)
    neigh = tg[["frame", "row", "col"]]
    key = lambda d: list(zip(d.frame.round(6), d.row.round(6), d.col.round(6)))
    parts, counts = [], {}
    for kinds, Uu in ((("blend", "c1"), U1), (("c2",), U2)):
        sub = tg[tg.kind.isin(kinds)].reset_index(drop=True)
        if not len(sub):
            continue
        obs, _, cnt = reduce_spots(sub[["frame", "row", "col", "extra_px", "extra_frames"]], raw, geom, scan, lattice, Uu, profile, dac=setup.dac,
                                   pol=setup.pol, mask=mask, crowd_px=6.0, neighbours=neigh)
        counts["+".join(kinds)] = cnt
        if len(obs):
            kk = {k_: i_ for i_, k_ in enumerate(key(sub))}
            idx = [kk[k_] for k_ in key(obs)]
            parts.append(obs.assign(kind=sub.kind.to_numpy()[idx], f1=sub.f1.to_numpy()[idx], f2=sub.f2.to_numpy()[idx], p0=sub.p0.to_numpy()[idx],
                                    p1=sub.p1.to_numpy()[idx], p2=sub.p2.to_numpy()[idx], sep_px=sub.sep_px.to_numpy()[idx]))
    O = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return TwinReduction(O, counts, U1, U2, geom, o1, pair_ok, profile, tg, spots, mask, info)


def _tth_max_deg(geom: Geometry) -> float:
    """Largest 2theta of the INSCRIBED detector edge, 3 px in from the farthest edge along either axis (the definition the validated chain used,
    ``simulate.Experiment.tth_max_deg``: atan((n/2 - 3) px / distance) for a centred beam). Not the corner: a corner-based default asks for
    reflections (a pool twice as large, 2theta 47 against 37 deg on the P11 geometry) that the validated runs never measured."""
    r_px = max(max(geom.bcz_px + 0.5, geom.n_pix_z - 0.5 - geom.bcz_px), max(geom.bcy_px + 0.5, geom.n_pix_y - 0.5 - geom.bcy_px)) - 3.0
    return float(np.degrees(np.arctan(r_px * geom.px_um / geom.lsd_um)))


def _warn(res: RefinementResult, what: str) -> RefinementResult:
    if not res.converged:
        hint = ""
        if "twin.m" in res.at_bound and res.twin_fraction > 0.49:
            hint = (" The twin fraction is on its 0.5 bound: domain 1 is taken to be the MAJORITY domain, so U0 may belong to the minority domain "
                    "(try U0 @ Rz(90 deg), or check that the twin law is right).")
        warnings.warn(f"{what}: refinement NOT converged ({res.reason}); parameters at a bound: {res.at_bound}; GooF {res.goof:.2f}.{hint}", RuntimeWarning, stacklevel=3)
    elif res.poor_fit:
        warnings.warn(f"{what}: GooF {res.goof:.2f} > 3 (a false minimum that touches no bound looks like this)", RuntimeWarning, stacklevel=3)
    return res


def refine_structure(crystal: Crystal, merged: pd.DataFrame, *, model_options: Optional[dict] = None, **refine_options) -> RefinementResult:
    """Weighted least-squares refinement of ``crystal`` (the START model) against ``merged`` (output of :func:`reduce_scan`).

    ``model_options`` go to :class:`midas_sx.refine.RefineModel` (``aniso_sites=``, ``refine_extinction=True`` + ``wavelength_A=`` ...);
    ``refine_options`` go to :func:`midas_sx.refine.refine` (``outlier_sigma=``, ``gate_free=`` ...). Read ``converged``, ``at_bound``,
    ``null_params`` and ``goof`` before quoting a parameter.
    """
    model = RefineModel(crystal, **(model_options or {}))
    res = refine(model, merged[["h", "k", "l"]].to_numpy(int), merged.F2.to_numpy(), merged.sigF2.to_numpy(), **refine_options)
    return _warn(res, "refine_structure")


def refine_twin90_structure(crystal: Crystal, tred: TwinReduction, *, twin_start: float = 0.15, model_options: Optional[dict] = None,
                            robust: bool = True, gate_free: Sequence[str] = (), **refine_options) -> RefinementResult:
    """Refine the structure and the twin fraction on a :class:`TwinReduction` (component flags and partner indices, PREREGISTER A26/A27).

    ``robust=True`` applies the studentized outlier rejection (``outlier_sigma=5``) of A27; ``gate_free`` names parameters whose bound hits
    do not fail convergence (A26 used the undetermined oxygen U). The twin fraction ``twin.m`` is the intensity weight of the second component
    in the model; for split spots (a != b) the A27 reading says to treat it as a nuisance weight, NOT a volume fraction (see the README).
    """
    O = tred.obs
    if len(O) < 20:
        raise ValueError(f"only {len(O)} observations: nothing to refine")
    H = O[["h", "k", "l"]].to_numpy(int)
    P = O[["p0", "p1", "p2"]].to_numpy(int)
    c2 = O.kind.to_numpy() == "c2"
    P_eff = P.copy(); P_eff[c2] = H[c2]                          # a c2 row is assigned under U2: its own index is already the domain-2 one
    fl = O[["f1", "f2"]].to_numpy(float)
    model = RefineModel(crystal, refine_twin=True, twin_start=twin_start, **(model_options or {}))
    kw = dict(refine_options)
    if robust:
        kw.setdefault("outlier_sigma", 5.0)
    if gate_free:
        kw.setdefault("gate_free", tuple(gate_free))
    res = refine(model, H, O.F2.to_numpy(), O.sigF2.to_numpy(), partner=P_eff, flags=fl, **kw)
    return _warn(res, "refine_twin90_structure")
