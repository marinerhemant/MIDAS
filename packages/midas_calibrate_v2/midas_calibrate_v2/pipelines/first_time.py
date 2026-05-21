"""First-time calibration — no v1 prior required.

Given only:

  - the calibrant material (lattice, space group),
  - the X-ray wavelength,
  - detector specs (pixel size, dimensions, optional panel layout),
  - one calibrant image,

return calibrated geometry with diagnostics.

The pipeline stages are:

  1. **Auto-seed** (Hough + arc-fit on the image) → BC and a rough Lsd.
  2. **Lsd-only LM** with very wide bounds → drag Lsd into the
     ±0.3 % LM basin (B6) from the rough Hough Lsd (~±20 % accurate, B5).
  3. **Geometry LM** (Lsd + BC + tilts) with distortion frozen →
     stabilise the linear geometry before letting the harmonics in.
  4. **Full LM** via :func:`autocalibrate_robust` → all parameters,
     plus the safety gates from :mod:`pipelines.diagnostics`.
  5. **Lsd grid sweep fallback**: if the gates fail, retry stages 2–4
     from each of {0.5×, 0.75×, 1.0×, 1.5×, 2.0×} Hough Lsd, returning
     the best-strain pass.

Usage::

    res, diag = first_time_calibrate(
        image,
        lattice=(5.4116, 5.4116, 5.4116, 90, 90, 90),
        space_group=225,                  # CeO2, Fm-3m
        wavelength_A=0.196,               # 63 keV
        pixel_size_um=200.0,
        n_pixels_y=2048, n_pixels_z=2048,
        lsd_initial_guess_um=900_000.0,   # rough order of magnitude OK
    )
    print(diag)              # gate report
    print(res.unpacked["Lsd"], res.unpacked["BC_y"], res.unpacked["BC_z"])

If the user has *no* idea about Lsd, pass ``lsd_initial_guess_um=None``
— the routine then defaults to ``300 mm``, which works as a Hough
starting point because the disambiguation step sweeps a 1.5× factor
above and below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import copy
import math
import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..compat.from_v1 import spec_from_v1_params, add_panel_parameters
from ..forward.distortion import P_COEF_NAMES
from ..forward.panels import PanelLayout
from ..parameters.spec import CalibrationSpec
from ..parameters.transforms import Logit
from ..seed.from_image import seed_from_image, SeedResult
from ..seed.hough import hough_seed_bc_lsd
from .diagnostics import (
    DiagnosticResult, run_all_gates, summarise, worst_severity,
)
from .robust import RobustCalibrationDiagnostics
from .single_pv import autocalibrate_pv, PVCalibrationResult


@dataclass
class FirstTimeResult:
    v1_params: V1Params                       # final calibrated geometry
    result: PVCalibrationResult                # full LM result
    diagnostics: RobustCalibrationDiagnostics
    seed: SeedResult                           # what auto-seed produced
    lsd_attempts: List[float] = field(default_factory=list)
                                                # Lsd values tried, in order
    stage_log: List[str] = field(default_factory=list)
                                                # human-readable per-stage log

    def __str__(self) -> str:
        lines = ["FirstTimeResult:",
                 f"  Lsd  = {float(self.result.unpacked['Lsd']):.1f} μm",
                 f"  BC_y = {float(self.result.unpacked['BC_y']):.3f} px",
                 f"  BC_z = {float(self.result.unpacked['BC_z']):.3f} px",
                 f"  ty   = {float(self.result.unpacked['ty']):+.4f} deg",
                 f"  tz   = {float(self.result.unpacked['tz']):+.4f} deg",
                 f"  strain = {self.result.history[-1].mean_strain_uE:.2f} μϵ",
                 f"  Lsd attempts ({len(self.lsd_attempts)}): "
                 f"{['%.0f' % v for v in self.lsd_attempts]}",
                 "  diagnostics:",
                 ]
        for d in self.diagnostics.results:
            lines.append(f"    [{d.severity}] {d.name}: {d.message}")
        return "\n".join(lines)


def _build_v1(
    *,
    pixel_size_um: float,
    n_pixels_y: int, n_pixels_z: int,
    lsd_um: float,
    bc_y: float, bc_z: float,
    wavelength_A: float,
    space_group: int,
    lattice: Tuple[float, float, float, float, float, float],
    max_ring_rad_px: Optional[float] = None,
    refine_per_panel: bool = False,
) -> V1Params:
    if max_ring_rad_px is None:
        # Default to "from BC to nearest detector edge" — covers all rings
        # the image actually contains.
        max_ring_rad_px = float(min(bc_y, n_pixels_y - bc_y,
                                       bc_z, n_pixels_z - bc_z) * 0.95)
    v1 = V1Params(
        NrPixelsY=int(n_pixels_y), NrPixelsZ=int(n_pixels_z),
        pxY=float(pixel_size_um), pxZ=float(pixel_size_um),
        Lsd=float(lsd_um), BC_y=float(bc_y), BC_z=float(bc_z),
        Wavelength=float(wavelength_A),
        SpaceGroup=int(space_group),
        LatticeConstant=tuple(lattice),
        MaxRingRad=float(max_ring_rad_px),
        Width=800.0, EtaBinSize=5.0, RBinSize=0.25,
    )
    # RhoD = BC-to-farthest-edge distance (µm): the natural distortion
    # normalisation radius for the automated / from-scratch path.
    from ..forward.sanity import detector_max_corner_dist_um
    v1.RhoD = detector_max_corner_dist_um(
        int(n_pixels_y), int(n_pixels_z), float(bc_y), float(bc_z),
        float(pixel_size_um), float(pixel_size_um),
    )
    v1.PerPanelLsd = bool(refine_per_panel)
    v1.PerPanelDistortion = bool(refine_per_panel)
    v1.validate()
    return v1


def _hough_seed(
    image: np.ndarray,
    *,
    v1: V1Params,
    panel_mask: Optional[np.ndarray] = None,
    sim_radii_px: Optional[np.ndarray] = None,
    method: str = "hough",            # "hough" (fast) | "arcs" (precise but slow)
) -> SeedResult:
    """Get (BC, Lsd) seeds from the image.

    Two methods are supported:

      - ``"hough"`` (default): fast Hough vote over a sweep of Lsd
        hypotheses.  ~3–10 s on 2880² images.  Recovers BC to ±1 px
        and Lsd to ±20 % per the B5 measurements.  Sufficient for the
        Lsd-only LM stage to take over.

      - ``"arcs"``: full arc-detection + chord-bisector / circle-fit
        pipeline from :func:`seed_from_image`.  More precise BC
        (±0.5 px on a clean image) but ~5–10 min on 2880² images
        because of dense connected-components labelling.
    """
    if sim_radii_px is None:
        from midas_calibrate.rings import build_ring_table
        rt = build_ring_table(v1)
        sim_radii_px = np.array(rt.r_ideal_px, dtype=np.float64)

    if method == "hough":
        # Downsample large images for the Hough vote — the BC seed
        # we need only requires ±1 px accuracy and the LM stages
        # take it from there.  4× downsampling cuts Hough wall by ~16×
        # on 2880² images.
        ds = 4 if max(image.shape) > 1024 else 1
        img_ds = image[::ds, ::ds] if ds > 1 else image
        mask_ds = (panel_mask[::ds, ::ds]
                   if (ds > 1 and panel_mask is not None)
                   else panel_mask)
        sim_radii_ds = sim_radii_px / ds
        # Constrain the Hough accumulator to a tight window around the
        # user's BC guess (±100 downsampled px = ±400 raw px on 4×).
        # Without this the vote drifts to bright off-center beam-stop or
        # asymmetric arc fragments.
        bc_guess_ds = (float(v1.BC_y) / ds, float(v1.BC_z) / ds)
        # The cone-aware refinement is shipped as an opt-in module
        # (cone_aware_bc_refine) but not engaged by default in this
        # pipeline; the Hough vote alone keeps the BC within the LM
        # basin width on the framework's documented tilt envelope.
        # Users with strongly tilted detectors may pass cone_aware=True
        # explicitly through the underlying hough_seed_bc_lsd entry.
        bc_y_ds, bc_z_ds, best_lsd, n_match = hough_seed_bc_lsd(
            img_ds,
            sim_radii_px=sim_radii_ds,
            initial_lsd=float(v1.Lsd),
            panel_mask=mask_ds,
            bc_guess=bc_guess_ds,
            bc_search_radius_px=100.0,      # ±400 raw px window
            lsd_search_factor=1.5,
            n_lsd_candidates=6,
            sample_step_pixels=1,           # already downsampled
        )
        bc_y = bc_y_ds * ds
        bc_z = bc_z_ds * ds
        return SeedResult(
            bc_y=float(bc_y), bc_z=float(bc_z), Lsd=float(best_lsd),
            n_arcs=0, n_rings=int(n_match),
            detected_radii_px=np.zeros(0, dtype=np.float64),
            arc_coords=np.zeros((0, 2), dtype=np.float64),
        )

    if method == "arcs":
        return seed_from_image(
            image,
            sim_radii_px=sim_radii_px,
            initial_lsd=float(v1.Lsd),
            npy=int(v1.NrPixelsY), npz=int(v1.NrPixelsZ),
            bc_guess=(float(v1.BC_y), float(v1.BC_z)),
            panel_mask=panel_mask,
        )

    raise ValueError(f"unknown seed method {method!r}; use 'hough' or 'arcs'")


def _lsd_only_spec(
    base_spec: CalibrationSpec,
    *,
    lsd_init: float,
    lsd_tol_factor: float = 0.5,        # ±50 %
) -> CalibrationSpec:
    """Return a deep-copy spec where every refinable parameter is
    frozen except ``Lsd``, with very wide Lsd bounds."""
    spec = copy.deepcopy(base_spec)
    for nm, p in spec.parameters.items():
        p.refined = (nm == "Lsd")
    if "Lsd" in spec.parameters:
        p = spec.parameters["Lsd"]
        p.init = float(lsd_init)
        lo = float(lsd_init) * (1.0 - lsd_tol_factor)
        hi = float(lsd_init) * (1.0 + lsd_tol_factor)
        p.bounds = (lo, hi)
        p.transform = Logit(lo, hi)
    return spec


def _bc_only_spec(
    base_spec: CalibrationSpec,
    *,
    bc_tol_px: float = 100.0,
) -> CalibrationSpec:
    """Refine Lsd + BC only; freeze tilts at 0 and distortion at 0.

    This is stage 2a: with a possibly-wrong BC seed, refining BC
    *before* unfreezing tilts prevents LM from walking tilts to
    absorb a BC offset (the joint Lsd+BC+tilts LM can land in a
    side basin where BC is fixed and tilts are wrong).
    """
    spec = copy.deepcopy(base_spec)
    for nm in P_COEF_NAMES:
        if nm in spec.parameters:
            spec.parameters[nm].refined = False
            spec.parameters[nm].init = 0.0
    for nm in ("ty", "tz"):
        if nm in spec.parameters:
            spec.parameters[nm].refined = False
            spec.parameters[nm].init = 0.0
    for nm in ("BC_y", "BC_z"):
        if nm in spec.parameters:
            p = spec.parameters[nm]
            cur = float(p.init)
            p.bounds = (cur - bc_tol_px, cur + bc_tol_px)
            p.transform = Logit(*p.bounds)
    return spec


def _geometry_only_spec(
    base_spec: CalibrationSpec,
    *,
    bc_tol_px: float = 30.0,            # tight after stage 2a
    tilts_tol_deg: float = 3.0,
) -> CalibrationSpec:
    """Stage 2b: refine Lsd + BC + tilts with TIGHT BC bounds; freeze
    every harmonic.

    Stage 2a (``_bc_only_spec``) brings BC within a few pixels of
    truth; stage 2b can then safely unfreeze tilts because the BC↔
    tilts trade-off no longer destabilises the optimum.
    """
    spec = copy.deepcopy(base_spec)
    for nm in P_COEF_NAMES:
        if nm in spec.parameters:
            spec.parameters[nm].refined = False
            spec.parameters[nm].init = 0.0
    for nm in ("BC_y", "BC_z"):
        if nm in spec.parameters:
            p = spec.parameters[nm]
            cur = float(p.init)
            p.bounds = (cur - bc_tol_px, cur + bc_tol_px)
            p.transform = Logit(*p.bounds)
    for nm in ("ty", "tz"):
        if nm in spec.parameters:
            p = spec.parameters[nm]
            cur = float(p.init)
            p.bounds = (cur - tilts_tol_deg, cur + tilts_tol_deg)
            p.transform = Logit(*p.bounds)
    return spec


def _try_calibrate_at_lsd(
    *,
    v1_seed: V1Params,
    image: np.ndarray,
    dark: Optional[np.ndarray],
    panel_layout: Optional[PanelLayout],
    lsd_um: float,
    lm_max_iter: int,
    n_iter: int,
    half_window_px: float,
    snr_min: float,
    trim_residual_pct: float,
    verbose: bool,
) -> Tuple[Optional[PVCalibrationResult], List[str]]:
    """Run the 3-stage refinement (Lsd-only → geometry → full) starting
    from ``lsd_um``.  Returns (result, log).  If a stage diverges, we
    return whatever we have plus the log."""
    log: List[str] = []
    v1_try = copy.copy(v1_seed)
    v1_try.Lsd = float(lsd_um)
    v1_try.RhoD = float(v1_try.MaxRingRad) * float(v1_try.pxY)
    v1_try.validate()

    base_spec = spec_from_v1_params(v1_try)

    # ---- stage 1: Lsd only with ±50 % bounds.
    spec1 = _lsd_only_spec(base_spec, lsd_init=lsd_um, lsd_tol_factor=0.5)
    if verbose:
        print(f"  [stage 1] Lsd-only LM, init={lsd_um:.0f} μm, "
              f"bounds=[{lsd_um*0.5:.0f}, {lsd_um*1.5:.0f}]", flush=True)
    try:
        res1 = autocalibrate_pv(
            v1_try, image, dark=dark, spec=spec1, panel_layout=panel_layout,
            n_iter=2, half_window_px=half_window_px, snr_min=snr_min,
            max_per_ring=None, trim_mode="stratified_multfactor",
            trim_residual_pct=trim_residual_pct, huber_delta=None,
            reuse_fits=True, lm_max_iter=lm_max_iter,
            verbose=False,
        )
        v1_try.Lsd = float(res1.unpacked["Lsd"])
        v1_try.RhoD = float(v1_try.MaxRingRad) * float(v1_try.pxY)
        log.append(f"stage1 Lsd→{v1_try.Lsd:.0f} μm  "
                   f"strain={res1.history[-1].mean_strain_uE:.1f} μϵ")
    except Exception as e:
        log.append(f"stage1 FAILED: {type(e).__name__}: {e}")
        return None, log

    # ---- stage 2a: Lsd + BC only (tilts frozen at 0, distortion frozen).
    # Decoupling BC refinement from tilts avoids the joint side-basin
    # where LM walks tilts to absorb a BC offset.
    #
    # CRITICAL: ``reuse_fits=False`` so the cake re-runs at each outer
    # iter with the updated BC.  With ``reuse_fits=True`` (which we use
    # in stages 2b/3 for stability) the fit positions are frozen at
    # the seed BC, and the LM can refine BC in the *spec* but the
    # underlying fits don't move — leading to a side-basin escape.
    base_spec2a = spec_from_v1_params(v1_try)
    spec2a = _bc_only_spec(base_spec2a, bc_tol_px=100.0)
    if verbose:
        print(f"  [stage 2a] BC LM (Lsd+BC, tilts frozen, cake re-runs)",
              flush=True)
    try:
        res2a = autocalibrate_pv(
            v1_try, image, dark=dark, spec=spec2a, panel_layout=panel_layout,
            n_iter=4, half_window_px=half_window_px, snr_min=snr_min,
            max_per_ring=None, trim_mode="stratified_multfactor",
            trim_residual_pct=trim_residual_pct, huber_delta=None,
            reuse_fits=False,                                  # ← key fix
            lm_max_iter=lm_max_iter,
            verbose=False,
        )
        for k in ("Lsd", "BC_y", "BC_z"):
            if k in res2a.unpacked:
                setattr(v1_try, k, float(res2a.unpacked[k]))
        v1_try.RhoD = float(v1_try.MaxRingRad) * float(v1_try.pxY)
        log.append(f"stage2a BC=({v1_try.BC_y:.1f},{v1_try.BC_z:.1f}) "
                   f"strain={res2a.history[-1].mean_strain_uE:.1f} μϵ")
    except Exception as e:
        log.append(f"stage2a FAILED: {type(e).__name__}: {e}")
        return None, log

    # ---- stage 2b: Lsd + BC + tilts (now BC is settled), distortion frozen.
    base_spec2b = spec_from_v1_params(v1_try)
    spec2b = _geometry_only_spec(base_spec2b, bc_tol_px=30.0,
                                   tilts_tol_deg=3.0)
    if verbose:
        print(f"  [stage 2b] geometry LM (Lsd+BC+tilts, distortion frozen)",
              flush=True)
    try:
        res2 = autocalibrate_pv(
            v1_try, image, dark=dark, spec=spec2b, panel_layout=panel_layout,
            n_iter=3, half_window_px=half_window_px, snr_min=snr_min,
            max_per_ring=None, trim_mode="stratified_multfactor",
            trim_residual_pct=trim_residual_pct, huber_delta=None,
            reuse_fits=True, lm_max_iter=lm_max_iter,
            verbose=False,
        )
        for k in ("Lsd", "BC_y", "BC_z", "ty", "tz"):
            if k in res2.unpacked:
                setattr(v1_try, k, float(res2.unpacked[k]))
        v1_try.RhoD = float(v1_try.MaxRingRad) * float(v1_try.pxY)
        log.append(f"stage2b strain={res2.history[-1].mean_strain_uE:.1f} μϵ "
                   f"BC=({v1_try.BC_y:.1f},{v1_try.BC_z:.1f})  "
                   f"tilts=({v1_try.ty:+.3f},{v1_try.tz:+.3f})")
    except Exception as e:
        log.append(f"stage2b FAILED: {type(e).__name__}: {e}")
        return None, log

    # ---- stage 3: full LM (every parameter unfrozen).
    if verbose:
        print(f"  [stage 3] full LM (all params)", flush=True)
    base_spec3 = spec_from_v1_params(v1_try)
    if panel_layout is not None and panel_layout.n_panels() > 1:
        add_panel_parameters(
            base_spec3, n_panels=panel_layout.n_panels(),
            tol_shift_px=4.0, tol_rot_deg=2.0,
            tol_lsd_um=2000.0, tol_p2=2e-2,
            enable_lsd=True, enable_p2=True,
        )
    try:
        res3 = autocalibrate_pv(
            v1_try, image, dark=dark, spec=base_spec3, panel_layout=panel_layout,
            n_iter=n_iter, half_window_px=half_window_px, snr_min=snr_min,
            max_per_ring=None, trim_mode="stratified_multfactor",
            trim_residual_pct=trim_residual_pct, huber_delta=None,
            reuse_fits=True, lm_max_iter=lm_max_iter,
            verbose=False,
        )
        log.append(f"stage3 strain={res3.history[-1].mean_strain_uE:.2f} μϵ "
                   f"Lsd={float(res3.unpacked['Lsd']):.1f}")
        return res3, log
    except Exception as e:
        log.append(f"stage3 FAILED: {type(e).__name__}: {e}")
        return None, log


def first_time_calibrate(
    image: np.ndarray,
    *,
    lattice: Tuple[float, float, float, float, float, float],
    space_group: int,
    wavelength_A: float,
    pixel_size_um: float,
    n_pixels_y: int, n_pixels_z: int,
    lsd_initial_guess_um: Optional[float] = None,
    bc_initial_guess: Optional[Tuple[float, float]] = None,
    dark: Optional[np.ndarray] = None,
    panel_layout: Optional[PanelLayout] = None,
    panel_mask: Optional[np.ndarray] = None,
    refine_per_panel: bool = False,
    strain_threshold_uE: float = 100.0,
    lsd_sweep_factors: Sequence[float] = (1.0, 0.5, 1.5, 0.75, 2.0, 0.33, 3.0),
    n_iter_full: int = 5,
    lm_max_iter: int = 200,
    half_window_px: float = 6.0,
    snr_min: float = 8.0,
    trim_residual_pct: float = 5.0,
    seed_method: str = "hough",         # "hough" (fast, default) | "arcs"
    tilt_prior_deg: Optional[Tuple[float, float]] = None,
    verbose: bool = True,
) -> FirstTimeResult:
    """Calibrate from material + wavelength + detector + image with no v1 prior.

    Parameters
    ----------
    image : 2-D np.ndarray
        Calibrant image.
    lattice : 6-tuple
        (a, b, c, α, β, γ) with lengths in Å and angles in degrees.
    space_group : int
        International table number.  225 for CeO₂ (Fm-3m).
    wavelength_A : float
        X-ray wavelength in Å.
    pixel_size_um : float
        Detector pixel size (assumed isotropic; pxY == pxZ).
    n_pixels_y, n_pixels_z : int
        Detector dimensions.
    lsd_initial_guess_um : optional float
        Rough Lsd guess in μm.  If None, defaults to 300 mm.  The Hough
        seeding step is robust to ±50 % error here.
    bc_initial_guess : optional (y, z) tuple
        Initial BC.  Defaults to image center.
    panel_layout, panel_mask : optional
        Multi-panel detector layout.  ``panel_mask`` is a boolean array
        marking valid pixels (True for live, False for gaps); used by
        the arc-detection step to ignore inter-panel gaps.
    refine_per_panel : bool
        If True, enables per-panel Lsd + p2 refinement in stage 3.
    strain_threshold_uE : float
        Cap for strain_cap diagnostic; runs above this are considered
        basin escapes.  Default 100 μϵ.
    lsd_sweep_factors : sequence of floats
        Lsd hypotheses (multipliers of the Hough Lsd) to try in order.
        We stop at the first one that passes strain_cap + basin_check;
        if all fail, we return the lowest-strain attempt.
    tilt_prior_deg : optional ``(t_y_deg, t_z_deg)`` tuple
        User-supplied tilt prior, accurate to within roughly
        :math:`\\pm 5^\\circ`.  When provided, the seed BC is refined by
        :func:`midas_calibrate_v2.seed.cone.cone_aware_bc_refine_with_tilt_prior`,
        which gates per-ring edges around the predicted ellipse-centre
        (computed from the prior) and extrapolates the per-ring centres
        to :math:`2\\theta \\to 0`.  This pulls the BC seed back inside
        the LM basin even on detectors tilted by 10--15 degrees, where
        the chord-bisector baseline drifts beyond the 60 px LM basin
        width.  Pass ``None`` (the default) for nominal-perpendicular
        detectors, where the chord-bisector seed is sufficient.

    Returns
    -------
    :class:`FirstTimeResult`.
    """
    # ---------- build a seed V1Params for the auto-seed.
    # Track whether BC was user-supplied; image-centre fallback below is
    # only used to construct v1_seed for build_ring_table (sim_radii_px
    # depends on Lsd, not BC).  The auto-seed branch then overrides it.
    user_supplied_bc = bc_initial_guess is not None
    if lsd_initial_guess_um is None:
        lsd_initial_guess_um = 300_000.0
    if bc_initial_guess is None:
        bc_initial_guess = (n_pixels_y / 2.0, n_pixels_z / 2.0)

    v1_seed = _build_v1(
        pixel_size_um=pixel_size_um,
        n_pixels_y=n_pixels_y, n_pixels_z=n_pixels_z,
        lsd_um=lsd_initial_guess_um,
        bc_y=bc_initial_guess[0], bc_z=bc_initial_guess[1],
        wavelength_A=wavelength_A,
        space_group=space_group, lattice=lattice,
        refine_per_panel=refine_per_panel,
    )

    # ---------- stage 0: auto-seed BC and Lsd from the calibrant image.
    # The fully-automated path uses the AutoCalibrateZarr recipe: median-
    # filter background subtraction, threshold, connected-component
    # labelling, chord-bisector per arc → median BC, mean radial
    # distance per arc → ring radii, multi-hypothesis ring-matching to
    # estimate Lsd.  No operator input beyond material+λ+detector specs.
    if user_supplied_bc:
        # Trust the user-supplied seed; build a no-op SeedResult for the
        # report.  This is the recommended path when the user has
        # eyeballed BC from the beamstop — usually faster and more
        # accurate than the auto-seed on images with bright beam stops.
        if verbose:
            print(f"[first_time] stage 0: using user-supplied BC seed "
                  f"= {bc_initial_guess} (auto-seed skipped)",
                  flush=True)
        seed = SeedResult(
            bc_y=float(bc_initial_guess[0]),
            bc_z=float(bc_initial_guess[1]),
            Lsd=float(lsd_initial_guess_um),
            n_arcs=0, n_rings=0,
            detected_radii_px=np.zeros(0, dtype=np.float64),
            arc_coords=np.zeros((0, 2), dtype=np.float64),
        )
    else:
        if verbose:
            print(f"[first_time] stage 0: fully-automated auto-seed "
                  f"(AutoCalibrateZarr recipe)", flush=True)
        # Use AutoCalibrateZarr-style seeding: median bg → threshold →
        # connected components → chord-bisector BC → multi-hypothesis Lsd.
        from ..seed.from_calibrant_image import auto_seed_calibrant
        from midas_calibrate.rings import build_ring_table
        import copy as _copy

        # Generate simulated ring radii at AutoCalibrateZarr's exact
        # anchor: initialLsd = 1_000_000 µm with the cone capped at the
        # detector's longest edge.  Decoupling the sim list from the
        # operator's Lsd guess prevents the matcher from drowning in
        # high-angle rings when the seed Lsd is much smaller than truth.
        ACZ_INITIAL_LSD_UM = 1_000_000.0
        v1_for_sim = _copy.deepcopy(v1_seed)
        v1_for_sim.Lsd = ACZ_INITIAL_LSD_UM
        v1_for_sim.MaxRingRad = float(max(n_pixels_y, n_pixels_z))
        v1_for_sim.RhoD = (float(v1_for_sim.MaxRingRad)
                            * float(v1_for_sim.pxY))
        rt = build_ring_table(v1_for_sim)
        sim_radii_px = np.asarray(getattr(rt, "r_ideal_px",
                                           getattr(rt, "radius_px", None)),
                                    dtype=np.float64)
        # Image needs to be passed in MIDAS convention (already loaded
        # via _load_image which applies the appropriate transform).
        # Note: dark may already be subtracted upstream; if the user
        # passes a dark frame explicitly, we subtract it here too.
        img_for_seed = image.astype(np.float64)
        if dark is not None:
            img_for_seed = img_for_seed - dark.astype(np.float64)

        try:
            auto = auto_seed_calibrant(
                img_for_seed,
                sim_radii_px=sim_radii_px,
                initial_lsd_um=ACZ_INITIAL_LSD_UM,
                min_area=300, median_kernel=101, median_iters=5,
                threshold=None, first_ring=1, max_ring=0,
                skip_median=(dark is not None),
            )
            if verbose:
                print(f"           BC=({auto.BC_y:.2f}, {auto.BC_z:.2f}) "
                      f"Lsd={auto.Lsd_um:.0f} μm  n_arcs={auto.n_arcs}  "
                      f"matched={auto.n_rings_matched} rings", flush=True)
            seed = SeedResult(
                bc_y=float(auto.BC_y), bc_z=float(auto.BC_z),
                Lsd=float(auto.Lsd_um),
                n_arcs=int(auto.n_arcs),
                n_rings=int(auto.n_rings_matched),
                detected_radii_px=auto.detected_radii_px,
                arc_coords=np.zeros((0, 2), dtype=np.float64),
            )
        except Exception as e:
            if verbose:
                print(f"           auto_seed_calibrant failed: {e}; "
                      f"falling back to image-centre + initial Lsd guess",
                      flush=True)
            seed = SeedResult(
                bc_y=float(v1_seed.BC_y), bc_z=float(v1_seed.BC_z),
                Lsd=float(lsd_initial_guess_um),
                n_arcs=0, n_rings=0,
                detected_radii_px=np.zeros(0, dtype=np.float64),
                arc_coords=np.zeros((0, 2), dtype=np.float64),
            )
        v1_seed.BC_y = float(seed.bc_y)
        v1_seed.BC_z = float(seed.bc_z)
        v1_seed.Lsd = float(seed.Lsd) if seed.Lsd > 0 else float(lsd_initial_guess_um)

        # When the user supplied a tilt prior, refine the seed BC by
        # cone-aware ellipse extrapolation: for each ring, predict the
        # ellipse-centre offset from the prior, gate edges around the
        # predicted centre, fit per-ring ellipses, and extrapolate the
        # centre sequence to 2theta -> 0.  This pulls the BC seed back
        # within the LM basin even on detectors tilted by 10-15
        # degrees, where the chord-bisector baseline drifts beyond the
        # 60 px LM basin width.
        if tilt_prior_deg is not None:
            from ..seed.cone import cone_aware_bc_refine_with_tilt_prior
            try:
                rt_for_cone = build_ring_table(v1_seed)
                radii_for_cone = np.asarray(rt_for_cone.r_ideal_px,
                                              dtype=np.float64)
                tt_for_cone = np.radians(np.asarray(
                    rt_for_cone.two_theta_deg, dtype=np.float64))
                ty_p, tz_p = (float(tilt_prior_deg[0]),
                               float(tilt_prior_deg[1]))
                bc_y_ref, bc_z_ref, n_used = (
                    cone_aware_bc_refine_with_tilt_prior(
                        img_for_seed,
                        bc_y_seed=float(v1_seed.BC_y),
                        bc_z_seed=float(v1_seed.BC_z),
                        Lsd_um=float(v1_seed.Lsd),
                        p_x_um=float(pixel_size_um),
                        sim_radii_px=radii_for_cone,
                        two_theta_rad=tt_for_cone,
                        tilt_y_prior_rad=math.radians(ty_p),
                        tilt_z_prior_rad=math.radians(tz_p),
                        panel_mask=panel_mask,
                    )
                )
                if n_used >= 2:
                    if verbose:
                        print(f"           cone-aware refine "
                              f"(tilt prior {ty_p:+.1f}°, {tz_p:+.1f}°): "
                              f"BC=({bc_y_ref:.2f}, {bc_z_ref:.2f}) "
                              f"from {n_used} rings", flush=True)
                    v1_seed.BC_y = float(bc_y_ref)
                    v1_seed.BC_z = float(bc_z_ref)
                elif verbose:
                    print(f"           cone-aware refine: only {n_used} "
                          f"valid rings; keeping chord-bisector BC",
                          flush=True)
            except Exception as e:
                if verbose:
                    print(f"           cone-aware refine failed: {e}; "
                          f"keeping chord-bisector BC", flush=True)
    v1_seed.RhoD = float(v1_seed.MaxRingRad) * float(v1_seed.pxY)

    # ---------- stages 1–3, with Lsd sweep on failure.
    best_result: Optional[PVCalibrationResult] = None
    best_strain = float("inf")
    best_diag: Optional[List[DiagnosticResult]] = None
    best_log: List[str] = []
    lsd_attempts: List[float] = []
    overall_log: List[str] = []

    for factor in lsd_sweep_factors:
        lsd_try = float(seed.Lsd) * float(factor)
        lsd_attempts.append(lsd_try)
        if verbose:
            print(f"\n[first_time] attempt Lsd = {lsd_try:.0f} μm "
                  f"(factor {factor:.2f}× Hough)", flush=True)
        res, stage_log = _try_calibrate_at_lsd(
            v1_seed=v1_seed, image=image, dark=dark,
            panel_layout=panel_layout, lsd_um=lsd_try,
            lm_max_iter=lm_max_iter,
            n_iter=n_iter_full,
            half_window_px=half_window_px,
            snr_min=snr_min,
            trim_residual_pct=trim_residual_pct,
            verbose=verbose,
        )
        overall_log.extend(stage_log)
        if res is None or not res.history:
            continue
        strain = float(res.history[-1].mean_strain_uE)
        # Require ≥ 4 distinct rings in the fit set.  With fewer rings,
        # strain is dominated by a single ring's fit quality and can
        # be artificially small at completely wrong geometries (the
        # ``strain = 0 at Lsd 5× truth'' failure mode where the wrong
        # Lsd happens to put one ring on a high-2θ pixel band).
        n_rings_in_fit = 0
        if res.fits_final is not None:
            try:
                n_rings_in_fit = int(len(set(res.fits_final.ring_idx.cpu().tolist())))
            except Exception:
                n_rings_in_fit = 0
        if n_rings_in_fit < 4:
            if verbose:
                print(f"  ⚠ skipping attempt: only {n_rings_in_fit} ring(s) "
                      f"in fit set (need ≥ 4 for a trustworthy strain)",
                      flush=True)
            continue
        # Build a v1_attempt at the converged params for the gates.
        v1_attempt = copy.copy(v1_seed)
        v1_attempt.Lsd = lsd_try            # for basin_check vs the seed used
        diag_results = run_all_gates(
            v1_init=v1_attempt,
            unpacked=res.unpacked,
            history=res.history,
            fits=res.fits_final,
            panel_layout=panel_layout,
            strain_threshold_uE=strain_threshold_uE,
        )
        sev = worst_severity(diag_results)
        if verbose:
            for d in diag_results:
                icon = {"ok": "  ✓", "warn": "  ⚠", "fail": "  ✗"}.get(
                    d.severity, "  ?")
                print(f"{icon} [{d.name}] {d.message}", flush=True)

        # Track best by strain among non-failed attempts.
        if strain < best_strain:
            best_strain = strain
            best_result = res
            best_diag = diag_results
            best_log = stage_log

        # Accept the first attempt where neither strain_cap nor basin
        # is "fail".  CV gate may flag basis-incompleteness — that's
        # diagnostic information, not a calibration rejection.
        critical_pass = all(
            d.severity != "fail" for d in diag_results
            if d.name in ("strain_cap", "basin_check")
        )
        if critical_pass and strain <= strain_threshold_uE:
            if verbose:
                print(f"[first_time] ✓ accepted Lsd={lsd_try:.0f} μm "
                      f"(strain={strain:.1f} μϵ, critical gates pass)",
                      flush=True)
            break

    if best_result is None:
        raise RuntimeError(
            "first_time_calibrate: every Lsd attempt failed.  "
            "Likely cause: too few rings detected, BC seed is wrong, or "
            "the calibrant isn't the one specified.\n" +
            "\n".join(overall_log)
        )

    # ---------- finalise diag wrapper.
    final_diag_results = best_diag or []
    diag = RobustCalibrationDiagnostics(
        severity=worst_severity(final_diag_results),
        results=final_diag_results,
        auto_seeded=True,
        seed_drift_px=float(((seed.bc_y - bc_initial_guess[0]) ** 2 +
                              (seed.bc_z - bc_initial_guess[1]) ** 2) ** 0.5),
    )

    # Build the final v1 from the converged unpacked dict.
    v1_final = copy.copy(v1_seed)
    for k in ("Lsd", "BC_y", "BC_z", "ty", "tz"):
        if k in best_result.unpacked:
            setattr(v1_final, k, float(best_result.unpacked[k]))
    for i, nm in enumerate(P_COEF_NAMES):
        if nm in best_result.unpacked:
            setattr(v1_final, f"p{i}", float(best_result.unpacked[nm]))
    v1_final.RhoD = float(v1_final.MaxRingRad) * float(v1_final.pxY)
    v1_final.validate()

    if verbose:
        print(f"\n[first_time] DONE.  severity={diag.severity}", flush=True)
        print(summarise(diag.results), flush=True)

    return FirstTimeResult(
        v1_params=v1_final,
        result=best_result,
        diagnostics=diag,
        seed=seed,
        lsd_attempts=lsd_attempts,
        stage_log=overall_log + best_log,
    )


__all__ = [
    "first_time_calibrate",
    "FirstTimeResult",
]
