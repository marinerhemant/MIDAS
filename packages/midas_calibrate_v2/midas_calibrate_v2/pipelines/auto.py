"""One-shot fully-automated calibration: image + wavelength → calibration.

Takes the bare minimum the caller can reasonably supply, runs the entire
v2 stack (seed → autocalibrate → empirical residual-correction map),
and returns a single result object plus an optional v1-compatible
binary residual map for downstream tools.

Typical usage::

    from midas_calibrate_v2 import calibrate
    result = calibrate(
        image,
        wavelength=0.184139,
        pxY=150.0,
        dark=dark_image,
        calibrant="CeO2",
        output_dir="./calib_out",
    )
    # result.Lsd, result.BC_y, result.BC_z, result.ty, result.tz,
    # result.post_residual_strain_uE, result.residual_corr_map, ...
"""
from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, Dict, List

import numpy as np
import torch


# ============================================================ calibrant DB

# Shared with midas_calibrate_v2.seed.auto_seed so calibrate() and
# make_seed() always resolve a calibrant (named or custom dict) to the
# exact same lattice — see midas_calibrate_v2.seed.calibrant for why the
# two used to be independent registries that could (and did) drift apart.
from ..seed.calibrant import (CALIBRANTS, resolve_calibrant, resolve_calibrants,
                              phases_from_calibrants)


def _generate_sim_radii_px(*, lattice_a: float, lattice_b: float, lattice_c: float,
                            alpha: float, beta: float, gamma: float,
                            wavelength: float, px: float,
                            sg: int = 225,
                            Lsd_nominal_um: float = 1_000_000.0,
                            max_2theta_deg: float = 28.0) -> np.ndarray:
    """Predict ring radii (px) at a nominal Lsd for the calibrant.

    Delegates HKL enumeration + Bragg-allowed selection to
    :func:`midas_hkls.generate_hkls`, which handles all 230 space groups
    via proper Hall-symbol symmetry operations (NOT a hand-rolled per-SG
    extinction switch — those silently get one of the 230 SGs wrong).
    Returns the deduplicated, sorted ring radii (px) at the nominal Lsd.
    """
    from midas_hkls import SpaceGroup, Lattice, generate_hkls

    lat = Lattice(
        a=lattice_a, b=lattice_b, c=lattice_c,
        alpha=alpha, beta=beta, gamma=gamma,
    )
    refs = generate_hkls(
        SpaceGroup.from_number(sg), lat,
        wavelength_A=wavelength,
        two_theta_max_deg=max_2theta_deg,
    )
    out = []
    for r in refs:
        R = Lsd_nominal_um * math.tan(math.radians(r.two_theta_deg)) / px
        out.append(R)
    return np.array(sorted(set(round(r, 3) for r in out)))


def _distortion_refine_flags(spec) -> Dict[str, bool]:
    """Map a distortion-block selector onto v1's ``Refine["p0".."p14"]`` flags.

    ``refine_distortion`` used to be a single bool covering all fifteen
    coefficients, so there was no way to act on the azimuth gate's advice
    ("refine the radial block only") without hand-building a spec.
    """
    from ..forward.distortion import (V2_TO_V1_DISTORTION,
                                      resolve_distortion_block)
    names = resolve_distortion_block(spec)
    flags = {f"p{i}": False for i in range(15)}
    for nm in names:
        flags[f"p{int(V2_TO_V1_DISTORTION[nm])}"] = True
    return flags


# ============================================================ result type

class SeedFallbackWarning(UserWarning):
    """The validated seeder failed and a last-resort seed was used instead.

    Not an error: a geometry is still returned.  It fires because the fallback
    is a materially weaker estimate, the refiner tends to stay wherever it is
    put, and ``basin_check`` scores the resulting zero drift as a pass — so
    without this warning a wrong geometry arrives carrying green ticks.
    """


@dataclass
class AutoCalibrationResult:
    """Everything the autocalibration pipeline produces."""

    # Refined geometry (µm and deg)
    Lsd: float
    BC_y: float
    BC_z: float
    tx: float
    ty: float
    tz: float
    # Refined distortion (v2 names: iso_R2/4/6, a1..a6, phi1..phi6)
    distortion: Dict[str, float] = field(default_factory=dict)
    # Detector + wavelength (echoed back for convenience)
    pxY: float = 0.0
    pxZ: float = 0.0
    NrPixelsY: int = 0
    NrPixelsZ: int = 0
    wavelength_A: float = 0.0
    # Quality + provenance
    post_residual_strain_uE: Optional[float] = None
    in_loop_strain_uE: Optional[float] = None
    #: Robust summaries of the SAME residuals as ``post_residual_strain_uE``.
    #:
    #: ``post_residual_strain_uE`` is the plain mean over EVERY fitted point.
    #: The v1 C tool (AutoCalibrateZarr / ``Mean_Difference_Refined``) rejects
    #: outlier fits before reporting, so its published number is a TRIMMED
    #: mean and is not comparable with the plain one — on the shipped 48-panel
    #: Pilatus the C reference reports 18.27 µε where this pipeline reports
    #: 40.76 µε for a geometry that agrees with it to 2.7 ppm in Lsd and
    #: 0.07 px in BC.
    #:
    #: ``IterRecord`` has always computed these; they simply were not surfaced.
    post_residual_strain_median_uE: Optional[float] = None
    post_residual_strain_trim_uE: Optional[float] = None
    # Per-parameter 1σ from the Gauss-Newton covariance at MAP, keyed by v2
    # parameter name.  Empty when the Jacobian is singular.  A refined
    # parameter whose |value| is well under its σ is not measured by the data,
    # however confident the point estimate looks.
    sigma: Dict[str, float] = field(default_factory=dict)
    # Refined parameters that are consistent with zero (|value| < n·σ) or are
    # sitting on a bound.  These are the ones to freeze and re-run.
    unconstrained: List[str] = field(default_factory=list)
    at_bounds: List[str] = field(default_factory=list)
    #: Per-ELEMENT 1σ for vector-valued refined parameters (``panel_delta_yz``
    #: is 48×2, ``panel_delta_lsd`` is 48, …), keyed by parameter name.
    #:
    #: ``sigma`` above holds scalars only, so before 2026-08-29 a run that
    #: refined nothing but vectors — the panel stage freezes every global —
    #: returned an EMPTY ``sigma`` and a trivially empty ``unconstrained``,
    #: which is indistinguishable from "everything is well determined". The
    #: Laplace always computed these numbers; they were simply not recorded.
    sigma_vector: Dict[str, "np.ndarray"] = field(default_factory=dict)
    #: Refined parameters (or vector ELEMENTS, named ``p[i]``) for which the
    #: Laplace produced NO information: σ is 0 or non-finite.
    #:
    #: **σ = 0 does not mean infinite precision — it means the opposite.**
    #: ``sigma_per_dim = sqrt(diag(cov).clamp(min=0))``, so a zero is a
    #: NON-POSITIVE variance clamped away: the Hessian is indefinite in that
    #: direction. A zero can also come from the bounded reparameterisation's
    #: Jacobian ``span·s·(1−s)`` vanishing when a parameter is railed at a
    #: bound. Either way the value is not measured, and the ``|value| < σ``
    #: test that populates ``unconstrained`` cannot fire for it — which is why
    #: these need their own list rather than being folded in.
    undetermined: List[str] = field(default_factory=list)
    # Calibrant name(s) actually used.
    calibrants: List[str] = field(default_factory=list)
    # Diagnostic gate results (azimuth coverage, RhoD scaling, ...).
    diagnostics: List = field(default_factory=list)
    residual_corr_map: Optional[torch.Tensor] = None     # [NrPixelsZ, NrPixelsY] px
    residual_corr_bin_path: Optional[str] = None
    seed_seconds: float = 0.0
    refine_seconds: float = 0.0
    # Raw seed + LM bits (for plotting / further analysis)
    seed_BC_y: float = 0.0
    seed_BC_z: float = 0.0
    seed_Lsd: float = 0.0
    # How the seed was obtained: "user" (BC_guess), "make_seed" (the validated
    # seeder) or "fallback" (last-resort chord-only arc seed).  A "fallback"
    # geometry is NOT of the same quality as a "make_seed" one and should not be
    # reported as though it were.  ``seed_note`` carries why.
    seed_method: str = "unknown"
    seed_note: str = ""
    iter_history: List[Dict] = field(default_factory=list)

    def to_integration_spec(
        self,
        *,
        RMin: float = 0.0,
        RMax: float = 0.0,
        RBinSize: float = 0.0,
        EtaMin: float = -180.0,
        EtaMax: float = 180.0,
        EtaBinSize: float = 5.0,
        RhoD: Optional[float] = None,
    ):
        """Convert this calibration into a ``midas_integrate_v2.IntegrationSpec``.

        Lives here rather than in each consumer so the mapping is written once,
        next to the fields it reads. Downstream packages (midas-dt, and
        anything else integrating against a fresh calibration) call this rather
        than transcribing eight numbers by hand -- which is exactly where a
        sign or unit error enters.

        Two things this handles that a hand-written conversion tends to miss:

        * **The spec's geometry fields are ``torch.Tensor``, not float.** They
          are declared that way so the spec stays differentiable for joint
          refinement. A plain float survives construction and then fails inside
          ``spec.device()``, a long way from the call site.
        * **Distortion naming.** ``self.distortion`` is already in v2 named
          form (``iso_R2``, ``a1..a6``, ``phi1..phi6``), which is what this
          pipeline refines, so it is forwarded as-is. Legacy ``p0..p3`` vectors
          are a DIFFERENT convention and the mapping between them is a
          permutation, not positional -- see
          ``midas_distortion.V1_TO_V2_DISTORTION``. Do not mix the two.

        R/eta binning is not part of a calibration, so pass it here or set it
        on the returned spec.
        """
        try:
            import torch
            from midas_integrate_v2 import IntegrationSpec
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "to_integration_spec() needs midas-integrate-v2 (and torch). "
                "Install with `pip install midas-integrate-v2`."
            ) from exc

        def _t(v):
            return torch.as_tensor(float(v), dtype=torch.float64)

        kw = dict(
            NrPixelsY=int(self.NrPixelsY),
            NrPixelsZ=int(self.NrPixelsZ),
            pxY=float(self.pxY),
            pxZ=float(self.pxZ or self.pxY),
            Lsd=_t(self.Lsd),
            BC_y=_t(self.BC_y),
            BC_z=_t(self.BC_z),
            tx=_t(self.tx), ty=_t(self.ty), tz=_t(self.tz),
            Wavelength=_t(self.wavelength_A),
            RMin=float(RMin), RMax=float(RMax), RBinSize=float(RBinSize),
            EtaMin=float(EtaMin), EtaMax=float(EtaMax),
            EtaBinSize=float(EtaBinSize),
        )
        if RhoD is not None:
            kw["RhoD"] = float(RhoD)
        kw.update({k: _t(v) for k, v in (self.distortion or {}).items()})
        return IntegrationSpec(**kw)


# ============================================================ entry point

def calibrate(
    image: np.ndarray,
    *,
    wavelength: float,
    pxY: float,
    pxZ: Optional[float] = None,
    dark: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    im_trans: tuple = (),
    calibrant: Union[str, Dict, Sequence[Union[str, Dict]]] = "CeO2",
    min_ring_separation_px: float = 0.0,
    blend_exclude_cross_phase_only: bool = False,
    min_eta_bins_per_ring: int = 0,
    min_ring_snr: float = 0.0,
    output_dir: Optional[Union[str, Path]] = None,
    initial_Lsd: float = 1_000_000.0,
    BC_guess: Optional[Tuple[float, float]] = None,
    lsd_window: Optional[float] = None,
    initial_BC_y: Optional[float] = None,
    initial_BC_z: Optional[float] = None,
    max_2theta_deg: float = 28.0,
    min_ring_radius_px: float = 120.0,
    max_ring_radius_px: Optional[float] = None,
    n_iter: int = 4,
    lm_max_iter: int = 200,
    build_residual_corr: bool = True,
    refine_tilts: bool = True,
    refine_distortion: Union[bool, str, Sequence[str]] = True,
    panel_layout=None,
    panel_mode: str = "radius",
    panel_tol_shift_px: float = 3.0,
    panel_tol_rot_deg: float = 1.0,
    panel_tol_radius_px: float = 2.0,
    eta_bin_size: Optional[float] = None,
    r_bin_size: Optional[float] = None,
    peak_width_um: Optional[float] = None,
    weight_by_radius: Optional[bool] = None,
    doublet_separation_px: Optional[float] = None,
    outlier_factor: Optional[float] = None,
    remove_outliers_between_iters: Optional[bool] = None,
    refine_panel_lsd: bool = False,
    refine_panel_p2: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = True,
) -> AutoCalibrationResult:
    """Fully-automated single-image calibration.

    Minimum inputs the caller must supply:

    * ``image``  — 2-D numpy array (raw or dark-subtracted; if dark is
      passed it will be subtracted here).  MIDAS convention:
      ``image.shape = (NrPixelsZ, NrPixelsY)``.
    * ``wavelength`` — X-ray wavelength in Å.
    * ``mask`` — optional bad-pixel mask in the RAW orientation, same shape as
      ``image``; ``im_trans`` is applied to it here so pass it exactly as read
      from disk. **Nonzero means BAD** (matching ``build_map`` and the shipped
      ``mask_upd.tif``, whose nonzero pixels coincide exactly with the dead
      pixels of the Pilatus frame — note the example ``parameters.txt`` comment
      "0 = masked" is wrong; that would mask 92 % of the detector). Masked
      pixels are removed from BOTH the numerator and the denominator of every
      cake cell. **New 2026-08-29** — before that there was no way to pass a
      mask into the calibration at all, so bad pixels entered the cake as
      genuine zeros and diluted every cell they touched.
    * ``eta_bin_size`` / ``r_bin_size`` / ``peak_width_um`` — cake binning for
      the E-step, in degrees / pixels / µm. ``None`` keeps
      ``CalibrationParams``' defaults (5.0 / 0.25 / 800.0). **New 2026-08-29**:
      these existed on the params object but were unreachable from here, and
      the shipped reference ``parameters.txt`` uses ``EtaBinSize 1`` — five
      times finer than the default this entry point forced.
    * ``weight_by_radius`` — weight fitted points by R/Rmax, emphasising the
      outer rings that carry the tilt and distortion leverage. The reference
      sets it; the default is off.
    * ``doublet_separation_px``, ``outlier_factor``,
      ``remove_outliers_between_iters`` — likewise previously unreachable.
    * ``pxY`` — pixel pitch in µm (square pixels assumed unless ``pxZ`` is
      also given).
    * ``dark`` — optional dark-frame array; subtracted from ``image``.
    * ``calibrant`` — name (``"CeO2"``, ``"LaB6"``, ``"Si"``, ``"Al2O3"``)
      or a dict with ``a``, optionally ``c``, ``alpha``, ``gamma``, ``sg``.
      **A list of either** calibrates against a mixed-calibrant exposure
      (e.g. ``["CeO2", "LaB6"]``): the ring table then carries both phases,
      and the result reports the residual per phase so the two can be checked
      against each other.  Seeding still uses the FIRST entry, so list the
      stronger, smoother powder first.
    * ``min_ring_separation_px`` — drop any ring whose nearest neighbour in
      radius is closer than this.  Two interleaved ring sets always produce a
      few collisions, and a blended ring's centroid is dragged by its
      neighbour.  Excluding them is cheaper than modelling them; on a 1-ID
      CeO2+LaB6 frame a 12 px cut costs 6 of 40 rings.  0 disables.
    * ``blend_exclude_cross_phase_only`` — restrict that exclusion to
      collisions between DIFFERENT calibrants, leaving same-phase doublets to
      the doublet co-fitter.
    * ``panel_layout`` — optional :class:`PanelLayout` for tiled-module
      detectors (Pilatus, Eiger).  Build it once with
      ``PanelLayout.regular(n_y, n_z, sy, sz, gap_y, gap_z)`` and pass it in;
      per-panel rigid-body shifts (δy, δz, δθ, and optionally δLsd / δp₂) are
      then refined in the M-step.  ``panel_tol_shift_px`` /
      ``panel_tol_rot_deg`` bound the shift and rotation; this is what brings
      a multi-module Pilatus from a few-hundred µε monolithic fit down to the
      sub-20 µε regime.  Leave ``None`` for monolithic detectors (GE, Varex).
    * ``BC_guess`` — optional ``(BC_y, BC_z)`` in pixels.  When supplied,
      the auto-seed step is **bypassed**; ``BC_guess`` + ``initial_Lsd``
      become the seed and LM refinement proceeds.  Use this as an explicit
      escape hatch for off-panel BC, very weak rings, or any image the
      auto-seed gets wrong.  Pair with a sensible ``initial_Lsd``.
    * ``initial_Lsd`` — nominal sample-to-detector distance (µm).  When
      this differs from the 1 m default, ``lsd_window`` is automatically
      set to ``1.5`` so the multi-hypothesis matcher trusts the prior
      (treats it as a ±50 % window).  Override ``lsd_window`` if you want
      a tighter or looser window.

    User-supplied seed (optional)
    -----------------------------
    Pass **both** ``initial_BC_y`` and ``initial_BC_z`` (and an
    ``initial_Lsd`` matching the geometry) to skip the automatic seeder
    entirely and start the LM refinement directly from those values.
    Useful when:

    * the auto-seeder fails (very sparse / spotty calibrant images,
      off-detector beam centre where chord-bisector can't anchor),
    * or the user already knows the BC from beamline alignment and just
      wants the LM to polish.

    When only one of ``initial_BC_y`` / ``initial_BC_z`` is set, the
    automatic seeder runs as before (the partial hint is ignored).

    Pipeline:

    1. Subtract dark (if given) and clip negatives.
    2. **Seed** BC and Lsd from the image via
       :func:`midas_calibrate_v2.seed.seed_from_image` (median filter +
       chord-bisector + multi-hypothesis Lsd).
    3. Run :func:`midas_calibrate_v2.pipelines.single.autocalibrate` with
       the seeded geometry, refining (Lsd, BC, ty, tz, all 15 distortion).
    4. **Build empirical residual-correction map** (port of v1 C
       ``dg_residual_corr_lookup``) and re-evaluate strain.
    5. Optionally save the residual map as a v1-compatible binary at
       ``output_dir/residual_corr.bin`` — directly consumable by
       ``midas_integrate``, ``midas_integrate_v2``, and the C
       ``CalibrantIntegratorOMP``.

    Returns
    -------
    AutoCalibrationResult
        Full record of refined parameters, residual map, timing,
        and seed provenance.
    """
    import time
    import midas_calibrate_v2.seed       # diplib preload (order matters)
    from midas_calibrate_v2.seed import seed_from_image
    from midas_calibrate_v2.pipelines.single import autocalibrate
    from midas_calibrate.params import CalibrationParams

    if image.ndim != 2:
        raise ValueError(f"image must be 2-D; got shape {image.shape}")
    # MIDAS image transforms (1=flip Y, 2=flip Z, 3=transpose): bring the raw
    # detector image into the geometry-model orientation. Applied to image AND
    # dark so they stay registered; done before BC/shape so everything
    # downstream works in the true frame.
    def _imtrans(arr):
        for opt in im_trans:
            if opt == 1:
                arr = arr[:, ::-1]
            elif opt == 2:
                arr = arr[::-1, :]
            elif opt == 3:
                arr = arr.T
        return np.ascontiguousarray(arr)
    if im_trans:
        image = _imtrans(image)
        if dark is not None:
            dark = _imtrans(dark)
        # The MASK must ride along. A mask left in the raw orientation while
        # the image is flipped masks the wrong pixels — silently, and worse
        # than no mask at all.
        if mask is not None:
            mask = _imtrans(mask)
    if mask is not None and mask.shape != image.shape:
        raise ValueError(
            f"mask shape {tuple(np.shape(mask))} != image shape "
            f"{tuple(image.shape)} (after im_trans={im_trans})")
    NZ, NY = image.shape
    if pxZ is None:
        pxZ = pxY

    # Resolve calibrant.  resolve_calibrant() validates dict calibrants against
    # the crystal system implied by 'sg' and returns the full (a,b,c,alpha,
    # beta,gamma) lattice — the same one make_seed() uses internally, so the
    # seed and the fit never disagree on what the calibrant's lattice is.
    # A list means a mixed-calibrant exposure.  Every phase goes into the ring
    # table; the FIRST is also the one the seeder works from, since the seeder
    # matches an arc pattern against a single ring table.
    cal_specs = resolve_calibrants(calibrant)
    cal = cal_specs[0]
    a, b, c = cal["a"], cal["b"], cal["c"]
    alpha, beta, gamma = cal["alpha"], cal["beta"], cal["gamma"]
    sg = cal["sg"]
    cal_name = cal["name"]
    cal_names = [s["name"] for s in cal_specs]
    seed_calibrant = calibrant[0] if isinstance(calibrant, (list, tuple)) else calibrant

    # 1. Background.  Detect bad-pixel / detector-gap sentinels from the RAW
    # image *before* clipping, so they can be masked out of seeding
    # (otherwise module gaps fragment the ring arcs and bias the beam centre).
    #   - signed dtypes: Pilatus / GE use -1, -2 as dead-pixel / gap markers.
    #   - uint32 dtypes:  Eiger writes 2^32-1 = 4294967295 for dead pixels.
    # The uint32 case is critical because such pixels otherwise blow up
    # ``np.std(corr)`` → the auto-threshold ``100·(1+std//100)`` becomes
    # astronomical and the seed fails with "no connected components above
    # threshold".
    sentinel_mask = np.zeros(image.shape, dtype=bool)
    if np.issubdtype(image.dtype, np.signedinteger) or np.issubdtype(image.dtype, np.floating):
        sentinel_mask |= (image == -1) | (image == -2)
    if image.dtype == np.uint32:
        sentinel_mask |= (image == np.iinfo(np.uint32).max)
    elif image.dtype == np.uint16:
        sentinel_mask |= (image == np.iinfo(np.uint16).max)
    sentinel_mask = sentinel_mask if bool(sentinel_mask.any()) else None
    if dark is not None:
        if dark.shape != image.shape:
            raise ValueError(
                f"dark shape {dark.shape} != image shape {image.shape}"
            )
        img = np.clip(image.astype(np.float32) - dark.astype(np.float32), 0, None)
    else:
        img = np.clip(image.astype(np.float32), 0, None)
    # Zero sentinel pixels in the float image too — uint32 sentinels cast to
    # ~4.3e9 which dwarfs any real signal and breaks both the auto-threshold
    # AND the LM forward model (the M-step would see absurd intensities).
    if sentinel_mask is not None:
        img[sentinel_mask] = 0.0

    # 2. Seed BC + Lsd.
    # Generate simulated ring radii for the nominal geometry — needed by the
    # automatic seeder and the seed_from_image fallback.
    if verbose:
        extra = (f" (of {len(cal_specs)} phases: {', '.join(cal_names)})"
                 if len(cal_specs) > 1 else "")
        print(f"[calibrate] STAGE 1: seeding from {cal_name} rings{extra}...",
              flush=True)
    sim_radii = _generate_sim_radii_px(
        lattice_a=a, lattice_b=b, lattice_c=c, alpha=alpha, beta=beta, gamma=gamma,
        wavelength=wavelength, px=pxY, sg=sg,
        Lsd_nominal_um=initial_Lsd, max_2theta_deg=max_2theta_deg,
    )
    if sim_radii.size < 3:
        raise RuntimeError(
            f"Only {sim_radii.size} simulated rings under "
            f"{max_2theta_deg}\u00b0 — check wavelength/lattice"
        )
    # Accept either BC_guess=(BC_y, BC_z) tuple or the initial_BC_y/initial_BC_z
    # float pair (merged from PR #52) — normalize to a single BC_guess tuple so
    # there is exactly one user-seed code path downstream.
    if BC_guess is None and initial_BC_y is not None and initial_BC_z is not None:
        BC_guess = (float(initial_BC_y), float(initial_BC_z))

    t0 = time.time()
    from types import SimpleNamespace
    seed = None
    seed_method, seed_note = "unknown", ""
    # PATH 1: caller supplied a beam centre -> bypass arc detection entirely.
    # Use it + initial_Lsd as the seed.  Explicit escape hatch for datasets
    # where the automatic seeder fails (off-panel BC, very weak rings).
    if BC_guess is not None:
        bcy, bcz = float(BC_guess[0]), float(BC_guess[1])
        seed = SimpleNamespace(bc_y=bcy, bc_z=bcz, Lsd=float(initial_Lsd),
                               n_arcs=0, n_rings=0)
        seed_method, seed_note = "user", "BC_guess supplied by caller"
        if verbose:
            print(f"[calibrate]   user-provided BC=({bcy:.3f}, {bcz:.3f})  "
                  f"Lsd={initial_Lsd/1000:.3f} mm — auto-seed bypassed",
                  flush=True)
    # PATH 2: robust automatic seeder (auto_seed.make_seed).  Self-determines
    # Lsd from the ring radii; handles monolithic, panel (Pilatus/Eiger) and
    # off-panel-BC geometries via circle-fit + RANSAC.  Validated on 4 real
    # CeO2 geometries to <=3.6 px BC / <=0.24% Lsd, ~10 s each
    # (dev/paper/midas_v2_test/run_seeder_baseline.py).
    # use_diplib=False: diplib's median filter segfaults on macOS and the scipy
    # path is equally accurate.  make_seed accepts either a registered name
    # (CeO2/LaB6/Si/Al2O3) or a custom lattice dict ({'a':..,'sg':..}); both
    # get the robust seeder.  On any failure it falls through to PATH 3.
    if seed is None:
        try:
            from ..seed.auto_seed import make_seed
            ms = make_seed(img, wavelength_A=wavelength, px_um=pxY,
                           calibrant=seed_calibrant, use_diplib=False)
            if ms.Lsd_um and ms.Lsd_um > 0:
                seed = SimpleNamespace(bc_y=ms.BC_y, bc_z=ms.BC_z, Lsd=ms.Lsd_um,
                                       n_arcs=0, n_rings=int(ms.n_measured))
                seed_method = "make_seed"
                seed_note = (f"{ms.n_measured} rings, first_ring={ms.first_ring}, "
                             f"rms={ms.rms_px:.2f} px")
                if verbose:
                    print(f"[calibrate]   make_seed ({seed_note})", flush=True)
            else:
                seed_note = f"make_seed returned Lsd={ms.Lsd_um!r}"
        except Exception as e:  # pragma: no cover - fall back on any seed failure
            seed_note = f"make_seed failed: {e}"
            if verbose:
                print(f"[calibrate]   {seed_note}; using arc seed fallback",
                      flush=True)
    # PATH 3: last-resort chord-only arc seed.
    #
    # This is a DEGRADED path, not an equivalent one: make_seed is the validated
    # seeder (<=3.6 px BC on four real geometries), while this is a chord-only
    # arc fit that can and does return a beam centre off the panel entirely
    # (measured: BC_y = -103 px on a 4148-wide EIGER frame).  The refiner then
    # stays in that basin, and `basin_check` reports the resulting ZERO
    # seed-to-MAP drift as "within safe basin" — so a wrong answer collects a
    # green tick.  Falling back must therefore be visible whatever `verbose` is
    # set to, and must be recorded on the result.
    if seed is None:
        seed = seed_from_image(
            image=img, sim_radii_px=sim_radii,
            initial_lsd=initial_Lsd, npy=NY, npz=NZ,
            bc_guess=BC_guess,
            skip_median=False, min_ring_radius_px=min_ring_radius_px,
        )
        seed_method = "fallback"
        warnings.warn(
            f"calibrate(): the validated seeder did not produce a seed "
            f"({seed_note}); fell back to the last-resort chord-only arc seed, "
            f"which gave BC=({seed.bc_y:.1f}, {seed.bc_z:.1f}) "
            f"Lsd={seed.Lsd / 1000:.1f} mm. Treat this geometry as unverified: "
            f"check the beam centre lies on the panel and that the seed-to-MAP "
            f"drift is not zero before using it.",
            SeedFallbackWarning, stacklevel=2)
    seed_time = time.time() - t0
    if verbose:
        print(f"[calibrate]   BC=({seed.bc_y:.3f}, {seed.bc_z:.3f})  "
              f"Lsd={seed.Lsd/1000:.3f} mm  ({seed.n_arcs} arcs, "
              f"{seed.n_rings} ring matches, {seed_time:.1f}s)", flush=True)

    # 3. Autocalibrate.
    if max_ring_radius_px is None:
        # Pull back ~3% from the actual corner distance — the corner pixels
        # are usually outside the well-illuminated detector area (vignetting,
        # beamstop arm shadow, panel-edge artefacts) and including them
        # gives the LM noisy high-Q fits that pull the geometry into the
        # wrong basin.  Caller can override if needed.
        corner_dist = math.sqrt(
            (max(seed.bc_y, NY - 1 - seed.bc_y)) ** 2
            + (max(seed.bc_z, NZ - 1 - seed.bc_z)) ** 2
        )
        max_ring_radius_px = corner_dist * 0.97 - 10.0
    RhoD_px = math.sqrt(
        (max(seed.bc_y, NY - 1 - seed.bc_y)) ** 2
        + (max(seed.bc_z, NZ - 1 - seed.bc_z)) ** 2
    )
    v1 = CalibrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=pxY, pxZ=pxZ,
        Lsd=seed.Lsd, BC_y=seed.bc_y, BC_z=seed.bc_z,
        tx=0.0, ty=0.0, tz=0.0,
        Wavelength=wavelength, SpaceGroup=sg,
        LatticeConstant=(a, b, c, alpha, beta, gamma),
        MaxRingRad=float(max_ring_radius_px),
        MinRingRad=float(min_ring_radius_px),
        RhoD=RhoD_px,
        nIterations=n_iter,
        Refine={"Lsd": True, "BC": True,
                "ty": bool(refine_tilts), "tz": bool(refine_tilts),
                "Wavelength": False, "Parallax": False,
                **_distortion_refine_flags(refine_distortion)},
        Device=device, Dtype="fp64" if dtype == torch.float64 else "fp32",
    )
    # Multi-calibrant: every phase enters the ring table (Phases wins over the
    # scalar SpaceGroup/LatticeConstant above, which stay set to phase 0 so
    # single-phase consumers of v1 keep working).
    v1.Phases = phases_from_calibrants(calibrant)
    v1.MinRingSeparation = float(min_ring_separation_px)
    v1.MinEtaBinsPerRing = int(min_eta_bins_per_ring)
    v1.MinRingSNR = float(min_ring_snr)
    v1.BlendExcludeCrossPhaseOnly = bool(blend_exclude_cross_phase_only)

    # E-step / objective knobs that CalibrationParams has always carried but
    # calibrate() could not reach. Left as None = keep the dataclass default,
    # so nothing changes for existing callers.
    #
    # These matter: the shipped reference `parameters.txt` sets EtaBinSize 1
    # against a default of 5 (five times coarser azimuthal binning) and
    # WeightByRadius 1 against a default of False, and neither could be
    # requested through this entry point.
    for _attr, _val in (("EtaBinSize", eta_bin_size),
                        ("RBinSize", r_bin_size),
                        ("Width", peak_width_um),
                        ("DoubletSeparation", doublet_separation_px),
                        ("OutlierFactor", outlier_factor)):
        if _val is not None:
            setattr(v1, _attr, float(_val))
    if weight_by_radius is not None:
        # CalibrationParams.WeightByRadius (params.py:119) is DECLARED AND READ
        # NOWHERE in the Python implementation — grepped 2026-08-29, the only
        # other hits are v1-C output files. Setting it changes nothing, and
        # measured: it gave a bit-identical 40.76 µε on the real Pilatus.
        # Carry the value so the params object stays faithful, but say so.
        v1.WeightByRadius = bool(weight_by_radius)
        if weight_by_radius:
            warnings.warn(
                "calibrate(): weight_by_radius has NO effect — "
                "CalibrationParams.WeightByRadius is not consumed anywhere in "
                "the Python calibration (it is a v1-C parameter-file key). The "
                "value is stored on the params object but no code reads it.",
                UserWarning, stacklevel=2)
    if remove_outliers_between_iters is not None:
        v1.RemoveOutliersBetweenIters = bool(remove_outliers_between_iters)

    bin_path = None
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        bin_path = str(out / "residual_corr.bin")

    if verbose:
        print(f"[calibrate] STAGE 2: autocalibrate + residual map...", flush=True)
    t1 = time.time()
    # Phase 1: monolithic geometry + distortion (no panels). On a tiled
    # detector this stalls in the few-hundred-µε range, but it locks the
    # global geometry and distortion that the panel phase builds on.  Defer
    # the residual map to the panel phase when one follows.
    # Snapshot the SEED geometry. ``autocalibrate`` refines ``v1`` in place, so
    # passing ``v1`` itself to ``basin_check`` later compares the final geometry
    # with itself: the drift is identically zero and the gate can only ever
    # report a pass. It has to hold the pre-refinement values to mean anything.
    v1_seed = copy.deepcopy(v1)

    cr = autocalibrate(
        v1, image, dark=dark, mask=mask,
        n_iter=n_iter, lm_max_iter=lm_max_iter,
        dtype=dtype, device=device, verbose=verbose,
        build_residual_corr=build_residual_corr and panel_layout is None,
        residual_corr_path=bin_path if panel_layout is None else None,
    )

    # Phase 2 (tiled detectors): freeze the global geometry + distortion and
    # refine ONLY per-panel rigid-body shifts.  Doing this jointly with the
    # alternating E↔M loop diverges — the E-step (peak extraction) has no
    # panel awareness, so the 5N panel DOF fight the global tilts every
    # iteration.  The C AutoCalibrateZarr refines panels as a separate locked
    # stage; we mirror that here, which is what reaches the sub-20 µε regime.
    if panel_layout is not None:
        from ..compat.from_v1 import (
            spec_from_v1_params, add_panel_parameters, add_panel_ring_radius,
        )
        # v1 has been mutated in-place to the phase-1 geometry; seed the
        # frozen spec from cr.unpacked so distortion (v2-named) carries over,
        # then freeze every global parameter.
        spec2 = spec_from_v1_params(v1)
        for name, prm in spec2.parameters.items():
            if name in cr.unpacked and cr.unpacked[name].numel() == 1:
                prm.init = float(cr.unpacked[name].detach().reshape(-1)[0])
            prm.refined = False
        if panel_mode == "radius":
            # `refine_panel_lsd` / `refine_panel_p2` are only wired into the
            # "shift" branch below. Silently ignoring them here is exactly the
            # class of no-op flag this codebase has been bitten by, so say so.
            if refine_panel_lsd or refine_panel_p2:
                warnings.warn(
                    "calibrate(): refine_panel_lsd / refine_panel_p2 have NO "
                    "effect with panel_mode='radius' — they are only used by "
                    "panel_mode='shift'. The per-(panel, ring) radial offset "
                    "that 'radius' fits already subsumes a per-panel Lsd, but "
                    "if you want the reference's PerPanelLsd / "
                    "PerPanelDistortion behaviour, pass panel_mode='shift'.",
                    UserWarning, stacklevel=2)
            # Per-(panel, ring) radial offset — nulls the radial calibrant
            # residual cell-by-cell.  The full ring table runs to high Q (200+
            # families); only the inner rings land on the detector, so probe
            # one E-step at the frozen geometry and size the parameter to just
            # the rings that carry fits (ring_idx indexes the ring table in
            # ascending-radius order, so max_idx+1 covers every populated ring
            # while keeping the LM Jacobian small).
            from ._common import run_estep_v1
            probe = run_estep_v1(v1, image, dark=dark, dtype=dtype, device=device)
            n_rings = int(probe.ring_idx.max().item()) + 1
            add_panel_ring_radius(spec2, panel_layout.n_panels(), n_rings,
                                  tol_px=panel_tol_radius_px)
            if verbose:
                print(f"[calibrate] STAGE 3: per-(panel, ring) radius refinement "
                      f"({panel_layout.n_panels()} panels × {n_rings} rings "
                      f"with fits, ±{panel_tol_radius_px}px)...", flush=True)
        elif panel_mode == "shift":
            add_panel_parameters(
                spec2, panel_layout.n_panels(),
                tol_shift_px=panel_tol_shift_px,
                tol_rot_deg=panel_tol_rot_deg,
                enable_lsd=refine_panel_lsd,
                enable_p2=refine_panel_p2,
            )
            if verbose:
                print(f"[calibrate] STAGE 3: per-panel shift refinement "
                      f"({panel_layout.n_panels()} panels, "
                      f"{panel_layout.n_panels_y}×{panel_layout.n_panels_z}, "
                      f"±{panel_tol_shift_px}px / ±{panel_tol_rot_deg}°)...", flush=True)
        else:
            raise ValueError(f"panel_mode must be 'radius' or 'shift'; got {panel_mode!r}")
        cr = autocalibrate(
            v1, image, dark=dark, mask=mask,
            spec=spec2, panel_layout=panel_layout,
            n_iter=n_iter, lm_max_iter=lm_max_iter,
            dtype=dtype, device=device, verbose=verbose,
            build_residual_corr=build_residual_corr,
            residual_corr_path=bin_path,
        )
    refine_time = time.time() - t1
    u = cr.unpacked

    # 3b. Per-parameter 1σ and the "is this actually measured?" audit.
    #
    # A point estimate says nothing about whether the data constrained it.
    # Refining the radial distortion on a narrow-azimuth frame returns
    # confident-looking coefficients whose σ is larger than the value — and
    # silently railed several of them.  Compute the Gauss-Newton covariance at
    # MAP and flag every refined parameter that is consistent with zero or is
    # sitting on a bound, so the caller can freeze it and re-run.
    sigma: Dict[str, float] = {}
    sigma_vector: Dict[str, np.ndarray] = {}
    unconstrained: List[str] = []
    undetermined: List[str] = []
    at_bounds: List[str] = []
    diag_results: List = []
    try:
        from ..inference.laplace import laplace_at_map
        from ..loss.pseudo_strain import pseudo_strain_residual
        fd = cr.fits_final
        if fd is not None:
            def _nll(unp):
                r = pseudo_strain_residual(
                    fd.Y_pix, fd.Z_pix, fd.ring_two_theta_deg, unp,
                    rho_d=fd.rho_d, weights=fd.weights,
                    panel_layout=panel_layout, panel_idx=fd.panel_idx,
                    ring_idx=fd.ring_idx)
                n_par = max(sum(1 for p in cr.spec.parameters.values() if p.refined), 1)
                dof = max(int(r.numel()) - n_par, 1)
                s2 = float((r * r).sum().detach()) / dof
                return 0.5 * (r * r).sum() / max(s2, 1e-30)
            lap = laplace_at_map(cr.spec, _nll, u)
            sig_all = lap.sigma_per_dim.detach().cpu().numpy()
            off = 0
            for nm, sz in zip(lap.refined_names, lap.refined_sizes):
                blk = sig_all[off:off + sz]
                if sz == 1:
                    sigma[nm] = float(blk[0])
                else:
                    # Vector-valued: record every element. These were dropped
                    # entirely before 2026-08-29 even though the Laplace
                    # computes them (240 dims on the 48-panel Pilatus).
                    sigma_vector[nm] = np.asarray(blk, dtype=float).copy()
                    val_v = np.asarray(
                        torch.as_tensor(u[nm]).detach().cpu().reshape(-1),
                        dtype=float)
                    for k in range(sz):
                        sg_k = float(blk[k])
                        if not np.isfinite(sg_k) or sg_k <= 0.0:
                            undetermined.append(f"{nm}[{k}]")
                        elif k < val_v.size and abs(val_v[k]) < sg_k:
                            unconstrained.append(f"{nm}[{k}]")
                off += sz
            # A gate that CAN fire: if most refined degrees of freedom are
            # not measured by the data, the model is over-parameterised and
            # the extra DOF are fitting noise. Measured on the real 48-panel
            # Pilatus with per-panel shift+Lsd+p2: of 240 refined DOF, 69 were
            # undetermined and 133 more consistent with zero — only 38 carried
            # information, and nothing said so.
            n_dof = sum(lap.refined_sizes) if lap.refined_names else 0
            n_dead = len(undetermined) + len(unconstrained)
            if n_dof >= 10 and n_dead > 0.5 * n_dof:
                warnings.warn(
                    f"calibrate(): {n_dead} of {n_dof} refined degrees of "
                    f"freedom are not measured by the data "
                    f"({len(undetermined)} undetermined, {len(unconstrained)} "
                    f"consistent with zero). The model is over-parameterised "
                    f"for this dataset — the surplus DOF are fitting noise. "
                    f"See `undetermined` / `unconstrained` / `sigma_vector`.",
                    UserWarning, stacklevel=2)
            if not sigma and not sigma_vector and lap.refined_names:
                warnings.warn(
                    "calibrate(): the Laplace produced no per-parameter sigma "
                    f"at all for ({', '.join(lap.refined_names)}). `sigma`, "
                    "`sigma_vector`, `unconstrained` and `at_bounds` are empty "
                    "because nothing was measured, not because everything is "
                    "fine.", UserWarning, stacklevel=2)
            for nm, sg_ in sigma.items():
                val = float(torch.as_tensor(u[nm]).reshape(-1)[0])
                if not np.isfinite(sg_) or sg_ <= 0.0:
                    # Not "perfectly determined" — see `undetermined`. The
                    # |value| < σ test below cannot fire for these, so without
                    # this they escaped every check silently.
                    undetermined.append(nm)
                elif abs(val) < sg_:
                    unconstrained.append(nm)
                bnds = cr.spec.parameters[nm].bounds
                if bnds:
                    span = abs(bnds[1] - bnds[0])
                    if span > 0 and min(abs(val - bnds[0]),
                                        abs(val - bnds[1])) < 1e-4 * span:
                        at_bounds.append(nm)
            from .diagnostics import run_all_gates
            diag_results = run_all_gates(
                v1_init=v1_seed, unpacked=u, history=cr.history, fits=fd,
                spec=cr.spec, panel_layout=panel_layout)
            from .diagnostics import seed_provenance_gate
            diag_results.insert(0, seed_provenance_gate(
                seed_method=seed_method, seed_note=seed_note,
                seed_BC_y=seed.bc_y, seed_BC_z=seed.bc_z,
                NrPixelsY=NY, NrPixelsZ=NZ))
    except Exception as e:                    # diagnostics must never fail a run
        # ...but it must not fail SILENTLY either. This used to report only
        # under `verbose`, so with the default verbose=False a failure here
        # returned a result whose `sigma`, `unconstrained`, `at_bounds` and
        # `diagnostics` were all empty, with nothing printed — and an empty
        # gate list used to read back as "ok" (see
        # diagnostics.worst_severity). Warn unconditionally.
        warnings.warn(
            f"calibrate(): uncertainty/diagnostics could not be computed "
            f"({type(e).__name__}: {e}). The returned result has NO sigma, NO "
            f"unconstrained/at_bounds list and NO gate results — do not read "
            f"their emptiness as a clean bill of health.",
            UserWarning, stacklevel=2)
        if verbose:
            print(f"[calibrate]   (uncertainty/diagnostics skipped: {e})",
                  flush=True)

    if verbose:
        if unconstrained:
            print(f"[calibrate] ⚠ refined but NOT determined by the data "
                  f"(|value| < 1σ): {', '.join(unconstrained)} — freeze these "
                  f"and re-run", flush=True)
        if at_bounds:
            print(f"[calibrate] ⚠ refined and sitting ON a bound: "
                  f"{', '.join(at_bounds)} — the bound, not the data, is "
                  f"setting these", flush=True)
        for d in diag_results:
            icon = {"ok": "✓", "warn": "⚠", "fail": "✗"}.get(d.severity, "?")
            print(f"[calibrate] {icon} [{d.name}] {d.message}", flush=True)
        if cr.fits_final is not None and cr.fits_final.phase_idx is not None \
                and len(cr.fits_final.phase_names) > 1:
            from ..loss.diagnostics import per_phase_summary
            from ..loss.pseudo_strain import pseudo_strain_residual as _psr
            with torch.no_grad():
                _r = _psr(cr.fits_final.Y_pix, cr.fits_final.Z_pix,
                          cr.fits_final.ring_two_theta_deg, u,
                          rho_d=cr.fits_final.rho_d, weights=None,
                          panel_layout=panel_layout,
                          panel_idx=cr.fits_final.panel_idx).abs() * 1e6
            print(per_phase_summary(_r, cr.fits_final.phase_idx,
                                    cr.fits_final.phase_names), flush=True)

    # 4. Persist summary JSON.
    if output_dir is not None:
        import json
        summary = {
            "calibrant": cal_name,
            "calibrants": cal_names,
            "sigma": sigma,
            "unconstrained": unconstrained,
            "at_bounds": at_bounds,
            "diagnostics": [{"name": d.name, "severity": d.severity,
                              "message": d.message} for d in diag_results],
            "wavelength_A": wavelength,
            "pxY_um": pxY, "pxZ_um": pxZ,
            "NrPixelsY": NY, "NrPixelsZ": NZ,
            "Lsd_um": float(u["Lsd"]),
            "BC_y_px": float(u["BC_y"]), "BC_z_px": float(u["BC_z"]),
            "tx_deg": float(u.get("tx", 0.0)),
            "ty_deg": float(u["ty"]), "tz_deg": float(u["tz"]),
            "distortion": {n: float(u[n]) for n in
                            ("iso_R2","iso_R4","iso_R6",
                             "a1","phi1","a2","phi2","a3","phi3",
                             "a4","phi4","a5","phi5","a6","phi6")
                            if n in u},
            # The adopted geometry is the BEST iterate, not the last (see
            # pipelines/single.py), so report the best iterate's strain here —
            # history[-1] would describe a geometry that was not returned.
            "in_loop_strain_uE": (min(h.mean_strain_uE for h in cr.history)
                                   if cr.history else None),
            "post_residual_strain_uE": cr.post_residual_strain_uE,
            "residual_corr_bin": bin_path,
            "seed_BC_y": seed.bc_y, "seed_BC_z": seed.bc_z,
            "seed_Lsd_um": seed.Lsd,
            "seed_seconds": seed_time,
            "refine_seconds": refine_time,
        }
        (Path(output_dir) / "calibration.json").write_text(json.dumps(summary, indent=2))

    return AutoCalibrationResult(
        Lsd=float(u["Lsd"]),
        BC_y=float(u["BC_y"]), BC_z=float(u["BC_z"]),
        tx=float(u.get("tx", 0.0)),
        ty=float(u["ty"]), tz=float(u["tz"]),
        distortion={n: float(u[n]) for n in
                     ("iso_R2","iso_R4","iso_R6",
                      "a1","phi1","a2","phi2","a3","phi3",
                      "a4","phi4","a5","phi5","a6","phi6") if n in u},
        pxY=pxY, pxZ=pxZ, NrPixelsY=NY, NrPixelsZ=NZ,
        wavelength_A=wavelength,
        post_residual_strain_uE=cr.post_residual_strain_uE,
        post_residual_strain_median_uE=(
            cr.history[-1].median_strain_uE if cr.history else None),
        post_residual_strain_trim_uE=(
            cr.history[-1].trim_strain_uE if cr.history else None),
        in_loop_strain_uE=(min(h.mean_strain_uE for h in cr.history)
                            if cr.history else None),
        residual_corr_map=cr.residual_corr_map,
        residual_corr_bin_path=bin_path,
        sigma=sigma, sigma_vector=sigma_vector,
        unconstrained=unconstrained, undetermined=undetermined,
        at_bounds=at_bounds,
        calibrants=cal_names, diagnostics=diag_results,
        seed_seconds=seed_time, refine_seconds=refine_time,
        seed_BC_y=seed.bc_y, seed_BC_z=seed.bc_z, seed_Lsd=seed.Lsd,
        seed_method=seed_method, seed_note=seed_note,
        iter_history=[{"iter": h.iteration, "strain_uE": h.mean_strain_uE,
                       "median_strain_uE": h.median_strain_uE,
                       "trim_strain_uE": h.trim_strain_uE,
                       "n_fitted": h.n_fitted,
                       "Lsd": h.Lsd, "BC_y": h.BC_y, "BC_z": h.BC_z,
                       "ty": h.ty, "tz": h.tz} for h in cr.history],
    )


__all__ = ["calibrate", "AutoCalibrationResult", "CALIBRANTS",
           "SeedFallbackWarning"]
