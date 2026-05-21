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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List

import numpy as np
import torch


# ============================================================ calibrant DB

# Canonical powder-calibrant lattice constants + space groups.  Add new
# entries here; everything else flows through.
CALIBRANTS: Dict[str, Dict] = {
    "CeO2":  {"a": 5.4116, "alpha": 90.0, "sg": 225},
    "LaB6":  {"a": 4.1569, "alpha": 90.0, "sg": 221},
    "Si":    {"a": 5.4310, "alpha": 90.0, "sg": 227},
    "Al2O3": {"a": 4.7589, "c": 12.9920, "alpha": 90.0, "gamma": 120.0, "sg": 167},
}


def _generate_sim_radii_px(*, lattice_a: float, lattice_c: float,
                            alpha: float, gamma: float,
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

    # Build the lattice with the canonical a-b-c, alpha-beta-gamma layout.
    # For cubic systems alpha=beta=gamma=90 and a=b=c.
    # For trigonal/hexagonal alpha=beta=90, gamma=120 and a=b!=c.
    lat = Lattice(
        a=lattice_a, b=lattice_a, c=lattice_c,
        alpha=alpha, beta=alpha, gamma=gamma,
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


# ============================================================ result type

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
    residual_corr_map: Optional[torch.Tensor] = None     # [NrPixelsZ, NrPixelsY] px
    residual_corr_bin_path: Optional[str] = None
    seed_seconds: float = 0.0
    refine_seconds: float = 0.0
    # Raw seed + LM bits (for plotting / further analysis)
    seed_BC_y: float = 0.0
    seed_BC_z: float = 0.0
    seed_Lsd: float = 0.0
    iter_history: List[Dict] = field(default_factory=list)


# ============================================================ entry point

def calibrate(
    image: np.ndarray,
    *,
    wavelength: float,
    pxY: float,
    pxZ: Optional[float] = None,
    dark: Optional[np.ndarray] = None,
    im_trans: tuple = (),
    calibrant: Union[str, Dict] = "CeO2",
    output_dir: Optional[Union[str, Path]] = None,
    initial_Lsd: float = 1_000_000.0,
    lsd_window: Optional[float] = None,
    max_2theta_deg: float = 28.0,
    min_ring_radius_px: float = 120.0,
    max_ring_radius_px: Optional[float] = None,
    n_iter: int = 4,
    lm_max_iter: int = 200,
    build_residual_corr: bool = True,
    refine_tilts: bool = True,
    refine_distortion: bool = True,
    panel_layout=None,
    panel_mode: str = "radius",
    panel_tol_shift_px: float = 3.0,
    panel_tol_rot_deg: float = 1.0,
    panel_tol_radius_px: float = 2.0,
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
    * ``pxY`` — pixel pitch in µm (square pixels assumed unless ``pxZ`` is
      also given).
    * ``dark`` — optional dark-frame array; subtracted from ``image``.
    * ``calibrant`` — name (``"CeO2"``, ``"LaB6"``, ``"Si"``, ``"Al2O3"``)
      or a dict with ``a``, optionally ``c``, ``alpha``, ``gamma``, ``sg``.
    * ``panel_layout`` — optional :class:`PanelLayout` for tiled-module
      detectors (Pilatus, Eiger).  Build it once with
      ``PanelLayout.regular(n_y, n_z, sy, sz, gap_y, gap_z)`` and pass it in;
      per-panel rigid-body shifts (δy, δz, δθ, and optionally δLsd / δp₂) are
      then refined in the M-step.  ``panel_tol_shift_px`` /
      ``panel_tol_rot_deg`` bound the shift and rotation; this is what brings
      a multi-module Pilatus from a few-hundred µε monolithic fit down to the
      sub-20 µε regime.  Leave ``None`` for monolithic detectors (GE, Varex).

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
    NZ, NY = image.shape
    if pxZ is None:
        pxZ = pxY

    # Resolve calibrant.
    if isinstance(calibrant, str):
        if calibrant not in CALIBRANTS:
            raise ValueError(
                f"Unknown calibrant {calibrant!r}; known: {sorted(CALIBRANTS)}"
            )
        cal = dict(CALIBRANTS[calibrant])
    else:
        cal = dict(calibrant)
    a = float(cal["a"])
    c = float(cal.get("c", a))
    alpha = float(cal.get("alpha", 90.0))
    gamma = float(cal.get("gamma", 90.0))
    sg = int(cal["sg"])
    cal_name = calibrant if isinstance(calibrant, str) else "<custom>"

    # 1. Background.  Detect bad-pixel / detector-gap sentinels (-1, -2) from
    # the RAW image *before* clipping, so they can be masked out of seeding
    # (otherwise module gaps fragment the ring arcs and bias the beam centre).
    sentinel_mask = (image == -1) | (image == -2)
    sentinel_mask = sentinel_mask if bool(sentinel_mask.any()) else None
    if dark is not None:
        if dark.shape != image.shape:
            raise ValueError(
                f"dark shape {dark.shape} != image shape {image.shape}"
            )
        img = np.clip(image.astype(np.float32) - dark.astype(np.float32), 0, None)
    else:
        img = np.clip(image.astype(np.float32), 0, None)

    # 2. Seed BC + Lsd.
    if verbose:
        print(f"[calibrate] STAGE 1: seeding from {cal_name} rings...", flush=True)
    sim_radii = _generate_sim_radii_px(
        lattice_a=a, lattice_c=c, alpha=alpha, gamma=gamma,
        wavelength=wavelength, px=pxY, sg=sg,
        Lsd_nominal_um=initial_Lsd, max_2theta_deg=max_2theta_deg,
    )
    if sim_radii.size < 3:
        raise RuntimeError(
            f"Only {sim_radii.size} simulated rings under "
            f"{max_2theta_deg}° — check wavelength/lattice"
        )
    t0 = time.time()
    # Robust seed first: connected-component arc detection → chord-bisector
    # beam centre (recovers OFF-PANEL BC) → multi-hypothesis Lsd. This is the
    # battle-tested recipe (formerly AutoCalibrateZarr); it handles off-centre
    # BC, masked detectors, and weak data where the chord-only arc seed
    # (seed_from_image) gets stuck. Fall back to seed_from_image otherwise.
    from types import SimpleNamespace
    seed = None
    try:
        from ..seed.from_calibrant_image import auto_seed_calibrant
        asr = auto_seed_calibrant(img, sim_radii_px=sim_radii,
                                  initial_lsd_um=initial_Lsd, mask=sentinel_mask,
                                  lsd_window=lsd_window)
        if asr.Lsd_um and asr.Lsd_um > 0 and asr.n_rings_matched >= 1:
            seed = SimpleNamespace(bc_y=asr.BC_y, bc_z=asr.BC_z, Lsd=asr.Lsd_um,
                                   n_arcs=asr.n_arcs, n_rings=asr.n_rings_matched)
            if verbose:
                print(f"[calibrate]   connected-component seed "
                      f"({asr.n_arcs} arcs, {asr.n_rings_matched} rings matched, "
                      f"thr={asr.threshold_used:.0f})", flush=True)
    except Exception as e:  # pragma: no cover - fall back on any seed failure
        if verbose:
            print(f"[calibrate]   auto_seed_calibrant failed ({e}); using arc seed",
                  flush=True)
    if seed is None:
        seed = seed_from_image(
            image=img, sim_radii_px=sim_radii,
            initial_lsd=initial_Lsd, npy=NY, npz=NZ,
            skip_median=False, min_ring_radius_px=min_ring_radius_px,
        )
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
        LatticeConstant=(a, a, c, alpha, alpha, gamma),
        MaxRingRad=float(max_ring_radius_px),
        MinRingRad=float(min_ring_radius_px),
        RhoD=RhoD_px,
        nIterations=n_iter,
        Refine={"Lsd": True, "BC": True,
                "ty": bool(refine_tilts), "tz": bool(refine_tilts),
                "Wavelength": False, "Parallax": False,
                **{f"p{i}": refine_distortion for i in range(15)}},
        Device=device, Dtype="fp64" if dtype == torch.float64 else "fp32",
    )

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
    cr = autocalibrate(
        v1, image, dark=dark,
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
            v1, image, dark=dark,
            spec=spec2, panel_layout=panel_layout,
            n_iter=n_iter, lm_max_iter=lm_max_iter,
            dtype=dtype, device=device, verbose=verbose,
            build_residual_corr=build_residual_corr,
            residual_corr_path=bin_path,
        )
    refine_time = time.time() - t1
    u = cr.unpacked

    # 4. Persist summary JSON.
    if output_dir is not None:
        import json
        summary = {
            "calibrant": cal_name,
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
            "in_loop_strain_uE": cr.history[-1].mean_strain_uE if cr.history else None,
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
        in_loop_strain_uE=cr.history[-1].mean_strain_uE if cr.history else None,
        residual_corr_map=cr.residual_corr_map,
        residual_corr_bin_path=bin_path,
        seed_seconds=seed_time, refine_seconds=refine_time,
        seed_BC_y=seed.bc_y, seed_BC_z=seed.bc_z, seed_Lsd=seed.Lsd,
        iter_history=[{"iter": h.iteration, "strain_uE": h.mean_strain_uE,
                       "Lsd": h.Lsd, "BC_y": h.BC_y, "BC_z": h.BC_z,
                       "ty": h.ty, "tz": h.tz} for h in cr.history],
    )


__all__ = ["calibrate", "AutoCalibrationResult", "CALIBRANTS"]
