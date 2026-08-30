"""Multi-image / multi-distance joint calibration.

Each image has per-image parameters (Lsd, BC, tilts) plus a shared block
(distortion harmonics, panels, pxY, pxZ).  Joint loss = Σ pseudo-strain.

Wright2022's grid panel calibration falls out as a special case (multiple
beam positions, shared panel shifts).  This is the milestone that
operationally unlocks pxY / pxZ / d-spacing fitting because the
rank-deficiency from a single image disappears when multiple geometries
share the intrinsic parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_peakfit import GenericLMConfig
from midas_peakfit.reparam import x_to_u, u_to_x
from midas_peakfit import lm_solve_generic

from ..compat.from_v1 import spec_from_v1_params
from ..forward.panels import PanelLayout
from ..loss.pseudo_strain import pseudo_strain_residual
from ..parameters.parameter import Parameter
from ..parameters.pack import (
    pack_multi, refined_indices, refined_bounds,
    unpack_spec, MultiPackInfo, pack_spec,
)
from ..parameters.spec import CalibrationSpec, MultiImageSpec
from ._common import FittedDataset, run_estep_v1


#: Parameters that describe the DETECTOR rather than the exposure.  Sharing
#: exactly these is the "same detector, different sample" model — see
#: :func:`build_multi_spec` under ``mode="same_detector"``.
_DETECTOR_NAMES = ["pxY", "pxZ", "RhoD", "tx", "ty", "tz"]
_PANEL_NAMES = ["panel_delta_yz", "panel_delta_theta",
                "panel_delta_lsd", "panel_delta_p2"]

MULTI_MODES = ("independent", "same_detector")


def build_multi_spec(
    v1_per_image: List[V1Params],
    *,
    shared_names: Optional[List[str]] = None,
    link_lsd: bool = False,
    mode: str = "independent",
) -> MultiImageSpec:
    """Construct a MultiImageSpec from per-image v1 params.

    ``mode="independent"`` (default, unchanged): shared block is pxY, pxZ,
    RhoD, all v2 distortion params, panel_*.  Per-image: Lsd, BC_y, BC_z, ty,
    tz, Wavelength, Parallax, tx.  Appropriate when the images really are
    different setups.

    ``mode="same_detector"``: additionally shares the **tilts** (tx, ty, tz).
    Use this whenever the images come from ONE detector that did not move --
    several phases on a single exposure, or repeated exposures of the same
    setup.  What stays per-image is then exactly ``Lsd``, ``BC_y``, ``BC_z``,
    i.e. that exposure's SAMPLE POSITION: a sample displaced along the beam
    changes its Lsd, and one displaced transversely moves the apparent beam
    centre by ``d/px`` pixels.  Passing the same frame twice with one
    calibrant each therefore fits the two powders' relative position.

    Leaving the tilts per-image when the detector did not move is not merely
    wasteful, it is wrong, and it biases what is left: on a real CeO2+LaB6
    1-ID frame, independently-refined tilts absorbed the difference between
    the two calibrants and reported a spurious 1.43 mm relative sample offset
    where sharing the tilts gives 72 +/- 34 um.

    ``link_lsd=True`` additionally shares ``Lsd`` and ``Wavelength``.  Use it
    with ``lsd_offsets_um`` on :func:`autocalibrate_multi`, which turns the
    shared ``Lsd`` into a single refined offset ``L0`` and reconstructs each
    image's distance as ``Lsd_i = L0 + Delta_i``.  See that function for why
    this — and not merely using several distances — is what makes the
    wavelength identifiable.
    """
    if mode not in MULTI_MODES:
        raise ValueError(f"mode must be one of {MULTI_MODES}; got {mode!r}")
    if shared_names is None:
        from ..forward.distortion import P_COEF_NAMES
        shared_names = (["pxY", "pxZ", "RhoD"]
                        + list(P_COEF_NAMES)
                        + list(_PANEL_NAMES))
        if mode == "same_detector":
            shared_names += ["tx", "ty", "tz"]
    elif mode == "same_detector":
        shared_names = list(shared_names) + [
            n for n in ("tx", "ty", "tz") if n not in shared_names]
    if link_lsd:
        shared_names = list(shared_names) + ["Lsd", "Wavelength"]

    specs = [spec_from_v1_params(p) for p in v1_per_image]
    shared_names = [n for n in shared_names if n in specs[0].parameters]
    return MultiImageSpec.from_calibration_specs(specs, shared_names)


@dataclass
class MultiResult:
    multi_spec: MultiImageSpec
    shared_unpacked: Dict[str, torch.Tensor]
    per_image_unpacked: List[Dict[str, torch.Tensor]]
    cost: float
    rc: int
    # Empirical residual-correction map built post-MAP (port of v1 C
    # dg_residual_corr_lookup).  None when ``build_residual_corr=False``.
    # Shape: ``[NrPixelsZ, NrPixelsY]`` float64, ΔR in pixels.
    residual_corr_map: Optional[torch.Tensor] = None
    # Mean weighted strain (µε) re-evaluated at MAP after the residual
    # map is applied — the honest "post-correction" number.  None when
    # ``build_residual_corr=False``.
    post_residual_strain_uE: Optional[List[float]] = None
    # Known-distance-travel mode only (``lsd_offsets_um``): the single refined
    # offset L0 (µm) and the reconstructed per-image Lsd_i = L0 + Delta_i.
    # ``shared_unpacked["Lsd"]`` holds L0, NOT a usable per-image distance, so
    # read the distances from here.  Deltas are re-centred to zero mean, so L0
    # is the distance at the centroid of the scan, not at any one image.
    L0_um: Optional[float] = None
    lsd_per_image_um: Optional[List[float]] = None


def _build_multi_indices(multi_spec: MultiImageSpec, info: MultiPackInfo,
                          dtype, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (refined_idx, lo, hi) over the multi-image flat tensor.

    ``refined_indices`` returns CPU tensors by default; move the concatenated
    result onto ``device`` so it lines up with ``x_full`` (which ``pack_multi``
    already places on ``device``). Without this, ``index_select`` raises a
    cross-device error the moment the caller is on a GPU.
    """
    s_spec = CalibrationSpec(parameters=multi_spec.shared)
    lo_s, hi_s = refined_bounds(s_spec, info.shared_info, dtype=dtype, device=device)
    lo_pieces = [lo_s]; hi_pieces = [hi_s]
    ref_idx_pieces = [refined_indices(info.shared_info)]
    for img_dict, img_info in zip(multi_spec.per_image, info.per_image_info):
        i_spec = CalibrationSpec(parameters=img_dict)
        lo_i, hi_i = refined_bounds(i_spec, img_info, dtype=dtype, device=device)
        lo_pieces.append(lo_i); hi_pieces.append(hi_i)
        ref_idx_pieces.append(refined_indices(img_info))
    refined_idx = torch.cat(ref_idx_pieces).to(device=device)
    return refined_idx, torch.cat(lo_pieces), torch.cat(hi_pieces)


def autocalibrate_multi(
    v1_per_image: List[V1Params],
    images: List[np.ndarray],
    darks: Optional[List[Optional[np.ndarray]]] = None,
    masks: Optional[List[Optional[np.ndarray]]] = None,
    *,
    multi_spec: Optional[MultiImageSpec] = None,
    panel_layout: Optional[PanelLayout] = None,
    n_iter: int = 5,
    lm_max_iter: int = 200,
    dtype=torch.float64, device: str = "cpu",
    verbose: bool = True,
    build_residual_corr: bool = True,
    residual_corr_outlier_pct: float = 90.0,
    residual_corr_path: Optional[str] = None,
    lsd_offsets_um: Optional[List[float]] = None,
) -> MultiResult:
    """Joint calibration over multiple images.

    Known-distance-travel mode (``lsd_offsets_um``)
    -----------------------------------------------
    Pass the *exactly known* relative travel of the detector stage, one value
    per image (µm, any common origin).  Each image's distance is then
    reconstructed as ``Lsd_i = L0 + Delta_i`` from a SINGLE shared refined
    ``L0`` (the unknown offset between the stage readback and the true
    sample-detector distance), instead of one free ``Lsd`` per image.

    This is what makes ``Wavelength`` identifiable, and it is worth being
    precise about why, because "just use several distances" does NOT do it.
    With one free ``Lsd`` per image the transformation

        lambda -> k*lambda,   Lsd_i -> k*Lsd_i  (all i)

    leaves every predicted ring radius unchanged to first order (2theta is
    small, R = Lsd*tan(2theta), 2theta ~ lambda/d), so lambda and the
    distances stay degenerate no matter how many images are stacked up --
    only the weak tan/arcsin non-linearity separates them.  Fixing the
    DIFFERENCES ``Delta_i`` forbids that rescaling: k*Lsd_i is no longer of
    the form L0' + Delta_i unless k == 1.  On a 28-image CeO2 scan spanning
    330-3000 mm this stiffens the soft (lambda, Lsd) Fisher direction by ~3
    orders of magnitude (condition number 3e13 -> 2e8).

    Requires ``link_lsd=True`` when building the spec (or a ``multi_spec``
    that already carries a shared ``Lsd``), so that the shared ``Lsd``
    parameter can play the role of ``L0``.

    After LM/EM convergence, optionally builds an empirical residual
    correction map from per-fit ``ΔR = R_forward - R_ideal`` and stores it on
    the result.  The map absorbs systematic deviations the harmonic
    distortion polynomial cannot capture (port of v1 C
    ``dg_residual_corr_lookup``) and brings v2 to v1-AutoCal accuracy.  When
    ``residual_corr_path`` is set, the map is also persisted as a
    v1-compatible binary readable by :mod:`midas_integrate` and
    ``CalibrantIntegratorOMP``.
    """
    # ---- Resolve RhoD to µm on every input image (RhoD enters only as
    # ρ = R_um / RhoD). Auto-detect units of the supplied value and default
    # to the BC-to-farthest-edge distance for the automated case.
    from ..forward.sanity import resolve_rho_d_um
    for i, v1 in enumerate(v1_per_image):
        rho_d_um, _ = resolve_rho_d_um(
            v1.RhoD if v1.RhoD > 0 else v1.MaxRingRad,
            NrPixelsY=int(v1.NrPixelsY), NrPixelsZ=int(v1.NrPixelsZ),
            BC_y=float(v1.BC_y), BC_z=float(v1.BC_z),
            pxY=float(v1.pxY), pxZ=float(v1.pxZ if v1.pxZ > 0 else v1.pxY),
        )
        v1.RhoD = rho_d_um   # canonical µm for E-step + forward distortion
    n_imgs = len(v1_per_image)
    if len(images) != n_imgs:
        raise ValueError("len(images) must match len(v1_per_image)")
    if darks is None:
        darks = [None] * n_imgs
    if masks is None:
        masks = [None] * n_imgs
    if len(masks) != n_imgs:
        raise ValueError(
            f"masks has {len(masks)} entries for {n_imgs} images")

    # ---- Known-distance-travel mode.  Delta_i are constants, not parameters:
    # they are re-centred to a zero-mean so the shared "Lsd" is L0 at the
    # centroid of the scan, which decorrelates L0 from lambda far better than
    # anchoring at an arbitrary end point.
    link_lsd = lsd_offsets_um is not None
    deltas_t: List[torch.Tensor] = []
    if link_lsd:
        if len(lsd_offsets_um) != n_imgs:
            raise ValueError(
                f"lsd_offsets_um has {len(lsd_offsets_um)} entries but there "
                f"are {n_imgs} images")
        d_arr = np.asarray(lsd_offsets_um, dtype=float)
        d_arr = d_arr - d_arr.mean()
        deltas_t = [torch.as_tensor(float(d), dtype=dtype, device=device)
                    for d in d_arr]
        # Seed the shared L0 at the mean of the per-image distances so the
        # first LM step starts on the constraint surface rather than off it.
        L0_seed = float(np.mean([v1.Lsd for v1 in v1_per_image]))
        for v1, d in zip(v1_per_image, d_arr):
            v1.Lsd = L0_seed + float(d)

    if multi_spec is None:
        multi_spec = build_multi_spec(v1_per_image, link_lsd=link_lsd)

    if link_lsd:
        if "Lsd" not in multi_spec.shared:
            raise ValueError(
                "lsd_offsets_um requires a shared 'Lsd' in the multi_spec "
                "(build it with build_multi_spec(..., link_lsd=True))")
        p_lsd = multi_spec.shared["Lsd"]
        # The shared block is taken from image 0, so this Parameter's bounds
        # are centred on image 0's DISTANCE, whereas it now has to represent
        # L0 at the centroid of the scan.  Re-centre the box on L0_seed as
        # well as the init: leaving the old bounds puts the start point far
        # outside them, the Logit transform saturates, and LM sits on a bound
        # burning its whole iteration budget without moving.
        half = 0.5 * abs(p_lsd.bounds[1] - p_lsd.bounds[0]) if p_lsd.bounds \
            else max(1.0e4, 0.05 * abs(L0_seed))
        p_lsd.init = L0_seed
        p_lsd.refined = True
        p_lsd.bounds = (L0_seed - half, L0_seed + half)
        p_lsd.transform = None      # rebuilt as Logit over the new box
        if verbose:
            print(f"[multi] linked Lsd: L0 seed {L0_seed/1e3:.4f} mm, "
                  f"box +/-{half/1e3:.1f} mm, Delta range "
                  f"{d_arr.min()/1e3:+.1f}..{d_arr.max()/1e3:+.1f} mm",
                  flush=True)

    cost_final = float("inf")
    rc_final = 1
    shared_dict_final: Dict[str, torch.Tensor] = {}
    per_dicts_final: List[Dict[str, torch.Tensor]] = []

    for it in range(n_iter):
        # E-step per image at current geometry.
        fits_per_image: List[FittedDataset] = []
        for i_img, (v1, img, drk, msk) in enumerate(zip(v1_per_image, images, darks, masks)):
            fd = run_estep_v1(v1, img, dark=drk, mask=msk, dtype=dtype, device=device)
            if fd.ring_d_spacing_A is None and fd.rt is not None:
                # Refining Wavelength needs the autograd chain to run through
                # lambda, which pseudo_strain_residual only does when per-point
                # d-spacings are supplied (otherwise it uses the pre-computed
                # constant 2theta and d(residual)/d(lambda) is exactly 0).
                d_ring = torch.as_tensor(fd.rt.d_spacing, dtype=dtype,
                                         device=device)
                fd.ring_d_spacing_A = d_ring[fd.ring_idx]
            # A refined Wavelength with no d-spacings is a SILENT no-op: the
            # parameter is packed, gets an all-zero Jacobian column, and comes
            # back at its initial value with no warning.  Refuse instead.
            if fd.ring_d_spacing_A is None:
                wl_shared = multi_spec.shared.get("Wavelength")
                wl_local = multi_spec.per_image[i_img].get("Wavelength")
                if (wl_shared is not None and wl_shared.refined) or \
                        (wl_local is not None and wl_local.refined):
                    raise RuntimeError(
                        "Wavelength is marked refined but the E-step produced no "
                        "per-fit d-spacings, so d(residual)/d(lambda) is "
                        "identically zero and the refinement would silently do "
                        "nothing. Ensure the fitted dataset carries "
                        "ring_d_spacing_A (run_estep_v1 populates it).")
            fits_per_image.append(fd)

        x_full, info = pack_multi(multi_spec, dtype=dtype, device=device)
        refined_idx, lo, hi = _build_multi_indices(multi_spec, info, dtype, device)
        x_ref = x_full.index_select(0, refined_idx)

        s_spec = CalibrationSpec(parameters=multi_spec.shared)
        per_specs = [CalibrationSpec(parameters=d) for d in multi_spec.per_image]

        # Σ panel = 0 zero-sum constraint (Wright 2022 §3.2): if the
        # multi_spec carries the flag (set via add_panel_zero_sum_constraint
        # on the shared block), append the constraint residual exactly once
        # to the joint residual (panels are shared, so one set of zero-sum
        # rows applies to all images).
        _zs_active = bool(getattr(multi_spec, "zero_sum_panels", False))
        _zs_lambda = float(getattr(multi_spec, "zero_sum_lambda", 1e6))
        if _zs_active:
            from ..loss.constraints import zero_sum_residual

        def residual_fn(u, lo_, hi_):
            x_ref_now = u_to_x(u, lo_, hi_).squeeze(0)
            x_full_now = x_full.clone()
            x_full_now[refined_idx] = x_ref_now
            shared_dict = unpack_spec(x_full_now, info.shared_info, s_spec)
            per_dicts = [unpack_spec(x_full_now, img_info, ps)
                          for img_info, ps in zip(info.per_image_info, per_specs)]
            r_pieces: List[torch.Tensor] = []
            for i_img, (fits, per_d) in enumerate(zip(fits_per_image, per_dicts)):
                merged = {**shared_dict, **per_d}
                if link_lsd:
                    # Lsd_i = L0 + Delta_i.  Built here (not packed as a
                    # parameter) so the autograd chain runs through the single
                    # shared L0 and the Delta_i stay exact constants.
                    merged["Lsd"] = shared_dict["Lsd"] + deltas_t[i_img]
                r = pseudo_strain_residual(
                    fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, merged,
                    rho_d=fits.rho_d, weights=fits.weights,
                    panel_layout=panel_layout, panel_idx=fits.panel_idx,
                    ring_idx=fits.ring_idx,
                    ring_d_spacing_A=fits.ring_d_spacing_A,
                )
                r_pieces.append(r)
            if _zs_active:
                zs = zero_sum_residual(shared_dict, lambda_zs=_zs_lambda)
                if zs.numel() > 0:
                    r_pieces.append(zs)
            return torch.cat(r_pieces).unsqueeze(0)

        x_final, cost, rc = lm_solve_generic(
            x_ref.unsqueeze(0), lo.unsqueeze(0), hi.unsqueeze(0),
            residual_fn=residual_fn,
            config=GenericLMConfig(max_iter=lm_max_iter, ftol_rel=1e-9, xtol_rel=1e-9),
        )

        with torch.no_grad():
            x_ref_final = x_final.squeeze(0)
            x_full_final = x_full.clone()
            x_full_final[refined_idx] = x_ref_final
            shared_dict_final = unpack_spec(x_full_final, info.shared_info, s_spec)
            per_dicts_final = [unpack_spec(x_full_final, img_info, ps)
                                for img_info, ps in zip(info.per_image_info, per_specs)]

        # Push back into spec inits and v1 params for next E-step.
        for name, val in shared_dict_final.items():
            init = val.detach().cpu() if val.numel() > 1 else float(val.detach())
            multi_spec.shared[name].init = init
            for i_img, v1 in enumerate(v1_per_image):
                if hasattr(v1, name) and val.numel() == 1:
                    try:
                        cur = getattr(v1, name)
                        v = float(val.detach())
                        if link_lsd and name == "Lsd":
                            # shared Lsd is L0; the image's actual distance for
                            # the next E-step is L0 + Delta_i.  Writing L0 to
                            # every image would collapse the whole scan onto
                            # one distance and the next E-step would look for
                            # rings at the wrong radii.
                            v = v + float(deltas_t[i_img])
                        setattr(v1, name, type(cur)(v))
                    except Exception:
                        pass
        for img_idx, per_d in enumerate(per_dicts_final):
            for name, val in per_d.items():
                multi_spec.per_image[img_idx][name].init = (
                    val.detach().cpu() if val.numel() > 1 else float(val.detach())
                )
                v1 = v1_per_image[img_idx]
                if hasattr(v1, name) and val.numel() == 1:
                    try:
                        cur = getattr(v1, name)
                        setattr(v1, name, type(cur)(float(val.detach())))
                    except Exception:
                        pass

        cost_final = float(cost.item())
        rc_final = int(rc.item())
        if verbose:
            # Honest strain at MAP: rebuild E-step at post-LM geometry. The
            # in-loop residual_fn still holds pre-LM fits_per_image, so
            # evaluating it at x_final reports the drift between pre-LM peak
            # positions and post-LM forward prediction — not fit quality.
            # v1_per_image has already been pushed to x_final above.
            with torch.no_grad():
                fits_at_map = [run_estep_v1(v1, img, dark=drk, mask=msk,
                                             dtype=dtype, device=device)
                                for v1, img, drk, msk in zip(v1_per_image, images, darks, masks)]
                r_pieces_map = []
                for fits, per_d in zip(fits_at_map, per_dicts_final):
                    merged = {**shared_dict_final, **per_d}
                    r = pseudo_strain_residual(
                        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, merged,
                        rho_d=fits.rho_d, weights=fits.weights,
                        panel_layout=panel_layout, panel_idx=fits.panel_idx,
                    )
                    r_pieces_map.append(r)
                mean_uE = float(torch.cat(r_pieces_map).abs().mean()) * 1e6
            print(f"[multi iter {it}] cost={cost_final:.6e}  rc={rc_final}  "
                  f"strain={mean_uE:8.1f}μϵ across {n_imgs} images")

    # ---- Post-MAP empirical residual-correction map (v1 parity stage) -----
    # After LM/EM converges, fit a smooth ΔR(Y, Z) spline to the per-fit
    # residuals (R_forward - R_ideal) and store it as a detector-resolution
    # grid.  Subsequent forward calls add ΔR via differentiable bilinear
    # lookup; this is the v2 port of v1 C ``dg_residual_corr_lookup`` and
    # closes the ~50–100 µε per-pixel gap between v2 and AutoCal/v1.
    residual_map: Optional[torch.Tensor] = None
    post_strain: Optional[List[float]] = None
    if build_residual_corr:
        from ..forward.residual_corr import (
            build_residual_corr_map, save_residual_corr_bin,
        )
        from ..forward.bragg import R_ideal_px

        # Aggregate non-outlier (Y, Z, ΔR_µm) across all images so the map
        # is shared (matches the multi-distance shared-detector setup).
        Y_all, Z_all, dR_all = [], [], []
        post_strain = []
        # Re-use fits_per_image and post-LM unpacked dicts from the LAST
        # EM iteration (still in scope from the for-loop).
        for fits, per_d in zip(fits_per_image, per_dicts_final):
            merged = {**shared_dict_final, **per_d}
            with torch.no_grad():
                r_un = pseudo_strain_residual(
                    fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, merged,
                    rho_d=fits.rho_d, weights=None,
                    panel_layout=panel_layout, panel_idx=fits.panel_idx,
                )
                # Default px for radial-µm conversion (mean of pxY/pxZ).
                pxY = float(merged.get("pxY", torch.as_tensor(200.0)))
                pxZ = float(merged.get("pxZ", torch.as_tensor(pxY)))
                px_mean = 0.5 * (pxY + pxZ)
                R_ideal = R_ideal_px(
                    fits.ring_two_theta_deg,
                    merged["Lsd"].detach(),
                    torch.as_tensor(px_mean, dtype=fits.Y_pix.dtype),
                )
                # r_un = 1 - R_fwd/R_ideal  →  ΔR_px = R_fwd - R_ideal = -R_ideal·r_un
                delta_R_um = (-R_ideal * r_un) * px_mean
                abs_r = r_un.abs().cpu().numpy()
                if abs_r.size > 0:
                    cutoff = float(np.percentile(abs_r, residual_corr_outlier_pct))
                    keep = torch.as_tensor(abs_r < cutoff)
                    Y_all.append(fits.Y_pix[keep].detach().cpu())
                    Z_all.append(fits.Z_pix[keep].detach().cpu())
                    dR_all.append(delta_R_um[keep].detach().cpu())

        if Y_all and sum(t.numel() for t in Y_all) >= 50:
            Y_cat = torch.cat(Y_all)
            Z_cat = torch.cat(Z_all)
            dR_cat = torch.cat(dR_all)
            NrPixelsY = int(v1_per_image[0].NrPixelsY)
            NrPixelsZ = int(v1_per_image[0].NrPixelsZ)
            pxY0 = float(v1_per_image[0].pxY)
            if verbose:
                print(f"[multi] building residual corr map from "
                      f"{Y_cat.numel()} non-outlier fits across {n_imgs} images...",
                      flush=True)
            residual_map = build_residual_corr_map(
                Y_cat, Z_cat, dR_cat,
                NrPixelsY=NrPixelsY, NrPixelsZ=NrPixelsZ, pxY=pxY0,
                dtype=dtype,
            ).to(device=device)
            if residual_corr_path is not None:
                save_residual_corr_bin(residual_map, residual_corr_path)
                if verbose:
                    print(f"[multi] saved residual map -> {residual_corr_path}",
                          flush=True)

            # Honest post-residual strain: rebuild E-step at MAP, evaluate
            # with the new map applied.
            with torch.no_grad():
                fits_post = [run_estep_v1(v1, img, dark=drk, mask=msk,
                                           dtype=dtype, device=device)
                              for v1, img, drk, msk in zip(v1_per_image, images, darks, masks)]
                for fits, per_d in zip(fits_post, per_dicts_final):
                    merged = {**shared_dict_final, **per_d,
                              "residual_corr_map": residual_map}
                    r = pseudo_strain_residual(
                        fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, merged,
                        rho_d=fits.rho_d, weights=fits.weights,
                        panel_layout=panel_layout, panel_idx=fits.panel_idx,
                    )
                    post_strain.append(float(r.abs().mean()) * 1e6)
                if verbose:
                    pretty = ", ".join(f"img{i}={s:.1f}μϵ" for i, s in enumerate(post_strain))
                    print(f"[multi] strain after residual map: {pretty}",
                          flush=True)
        elif verbose:
            print(f"[multi] residual corr map skipped: only "
                  f"{sum(t.numel() for t in Y_all)} non-outlier fits "
                  f"(need >=50)", flush=True)

    L0_out = None
    lsd_out = None
    if link_lsd:
        L0_out = float(shared_dict_final["Lsd"].detach())
        lsd_out = [L0_out + float(d) for d in deltas_t]

    return MultiResult(
        multi_spec=multi_spec,
        shared_unpacked=shared_dict_final,
        per_image_unpacked=per_dicts_final,
        cost=cost_final, rc=rc_final,
        residual_corr_map=residual_map,
        post_residual_strain_uE=post_strain,
        L0_um=L0_out,
        lsd_per_image_um=lsd_out,
    )


__all__ = ["build_multi_spec", "MultiResult", "autocalibrate_multi",
           "MULTI_MODES"]
