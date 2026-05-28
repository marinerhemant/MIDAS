"""Write a v2 result back to a v1-compatible paramstest.txt.

Downstream MIDAS HEDM tools consume the v1 format; v2 exports here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from midas_calibrate.params import CalibrationParams as V1Params

from ..parameters.spec import CalibrationSpec


# v2 distortion name → v1 p-index (inverse of V1_TO_V2_DISTORTION in from_v1).
_V2_TO_V1_DISTORTION = {
    "iso_R2": "p2", "iso_R4": "p5", "iso_R6": "p4",
    "a1": "p7",  "phi1": "p8",
    "a2": "p0",  "phi2": "p6",
    "a3": "p9",  "phi3": "p10",
    "a4": "p1",  "phi4": "p3",
    "a5": "p11", "phi5": "p12",
    "a6": "p13", "phi6": "p14",
}


def unpacked_to_v1_params(
    unpacked: Dict[str, torch.Tensor],
    template: V1Params,
) -> V1Params:
    """Copy refined values from a v2 unpacked dict back into a v1 params object.

    v2's distortion names (``iso_R2``..``a6/phi6``) are translated back to
    the v1 ``p₀``..``p₁₄`` slots so downstream HEDM tools see the same file.
    """
    out = V1Params(**{k: getattr(template, k) for k in template.__dict__})
    for name, val in unpacked.items():
        if name in ("panel_delta_yz", "panel_delta_theta",
                    "panel_delta_lsd", "panel_delta_p2"):
            continue   # panel data goes to a separate file
        scalar = val.detach().reshape(-1)[0].item() if val.ndim > 0 else val.item()
        # Map v2 distortion names back to v1 p-indices for output.
        target = _V2_TO_V1_DISTORTION.get(name, name)
        if hasattr(out, target):
            cur = getattr(out, target)
            try:
                setattr(out, target, type(cur)(scalar))
            except Exception:
                setattr(out, target, scalar)
    return out


def write_v1_paramstest(
    unpacked: Dict[str, torch.Tensor],
    template: V1Params,
    path: Path | str,
) -> None:
    """Write a v1-compatible paramstest.txt at the given path."""
    out = unpacked_to_v1_params(unpacked, template)
    out.write(path)


def write_ff_paramstest(
    result,
    template: V1Params,
    out_dir: Path | str,
    *,
    paramstest_name: str = "paramstest.txt",
    spline_filename: str = "residual_corr.bin",
) -> Path:
    """Export a v2 calibration result as the FF paramstest consumed by
    midas-peakfit + midas-transforms (via the zipper's analysis_parameters).

    The FF analogue of :func:`to_integrate.to_integrate_params`: it writes the
    refined geometry + ``p0..p14`` distortion (translated from v2 names) AND,
    when the result carries a Stage-4 residual spline, evaluates it on the
    detector grid into ``out_dir/spline_filename`` and records a
    ``ResidualCorrectionMap`` line so both consumers apply the full v2
    distortion (analytical harmonic + spline residual).

    ``result`` may be a ``FourStageResult`` (``.stage2.unpacked`` +
    ``.stage3_spline_fn``) or any object exposing ``.unpacked``; the spline is
    only written if present. ``template`` supplies fixed fields (NrPixelsY/Z,
    px, lattice, etc.). Returns the paramstest path.
    """
    from .to_integrate import write_residual_correction_from_spline

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "stage2"):
        unpacked = result.stage2.unpacked
        spline_fn = getattr(result, "stage3_spline_fn", None)
    elif hasattr(result, "unpacked"):
        unpacked = result.unpacked
        spline_fn = getattr(result, "stage3_spline_fn", None)
    else:
        raise TypeError(
            f"result type {type(result).__name__} has neither .unpacked nor "
            ".stage2 — pass a FourStageResult or a result exposing .unpacked"
        )

    params = unpacked_to_v1_params(unpacked, template)

    if spline_fn is not None:
        spline_path = out_dir / spline_filename
        px_mean = (0.5 * (params.pxY + params.pxZ)
                   if params.pxZ > 0 else params.pxY)
        write_residual_correction_from_spline(
            spline_fn,
            NrPixelsY=params.NrPixelsY, NrPixelsZ=params.NrPixelsZ,
            px_mean_um=px_mean,
            out_path=spline_path,
        )
        # Carried verbatim by ff_zip.write_analysis_parameters into the zarr.
        params.extra["ResidualCorrectionMap"] = str(spline_path)

    ptest = out_dir / paramstest_name
    params.write(ptest)
    return ptest


def ff_paramstest_from_auto_result(
    result,
    template_ps: Path | str,
    out_path: Path | str,
    *,
    raw_folder: str | None = None,
    n_pixels_y: int | None = None,
    n_pixels_z: int | None = None,
) -> Path:
    """Write an FF paramstest from an :class:`AutoCalibrationResult`, carrying
    the **full** refined geometry + distortion + residual map from the powder
    calibration into the HEDM reconstruction — no zeroing, no manual ``p``-slots.

    The v2 result is the single source of truth, written in the **v2 harmonic
    naming** — no ``p0..p14``:
      * geometry  — ``Lsd, BC_y, BC_z, tx, ty, tz`` (µm, deg);
      * distortion — the v2-named harmonics in ``result.distortion``
        (``iso_R2/4/6, a1..a6, phi1..phi6``), written verbatim. The zipper
        carries them into the zarr; peakfit + transforms read v2 names natively
        (legacy ``p0..p14`` still accepted for old files);
      * spline residual — ``result.residual_corr_bin_path`` recorded as
        ``ResidualCorrectionMap`` so peakfit + transforms apply it.

    Indexing/refinement thresholds (``RingThresh``, ``MinNrSpots``, …) are
    carried verbatim from ``template_ps``; only the geometry/distortion/detector
    keys are replaced. Caller supplies ``raw_folder`` (sample frames) and, if the
    detector size differs from the template, ``n_pixels_y/z``.

    Returns the written paramstest path.
    """
    from midas_distortion import P_COEF_NAMES

    dist = dict(getattr(result, "distortion", {}) or {})
    ny = int(n_pixels_y if n_pixels_y is not None else getattr(result, "NrPixelsY", 0))
    nz = int(n_pixels_z if n_pixels_z is not None else getattr(result, "NrPixelsZ", 0))

    # Keys we replace from the v2 result; everything else in the template
    # (thresholds, ring numbers, omega scan, …) is carried verbatim. We strip
    # both the v2 distortion names and any legacy p0..p14 so the output is
    # unambiguously v2-named. ``px`` is replaced too — without it, downstream
    # tools (midas-joint-ff-calibrate, peakfit) silently default to 0 for the
    # pixel↔µm conversion and every spot prediction lands at the wrong
    # position, dropping the match count to zero.
    replaced = {"Lsd", "BC", "tx", "ty", "tz", "RawFolder",
                "NrPixelsY", "NrPixelsZ", "px", "ResidualCorrectionMap",
                *P_COEF_NAMES, *(f"p{i}" for i in range(15))}

    kept: list[str] = []
    for ln in Path(template_ps).read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.split()[0] not in replaced:
            kept.append(ln)

    inj = [
        "# --- geometry + FULL distortion from midas-calibrate-v2 (powder) ---",
        f"Lsd {result.Lsd:.10g}",
        f"BC {result.BC_y:.10g} {result.BC_z:.10g}",
        f"tx {result.tx:.10g}", f"ty {result.ty:.10g}", f"tz {result.tz:.10g}",
        "# v2 distortion harmonics (carried natively; no p0..p14)",
        *[f"{nm} {float(dist.get(nm, 0.0)):.10g}" for nm in P_COEF_NAMES],
    ]
    # Pixel size: v1's ``px`` is the (Y,Z)-mean — same convention to_integrate
    # and the rest of the codebase use. Fall back to ``pxY`` if pxZ is absent
    # (single-axis calibrations). Skip if the result doesn't carry pixel size.
    pxY = float(getattr(result, "pxY", 0.0) or 0.0)
    pxZ = float(getattr(result, "pxZ", 0.0) or 0.0)
    if pxY > 0:
        px_mean = 0.5 * (pxY + pxZ) if pxZ > 0 else pxY
        inj.append(f"px {px_mean:.10g}")
    if ny > 0 and nz > 0:
        inj += [f"NrPixelsY {ny}", f"NrPixelsZ {nz}"]
    if raw_folder:
        rf = str(raw_folder)
        inj.append(f"RawFolder {rf if rf.endswith('/') else rf + '/'}")
    # Only apply the residual-correction map if calibration KEPT it. It sets
    # ``residual_corr_map=None`` when the empirical map worsened strain and was
    # discarded — the .bin is still on disk and ``residual_corr_bin_path`` stays
    # set, but applying it would degrade the reconstruction.
    rcm = getattr(result, "residual_corr_bin_path", None)
    rcm_kept = getattr(result, "residual_corr_map", None) is not None
    if rcm and rcm_kept:
        inj += ["# Stage-4 spline residual-correction map (px), applied by "
                "peakfit + transforms", f"ResidualCorrectionMap {rcm}"]
    elif rcm and not rcm_kept:
        inj += ["# residual-correction map was discarded by calibration "
                "(did not reduce strain) — intentionally NOT applied"]

    out_path = Path(out_path)
    out_path.write_text("\n".join(kept + inj) + "\n")
    return out_path


def write_panel_shifts_file(
    unpacked: Dict[str, torch.Tensor],
    path: Path | str,
) -> None:
    """Write a v1-compatible PanelShiftsFile (text, six columns).

    Columns: panel_id, δy, δz, δθ, δLsd, δp₂.
    """
    dyz = unpacked.get("panel_delta_yz")
    dth = unpacked.get("panel_delta_theta")
    dl  = unpacked.get("panel_delta_lsd")
    dp2 = unpacked.get("panel_delta_p2")
    if dyz is None or dth is None:
        raise ValueError("panel_delta_yz and panel_delta_theta are required")
    n = dyz.shape[0]
    if dl is None:
        dl = torch.zeros(n)
    if dp2 is None:
        dp2 = torch.zeros(n)
    lines = []
    for k in range(n):
        lines.append(f"{k:5d}  "
                     f"{float(dyz[k, 0]):+.6f}  {float(dyz[k, 1]):+.6f}  "
                     f"{float(dth[k]):+.6e}  "
                     f"{float(dl[k]):+.4f}  {float(dp2[k]):+.6e}")
    Path(path).write_text("\n".join(lines) + "\n")


__all__ = ["unpacked_to_v1_params", "write_v1_paramstest", "write_ff_paramstest",
           "ff_paramstest_from_auto_result", "write_panel_shifts_file"]
