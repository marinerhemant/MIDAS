"""Convert v2 calibration outputs to formats consumed by ``midas-integrate``.

Two cases that the v1 paramstest export (``compat/to_v1.py``) cannot
carry through:

1. The Stage 4 thin-plate spline residual correction
   (:class:`pipelines.four_stage.FourStageResult.stage3_spline_fn`).
   Evaluated on the full detector grid here and written as the binary
   ``ΔR(Y, Z)`` lookup file that ``midas_integrate.residual_corr``
   expects: shape ``(NrPixelsZ, NrPixelsY)`` of float64, row-major,
   units of pixels.

2. The per-ring offset ``δr_k`` (F2 fix).  ``integrate`` v1 has no
   per-ring concept in its radial map, so we cannot apply it inside
   the integration kernel.  We emit a small companion JSON file that
   downstream peak-fit / Rietveld tools can consume; the integration
   output remains uncorrected for δr_k and the user is warned.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch


def write_residual_correction_from_spline(
    spline_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    NrPixelsY: int, NrPixelsZ: int,
    px_mean_um: float,
    out_path: Path | str,
    chunk_size: int = 500_000,
) -> None:
    """Evaluate v2's Stage 4 RBF spline on the full detector grid and
    write the binary file consumed by
    :func:`midas_integrate.residual_corr.load_residual_correction_map`.

    Parameters
    ----------
    spline_predict :
        Callable ``predict(Y_pix, Z_pix) -> ΔR_um`` returned by
        :func:`midas_calibrate_v2.pipelines.four_stage.autocalibrate_four_stage`
        as ``result.stage3_spline_fn``.  Inputs are 1-D numpy arrays of
        pixel positions; output is ΔR in micrometres.
    NrPixelsY, NrPixelsZ :
        Detector dimensions (must match the calibration's spec).
    px_mean_um :
        Pixel pitch in µm; used to convert the spline's ΔR (µm) to the
        binary file's units (pixels).  For non-square pixels pass
        ``(pxY + pxZ) / 2``.
    out_path :
        Where to write the binary file (typical convention:
        ``<image>.residual_corr.bin`` next to the image).
    chunk_size :
        Pixels per RBF query.  ``RBFInterpolator(neighbors=200)`` cost
        per query is O(N · 200); chunking keeps peak memory bounded
        even on 16 MP detectors (≈ 16 M pixels / 500 k chunks → 33
        chunks).
    """
    out_path = Path(out_path)
    n_total = NrPixelsY * NrPixelsZ
    grid = np.empty((NrPixelsZ, NrPixelsY), dtype=np.float64)

    # Build the full (Y, Z) coordinate grid in row-major order to match
    # integrate's storage convention (z-outer, y-inner).
    for z_start in range(0, NrPixelsZ, max(1, chunk_size // NrPixelsY)):
        z_end = min(NrPixelsZ, z_start + max(1, chunk_size // NrPixelsY))
        zs = np.repeat(np.arange(z_start, z_end, dtype=np.float64), NrPixelsY)
        ys = np.tile(np.arange(NrPixelsY, dtype=np.float64), z_end - z_start)
        dR_um = spline_predict(ys, zs)
        # ΔR µm → pixels
        dR_px = dR_um / px_mean_um
        grid[z_start:z_end, :] = dR_px.reshape(z_end - z_start, NrPixelsY)

    grid.astype(np.float64).tofile(out_path)


def write_per_ring_offsets_json(
    unpacked: Dict[str, torch.Tensor],
    *,
    ring_d_spacing_A: np.ndarray,
    ring_two_theta_deg: np.ndarray,
    out_path: Path | str,
) -> None:
    """Emit a JSON sidecar describing v2's per-ring radial offsets ``δr_k``.

    ``midas-integrate`` v1's radial map cannot apply per-ring shifts
    (every pixel is binned by its (R, η) regardless of which ring it
    actually came from), so when the v2 calibration uses the F2 fix the
    correction has to be applied at the downstream peak-fit / Rietveld
    stage.  This sidecar carries the information through.

    File schema (JSON)::

        {
          "version": 1,
          "n_rings": N,
          "rings": [
            {"k": 0, "d_spacing_A": 3.124, "two_theta_deg": 3.61,
             "delta_r_px": -0.0123},
            ...
          ]
        }

    No-op if ``unpacked`` does not carry a ``delta_r_k`` parameter.
    """
    dr = unpacked.get("delta_r_k")
    if dr is None:
        return
    dr = dr.detach().cpu().numpy()
    n = int(dr.shape[0])
    if len(ring_d_spacing_A) != n or len(ring_two_theta_deg) != n:
        raise ValueError(
            f"delta_r_k length ({n}) does not match ring_d_spacing_A "
            f"({len(ring_d_spacing_A)}) or ring_two_theta_deg "
            f"({len(ring_two_theta_deg)})"
        )
    payload = {
        "version": 1,
        "n_rings": n,
        "rings": [
            {
                "k": int(k),
                "d_spacing_A": float(ring_d_spacing_A[k]),
                "two_theta_deg": float(ring_two_theta_deg[k]),
                "delta_r_px": float(dr[k]),
            }
            for k in range(n)
        ],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))


def to_integrate_params(
    result: object,
    *,
    template,           # midas_integrate.params.IntegrationParams
    output_dir: Optional[Path | str] = None,
    ring_d_spacing_A: Optional[np.ndarray] = None,
    ring_two_theta_deg: Optional[np.ndarray] = None,
    spline_filename: str = "residual_corr.bin",
    delta_r_filename: str = "delta_r_k.json",
    warn_on_dropped: bool = True,
):
    """One-shot v2-result-to-integrate handoff.

    Accepts any v2 pipeline result that exposes ``.unpacked``
    (``PVCalibrationResult``) or its ``.stage2.unpacked`` plus
    ``.stage3_spline_fn`` (``FourStageResult``), and produces a fully
    populated :class:`midas_integrate.params.IntegrationParams` together
    with any sidecar files needed for downstream tooling:

    - **Stage 4 spline** (``FourStageResult.stage3_spline_fn``): if
      present and ``output_dir`` is given, the spline is evaluated on the
      detector grid and written as ``output_dir / spline_filename``;
      ``IntegrationParams.ResidualCorrectionMap`` is set to that path.
    - **Per-ring δr_k**: if present in ``unpacked`` and ``ring_d_spacing_A``
      / ``ring_two_theta_deg`` and ``output_dir`` are given, the JSON
      sidecar is written to ``output_dir / delta_r_filename``. δr_k cannot
      be applied inside the v1 integration kernel; downstream peak-fit /
      Rietveld tools should pick up the sidecar.

    Parameters
    ----------
    result :
        ``PVCalibrationResult`` or ``FourStageResult`` (anything with
        ``.unpacked`` or ``.stage2.unpacked``).
    template :
        v1 :class:`IntegrationParams` carrying NrPixelsY/Z, RhoD, binning,
        TransOpt, mask path, etc. Easiest to construct via
        :func:`midas_integrate.params.parse_params` from the seed paramstest
        the v2 calibration started from.
    output_dir :
        If given, sidecar files are written here. The directory must exist.

    Returns
    -------
    A new :class:`IntegrationParams` ready for
    :func:`midas_integrate.detector_mapper.build_map`.
    """
    from midas_integrate.compat.from_v2 import params_from_v2_unpacked

    # Normalise result types: PVCalibrationResult exposes .unpacked
    # directly; FourStageResult lives one level down at .stage2.
    if hasattr(result, "unpacked"):
        unpacked = result.unpacked
        spline_fn = None
    elif hasattr(result, "stage2"):
        unpacked = result.stage2.unpacked
        spline_fn = getattr(result, "stage3_spline_fn", None)
    else:
        raise TypeError(
            f"result type {type(result).__name__} has neither .unpacked nor "
            ".stage2 — pass a PVCalibrationResult or FourStageResult"
        )

    ip = params_from_v2_unpacked(unpacked, template=template,
                                  warn_on_dropped=warn_on_dropped)

    if output_dir is None:
        return ip

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"output_dir does not exist: {output_dir}"
        )

    if spline_fn is not None:
        spline_path = output_dir / spline_filename
        px_mean = 0.5 * (ip.pxY + ip.pxZ) if ip.pxZ > 0 else ip.pxY
        write_residual_correction_from_spline(
            spline_fn,
            NrPixelsY=ip.NrPixelsY, NrPixelsZ=ip.NrPixelsZ,
            px_mean_um=px_mean,
            out_path=spline_path,
        )
        ip.ResidualCorrectionMap = str(spline_path)

    if "delta_r_k" in unpacked:
        if ring_d_spacing_A is not None and ring_two_theta_deg is not None:
            write_per_ring_offsets_json(
                unpacked,
                ring_d_spacing_A=ring_d_spacing_A,
                ring_two_theta_deg=ring_two_theta_deg,
                out_path=output_dir / delta_r_filename,
            )
        # If ring info is missing, params_from_v2_unpacked already warned.

    return ip


def spec_from_calibration_result(
    result,
    *,
    RBinSize: float,
    EtaBinSize: float = 5.0,
    RMin: float = 10.0,
    RMax: Optional[float] = None,
    EtaMin: float = -180.0,
    EtaMax: float = 180.0,
    residual_corr_map: Optional[str] = None,
):
    """Build a ``midas_integrate_v2`` ``IntegrationSpec`` from a
    :class:`~midas_calibrate_v2.pipelines.auto.AutoCalibrationResult`
    (the object returned by :func:`midas_calibrate_v2.calibrate`).

    The calibration result carries the full refined geometry + distortion +
    residual-map path, but NOT the integration binning
    (``RMin``/``RMax``/``RBinSize``/``EtaBinSize``) — those are an integration
    choice, supplied here. ``RBinSize`` is required; ``RMax=None`` defaults to
    the beam-centre-to-farthest-corner distance (the full detector).

    The returned spec is the single source for **both** integrators:

    * **v2** — feed straight to ``midas_integrate_v2.build_geometry`` /
      ``integrate``.
    * **v1** — convert with
      ``midas_integrate_v2.compat.to_v1.v1_params_from_spec(spec)`` for the
      legacy ``midas_integrate`` path.

    Notes
    -----
    ``RhoD`` is set in **pixels** (the v2 spec convention; the forward model
    converts internally). Distortion harmonic names are shared between
    ``midas_calibrate_v2`` and ``midas_integrate_v2``, so they copy across with
    no translation.
    """
    try:
        import torch
        from midas_integrate_v2.spec import IntegrationSpec, DISTORTION_NAMES
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "spec_from_calibration_result requires midas-integrate-v2 "
            "(pip install midas-integrate-v2)"
        ) from e

    def _t(x):
        return torch.as_tensor(x, dtype=torch.float64)

    s = IntegrationSpec()
    s.NrPixelsY = int(result.NrPixelsY)
    s.NrPixelsZ = int(result.NrPixelsZ)
    s.pxY = float(result.pxY)
    s.pxZ = float(result.pxZ)
    s.Lsd = _t(result.Lsd)
    s.BC_y = _t(result.BC_y)
    s.BC_z = _t(result.BC_z)
    s.tx = _t(result.tx)
    s.ty = _t(result.ty)
    s.tz = _t(result.tz)
    s.Wavelength = _t(result.wavelength_A)
    s.Parallax = _t(0.0)

    NY, NZ = s.NrPixelsY, s.NrPixelsZ
    # RhoD is the distortion normalisation radius and MUST be in micrometres
    # (the forward model uses ρ = R_µm / RhoD). It is the beam-centre-to-
    # farthest-corner distance in px times the pixel pitch — the same value the
    # calibration used. (Setting it in pixels makes ρ ~px/µm too large by the
    # pitch, so the distortion polynomial explodes and washes out the rings.)
    px_mean = 0.5 * (float(result.pxY) + float(result.pxZ))
    corner_px = math.sqrt(
        max(result.BC_y, NY - 1 - result.BC_y) ** 2
        + max(result.BC_z, NZ - 1 - result.BC_z) ** 2
    )
    s.RhoD = corner_px * px_mean

    for name in DISTORTION_NAMES:
        if name in result.distortion and hasattr(s, name):
            setattr(s, name, _t(result.distortion[name]))

    rcm = residual_corr_map
    if rcm is None:
        rcm = getattr(result, "residual_corr_bin_path", None)
    if rcm:
        s.ResidualCorrectionMap = rcm

    s.RMin = float(RMin)
    # RMin/RMax/RBinSize are in PIXELS (RhoD above is µm — different unit).
    s.RMax = float(RMax) if RMax is not None else float(corner_px)
    s.RBinSize = float(RBinSize)
    s.EtaMin = float(EtaMin)
    s.EtaMax = float(EtaMax)
    s.EtaBinSize = float(EtaBinSize)

    s.validate()
    return s


def spec_from_calibration_json(
    path,
    *,
    RBinSize: float,
    EtaBinSize: float = 5.0,
    RMin: float = 10.0,
    RMax: Optional[float] = None,
    EtaMin: float = -180.0,
    EtaMax: float = 180.0,
    residual_corr_map: Optional[str] = None,
):
    """Build a ``midas_integrate_v2`` ``IntegrationSpec`` from a saved
    ``calibration.json`` (the file ``calibrate(output_dir=...)`` writes).

    Convenience wrapper over :func:`spec_from_calibration_result` for batch
    workflows: calibrate once, then build the spec straight from disk for many
    sample frames — no need to keep the live ``result`` object. The JSON keys
    (``Lsd_um``, ``BC_y_px``, ``tx_deg`` …) are mapped to the bare result
    field names. ``RBinSize`` is required; the bin parameters are an
    integration choice (not stored in the calibration). If ``residual_corr_map``
    is None, the JSON's ``residual_corr_bin`` path is used.
    """
    import json
    from types import SimpleNamespace

    with open(path) as fh:
        c = json.load(fh)
    result = SimpleNamespace(
        NrPixelsY=c["NrPixelsY"], NrPixelsZ=c["NrPixelsZ"],
        pxY=c["pxY_um"], pxZ=c["pxZ_um"],
        Lsd=c["Lsd_um"], BC_y=c["BC_y_px"], BC_z=c["BC_z_px"],
        tx=c["tx_deg"], ty=c["ty_deg"], tz=c["tz_deg"],
        wavelength_A=c["wavelength_A"],
        distortion=c.get("distortion", {}),
        residual_corr_bin_path=c.get("residual_corr_bin"),
    )
    return spec_from_calibration_result(
        result, RBinSize=RBinSize, EtaBinSize=EtaBinSize,
        RMin=RMin, RMax=RMax, EtaMin=EtaMin, EtaMax=EtaMax,
        residual_corr_map=residual_corr_map,
    )


__all__ = [
    "write_residual_correction_from_spline",
    "write_per_ring_offsets_json",
    "to_integrate_params",
    "spec_from_calibration_result",
    "spec_from_calibration_json",
]
