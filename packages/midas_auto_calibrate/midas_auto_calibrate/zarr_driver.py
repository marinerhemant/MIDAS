"""Programmatic wrapper around the vendored ``AutoCalibrateZarr.main()``.

The wrapper exposes :func:`calibrate_zarr` — the paper's end-to-end
auto-calibration pipeline including outlier rejection, interleaved
tilts / distortion / panel refinement, and optional Stage-3 TPS
residual-correction map. :func:`calibrate_progressive` stays the lighter
two-stage alternative that runs a few iterations without the full
re-integration loop; ``calibrate_zarr`` is what matches the paper numbers.

Example
-------
>>> from midas_auto_calibrate import CalibrationConfig, calibrate_zarr
>>> cfg = CalibrationConfig(
...     material="CeO2", wavelength=0.196793, pixel_size=150.0,
...     lsd=895_930, ybc=1446.97, zbc=1468.91, rho_d=309_094.28,
...     nr_pixels_y=2880, nr_pixels_z=2880, im_trans_opt=[2],
... )
>>> result = calibrate_zarr(cfg, "Ceria.tif", work_dir="cal/", fit_p_models="all")
>>> print(result.pseudo_strain)        # ~5 µε on the bundled Varex frame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from .geometry import DetectorGeometry


@dataclass
class AutoCalResult:
    """Outputs of :func:`calibrate_zarr`."""

    geometry: DetectorGeometry
    pseudo_strain: float                       # mean strain, microstrain
    pseudo_strain_std: float                   # std of strain
    refined_params_file: Path                  # MIDAS Parameters.txt
    work_dir: Path
    log_lines: list[str] = field(default_factory=list)


def _parse_refined_params(path: Path) -> dict[str, Any]:
    """Read the MIDAS-style Parameters.txt the driver writes on success."""
    params: dict[str, Any] = {}
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] in "#%":
                continue
            parts = line.split(None, 1)
            if len(parts) == 1:
                continue
            key, value = parts[0], parts[1].strip()
            if key in params:
                if not isinstance(params[key], list):
                    params[key] = [params[key]]
                params[key].append(value)
            else:
                params[key] = value
    return params


def _first_refined_params(work_dir: Path) -> Path:
    """Find the refined_MIDAS_params_<stem>.txt in work_dir (newest wins)."""
    matches = sorted(
        work_dir.glob("refined_MIDAS_params_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"calibrate_zarr: no refined_MIDAS_params_*.txt written in "
            f"{work_dir}. The driver likely aborted before converging."
        )
    return matches[0]


def _geometry_from_refined(params: dict[str, Any],
                            config: "CalibrationConfig") -> DetectorGeometry:
    """Construct a ``DetectorGeometry`` from the refined params dict."""
    def g(key: str, default: float = 0.0) -> float:
        v = params.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    ybc, zbc = 0.0, 0.0
    if "BC" in params:
        parts = str(params["BC"]).split()
        if len(parts) >= 2:
            ybc, zbc = float(parts[0]), float(parts[1])

    return DetectorGeometry(
        lsd=g("Lsd", config.lsd),
        ybc=ybc or config.ybc, zbc=zbc or config.zbc,
        tx=g("tx"), ty=g("ty"), tz=g("tz"),
        p0=g("p0"), p1=g("p1"), p2=g("p2"), p3=g("p3"), p4=g("p4"),
        p5=g("p5"), p6=g("p6"), p7=g("p7"), p8=g("p8"), p9=g("p9"),
        p10=g("p10"), p11=g("p11"), p12=g("p12"),
        p13=g("p13"), p14=g("p14"),
        px=g("px", config.pixel_size),
        nr_pixels_y=int(g("NrPixelsY", config.nr_pixels_y)),
        nr_pixels_z=int(g("NrPixelsZ", config.nr_pixels_z)),
    )


def calibrate_zarr(
    config: "CalibrationConfig",
    image: Union[str, Path],
    *,
    work_dir: Union[str, Path],
    fit_p_models: str = "tilt,spherical,dipole,trefoil,octupole",
    n_iterations: int = 40,
    outlier_iterations: int = 3,
    mult_factor: float = 5.0,
    first_ring: int = 1,
    max_ring: Optional[int] = None,
    fit_residual_map: bool = True,
    gradient_correction: bool = True,
    n_cpus: int = 4,
    extra_argv: Sequence[str] = (),
    params_file: Optional[Union[str, Path]] = None,
) -> AutoCalResult:
    """Run the vendored ``AutoCalibrateZarr`` pipeline on ``image``.

    This is the paper-quality path: it matches ``AutoCalibrateZarr.py`` output
    to the parameter-file numbers because it *is* that code, called
    programmatically with argv synthesised from ``config`` + the kwargs.

    Parameters
    ----------
    config : CalibrationConfig
        Detector, calibrant, and wavelength seeds. Written to a
        Parameters.txt in ``work_dir`` and passed to the driver as
        ``--params``. Any entries in ``config.extra_params`` flow through.
    image : path
        Raw calibrant frame (TIFF/HDF5/GE/Zarr — auto-detected).
    work_dir : path
        Workdir the driver runs inside; it writes the refined params file
        and intermediate artifacts here.
    fit_p_models : str
        Higher-order distortion seeds, e.g. ``"all"``,
        ``"tilt,spherical,dipole,trefoil,octupole"``.
    n_iterations : int
        Outer refinement iterations (driver's ``-n``; 40 = paper default).
    outlier_iterations : int
        Outlier-ring rejection passes per outer iter (driver's default: 3).
    mult_factor, first_ring, max_ring
        Ring-fit weighting knobs.
    fit_residual_map : bool
        Enable Stage-3 TPS residual correction spline (paper's sub-5-µε
        final stage). Disable for apples-to-apples comparison with the
        lighter :func:`calibrate_progressive`.
    gradient_correction : bool
        Enable the paper's gradient-based peak-fit refinement.
    n_cpus : int
        OpenMP threads for the C inner loop.
    extra_argv : list[str]
        Passed verbatim to the driver's argparse (escape hatch for
        flags not exposed as kwargs).
    params_file : path, optional
        Use this Parameters.txt instead of auto-generating one from
        ``config``. Rare; the auto-generated one is usually correct.
    """
    from . import _autocal_driver as driver
    from ._config import write_params_file

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    image = Path(image).resolve()

    # Write Parameters.txt from config if the caller didn't hand one over.
    if params_file is None:
        params = config.to_params()
        # Driver writes its own Folder/FileStem/Ext, so strip those so we
        # don't fight with the auto-detection.
        for k in ("Folder", "FileStem", "StartNr", "EndNr", "Padding", "Ext"):
            params.pop(k, None)
        ps_path = work_dir / "Parameters_autocal.txt"
        write_params_file(ps_path, params)
    else:
        ps_path = Path(params_file).resolve()

    argv = [
        "--data", str(image),
        "--params", str(ps_path),
        "--fit-p-models", fit_p_models,
        "--n-iterations", str(n_iterations),
        "--outlier-iterations", str(outlier_iterations),
        "--mult-factor", str(mult_factor),
        "--first-ring", str(first_ring),
        "--fit-residual-map", "1" if fit_residual_map else "0",
        "--gradient-correction", "1" if gradient_correction else "0",
        "--plots", "0",
        "--cpus", str(n_cpus),
    ]
    if max_ring is not None:
        argv += ["--max-ring", str(max_ring)]
    argv += list(extra_argv)

    # Driver's main() parses sys.argv; patch it for the duration of the
    # call and swap cwd so its relative-path writes land in work_dir.
    import os
    import sys
    saved_argv = sys.argv
    saved_cwd = os.getcwd()
    try:
        sys.argv = ["midas-auto-calibrate"] + argv
        os.chdir(work_dir)
        rc = driver.main()
        if rc not in (None, 0):
            raise RuntimeError(
                f"AutoCalibrateZarr.main() returned exit code {rc}. "
                f"See logs in {work_dir}."
            )
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)

    refined_path = _first_refined_params(work_dir)
    refined = _parse_refined_params(refined_path)

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(refined.get(key, default))
        except (TypeError, ValueError):
            return default

    return AutoCalResult(
        geometry=_geometry_from_refined(refined, config),
        pseudo_strain=_f("MeanStrain", 0.0) * 1e6,
        pseudo_strain_std=_f("StdStrain", 0.0) * 1e6,
        refined_params_file=refined_path,
        work_dir=work_dir,
    )
