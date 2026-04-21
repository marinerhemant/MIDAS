"""Command-line entry points.

``midas-auto-calibrate`` drives :func:`calibrate_progressive` — the
recommended two-stage workflow (geometry lock in Stage 1, higher-order
distortion in Stage 2 with per-p seeds). A MIDAS Parameters.txt provides
the detector + calibrant knobs; the CLI writes the refined geometry as
``geometry.json`` in ``--work-dir``.

``midas-calib-validate`` / ``midas-calib-benchmark`` remain stubs —
implementation is scheduled in the release plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_VALUE_LISTS = {"ImTransOpt", "RingsToExclude", "BoxSizes", "PanelGapsY",
                "PanelGapsZ", "RingThresh"}


def _parse_params_file(path: Path) -> dict[str, Any]:
    """Minimal MIDAS Parameters.txt → dict reader.

    Supports ``key value`` lines, ``#`` / ``%`` comment lines, and list-valued
    keys that appear multiple times (aggregated into a list). Values are
    left as strings; numeric coercion happens at the callsite where the
    target type is known.
    """
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
            if key in _VALUE_LISTS:
                params.setdefault(key, []).append(value.split())
            elif key in params:
                if not isinstance(params[key], list):
                    params[key] = [params[key]]
                params[key].append(value)
            else:
                params[key] = value
    return params


def _try_float(s: Any) -> Any:
    try:
        return float(s)
    except (TypeError, ValueError):
        return s


def _config_from_params(params: dict[str, Any]):
    from ._config import CalibrationConfig

    def f(key, default=None):
        return _try_float(params.get(key, default))

    def i(key, default=None):
        v = params.get(key, default)
        try:
            return int(float(v)) if v is not None else default
        except (TypeError, ValueError):
            return default

    lattice = None
    if "LatticeConstant" in params:
        vals = str(params["LatticeConstant"]).split()
        lattice = tuple(float(x) for x in vals)
    material = str(params.get("SpaceGroup") or params.get("Material") or "CeO2")

    im_trans_raw = params.get("ImTransOpt", [])
    # Normalise to flat [int, ...] — multi-line repeats arrive as a list
    # of token-lists, single-line values arrive as a whitespace string.
    im_trans: list[int] = []
    if isinstance(im_trans_raw, list):
        for row in im_trans_raw:
            im_trans.extend(int(float(t)) for t in row)
    else:
        im_trans = [int(float(t)) for t in str(im_trans_raw).split()]
    im_trans = im_trans or [0]

    structured = {
        "LatticeConstant", "SpaceGroup", "Material", "Wavelength", "px",
        "PxY", "Lsd", "BC", "NrPixelsY", "NrPixelsZ", "RhoD", "ImTransOpt",
        "Dark", "ImageMaskFile", "RMax",
    }
    extra: dict[str, Any] = {}
    for k, v in params.items():
        if k in structured:
            continue
        extra[k] = v if isinstance(v, list) else _try_float(v)

    ybc, zbc = 0.0, 0.0
    if "BC" in params:
        bc_parts = str(params["BC"]).split()
        if len(bc_parts) >= 2:
            ybc, zbc = float(bc_parts[0]), float(bc_parts[1])

    return CalibrationConfig(
        material=material,
        lattice_params=lattice,
        wavelength=f("Wavelength", 0.0),
        pixel_size=f("px", f("PxY", 200.0)),
        lsd=f("Lsd", 1_000_000.0),
        ybc=ybc, zbc=zbc,
        rho_d=f("RhoD", None),
        nr_pixels_y=i("NrPixelsY", 2048),
        nr_pixels_z=i("NrPixelsZ", 2048),
        r_max=f("RMax", None),
        dark_file=str(params.get("Dark", "")),
        mask_file=str(params.get("ImageMaskFile", "")),
        im_trans_opt=im_trans,
        extra_params=extra,
    )


def main_calibrate(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midas-auto-calibrate",
        description="Two-stage auto-calibration (geometry → distortion). "
                    "Wraps calibrate_progressive() for CLI use.",
    )
    parser.add_argument("params", type=Path,
                        help="MIDAS Parameters.txt with detector + calibrant knobs.")
    parser.add_argument("image", type=Path, help="Raw calibrant image.")
    parser.add_argument("--work-dir", type=Path, default=Path("calibration"),
                        help="Dir for stage subdirs + output geometry.json.")
    parser.add_argument("--fit-p-models", default="all",
                        help="Higher-order distortion modes seeded in Stage 2 "
                             "(e.g. 'all', 'tilt,spherical', 'tilt,dipole,"
                             "trefoil'). Default: all.")
    parser.add_argument("--n-iter-stage1", type=int, default=5)
    parser.add_argument("--n-iter-stage2", type=int, default=5)
    parser.add_argument("-n", "--n-cpus", type=int, default=4)
    parser.add_argument("--single-stage", action="store_true",
                        help="Skip Stage 2. Not recommended for paper-quality.")
    args = parser.parse_args(argv)

    if not args.params.exists():
        parser.error(f"params file not found: {args.params}")
    if not args.image.exists():
        parser.error(f"image not found: {args.image}")

    params = _parse_params_file(args.params)
    cfg = _config_from_params(params)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.single_stage:
        from .calibrate import auto_calibrate
        result = auto_calibrate(cfg, args.image, work_dir=args.work_dir,
                                n_cpus=args.n_cpus)
        geom = result.geometry
        residual = result.pseudo_strain
        print(f"Single-stage residual: {residual:.2f} µε")
    else:
        from .progressive import calibrate_progressive
        result = calibrate_progressive(
            cfg, args.image, work_dir=args.work_dir,
            fit_p_models=args.fit_p_models,
            n_iterations_stage1=args.n_iter_stage1,
            n_iterations_stage2=args.n_iter_stage2,
            n_cpus=args.n_cpus,
        )
        geom = result.geometry
        residual = result.pseudo_strain
        for stage_name, stage_result in result.stages:
            print(f"{stage_name}: {stage_result.pseudo_strain:.2f} µε")
        print(f"Final residual: {residual:.2f} µε")

    out_json = args.work_dir / "geometry.json"
    out_json.write_text(json.dumps({
        "lsd": geom.lsd, "ybc": geom.ybc, "zbc": geom.zbc,
        "tx": geom.tx, "ty": geom.ty, "tz": geom.tz,
        **{f"p{i}": getattr(geom, f"p{i}") for i in range(15)},
        "px": geom.px, "nr_pixels_y": geom.nr_pixels_y,
        "nr_pixels_z": geom.nr_pixels_z,
        "pseudo_strain_ustrain": float(residual),
    }, indent=2))
    print(f"Wrote {out_json}")
    return 0


def _not_implemented(name: str) -> int:
    sys.stderr.write(
        f"{name}: implementation pending. See "
        "packages/midas_auto_calibrate/ release plan for timeline.\n"
    )
    return 2


def main_validate() -> int:
    return _not_implemented("midas-calib-validate")


def main_benchmark() -> int:
    return _not_implemented("midas-calib-benchmark")
