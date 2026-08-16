"""Command-line entry points for midas-calibrate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-calibrate"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)



def _load_image(path: Path) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(p)
    if p.suffix.lower() in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            keys = list(f.keys())
            return np.asarray(f[keys[0]])
    if p.suffix.lower() in (".npy",):
        return np.load(p)
    raise ValueError(f"unknown image extension: {p.suffix}")


def main(argv: Sequence[str] | None = None) -> int:
    return autocalibrate_main(argv)


def autocalibrate_main(argv: Sequence[str] | None = None) -> int:
    parser = _midas_make_parser(prog="midas-autocalibrate",
                                     description="Native Python detector calibration")
    parser.add_argument("params_file", type=Path, help="CalibrationParams .txt")
    parser.add_argument("--image", type=Path, default=None,
                        help="calibrant image (overrides ImagePath in params)")
    parser.add_argument("--dark", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="path to write parameters_refined.txt (default: alongside input)")
    parser.add_argument("--engine", choices=["alternating", "joint"], default="alternating")
    parser.add_argument("--n-iters", type=int, default=None,
                        help="override nIterations from the params file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    from .orchestrator import autocalibrate
    from .params import CalibrationParams

    params = CalibrationParams.from_file(args.params_file)
    if args.n_iters is not None:
        params.nIterations = args.n_iters

    image_path = args.image or Path(params.ImagePath)
    image = _load_image(image_path)
    dark = _load_image(args.dark) if args.dark else None

    if args.engine == "joint":
        try:
            from .joint import autocalibrate_joint
            result = autocalibrate_joint(params, image, dark=dark, verbose=not args.quiet)
        except ImportError:
            print("joint engine not yet available; falling back to alternating", file=sys.stderr)
            result = autocalibrate(params, image, dark=dark, verbose=not args.quiet)
    else:
        result = autocalibrate(params, image, dark=dark, verbose=not args.quiet)

    out = args.output or args.params_file.with_name(args.params_file.stem + "_refined.txt")
    result.params.write(out)
    if not args.quiet:
        print(f"\nFinal mean strain: {result.history[-1].mean_strain_uE:.1f} μϵ")
        print(f"Refined parameters written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
