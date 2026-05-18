"""Command-line entry point — ``midas-calibrate-v2``.

Mirrors the v1 CLI (``midas-calibrate``) for the single-image case so users
can opt in by changing the binary name; v2-specific flags select the
multi-image, Bayesian, and NN-residual pipelines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _load_image(path: Path):
    import numpy as np
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p)
    if p.endswith(".tif") or p.endswith(".tiff"):
        import tifffile
        return tifffile.imread(p)
    if p.endswith(".h5") or p.endswith(".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            keys = list(f.keys())
            return f[keys[0]][...]
    raise ValueError(f"unsupported image format: {p}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser("midas-calibrate-v2",
                                  description="Differentiable detector calibration v2.")
    p.add_argument("paramsfile", type=Path, help="v1-format paramstest.txt")
    p.add_argument("--mode", choices=("single", "multi", "bayesian",
                                      "nn", "joint", "sensitivity"),
                   default="single",
                   help="pipeline to run")
    p.add_argument("--image", type=Path, help="image file (single-image modes)")
    p.add_argument("--dark", type=Path, default=None)
    p.add_argument("--images", type=Path, nargs="+",
                   help="image files (multi-image mode)")
    p.add_argument("--paramsfiles", type=Path, nargs="+",
                   help="per-image paramstest files (multi-image mode)")
    p.add_argument("--bayesian-mode", choices=("laplace", "vi", "hmc"),
                   default="laplace")
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("paramstest_v2.txt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args(argv)

    from midas_calibrate.params import CalibrationParams as V1Params
    from .compat.to_v1 import write_v1_paramstest

    v1 = V1Params.from_file(args.paramsfile)

    if args.mode == "single":
        from .pipelines.single import autocalibrate
        if args.image is None:
            raise SystemExit("--image is required for single-image mode")
        image = _load_image(args.image)
        dark = _load_image(args.dark) if args.dark else None
        result = autocalibrate(v1, image, dark=dark, n_iter=args.n_iter,
                                device=args.device, verbose=args.verbose)
        write_v1_paramstest(result.unpacked, v1, args.output)
        print(f"wrote {args.output}")
        return 0

    if args.mode == "multi":
        from .pipelines.multi import autocalibrate_multi
        if not args.images or not args.paramsfiles:
            raise SystemExit("--images and --paramsfiles are required for multi mode")
        if len(args.images) != len(args.paramsfiles):
            raise SystemExit("len(--images) must equal len(--paramsfiles)")
        v1s = [V1Params.from_file(p) for p in args.paramsfiles]
        imgs = [_load_image(p) for p in args.images]
        result = autocalibrate_multi(v1s, imgs, n_iter=args.n_iter,
                                       device=args.device, verbose=args.verbose)
        # Write per-image paramstest files.
        for i, (per, v1_i) in enumerate(zip(result.per_image_unpacked, v1s)):
            unpacked = {**result.shared_unpacked, **per}
            out = args.output.with_suffix(f".image{i}.txt")
            write_v1_paramstest(unpacked, v1_i, out)
            print(f"wrote {out}")
        return 0

    if args.mode == "bayesian":
        from .pipelines.bayesian import autocalibrate_bayesian
        if args.image is None:
            raise SystemExit("--image is required for bayesian mode")
        image = _load_image(args.image)
        dark = _load_image(args.dark) if args.dark else None
        result = autocalibrate_bayesian(v1, image, dark=dark,
                                          mode=args.bayesian_mode,
                                          device=args.device, verbose=args.verbose)
        write_v1_paramstest(result.map_unpacked, v1, args.output)
        print(f"wrote {args.output}; Laplace covariance computed.")
        return 0

    if args.mode == "nn":
        from .pipelines.nn_residual import autocalibrate_nn
        if args.image is None:
            raise SystemExit("--image is required for nn mode")
        image = _load_image(args.image)
        dark = _load_image(args.dark) if args.dark else None
        result = autocalibrate_nn(v1, image, dark=dark,
                                    device=args.device, verbose=args.verbose)
        write_v1_paramstest(result.map_unpacked, v1, args.output)
        print(f"wrote {args.output}; NN-residual training complete.")
        return 0

    if args.mode == "joint":
        from .pipelines.joint_cake import autocalibrate_joint
        if args.image is None:
            raise SystemExit("--image is required for joint-cake mode")
        image = _load_image(args.image)
        dark = _load_image(args.dark) if args.dark else None
        result = autocalibrate_joint(v1, image, dark=dark,
                                      device=args.device, verbose=args.verbose)
        write_v1_paramstest(result.map_unpacked, v1, args.output)
        print(f"wrote {args.output}; joint forward cake fit complete.")
        return 0

    if args.mode == "sensitivity":
        raise SystemExit(
            "sensitivity mode requires a user-supplied differentiable HEDM "
            "evaluator; use the Python API:\n"
            "    from midas_calibrate_v2.pipelines.downstream import "
            "sensitivity_diagnostic"
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
