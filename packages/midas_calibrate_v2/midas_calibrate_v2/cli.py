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
    p.add_argument("paramsfile", type=Path, nargs="?", default=None,
                   help="v1-format paramstest.txt. In ff mode this is the "
                        "TEMPLATE; omit it and pass the --from-scratch group "
                        "instead.")
    p.add_argument("--mode", choices=("single", "multi", "bayesian",
                                      "nn", "joint", "sensitivity", "ff"),
                   default="single",
                   help="pipeline to run; 'ff' = powder standard -> FF-HEDM "
                        "parameter file (see --raw-folder)")
    p.add_argument("--image", type=Path, help="image file (single-image modes)")
    p.add_argument("--dark", type=Path, default=None)
    p.add_argument("--images", type=Path, nargs="+",
                   help="image files (multi-image mode)")
    p.add_argument("--paramsfiles", type=Path, nargs="+",
                   help="per-image paramstest files (multi-image mode)")
    p.add_argument("--lsd-offsets", type=float, nargs="+", default=None,
                   metavar="UM",
                   help="multi mode: EXACTLY known relative detector travel, "
                        "one per image, in µm (any common origin — e.g. the "
                        "stage readback). Switches on linked-distance mode: "
                        "Lsd_i = L0 + Delta_i from a single shared L0, with a "
                        "shared refined Wavelength. This — not merely using "
                        "several distances — is what makes the wavelength "
                        "identifiable; with a free Lsd per image, lambda and "
                        "the distances rescale together and stay degenerate.")
    p.add_argument("--bayesian-mode", choices=("laplace", "vi", "hmc"),
                   default="laplace")
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path("paramstest_v2.txt"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true", default=True)
    # ---- ff mode -------------------------------------------------------
    ff = p.add_argument_group(
        "ff mode", "powder standard -> FF-HEDM parameter file. The positional "
        "paramsfile is the TEMPLATE: its thresholds, ring numbers, omega scan "
        "and lattice are carried over; geometry, distortion and RhoD are "
        "replaced.")
    ff.add_argument("--calibrant", default="CeO2",
                    help="CeO2 (default), LaB6, Si, Al2O3")
    ff.add_argument("--image-group", default="exchange/data",
                    help="HDF5 dataset holding the calibrant frames")
    ff.add_argument("--dark-group", default=None,
                    help="HDF5 dataset holding the dark. NB 20-ID Varex keeps "
                         "it in /exchange/bright; /exchange/dark is all zeros")
    ff.add_argument("--reduce", choices=("median", "mean"), default="median")
    ff.add_argument("--wavelength", type=float, default=None,
                    help="Angstrom; defaults to the template's value")
    ff.add_argument("--initial-lsd", type=float, default=1_000_000.0,
                    help="um; known to a few cm is enough. BC is auto-seeded "
                         "and must not be guessed")
    ff.add_argument("--im-trans", type=int, nargs="*", default=None,
                    help="MIDAS ImTransOpt codes (1 flip-Y, 2 flip-Z, "
                         "3 transpose); defaults to the template. MUST match "
                         "the reconstruction")
    ff.add_argument("--raw-folder", default=None,
                    help="the SAMPLE data folder to record in the new file")
    ff.add_argument("--strain-gate", type=float, default=100.0,
                    help="ue; exit non-zero above this (default 100)")
    ff.add_argument("--no-overlay", action="store_true")
    # ---- ff mode with no template --------------------------------------
    fs = p.add_argument_group(
        "ff mode without a template",
        "Omit the positional paramsfile and give these instead. They are the "
        "keys that describe the experiment rather than the detector, which is "
        "everything nothing else can guess. Search bounds, margins and bin "
        "sizes get known-good FF defaults (see FF_DEFAULTS); override any of "
        "them with --set KEY=VALUE.")
    fs.add_argument("--px", type=float, help="pixel pitch, um")
    fs.add_argument("--n-pixels", type=int, help="detector is n x n")
    fs.add_argument("--lattice", type=float, nargs=6,
                    metavar=("A", "B", "C", "ALPHA", "BETA", "GAMMA"),
                    help="SAMPLE lattice (not the calibrant's)")
    fs.add_argument("--space-group", type=int, help="sample space group")
    fs.add_argument("--file-stem")
    fs.add_argument("--start-file-nr", type=int)
    fs.add_argument("--ext", default=".h5")
    fs.add_argument("--padding", type=int, default=6)
    fs.add_argument("--dark-file", help="the sample's dark exposure")
    fs.add_argument("--dark-loc", help="HDF5 path to the dark inside it")
    fs.add_argument("--omega-start", type=float)
    fs.add_argument("--omega-step", type=float)
    fs.add_argument("--skip-frame", type=int, default=0)
    fs.add_argument("--n-files-per-sweep", type=int, default=1)
    fs.add_argument("--ring-thresh", action="append", metavar="RING:THRESH",
                    help="repeatable, e.g. --ring-thresh 1:75 --ring-thresh 4:70")
    fs.add_argument("--ring-to-index", type=int, default=None,
                    help="ring the indexer seeds from (OverAllRingToIndex). "
                         "Pick a STRONG, well-populated ring: it sets how many "
                         "seeds indexing gets. Defaults to the lowest ring "
                         "given, which is often not the best choice.")
    fs.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override any default, repeatable")
    args = p.parse_args(argv)

    from midas_calibrate.params import CalibrationParams as V1Params
    from .compat.to_v1 import write_v1_paramstest

    # ff mode is the one mode that can run without a template, so it is
    # dispatched before the template is loaded.
    if args.paramsfile is None and args.mode != "ff":
        raise SystemExit(f"--mode {args.mode} needs the positional paramsfile")
    v1 = V1Params.from_file(args.paramsfile) if args.paramsfile else None

    if args.mode == "ff":
        from .pipelines.ff_calibrate import (calibrate_ff_from_files,
                                             synthesize_template)
        if args.image is None:
            raise SystemExit("--image is required for ff mode (the calibrant "
                             "exposure)")
        template = args.paramsfile
        if template is None:
            need = {"--px": args.px, "--n-pixels": args.n_pixels,
                    "--lattice": args.lattice, "--space-group": args.space_group,
                    "--file-stem": args.file_stem,
                    "--start-file-nr": args.start_file_nr,
                    "--omega-start": args.omega_start,
                    "--omega-step": args.omega_step,
                    "--raw-folder": args.raw_folder,
                    "--wavelength": args.wavelength,
                    "--ring-thresh": args.ring_thresh}
            missing = [k for k, v in need.items() if v is None]
            if missing:
                raise SystemExit(
                    "no template given, so these are required: "
                    + ", ".join(missing)
                    + "\n(or pass an existing FF parameter file as the "
                      "positional argument and they are read from it)")
            try:
                rt = [tuple(float(x) for x in s.split(":", 1))
                      for s in args.ring_thresh]
            except ValueError:
                raise SystemExit("--ring-thresh wants RING:THRESH, "
                                 "e.g. --ring-thresh 1:75")
            # Extra parameter-file keys, from --set KEY=VALUE. Starts empty:
            # the template defaults live in synthesize_template, and anything
            # not overridden here is left to it.
            over: dict = {}
            for kv in (args.set or []):
                if "=" not in kv:
                    raise SystemExit(f"--set wants KEY=VALUE, got {kv!r}")
                k, v = kv.split("=", 1)
                over[k.strip()] = v.strip()
            over.setdefault("SkipFrame", str(args.skip_frame))
            over.setdefault("NrFilesPerSweep", str(args.n_files_per_sweep))
            over.setdefault("Padding", str(args.padding))
            template = Path(args.output).parent / "template_from_flags.txt"
            synthesize_template(
                template, wavelength_A=args.wavelength, px_um=args.px,
                n_pixels=args.n_pixels,
                im_trans=tuple(args.im_trans) if args.im_trans else (),
                lattice=args.lattice, space_group=args.space_group,
                raw_folder=args.raw_folder, file_stem=args.file_stem,
                start_file_nr=args.start_file_nr, ext=args.ext,
                omega_start=args.omega_start, omega_step=args.omega_step,
                ring_thresh=rt, dark_file=args.dark_file,
                dark_loc=args.dark_loc, ring_to_index=args.ring_to_index,
                overrides=over)
            print(f"[ff-calib] no template given; wrote {template}")
        res = calibrate_ff_from_files(
            args.image, template, args.output,
            raw_folder=args.raw_folder, calibrant=args.calibrant,
            wavelength_A=args.wavelength,
            data_group=args.image_group, dark_group=args.dark_group,
            reduce=args.reduce,
            im_trans=tuple(args.im_trans) if args.im_trans else (),
            initial_Lsd_um=args.initial_lsd, n_iter=args.n_iter,
            strain_gate_uE=args.strain_gate, overlay=not args.no_overlay,
            device=args.device, verbose=args.verbose)
        return 0 if res.passes_gate else 1

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
        if args.lsd_offsets is not None and len(args.lsd_offsets) != len(imgs):
            raise SystemExit(
                f"--lsd-offsets has {len(args.lsd_offsets)} values but "
                f"{len(imgs)} images were given; pass one per image")
        result = autocalibrate_multi(v1s, imgs, n_iter=args.n_iter,
                                       device=args.device, verbose=args.verbose,
                                       lsd_offsets_um=args.lsd_offsets)
        if result.L0_um is not None:
            lam = float(result.shared_unpacked["Wavelength"].detach())
            print(f"linked-distance fit: L0 = {result.L0_um/1e3:.4f} mm, "
                  f"Wavelength = {lam:.7f} A "
                  f"({12.398419843320026/lam:.4f} keV)")
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
