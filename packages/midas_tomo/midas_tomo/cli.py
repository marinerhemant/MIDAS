"""``midas-tomo`` command line entry point.

Replaces the legacy ``process_hdf.py``. Same flags, plus ``--out`` and
``--deterministic``, and with three behaviour changes that are called out in
``--help`` because they alter results rather than just ergonomics:

* Ring removal is **off** unless asked for. The legacy driver wrote
  ``ringRemovalCoefficient 1.0`` unconditionally, so every reconstruction it
  produced had ring removal on whether or not that was wanted.
* Outputs go to ``--out`` (default: next to the input) rather than being
  scattered through the current working directory.
* A ``CropXR``/``CropZR`` of 0 means "crop nothing on that side" instead of
  silently producing an empty array. See :func:`midas_tomo.hdf5.crop_slice`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from . import __version__, backend_c

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-tomo"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)


log = logging.getLogger("midas_tomo")


def _build_parser() -> argparse.ArgumentParser:
    p = _midas_make_parser(
        prog="midas-tomo",
        description="Gridrec CT reconstruction from an APS /exchange HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Differences from the legacy TOMO/process_hdf.py:\n"
            "  * ring removal is OFF by default (it was unconditionally ON);\n"
            "    pass --ringRemoval 1.0 to restore the old behaviour\n"
            "  * outputs go to --out, not the current directory\n"
            "  * a right-crop of 0 now means 'no crop' instead of 'empty array'"
        ),
    )
    p.add_argument("-dataFN", "--data", dest="data_fn", type=Path, required=True,
                   help="input HDF5 file (/exchange layout)")
    p.add_argument("-nCPUs", "--ncpus", dest="n_cpus", type=int, default=8,
                   help="OpenMP threads")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: alongside the input)")
    p.add_argument("--shifts", type=float, nargs=3, metavar=("START", "END", "STEP"),
                   default=None,
                   help="sweep the rotation-axis shift instead of using the "
                        "single value stored in the file")
    p.add_argument("--find-shift", dest="find_shift", action="store_true",
                   help="pick the rotation-axis shift automatically from the "
                        "sweep given by --shifts, using two independent "
                        "sharpness criteria. Refuses to report a single value "
                        "when they disagree -- see midas_tomo.center.")
    p.add_argument("--find-shift-slices", type=int, nargs="+", default=None,
                   help="slice indices to score (default: four spread through "
                        "the stack).")
    p.add_argument("--filter", dest="filter_nr", type=int, default=2,
                   choices=[0, 1, 2, 3, 4],
                   help="0 none, 1 Shepp-Logan, 2 Hann, 3 Hamming, 4 ramp")
    p.add_argument("--ringRemoval", type=float, default=0.0,
                   help="ring-removal coefficient; 0 disables it")
    p.add_argument("--extraPad", action="store_true",
                   help="pad to 2x the next power of two")
    p.add_argument("--noLog", action="store_true",
                   help="skip the -log transmission step (use intensities directly)")
    p.add_argument("--deterministic", action="store_true",
                   help="plan FFTs with FFTW_ESTIMATE: reproducible across runs "
                        "and machines, no wisdom file written, slightly slower")
    p.add_argument("--gpu", action="store_true", help="use the CUDA engine if built")
    p.add_argument("--hdf5-out", action="store_true",
                   help="also write the reconstruction as HDF5 with its shape "
                        "and angles attached")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--version", action="version", version=f"midas-tomo {__version__}")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--tuneCleanup", nargs="?", const="__default__", default=None,
                   metavar="GRID.TXT",
                   help="sweep Vo stripe-removal parameters on a thin slab first "
                        "and use the winner; with no value uses the built-in grid")
    g.add_argument("--cleanup", type=float, nargs=3, metavar=("SNR", "LA", "SM"),
                   default=None, help="use this fixed stripe-removal config")
    p.add_argument("--tuningSlices", type=int, default=4,
                   help="mid-stack slices used while tuning")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not backend_c.available(gpu=args.gpu):
        print(backend_c.why_unavailable(gpu=args.gpu), file=sys.stderr)
        return 2

    if args.deterministic and not backend_c.supports_deterministic(gpu=args.gpu):
        print(
            "--deterministic is not supported by the installed binary. It would "
            "be ignored silently, so refusing rather than pretending.",
            file=sys.stderr,
        )
        return 2

    from .api import run_tomo
    from .cleanup import run_tomo_cleanup_sweep
    from .hdf5 import read_exchange, write_recon_hdf5

    if not args.data_fn.is_file():
        print(f"no such file: {args.data_fn}", file=sys.stderr)
        return 2

    out_dir = args.out or args.data_fn.parent / f"{args.data_fn.stem}_recon"
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = read_exchange(args.data_fn)

    # ---- stripe-removal config
    stripe_kw: dict = {}
    if args.cleanup is not None:
        snr, la, sm = float(args.cleanup[0]), int(args.cleanup[1]), int(args.cleanup[2])
        stripe_kw = dict(do_stripe_removal=True, stripe_snr=snr,
                         stripe_la_size=la, stripe_sm_size=sm)
        log.info("stripe removal: fixed snr=%s la=%s sm=%s", snr, la, sm)
    elif args.tuneCleanup is not None:
        half = max(1, args.tuningSlices // 2)
        mid = scan.det_ydim // 2
        z0 = max(0, mid - half)
        z1 = min(scan.det_ydim, z0 + 2 * half)
        slab = read_exchange(args.data_fn, slab=(z0, z1))
        grid = None if args.tuneCleanup == "__default__" else args.tuneCleanup
        res = run_tomo_cleanup_sweep(
            slab.data, slab.dark, slab.whites, out_dir / "cleanup_tuning",
            slab.angles, cleanup_configs=grid, shift=scan.shift,
            tuning_slices=list(range(slab.data.shape[1])), n_cpus=args.n_cpus,
        )
        bc = res["best_config"]
        if bc["snr"] > 0:
            stripe_kw = dict(do_stripe_removal=True, stripe_snr=bc["snr"],
                             stripe_la_size=bc["la"], stripe_sm_size=bc["sm"])
            log.info("stripe removal: tuned snr=%s la=%s sm=%s",
                     bc["snr"], bc["la"], bc["sm"])
        else:
            log.info("stripe removal: tuning preferred the baseline; leaving it off")

    shifts = tuple(args.shifts) if args.shifts is not None else scan.shift

    recon = run_tomo(
        scan.data, scan.dark, scan.whites, out_dir, scan.angles,
        shifts=shifts,
        filter_nr=args.filter_nr,
        do_log=not args.noLog,
        extra_pad=args.extraPad,
        ring_removal=args.ringRemoval,
        n_cpus=args.n_cpus,
        use_gpu=args.gpu,
        deterministic=args.deterministic,
        **stripe_kw,
    )

    if args.find_shift:
        from .center import find_center_consensus

        if args.shifts is None:
            print("--find-shift needs a sweep to choose from; pass --shifts "
                  "START END STEP.", file=sys.stderr)
            return 2
        if recon.shape[0] < 3:
            print(f"--find-shift needs at least 3 candidate shifts, the sweep "
                  f"produced {recon.shape[0]}.", file=sys.stderr)
            return 2

        c = find_center_consensus(recon, tuple(args.shifts),
                                  slices=args.find_shift_slices)
        v = c["per_method"]["variance"]
        tvm = c["per_method"]["tv"]
        print(f"rotation-axis shift, scored on slices {c['slices']}:")
        print(f"  variance        {v['median']:+.3f}  "
              f"(per-slice {[f'{x:+.2f}' for x in v['picks']]}, "
              f"well-determined={v['well_determined']})")
        print(f"  total variation {tvm['median']:+.3f}  "
              f"(per-slice {[f'{x:+.2f}' for x in tvm['picks']]}, "
              f"well-determined={tvm['well_determined']})")
        if c["trustworthy"]:
            print(f"  ==> shift {c['best_shift']:+.3f}   "
                  f"(criteria agree to {c['disagreement']:.3f})")
        else:
            # Not an error: the reconstruction is still written for
            # inspection. But no single number is reported as the answer.
            print(f"  ==> NOT DETERMINED: {c['reason']}", file=sys.stderr)
            print("      The sweep is written out; look at it rather than "
                  "taking a number from a curve whose argmax means nothing.",
                  file=sys.stderr)

    bin_out = out_dir / f"{args.data_fn.stem}_recon_float32.bin"
    recon.astype(np.float32).tofile(bin_out)
    print(f"wrote {bin_out}  shape={recon.shape} (shift, slice, y, x)")

    if args.hdf5_out:
        start, end, step, n = (
            (*args.shifts, 0) if args.shifts is not None
            else (scan.shift, scan.shift, 1.0, 1)
        )
        shift_axis = (np.array([scan.shift]) if args.shifts is None
                      else np.linspace(start, end, recon.shape[0]))
        h5_out = write_recon_hdf5(
            out_dir / f"{args.data_fn.stem}_recon.h5", recon,
            angles=scan.angles, shifts=shift_axis,
            metadata={"source": str(args.data_fn),
                      "filter": args.filter_nr,
                      "ring_removal": args.ringRemoval,
                      "deterministic": args.deterministic},
        )
        print(f"wrote {h5_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
