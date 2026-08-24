"""``midas-tomo-reconstruct`` — scan record in, registered reconstruction out.

A separate console script rather than a subcommand of ``midas-tomo``: that
command's flat ``-dataFN`` interface is a documented contract used by existing
scripts, and bolting a subcommand onto it would change how its arguments parse.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midas-tomo-reconstruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Reconstruct a 1-ID tomography scan from its own scan record, "
            "finding the rotation-axis shift automatically."
        ),
        epilog=(
            "The scan record is <prefix>_TomoFastScan.dat, normally under\n"
            "  <expt>/metadata/<expt>/<scan>/\n"
            "It carries the pixel size, propagation distance, energy, angles\n"
            "and frame layout. Do NOT take the pixel size from\n"
            "tomocupy_args.yml -- measured wrong for both beamtimes surveyed\n"
            "(it holds a different camera's value).\n"
        ),
    )
    p.add_argument("scan_record", type=Path,
                   help="<prefix>_TomoFastScan.dat")
    p.add_argument("--root", type=Path, required=True,
                   help="local directory holding the scan's image folder "
                        "(the Path: inside the record is the acquisition "
                        "machine's view and is usually not mounted here)")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--ext", default=".tif", help="frame extension (.tif/.tiff)")
    p.add_argument("--crop", type=int, nargs=4, default=None,
                   metavar=("ROW0", "ROW1", "COL0", "COL1"),
                   help="crop every frame. NOT inferred: cropping chooses "
                        "which part of the specimen is reconstructed.")

    g = p.add_argument_group("rotation-axis shift")
    g.add_argument("--coarse", type=float, nargs=3, default=(-25.0, 25.0, 1.0),
                   metavar=("START", "END", "STEP"))
    g.add_argument("--fine-half-width", type=float, default=2.0)
    g.add_argument("--fine-step", type=float, default=0.1)
    g.add_argument("--centre-slab", type=int, default=16,
                   help="detector rows used for the shift sweeps (default 16). "
                        "The sweep cube is n_shifts x n_slices x X x X, so a "
                        "full 2048-row 2320-wide scan would need 111 GB; the "
                        "final reconstruction still uses every row.")
    g.add_argument("--no-strict", action="store_true",
                   help="reconstruct even when centring cannot certify the "
                        "shift. The output is then marked unverified.")

    g = p.add_argument_group("corrections")
    g.add_argument("--delta-beta", type=float, default=0.0,
                   help="Paganin phase retrieval strength. 0 (default) is "
                        "bit-exactly no filtering. It is a strong low-pass "
                        "and sets how large the specimen reconstructs, so "
                        "sweep it rather than picking one.")
    g.add_argument("--stripe", type=float, nargs=3, default=None,
                   metavar=("SNR", "LA", "SM"), help="stripe removal")
    g.add_argument("--filter", dest="filter_nr", type=int, default=2)
    g.add_argument("--measure-tilt", action="store_true",
                   help="also measure the detector roll from the flat field")

    p.add_argument("-nCPUs", "--ncpus", dest="n_cpus", type=int, default=8)
    p.add_argument("--no-hdf5", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--version", action="version",
                   version=f"midas-tomo {__version__}")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.scan_record.is_file():
        print(f"no such scan record: {args.scan_record}", file=sys.stderr)
        return 2
    if not Path(args.root).is_dir():
        print(f"--root is not a directory: {args.root}", file=sys.stderr)
        return 2

    from .workflow import reconstruct_scan

    stripe = None
    if args.stripe:
        stripe = dict(do_stripe_removal=True, stripe_snr=float(args.stripe[0]),
                      stripe_la_size=int(args.stripe[1]),
                      stripe_sm_size=int(args.stripe[2]))

    try:
        res = reconstruct_scan(
            args.scan_record, args.out, root=args.root,
            crop=tuple(args.crop) if args.crop else None, ext=args.ext,
            coarse=tuple(args.coarse), fine_half_width=args.fine_half_width,
            fine_step=args.fine_step, delta_beta=args.delta_beta,
            centre_slab=args.centre_slab,
            measure_tilt=args.measure_tilt, n_cpus=args.n_cpus,
            filter_nr=args.filter_nr, strict=not args.no_strict,
            stripe=stripe, write_hdf5=not args.no_hdf5,
            progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True),
        )
    except RuntimeError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    print("\n" + res.summary(), file=sys.stderr)
    print("\nTo use this as a sample shape:", file=sys.stderr)
    print(res.sample_shape_hint(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
