"""``midas-dt`` command line entry point.

Replaces ``DT/runDTrecon.py``, which was a 2022 session transcript with
hard-coded paths, two mutually inconsistent methods run unconditionally, and a
call to a binary whose name it got wrong (with the return code ignored).

Differences that change results, stated in ``--help`` rather than buried:

* the branch is **chosen**, not both-run-and-hope
* non-additive outputs use the weighted-moment form by default
* the snake correction is **detected** from the data, not asserted by a flag
* omega is negated once, in one place
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import __version__

# ── MIDAS preflight: richer argument errors when midas-params is installed ───
_MIDAS_DIST = "midas-dt"


def _midas_make_parser(*a, **kw):
    """ArgumentParser factory. Uses midas_params' subclass when available so
    argument errors carry the running version and a did-you-mean; falls back to
    stock argparse otherwise, so this stays an optional dependency."""
    try:
        from midas_params.preflight import MidasArgumentParser
    except Exception:
        return argparse.ArgumentParser(*a, **kw)
    return MidasArgumentParser(*a, package=_MIDAS_DIST, **kw)


log = logging.getLogger("midas_dt")


def _build_parser() -> argparse.ArgumentParser:
    p = _midas_make_parser(
        prog="midas-dt",
        description="XRD-CT reconstruction: raw frames to per-voxel maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Differences from DT/runDTrecon.py:\n"
            "  * you choose ONE branch; it ran both and reported both\n"
            "  * non-additive outputs (RMEAN, Sigma*) use the weighted-moment\n"
            "    form -- back-projecting them directly has no physical meaning\n"
            "  * the snake correction is detected from the data, not asserted\n"
            "  * reconstruction goes through midas-tomo, never skimage.iradon"
        ),
    )
    p.add_argument("--params", type=Path, required=True,
                   help="legacy DT parameter file (ps_dt*.txt) for the geometry")
    p.add_argument("--raw-dir", type=Path, required=True, help="directory of .raw files")
    p.add_argument("--stem", required=True, help="file stem, e.g. dm_dt_pf_U3O8_600A")
    p.add_argument("--start", type=int, required=True, help="first file number")
    p.add_argument("--end", type=int, required=True, help="last file number")
    p.add_argument("--dark", type=Path, default=None, help="dark reference .raw")
    p.add_argument("--out", type=Path, required=True, help="output directory")

    p.add_argument("--branch", choices=["fit-recon", "recon-fit"], default="recon-fit",
                   help="fit-then-reconstruct (cheap) or reconstruct-then-fit (exact)")
    p.add_argument("--weighting", choices=["intensity", "none"], default="intensity",
                   help="fit-recon only: how non-additive outputs are handled")
    p.add_argument("--compare", action="store_true",
                   help="run BOTH branches and report the per-output discrepancy")

    p.add_argument("--r-min", type=float, required=True, help="radius window, px")
    p.add_argument("--r-max", type=float, required=True)
    p.add_argument("--r-bin", type=float, default=0.25)
    p.add_argument("--eta-min", type=float, default=-180.0)
    p.add_argument("--eta-max", type=float, default=180.0)
    p.add_argument("--eta-bin", type=float, default=360.0)
    p.add_argument("--n-peaks", type=int, default=1)

    p.add_argument("--shift", type=float, default=None,
                   help="rotation-axis offset in px; estimated from the data if omitted")
    p.add_argument("--variance-samples", type=int, default=0,
                   help="Monte-Carlo samples for per-voxel sigma (0 = none)")
    p.add_argument("--frames", type=int, default=None,
                   help="use only the first N rotations (for a quick look)")
    p.add_argument("--n-cpus", type=int, default=8)
    p.add_argument("--no-snake-detect", action="store_true",
                   help="skip snake detection and assume unidirectional")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--version", action="version", version=f"midas-dt {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * min(args.verbose, 2),
                        format="%(levelname)s %(name)s: %(message)s")

    from .branches import compare, format_comparison, run_fit_then_recon, run_recon_then_fit
    from .channels import Channel
    from .geometry import geometry_from_legacy_params, parse_legacy_params
    from .io import write_result
    from .reduce import FrameReducer
    from .scan import DTScan, detect_snake
    from .sinogram import assemble

    if not args.params.is_file():
        print(f"no such parameter file: {args.params}", file=sys.stderr)
        return 2

    geo = geometry_from_legacy_params(args.params)
    params = parse_legacy_params(args.params)
    print(f"geometry: {geo.describe()}")

    scan = DTScan.from_stem(
        args.raw_dir, args.stem, args.start, args.end,
        start_omega=params.get("startOme", 180.25),
        omega_step=params.get("omeStep", -0.25),
        dark_file=args.dark,
    )
    print(f"scan: {scan.describe()}")

    channel = Channel(args.r_min, args.r_max, eta_min=args.eta_min,
                      eta_max=args.eta_max, r_bin=args.r_bin,
                      eta_bin=args.eta_bin, n_peaks=args.n_peaks)
    print(f"channel: {channel.describe()}")

    frames = (list(range(min(args.frames, scan.n_frames)))
              if args.frames else list(range(scan.n_frames)))
    reducer = FrameReducer(geo, channel, dark=scan.dark())

    print(f"reducing {scan.n_translations} translations x {len(frames)} frames ...")
    inten, var = [], []
    for t in range(scan.n_translations):
        i, v = reducer.reduce_translation(scan, t, frames=frames)
        inten.append(i)
        var.append(v)
    inten = np.stack(inten)
    var = np.stack(var)

    snake = False
    if not args.no_snake_detect:
        profiles = inten.reshape(inten.shape[0], inten.shape[1], -1).sum(axis=2)
        snake, gain = detect_snake(profiles)
        print(f"snake detection: {'SNAKE' if snake else 'unidirectional'} (gain {gain:.2f})")

    stack = assemble(inten, var, scan.omega_deg[frames], channel, snake=snake)
    print(f"sinograms: {stack.describe()}")

    shift = args.shift
    if shift is None:
        from .center import find_centre
        res = find_centre(stack, method="com", cross_check=False)
        shift = res.shift
        print(f"rotation axis: {res.describe()}")

    args.out.mkdir(parents=True, exist_ok=True)
    recon_kw = dict(shift=shift, n_cpus=args.n_cpus,
                    variance_samples=args.variance_samples)

    if args.compare:
        a = run_fit_then_recon(stack, weighting=args.weighting, **recon_kw)
        b = run_recon_then_fit(stack, **recon_kw)
        stats = compare(a, b)
        text = format_comparison(stats, a, b)
        print("\n" + text)
        (args.out / "branch_comparison.txt").write_text(text + "\n")
        (args.out / "branch_comparison.json").write_text(json.dumps(stats, indent=2) + "\n")
        write_result(a, args.out / "fit_then_recon")
        write_result(b, args.out / "recon_then_fit")
    elif args.branch == "fit-recon":
        r = run_fit_then_recon(stack, weighting=args.weighting, **recon_kw)
        print(r.describe())
        write_result(r, args.out)
    else:
        r = run_recon_then_fit(stack, **recon_kw)
        print(r.describe())
        write_result(r, args.out)

    for w in stack.limits.warnings():
        print(f"NOTE: {w}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
