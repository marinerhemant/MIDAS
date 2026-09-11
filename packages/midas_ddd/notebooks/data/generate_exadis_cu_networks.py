#!/usr/bin/env python
"""Generate the ExaDiS networks that ship with notebook 01 (midas_ddd).

Run wherever ExaDiS is built (``pyexadis`` on PYTHONPATH). It writes ParaDiS
``.data`` files, which ``midas_ddd.read_paradis`` reads with no ExaDiS installed --
that is the file bridge the notebook demonstrates.

Cu, fcc, periodic cubic box of 1000 b (b = 2.556 A, so 255.6 nm on a side):

* loops -- 167 perfect prismatic loops, b = 1/2<110>, radius 15-30 b (3.8-7.7 nm),
  discretised at 4 b. 167 / (255.6 nm)^3 = 1.0e22 m^-3.
* lines -- 24 straight infinite lines, balanced +/- b so they form dipoles
  (ExaDiS's own advice: a multiple of 24 for fcc), discretised at 20 b.

Validation deliberately does NOT happen here: the notebook validates the files
with midas_ddd, which is the point of shipping them.

    PYTHONPATH=<exadis>/python python generate_exadis_cu_networks.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _write(utils, N, path):
    """ExaDiS's generator returns an ExaDisNet, but utils.write_data wants a
    DisNetManager; ExaDisNet carries its own writer, so use whichever exists."""
    if hasattr(N, "get_disnet"):
        utils.write_data(N, path)
    else:
        N.write_data(path)


def _counts(N):
    disnet = N.get_disnet() if hasattr(N, "get_disnet") else N
    return dict(nodes=len(disnet.get_nodes_data()["positions"]),
                segments=len(disnet.get_segs_data()["nodeids"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-b", type=float, default=1000.0)
    ap.add_argument("--loops", type=int, default=167)
    ap.add_argument("--rmin-b", type=float, default=15.0)
    ap.add_argument("--rmax-b", type=float, default=30.0)
    ap.add_argument("--loop-maxseg-b", type=float, default=4.0)
    ap.add_argument("--lines", type=int, default=24)
    ap.add_argument("--line-maxseg-b", type=float, default=20.0)
    ap.add_argument("--theta-deg", type=float, nargs="*", default=None,
                    help="allowed character angles (0 = screw, 90 = edge); ExaDiS's default "
                         "made 24 pure screws, which have no small-angle signal in Cu")
    ap.add_argument("--seed-loops", type=int, default=20260910)
    ap.add_argument("--seed-lines", type=int, default=20260911)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import pyexadis
    import pyexadis_utils as utils

    repo = os.path.dirname(os.path.dirname(os.path.abspath(pyexadis.__file__)))
    git = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip()

    pyexadis.initialize()          # Kokkos: every pyexadis call segfaults without it
    try:
        t0 = time.time()
        loops = utils.generate_prismatic_config(
            "fcc", args.box_b, args.loops, [args.rmin_b, args.rmax_b],
            maxseg=args.loop_maxseg_b, seed=args.seed_loops, uniform=True)
        loops_path = os.path.join(args.out, "exadis_cu_loops.data")
        _write(utils, loops, loops_path)

        lines = utils.generate_line_config(
            "fcc", args.box_b, args.lines, theta=args.theta_deg, maxseg=args.line_maxseg_b,
            seed=args.seed_lines, verbose=False)
        lines_path = os.path.join(args.out, "exadis_cu_lines.data")
        _write(utils, lines, lines_path)

        meta = dict(argv=sys.argv, exadis_repo=repo, exadis_git=git,
                    generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                    seconds=round(time.time() - t0, 2), b_magnitude_A=2.556,
                    box_b=args.box_b, theta_deg=args.theta_deg,
                    loops=dict(file=os.path.basename(loops_path), **_counts(loops)),
                    lines=dict(file=os.path.basename(lines_path), **_counts(lines)))
        with open(os.path.join(args.out, "exadis_cu_networks.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        print(json.dumps(meta, indent=1))
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
