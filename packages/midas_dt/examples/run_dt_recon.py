"""The 2022 U3O8 workflow, as a runnable example.

Successor to ``DT/runDTrecon.py``. That script was a session transcript: paths
hard-coded to a machine that no longer exists, two mutually inconsistent
methods run unconditionally, and a call to ``DetectorMapper`` when the binary
is named ``DetectorMapperDT`` -- with the return code ignored, so it had been
failing silently.

This does the same science, configurably, and says what it is doing.

Data (verified 2026-08-13; reach: ssh chiltepin -> ssh haydn, both s1iduser):

    raw     /scratch/s1iduser/mpe_nov22_midas2/mpe_nov22/
    600 A   dm_dt_pf_U3O8_600A_000161..000215   (55 translations)
    dark    dark_before_000159.raw
    params  /scratch/s1iduser/DTnewversion/
              ps_dt_u3o8_600A_fileNr_161_215_rad_105_525.txt

The 43-97 range the original script names is NOT on haydn. 161-215 is the same
600 A sample and has 2023 reference output, so this uses that.

Run:

    python examples/run_dt_recon.py --quick        # 40 rotations, one channel
    python examples/run_dt_recon.py --compare      # both branches, full scan
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from midas_dt import (                                    # noqa: E402
    Channel, DTScan, assemble, compare, detect_snake, find_centre,
    format_comparison, geometry_from_legacy_params, parse_legacy_params,
    run_fit_then_recon, run_recon_then_fit, write_result,
)
from midas_dt.reduce import FrameReducer                  # noqa: E402

RAW = Path("/scratch/s1iduser/mpe_nov22_midas2/mpe_nov22")
PARAMS = Path("/scratch/s1iduser/DTnewversion/"
              "ps_dt_u3o8_600A_fileNr_161_215_rad_105_525.txt")
STEM, START, END = "dm_dt_pf_U3O8_600A", 161, 215
DARK = RAW / "dark_before_000159.raw"

# The 2023 runs fitted this window; keeping it makes the results comparable.
CHANNEL = Channel(105.0, 125.0, eta_min=-180.0, eta_max=180.0,
                  r_bin=0.5, eta_bin=360.0, label="U3O8_rad_105_125")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="40 rotations instead of 1440 -- minutes, not hours")
    ap.add_argument("--compare", action="store_true",
                    help="run BOTH branches and report the discrepancy")
    ap.add_argument("--out", type=Path, default=Path("./u3o8_600A_out"))
    ap.add_argument("--n-cpus", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not RAW.is_dir():
        print(f"raw data not found at {RAW}.\nThis example expects the U3O8 "
              f"scan on haydn; see the module docstring.", file=sys.stderr)
        return 2

    geo = geometry_from_legacy_params(PARAMS)
    params = parse_legacy_params(PARAMS)
    print(f"geometry : {geo.describe()}")
    print(f"           energy {geo.energy_kev:.2f} keV  "
          f"(the file's comment says 55.618 keV; it is stale)")

    scan = DTScan.from_stem(RAW, STEM, START, END,
                            start_omega=params["startOme"],
                            omega_step=params["omeStep"], dark_file=DARK)
    print(f"scan     : {scan.describe()}")
    print(f"channel  : {CHANNEL.describe()}")

    n = 40 if args.quick else scan.n_frames
    frames = list(np.linspace(0, scan.n_frames - 1, n).astype(int))
    print(f"\nreducing {scan.n_translations} x {len(frames)} frames ...")

    reducer = FrameReducer(geo, CHANNEL, dark=scan.dark())
    inten, var = [], []
    for t in range(scan.n_translations):
        i, v = reducer.reduce_translation(scan, t, frames=frames)
        inten.append(i)
        var.append(v)
        if t % 10 == 0:
            print(f"  translation {t}/{scan.n_translations}")
    inten, var = np.stack(inten), np.stack(var)

    # The parameter file says BadRotation 1. Detect it anyway -- a flag set by
    # hand, wrong in either direction, gives a plausible image of the wrong
    # object and nothing downstream notices.
    profiles = inten.reshape(inten.shape[0], inten.shape[1], -1).sum(axis=2)
    snake, gain = detect_snake(profiles)
    print(f"\nsnake    : {'DETECTED' if snake else 'not detected'} "
          f"(gain {gain:.2f}); file declares BadRotation="
          f"{params.get('BadRotation', '?')}")

    stack = assemble(inten, var, scan.omega_deg[frames], CHANNEL, snake=snake)
    print(f"sinograms: {stack.describe()}")

    centre = find_centre(stack, method="com", cross_check=False)
    print(f"axis     : {centre.describe()}")

    args.out.mkdir(parents=True, exist_ok=True)
    kw = dict(shift=centre.shift, n_cpus=args.n_cpus)

    if args.compare:
        a = run_fit_then_recon(stack, weighting="intensity", **kw)
        b = run_recon_then_fit(stack, **kw)
        text = format_comparison(compare(a, b), a, b)
        print("\n" + text)
        (args.out / "branch_comparison.txt").write_text(text + "\n")
        write_result(a, args.out / "fit_then_recon")
        write_result(b, args.out / "recon_then_fit")
    else:
        r = run_recon_then_fit(stack, **kw)
        print(f"\n{r.describe()}")
        write_result(r, args.out)

    print(f"\nwrote {args.out}")
    for w in stack.limits.warnings():
        print(f"NOTE: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
