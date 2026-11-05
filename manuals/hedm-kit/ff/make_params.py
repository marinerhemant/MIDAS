#!/usr/bin/env python3
"""Build a run's Parameters.txt from a midas-calibrate-v2 --mode ff output plus the sample keys.

Usage:  python make_params.py <ps_calibrated.txt> <out_Parameters.txt> KEY=VALUE [KEY=VALUE ...]

Every KEY given REPLACES all existing lines of that key (a multi-value key such as RingThresh takes
'RingThresh=3:100,5:200' -> one line per ring); a key not in the file is appended.
The calibration writes ResidualCorrectionMap as a path RELATIVE to where the calibration ran: pass it
absolute (ResidualCorrectionMap=/abs/path/residual_corr.bin) or the run cannot find it.

What the calibration output gets wrong for a real run, so you must set it:
  SpaceGroup, LatticeConstant  (they are the calibrant-run placeholders)
  MarginOme   (calibrate-v2's from-scratch value 0.2 is BELOW a 0.25 deg omega step: use >= 2 steps)
  Completeness, RingThresh, OverAllRingToIndex, Dark/darkLoc, Wedge
"""
import os
import sys


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    sets = {}
    for a in sys.argv[3:]:
        k, v = a.split("=", 1)
        sets[k] = v
    lines = open(src).read().splitlines()
    keep, seen = [], set()
    for ln in lines:
        s = ln.split()
        if s and not ln.lstrip().startswith("#") and s[0] in sets:
            continue                                # replaced below
        keep.append(ln)
    keep.append("")
    keep.append("# --- run settings (hedm-kit make_params.py) ---")
    for k, v in sets.items():
        if k == "RingThresh":
            for item in v.split(","):
                r, t = item.split(":")
                keep.append(f"RingThresh {r} {t}")
        else:
            keep.append(f"{k} {v}")
    open(out, "w").write("\n".join(keep) + "\n")
    print(f"wrote {out}: {len(keep)} lines, set {sorted(sets)}")


if __name__ == "__main__":
    main()
