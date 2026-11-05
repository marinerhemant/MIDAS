#!/usr/bin/env python3
"""Compare a real arm with a null arm: the pre-set statistic N(c) and the descriptive tables.

Usage:  python arm_compare.py <real Grains.csv> <null Grains.csv> <space-group> [out.json]

N(c) = number of DISTINCT orientations among grains with Confidence >= c, by greedy leader clustering (highest
Confidence first) at a misorientation tolerance, using midas_stress.misorientation_om_batch with the sample's space
group (orientation code is not re-implemented here). Reported for c in {0.5..0.9} and tolerances 0.25/0.5/1/2/5 deg
(known-limits KL-070: a clustered count is a count of orientation tiles; the sweep shows how tolerance-stable it is).
The verdict-bearing tolerance is 2 deg. Grains.csv columns (MIDAS): 1:10 = orientation matrix, 13 = lattice a,
19 DiffPos, 20 DiffOme, 21 DiffAngle, 22 GrainRadius, 23 Confidence.
"""
import json
import sys

import numpy as np
from midas_stress.orientation import misorientation_om_batch

GATES = [0.5, 0.6, 0.7, 0.8, 0.9]
TOLS = [0.25, 0.5, 1.0, 2.0, 5.0]


def load(fn):
    g = np.genfromtxt(fn, comments="%")
    return {"OM": g[:, 1:10], "conf": g[:, 23], "dpos": g[:, 19], "dang": g[:, 21], "rad": g[:, 22], "n": len(g)}


def n_distinct(OM, conf, gate, tol_deg, sg):
    sel = np.where(conf >= gate)[0]
    if len(sel) == 0:
        return 0
    order = sel[np.argsort(-conf[sel])]
    left = order.copy()
    n = 0
    while len(left):
        lead = left[0]
        n += 1
        if len(left) == 1:
            break
        m = np.degrees(misorientation_om_batch(np.repeat(OM[lead][None], len(left) - 1, 0), OM[left[1:]], sg))
        left = left[1:][m >= tol_deg]
    return n


def describe(d):
    q = np.percentile(d["conf"], [5, 25, 50, 75, 95]) if d["n"] else []
    return {"n_grains": d["n"], "conf_quantiles_5_25_50_75_95": [float(x) for x in q],
            "conf_bins": {f"[{lo:.1f},{hi:.1f})": int(((d['conf'] >= lo) & (d['conf'] < hi)).sum())
                          for lo, hi in [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]},
            "median_DiffPos_um": float(np.median(d["dpos"])) if d["n"] else None,
            "median_DiffAngle_deg": float(np.median(d["dang"])) if d["n"] else None,
            "median_GrainRadius_um": float(np.median(d["rad"])) if d["n"] else None}


def main():
    real, null, sg = load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3])
    out = {"real": describe(real), "null": describe(null), "N": {}}
    for name, d in (("real", real), ("null", null)):
        out["N"][name] = {f"c>={c}": {f"{t}deg": n_distinct(d["OM"], d["conf"], c, t, sg) for t in TOLS} for c in GATES}
    go = []
    for c in GATES:
        nr, nn = out["N"]["real"][f"c>={c}"]["2.0deg"], out["N"]["null"][f"c>={c}"]["2.0deg"]
        ok = nr >= 10 and nr >= 3 * max(1, nn)
        go.append({"gate": c, "N_real_2deg": nr, "N_null_2deg": nn, "passes": bool(ok)})
    out["go_rule"] = go
    out["GO"] = any(x["passes"] for x in go)
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 4:
        json.dump(out, open(sys.argv[4], "w"), indent=1)


if __name__ == "__main__":
    main()
