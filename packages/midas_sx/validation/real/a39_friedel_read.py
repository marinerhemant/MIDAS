"""Apply the PREREG_friedel_geometry.md read rules mechanically to friedel_geometry_<run>.json.  python -I a39_friedel_read.py <dir with friedel_geometry_rNN.json> <runs>
sigma used: tx -> bootstrap, wedge -> formal (the choice made after the positive control; see the PREREG read section)."""
import json, os, sys
import numpy as np
d, runs = sys.argv[1], sys.argv[2].split(",")
ok_runs = []
for r in runs:
    j = json.load(open(os.path.join(d, f"friedel_geometry_{r}.json")))[r]
    t = {p: i for i, p in enumerate(j["header start"]["table"]["param"])}
    tab = j["header start"]["table"]
    null = j["null"]
    line, ok = [], True
    for p, sig_key in (("tx", "sigma_boot"), ("wedge", "sigma_formal")):
        i = t[p]; val, fit, sig = tab["value"][i], tab["orientation_fit"][i], tab[sig_key][i]
        diff = val - fit; tol = max(2 * sig, 0.03)
        A = abs(diff) <= tol and abs(diff) <= 0.07
        sig_ok = sig <= (0.05 if p == "tx" else 0.10)
        null_med = float(np.median(null[p]))
        C = abs(null_med - fit) > 0.03                                   # the scrambled-mate estimate must not already be within the tolerance of the fit value
        line.append(f"{p}: pair {val:+.4f} +/- {sig:.4f} ({sig_key[6:]}), fit {fit:+.4f}, diff {diff:+.4f} (tol {tol:.3f}): agree {A}; sigma ok {sig_ok}; null median {null_med:+.3f} -> null does not reproduce {C}")
        ok &= bool(A and sig_ok and C)
        refute = abs(diff) > 3 * sig and abs(diff) > 0.10
        line.append(f"   (refute condition met: {refute})")
    print(f"{r}: " + " | ".join(line), "->", "ALL CONDITIONS MET" if ok else "not all conditions met")
    ok_runs.append(ok)
print(f"\nruns meeting every condition: {sum(ok_runs)} of {len(runs)} (CONFIRMED needs >= 2)")
