"""Read friedel_sim.py output against PREREG_friedel_geometry.md positive-control criteria.  python -I friedel_sim_read.py <csv>"""
import sys
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); pd.set_option("display.width", 220)
P = ["tx", "ty", "tz", "wedge", "bcz", "bcy", "lsd"]
print("n reps per scenario/kind:"); print(d.groupby(["scenario", "kind"]).size().to_string())
ok_all = True
for (sc, kind), g in d.groupby(["scenario", "kind"]):
    if kind == "scrambled~100":
        continue
    print(f"\n== {sc} | {kind} | n={len(g)} | pairs median {g.n_pairs.median():.0f}")
    rows = []
    for p in P:
        err = g[p] - g["true_" + p]; sd = err.std(ddof=1); sg = g["sig_" + p]
        rows.append(dict(param=p, bias=err.mean(), emp_sd=sd, mean_sigma_formal=sg.mean(), sd_over_sigma=sd / sg.mean(), coverage68=float((err.abs() < sg).mean()), bias_over_sd=err.mean() / sd))
    t = pd.DataFrame(rows).round(4); print(t.to_string(index=False))
    if kind == "pairs~100":
        for p in ("tx", "wedge"):
            r = t[t.param == p].iloc[0]
            ok = abs(r.bias) <= 0.02 and abs(r.bias) <= 0.5 * r.emp_sd and 0.7 <= r.sd_over_sigma <= 1.4 and 0.55 <= r.coverage68 <= 0.80
            print(f"   PASS {p}: {ok}  (|bias| {abs(r.bias):.4f} <= min(0.02, 0.5 sd {0.5*r.emp_sd:.4f}); sd/sigma {r.sd_over_sigma:.2f} in [0.7,1.4]; coverage {r.coverage68:.2f} in [0.55,0.80])")
            ok_all &= bool(ok)
print("\nscrambled-mate null: rms error of the estimate about the TRUTH, scrambled vs true pairs (registered: scrambled >= 3 x true-pair)")
for sc, g in d.groupby("scenario"):
    a, b = g[g.kind == "pairs~100"], g[g.kind == "scrambled~100"]
    out = []
    for p in ("tx", "wedge"):
        ra, rb = float(np.sqrt(np.mean((a[p] - a["true_" + p]) ** 2))), float(np.sqrt(np.mean((b[p] - b["true_" + p]) ** 2)))
        out.append(f"{p}: true-pair rms error {ra:.4f}, scrambled {rb:.3f} ({rb / ra:.0f}x; median scrambled estimate {b[p].median():+.3f}, truth {a['true_' + p].iloc[0]:+.2f})")
    print(sc, "|", " | ".join(out), f"| rms residual scrambled {b.rms.median():.3f} vs {a.rms.median():.4f}")
print("\nPOSITIVE CONTROL (S1 tx and wedge, registered criteria):", "PASS" if ok_all else "FAIL")
