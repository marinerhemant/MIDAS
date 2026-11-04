"""PREREGISTER A32 analysis: python analyse_u_precision.py > results/uprec_analysis.txt"""
import glob
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/uprec_*.csv"))], ignore_index=True)
params = [c[2:] for c in d.columns if c.startswith("x:")]
tol = {p: (0.00127 if p.endswith(".U") else 0.02) for p in params}
d0 = d[d.theta == 0.0].set_index("seed")
print(f"# A32 analysis: {d.seed.nunique()} seeds, levels {sorted(d.theta.unique())}; converged at theta=0: {int(d0.converged.sum())}/{len(d0)}; n_shared median {int(d0.n_shared.median())}")
det = [p for p in params if np.nanmedian(d0[f"s:{p}"]) < tol[p]]
print("determined set (median formal sigma at theta=0 below 0.02 A / 0.00127 A^2):", det, "| not determined:", [p for p in params if p not in det])
C3_01 = {"La2.z": 1.38e-3, "Ni.z": 6.16e-4, "O1.z": -5.57e-3, "O2.z": -6.55e-3, "La1.U": 2.2e-4, "La2.U": -2.1e-4, "Ni.U": 2.69e-4, "O1.U": 1.68e-3, "O2.U": -3.75e-4, "O3.U": 6.55e-3}   # paired map, 0.1 deg about (1,1,1)
rows = []
for th in (0.01, 0.02, 0.05):
    g = d[d.theta == th].set_index("seed")
    ok = g.converged & d0.converged.reindex(g.index)
    for p in params:
        shift = (g[f"x:{p}"] - d0[f"x:{p}"].reindex(g.index))[ok]
        z = (shift / d0[f"s:{p}"].reindex(g.index)[ok]).abs()
        rows.append(dict(theta=th, param=p, det=p in det, n=int(ok.sum()), med_abs_shift=float(shift.abs().median()), med_z=float(z.median()), p90_z=float(z.quantile(0.9))))
r = pd.DataFrame(rows)
for th in (0.01, 0.02, 0.05):
    print(f"\n== theta = {th} deg  (|paired shift| / formal sigma at theta = 0)")
    print(r[r.theta == th].drop(columns="theta").round(3).to_string(index=False))
print("\nReadings against A32:")
g = r[(r.theta == 0.02) & r.det]
conf1 = bool((g.med_z <= 0.5).all() and (g.p90_z <= 1.0).all())
refute = bool((g.med_z > 1.0).any())
print(f"theta = 0.02: determined parameters with median |z| <= 0.5 and p90 <= 1.0: {int(((g.med_z <= 0.5) & (g.p90_z <= 1.0)).sum())}/{len(g)}; max median |z| {g.med_z.max():.2f} ({g.loc[g.med_z.idxmax(), 'param']}); max p90 {g.p90_z.max():.2f}")
lin = []
for p, v in C3_01.items():
    a = r[(r.theta == 0.05) & (r.param == p)].med_abs_shift.iloc[0]
    lin.append((p, a, 0.5 * abs(v)))
print("linearity (theta = 0.05 median |shift| vs 0.5 x the 0.1-deg C3 value, axis (1,1,1) only so a rough check):")
for p, a, e in lin:
    print(f"   {p:6s} measured {a:.2e}  predicted {e:.2e}  ratio {a / e if e else float('nan'):.2f}")
big = max(lin, key=lambda t: t[2])
lin_ok = 0.5 <= big[1] / big[2] <= 2.0
print(f"most sensitive in C3 ({big[0]}): ratio {big[1] / big[2]:.2f} -> {'within a factor 2' if lin_ok else 'NOT within a factor 2'}")
print("\nREGISTERED READING:", "REFUTED (joint refinement needed at 0.02 deg)" if refute else ("CONFIRMED" if (conf1 and lin_ok) else "INCONCLUSIVE"))
