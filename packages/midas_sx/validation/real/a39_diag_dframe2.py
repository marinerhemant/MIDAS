"""Fork A: registered statistics of PREREG_lowangle_dframe.md in FRAME units (frame_meas - frame_calc; omega slope read from the data) + post hoc eta/tth look.  python -I a39_diag_dframe2.py <dframe_table.pkl dir>"""
import os, sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250)
t = pd.read_pickle(os.path.join(sys.argv[1], "dframe_table.pkl"))
slope = np.polyfit(t.frame, t.omega_meas, 1)[0]
print(f"omega_meas vs frame slope {slope:+.5f} deg/frame (intercept check below)")
# omega_meas may be wrapped (mod 360); use local slope check on unwrapped difference
dom = ((t.omega_meas - t.omega + 180) % 360) - 180
t["dfr"] = dom / slope                          # frame_meas - frame_calc
mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
c = t[(t.integrated > 1000) & (t.full_n_sub_peaks == 1) & (t.dfr.abs() < 4)].copy()
print(f"clean {len(c)} of {len(t)}")
for r, g in c.groupby("run"):
    lo = g[g.tth < 20].copy()
    lo["t3"] = pd.qcut(lo.skew_omega.abs(), 3, labels=["low", "mid", "high"])
    X = pd.get_dummies(pd.cut(g.tth, [0, 10, 15, 20, 25, 30, 40, 60]), drop_first=True).astype(float); X["lognf"] = np.log(g.n_frames.clip(1)); X["L"] = g.L; X["skew"] = g.skew_omega; X["const"] = 1.0
    ok = X.notna().all(axis=1); Xo, y = X[ok].to_numpy(float), g.dfr[ok].to_numpy(float)
    b = np.linalg.lstsq(Xo, y, rcond=None)[0]; rs = y - Xo @ b; cov = (rs @ rs / (len(y) - Xo.shape[1])) * np.linalg.inv(Xo.T @ Xo); j = list(X.columns).index("skew")
    print(f"{r}: tth<20 n {len(lo)} median dframe {lo.dfr.median():+.3f} (bootstrap 95% {np.percentile([lo.dfr.sample(len(lo), replace=True, random_state=i).median() for i in range(500)], [2.5, 97.5]).round(2)}); lowest-|skew| tertile median {lo[lo.t3=='low'].dfr.median():+.3f} (n {int((lo.t3=='low').sum())}); skew slope {b[j]:+.3f} t {b[j]/np.sqrt(cov[j,j]):+.1f}")
print("\npooled per run, dframe by tth bin (frames, blob - calc):")
c["tb"] = pd.cut(c.tth, [0, 15, 20, 25, 30, 40, 60]); print(c.pivot_table(index="tb", columns="run", values="dfr", aggfunc="median", observed=True).round(2).to_string())
print("\nPOST HOC: dframe by eta bin (30 deg) per run (n in parentheses via second table):")
c["eb"] = pd.cut(c.eta, np.arange(-180, 181, 30)); print(c.pivot_table(index="eb", columns="run", values="dfr", aggfunc="median", observed=True).round(2).to_string()); print(c.pivot_table(index="eb", columns="run", values="dfr", aggfunc="size", observed=True).to_string())
print("\nPOST HOC: eta<0 vs eta>0 at fixed tth bin"); c["half"] = np.where(c.eta < 0, "eta<0", "eta>0"); print(c.pivot_table(index=["tb", "half"], columns="run", values="dfr", aggfunc="median", observed=True).round(2).to_string())
print("\nPOST HOC: dframe vs row (detector), per run"); c["rb"] = pd.cut(c.row, np.linspace(0, 1679, 8)); print(c.pivot_table(index="rb", columns="run", values="dfr", aggfunc="median", observed=True).round(2).to_string())
print("\nPOST HOC: dframe vs col (detector), per run"); c["cb"] = pd.cut(c.col, np.linspace(0, 1475, 8)); print(c.pivot_table(index="cb", columns="run", values="dfr", aggfunc="median", observed=True).round(2).to_string())
