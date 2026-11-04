"""Fork A / PREREG_lowangle_dframe.md: dframe = (omega_meas - omega_calc)/step of reduced observations, per run, against tth, L, omega extent, skew_omega, eta, omega.
python -I a39_diag_dframe.py <out dir> <runs>"""
import os, sys
import numpy as np, pandas as pd
d, runs = sys.argv[1], sys.argv[2].split(",")
STEP = 0.2
pd.set_option("display.width", 250)
mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
tabs = {}
for r in runs:
    o = pd.read_pickle(os.path.join(d, r, "obs.pkl")); s = pd.read_csv(os.path.join(d, r, "spots_blind.csv"))
    # omega_meas is the blob omega; join to the blob by (frame, row, col) nearest-exact
    o["key"] = list(zip(o.frame.round(4), o.row.round(3), o.col.round(3))); s["key"] = list(zip(s.frame.round(4), s.row.round(3), s.col.round(3)))
    m = o.merge(s.drop_duplicates("key")[["key", "integrated", "skew_omega", "kurt_omega", "omega_width_frames", "n_frames", "full_n_sub_peaks", "aspect", "length_px"]], on="key", how="left")
    m["dfr"] = (m.omega_meas - m.omega) / STEP
    # wrap
    m["dfr"] = (m.dfr + 900 / STEP * 0) 
    m["om"] = ((m.omega + 180) % 360) - 180
    tabs[r] = m
    print(f"== {r}: {len(o)} obs, joined {m.integrated.notna().sum()}; dfr median {m.dfr.median():+.3f} mad {mad(m.dfr):.3f}")
    c = m[(m.integrated > 1000) & (m.full_n_sub_peaks == 1) & (m.dfr.abs() < 4)].copy()
    print(f"   clean: {len(c)}")
    c["tb"] = pd.cut(c.tth, [0, 10, 15, 20, 25, 30, 40, 60])
    print(c.groupby("tb", observed=True).agg(n=("dfr", "size"), dfr_med=("dfr", "median"), dfr_mad=("dfr", mad), L=("L", "median"), nfr=("n_frames", "median"), skew=("skew_omega", "median")).round(2).to_string())
    c["Lb"] = pd.cut(c.L, [0, 1.5, 2, 3, 5, 10, 1e3]); print(c.groupby("Lb", observed=True).agg(n=("dfr", "size"), dfr_med=("dfr", "median"), tth=("tth", "median")).round(2).T.to_string())
    c["eb"] = pd.cut(c.eta, np.arange(-180, 181, 60)); print(c.groupby("eb", observed=True).agg(n=("dfr", "size"), dfr_med=("dfr", "median")).round(2).T.to_string())
    lo = c[c.tth < 20].copy()
    if len(lo) > 12:
        lo["t3"] = pd.qcut(lo.skew_omega.abs(), 3, labels=["low|skew|", "mid", "high"])
        print("   tth<20 by |skew_omega| tertile:"); print(lo.groupby("t3", observed=True).agg(n=("dfr", "size"), dfr_med=("dfr", "median"), absskew=("skew_omega", lambda x: x.abs().median())).round(2).to_string())
        lo["sb"] = pd.cut(lo.skew_omega, [-9, -1, -0.3, 0.3, 1, 9]); print("   tth<20 by signed skew_omega:"); print(lo.groupby("sb", observed=True).agg(n=("dfr", "size"), dfr_med=("dfr", "median")).round(2).T.to_string())
    # OLS partial slope on skew controlling tth bin, log nfr, L
    X = pd.get_dummies(c.tb, drop_first=True).astype(float); X["lognf"] = np.log(c.n_frames.clip(1)); X["L"] = c.L; X["skew"] = c.skew_omega; X["const"] = 1.0
    ok = X.notna().all(axis=1) & c.dfr.notna(); Xo, y = X[ok].to_numpy(float), c.dfr[ok].to_numpy(float)
    b, res, *_ = np.linalg.lstsq(Xo, y, rcond=None); rs = y - Xo @ b; s2 = rs @ rs / (len(y) - Xo.shape[1]); cov = s2 * np.linalg.inv(Xo.T @ Xo)
    j = list(X.columns).index("skew"); print(f"   OLS dfr ~ tthbin + log nframes + L + skew_omega: skew slope {b[j]:+.4f} frames/unit skew, t = {b[j]/np.sqrt(cov[j,j]):+.1f}; L slope {b[list(X.columns).index('L')]:+.4f} t = {b[list(X.columns).index('L')]/np.sqrt(cov[list(X.columns).index('L'),list(X.columns).index('L')]):+.1f}")
    # robust-ish: omega dependence
    c["ob"] = pd.cut(c.om, np.arange(-160, 41, 40)); print(c.groupby("ob", observed=True).agg(n=("dfr", "size"), dfr_med=("dfr", "median")).round(2).T.to_string())
pd.concat([t.assign(run=r) for r, t in tabs.items()]).to_pickle(os.path.join(d, "dframe_table.pkl"))
