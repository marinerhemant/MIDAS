"""Fork A POST HOC (not registered): where does the dframe offset live on the detector? Finer column/row bins, pooled runs, module edges.  python -I a39_diag_dframe3.py <dir>"""
import os, sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
t = pd.read_pickle(os.path.join(sys.argv[1], "dframe_table.pkl"))
t["dfr"] = (((t.omega_meas - t.omega + 180) % 360) - 180) / -0.2
c = t[(t.integrated > 1000) & (t.full_n_sub_peaks == 1) & (t.dfr.abs() < 4)].copy()
print("col bins of 60 px, cols 0-900 (pooled runs): n, median dframe, median tth, median eta, median L")
c["cb"] = pd.cut(c.col, np.arange(0, 961, 60))
print(c.groupby("cb", observed=True).agg(n=("dfr", "size"), dfr=("dfr", "median"), tth=("tth", "median"), eta=("eta", "median"), L=("L", "median"), nfr=("n_frames", "median")).round(2).to_string())
print("\nPilatus 2M module column edges: 0-486 | gap 487-493 | 494-980 | gap | 988-1474")
print("\nat fixed 2theta 20-30: col < 487 vs col >= 487 (pooled), and eta sign")
m = c[(c.tth > 20) & (c.tth < 30)].copy(); m["mod"] = np.where(m.col < 487, "col<487", "col>=487"); m["half"] = np.where(m.eta < 0, "eta<0", "eta>0")
print(m.groupby(["mod", "half"]).agg(n=("dfr", "size"), dfr=("dfr", "median"), row=("row", "median"), col=("col", "median")).round(2).to_string())
print("\nrow bins (195+17 px modules) for col<487 only")
x = c[c.col < 487].copy(); x["rb"] = pd.cut(x.row, np.arange(0, 1700, 212)); print(x.groupby("rb", observed=True).agg(n=("dfr", "size"), dfr=("dfr", "median"), tth=("tth", "median")).round(2).to_string())
print("\ncol<487, dframe vs tth bins"); x["tb"] = pd.cut(x.tth, [0, 10, 15, 20, 25, 30, 40, 60]); print(x.groupby("tb", observed=True).agg(n=("dfr", "size"), dfr=("dfr", "median")).round(2).T.to_string())
print("\nrow vs col scatter summary of frame offset: median dfr in 3x3 detector regions (rows thirds x cols 0-487/487-980/980+)")
c["R"] = pd.cut(c.row, [0, 560, 1120, 1700], labels=["rows0-560", "560-1120", "1120+"]); c["C"] = pd.cut(c.col, [0, 487, 980, 1500], labels=["c<487", "487-980", "980+"])
print(c.pivot_table(index="R", columns="C", values="dfr", aggfunc="median", observed=True).round(2).to_string()); print(c.pivot_table(index="R", columns="C", values="dfr", aggfunc="size", observed=True).to_string())
