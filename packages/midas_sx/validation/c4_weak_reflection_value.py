"""A6 decision input: what would integrating the weak reflections (lower detectability cut) buy? (dev)  python c4_weak_reflection_value.py"""
import sys; sys.argv = ["x", "none"]
import numpy as np, pandas as pd
import forecast_c4 as F
pd.set_option("display.width", 220)
rows = []
for cm in (1138, 500, 250, 120, 60):
    pts = [p for p in (F.run_point(o, 40.0, 35.0, 1e5, cm) for o in range(1, 9)) if p]
    d = pd.DataFrame(pts)
    r = dict(c_min=cm, n_unique=d.n_unique.median())
    for c in [c for c in d.columns if c.startswith("s:") and c != "s:scale"]:
        r[c[2:]] = d[c].replace(np.inf, np.nan).median()
    rows.append(r)
t = pd.DataFrame(rows)
lines = ["Nominal geometry (W=40, alpha=35, n_int=1e5), median over 8 orientations; formal sigma in A (z) / A^2 (U).",
         "Lower c_min = weaker reflections also integrated (what predict-and-integrate would add):", t.to_string(index=False, float_format=lambda v: f"{v:.4g}"),
         "", "ratio of sigma to the c_min=1138 (blob-finder) case:",
         (t.set_index("c_min").drop(columns="n_unique") / t.iloc[0].drop(["c_min", "n_unique"])).round(2).to_string()]
print("\n".join(lines))
open("results/c4_weak_reflection_value.txt", "w").write("\n".join(lines) + "\n")
