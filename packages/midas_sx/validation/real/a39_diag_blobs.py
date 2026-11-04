import json, os, sys
import numpy as np, pandas as pd
out = sys.argv[1]
sp = pd.read_csv(os.path.join(out, "spots.csv"))
print(sp.columns.tolist(), len(sp))
sp["win"] = (sp.frame // 100) * 100
print("blobs per 100-frame window:", sp.groupby("win").size().to_dict())
print("median integrated per window:", sp.groupby("win").integrated.median().round(0).to_dict())
top = sp.sort_values("integrated", ascending=False).head(600)
print("share of the strongest 600 per window:", top.groupby("win").size().to_dict())
print(sp.sort_values("integrated", ascending=False).head(12)[["frame", "row", "col", "integrated"]].round(1).to_string())
for a in (600, 700, 800):
    w = sp[(sp.frame >= a) & (sp.frame < a + 50)]
    print(a, "50 frames: n", len(w), "row/col quantiles", np.percentile(w.row, [5, 50, 95]).round(0), np.percentile(w.col, [5, 50, 95]).round(0))
