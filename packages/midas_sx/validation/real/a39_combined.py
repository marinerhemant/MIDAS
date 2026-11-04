"""A39-FLOOR combination (reference-free): scale_groups | scale_smooth, then merge_observations default | robust. R_int and R_split ({02,03} vs {04}, {02,04} vs {03}, {03,04} vs {02}), f.
python -I a39_combined.py <out dir> <runs>"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_hkls import SpaceGroup
from midas_sx.merge import merge_observations, r_int, scale_groups, scale_smooth
out, runs = sys.argv[1], sys.argv[2].split(",")
sg = SpaceGroup.from_hm("P 21 21 21")
O = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
def rs(d, a, b, kw):
    A = merge_observations(d[d.run.isin(a)], sg, **kw).set_index(["h", "k", "l"]); B = merge_observations(d[d.run.isin(b)], sg, **kw).set_index(["h", "k", "l"])
    j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.F2_a > 0) & (j.F2_b > 0)]
    return float((j.F2_a - j.F2_b).abs().sum() / (0.5 * (j.F2_a + j.F2_b)).sum()), len(j)
G, _ = scale_groups(O, sg); S, _ = scale_smooth(O, sg)
for sname, D in (("scale_groups", G), ("scale_smooth", S)):
    for mname, kw in (("default merge", {}), ("robust merge", dict(robust=True))):
        splits = [rs(D, ["r02", "r03"], ["r04"], kw), rs(D, ["r02", "r04"], ["r03"], kw), rs(D, ["r03", "r04"], ["r02"], kw)]
        print(f"{sname:13s} + {mname:13s} R_int={r_int(D, sg, merge_kwargs=kw) if kw else r_int(D, sg):.4f}  R_split(02+03|04)={splits[0][0]:.4f}  (02+04|03)={splits[1][0]:.4f}  (03+04|02)={splits[2][0]:.4f}  n_common={splits[0][1]}", flush=True)
M = merge_observations(S, sg, robust=True); M.to_csv(os.path.join(out, "merged_smooth_robust.csv"), index=False); print("wrote", os.path.join(out, "merged_smooth_robust.csv"), len(M), "unique;", int((M.n >= 2).sum()), "n>=2")
