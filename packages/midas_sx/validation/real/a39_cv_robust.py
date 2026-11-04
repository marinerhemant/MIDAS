"""A39-FLOOR: cross-validated R_int and R_split for scale_smooth + robust merge. Scale fitted from a random half of the unique reflections, evaluated on the other half (5 random splits x 2).
python -I a39_cv_robust.py <out dir> <runs>"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_hkls import SpaceGroup
from midas_sx.merge import merge_observations, r_int, scale_groups, scale_smooth
out, runs = sys.argv[1], sys.argv[2].split(",")
sg = SpaceGroup.from_hm("P 21 21 21")
O = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
O["u"] = pd.factorize(pd.Series([tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in O[["h", "k", "l"]].to_numpy()]))[0]
G, _ = scale_groups(O, sg); rng = np.random.default_rng(2027); rows = []
def rs(d, kw):
    A = merge_observations(d[d.run.isin(["r02", "r03"])], sg, **kw).set_index(["h", "k", "l"]); B = merge_observations(d[d.run == "r04"], sg, **kw).set_index(["h", "k", "l"])
    j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.F2_a > 0) & (j.F2_b > 0)]
    return float((j.F2_a - j.F2_b).abs().sum() / (0.5 * (j.F2_a + j.F2_b)).sum())
for sp in range(5):
    half = rng.random(O.u.max() + 1) < 0.5; mA = half[O.u.to_numpy()]
    for tag, fitm in (("fitA->evalB", mA), ("fitB->evalA", ~mA)):
        S, _ = scale_smooth(O, sg, fit_mask=fitm); ev = ~fitm
        rows.append(dict(split=sp, tag=tag, rint_base=r_int(G[ev], sg), rint_smooth=r_int(S[ev], sg), rint_smooth_robust=r_int(S[ev], sg, merge_kwargs=dict(robust=True)),
                         rsplit_base=rs(G[ev], {}), rsplit_smooth_robust=rs(S[ev], dict(robust=True))))
R = pd.DataFrame(rows); print(R.round(4).to_string(index=False)); print("mean:", R.drop(columns=["split", "tag"]).mean().round(4).to_dict())
