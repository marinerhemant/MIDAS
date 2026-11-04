"""A39-FLOOR step 4 CONFIRMATORY (registered, Amendment 1): smooth per-run scale vs frame (midas_sx.merge.scale_smooth, K = 8), cross-validated over reflections.
For each of 5 random splits of the unique reflections: fit the scale from half A, evaluate f and R_int on half B (and vice versa), against scale_groups on the same half; PLACEBO = frame labels permuted within each run.
Also: tuning merge (runs 02+03) vs hold-out (run 04) f, 3-run R_int, R_split = merge{02,03} vs {04}.   python -I a39_scale_cv.py <out dir> <runs> [K=8]"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_hkls import SpaceGroup
from midas_sx.merge import merge_observations, r_int, scale_groups, scale_smooth
out, runs = sys.argv[1], sys.argv[2].split(","); K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sg = SpaceGroup.from_hm("P 21 21 21")
O = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
O["u"] = pd.factorize(pd.Series([tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in O[["h", "k", "l"]].to_numpy()]))[0]

def f_est(d):
    d = d.copy(); d["n"] = d.groupby("u").F2.transform("size"); d = d[(d.n >= 2) & (d.F2 > 0)].copy(); s = d.groupby("u").F2.transform("sum")
    d["loo"] = (s - d.F2) / (d.n - 1); d = d[d.loo > 0].copy(); d["dev"] = np.log(d.F2 / d.loo); d["ps"] = (d.sigF2 / d.F2) ** 2 * (1 + 1 / (d.n - 1))
    v = (1.4826 * np.median(np.abs(d.dev - np.median(d.dev)))) ** 2; return float(np.sqrt(max(0.0, v - np.median(d.ps)) / 2))

def report(label, d):
    a, b = d[d.run.isin(["r02", "r03"])], d[d.run == "r04"]
    print(f"{label:34s} f_all={f_est(d):.3f}  f_02_03={f_est(a):.3f}  f_04={f_est(b):.3f}  R_int(all)={r_int(d, sg):.4f}  R_int(02+03)={r_int(a, sg):.4f}", flush=True)

def r_split(d):
    A = merge_observations(d[d.run.isin(["r02", "r03"])], sg).set_index(["h", "k", "l"]); B = merge_observations(d[d.run == "r04"], sg).set_index(["h", "k", "l"])
    j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner"); j = j[(j.F2_a > 0) & (j.F2_b > 0)]
    return float((j.F2_a - j.F2_b).abs().sum() / (0.5 * (j.F2_a + j.F2_b)).sum()), len(j)

print("== all reflections used to fit and to evaluate (in-sample, as the post hoc diagnostic)")
G, tg = scale_groups(O, sg); report("scale_groups (baseline)", G); print("   R_split baseline: %.4f (n %d)" % r_split(G))
S, ts = scale_smooth(O, sg, n_knots=K); report(f"scale_smooth K={K}", S); print("   R_split smooth:   %.4f (n %d)" % r_split(S)); print(ts.round(3).to_string(index=False))
print("\n== cross-validated: fit from one random half of the unique reflections, evaluate on the OTHER half (5 splits)")
rng = np.random.default_rng(2026); res = []
for sp in range(5):
    half = rng.random(O.u.max() + 1) < 0.5; mA = half[O.u.to_numpy()]
    for tag, fitm in (("fitA->evalB", mA), ("fitB->evalA", ~mA)):
        S2, _ = scale_smooth(O, sg, n_knots=K, fit_mask=fitm); ev = ~fitm
        base = O[ev]; Gb, _ = scale_groups(O, sg); Gb = Gb[ev]; Sb = S2[ev]
        res.append(dict(split=sp, tag=tag, f_base=f_est(Gb), f_cv=f_est(Sb), rint_base=r_int(Gb, sg), rint_cv=r_int(Sb, sg)))
R = pd.DataFrame(res); print(R.round(4).to_string(index=False)); print("mean:", R[["f_base", "f_cv", "rint_base", "rint_cv"]].mean().round(4).to_dict())
print("\n== placebo (frame labels permuted within each run), in-sample")
for seed in (1, 2, 3):
    P = O.copy(); r2 = np.random.default_rng(seed)
    for r in runs: m = (P.run == r).to_numpy(); P.loc[m, "frame"] = r2.permutation(P.loc[m, "frame"].to_numpy())
    Sp, _ = scale_smooth(P, sg, n_knots=K); report(f"placebo seed {seed}", Sp)
