"""A39-FLOOR steps 3 and 4 as POST HOC reference-free diagnostics on stored observations:
 (3) floor f without observations whose blob peak counts exceed a limit (count-rate losses);
 (4) floor f after a per-run smooth scale vs frame fitted from the equivalents (piecewise-linear log scale, K knots), with a PLACEBO (frame labels permuted within each run, same K).
f = estimator of a39_diag_budget (leave-one-out log deviation, robust, minus counting prediction, / sqrt 2).  python -I a39_diag_scalefit.py <out dir> <cif> <runs> [K=8]"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_hkls import SpaceGroup
from midas_sx.merge import scale_groups
out, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(","); K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
sg = SpaceGroup.from_hm("P 21 21 21")
parts = []
for r in runs:
    o = pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r)
    s = pd.read_csv(os.path.join(out, r, "spots_blind.csv"), usecols=["frame", "row", "col", "peak_counts"])
    o = o.merge(s, on=["frame", "row", "col"], how="left"); parts.append(o)
O = pd.concat(parts, ignore_index=True); print("obs", len(O), "peak_counts missing", int(O.peak_counts.isna().sum()))
O, _ = scale_groups(O, sg); O["u"] = [tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in O[["h", "k", "l"]].to_numpy()]

def f_of(d, label, show_L=False):
    d = d.copy(); g = d.groupby("u"); d["n"] = g.F2.transform("size"); d = d[(d.n >= 2) & (d.F2 > 0)].copy(); s = g.F2.transform("sum").loc[d.index] if False else d.groupby("u").F2.transform("sum")
    d["loo"] = (s - d.F2) / (d.n - 1); d = d[d.loo > 0].copy(); d["dev"] = np.log(d.F2 / d.loo); d["ps"] = (d.sigF2 / d.F2) ** 2 * (1 + 1 / (d.n - 1))
    def fl(x):
        v = (1.4826 * np.median(np.abs(x.dev - np.median(x.dev)))) ** 2; return np.sqrt(max(0.0, v - np.median(x.ps)) / 2)
    res = dict(all=fl(d), r02_03=fl(d[d.run.isin(["r02", "r03"])]), r04=fl(d[d.run == "r04"]), L_lt1p5=fl(d[d.L < 1.5]), L_gt3=fl(d[d.L > 3]), n=len(d))
    print(f"{label:46s} " + "  ".join(f"{k}={v:.3f}" if k != "n" else f"n={v}" for k, v in res.items()), flush=True)
    return res

def smooth_scale(d, frame_col="frame", seed=None, iters=30):
    d = d[(d.F2 > 0)].copy(); d["n"] = d.groupby("u").F2.transform("size"); d = d[d.n >= 2].copy()
    fr = d[frame_col].to_numpy().copy()
    if seed is not None:                                   # placebo: permute the frame labels within each run
        rng = np.random.default_rng(seed)
        for r in runs: m = (d.run == r).to_numpy(); fr[m] = rng.permutation(fr[m])
    lf = np.log(d.F2.to_numpy()); uid = pd.factorize(d.u)[0]; run_id = pd.factorize(d.run)[0]
    # piecewise-linear basis per run, K knots over 0..850
    knots = np.linspace(0, 850, K); B = np.zeros((len(d), K * len(runs)))
    for i in range(len(d)):
        t = np.clip(fr[i], 0, 850); j = min(int(t / (850 / (K - 1))), K - 2); w = (t - knots[j]) / (knots[j + 1] - knots[j]); c = run_id[i] * K
        B[i, c + j] = 1 - w; B[i, c + j + 1] = w
    g = np.zeros(len(d)); coef = np.zeros(B.shape[1])
    for _ in range(iters):
        r = lf - g; lu = pd.Series(r).groupby(uid).transform("mean").to_numpy()      # per-reflection mean of the corrected log
        coef, *_ = np.linalg.lstsq(B, lf - lu, rcond=None); g = B @ coef; g -= g.mean()
    d["F2"] = d.F2 / np.exp(g); d["sigF2"] = d.sigF2 / np.exp(g)
    return d, g

print("\n== baseline"); f_of(O, "baseline (scale_groups only)")
print("\n== step 3: count-rate (post hoc exclusion by blob peak counts)")
for lim in (1e5, 3e4, 1e4):
    f_of(O[~(O.peak_counts > lim)], f"exclude peak_counts > {lim:.0e} ({int((O.peak_counts > lim).sum())} obs)")
print("\n== step 4: smooth per-run scale vs frame (K=%d knots)" % K)
d4, g4 = smooth_scale(O); f_of(d4, "smooth scale fitted"); print("   fitted log-scale range (max-min) per run:", {r: round(float(np.ptp(g4[(d4.run == r).to_numpy()])), 3) for r in runs})
for seed in (1, 2, 3, 4, 5):
    dp, _ = smooth_scale(O, seed=seed); f_of(dp, f"PLACEBO frames permuted (seed {seed})")
print("\n== steps 3+4 combined"); d34, _ = smooth_scale(O[~(O.peak_counts > 3e4)]); f_of(d34, "exclude >3e4 + smooth scale")
