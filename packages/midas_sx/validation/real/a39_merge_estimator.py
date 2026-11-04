"""A39-FLOOR step 1b: old vs robust merge estimator, reference-free (no COD, no refinement).
Metrics: R_split = merge {A runs} vs merge {B run} on the common unique reflections, sum|Fa-Fb| / sum (Fa+Fb)/2 (registered split A = {r02, r03}, B = {r04}; the two other splits are shown too);
R_int (package definition, mean from the same estimator; also over the retained observations only); rejected fraction; the named cases (0 0 8) and (1 0 2).
python -I a39_merge_estimator.py <out dir holding r02 r03 r04> [runs=r02,r03,r04]"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_hkls import SpaceGroup
from midas_sx.merge import estimate_error_model_g, merge_observations, r_int, scale_groups
out = sys.argv[1]; runs = (sys.argv[2] if len(sys.argv) > 2 else "r02,r03,r04").split(",")
sg = SpaceGroup.from_hm("P 21 21 21")
O = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
O, tab = scale_groups(O, sg); print("run scales", dict(zip(tab.group, tab.scale.round(3))), " observations", len(O))
g = estimate_error_model_g(O, sg); print(f"estimated error-model g = {g:.4f}")
KW = dict(robust=True, error_g=g)

def keyed(m): return m.set_index(["h", "k", "l"])
def rsplit(A, B, kw):
    a, b = keyed(merge_observations(A, sg, **kw)), keyed(merge_observations(B, sg, **kw)); c = a.index.intersection(b.index)
    fa, fb = a.loc[c, "F2"].to_numpy(), b.loc[c, "F2"].to_numpy(); return float(np.abs(fa - fb).sum() / ((fa + fb) / 2).sum()), len(c)
def rint_retained(obs, kw):
    """R_int over the observations the robust estimator RETAINS (mean from the same estimator)."""
    from midas_sx.merge import _robust_group
    m = merge_observations(obs, sg, **kw); mean = {(r.h, r.k, r.l): r.F2 for r in m.itertuples()}
    o = obs.copy(); o["_u"] = [tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in o[["h", "k", "l"]].to_numpy()]
    num = den = 0.0
    for u, d in o.groupby("_u"):
        x = d.F2.to_numpy(float); s_ = np.maximum(d.sigF2.to_numpy(float), 1e-12)
        keep, _ = _robust_group(x, s_, kw["error_g"], kw.get("reject_sigma", 4.0)) if kw.get("robust") else (np.ones(len(x), bool), None)
        if keep.sum() < 2: continue
        num += np.abs(x[keep] - mean[u]).sum(); den += np.abs(x[keep]).sum()
    return num / den
print("\n%-34s %10s %10s %10s" % ("split (A | B)", "old", "robust", "n common"))
for held in runs[::-1]:
    A, B = O[O.run != held], O[O.run == held]
    (so, n1), (sn, n2) = rsplit(A, B, {}), rsplit(A, B, KW)
    tag = " <- registered" if held == "r04" else ""
    print("%-34s %9.2f%% %9.2f%% %10d%s" % (f"{'+'.join(sorted(set(A.run)))} | {held}", 100 * so, 100 * sn, n1, tag))
print("\nablation on the registered split (A = all runs but r04, B = r04):")
A, B = O[O.run != "r04"], O[O.run == "r04"]
for lab, kw in (("old", {}), ("weights only (error model, no rejection)", dict(robust=True, error_g=g, reject_sigma=1e9)), ("rejection only (g = 0)", dict(robust=True, error_g=0.0)), ("both (registered robust)", KW)):
    print(f"  {lab:44s} R_split {100 * rsplit(A, B, kw)[0]:.2f}%")
rng = np.random.default_rng(0); a, b = keyed(merge_observations(A, sg)), keyed(merge_observations(B, sg)); a2, b2 = keyed(merge_observations(A, sg, **KW)), keyed(merge_observations(B, sg, **KW)); c = a.index.intersection(b.index)
fa, fb, ga, gb = (v.loc[c, "F2"].to_numpy() for v in (a, b, a2, b2)); diffs = []
for _ in range(2000):
    i = rng.integers(0, len(c), len(c)); diffs.append(np.abs(fa[i] - fb[i]).sum() / ((fa[i] + fb[i]) / 2).sum() - np.abs(ga[i] - gb[i]).sum() / ((ga[i] + gb[i]) / 2).sum())
print(f"  paired bootstrap (2000, reflections resampled) of R_split(old) - R_split(robust): mean {100 * np.mean(diffs):.2f} points, 95% interval [{100 * np.percentile(diffs, 2.5):.2f}, {100 * np.percentile(diffs, 97.5):.2f}]")
mo, mn = merge_observations(O, sg), merge_observations(O, sg, **KW)
print(f"\nmerged unique reflections: old {len(mo)} (n>=2: {(mo.n >= 2).sum()}), robust {len(mn)} (n>=2: {(mn.n >= 2).sum()}); observations rejected: {int(mn.n_rej.sum())} of {len(O)} ({100 * mn.n_rej.sum() / len(O):.2f}%), reflections with a rejection: {(mn.n_rej > 0).sum()}")
print("R_int (package definition, scaled):  old %.2f%%   robust (rejected obs still counted against the mean) %.2f%%   robust (retained obs only) %.2f%%" % (
    100 * r_int(O, sg), 100 * r_int(O, sg, merge_kwargs=KW), 100 * rint_retained(O, KW)))
print("\nnamed cases (merged over all runs):")
for hk in [(0, 0, 8), (1, 0, 2)]:
    u = tuple(sg.asu_representative(*hk)); sel = [i for i, r in enumerate(O.itertuples()) if tuple(sg.asu_representative(int(r.h), int(r.k), int(r.l))) == u]
    d = O.iloc[sel]; print(f"  {hk} -> {u}: obs F2 {np.round(d.F2.to_numpy(), 1).tolist()} sig {np.round(d.sigF2.to_numpy(), 1).tolist()}")
    for lab, m in (("old", mo), ("robust", mn)):
        r = m[(m.h == u[0]) & (m.k == u[1]) & (m.l == u[2])].iloc[0]; print(f"     {lab:7s} n {int(r.n)}  F2 {r.F2:9.1f}  sigF2 {r.sigF2:8.1f}" + (f"  rejected {int(r.n_rej)}" if lab == "robust" else ""))
# rejected observations: what are they
rej = []
O["_u"] = [tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in O[["h", "k", "l"]].to_numpy()]
for u, d in O.groupby("_u"):
    if len(d) < 3: continue
    x0 = d.F2.to_numpy(float); s0 = np.maximum(d.sigF2.to_numpy(float), 1e-12)
    from midas_sx.merge import _robust_group
    keep, sp = _robust_group(x0, s0, g, 4.0)
    for i in np.flatnonzero(~keep): rej.append(dict(u=u, run=d.run.iloc[i], F2=x0[i], sig=s0[i], L=d.L.iloc[i], tth=d.tth.iloc[i], others_median=float(np.median(np.delete(x0, i)))))
R = pd.DataFrame(rej); print(f"\nrejected observations: {len(R)}; median L {R.L.median():.2f} (all obs {O.L.median():.2f}); median tth {R.tth.median():.1f} (all {O.tth.median():.1f}); F2 < 0.2 x others' median: {(R.F2 < 0.2 * R.others_median).mean():.2f}; by run: {R.run.value_counts().to_dict()}")
