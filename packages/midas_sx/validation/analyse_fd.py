"""Read results/fd_*.csv and fd_single_*.csv against PREREGISTER A25 (dev).   python analyse_fd.py"""
import glob, json
import numpy as np, pandas as pd
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return pd.DataFrame(rows) if rows else None
D = load("results/fd_[0-9]*.jsonl"); S = load("results/fd_single_[0-9]*.jsonl")
sig = json.load(open("results/fd_sigma_rtn.json"))
say(f"# A25 analysis. sigma_rtn used (calibrated on unscored seeds 100-109): {[round(x, 4) for x in sig]}")

def top2(r):
    """The two found domains with most reflections: (n, best truth id, fold angle, convention) using the better of U / U^T per domain."""
    ds = []
    for k in range(4):
        if f"d{k}_n" in r and pd.notna(r.get(f"d{k}_n")):
            c = {(1, "U"): r[f"d{k}_fold1"], (2, "U"): r[f"d{k}_fold2"], (1, "T"): r[f"d{k}_fold1_T"], (2, "T"): r[f"d{k}_fold2_T"]}
            (tid, conv), ang = min(c.items(), key=lambda kv: kv[1])
            ds.append(dict(k=k, n=r[f"d{k}_n"], truth=tid, conv=conv, fold=ang, a=r[f"d{k}_a"], b=r[f"d{k}_b"], c=r[f"d{k}_c"]))
    ds.sort(key=lambda d: -d["n"])
    return ds[:2], len(ds)

rows = []
for _, r in D.iterrows():
    if "error" in D.columns and pd.notna(r.get("error")):
        rows.append(dict(seed=r.seed, n_found=-1, ok2=False)); continue
    t2, nf = top2(r)
    ok2 = len(t2) == 2 and t2[0]["truth"] != t2[1]["truth"]
    rows.append(dict(seed=r.seed, n_found=int(r.n_found), ok2=ok2, f_big=t2[0]["fold"] if t2 else np.nan, f_small=t2[1]["fold"] if len(t2) > 1 else np.nan,
                     n_big=t2[0]["n"] if t2 else np.nan, n_small=t2[1]["n"] if len(t2) > 1 else np.nan,
                     both_within_1deg=ok2 and all(d["fold"] <= 1.0 for d in t2), any_over_2deg=any(d["fold"] > 2.0 for d in t2),
                     cell_ok=ok2 and all(abs(d["a"] / 3.83 - 1) <= 0.01 and abs(d["c"] / 20.5 - 1) <= 0.01 for d in t2),
                     conv="".join(d["conv"] for d in t2), a_over_b=[d["b"] / d["a"] for d in t2] if t2 else [],
                     c_rel=[d["c"] / 20.5 - 1 for d in t2] if t2 else []))
R = pd.DataFrame(rows)
n = len(R)
say(f"\n## two-domain frames: {n} seeds")
say("domains found per seed: " + ", ".join(f"{k}: {v}" for k, v in R.n_found.value_counts().sort_index().items()))
say(f"two largest found domains match DIFFERENT truth domains: {100*R.ok2.mean():.0f} %")
say(f"both within 1 deg (after D4 folding): {100*R.both_within_1deg.mean():.0f} %; any matched domain beyond 2 deg: {100*R.any_over_2deg.mean():.0f} %")
say(f"fold angle, largest domain: median {R.f_big.median():.3f} deg (p90 {R.f_big.quantile(.9):.3f}); second: median {R.f_small.median():.3f} deg (p90 {R.f_small.quantile(.9):.3f})")
say(f"reflections claimed: largest {R.n_big.median():.0f}, second {R.n_small.median():.0f} (median)")
say(f"cell a and c both within 1 % of 3.83 / 20.5 for both domains: {100*R.cell_ok.mean():.0f} % (c relative error: " + ", ".join(f"{np.median([x for c in R.c_rel for x in c]):+.4f} median" for _ in [0]) + f", range {min(x for c in R.c_rel for x in c):+.4f}..{max(x for c in R.c_rel for x in c):+.4f}); b/a median {np.median([x for c in R.a_over_b for x in c]):.4f}")
conv_U = (R.conv.str.count("U") == 2).mean() if len(R) else np.nan
say(f"convention: both domains' U (not U^T) fold best in {100*conv_U:.0f} % of seeds")
# C3: refinement from the found U
ref = []
for _, r in D.iterrows():
    if pd.notna(r.get("o1_err_vs_1")) and pd.notna(r.get("o2_err_vs_1")):
        e1 = min(r.o1_err_vs_1, r.o1_err_vs_2); e2 = min(r.o2_err_vs_1, r.o2_err_vs_2)
        diff = (r.o1_err_vs_1 < r.o1_err_vs_2) != (r.o2_err_vs_1 < r.o2_err_vs_2)
        ref.append(dict(e1=e1, e2=e2, diff=diff, conv=bool(r.o1_converged) and bool(r.o2_converged)))
Rf = pd.DataFrame(ref)
if len(Rf):
    ok = (Rf.e1 <= 0.05) & (Rf.e2 <= 0.05) & Rf["diff"]
    say(f"after index_and_refine from the found U: domain 1 error median {Rf.e1.median():.4f} deg, domain 2 {Rf.e2.median():.4f} deg; both <= 0.05 deg and different truth domains in {100*ok.sum()/n:.0f} % of all seeds ({100*ok.mean():.0f} % of those refined, {len(Rf)} refined)")
    c3 = ok.sum() / n
else:
    c3 = 0.0; say("no refinement results")
c1 = R.both_within_1deg.mean()
say(f"\nC1 (both domains found, each within 1 deg, >= 90 %): {100*c1:.0f} %  => {'MET' if c1 >= 0.9 else 'NOT MET'}")
c2 = R.cell_ok.mean()
say(f"C2 (cell within 1 %): {100*c2:.0f} %  => {'MET (all of the C1 seeds)' if c2 >= 0.9 else 'NOT MET'}  [the start geometry has a +-1 % distance error, which scales the recovered cell]")
say(f"C3 (<= 0.05 deg after refinement, >= 90 %): {100*c3:.0f} %  => {'MET' if c3 >= 0.9 else 'NOT MET'}")
if S is not None:
    ns = S.n_found.value_counts().sort_index()
    one = (S.n_found == 1).mean()
    say(f"\n## single-domain frames (null): {len(S)} seeds; domains found: " + ", ".join(f"{k}: {v}" for k, v in ns.items()))
    c4 = one
    say(f"C4 exactly one domain in {100*one:.0f} % (>= 90 % needed)  => {'MET' if c4 >= 0.9 else 'NOT MET'}")
    say(f"   the domain found in the single-domain frames lies within 1 deg of the truth in {100*(S[['d0_fold1', 'd0_fold1_T']].min(axis=1) <= 1.0).mean():.0f} %")
else:
    c4 = np.nan; say("no single-domain data")
ref_c = []
if (R.n_found >= 2).mean() < 0.8: ref_c.append("< 80 % of two-domain seeds find two domains")
if R.any_over_2deg.mean() > 0.10: ref_c.append("a matched domain beyond 2 deg in > 10 %")
if S is not None and (S.n_found >= 2).mean() > 0.20: ref_c.append("null returns a second domain in > 20 %")
say(f"\nREFUTE conditions triggered: {ref_c or 'none'}")
say("OVERALL: " + ("CONFIRM" if (c1 >= 0.9 and c2 >= 0.9 and c3 >= 0.9 and c4 >= 0.9 and not ref_c) else "NOT MET (see items)"))
open("results/fd_analysis.txt", "w").write("\n".join(out) + "\n")
