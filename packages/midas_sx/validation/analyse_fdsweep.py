"""Read results/fdsweep_*.jsonl against PREREGISTER A28 / A28a (dev).   python analyse_fdsweep.py"""
import glob, json
import numpy as np, pandas as pd
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
rows = [json.loads(l) for f in sorted(glob.glob("results/fdsweep_*.jsonl")) for l in open(f) if l.strip()]
D = pd.DataFrame(rows)
if "error" in D.columns:
    say(f"# rows with an error: {int(D.error.notna().sum())} of {len(D)}: " + "; ".join(sorted(set(D.error.dropna().str.slice(0, 60)))[:3]))
    D = D[D.error.isna()]
order = ["iso-5", "iso-3", "iso-1.5", "iso0", "iso+1.5", "iso+3", "iso+5", "c-3", "c+3", "a-3", "a+3"]
say(f"# A28 analysis: {D.seed.nunique()} seeds x {D.cond.nunique()} conditions")
recs = []
for c in order:
    g = D[D.cond == c]
    if not len(g):
        continue
    r = dict(cond=c, n=len(g))
    for P in ("P1", "P2"):
        ok = np.array([x["ok"] for x in g[P]]); nf = np.array([x["n_found"] for x in g[P]])
        r[f"{P}_success"] = ok.mean(); r[f"{P}_one"] = (nf == 1).mean(); r[f"{P}_spurious"] = (nf > 2).mean(); r[f"{P}_zero"] = (nf == 0).mean()
        folds = [d["fold"] for x in g[P] for d in x["domains"][:2] if x["ok"]]
        r[f"{P}_fold_med"] = float(np.median(folds)) if folds else np.nan
    r["P2_rerun"] = float(g.P2_used_rerun.mean())
    # the cell the tool returns for the largest domain (P1 / P2) relative to the TRUE cell
    c1 = [x["domains"][0]["c"] / 20.5 - 1 for x in g.P1 if x["domains"]]; c2 = [x["domains"][0]["c"] / 20.5 - 1 for x in g.P2 if x["domains"]]
    r["P1_c_rel"] = float(np.median(c1)) if c1 else np.nan; r["P2_c_rel"] = float(np.median(c2)) if c2 else np.nan
    recs.append(r)
T = pd.DataFrame(recs).set_index("cond")
pd.set_option("display.width", 250)
say(T[["n", "P1_success", "P2_success", "P1_one", "P2_one", "P1_zero", "P2_zero", "P1_spurious", "P2_spurious", "P2_rerun", "P1_c_rel", "P2_c_rel", "P1_fold_med", "P2_fold_med"]].to_string(float_format=lambda x: f"{x:.3f}"))
iso = [c for c in order[:7] if c in T.index]; axes = [c for c in order[7:] if c in T.index]
tol = lambda P: max([abs(float(c.replace('iso', '') or 0)) for c in iso if T.loc[c, f"{P}_success"] >= 0.9] or [float('nan')])
say(f"\ntolerance (largest |s| of the isotropic conditions with success >= 90 %, no gaps assumed): P1 {tol('P1')} %, P2 {tol('P2')} %")
ok_iso = all(T.loc[c, "P2_success"] >= 0.9 for c in iso); ok_ax = all(T.loc[c, "P2_success"] >= 0.9 for c in axes)
need = any(T.loc[c, "P1_success"] < 0.9 for c in T.index)
sp0 = T.loc["iso0", "P2_spurious"] if "iso0" in T.index else np.nan
say(f"CONFIRM items: P2 >= 90 % in every isotropic condition: {ok_iso}; in all single-axis conditions: {ok_ax}; P1 < 90 % somewhere (second pass needed): {need}; P2 spurious rate at s = 0 <= 20 %: {sp0 <= 0.2} ({sp0:.2f})")
ref = []
for c in iso:
    if abs(float(c.replace('iso', '') or 0)) <= 3 and T.loc[c, "P2_success"] < 0.5:
        ref.append(f"P2 success {T.loc[c, 'P2_success']:.2f} < 0.5 at {c}")
worse = [c for c in T.index if T.loc[c, "P2_success"] < T.loc[c, "P1_success"]]
if len(worse) > 3:
    ref.append(f"P2 worse than P1 in {len(worse)} conditions: {worse}")
say(f"REFUTE conditions triggered: {ref or 'none'}")
say("OVERALL: " + ("CONFIRM" if (ok_iso and ok_ax and need and sp0 <= 0.2 and not ref) else "NOT MET (see items)"))
open("results/fdsweep_analysis.txt", "w").write("\n".join(out) + "\n")
