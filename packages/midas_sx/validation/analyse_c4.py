"""Read the C4 forecast (PREREGISTER A5, A9, A10) (dev).   python analyse_c4.py <c_min> [c2 prefix: c2v2|c2]"""
import sys, glob
import numpy as np, pandas as pd
cm = sys.argv[1]
PFX = sys.argv[2] if len(sys.argv) > 2 else "c2v2"
f = pd.read_csv(f"results/c4_forecast_cmin{cm}.csv")
pcols = [c for c in f.columns if c.startswith("s:") and c != "s:scale"]
TOL_X, TOL_U = 0.02, 0.00127
tol = {c: (TOL_U if c.endswith(".U") else TOL_X) for c in pcols}
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
say(f"# C4 forecast, detectability cut c_min = {cm} counts   (C2 comparison prefix: {PFX})")
say("parameters:", ", ".join(c[2:] for c in pcols), "| determined = formal sigma below", f"{TOL_X} A (coordinates) / {TOL_U} A^2 (U_iso)")

# ---- A5 check at the nominal geometry, orientation 1, against C2's measured formal sigma (N3p)
c2 = pd.concat([pd.read_csv(g) for g in sorted(glob.glob(f"results/{PFX}_S2_bilayer_like_*.csv"))], ignore_index=True)
c2 = c2[(c2.start == "N3p") & c2.converged]
nom = f[(f.W == 40) & (f.alpha == 35) & (f.n_int == 1e5)]
o1 = nom[nom.orient == 1].iloc[0]
say(f"\n== A5 check: forecast (orientation 1, nominal) vs C2 {PFX} N3p median formal sigma over {len(c2)} converged seeds; forecast n_unique {int(o1.n_unique)} vs C2 median {c2.n_unique.median():.0f}")
rows = []
for c in pcols:
    emp = c2["s:" + c[2:]].replace(np.inf, np.nan).median()
    rows.append(dict(param=c[2:], forecast=o1[c], c2_formal=emp, ratio=o1[c] / emp, c2_rms_error=np.sqrt((c2["e:" + c[2:]] ** 2).mean()),
                     within30=abs(o1[c] / emp - 1) <= 0.30, determined_c2=bool(emp < tol[c])))
t = pd.DataFrame(rows)
pd.set_option("display.width", 220)
say(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
det = t[t.determined_c2]
agree = bool(det.within30.all())
say(f"   forecast within 30 % of the C2 formal sigma for all {len(det)} determined parameters: {'YES' if agree else 'NO -> C4 INCONCLUSIVE'}")
gf = c2.goof.median()
say(f"   C2 median GooF {gf:.2f}: actual errors exceed formal sigma by about that factor if the reduction has systematics; "
    f"actual RMS error / forecast sigma (determined): " + ", ".join(f"{r.param} {r.c2_rms_error / r.forecast:.2f}" for r in det.itertuples()))

# ---- the coverage table
def block(n_int):
    g = f[f.n_int == n_int]
    rows = []
    for (a, W), h in g.groupby(["alpha", "W"]):
        r = dict(alpha=a, W=W, n_unique=h.n_unique.median(), n_orient=len(h), full_rank=float(h.full_rank.mean()))
        for c in pcols:
            r[c[2:]] = h[c].replace(np.inf, np.nan).median()
        r["n_determined"] = int(sum((h[c].replace(np.inf, np.nan).median() < tol[c]) for c in pcols))
        r["undetermined"] = ",".join(c[2:] for c in pcols if not (h[c].replace(np.inf, np.nan).median() < tol[c])) or "-"
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["alpha", "W"]) if rows else pd.DataFrame()
for n_int in (1e5, 1e4, 1e3):
    say(f"\n== coverage, strongest-reflection integrated counts {n_int:g} (median over {f.orient.nunique()} orientations; formal sigma, A for z, A^2 for U)")
    b = block(n_int)
    if b.empty:
        say('   (no reflection reaches the detectability cut at these counts)'); continue
    say(b.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    # does omega range beyond the opening angle add anything?
    for a in (25.0, 30.0, 35.0, 40.0):
        s = b[b.alpha == a].set_index("W").n_unique
        if len(s) == 3:
            say(f"   alpha={a:.0f}: unique reflections at W=30/45/60: {s[30.0]:.0f}/{s[45.0]:.0f}/{s[60.0]:.0f}")
open(f"results/c4_analysis_cmin{cm}.txt", "w").write("\n".join(out) + "\n")
