"""Read results/ext_<variant>_*.csv against PREREGISTER A21 / A21a / A21b (dev).   python analyse_ext.py
Criteria are exactly those in PREREGISTER.md A21 (E1 on T1_5 and T1_10, E2 on T2, E3 on T0, informativeness K vs O, REFUTE list)."""
import glob
import numpy as np, pandas as pd
DET = ["La2.z", "Ni.z", "O1.z", "O2.z", "La1.U", "La2.U", "Ni.U", "O1.U"]
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def load(v):
    fs = sorted(glob.glob(f"results/ext_{v}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None

def piv(d, arm, col):
    g = d[d.arm == arm].set_index("seed")
    return g[col]

res = {}
for v in ("T0", "T1_5", "T1_10", "T2"):
    d = load(v)
    if d is None:
        say(f"\n## {v}: no data"); continue
    say(f"\n## {v}: {d.seed.nunique()} seeds")
    E, K, O = (d[d.arm == a].set_index("seed") for a in "EKO")
    for a, g in (("K", K), ("E", E), ("O", O)):
        say(f"   arm {a}: converged {int(g.converged.sum())}/{len(g)}; GooF median {g.goof.median():.2f} (IQR {g.goof.quantile(.25):.2f}-{g.goof.quantile(.75):.2f}); "
            f"poor_fit {int(g.poor_fit.sum())}")
    nc = E[~E.converged]
    if len(nc):
        say("   E not converged: " + "; ".join(f"{r}x{c}" for r, c in nc.reason.str.replace(r"\(.*?\)", "", regex=True).str.slice(0, 50).value_counts().items()))
    ce = E[E.converged]
    rows = []
    for p in DET:
        sO = O[f"s:{p}"].median()
        eE, eK, eO = E[f"e:{p}"], K[f"e:{p}"], O[f"e:{p}"]
        sE = ce[f"s:{p}"]
        pull = ce[f"e:{p}"] / sE
        pe_o = ((E[f"e:{p}"] - eO) / O[f"s:{p}"]).loc[ce.index]
        pe_k = ((E[f"e:{p}"] - eK) / K[f"s:{p}"]).loc[ce.index]
        k_o = ((eK - eO) / O[f"s:{p}"])
        rows.append(dict(param=p, sigma_O=sO, sigma_E_over_K=(ce[f"s:{p}"] / K[f"s:{p}"].loc[ce.index]).median(),
                         E_bias_sig=eE.mean() / sO, E_pull_mean=pull.mean(), E_pull_sd=pull.std(ddof=1),
                         EminusO_mean=pe_o.mean(), EminusO_rms=np.sqrt((pe_o ** 2).mean()),
                         EminusK_rms=np.sqrt((pe_k ** 2).mean()),
                         KminusO_mean=k_o.mean(), KminusO_rms=np.sqrt((k_o ** 2).mean()), K_pull_mean=(eK / K[f"s:{p}"]).mean()))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    say(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    tb = ce.t_b
    say(f"   t_b (E, converged): median {tb.median():.3f} um (truth {E.t_true.iloc[0]}), IQR {tb.quantile(.25):.3f}-{tb.quantile(.75):.3f}; median formal sigma {ce.sigma_t_b.median():.3g}; "
        f"at upper bound (reason mentions ext): {int(E.reason.str.contains('ext.t2').sum())}/{len(E)}")
    cc = ce[[c for c in ce.columns if c.startswith('c:')]].abs()
    say(f"   |corr(t_b, other)|: max over partners per seed, median {cc.max(axis=1).median():.3f}, fraction of seeds > 0.98: {(cc.max(axis=1) > 0.98).mean():.2f}; "
        f"largest median partner: {cc.median().idxmax()} ({cc.median().max():.3f})")
    res[v] = dict(E=E, K=K, O=O, ce=ce, t=t, tb=tb)

say("\n# Verdicts against A21")
inf = []
for v in ("T1_5", "T1_10"):
    if v not in res: continue
    r = res[v]; t = r["t"]; tt = {"T1_5": 5.0, "T1_10": 10.0}[v]
    matter = (t.KminusO_mean.abs() > 1.0).any()
    removes = ((t.EminusO_mean.abs() < 0.5 * t.KminusO_mean.abs()) | (t.KminusO_mean.abs() <= 1.0)).all()
    ok_sd = t.E_pull_sd.between(0.7, 1.4).all()
    ok_m = (t.EminusO_mean.abs() <= 0.3).all(); ok_r = (t.EminusO_rms <= 0.5).all()
    ok_t = abs(r["tb"].median() / tt - 1) <= 0.15
    ok_c = len(r["ce"]) / len(r["E"]) >= 0.9
    n_ub = int(r["E"].reason.str.contains("ext.t2").sum()) / len(r["E"])
    ident = (r["ce"][[c for c in r["ce"].columns if c.startswith('c:')]].abs().max(axis=1) > 0.98).mean()
    say(f"E1 {v}: informative (K vs O shifts a determined parameter by >1 sigma): {matter}; pull sd in [0.7,1.4] {ok_sd}; E-O mean<=0.3 {ok_m}; E-O rms<=0.5 {ok_r}; "
        f"t_b within 15 % {ok_t} (median {r['tb'].median():.2f} vs {tt}); >=90 % converged {ok_c}  => " + ("CONFIRM" if all([ok_sd, ok_m, ok_r, ok_t, ok_c]) else "NOT MET"))
    say(f"   REFUTE checks {v}: K shift >1 sigma and E removes < half: {not removes}; t_b at bound in {100*n_ub:.0f} % (>20 % refutes); |corr|>0.98 in {100*ident:.0f} % of seeds (>50 % refutes)")
if "T2" in res:
    r = res["T2"]; t = r["t"]
    ok1 = (t.E_pull_mean.abs() <= 1.0).all(); ok2 = (t.EminusO_rms <= 1.0).all()
    say(f"E2 T2 (misspecified): |mean pull| <= 1 for all determined {ok1}; E-O rms <= 1 sigma {ok2}; effective t_b median {r['tb'].median():.2f} um  => " + ("CONFIRM" if ok1 and ok2 else "NOT MET"))
    say(f"   informative (K-O mean >1 sigma somewhere): {(t.KminusO_mean.abs() > 1.0).any()}")
if "T0" in res:
    r = res["T0"]; t = r["t"]
    frac_small = (r["E"].t_b <= 1.0).mean()
    infl = t.sigma_E_over_K.max()
    ok = frac_small >= 0.8 and (t.sigma_E_over_K <= 1.5).all() and (t.EminusK_rms <= 0.5).all()
    say(f"E3 T0 (null): t_b <= 1 um in {100*frac_small:.0f} % (>=80 % needed); sigma inflation E/K max {infl:.2f} (<=1.5, refute >2); E-K rms max {t.EminusK_rms.max():.2f} (<=0.5)  => " + ("CONFIRM" if ok else "NOT MET"))
    say(f"   REFUTE check T0: inflation >2 {infl > 2.0}; |E bias| > 1 sigma on a determined parameter {(t.E_bias_sig.abs() > 1.0).any()}")
open("results/ext_analysis.txt", "w").write("\n".join(out) + "\n")
