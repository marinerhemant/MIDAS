"""Read results/twin2_<variant>_*.csv against PREREGISTER A24 (dev).   python analyse_twin2.py"""
import glob
import numpy as np, pandas as pd
TOL_X, TOL_U = 0.02, 0.00127
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def load(v):
    fs = sorted(glob.glob(f"results/twin2_{v}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
D = {v: load(v) for v in ("tw30", "tw10", "tw00")}
T0 = D["tw00"]
if T0 is None:
    raise SystemExit("no tw00 data")
A0 = T0[T0.arm == "T"]
pcols = [c[2:] for c in A0.columns if c.startswith("e:") and c != "e:twin.m"]
det = [p for p in pcols if np.nanmedian(A0[f"s:{p}"].replace(np.inf, np.nan)) < (TOL_X if p.split(".")[1] in ("x", "y", "z") else TOL_U)]
say(f"# A24 analysis. Determined set (rule: median formal sigma in arm T on tw00 below 0.02 A / 0.00127 A^2): {det}")
say(f"  undetermined: {[p for p in pcols if p not in det]}")
coords = [p for p in det if p.split('.')[1] in ('x', 'y', 'z')]
res = {}
for v, d in D.items():
    if d is None:
        say(f"\n## {v}: no data"); continue
    mt = d.m_true.iloc[0]
    U, T = d[d.arm == "U"].set_index("seed"), d[d.arm == "T"].set_index("seed")
    say(f"\n## {v} (m = {mt}): {len(T)} seeds")
    for lab, g in (("U", U), ("T", T)):
        say(f"   arm {lab}: converged {int(g.converged.sum())}/{len(g)}; GooF median {g.goof.median():.2f} (IQR {g.goof.quantile(.25):.2f}-{g.goof.quantile(.75):.2f}); poor_fit {int(g.poor_fit.sum())}")
        nc = g[~g.converged]
        if len(nc):
            say("      not converged: " + "; ".join(f"{r}x{c}" for r, c in nc.reason.str.replace(r"\(.*?\)", "", regex=True).str.slice(0, 60).value_counts().items()))
    cT = T[T.converged]
    m_med = T.m.median()
    say(f"   m (arm T, all seeds): median {m_med:.4f} (IQR {T.m.quantile(.25):.4f}-{T.m.quantile(.75):.4f}), mean {T.m.mean():.4f}; median formal sigma_m {T.sigma_m.median():.4f}; m > 2 sigma_m in {100*(T.m > 2*T.sigma_m).mean():.0f} %; m <= 0.05 in {100*(T.m <= 0.05).mean():.0f} %; m > 0.10 in {100*(T.m > 0.10).mean():.0f} %")
    pm = (cT["e:twin.m"] / cT["s:twin.m"]) if "e:twin.m" in cT else None
    if pm is not None and mt > 0:
        say(f"   pull of m: mean {pm.mean():+.2f}, sd {pm.std(ddof=1):.2f}")
    cc = cT[[c for c in cT.columns if c.startswith('c:')]].abs()
    say(f"   |corr(m, other)|: max per seed median {cc.max(axis=1).median():.3f}; fraction of seeds > 0.98 {(cc.max(axis=1) > 0.98).mean():.2f}; largest median partner {cc.median().idxmax()} ({cc.median().max():.3f})")
    rows = []
    for p in det:
        sT = cT[f"s:{p}"]
        pull = cT[f"e:{p}"] / sT
        row = dict(param=p, sigma_T=sT.median(), T_bias_sig=T[f"e:{p}"].mean() / sT.median(), T_pull_mean=pull.mean(), T_pull_sd=pull.std(ddof=1),
                   U_bias_sig_T=U[f"e:{p}"].mean() / sT.median(), sigT_over_sigU=(sT / U.loc[sT.index, f"s:{p}"]).median(),
                   TminusU_rms_sigU=np.sqrt((((T[f"e:{p}"] - U[f"e:{p}"]) / U[f"s:{p}"]) ** 2).mean()))
        rows.append(row)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    say(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    res[v] = dict(T=T, U=U, t=t, cT=cT, mt=mt, pm=pm, cc=cc)

say("\n# Verdicts against A24")
for v in ("tw30", "tw10"):
    if v not in res: continue
    r = res[v]; t, T, U, mt = r["t"], r["T"], r["U"], r["mt"]
    ok_conv = T.converged.mean() >= 0.9
    ok_m = abs(T.m.median() - mt) <= 0.02
    ok_pull = t.T_pull_sd.between(0.7, 1.4).all() and (t.T_pull_mean.abs() <= 0.5).all() and (0.7 <= r["pm"].std(ddof=1) <= 1.4) and abs(r["pm"].mean()) <= 0.5
    ok_goof = T.goof.median() <= 1.2
    info = (U.goof.median() >= 1.5 * T.goof.median()) or (t.U_bias_sig_T.abs() > 1).any()
    say(f"E {v}: converged >= 90 % {ok_conv} ({100*T.converged.mean():.0f} %); |median m - {mt}| <= 0.02: {ok_m} (median {T.m.median():.4f}); pulls (m and determined) calibrated {ok_pull}; GooF median <= 1.2 {ok_goof}; "
        f"informative (U GooF >= 1.5 T or a coordinate/U shift > 1 sigma_T): {info}  => " + ("CONFIRM" if ok_conv and ok_m and ok_pull and ok_goof and info else "NOT MET"))
    ref = []
    if abs(T.m.median() - mt) > 0.05 and v == "tw30": ref.append("median m off by more than 0.05")
    if T.converged.mean() < 0.8: ref.append("T converged < 80 %")
    if not r["pm"].std(ddof=1) < 2 or not r["pm"].std(ddof=1) > 0.5 or (~t.T_pull_sd.between(0.5, 2)).any(): ref.append("pull sd outside [0.5, 2]")
    if (r["cc"].max(axis=1) > 0.98).mean() > 0.5: ref.append("|corr(m, ...)| > 0.98 in more than half the seeds")
    say(f"   REFUTE conditions: {ref or 'none'}")
if "tw00" in res:
    r = res["tw00"]; t, T, U = r["t"], r["T"], r["U"]
    ok_m = (T.m <= 0.05).mean() >= 0.9
    ok_inf = (t.sigT_over_sigU <= 1.5).all()
    exp = np.sqrt(np.maximum(t.sigT_over_sigU ** 2 - 1, 0))
    ok_shift = (t.TminusU_rms_sigU <= 1.2 * exp + 0.1).all()
    say(f"E tw00 (null): m <= 0.05 in {100*(T.m <= 0.05).mean():.0f} % (>= 90 %) {ok_m}; sigma_T/sigma_U <= 1.5 {ok_inf} (max {t.sigT_over_sigU.max():.2f}); T-U paired rms under 1.2 x nested expectation + 0.1 {ok_shift}  => " + ("CONFIRM" if ok_m and ok_inf and ok_shift else "NOT MET"))
    for _, q in t.iterrows():
        say(f"      {q.param}: T-U rms {q.TminusU_rms_sigU:.3f} vs bound {1.2*np.sqrt(max(q.sigT_over_sigU**2-1,0))+0.1:.3f}")
    say(f"   REFUTE: m > 0.10 in {100*(T.m > 0.10).mean():.0f} % of seeds (> 10 % refutes); inflation > 2: {(t.sigT_over_sigU > 2).any()}")
open("results/twin2_analysis.txt", "w").write("\n".join(out) + "\n")
