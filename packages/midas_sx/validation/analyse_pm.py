"""Read results/pm_{P11,P30,P00}_*.csv against PREREGISTER A26 (dev).   python analyse_pm.py"""
import glob
import numpy as np, pandas as pd
TOL_X, TOL_U = 0.02, 0.00127
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def load(v):
    fs = sorted(glob.glob(f"results/pm_{v}_[0-9]*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
    if d is not None and "note" in d.columns:
        bad = d[d.note.notna()]
        if len(bad):
            say(f"# {v}: seeds without results: " + "; ".join(f"{int(r.seed)}: {r.note}" for r in bad.itertuples()))
        d = d[d.note.isna()]
    return d
D = {v: load(v) for v in ("P11", "P30", "P00")}
P0 = D["P00"]
if P0 is None:
    raise SystemExit("no P00 data")
T0 = P0[P0.arm == "T"]
pcols = [c[2:] for c in T0.columns if c.startswith("e:") and c != "e:twin.m"]
det = [p for p in pcols if np.nanmedian(T0[f"s:{p}"].replace(np.inf, np.nan)) < (TOL_X if p.split(".")[1] in ("x", "y", "z") else TOL_U)]
say(f"# A26 analysis. Determined set (rule: arm T on P00): {det}")
res = {}
for v, d in D.items():
    if d is None:
        say(f"\n## {v}: no data"); continue
    mt = d.m_true.iloc[0]
    U, T, X = (d[d.arm == a].set_index("seed") for a in "UTX")
    say(f"\n## {v} (b/a = {d.b_over_a.iloc[0]}, m = {mt}): {len(T)} seeds")
    say(f"   targets/obs per seed (median): obs {T.n_obs.median():.0f} = blend {T.n_blend.median():.0f} + c1 {T.n_c1.median():.0f} + c2 {T.n_c2.median():.0f}; median separation of resolved/blended pairs {T.sep_med.median():.2f} px")
    say(f"   domain-1 orientation after refinement: median {T.u1_err_deg.median():.4f} deg, p90 {T.u1_err_deg.quantile(.9):.4f}, max {T.u1_err_deg.max():.4f}; > 0.10 deg in {100*(T.u1_err_deg>0.10).mean():.0f} %; optimiser converged {100*T.o1_converged.astype(bool).mean():.0f} %")
    for lab, g in (("U", U), ("T", T), ("X", X)):
        say(f"   arm {lab}: converged {int(g.converged.sum())}/{len(g)}; GooF median {g.goof.median():.2f} (IQR {g.goof.quantile(.25):.2f}-{g.goof.quantile(.75):.2f})" + (f"; m median {g.m.median():.4f} sigma_m {g.sigma_m.median():.4f}" if lab != "U" else ""))
    cT = T[T.converged]
    pm = cT["e:twin.m"] / cT["s:twin.m"] if mt > 0 else None
    say(f"   arm T m: median {T.m.median():.4f} (IQR {T.m.quantile(.25):.4f}-{T.m.quantile(.75):.4f}); m > 2 sigma_m in {100*(T.m > 2*T.sigma_m).mean():.0f} %; m <= 0.05 in {100*(T.m<=0.05).mean():.0f} %; m > 0.10 in {100*(T.m>0.10).mean():.0f} %" + (f"; pull of m mean {pm.mean():+.2f} sd {pm.std(ddof=1):.2f}" if pm is not None else ""))
    cc = cT[[c for c in cT.columns if c.startswith('c:')]].abs()
    say(f"   |corr(m, other)| max per seed median {cc.max(axis=1).median():.3f}; > 0.98 in {100*(cc.max(axis=1)>0.98).mean():.0f} % of seeds")
    rows = []
    for p in det:
        sT = cT[f"s:{p}"]
        pull = cT[f"e:{p}"] / sT
        sU = U.loc[sT.index, f"s:{p}"]
        rows.append(dict(param=p, sigma_T=sT.median(), T_bias_sig=T[f"e:{p}"].mean() / sT.median(), T_pull_mean=pull.mean(), T_pull_sd=pull.std(ddof=1),
                         U_bias_sig_T=U[f"e:{p}"].mean() / sT.median(), X_bias_sig_T=X[f"e:{p}"].mean() / sT.median(), sigT_over_sigU=(sT / sU).median(),
                         TminusU_rms_sigU=np.sqrt((((T[f"e:{p}"] - U[f"e:{p}"]) / U[f"s:{p}"]) ** 2).mean())))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    say(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    res[v] = dict(T=T, U=U, t=t, pm=pm, cc=cc, mt=mt)
say("\n# Verdicts against A26")
for v in ("P11", "P30"):
    if v not in res: continue
    r = res[v]; T, U, t, pm = r["T"], r["U"], r["t"], r["pm"]
    ok_conv = T.converged.mean() >= 0.9; ok_m = abs(T.m.median() - 0.30) <= 0.02
    ok_pull = t.T_pull_sd.between(0.7, 1.4).all() and (t.T_pull_mean.abs() <= 0.5).all() and 0.7 <= pm.std(ddof=1) <= 1.4 and abs(pm.mean()) <= 0.5
    ok_goof = T.goof.median() <= 1.2
    info = (U.goof.median() >= 1.5 * T.goof.median()) or (t.U_bias_sig_T.abs() > 1).any()
    ok_u = (T.u1_err_deg <= 0.10).mean() >= 0.9
    say(f"E {v}: converged >= 90 % {ok_conv} ({100*T.converged.mean():.0f} %); |median m - 0.30| <= 0.02 {ok_m} ({T.m.median():.4f}); pulls calibrated {ok_pull}; GooF median <= 1.2 {ok_goof}; informative {info}; U1 within 0.10 deg in >= 90 % {ok_u} ({100*(T.u1_err_deg<=0.10).mean():.0f} %)  => " + ("CONFIRM" if all([ok_conv, ok_m, ok_pull, ok_goof, info, ok_u]) else "NOT MET"))
    ref = []
    if abs(T.m.median() - 0.30) > 0.05: ref.append("median m off by > 0.05")
    if T.converged.mean() < 0.8: ref.append("T converged < 80 %")
    if not (0.5 <= pm.std(ddof=1) <= 2) or (~t.T_pull_sd.between(0.5, 2)).any(): ref.append("pull sd outside [0.5, 2]")
    if (T.u1_err_deg > 0.10).mean() > 0.10: ref.append("U1 error > 0.10 deg in > 10 %")
    if (r["cc"].max(axis=1) > 0.98).mean() > 0.5: ref.append("|corr(m, ..)| > 0.98 in more than half the seeds")
    say(f"   REFUTE conditions: {ref or 'none'}")
if "P00" in res:
    r = res["P00"]; T, t = r["T"], r["t"]
    ok_m = (T.m <= 0.05).mean() >= 0.9; ok_inf = (t.sigT_over_sigU <= 1.5).all()
    exp = np.sqrt(np.maximum(t.sigT_over_sigU ** 2 - 1, 0)); ok_sh = (t.TminusU_rms_sigU <= 1.2 * exp + 0.1).all()
    say(f"E P00 (null): m <= 0.05 in {100*(T.m<=0.05).mean():.0f} % {ok_m}; sigma inflation <= 1.5 {ok_inf} (max {t.sigT_over_sigU.max():.2f}); T-U under nested bound {ok_sh}  => " + ("CONFIRM" if ok_m and ok_inf and ok_sh else "NOT MET"))
    ref = []
    if (T.m > 0.10).mean() > 0.10: ref.append("m > 0.10 in > 10 %")
    if (T.u1_err_deg > 0.10).mean() > 0.10: ref.append("U1 error > 0.10 deg in > 10 %")
    say(f"   REFUTE conditions: {ref or 'none'}")
open("results/pm_analysis.txt", "w").write("\n".join(out) + "\n")
