"""Read the A27 arms against PREREGISTER A27 (dev).   python analyse_a27.py

A0 = results/pm_<v>_*.csv arm T (A26, plain); A1 = results/pmr_pm_merged_<v>_*.csv arm Tr; B0 = results/pmr_pm2_merged_<v>_*.csv arm T; B1 = same, arm Tr
(Ur = untwinned robust, for the null's sigma-inflation). Orientation diagnostics: results/pm2_<v>_*.csv."""
import glob
import numpy as np, pandas as pd
DET = ["La2.z", "Ni.z", "O1a.z", "O1b.z", "O2.z", "La1.U", "La2.U", "Ni.U"]
A26_U1 = {"P11": 0.0491, "P30": 0.0287}                                  # median domain-1 orientation error in A26 (results/pm_analysis.txt)
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def load(pat):
    fs = sorted(glob.glob(pat))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
    return d
def arm(d, a):
    return None if d is None else d[d.arm == a].set_index("seed")
R = {}
for v in ("P11", "P30", "P00"):
    A26 = load(f"results/pm_{v}_[0-9]*.csv"); A26 = A26[A26.arm.isin(["U", "T", "X"])] if A26 is not None else None
    Ar = load(f"results/pmr_pm_merged_{v}_[0-9]*.csv"); Br = load(f"results/pmr_pm2_merged_{v}_[0-9]*.csv")
    O = load(f"results/pm2_{v}_[0-9]*.csv")
    if O is not None and "note" in O.columns:
        O = O[O.note.isna()] if O.note.notna().any() else O
    R[v] = dict(A0=arm(A26, "T"), A1=arm(Ar, "Tr"), B0=arm(Br, "T"), B1=arm(Br, "Tr"), B1U=arm(Br, "Ur"), A1U=arm(Ar, "Ur"), O=O)
def block(name, T, mt, gate_note=""):
    if T is None or not len(T):
        say(f"   {name}: no data"); return None
    c = T[T.converged]
    pm = (c["e:twin.m"] / c["s:twin.m"]) if mt > 0 else None
    row = f"   {name}: converged {int(T.converged.sum())}/{len(T)}; GooF median {T.goof.median():.2f}; m median {T.m.median():.4f} (IQR {T.m.quantile(.25):.4f}-{T.m.quantile(.75):.4f}) sigma_m {T.sigma_m.median():.4f}"
    if pm is not None:
        row += f"; m pull mean {pm.mean():+.2f} sd {pm.std(ddof=1):.2f}"
    if "n_rejected" in T:
        row += f"; rows rejected per seed: median {T.n_rejected.median():.0f}, max {T.n_rejected.max()}"
    say(row + gate_note)
    return c
say("# A27 analysis (PREREGISTER A27). Determined set fixed in A26: " + str(DET))
verd = {}
for v, mt in (("P11", 0.3), ("P30", 0.3), ("P00", 0.0)):
    r = R[v]
    say(f"\n## {v} (m = {mt})")
    if r["O"] is not None and "u1_err_deg" in r["O"].columns:
        O = r["O"]
        say(f"   domain-1 orientation: pair model median {O.u1_err_deg.median():.4f} deg (p90 {O.u1_err_deg.quantile(.9):.4f}, max {O.u1_err_deg.max():.4f}; > 0.10 deg in {100*(O.u1_err_deg>0.10).mean():.0f} %); "
            f"centroid model (the A26 method, same frames) median {O.u1_err_centroid_deg.median():.4f} deg; pair model used (converged) in {100*O.pair_model_used.astype(bool).mean():.0f} % of seeds; fitted blend weight m_orient median {O.m_orient.median():.3f}")
    for nm in ("A0", "A1", "B0", "B1"):
        block({"A0": "A0 (A26 tables, plain T)", "A1": "A1 (A26 tables, robust Tr)", "B0": "B0 (pair-model tables, plain T)", "B1": "B1 (pair-model tables, robust Tr)"}[nm], r[nm], mt)
    T = r["B1"]
    if T is None:
        continue
    c = T[T.converged]
    rows = []
    for p in DET:
        pull = c[f"e:{p}"] / c[f"s:{p}"]
        sU = r["B1U"].loc[c.index, f"s:{p}"] if r["B1U"] is not None else None
        rows.append(dict(param=p, pull_mean=pull.mean(), pull_sd=pull.std(ddof=1), sigT_over_sigU=(c[f"s:{p}"] / sU).median() if sU is not None else np.nan))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    say("   B1 determined parameters (converged seeds):"); say(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    verd[v] = (T, c, t)
say("\n# Verdicts against A27 (primary arm B1)")
for v in ("P11", "P30"):
    if v not in verd: continue
    T, c, t = verd[v]; O = R[v]["O"]
    pm = c["e:twin.m"] / c["s:twin.m"]
    c1 = T.converged.mean() >= 0.9 and len(c) >= 45
    c2 = abs(T.m.median() - 0.30) <= 0.02
    c3 = 0.7 <= pm.std(ddof=1) <= 1.4 and abs(pm.mean()) <= 0.5 and t.pull_sd.between(0.7, 1.4).all() and (t.pull_mean.abs() <= 0.5).all()
    c4 = T.goof.median() <= 1.2
    u = O.u1_err_deg
    c5 = u.median() <= 0.5 * A26_U1[v] and (u <= 0.10).mean() >= 0.98
    c6 = T.n_rejected.median() <= 3
    say(f"E {v}: (1) revised-gate convergence >= 90 % and >= 45 seeds: {c1} ({100*T.converged.mean():.0f} %, {len(c)}); (2) |median m - 0.30| <= 0.02: {c2} ({T.m.median():.4f}); (3) pulls calibrated: {c3} "
        f"(m {pm.mean():+.2f}/{pm.std(ddof=1):.2f}; params mean max {t.pull_mean.abs().max():.2f}, sd range {t.pull_sd.min():.2f}-{t.pull_sd.max():.2f}); (4) GooF median <= 1.2: {c4} ({T.goof.median():.2f}); "
        f"(5) orientation median <= {0.5*A26_U1[v]:.4f} deg and <= 0.10 deg in >= 98 %: {c5} ({u.median():.4f}, {100*(u<=0.10).mean():.0f} %); (6) rejected rows median <= 3: {c6} ({T.n_rejected.median():.0f})  => "
        + ("CONFIRM" if all([c1, c2, c3, c4, c5, c6]) else "NOT MET"))
    ref = []
    if T.converged.mean() < 0.8: ref.append("B1 revised-gate convergence < 80 %")
    if abs(T.m.median() - 0.30) > 0.05: ref.append("median m off by > 0.05")
    if not (0.5 <= pm.std(ddof=1) <= 2): ref.append("m pull sd outside [0.5, 2]")
    if u.median() >= O.u1_err_centroid_deg.median(): ref.append("pair-model orientation not smaller than the centroid model")
    say(f"   REFUTE conditions: {ref or 'none'}")
if "P00" in verd:
    T, c, t = verd["P00"]
    ok_m = (T.m <= 0.05).mean() >= 0.9; ok_i = (t.sigT_over_sigU <= 1.5).all(); ok_hi = (T.m > 0.10).mean() <= 0.10
    say(f"E P00 (null, B1): m <= 0.05 in {100*(T.m<=0.05).mean():.0f} % {ok_m}; sigma inflation Tr/Ur <= 1.5 {ok_i} (max {t.sigT_over_sigU.max():.2f}); m > 0.10 in {100*(T.m>0.10).mean():.0f} % (<= 10 %) {ok_hi}  => "
        + ("CONFIRM" if ok_m and ok_i and ok_hi else "NOT MET"))
open("results/a27_analysis.txt", "w").write("\n".join(out) + "\n")
