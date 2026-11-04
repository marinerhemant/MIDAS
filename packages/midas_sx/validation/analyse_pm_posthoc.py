"""POST HOC (after the A26 read; does not change it): is the 'not converged' of A26 only an oxygen-U bound hit?   python analyse_pm_posthoc.py"""
import glob, re
import numpy as np, pandas as pd
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
UNDET = {"O1a.U", "O1b.U", "O2.U", "O3.U"}                  # the parameters outside the determined set of the A26 rule
DET = ["La2.z", "Ni.z", "O1a.z", "O1b.z", "O2.z", "La1.U", "La2.U", "Ni.U"]
def load(v):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"results/pm_{v}_[0-9]*.csv"))], ignore_index=True)
    return d[d.arm.isin(["U", "T", "X"])]
def gate_only(reason):
    if reason == "ok": return True
    m = re.fullmatch(r"on bound: ([A-Za-z0-9.,]+)", reason.split(";")[0].strip()) if ";" not in reason else None
    return bool(m) and set(m.group(1).split(",")) <= UNDET
say("# POST HOC: the A26 convergence gate counts a bound hit of an UNDETERMINED oxygen U (O1a.U, O1b.U, O2.U, O3.U) as non-convergence")
for v in ("P11", "P30", "P00"):
    d = load(v)
    for arm in ("U", "T"):
        g = d[d.arm == arm]
        go = g.reason.map(gate_only)
        say(f"{v} arm {arm}: gate-converged {int(g.converged.sum())}/{len(g)}; converged apart from oxygen-U bound hits {int(go.sum())}/{len(g)}; "
            f"other failures {len(g) - int(go.sum())}: " + "; ".join(sorted(set(g.reason[~go].str.slice(0, 50)))[:3]))
    T = d[d.arm == "T"]; U = d[d.arm == "U"].set_index("seed"); Tg = T[T.reason.map(gate_only)]
    mt = T.m_true.iloc[0]
    if mt > 0:
        pm = (Tg["e:twin.m"] / Tg["s:twin.m"])
        say(f"   {v} arm T on those {len(Tg)} seeds: m median {Tg.m.median():.4f} (IQR {Tg.m.quantile(.25):.4f}-{Tg.m.quantile(.75):.4f}), pull of m mean {pm.mean():+.2f} sd {pm.std(ddof=1):.2f}")
    rows = []
    for p in DET:
        pull = Tg[f"e:{p}"] / Tg[f"s:{p}"]
        rows.append(dict(param=p, pull_mean=pull.mean(), pull_sd=pull.std(ddof=1), U_bias_sig_T=U.loc[Tg.seed, f"e:{p}"].mean() / Tg[f"s:{p}"].median()))
    say(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
open("results/pm_posthoc.txt", "w").write("\n".join(out) + "\n")
