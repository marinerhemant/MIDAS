"""PREREGISTER A36 T2(b)(c) analysis: python analyse_t2_a36.py > results/a36t2_analysis.txt
Arm Tr of the A27 chain (run_pm2.py + refit_pm.py), held-out seeds 70-89, background mean (baseline, same seeds) vs quadratic.
(b) P11: La2.U mean pull <= +0.6 and Ni.U within +-0.6, m median within 0.03 of 0.30; P30 and P00 |mean pull| not worse by more than 0.3;
(c) the formal sigma of m (median) not inflated by more than 25 %. (a) the blend-row z is read by analyse_t2a_a36.py."""
import numpy as np, pandas as pd
DET = ["La2.z", "Ni.z", "O1a.z", "O1b.z", "O2.z", "La1.U", "La2.U", "Ni.U"]
res = {}
for v in ("P11", "P30", "P00"):
    for arm, tab in (("mean", "pm2_merged"), ("quadratic", "pm2q_merged")):
        d = pd.read_csv(f"results/pmr_{tab}_{v}_70.csv")
        d = d[d.arm == "Tr"]
        res[(v, arm)] = d
    print(f"## {v}: seeds mean {res[(v,'mean')].seed.nunique()} quadratic {res[(v,'quadratic')].seed.nunique()}")
    for arm in ("mean", "quadratic"):
        d = res[(v, arm)]
        c = d[d.converged.astype(bool)]
        print(f"  {arm:9s} converged {len(c)}/{len(d)}; GooF median {d.goof.median():.2f}; m median {d.m.median():.4f} (sigma_m median {d.sigma_m.median():.4f})")
    rows = []
    for p in DET:
        r = {"param": p}
        for arm in ("mean", "quadratic"):
            c = res[(v, arm)]; c = c[c.converged.astype(bool)]
            pull = c[f"e:{p}"] / c[f"s:{p}"]
            r[f"{arm}_pull_mean"], r[f"{arm}_pull_sd"] = pull.mean(), pull.std(ddof=1)
        rows.append(r)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
def pm(v, arm):
    c = res[(v, arm)]; c = c[c.converged.astype(bool)]
    return (c["e:twin.m"] / c["s:twin.m"]) if v != "P00" else None
def pull(v, arm, p):
    c = res[(v, arm)]; c = c[c.converged.astype(bool)]
    return (c[f"e:{p}"] / c[f"s:{p}"]).mean()
b1 = pull("P11", "quadratic", "La2.U") <= 0.6 and abs(pull("P11", "quadratic", "Ni.U")) <= 0.6 and abs(res[("P11", "quadratic")].m.median() - 0.30) <= 0.03
b2 = all(max(abs(pull(v, "quadratic", p)) - abs(pull(v, "mean", p)) for p in DET) <= 0.3 for v in ("P30", "P00"))
c1 = res[("P11", "quadratic")].sigma_m.median() <= 1.25 * res[("P11", "mean")].sigma_m.median()
print(f"\nT2(b) P11: La2.U pull {pull('P11','quadratic','La2.U'):+.2f} (<= +0.6), Ni.U {pull('P11','quadratic','Ni.U'):+.2f} (within 0.6), m median {res[('P11','quadratic')].m.median():.4f} -> {'MET' if b1 else 'NOT MET'}")
print(f"T2(b) P30 / P00 no worse by > 0.3 in |mean pull| -> {'MET' if b2 else 'NOT MET'}")
print(f"T2(c) sigma_m inflation P11: {res[('P11','quadratic')].sigma_m.median() / res[('P11','mean')].sigma_m.median():.3f} (<= 1.25) -> {'MET' if c1 else 'NOT MET'}")
