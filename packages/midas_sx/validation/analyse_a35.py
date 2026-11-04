"""PREREGISTER A35 analysis: python analyse_a35.py > results/a35_analysis.txt"""
import glob
import numpy as np, pandas as pd
pd.set_option("display.width", 200)
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/a35_fit_*.csv"))], ignore_index=True)
DET = ["La2.z", "Ni.z", "O1a.z", "O1b.z", "O2.z", "La1.U", "La2.U", "Ni.U"]
print(f"# A35: P11, seeds {d.seed.min()}..{d.seed.max()} ({d.seed.nunique()}), arm Tr on tables reduced with the estimated shell background and with the oracle background\n")
out = {}
for v in ("est_bg", "oracle_bg0"):
    g = d[(d.variant == v) & d.converged.astype(bool)]
    print(f"## {v}: converged {len(g)}/{int((d.variant == v).sum())}; GooF median {g.goof.median():.3f}; m median {g.m.median():.4f} (sigma_m {g.sigma_m.median():.4f}); rejected rows per seed median {g.n_rej.median():.0f}")
    rows = []
    for p in DET:
        z = (g[f"e:{p}"] / g[f"s:{p}"]).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(param=p, n=len(z), pull_mean=z.mean(), pull_sd=z.std(), mean_err=g[f"e:{p}"].mean(), median_sigma=g[f"s:{p}"].median()))
    t = pd.DataFrame(rows); out[v] = t.set_index("param")
    print(t.round(4).to_string(index=False)); print()
a, b = out["est_bg"], out["oracle_bg0"]
rep = a.loc["La2.U", "pull_mean"] >= 1.0 and a.loc["Ni.U", "pull_mean"] <= -0.5
clean = bool((b.pull_mean.abs() <= 0.5).all() and b.pull_sd.between(0.7, 1.4).all())
refute = b.loc["La2.U", "pull_mean"] >= 1.0
print(f"(a) est_bg reproduces the bias with true U (La2.U >= +1.0 and Ni.U <= -0.5): La2.U {a.loc['La2.U','pull_mean']:+.2f}, Ni.U {a.loc['Ni.U','pull_mean']:+.2f} -> {rep}")
print(f"(b) oracle: all eight |mean pull| <= 0.5 and pull sd in [0.7, 1.4]: {clean}  (La2.U {b.loc['La2.U','pull_mean']:+.2f}, Ni.U {b.loc['Ni.U','pull_mean']:+.2f}; max |mean pull| {b.pull_mean.abs().max():.2f}, sd range {b.pull_sd.min():.2f}-{b.pull_sd.max():.2f})")
print("REGISTERED READING:", "REFUTED (the bias survives a perfect background)" if refute else "CONFIRMED" if (rep and clean) else "INCONCLUSIVE")
