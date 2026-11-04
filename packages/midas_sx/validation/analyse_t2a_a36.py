"""PREREGISTER A36 T2(a): blend rows on the NOISY frames, exp_tot < 3000 counts: median z = (I - exp)/sigI for background mean vs quadratic (seeds 70-89)."""
import glob
import numpy as np, pandas as pd
ok = True
for v in ("P11", "P30", "P00"):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/a36t2a_rows_{v}_*.csv")], ignore_index=True)
    b = d[(d.kind == "blend") & (d.exp_tot > 0) & (d.exp_tot < 3000)].copy()
    b["z"] = (b.I - b.exp_tot) / b.sigI
    print(f"## {v}: {d.seed.nunique()} seeds")
    for arm in ("mean", "quadratic"):
        g = b[b.frames == arm]
        se = 1.25 / np.sqrt(len(g))
        print(f"  {arm:9s} n={len(g):5d}  median z {g.z.median():+.3f} (se {se:.3f})  median (I - exp) {np.median(g.I - g.exp_tot):+7.2f}  mean sigI {g.sigI.mean():.0f}")
        if arm == "quadratic":
            ok &= abs(g.z.median()) <= 0.05
print("\nT2(a): quadratic median z within +-0.05 in all three variants ->", "MET" if ok else "NOT MET")
