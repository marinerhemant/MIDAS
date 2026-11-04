"""PREREGISTER A36 T1 analysis: python analyse_t1_a36.py > results/a36t1_analysis.txt
Blend rows (kind == blend, exp_tot > 0), exp_tot < 3000 counts: median (I - exp) and the fit I = a exp + b (exp < 6000), mean vs quadratic, per variant.
CONFIRM (T1): in all three variants, quadratic |median| <= 3 counts, 0.99 <= a <= 1.01, |b| <= 3. REFUTE: P11 |median| > 10 counts. Between: INCONCLUSIVE."""
import glob
import numpy as np, pandas as pd
ok_all, worst = True, 0.0
print("# A36 T1: noise-free (clean) frames, true U and geometry, mean vs quadratic shell background\n")
for var in ("P11", "P30", "P00"):
    fs = glob.glob(f"results/a36t1_rows_{var}_*.csv")
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    b = d[(d.kind == "blend") & (d.exp_tot > 0)]
    print(f"## {var}: {d.seed.nunique()} seeds ({sorted(d.seed.unique())}), {len(b) // 2} blend rows per variant")
    for v in ("mean", "quadratic"):
        g = b[b.variant == v]
        w, h = g[g.exp_tot < 3000], g[g.exp_tot < 6000]
        a, c = np.polyfit(h.exp_tot, h.I, 1)
        med = float((w.I - w.exp_tot).median())
        print(f"  {v:10s} exp<3000: n={len(w):5d}  median (I - exp) {med:+8.2f} counts  median n_box {w.n_box.median():.0f}  fit I = {a:.4f} exp {c:+.2f}")
        if v == "quadratic":
            ok = abs(med) <= 3 and 0.99 <= a <= 1.01 and abs(c) <= 3
            ok_all &= ok
            worst = max(worst, abs(med)) if var == "P11" else worst
            print(f"  -> T1 criteria for {var}: {'MET' if ok else 'NOT MET'}")
print("\nREGISTERED READING (T1):", "CONFIRMED" if ok_all else "REFUTED" if worst > 10 else "INCONCLUSIVE")
