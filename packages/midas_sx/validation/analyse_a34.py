"""PREREGISTER A34 analysis: python analyse_a34.py > results/a34_analysis.txt"""
import glob
import numpy as np, pandas as pd
res = {}
print("# A34: blend rows (kind == blend, exp_tot > 0): estimated shell background vs the oracle (signal-only frames, background 0)\n")
for var in ("P11", "P30", "P00"):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/a34_rows_{var}_*.csv")], ignore_index=True)
    b = d[(d.kind == "blend") & (d.exp_tot > 0)]
    print(f"## {var}: {d.seed.nunique()} seeds, {len(b) // 2} blend rows per variant")
    for v in ("est_bg", "oracle_bg0"):
        g = b[b.variant == v]
        w = g[g.exp_tot < 3000]
        h = g[g.exp_tot < 6000]
        a, c = np.polyfit(h.exp_tot, h.I, 1)
        med = float((w.I - w.exp_tot).median())
        per_vox = float(((w.I - w.exp_tot) / w.n_box).median())
        print(f"  {v:11s} exp<3000: n={len(w):5d}  median (I - exp) {med:+8.1f} counts   median ratio {float((w.I / w.exp_tot).median()):.3f}   per voxel {per_vox:+.4f}   median n_box {w.n_box.median():.0f}   fit I = {a:.4f} exp {c:+.1f}")
        res[(var, v)] = (med, a, c)
print("\nReadings against A34:")
e, o = res[("P11", "est_bg")], res[("P11", "oracle_bg0")]
confirm = (abs(o[0]) <= 5 and 0.99 <= o[1] <= 1.01 and all(abs(res[(v, "oracle_bg0")][0]) <= 5 for v in ("P30", "P00")) and e[2] < -50)
refute = o[2] <= -20
print(f"  (a) estimated bg reproduces A29 in P11: b = {e[2]:+.1f} counts (A29: -79)")
print(f"  (b) oracle P11: median offset {o[0]:+.1f}, slope {o[1]:.4f}; P30 {res[('P30','oracle_bg0')][0]:+.1f}, P00 {res[('P00','oracle_bg0')][0]:+.1f}")
print("  REGISTERED READING:", "REFUTED (the cause is elsewhere)" if refute else "CONFIRMED (the shell background estimate causes the P11 blend deficit)" if confirm else "INCONCLUSIVE")
