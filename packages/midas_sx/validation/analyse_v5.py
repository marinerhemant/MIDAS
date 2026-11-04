"""A20 control: split the paired shift of the refined-U run (v4) from the exact-U run (v3) into  v4 - v5 (effect of the U and geometry estimate)
and  v5 - v3 (spot finding with the wrong-start geometry: pipeline selection noise). Seeds present in all three. N3p start. (dev)"""
import glob, numpy as np, pandas as pd
rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))], ignore_index=True)
g = lambda df: df[df.start == "N3p"].set_index("seed")
v3, v4, v5 = g(rd("results/c2v3_S2_bilayer_like_*.csv")), g(rd("results/c2v4_S2_bilayer_like_*.csv")), g(rd("results/c2v5_S2_bilayer_like_*.csv"))
seeds = sorted(set(v3.index) & set(v4.index) & set(v5.index))
det = ["La2.z", "Ni.z", "O1.z", "O2.z", "La1.U", "La2.U", "Ni.U", "O1.U"]
out = [f"A20 control, {len(seeds)} seeds (N3p, converged in all three): paired shifts in units of the exact-U (v3) formal sigma"]
ok = [s for s in seeds if v3.loc[s, "converged"] and v4.loc[s, "converged"] and v5.loc[s, "converged"]]
rows = []
for p in det:
    sg = v3.loc[ok, "s:" + p]
    a = (v4.loc[ok, "e:" + p] - v5.loc[ok, "e:" + p]) / sg         # effect of the refined U and geometry
    b = (v5.loc[ok, "e:" + p] - v3.loc[ok, "e:" + p]) / sg         # wrong-start spot finding only
    c = (v4.loc[ok, "e:" + p] - v3.loc[ok, "e:" + p]) / sg         # total (the registered quantity)
    rows.append(dict(param=p, rms_v4_minus_v5=np.sqrt((a ** 2).mean()), rms_v5_minus_v3=np.sqrt((b ** 2).mean()), rms_v4_minus_v3=np.sqrt((c ** 2).mean()),
                     max_v4_minus_v5=a.abs().max(), mean_v4_minus_v5=a.mean()))
t = pd.DataFrame(rows)
out.append(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
iso = bool((t.rms_v4_minus_v5 <= 0.5).all())
out.append(f"\nrms(v4 - v5) <= 0.5 sigma for all {len(det)}: {'YES: the refined U and geometry change the answer by less than the registered bound; the unmet A19 criterion is spot-finding / selection noise' if iso else 'NO: the U and geometry estimate itself contributes beyond 0.5 sigma'}")
out.append(f"U error of the refined orientation on these seeds: median {v4.loc[ok, 'u_err_deg'].median():.4f} deg; n_unique v3/v5/v4 medians {v3.loc[ok, 'n_unique'].median():.0f}/{v5.loc[ok, 'n_unique'].median():.0f}/{v4.loc[ok, 'n_unique'].median():.0f}")
print("\n".join(out)); open("results/c2v5_control_analysis.txt", "w").write("\n".join(out) + "\n")
