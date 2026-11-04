"""PREREGISTER A33 analysis: python analyse_joint.py > results/joint_analysis.txt"""
import glob
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/joint_*.csv"))], ignore_index=True)
F = d[d.arm == "F"]
conv = F[F.converged.astype(bool)]
print(f"# A33 joint (S2OA: aniso cations + twin m 0.30 + per-domain extinction t_b 5 um): {d.seed.nunique()} seeds\n")
print("arm   converged  GooF median  n_par  n_unique")
for arm in ("F", "FmE", "FmA", "FmT", "K"):
    g = d[d.arm == arm]
    print(f"{arm:5s} {int(g.converged.astype(bool).sum()):3d}/{len(g):<3d}   {g.goof.median():8.3f}   {int(g.n_par.median()):4d}   {int(g.n_unique.median()):4d}")


def thr(p):
    if p == "t_b":
        return 2.0
    if p.startswith("twin"):
        return 0.02
    return 0.02 if p.endswith(".z") else 0.00127


names = [c[2:] for c in F.columns if c.startswith("e:")]
det = [p for p in names if np.nanmedian(conv[f"s:{p}"].replace([np.inf, -np.inf], np.nan)) < thr(p)]
und = [p for p in names if p not in det]
print(f"\ndeterminable (median formal sigma below threshold): {len(det)}: {det}\nnot determined: {und}")
rows = []
for p in names:
    e, s = conv[f"e:{p}"], conv[f"s:{p}"]
    ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
    z = e[ok] / s[ok]
    rows.append(dict(param=p, det=p in det, n=int(ok.sum()), mean_err=e[ok].mean(), med_sigma=s[ok].median(), pull_mean=z.mean(), pull_sd=z.std(), wrong_gt3=int((z.abs() > 3).sum())))
t = pd.DataFrame(rows)
print("\nArm F, per parameter (error = estimate - truth):")
print(t.round(4).to_string(index=False))
m, tb = conv["x:twin.m"], conv["x:t_b"]
print(f"\nm: median {m.median():.4f} (true 0.30), sigma_m median {conv['s:twin.m'].median():.4f};  t_b: median {tb.median():.3f} um (true 5.0), sigma median {conv['s:t_b'].median():.3f}")
cc = [c for c in conv.columns if c.startswith("c:")]
print("\ncorrelations (arm F, converged seeds): median / fraction of seeds with |corr| > 0.98")
bad_any = (conv[cc].abs() > 0.98).any(axis=1).mean()
for c in cc:
    print(f"  {c:28s} median {conv[c].median():+.3f}   |corr|>0.98 in {100 * (conv[c].abs() > 0.98).mean():.0f} %")
print(f"  seeds with ANY |corr| > 0.98 among these pairs: {100 * bad_any:.0f} %")
print("\nNecessity of each effect (ablated arm vs arm F; 'needs' = GooF >= 1.5 x F's OR a determined parameter of F with |mean ablated-arm error| > 1 sigma_F):")
for arm, eff in (("FmE", "extinction"), ("FmA", "anisotropic U"), ("FmT", "twin fraction"), ("K", "all three")):
    g = d[d.arm == arm]; gc = g[g.converged.astype(bool)]
    gr = g.goof.median() / F.goof.median()
    sh = {}
    for p in det:
        if f"e:{p}" in gc and gc[f"e:{p}"].notna().any():
            sh[p] = abs(gc[f"e:{p}"].mean()) / conv[f"s:{p}"].median()
    best = max(sh, key=sh.get) if sh else None
    need = gr >= 1.5 or (best is not None and sh[best] > 1.0)
    print(f"  {arm:4s} (no {eff:13s}): converged {len(gc)}/{len(g)}, GooF {g.goof.median():.2f} (x{gr:.2f}), largest |mean error|/sigma_F {sh[best] if best else float('nan'):.2f} ({best}) -> {'NEEDED' if need else 'not needed at this size'}")
dd = t[t.det]
conf = (len(conv) >= 27 and abs(m.median() - 0.30) <= 0.03 and abs(tb.median() - 5.0) <= 1.0 and dd.pull_sd.between(0.7, 1.4).all() and (dd.pull_mean.abs() <= 0.5).all()
        and F.goof.median() <= 1.2 and bad_any <= 0.5)
ref = (dd.pull_sd.lt(0.5) | dd.pull_sd.gt(2.0)).any() or len(conv) < 24 or abs(m.median() - 0.30) > 0.06 or bad_any > 0.5
print("\nREGISTERED READING:", "UNINFORMATIVE (< 6 determined parameters)" if len(det) < 6 else "REFUTED" + (" (identifiability limit: |corr| > 0.98 in more than half the seeds)" if bad_any > 0.5 else "") if ref else "CONFIRMED" if conf else "INCONCLUSIVE")
