"""Read results/twin_*.csv against PREREGISTER A23 (dev).   python analyse_twin.py"""
import glob, json
import numpy as np, pandas as pd
DET = ["La2.z", "Ni.z", "O1.z", "O2.z", "La1.U", "La2.U", "Ni.U", "O1.U"]
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/twin_[0-9]*.csv"))], ignore_index=True)
if "note" in d.columns:
    bad = d[d.note.notna()]
    if len(bad):
        say(f"# seeds with a failed chain (no result rows): {len(bad)}: " + "; ".join(f"seed {int(r.seed)}: {r.note}" for r in bad.itertuples()))
    d = d[d.note.isna()]
S1, M = d[d.arm == "S1"].set_index("seed"), d[d.arm == "M"].set_index("seed")
say(f"# A23 two-domain analysis: {len(M)} seeds with results")
# C1 orientations
for k, lab in (("u1_err_deg", "domain 1"), ("u2_err_deg", "domain 2")):
    v = M[k]
    say(f"C1 {lab}: U error median {v.median():.4f} deg (p90 {v.quantile(.9):.4f}, max {v.max():.4f}); within 0.05 deg {100*(v<=0.05).mean():.0f} %; above 0.10 deg {100*(v>0.10).mean():.0f} %; "
        f"optimiser converged {100*M['o1_converged' if lab=='domain 1' else 'o2_converged'].astype(bool).mean():.0f} %")
ok_c1 = all((M[k] <= 0.05).mean() >= 0.9 for k in ("u1_err_deg", "u2_err_deg"))
ref_c1 = any((M[k] > 0.10).mean() > 0.10 for k in ("u1_err_deg", "u2_err_deg"))
# C2 arm M
cM = M[M.converged]
rows = []
for p in DET:
    pull = cM[f"e:{p}"] / cM[f"s:{p}"]
    sS = S1[S1.converged][f"s:{p}"]
    rows.append(dict(param=p, sigma_M=cM[f"s:{p}"].median(), sigma_S1=sS.median(), ratio=cM[f"s:{p}"].median() / sS.median(),
                     M_pull_mean=pull.mean(), M_pull_sd=pull.std(ddof=1),
                     S1_pull_mean=(S1[S1.converged][f"e:{p}"] / sS).mean(), S1_pull_sd=(S1[S1.converged][f"e:{p}"] / sS).std(ddof=1)))
t = pd.DataFrame(rows)
pd.set_option("display.width", 220)
say(f"\narm S1: converged {int(S1.converged.sum())}/{len(S1)}; GooF median {S1.goof.median():.2f}; unique reflections median {S1.n_unique.median():.0f}")
say(f"arm M : converged {int(M.converged.sum())}/{len(M)}; GooF median {M.goof.median():.2f}; unique reflections median {M.n_unique.median():.0f} (domain 1 {M.n_unique_A.median():.0f}, domain 2 {M.n_unique_B.median():.0f})")
nc = M[~M.converged]
if len(nc):
    say("   M not converged: " + "; ".join(f"{r}x{c}" for r, c in nc.reason.str.replace(r"\(.*?\)", "", regex=True).str.slice(0, 60).value_counts().items()))
say(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
ok_conv = M.converged.mean() >= 0.9
ok_pull = t.M_pull_sd.between(0.7, 1.4).all() and (t.M_pull_mean.abs() <= 0.5).all()
ok_goof = M.goof.median() <= 1.2
ratio_med = M.k_ratio.median(); true_r = M.true_ratio.iloc[0]
ok_c3 = abs(ratio_med / true_r - 1) <= 0.05
ok_c4 = int((t.ratio < 1).sum()) >= 6
say(f"\nC2 arm M: >= 90 % converged {ok_conv} ({100*M.converged.mean():.0f} %); pull sd in [0.7,1.4] and |mean| <= 0.5 {ok_pull}; GooF median <= 1.2 {ok_goof}")
say(f"C3 domain ratio k2/k1: median {ratio_med:.4f} (IQR {M.k_ratio.quantile(.25):.4f}-{M.k_ratio.quantile(.75):.4f}) vs true {true_r:.4f}: relative error {100*(ratio_med/true_r-1):+.1f} %  => {'MET' if ok_c3 else 'NOT MET'}")
say(f"C4 sigma_M below sigma_S1 for {int((t.ratio < 1).sum())}/8 determined parameters (median ratio {t.ratio.median():.3f}) => {'MET' if ok_c4 else 'NOT MET'}")
ref = []
if ref_c1: ref.append("a domain U error > 0.10 deg in > 10 % of seeds")
if (~t.M_pull_sd.between(0.5, 2)).any(): ref.append("M pull sd outside [0.5, 2]")
if abs(ratio_med / true_r - 1) > 0.20: ref.append("k2/k1 off by > 20 %")
if M.converged.mean() < 0.8: ref.append("M converged < 80 %")
if int((t.ratio > 1).sum()) > 4: ref.append("M sigma larger than S1 for more than half")
say(f"REFUTE conditions triggered: {ref or 'none'}")
say(f"\nOVERALL: " + ("CONFIRM" if (ok_c1 and ok_conv and ok_pull and ok_goof and ok_c3 and ok_c4 and not ref) else "NOT MET (see items)"))
# reported
say("\n# Reported, not judged")
say(f"unique reflections S1 {S1.n_unique.median():.0f} vs M {M.n_unique.median():.0f}; contamination (kept domain-1 reflections with a TRUE domain-2 spot in the support box): median {M.contamination.median():.3f}")
say("rejections per seed, S1 (median counts): " + json.dumps(pd.DataFrame([json.loads(x) for x in S1.rejected]).median().round(1).to_dict()))
say("rejections per seed, M  (median counts): " + json.dumps(pd.DataFrame([json.loads(x) for x in M.rejected]).median().round(1).to_dict()))
try:
    ref_ = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/c2v3_S2_bilayer_like_*.csv"))]).query("start == 'N3p'").set_index("seed")
    sh = {}
    for p in DET:
        j = S1.index.intersection(ref_.index)
        sh[p] = ((S1.loc[j, f"e:{p}"] - ref_.loc[j, f"e:{p}"]) / ref_.loc[j, f"s:{p}"])
    sh = pd.DataFrame(sh)
    say("contamination effect: S1 (two-domain frames) minus the single-domain c2v3 run of the same seeds, in c2v3 formal sigma: mean " + ", ".join(f"{p} {sh[p].mean():+.2f}" for p in DET) + "; rms " + ", ".join(f"{p} {np.sqrt((sh[p]**2).mean()):.2f}" for p in DET))
except Exception as e:                                                     # noqa: BLE001
    say(f"(c2v3 comparison unavailable: {e})")
# ---- exploratory M2 (PREREGISTER A23a): reported, does not change the A23 verdict
M2 = d[d.arm == "M2"].set_index("seed")
if len(M2):
    say("\n# EXPLORATORY arm M2 (A23a): ambiguous only when both domains are inside the 0.15 gate")
    c2 = M2[M2.converged]
    r2 = []
    for p in DET:
        pull = c2[f"e:{p}"] / c2[f"s:{p}"]
        r2.append(dict(param=p, sigma_M2=c2[f"s:{p}"].median(), ratio_vs_S1=c2[f"s:{p}"].median() / S1[S1.converged][f"s:{p}"].median(),
                       ratio_vs_M=c2[f"s:{p}"].median() / cM[f"s:{p}"].median(), pull_mean=pull.mean(), pull_sd=pull.std(ddof=1)))
    t2 = pd.DataFrame(r2)
    say(f"M2: converged {int(M2.converged.sum())}/{len(M2)}; GooF median {M2.goof.median():.2f}; unique reflections median {M2.n_unique.median():.0f} (domain 1 {M2.n_unique_A2.median():.0f}, domain 2 {M2.n_unique_B2.median():.0f}); k2/k1 median {M2.k_ratio.median():.4f}")
    say(t2.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    keep = M2.n_unique_A2.median() >= 52
    gain = int((t2.ratio_vs_S1 < 1).sum()) >= 6
    say(f"A23a reading: domain-1 unique reflections >= 52: {keep}; sigma below S1 for {int((t2.ratio_vs_S1 < 1).sum())}/8 (>= 6 needed): {gain}  => " + ("cause supported" if keep and gain else "cause NOT supported (or only partly)"))
try:
    first = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/_twin_first/twin_*.csv"))])
    for arm in ("S1", "M"):
        a_ = first[(first.arm == arm)].set_index("seed"); b_ = d[d.arm == arm].set_index("seed")
        j = a_.index.intersection(b_.index)
        dif = max(np.nanmax(np.abs(a_.loc[j, f"e:{p}"] - b_.loc[j, f"e:{p}"])) for p in DET)
        say(f"reproducibility check, arm {arm}: max |difference| of the determined-parameter errors against the first run {dif:.2e}")
except Exception as e:                                                     # noqa: BLE001
    say(f"(reproducibility check unavailable: {e})")
open("results/twin_analysis.txt", "w").write("\n".join(out) + "\n")
