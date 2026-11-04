"""Read results/c2_<S>.csv against PREREGISTER A4/A8 (dev).   python analyse_c2.py <S1|S2>"""
import sys
import numpy as np, pandas as pd
which = sys.argv[1]
PFX = {"v2": "c2v2", "v3": "c2v3"}.get(sys.argv[2] if len(sys.argv) > 2 else "", "c2")
name = "S1_Cu" if which == "S1" else "S2_bilayer_like"
import glob
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"results/{PFX}_{name}_*.csv"))], ignore_index=True)
d = d[d["start"] != "-"]
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
say(f"# C2 analysis ({PFX}) {name}: seeds={d.seed.nunique()} rows={len(d)}   (PREREGISTER A4, A8)")
TOL_X, TOL_U = 0.02, 0.00127            # A, A^2: 'determined' = median formal sigma below this (A4)
pcols = [c[2:] for c in d.columns if c.startswith("e:")]
for start in ("N3", "N3p"):
    g = d[d.start == start]
    n = len(g); conv = g[g.converged]
    say(f"\n== start {start}: {n} seeds; converged {len(conv)} ({100*len(conv)/max(n,1):.0f} %); "
        f"converged AND GooF<=3: {int((conv.goof <= 3).sum())}; poor fit (GooF>3) among converged: {int((conv.goof > 3).sum())}")
    nc = g[~g.converged]
    if len(nc):
        reasons = nc.reason.str.replace(r"\(.*?\)", "", regex=True).str.slice(0, 60).value_counts()
        say("   not converged, by reason: " + "; ".join(f"{r} x{c}" for r, c in reasons.items()))
    say(f"   unique reflections per seed: median {g.n_unique.median():.0f} (min {g.n_unique.min()}, max {g.n_unique.max()}); n_par {int(g.n_par.iloc[0])}; "
        f"GooF median {conv.goof.median():.2f} (IQR {conv.goof.quantile(.25):.2f}-{conv.goof.quantile(.75):.2f})")
    if len(conv) < 5:
        say("   too few converged seeds to read pulls"); continue
    rows = []
    for p in pcols:
        sig = conv[f"s:{p}"].replace([np.inf], np.nan)
        err = conv[f"e:{p}"]
        tol = TOL_U if p.endswith(".U") else TOL_X
        determined = bool(np.nanmedian(sig) < tol)                    # fixed from the formal sigma ALONE (A4)
        pull = err / sig
        rows.append(dict(param=p, determined=determined, med_sigma=np.nanmedian(sig), tol=tol, bias=err.mean(),
                         rms_err=np.sqrt((err ** 2).mean()), pull_mean=pull.mean(), pull_sd=pull.std(ddof=1),
                         pull_mean_se=pull.std(ddof=1) / np.sqrt(len(pull)),
                         pull_sd_rescaled=(err / (sig * conv.goof)).std(ddof=1), n=len(pull)))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    say(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    det = t[t.determined]
    ok_mean = (det.pull_mean.abs() <= 0.3).all() if len(det) else False
    ok_sd = det.pull_sd.between(0.7, 1.4).all() if len(det) else False
    say(f"   determined parameters ({len(det)}): {', '.join(det.param)}")
    say(f"   undetermined (median formal sigma above tolerance): {', '.join(t[~t.determined].param) or 'none'}")
    say(f"   C2 pull criteria on the determined set (|mean|<=0.3 and 0.7<=sd<=1.4): mean {'MET' if ok_mean else 'NOT MET'}, sd {'MET' if ok_sd else 'NOT MET'}")
    say(f"   coordinate bias <= 0.02 A on determined coordinates: " + ("MET" if (det[~det.param.str.endswith('.U')].bias.abs() <= 0.02).all() else "NOT MET"))
open(f"results/{PFX}_analysis_{name}.txt", "w").write("\n".join(out) + "\n")
