"""PREREGISTER A30 analysis: python analyse_a30.py > results/a30_analysis.txt

Per condition (S2A x O4, S2A x O6, S2O-tw30 x O4, S2O-tw30 x O6): determined set from the condition's own arm A / arm T formal sigmas (A30: < 0.02 A coordinates,
< 0.00127 A^2 U/Uij, < 0.02 for m), then the A22 / A24 CONFIRM and REFUTE conditions with 25 seeds."""
import glob
import numpy as np, pandas as pd
pd.set_option("display.width", 220)


def thr(name):
    if name.startswith("twin"):
        return 0.02
    return 0.02 if name.endswith(".z") else 0.00127


def load(pattern):
    fs = sorted(glob.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def params_of(d, arm):
    return [c[2:] for c in d.columns if c.startswith("s:") and c != "s:scale" and d[d.arm == arm][c].notna().any()]


def report(title, d, arm_main, arm_null, kind, o):
    print(f"\n## {title}")
    if d.empty:
        print("  no data yet"); return None
    nseed = d.seed.nunique()
    main = d[d.arm == arm_main]; null = d[d.arm == arm_null]
    conv = main[main.converged.astype(bool)]
    print(f"  seeds {nseed}; arm {arm_main}: converged {len(conv)}/{len(main)}; GooF median {main.goof.median():.3f} (arm {arm_null}: {null.goof.median():.3f}); n_unique median {main.n_unique.median():.0f}; n_par {int(main.n_par.median())}")
    P = params_of(d, arm_main)
    # A30 rule as REGISTERED: median formal sigma of the condition's own fits (ALL fits of the main arm, converged or not; the first version of this script used converged fits only,
    # which empties the set when almost nothing converges)
    det = [p for p in P if np.nanmedian(main[f"s:{p}"].replace([np.inf, -np.inf], np.nan)) < thr(p)]
    und = [p for p in P if p not in det]
    print(f"  determined ({len(det)}): {det}\n  undetermined ({len(und)}): {und}")
    rows = []
    for p in P:
        e, s = main[f"e:{p}"], main[f"s:{p}"]                       # all fits of the main arm (pulls of non-converged fits are reported, flagged by n_conv below)
        ok = np.isfinite(e) & np.isfinite(s) & (s > 0)
        z = (e[ok] / s[ok])
        rows.append(dict(param=p, det=p in det, n=int(ok.sum()), n_conv=int((ok & main.converged.astype(bool)).sum()), med_sigma=float(np.median(s[ok])) if ok.any() else np.nan, pull_mean=float(z.mean()) if len(z) else np.nan,
                         pull_sd=float(z.std()) if len(z) else np.nan, conf_wrong_gt3=int((z.abs() > 3).sum())))
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    if kind == "twin":
        m = conv["x:twin.m"]
        print(f"  m: median {m.median():.4f} (true 0.30), sigma_m median {conv['s:twin.m'].median():.4f}, detection m > 2 sigma_m {int((m > 2 * conv['s:twin.m']).sum())}/{len(conv)}")
        cc = [c for c in conv.columns if c.startswith("c:")]
        big = (conv[cc].abs() > 0.98).any(axis=1).mean() if cc else np.nan
        print(f"  seeds with |corr(m, any parameter)| > 0.98: {100 * big:.0f} %")
        med_off = abs(m.median() - 0.30)
    else:
        big = np.nanmean([(conv[f"c:{lab}"].abs() > 0.98).mean() for lab in ("La1", "La2", "Ni") if f"c:{lab}" in conv] or [np.nan])
        print(f"  seeds with |corr(Uxx, Uzz)| > 0.98 (mean over sites): {100 * big:.0f} %")
        med_off = 0.0
    # informative
    gI = null.goof.median() / main.goof.median()
    shift = []
    for p in det:
        if f"e:{p}" in null and f"s:{p}" in main:
            en = null[f"e:{p}"][null.converged.astype(bool)]
            sn = main[f"s:{p}"].median()
            if np.isfinite(sn) and sn > 0 and len(en):
                shift.append(abs(en.mean()) / sn)
    informative = gI >= 1.5 or any(v > 1.0 for v in shift)
    print(f"  informative: GooF ratio null/main {gI:.2f}; max |mean shift of the null arm| / sigma_main over shared determined parameters {max(shift) if shift else float('nan'):.2f} -> {informative}")
    dd = t[t.det]
    conf = (len(conv) >= 23) and (dd.pull_sd.between(0.7, 1.4).all()) and ((dd.pull_mean.abs() <= 0.5).all()) and (main.goof.median() <= 1.2) and informative and \
        ((med_off <= 0.02) if kind == "twin" else True)
    ref = (dd.pull_sd.lt(0.5) | dd.pull_sd.gt(2.0)).any() or (len(conv) < 20) or (big > 0.5) or ((med_off > 0.05) if kind == "twin" else False)
    word = "UNINFORMATIVE (< 4 determined parameters)" if len(det) < 4 else "REFUTED" if ref else "CONFIRMED" if conf else "INCONCLUSIVE"
    print(f"  REGISTERED READING: {word}")
    return word


reads = {}
for o in (4, 6):
    reads[f"S2A x O{o}"] = report(f"Anisotropic U (A22 protocol), orientation seed {o}", load(f"results/aniso_S2A_o{o}_*.csv"), "A", "I", "aniso", o)
for o in (4, 6):
    reads[f"S2O-tw30 x O{o}"] = report(f"Exact twin fraction m = 0.30 (A24 protocol), orientation seed {o}", load(f"results/twin2_tw30_o{o}_*.csv"), "T", "U", "twin", o)
print("\n# Summary:", reads)
