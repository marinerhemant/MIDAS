"""PREREGISTER A29 analysis: reads results/a29_rows_<var>_*.csv and results/a29_orient_<var>_*.csv and prints the readings against the thresholds fixed in A29.

    python analyse_a29.py > results/a29_analysis.txt
"""
import glob
import numpy as np, pandas as pd

R = Path = None
pd.set_option("display.width", 200)


def load(pat):
    fs = sorted(glob.glob(pat))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def med_z(d):
    """median z and its approximate standard error 1.25/sqrt(n)."""
    if len(d) == 0:
        return np.nan, np.nan, 0
    return float(d.z.median()), 1.25 / np.sqrt(len(d)), len(d)


print("# A29 analysis (PREREGISTER A29). z = (I - exp_tot) / sigI, sigI from the noisy reduction of the same row.\n")
reads = {}
rows_all = {}
for var in ("P11", "P30", "P00"):
    d = load(f"results/a29_rows_{var}_*.csv")
    if var == "P00" and not d.empty:
        d = d[d.kind != "c2"]                      # the untwinned control has no second component: its c2 targets measure nothing (the first 9 seeds were written with a spurious truth for them)
    if d.empty:
        print(f"## {var}: no rows\n"); continue
    noisy = d[d.frames == "noisy"][["seed", "kind", "h", "k", "l", "sigI"]].rename(columns={"sigI": "sig_n"})
    d = d.merge(noisy.drop_duplicates(["seed", "kind", "h", "k", "l"]), on=["seed", "kind", "h", "k", "l"], how="left")
    d["z"] = (d.I - d.exp_tot) / d.sig_n
    d["matched"] = d.exp_tot > 0
    rows_all[var] = d
    print(f"## {var}: {d.seed.nunique()} seeds, {len(d)} rows")
    for fr in ("clean", "noisy"):
        for kind in ("c1", "c2", "blend"):
            for lab, sel in (("exp_tot < 3000", d.exp_tot < 3000), ("exp_tot >= 3000", d.exp_tot >= 3000)):
                g = d[(d.frames == fr) & (d.kind == kind) & sel & d.matched]
                if len(g) < 5:
                    continue
                m, se, n = med_z(g)
                print(f"  {fr:5s} {kind:5s} {lab:16s} n={n:4d}  median z {m:+.3f} (se {se:.3f})  median ratio I/exp {np.median(g.I / g.exp_tot):.4f}  median I-exp {np.median(g.I - g.exp_tot):+.1f}")
    unm = d[(d.frames == "clean")].groupby("kind").matched.agg(lambda s: (~s).mean())
    print("  fraction of rows with NO truth match (exp = 0), clean frames:", {k: round(v, 3) for k, v in unm.items()})
    b = d[(d.frames == "clean") & (d.kind == "blend") & d.matched & (d.exp_tot < 3000)]
    if len(b):
        b = b.assign(sep_bin=pd.cut(b.sep_px, [0, 1, 2, 3, 5, 100]))
        print("  blend rows, clean, exp_tot < 3000, by separation (px):")
        print(b.groupby("sep_bin", observed=True).z.agg(["count", "median"]).round(3).to_string())
    print()

print("# D1 readings against A29")
def reading(var):
    d = rows_all.get(var)
    if d is None:
        return None
    cl = d[(d.frames == "clean") & d.matched & (d.exp_tot < 3000)]
    nz = d[(d.frames == "noisy") & d.matched & (d.exp_tot < 3000)]
    bc, sec, nbc = med_z(cl[cl.kind == "blend"]); bn, sen, nbn = med_z(nz[nz.kind == "blend"])
    c1, _, n1 = med_z(cl[cl.kind == "c1"])
    return dict(blend_clean=bc, se_clean=sec, n_clean=nbc, blend_noisy=bn, se_noisy=sen, n_noisy=nbn, c1_clean=c1, n_c1=n1)

ctrl = reading("P00")
if ctrl is not None:
    ok = abs(ctrl["c1_clean"]) < 0.10
    print(f"P00 control: c1 clean median z {ctrl['c1_clean']:+.3f} (n={ctrl['n_c1']}) -> control {'OK' if ok else 'FAILS: D1 is INCONCLUSIVE'}")
else:
    ok = False
for var in ("P11", "P30"):
    r = reading(var)
    if r is None:
        continue
    A = (r["blend_clean"] <= -0.25) and (abs(r["blend_clean"]) / r["se_clean"] >= 4) and (abs(r["c1_clean"]) < 0.10)
    Bn = (r["blend_noisy"] <= -0.25) and (abs(r["blend_noisy"]) / r["se_noisy"] >= 4)
    B = Bn and abs(r["blend_clean"]) < 0.10
    C = abs(r["blend_clean"]) < 0.10 and abs(r["blend_noisy"]) < 0.10
    word = "H-A (reduction offset on blends)" if A else "H-B (noise-driven)" if B else "H-C (not the reduction)" if C else "INCONCLUSIVE"
    if not ok:
        word = "INCONCLUSIVE (control failed)"
    print(f"{var}: blend clean {r['blend_clean']:+.3f} (se {r['se_clean']:.3f}, n={r['n_clean']}), blend noisy {r['blend_noisy']:+.3f} (se {r['se_noisy']:.3f}, n={r['n_noisy']}), "
          f"c1 clean {r['c1_clean']:+.3f}  =>  {word}")

print("\n# D2 orientation: domain-1 error after index_and_refine, noisy vs clean frames (same blobs positions NOT shared; clean frames use their own spot list)")
for var in ("P11", "P30", "P00"):
    o = load(f"results/a29_orient_{var}_*.csv")
    if o.empty or "err_clean_deg" not in o:
        print(f"{var}: no data"); continue
    o = o.dropna(subset=["err_clean_deg"])
    mn, mc = o.err_noisy_deg.median(), o.err_clean_deg.median()
    word = "H-D (noise floor)" if mc <= 0.3 * mn else "H-E (bias floor)" if mc >= 0.7 * mn else "INCONCLUSIVE"
    print(f"{var}: n={len(o)} seeds, noisy median {mn:.4f} deg, clean median {mc:.4f} deg, ratio {mc / mn:.2f}, difference {mn - mc:+.4f} deg  =>  {word}")
