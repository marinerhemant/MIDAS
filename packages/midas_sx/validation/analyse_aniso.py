"""Read results/aniso_{S2A,T0}_*.csv against PREREGISTER A22 (dev).   python analyse_aniso.py"""
import glob
import numpy as np, pandas as pd
TOL_X, TOL_U = 0.02, 0.00127
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
def load(d):
    fs = sorted(glob.glob(f"results/aniso_{d}_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
T0, S2A = load("T0"), load("S2A")
if T0 is None:
    raise SystemExit("no T0 data")
A0 = T0[T0.arm == "A"]
pcols = [c[2:] for c in A0.columns if c.startswith("e:") and not c.endswith(".Ueq")]
det = []
for p in pcols:
    tol = TOL_X if p.split(".")[1] in ("x", "y", "z") else TOL_U
    if np.nanmedian(A0[f"s:{p}"].replace(np.inf, np.nan)) < tol:
        det.append(p)
say(f"# A22 analysis. Determined set (rule: median formal sigma in arm A on T0 below 0.02 A / 0.00127 A^2): {det}")
undet = [p for p in pcols if p not in det]
say(f"  undetermined: {undet}")
coords = [p for p in det if p.split('.')[1] in ('x', 'y', 'z')]
iso_coords = [p for p in det if p in [c for c in pcols if True]]  # alias

def block(d, name):
    I, A = d[d.arm == "I"].set_index("seed"), d[d.arm == "A"].set_index("seed")
    say(f"\n## {name}: {d.seed.nunique()} seeds")
    for lab, g in (("I", I), ("A", A)):
        say(f"   arm {lab}: converged {int(g.converged.sum())}/{len(g)}; GooF median {g.goof.median():.2f} (IQR {g.goof.quantile(.25):.2f}-{g.goof.quantile(.75):.2f}); poor_fit {int(g.poor_fit.sum())}")
        nc = g[~g.converged]
        if len(nc):
            say("      not converged: " + "; ".join(f"{r}x{c}" for r, c in nc.reason.str.replace(r"\(.*?\)", "", regex=True).str.slice(0, 60).value_counts().items()))
    cA = A[A.converged]
    rows = []
    for p in det:
        sA = cA[f"s:{p}"]
        pull = cA[f"e:{p}"] / sA
        row = dict(param=p, sigma_A=sA.median(), A_bias_sig=A[f"e:{p}"].mean() / sA.median(), A_pull_mean=pull.mean(), A_pull_sd=pull.std(ddof=1))
        if f"e:{p}" in I.columns and f"s:{p}" in I.columns and not I[f"e:{p}"].isna().all():
            sI = I[f"s:{p}"]
            row.update(I_bias_sig_A=I[f"e:{p}"].mean() / sA.median(), sigA_over_sigI=(sA / sI.loc[sA.index]).median(),
                       AminusI_rms_sigI=np.sqrt((((A[f"e:{p}"] - I[f"e:{p}"]) / I[f"s:{p}"]) ** 2).mean()))
        rows.append(row)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    say(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    # cation U_eq (both arms)
    for lab in ("La1", "La2", "Ni"):
        say(f"   U_eq {lab}: A bias {A[f'e:{lab}.Ueq'].mean():+.2e} (sd {A[f'e:{lab}.Ueq'].std():.1e}); I bias {I[f'e:{lab}.U'].mean():+.2e} (sd {I[f'e:{lab}.U'].std():.1e})")
    cc = {lab: A[f"c:{lab}"].abs() for lab in ("La1", "La2", "Ni")}
    say("   |corr(Uxx,Uzz)| median: " + ", ".join(f"{k} {v.median():.2f} (>0.98 in {100*(v>0.98).mean():.0f} %)" for k, v in cc.items()) + f"; smallest Cartesian-U eigenvalue min over seeds {A.min_eig.min():.2e}")
    return I, A, t

if S2A is not None:
    I, A, t = block(S2A, "S2A (anisotropic truth)")
    tc = t[t.param.isin(coords)]
    informative = (tc.I_bias_sig_A.abs() > 1).any() or (I.goof.median() >= 1.5 * A.goof.median())
    ok_conv = A.converged.mean() >= 0.9
    ok_pull = t.A_pull_sd.between(0.7, 1.4).all() and (t.A_pull_mean.abs() <= 0.5).all()
    ok_goof = A.goof.median() <= 1.2
    say(f"\nE1 S2A: informative (I: coordinate mean shift >1 sigma_A or GooF_I >= 1.5 GooF_A): {informative}; A converged >= 90 %: {ok_conv} ({100*A.converged.mean():.0f} %); "
        f"pull sd in [0.7,1.4] and |mean pull| <= 0.5 for all determined: {ok_pull}; GooF median <= 1.2: {ok_goof}  => " + ("CONFIRM" if ok_conv and ok_pull and ok_goof else "NOT MET"))
    bias_I = tc.I_bias_sig_A.abs().mean(); bias_A = tc.A_bias_sig.abs().mean()
    say(f"   REFUTE checks: pull sd outside [0.5,2]: {(~t.A_pull_sd.between(0.5, 2)).any()}; A converged < 80 %: {A.converged.mean() < 0.8}; "
        f"corr>0.98 in > half the seeds for some site: {any(((A[f'c:{l}'].abs() > 0.98).mean() > 0.5) for l in ('La1','La2','Ni'))}; "
        f"mean |coordinate bias|/sigma_A: A {bias_A:.2f} vs I {bias_I:.2f} (A no closer while I informative: {informative and bias_A >= bias_I})")
if T0 is not None:
    I, A, t = block(T0, "T0 (isotropic truth, null)")
    ds = {lab: (A[f"d:{lab}"] / A[f"ds:{lab}"]).abs() for lab in ("La1", "La2", "Ni")}
    frac = {k: float((v <= 2).mean()) for k, v in ds.items()}
    tcT = t[t.param.isin(coords)]
    ratio_ok = (tcT.sigA_over_sigI <= 1.5).all()
    exp = np.sqrt(np.maximum(tcT.sigA_over_sigI ** 2 - 1, 0))
    shift_ok = (tcT.AminusI_rms_sigI <= 1.2 * exp + 0.1).all()
    say(f"\nE2 T0: (Uzz-Uxx)/sigma within 2 in {', '.join(f'{k} {100*v:.0f} %' for k, v in frac.items())} (need >= 90 % each: {all(v >= 0.9 for v in frac.values())}); "
        f"sigma_A/sigma_I of determined coordinates <= 1.5: {ratio_ok} (max {tcT.sigA_over_sigI.max():.2f}); A-I paired rms <= 1.2 x nested expectation + 0.1: {shift_ok}")
    for _, r_ in tcT.iterrows():
        say(f"      {r_.param}: A-I rms {r_.AminusI_rms_sigI:.3f} vs bound {1.2*np.sqrt(max(r_.sigA_over_sigI**2-1,0))+0.1:.3f}")
    say("   => " + ("CONFIRM" if all(v >= 0.9 for v in frac.values()) and ratio_ok and shift_ok else "NOT MET") + f"; REFUTE check (inflation > 2): {(tcT.sigA_over_sigI > 2).any()}")
open("results/aniso_analysis.txt", "w").write("\n".join(out) + "\n")
