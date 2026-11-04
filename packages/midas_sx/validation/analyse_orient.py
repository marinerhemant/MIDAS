"""Read results/c2v4_S2_*.csv against PREREGISTER A19 exactly as registered (dev).   python analyse_orient.py"""
import glob
import numpy as np, pandas as pd
rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))], ignore_index=True)
v4 = rd("results/c2v4_S2_bilayer_like_*.csv"); v3 = rd("results/c2v3_S2_bilayer_like_*.csv")
out = []
def say(*a):
    t = " ".join(str(x) for x in a); print(t); out.append(t)
TOL = {"z": 0.02, "U": 0.00127}
ref = v4[v4.start == "N3p"].set_index("seed"); nul = v4[v4.start == "null_N3p"].set_index("seed"); ex = v3[v3.start == "N3p"].set_index("seed")
n = len(ref)
say(f"# A19 orientation + geometry refinement from a wrong start: {n} seeds (start: U 0.3 deg off, beam centre 0.5 px off, distance 1 % off)")
# ---- O1
u = ref.u_err_deg
say(f"\nO1  refined U error [deg]: median {u.median():.4f}  mean {u.mean():.4f}  p90 {u.quantile(.9):.4f}  max {u.max():.4f}")
f05 = float((u <= 0.05).mean())
say(f"    fraction within 0.05 deg: {f05:.0%}  (need >= 90 %);  median <= 0.03: {'yes' if u.median() <= 0.03 else 'no'}  -> O1 {'MET' if (f05 >= 0.9 and u.median() <= 0.03) else 'NOT MET'}")
say(f"    existing direction-only tool from the same start: median {ref.u_err_existing_deg.median():.4f} deg, p90 {ref.u_err_existing_deg.quantile(.9):.4f}, within 0.05 deg: {float((ref.u_err_existing_deg <= 0.05).mean()):.0%}")
say(f"    optimiser converged: {int(ref.o_converged.sum())}/{n}; spots used median {ref.o_n_used.median():.0f} of {ref.o_n_spots.median():.0f} blobs; rms residual {ref.o_rms_px.median():.3f} px, {ref.o_rms_frame.median():.3f} frames")
say(f"    formal sigma of the rotation (deg) median x/y/z: {ref.o_sig_x.median():.4f} / {ref.o_sig_y.median():.4f} / {ref.o_sig_z.median():.4f}")
say(f"    geometry recovered (error after refinement; start error): beam centre row {ref.bc_row_err.mean():+.3f} +- {ref.bc_row_err.std():.3f} px (start rms {np.sqrt((ref.o_start_bc_row**2).mean()):.2f}), "
    f"col {ref.bc_col_err.mean():+.3f} +- {ref.bc_col_err.std():.3f} px, distance {ref.dlsd_rel_err.mean():+.2e} +- {ref.dlsd_rel_err.std():.1e} (start 1e-2)")
# ---- refute conditions
say("\nREFUTE conditions (A19):")
c1 = float((u > 0.10).mean()); say(f"    refined U error > 0.10 deg in {c1:.0%} of seeds (refuted if > 10 %): {'REFUTED' if c1 > 0.10 else 'not triggered'}")
c3 = float((ref.o_n_used < 0.6 * 142).mean()); say(f"    hkl assignment collapse (< 85 spots used) in {c3:.0%} of seeds (refuted if > 10 %): {'REFUTED' if c3 > 0.10 else 'not triggered'}")
# ---- structure
pn = [c[2:] for c in ref.columns if c.startswith("e:")]
sig = {p: ref["s:" + p].replace(np.inf, np.nan).median() for p in pn}
det = [p for p in pn if sig[p] < (TOL["U"] if p.endswith(".U") else TOL["z"])]
conv = ref[ref.converged]
say(f"\nstructure refinement with the refined U and geometry (N3p start): converged {len(conv)}/{n}; GooF median {conv.goof.median():.2f}; determined set {det}")
rows = []
paired_max_any = np.zeros(len(conv), bool)
for p in det:
    e4 = conv["e:" + p]; s4 = conv["s:" + p]
    common = conv.index.intersection(ex.index)
    d_sig = (conv.loc[common, "e:" + p] - ex.loc[common, "e:" + p]) / ex.loc[common, "s:" + p]
    pull = e4 / s4
    rows.append(dict(param=p, pull_sd=pull.std(ddof=1), pull_mean=pull.mean(), paired_mean=d_sig.mean(), paired_rms=np.sqrt((d_sig ** 2).mean()),
                     paired_max=d_sig.abs().max(), frac_gt1=float((d_sig.abs() > 1).mean())))
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
say(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
sd_ok = bool(t.pull_sd.between(0.7, 1.4).all()); pm_ok = bool((t.paired_mean.abs() <= 0.3).all()); pr_ok = bool((t.paired_rms <= 0.5).all())
say(f"\nO2  pull sd in [0.7,1.4] for all {len(det)}: {'MET' if sd_ok else 'NOT MET'};  paired mean shift vs exact U <= 0.3 sigma: {'MET' if pm_ok else 'NOT MET'};  paired rms <= 0.5 sigma: {'MET' if pr_ok else 'NOT MET'}"
    f"   -> O2 {'MET' if (sd_ok and pm_ok and pr_ok) else 'NOT MET'}")
say(f"    (reported, not the criterion: |mean pull| <= 0.3 for all: {'yes' if bool((t.pull_mean.abs() <= 0.3).all()) else 'no'}; max |mean pull| {t.pull_mean.abs().max():.3f})")
hit = []
for sd_ in conv.index.intersection(ex.index):
    d_ = (conv.loc[sd_, ["e:" + q for q in det]].to_numpy(float) - ex.loc[sd_, ["e:" + q for q in det]].to_numpy(float)) / ex.loc[sd_, ["s:" + q for q in det]].to_numpy(float)
    hit.append(bool(np.abs(d_).max() > 1.0))
fr = float(np.mean(hit))
say(f"    REFUTE: seeds with a paired shift > 1 sigma on ANY determined parameter: {fr:.0%} (refuted if > 10 %): {'REFUTED' if fr > 0.10 else 'not triggered'}")
# ---- null arm
nc = nul[nul.converged]
rows = []
for p in det:
    common = nc.index.intersection(ex.index)
    d_sig = (nc.loc[common, "e:" + p] - ex.loc[common, "e:" + p]) / ex.loc[common, "s:" + p]
    rows.append(dict(param=p, null_paired_mean=d_sig.mean(), null_paired_rms=np.sqrt((d_sig ** 2).mean()), null_paired_max=d_sig.abs().max(), null_n_unique=nc.n_unique.median()))
say("\nNULL arm (the wrong start left unrefined): paired shift vs exact U in sigma")
say(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
open("results/c2v4_analysis_orientation.txt", "w").write("\n".join(out) + "\n")
