"""PREREGISTER A36 T3: untwinned S2 orientation 1, v3 chain, seeds 0-24, SAME frames, background mean vs quadratic. Run: python analyse_t3_a36.py > results/a36t3_analysis.txt
Criteria (CONFIRM): median over the determined parameters of sigma_quad / sigma_mean <= 1.10; R_F2 of the merged table vs truth (strong, one global scale) not worse by more than
0.5 percentage points; pull sd in [0.7, 1.4] and |mean pull| <= 0.5 for the determined parameters; GooF median <= 1.2; >= 90 % converged. REFUTE: sigma inflation > 1.30 or pulls sd outside
[0.5, 2]. The determined set is the A26 rule applied to the MEAN arm: median formal sigma < 0.02 A (coordinates) / 0.00127 A^2 (U)."""
import glob, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import simulate as sim

arms = {"mean": "results/c2v3_t3_S2_bilayer_like_0.csv", "quadratic": "results/c2v3_bgq_t3_S2_bilayer_like_0.csv"}
mer = {"mean": "results/c2v3_t3_merged/S2_bilayer_like_seed{}.npz", "quadratic": "results/c2v3_bgq_t3_merged/S2_bilayer_like_seed{}.npz"}
D = {k: pd.read_csv(v) for k, v in arms.items()}
D = {k: d[(d.start == "N3p") & d.note.isna()] if "note" in d else d[d.start == "N3p"] for k, d in D.items()}
seeds = sorted(set(D["mean"].seed) & set(D["quadratic"].seed))
print(f"# A36 T3: {len(seeds)} paired seeds (N3p start, 0.10 A)")
pars = [c[2:] for c in D["mean"].columns if c.startswith("e:")]
det = [p for p in pars if np.nanmedian(D["mean"][f"s:{p}"].replace(np.inf, np.nan)) < (0.02 if p.split(".")[1] in ("x", "y", "z") else 0.00127)]
print("determined set:", det)
rows = []
for p in det:
    r = {"param": p}
    for arm, d in D.items():
        d = d[d.seed.isin(seeds)]
        c = d[d.converged.astype(bool)]
        pull = c[f"e:{p}"] / c[f"s:{p}"]
        r[f"{arm}_sigma"] = c[f"s:{p}"].median(); r[f"{arm}_pull_mean"] = pull.mean(); r[f"{arm}_pull_sd"] = pull.std(ddof=1)
    r["sigma_ratio"] = r["quadratic_sigma"] / r["mean_sigma"]
    rows.append(r)
t = pd.DataFrame(rows); pd.set_option("display.width", 220)
print(t.round(3).to_string(index=False))
for arm, d in D.items():
    d = d[d.seed.isin(seeds)]
    print(f"{arm:9s} converged {int(d.converged.astype(bool).sum())}/{len(d)}; GooF median {d.goof.median():.2f}")
# R_F2 vs truth on strong merged reflections
def rf2(arm):
    out = []
    for s in seeds:
        z = np.load(mer[arm].format(s))
        hkl, F2, sg = z["hkl"], z["F2"], z["sig"]
        ft = sim.f2_truth(sim.structure_s2_bilayer(), hkl)
        m = F2 / sg > 20
        k = (F2[m] * ft[m]).sum() / (ft[m] ** 2).sum()
        out.append(np.abs(F2[m] - k * ft[m]).sum() / F2[m].sum())
    return np.array(out)
ra, rb = rf2("mean"), rf2("quadratic")
print(f"R_F2 (strong, global scale): mean {100*np.median(ra):.2f} %  quadratic {100*np.median(rb):.2f} %  difference {100*(np.median(rb)-np.median(ra)):+.2f} pp")
mc = D["mean"][D["mean"].seed.isin(seeds)]; qc = D["quadratic"][D["quadratic"].seed.isin(seeds)]
c1 = t.sigma_ratio.median() <= 1.10; c2 = (np.median(rb) - np.median(ra)) <= 0.005
c3 = t.quadratic_pull_sd.between(0.7, 1.4).all() and (t.quadratic_pull_mean.abs() <= 0.5).all()
c4 = qc.goof.median() <= 1.2; c5 = qc.converged.astype(bool).mean() >= 0.9
print(f"\nsigma ratio median {t.sigma_ratio.median():.3f} (<= 1.10) {c1}; R_F2 {c2}; pulls {c3}; GooF {c4}; converged {c5}")
refute = t.sigma_ratio.median() > 1.30 or (~t.quadratic_pull_sd.between(0.5, 2)).any()
print("T3 reading:", "CONFIRMED" if all([c1, c2, c3, c4, c5]) else "REFUTED" if refute else "INCONCLUSIVE")
