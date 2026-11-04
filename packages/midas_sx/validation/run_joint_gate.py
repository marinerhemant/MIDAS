"""POST HOC (after the registered A33 reading REFUTED on convergence): arm F only, gate_free extended by the in-plane cation U components (La1/La2/Ni Uxx, Uyy), which are
undetermined (formal sigma ~0.003 against 0.00127) and are the parameters that touched a bound in all 10 failed fits. Same tables, same start.  python run_joint_gate.py > results/joint_gate_analysis.txt"""
import sys, glob
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
exec(open("run_joint_refine.py").read().split("for seed in range(lo_s, hi_s):")[0].replace("lo_s, hi_s = int(sys.argv[1]), int(sys.argv[2])", "lo_s, hi_s = 0, 30"))
GF = GATE_FREE + tuple(f"{a}.U{c}" for a in ANISO for c in ("xx", "yy"))
rows = []
for seed in range(0, 30):
    z = np.load(str(tab).format(seed))
    start = perturbed_crystal(start_crystal, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    model = RefineModel(start, **ARMS["F"])
    r = refine(model, z["hkl"], z["F2"], z["sig"], gate_free=GF)
    err = errors(r, model)
    row = dict(seed=seed, converged=r.converged, reason=r.reason, goof=r.goof, m=r.twin_fraction, sm=r.sigma_twin_fraction, tb=r.t_b_um, stb=r.sigma_t_b_um, ignored=";".join(r.at_bound_ignored))
    for j, nm in enumerate(r.names):
        if nm in err:
            row[f"e:{nm}"] = err[nm]; row[f"s:{nm}"] = r.sigma[j]
    row["e:t_b"] = err["t_b"]; row["s:t_b"] = r.sigma_t_b_um
    rows.append(row)
d = pd.DataFrame(rows); d.to_csv("results/joint_gate_0.csv", index=False)
c = d[d.converged]
print(f"POST HOC revised gate (gate_free = oxygen U + cation in-plane U): converged {len(c)}/30; GooF median {d.goof.median():.3f}; m median {c.m.median():.4f} (sigma {c.sm.median():.4f}); t_b median {c.tb.median():.3f} um (sigma {c.stb.median():.3f})")
DET = ["La2.z", "Ni.z", "O1a.z", "O1b.z", "O2.z", "La2.Uzz", "Ni.Uzz", "t_b"]
for p in DET + ["twin.m", "La1.Uzz"]:
    e, s = c[f"e:{p}"], c[f"s:{p}"]
    z_ = (e / s).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  {p:8s} n={len(z_):2d} pull mean {z_.mean():+.3f} sd {z_.std():.3f}  mean err {e.mean():+.5f}  median sigma {s.median():.5f}")
print("bound hits ignored per seed (median count):", d.ignored.fillna("").map(lambda s: len([x for x in s.split(';') if x])).median())
