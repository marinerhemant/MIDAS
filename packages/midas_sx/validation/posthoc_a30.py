"""POST HOC after the registered A30 reads (INCONCLUSIVE at O4 on convergence / a mean pull just above 0.5): the same arms with gate_free = the undetermined oxygen U.
    python posthoc_a30.py > results/a30_posthoc.txt        (same tables, same starts as the registered runs)"""
import sys, os
sys.path.insert(0, "."); sys.path.insert(0, "..")
import numpy as np, pandas as pd
os.environ["ORI"] = "4"
res = {}
for name, argv, split, arm, gf in (("S2A x O4 (arm A)", ["x", "S2A", "0", "25"], "for seed in range(lo_s, hi_s):", "A", ("O1.U", "O2.U", "O3.U")),):
    sys.argv = argv
    src = open("run_aniso_refine.py").read()
    head = src.split(split)[0]
    g = {"__name__": "ph"}; exec(head, g)
    rows = []
    for seed in range(0, 25):
        z = np.load(str(g["tab"]).format(seed)); start = g["perturbed_crystal"](g["start_crystal"], np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
        model = g["RefineModel"](start, aniso_sites=g["ANISO"])
        r = g["refine"](model, z["hkl"], z["F2"], z["sig"], gate_free=gf)
        err = g["errors"](r, model)
        row = dict(seed=seed, converged=r.converged, goof=r.goof, reason=r.reason)
        for j, nm in enumerate(r.names):
            if nm in err:
                row[f"e:{nm}"] = err[nm]; row[f"s:{nm}"] = r.sigma[j]
        rows.append(row)
    res[name] = pd.DataFrame(rows)
sys.argv = ["x", "tw30", "0", "25"]
src = open("run_twin2_refine.py").read(); head = src.split("for seed in range(lo_s, hi_s):")[0]
g = {"__name__": "ph"}; exec(head, g)
rows = []
for seed in range(0, 25):
    f = g["Path"](f"results/c2v3_tw30_o4_merged") / f"S2O_immm_seed{seed}.npz"; z = np.load(f)
    start = g["perturbed_crystal"](g["truth"], np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    model = g["RefineModel"](start, refine_twin=True, twin_start=0.15)
    r = g["refine"](model, z["hkl"], z["F2"], z["sig"], gate_free=("O1a.U", "O1b.U", "O2.U", "O3.U"))
    err = g["errors"](r, model)
    row = dict(seed=seed, converged=r.converged, goof=r.goof, m=r.twin_fraction, sm=r.sigma_twin_fraction)
    for j, nm in enumerate(r.names):
        if nm in err:
            row[f"e:{nm}"] = err[nm]; row[f"s:{nm}"] = r.sigma[j]
    rows.append(row)
res["S2O-tw30 x O4 (arm T)"] = pd.DataFrame(rows)
for name, d in res.items():
    c = d[d.converged]
    print(f"\n== POST HOC {name}, gate_free = oxygen U: converged {len(c)}/25; GooF median {d.goof.median():.3f}")
    names = [k[2:] for k in d.columns if k.startswith("e:")]
    for p in names:
        e, s = c[f"e:{p}"], c[f"s:{p}"]
        z_ = (e / s).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {p:8s} n={len(z_):2d} pull mean {z_.mean():+.3f} sd {z_.std():.3f}  median sigma {s.median():.5f}")
    if "m" in d:
        print(f"  m median {c.m.median():.4f}, sigma_m {c.sm.median():.4f}")
