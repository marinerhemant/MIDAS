"""PREREGISTER A24: refine merged F2 of the merohedral-twin variants with arm U (no twin) and arm T (twin fraction free, start 0.15, bounds [0, 0.5]).

    python run_twin2_refine.py <tw30|tw10|tw00> <seed_lo> <seed_hi>     # appends results/twin2_<variant>_<lo>.csv

Merged tables: results/c2v3_<variant>_merged/S2O_immm_seed<N>.npz (run_c2.py S2O ... v3 with TWIN=<percent>). Both arms start from the truth structure displaced
0.10 A along the allowed shifts (seed 10000*seed+2).
"""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

var, lo_s, hi_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m_true = int(var[2:]) / 100.0
st = sim.structure_s2o()
truth = Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup),
                [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
labels = [a.label for a in truth.atoms]
import os
SFX = f"_o{os.environ['ORI']}" if os.environ.get("ORI") else ""            # A30: another orientation
csv = Path("results") / f"twin2_{var}{SFX}_{lo_s}.csv"


def errors(r, model):
    o = {}
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind in ("scale", "ext"):
            continue
        if kind == "twin":
            o[nm] = r.x[j] - m_true; continue
        ia = labels.index(nm.split(".")[0])
        if kind == "U":
            o[nm] = r.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2
        else:
            b = model.basis_frac[j]
            d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            o[nm] = float(d @ b) / float(b @ b)
    return o


for seed in range(lo_s, hi_s):
    f = Path(f"results/c2v3_{var}{SFX}_merged") / f"S2O_immm_seed{seed}.npz"
    if not f.exists():
        print(f"seed {seed}: no merged table"); continue
    z = np.load(f)
    hkl, F2, sig = z["hkl"], z["F2"], z["sig"]
    start = perturbed_crystal(truth, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    rows = []
    for arm in ("U", "T"):
        t0 = time.time()
        model = RefineModel(start, refine_twin=(arm == "T"), twin_start=0.15)
        r = refine(model, hkl, F2, sig)
        err = errors(r, model)
        row = dict(variant=var, seed=seed, arm=arm, converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2,
                   n_unique=len(hkl), n_par=r.n_par, nfev=r.nfev, sv_ratio=r.sv_ratio, seconds=time.time() - t0, m_true=m_true,
                   m=r.twin_fraction, sigma_m=r.sigma_twin_fraction)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        if arm == "T":
            C = r.correlation(); im = r.names.index("twin.m")
            for j, nm in enumerate(r.names):
                if j != im:
                    row[f"c:{nm}"] = C[im, j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed} {var}: " + "; ".join(f"{r_['arm']} conv={r_['converged']} GooF {r_['goof']:.2f}" for r_ in rows) + f"; m {rows[1]['m']:.3f} +- {rows[1]['sigma_m']:.3f}", flush=True)
