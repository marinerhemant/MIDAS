"""PREREGISTER A27: refit the A26 (or A27) observation tables with the outlier-aware refinement and the revised gate.

    python refit_pm.py <P11|P30|P00> <seed_lo> <seed_hi> [tables=pm_merged|pm2_merged]    # appends results/pmr_<tables>_<variant>_<lo>.csv

Arms (same rows, same start as run_pm.py): U = untwinned domain-1 components, T = twin fraction with component flags, each in two versions:
  plain  (as A26)                                      -> arms U, T
  robust (outlier_sigma = 5, gate_free = the undetermined oxygen U)  -> arms Ur, Tr
gate_free is the set fixed in A26 by the rule 'median formal sigma in arm T on P00 above 0.02 A / 0.00127 A^2': O1a.U, O1b.U, O2.U, O3.U.
"""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

var, lo_s, hi_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tabdir = sys.argv[4] if len(sys.argv) > 4 else "pm_merged"
ARMS = tuple((sys.argv[5] if len(sys.argv) > 5 else "U,T,Ur,Tr").split(","))        # which arms to run (each robust arm is slow: a fit per rejected row)
B_OVER_A = {"P11": 1.011, "P30": 1.03, "P00": 1.011}[var]
M_TRUE = 0.0 if var == "P00" else 0.30
GATE_FREE = ("O1a.U", "O1b.U", "O2.U", "O3.U")
st = sim.structure_s2p(B_OVER_A)
truth = Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
labels = [a.label for a in truth.atoms]
csv = Path("results") / f"pmr_{tabdir}_{var}_{lo_s}.csv"


def errors(r, model):
    o = {}
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind in ("scale", "ext"):
            continue
        if kind == "twin":
            o[nm] = r.x[j] - M_TRUE; continue
        ia = labels.index(nm.split(".")[0])
        if kind == "U":
            o[nm] = r.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2
        else:
            b = model.basis_frac[j]
            d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            o[nm] = float(d @ b) / float(b @ b)
    return o


for seed in range(lo_s, hi_s):
    f = Path("results") / tabdir / f"{var}_seed{seed}.npz"
    if not f.exists():
        print(f"seed {seed}: no table"); continue
    z = np.load(f, allow_pickle=True)
    H = np.c_[z["h"], z["k"], z["l"]].astype(int); P = np.c_[z["p0"], z["p1"], z["p2"]].astype(int)
    F2o, sg_ = z["F2"], z["sigF2"]; fl = np.c_[z["f1"], z["f2"]].astype(float); kind = z["kind"].astype(str)
    c2m = kind == "c2"
    H_eff = H.copy(); P_eff = P.copy(); P_eff[c2m] = H[c2m]
    start = perturbed_crystal(truth, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    rows = []
    for arm in ARMS:
        t0 = time.time()
        robust = arm.endswith("r")
        kw = dict(outlier_sigma=5.0, gate_free=GATE_FREE) if robust else {}
        if arm[0] == "U":
            sel = ~c2m & (fl[:, 0] == 1)
            model = RefineModel(start)
            r = refine(model, H[sel], F2o[sel], sg_[sel], **kw)
        else:
            model = RefineModel(start, refine_twin=True, twin_start=0.15)
            r = refine(model, H_eff, F2o, sg_, partner=P_eff, flags=fl, **kw)
        err = errors(r, model)
        row = dict(variant=var, seed=seed, arm=arm, converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2, n_par=r.n_par, n_obs=r.n_obs,
                   n_rejected=len(r.rejected), rejected_kinds=",".join(sorted(set(kind[(~c2m & (fl[:, 0] == 1)) if arm[0] == "U" else slice(None)][r.rejected]))) if len(r.rejected) else "",
                   ignored=",".join(r.at_bound_ignored), m_true=M_TRUE, m=r.twin_fraction, sigma_m=r.sigma_twin_fraction, seconds=time.time() - t0)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed} {var}: " + "; ".join(f"{r_['arm']} conv={r_['converged']} GooF {r_['goof']:.2f} rej {r_['n_rejected']}" for r_ in rows) + f"; m (last arm) {rows[-1]['m']:.3f}", flush=True)
